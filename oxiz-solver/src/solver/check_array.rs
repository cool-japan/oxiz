//! Array theory constraint checking

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::Solver;

impl Solver {
    pub(super) fn check_array_constraints(&self, manager: &TermManager) -> bool {
        // Collect select constraints: (select a i) = v
        let mut select_values: FxHashMap<(TermId, TermId), TermId> = FxHashMap::default();
        // Collect store-select patterns: (select (store a i v) i)
        let mut store_select_same_index: Vec<(TermId, TermId, TermId, TermId)> = Vec::new(); // (array, index, stored_val, result)
        // Collect array equalities: a = b
        let mut array_equalities: Vec<(TermId, TermId)> = Vec::new();
        // Collect all select assertions: (select_term, asserted_value)
        let mut select_assertions: Vec<(TermId, TermId)> = Vec::new();
        // Collect negated select assertions: not(= (select ...) val) -> (select_term, val)
        let mut negated_select_assertions: Vec<(TermId, TermId)> = Vec::new();
        // Collect read-consistency conflicts detected during collection
        let mut read_conflicts: Vec<(TermId, TermId)> = Vec::new();

        // Collect array variable aliases: array_var → store_term
        // For assertions of the form (= B (store A i v)), maps B → the store term.
        // This allows select(B, i) to be resolved via the read-over-write axiom.
        let array_var_aliases = self.collect_array_var_aliases(manager);

        for &assertion in &self.assertions {
            self.collect_array_constraints(
                assertion,
                manager,
                &mut select_values,
                &mut store_select_same_index,
                &mut array_equalities,
                &mut select_assertions,
                &mut negated_select_assertions,
                &mut read_conflicts,
            );
        }

        // Resolve alias-based conflicts:
        // For each select_assertion (select_term, asserted_value) where the array in
        // select_term is an alias for a store, also check the aliased version.
        let mut alias_select_assertions: Vec<(TermId, TermId)> = Vec::new();
        for &(select_term, asserted_value) in &select_assertions {
            if let Some(resolved) =
                self.resolve_select_through_alias(select_term, &array_var_aliases, manager)
            {
                alias_select_assertions.push((resolved, asserted_value));
            }
        }
        select_assertions.extend(alias_select_assertions);

        // Similarly resolve negated select assertions through aliases.
        let mut alias_negated: Vec<(TermId, TermId)> = Vec::new();
        for &(select_term, negated_val) in &negated_select_assertions {
            if let Some(resolved) =
                self.resolve_select_through_alias(select_term, &array_var_aliases, manager)
            {
                alias_negated.push((resolved, negated_val));
            }
        }
        negated_select_assertions.extend(alias_negated);

        // Also inject alias-aware values into select_values so BV cross-theory checks work.
        // When (= val (select B i)) and B aliases store(A, i, v), we can infer val = v.
        self.inject_alias_select_values(
            &array_var_aliases,
            &mut select_values,
            manager,
        );

        // Check: Alias-derived BV ordering conflicts.
        // When (= val (select B i)), B aliases (store A i w), we infer val = w.
        // If BV ordering constraints on val are inconsistent with val = w, detect UNSAT.
        if !array_var_aliases.is_empty()
            && self.check_alias_bv_ordering_conflict(&array_var_aliases, manager)
        {
            return true;
        }

        // Check: Read-consistency conflicts (same array, same index, different values)
        for &(existing_val, new_val) in &read_conflicts {
            if self.are_different_values(existing_val, new_val, manager) {
                return true; // Conflict: (select a i) = v1 and (select a i) = v2 with v1 != v2
            }
        }

        // Check: Read-over-write with same index (array_03)
        // The axiom says: select(store(a, i, v), i) = v
        // So if we have assertion (= (select (store a i stored_val) i) result)
        // Then result MUST equal stored_val. If they're different, it's UNSAT.
        for &(_array, _index, stored_val, result) in &store_select_same_index {
            if result != stored_val {
                // Check if they're actually different concrete values
                if self.are_different_values(stored_val, result, manager) {
                    return true; // Conflict: axiom says result should be stored_val
                }
            }
        }

        // Check: Nested array read-over-write (array_08)
        // For each select assertion (select X i) = v, recursively evaluate X
        // to see if it simplifies via the axiom to a different value.
        // We also apply alias-aware evaluation to handle the pattern:
        //   (= B (store A i w))  +  (= (select B i) v)  where v ≠ w → UNSAT
        for &(select_term, asserted_value) in &select_assertions {
            // Standard evaluation (handles direct store terms in the array position).
            if let Some(evaluated_value) = self.evaluate_select_axiom(select_term, manager) {
                if evaluated_value != asserted_value
                    && self.are_different_values(evaluated_value, asserted_value, manager)
                {
                    return true; // Conflict: axiom says it should be evaluated_value
                }
            }
            // Alias-aware evaluation (handles array variables aliased to store terms).
            if !array_var_aliases.is_empty() {
                if let Some(evaluated_value) = self.evaluate_select_axiom_with_alias(
                    select_term,
                    &array_var_aliases,
                    manager,
                ) {
                    if evaluated_value != asserted_value
                        && self.are_different_values(evaluated_value, asserted_value, manager)
                    {
                        return true;
                    }
                }
            }
        }

        // Check: Negated store-select axiom enforcement
        // For each not(= (select X i) val), if the read-over-write axiom implies
        // select(X, i) = axiom_val, and axiom_val equals negated_val (directly or via positive
        // equalities), then we have a direct contradiction.
        // This handles two cases:
        //   1. Direct: not(= (select (store a 3 5) 3) 5) → axiom gives 5, negated is 5 → UNSAT
        //   2. Indirect: not(= (select (store a i v) i) 42) with (= v 42) → axiom gives v,
        //      v is constrained to 42 by positive assertion → UNSAT
        for &(select_term, negated_val) in &negated_select_assertions {
            // Standard evaluation.
            if let Some(axiom_val) = self.evaluate_select_axiom(select_term, manager) {
                if axiom_val == negated_val
                    || self.values_equal_concrete(axiom_val, negated_val, manager)
                    || self.is_value_constrained_to(axiom_val, negated_val, manager, &select_values)
                {
                    return true; // Contradiction: axiom forces this value, assertion denies it
                }
            }
            // Alias-aware evaluation.
            if !array_var_aliases.is_empty() {
                if let Some(axiom_val) = self.evaluate_select_axiom_with_alias(
                    select_term,
                    &array_var_aliases,
                    manager,
                ) {
                    if axiom_val == negated_val
                        || self.values_equal_concrete(axiom_val, negated_val, manager)
                        || self.is_value_constrained_to(
                            axiom_val,
                            negated_val,
                            manager,
                            &select_values,
                        )
                    {
                        return true;
                    }
                }
            }
            // Also check via direct store-select (one level, without recursive evaluation)
            if let Some(stored_val) = self.get_store_select_same_index_value(select_term, manager) {
                if stored_val == negated_val
                    || self.values_equal_concrete(stored_val, negated_val, manager)
                    || self.is_value_constrained_to(
                        stored_val,
                        negated_val,
                        manager,
                        &select_values,
                    )
                {
                    return true; // Contradiction: direct store axiom value matches negated value
                }
            }
        }

        // Check: Extensionality (array_06)
        // If a = b, then (select a i) = (select b i) for all i
        for &(array_a, array_b) in &array_equalities {
            // Check if there's a constraint that says select(a, i) != select(b, i) for some i
            for (&(sel_array, sel_index), &sel_val) in &select_values {
                if sel_array == array_a {
                    // Look for select(b, same_index) with different value
                    if let Some(&other_val) = select_values.get(&(array_b, sel_index)) {
                        if sel_val != other_val {
                            // Check if they're different literals
                            if self.are_different_values(sel_val, other_val, manager) {
                                return true;
                            }
                        }
                    }
                }
            }
            // Check for not(= (select a i) (select b i)) assertions
            for &assertion in &self.assertions {
                if self.is_select_inequality_assertion(assertion, array_a, array_b, manager) {
                    return true;
                }
            }
        }

        // Check: Cross-theory conflict (QF_ABV with variable equalities + BV arithmetic)
        // Example: x=#x05, select(a,x)=bvadd(x,#x01), select(a,#x05)=#x10
        // select(a,x) evaluates via x=5 to select(a,5)=6, but select(a,5)=16 → conflict
        {
            let var_equalities = self.collect_bv_var_equalities(manager);
            if !var_equalities.is_empty() {
                if self.check_cross_theory_conflict(&select_values, &var_equalities, manager) {
                    return true;
                }
            }
        }

        false
    }

    /// Evaluate a select term by recursively applying the read-over-write axiom
    /// select(store(a, i, v), i) = v
    /// Returns Some(value) if the select can be evaluated to a concrete value
    fn evaluate_select_axiom(&self, term: TermId, manager: &TermManager) -> Option<TermId> {
        let term_data = manager.get(term)?;

        if let TermKind::Select(array, index) = &term_data.kind {
            // First, check if the array itself needs simplification (recursive call)
            let simplified_array = self.simplify_array_term(*array, manager);

            // Check if simplified_array is a store with the same index
            if let Some(simplified_data) = manager.get(simplified_array) {
                if let TermKind::Store(_base, store_idx, stored_val) = &simplified_data.kind {
                    if self.terms_equal_simple(*store_idx, *index, manager) {
                        // select(store(a, i, v), i) = v
                        // Recursively evaluate the stored value
                        return Some(
                            self.evaluate_select_axiom(*stored_val, manager)
                                .unwrap_or(*stored_val),
                        );
                    }
                }
            }

            // Also check the original array if it's a store
            if let Some(array_data) = manager.get(*array) {
                if let TermKind::Store(_base, store_idx, stored_val) = &array_data.kind {
                    if self.terms_equal_simple(*store_idx, *index, manager) {
                        // select(store(a, i, v), i) = v
                        return Some(
                            self.evaluate_select_axiom(*stored_val, manager)
                                .unwrap_or(*stored_val),
                        );
                    }
                }
            }
        }

        None
    }

    /// Simplify an array term by applying the read-over-write axiom
    /// If the term is select(store(a, i, v), i), return v
    fn simplify_array_term(&self, term: TermId, manager: &TermManager) -> TermId {
        let Some(term_data) = manager.get(term) else {
            return term;
        };

        if let TermKind::Select(array, index) = &term_data.kind {
            // Check if array is a store with the same index
            if let Some(array_data) = manager.get(*array) {
                if let TermKind::Store(_base, store_idx, stored_val) = &array_data.kind {
                    if self.terms_equal_simple(*store_idx, *index, manager) {
                        // select(store(a, i, v), i) = v
                        // Recursively simplify the result
                        return self.simplify_array_term(*stored_val, manager);
                    }
                }
            }
        }

        term
    }

    /// Check if two terms represent different concrete values
    fn are_different_values(&self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        if a == b {
            return false;
        }
        let (Some(a_data), Some(b_data)) = (manager.get(a), manager.get(b)) else {
            return false;
        };
        match (&a_data.kind, &b_data.kind) {
            (TermKind::IntConst(s1), TermKind::IntConst(s2)) => s1 != s2,
            (
                TermKind::BitVecConst {
                    value: v1,
                    width: w1,
                },
                TermKind::BitVecConst {
                    value: v2,
                    width: w2,
                },
            ) => w1 == w2 && v1 != v2,
            (TermKind::RealConst(r1), TermKind::RealConst(r2)) => r1 != r2,
            _ => false,
        }
    }

    /// Collect array constraints from a term
    /// `in_positive_context` tracks whether we're in a positive (true) or negative (not) context
    fn collect_array_constraints(
        &self,
        term: TermId,
        manager: &TermManager,
        select_values: &mut FxHashMap<(TermId, TermId), TermId>,
        store_select_same_index: &mut Vec<(TermId, TermId, TermId, TermId)>,
        array_equalities: &mut Vec<(TermId, TermId)>,
        select_assertions: &mut Vec<(TermId, TermId)>,
        negated_select_assertions: &mut Vec<(TermId, TermId)>,
        read_conflicts: &mut Vec<(TermId, TermId)>,
    ) {
        self.collect_array_constraints_inner(
            term,
            manager,
            select_values,
            store_select_same_index,
            array_equalities,
            select_assertions,
            negated_select_assertions,
            read_conflicts,
            true,
        );
    }

    fn collect_array_constraints_inner(
        &self,
        term: TermId,
        manager: &TermManager,
        select_values: &mut FxHashMap<(TermId, TermId), TermId>,
        store_select_same_index: &mut Vec<(TermId, TermId, TermId, TermId)>,
        array_equalities: &mut Vec<(TermId, TermId)>,
        select_assertions: &mut Vec<(TermId, TermId)>,
        negated_select_assertions: &mut Vec<(TermId, TermId)>,
        read_conflicts: &mut Vec<(TermId, TermId)>,
        in_positive_context: bool,
    ) {
        let Some(term_data) = manager.get(term) else {
            return;
        };

        match &term_data.kind {
            TermKind::Eq(lhs, rhs) => {
                // Only check for array equality when in positive context (not inside a Not)
                // Array equality like (= a b) only means a equals b when it's asserted directly,
                // not when it's negated as (not (= a b))
                if in_positive_context {
                    if self.is_array_variable(*lhs, manager)
                        && self.is_array_variable(*rhs, manager)
                    {
                        array_equalities.push((*lhs, *rhs));
                    }
                }

                // Check for (select a i) = v — only in positive context
                if in_positive_context {
                    if let Some((array, index)) = self.extract_select(*lhs, manager) {
                        if let Some(&existing_val) = select_values.get(&(array, index)) {
                            if existing_val != *rhs {
                                read_conflicts.push((existing_val, *rhs));
                            }
                        } else {
                            select_values.insert((array, index), *rhs);
                        }
                        // Also record for nested array evaluation (array_08)
                        select_assertions.push((*lhs, *rhs));
                    }
                    if let Some((array, index)) = self.extract_select(*rhs, manager) {
                        if let Some(&existing_val) = select_values.get(&(array, index)) {
                            if existing_val != *lhs {
                                read_conflicts.push((existing_val, *lhs));
                            }
                        } else {
                            select_values.insert((array, index), *lhs);
                        }
                        // Also record for nested array evaluation (array_08)
                        select_assertions.push((*rhs, *lhs));
                    }

                    // Check for (select (store a i v) i) = result
                    if let Some((inner_array, outer_index)) = self.extract_select(*lhs, manager) {
                        if let Some((base_array, store_index, stored_val)) =
                            self.extract_store(inner_array, manager)
                        {
                            // Check if indices are the same
                            if self.terms_equal_simple(outer_index, store_index, manager) {
                                store_select_same_index.push((
                                    base_array,
                                    store_index,
                                    stored_val,
                                    *rhs,
                                ));
                            }
                        }
                    }
                    if let Some((inner_array, outer_index)) = self.extract_select(*rhs, manager) {
                        if let Some((base_array, store_index, stored_val)) =
                            self.extract_store(inner_array, manager)
                        {
                            if self.terms_equal_simple(outer_index, store_index, manager) {
                                store_select_same_index.push((
                                    base_array,
                                    store_index,
                                    stored_val,
                                    *lhs,
                                ));
                            }
                        }
                    }
                } else {
                    // Negative context: we are inside a not(= ...) expression.
                    // Collect negated select assertions: not(= (select array idx) val)
                    // These mean the assertion claims select(array, idx) != val.
                    // If the store-select axiom forces select(array, idx) = val, contradiction.
                    if self.extract_select(*lhs, manager).is_some() {
                        negated_select_assertions.push((*lhs, *rhs));
                    }
                    if self.extract_select(*rhs, manager).is_some() {
                        negated_select_assertions.push((*rhs, *lhs));
                    }
                }

                self.collect_array_constraints_inner(
                    *lhs,
                    manager,
                    select_values,
                    store_select_same_index,
                    array_equalities,
                    select_assertions,
                    negated_select_assertions,
                    read_conflicts,
                    in_positive_context,
                );
                self.collect_array_constraints_inner(
                    *rhs,
                    manager,
                    select_values,
                    store_select_same_index,
                    array_equalities,
                    select_assertions,
                    negated_select_assertions,
                    read_conflicts,
                    in_positive_context,
                );
            }
            TermKind::And(args) => {
                for &arg in args {
                    self.collect_array_constraints_inner(
                        arg,
                        manager,
                        select_values,
                        store_select_same_index,
                        array_equalities,
                        select_assertions,
                        negated_select_assertions,
                        read_conflicts,
                        in_positive_context,
                    );
                }
            }
            TermKind::Or(_args) => {
                // Don't collect from OR branches - they represent disjunctions
            }
            TermKind::Not(inner) => {
                // Flip the context when entering a Not
                self.collect_array_constraints_inner(
                    *inner,
                    manager,
                    select_values,
                    store_select_same_index,
                    array_equalities,
                    select_assertions,
                    negated_select_assertions,
                    read_conflicts,
                    !in_positive_context,
                );
            }
            _ => {}
        }
    }

    /// Check if term is an array variable
    fn is_array_variable(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(term_data) = manager.get(term) else {
            return false;
        };
        if let TermKind::Var(_) = &term_data.kind {
            // Check if the sort is an array sort
            if let Some(sort) = manager.sorts.get(term_data.sort) {
                return matches!(sort.kind, oxiz_core::SortKind::Array { .. });
            }
        }
        false
    }

    /// Extract (select array index) pattern
    fn extract_select(&self, term: TermId, manager: &TermManager) -> Option<(TermId, TermId)> {
        let term_data = manager.get(term)?;
        if let TermKind::Select(array, index) = &term_data.kind {
            Some((*array, *index))
        } else {
            None
        }
    }

    /// Extract (store array index value) pattern
    fn extract_store(
        &self,
        term: TermId,
        manager: &TermManager,
    ) -> Option<(TermId, TermId, TermId)> {
        let term_data = manager.get(term)?;
        if let TermKind::Store(array, index, value) = &term_data.kind {
            Some((*array, *index, *value))
        } else {
            None
        }
    }

    /// Check if two terms are structurally equal (simple comparison)
    fn terms_equal_simple(&self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        if a == b {
            return true;
        }
        let (Some(a_data), Some(b_data)) = (manager.get(a), manager.get(b)) else {
            return false;
        };
        match (&a_data.kind, &b_data.kind) {
            (TermKind::IntConst(s1), TermKind::IntConst(s2)) => s1 == s2,
            _ => false,
        }
    }

    /// Check if assertion says (= term1 term2)
    fn asserts_equality(
        &self,
        assertion: TermId,
        term1: TermId,
        term2: TermId,
        manager: &TermManager,
    ) -> bool {
        let Some(assertion_data) = manager.get(assertion) else {
            return false;
        };
        if let TermKind::Eq(lhs, rhs) = &assertion_data.kind {
            (*lhs == term1 && *rhs == term2) || (*lhs == term2 && *rhs == term1)
        } else {
            false
        }
    }

    /// Check if two terms represent equal concrete values (both are concrete literals with same value).
    /// Unlike `are_different_values`, this returns true when the values are provably equal.
    fn values_equal_concrete(&self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        if a == b {
            return true;
        }
        let (Some(a_data), Some(b_data)) = (manager.get(a), manager.get(b)) else {
            return false;
        };
        match (&a_data.kind, &b_data.kind) {
            (TermKind::IntConst(s1), TermKind::IntConst(s2)) => s1 == s2,
            (
                TermKind::BitVecConst {
                    value: v1,
                    width: w1,
                },
                TermKind::BitVecConst {
                    value: v2,
                    width: w2,
                },
            ) => w1 == w2 && v1 == v2,
            (TermKind::RealConst(r1), TermKind::RealConst(r2)) => r1 == r2,
            _ => false,
        }
    }

    /// For a select term `(select array index)`, if `array` is a store expression
    /// `(store base store_idx stored_val)` and `index == store_idx`, return `stored_val`.
    /// This directly applies the read-over-write axiom at one level.
    fn get_store_select_same_index_value(
        &self,
        select_term: TermId,
        manager: &TermManager,
    ) -> Option<TermId> {
        let term_data = manager.get(select_term)?;
        if let TermKind::Select(array, index) = &term_data.kind {
            let array_data = manager.get(*array)?;
            if let TermKind::Store(_base, store_idx, stored_val) = &array_data.kind {
                if self.terms_equal_simple(*store_idx, *index, manager) {
                    return Some(*stored_val);
                }
            }
        }
        None
    }

    /// Check if `value_term` is constrained by positive select-equality assertions to equal
    /// `target_val`. Used to detect cases where the stored variable is pinned to a concrete
    /// value that conflicts with a negated assertion.
    /// For example: `(= v 42)` asserted, and we want to know if `v` is constrained to equal 42.
    fn is_value_constrained_to(
        &self,
        value_term: TermId,
        target_val: TermId,
        manager: &TermManager,
        select_values: &FxHashMap<(TermId, TermId), TermId>,
    ) -> bool {
        // Direct identity check
        if value_term == target_val {
            return true;
        }
        if self.values_equal_concrete(value_term, target_val, manager) {
            return true;
        }

        // Check if there is a positive equality assertion (= value_term target_val)
        // by scanning the assertions for direct equalities.
        for &assertion in &self.assertions {
            let Some(assertion_data) = manager.get(assertion) else {
                continue;
            };
            if let TermKind::Eq(lhs, rhs) = &assertion_data.kind {
                // Check (= value_term target_val) or (= target_val value_term)
                if (*lhs == value_term && self.values_equal_concrete(*rhs, target_val, manager))
                    || (*rhs == value_term && self.values_equal_concrete(*lhs, target_val, manager))
                {
                    return true;
                }
                // Also check if value_term is bound to a select result that maps to target_val
                if *lhs == value_term {
                    if let Some((sel_array, sel_index)) = self.extract_select(*rhs, manager) {
                        if let Some(&mapped_val) = select_values.get(&(sel_array, sel_index)) {
                            if self.values_equal_concrete(mapped_val, target_val, manager) {
                                return true;
                            }
                        }
                    }
                }
                if *rhs == value_term {
                    if let Some((sel_array, sel_index)) = self.extract_select(*lhs, manager) {
                        if let Some(&mapped_val) = select_values.get(&(sel_array, sel_index)) {
                            if self.values_equal_concrete(mapped_val, target_val, manager) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Collect BV variable-to-constant equalities from assertions.
    /// For each assertion of the form `(= Var BitVecConst)` or `(= BitVecConst Var)`,
    /// record the mapping from the variable TermId to (concrete_value, width).
    fn collect_bv_var_equalities(
        &self,
        manager: &TermManager,
    ) -> FxHashMap<TermId, (num_bigint::BigInt, u32)> {
        let mut result: FxHashMap<TermId, (num_bigint::BigInt, u32)> = FxHashMap::default();
        for &assertion in &self.assertions {
            let Some(data) = manager.get(assertion) else {
                continue;
            };
            if let TermKind::Eq(lhs, rhs) = &data.kind {
                self.try_record_var_const_eq(*lhs, *rhs, manager, &mut result);
                self.try_record_var_const_eq(*rhs, *lhs, manager, &mut result);
            }
        }
        result
    }

    /// If `var_term` is a Var and `val_term` is a BitVecConst, record the mapping.
    fn try_record_var_const_eq(
        &self,
        var_term: TermId,
        val_term: TermId,
        manager: &TermManager,
        result: &mut FxHashMap<TermId, (num_bigint::BigInt, u32)>,
    ) {
        let (Some(var_data), Some(val_data)) = (manager.get(var_term), manager.get(val_term))
        else {
            return;
        };
        if let TermKind::Var(_) = &var_data.kind {
            if let TermKind::BitVecConst { value, width } = &val_data.kind {
                result.insert(var_term, (value.clone(), *width));
            }
        }
    }

    /// Compute the modular mask for a given bit width: (2^width - 1).
    /// Returns a BigInt that can be used for masking.
    fn bv_mask(width: u32) -> num_bigint::BigInt {
        use num_bigint::BigInt;
        use num_traits::One;
        (BigInt::one() << width as usize) - BigInt::one()
    }

    /// Evaluate a BV expression to a concrete (value, width) pair given variable bindings.
    /// Returns None if the expression cannot be fully evaluated.
    fn evaluate_bv_expr(
        &self,
        term: TermId,
        var_equalities: &FxHashMap<TermId, (num_bigint::BigInt, u32)>,
        manager: &TermManager,
    ) -> Option<(num_bigint::BigInt, u32)> {
        use num_bigint::BigInt;
        use num_traits::Zero;

        let term_data = manager.get(term)?;
        match &term_data.kind {
            TermKind::BitVecConst { value, width } => Some((value.clone(), *width)),
            TermKind::Var(_) => var_equalities.get(&term).cloned(),
            TermKind::BvAdd(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                let mask = Self::bv_mask(wa);
                Some(((va + vb) & &mask, wa))
            }
            TermKind::BvSub(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                let mask = Self::bv_mask(wa);
                // Add mask+1 to avoid negative results before masking
                Some(((va - vb + (&mask + BigInt::from(1i32))) & &mask, wa))
            }
            TermKind::BvMul(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                let mask = Self::bv_mask(wa);
                Some(((va * vb) & &mask, wa))
            }
            TermKind::BvUdiv(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                if vb.is_zero() {
                    // BV unsigned division by zero is defined as all-ones
                    return Some((Self::bv_mask(wa), wa));
                }
                Some((va / vb, wa))
            }
            TermKind::BvUrem(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                if vb.is_zero() {
                    return Some((va, wa));
                }
                Some((va % vb, wa))
            }
            TermKind::BvAnd(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                Some((va & vb, wa))
            }
            TermKind::BvOr(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                Some((va | vb, wa))
            }
            TermKind::BvXor(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                if wa != wb {
                    return None;
                }
                Some((va ^ vb, wa))
            }
            TermKind::BvNot(a) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let mask = Self::bv_mask(wa);
                // NOT in BV: flip all bits within the width
                Some((!va & mask, wa))
            }
            TermKind::BvShl(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, _wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                let mask = Self::bv_mask(wa);
                // Convert shift amount to usize safely
                let shift: usize = vb.to_u64_digits().1.first().copied().unwrap_or(0) as usize;
                if shift >= wa as usize {
                    return Some((BigInt::zero(), wa));
                }
                Some(((va << shift) & mask, wa))
            }
            TermKind::BvLshr(a, b) => {
                let (va, wa) = self.evaluate_bv_expr(*a, var_equalities, manager)?;
                let (vb, _wb) = self.evaluate_bv_expr(*b, var_equalities, manager)?;
                let shift: usize = vb.to_u64_digits().1.first().copied().unwrap_or(0) as usize;
                if shift >= wa as usize {
                    return Some((BigInt::zero(), wa));
                }
                Some((va >> shift, wa))
            }
            _ => None,
        }
    }

    /// Cross-theory conflict check: detect conflicts that require variable substitution.
    ///
    /// Given:
    ///   var_equalities:  x → (5, 8)      from `(= x #x05)`
    ///   select_values:   (a, x)   → bvadd(x, #x01)   from `(= (select a x) (bvadd x #x01))`
    ///                    (a, #x05) → #x10             from `(= (select a #x05) #x10)`
    ///
    /// After evaluating indices and values:
    ///   (a, x) index evaluates to 5, value bvadd(x,1) evaluates to 6
    ///   (a, #x05) index is 5, value #x10 is 16
    ///   Same index, different values → UNSAT
    fn check_cross_theory_conflict(
        &self,
        select_values: &FxHashMap<(TermId, TermId), TermId>,
        var_equalities: &FxHashMap<TermId, (num_bigint::BigInt, u32)>,
        manager: &TermManager,
    ) -> bool {
        use num_bigint::BigInt;

        // Build a list of (array_term, evaluated_index: (BigInt,u32), evaluated_value: (BigInt,u32))
        // for all select_values entries that can be fully evaluated.
        struct EvalEntry {
            array: TermId,
            index_val: (BigInt, u32),
            value_val: (BigInt, u32),
        }

        let mut evaluated: Vec<EvalEntry> = Vec::new();

        for (&(array, index_term), &value_term) in select_values {
            let Some(index_val) = self.evaluate_bv_expr(index_term, var_equalities, manager) else {
                continue;
            };
            let Some(value_val) = self.evaluate_bv_expr(value_term, var_equalities, manager) else {
                continue;
            };
            evaluated.push(EvalEntry {
                array,
                index_val,
                value_val,
            });
        }

        // Check all pairs with the same array and same evaluated index
        for i in 0..evaluated.len() {
            for j in (i + 1)..evaluated.len() {
                let ei = &evaluated[i];
                let ej = &evaluated[j];
                if ei.array != ej.array {
                    continue;
                }
                // Indices must have same width and value to be considered identical
                if ei.index_val != ej.index_val {
                    continue;
                }
                // Same array, same index → must have same value
                if ei.value_val != ej.value_val {
                    return true; // Conflict
                }
            }
        }

        false
    }

    /// Check if assertion says not(= (select a i) (select b i))
    fn is_select_inequality_assertion(
        &self,
        assertion: TermId,
        array_a: TermId,
        array_b: TermId,
        manager: &TermManager,
    ) -> bool {
        let Some(assertion_data) = manager.get(assertion) else {
            return false;
        };
        if let TermKind::Not(inner) = &assertion_data.kind {
            let Some(inner_data) = manager.get(*inner) else {
                return false;
            };
            if let TermKind::Eq(lhs, rhs) = &inner_data.kind {
                // Check if lhs is select(a, i) and rhs is select(b, i)
                if let (Some((sel_a, idx_a)), Some((sel_b, idx_b))) = (
                    self.extract_select(*lhs, manager),
                    self.extract_select(*rhs, manager),
                ) {
                    if ((sel_a == array_a && sel_b == array_b)
                        || (sel_a == array_b && sel_b == array_a))
                        && self.terms_equal_simple(idx_a, idx_b, manager)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Array variable alias resolution
    //
    // These methods handle the pattern:
    //   (declare-const B Array)
    //   (assert (= B (store A i v)))   ← B is an alias for the store term
    //   (assert (= (select B i) W))    ← select must resolve through B's alias
    // ──────────────────────────────────────────────────────────────────────────

    /// Collect array variable aliases from assertions.
    ///
    /// For each assertion `(= B (store A i v))` or `(= (store A i v) B)` where
    /// `B` is a variable with an array sort, record `B → store_term_id`.
    ///
    /// Multiple levels of aliasing (B → store1 → store2) are handled by a
    /// fixpoint iteration: at most `max_iters` passes to resolve chains.
    fn collect_array_var_aliases(
        &self,
        manager: &TermManager,
    ) -> FxHashMap<TermId, TermId> {
        let mut aliases: FxHashMap<TermId, TermId> = FxHashMap::default();

        for &assertion in &self.assertions {
            let Some(data) = manager.get(assertion) else {
                continue;
            };
            if let TermKind::Eq(lhs, rhs) = &data.kind {
                self.try_record_array_alias(*lhs, *rhs, manager, &mut aliases);
                self.try_record_array_alias(*rhs, *lhs, manager, &mut aliases);
            }
        }

        // Fixpoint: resolve transitive aliases (B → C, C → store(…) becomes B → store(…))
        let max_iters = 8;
        for _ in 0..max_iters {
            let mut changed = false;
            let keys: Vec<TermId> = aliases.keys().copied().collect();
            for key in keys {
                let target = aliases[&key];
                // If the alias target is itself aliased to a store, follow through.
                if let Some(&next_target) = aliases.get(&target) {
                    let target_data = manager.get(next_target);
                    if target_data.is_some_and(|d| matches!(d.kind, TermKind::Store(..))) {
                        aliases.insert(key, next_target);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        aliases
    }

    /// If `var_term` is an array variable and `store_term` is a Store expression,
    /// record `var_term → store_term` in `aliases`.
    fn try_record_array_alias(
        &self,
        var_term: TermId,
        store_term: TermId,
        manager: &TermManager,
        aliases: &mut FxHashMap<TermId, TermId>,
    ) {
        let (Some(var_data), Some(store_data)) = (manager.get(var_term), manager.get(store_term))
        else {
            return;
        };
        // LHS must be a variable (declared array constant).
        if !matches!(var_data.kind, TermKind::Var(_)) {
            return;
        }
        // RHS must be a Store expression.
        if !matches!(store_data.kind, TermKind::Store(..)) {
            return;
        }
        aliases.insert(var_term, store_term);
    }

    /// Given a select term `(select B i)`, check if `B` is in `array_var_aliases`.
    /// If so, return a *virtual* select term that uses the aliased store expression.
    ///
    /// Because we cannot create new TermIds here (no mutable manager), we instead
    /// return the *original select term* with its array replaced by the alias
    /// target if the alias target is a Store — but since we cannot mutate the
    /// term graph, we return a synthetic representation by returning the raw
    /// store term that would be the array operand.
    ///
    /// Concretely, we return `Some(virtual_select_id)` only when the resolved
    /// array is a `Store` and the select index matches the store index, letting
    /// the caller use `evaluate_select_axiom_with_alias` to get the stored value.
    fn resolve_select_through_alias(
        &self,
        select_term: TermId,
        aliases: &FxHashMap<TermId, TermId>,
        manager: &TermManager,
    ) -> Option<TermId> {
        let term_data = manager.get(select_term)?;
        let TermKind::Select(array, _index) = &term_data.kind else {
            return None;
        };
        // Only apply when the array is a Var that has an alias.
        let array_data = manager.get(*array)?;
        if !matches!(array_data.kind, TermKind::Var(_)) {
            return None;
        }
        aliases.get(array)?;
        // Return the select term unchanged — evaluate_select_axiom will resolve
        // it via the alias map passed through the wrapper.
        // (We signal "this needs alias resolution" by returning Some(select_term).)
        Some(select_term)
    }

    /// Evaluate a select term with awareness of array variable aliases.
    ///
    /// Like `evaluate_select_axiom`, but before checking if the array is a Store,
    /// first resolves the array through `aliases`.
    fn evaluate_select_axiom_with_alias(
        &self,
        select_term: TermId,
        aliases: &FxHashMap<TermId, TermId>,
        manager: &TermManager,
    ) -> Option<TermId> {
        let term_data = manager.get(select_term)?;
        let TermKind::Select(array, index) = &term_data.kind else {
            return None;
        };

        // Resolve the array through the alias map.
        let resolved_array = {
            let arr_data = manager.get(*array)?;
            if matches!(arr_data.kind, TermKind::Var(_)) {
                aliases.get(array).copied().unwrap_or(*array)
            } else {
                *array
            }
        };

        // Apply read-over-write axiom on the resolved array.
        let resolved_data = manager.get(resolved_array)?;
        if let TermKind::Store(_base, store_idx, stored_val) = &resolved_data.kind {
            if self.terms_equal_simple(*store_idx, *index, manager) {
                return Some(*stored_val);
            }
        }

        // Fall back to the standard evaluation (handles nested stores).
        self.evaluate_select_axiom(select_term, manager)
    }

    /// Inject alias-derived values into `select_values`.
    ///
    /// For every `(= val (select B i))` assertion where `B` is aliased to a
    /// store expression `(store A i v)`, we know `val = v` by the read-over-write
    /// axiom.  Add `(virtual_B_store, i) → v` and `(B_var, i) → v` into
    /// `select_values` so that downstream BV cross-theory checks (which use
    /// `select_values`) can detect contradictions involving `val`.
    ///
    /// We also directly check for UNSAT: if `(B_var, i)` already maps to a
    /// different concrete value than `v`, we detect a conflict.
    fn inject_alias_select_values(
        &self,
        aliases: &FxHashMap<TermId, TermId>,
        select_values: &mut FxHashMap<(TermId, TermId), TermId>,
        manager: &TermManager,
    ) {
        if aliases.is_empty() {
            return;
        }

        // Scan assertions for `(= val (select B i))` or `(= (select B i) val)`.
        for &assertion in &self.assertions {
            let Some(data) = manager.get(assertion) else {
                continue;
            };
            if let TermKind::Eq(lhs, rhs) = &data.kind {
                self.maybe_inject_alias_value(*lhs, *rhs, aliases, select_values, manager);
                self.maybe_inject_alias_value(*rhs, *lhs, aliases, select_values, manager);
            }
        }
    }

    /// Helper: if `select_term` is `(select B i)` and `B` aliases `(store A i v)`,
    /// record `(B, i) → v` and `(store_term, i) → v` in `select_values`.
    fn maybe_inject_alias_value(
        &self,
        select_term: TermId,
        _value_term: TermId,
        aliases: &FxHashMap<TermId, TermId>,
        select_values: &mut FxHashMap<(TermId, TermId), TermId>,
        manager: &TermManager,
    ) {
        let Some(data) = manager.get(select_term) else {
            return;
        };
        let TermKind::Select(array, index) = &data.kind else {
            return;
        };
        let Some(arr_data) = manager.get(*array) else {
            return;
        };
        if !matches!(arr_data.kind, TermKind::Var(_)) {
            return;
        }
        let Some(&store_term) = aliases.get(array) else {
            return;
        };
        let Some(store_data) = manager.get(store_term) else {
            return;
        };
        let TermKind::Store(_base, store_idx, stored_val) = &store_data.kind else {
            return;
        };
        if !self.terms_equal_simple(*store_idx, *index, manager) {
            return;
        }
        // `stored_val` is the value that the axiom forces at this index.
        // Record under the original array variable key so downstream checks
        // can look it up.
        select_values
            .entry((*array, *index))
            .or_insert(*stored_val);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Alias-derived BV ordering conflict detection
    //
    // Pattern:
    //   (= B (store A i w))           ← B aliases the store
    //   (= val (select B i))           ← val is bound to select(B, i) = w (by axiom)
    //   (bvugt val w)                  ← requires val > w, but val = w → UNSAT
    // ──────────────────────────────────────────────────────────────────────────

    /// Check for BV ordering conflicts that arise when a variable is derived
    /// from an alias-resolved array select.
    ///
    /// Specifically:
    /// 1. Build a map `derived_values: TermId → (BigInt, u32)` mapping scalar
    ///    variables to their concrete BV values implied by array axiom + aliases.
    /// 2. Scan BV ordering assertions (`bvugt`, `bvult`, `bvuge`, `bvule`) and
    ///    check if the derived value violates the ordering constraint.
    fn check_alias_bv_ordering_conflict(
        &self,
        aliases: &FxHashMap<TermId, TermId>,
        manager: &TermManager,
    ) -> bool {
        use num_bigint::BigInt;

        // Step 1: Build derived_values map.
        // For each assertion (= val (select B i)) where B aliases (store A i w)
        // and w is a concrete BV constant, record val → (w_value, width).
        let mut derived_values: FxHashMap<TermId, (BigInt, u32)> = FxHashMap::default();

        for &assertion in &self.assertions {
            let Some(data) = manager.get(assertion) else {
                continue;
            };
            if let TermKind::Eq(lhs, rhs) = &data.kind {
                self.try_derive_bv_from_alias_select(*lhs, *rhs, aliases, &mut derived_values, manager);
                self.try_derive_bv_from_alias_select(*rhs, *lhs, aliases, &mut derived_values, manager);
            }
        }

        if derived_values.is_empty() {
            return false;
        }

        // Step 2: Scan BV ordering assertions and check for conflicts.
        for &assertion in &self.assertions {
            let Some(data) = manager.get(assertion) else {
                continue;
            };
            if self.check_bv_ordering_against_derived(assertion, &derived_values, manager) {
                return true;
            }
            // Also check negated equalities: (not (= a b)) where a is derived and b is concrete.
            if let TermKind::Not(inner) = &data.kind {
                if let Some(inner_data) = manager.get(*inner) {
                    if let TermKind::Eq(lhs, rhs) = &inner_data.kind {
                        // not(= val concrete): val must be concrete by axiom but the negation says it isn't
                        if let Some((derived_val, dw)) = derived_values.get(lhs) {
                            if let Some(rhs_data) = manager.get(*rhs) {
                                if let TermKind::BitVecConst { value, width } = &rhs_data.kind {
                                    if *width == *dw && derived_val == value {
                                        return true;
                                    }
                                }
                            }
                        }
                        if let Some((derived_val, dw)) = derived_values.get(rhs) {
                            if let Some(lhs_data) = manager.get(*lhs) {
                                if let TermKind::BitVecConst { value, width } = &lhs_data.kind {
                                    if *width == *dw && derived_val == value {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        false
    }

    /// If `select_like` is `(select B i)` where `B` is in `aliases` and the stored
    /// value at index `i` is a concrete BV constant, and `scalar` is a Var,
    /// record `scalar → (stored_value, width)` in `derived_values`.
    fn try_derive_bv_from_alias_select(
        &self,
        scalar: TermId,
        select_like: TermId,
        aliases: &FxHashMap<TermId, TermId>,
        derived_values: &mut FxHashMap<TermId, (num_bigint::BigInt, u32)>,
        manager: &TermManager,
    ) {
        // `scalar` must be a Var (not a constant expression).
        let Some(scalar_data) = manager.get(scalar) else {
            return;
        };
        if !matches!(scalar_data.kind, TermKind::Var(_)) {
            return;
        }
        // `select_like` must be a Select.
        let Some(sel_data) = manager.get(select_like) else {
            return;
        };
        let TermKind::Select(array, index) = &sel_data.kind else {
            return;
        };
        // `array` must be a Var that is in the alias map.
        let Some(arr_data) = manager.get(*array) else {
            return;
        };
        if !matches!(arr_data.kind, TermKind::Var(_)) {
            return;
        }
        let Some(&store_term) = aliases.get(array) else {
            return;
        };
        let Some(store_data) = manager.get(store_term) else {
            return;
        };
        let TermKind::Store(_base, store_idx, stored_val) = &store_data.kind else {
            return;
        };
        // Indices must match.
        if !self.terms_equal_simple(*store_idx, *index, manager) {
            return;
        }
        // `stored_val` must be a concrete BV constant.
        let Some(stored_data) = manager.get(*stored_val) else {
            return;
        };
        if let TermKind::BitVecConst { value, width } = &stored_data.kind {
            derived_values.insert(scalar, (value.clone(), *width));
        }
    }

    /// Check whether a single BV ordering assertion conflicts with the given
    /// `derived_values` map.
    ///
    /// Handles: `bvugt`, `bvult`, `bvuge`, `bvule`, `bvsgt`, `bvslt`, `bvsge`,
    /// `bvsle`.  For each ordering assertion `(op a b)`, if one side is in
    /// `derived_values`, evaluate the constraint and return `true` if it is
    /// provably false.
    fn check_bv_ordering_against_derived(
        &self,
        assertion: TermId,
        derived_values: &FxHashMap<TermId, (num_bigint::BigInt, u32)>,
        manager: &TermManager,
    ) -> bool {
        use num_bigint::BigInt;
        use num_traits::One;

        let Some(data) = manager.get(assertion) else {
            return false;
        };

        // Helper: get concrete BV value for a term (either from derived_values or literal).
        let get_bv_val = |term: TermId| -> Option<(BigInt, u32)> {
            if let Some(v) = derived_values.get(&term) {
                return Some(v.clone());
            }
            let d = manager.get(term)?;
            if let TermKind::BitVecConst { value, width } = &d.kind {
                return Some((value.clone(), *width));
            }
            None
        };

        match &data.kind {
            TermKind::BvUlt(a, b) => {
                // a < b (unsigned)
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        return va >= vb; // conflict when NOT (va < vb)
                    }
                }
            }
            TermKind::BvUle(a, b) => {
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        return va > vb;
                    }
                }
            }
            TermKind::BvUgt(a, b) => {
                // a > b (unsigned) — equivalent to b < a
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        return va <= vb; // conflict when NOT (va > vb)
                    }
                }
            }
            TermKind::BvUge(a, b) => {
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        return va < vb;
                    }
                }
            }
            TermKind::BvSlt(a, b) => {
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        let half = BigInt::one() << (wa as usize - 1);
                        let mod_val = BigInt::one() << wa as usize;
                        let signed_a =
                            if va >= half { va.clone() - &mod_val } else { va.clone() };
                        let signed_b =
                            if vb >= half { vb.clone() - &mod_val } else { vb.clone() };
                        return signed_a >= signed_b;
                    }
                }
            }
            TermKind::BvSle(a, b) => {
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        let half = BigInt::one() << (wa as usize - 1);
                        let mod_val = BigInt::one() << wa as usize;
                        let signed_a =
                            if va >= half { va.clone() - &mod_val } else { va.clone() };
                        let signed_b =
                            if vb >= half { vb.clone() - &mod_val } else { vb.clone() };
                        return signed_a > signed_b;
                    }
                }
            }
            TermKind::BvSgt(a, b) => {
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        let half = BigInt::one() << (wa as usize - 1);
                        let mod_val = BigInt::one() << wa as usize;
                        let signed_a =
                            if va >= half { va.clone() - &mod_val } else { va.clone() };
                        let signed_b =
                            if vb >= half { vb.clone() - &mod_val } else { vb.clone() };
                        return signed_a <= signed_b;
                    }
                }
            }
            TermKind::BvSge(a, b) => {
                if let (Some((va, wa)), Some((vb, wb))) = (get_bv_val(*a), get_bv_val(*b)) {
                    if wa == wb {
                        let half = BigInt::one() << (wa as usize - 1);
                        let mod_val = BigInt::one() << wa as usize;
                        let signed_a =
                            if va >= half { va.clone() - &mod_val } else { va.clone() };
                        let signed_b =
                            if vb >= half { vb.clone() - &mod_val } else { vb.clone() };
                        return signed_a < signed_b;
                    }
                }
            }
            _ => {}
        }

        false
    }
}
