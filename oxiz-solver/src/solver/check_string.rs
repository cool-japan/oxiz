//! String theory constraint checking

#[allow(unused_imports)]
use crate::prelude::*;
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::Solver;

impl Solver {
    pub(super) fn check_string_constraints(&self, manager: &TermManager) -> bool {
        // Collect string variable assignments and length constraints
        let mut string_assignments: FxHashMap<TermId, String> = FxHashMap::default();
        let mut length_constraints: FxHashMap<TermId, i64> = FxHashMap::default();
        let mut concat_equalities: Vec<(Vec<TermId>, String)> = Vec::new();
        let mut replace_all_constraints: Vec<(TermId, TermId, String, String, String)> = Vec::new();

        // First pass: collect all string assignments and constraints from assertions
        for &assertion in &self.assertions {
            self.collect_string_constraints(
                assertion,
                manager,
                &mut string_assignments,
                &mut length_constraints,
                &mut concat_equalities,
                &mut replace_all_constraints,
            );
        }

        // Second pass: Now that all variable assignments are collected, resolve replace_all constraints
        // where source was a variable that is now known
        for &assertion in &self.assertions {
            self.collect_replace_all_with_resolved_vars(
                assertion,
                manager,
                &string_assignments,
                &mut replace_all_constraints,
            );
        }

        // Check 1: Length vs concrete string conflicts (string_04 fix)
        // If we have len(x) = n and x = "literal", check if len("literal") == n
        for (&var, &declared_len) in &length_constraints {
            if let Some(value) = string_assignments.get(&var) {
                let actual_len = value.chars().count() as i64;
                if actual_len != declared_len {
                    return true; // Conflict: declared length != actual length
                }
            }
        }

        // Check 2: Concatenation length consistency (string_02 fix)
        // If we have concat(a, b, c) = "result", check if sum of lengths is consistent
        for (operands, result_str) in &concat_equalities {
            let result_len = result_str.chars().count() as i64;
            let mut total_declared_len = 0i64;
            let mut all_have_length = true;

            for operand in operands {
                if let Some(&len) = length_constraints.get(operand) {
                    total_declared_len += len;
                } else if let Some(value) = string_assignments.get(operand) {
                    total_declared_len += value.chars().count() as i64;
                } else {
                    all_have_length = false;
                    break;
                }
            }

            if all_have_length && total_declared_len != result_len {
                return true; // Conflict: sum of operand lengths != result length
            }
        }

        // Check 3: Replace-all operation semantics (string_08 fix)
        // If we have replace_all(s, old, new) = result, with s, old, new, result all known,
        // verify that the operation produces the expected result
        for (result_var, source_var, source_val, pattern, replacement) in &replace_all_constraints {
            // Check if result is assigned to a concrete value
            if let Some(result_val) = string_assignments.get(result_var) {
                // If source contains the pattern and pattern != replacement,
                // then result cannot equal source
                if !pattern.is_empty() && source_val.contains(pattern) && pattern != replacement {
                    // Compute actual result
                    let actual_result = source_val.replace(pattern, replacement);
                    if &actual_result != result_val {
                        return true; // Conflict: replace_all result mismatch
                    }
                }
            }
            // Also check if source is concrete but has a length constraint
            // The source_var might not be concrete but the source_val is already collected
            if length_constraints.contains_key(source_var) {
                if let Some(result_val) = string_assignments.get(result_var) {
                    // Source is constrained but result is concrete - check pattern effects
                    if !pattern.is_empty() {
                        // Check if pattern exists in source - if so, result must be different
                        if source_val.contains(pattern) && pattern != replacement {
                            // If source and result are claimed to be equal, but replacement would change it
                            if source_val == result_val.as_str() {
                                return true; // Conflict
                            }
                        }
                    }
                }
            }
        }

        false // No conflict found
    }

    /// Recursively collect string constraints from a term
    fn collect_string_constraints(
        &self,
        term: TermId,
        manager: &TermManager,
        string_assignments: &mut FxHashMap<TermId, String>,
        length_constraints: &mut FxHashMap<TermId, i64>,
        concat_equalities: &mut Vec<(Vec<TermId>, String)>,
        replace_all_constraints: &mut Vec<(TermId, TermId, String, String, String)>,
    ) {
        let Some(term_data) = manager.get(term) else {
            return;
        };

        match &term_data.kind {
            // Handle equality: look for string-related equalities
            TermKind::Eq(lhs, rhs) => {
                // Check for variable = string literal
                if let Some(lit) = self.get_string_literal(*rhs, manager) {
                    // lhs = "literal"
                    if self.is_string_variable(*lhs, manager) {
                        string_assignments.insert(*lhs, lit);
                    }
                } else if let Some(lit) = self.get_string_literal(*lhs, manager) {
                    // "literal" = rhs
                    if self.is_string_variable(*rhs, manager) {
                        string_assignments.insert(*rhs, lit);
                    }
                }

                // Check for length constraint: (= (str.len x) n)
                if let Some((var, len)) = self.extract_length_constraint(*lhs, *rhs, manager) {
                    length_constraints.insert(var, len);
                } else if let Some((var, len)) = self.extract_length_constraint(*rhs, *lhs, manager)
                {
                    length_constraints.insert(var, len);
                }

                // Check for concat equality: (= (str.++ a b c) "result")
                if let Some(result_str) = self.get_string_literal(*rhs, manager) {
                    if let Some(operands) = self.extract_concat_operands(*lhs, manager) {
                        concat_equalities.push((operands, result_str));
                    }
                } else if let Some(result_str) = self.get_string_literal(*lhs, manager) {
                    if let Some(operands) = self.extract_concat_operands(*rhs, manager) {
                        concat_equalities.push((operands, result_str));
                    }
                }

                // Check for replace_all: (= result (str.replace_all s old new))
                if let Some((source, pattern, replacement)) =
                    self.extract_replace_all(*rhs, manager)
                {
                    // Get source value either directly or via variable assignment
                    let source_val = self
                        .get_string_literal(source, manager)
                        .or_else(|| string_assignments.get(&source).cloned());
                    if let Some(source_val) = source_val {
                        if let Some(pattern_val) = self.get_string_literal(pattern, manager) {
                            if let Some(replacement_val) =
                                self.get_string_literal(replacement, manager)
                            {
                                replace_all_constraints.push((
                                    *lhs,
                                    source,
                                    source_val,
                                    pattern_val,
                                    replacement_val,
                                ));
                            }
                        }
                    }
                } else if let Some((source, pattern, replacement)) =
                    self.extract_replace_all(*lhs, manager)
                {
                    // Get source value either directly or via variable assignment
                    let source_val = self
                        .get_string_literal(source, manager)
                        .or_else(|| string_assignments.get(&source).cloned());
                    if let Some(source_val) = source_val {
                        if let Some(pattern_val) = self.get_string_literal(pattern, manager) {
                            if let Some(replacement_val) =
                                self.get_string_literal(replacement, manager)
                            {
                                replace_all_constraints.push((
                                    *rhs,
                                    source,
                                    source_val,
                                    pattern_val,
                                    replacement_val,
                                ));
                            }
                        }
                    }
                }

                // Recursively check children
                self.collect_string_constraints(
                    *lhs,
                    manager,
                    string_assignments,
                    length_constraints,
                    concat_equalities,
                    replace_all_constraints,
                );
                self.collect_string_constraints(
                    *rhs,
                    manager,
                    string_assignments,
                    length_constraints,
                    concat_equalities,
                    replace_all_constraints,
                );
            }

            // Handle And: recurse into all conjuncts
            TermKind::And(args) => {
                for &arg in args {
                    self.collect_string_constraints(
                        arg,
                        manager,
                        string_assignments,
                        length_constraints,
                        concat_equalities,
                        replace_all_constraints,
                    );
                }
            }

            // Handle other compound terms
            TermKind::Or(args) => {
                for &arg in args {
                    self.collect_string_constraints(
                        arg,
                        manager,
                        string_assignments,
                        length_constraints,
                        concat_equalities,
                        replace_all_constraints,
                    );
                }
            }

            TermKind::Not(inner) => {
                self.collect_string_constraints(
                    *inner,
                    manager,
                    string_assignments,
                    length_constraints,
                    concat_equalities,
                    replace_all_constraints,
                );
            }

            TermKind::Implies(lhs, rhs) => {
                self.collect_string_constraints(
                    *lhs,
                    manager,
                    string_assignments,
                    length_constraints,
                    concat_equalities,
                    replace_all_constraints,
                );
                self.collect_string_constraints(
                    *rhs,
                    manager,
                    string_assignments,
                    length_constraints,
                    concat_equalities,
                    replace_all_constraints,
                );
            }

            _ => {}
        }
    }

    /// Get string literal value from a term
    fn get_string_literal(&self, term: TermId, manager: &TermManager) -> Option<String> {
        let term_data = manager.get(term)?;
        if let TermKind::StringLit(s) = &term_data.kind {
            Some(s.clone())
        } else {
            None
        }
    }

    /// Check if a term is a string variable (not a literal or operation)
    fn is_string_variable(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(term_data) = manager.get(term) else {
            return false;
        };
        matches!(term_data.kind, TermKind::Var(_))
    }

    /// Extract length constraint: (str.len var) = n
    fn extract_length_constraint(
        &self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
    ) -> Option<(TermId, i64)> {
        let lhs_data = manager.get(lhs)?;
        let rhs_data = manager.get(rhs)?;

        // Check if lhs is (str.len var) and rhs is an integer constant
        if let TermKind::StrLen(inner) = &lhs_data.kind {
            if let TermKind::IntConst(n) = &rhs_data.kind {
                return n.to_i64().map(|len| (*inner, len));
            }
        }

        None
    }

    /// Extract operands from a concat expression
    fn extract_concat_operands(&self, term: TermId, manager: &TermManager) -> Option<Vec<TermId>> {
        let term_data = manager.get(term)?;

        match &term_data.kind {
            TermKind::StrConcat(lhs, rhs) => {
                let mut operands = Vec::new();
                // Flatten nested concats
                self.flatten_concat(*lhs, manager, &mut operands);
                self.flatten_concat(*rhs, manager, &mut operands);
                Some(operands)
            }
            _ => None,
        }
    }

    /// Flatten a concat tree into a list of operands
    fn flatten_concat(&self, term: TermId, manager: &TermManager, operands: &mut Vec<TermId>) {
        let Some(term_data) = manager.get(term) else {
            operands.push(term);
            return;
        };

        match &term_data.kind {
            TermKind::StrConcat(lhs, rhs) => {
                self.flatten_concat(*lhs, manager, operands);
                self.flatten_concat(*rhs, manager, operands);
            }
            _ => {
                operands.push(term);
            }
        }
    }

    /// Extract replace_all operation: (str.replace_all source pattern replacement)
    fn extract_replace_all(
        &self,
        term: TermId,
        manager: &TermManager,
    ) -> Option<(TermId, TermId, TermId)> {
        let term_data = manager.get(term)?;
        if let TermKind::StrReplaceAll(source, pattern, replacement) = &term_data.kind {
            Some((*source, *pattern, *replacement))
        } else {
            None
        }
    }

    /// Second pass collection for replace_all with resolved variable assignments
    fn collect_replace_all_with_resolved_vars(
        &self,
        term: TermId,
        manager: &TermManager,
        string_assignments: &FxHashMap<TermId, String>,
        replace_all_constraints: &mut Vec<(TermId, TermId, String, String, String)>,
    ) {
        let Some(term_data) = manager.get(term) else {
            return;
        };

        match &term_data.kind {
            TermKind::Eq(lhs, rhs) => {
                // Check for replace_all with variable source that is now resolved
                if let Some((source, pattern, replacement)) =
                    self.extract_replace_all(*rhs, manager)
                {
                    // Try to resolve source from assignments
                    if let Some(source_val) = string_assignments.get(&source) {
                        if let Some(pattern_val) = self.get_string_literal(pattern, manager) {
                            if let Some(replacement_val) =
                                self.get_string_literal(replacement, manager)
                            {
                                // Only add if not already present
                                let entry = (
                                    *lhs,
                                    source,
                                    source_val.clone(),
                                    pattern_val,
                                    replacement_val,
                                );
                                if !replace_all_constraints.contains(&entry) {
                                    replace_all_constraints.push(entry);
                                }
                            }
                        }
                    }
                } else if let Some((source, pattern, replacement)) =
                    self.extract_replace_all(*lhs, manager)
                {
                    // Try to resolve source from assignments
                    if let Some(source_val) = string_assignments.get(&source) {
                        if let Some(pattern_val) = self.get_string_literal(pattern, manager) {
                            if let Some(replacement_val) =
                                self.get_string_literal(replacement, manager)
                            {
                                // Only add if not already present
                                let entry = (
                                    *rhs,
                                    source,
                                    source_val.clone(),
                                    pattern_val,
                                    replacement_val,
                                );
                                if !replace_all_constraints.contains(&entry) {
                                    replace_all_constraints.push(entry);
                                }
                            }
                        }
                    }
                }

                // Recursively check children
                self.collect_replace_all_with_resolved_vars(
                    *lhs,
                    manager,
                    string_assignments,
                    replace_all_constraints,
                );
                self.collect_replace_all_with_resolved_vars(
                    *rhs,
                    manager,
                    string_assignments,
                    replace_all_constraints,
                );
            }
            TermKind::And(args) => {
                for &arg in args {
                    self.collect_replace_all_with_resolved_vars(
                        arg,
                        manager,
                        string_assignments,
                        replace_all_constraints,
                    );
                }
            }
            TermKind::Or(args) => {
                for &arg in args {
                    self.collect_replace_all_with_resolved_vars(
                        arg,
                        manager,
                        string_assignments,
                        replace_all_constraints,
                    );
                }
            }
            TermKind::Not(inner) => {
                self.collect_replace_all_with_resolved_vars(
                    *inner,
                    manager,
                    string_assignments,
                    replace_all_constraints,
                );
            }
            _ => {}
        }
    }
}
