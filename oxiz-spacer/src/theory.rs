//! Theory-aware operations for Spacer.
//!
//! This module provides theory-specific enhancements for PDR/IC3,
//! including theory-aware generalization, interpolation, and model projection.
//!
//! Reference: Z3's `muz/spacer/spacer_context.cpp` theory integration

use crate::chc::PredId;
use oxiz_core::{SortId, TermId, TermManager};
use smallvec::SmallVec;

/// Outcome of one reduction step of [`TheoryIntegration::project_variables`].
enum ProjectStep {
    /// The subformula is fully projected.
    Done(TermId),
    /// The subformula is a conjunction; project these arguments first.
    Conjuncts(Vec<TermId>),
    /// The subformula is a disjunction; project these arguments first.
    Disjuncts(Vec<TermId>),
}

/// Theory integration for Spacer
pub struct TheoryIntegration;

impl TheoryIntegration {
    /// Create a new theory integration
    pub fn new() -> Self {
        Self
    }

    /// Check if a term involves linear arithmetic
    pub fn is_linear_arithmetic(term: TermId, manager: &TermManager) -> bool {
        if let Some(t) = manager.get(term) {
            matches!(
                t.sort,
                sort if sort == manager.sorts.int_sort || sort == manager.sorts.real_sort
            )
        } else {
            false
        }
    }

    /// Check if a cube *literal* is an arithmetic comparison (`<`, `<=`,
    /// `>`, `>=`, or `=`) over int/real-sorted operands.
    ///
    /// [`Self::is_linear_arithmetic`] checks a term's *own* sort, which
    /// is the right question for a bare operand (e.g. a variable) but the
    /// wrong one for a cube literal: a comparison like `x < 10` is itself
    /// `Bool`-sorted, so `is_linear_arithmetic(x_lt_10, ..)` is always
    /// `false` regardless of `x`'s sort. `theory_generalize` used exactly
    /// that (wrong) check to decide which literals to run its
    /// arithmetic-specific rewriting on, which meant that code path was
    /// unreachable for every realistic cube (cube literals are
    /// comparisons, not bare operands).
    fn is_arithmetic_literal(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        match manager.get(term).map(|t| &t.kind) {
            Some(
                TermKind::Lt(a, _) | TermKind::Le(a, _) | TermKind::Gt(a, _) | TermKind::Ge(a, _),
            ) => Self::is_linear_arithmetic(*a, manager),
            Some(TermKind::Eq(a, b)) => {
                Self::is_linear_arithmetic(*a, manager) || Self::is_linear_arithmetic(*b, manager)
            }
            _ => false,
        }
    }

    /// Check if a term involves arrays
    pub fn is_array_term(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::{SortKind, TermKind};

        if let Some(t) = manager.get(term) {
            // Check if the term is an array operation (Select or Store)
            match &t.kind {
                TermKind::Select(_, _) | TermKind::Store(_, _, _) => return true,
                _ => {}
            }

            // Check if the term's sort is an array sort
            if let Some(sort) = manager.sorts.get(t.sort)
                && matches!(sort.kind, SortKind::Array { .. })
            {
                return true;
            }
        }

        false
    }

    /// Check if a term involves bitvectors
    pub fn is_bitvector_term(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::{SortKind, TermKind};

        if let Some(t) = manager.get(term) {
            // Check if the term is a bitvector operation
            match &t.kind {
                TermKind::BitVecConst { .. }
                | TermKind::BvNot(_)
                | TermKind::BvAnd(_, _)
                | TermKind::BvOr(_, _)
                | TermKind::BvXor(_, _)
                | TermKind::BvAdd(_, _)
                | TermKind::BvSub(_, _)
                | TermKind::BvMul(_, _)
                | TermKind::BvUdiv(_, _)
                | TermKind::BvSdiv(_, _)
                | TermKind::BvUrem(_, _)
                | TermKind::BvSrem(_, _)
                | TermKind::BvShl(_, _)
                | TermKind::BvLshr(_, _)
                | TermKind::BvAshr(_, _)
                | TermKind::BvConcat(_, _)
                | TermKind::BvExtract { .. } => return true,
                _ => {}
            }

            // Check if the term's sort is a bitvector sort
            if let Some(sort) = manager.sorts.get(t.sort)
                && matches!(sort.kind, SortKind::BitVec(_))
            {
                return true;
            }
        }

        false
    }

    /// Project a formula over specific variables (theory-aware).
    ///
    /// For LIA this keeps conjuncts that mention only `vars_to_keep` (or are
    /// ground); arrays and bitvectors get their own theory-specific handling.
    ///
    /// This used to be mutually recursive with `project_not` (since inlined), and
    /// *both* directions consumed native stack while simultaneously
    /// allocating new terms at every level, so a deep formula exhausted the
    /// stack and the term manager together. The traversal is an explicit
    /// work stack now.
    ///
    /// The negation handling is unchanged in meaning: `project_not` never
    /// did anything but reduce `¬arg` to another `project_variables` call
    /// (`¬¬p → p`, De Morgan) or terminate on an atom, so it is expressed
    /// here as a rewrite loop over `current` rather than a second function.
    /// The array/bitvector checks are re-run after each such rewrite, which
    /// is exactly what re-entering `project_variables` used to do.
    pub fn project_variables(
        formula: TermId,
        vars_to_keep: &[TermId],
        manager: &mut TermManager,
    ) -> TermId {
        /// One step of the explicit projection stack.
        enum Work {
            /// Project this subformula.
            Eval(TermId),
            /// Combine the top `n` results into a conjunction, dropping any
            /// that mentions a variable outside `vars_to_keep`.
            Conjoin(usize),
            /// Combine the top `n` results into a disjunction.
            Disjoin(usize),
        }

        let mut work: Vec<Work> = vec![Work::Eval(formula)];
        let mut values: Vec<TermId> = Vec::new();

        while let Some(item) = work.pop() {
            match item {
                Work::Eval(start) => match Self::project_step(start, vars_to_keep, manager) {
                    ProjectStep::Done(result) => values.push(result),
                    ProjectStep::Conjuncts(args) => {
                        work.push(Work::Conjoin(args.len()));
                        for arg in args.into_iter().rev() {
                            work.push(Work::Eval(arg));
                        }
                    }
                    ProjectStep::Disjuncts(args) => {
                        work.push(Work::Disjoin(args.len()));
                        for arg in args.into_iter().rev() {
                            work.push(Work::Eval(arg));
                        }
                    }
                },
                Work::Conjoin(count) => {
                    let projected = values.split_off(values.len().saturating_sub(count));
                    let kept: Vec<TermId> = projected
                        .into_iter()
                        .filter(|&proj| {
                            Self::uses_only_vars(proj, vars_to_keep, manager)
                                || Self::is_ground_constraint(proj, manager)
                        })
                        .collect();
                    let combined = match kept.as_slice() {
                        [] => manager.mk_true(),
                        [only] => *only,
                        _ => manager.mk_and(kept),
                    };
                    values.push(combined);
                }
                Work::Disjoin(count) => {
                    let projected = values.split_off(values.len().saturating_sub(count));
                    let combined = manager.mk_or(projected);
                    values.push(combined);
                }
            }
        }

        // Exactly one value is produced for the root formula.
        values.pop().unwrap_or(formula)
    }

    /// Reduce `formula` until it is either fully projected or exposed as a
    /// conjunction/disjunction whose arguments must be projected first.
    ///
    /// The loop performs the negation pushing that
    /// `project_not` used to perform by re-entering `project_variables`:
    /// `¬¬p` becomes `p`, `¬(a ∧ b)` becomes `¬a ∨ ¬b`, `¬(a ∨ b)` becomes
    /// `¬a ∧ ¬b`, and a negated atom is projected like any other atom.
    ///
    /// Negation must *not* be handled by projecting `arg` and negating the
    /// result: `project_variables` guarantees `phi ⇒ psi(phi)`, and negating
    /// both sides yields an UNDER-approximation of `¬arg`, breaking the
    /// over-approximation contract a caller building a frame invariant
    /// relies on.
    fn project_step(
        formula: TermId,
        vars_to_keep: &[TermId],
        manager: &mut TermManager,
    ) -> ProjectStep {
        use oxiz_core::TermKind;

        let mut current = formula;
        loop {
            let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
                return ProjectStep::Done(current);
            };

            if Self::is_array_term(current, manager) {
                return ProjectStep::Done(Self::project_array_term(current, vars_to_keep, manager));
            }
            if Self::is_bitvector_term(current, manager) {
                return ProjectStep::Done(Self::project_bitvector_term(
                    current,
                    vars_to_keep,
                    manager,
                ));
            }

            match kind {
                TermKind::And(args) => return ProjectStep::Conjuncts(args.to_vec()),
                TermKind::Or(args) => return ProjectStep::Disjuncts(args.to_vec()),
                TermKind::Not(arg) => match manager.get(arg).map(|t| t.kind.clone()) {
                    // ¬¬p = p
                    Some(TermKind::Not(inner)) => current = inner,
                    // ¬(a ∧ b ∧ …) = ¬a ∨ ¬b ∨ …
                    Some(TermKind::And(args)) => {
                        let negated: Vec<TermId> =
                            args.iter().map(|&a| manager.mk_not(a)).collect();
                        current = manager.mk_or(negated);
                    }
                    // ¬(a ∨ b ∨ …) = ¬a ∧ ¬b ∧ …
                    Some(TermKind::Or(args)) => {
                        let negated: Vec<TermId> =
                            args.iter().map(|&a| manager.mk_not(a)).collect();
                        current = manager.mk_and(negated);
                    }
                    // Atomic (or unknown) formula under negation.
                    _ => {
                        let whole = manager.mk_not(arg);
                        return ProjectStep::Done(Self::project_atom(whole, vars_to_keep, manager));
                    }
                },
                _ => return ProjectStep::Done(Self::project_atom(current, vars_to_keep, manager)),
            }
        }
    }

    /// Keep an atomic formula verbatim if it only mentions `vars_to_keep`
    /// (or is ground), otherwise replace it by `true` — always a sound
    /// over-approximation of anything.
    fn project_atom(atom: TermId, vars_to_keep: &[TermId], manager: &mut TermManager) -> TermId {
        if Self::uses_only_vars(atom, vars_to_keep, manager)
            || Self::is_ground_constraint(atom, manager)
        {
            atom
        } else {
            manager.mk_true()
        }
    }

    /// Check that every variable occurring in `term` is one of `vars`.
    ///
    /// Walked iteratively with a visited set (see [`crate::walk`]). The old
    /// recursive form had three problems: unbounded stack depth on nested
    /// input, exponential re-expansion of a shared DAG, and a `_ => false`
    /// arm that reported *any* term built from an operator it did not
    /// enumerate (`Ite`, `Implies`, `Apply`, `Select`/`Store`, bitvector and
    /// string operations, …) as "uses other variables" even when it
    /// mentioned none at all. That last one is safe in direction — it only
    /// over-projects — but it silently disabled projection for whole
    /// theories. Descending uniformly answers the question exactly.
    ///
    /// A dangling id still answers `false`: nothing is known about it, and
    /// "may use other variables" is the conservative reading.
    fn uses_only_vars(term: TermId, vars: &[TermId], manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        !crate::walk::any_node(manager, term, |id, kind| match kind {
            Some(TermKind::Var(_)) => !vars.contains(&id),
            Some(_) => false,
            None => true,
        })
    }

    /// Check if a term is a ground constraint (no variables).
    ///
    /// Iterative walk with a visited set. As with [`Self::uses_only_vars`],
    /// the old recursive version's `_ => false` arm declared any term
    /// containing an operator outside its enumeration "not ground" even when
    /// it was variable-free; descending uniformly answers exactly. A
    /// dangling id answers `false` (unknown, so not provably ground).
    fn is_ground_constraint(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        !crate::walk::any_node(manager, term, |_, kind| match kind {
            Some(TermKind::Var(_)) | None => true,
            Some(_) => false,
        })
    }

    /// Project an array term over specific variables
    fn project_array_term(
        term: TermId,
        vars_to_keep: &[TermId],
        manager: &mut TermManager,
    ) -> TermId {
        use oxiz_core::TermKind;

        let Some(t) = manager.get(term) else {
            return term;
        };

        match &t.kind {
            TermKind::Select(array, index) => {
                // Keep select if array or index are in vars_to_keep
                if vars_to_keep.contains(array) || vars_to_keep.contains(index) {
                    term
                } else {
                    manager.mk_true() // Project out
                }
            }
            TermKind::Store(array, index, value) => {
                // Keep store if any component is in vars_to_keep
                if vars_to_keep.contains(array)
                    || vars_to_keep.contains(index)
                    || vars_to_keep.contains(value)
                {
                    term
                } else {
                    *array // Return base array, projecting out the store
                }
            }
            _ => {
                // For other array-typed terms, use default projection
                if Self::uses_only_vars(term, vars_to_keep, manager) {
                    term
                } else {
                    manager.mk_true()
                }
            }
        }
    }

    /// Project a bitvector term over specific variables
    fn project_bitvector_term(
        term: TermId,
        vars_to_keep: &[TermId],
        manager: &mut TermManager,
    ) -> TermId {
        use oxiz_core::TermKind;

        let Some(t) = manager.get(term) else {
            return term;
        };

        match &t.kind {
            // For bitvector operations, recursively project operands
            TermKind::BvAnd(a, b)
            | TermKind::BvOr(a, b)
            | TermKind::BvXor(a, b)
            | TermKind::BvAdd(a, b)
            | TermKind::BvSub(a, b)
            | TermKind::BvMul(a, b) => {
                let a_keep = vars_to_keep.contains(a) || Self::uses_vars(*a, vars_to_keep, manager);
                let b_keep = vars_to_keep.contains(b) || Self::uses_vars(*b, vars_to_keep, manager);

                if a_keep && b_keep {
                    term // Keep entire operation
                } else if a_keep {
                    *a // Project to just first operand
                } else if b_keep {
                    *b // Project to just second operand
                } else {
                    manager.mk_true() // Project out entirely
                }
            }
            TermKind::BvNot(arg) => {
                if vars_to_keep.contains(arg) || Self::uses_vars(*arg, vars_to_keep, manager) {
                    term
                } else {
                    manager.mk_true()
                }
            }
            TermKind::BvExtract { arg, .. } => {
                if vars_to_keep.contains(arg) || Self::uses_vars(*arg, vars_to_keep, manager) {
                    term
                } else {
                    manager.mk_true()
                }
            }
            _ => {
                // For other bitvector terms, use default projection
                if Self::uses_only_vars(term, vars_to_keep, manager) {
                    term
                } else {
                    manager.mk_true()
                }
            }
        }
    }

    /// Check if a term uses any of the specified variables.
    ///
    /// Iterative walk with a visited set. The old recursive form ended in
    /// `_ => false`, so an occurrence under any unenumerated operator was
    /// reported as absent; descending uniformly via
    /// [`oxiz_core::ast::traversal::get_children`] answers exactly. A
    /// dangling id answers `false`, matching the old behaviour (there is no
    /// variable to find in a term that does not exist).
    fn uses_vars(term: TermId, vars: &[TermId], manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        crate::walk::any_node(manager, term, |id, kind| {
            matches!(kind, Some(TermKind::Var(_))) && vars.contains(&id)
        })
    }

    /// Strengthen a lemma using theory-specific information
    pub fn theory_strengthen(
        lemma: TermId,
        _pred: PredId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        // Theory-specific lemma strengthening
        // For LIA: Add bounds, octagon constraints
        // For Arrays: Add array axioms, extensionality
        // For BV: Add bit-level constraints
        // For ADT: Add constructor constraints

        use oxiz_core::TermKind;

        // Enhanced: try to add theory-specific constraints for linear arithmetic
        if Self::is_linear_arithmetic(lemma, manager) {
            // Extract term kind and operands first before mutable borrow
            let term_info = manager.get(lemma).map(|t| t.kind.clone());

            let kind = term_info?;

            // For linear arithmetic, we can add implied bounds
            match kind {
                TermKind::Eq(a, b) => {
                    // x = y implies x <= y AND x >= y
                    let le = manager.mk_le(a, b);
                    let ge = manager.mk_ge(a, b);
                    Some(manager.mk_and(vec![lemma, le, ge]))
                }
                TermKind::Lt(a, b) => {
                    // x < y implies x <= y
                    let le = manager.mk_le(a, b);
                    Some(manager.mk_and(vec![lemma, le]))
                }
                TermKind::Gt(a, b) => {
                    // x > y implies x >= y
                    let ge = manager.mk_ge(a, b);
                    Some(manager.mk_and(vec![lemma, ge]))
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Extract theory-specific witnesses from a model
    #[allow(dead_code)]
    pub fn extract_witness(term: TermId, sort: SortId, manager: &TermManager) -> Option<Witness> {
        // Extract concrete values for different theories
        // For LIA: Extract integer/real values
        // For Arrays: Extract array contents as map
        // For BV: Extract bitvector values
        // For ADT: Extract constructor applications

        use oxiz_core::TermKind;

        let t = manager.get(term)?;

        // Enhanced: extract witnesses for basic theories
        if sort == manager.sorts.bool_sort {
            // Boolean witness
            match &t.kind {
                TermKind::True => Some(Witness::Bool(true)),
                TermKind::False => Some(Witness::Bool(false)),
                _ => None,
            }
        } else {
            // For integer/real/other theories, would need additional dependencies
            // or model extraction from solver
            // Placeholder: return None for now
            let _ = term; // Suppress warning
            None
        }
    }

    /// Generalize a cube using theory-specific techniques
    pub fn theory_generalize(cube: &[TermId], manager: &mut TermManager) -> SmallVec<[TermId; 8]> {
        // Theory-aware generalization
        // For LIA: Widen bounds, drop disjuncts, merge intervals
        // For Arrays: Generalize array properties
        // For BV: Generalize bit patterns
        // For ADT: Generalize constructor patterns

        use oxiz_core::TermKind;

        let mut generalized = SmallVec::new();

        // First pass: collect constraints and categorize them
        let mut arithmetic_constraints = Vec::new();
        let mut other_constraints = Vec::new();

        for &lit in cube {
            if Self::is_arithmetic_literal(lit, manager) {
                arithmetic_constraints.push(lit);
            } else {
                other_constraints.push(lit);
            }
        }

        // Enhanced arithmetic generalization
        for &lit in &arithmetic_constraints {
            let Some(kind) = manager.get(lit).map(|t| t.kind.clone()) else {
                generalized.push(lit);
                continue;
            };

            match kind {
                // For strict inequalities over integers, convert to the
                // exactly-equivalent non-strict form: `x < b` <=> `x <=
                // b-1` (and dually for `>`). This is *only* exact for
                // integers -- reals are dense, so `x < b` has no
                // non-strict equivalent there at all. The previous code
                // rewrote `x < b` to the *weaker* `x <= b` (dropping the
                // `-1`/`+1` entirely, contradicting its own comment) for
                // every sort including Real, which for Real additionally
                // fabricates a claimed-equivalent constraint that isn't
                // even approximately the same set (`x <= b` admits
                // `x = b`, which `x < b` explicitly excludes).
                TermKind::Lt(a, b) if Self::is_int_sorted(a, manager) => {
                    let one = manager.mk_int(1);
                    let b_minus_one = manager.mk_sub(b, one);
                    let le = manager.mk_le(a, b_minus_one);
                    generalized.push(le);
                }
                TermKind::Gt(a, b) if Self::is_int_sorted(a, manager) => {
                    let one = manager.mk_int(1);
                    let b_plus_one = manager.mk_add([b, one]);
                    let ge = manager.mk_ge(a, b_plus_one);
                    generalized.push(ge);
                }
                // Non-integer strict inequalities (e.g. Real): no exact
                // non-strict equivalent exists, so keep the literal as-is
                // rather than fabricate an incorrect one.
                TermKind::Lt(_, _) | TermKind::Gt(_, _) => {
                    generalized.push(lit);
                }
                // For equalities, try to weaken to interval constraints
                TermKind::Eq(a, b) if Self::can_weaken_equality(a, b, manager) => {
                    // x = c can be weakened to x >= c AND x <= c
                    // but we keep just the equality for precision
                    // A more aggressive generalization could drop the equality
                    generalized.push(lit);
                }
                // Keep bounds as-is (they're already general)
                TermKind::Le(_, _) | TermKind::Ge(_, _) => {
                    generalized.push(lit);
                }
                // For other arithmetic constraints
                _ => {
                    generalized.push(lit);
                }
            }
        }

        // Add non-arithmetic constraints unchanged
        generalized.extend(other_constraints);

        // Additional optimization: merge overlapping bounds
        Self::merge_arithmetic_bounds(&mut generalized, manager);

        generalized
    }

    /// Check if an equality can be safely weakened
    fn can_weaken_equality(a: TermId, b: TermId, manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        // Can weaken x = c where c is a constant
        let a_is_const = matches!(
            manager.get(a).map(|t| &t.kind),
            Some(TermKind::IntConst(_) | TermKind::RealConst(_))
        );
        let b_is_const = matches!(
            manager.get(b).map(|t| &t.kind),
            Some(TermKind::IntConst(_) | TermKind::RealConst(_))
        );

        a_is_const || b_is_const
    }

    /// Check whether `term` has `Int` sort -- the only sort for which a
    /// strict inequality has an exact non-strict equivalent.
    fn is_int_sorted(term: TermId, manager: &TermManager) -> bool {
        manager
            .get(term)
            .is_some_and(|t| t.sort == manager.sorts.int_sort)
    }

    /// Merge overlapping arithmetic bounds
    /// For example: x <= 5 AND x <= 10 becomes just x <= 5
    fn merge_arithmetic_bounds(
        constraints: &mut SmallVec<[TermId; 8]>,
        _manager: &mut TermManager,
    ) {
        // Advanced optimization: detect and merge redundant bounds
        // For now, this is a placeholder for future optimization
        // Full implementation would:
        // 1. Group constraints by variable
        // 2. Identify redundant bounds (x <= 5 subsumes x <= 10)
        // 3. Remove subsumed constraints

        // Placeholder: no merging yet, just return as-is
        // This prevents unnecessary code churn while keeping the structure
        let _ = constraints;
    }

    /// Check if a term is integer zero
    fn is_int_zero(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        manager
            .get(term)
            .is_some_and(|t| matches!(&t.kind, TermKind::IntConst(n) if n.to_string() == "0"))
    }

    /// Check if a term is integer one
    fn is_int_one(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        manager
            .get(term)
            .is_some_and(|t| matches!(&t.kind, TermKind::IntConst(n) if n.to_string() == "1"))
    }

    /// Simplify arithmetic expressions using theory-specific rules
    pub fn arithmetic_simplify(expr: TermId, manager: &mut TermManager) -> TermId {
        use oxiz_core::TermKind;

        let Some(term) = manager.get(expr) else {
            return expr;
        };

        match term.kind.clone() {
            // Simplify x + 0 to x
            TermKind::Add(args) => {
                let simplified_args: Vec<TermId> = args
                    .iter()
                    .filter(|&&arg| !Self::is_int_zero(arg, manager))
                    .copied()
                    .collect();

                if simplified_args.is_empty() {
                    manager.mk_int(0)
                } else if simplified_args.len() == 1 {
                    simplified_args[0]
                } else if simplified_args.len() < args.len() {
                    manager.mk_add(simplified_args)
                } else {
                    expr
                }
            }
            // Simplify x * 1 to x
            TermKind::Mul(args) => {
                let has_zero = args.iter().any(|&arg| Self::is_int_zero(arg, manager));

                if has_zero {
                    return manager.mk_int(0);
                }

                let simplified_args: Vec<TermId> = args
                    .iter()
                    .filter(|&&arg| !Self::is_int_one(arg, manager))
                    .copied()
                    .collect();

                if simplified_args.is_empty() {
                    manager.mk_int(1)
                } else if simplified_args.len() == 1 {
                    simplified_args[0]
                } else if simplified_args.len() < args.len() {
                    manager.mk_mul(simplified_args)
                } else {
                    expr
                }
            }
            // x - 0 = x
            TermKind::Sub(a, b) => {
                if Self::is_int_zero(b, manager) {
                    a
                } else {
                    expr
                }
            }
            _ => expr,
        }
    }
}

impl Default for TheoryIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// A concrete witness value from the model
#[derive(Debug, Clone)]
pub enum Witness {
    /// Integer value
    Int(i64),
    /// Real value (as rational)
    Real(i64, u64), // numerator, denominator
    /// Boolean value
    Bool(bool),
    /// Array value (map from indices to elements)
    Array(SmallVec<[(Box<Witness>, Box<Witness>); 4]>, Box<Witness>), // entries + default
    /// Bitvector value
    BitVector(u64, u32), // value, width
    /// Constructor application
    Constructor(String, SmallVec<[Box<Witness>; 4]>), // name, arguments
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stack size and nesting depth for the deep-recursion test below.
    ///
    /// The two are scaled together on purpose: what the test actually pins is
    /// the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). Natively recursive projection needs far more than
    /// that per frame and still overflows, so the regression keeps every bit
    /// of its detection power. The pair used to be 1 MiB / 50_000 -- the same
    /// 21 bytes. `mk_and` flattens its arguments (so `acc = mk_and([acc,
    /// atom])` never actually nests, and is quadratic to boot), so the deep
    /// term below is built with `TermManager::intern_term` directly, which
    /// neither flattens nor re-interns already-built prefixes. Never raise
    /// `DEEP_DEPTH` without raising `DEEP_STACK` by the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

    #[test]
    fn test_theory_integration_creation() {
        let theory = TheoryIntegration::new();
        let _ = theory; // Just check it compiles
    }

    #[test]
    fn test_is_linear_arithmetic() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);

        assert!(TheoryIntegration::is_linear_arithmetic(x, &manager));

        let y = manager.mk_var("y", manager.sorts.real_sort);
        assert!(TheoryIntegration::is_linear_arithmetic(y, &manager));

        let b = manager.mk_var("b", manager.sorts.bool_sort);
        assert!(!TheoryIntegration::is_linear_arithmetic(b, &manager));
    }

    #[test]
    fn test_project_variables() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let formula = manager.mk_eq(x, zero);

        let projected = TheoryIntegration::project_variables(formula, &[x], &mut manager);
        assert_eq!(projected, formula); // Placeholder returns formula as-is
    }

    #[test]
    fn test_theory_generalize() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let cube = [x];

        let generalized = TheoryIntegration::theory_generalize(&cube, &mut manager);
        assert_eq!(generalized.len(), 1);
        assert_eq!(generalized[0], x);
    }

    // -----------------------------------------------------------------------
    // Regression tests for the `sweep-backend-misc` triage sweep.
    // -----------------------------------------------------------------------

    /// `theory_generalize` used to rewrite `x < b` to `x <= b` (dropping
    /// the `-1` its own comment claimed to apply), for every sort
    /// including Real. This verifies the Int case now produces the truly
    /// equivalent `x <= b - 1`.
    #[test]
    fn test_theory_generalize_int_lt_becomes_le_b_minus_one() {
        use oxiz_core::TermKind;

        let mut manager = TermManager::new();
        let x = manager.mk_var("gen_int_x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(x, ten);

        let generalized = TheoryIntegration::theory_generalize(&[lt], &mut manager);
        assert_eq!(generalized.len(), 1);

        let one = manager.mk_int(1);
        let expected_rhs = manager.mk_sub(ten, one);
        let expected = manager.mk_le(x, expected_rhs);
        assert_eq!(
            generalized[0], expected,
            "x < 10 (Int) must generalize to exactly x <= 10 - 1, not x <= 10"
        );

        let result_kind = manager
            .get(generalized[0])
            .expect("term exists")
            .kind
            .clone();
        assert!(
            matches!(result_kind, TermKind::Le(a, b) if a == x && b == expected_rhs),
            "unexpected structure: {result_kind:?}"
        );
    }

    /// Dual check for `>`: `x > b` must become `x >= b + 1` for Int, not
    /// the bare (weaker, off-by-one) `x >= b`.
    #[test]
    fn test_theory_generalize_int_gt_becomes_ge_b_plus_one() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("gen_int_x2", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let gt = manager.mk_gt(x, ten);

        let generalized = TheoryIntegration::theory_generalize(&[gt], &mut manager);
        assert_eq!(generalized.len(), 1);

        let one = manager.mk_int(1);
        let expected_rhs = manager.mk_add([ten, one]);
        let expected = manager.mk_ge(x, expected_rhs);
        assert_eq!(
            generalized[0], expected,
            "x > 10 (Int) must generalize to exactly x >= 10 + 1, not x >= 10"
        );
    }

    /// For non-integer sorts (Real is dense: there is no exact non-strict
    /// equivalent to a strict inequality), the strict literal must be
    /// left unchanged rather than fabricate an incorrect `x <= b` that
    /// admits `x = b`.
    #[test]
    fn test_theory_generalize_real_lt_left_unchanged() {
        use num_rational::Rational64;

        let mut manager = TermManager::new();
        let x = manager.mk_var("gen_real_x", manager.sorts.real_sort);
        let ten_half = manager.mk_real(Rational64::new(21, 2));
        let lt = manager.mk_lt(x, ten_half);

        let generalized = TheoryIntegration::theory_generalize(&[lt], &mut manager);
        assert_eq!(generalized.len(), 1);
        assert_eq!(
            generalized[0], lt,
            "a Real strict inequality has no exact non-strict equivalent \
             and must be left unchanged, not rewritten to an incorrect x <= b"
        );
    }

    /// `project_variables` used to recurse through `Not` by projecting
    /// the inner formula (an over-approximation, `arg => psi(arg)`) and
    /// then negating the result -- which produces `¬psi(arg) => ¬arg`,
    /// an UNDER-approximation of `¬arg`, not the over-approximation this
    /// function's whole contract promises. Verify a `Not`-wrapped
    /// formula whose inner conjunction mixes a kept and a dropped
    /// variable is over-approximated soundly: since the whole `¬(x=0 ∧
    /// y=0)` mentions the dropped variable `y`, the only sound result
    /// (given this function's atomic-fallback strategy) is `true`, never
    /// a term that excludes states satisfying the original formula.
    #[test]
    fn test_project_variables_not_is_sound_over_approximation() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("proj_not_x", manager.sorts.int_sort);
        let y = manager.mk_var("proj_not_y", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let x_eq_0 = manager.mk_eq(x, zero);
        let y_eq_0 = manager.mk_eq(y, zero);
        let conj = manager.mk_and([x_eq_0, y_eq_0]);
        let formula = manager.mk_not(conj);

        // Old (buggy) behavior would recurse into `project_variables(x=0
        // ∧ y=0)`, which drops the `y=0` conjunct (since `y` is not
        // kept) yielding `x=0`, then negate it to `¬(x=0)` -- but `¬(x=0
        // ∧ y=0)` does NOT imply `¬(x=0)` (e.g. x=0, y=1 satisfies the
        // former and violates the latter), so that would be unsound.
        let projected = TheoryIntegration::project_variables(formula, &[x], &mut manager);

        let unsound_result = manager.mk_not(x_eq_0);
        assert_ne!(
            projected, unsound_result,
            "must not produce the unsound ¬(x=0) via naive Not-recursion"
        );
        assert_eq!(
            projected,
            manager.mk_true(),
            "the whole ¬(x=0 ∧ y=0) mentions the dropped variable y, so \
             the only sound projection is `true`"
        );
    }

    /// Double negation must simplify soundly rather than trigger the
    /// same under-approximation bug one level deeper.
    #[test]
    fn test_project_variables_double_negation_sound() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("proj_dneg_x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let x_eq_0 = manager.mk_eq(x, zero);
        let not_x_eq_0 = manager.mk_not(x_eq_0);
        let not_not = manager.mk_not(not_x_eq_0);

        let projected = TheoryIntegration::project_variables(not_not, &[x], &mut manager);
        assert_eq!(
            projected, x_eq_0,
            "¬¬(x=0) over only kept variables must simplify exactly to x=0"
        );
    }

    /// Deeply nested conjunction/negation must not overflow: this used to be
    /// mutual recursion between `project_variables` and `project_not`.
    #[test]
    fn project_variables_survives_deep_nesting() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                use oxiz_core::TermKind;

                let mut manager = TermManager::new();
                let bool_sort = manager.sorts.bool_sort;
                let int_sort = manager.sorts.int_sort;
                let x = manager.mk_var("x", int_sort);
                let zero = manager.mk_int(0);
                let mut formula = manager.mk_ge(x, zero);
                // Interned directly (not via `mk_and`, which would flatten
                // this back into one wide `And`) so the tree really is
                // `DEEP_DEPTH` deep.
                for i in 0..DEEP_DEPTH {
                    let v = manager.mk_var(&format!("v{i}"), int_sort);
                    let atom = manager.mk_ge(v, zero);
                    formula =
                        manager.intern_term(TermKind::And(vec![formula, atom].into()), bool_sort);
                }
                let projected = TheoryIntegration::project_variables(formula, &[x], &mut manager);
                assert!(
                    manager.get(projected).is_some(),
                    "deep projection must return a term"
                );
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep project_variables must return");
    }

    /// `uses_only_vars` must be exact rather than answering "no" for every
    /// operator outside a short enumeration.
    #[test]
    fn uses_only_vars_accepts_unenumerated_operators() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let cond = manager.mk_var("c", bool_sort);
        // `Ite(c, x, y)` mentions exactly {c, x, y}.
        let ite = manager.mk_ite(cond, x, y);
        assert!(
            TheoryIntegration::uses_only_vars(ite, &[cond, x, y], &manager),
            "an Ite over kept variables uses only kept variables"
        );
        assert!(
            !TheoryIntegration::uses_only_vars(ite, &[cond, x], &manager),
            "dropping `y` from the kept set must make the answer `false`"
        );
    }

    /// `is_ground_constraint` must recognise a variable-free term regardless
    /// of which operators build it, and reject one containing a variable.
    #[test]
    fn is_ground_constraint_is_exact() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let sum = manager.mk_add([one, two]);
        let ground = manager.mk_le(sum, two);
        assert!(TheoryIntegration::is_ground_constraint(ground, &manager));

        let x = manager.mk_var("x", int_sort);
        let non_ground = manager.mk_le(x, two);
        assert!(!TheoryIntegration::is_ground_constraint(
            non_ground, &manager
        ));
    }

    /// `uses_vars` finds an occurrence under an operator the old walk had no
    /// arm for, and reports absence correctly.
    #[test]
    fn uses_vars_sees_through_ite() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let cond = manager.mk_var("c", bool_sort);
        let ite = manager.mk_ite(cond, x, y);
        assert!(TheoryIntegration::uses_vars(ite, &[y], &manager));
        let z = manager.mk_var("z", int_sort);
        assert!(!TheoryIntegration::uses_vars(ite, &[z], &manager));
    }
}
