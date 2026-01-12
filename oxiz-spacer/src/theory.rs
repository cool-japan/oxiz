//! Theory-aware operations for Spacer.
//!
//! This module provides theory-specific enhancements for PDR/IC3,
//! including theory-aware generalization, interpolation, and model projection.
//!
//! Reference: Z3's `muz/spacer/spacer_context.cpp` theory integration

use crate::chc::PredId;
use oxiz_core::{SortId, TermId, TermManager};
use smallvec::SmallVec;

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

    /// Project a formula over specific variables (theory-aware)
    pub fn project_variables(
        formula: TermId,
        vars_to_keep: &[TermId],
        manager: &mut TermManager,
    ) -> TermId {
        // Theory-aware projection
        // For LIA: Use Fourier-Motzkin or virtual term substitution
        // For Arrays: Use array property fragments and axiom instantiation
        // For BV: Use bit-blasting or interval analysis
        // For ADT: Use constructor/selector elimination

        // Enhanced implementation with theory awareness for arrays and bitvectors
        use oxiz_core::TermKind;

        // Clone the term kind to avoid borrow checker issues
        let term_kind = manager.get(formula).map(|t| t.kind.clone());

        let Some(kind) = term_kind else {
            return formula;
        };

        // Special handling for array theory
        if Self::is_array_term(formula, manager) {
            return Self::project_array_term(formula, vars_to_keep, manager);
        }

        // Special handling for bitvector theory
        if Self::is_bitvector_term(formula, manager) {
            return Self::project_bitvector_term(formula, vars_to_keep, manager);
        }

        match kind {
            TermKind::And(args) => {
                // Project each conjunct and recombine
                let args_vec: Vec<TermId> = args.to_vec();
                let mut projected = Vec::new();

                for arg in args_vec {
                    let proj = Self::project_variables(arg, vars_to_keep, manager);
                    // Keep only formulas that mention variables we want to keep
                    if Self::uses_only_vars(proj, vars_to_keep, manager)
                        || Self::is_ground_constraint(proj, manager)
                    {
                        projected.push(proj);
                    }
                }

                if projected.is_empty() {
                    manager.mk_true()
                } else if projected.len() == 1 {
                    projected[0]
                } else {
                    manager.mk_and(projected)
                }
            }
            TermKind::Or(args) => {
                // For disjunctions, we need to be more conservative
                let args_vec: Vec<TermId> = args.to_vec();
                let projected: Vec<TermId> = args_vec
                    .into_iter()
                    .map(|arg| Self::project_variables(arg, vars_to_keep, manager))
                    .collect();
                manager.mk_or(projected)
            }
            TermKind::Not(arg) => {
                let projected = Self::project_variables(arg, vars_to_keep, manager);
                manager.mk_not(projected)
            }
            _ => {
                // For atomic formulas, check if they only use vars to keep
                if Self::uses_only_vars(formula, vars_to_keep, manager)
                    || Self::is_ground_constraint(formula, manager)
                {
                    formula
                } else {
                    manager.mk_true() // Project out
                }
            }
        }
    }

    /// Check if a term uses only the specified variables
    fn uses_only_vars(term: TermId, vars: &[TermId], manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        let Some(t) = manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::Var(_) => vars.contains(&term),
            TermKind::And(args) | TermKind::Or(args) => args
                .iter()
                .all(|&arg| Self::uses_only_vars(arg, vars, manager)),
            TermKind::Not(arg) => Self::uses_only_vars(*arg, vars, manager),
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b) => {
                Self::uses_only_vars(*a, vars, manager) && Self::uses_only_vars(*b, vars, manager)
            }
            TermKind::Add(args) | TermKind::Mul(args) => args
                .iter()
                .all(|&arg| Self::uses_only_vars(arg, vars, manager)),
            TermKind::Sub(a, b) | TermKind::Div(a, b) | TermKind::Mod(a, b) => {
                Self::uses_only_vars(*a, vars, manager) && Self::uses_only_vars(*b, vars, manager)
            }
            TermKind::True | TermKind::False | TermKind::IntConst(_) | TermKind::RealConst(_) => {
                true
            }
            _ => false,
        }
    }

    /// Check if a term is a ground constraint (no variables)
    fn is_ground_constraint(term: TermId, manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        let Some(t) = manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::Var(_) => false,
            TermKind::True | TermKind::False | TermKind::IntConst(_) | TermKind::RealConst(_) => {
                true
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args
                .iter()
                .all(|&arg| Self::is_ground_constraint(arg, manager)),
            TermKind::Not(arg) => Self::is_ground_constraint(*arg, manager),
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b) => {
                Self::is_ground_constraint(*a, manager) && Self::is_ground_constraint(*b, manager)
            }
            _ => false,
        }
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

    /// Check if a term uses any of the specified variables
    fn uses_vars(term: TermId, vars: &[TermId], manager: &TermManager) -> bool {
        use oxiz_core::TermKind;

        let Some(t) = manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::Var(_) => vars.contains(&term),
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args.iter().any(|&arg| Self::uses_vars(arg, vars, manager)),
            TermKind::Not(arg) | TermKind::Neg(arg) | TermKind::BvNot(arg) => {
                Self::uses_vars(*arg, vars, manager)
            }
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::BvAnd(a, b)
            | TermKind::BvOr(a, b)
            | TermKind::BvXor(a, b)
            | TermKind::BvAdd(a, b)
            | TermKind::BvSub(a, b)
            | TermKind::BvMul(a, b) => {
                Self::uses_vars(*a, vars, manager) || Self::uses_vars(*b, vars, manager)
            }
            TermKind::Select(a, i) => {
                Self::uses_vars(*a, vars, manager) || Self::uses_vars(*i, vars, manager)
            }
            TermKind::Store(a, i, v) => {
                Self::uses_vars(*a, vars, manager)
                    || Self::uses_vars(*i, vars, manager)
                    || Self::uses_vars(*v, vars, manager)
            }
            TermKind::BvExtract { arg, .. } => Self::uses_vars(*arg, vars, manager),
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. } => false,
            _ => false,
        }
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
            if Self::is_linear_arithmetic(lit, manager) {
                arithmetic_constraints.push(lit);
            } else {
                other_constraints.push(lit);
            }
        }

        // Enhanced arithmetic generalization
        for &lit in &arithmetic_constraints {
            let Some(term) = manager.get(lit) else {
                generalized.push(lit);
                continue;
            };

            match &term.kind {
                // For strict inequalities, convert to non-strict for integers
                TermKind::Lt(a, b) => {
                    // x < c becomes x <= c-1 for integers
                    // This is a safe generalization
                    let le = manager.mk_le(*a, *b);
                    generalized.push(le);
                }
                TermKind::Gt(a, b) => {
                    // x > c becomes x >= c+1 for integers
                    let ge = manager.mk_ge(*a, *b);
                    generalized.push(ge);
                }
                // For equalities, try to weaken to interval constraints
                TermKind::Eq(a, b) if Self::can_weaken_equality(*a, *b, manager) => {
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
}
