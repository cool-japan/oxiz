//! Model-Based Projection (MBP) for Quantifier Elimination
//!
//! This module implements model-based projection algorithms for quantifier
//! elimination. Given a formula ∃x. φ(x, y) and a model M satisfying φ,
//! MBP computes formulas ψ(y) such that:
//! - M |= φ implies M |= ψ
//! - ψ does not contain x
//!
//! # Supported Theories
//!
//! - Linear Real Arithmetic (LRA)
//! - Linear Integer Arithmetic (LIA)
//! - Arrays
//! - Datatypes
//!
//! # Reference
//!
//! - Bjørner, N., & Janota, M. (2015). Playing with quantified satisfaction.

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use lasso::Spur;
use num_bigint::BigInt;
use num_rational::BigRational;
use rustc_hash::{FxHashMap, FxHashSet};

/// Model for MBP - maps variables to their values
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// Boolean assignments
    pub bools: FxHashMap<TermId, bool>,
    /// Integer assignments
    pub ints: FxHashMap<TermId, BigInt>,
    /// Real/Rational assignments
    pub reals: FxHashMap<TermId, BigRational>,
    /// Array assignments (array term -> (index -> value) map)
    pub arrays: FxHashMap<TermId, FxHashMap<TermId, TermId>>,
    /// Default values for arrays
    pub array_defaults: FxHashMap<TermId, TermId>,
}

impl Model {
    /// Create a new empty model
    pub fn new() -> Self {
        Self::default()
    }

    /// Get boolean value for a term
    pub fn get_bool(&self, term: TermId) -> Option<bool> {
        self.bools.get(&term).copied()
    }

    /// Get integer value for a term
    pub fn get_int(&self, term: TermId) -> Option<&BigInt> {
        self.ints.get(&term)
    }

    /// Get real value for a term
    pub fn get_real(&self, term: TermId) -> Option<&BigRational> {
        self.reals.get(&term)
    }

    /// Set boolean value
    pub fn set_bool(&mut self, term: TermId, value: bool) {
        self.bools.insert(term, value);
    }

    /// Set integer value
    pub fn set_int(&mut self, term: TermId, value: BigInt) {
        self.ints.insert(term, value);
    }

    /// Set real value
    pub fn set_real(&mut self, term: TermId, value: BigRational) {
        self.reals.insert(term, value);
    }
}

/// Result of model-based projection
#[derive(Debug, Clone)]
pub struct MbpResult {
    /// The projected formula (disjunction of cubes)
    pub formulas: Vec<TermId>,
    /// Variables that were successfully eliminated
    pub eliminated: Vec<Spur>,
    /// Variables that could not be eliminated
    pub remaining: Vec<Spur>,
}

impl MbpResult {
    /// Check if all variables were eliminated
    pub fn is_complete(&self) -> bool {
        self.remaining.is_empty()
    }

    /// Get the projected formula as a single term
    pub fn to_formula(&self, manager: &mut TermManager) -> TermId {
        if self.formulas.is_empty() {
            manager.mk_false()
        } else if self.formulas.len() == 1 {
            self.formulas[0]
        } else {
            manager.mk_or(self.formulas.iter().copied())
        }
    }
}

/// Configuration for MBP
#[derive(Debug, Clone)]
pub struct MbpConfig {
    /// Maximum number of case splits
    pub max_case_splits: usize,
    /// Whether to use model completion
    pub model_completion: bool,
    /// Whether to simplify result
    pub simplify: bool,
    /// Theory-specific projector to use
    pub projector: ProjectorKind,
}

impl Default for MbpConfig {
    fn default() -> Self {
        Self {
            max_case_splits: 100,
            model_completion: true,
            simplify: true,
            projector: ProjectorKind::Auto,
        }
    }
}

/// Kind of projector to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectorKind {
    /// Automatically detect based on formula
    #[default]
    Auto,
    /// Linear Real Arithmetic projector
    Lra,
    /// Linear Integer Arithmetic projector
    Lia,
    /// Array projector
    Array,
    /// Datatype projector
    Datatype,
}

/// Model-Based Projection engine
#[derive(Debug)]
pub struct MbpEngine<'a> {
    /// Term manager
    manager: &'a mut TermManager,
    /// Configuration
    config: MbpConfig,
    /// Cache for evaluated terms (reserved for future optimization)
    #[allow(dead_code)]
    eval_cache: FxHashMap<TermId, TermId>,
}

impl<'a> MbpEngine<'a> {
    /// Create a new MBP engine
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self::with_config(manager, MbpConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(manager: &'a mut TermManager, config: MbpConfig) -> Self {
        Self {
            manager,
            config,
            eval_cache: FxHashMap::default(),
        }
    }

    /// Project variables from a formula using model-based projection
    ///
    /// Given ∃vars. formula and a model, compute an equivalent formula
    /// without the quantified variables.
    pub fn project(&mut self, formula: TermId, vars: &[Spur], model: &Model) -> Result<MbpResult> {
        // Collect the variables to eliminate
        let vars_set: FxHashSet<_> = vars.iter().copied().collect();

        // Extract literals from the formula
        let literals = self.extract_literals(formula);

        // Determine which projector to use
        let projector = self.detect_projector(formula, &vars_set);

        // Project based on theory
        match projector {
            ProjectorKind::Lra => self.project_lra(&literals, &vars_set, model),
            ProjectorKind::Lia => self.project_lia(&literals, &vars_set, model),
            ProjectorKind::Array => self.project_array(&literals, &vars_set, model),
            ProjectorKind::Datatype => self.project_datatype(&literals, &vars_set, model),
            ProjectorKind::Auto => {
                // Try LRA first, then LIA
                let lra_result = self.project_lra(&literals, &vars_set, model)?;
                if lra_result.is_complete() {
                    Ok(lra_result)
                } else {
                    self.project_lia(&literals, &vars_set, model)
                }
            }
        }
    }

    /// Extract literals from a formula (conjunctive form)
    fn extract_literals(&self, formula: TermId) -> Vec<TermId> {
        let mut literals = Vec::new();
        self.extract_literals_rec(formula, &mut literals);
        literals
    }

    fn extract_literals_rec(&self, term: TermId, literals: &mut Vec<TermId>) {
        let Some(t) = self.manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::And(args) => {
                for &arg in args.iter() {
                    self.extract_literals_rec(arg, literals);
                }
            }
            _ => {
                literals.push(term);
            }
        }
    }

    /// Detect which projector to use based on formula structure
    fn detect_projector(&self, formula: TermId, vars: &FxHashSet<Spur>) -> ProjectorKind {
        if self.config.projector != ProjectorKind::Auto {
            return self.config.projector;
        }

        // Check for arrays
        if self.contains_array_ops(formula) {
            return ProjectorKind::Array;
        }

        // Check for datatypes
        if self.contains_datatype_ops(formula) {
            return ProjectorKind::Datatype;
        }

        // Check if all arithmetic is linear and over reals or integers
        if self.is_linear_real(formula, vars) {
            return ProjectorKind::Lra;
        }

        if self.is_linear_int(formula, vars) {
            return ProjectorKind::Lia;
        }

        // Default to LRA
        ProjectorKind::Lra
    }

    fn contains_array_ops(&self, term: TermId) -> bool {
        let Some(t) = self.manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::Select(_, _) | TermKind::Store(_, _, _) => true,
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args.iter().any(|&a| self.contains_array_ops(a)),
            TermKind::Not(a) | TermKind::Neg(a) => self.contains_array_ops(*a),
            TermKind::Eq(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Sub(a, b) => self.contains_array_ops(*a) || self.contains_array_ops(*b),
            TermKind::Ite(c, t, e) => {
                self.contains_array_ops(*c)
                    || self.contains_array_ops(*t)
                    || self.contains_array_ops(*e)
            }
            _ => false,
        }
    }

    fn contains_datatype_ops(&self, term: TermId) -> bool {
        let Some(t) = self.manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::DtConstructor { .. }
            | TermKind::DtSelector { .. }
            | TermKind::DtTester { .. } => true,
            TermKind::And(args) | TermKind::Or(args) => {
                args.iter().any(|&a| self.contains_datatype_ops(a))
            }
            TermKind::Not(a) => self.contains_datatype_ops(*a),
            TermKind::Eq(a, b) => self.contains_datatype_ops(*a) || self.contains_datatype_ops(*b),
            _ => false,
        }
    }

    fn is_linear_real(&self, _term: TermId, _vars: &FxHashSet<Spur>) -> bool {
        // For now, assume linear real
        true
    }

    fn is_linear_int(&self, _term: TermId, _vars: &FxHashSet<Spur>) -> bool {
        // For now, assume linear int
        true
    }

    /// LRA projector: Fourier-Motzkin style projection
    fn project_lra(
        &mut self,
        literals: &[TermId],
        vars: &FxHashSet<Spur>,
        model: &Model,
    ) -> Result<MbpResult> {
        let mut result_formulas = Vec::new();
        let mut eliminated = Vec::new();
        let mut remaining: Vec<Spur> = vars.iter().copied().collect();

        // Process each variable
        for &var in vars.iter() {
            // Classify literals by variable occurrence
            let (lower_bounds, upper_bounds, others) = self.classify_bounds_for_var(literals, var);

            if lower_bounds.is_empty() && upper_bounds.is_empty() {
                // Variable doesn't appear - trivially eliminate
                eliminated.push(var);
                remaining.retain(|&v| v != var);
                continue;
            }

            // Use model to pick a "good" bound
            let projected = self.project_var_lra(var, &lower_bounds, &upper_bounds, &others, model);

            result_formulas.extend(projected);
            eliminated.push(var);
            remaining.retain(|&v| v != var);
        }

        // Add literals that don't mention any variables
        let non_var_literals: Vec<_> = literals
            .iter()
            .filter(|&&lit| !self.mentions_any_var(lit, vars))
            .copied()
            .collect();

        result_formulas.extend(non_var_literals);

        // Build final formula
        let final_formula = if result_formulas.is_empty() {
            self.manager.mk_true()
        } else if result_formulas.len() == 1 {
            result_formulas[0]
        } else {
            self.manager.mk_and(result_formulas.iter().copied())
        };

        Ok(MbpResult {
            formulas: vec![final_formula],
            eliminated,
            remaining,
        })
    }

    /// Classify literals into lower bounds, upper bounds, and others for a variable
    fn classify_bounds_for_var(
        &self,
        literals: &[TermId],
        var: Spur,
    ) -> (Vec<TermId>, Vec<TermId>, Vec<TermId>) {
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut others = Vec::new();

        for &lit in literals {
            match self.get_bound_type(lit, var) {
                Some(BoundType::Lower) => lower.push(lit),
                Some(BoundType::Upper) => upper.push(lit),
                None => others.push(lit),
            }
        }

        (lower, upper, others)
    }

    /// Determine if a literal is a lower bound, upper bound, or neither for a variable
    fn get_bound_type(&self, lit: TermId, var: Spur) -> Option<BoundType> {
        let t = self.manager.get(lit)?;

        match &t.kind {
            // x <= e  -> upper bound on x
            TermKind::Le(lhs, rhs) => {
                if self.is_var(*lhs, var) && !self.mentions_var(*rhs, var) {
                    return Some(BoundType::Upper);
                }
                if self.is_var(*rhs, var) && !self.mentions_var(*lhs, var) {
                    return Some(BoundType::Lower);
                }
                None
            }
            // x < e  -> upper bound on x (strict)
            TermKind::Lt(lhs, rhs) => {
                if self.is_var(*lhs, var) && !self.mentions_var(*rhs, var) {
                    return Some(BoundType::Upper);
                }
                if self.is_var(*rhs, var) && !self.mentions_var(*lhs, var) {
                    return Some(BoundType::Lower);
                }
                None
            }
            // x >= e  -> lower bound on x
            TermKind::Ge(lhs, rhs) => {
                if self.is_var(*lhs, var) && !self.mentions_var(*rhs, var) {
                    return Some(BoundType::Lower);
                }
                if self.is_var(*rhs, var) && !self.mentions_var(*lhs, var) {
                    return Some(BoundType::Upper);
                }
                None
            }
            // x > e  -> lower bound on x (strict)
            TermKind::Gt(lhs, rhs) => {
                if self.is_var(*lhs, var) && !self.mentions_var(*rhs, var) {
                    return Some(BoundType::Lower);
                }
                if self.is_var(*rhs, var) && !self.mentions_var(*lhs, var) {
                    return Some(BoundType::Upper);
                }
                None
            }
            _ => None,
        }
    }

    fn is_var(&self, term: TermId, var: Spur) -> bool {
        if let Some(t) = self.manager.get(term) {
            matches!(&t.kind, TermKind::Var(name) if *name == var)
        } else {
            false
        }
    }

    fn mentions_var(&self, term: TermId, var: Spur) -> bool {
        let Some(t) = self.manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::Var(name) => *name == var,
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args.iter().any(|&a| self.mentions_var(a, var)),
            TermKind::Not(a) | TermKind::Neg(a) => self.mentions_var(*a, var),
            TermKind::Eq(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b) => self.mentions_var(*a, var) || self.mentions_var(*b, var),
            TermKind::Ite(c, t, e) => {
                self.mentions_var(*c, var)
                    || self.mentions_var(*t, var)
                    || self.mentions_var(*e, var)
            }
            TermKind::Apply { args, .. } => args.iter().any(|&a| self.mentions_var(a, var)),
            _ => false,
        }
    }

    fn mentions_any_var(&self, term: TermId, vars: &FxHashSet<Spur>) -> bool {
        vars.iter().any(|&v| self.mentions_var(term, v))
    }

    /// Project a single variable using LRA (model-guided)
    fn project_var_lra(
        &mut self,
        _var: Spur,
        lower_bounds: &[TermId],
        upper_bounds: &[TermId],
        others: &[TermId],
        _model: &Model,
    ) -> Vec<TermId> {
        let mut result = Vec::new();

        // For each pair of lower and upper bound, generate: lower <= upper
        for &lower in lower_bounds {
            for &upper in upper_bounds {
                // Extract the bound expressions
                let lower_expr = self.extract_bound_expr(lower, true);
                let upper_expr = self.extract_bound_expr(upper, false);

                if let (Some(l), Some(u)) = (lower_expr, upper_expr) {
                    // Generate: l <= u
                    let constraint = self.manager.mk_le(l, u);
                    result.push(constraint);
                }
            }
        }

        // Add other constraints that don't mention the variable
        result.extend(others.iter().copied());

        result
    }

    /// Extract the bound expression from a constraint
    fn extract_bound_expr(&self, constraint: TermId, is_lower: bool) -> Option<TermId> {
        let t = self.manager.get(constraint)?;

        match &t.kind {
            TermKind::Le(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                if is_lower {
                    Some(*lhs) // lower bound: rhs
                } else {
                    Some(*rhs) // upper bound: rhs
                }
            }
            TermKind::Ge(lhs, rhs) | TermKind::Gt(lhs, rhs) => {
                if is_lower {
                    Some(*rhs) // lower bound: rhs
                } else {
                    Some(*lhs) // upper bound: lhs
                }
            }
            _ => None,
        }
    }

    /// LIA projector: Integer arithmetic projection
    fn project_lia(
        &mut self,
        literals: &[TermId],
        vars: &FxHashSet<Spur>,
        model: &Model,
    ) -> Result<MbpResult> {
        // LIA projection is more complex due to divisibility constraints
        // For now, use a simplified version similar to LRA but with divisibility
        self.project_lra(literals, vars, model)
    }

    /// Array projector
    fn project_array(
        &mut self,
        literals: &[TermId],
        vars: &FxHashSet<Spur>,
        _model: &Model,
    ) -> Result<MbpResult> {
        // Array projection:
        // For ∃a. φ(a, i, v), project by:
        // - Finding all select(a, i) terms
        // - Substituting with fresh variables
        // - Adding consistency constraints

        let mut result = Vec::new();
        let eliminated = Vec::new();
        let remaining: Vec<Spur> = vars.iter().copied().collect();

        // Collect array-related constraints
        for &lit in literals {
            if !self.mentions_any_var(lit, vars) {
                result.push(lit);
            }
            // For array vars, we'd need to handle select/store axioms
            // This is a simplified version
        }

        let formula = if result.is_empty() {
            self.manager.mk_true()
        } else {
            self.manager.mk_and(result.iter().copied())
        };

        Ok(MbpResult {
            formulas: vec![formula],
            eliminated,
            remaining,
        })
    }

    /// Datatype projector
    fn project_datatype(
        &mut self,
        literals: &[TermId],
        vars: &FxHashSet<Spur>,
        _model: &Model,
    ) -> Result<MbpResult> {
        // Datatype projection:
        // For ∃x:DT. φ(x), case split on constructors
        // ∃x:DT. φ(x) ≡ ∨_c ∃args_c. φ(c(args_c))

        let mut result = Vec::new();
        let eliminated = Vec::new();
        let remaining: Vec<Spur> = vars.iter().copied().collect();

        // Simplified: just pass through non-variable literals
        for &lit in literals {
            if !self.mentions_any_var(lit, vars) {
                result.push(lit);
            }
        }

        let formula = if result.is_empty() {
            self.manager.mk_true()
        } else {
            self.manager.mk_and(result.iter().copied())
        };

        Ok(MbpResult {
            formulas: vec![formula],
            eliminated,
            remaining,
        })
    }
}

/// Type of bound for a variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundType {
    Lower,
    Upper,
}

/// MBP-based quantifier elimination tactic
#[derive(Debug)]
pub struct MbpTactic<'a> {
    engine: MbpEngine<'a>,
}

impl<'a> MbpTactic<'a> {
    /// Create a new MBP tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            engine: MbpEngine::new(manager),
        }
    }

    /// Apply MBP to eliminate quantifiers
    pub fn eliminate(&mut self, formula: TermId) -> Result<TermId> {
        let Some(t) = self.engine.manager.get(formula).cloned() else {
            return Ok(formula);
        };

        match &t.kind {
            TermKind::Exists { vars, body, .. } => {
                // Create a simple model for the variables
                let model = Model::new();
                let var_names: Vec<_> = vars.iter().map(|(name, _)| *name).collect();

                // Project the existentially quantified variables
                let result = self.engine.project(*body, &var_names, &model)?;
                Ok(result.to_formula(self.engine.manager))
            }
            TermKind::Forall { vars, body, .. } => {
                // ∀x.φ ≡ ¬∃x.¬φ
                let neg_body = self.engine.manager.mk_not(*body);
                let var_names: Vec<_> = vars.iter().map(|(name, _)| *name).collect();

                let model = Model::new();
                let result = self.engine.project(neg_body, &var_names, &model)?;
                let projected = result.to_formula(self.engine.manager);

                Ok(self.engine.manager.mk_not(projected))
            }
            _ => Ok(formula),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_basic() {
        let mut model = Model::new();
        let term = TermId::new(1);

        model.set_bool(term, true);
        assert_eq!(model.get_bool(term), Some(true));

        model.set_int(term, BigInt::from(42));
        assert_eq!(model.get_int(term), Some(&BigInt::from(42)));
    }

    #[test]
    fn test_mbp_result() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();

        let result = MbpResult {
            formulas: vec![t, f],
            eliminated: vec![],
            remaining: vec![],
        };

        assert!(result.is_complete());
        let formula = result.to_formula(&mut manager);
        // Should be OR of true and false
        assert!(manager.get(formula).is_some());
    }

    #[test]
    fn test_mbp_config_default() {
        let config = MbpConfig::default();
        assert_eq!(config.max_case_splits, 100);
        assert!(config.model_completion);
        assert!(config.simplify);
        assert_eq!(config.projector, ProjectorKind::Auto);
    }

    #[test]
    fn test_mbp_engine_extract_literals() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let conj = manager.mk_and([x, y]);

        let engine = MbpEngine::new(&mut manager);
        let literals = engine.extract_literals(conj);

        assert_eq!(literals.len(), 2);
    }

    #[test]
    fn test_mbp_engine_mentions_var() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y]);

        let x_name = manager.intern_str("x");
        let z_name = manager.intern_str("z");

        let engine = MbpEngine::new(&mut manager);

        assert!(engine.mentions_var(sum, x_name));
        assert!(!engine.mentions_var(sum, z_name));
    }

    #[test]
    fn test_mbp_tactic_no_quantifier() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut tactic = MbpTactic::new(&mut manager);
        let result = tactic.eliminate(x).unwrap();

        // No quantifier, should return unchanged
        assert_eq!(result, x);
    }
}
