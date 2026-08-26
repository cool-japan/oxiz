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
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_rational::BigRational;

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
    /// Nonlinear arithmetic: no complete linear projector applies. Performs a
    /// *sound partial* projection (eliminate target variables only from
    /// genuinely linear literals; keep nonlinear literals opaque and leave the
    /// variables they constrain uneliminated) rather than silently applying
    /// the LRA/LIA projector to nonlinear input. See [`MbpEngine::project`].
    Nonlinear,
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

        // Determine which projector to use.
        let detected = self.detect_projector(formula, &vars_set);

        // Defense in depth: even when a caller bypasses `detect_projector` by
        // explicitly requesting a linear projector (`config.projector` =
        // Lra/Lia), never run the linear projector's completeness assumptions
        // over nonlinear input — route it to the honest partial handler
        // instead (TODO-940: "never silently linear-project nonlinear terms").
        let projector = if matches!(detected, ProjectorKind::Lra | ProjectorKind::Lia)
            && self.contains_nonlinear_arith(formula)
        {
            ProjectorKind::Nonlinear
        } else {
            detected
        };

        // Project based on theory
        match projector {
            ProjectorKind::Lra => self.project_lra(&literals, &vars_set, model),
            ProjectorKind::Lia => self.project_lia(&literals, &vars_set, model),
            ProjectorKind::Array => self.project_array(&literals, &vars_set, model),
            ProjectorKind::Datatype => self.project_datatype(&literals, &vars_set, model),
            ProjectorKind::Nonlinear => self.project_nonlinear(&literals, &vars_set, model),
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

    /// Flatten a conjunction into its literals.
    ///
    /// Uses an explicit heap stack rather than native recursion: the `And`
    /// nesting depth is attacker-controlled (a validly constructed
    /// `(and (and (and ...)))` chain), the return type is `()`, and a depth
    /// cap on a literal *collector* could only drop literals -- i.e. weaken
    /// the projected formula into an unsound over-approximation. See the
    /// module doc comment on `contains_array_ops` for the same reasoning
    /// applied to the predicate walks.
    ///
    /// Note that unlike the predicate walks below this one deliberately does
    /// *not* deduplicate on `TermId`: `extract_literals` returns a multiset
    /// in source order and the projectors index into it positionally.
    /// Re-reaching a shared `And` node twice through two different parents is
    /// therefore reproduced faithfully, exactly as the previous recursive
    /// version did.
    fn extract_literals_rec(&self, term: TermId, literals: &mut Vec<TermId>) {
        // Reverse-push so children come off the stack in source order.
        let mut stack = vec![term];
        while let Some(id) = stack.pop() {
            match self.manager.get(id).map(|t| &t.kind) {
                Some(TermKind::And(args)) => {
                    stack.extend(args.iter().rev().copied());
                }
                // Every non-conjunction node -- including a dangling id with
                // no term behind it -- is a literal of the conjunction. This
                // catch-all is a pure leaf classification (`And` is the one
                // connective this function decomposes), so a future
                // `TermKind` variant is correctly treated as an opaque
                // literal rather than silently dropped.
                _ => literals.push(id),
            }
        }
    }

    /// Iteratively test whether any node of `root`'s term DAG satisfies
    /// `hit`.
    ///
    /// Shared by [`Self::contains_array_ops`],
    /// [`Self::contains_datatype_ops`], [`Self::contains_nonlinear_arith`]
    /// and [`Self::mentions_var`], each of which used to carry its own
    /// hand-written recursive walk with a `_ => false` catch-all. Two
    /// separate defects came out of that shape:
    ///
    /// * **Unbounded native recursion** on a `-> bool` return type. A depth
    ///   cap is not available as a fix here: every one of these predicates
    ///   answers "does this formula contain X", so a capped `false` is a
    ///   *wrong answer*, not a weaker one (see each caller's doc comment).
    /// * **Silently unvisited children.** The hand-written matches each
    ///   enumerated only the handful of `TermKind` variants their author had
    ///   in mind; every other variant -- `Store`, `Select`, `Distinct`,
    ///   `Implies`, `Xor`, `Mod`, `Let`, all of `Str*`/`Bv*`/`Fp*`, the
    ///   quantifiers -- fell into `_ => false` and had its children *never
    ///   looked at*. So e.g. `mentions_var((store a x v), x)` answered
    ///   `false`.
    ///
    /// Descending is now delegated to [`crate::ast::traversal::get_children`],
    /// which matches `TermKind` exhaustively with no catch-all, so a newly
    /// added variant is a compile error there rather than a silent
    /// truncation here. `visited` makes this linear in DAG size rather than
    /// exponential in the tree unfolding; that is unconditionally sound for
    /// all four callers because each asks a question about a subterm that
    /// does not depend on the path taken to reach it.
    ///
    /// Trigger patterns on `Forall`/`Exists` are not descended into, matching
    /// `get_children` (and therefore every other generic walk in the crate).
    /// A `:pattern` annotation is a matching hint that carries no assertional
    /// content, so it cannot change what theory a formula lies in nor which
    /// variables the formula constrains.
    fn any_subterm<F>(&self, root: TermId, mut hit: F) -> bool
    where
        F: FnMut(&TermKind) -> bool,
    {
        let mut stack = vec![root];
        let mut visited = FxHashSet::default();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(term) = self.manager.get(id) else {
                continue;
            };
            if hit(&term.kind) {
                return true;
            }
            stack.extend(crate::ast::traversal::get_children(&term.kind));
        }
        false
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

        // Nonlinear arithmetic (e.g. `x * y`, `x div y`) has no complete
        // Fourier-Motzkin / Loos-Weispfenning linear projector. Route it to
        // the explicit nonlinear handler instead of falling through to the
        // LRA default, which would misrepresent a complete linear elimination
        // over input it cannot soundly eliminate (TODO-940).
        if self.contains_nonlinear_arith(formula) {
            return ProjectorKind::Nonlinear;
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

    /// Whether `term` mentions any array operation anywhere.
    ///
    /// Answering `false` for a formula that *does* contain arrays makes
    /// `detect_projector` route it to the arithmetic projector, which has no
    /// array reasoning at all -- so this must be a complete walk, not a
    /// capped one. See [`Self::any_subterm`].
    fn contains_array_ops(&self, term: TermId) -> bool {
        self.any_subterm(term, |kind| {
            matches!(kind, TermKind::Select(_, _) | TermKind::Store(_, _, _))
        })
    }

    /// Whether `term` mentions any algebraic-datatype operation anywhere.
    ///
    /// Same reachability argument as [`Self::contains_array_ops`]: a `false`
    /// here silently selects a projector that cannot reason about
    /// constructors, testers or selectors.
    fn contains_datatype_ops(&self, term: TermId) -> bool {
        self.any_subterm(term, |kind| {
            matches!(
                kind,
                TermKind::DtConstructor { .. }
                    | TermKind::DtSelector { .. }
                    | TermKind::DtTester { .. }
            )
        })
    }

    /// Whether `formula` contains only linear real arithmetic.
    ///
    /// Linearity is theory-agnostic (a product of two variables is
    /// nonlinear whether the variables range over `Int` or `Real`), so this
    /// currently coincides with [`Self::is_linear_int`]; `detect_projector`
    /// tries this one first and falls back to the LIA check, matching the
    /// existing "try LRA, then LIA" `Auto` dispatch order.
    fn is_linear_real(&self, term: TermId, _vars: &FxHashSet<Spur>) -> bool {
        !self.contains_nonlinear_arith(term)
    }

    /// Whether `formula` contains only linear integer arithmetic. See
    /// [`Self::is_linear_real`].
    fn is_linear_int(&self, term: TermId, _vars: &FxHashSet<Spur>) -> bool {
        !self.contains_nonlinear_arith(term)
    }

    /// Detect nonlinear arithmetic anywhere in `term`: a product of two or
    /// more non-constant factors (e.g. `x * y`), or division/modulo by a
    /// non-constant divisor (e.g. `x div y`).
    ///
    /// `project_lra`/`project_lia` implement Fourier-Motzkin /
    /// Loos-Weispfenning-style elimination, which is only sound for
    /// literals linear in the eliminated variables; running a nonlinear
    /// literal through it would silently produce a wrong projection. This
    /// check feeds [`Self::is_linear_real`]/[`Self::is_linear_int`] (used by
    /// `detect_projector`'s theory selection), and `project_lra` also
    /// independently refuses to eliminate a variable through any literal it
    /// cannot decompose into a recognized bound -- see the `unprojectable`
    /// handling there -- as defense in depth against exactly this class of
    /// bug even when a caller bypasses `detect_projector` by requesting a
    /// specific `ProjectorKind`.
    fn contains_nonlinear_arith(&self, term: TermId) -> bool {
        self.any_subterm(term, |kind| match kind {
            TermKind::Mul(args) => {
                args.iter().filter(|&&a| !self.is_arith_constant(a)).count() >= 2
            }
            TermKind::Div(_, rhs) | TermKind::Mod(_, rhs) => !self.is_arith_constant(*rhs),
            _ => false,
        })
    }

    /// Whether `term` is a numeric literal (no variables), used by
    /// [`Self::contains_nonlinear_arith`] to recognize a constant
    /// coefficient in a product.
    fn is_arith_constant(&self, term: TermId) -> bool {
        matches!(
            self.manager.get(term).map(|t| &t.kind),
            Some(TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. })
        )
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

            // Literals that mention `var` but weren't recognized as a bound
            // on it -- e.g. it appears inside a nonlinear term like
            // `x * y <= 5`, or a linear combination such as `x + y <= 5`
            // that this bare-variable bound matcher doesn't decompose --
            // cannot be soundly eliminated here: doing so anyway would drop
            // `var`'s occurrence from the projected formula while still
            // reporting `var` as eliminated, silently changing the
            // formula's meaning (this is what let nonlinear literals slip
            // through as if they were linear). Leave `var` in `remaining`
            // and keep the untouched literal(s) instead of fabricating an
            // unsound projection.
            let unprojectable: Vec<TermId> = others
                .iter()
                .copied()
                .filter(|&lit| self.mentions_var(lit, var))
                .collect();

            if lower_bounds.is_empty() && upper_bounds.is_empty() {
                if unprojectable.is_empty() {
                    // Variable doesn't appear at all - trivially eliminate.
                    eliminated.push(var);
                    remaining.retain(|&v| v != var);
                } else {
                    result_formulas.extend(unprojectable);
                }
                continue;
            }

            if !unprojectable.is_empty() {
                // At least one occurrence of `var` can't be decomposed into
                // a bound; keep every literal mentioning `var` unchanged
                // rather than eliminating `var` only partially/unsoundly.
                result_formulas.extend(lower_bounds.iter().copied());
                result_formulas.extend(upper_bounds.iter().copied());
                result_formulas.extend(unprojectable);
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

    /// Whether the variable named `var` occurs anywhere in `term`.
    ///
    /// A `false` here is what tells the projectors "this literal is
    /// independent of the variable being eliminated, keep it verbatim in the
    /// residue". Answering `false` for a literal that *does* mention `var`
    /// therefore leaves the supposedly-eliminated variable free in the
    /// projection -- an unsound quantifier elimination -- which is why this
    /// is a complete iterative walk with no depth cap. See
    /// [`Self::any_subterm`].
    fn mentions_var(&self, term: TermId, var: Spur) -> bool {
        self.any_subterm(
            term,
            |kind| matches!(kind, TermKind::Var(name) if *name == var),
        )
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

    /// Nonlinear projector: sound *partial* projection for formulas containing
    /// nonlinear arithmetic.
    ///
    /// `oxiz-core`'s MBP has no NLSAT/CAD-based projector, so nonlinear input
    /// cannot be given a complete quantifier elimination here. Rather than
    /// silently applying the LRA projector's completeness assumptions to
    /// nonlinear literals (TODO-940), this handler eliminates each target
    /// variable **only** through literals that are genuinely linear bounds on
    /// it, keeps every nonlinear literal opaque and unchanged, and leaves any
    /// variable occurring in a nonlinear literal in [`MbpResult::remaining`]
    /// (so [`MbpResult::is_complete`] honestly reports `false`).
    ///
    /// The bound-classification / `unprojectable` logic in [`Self::project_lra`]
    /// already implements exactly this per-variable discipline — a literal
    /// that does not decompose into a recognized linear bound on the variable
    /// (which includes every nonlinear literal) is preserved verbatim and the
    /// variable is not reported as eliminated — so this method delegates to it.
    /// The distinct routing exists so nonlinear input is never *labelled* as a
    /// complete LRA/LIA elimination.
    fn project_nonlinear(
        &mut self,
        literals: &[TermId],
        vars: &FxHashSet<Spur>,
        model: &Model,
    ) -> Result<MbpResult> {
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
        let result = tactic.eliminate(x).expect("test operation should succeed");

        // No quantifier, should return unchanged
        assert_eq!(result, x);
    }

    // ── TODO-940: nonlinear input must not be linear-projected ────────────

    /// Regression: `mentions_var` used to end in `_ => false`, so it never
    /// looked inside `Store`/`Select` (or any other unlisted kind) and
    /// reported `false` for a literal that plainly mentions the variable.
    /// The projectors read that `false` as "this literal is independent of
    /// the variable being eliminated, keep it verbatim", which leaves the
    /// supposedly-eliminated variable free in the projection.
    #[test]
    fn mentions_var_sees_through_array_and_other_unlisted_kinds() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);

        let a = manager.mk_var("a", array_sort);
        let x = manager.mk_var("x", int_sort);
        let v = manager.mk_int(7);
        let stored = manager.mk_store(a, x, v);
        let selected = manager.mk_select(stored, x);
        let distinct = manager.mk_distinct([x, v]);

        let x_name = manager.intern_str("x");
        let z_name = manager.intern_str("z");
        let engine = MbpEngine::new(&mut manager);

        assert!(engine.mentions_var(stored, x_name));
        assert!(engine.mentions_var(selected, x_name));
        assert!(engine.mentions_var(distinct, x_name));
        assert!(!engine.mentions_var(stored, z_name));
    }

    /// Same defect on the projector-selection side: an array operation buried
    /// under an uninterpreted application used to be invisible, routing an
    /// array formula to the arithmetic projector.
    #[test]
    fn contains_array_ops_sees_through_an_uninterpreted_application() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);

        let a = manager.mk_var("a", array_sort);
        let i = manager.mk_var("i", int_sort);
        let sel = manager.mk_select(a, i);
        let p = manager.mk_apply("P", [sel], bool_sort);

        let engine = MbpEngine::new(&mut manager);
        assert!(engine.contains_array_ops(p));
    }

    /// The predicate walks are iterative now: a chain far deeper than any
    /// native stack could hold must simply return. Returning at all is the
    /// assertion -- an overflow aborts the process.
    #[test]
    fn predicates_survive_a_deep_chain_on_a_tiny_stack() {
        const DEPTH: usize = 7_500;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let x = manager.mk_var("x", int_sort);
                let one = manager.mk_int(1);

                let mut chain = x;
                for _ in 0..DEPTH {
                    chain = manager.mk_add([chain, one]);
                }

                let x_name = manager.intern_str("x");
                let engine = MbpEngine::new(&mut manager);
                (
                    engine.mentions_var(chain, x_name),
                    engine.contains_array_ops(chain),
                    engine.contains_nonlinear_arith(chain),
                )
            })
            .expect("test thread must spawn");

        assert_eq!(handle.join().ok(), Some((true, false, false)));
    }

    #[test]
    fn test_detect_projector_routes_nonlinear() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let three = manager.mk_int(3);
        let xy = manager.mk_mul([x, y]); // nonlinear x*y
        let nl = manager.mk_ge(xy, three); // x*y >= 3
        let x_name = manager.intern_str("x");

        let vars: FxHashSet<Spur> = [x_name].into_iter().collect();
        let engine = MbpEngine::new(&mut manager);
        assert_eq!(
            engine.detect_projector(nl, &vars),
            ProjectorKind::Nonlinear,
            "a formula containing x*y must route to the Nonlinear projector"
        );
    }

    #[test]
    fn test_project_nonlinear_leaves_variable_uneliminated() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let three = manager.mk_int(3);
        let five = manager.mk_int(5);
        let xy = manager.mk_mul([x, y]); // nonlinear x*y
        let nl = manager.mk_ge(xy, three); // x*y >= 3
        let lin = manager.mk_le(x, five); // x <= 5
        let conj = manager.mk_and([nl, lin]);
        let x_name = manager.intern_str("x");

        let mut engine = MbpEngine::new(&mut manager);
        let model = Model::new();
        let result = engine
            .project(conj, &[x_name], &model)
            .expect("project must not error");

        // x occurs in the nonlinear literal x*y >= 3, so it must NOT be
        // reported as eliminated, and the projection cannot be complete.
        assert!(
            result.remaining.contains(&x_name),
            "x must remain (constrained by a nonlinear literal)"
        );
        assert!(!result.eliminated.contains(&x_name));
        assert!(!result.is_complete());

        // The nonlinear literal must be preserved verbatim (x still occurs).
        let out = result.formulas[0];
        assert!(
            engine.mentions_var(out, x_name),
            "the nonlinear literal mentioning x must be preserved, not dropped"
        );
    }

    #[test]
    fn test_project_linear_still_eliminates() {
        // Guard against over-restriction: a genuinely linear variable must
        // still be eliminated.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let one = manager.mk_int(1);
        let five = manager.mk_int(5);
        let lower = manager.mk_ge(x, one); // x >= 1
        let upper = manager.mk_le(x, five); // x <= 5
        let conj = manager.mk_and([lower, upper]);
        let x_name = manager.intern_str("x");

        let mut engine = MbpEngine::new(&mut manager);
        let model = Model::new();
        let result = engine
            .project(conj, &[x_name], &model)
            .expect("project must not error");

        assert!(
            result.eliminated.contains(&x_name),
            "a linear variable with lower/upper bounds must be eliminated"
        );
        assert!(result.is_complete());
    }
}
