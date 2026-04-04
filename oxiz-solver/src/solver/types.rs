//! Types and data structures for the SMT solver

#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use oxiz_core::ast::{RoundingMode, TermId, TermKind, TermManager};
use oxiz_sat::{Lit, RestartStrategy, Var};
use smallvec::SmallVec;

/// Proof step for resolution-based proofs
#[derive(Debug, Clone)]
pub enum ProofStep {
    /// Input clause (from the original formula)
    Input {
        /// Clause index
        index: u32,
        /// The clause (as a disjunction of literals)
        clause: Vec<Lit>,
    },
    /// Resolution step
    Resolution {
        /// Index of this proof step
        index: u32,
        /// Left parent clause index
        left: u32,
        /// Right parent clause index
        right: u32,
        /// Pivot variable (the variable resolved on)
        pivot: Var,
        /// Resulting clause
        clause: Vec<Lit>,
    },
    /// Theory lemma (from a theory solver)
    TheoryLemma {
        /// Index of this proof step
        index: u32,
        /// The theory that produced this lemma
        theory: String,
        /// The lemma clause
        clause: Vec<Lit>,
        /// Explanation terms
        explanation: Vec<TermId>,
    },
}

/// A proof of unsatisfiability
#[derive(Debug, Clone)]
pub struct Proof {
    /// Sequence of proof steps leading to the empty clause
    steps: Vec<ProofStep>,
    /// Index of the final empty clause (proving unsat)
    empty_clause_index: Option<u32>,
}

impl Proof {
    /// Create a new empty proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            empty_clause_index: None,
        }
    }

    /// Add a proof step
    pub fn add_step(&mut self, step: ProofStep) {
        self.steps.push(step);
    }

    /// Set the index of the empty clause (final step proving unsat)
    pub fn set_empty_clause(&mut self, index: u32) {
        self.empty_clause_index = Some(index);
    }

    /// Check if the proof is complete (has an empty clause)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.empty_clause_index.is_some()
    }

    /// Get the number of proof steps
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Iterate over all proof steps
    pub fn steps(&self) -> impl Iterator<Item = &ProofStep> {
        self.steps.iter()
    }

    /// Format the proof as a string (for debugging or output)
    #[must_use]
    pub fn format(&self) -> String {
        let mut result = String::from("(proof\n");
        for step in &self.steps {
            match step {
                ProofStep::Input { index, clause } => {
                    result.push_str(&format!("  (input {} {:?})\n", index, clause));
                }
                ProofStep::Resolution {
                    index,
                    left,
                    right,
                    pivot,
                    clause,
                } => {
                    result.push_str(&format!(
                        "  (resolution {} {} {} {:?} {:?})\n",
                        index, left, right, pivot, clause
                    ));
                }
                ProofStep::TheoryLemma {
                    index,
                    theory,
                    clause,
                    ..
                } => {
                    result.push_str(&format!(
                        "  (theory-lemma {} {} {:?})\n",
                        index, theory, clause
                    ));
                }
            }
        }
        if let Some(idx) = self.empty_clause_index {
            result.push_str(&format!("  (empty-clause {})\n", idx));
        }
        result.push_str(")\n");
        result
    }
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a theory constraint associated with a boolean variable
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum Constraint {
    /// Equality constraint: lhs = rhs
    Eq(TermId, TermId),
    /// Disequality constraint: lhs != rhs (negation of equality)
    Diseq(TermId, TermId),
    /// Less-than constraint: lhs < rhs
    Lt(TermId, TermId),
    /// Less-than-or-equal constraint: lhs <= rhs
    Le(TermId, TermId),
    /// Greater-than constraint: lhs > rhs
    Gt(TermId, TermId),
    /// Greater-than-or-equal constraint: lhs >= rhs
    Ge(TermId, TermId),
    /// Boolean-valued uninterpreted function application.
    /// When the SAT solver assigns this variable true/false, we must inform
    /// the EUF solver so that congruence closure can detect conflicts
    /// (e.g., `t(m) = true` and `t(co) = false` but `m = co`).
    BoolApp(TermId),
}

/// Type of arithmetic constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArithConstraintType {
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Le,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Ge,
}

/// Parsed arithmetic constraint with extracted linear expression
/// Represents: sum of (term, coefficient) <= constant OR < constant (if strict)
#[derive(Debug, Clone)]
pub(crate) struct ParsedArithConstraint {
    /// Linear terms: (variable_term, coefficient)
    pub(crate) terms: SmallVec<[(TermId, Rational64); 4]>,
    /// Constant bound (RHS)
    pub(crate) constant: Rational64,
    /// Type of constraint
    pub(crate) constraint_type: ArithConstraintType,
    /// The original term (for conflict explanation)
    pub(crate) reason_term: TermId,
}

/// Polarity of a term in the formula
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Polarity {
    /// Term appears only positively
    Positive,
    /// Term appears only negatively
    Negative,
    /// Term appears in both polarities
    Both,
}

/// Result of SMT solving
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout, incomplete, etc.)
    Unknown,
}

/// Theory checking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TheoryMode {
    /// Eager theory checking (check on every assignment)
    Eager,
    /// Lazy theory checking (check only on complete assignments)
    Lazy,
}

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,
    /// Enable parallel solving
    pub parallel: bool,
    /// Number of threads for parallel solving
    pub num_threads: usize,
    /// Enable proof generation
    pub proof: bool,
    /// Enable model generation
    pub model: bool,
    /// Theory checking mode
    pub theory_mode: TheoryMode,
    /// Enable preprocessing/simplification
    pub simplify: bool,
    /// Maximum number of conflicts before giving up (0 = unlimited)
    pub max_conflicts: u64,
    /// Maximum number of decisions before giving up (0 = unlimited)
    pub max_decisions: u64,
    /// Restart strategy for SAT solver
    pub restart_strategy: RestartStrategy,
    /// Enable clause minimization (recursive minimization of learned clauses)
    pub enable_clause_minimization: bool,
    /// Enable learned clause subsumption
    pub enable_clause_subsumption: bool,
    /// Enable variable elimination during preprocessing
    pub enable_variable_elimination: bool,
    /// Variable elimination limit (max clauses to produce)
    pub variable_elimination_limit: usize,
    /// Enable blocked clause elimination during preprocessing
    pub enable_blocked_clause_elimination: bool,
    /// Enable symmetry breaking predicates
    pub enable_symmetry_breaking: bool,
    /// Enable inprocessing (periodic preprocessing during search)
    pub enable_inprocessing: bool,
    /// Inprocessing interval (number of conflicts between inprocessing)
    pub inprocessing_interval: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl SolverConfig {
    /// Create a configuration optimized for speed (minimal preprocessing)
    /// Best for easy problems or when quick results are needed
    #[must_use]
    pub fn fast() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 4,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Eager,
            simplify: true, // Keep basic simplification
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Geometric, // Faster than Glucose
            enable_clause_minimization: true,             // Keep this, it's fast
            enable_clause_subsumption: false,             // Skip for speed
            enable_variable_elimination: false,           // Skip preprocessing
            variable_elimination_limit: 0,
            enable_blocked_clause_elimination: false, // Skip preprocessing
            enable_symmetry_breaking: false,
            enable_inprocessing: false, // No inprocessing for speed
            inprocessing_interval: 0,
        }
    }

    /// Create a balanced configuration (default)
    /// Good balance between preprocessing and solving speed
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 4,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Eager,
            simplify: true,
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Glucose, // Adaptive restarts
            enable_clause_minimization: true,
            enable_clause_subsumption: true,
            enable_variable_elimination: true,
            variable_elimination_limit: 1000, // Conservative limit
            enable_blocked_clause_elimination: true,
            enable_symmetry_breaking: false, // Still expensive
            enable_inprocessing: true,
            inprocessing_interval: 10000,
        }
    }

    /// Create a configuration optimized for hard problems
    /// Uses aggressive preprocessing and symmetry breaking
    #[must_use]
    pub fn thorough() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 4,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Eager,
            simplify: true,
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Glucose,
            enable_clause_minimization: true,
            enable_clause_subsumption: true,
            enable_variable_elimination: true,
            variable_elimination_limit: 5000, // More aggressive
            enable_blocked_clause_elimination: true,
            enable_symmetry_breaking: true, // Enable for hard problems
            enable_inprocessing: true,
            inprocessing_interval: 5000, // More frequent inprocessing
        }
    }

    /// Create a minimal configuration (almost all features disabled)
    /// Useful for debugging or when you want full control
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            timeout_ms: 0,
            parallel: false,
            num_threads: 1,
            proof: false,
            model: true,
            theory_mode: TheoryMode::Lazy, // Lazy for minimal overhead
            simplify: false,
            max_conflicts: 0,
            max_decisions: 0,
            restart_strategy: RestartStrategy::Geometric,
            enable_clause_minimization: false,
            enable_clause_subsumption: false,
            enable_variable_elimination: false,
            variable_elimination_limit: 0,
            enable_blocked_clause_elimination: false,
            enable_symmetry_breaking: false,
            enable_inprocessing: false,
            inprocessing_interval: 0,
        }
    }

    /// Enable proof generation
    #[must_use]
    pub fn with_proof(mut self) -> Self {
        self.proof = true;
        self
    }

    /// Set timeout in milliseconds
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set maximum number of conflicts
    #[must_use]
    pub fn with_max_conflicts(mut self, max_conflicts: u64) -> Self {
        self.max_conflicts = max_conflicts;
        self
    }

    /// Set maximum number of decisions
    #[must_use]
    pub fn with_max_decisions(mut self, max_decisions: u64) -> Self {
        self.max_decisions = max_decisions;
        self
    }

    /// Enable parallel solving
    #[must_use]
    pub fn with_parallel(mut self, num_threads: usize) -> Self {
        self.parallel = true;
        self.num_threads = num_threads;
        self
    }

    /// Set restart strategy
    #[must_use]
    pub fn with_restart_strategy(mut self, strategy: RestartStrategy) -> Self {
        self.restart_strategy = strategy;
        self
    }

    /// Set theory mode
    #[must_use]
    pub fn with_theory_mode(mut self, mode: TheoryMode) -> Self {
        self.theory_mode = mode;
        self
    }
}

/// Solver statistics
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Number of decisions made
    pub decisions: u64,
    /// Number of conflicts encountered
    pub conflicts: u64,
    /// Number of propagations performed
    pub propagations: u64,
    /// Number of restarts performed
    pub restarts: u64,
    /// Number of learned clauses
    pub learned_clauses: u64,
    /// Number of theory propagations
    pub theory_propagations: u64,
    /// Number of theory conflicts
    pub theory_conflicts: u64,
}

impl Statistics {
    /// Create new statistics with all counters set to zero
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A model (assignment to variables)
#[derive(Debug, Clone)]
pub struct Model {
    /// Variable assignments
    assignments: FxHashMap<TermId, TermId>,
}

impl Model {
    /// Create a new empty model
    #[must_use]
    pub fn new() -> Self {
        Self {
            assignments: FxHashMap::default(),
        }
    }

    /// Get the value of a term in the model
    #[must_use]
    pub fn get(&self, term: TermId) -> Option<TermId> {
        self.assignments.get(&term).copied()
    }

    /// Set a value in the model
    pub fn set(&mut self, term: TermId, value: TermId) {
        self.assignments.insert(term, value);
    }

    /// Minimize the model by removing redundant assignments
    /// Returns a new minimized model containing only essential assignments
    pub fn minimize(&self, essential_vars: &[TermId]) -> Model {
        let mut minimized = Model::new();

        // Only keep assignments for essential variables
        for &var in essential_vars {
            if let Some(&value) = self.assignments.get(&var) {
                minimized.set(var, value);
            }
        }

        minimized
    }

    /// Get the number of assignments in the model
    #[must_use]
    pub fn size(&self) -> usize {
        self.assignments.len()
    }

    /// Get the assignments map (for MBQI integration)
    #[must_use]
    pub fn assignments(&self) -> &FxHashMap<TermId, TermId> {
        &self.assignments
    }

    /// Evaluate a term in this model
    /// Returns the simplified/evaluated term
    pub fn eval(&self, term: TermId, manager: &mut TermManager) -> TermId {
        // First check if we have a direct assignment
        if let Some(val) = self.get(term) {
            return val;
        }

        // Otherwise, recursively evaluate based on term structure
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match t.kind {
            // Constants evaluate to themselves
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. } => term,

            // Variables: look up in model or return the variable itself
            TermKind::Var(_) => self.get(term).unwrap_or(term),

            // Boolean operations
            TermKind::Not(arg) => {
                let arg_val = self.eval(arg, manager);
                if let Some(t) = manager.get(arg_val) {
                    match t.kind {
                        TermKind::True => manager.mk_false(),
                        TermKind::False => manager.mk_true(),
                        _ => manager.mk_not(arg_val),
                    }
                } else {
                    manager.mk_not(arg_val)
                }
            }

            TermKind::And(ref args) => {
                let mut eval_args = Vec::new();
                for &arg in args {
                    let val = self.eval(arg, manager);
                    if let Some(t) = manager.get(val) {
                        if matches!(t.kind, TermKind::False) {
                            return manager.mk_false();
                        }
                        if !matches!(t.kind, TermKind::True) {
                            eval_args.push(val);
                        }
                    } else {
                        eval_args.push(val);
                    }
                }
                if eval_args.is_empty() {
                    manager.mk_true()
                } else if eval_args.len() == 1 {
                    eval_args[0]
                } else {
                    manager.mk_and(eval_args)
                }
            }

            TermKind::Or(ref args) => {
                let mut eval_args = Vec::new();
                for &arg in args {
                    let val = self.eval(arg, manager);
                    if let Some(t) = manager.get(val) {
                        if matches!(t.kind, TermKind::True) {
                            return manager.mk_true();
                        }
                        if !matches!(t.kind, TermKind::False) {
                            eval_args.push(val);
                        }
                    } else {
                        eval_args.push(val);
                    }
                }
                if eval_args.is_empty() {
                    manager.mk_false()
                } else if eval_args.len() == 1 {
                    eval_args[0]
                } else {
                    manager.mk_or(eval_args)
                }
            }

            TermKind::Implies(lhs, rhs) => {
                let lhs_val = self.eval(lhs, manager);
                let rhs_val = self.eval(rhs, manager);

                if let Some(t) = manager.get(lhs_val) {
                    if matches!(t.kind, TermKind::False) {
                        return manager.mk_true();
                    }
                    if matches!(t.kind, TermKind::True) {
                        return rhs_val;
                    }
                }

                if let Some(t) = manager.get(rhs_val)
                    && matches!(t.kind, TermKind::True)
                {
                    return manager.mk_true();
                }

                manager.mk_implies(lhs_val, rhs_val)
            }

            TermKind::Ite(cond, then_br, else_br) => {
                let cond_val = self.eval(cond, manager);

                if let Some(t) = manager.get(cond_val) {
                    match t.kind {
                        TermKind::True => return self.eval(then_br, manager),
                        TermKind::False => return self.eval(else_br, manager),
                        _ => {}
                    }
                }

                let then_val = self.eval(then_br, manager);
                let else_val = self.eval(else_br, manager);
                manager.mk_ite(cond_val, then_val, else_val)
            }

            TermKind::Eq(lhs, rhs) => {
                let lhs_val = self.eval(lhs, manager);
                let rhs_val = self.eval(rhs, manager);

                if lhs_val == rhs_val {
                    return manager.mk_true();
                }

                // Simplify boolean equalities with constants:
                // x = true  => x
                // x = false => NOT x
                // true = x  => x
                // false = x => NOT x
                if let Some(lhs_term) = manager.get(lhs_val)
                    && lhs_term.sort == manager.sorts.bool_sort
                {
                    // Check if rhs is a boolean constant
                    if let Some(rhs_term) = manager.get(rhs_val) {
                        match rhs_term.kind {
                            TermKind::True => return lhs_val,
                            TermKind::False => return manager.mk_not(lhs_val),
                            _ => {}
                        }
                    }
                    // Check if lhs is a boolean constant
                    match lhs_term.kind {
                        TermKind::True => return rhs_val,
                        TermKind::False => return manager.mk_not(rhs_val),
                        _ => {}
                    }
                }

                manager.mk_eq(lhs_val, rhs_val)
            }

            // Arithmetic operations - basic constant folding
            TermKind::Neg(arg) => {
                let arg_val = self.eval(arg, manager);
                if let Some(t) = manager.get(arg_val) {
                    match &t.kind {
                        TermKind::IntConst(n) => return manager.mk_int(-n),
                        TermKind::RealConst(r) => return manager.mk_real(-r),
                        _ => {}
                    }
                }
                manager.mk_not(arg_val)
            }

            TermKind::Add(ref args) => {
                let eval_args: Vec<_> = args.iter().map(|&a| self.eval(a, manager)).collect();
                manager.mk_add(eval_args)
            }

            TermKind::Sub(lhs, rhs) => {
                let lhs_val = self.eval(lhs, manager);
                let rhs_val = self.eval(rhs, manager);
                manager.mk_sub(lhs_val, rhs_val)
            }

            TermKind::Mul(ref args) => {
                let eval_args: Vec<_> = args.iter().map(|&a| self.eval(a, manager)).collect();
                manager.mk_mul(eval_args)
            }

            // For other operations, just return the term or look it up
            _ => self.get(term).unwrap_or(term),
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    /// Pretty print the model in SMT-LIB2 format
    #[cfg(feature = "std")]
    pub fn pretty_print(&self, manager: &TermManager) -> String {
        if self.assignments.is_empty() {
            return "(model)".to_string();
        }

        let mut lines = vec!["(model".to_string()];
        let printer = oxiz_core::smtlib::Printer::new(manager);

        for (&var, &value) in &self.assignments {
            if let Some(term) = manager.get(var) {
                // Only print top-level variables, not internal encoding variables
                if let TermKind::Var(name) = &term.kind {
                    let sort_str = Self::format_sort(term.sort, manager);
                    let value_str = printer.print_term(value);
                    // Use Debug format for the symbol name
                    let name_str = format!("{:?}", name);
                    lines.push(format!(
                        "  (define-fun {} () {} {})",
                        name_str, sort_str, value_str
                    ));
                }
            }
        }
        lines.push(")".to_string());
        lines.join("\n")
    }

    /// Format a sort ID to its SMT-LIB2 representation
    fn format_sort(sort: oxiz_core::sort::SortId, manager: &TermManager) -> String {
        if sort == manager.sorts.bool_sort {
            "Bool".to_string()
        } else if sort == manager.sorts.int_sort {
            "Int".to_string()
        } else if sort == manager.sorts.real_sort {
            "Real".to_string()
        } else if let Some(s) = manager.sorts.get(sort) {
            if let Some(w) = s.bitvec_width() {
                format!("(_ BitVec {})", w)
            } else {
                "Unknown".to_string()
            }
        } else {
            "Unknown".to_string()
        }
    }
}

/// A named assertion for unsat core tracking
#[derive(Debug, Clone)]
pub struct NamedAssertion {
    /// The assertion term (kept for potential future use in minimization)
    #[allow(dead_code)]
    pub term: TermId,
    /// The name (if any)
    pub name: Option<String>,
    /// Index of this assertion
    pub index: u32,
}

/// An unsat core - a minimal set of assertions that are unsatisfiable
#[derive(Debug, Clone)]
pub struct UnsatCore {
    /// The names of assertions in the core
    pub names: Vec<String>,
    /// The indices of assertions in the core
    pub indices: Vec<u32>,
}

impl UnsatCore {
    /// Create a new empty unsat core
    #[must_use]
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Check if the core is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get the number of assertions in the core
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

impl Default for UnsatCore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached FP constraint data for a single assertion term.
#[derive(Debug, Clone)]
pub struct FpConstraintData {
    pub additions: Vec<(TermId, TermId, TermId, TermId, RoundingMode)>,
    pub divisions: Vec<(TermId, TermId, TermId, TermId, RoundingMode)>,
    pub multiplications: Vec<(TermId, TermId, TermId, TermId, RoundingMode)>,
    pub comparisons: Vec<(TermId, TermId, bool)>,
    pub equalities: Vec<(TermId, TermId)>,
    pub literals: FxHashMap<TermId, f64>,
    pub rounding_add_results: FxHashMap<(TermId, TermId, RoundingMode), TermId>,
    pub is_zero: FxHashSet<TermId>,
    pub is_positive: FxHashSet<TermId>,
    pub is_negative: FxHashSet<TermId>,
    pub not_nan: FxHashSet<TermId>,
    pub gt_comparisons: Vec<(TermId, TermId)>,
    pub lt_comparisons: Vec<(TermId, TermId)>,
    pub conversions: Vec<(TermId, u32, u32, TermId)>,
    pub real_to_fp_conversions: Vec<(TermId, u32, u32, TermId)>,
    pub subtractions: Vec<(TermId, TermId, TermId)>,
}

impl FpConstraintData {
    #[must_use]
    pub fn new() -> Self {
        Self {
            additions: Vec::new(),
            divisions: Vec::new(),
            multiplications: Vec::new(),
            comparisons: Vec::new(),
            equalities: Vec::new(),
            literals: FxHashMap::default(),
            rounding_add_results: FxHashMap::default(),
            is_zero: FxHashSet::default(),
            is_positive: FxHashSet::default(),
            is_negative: FxHashSet::default(),
            not_nan: FxHashSet::default(),
            gt_comparisons: Vec::new(),
            lt_comparisons: Vec::new(),
            conversions: Vec::new(),
            real_to_fp_conversions: Vec::new(),
            subtractions: Vec::new(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.additions.is_empty()
            && self.divisions.is_empty()
            && self.multiplications.is_empty()
            && self.comparisons.is_empty()
            && self.equalities.is_empty()
    }

    pub fn merge(&mut self, other: &FpConstraintData) {
        self.additions.extend_from_slice(&other.additions);
        self.divisions.extend_from_slice(&other.divisions);
        self.multiplications
            .extend_from_slice(&other.multiplications);
        self.comparisons.extend_from_slice(&other.comparisons);
        self.equalities.extend_from_slice(&other.equalities);
        for (&k, &v) in &other.literals {
            self.literals.insert(k, v);
        }
        for (&k, &v) in &other.rounding_add_results {
            self.rounding_add_results.insert(k, v);
        }
        self.is_zero.extend(other.is_zero.iter().copied());
        self.is_positive.extend(other.is_positive.iter().copied());
        self.is_negative.extend(other.is_negative.iter().copied());
        self.not_nan.extend(other.not_nan.iter().copied());
        self.gt_comparisons.extend_from_slice(&other.gt_comparisons);
        self.lt_comparisons.extend_from_slice(&other.lt_comparisons);
        self.conversions.extend_from_slice(&other.conversions);
        self.real_to_fp_conversions
            .extend_from_slice(&other.real_to_fp_conversions);
        self.subtractions.extend_from_slice(&other.subtractions);
    }
}

impl Default for FpConstraintData {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy model evaluation cache.
#[derive(Debug)]
pub struct ModelCache {
    model: Model,
    eval_cache: FxHashMap<TermId, TermId>,
    cache_hits: u64,
    cache_misses: u64,
}

impl ModelCache {
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            eval_cache: FxHashMap::default(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[must_use]
    pub fn get_direct(&self, term: TermId) -> Option<TermId> {
        self.model.get(term)
    }

    pub fn eval_lazy(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        if let Some(&cached) = self.eval_cache.get(&term) {
            self.cache_hits += 1;
            return cached;
        }
        self.cache_misses += 1;
        let result = self.model.eval(term, manager);
        self.eval_cache.insert(term, result);
        result
    }

    pub fn eval_batch(
        &mut self,
        terms: &[TermId],
        manager: &mut TermManager,
    ) -> SmallVec<[TermId; 8]> {
        terms
            .iter()
            .map(|&t| {
                if let Some(&cached) = self.eval_cache.get(&t) {
                    self.cache_hits += 1;
                    cached
                } else {
                    self.cache_misses += 1;
                    let result = self.model.eval(t, manager);
                    self.eval_cache.insert(t, result);
                    result
                }
            })
            .collect()
    }

    pub fn invalidate(&mut self) {
        self.eval_cache.clear();
    }

    pub fn invalidate_term(&mut self, term: TermId) {
        self.eval_cache.remove(&term);
    }

    #[must_use]
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }

    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.eval_cache.len()
    }

    #[must_use]
    pub fn model_size(&self) -> usize {
        self.model.size()
    }

    #[must_use]
    pub fn is_cached(&self, term: TermId) -> bool {
        self.eval_cache.contains_key(&term)
    }

    #[must_use]
    pub fn into_model(self) -> Model {
        self.model
    }
}
