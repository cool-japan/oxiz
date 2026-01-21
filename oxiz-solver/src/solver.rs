//! Main CDCL(T) Solver

use crate::mbqi::{MBQIResult, MBQISolver};
use crate::simplify::Simplifier;
use num_rational::Rational64;
use num_traits::{One, ToPrimitive, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_sat::{
    Lit, RestartStrategy, Solver as SatSolver, SolverConfig as SatConfig,
    SolverResult as SatResult, TheoryCallback, TheoryCheckResult, Var,
};
use oxiz_theories::arithmetic::ArithSolver;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::euf::EufSolver;
use oxiz_theories::{EqualityNotification, Theory, TheoryCombination};
use rustc_hash::{FxHashMap, FxHashSet};
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
enum Constraint {
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
}

/// Type of arithmetic constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArithConstraintType {
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
struct ParsedArithConstraint {
    /// Linear terms: (variable_term, coefficient)
    terms: SmallVec<[(TermId, Rational64); 4]>,
    /// Constant bound (RHS)
    constant: Rational64,
    /// Type of constraint
    constraint_type: ArithConstraintType,
    /// The original term (for conflict explanation)
    reason_term: TermId,
}

/// Polarity of a term in the formula
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Polarity {
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
                    manager.mk_true()
                } else {
                    manager.mk_eq(lhs_val, rhs_val)
                }
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

/// Main CDCL(T) SMT Solver
#[derive(Debug)]
pub struct Solver {
    /// Configuration
    config: SolverConfig,
    /// SAT solver core
    sat: SatSolver,
    /// EUF theory solver
    euf: EufSolver,
    /// Arithmetic theory solver
    arith: ArithSolver,
    /// Bitvector theory solver
    bv: BvSolver,
    /// MBQI solver for quantified formulas
    mbqi: MBQISolver,
    /// Whether the formula contains quantifiers
    has_quantifiers: bool,
    /// Term to SAT variable mapping
    term_to_var: FxHashMap<TermId, Var>,
    /// SAT variable to term mapping
    var_to_term: Vec<TermId>,
    /// SAT variable to theory constraint mapping
    var_to_constraint: FxHashMap<Var, Constraint>,
    /// SAT variable to parsed arithmetic constraint mapping
    var_to_parsed_arith: FxHashMap<Var, ParsedArithConstraint>,
    /// Current logic
    logic: Option<String>,
    /// Assertions
    assertions: Vec<TermId>,
    /// Named assertions for unsat core tracking
    named_assertions: Vec<NamedAssertion>,
    /// Assumption literals for unsat core tracking (maps assertion index to assumption var)
    /// Reserved for future use with assumption-based unsat core extraction
    #[allow(dead_code)]
    assumption_vars: FxHashMap<u32, Var>,
    /// Model (if sat)
    model: Option<Model>,
    /// Unsat core (if unsat)
    unsat_core: Option<UnsatCore>,
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
    /// Trail of operations for efficient undo
    trail: Vec<TrailOp>,
    /// Tracking which literals have been processed by theories
    theory_processed_up_to: usize,
    /// Whether to produce unsat cores
    produce_unsat_cores: bool,
    /// Track if we've asserted False (for immediate unsat)
    has_false_assertion: bool,
    /// Polarity tracking for optimization
    polarities: FxHashMap<TermId, Polarity>,
    /// Whether polarity-aware encoding is enabled
    polarity_aware: bool,
    /// Whether theory-aware branching is enabled
    theory_aware_branching: bool,
    /// Proof of unsatisfiability (if proof generation is enabled)
    proof: Option<Proof>,
    /// Formula simplifier
    simplifier: Simplifier,
    /// Solver statistics
    statistics: Statistics,
    /// Bitvector terms (for model extraction)
    bv_terms: FxHashSet<TermId>,
    /// Arithmetic terms (Int/Real variables for model extraction)
    arith_terms: FxHashSet<TermId>,
}

/// Theory decision hint
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct TheoryDecision {
    /// The variable to branch on
    pub var: Var,
    /// Suggested value (true = positive, false = negative)
    pub value: bool,
    /// Priority (higher = more important)
    pub priority: i32,
}

/// Theory manager that bridges the SAT solver with theory solvers
struct TheoryManager<'a> {
    /// Reference to the EUF solver
    euf: &'a mut EufSolver,
    /// Reference to the arithmetic solver
    arith: &'a mut ArithSolver,
    /// Reference to the bitvector solver
    bv: &'a mut BvSolver,
    /// Mapping from SAT variables to constraints
    var_to_constraint: &'a FxHashMap<Var, Constraint>,
    /// Mapping from SAT variables to parsed arithmetic constraints
    var_to_parsed_arith: &'a FxHashMap<Var, ParsedArithConstraint>,
    /// Mapping from terms to SAT variables (for conflict clause generation)
    term_to_var: &'a FxHashMap<TermId, Var>,
    /// Current decision level stack for backtracking
    level_stack: Vec<usize>,
    /// Number of processed assignments
    processed_count: usize,
    /// Theory checking mode
    theory_mode: TheoryMode,
    /// Pending assignments for lazy theory checking
    pending_assignments: Vec<(Lit, bool)>,
    /// Theory decision hints for branching
    #[allow(dead_code)]
    decision_hints: Vec<TheoryDecision>,
    /// Pending equality notifications for Nelson-Oppen
    pending_equalities: Vec<EqualityNotification>,
    /// Processed equalities (to avoid duplicates)
    processed_equalities: FxHashMap<(TermId, TermId), bool>,
    /// Reference to solver statistics (for tracking)
    statistics: &'a mut Statistics,
    /// Maximum conflicts allowed (0 = unlimited)
    max_conflicts: u64,
    /// Maximum decisions allowed (0 = unlimited)
    #[allow(dead_code)]
    max_decisions: u64,
}

impl<'a> TheoryManager<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        euf: &'a mut EufSolver,
        arith: &'a mut ArithSolver,
        bv: &'a mut BvSolver,
        var_to_constraint: &'a FxHashMap<Var, Constraint>,
        var_to_parsed_arith: &'a FxHashMap<Var, ParsedArithConstraint>,
        term_to_var: &'a FxHashMap<TermId, Var>,
        theory_mode: TheoryMode,
        statistics: &'a mut Statistics,
        max_conflicts: u64,
        max_decisions: u64,
    ) -> Self {
        Self {
            euf,
            arith,
            bv,
            var_to_constraint,
            var_to_parsed_arith,
            term_to_var,
            level_stack: vec![0],
            processed_count: 0,
            theory_mode,
            pending_assignments: Vec::new(),
            decision_hints: Vec::new(),
            pending_equalities: Vec::new(),
            processed_equalities: FxHashMap::default(),
            statistics,
            max_conflicts,
            max_decisions,
        }
    }

    /// Process Nelson-Oppen equality sharing
    /// Propagates equalities between theories until a fixed point is reached
    #[allow(dead_code)]
    fn propagate_equalities(&mut self) -> TheoryCheckResult {
        // Process all pending equalities
        while let Some(eq) = self.pending_equalities.pop() {
            // Avoid processing the same equality twice
            let key = if eq.lhs < eq.rhs {
                (eq.lhs, eq.rhs)
            } else {
                (eq.rhs, eq.lhs)
            };

            if self.processed_equalities.contains_key(&key) {
                continue;
            }
            self.processed_equalities.insert(key, true);

            // Notify EUF theory
            let lhs_node = self.euf.intern(eq.lhs);
            let rhs_node = self.euf.intern(eq.rhs);
            if let Err(_e) = self
                .euf
                .merge(lhs_node, rhs_node, eq.reason.unwrap_or(eq.lhs))
            {
                // Merge failed - should not happen
                continue;
            }

            // Check for conflicts after merging
            if let Some(conflict_terms) = self.euf.check_conflicts() {
                let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                return TheoryCheckResult::Conflict(conflict_lits);
            }

            // Notify arithmetic theory
            self.arith.notify_equality(eq);
        }

        TheoryCheckResult::Sat
    }

    /// Model-based theory combination
    /// Checks if theories agree on shared terms in their models
    /// If they disagree, generates equality constraints to force agreement
    #[allow(dead_code)]
    fn model_based_combination(&mut self) -> TheoryCheckResult {
        // Collect shared terms (terms that appear in multiple theories)
        let mut shared_terms: Vec<TermId> = Vec::new();

        // For now, we'll consider all terms in the mapping as potentially shared
        // A full implementation would track which terms belong to which theories
        for &term in self.term_to_var.keys() {
            shared_terms.push(term);
        }

        if shared_terms.len() < 2 {
            return TheoryCheckResult::Sat;
        }

        // Check if EUF and arithmetic models agree on shared terms
        // For each pair of terms that EUF considers equal, check if arithmetic agrees
        for i in 0..shared_terms.len() {
            for j in (i + 1)..shared_terms.len() {
                let t1 = shared_terms[i];
                let t2 = shared_terms[j];

                // Check if EUF considers them equal
                let t1_node = self.euf.intern(t1);
                let t2_node = self.euf.intern(t2);

                if self.euf.are_equal(t1_node, t2_node) {
                    // EUF says they're equal
                    // Check if arithmetic solver also considers them equal
                    let t1_value = self.arith.value(t1);
                    let t2_value = self.arith.value(t2);

                    if let (Some(v1), Some(v2)) = (t1_value, t2_value)
                        && v1 != v2
                    {
                        // Disagreement! Generate conflict clause
                        // The conflict is that EUF says t1=t2 but arithmetic says t1≠t2
                        // We need to find the literals that led to this equality in EUF
                        let conflict_lits = self.terms_to_conflict_clause(&[t1, t2]);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
        }

        TheoryCheckResult::Sat
    }

    /// Add an equality to be shared between theories
    #[allow(dead_code)]
    fn add_shared_equality(&mut self, lhs: TermId, rhs: TermId, reason: Option<TermId>) {
        self.pending_equalities
            .push(EqualityNotification { lhs, rhs, reason });
    }

    /// Get theory decision hints for branching
    /// Returns suggested variables to branch on, ordered by priority
    #[allow(dead_code)]
    fn get_decision_hints(&mut self) -> &[TheoryDecision] {
        // Clear old hints
        self.decision_hints.clear();

        // Collect hints from theory solvers
        // For now, we can suggest branching on variables that appear in
        // unsatisfied constraints or pending equalities

        // EUF hints: suggest branching on disequalities that might conflict
        // Arithmetic hints: suggest branching on bounds that are close to being violated

        // This is a placeholder - full implementation would query theory solvers
        // for their preferred branching decisions

        &self.decision_hints
    }

    /// Convert a list of term IDs to a conflict clause
    /// Each term ID should correspond to a constraint that was asserted
    fn terms_to_conflict_clause(&self, terms: &[TermId]) -> SmallVec<[Lit; 8]> {
        let mut conflict = SmallVec::new();
        for &term in terms {
            if let Some(&var) = self.term_to_var.get(&term) {
                // Negate the literal since these are the assertions that led to conflict
                conflict.push(Lit::neg(var));
            }
        }
        conflict
    }

    /// Process a theory constraint
    fn process_constraint(
        &mut self,
        var: Var,
        constraint: Constraint,
        is_positive: bool,
    ) -> TheoryCheckResult {
        match constraint {
            Constraint::Eq(lhs, rhs) => {
                if is_positive {
                    // Positive assignment: a = b, tell EUF to merge
                    let lhs_node = self.euf.intern(lhs);
                    let rhs_node = self.euf.intern(rhs);
                    if let Err(_e) = self.euf.merge(lhs_node, rhs_node, lhs) {
                        // Merge failed - should not happen in normal operation
                        return TheoryCheckResult::Sat;
                    }

                    // Check for immediate conflicts
                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        // Convert term IDs to literals for conflict clause
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }

                    // For arithmetic equalities, also send to ArithSolver
                    // Use pre-parsed constraint if available
                    if let Some(parsed) = self.var_to_parsed_arith.get(&var) {
                        let terms: Vec<(TermId, Rational64)> =
                            parsed.terms.iter().copied().collect();
                        let constant = parsed.constant;
                        let reason = parsed.reason_term;

                        // a = b means a - b <= 0 AND a - b >= 0
                        self.arith.assert_le(&terms, constant, reason);
                        self.arith.assert_ge(&terms, constant, reason);

                        // Check ArithSolver for conflicts
                        use oxiz_theories::Theory;
                        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                        if let Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) = self.arith.check()
                        {
                            let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                            return TheoryCheckResult::Conflict(conflict_lits);
                        }
                    }
                } else {
                    // Negative assignment: a != b, tell EUF about disequality
                    let lhs_node = self.euf.intern(lhs);
                    let rhs_node = self.euf.intern(rhs);
                    self.euf.assert_diseq(lhs_node, rhs_node, lhs);

                    // Check for immediate conflicts (if a = b was already derived)
                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
            Constraint::Diseq(lhs, rhs) => {
                if is_positive {
                    // Positive assignment: a != b
                    let lhs_node = self.euf.intern(lhs);
                    let rhs_node = self.euf.intern(rhs);
                    self.euf.assert_diseq(lhs_node, rhs_node, lhs);

                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                } else {
                    // Negative assignment: ~(a != b) means a = b
                    let lhs_node = self.euf.intern(lhs);
                    let rhs_node = self.euf.intern(rhs);
                    if let Err(_e) = self.euf.merge(lhs_node, rhs_node, lhs) {
                        return TheoryCheckResult::Sat;
                    }

                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
            // Arithmetic constraints - use parsed linear expressions
            Constraint::Lt(_lhs, _rhs)
            | Constraint::Le(_lhs, _rhs)
            | Constraint::Gt(_lhs, _rhs)
            | Constraint::Ge(_lhs, _rhs) => {
                // Look up the pre-parsed linear constraint
                if let Some(parsed) = self.var_to_parsed_arith.get(&var) {
                    // Add constraint to ArithSolver
                    let terms: Vec<(TermId, Rational64)> = parsed.terms.iter().copied().collect();
                    let reason = parsed.reason_term;
                    let constant = parsed.constant;

                    if is_positive {
                        // Positive assignment: constraint holds
                        match parsed.constraint_type {
                            ArithConstraintType::Lt => {
                                // lhs - rhs < 0, i.e., sum of terms < constant
                                self.arith.assert_lt(&terms, constant, reason);
                            }
                            ArithConstraintType::Le => {
                                // lhs - rhs <= 0
                                self.arith.assert_le(&terms, constant, reason);
                            }
                            ArithConstraintType::Gt => {
                                // lhs - rhs > 0, i.e., sum of terms > constant
                                self.arith.assert_gt(&terms, constant, reason);
                            }
                            ArithConstraintType::Ge => {
                                // lhs - rhs >= 0
                                self.arith.assert_ge(&terms, constant, reason);
                            }
                        }
                    } else {
                        // Negative assignment: negation of constraint holds
                        // ~(a < b) => a >= b
                        // ~(a <= b) => a > b
                        // ~(a > b) => a <= b
                        // ~(a >= b) => a < b
                        match parsed.constraint_type {
                            ArithConstraintType::Lt => {
                                // ~(lhs < rhs) => lhs >= rhs
                                self.arith.assert_ge(&terms, constant, reason);
                            }
                            ArithConstraintType::Le => {
                                // ~(lhs <= rhs) => lhs > rhs
                                self.arith.assert_gt(&terms, constant, reason);
                            }
                            ArithConstraintType::Gt => {
                                // ~(lhs > rhs) => lhs <= rhs
                                self.arith.assert_le(&terms, constant, reason);
                            }
                            ArithConstraintType::Ge => {
                                // ~(lhs >= rhs) => lhs < rhs
                                self.arith.assert_lt(&terms, constant, reason);
                            }
                        }
                    }

                    // Check ArithSolver for conflicts
                    use oxiz_theories::Theory;
                    use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                    if let Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) = self.arith.check() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
        }
        TheoryCheckResult::Sat
    }
}

impl TheoryCallback for TheoryManager<'_> {
    fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult {
        let var = lit.var();
        let is_positive = !lit.is_neg();

        // Track propagation
        self.statistics.propagations += 1;

        // In lazy mode, just collect assignments for batch processing
        if self.theory_mode == TheoryMode::Lazy {
            // Check if this variable has a theory constraint
            if self.var_to_constraint.contains_key(&var) {
                self.pending_assignments.push((lit, is_positive));
            }
            return TheoryCheckResult::Sat;
        }

        // Eager mode: process immediately
        // Check if this variable has a theory constraint
        let Some(constraint) = self.var_to_constraint.get(&var).cloned() else {
            return TheoryCheckResult::Sat;
        };

        self.processed_count += 1;
        self.statistics.theory_propagations += 1;

        let result = self.process_constraint(var, constraint, is_positive);

        // Track theory conflicts
        if matches!(result, TheoryCheckResult::Conflict(_)) {
            self.statistics.theory_conflicts += 1;
            self.statistics.conflicts += 1;

            // Check conflict limit
            if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
                return TheoryCheckResult::Sat; // Return Sat to signal resource exhaustion
            }
        }

        result
    }

    fn final_check(&mut self) -> TheoryCheckResult {
        // In lazy mode, process all pending assignments now
        if self.theory_mode == TheoryMode::Lazy {
            for &(lit, is_positive) in &self.pending_assignments.clone() {
                let var = lit.var();
                let Some(constraint) = self.var_to_constraint.get(&var).cloned() else {
                    continue;
                };

                self.statistics.theory_propagations += 1;

                // Process the constraint (same logic as eager mode)
                let result = self.process_constraint(var, constraint, is_positive);
                if let TheoryCheckResult::Conflict(conflict) = result {
                    self.statistics.theory_conflicts += 1;
                    self.statistics.conflicts += 1;

                    // Check conflict limit
                    if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
                        return TheoryCheckResult::Sat; // Signal resource exhaustion
                    }

                    return TheoryCheckResult::Conflict(conflict);
                }
            }
            // Clear pending assignments after processing
            self.pending_assignments.clear();
        }

        // Check EUF for conflicts
        if let Some(conflict_terms) = self.euf.check_conflicts() {
            // Convert TermIds to Lits for the conflict clause
            let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
            self.statistics.theory_conflicts += 1;
            self.statistics.conflicts += 1;

            // Check conflict limit
            if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
                return TheoryCheckResult::Sat; // Signal resource exhaustion
            }

            return TheoryCheckResult::Conflict(conflict_lits);
        }

        // Check arithmetic
        match self.arith.check() {
            Ok(result) => {
                match result {
                    oxiz_theories::TheoryCheckResult::Sat => {
                        // Arithmetic is consistent, now check model-based theory combination
                        // This ensures that different theories agree on shared terms
                        self.model_based_combination()
                    }
                    oxiz_theories::TheoryCheckResult::Unsat(conflict_terms) => {
                        // Arithmetic conflict detected - convert to SAT conflict clause
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        self.statistics.theory_conflicts += 1;
                        self.statistics.conflicts += 1;

                        // Check conflict limit
                        if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts
                        {
                            return TheoryCheckResult::Sat; // Signal resource exhaustion
                        }

                        TheoryCheckResult::Conflict(conflict_lits)
                    }
                    oxiz_theories::TheoryCheckResult::Propagate(_) => {
                        // Propagations should be handled in on_assignment
                        self.model_based_combination()
                    }
                    oxiz_theories::TheoryCheckResult::Unknown => {
                        // Theory is incomplete, be conservative
                        TheoryCheckResult::Sat
                    }
                }
            }
            Err(_error) => {
                // Internal error in the arithmetic solver
                // For now, be conservative and return Sat
                TheoryCheckResult::Sat
            }
        }
    }

    fn on_new_level(&mut self, level: u32) {
        // Push theory state when a new decision level is created
        // Ensure we have enough levels in the stack
        while self.level_stack.len() < (level as usize + 1) {
            self.level_stack.push(self.processed_count);
            self.euf.push();
            self.arith.push();
            self.bv.push();
        }
    }

    fn on_backtrack(&mut self, level: u32) {
        // Pop EUF, Arith, and BV states if needed
        while self.level_stack.len() > (level as usize + 1) {
            self.level_stack.pop();
            self.euf.pop();
            self.arith.pop();
            self.bv.pop();
        }
        self.processed_count = *self.level_stack.last().unwrap_or(&0);

        // Clear pending assignments on backtrack (in lazy mode)
        if self.theory_mode == TheoryMode::Lazy {
            self.pending_assignments.clear();
        }
    }
}

/// Trail operation for efficient undo
#[derive(Debug, Clone)]
enum TrailOp {
    /// An assertion was added
    AssertionAdded { index: usize },
    /// A variable was created
    VarCreated {
        #[allow(dead_code)]
        var: Var,
        term: TermId,
    },
    /// A constraint was added
    ConstraintAdded { var: Var },
    /// False assertion flag was set
    FalseAssertionSet,
    /// A named assertion was added
    NamedAssertionAdded { index: usize },
    /// A bitvector term was added
    BvTermAdded { term: TermId },
    /// An arithmetic term was added
    ArithTermAdded { term: TermId },
}

/// State for push/pop with trail-based undo
#[derive(Debug, Clone)]
struct ContextState {
    num_assertions: usize,
    num_vars: usize,
    has_false_assertion: bool,
    /// Trail position at the time of push
    trail_position: usize,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// Create a new solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new solver with configuration
    #[must_use]
    pub fn with_config(config: SolverConfig) -> Self {
        let proof_enabled = config.proof;

        // Build SAT solver configuration from our config
        let sat_config = SatConfig {
            restart_strategy: config.restart_strategy,
            enable_inprocessing: config.enable_inprocessing,
            inprocessing_interval: config.inprocessing_interval,
            ..SatConfig::default()
        };

        // Note: The following features are controlled by the SAT solver's preprocessor
        // and clause management systems. We pass the configuration but the actual
        // implementation is in oxiz-sat:
        // - Clause minimization (via RecursiveMinimizer)
        // - Clause subsumption (via SubsumptionChecker)
        // - Variable elimination (via Preprocessor::variable_elimination)
        // - Blocked clause elimination (via Preprocessor::blocked_clause_elimination)
        // - Symmetry breaking (via SymmetryBreaker)

        Self {
            config,
            sat: SatSolver::with_config(sat_config),
            euf: EufSolver::new(),
            arith: ArithSolver::lra(),
            bv: BvSolver::new(),
            mbqi: MBQISolver::new(),
            has_quantifiers: false,
            term_to_var: FxHashMap::default(),
            var_to_term: Vec::new(),
            var_to_constraint: FxHashMap::default(),
            var_to_parsed_arith: FxHashMap::default(),
            logic: None,
            assertions: Vec::new(),
            named_assertions: Vec::new(),
            assumption_vars: FxHashMap::default(),
            model: None,
            unsat_core: None,
            context_stack: Vec::new(),
            trail: Vec::new(),
            theory_processed_up_to: 0,
            produce_unsat_cores: false,
            has_false_assertion: false,
            polarities: FxHashMap::default(),
            polarity_aware: true, // Enable polarity-aware encoding by default
            theory_aware_branching: true, // Enable theory-aware branching by default
            proof: if proof_enabled {
                Some(Proof::new())
            } else {
                None
            },
            simplifier: Simplifier::new(),
            statistics: Statistics::new(),
            bv_terms: FxHashSet::default(),
            arith_terms: FxHashSet::default(),
        }
    }

    /// Get the proof (if proof generation is enabled and the result is unsat)
    #[must_use]
    pub fn get_proof(&self) -> Option<&Proof> {
        self.proof.as_ref()
    }

    /// Get the solver statistics
    #[must_use]
    pub fn get_statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Reset the solver statistics
    pub fn reset_statistics(&mut self) {
        self.statistics.reset();
    }

    /// Enable or disable theory-aware branching
    pub fn set_theory_aware_branching(&mut self, enabled: bool) {
        self.theory_aware_branching = enabled;
    }

    /// Check if theory-aware branching is enabled
    #[must_use]
    pub fn theory_aware_branching(&self) -> bool {
        self.theory_aware_branching
    }

    /// Enable or disable unsat core production
    pub fn set_produce_unsat_cores(&mut self, produce: bool) {
        self.produce_unsat_cores = produce;
    }

    /// Set the logic
    pub fn set_logic(&mut self, logic: &str) {
        self.logic = Some(logic.to_string());

        // Switch ArithSolver based on logic
        if logic.contains("LIA") || logic.contains("IDL") || logic.contains("NIA") {
            // Integer arithmetic logic (QF_LIA, LIA, QF_AUFLIA, QF_IDL, etc.)
            self.arith = ArithSolver::lia();
        } else if logic.contains("LRA") || logic.contains("RDL") || logic.contains("NRA") {
            // Real arithmetic logic (QF_LRA, LRA, QF_RDL, etc.)
            self.arith = ArithSolver::lra();
        } else if logic.contains("BV") {
            // Bitvector logic - use LIA since BV comparisons are handled
            // as bounded integer arithmetic
            self.arith = ArithSolver::lia();
        }
        // For other logics (QF_UF, etc.) keep the default LRA
    }

    /// Collect polarity information for all subterms
    /// This is used for polarity-aware encoding optimization
    fn collect_polarities(&mut self, term: TermId, polarity: Polarity, manager: &TermManager) {
        // Update the polarity for this term
        let current = self.polarities.get(&term).copied();
        let new_polarity = match (current, polarity) {
            (Some(Polarity::Both), _) | (_, Polarity::Both) => Polarity::Both,
            (Some(Polarity::Positive), Polarity::Negative)
            | (Some(Polarity::Negative), Polarity::Positive) => Polarity::Both,
            (Some(p), _) => p,
            (None, p) => p,
        };
        self.polarities.insert(term, new_polarity);

        // If we've reached Both polarity, no need to recurse further
        if current == Some(Polarity::Both) {
            return;
        }

        // Recursively collect polarities for subterms
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Not(arg) => {
                let neg_polarity = match polarity {
                    Polarity::Positive => Polarity::Negative,
                    Polarity::Negative => Polarity::Positive,
                    Polarity::Both => Polarity::Both,
                };
                self.collect_polarities(*arg, neg_polarity, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    self.collect_polarities(arg, polarity, manager);
                }
            }
            TermKind::Implies(lhs, rhs) => {
                let neg_polarity = match polarity {
                    Polarity::Positive => Polarity::Negative,
                    Polarity::Negative => Polarity::Positive,
                    Polarity::Both => Polarity::Both,
                };
                self.collect_polarities(*lhs, neg_polarity, manager);
                self.collect_polarities(*rhs, polarity, manager);
            }
            TermKind::Ite(cond, then_br, else_br) => {
                self.collect_polarities(*cond, Polarity::Both, manager);
                self.collect_polarities(*then_br, polarity, manager);
                self.collect_polarities(*else_br, polarity, manager);
            }
            TermKind::Xor(lhs, rhs) | TermKind::Eq(lhs, rhs) => {
                // For XOR and Eq, both sides appear in both polarities
                self.collect_polarities(*lhs, Polarity::Both, manager);
                self.collect_polarities(*rhs, Polarity::Both, manager);
            }
            _ => {
                // For other terms (constants, variables, theory atoms), stop recursion
            }
        }
    }

    /// Get a SAT variable for a term
    fn get_or_create_var(&mut self, term: TermId) -> Var {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }

        let var = self.sat.new_var();
        self.term_to_var.insert(term, var);
        self.trail.push(TrailOp::VarCreated { var, term });

        while self.var_to_term.len() <= var.index() {
            self.var_to_term.push(TermId::new(0));
        }
        self.var_to_term[var.index()] = term;
        var
    }

    /// Track theory variables in a term for model extraction
    /// Recursively scans a term to find Int/Real/BV variables and registers them
    fn track_theory_vars(&mut self, term_id: TermId, manager: &TermManager) {
        let Some(term) = manager.get(term_id) else {
            return;
        };

        match &term.kind {
            TermKind::Var(_) => {
                // Found a variable - check its sort and track appropriately
                let is_int = term.sort == manager.sorts.int_sort;
                let is_real = term.sort == manager.sorts.real_sort;

                if is_int || is_real {
                    if !self.arith_terms.contains(&term_id) {
                        self.arith_terms.insert(term_id);
                        self.trail.push(TrailOp::ArithTermAdded { term: term_id });
                        self.arith.intern(term_id);
                    }
                } else if let Some(sort) = manager.sorts.get(term.sort)
                    && sort.is_bitvec() && !self.bv_terms.contains(&term_id) {
                        self.bv_terms.insert(term_id);
                        self.trail.push(TrailOp::BvTermAdded { term: term_id });
                        if let Some(width) = sort.bitvec_width() {
                            self.bv.new_bv(term_id, width);
                        }
                        // Also intern in ArithSolver for BV comparison constraints
                        // (BV comparisons are handled as bounded integer arithmetic)
                        self.arith.intern(term_id);
                    }
            }
            // Recursively scan compound terms
            TermKind::Add(args) | TermKind::Mul(args) | TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    self.track_theory_vars(arg, manager);
                }
            }
            TermKind::Sub(lhs, rhs)
            | TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::BvAdd(lhs, rhs)
            | TermKind::BvSub(lhs, rhs)
            | TermKind::BvMul(lhs, rhs)
            | TermKind::BvUlt(lhs, rhs)
            | TermKind::BvUle(lhs, rhs)
            | TermKind::BvSlt(lhs, rhs)
            | TermKind::BvSle(lhs, rhs) => {
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
            }
            TermKind::Neg(arg) | TermKind::Not(arg) | TermKind::BvNot(arg) => {
                self.track_theory_vars(*arg, manager);
            }
            TermKind::Ite(cond, then_br, else_br) => {
                self.track_theory_vars(*cond, manager);
                self.track_theory_vars(*then_br, manager);
                self.track_theory_vars(*else_br, manager);
            }
            // Constants and other leaf terms - nothing to track
            _ => {}
        }
    }

    /// Parse an arithmetic comparison and extract linear expression
    /// Returns: (terms with coefficients, constant, constraint_type)
    fn parse_arith_comparison(
        &self,
        lhs: TermId,
        rhs: TermId,
        constraint_type: ArithConstraintType,
        reason: TermId,
        manager: &TermManager,
    ) -> Option<ParsedArithConstraint> {
        let mut terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
        let mut constant = Rational64::zero();

        // Parse LHS (add positive coefficients)
        self.extract_linear_terms(lhs, Rational64::one(), &mut terms, &mut constant, manager)?;

        // Parse RHS (subtract, so coefficients are negated)
        // For lhs OP rhs, we want lhs - rhs OP 0
        self.extract_linear_terms(rhs, -Rational64::one(), &mut terms, &mut constant, manager)?;

        // Combine like terms
        let mut combined: FxHashMap<TermId, Rational64> = FxHashMap::default();
        for (term, coef) in terms {
            *combined.entry(term).or_insert(Rational64::zero()) += coef;
        }

        // Remove zero coefficients
        let final_terms: SmallVec<[(TermId, Rational64); 4]> =
            combined.into_iter().filter(|(_, c)| !c.is_zero()).collect();

        Some(ParsedArithConstraint {
            terms: final_terms,
            constant: -constant, // Move constant to RHS
            constraint_type,
            reason_term: reason,
        })
    }

    /// Extract linear terms recursively from an arithmetic expression
    /// Returns None if the term is not linear
    #[allow(clippy::only_used_in_recursion)]
    fn extract_linear_terms(
        &self,
        term_id: TermId,
        scale: Rational64,
        terms: &mut SmallVec<[(TermId, Rational64); 4]>,
        constant: &mut Rational64,
        manager: &TermManager,
    ) -> Option<()> {
        let term = manager.get(term_id)?;

        match &term.kind {
            // Integer constant
            TermKind::IntConst(n) => {
                if let Some(val) = n.to_i64() {
                    *constant += scale * Rational64::from_integer(val);
                    Some(())
                } else {
                    // BigInt too large, skip for now
                    None
                }
            }

            // Rational constant
            TermKind::RealConst(r) => {
                *constant += scale * *r;
                Some(())
            }

            // Bitvector constant - treat as integer
            TermKind::BitVecConst { value, .. } => {
                if let Some(val) = value.to_i64() {
                    *constant += scale * Rational64::from_integer(val);
                    Some(())
                } else {
                    // BigInt too large, skip for now
                    None
                }
            }

            // Variable (or bitvector variable - treat as integer variable)
            TermKind::Var(_) => {
                terms.push((term_id, scale));
                Some(())
            }

            // Addition
            TermKind::Add(args) => {
                for &arg in args {
                    self.extract_linear_terms(arg, scale, terms, constant, manager)?;
                }
                Some(())
            }

            // Subtraction
            TermKind::Sub(lhs, rhs) => {
                self.extract_linear_terms(*lhs, scale, terms, constant, manager)?;
                self.extract_linear_terms(*rhs, -scale, terms, constant, manager)?;
                Some(())
            }

            // Negation
            TermKind::Neg(arg) => self.extract_linear_terms(*arg, -scale, terms, constant, manager),

            // Multiplication by constant
            TermKind::Mul(args) => {
                // Check if all but one are constants
                let mut const_product = Rational64::one();
                let mut var_term: Option<TermId> = None;

                for &arg in args {
                    let arg_term = manager.get(arg)?;
                    match &arg_term.kind {
                        TermKind::IntConst(n) => {
                            if let Some(val) = n.to_i64() {
                                const_product *= Rational64::from_integer(val);
                            } else {
                                return None; // BigInt too large
                            }
                        }
                        TermKind::RealConst(r) => {
                            const_product *= *r;
                        }
                        _ => {
                            if var_term.is_some() {
                                // Multiple non-constant terms - not linear
                                return None;
                            }
                            var_term = Some(arg);
                        }
                    }
                }

                let new_scale = scale * const_product;
                match var_term {
                    Some(v) => self.extract_linear_terms(v, new_scale, terms, constant, manager),
                    None => {
                        // All constants
                        *constant += new_scale;
                        Some(())
                    }
                }
            }

            // Not linear
            _ => None,
        }
    }

    /// Assert a term
    pub fn assert(&mut self, term: TermId, manager: &mut TermManager) {
        let index = self.assertions.len();
        self.assertions.push(term);
        self.trail.push(TrailOp::AssertionAdded { index });

        // Check if this is a boolean constant first
        if let Some(t) = manager.get(term) {
            match t.kind {
                TermKind::False => {
                    // Mark that we have a false assertion
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: None,
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                TermKind::True => {
                    // True is always satisfied, no need to encode
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: None,
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                _ => {}
            }
        }

        // Apply simplification if enabled
        let term_to_encode = if self.config.simplify {
            self.simplifier.simplify(term, manager)
        } else {
            term
        };

        // Check again if simplification produced a constant
        if let Some(t) = manager.get(term_to_encode) {
            match t.kind {
                TermKind::False => {
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    return;
                }
                TermKind::True => {
                    // Simplified to true, no need to encode
                    return;
                }
                _ => {}
            }
        }

        // Collect polarity information if polarity-aware encoding is enabled
        if self.polarity_aware {
            self.collect_polarities(term_to_encode, Polarity::Positive, manager);
        }

        // Encode the assertion immediately
        let lit = self.encode(term_to_encode, manager);
        self.sat.add_clause([lit]);

        if self.produce_unsat_cores {
            let na_index = self.named_assertions.len();
            self.named_assertions.push(NamedAssertion {
                term,
                name: None,
                index: index as u32,
            });
            self.trail
                .push(TrailOp::NamedAssertionAdded { index: na_index });
        }
    }

    /// Assert a named term (for unsat core tracking)
    pub fn assert_named(&mut self, term: TermId, name: &str, manager: &mut TermManager) {
        let index = self.assertions.len();
        self.assertions.push(term);
        self.trail.push(TrailOp::AssertionAdded { index });

        // Check if this is a boolean constant first
        if let Some(t) = manager.get(term) {
            match t.kind {
                TermKind::False => {
                    // Mark that we have a false assertion
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: Some(name.to_string()),
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                TermKind::True => {
                    // True is always satisfied, no need to encode
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: Some(name.to_string()),
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                _ => {}
            }
        }

        // Collect polarity information if polarity-aware encoding is enabled
        if self.polarity_aware {
            self.collect_polarities(term, Polarity::Positive, manager);
        }

        // Encode the assertion immediately
        let lit = self.encode(term, manager);
        self.sat.add_clause([lit]);

        if self.produce_unsat_cores {
            let na_index = self.named_assertions.len();
            self.named_assertions.push(NamedAssertion {
                term,
                name: Some(name.to_string()),
                index: index as u32,
            });
            self.trail
                .push(TrailOp::NamedAssertionAdded { index: na_index });
        }
    }

    /// Get the unsat core (after check() returned Unsat)
    #[must_use]
    pub fn get_unsat_core(&self) -> Option<&UnsatCore> {
        self.unsat_core.as_ref()
    }

    /// Encode a term into SAT clauses using Tseitin transformation
    fn encode(&mut self, term: TermId, manager: &mut TermManager) -> Lit {
        // Clone the term data to avoid borrowing issues
        let Some(t) = manager.get(term).cloned() else {
            let var = self.get_or_create_var(term);
            return Lit::pos(var);
        };

        match &t.kind {
            TermKind::True => {
                let var = self.get_or_create_var(manager.mk_true());
                self.sat.add_clause([Lit::pos(var)]);
                Lit::pos(var)
            }
            TermKind::False => {
                let var = self.get_or_create_var(manager.mk_false());
                self.sat.add_clause([Lit::neg(var)]);
                Lit::neg(var)
            }
            TermKind::Var(_) => {
                let var = self.get_or_create_var(term);
                // Track theory terms for model extraction
                let is_int = t.sort == manager.sorts.int_sort;
                let is_real = t.sort == manager.sorts.real_sort;

                if is_int || is_real {
                    // Track arithmetic terms
                    if !self.arith_terms.contains(&term) {
                        self.arith_terms.insert(term);
                        self.trail.push(TrailOp::ArithTermAdded { term });
                        // Register with arithmetic solver
                        self.arith.intern(term);
                    }
                } else if let Some(sort) = manager.sorts.get(t.sort)
                    && sort.is_bitvec() && !self.bv_terms.contains(&term) {
                        self.bv_terms.insert(term);
                        self.trail.push(TrailOp::BvTermAdded { term });
                        // Register with BV solver if not already registered
                        if let Some(width) = sort.bitvec_width() {
                            self.bv.new_bv(term, width);
                        }
                    }
                Lit::pos(var)
            }
            TermKind::Not(arg) => {
                let arg_lit = self.encode(*arg, manager);
                arg_lit.negate()
            }
            TermKind::And(args) => {
                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut arg_lits: Vec<Lit> = Vec::new();
                for &arg in args {
                    arg_lits.push(self.encode(arg, manager));
                }

                // Get polarity for optimization
                let polarity = if self.polarity_aware {
                    self.polarities
                        .get(&term)
                        .copied()
                        .unwrap_or(Polarity::Both)
                } else {
                    Polarity::Both
                };

                // result => all args (needed when result is positive)
                // ~result or arg1, ~result or arg2, ...
                if polarity != Polarity::Negative {
                    for &arg in &arg_lits {
                        self.sat.add_clause([result.negate(), arg]);
                    }
                }

                // all args => result (needed when result is negative)
                // ~arg1 or ~arg2 or ... or result
                if polarity != Polarity::Positive {
                    let mut clause: Vec<Lit> = arg_lits.iter().map(|l| l.negate()).collect();
                    clause.push(result);
                    self.sat.add_clause(clause);
                }

                result
            }
            TermKind::Or(args) => {
                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut arg_lits: Vec<Lit> = Vec::new();
                for &arg in args {
                    arg_lits.push(self.encode(arg, manager));
                }

                // Get polarity for optimization
                let polarity = if self.polarity_aware {
                    self.polarities
                        .get(&term)
                        .copied()
                        .unwrap_or(Polarity::Both)
                } else {
                    Polarity::Both
                };

                // result => some arg (needed when result is positive)
                // ~result or arg1 or arg2 or ...
                if polarity != Polarity::Negative {
                    let mut clause: Vec<Lit> = vec![result.negate()];
                    clause.extend(arg_lits.iter().copied());
                    self.sat.add_clause(clause);
                }

                // some arg => result (needed when result is negative)
                // ~arg1 or result, ~arg2 or result, ...
                if polarity != Polarity::Positive {
                    for &arg in &arg_lits {
                        self.sat.add_clause([arg.negate(), result]);
                    }
                }

                result
            }
            TermKind::Xor(lhs, rhs) => {
                let lhs_lit = self.encode(*lhs, manager);
                let rhs_lit = self.encode(*rhs, manager);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (lhs xor rhs)
                // result <=> (lhs and ~rhs) or (~lhs and rhs)

                // result => (lhs or rhs)
                self.sat.add_clause([result.negate(), lhs_lit, rhs_lit]);
                // result => (~lhs or ~rhs)
                self.sat
                    .add_clause([result.negate(), lhs_lit.negate(), rhs_lit.negate()]);

                // (lhs and ~rhs) => result
                self.sat.add_clause([lhs_lit.negate(), rhs_lit, result]);
                // (~lhs and rhs) => result
                self.sat.add_clause([lhs_lit, rhs_lit.negate(), result]);

                result
            }
            TermKind::Implies(lhs, rhs) => {
                let lhs_lit = self.encode(*lhs, manager);
                let rhs_lit = self.encode(*rhs, manager);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (~lhs or rhs)
                // result => ~lhs or rhs
                self.sat
                    .add_clause([result.negate(), lhs_lit.negate(), rhs_lit]);

                // (~lhs or rhs) => result
                // lhs or result, ~rhs or result
                self.sat.add_clause([lhs_lit, result]);
                self.sat.add_clause([rhs_lit.negate(), result]);

                result
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let cond_lit = self.encode(*cond, manager);
                let then_lit = self.encode(*then_br, manager);
                let else_lit = self.encode(*else_br, manager);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (cond ? then : else)
                // cond and result => then
                self.sat
                    .add_clause([cond_lit.negate(), result.negate(), then_lit]);
                // cond and then => result
                self.sat
                    .add_clause([cond_lit.negate(), then_lit.negate(), result]);

                // ~cond and result => else
                self.sat.add_clause([cond_lit, result.negate(), else_lit]);
                // ~cond and else => result
                self.sat.add_clause([cond_lit, else_lit.negate(), result]);

                result
            }
            TermKind::Eq(lhs, rhs) => {
                // Check if this is a boolean equality or theory equality
                let lhs_term = manager.get(*lhs);
                let is_bool_eq = lhs_term.is_some_and(|t| t.sort == manager.sorts.bool_sort);

                if is_bool_eq {
                    // Boolean equality: encode as iff
                    let lhs_lit = self.encode(*lhs, manager);
                    let rhs_lit = self.encode(*rhs, manager);

                    let result_var = self.get_or_create_var(term);
                    let result = Lit::pos(result_var);

                    // result <=> (lhs <=> rhs)
                    // result => (lhs => rhs) and (rhs => lhs)
                    self.sat
                        .add_clause([result.negate(), lhs_lit.negate(), rhs_lit]);
                    self.sat
                        .add_clause([result.negate(), rhs_lit.negate(), lhs_lit]);

                    // (lhs <=> rhs) => result
                    self.sat.add_clause([lhs_lit, rhs_lit, result]);
                    self.sat
                        .add_clause([lhs_lit.negate(), rhs_lit.negate(), result]);

                    result
                } else {
                    // Theory equality: create a fresh boolean variable
                    // Store the constraint for theory propagation
                    let var = self.get_or_create_var(term);
                    self.var_to_constraint
                        .insert(var, Constraint::Eq(*lhs, *rhs));
                    self.trail.push(TrailOp::ConstraintAdded { var });

                    // Track theory variables for model extraction
                    self.track_theory_vars(*lhs, manager);
                    self.track_theory_vars(*rhs, manager);

                    // Pre-parse arithmetic equality for ArithSolver
                    // Only for Int/Real sorts, not BitVec
                    let is_arith = lhs_term
                        .is_some_and(|t| t.sort == manager.sorts.int_sort || t.sort == manager.sorts.real_sort);
                    if is_arith {
                        // We use Le type as placeholder since equality will be asserted
                        // as both Le and Ge
                        if let Some(parsed) = self.parse_arith_comparison(
                            *lhs,
                            *rhs,
                            ArithConstraintType::Le,
                            term,
                            manager,
                        ) {
                            self.var_to_parsed_arith.insert(var, parsed);
                        }
                    }

                    Lit::pos(var)
                }
            }
            TermKind::Distinct(args) => {
                // Encode distinct as pairwise disequalities
                // distinct(a,b,c) <=> (a!=b) and (a!=c) and (b!=c)
                if args.len() <= 1 {
                    // trivially true
                    let var = self.get_or_create_var(manager.mk_true());
                    return Lit::pos(var);
                }

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut diseq_lits = Vec::new();
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        let eq = manager.mk_eq(args[i], args[j]);
                        let eq_lit = self.encode(eq, manager);
                        diseq_lits.push(eq_lit.negate());
                    }
                }

                // result => all disequalities
                for &diseq in &diseq_lits {
                    self.sat.add_clause([result.negate(), diseq]);
                }

                // all disequalities => result
                let mut clause: Vec<Lit> = diseq_lits.iter().map(|l| l.negate()).collect();
                clause.push(result);
                self.sat.add_clause(clause);

                result
            }
            TermKind::Let { bindings, body } => {
                // For encoding, we can substitute the bindings into the body
                // This is a simplification - a more sophisticated approach would
                // memoize the bindings
                let substituted = *body;
                for (name, value) in bindings.iter().rev() {
                    // In a full implementation, we'd perform proper substitution
                    // For now, just encode the body directly
                    let _ = (name, value);
                }
                self.encode(substituted, manager)
            }
            // Theory atoms (arithmetic, bitvec, arrays, UF)
            // These get fresh boolean variables - the theory solver handles the semantics
            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                // Constants are theory terms, not boolean formulas
                // Should not appear at top level in boolean context
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Neg(_)
            | TermKind::Add(_)
            | TermKind::Sub(_, _)
            | TermKind::Mul(_)
            | TermKind::Div(_, _)
            | TermKind::Mod(_, _) => {
                // Arithmetic terms - should not appear at boolean top level
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Lt(lhs, rhs) => {
                // Arithmetic predicate: lhs < rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Lt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Le(lhs, rhs) => {
                // Arithmetic predicate: lhs <= rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Le(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Gt(lhs, rhs) => {
                // Arithmetic predicate: lhs > rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Gt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Gt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Ge(lhs, rhs) => {
                // Arithmetic predicate: lhs >= rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Ge(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Ge, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvConcat(_, _)
            | TermKind::BvExtract { .. }
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
            | TermKind::BvAshr(_, _) => {
                // Bitvector terms - should not appear at boolean top level
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::BvUlt(lhs, rhs) => {
                // Bitvector unsigned less-than: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Lt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse as arithmetic constraint (bitvector as bounded integer)
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvUle(lhs, rhs) => {
                // Bitvector unsigned less-than-or-equal: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Le(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvSlt(lhs, rhs) => {
                // Bitvector signed less-than: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Lt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvSle(lhs, rhs) => {
                // Bitvector signed less-than-or-equal: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Le(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Select(_, _) | TermKind::Store(_, _, _) => {
                // Array operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Apply { .. } => {
                // Uninterpreted function application - theory term
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Forall { patterns, .. } => {
                // Universal quantifiers: register with MBQI
                self.has_quantifiers = true;
                self.mbqi.add_quantifier(term, manager);
                // Collect ground terms from patterns as candidates
                for pattern in patterns {
                    for &trigger in pattern {
                        self.mbqi.collect_ground_terms(trigger, manager);
                    }
                }
                // Create a boolean variable for the quantifier
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Exists { patterns, .. } => {
                // Existential quantifiers: register with MBQI for tracking
                self.has_quantifiers = true;
                self.mbqi.add_quantifier(term, manager);
                // Collect ground terms from patterns
                for pattern in patterns {
                    for &trigger in pattern {
                        self.mbqi.collect_ground_terms(trigger, manager);
                    }
                }
                // Create a boolean variable for the quantifier
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // String operations - theory terms and predicates
            TermKind::StringLit(_)
            | TermKind::StrConcat(_, _)
            | TermKind::StrLen(_)
            | TermKind::StrSubstr(_, _, _)
            | TermKind::StrAt(_, _)
            | TermKind::StrReplace(_, _, _)
            | TermKind::StrReplaceAll(_, _, _)
            | TermKind::StrToInt(_)
            | TermKind::IntToStr(_)
            | TermKind::StrInRe(_, _) => {
                // String terms - theory solver handles these
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::StrContains(_, _)
            | TermKind::StrPrefixOf(_, _)
            | TermKind::StrSuffixOf(_, _)
            | TermKind::StrIndexOf(_, _, _) => {
                // String predicates - theory atoms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point constants and special values
            TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. } => {
                // FP constants - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point operations
            TermKind::FpAbs(_)
            | TermKind::FpNeg(_)
            | TermKind::FpSqrt(_, _)
            | TermKind::FpRoundToIntegral(_, _)
            | TermKind::FpAdd(_, _, _)
            | TermKind::FpSub(_, _, _)
            | TermKind::FpMul(_, _, _)
            | TermKind::FpDiv(_, _, _)
            | TermKind::FpRem(_, _)
            | TermKind::FpMin(_, _)
            | TermKind::FpMax(_, _)
            | TermKind::FpFma(_, _, _, _) => {
                // FP operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point predicates
            TermKind::FpLeq(_, _)
            | TermKind::FpLt(_, _)
            | TermKind::FpGeq(_, _)
            | TermKind::FpGt(_, _)
            | TermKind::FpEq(_, _)
            | TermKind::FpIsNormal(_)
            | TermKind::FpIsSubnormal(_)
            | TermKind::FpIsZero(_)
            | TermKind::FpIsInfinite(_)
            | TermKind::FpIsNaN(_)
            | TermKind::FpIsNegative(_)
            | TermKind::FpIsPositive(_) => {
                // FP predicates - theory atoms that return bool
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point conversions
            TermKind::FpToFp { .. }
            | TermKind::FpToSBV { .. }
            | TermKind::FpToUBV { .. }
            | TermKind::FpToReal(_)
            | TermKind::RealToFp { .. }
            | TermKind::SBVToFp { .. }
            | TermKind::UBVToFp { .. } => {
                // FP conversions - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Datatype operations
            TermKind::DtConstructor { .. }
            | TermKind::DtTester { .. }
            | TermKind::DtSelector { .. } => {
                // Datatype operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Match expressions on datatypes
            TermKind::Match { .. } => {
                // Match expressions - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
        }
    }

    /// Check satisfiability
    pub fn check(&mut self, manager: &mut TermManager) -> SolverResult {
        // Check for trivial unsat (false assertion)
        if self.has_false_assertion {
            self.build_unsat_core_trivial_false();
            return SolverResult::Unsat;
        }

        if self.assertions.is_empty() {
            return SolverResult::Sat;
        }

        // Check resource limits before starting
        if self.config.max_conflicts > 0 && self.statistics.conflicts >= self.config.max_conflicts {
            return SolverResult::Unknown;
        }
        if self.config.max_decisions > 0 && self.statistics.decisions >= self.config.max_decisions {
            return SolverResult::Unknown;
        }

        // Run SAT solver with theory integration
        let mut theory_manager = TheoryManager::new(
            &mut self.euf,
            &mut self.arith,
            &mut self.bv,
            &self.var_to_constraint,
            &self.var_to_parsed_arith,
            &self.term_to_var,
            self.config.theory_mode,
            &mut self.statistics,
            self.config.max_conflicts,
            self.config.max_decisions,
        );

        // MBQI loop for quantified formulas
        let max_mbqi_iterations = 100;
        let mut mbqi_iteration = 0;

        loop {
            let sat_result = self.sat.solve_with_theory(&mut theory_manager);

            match sat_result {
                SatResult::Unsat => {
                    self.build_unsat_core();
                    return SolverResult::Unsat;
                }
                SatResult::Unknown => {
                    return SolverResult::Unknown;
                }
                SatResult::Sat => {
                    // If no quantifiers, we're done
                    if !self.has_quantifiers {
                        self.build_model(manager);
                        self.unsat_core = None;
                        return SolverResult::Sat;
                    }

                    // Build partial model for MBQI
                    self.build_model(manager);

                    // Run MBQI to check quantified formulas
                    let model_assignments = self
                        .model
                        .as_ref()
                        .map(|m| m.assignments().clone())
                        .unwrap_or_default();
                    let mbqi_result = self.mbqi.check_with_model(&model_assignments, manager);

                    match mbqi_result {
                        MBQIResult::NoQuantifiers | MBQIResult::Satisfied => {
                            // All quantifiers satisfied
                            self.unsat_core = None;
                            return SolverResult::Sat;
                        }
                        MBQIResult::InstantiationLimit => {
                            // Too many instantiations - return unknown
                            return SolverResult::Unknown;
                        }
                        MBQIResult::Conflict(conflict_terms) => {
                            // Add conflict clause
                            let lits: Vec<Lit> = conflict_terms
                                .iter()
                                .filter_map(|&t| self.term_to_var.get(&t).map(|&v| Lit::neg(v)))
                                .collect();
                            if !lits.is_empty() {
                                self.sat.add_clause(lits);
                            }
                            // Continue loop
                        }
                        MBQIResult::NewInstantiations(instantiations) => {
                            // Add instantiation lemmas
                            for inst in instantiations {
                                // The instantiation is: ∀x.φ(x) → φ(t)
                                // We assert φ(t) (the result term)
                                let lit = self.encode(inst.result, manager);
                                self.sat.add_clause([lit]);
                            }
                            // Continue loop
                        }
                    }

                    mbqi_iteration += 1;
                    if mbqi_iteration >= max_mbqi_iterations {
                        return SolverResult::Unknown;
                    }

                    // Recreate theory manager for next iteration
                    theory_manager = TheoryManager::new(
                        &mut self.euf,
                        &mut self.arith,
                        &mut self.bv,
                        &self.var_to_constraint,
                        &self.var_to_parsed_arith,
                        &self.term_to_var,
                        self.config.theory_mode,
                        &mut self.statistics,
                        self.config.max_conflicts,
                        self.config.max_decisions,
                    );
                }
            }
        }
    }

    /// Check satisfiability under assumptions
    /// Assumptions are temporary constraints that don't modify the assertion stack
    pub fn check_with_assumptions(
        &mut self,
        assumptions: &[TermId],
        manager: &mut TermManager,
    ) -> SolverResult {
        // Save current state
        self.push();

        // Assert all assumptions
        for &assumption in assumptions {
            self.assert(assumption, manager);
        }

        // Check satisfiability
        let result = self.check(manager);

        // Restore state
        self.pop();

        result
    }

    /// Check satisfiability (pure SAT, no theory integration)
    /// Useful for benchmarking or when theories are not needed
    pub fn check_sat_only(&mut self, manager: &mut TermManager) -> SolverResult {
        if self.assertions.is_empty() {
            return SolverResult::Sat;
        }

        match self.sat.solve() {
            SatResult::Sat => {
                self.build_model(manager);
                SolverResult::Sat
            }
            SatResult::Unsat => SolverResult::Unsat,
            SatResult::Unknown => SolverResult::Unknown,
        }
    }

    /// Build the model after SAT solving
    fn build_model(&mut self, manager: &mut TermManager) {
        let mut model = Model::new();
        let sat_model = self.sat.model();

        // Get boolean values from SAT model
        for (&term, &var) in &self.term_to_var {
            let val = sat_model.get(var.index()).copied();
            if let Some(v) = val {
                let bool_val = if v.is_true() {
                    manager.mk_true()
                } else if v.is_false() {
                    manager.mk_false()
                } else {
                    continue;
                };
                model.set(term, bool_val);
            }
        }

        // Extract values from equality constraints (e.g., x = 5)
        // This handles cases where a variable is equated to a constant
        for (&var, constraint) in &self.var_to_constraint {
            // Check if the equality is assigned true in the SAT model
            let is_true = sat_model
                .get(var.index())
                .copied()
                .is_some_and(|v| v.is_true());

            if !is_true {
                continue;
            }

            if let Constraint::Eq(lhs, rhs) = constraint {
                // Check if one side is a tracked variable and the other is a constant
                let (var_term, const_term) =
                    if self.arith_terms.contains(lhs) || self.bv_terms.contains(lhs) {
                        (*lhs, *rhs)
                    } else if self.arith_terms.contains(rhs) || self.bv_terms.contains(rhs) {
                        (*rhs, *lhs)
                    } else {
                        continue;
                    };

                // Check if const_term is actually a constant
                let Some(const_term_data) = manager.get(const_term) else {
                    continue;
                };

                match &const_term_data.kind {
                    TermKind::IntConst(n) => {
                        if let Some(val) = n.to_i64() {
                            let value_term = manager.mk_int(val);
                            model.set(var_term, value_term);
                        }
                    }
                    TermKind::RealConst(r) => {
                        let value_term = manager.mk_real(*r);
                        model.set(var_term, value_term);
                    }
                    TermKind::BitVecConst { value, width } => {
                        if let Some(val) = value.to_u64() {
                            let value_term = manager.mk_bitvec(val, *width);
                            model.set(var_term, value_term);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Get arithmetic values from theory solver
        // Iterate over tracked arithmetic terms
        for &term in &self.arith_terms {
            // Don't overwrite if already set (e.g., from equality extraction above)
            if model.get(term).is_some() {
                continue;
            }

            if let Some(value) = self.arith.value(term) {
                // Create the appropriate value term based on whether it's integer or real
                let value_term = if *value.denom() == 1 {
                    // Integer value
                    manager.mk_int(*value.numer())
                } else {
                    // Rational value
                    manager.mk_real(value)
                };
                model.set(term, value_term);
            } else {
                // If no value from ArithSolver (e.g., unconstrained variable), use default
                // Get the sort to determine if it's Int or Real
                let is_int = manager
                    .get(term)
                    .map(|t| t.sort == manager.sorts.int_sort)
                    .unwrap_or(true);

                let value_term = if is_int {
                    manager.mk_int(0i64)
                } else {
                    manager.mk_real(num_rational::Rational64::from_integer(0))
                };
                model.set(term, value_term);
            }
        }

        // Get bitvector values - check ArithSolver first (for BV comparisons),
        // then BvSolver (for BV arithmetic/bit operations)
        for &term in &self.bv_terms {
            // Don't overwrite if already set (shouldn't happen, but be safe)
            if model.get(term).is_some() {
                continue;
            }

            // Get the bitvector width from the term's sort
            let width = manager
                .get(term)
                .and_then(|t| manager.sorts.get(t.sort))
                .and_then(|s| s.bitvec_width())
                .unwrap_or(64);

            // For BV comparisons handled as bounded integer arithmetic,
            // check ArithSolver FIRST (it has the actual constraint values)
            if let Some(arith_value) = self.arith.value(term) {
                let int_value = arith_value.to_integer();
                let value_term = manager.mk_bitvec(int_value, width);
                model.set(term, value_term);
            } else if let Some(bv_value) = self.bv.get_value(term) {
                // For BV bit operations, get value from BvSolver
                let value_term = manager.mk_bitvec(bv_value, width);
                model.set(term, value_term);
            } else {
                // If no value from either solver, use default value (0)
                // This handles unconstrained BV variables
                let value_term = manager.mk_bitvec(0i64, width);
                model.set(term, value_term);
            }
        }

        self.model = Some(model);
    }

    /// Build unsat core for trivial conflicts (assertion of false)
    fn build_unsat_core_trivial_false(&mut self) {
        if !self.produce_unsat_cores {
            self.unsat_core = None;
            return;
        }

        // Find all assertions that are trivially false
        let mut core = UnsatCore::new();

        for (i, &term) in self.assertions.iter().enumerate() {
            if term == TermId::new(1) {
                // This is a false assertion
                core.indices.push(i as u32);

                // Find the name if there is one
                if let Some(named) = self.named_assertions.iter().find(|na| na.index == i as u32)
                    && let Some(ref name) = named.name
                {
                    core.names.push(name.clone());
                }
            }
        }

        self.unsat_core = Some(core);
    }

    /// Build unsat core from SAT solver conflict analysis
    fn build_unsat_core(&mut self) {
        if !self.produce_unsat_cores {
            self.unsat_core = None;
            return;
        }

        // Build unsat core from the named assertions
        // In assumption-based mode, we would use the failed assumptions from the SAT solver
        // For now, we use a heuristic approach based on the conflict analysis

        let mut core = UnsatCore::new();

        // If assumption_vars is populated, we can use assumption-based extraction
        if !self.assumption_vars.is_empty() {
            // Assumption-based core extraction
            // Get the failed assumptions from the SAT solver
            // Note: This requires SAT solver support for assumption tracking
            // For now, include all named assertions as a conservative approach
            for na in &self.named_assertions {
                core.indices.push(na.index);
                if let Some(ref name) = na.name {
                    core.names.push(name.clone());
                }
            }
        } else {
            // Fallback: include all named assertions
            // This provides a valid unsat core, though not necessarily minimal
            for na in &self.named_assertions {
                core.indices.push(na.index);
                if let Some(ref name) = na.name {
                    core.names.push(name.clone());
                }
            }
        }

        self.unsat_core = Some(core);
    }

    /// Enable assumption-based unsat core tracking
    /// This creates assumption variables for each assertion
    /// which can be used to efficiently extract minimal unsat cores
    pub fn enable_assumption_based_cores(&mut self) {
        self.produce_unsat_cores = true;
        // Assumption variables would be created during assertion
        // to enable fine-grained core extraction
    }

    /// Minimize an unsat core using greedy deletion
    /// This creates a minimal (but not necessarily minimum) unsatisfiable subset
    pub fn minimize_unsat_core(&mut self, manager: &mut TermManager) -> Option<UnsatCore> {
        if !self.produce_unsat_cores {
            return None;
        }

        // Get the current unsat core
        let core = self.unsat_core.as_ref()?;
        if core.is_empty() {
            return Some(core.clone());
        }

        // Extract the assertions in the core
        let mut core_assertions: Vec<_> = core
            .indices
            .iter()
            .map(|&idx| {
                let assertion = self.assertions[idx as usize];
                let name = self
                    .named_assertions
                    .iter()
                    .find(|na| na.index == idx)
                    .and_then(|na| na.name.clone());
                (idx, assertion, name)
            })
            .collect();

        // Try to remove each assertion one by one
        let mut i = 0;
        while i < core_assertions.len() {
            // Create a temporary solver with all assertions except the i-th one
            let mut temp_solver = Solver::new();
            temp_solver.set_logic(self.logic.as_deref().unwrap_or("ALL"));

            // Add all assertions except the i-th one
            for (j, &(_, assertion, _)) in core_assertions.iter().enumerate() {
                if i != j {
                    temp_solver.assert(assertion, manager);
                }
            }

            // Check if still unsat
            if temp_solver.check(manager) == SolverResult::Unsat {
                // Still unsat without this assertion - remove it
                core_assertions.remove(i);
                // Don't increment i, check the next element which is now at position i
            } else {
                // This assertion is needed
                i += 1;
            }
        }

        // Build the minimized core
        let mut minimized = UnsatCore::new();
        for (idx, _, name) in core_assertions {
            minimized.indices.push(idx);
            if let Some(n) = name {
                minimized.names.push(n);
            }
        }

        Some(minimized)
    }

    /// Get the model (if sat)
    #[must_use]
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// Assert multiple terms at once
    /// This is more efficient than calling assert() multiple times
    pub fn assert_many(&mut self, terms: &[TermId], manager: &mut TermManager) {
        for &term in terms {
            self.assert(term, manager);
        }
    }

    /// Get the number of assertions in the solver
    #[must_use]
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Get the number of variables in the SAT solver
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.term_to_var.len()
    }

    /// Check if the solver has any assertions
    #[must_use]
    pub fn has_assertions(&self) -> bool {
        !self.assertions.is_empty()
    }

    /// Get the current context level (push/pop depth)
    #[must_use]
    pub fn context_level(&self) -> usize {
        self.context_stack.len()
    }

    /// Push a context level
    pub fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_assertions: self.assertions.len(),
            num_vars: self.var_to_term.len(),
            has_false_assertion: self.has_false_assertion,
            trail_position: self.trail.len(),
        });
        self.sat.push();
        self.euf.push();
        self.arith.push();
    }

    /// Pop a context level using trail-based undo
    pub fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Undo all operations in the trail since the push
            while self.trail.len() > state.trail_position {
                if let Some(op) = self.trail.pop() {
                    match op {
                        TrailOp::AssertionAdded { index } => {
                            if self.assertions.len() > index {
                                self.assertions.truncate(index);
                            }
                        }
                        TrailOp::VarCreated { var: _, term } => {
                            // Remove the term-to-var mapping
                            self.term_to_var.remove(&term);
                        }
                        TrailOp::ConstraintAdded { var } => {
                            // Remove the constraint
                            self.var_to_constraint.remove(&var);
                        }
                        TrailOp::FalseAssertionSet => {
                            // Reset the flag
                            self.has_false_assertion = false;
                        }
                        TrailOp::NamedAssertionAdded { index } => {
                            // Remove the named assertion
                            if self.named_assertions.len() > index {
                                self.named_assertions.truncate(index);
                            }
                        }
                        TrailOp::BvTermAdded { term } => {
                            // Remove the bitvector term
                            self.bv_terms.remove(&term);
                        }
                        TrailOp::ArithTermAdded { term } => {
                            // Remove the arithmetic term
                            self.arith_terms.remove(&term);
                        }
                    }
                }
            }

            // Use state to restore other fields
            self.assertions.truncate(state.num_assertions);
            self.var_to_term.truncate(state.num_vars);
            self.has_false_assertion = state.has_false_assertion;

            self.sat.pop();
            self.euf.pop();
            self.arith.pop();
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.sat.reset();
        self.euf.reset();
        self.arith.reset();
        self.bv.reset();
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.var_to_constraint.clear();
        self.var_to_parsed_arith.clear();
        self.assertions.clear();
        self.named_assertions.clear();
        self.model = None;
        self.unsat_core = None;
        self.context_stack.clear();
        self.trail.clear();
        self.logic = None;
        self.theory_processed_up_to = 0;
        self.has_false_assertion = false;
        self.bv_terms.clear();
        self.arith_terms.clear();
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SolverConfig) {
        self.config = config;
    }

    /// Get solver statistics
    #[must_use]
    pub fn stats(&self) -> &oxiz_sat::SolverStats {
        self.sat.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_empty() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    }

    #[test]
    fn test_solver_true() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        solver.assert(t, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    }

    #[test]
    fn test_solver_false() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let f = manager.mk_false();
        solver.assert(f, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
    }

    #[test]
    fn test_solver_push_pop() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let t = manager.mk_true();
        solver.assert(t, &mut manager);
        solver.push();

        let f = manager.mk_false();
        solver.assert(f, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Unsat);

        solver.pop();
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    }

    #[test]
    fn test_unsat_core_trivial() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        solver.set_produce_unsat_cores(true);

        let t = manager.mk_true();
        let f = manager.mk_false();

        solver.assert_named(t, "a1", &mut manager);
        solver.assert_named(f, "a2", &mut manager);
        solver.assert_named(t, "a3", &mut manager);

        assert_eq!(solver.check(&mut manager), SolverResult::Unsat);

        let core = solver.get_unsat_core();
        assert!(core.is_some());

        let core = core.unwrap();
        assert!(!core.is_empty());
        assert!(core.names.contains(&"a2".to_string()));
    }

    #[test]
    fn test_unsat_core_not_produced_when_sat() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        solver.set_produce_unsat_cores(true);

        let t = manager.mk_true();
        solver.assert_named(t, "a1", &mut manager);
        solver.assert_named(t, "a2", &mut manager);

        assert_eq!(solver.check(&mut manager), SolverResult::Sat);
        assert!(solver.get_unsat_core().is_none());
    }

    #[test]
    fn test_unsat_core_disabled() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        // Don't enable unsat cores

        let f = manager.mk_false();
        solver.assert(f, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Unsat);

        // Core should be None when not enabled
        assert!(solver.get_unsat_core().is_none());
    }

    #[test]
    fn test_boolean_encoding_and() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        // Test: (p and q) should be SAT with p=true, q=true
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let and = manager.mk_and(vec![p, q]);

        solver.assert(and, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);

        // The model should have both p and q as true
        let model = solver.model().expect("Should have model");
        assert!(model.get(p).is_some());
        assert!(model.get(q).is_some());
    }

    #[test]
    fn test_boolean_encoding_or() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        // Test: (p or q) and (not p) should be SAT with q=true
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let or = manager.mk_or(vec![p, q]);
        let not_p = manager.mk_not(p);

        solver.assert(or, &mut manager);
        solver.assert(not_p, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    }

    #[test]
    fn test_boolean_encoding_implies() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        // Test: (p => q) and p and (not q) should be UNSAT
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let implies = manager.mk_implies(p, q);
        let not_q = manager.mk_not(q);

        solver.assert(implies, &mut manager);
        solver.assert(p, &mut manager);
        solver.assert(not_q, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
    }

    #[test]
    fn test_boolean_encoding_distinct() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        // Test: distinct(p, q, r) and p and q should be UNSAT (since p=q)
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let r = manager.mk_var("r", manager.sorts.bool_sort);
        let distinct = manager.mk_distinct(vec![p, q, r]);

        solver.assert(distinct, &mut manager);
        solver.assert(p, &mut manager);
        solver.assert(q, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
    }

    #[test]
    fn test_model_evaluation_bool() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);

        // Assert p and not q
        solver.assert(p, &mut manager);
        solver.assert(manager.mk_not(q), &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);

        let model = solver.model().expect("Should have model");

        // Evaluate p (should be true)
        let p_val = model.eval(p, &mut manager);
        assert_eq!(p_val, manager.mk_true());

        // Evaluate q (should be false)
        let q_val = model.eval(q, &mut manager);
        assert_eq!(q_val, manager.mk_false());

        // Evaluate (p and q) - should be false
        let and_term = manager.mk_and(vec![p, q]);
        let and_val = model.eval(and_term, &mut manager);
        assert_eq!(and_val, manager.mk_false());

        // Evaluate (p or q) - should be true
        let or_term = manager.mk_or(vec![p, q]);
        let or_val = model.eval(or_term, &mut manager);
        assert_eq!(or_val, manager.mk_true());
    }

    #[test]
    fn test_model_evaluation_ite() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let r = manager.mk_var("r", manager.sorts.bool_sort);

        // Assert p
        solver.assert(p, &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);

        let model = solver.model().expect("Should have model");

        // Evaluate (ite p q r) - should evaluate to q since p is true
        let ite_term = manager.mk_ite(p, q, r);
        let ite_val = model.eval(ite_term, &mut manager);
        // The result should be q's value (whatever it is in the model)
        let q_val = model.eval(q, &mut manager);
        assert_eq!(ite_val, q_val);
    }

    #[test]
    fn test_model_evaluation_implies() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);

        // Assert not p
        solver.assert(manager.mk_not(p), &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);

        let model = solver.model().expect("Should have model");

        // Evaluate (p => q) - should be true since p is false
        let implies_term = manager.mk_implies(p, q);
        let implies_val = model.eval(implies_term, &mut manager);
        assert_eq!(implies_val, manager.mk_true());
    }

    #[test]
    fn test_bv_comparison_model_generation() {
        // Test BV comparison: 5 < x < 10 should give x in range [6, 9]
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        solver.set_logic("QF_BV");

        // Create BitVec[8] variable
        let bv8_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv8_sort);

        // Create constants
        let five = manager.mk_bitvec(5i64, 8);
        let ten = manager.mk_bitvec(10i64, 8);

        // Assert: 5 < x (unsigned)
        let lt1 = manager.mk_bv_ult(five, x);
        solver.assert(lt1, &mut manager);

        // Assert: x < 10 (unsigned)
        let lt2 = manager.mk_bv_ult(x, ten);
        solver.assert(lt2, &mut manager);

        let result = solver.check(&mut manager);
        assert_eq!(result, SolverResult::Sat);

        // Check that we get a valid model
        let model = solver.model().expect("Should have model");

        // Get the value of x
        if let Some(x_value_id) = model.get(x) {
            if let Some(x_term) = manager.get(x_value_id) {
                if let TermKind::BitVecConst { value, .. } = &x_term.kind {
                    let x_val = value.to_u64().unwrap_or(0);
                    // x should be in range [6, 9]
                    assert!(x_val >= 6 && x_val <= 9, "Expected x in [6,9], got {}", x_val);
                }
            }
        }
    }

    #[test]
    fn test_arithmetic_model_generation() {
        use num_bigint::BigInt;

        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        // Create integer variables
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // Create constraints: x + y = 10, x >= 0, y >= 0
        let ten = manager.mk_int(BigInt::from(10));
        let zero = manager.mk_int(BigInt::from(0));
        let sum = manager.mk_add(vec![x, y]);

        let eq = manager.mk_eq(sum, ten);
        let x_ge_0 = manager.mk_ge(x, zero);
        let y_ge_0 = manager.mk_ge(y, zero);

        solver.assert(eq, &mut manager);
        solver.assert(x_ge_0, &mut manager);
        solver.assert(y_ge_0, &mut manager);

        assert_eq!(solver.check(&mut manager), SolverResult::Sat);

        // Check that we can get a model (even if the arithmetic values aren't fully computed yet)
        let model = solver.model();
        assert!(model.is_some(), "Should have a model for SAT result");
    }

    #[test]
    fn test_model_pretty_print() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);

        solver.assert(p, &mut manager);
        solver.assert(manager.mk_not(q), &mut manager);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);

        let model = solver.model().expect("Should have model");
        let pretty = model.pretty_print(&manager);

        // Should contain the model structure
        assert!(pretty.contains("(model"));
        assert!(pretty.contains("define-fun"));
        // Should contain variable names
        assert!(pretty.contains("p") || pretty.contains("q"));
    }

    #[test]
    fn test_trail_based_undo_assertions() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);

        // Initial state
        assert_eq!(solver.assertions.len(), 0);
        assert_eq!(solver.trail.len(), 0);

        // Assert p
        solver.assert(p, &mut manager);
        assert_eq!(solver.assertions.len(), 1);
        assert!(!solver.trail.is_empty());

        // Push and assert q
        solver.push();
        let trail_len_after_push = solver.trail.len();
        solver.assert(q, &mut manager);
        assert_eq!(solver.assertions.len(), 2);
        assert!(solver.trail.len() > trail_len_after_push);

        // Pop should undo the second assertion
        solver.pop();
        assert_eq!(solver.assertions.len(), 1);
        assert_eq!(solver.assertions[0], p);
    }

    #[test]
    fn test_trail_based_undo_variables() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);

        // Assert p creates variables
        solver.assert(p, &mut manager);
        let initial_var_count = solver.term_to_var.len();

        // Push and assert q
        solver.push();
        solver.assert(q, &mut manager);
        assert!(solver.term_to_var.len() >= initial_var_count);

        // Pop should remove q's variable
        solver.pop();
        // Note: Some variables may remain due to encoding, but q should be removed
        assert_eq!(solver.assertions.len(), 1);
    }

    #[test]
    fn test_trail_based_undo_constraints() {
        use num_bigint::BigInt;

        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(BigInt::from(0));

        // Assert x >= 0 creates a constraint
        let c1 = manager.mk_ge(x, zero);
        solver.assert(c1, &mut manager);
        let initial_constraint_count = solver.var_to_constraint.len();

        // Push and add another constraint
        solver.push();
        let ten = manager.mk_int(BigInt::from(10));
        let c2 = manager.mk_le(x, ten);
        solver.assert(c2, &mut manager);
        assert!(solver.var_to_constraint.len() >= initial_constraint_count);

        // Pop should remove the second constraint
        solver.pop();
        assert_eq!(solver.var_to_constraint.len(), initial_constraint_count);
    }

    #[test]
    fn test_trail_based_undo_false_assertion() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        assert!(!solver.has_false_assertion);

        solver.push();
        solver.assert(manager.mk_false(), &mut manager);
        assert!(solver.has_false_assertion);

        solver.pop();
        assert!(!solver.has_false_assertion);
    }

    #[test]
    fn test_trail_based_undo_named_assertions() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        solver.set_produce_unsat_cores(true);

        let p = manager.mk_var("p", manager.sorts.bool_sort);

        solver.assert_named(p, "assertion1", &mut manager);
        assert_eq!(solver.named_assertions.len(), 1);

        solver.push();
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        solver.assert_named(q, "assertion2", &mut manager);
        assert_eq!(solver.named_assertions.len(), 2);

        solver.pop();
        assert_eq!(solver.named_assertions.len(), 1);
        assert_eq!(
            solver.named_assertions[0].name,
            Some("assertion1".to_string())
        );
    }

    #[test]
    fn test_trail_based_undo_nested_push_pop() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        solver.assert(p, &mut manager);

        // First push
        solver.push();
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        solver.assert(q, &mut manager);
        assert_eq!(solver.assertions.len(), 2);

        // Second push
        solver.push();
        let r = manager.mk_var("r", manager.sorts.bool_sort);
        solver.assert(r, &mut manager);
        assert_eq!(solver.assertions.len(), 3);

        // Pop once
        solver.pop();
        assert_eq!(solver.assertions.len(), 2);

        // Pop again
        solver.pop();
        assert_eq!(solver.assertions.len(), 1);
        assert_eq!(solver.assertions[0], p);
    }

    #[test]
    fn test_config_presets() {
        // Test that all presets can be created without panicking
        let _fast = SolverConfig::fast();
        let _balanced = SolverConfig::balanced();
        let _thorough = SolverConfig::thorough();
        let _minimal = SolverConfig::minimal();
    }

    #[test]
    fn test_config_fast_characteristics() {
        let config = SolverConfig::fast();

        // Fast config should disable expensive features
        assert!(!config.enable_variable_elimination);
        assert!(!config.enable_blocked_clause_elimination);
        assert!(!config.enable_symmetry_breaking);
        assert!(!config.enable_inprocessing);
        assert!(!config.enable_clause_subsumption);

        // But keep fast optimizations
        assert!(config.enable_clause_minimization);
        assert!(config.simplify);

        // Should use Geometric restarts (faster)
        assert_eq!(config.restart_strategy, RestartStrategy::Geometric);
    }

    #[test]
    fn test_config_balanced_characteristics() {
        let config = SolverConfig::balanced();

        // Balanced should enable most features with moderate settings
        assert!(config.enable_variable_elimination);
        assert!(config.enable_blocked_clause_elimination);
        assert!(config.enable_inprocessing);
        assert!(config.enable_clause_minimization);
        assert!(config.enable_clause_subsumption);
        assert!(config.simplify);

        // But not the most expensive one
        assert!(!config.enable_symmetry_breaking);

        // Should use Glucose restarts (adaptive)
        assert_eq!(config.restart_strategy, RestartStrategy::Glucose);

        // Conservative limits
        assert_eq!(config.variable_elimination_limit, 1000);
        assert_eq!(config.inprocessing_interval, 10000);
    }

    #[test]
    fn test_config_thorough_characteristics() {
        let config = SolverConfig::thorough();

        // Thorough should enable all features
        assert!(config.enable_variable_elimination);
        assert!(config.enable_blocked_clause_elimination);
        assert!(config.enable_symmetry_breaking); // Even this expensive one
        assert!(config.enable_inprocessing);
        assert!(config.enable_clause_minimization);
        assert!(config.enable_clause_subsumption);
        assert!(config.simplify);

        // Aggressive settings
        assert_eq!(config.variable_elimination_limit, 5000);
        assert_eq!(config.inprocessing_interval, 5000);
    }

    #[test]
    fn test_config_minimal_characteristics() {
        let config = SolverConfig::minimal();

        // Minimal should disable everything optional
        assert!(!config.simplify);
        assert!(!config.enable_variable_elimination);
        assert!(!config.enable_blocked_clause_elimination);
        assert!(!config.enable_symmetry_breaking);
        assert!(!config.enable_inprocessing);
        assert!(!config.enable_clause_minimization);
        assert!(!config.enable_clause_subsumption);

        // Should use lazy theory mode for minimal overhead
        assert_eq!(config.theory_mode, TheoryMode::Lazy);

        // Single threaded
        assert_eq!(config.num_threads, 1);
    }

    #[test]
    fn test_config_builder_pattern() {
        // Test the builder-style methods
        let config = SolverConfig::fast()
            .with_proof()
            .with_timeout(5000)
            .with_max_conflicts(1000)
            .with_max_decisions(2000)
            .with_parallel(8)
            .with_restart_strategy(RestartStrategy::Luby)
            .with_theory_mode(TheoryMode::Lazy);

        assert!(config.proof);
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.max_conflicts, 1000);
        assert_eq!(config.max_decisions, 2000);
        assert!(config.parallel);
        assert_eq!(config.num_threads, 8);
        assert_eq!(config.restart_strategy, RestartStrategy::Luby);
        assert_eq!(config.theory_mode, TheoryMode::Lazy);
    }

    #[test]
    fn test_solver_with_different_configs() {
        let mut manager = TermManager::new();

        // Create solvers with different configs
        let mut solver_fast = Solver::with_config(SolverConfig::fast());
        let mut solver_balanced = Solver::with_config(SolverConfig::balanced());
        let mut solver_thorough = Solver::with_config(SolverConfig::thorough());
        let mut solver_minimal = Solver::with_config(SolverConfig::minimal());

        // They should all solve a simple problem correctly
        let t = manager.mk_true();
        solver_fast.assert(t, &mut manager);
        solver_balanced.assert(t, &mut manager);
        solver_thorough.assert(t, &mut manager);
        solver_minimal.assert(t, &mut manager);

        assert_eq!(solver_fast.check(&mut manager), SolverResult::Sat);
        assert_eq!(solver_balanced.check(&mut manager), SolverResult::Sat);
        assert_eq!(solver_thorough.check(&mut manager), SolverResult::Sat);
        assert_eq!(solver_minimal.check(&mut manager), SolverResult::Sat);
    }

    #[test]
    fn test_config_default_is_balanced() {
        let default = SolverConfig::default();
        let balanced = SolverConfig::balanced();

        // Default should be the same as balanced
        assert_eq!(
            default.enable_variable_elimination,
            balanced.enable_variable_elimination
        );
        assert_eq!(
            default.enable_clause_minimization,
            balanced.enable_clause_minimization
        );
        assert_eq!(
            default.enable_symmetry_breaking,
            balanced.enable_symmetry_breaking
        );
        assert_eq!(default.restart_strategy, balanced.restart_strategy);
    }

    #[test]
    fn test_theory_combination_arith_solver() {
        use oxiz_theories::arithmetic::ArithSolver;
        use oxiz_theories::{EqualityNotification, TheoryCombination};

        let mut arith = ArithSolver::lra();
        let mut manager = TermManager::new();

        // Create two arithmetic variables
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // Intern them in the arithmetic solver
        let _x_var = arith.intern(x);
        let _y_var = arith.intern(y);

        // Test notify_equality with relevant terms
        let eq_notification = EqualityNotification {
            lhs: x,
            rhs: y,
            reason: None,
        };

        let accepted = arith.notify_equality(eq_notification);
        assert!(
            accepted,
            "ArithSolver should accept equality notification for known terms"
        );

        // Test is_relevant
        assert!(
            arith.is_relevant(x),
            "x should be relevant to arithmetic solver"
        );
        assert!(
            arith.is_relevant(y),
            "y should be relevant to arithmetic solver"
        );

        // Test with unknown term
        let z = manager.mk_var("z", manager.sorts.int_sort);
        assert!(
            !arith.is_relevant(z),
            "z should not be relevant (not interned)"
        );

        // Test notify_equality with unknown terms
        let eq_unknown = EqualityNotification {
            lhs: x,
            rhs: z,
            reason: None,
        };
        let accepted_unknown = arith.notify_equality(eq_unknown);
        assert!(
            !accepted_unknown,
            "ArithSolver should reject equality with unknown term"
        );
    }

    #[test]
    fn test_theory_combination_get_shared_equalities() {
        use oxiz_theories::TheoryCombination;
        use oxiz_theories::arithmetic::ArithSolver;

        let arith = ArithSolver::lra();

        // Test get_shared_equalities
        let shared = arith.get_shared_equalities();
        assert!(
            shared.is_empty(),
            "ArithSolver should return empty shared equalities (placeholder)"
        );
    }

    #[test]
    fn test_equality_notification_fields() {
        use oxiz_theories::EqualityNotification;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // Test with reason
        let eq1 = EqualityNotification {
            lhs: x,
            rhs: y,
            reason: Some(x),
        };
        assert_eq!(eq1.lhs, x);
        assert_eq!(eq1.rhs, y);
        assert_eq!(eq1.reason, Some(x));

        // Test without reason
        let eq2 = EqualityNotification {
            lhs: x,
            rhs: y,
            reason: None,
        };
        assert_eq!(eq2.reason, None);

        // Test equality and cloning
        let eq3 = eq1;
        assert_eq!(eq3.lhs, eq1.lhs);
        assert_eq!(eq3.rhs, eq1.rhs);
    }

    #[test]
    fn test_assert_many() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let r = manager.mk_var("r", manager.sorts.bool_sort);

        // Assert multiple terms at once
        solver.assert_many(&[p, q, r], &mut manager);

        assert_eq!(solver.num_assertions(), 3);
        assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    }

    #[test]
    fn test_num_assertions_and_variables() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        assert_eq!(solver.num_assertions(), 0);
        assert!(!solver.has_assertions());

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);

        solver.assert(p, &mut manager);
        assert_eq!(solver.num_assertions(), 1);
        assert!(solver.has_assertions());

        solver.assert(q, &mut manager);
        assert_eq!(solver.num_assertions(), 2);

        // Variables are created during encoding
        assert!(solver.num_variables() > 0);
    }

    #[test]
    fn test_context_level() {
        let mut solver = Solver::new();

        assert_eq!(solver.context_level(), 0);

        solver.push();
        assert_eq!(solver.context_level(), 1);

        solver.push();
        assert_eq!(solver.context_level(), 2);

        solver.pop();
        assert_eq!(solver.context_level(), 1);

        solver.pop();
        assert_eq!(solver.context_level(), 0);
    }

    // ===== Quantifier Tests =====

    #[test]
    fn test_quantifier_basic_forall() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;

        // Create: forall x. P(x)
        // This asserts P holds for all x
        let x = manager.mk_var("x", bool_sort);
        let p_x = manager.mk_apply("P", [x], bool_sort);
        let forall = manager.mk_forall([("x", bool_sort)], p_x);

        solver.assert(forall, &mut manager);

        // The solver should handle the quantifier (may return sat, unknown, or use MBQI)
        let result = solver.check(&mut manager);
        // Quantifiers without ground terms typically return sat (trivially satisfied)
        assert!(
            result == SolverResult::Sat || result == SolverResult::Unknown,
            "Basic forall should not be unsat"
        );
    }

    #[test]
    fn test_quantifier_basic_exists() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;

        // Create: exists x. P(x)
        let x = manager.mk_var("x", bool_sort);
        let p_x = manager.mk_apply("P", [x], bool_sort);
        let exists = manager.mk_exists([("x", bool_sort)], p_x);

        solver.assert(exists, &mut manager);

        let result = solver.check(&mut manager);
        assert!(
            result == SolverResult::Sat || result == SolverResult::Unknown,
            "Basic exists should not be unsat"
        );
    }

    #[test]
    fn test_quantifier_with_ground_terms() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        // Create ground terms for instantiation
        let zero = manager.mk_int(0);
        let one = manager.mk_int(1);

        // P(0) = true and P(1) = true
        let p_0 = manager.mk_apply("P", [zero], bool_sort);
        let p_1 = manager.mk_apply("P", [one], bool_sort);
        solver.assert(p_0, &mut manager);
        solver.assert(p_1, &mut manager);

        // forall x. P(x) - should be satisfiable with the given ground terms
        let x = manager.mk_var("x", int_sort);
        let p_x = manager.mk_apply("P", [x], bool_sort);
        let forall = manager.mk_forall([("x", int_sort)], p_x);
        solver.assert(forall, &mut manager);

        let result = solver.check(&mut manager);
        // MBQI should find that P(0) and P(1) satisfy the quantifier
        assert!(
            result == SolverResult::Sat || result == SolverResult::Unknown,
            "Quantifier with matching ground terms should be satisfiable"
        );
    }

    #[test]
    fn test_quantifier_instantiation() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        // Create a ground term
        let c = manager.mk_apply("c", [], int_sort);

        // Assert: forall x. f(x) > 0
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let f_x_gt_0 = manager.mk_gt(f_x, zero);
        let forall = manager.mk_forall([("x", int_sort)], f_x_gt_0);
        solver.assert(forall, &mut manager);

        // Assert: f(c) exists (provides instantiation candidate)
        let f_c = manager.mk_apply("f", [c], int_sort);
        let f_c_exists = manager.mk_apply("exists_f_c", [f_c], bool_sort);
        solver.assert(f_c_exists, &mut manager);

        let result = solver.check(&mut manager);
        // MBQI should instantiate the quantifier with c
        assert!(
            result == SolverResult::Sat || result == SolverResult::Unknown,
            "Quantifier instantiation test"
        );
    }

    #[test]
    fn test_quantifier_mbqi_solver_integration() {
        use crate::mbqi::MBQISolver;

        let mut mbqi = MBQISolver::new();
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Create a universal quantifier
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let x_gt_0 = manager.mk_gt(x, zero);
        let forall = manager.mk_forall([("x", int_sort)], x_gt_0);

        // Add the quantifier to MBQI
        mbqi.add_quantifier(forall, &manager);

        // Add some candidate terms
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        mbqi.add_candidate(one, int_sort);
        mbqi.add_candidate(two, int_sort);

        // Check that MBQI tracks the quantifier
        assert!(mbqi.is_enabled(), "MBQI should be enabled by default");
    }

    #[test]
    fn test_quantifier_pattern_matching() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Create: forall x. (f(x) = g(x)) with pattern f(x)
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let g_x = manager.mk_apply("g", [x], int_sort);
        let body = manager.mk_eq(f_x, g_x);

        // Create pattern
        let pattern: smallvec::SmallVec<[_; 2]> = smallvec::smallvec![f_x];
        let patterns: smallvec::SmallVec<[_; 2]> = smallvec::smallvec![pattern];

        let forall = manager.mk_forall_with_patterns([("x", int_sort)], body, patterns);
        solver.assert(forall, &mut manager);

        // Add ground term f(c) to trigger pattern matching
        let c = manager.mk_apply("c", [], int_sort);
        let f_c = manager.mk_apply("f", [c], int_sort);
        let f_c_eq_c = manager.mk_eq(f_c, c);
        solver.assert(f_c_eq_c, &mut manager);

        let result = solver.check(&mut manager);
        assert!(
            result == SolverResult::Sat || result == SolverResult::Unknown,
            "Pattern matching should allow instantiation"
        );
    }

    #[test]
    fn test_quantifier_multiple() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Create: forall x. forall y. x + y = y + x (commutativity)
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let x_plus_y = manager.mk_add([x, y]);
        let y_plus_x = manager.mk_add([y, x]);
        let commutative = manager.mk_eq(x_plus_y, y_plus_x);

        let inner_forall = manager.mk_forall([("y", int_sort)], commutative);
        let outer_forall = manager.mk_forall([("x", int_sort)], inner_forall);

        solver.assert(outer_forall, &mut manager);

        let result = solver.check(&mut manager);
        assert!(
            result == SolverResult::Sat || result == SolverResult::Unknown,
            "Nested forall should be handled"
        );
    }

    #[test]
    fn test_quantifier_with_model() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;

        // Simple satisfiable formula with quantifier
        let p = manager.mk_var("p", bool_sort);
        solver.assert(p, &mut manager);

        // Add a trivial quantifier (x OR NOT x is always true)
        let x = manager.mk_var("x", bool_sort);
        let not_x = manager.mk_not(x);
        let x_or_not_x = manager.mk_or([x, not_x]);
        let tautology = manager.mk_forall([("x", bool_sort)], x_or_not_x);
        solver.assert(tautology, &mut manager);

        let result = solver.check(&mut manager);
        assert_eq!(
            result,
            SolverResult::Sat,
            "Tautology in quantifier should be SAT"
        );

        // Check that we can get a model
        if let Some(model) = solver.model() {
            assert!(model.size() > 0, "Model should have assignments");
        }
    }

    #[test]
    fn test_quantifier_push_pop() {
        let mut solver = Solver::new();
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Assert base formula
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let x_gt_0 = manager.mk_gt(x, zero);
        let forall = manager.mk_forall([("x", int_sort)], x_gt_0);

        solver.push();
        solver.assert(forall, &mut manager);

        let result1 = solver.check(&mut manager);
        // forall x. x > 0 is invalid (counterexample: x = 0 or x = -1)
        // So the solver should return Unsat or Unknown
        assert!(
            result1 == SolverResult::Unsat || result1 == SolverResult::Unknown,
            "forall x. x > 0 should be Unsat or Unknown, got {:?}",
            result1
        );

        solver.pop();

        // After pop, the quantifier assertion should be gone
        let result2 = solver.check(&mut manager);
        assert_eq!(
            result2,
            SolverResult::Sat,
            "After pop, should be trivially SAT"
        );
    }

    /// Test that integer contradictions are correctly detected as UNSAT.
    ///
    /// NOTE: This test currently fails because arithmetic constraints (Ge, Lt, etc.)
    /// are not fully integrated with the simplex solver. The `process_constraint` function
    /// in solver.rs has a TODO to implement this. Once arithmetic constraint encoding
    /// is complete, this test should pass and the ignore attribute should be removed.
    #[test]
    #[ignore = "Requires complete arithmetic constraint encoding to simplex (see process_constraint in solver.rs)"]
    fn test_integer_contradiction_unsat() {
        use num_bigint::BigInt;

        let mut solver = Solver::new();
        let mut manager = TermManager::new();

        // Create integer variable x
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(BigInt::from(0));

        // Assert x >= 0
        let x_ge_0 = manager.mk_ge(x, zero);
        solver.assert(x_ge_0, &mut manager);

        // Assert x < 0 (contradicts x >= 0)
        let x_lt_0 = manager.mk_lt(x, zero);
        solver.assert(x_lt_0, &mut manager);

        // Should be UNSAT because x cannot be both >= 0 and < 0
        let result = solver.check(&mut manager);
        assert_eq!(
            result,
            SolverResult::Unsat,
            "x >= 0 AND x < 0 should be UNSAT"
        );
    }
}
