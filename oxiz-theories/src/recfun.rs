//! Recursive Function Theory
//!
//! A **registry** for recursive function definitions and the applications seen
//! of them, with push/pop scoping. Concretely it provides:
//!
//! - a function-definition registry keyed by interned name;
//! - application tracking, with a per-application depth counter and a
//!   configurable ceiling ([`RecFunConfig::max_depth`]) above which an
//!   application stops being recorded;
//! - push/pop context management over the recorded applications.
//!
//! It does **not** unroll anything. The doc comment here used to advertise
//! "depth-limited unrolling", which was never true of this file: no method
//! instantiates a definition, substitutes an argument, or emits a lemma.
//! Unfolding the definitional axiom `forall x. f(x) = body` — and deciding what
//! verdict a bounded unfolding justifies — is driven from the solver context
//! (`oxiz_solver::context::recfun`), which owns the term manager and the solver
//! scopes that instantiation needs.

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::TermId;
use oxiz_core::error::OxizError;
use oxiz_core::interner::Spur;
use oxiz_core::{Result, SortId};

/// Unique identifier for a recursive function definition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecFunId(usize);

impl RecFunId {
    /// Create a new RecFunId
    pub fn new(id: usize) -> Self {
        RecFunId(id)
    }

    /// Get the inner ID
    pub fn id(self) -> usize {
        self.0
    }
}

/// One case of a recursive function definition
#[derive(Debug, Clone)]
pub struct CaseDef {
    /// Guards (conditions) for this case
    pub guards: Vec<TermId>,
    /// Right-hand side (body) of this case
    pub rhs: TermId,
    /// Whether this case is "immediate" (contains no recursive calls)
    pub is_immediate: bool,
    /// Case predicate symbol (C_f_i)
    pub predicate: Spur,
}

impl CaseDef {
    /// Create a new case definition
    pub fn new(guards: Vec<TermId>, rhs: TermId, predicate: Spur) -> Self {
        Self {
            guards,
            rhs,
            is_immediate: false,
            predicate,
        }
    }

    /// The single, unguarded case of a definition written as one plain body —
    /// the shape `(define-fun-rec f ((x S)) R body)` produces.
    ///
    /// `guards` stays empty: the case applies to every argument tuple, which is
    /// exactly what the SMT-LIB surface syntax says. The multi-case form (a
    /// definition split into guarded branches with one case predicate each, so
    /// that only the relevant branch has to be instantiated) is what `guards`
    /// and [`CaseDef::predicate`] are reserved for; nothing constructs it yet,
    /// and a single unguarded case is the honest encoding of a plain body
    /// rather than a placeholder for one.
    pub fn plain(rhs: TermId, predicate: Spur) -> Self {
        Self::new(Vec::new(), rhs, predicate)
    }
}

/// Definition of a recursive function
#[derive(Debug, Clone)]
pub struct RecFunDef {
    /// Unique identifier
    pub id: RecFunId,
    /// Function name
    pub name: Spur,
    /// Parameter names
    pub params: Vec<Spur>,
    /// Parameter sorts
    pub param_sorts: Vec<SortId>,
    /// Return sort
    pub return_sort: SortId,
    /// Cases (pattern match branches)
    pub cases: Vec<CaseDef>,
    /// Whether this is a macro (single immediate case)
    pub is_macro: bool,
}

impl RecFunDef {
    /// Create a new recursive function definition
    pub fn new(
        id: RecFunId,
        name: Spur,
        params: Vec<Spur>,
        param_sorts: Vec<SortId>,
        return_sort: SortId,
        cases: Vec<CaseDef>,
    ) -> Self {
        let is_macro = cases.len() == 1 && cases[0].is_immediate;
        Self {
            id,
            name,
            params,
            param_sorts,
            return_sort,
            cases,
            is_macro,
        }
    }

    /// Get the arity (number of parameters)
    pub fn arity(&self) -> usize {
        self.params.len()
    }

    /// Check if this is a macro definition
    pub fn is_macro(&self) -> bool {
        self.is_macro
    }
}

/// Statistics for recursive function solving
#[derive(Debug, Clone, Default)]
pub struct RecFunStats {
    /// Number of case expansions
    pub case_expansions: usize,
    /// Number of body expansions
    pub body_expansions: usize,
    /// Number of macro expansions
    pub macro_expansions: usize,
    /// Number of depth limit hits
    pub depth_limits: usize,
}

impl RecFunStats {
    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Configuration for recursive function solving
#[derive(Debug, Clone)]
pub struct RecFunConfig {
    /// Maximum unrolling depth
    pub max_depth: usize,
    /// Enable immediate case optimization
    pub enable_immediate: bool,
    /// Enable macro expansion
    pub enable_macros: bool,
}

impl Default for RecFunConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            enable_immediate: true,
            enable_macros: true,
        }
    }
}

/// Application of a recursive function
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FunApp {
    /// Function ID
    fun_id: RecFunId,
    /// Function symbol
    func: Spur,
    /// Arguments
    args: Vec<TermId>,
}

/// Recursive function solver
pub struct RecFunSolver {
    /// Configuration
    config: RecFunConfig,
    /// Statistics
    stats: RecFunStats,
    /// Function definitions by ID
    defs: HashMap<RecFunId, RecFunDef>,
    /// Function name to ID mapping
    name_to_id: HashMap<Spur, RecFunId>,
    /// Next function ID
    next_id: usize,
    /// Seen function applications
    seen_apps: HashSet<FunApp>,
    /// Applications in the order they were first recorded.
    ///
    /// `seen_apps` alone cannot support `pop`: a `HashSet` has no insertion
    /// order, so "the entries added since the push" is not a question it can
    /// answer. This trail is that answer, and it is why [`RecFunSolver::pop`]
    /// retracts exactly the applications the popped scope introduced.
    app_trail: Vec<FunApp>,
    /// Current depth of each function application
    app_depth: HashMap<FunApp, usize>,
    /// Context stack for push/pop: the `app_trail` length at each push.
    context_stack: Vec<usize>,
}

impl RecFunSolver {
    /// Create a new recursive function solver
    pub fn new() -> Self {
        Self::with_config(RecFunConfig::default())
    }

    /// Create a new solver with custom configuration
    pub fn with_config(config: RecFunConfig) -> Self {
        Self {
            config,
            stats: RecFunStats::default(),
            defs: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: 0,
            seen_apps: HashSet::new(),
            app_trail: Vec::new(),
            app_depth: HashMap::new(),
            context_stack: Vec::new(),
        }
    }

    /// Define a new recursive function
    pub fn define_function(
        &mut self,
        name: Spur,
        params: Vec<Spur>,
        param_sorts: Vec<SortId>,
        return_sort: SortId,
        cases: Vec<CaseDef>,
    ) -> Result<RecFunId> {
        if self.name_to_id.contains_key(&name) {
            return Err(OxizError::Internal("Function already defined".to_string()));
        }

        let id = RecFunId::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| OxizError::Internal("RecFunId overflow".to_string()))?;

        let def = RecFunDef::new(id, name, params, param_sorts, return_sort, cases);
        self.defs.insert(id, def);
        self.name_to_id.insert(name, id);

        Ok(id)
    }

    /// Get a function definition by ID
    pub fn get_def(&self, id: RecFunId) -> Option<&RecFunDef> {
        self.defs.get(&id)
    }

    /// Get a function definition by name
    pub fn get_def_by_name(&self, name: Spur) -> Option<&RecFunDef> {
        self.name_to_id.get(&name).and_then(|id| self.defs.get(id))
    }

    /// Check if a function is defined
    pub fn is_defined(&self, name: Spur) -> bool {
        self.name_to_id.contains_key(&name)
    }

    /// Mark a function application as seen (for expansion tracking)
    pub fn register_application(&mut self, func: Spur, args: Vec<TermId>) -> Result<()> {
        let fun_id = match self.name_to_id.get(&func) {
            Some(&id) => id,
            None => return Ok(()), // Not a recursive function
        };

        let app = FunApp { fun_id, func, args };

        // Check depth limit
        let depth = self.app_depth.get(&app).copied().unwrap_or(0);
        if depth >= self.config.max_depth {
            self.stats.depth_limits = self.stats.depth_limits.saturating_add(1);
            return Ok(());
        }

        if self.seen_apps.insert(app.clone()) {
            self.app_trail.push(app);
        }
        Ok(())
    }

    /// Push a new context level
    pub fn push(&mut self) {
        self.context_stack.push(self.app_trail.len());
    }

    /// Pop a context level, retracting exactly the applications recorded since
    /// the matching [`RecFunSolver::push`].
    ///
    /// This used to read `seen_apps.iter().skip(size)`, which is wrong twice
    /// over: a `HashSet` iterates in an unspecified, hash-dependent order, so
    /// `skip(size)` names an **arbitrary** subset of the applications rather
    /// than the ones the scope introduced — retracting applications from an
    /// enclosing scope (they then look unseen, and get re-registered and
    /// re-counted) while keeping ones the scope owned (they survive their own
    /// retraction, and a later re-entry into the scope silently skips them).
    /// The insertion-ordered `app_trail` is what makes the retraction exact.
    pub fn pop(&mut self) {
        if let Some(mark) = self.context_stack.pop() {
            while self.app_trail.len() > mark {
                if let Some(app) = self.app_trail.pop() {
                    self.seen_apps.remove(&app);
                    self.app_depth.remove(&app);
                }
            }
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.seen_apps.clear();
        self.app_trail.clear();
        self.app_depth.clear();
        self.context_stack.clear();
        self.stats.reset();
    }

    /// Get statistics
    pub fn stats(&self) -> &RecFunStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &RecFunConfig {
        &self.config
    }
}

impl Default for RecFunSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::interner::Key;

    #[test]
    fn test_recfun_id() {
        let id1 = RecFunId::new(0);
        let id2 = RecFunId::new(1);
        assert_ne!(id1, id2);
        assert_eq!(id1.id(), 0);
        assert_eq!(id2.id(), 1);
    }

    #[test]
    fn test_case_def_creation() {
        let guards = vec![];
        let rhs = TermId::new(1);
        let predicate = Spur::try_from_usize(0).expect("valid spur");

        let case = CaseDef::new(guards, rhs, predicate);
        assert!(!case.is_immediate);
        assert_eq!(case.guards.len(), 0);
    }

    #[test]
    fn test_recfun_solver_creation() {
        let solver = RecFunSolver::new();
        assert_eq!(solver.stats().case_expansions, 0);
        assert_eq!(solver.stats().body_expansions, 0);
        assert_eq!(solver.stats().macro_expansions, 0);
    }

    #[test]
    fn test_recfun_config() {
        let config = RecFunConfig::default();
        assert_eq!(config.max_depth, 10);
        assert!(config.enable_immediate);
        assert!(config.enable_macros);
    }

    #[test]
    fn test_push_pop() {
        let mut solver = RecFunSolver::new();
        solver.push();
        solver.push();
        solver.pop();
        solver.pop();
    }

    /// A solver with a definition registered and nothing else.
    fn solver_with_one_definition() -> (RecFunSolver, Spur) {
        let mut solver = RecFunSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let predicate = Spur::try_from_usize(1).expect("valid spur");
        solver
            .define_function(
                name,
                vec![Spur::try_from_usize(2).expect("valid spur")],
                vec![SortId::new(0)],
                SortId::new(0),
                vec![CaseDef::plain(TermId::new(0), predicate)],
            )
            .expect("definition should register");
        (solver, name)
    }

    /// `pop` must retract exactly the applications recorded since the matching
    /// `push` — no more (an enclosing scope's application would look unseen and
    /// be re-counted) and no fewer (a retracted scope's application would
    /// survive, so re-entering the scope would silently skip it).
    ///
    /// The old implementation took `seen_apps.iter().skip(n)`, which names an
    /// arbitrary hash-ordered subset. Thirty-two outer applications are enough
    /// that the set's iteration order differs from the insertion order, so this
    /// test fails against it.
    #[test]
    fn pop_retracts_exactly_the_applications_of_the_popped_scope() {
        let (mut solver, name) = solver_with_one_definition();

        for i in 0..32u32 {
            solver
                .register_application(name, vec![TermId::new(i)])
                .expect("registration should succeed");
        }
        let outer = solver.seen_apps.clone();
        assert_eq!(outer.len(), 32);

        solver.push();
        for i in 100..116u32 {
            solver
                .register_application(name, vec![TermId::new(i)])
                .expect("registration should succeed");
        }
        assert_eq!(solver.seen_apps.len(), 48, "the scope adds 16 applications");

        solver.pop();
        assert_eq!(
            solver.seen_apps, outer,
            "pop must restore exactly the pre-push application set"
        );
        assert_eq!(solver.app_trail.len(), 32, "the trail shrinks with the set");
        assert!(
            solver.app_depth.keys().all(|app| outer.contains(app)),
            "no depth entry may outlive its application"
        );
    }

    /// Re-registering an application already seen must not grow the trail, or
    /// `pop` would retract it once per registration and lose an enclosing
    /// scope's entry.
    #[test]
    fn repeated_registration_does_not_double_count_the_trail() {
        let (mut solver, name) = solver_with_one_definition();
        solver.push();
        for _ in 0..5 {
            solver
                .register_application(name, vec![TermId::new(7)])
                .expect("registration should succeed");
        }
        assert_eq!(solver.app_trail.len(), 1);
        solver.pop();
        assert!(solver.seen_apps.is_empty());
        assert!(solver.app_trail.is_empty());
    }

    #[test]
    fn test_define_function() {
        let mut solver = RecFunSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let params = vec![];
        let param_sorts = vec![];
        let return_sort = SortId::new(0);
        let cases = vec![CaseDef::new(
            vec![],
            TermId::new(0),
            Spur::try_from_usize(1).expect("valid spur"),
        )];

        let id = solver.define_function(name, params, param_sorts, return_sort, cases);
        assert!(id.is_ok());
        assert!(solver.is_defined(name));
    }
}
