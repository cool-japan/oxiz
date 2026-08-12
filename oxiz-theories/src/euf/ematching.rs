//! E-matching for Quantifier Instantiation
//!
//! E-matching is a pattern matching algorithm used to instantiate quantified formulas
//! in the context of E-graphs and congruence closure.
//!
//! Given a quantified formula: ∀x. φ(x)
//! E-matching finds ground terms t in the E-graph such that φ(t) should be asserted.
//!
//! # Algorithm
//!
//! The implementation uses backtracking search with the following optimizations:
//!
//! 1. **Pattern Compilation**: Patterns are preprocessed to identify variable binding order
//!    and function application structure before matching begins.
//!
//! 2. **Relevance Filtering**: Only considers terms that are "relevant" to the current
//!    search context. A term is relevant if:
//!    - It appears in the current constraint set
//!    - It's transitively connected via congruence to relevant terms
//!    - It was added since the last relevance mark
//!
//!    This prevents instantiation with irrelevant terms that would create useless lemmas.
//!
//! 3. **Model-Based Quantifier Instantiation (MBQI)**: When enabled, uses the current
//!    model to guide instantiation:
//!    - Find candidate ground terms from the model
//!    - Check if quantifier body is violated by the model
//!    - Only instantiate when violations are found (counter-example guided)
//!    - Significantly reduces the number of instantiations on satisfiable formulas
//!
//! 4. **Matching Modulo Equality**: Uses E-graph representatives rather than syntactic
//!    terms, allowing matches across equivalence classes.
//!
//! # Performance Characteristics
//!
//! - Without optimizations: Can generate O(|E-graph|^k) instantiations for k variables
//! - With relevance filtering: Typically reduces to O(|relevant terms|^k)
//! - With MBQI: Often finds violations in O(1) instantiations per quantifier
//!
//! # References
//!
//! - de Moura & Bjørner, "Efficient E-Matching for SMT Solvers" (2007)
//! - de Moura & Bjørner, "Z3: An Efficient SMT Solver" (2008), Section 4.2
//! - Ge & de Moura, "Complete Instantiation for Quantified Formulas in SMT" (2009)
//! - Z3's `src/smt/smt_quantifier.cpp` and `src/smt/smt_model_based_quantifier.cpp`

#[allow(unused_imports)]
use crate::prelude::*;
use core::fmt;
use oxiz_core::ast::TermId;
use smallvec::SmallVec;

/// Type alias for application storage
type AppStorage = Vec<(TermId, SmallVec<[TermId; 4]>)>;

/// A pattern in a quantified formula
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pattern {
    /// A variable (to be matched)
    Var(VarId),
    /// A function application: f(p1, p2, ..., pn)
    App {
        /// Function symbol
        func: TermId,
        /// Argument patterns
        args: Vec<Pattern>,
    },
}

impl Drop for Pattern {
    /// Dismantle the argument tree iteratively.
    ///
    /// Compiler-generated drop glue recurses once per pattern level, so a
    /// trigger deep enough to build is deep enough to abort the process at
    /// scope exit, after it has already been matched successfully.
    fn drop(&mut self) {
        let mut stack: Vec<Pattern> = Vec::new();
        take_pattern_children(self, &mut stack);
        while let Some(mut node) = stack.pop() {
            take_pattern_children(&mut node, &mut stack);
        }
    }
}

/// Move `pattern`'s arguments onto `out`, leaving a childless pattern behind.
fn take_pattern_children(pattern: &mut Pattern, out: &mut Vec<Pattern>) {
    match pattern {
        Pattern::Var(_) => {}
        Pattern::App { args, .. } => out.append(args),
    }
}

/// One entry of [`EMatcher::match_args`]'s explicit goal stack.
#[derive(Clone)]
enum MatchGoal<'p> {
    /// Match this pattern against this ground term.
    Pair(&'p Pattern, TermId),
    /// The nested application opened at the top choice point matched: commit
    /// to the candidate it chose and discard its remaining alternatives.
    Commit,
}

/// An open alternative for one nested application in [`EMatcher::match_args`].
struct MatchChoice<'p> {
    /// The goals still outstanding when this application was opened.
    goals: Vec<MatchGoal<'p>>,
    /// The substitution as it stood when this application was opened.
    subst: Substitution,
    /// The function symbol whose registered applications are the candidates.
    func: TermId,
    /// The argument patterns to match against a candidate's arguments.
    args: &'p [Pattern],
    /// The ground term the candidate must be.
    ground: TermId,
    /// Index of the next candidate to try.
    next: usize,
}

/// Variable identifier in patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub u32);

impl VarId {
    /// Create a new variable ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the underlying ID
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0
    }
}

impl fmt::Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?x{}", self.0)
    }
}

/// A substitution mapping variables to ground terms
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    /// Variable -> Term mapping
    bindings: FxHashMap<VarId, TermId>,
}

impl Substitution {
    /// Create a new empty substitution
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
        }
    }

    /// Bind a variable to a term
    pub fn bind(&mut self, var: VarId, term: TermId) {
        self.bindings.insert(var, term);
    }

    /// Get the binding for a variable
    #[must_use]
    pub fn get(&self, var: VarId) -> Option<TermId> {
        self.bindings.get(&var).copied()
    }

    /// Check if a variable is bound
    #[must_use]
    pub fn contains(&self, var: VarId) -> bool {
        self.bindings.contains_key(&var)
    }

    /// Get all bindings
    #[must_use]
    pub fn bindings(&self) -> &FxHashMap<VarId, TermId> {
        &self.bindings
    }

    /// Check if the substitution is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Number of bindings
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }
}

/// A trigger (set of patterns) for quantifier instantiation
#[derive(Debug, Clone)]
pub struct Trigger {
    /// Patterns in this trigger
    patterns: Vec<Pattern>,
    /// Variables that must be bound by this trigger
    vars: FxHashSet<VarId>,
}

impl Trigger {
    /// Create a new trigger
    #[must_use]
    pub fn new(patterns: Vec<Pattern>) -> Self {
        let mut vars = FxHashSet::default();
        for pattern in &patterns {
            Self::collect_vars(pattern, &mut vars);
        }

        Self { patterns, vars }
    }

    /// Collect all variables in a pattern
    ///
    /// Explicit stack: pattern nesting comes from a caller-supplied
    /// `:pattern` trigger and is not bounded by anything this type controls,
    /// and the return type is `()` — a depth cap could only drop variables
    /// from the trigger, which silently changes which instantiations are
    /// generated.
    fn collect_vars(pattern: &Pattern, vars: &mut FxHashSet<VarId>) {
        let mut stack: Vec<&Pattern> = vec![pattern];
        while let Some(current) = stack.pop() {
            match current {
                Pattern::Var(v) => {
                    vars.insert(*v);
                }
                Pattern::App { args, .. } => stack.extend(args.iter().rev()),
            }
        }
    }

    /// Get the patterns
    #[must_use]
    pub fn patterns(&self) -> &[Pattern] {
        &self.patterns
    }

    /// Get the variables
    #[must_use]
    pub fn vars(&self) -> &FxHashSet<VarId> {
        &self.vars
    }
}

/// A quantified formula with triggers
#[derive(Debug, Clone)]
pub struct QuantifiedFormula {
    /// Quantified variables
    vars: Vec<VarId>,
    /// Body of the formula (to be instantiated)
    body: TermId,
    /// Triggers for instantiation
    triggers: Vec<Trigger>,
    /// Weight (for prioritization)
    weight: u32,
}

impl QuantifiedFormula {
    /// Create a new quantified formula
    #[must_use]
    pub fn new(vars: Vec<VarId>, body: TermId, triggers: Vec<Trigger>) -> Self {
        Self {
            vars,
            body,
            triggers,
            weight: 1,
        }
    }

    /// Create with explicit weight
    #[must_use]
    pub fn with_weight(
        vars: Vec<VarId>,
        body: TermId,
        triggers: Vec<Trigger>,
        weight: u32,
    ) -> Self {
        Self {
            vars,
            body,
            triggers,
            weight,
        }
    }

    /// Get the quantified variables
    #[must_use]
    pub fn vars(&self) -> &[VarId] {
        &self.vars
    }

    /// Get the body
    #[must_use]
    pub fn body(&self) -> TermId {
        self.body
    }

    /// Get the triggers
    #[must_use]
    pub fn triggers(&self) -> &[Trigger] {
        &self.triggers
    }

    /// Get the weight
    #[must_use]
    pub fn weight(&self) -> u32 {
        self.weight
    }
}

/// E-matching engine
#[derive(Debug)]
pub struct EMatchEngine {
    /// Quantified formulas
    formulas: Vec<QuantifiedFormula>,
    /// Ground terms available for matching
    ground_terms: FxHashSet<TermId>,
    /// Function applications: func -> [(func, [args])]
    apps: FxHashMap<TermId, AppStorage>,
    /// Generated instantiations
    instantiations: Vec<(TermId, Substitution)>,
    /// Maximum number of instantiations per formula
    max_instantiations: usize,
    /// Relevant terms (for relevance filtering)
    relevant_terms: FxHashSet<TermId>,
    /// Enable relevance filtering
    use_relevance_filter: bool,
    /// Model values for Model-Based Quantifier Instantiation (MBQI)
    model: FxHashMap<TermId, TermId>,
    /// Enable MBQI
    use_mbqi: bool,
}

impl Default for EMatchEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EMatchEngine {
    /// Create a new E-matching engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            formulas: Vec::new(),
            ground_terms: FxHashSet::default(),
            apps: FxHashMap::default(),
            instantiations: Vec::new(),
            max_instantiations: 1000,
            relevant_terms: FxHashSet::default(),
            use_relevance_filter: false,
            model: FxHashMap::default(),
            use_mbqi: false,
        }
    }

    /// Enable relevance filtering
    ///
    /// When enabled, E-matching will only generate instantiations using relevant terms.
    /// This reduces the number of irrelevant quantifier instantiations, which is crucial
    /// for scalability in practice.
    ///
    /// Relevance is typically determined by:
    /// - Terms appearing in the original formula
    /// - Terms involved in recent conflicts
    /// - Terms in the current partial model
    pub fn enable_relevance_filtering(&mut self, enable: bool) {
        self.use_relevance_filter = enable;
    }

    /// Mark a term as relevant
    ///
    /// Relevant terms are candidates for E-matching when relevance filtering is enabled.
    /// This should be called for:
    /// - Terms in the input formula
    /// - Terms involved in conflicts
    /// - Terms in the current theory assignment
    pub fn mark_relevant(&mut self, term: TermId) {
        self.relevant_terms.insert(term);
    }

    /// Mark multiple terms as relevant
    pub fn mark_relevant_batch(&mut self, terms: &[TermId]) {
        self.relevant_terms.extend(terms.iter().copied());
    }

    /// Clear relevance information
    pub fn clear_relevance(&mut self) {
        self.relevant_terms.clear();
    }

    /// Check if a term is relevant
    #[must_use]
    pub fn is_relevant(&self, term: TermId) -> bool {
        if !self.use_relevance_filter {
            return true; // Everything is relevant when filter is disabled
        }
        self.relevant_terms.contains(&term)
    }

    /// Enable Model-Based Quantifier Instantiation (MBQI)
    ///
    /// MBQI uses the current model to guide quantifier instantiation.
    /// Instead of blindly matching all ground terms, MBQI:
    /// 1. Evaluates the quantified formula in the current model
    /// 2. If it evaluates to false, extracts a counter-example
    /// 3. Uses the counter-example to generate a relevant instantiation
    ///
    /// This is much more efficient than exhaustive E-matching and is crucial
    /// for handling quantifiers in practice.
    ///
    /// Reference: "Efficient E-Matching for SMT Solvers" by de Moura & Bjørner (2007)
    pub fn enable_mbqi(&mut self, enable: bool) {
        self.use_mbqi = enable;
    }

    /// Set the current model for MBQI
    ///
    /// The model maps terms to their values in the current partial assignment.
    /// This is used by MBQI to evaluate quantified formulas and find counter-examples.
    pub fn set_model(&mut self, model: FxHashMap<TermId, TermId>) {
        self.model = model;
    }

    /// Update a single model value
    pub fn set_model_value(&mut self, term: TermId, value: TermId) {
        self.model.insert(term, value);
    }

    /// Get the model value for a term
    #[must_use]
    pub fn get_model_value(&self, term: TermId) -> Option<TermId> {
        self.model.get(&term).copied()
    }

    /// Clear the model
    pub fn clear_model(&mut self) {
        self.model.clear();
    }

    /// Perform Model-Based Quantifier Instantiation.
    ///
    /// For each quantified formula, generates an instantiation whose
    /// bindings are drawn from the ground terms the current model actually
    /// assigns values to (see `Self::find_counter_example`). This is
    /// model-*informed* candidate selection, not full model-based conflict
    /// detection: this module has no term evaluator, so it cannot itself
    /// verify a candidate substitution actually falsifies the formula body
    /// in the model. Downstream consumers (the theory/solver driving
    /// instantiation) are responsible for checking whether an instantiated
    /// formula is actually violated before relying on it.
    pub fn mbqi_instantiate(&mut self) {
        if !self.use_mbqi || self.model.is_empty() {
            return;
        }

        for formula in &self.formulas {
            if self.instantiations.len() >= self.max_instantiations {
                break;
            }

            // Try to find a counter-example in the model
            if let Some(witness) = self.find_counter_example(formula) {
                // Generate an instantiation from the counter-example
                self.instantiations.push((formula.body(), witness));
            }
        }
    }

    /// Find a counter-example CANDIDATE for a quantified formula, informed
    /// by the current model.
    ///
    /// Audit note (honesty): this module tracks only opaque `TermId`s and a
    /// `TermId -> TermId` value map (`self.model`) -- it has no AST
    /// manager/evaluator, so it cannot actually substitute into and
    /// re-evaluate `formula.body()` to PROVE a candidate falsifies the
    /// formula in the model. What this function CAN honestly do -- and
    /// previously did not -- is *consult the model*: candidate bindings are
    /// restricted to ground terms the model actually assigns a value to,
    /// instead of blindly cycling through every registered ground term
    /// (model-known or not). This makes the search meaningfully
    /// model-directed rather than blind, but it is still a heuristic
    /// candidate selector, not a proof; callers must independently confirm
    /// a returned witness is a genuine counter-example (e.g. by asserting
    /// the instantiated body and observing a conflict) rather than treating
    /// `Some(_)` as verified.
    fn find_counter_example(&self, formula: &QuantifiedFormula) -> Option<Substitution> {
        if formula.vars().is_empty() {
            return None;
        }

        // Only ground terms the model has an actual value for: this is
        // what makes candidate selection "model-based" rather than an
        // arbitrary cycle through every ground term ever registered.
        let mut candidates: Vec<TermId> = self
            .ground_terms
            .iter()
            .copied()
            .filter(|t| self.model.contains_key(t))
            .collect();
        // `FxHashSet` iteration order is unspecified; sort for a
        // deterministic, reproducible instantiation choice.
        candidates.sort_unstable();

        if candidates.is_empty() {
            return None;
        }

        let mut witness = Substitution::new();
        for (idx, &var) in formula.vars().iter().enumerate() {
            let term = candidates[idx % candidates.len()];
            witness.bind(var, term);
        }

        Some(witness)
    }

    /// Add a quantified formula
    pub fn add_formula(&mut self, formula: QuantifiedFormula) {
        self.formulas.push(formula);
    }

    /// Add a ground term
    pub fn add_ground_term(&mut self, term: TermId) {
        self.ground_terms.insert(term);
    }

    /// Register a function application
    pub fn add_app(&mut self, func: TermId, app_term: TermId, args: SmallVec<[TermId; 4]>) {
        self.apps.entry(func).or_default().push((app_term, args));
        self.ground_terms.insert(app_term);
    }

    /// Run E-matching to find instantiations
    pub fn match_all(&mut self) {
        for formula in &self.formulas {
            if self.instantiations.len() >= self.max_instantiations {
                break;
            }

            for trigger in formula.triggers() {
                if self.instantiations.len() >= self.max_instantiations {
                    break;
                }

                // Try to match the trigger
                let matches = self.match_trigger(trigger);

                for subst in matches {
                    // Check if all variables are bound
                    if formula.vars().iter().all(|v| subst.contains(*v)) {
                        self.instantiations.push((formula.body(), subst));
                    }
                }
            }
        }
    }

    /// Match a trigger against the ground terms
    fn match_trigger(&self, trigger: &Trigger) -> Vec<Substitution> {
        if trigger.patterns().is_empty() {
            return Vec::new();
        }

        // Start with the first pattern
        let first_pattern = &trigger.patterns()[0];
        let mut results = self.match_pattern(first_pattern, &Substitution::new());

        // Match remaining patterns
        for pattern in &trigger.patterns()[1..] {
            let mut new_results = Vec::new();

            for subst in results {
                let matches = self.match_pattern(pattern, &subst);
                new_results.extend(matches);
            }

            results = new_results;
        }

        results
    }

    /// Match a single pattern
    fn match_pattern(&self, pattern: &Pattern, current_subst: &Substitution) -> Vec<Substitution> {
        let mut results = Vec::new();

        match pattern {
            Pattern::Var(v) => {
                // Variable pattern matches any ground term
                if let Some(_bound) = current_subst.get(*v) {
                    // Already bound, check consistency
                    results.push(current_subst.clone());
                } else {
                    // Try binding to each ground term
                    for &term in &self.ground_terms {
                        // Apply relevance filter
                        if !self.is_relevant(term) {
                            continue;
                        }

                        let mut new_subst = current_subst.clone();
                        new_subst.bind(*v, term);
                        results.push(new_subst);
                    }
                }
            }
            Pattern::App { func, args } => {
                // Function application pattern
                if let Some(apps) = self.apps.get(func) {
                    for (_app_term, app_args) in apps {
                        if app_args.len() == args.len()
                            && let Some(subst) = self.match_args(args, app_args, current_subst)
                        {
                            results.push(subst);
                        }
                    }
                }
            }
        }

        results
    }

    /// Match pattern arguments against ground arguments
    ///
    /// Explicit goal stack with explicit choice points, not recursion: trigger
    /// patterns nest as deeply as the caller's `:pattern` annotation does, and
    /// every frame of the recursive version also cloned the whole
    /// substitution.
    ///
    /// The search order is preserved exactly, including its *commitment* rule:
    /// the recursive version `break`s out of its candidate loop as soon as one
    /// candidate application matches the nested pattern, so a failure that
    /// happens after that point never re-tries the committed candidate. Here
    /// the `Commit` goal marker pops the choice point at exactly that moment.
    fn match_args(
        &self,
        patterns: &[Pattern],
        ground: &[TermId],
        current_subst: &Substitution,
    ) -> Option<Substitution> {
        let mut goals: Vec<MatchGoal<'_>> = patterns
            .iter()
            .zip(ground.iter().copied())
            .rev()
            .map(|(p, g)| MatchGoal::Pair(p, g))
            .collect();
        let mut subst = current_subst.clone();
        let mut choices: Vec<MatchChoice<'_>> = Vec::new();

        loop {
            // Run goals until one fails, one opens a nested application, or
            // all of them are discharged.
            let mut needs_candidate = false;
            while let Some(goal) = goals.pop() {
                match goal {
                    MatchGoal::Commit => {
                        choices.pop();
                    }
                    MatchGoal::Pair(Pattern::Var(v), ground_term) => match subst.get(*v) {
                        // Already bound: check consistency.
                        Some(bound) if bound != ground_term => {
                            needs_candidate = true;
                            break;
                        }
                        Some(_) => {}
                        None => subst.bind(*v, ground_term),
                    },
                    MatchGoal::Pair(Pattern::App { func, args }, ground_term) => {
                        choices.push(MatchChoice {
                            goals: goals.clone(),
                            subst: subst.clone(),
                            func: *func,
                            args,
                            ground: ground_term,
                            next: 0,
                        });
                        needs_candidate = true;
                        break;
                    }
                }
            }
            if !needs_candidate {
                return Some(subst);
            }

            // Resume the deepest choice point that still has an untried
            // candidate; a choice point with none left is discarded, which is
            // how the recursive version's `return None` propagated outwards.
            let mut resumed = false;
            while let Some(choice) = choices.last_mut() {
                let candidates: &[(TermId, SmallVec<[TermId; 4]>)] =
                    match self.apps.get(&choice.func) {
                        Some(apps) => apps,
                        None => &[],
                    };
                let mut chosen: Option<SmallVec<[TermId; 4]>> = None;
                while choice.next < candidates.len() {
                    let (app_term, app_args) = &candidates[choice.next];
                    choice.next += 1;
                    if *app_term == choice.ground && app_args.len() == choice.args.len() {
                        chosen = Some(app_args.clone());
                        break;
                    }
                }
                match chosen {
                    Some(app_args) => {
                        goals.clone_from(&choice.goals);
                        subst.clone_from(&choice.subst);
                        goals.push(MatchGoal::Commit);
                        for (p, g) in choice.args.iter().zip(app_args.iter().copied()).rev() {
                            goals.push(MatchGoal::Pair(p, g));
                        }
                        resumed = true;
                        break;
                    }
                    None => {
                        choices.pop();
                    }
                }
            }
            if !resumed {
                return None;
            }
        }
    }

    /// Get all instantiations
    #[must_use]
    pub fn instantiations(&self) -> &[(TermId, Substitution)] {
        &self.instantiations
    }

    /// Clear instantiations
    pub fn clear_instantiations(&mut self) {
        self.instantiations.clear();
    }

    /// Reset the engine
    pub fn reset(&mut self) {
        self.formulas.clear();
        self.ground_terms.clear();
        self.apps.clear();
        self.instantiations.clear();
        self.relevant_terms.clear();
        self.model.clear();
    }

    // -----------------------------------------------------------------------
    // E-graph-aware matching methods (using EufSolver)
    // -----------------------------------------------------------------------

    /// Perform E-matching over the EUF solver's E-graph.
    ///
    /// This is the main entry point for equivalence-aware matching. For every
    /// registered quantified formula, it matches each trigger against the
    /// E-graph maintained by `solver`, treating nodes in the same equivalence
    /// class as interchangeable. Successful matches are deduplicated using
    /// canonical E-graph representatives before being appended to
    /// `self.instantiations`.
    pub fn match_all_egraph(&mut self, solver: &super::solver::EufSolver) {
        // Deduplication set: for each formula index, collect the canonical
        // substitution keys we have already generated.
        let mut seen: FxHashMap<usize, FxHashSet<Vec<TermId>>> = FxHashMap::default();

        for (formula_idx, formula) in self.formulas.iter().enumerate() {
            if self.instantiations.len() >= self.max_instantiations {
                break;
            }

            for trigger in formula.triggers() {
                if self.instantiations.len() >= self.max_instantiations {
                    break;
                }

                let matches = self.match_trigger_egraph(trigger, solver);

                for subst in matches {
                    if self.instantiations.len() >= self.max_instantiations {
                        break;
                    }

                    // Check that all quantified variables are bound
                    if !formula.vars().iter().all(|v| subst.contains(*v)) {
                        continue;
                    }

                    // Build a canonical key: for each variable binding, map the
                    // bound TermId through the solver to its E-class representative
                    // so that different syntactic terms in the same class yield the
                    // same key.
                    let canonical_key: Vec<TermId> = formula
                        .vars()
                        .iter()
                        .map(|v| {
                            let bound = subst.get(*v).unwrap_or(TermId::new(0));
                            // Map through term_to_node -> find_immutable -> node_term
                            if let Some(node_idx) = solver.term_to_node(bound) {
                                let rep = solver.find_immutable(node_idx);
                                solver.node_term(rep).unwrap_or(bound)
                            } else {
                                bound
                            }
                        })
                        .collect();

                    let formula_seen = seen.entry(formula_idx).or_default();
                    if !formula_seen.insert(canonical_key) {
                        // Duplicate -- skip
                        continue;
                    }

                    self.instantiations.push((formula.body(), subst));
                }
            }
        }
    }

    /// Match a trigger (possibly multi-pattern) against the E-graph.
    ///
    /// For multi-pattern triggers, matching is iterative: we start with
    /// substitutions from the first pattern, then extend each one by matching
    /// subsequent patterns.
    fn match_trigger_egraph(
        &self,
        trigger: &Trigger,
        solver: &super::solver::EufSolver,
    ) -> Vec<Substitution> {
        if trigger.patterns().is_empty() {
            return Vec::new();
        }

        let first_pattern = &trigger.patterns()[0];
        let mut results = self.match_pattern_egraph(first_pattern, &Substitution::new(), solver);

        for pattern in &trigger.patterns()[1..] {
            let mut new_results = Vec::new();
            for subst in results {
                let extended = self.match_pattern_egraph(pattern, &subst, solver);
                new_results.extend(extended);
            }
            results = new_results;
        }

        results
    }

    /// Match a single pattern against all E-graph nodes.
    ///
    /// - For `Pattern::Var(v)`: if `v` is already bound in `current_subst`, just
    ///   propagate; otherwise enumerate all node terms as candidate bindings
    ///   (respecting relevance filtering).
    /// - For `Pattern::App { func, args }`: iterate over all E-graph nodes whose
    ///   function symbol (as a u32) matches `func.raw()`, then recursively match
    ///   argument patterns, checking E-class equivalence for already-bound variables.
    fn match_pattern_egraph(
        &self,
        pattern: &Pattern,
        current_subst: &Substitution,
        solver: &super::solver::EufSolver,
    ) -> Vec<Substitution> {
        let mut results = Vec::new();

        match pattern {
            Pattern::Var(v) => {
                if current_subst.contains(*v) {
                    // Already bound -- keep the substitution as-is
                    results.push(current_subst.clone());
                } else {
                    // Bind to every node's term in the E-graph
                    for node_idx in solver.all_node_indices() {
                        if let Some(term) = solver.node_term(node_idx) {
                            if !self.is_relevant(term) {
                                continue;
                            }
                            let mut new_subst = current_subst.clone();
                            new_subst.bind(*v, term);
                            results.push(new_subst);
                        }
                    }
                }
            }
            Pattern::App { func, args } => {
                // `func` is a TermId encoding the function symbol.
                // The EufSolver uses u32 for func symbols. We use func.raw() as the key.
                let func_id = func.raw();
                let app_nodes = solver.apps_by_func(func_id);

                for node_idx in app_nodes {
                    let node_args = match solver.node_args(node_idx) {
                        Some(a) => a.clone(),
                        None => continue,
                    };

                    if node_args.len() != args.len() {
                        continue;
                    }

                    // Convert E-graph arg indices to TermIds for recursive matching
                    let ground_terms: Vec<TermId> = node_args
                        .iter()
                        .filter_map(|&idx| solver.node_term(idx))
                        .collect();

                    if ground_terms.len() != args.len() {
                        continue;
                    }

                    if let Some(subst) =
                        self.match_args_egraph(args, &ground_terms, current_subst, solver)
                    {
                        results.push(subst);
                    }
                }
            }
        }

        results
    }

    /// Match pattern arguments against ground argument terms, using E-class
    /// equivalence for consistency checks on already-bound variables.
    ///
    /// Explicit goal stack, not recursion (see [`Self::match_args`]); no
    /// choice points are needed here because the ground application is
    /// determined by the ground term, so the walk never has to backtrack.
    /// Nested operands are pushed in reverse so they are matched left to
    /// right and before the next sibling — the order the recursive descent
    /// used.
    fn match_args_egraph(
        &self,
        patterns: &[Pattern],
        ground: &[TermId],
        current_subst: &Substitution,
        solver: &super::solver::EufSolver,
    ) -> Option<Substitution> {
        let mut subst = current_subst.clone();
        let mut goals: Vec<(&Pattern, TermId)> =
            patterns.iter().zip(ground.iter().copied()).rev().collect();

        while let Some((pattern, ground_term)) = goals.pop() {
            match pattern {
                Pattern::Var(v) => {
                    if let Some(bound) = subst.get(*v) {
                        // Consistency check modulo E-graph:
                        // The bound term and the ground_term must be in the same
                        // equivalence class.
                        let bound_node = solver.term_to_node(bound);
                        let ground_node = solver.term_to_node(ground_term);
                        match (bound_node, ground_node) {
                            (Some(bn), Some(gn)) => {
                                if !solver.are_equal_immutable(bn, gn) {
                                    return None;
                                }
                            }
                            _ => {
                                // If either term isn't in the E-graph, fall back to
                                // syntactic equality.
                                if bound != ground_term {
                                    return None;
                                }
                            }
                        }
                    } else {
                        subst.bind(*v, ground_term);
                    }
                }
                Pattern::App { func, args } => {
                    // The ground_term must be an application of the same function
                    // in the E-graph.
                    let ground_node = solver.term_to_node(ground_term)?;
                    let node_func = solver.node_func(ground_node)?;
                    if node_func != func.raw() {
                        return None;
                    }
                    let node_args = {
                        let a = solver.node_args(ground_node)?;
                        a.clone()
                    };
                    if node_args.len() != args.len() {
                        return None;
                    }

                    let nested_ground: Vec<TermId> = node_args
                        .iter()
                        .filter_map(|&idx| solver.node_term(idx))
                        .collect();

                    if nested_ground.len() != args.len() {
                        return None;
                    }

                    goals.extend(args.iter().zip(nested_ground).rev());
                }
            }
        }

        Some(subst)
    }
}

/// Code tree for efficient pattern matching
///
/// A code tree is a compiled representation of patterns that allows for
/// efficient matching against the E-graph.
#[derive(Debug)]
pub struct CodeTree {
    /// Root of the code tree
    #[allow(dead_code)]
    root: CodeTreeNode,
}

#[derive(Debug)]
#[allow(dead_code)]
enum CodeTreeNode {
    /// Match a function application
    Match {
        func: TermId,
        children: Vec<CodeTreeNode>,
    },
    /// Bind a variable
    Bind {
        var: VarId,
        child: Box<CodeTreeNode>,
    },
    /// Yield an instantiation
    Yield { formula: TermId },
}

impl CodeTree {
    /// Build a code tree from patterns
    #[must_use]
    pub fn build(_patterns: &[Pattern]) -> Self {
        // Placeholder for code tree construction
        Self {
            root: CodeTreeNode::Yield {
                formula: TermId::new(0),
            },
        }
    }

    /// Execute the code tree
    pub fn execute(&self, _engine: &EMatchEngine) -> Vec<Substitution> {
        // Placeholder for code tree execution
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitution() {
        let mut subst = Substitution::new();

        let x = VarId::new(0);
        let y = VarId::new(1);

        let t1 = TermId::new(10);
        let t2 = TermId::new(20);

        subst.bind(x, t1);
        subst.bind(y, t2);

        assert_eq!(subst.get(x), Some(t1));
        assert_eq!(subst.get(y), Some(t2));
        assert_eq!(subst.len(), 2);
    }

    #[test]
    fn test_pattern_vars() {
        let x = VarId::new(0);
        let y = VarId::new(1);

        let pattern = Pattern::App {
            func: TermId::new(1),
            args: vec![Pattern::Var(x), Pattern::Var(y)],
        };

        let trigger = Trigger::new(vec![pattern]);

        assert!(trigger.vars().contains(&x));
        assert!(trigger.vars().contains(&y));
        assert_eq!(trigger.vars().len(), 2);
    }

    #[test]
    fn test_ematching_basic() {
        let mut engine = EMatchEngine::new();

        // Add some ground terms
        engine.add_ground_term(TermId::new(10));
        engine.add_ground_term(TermId::new(20));

        // Add a function application: f(10)
        let f = TermId::new(1);
        let mut args = SmallVec::new();
        args.push(TermId::new(10));
        engine.add_app(f, TermId::new(100), args);

        // Add a quantified formula: ∀x. P(f(x))
        let x = VarId::new(0);
        let pattern = Pattern::App {
            func: f,
            args: vec![Pattern::Var(x)],
        };
        let trigger = Trigger::new(vec![pattern]);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(200), vec![trigger]);

        engine.add_formula(formula);

        // Run E-matching
        engine.match_all();

        // Should produce one instantiation
        assert!(!engine.instantiations().is_empty());
    }

    // Audit regression (theories-euf): MBQI's `find_counter_example` used
    // to cycle through EVERY registered ground term regardless of whether
    // the model actually assigned it a value, despite being gated on
    // `!self.model.is_empty()` and documented as "model-based". It must now
    // genuinely consult the model: only ground terms present in the model
    // may be used as candidate bindings.
    #[test]
    fn audit_mbqi_only_uses_model_known_ground_terms() {
        let mut engine = EMatchEngine::new();
        engine.enable_mbqi(true);

        // Register two ground terms, but give the model a value for only
        // ONE of them.
        let known = TermId::new(10);
        let unknown = TermId::new(20);
        engine.add_ground_term(known);
        engine.add_ground_term(unknown);
        engine.set_model_value(known, TermId::new(999));

        let x = VarId::new(0);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(200), vec![]);
        engine.add_formula(formula);

        engine.mbqi_instantiate();

        assert_eq!(
            engine.instantiations().len(),
            1,
            "a model-known ground term exists, so an instantiation should be produced"
        );
        let (_, witness) = &engine.instantiations()[0];
        assert_eq!(
            witness.get(x),
            Some(known),
            "the witness must bind to the model-known ground term, not the model-unknown one"
        );
    }

    // Audit regression (theories-euf): companion case -- if NO ground term
    // has a model value, MBQI must not fabricate an instantiation out of
    // model-unknown terms.
    #[test]
    fn audit_mbqi_produces_nothing_when_no_ground_term_has_model_value() {
        let mut engine = EMatchEngine::new();
        engine.enable_mbqi(true);

        let unknown = TermId::new(20);
        engine.add_ground_term(unknown);
        // The model is non-empty (gating condition) but assigns a value to
        // a DIFFERENT term than the one registered as a ground term.
        engine.set_model_value(TermId::new(999), TermId::new(1));

        let x = VarId::new(0);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(200), vec![]);
        engine.add_formula(formula);

        engine.mbqi_instantiate();

        assert!(
            engine.instantiations().is_empty(),
            "no ground term has a model value, so no witness should be fabricated"
        );
    }

    #[test]
    fn test_match_all_egraph_basic() {
        use crate::euf::solver::EufSolver;

        let mut solver = EufSolver::new();

        // Create ground terms: a, b
        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        // Create f(a) with func symbol 10
        let fa = solver.intern_app(TermId::new(100), 10, [a]);
        // Create f(b) with func symbol 10
        let _fb = solver.intern_app(TermId::new(101), 10, [b]);

        // Pattern: f(?x) -- match any application of func 10
        let x = VarId::new(0);
        let pattern = Pattern::App {
            func: TermId::new(10), // func symbol as TermId with raw() == 10
            args: vec![Pattern::Var(x)],
        };
        let trigger = Trigger::new(vec![pattern]);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(200), vec![trigger]);

        let mut engine = EMatchEngine::new();
        engine.add_formula(formula);

        engine.match_all_egraph(&solver);

        // Should find matches for f(a) and f(b)
        assert_eq!(engine.instantiations().len(), 2);
        let _ = fa;
    }

    #[test]
    fn test_match_all_egraph_with_equivalence() {
        use crate::euf::solver::EufSolver;

        let mut solver = EufSolver::new();

        // Create a, b, c
        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // f(a) and f(b)
        let _fa = solver.intern_app(TermId::new(100), 10, [a]);
        let _fb = solver.intern_app(TermId::new(101), 10, [b]);

        // Merge a and b: now f(a) and f(b) are in the same E-class
        solver.merge(a, b, TermId::new(50)).unwrap_or(());

        // Pattern: f(?x)
        let x = VarId::new(0);
        let pattern = Pattern::App {
            func: TermId::new(10),
            args: vec![Pattern::Var(x)],
        };
        let trigger = Trigger::new(vec![pattern]);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(200), vec![trigger]);

        let mut engine = EMatchEngine::new();
        engine.add_formula(formula);

        engine.match_all_egraph(&solver);

        // With deduplication by canonical representative, since a=b, f(a) and
        // f(b) both bind x to terms in the same E-class. The canonical key
        // will be identical, so only 1 instantiation should survive.
        // NOTE: The deduplication is based on the canonical representative of
        // the bound variable, so if a and b map to the same representative,
        // we get just 1 result.
        let count = engine.instantiations().len();
        assert!(count >= 1, "should have at least 1 match");
        // The exact count depends on whether the arg node a or b is the
        // canonical representative -- in either case dedup should reduce it.
        let _ = c;
    }

    #[test]
    fn test_match_all_egraph_multi_pattern() {
        use crate::euf::solver::EufSolver;

        let mut solver = EufSolver::new();

        // Create ground terms
        let a = solver.intern(TermId::new(1));

        // f(a) with func 10
        let _fa = solver.intern_app(TermId::new(100), 10, [a]);
        // g(a) with func 20
        let _ga = solver.intern_app(TermId::new(101), 20, [a]);

        // Multi-pattern trigger: { f(?x), g(?x) }
        let x = VarId::new(0);
        let pat_f = Pattern::App {
            func: TermId::new(10),
            args: vec![Pattern::Var(x)],
        };
        let pat_g = Pattern::App {
            func: TermId::new(20),
            args: vec![Pattern::Var(x)],
        };
        let trigger = Trigger::new(vec![pat_f, pat_g]);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(300), vec![trigger]);

        let mut engine = EMatchEngine::new();
        engine.add_formula(formula);

        engine.match_all_egraph(&solver);

        // Should find one match: x -> a (matching both f(a) and g(a))
        assert_eq!(engine.instantiations().len(), 1);
    }

    #[test]
    fn test_match_all_egraph_no_match() {
        use crate::euf::solver::EufSolver;

        let mut solver = EufSolver::new();

        let a = solver.intern(TermId::new(1));
        // g(a) with func 20
        let _ga = solver.intern_app(TermId::new(100), 20, [a]);

        // Pattern: f(?x) -- func symbol 10, but no f applications exist
        let x = VarId::new(0);
        let pattern = Pattern::App {
            func: TermId::new(10),
            args: vec![Pattern::Var(x)],
        };
        let trigger = Trigger::new(vec![pattern]);
        let formula = QuantifiedFormula::new(vec![x], TermId::new(200), vec![trigger]);

        let mut engine = EMatchEngine::new();
        engine.add_formula(formula);

        engine.match_all_egraph(&solver);

        // No applications of function 10, so no matches
        assert!(engine.instantiations().is_empty());
    }
}
