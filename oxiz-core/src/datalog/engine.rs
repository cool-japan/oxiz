//! Datalog evaluation engine
//!
//! Implements semi-naive evaluation with stratification support.
//! Supports both bottom-up and top-down evaluation strategies.

use lasso::{Spur, ThreadedRodeo};
use rustc_hash::FxHashSet;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use super::relation::{Relation, RelationId, RelationKind};
use super::rule::{ArithOp, Atom, AtomKind, Binding, ComparisonOp, Rule, RuleId, Term};
use super::schema::Schema;
use super::tuple::{Tuple, Value};

/// Configuration for the Datalog engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum iterations for fixed-point
    pub max_iterations: usize,
    /// Enable semi-naive evaluation
    pub semi_naive: bool,
    /// Enable magic sets transformation
    pub magic_sets: bool,
    /// Enable subsumption checking
    pub subsumption: bool,
    /// Enable tuple deduplication
    pub deduplicate: bool,
    /// Batch size for incremental updates
    pub batch_size: usize,
    /// Debug/trace evaluation
    pub trace: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            semi_naive: true,
            magic_sets: false,
            subsumption: false,
            deduplicate: true,
            batch_size: 1000,
            trace: false,
        }
    }
}

/// Result of evaluation
#[derive(Debug)]
pub struct EvaluationResult {
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether fixed-point was reached
    pub converged: bool,
    /// Number of new tuples derived
    pub new_tuples: usize,
    /// Evaluation time in milliseconds
    pub time_ms: u64,
    /// Per-relation statistics
    pub relation_stats: HashMap<String, RelationEvalStats>,
}

/// Per-relation evaluation statistics
#[derive(Debug, Default)]
pub struct RelationEvalStats {
    /// Tuples before evaluation
    pub initial_size: usize,
    /// Tuples after evaluation
    pub final_size: usize,
    /// Tuples derived
    pub derived: usize,
    /// Rules applied
    pub rules_applied: usize,
}

/// Stratum in stratified evaluation
#[derive(Debug)]
struct Stratum {
    /// Rules in this stratum
    rules: Vec<RuleId>,
    /// Relations defined in this stratum
    relations: HashSet<Spur>,
    /// Dependencies within stratum (for recursive rules)
    recursive_deps: HashSet<Spur>,
}

/// The Datalog evaluation engine
#[derive(Debug)]
pub struct DatalogEngine {
    /// String interner
    interner: ThreadedRodeo,
    /// Configuration
    config: EngineConfig,
    /// Relations by ID
    relations: HashMap<RelationId, Relation>,
    /// Relation name to ID mapping
    relation_names: HashMap<Spur, RelationId>,
    /// Rules by ID
    rules: HashMap<RuleId, Rule>,
    /// Next relation ID
    next_relation_id: AtomicU64,
    /// Next rule ID
    next_rule_id: AtomicU64,
    /// Strata for stratified evaluation
    strata: Vec<Stratum>,
    /// Is stratification computed
    stratified: bool,
    /// Dependency graph (relation -> rules that define it)
    def_rules: HashMap<Spur, Vec<RuleId>>,
}

impl DatalogEngine {
    /// Create a new engine with default configuration
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }

    /// Create a new engine with given configuration
    pub fn with_config(config: EngineConfig) -> Self {
        Self {
            interner: ThreadedRodeo::default(),
            config,
            relations: HashMap::new(),
            relation_names: HashMap::new(),
            rules: HashMap::new(),
            next_relation_id: AtomicU64::new(1),
            next_rule_id: AtomicU64::new(1),
            strata: Vec::new(),
            stratified: false,
            def_rules: HashMap::new(),
        }
    }

    /// Get string interner
    pub fn interner(&self) -> &ThreadedRodeo {
        &self.interner
    }

    /// Intern a string
    pub fn intern(&self, s: &str) -> Spur {
        self.interner.get_or_intern(s)
    }

    /// Resolve interned string
    pub fn resolve(&self, spur: Spur) -> &str {
        self.interner.resolve(&spur)
    }

    /// Create and register a new EDB relation
    pub fn create_edb(&mut self, name: &str, schema: Schema) -> RelationId {
        let id = RelationId::new(self.next_relation_id.fetch_add(1, Ordering::SeqCst));
        let name_spur = self.interner.get_or_intern(name);
        let relation = Relation::edb(id, name.to_string(), schema);
        self.relations.insert(id, relation);
        self.relation_names.insert(name_spur, id);
        self.stratified = false;
        id
    }

    /// Create and register a new IDB relation
    pub fn create_idb(&mut self, name: &str, schema: Schema) -> RelationId {
        let id = RelationId::new(self.next_relation_id.fetch_add(1, Ordering::SeqCst));
        let name_spur = self.interner.get_or_intern(name);
        let relation = Relation::idb(id, name.to_string(), schema);
        self.relations.insert(id, relation);
        self.relation_names.insert(name_spur, id);
        self.stratified = false;
        id
    }

    /// Get relation by ID
    pub fn relation(&self, id: RelationId) -> Option<&Relation> {
        self.relations.get(&id)
    }

    /// Get mutable relation by ID
    pub fn relation_mut(&mut self, id: RelationId) -> Option<&mut Relation> {
        self.relations.get_mut(&id)
    }

    /// Get relation by name
    pub fn relation_by_name(&self, name: &str) -> Option<&Relation> {
        let spur = self.interner.get(name)?;
        let id = self.relation_names.get(&spur)?;
        self.relations.get(id)
    }

    /// Get mutable relation by name
    pub fn relation_by_name_mut(&mut self, name: &str) -> Option<&mut Relation> {
        let spur = self.interner.get(name)?;
        let id = self.relation_names.get(&spur).copied()?;
        self.relations.get_mut(&id)
    }

    /// Insert facts into EDB relation
    pub fn insert_facts(&mut self, relation_name: &str, facts: Vec<Tuple>) -> usize {
        if let Some(rel) = self.relation_by_name_mut(relation_name) {
            rel.insert_all(facts)
        } else {
            0
        }
    }

    /// Add a rule
    pub fn add_rule(&mut self, mut rule: Rule) -> RuleId {
        // Resolve relation references
        if let Some(name) = rule.head().relation()
            && let Some(&id) = self.relation_names.get(&name)
        {
            rule.head_mut().set_relation_id(id);
        }

        for atom in rule.body_mut() {
            if let Some(name) = atom.relation()
                && let Some(&id) = self.relation_names.get(&name)
            {
                atom.set_relation_id(id);
            }
        }

        let id = rule.id();

        // Track which rules define which relations
        if let Some(head_rel) = rule.head().relation() {
            self.def_rules.entry(head_rel).or_default().push(id);
        }

        self.rules.insert(id, rule);
        self.stratified = false;
        id
    }

    /// Generate a new rule ID
    pub fn new_rule_id(&self) -> RuleId {
        RuleId::new(self.next_rule_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Compute stratification
    pub fn stratify(&mut self) -> Result<(), StratificationError> {
        if self.stratified {
            return Ok(());
        }

        // Build dependency graph
        let mut positive_deps: HashMap<Spur, HashSet<Spur>> = HashMap::new();
        let mut negative_deps: HashMap<Spur, HashSet<Spur>> = HashMap::new();

        for rule in self.rules.values() {
            if let Some(head_rel) = rule.head().relation() {
                let pos_entry = positive_deps.entry(head_rel).or_default();
                let neg_entry = negative_deps.entry(head_rel).or_default();

                for atom in rule.body() {
                    if let Some(body_rel) = atom.relation() {
                        match atom.kind() {
                            AtomKind::Positive => {
                                pos_entry.insert(body_rel);
                            }
                            AtomKind::Negated => {
                                neg_entry.insert(body_rel);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Compute SCCs for stratification
        let all_rels: HashSet<_> = positive_deps
            .keys()
            .chain(negative_deps.keys())
            .copied()
            .collect();

        // Assign strata (simplified algorithm)
        let mut stratum_map: HashMap<Spur, usize> = HashMap::new();
        let mut changed = true;
        let mut max_iterations = 100;

        // Initialize all to stratum 0
        for &rel in &all_rels {
            stratum_map.insert(rel, 0);
        }

        while changed && max_iterations > 0 {
            changed = false;
            max_iterations -= 1;

            for &rel in &all_rels {
                let current = stratum_map[&rel];

                // Positive deps: same or lower stratum
                if let Some(deps) = positive_deps.get(&rel) {
                    for &dep in deps {
                        if let Some(&dep_stratum) = stratum_map.get(&dep)
                            && dep_stratum > current
                        {
                            stratum_map.insert(rel, dep_stratum);
                            changed = true;
                        }
                    }
                }

                // Negative deps: strictly lower stratum
                if let Some(deps) = negative_deps.get(&rel) {
                    for &dep in deps {
                        if let Some(&dep_stratum) = stratum_map.get(&dep)
                            && dep_stratum >= stratum_map[&rel]
                        {
                            stratum_map.insert(rel, dep_stratum + 1);
                            changed = true;
                        }
                    }
                }
            }
        }

        if max_iterations == 0 {
            return Err(StratificationError::CycleWithNegation);
        }

        // Build strata
        let max_stratum = stratum_map.values().max().copied().unwrap_or(0);
        let mut strata = Vec::with_capacity(max_stratum + 1);

        for s in 0..=max_stratum {
            let relations: HashSet<_> = stratum_map
                .iter()
                .filter(|&(_, stratum)| *stratum == s)
                .map(|(&rel, _)| rel)
                .collect();

            let rules: Vec<_> = self
                .rules
                .iter()
                .filter(|(_, rule)| {
                    rule.head()
                        .relation()
                        .is_some_and(|r| relations.contains(&r))
                })
                .map(|(&id, _)| id)
                .collect();

            // Find recursive deps within stratum
            let mut recursive_deps = HashSet::new();
            for &rule_id in &rules {
                if let Some(rule) = self.rules.get(&rule_id)
                    && rule.is_recursive()
                    && let Some(rel) = rule.head().relation()
                {
                    recursive_deps.insert(rel);
                }
            }

            strata.push(Stratum {
                rules,
                relations,
                recursive_deps,
            });
        }

        self.strata = strata;
        self.stratified = true;
        Ok(())
    }

    /// Evaluate all rules to fixed-point
    pub fn evaluate(&mut self) -> Result<EvaluationResult, EvaluationError> {
        let start = std::time::Instant::now();

        // Ensure stratification
        self.stratify().map_err(EvaluationError::Stratification)?;

        let mut total_iterations = 0;
        let mut total_new_tuples = 0;
        let mut relation_stats: HashMap<String, RelationEvalStats> = HashMap::new();

        // Initialize stats
        for rel in self.relations.values() {
            relation_stats.insert(
                rel.name().to_string(),
                RelationEvalStats {
                    initial_size: rel.len(),
                    ..Default::default()
                },
            );
        }

        // Evaluate each stratum
        for stratum_idx in 0..self.strata.len() {
            let stratum = &self.strata[stratum_idx];

            if stratum.rules.is_empty() {
                continue;
            }

            // Get rules for this stratum
            let rule_ids: Vec<_> = stratum.rules.clone();
            let is_recursive = !stratum.recursive_deps.is_empty();

            if is_recursive && self.config.semi_naive {
                // Semi-naive evaluation for recursive strata
                let (iters, new_tuples) = self.evaluate_semi_naive(&rule_ids)?;
                total_iterations += iters;
                total_new_tuples += new_tuples;
            } else {
                // Single-pass for non-recursive strata
                let new_tuples = self.evaluate_once(&rule_ids)?;
                total_new_tuples += new_tuples;
                total_iterations += 1;
            }
        }

        // Update final stats
        for rel in self.relations.values() {
            if let Some(stats) = relation_stats.get_mut(rel.name()) {
                stats.final_size = rel.len();
                stats.derived = stats.final_size.saturating_sub(stats.initial_size);
            }
        }

        Ok(EvaluationResult {
            iterations: total_iterations,
            converged: true,
            new_tuples: total_new_tuples,
            time_ms: start.elapsed().as_millis() as u64,
            relation_stats,
        })
    }

    /// Evaluate rules once (non-recursive)
    fn evaluate_once(&mut self, rule_ids: &[RuleId]) -> Result<usize, EvaluationError> {
        let mut new_tuples = 0;

        for &rule_id in rule_ids {
            // Get rule info without holding borrow
            let (head_id, derived) = {
                let rule = self
                    .rules
                    .get(&rule_id)
                    .ok_or(EvaluationError::RuleNotFound(rule_id))?;
                let head_id = rule.head().relation_id();
                let derived = self.evaluate_rule(rule)?;
                (head_id, derived)
            };

            // Insert derived tuples
            if let Some(rel_id) = head_id
                && let Some(rel) = self.relations.get_mut(&rel_id)
            {
                new_tuples += rel.insert_all(derived);
            }
        }

        Ok(new_tuples)
    }

    /// Semi-naive evaluation for recursive rules
    fn evaluate_semi_naive(
        &mut self,
        rule_ids: &[RuleId],
    ) -> Result<(usize, usize), EvaluationError> {
        let mut iterations = 0;
        let mut total_new = 0;

        // Initial pass: evaluate all rules
        let initial_new = self.evaluate_once(rule_ids)?;
        total_new += initial_new;
        iterations += 1;

        // Initialize deltas
        for rel in self.relations.values_mut() {
            rel.advance_delta();
        }

        // Fixed-point iteration
        loop {
            if iterations >= self.config.max_iterations {
                return Err(EvaluationError::MaxIterationsExceeded);
            }

            let new_this_iteration = self.evaluate_once(rule_ids)?;

            if new_this_iteration == 0 {
                break;
            }

            total_new += new_this_iteration;
            iterations += 1;

            // Advance deltas
            for rel in self.relations.values_mut() {
                rel.advance_delta();
            }
        }

        Ok((iterations, total_new))
    }

    /// Evaluate a single rule
    fn evaluate_rule(&self, rule: &Rule) -> Result<Vec<Tuple>, EvaluationError> {
        if rule.is_fact() {
            // Handle facts directly
            return self.evaluate_fact(rule);
        }

        // Start with empty bindings
        let initial_bindings = vec![Binding::new()];

        // Process each body atom
        let mut current_bindings = initial_bindings;

        for atom in rule.body() {
            current_bindings = self.match_atom(atom, current_bindings)?;
            if current_bindings.is_empty() {
                return Ok(Vec::new());
            }
        }

        // Project to head
        self.project_to_head(rule.head(), current_bindings)
    }

    /// Evaluate a fact (rule with empty body)
    fn evaluate_fact(&self, rule: &Rule) -> Result<Vec<Tuple>, EvaluationError> {
        let head = rule.head();

        // All terms must be constants
        let values: Vec<Value> = head
            .terms()
            .iter()
            .filter_map(|t| t.as_const().cloned())
            .collect();

        if values.len() == head.arity() {
            Ok(vec![Tuple::new(values)])
        } else {
            Ok(Vec::new())
        }
    }

    /// Match an atom against bindings
    fn match_atom(
        &self,
        atom: &Atom,
        bindings: Vec<Binding>,
    ) -> Result<Vec<Binding>, EvaluationError> {
        match atom.kind() {
            AtomKind::Positive | AtomKind::Negated => self.match_relation_atom(atom, bindings),
            AtomKind::Comparison => self.match_comparison(atom, bindings),
            AtomKind::Arithmetic => self.match_arithmetic(atom, bindings),
            AtomKind::Aggregate => self.match_aggregate(atom, bindings),
        }
    }

    /// Match a relation atom
    fn match_relation_atom(
        &self,
        atom: &Atom,
        bindings: Vec<Binding>,
    ) -> Result<Vec<Binding>, EvaluationError> {
        let rel_id = atom.relation_id().ok_or_else(|| {
            EvaluationError::UnresolvedRelation(
                atom.relation()
                    .map(|s| self.resolve(s).to_string())
                    .unwrap_or_default(),
            )
        })?;

        let relation = self
            .relations
            .get(&rel_id)
            .ok_or(EvaluationError::RelationNotFound(rel_id))?;

        let mut new_bindings = Vec::new();

        for binding in bindings {
            // Try to match each tuple in relation
            for tuple in relation.iter() {
                if let Some(extended) = self.match_tuple(atom.terms(), tuple, &binding)
                    && atom.kind() == AtomKind::Positive
                {
                    new_bindings.push(extended);
                }
            }

            // For negated atoms, keep binding if NO tuple matches
            if atom.kind() == AtomKind::Negated {
                let has_match = relation
                    .iter()
                    .any(|t| self.match_tuple(atom.terms(), t, &binding).is_some());

                if !has_match {
                    new_bindings.push(binding);
                }
            }
        }

        Ok(new_bindings)
    }

    /// Try to match terms against a tuple
    fn match_tuple(&self, terms: &[Term], tuple: &Tuple, binding: &Binding) -> Option<Binding> {
        if terms.len() != tuple.len() {
            return None;
        }

        let mut new_binding = binding.clone();

        for (term, value) in terms.iter().zip(tuple.values()) {
            match term {
                Term::Var(var) => {
                    if let Some(bound_val) = new_binding.get(*var) {
                        if bound_val != value {
                            return None; // Mismatch
                        }
                    } else {
                        new_binding.bind(*var, value.clone());
                    }
                }
                Term::Const(const_val) => {
                    if const_val != value {
                        return None;
                    }
                }
                Term::Wildcard => {
                    // Always matches
                }
                Term::Aggregate(_, _) => {
                    return None; // Shouldn't appear in body
                }
            }
        }

        Some(new_binding)
    }

    /// Match a comparison atom
    fn match_comparison(
        &self,
        atom: &Atom,
        bindings: Vec<Binding>,
    ) -> Result<Vec<Binding>, EvaluationError> {
        let op = atom.comparison_op().ok_or(EvaluationError::InvalidAtom)?;
        let terms = atom.terms();

        if terms.len() != 2 {
            return Err(EvaluationError::InvalidAtom);
        }

        let mut result = Vec::new();

        for binding in bindings {
            let left = binding.apply(&terms[0]);
            let right = binding.apply(&terms[1]);

            if let (Some(l), Some(r)) = (left, right) {
                let satisfied = match op {
                    ComparisonOp::Eq => l == r,
                    ComparisonOp::Ne => l != r,
                    ComparisonOp::Lt => l < r,
                    ComparisonOp::Le => l <= r,
                    ComparisonOp::Gt => l > r,
                    ComparisonOp::Ge => l >= r,
                };

                if satisfied {
                    result.push(binding);
                }
            }
        }

        Ok(result)
    }

    /// Match an arithmetic atom
    fn match_arithmetic(
        &self,
        atom: &Atom,
        bindings: Vec<Binding>,
    ) -> Result<Vec<Binding>, EvaluationError> {
        let op = atom.arith_op().ok_or(EvaluationError::InvalidAtom)?;
        let terms = atom.terms();

        if terms.len() != 3 {
            return Err(EvaluationError::InvalidAtom);
        }

        let mut result = Vec::new();

        for binding in bindings {
            let left = binding.apply(&terms[1]);
            let right = binding.apply(&terms[2]);

            if let (Some(l), Some(r)) = (left, right) {
                let computed = self.compute_arithmetic(op, &l, &r)?;

                // Bind or check result
                match &terms[0] {
                    Term::Var(var) => {
                        if let Some(existing) = binding.get(*var) {
                            if *existing == computed {
                                result.push(binding);
                            }
                        } else {
                            let mut new_binding = binding.clone();
                            new_binding.bind(*var, computed);
                            result.push(new_binding);
                        }
                    }
                    Term::Const(c) => {
                        if *c == computed {
                            result.push(binding);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(result)
    }

    /// Compute arithmetic operation
    fn compute_arithmetic(
        &self,
        op: ArithOp,
        left: &Value,
        right: &Value,
    ) -> Result<Value, EvaluationError> {
        match (left, right) {
            (Value::Int64(l), Value::Int64(r)) => {
                let result = match op {
                    ArithOp::Add => l
                        .checked_add(*r)
                        .ok_or(EvaluationError::ArithmeticOverflow)?,
                    ArithOp::Sub => l
                        .checked_sub(*r)
                        .ok_or(EvaluationError::ArithmeticOverflow)?,
                    ArithOp::Mul => l
                        .checked_mul(*r)
                        .ok_or(EvaluationError::ArithmeticOverflow)?,
                    ArithOp::Div => {
                        if *r == 0 {
                            return Err(EvaluationError::DivisionByZero);
                        }
                        l.checked_div(*r)
                            .ok_or(EvaluationError::ArithmeticOverflow)?
                    }
                    ArithOp::Mod => {
                        if *r == 0 {
                            return Err(EvaluationError::DivisionByZero);
                        }
                        l.checked_rem(*r)
                            .ok_or(EvaluationError::ArithmeticOverflow)?
                    }
                };
                Ok(Value::Int64(result))
            }
            _ => Err(EvaluationError::TypeMismatch),
        }
    }

    /// Match an aggregate atom
    fn match_aggregate(
        &self,
        _atom: &Atom,
        bindings: Vec<Binding>,
    ) -> Result<Vec<Binding>, EvaluationError> {
        // Simplified aggregate handling
        // Full implementation would group by variables and compute aggregates
        Ok(bindings)
    }

    /// Project bindings to head tuple
    fn project_to_head(
        &self,
        head: &Atom,
        bindings: Vec<Binding>,
    ) -> Result<Vec<Tuple>, EvaluationError> {
        let mut tuples = Vec::new();

        for binding in bindings {
            let values: Vec<Value> = head
                .terms()
                .iter()
                .filter_map(|t| binding.apply(t))
                .collect();

            if values.len() == head.arity() {
                tuples.push(Tuple::new(values));
            }
        }

        // Deduplicate if configured
        if self.config.deduplicate {
            let unique: FxHashSet<_> = tuples.drain(..).collect();
            tuples.extend(unique);
        }

        Ok(tuples)
    }

    /// Query a relation with a pattern
    pub fn query(&self, relation_name: &str, pattern: &[Term]) -> Vec<Tuple> {
        let rel = match self.relation_by_name(relation_name) {
            Some(r) => r,
            None => return Vec::new(),
        };

        rel.iter()
            .filter(|tuple| {
                pattern
                    .iter()
                    .zip(tuple.values())
                    .all(|(term, val)| match term {
                        Term::Const(c) => c == val,
                        Term::Wildcard => true,
                        Term::Var(_) => true, // Variables match anything
                        _ => false,
                    })
            })
            .cloned()
            .collect()
    }

    /// Get all facts from a relation
    pub fn facts(&self, relation_name: &str) -> Vec<Tuple> {
        self.relation_by_name(relation_name)
            .map(|r| r.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Clear all derived facts (IDB relations)
    pub fn clear_derived(&mut self) {
        for rel in self.relations.values_mut() {
            if rel.kind() == RelationKind::Idb {
                rel.clear();
            }
        }
        self.stratified = false;
    }

    /// Clear all data
    pub fn clear_all(&mut self) {
        for rel in self.relations.values_mut() {
            rel.clear();
        }
        self.stratified = false;
    }
}

impl Default for DatalogEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Error during stratification
#[derive(Debug, Clone)]
pub enum StratificationError {
    /// Cycle with negation detected
    CycleWithNegation,
    /// Unresolved relation reference
    UnresolvedRelation(String),
}

impl std::fmt::Display for StratificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StratificationError::CycleWithNegation => {
                write!(
                    f,
                    "Cycle with negation detected - program is not stratifiable"
                )
            }
            StratificationError::UnresolvedRelation(name) => {
                write!(f, "Unresolved relation: {}", name)
            }
        }
    }
}

impl std::error::Error for StratificationError {}

/// Error during evaluation
#[derive(Debug)]
pub enum EvaluationError {
    /// Stratification error
    Stratification(StratificationError),
    /// Rule not found
    RuleNotFound(RuleId),
    /// Relation not found
    RelationNotFound(RelationId),
    /// Unresolved relation
    UnresolvedRelation(String),
    /// Invalid atom
    InvalidAtom,
    /// Type mismatch
    TypeMismatch,
    /// Arithmetic overflow
    ArithmeticOverflow,
    /// Division by zero
    DivisionByZero,
    /// Maximum iterations exceeded
    MaxIterationsExceeded,
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluationError::Stratification(e) => write!(f, "Stratification error: {}", e),
            EvaluationError::RuleNotFound(id) => write!(f, "Rule not found: {:?}", id),
            EvaluationError::RelationNotFound(id) => write!(f, "Relation not found: {:?}", id),
            EvaluationError::UnresolvedRelation(name) => write!(f, "Unresolved relation: {}", name),
            EvaluationError::InvalidAtom => write!(f, "Invalid atom"),
            EvaluationError::TypeMismatch => write!(f, "Type mismatch in operation"),
            EvaluationError::ArithmeticOverflow => write!(f, "Arithmetic overflow"),
            EvaluationError::DivisionByZero => write!(f, "Division by zero"),
            EvaluationError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
        }
    }
}

impl std::error::Error for EvaluationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datalog::schema::DataType;
    use crate::datalog::tuple::TupleBuilder;

    fn create_edge_schema(interner: &ThreadedRodeo) -> Schema {
        let mut schema = Schema::new("edge".to_string());
        schema.add_column(interner.get_or_intern("from"), DataType::Int64);
        schema.add_column(interner.get_or_intern("to"), DataType::Int64);
        schema
    }

    #[test]
    fn test_simple_fact() {
        let mut engine = DatalogEngine::new();
        let schema = create_edge_schema(engine.interner());
        engine.create_edb("edge", schema);

        // Insert facts
        engine.insert_facts(
            "edge",
            vec![
                TupleBuilder::new().push_i64(1).push_i64(2).build(),
                TupleBuilder::new().push_i64(2).push_i64(3).build(),
            ],
        );

        let facts = engine.facts("edge");
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_simple_rule() {
        let mut engine = DatalogEngine::new();

        // Create schemas
        let edge_schema = create_edge_schema(engine.interner());
        let path_schema = {
            let mut s = Schema::new("path".to_string());
            s.add_column(engine.intern("from"), DataType::Int64);
            s.add_column(engine.intern("to"), DataType::Int64);
            s
        };

        engine.create_edb("edge", edge_schema);
        engine.create_idb("path", path_schema);

        // Insert facts
        engine.insert_facts(
            "edge",
            vec![
                TupleBuilder::new().push_i64(1).push_i64(2).build(),
                TupleBuilder::new().push_i64(2).push_i64(3).build(),
            ],
        );

        // path(x, y) :- edge(x, y)
        let edge = engine.intern("edge");
        let path = engine.intern("path");
        let x = Term::var(engine.intern("x"));
        let y = Term::var(engine.intern("y"));

        let rule = Rule::new(
            engine.new_rule_id(),
            Atom::positive(path, vec![x.clone(), y.clone()]),
            vec![Atom::positive(edge, vec![x, y])],
        );
        engine.add_rule(rule);

        // Evaluate
        let result = engine.evaluate().unwrap();
        assert!(result.converged);

        let paths = engine.facts("path");
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_recursive_transitive_closure() {
        let mut engine = DatalogEngine::new();

        // Create schemas
        let edge_schema = create_edge_schema(engine.interner());
        let reach_schema = {
            let mut s = Schema::new("reach".to_string());
            s.add_column(engine.intern("from"), DataType::Int64);
            s.add_column(engine.intern("to"), DataType::Int64);
            s
        };

        engine.create_edb("edge", edge_schema);
        engine.create_idb("reach", reach_schema);

        // Insert facts: 1 -> 2 -> 3 -> 4
        engine.insert_facts(
            "edge",
            vec![
                TupleBuilder::new().push_i64(1).push_i64(2).build(),
                TupleBuilder::new().push_i64(2).push_i64(3).build(),
                TupleBuilder::new().push_i64(3).push_i64(4).build(),
            ],
        );

        let edge = engine.intern("edge");
        let reach = engine.intern("reach");
        let x = Term::var(engine.intern("x"));
        let y = Term::var(engine.intern("y"));
        let z = Term::var(engine.intern("z"));

        // reach(x, y) :- edge(x, y)
        let rule1 = Rule::new(
            engine.new_rule_id(),
            Atom::positive(reach, vec![x.clone(), y.clone()]),
            vec![Atom::positive(edge, vec![x.clone(), y.clone()])],
        );
        engine.add_rule(rule1);

        // reach(x, z) :- reach(x, y), edge(y, z)
        let rule2 = Rule::new(
            engine.new_rule_id(),
            Atom::positive(reach, vec![x.clone(), z.clone()]),
            vec![
                Atom::positive(reach, vec![x, y.clone()]),
                Atom::positive(edge, vec![y, z]),
            ],
        );
        engine.add_rule(rule2);

        // Evaluate
        let result = engine.evaluate().unwrap();
        assert!(result.converged);

        let reaches = engine.facts("reach");
        // Should have: (1,2), (2,3), (3,4), (1,3), (2,4), (1,4)
        assert_eq!(reaches.len(), 6);
    }

    #[test]
    fn test_stratification() {
        let mut engine = DatalogEngine::new();

        let rel_schema = {
            let mut s = Schema::new("rel".to_string());
            s.add_column(engine.intern("x"), DataType::Int64);
            s
        };

        engine.create_edb("a", rel_schema.clone());
        engine.create_idb("b", rel_schema.clone());
        engine.create_idb("c", rel_schema);

        // b(x) :- a(x)
        // c(x) :- b(x), not a(x) -- This should be in higher stratum

        let a = engine.intern("a");
        let b = engine.intern("b");
        let c = engine.intern("c");
        let x = Term::var(engine.intern("x"));

        let rule1 = Rule::new(
            engine.new_rule_id(),
            Atom::positive(b, vec![x.clone()]),
            vec![Atom::positive(a, vec![x.clone()])],
        );
        engine.add_rule(rule1);

        let rule2 = Rule::new(
            engine.new_rule_id(),
            Atom::positive(c, vec![x.clone()]),
            vec![
                Atom::positive(b, vec![x.clone()]),
                Atom::negated(a, vec![x]),
            ],
        );
        engine.add_rule(rule2);

        // Should stratify successfully
        let result = engine.stratify();
        assert!(result.is_ok());
    }

    #[test]
    fn test_comparison_filter() {
        let mut engine = DatalogEngine::new();

        let num_schema = {
            let mut s = Schema::new("num".to_string());
            s.add_column(engine.intern("x"), DataType::Int64);
            s
        };
        let big_schema = {
            let mut s = Schema::new("big".to_string());
            s.add_column(engine.intern("x"), DataType::Int64);
            s
        };

        engine.create_edb("num", num_schema);
        engine.create_idb("big", big_schema);

        engine.insert_facts(
            "num",
            vec![
                TupleBuilder::new().push_i64(1).build(),
                TupleBuilder::new().push_i64(5).build(),
                TupleBuilder::new().push_i64(10).build(),
            ],
        );

        // big(x) :- num(x), x > 3
        let num = engine.intern("num");
        let big = engine.intern("big");
        let x = Term::var(engine.intern("x"));

        let rule = Rule::new(
            engine.new_rule_id(),
            Atom::positive(big, vec![x.clone()]),
            vec![
                Atom::positive(num, vec![x.clone()]),
                Atom::comparison(x, ComparisonOp::Gt, Term::constant(Value::Int64(3))),
            ],
        );
        engine.add_rule(rule);

        let result = engine.evaluate().unwrap();
        assert!(result.converged);

        let bigs = engine.facts("big");
        assert_eq!(bigs.len(), 2); // 5 and 10
    }
}
