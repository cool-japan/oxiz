//! Constrained Horn Clause (CHC) representation.
//!
//! CHC format: `forall X. (body => head)`
//! where body is a conjunction of constraints and predicate applications,
//! and head is a predicate application or false (query).
//!
//! Reference: Z3's `muz/spacer/` implementation.

use indexmap::IndexMap;
use oxiz_core::{TermId, TermManager};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};

/// Unique identifier for a predicate declaration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PredId(pub u32);

impl PredId {
    /// Create a new predicate ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Unique identifier for a CHC rule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleId(pub u32);

impl RuleId {
    /// Create a new rule ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A predicate declaration with arity and parameter sorts
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Unique identifier
    pub id: PredId,
    /// Name of the predicate
    pub name: String,
    /// Parameter sorts (arity = params.len())
    pub params: SmallVec<[oxiz_core::SortId; 4]>,
}

impl Predicate {
    /// Get the arity of this predicate
    #[inline]
    #[must_use]
    pub fn arity(&self) -> usize {
        self.params.len()
    }
}

/// An application of a predicate to arguments
#[derive(Debug, Clone)]
pub struct PredicateApp {
    /// The predicate being applied
    pub pred: PredId,
    /// Arguments (must match arity)
    pub args: SmallVec<[TermId; 4]>,
}

impl PredicateApp {
    /// Create a new predicate application
    pub fn new(pred: PredId, args: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            pred,
            args: args.into_iter().collect(),
        }
    }
}

/// The head of a CHC rule: either a predicate application or `false` (query)
#[derive(Debug, Clone)]
pub enum RuleHead {
    /// Predicate application (non-query rule)
    Predicate(PredicateApp),
    /// Query rule (head is false)
    Query,
}

impl RuleHead {
    /// Check if this is a query (false head)
    #[inline]
    #[must_use]
    pub fn is_query(&self) -> bool {
        matches!(self, RuleHead::Query)
    }

    /// Get the predicate application if not a query
    #[inline]
    #[must_use]
    pub fn as_predicate(&self) -> Option<&PredicateApp> {
        match self {
            RuleHead::Predicate(app) => Some(app),
            RuleHead::Query => None,
        }
    }
}

/// The body of a CHC rule: conjunction of predicate applications and constraints
#[derive(Debug, Clone)]
pub struct RuleBody {
    /// Predicate applications in the body (uninterpreted tail)
    pub predicates: SmallVec<[PredicateApp; 2]>,
    /// Interpreted constraint (conjunction)
    pub constraint: TermId,
}

impl RuleBody {
    /// Create a body with no predicate applications (init rule)
    pub fn init(constraint: TermId) -> Self {
        Self {
            predicates: SmallVec::new(),
            constraint,
        }
    }

    /// Create a body with predicate applications
    pub fn new(predicates: impl IntoIterator<Item = PredicateApp>, constraint: TermId) -> Self {
        Self {
            predicates: predicates.into_iter().collect(),
            constraint,
        }
    }

    /// Check if this is an init body (no predicate applications)
    #[inline]
    #[must_use]
    pub fn is_init(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Get the number of predicate applications (uninterpreted tail size)
    #[inline]
    #[must_use]
    pub fn uninterpreted_tail_size(&self) -> usize {
        self.predicates.len()
    }
}

/// A CHC rule: `forall vars. (body => head)`
#[derive(Debug, Clone)]
pub struct Rule {
    /// Unique identifier
    pub id: RuleId,
    /// Universally quantified variables
    pub vars: SmallVec<[(String, oxiz_core::SortId); 4]>,
    /// Body of the rule
    pub body: RuleBody,
    /// Head of the rule
    pub head: RuleHead,
    /// Optional name/label for the rule
    pub name: Option<String>,
}

impl Rule {
    /// Check if this is an init rule (no predicates in body)
    #[inline]
    #[must_use]
    pub fn is_init(&self) -> bool {
        self.body.is_init()
    }

    /// Check if this is a query rule (head is false)
    #[inline]
    #[must_use]
    pub fn is_query(&self) -> bool {
        self.head.is_query()
    }

    /// Get the head predicate if not a query
    #[inline]
    #[must_use]
    pub fn head_predicate(&self) -> Option<PredId> {
        match &self.head {
            RuleHead::Predicate(app) => Some(app.pred),
            RuleHead::Query => None,
        }
    }

    /// Get all body predicate IDs
    pub fn body_predicates(&self) -> impl Iterator<Item = PredId> + '_ {
        self.body.predicates.iter().map(|app| app.pred)
    }
}

/// A complete CHC system containing predicates and rules
#[derive(Debug)]
pub struct ChcSystem {
    /// Predicate declarations indexed by ID
    predicates: Vec<Predicate>,
    /// Predicate lookup by name
    pred_by_name: FxHashMap<String, PredId>,
    /// Next predicate ID
    next_pred_id: AtomicU32,

    /// Rules indexed by ID
    rules: Vec<Rule>,
    /// Next rule ID
    next_rule_id: AtomicU32,

    /// Rules grouped by head predicate (for forward analysis)
    rules_by_head: IndexMap<PredId, SmallVec<[RuleId; 4]>>,
    /// Rules that use a predicate in the body (for backward analysis)
    rules_by_body: IndexMap<PredId, SmallVec<[RuleId; 4]>>,

    /// Query rules (head is false)
    queries: SmallVec<[RuleId; 2]>,
    /// Entry rules (no predicates in body)
    entries: SmallVec<[RuleId; 2]>,
}

impl Default for ChcSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ChcSystem {
    /// Create a new empty CHC system
    pub fn new() -> Self {
        Self {
            predicates: Vec::new(),
            pred_by_name: FxHashMap::default(),
            next_pred_id: AtomicU32::new(0),
            rules: Vec::new(),
            next_rule_id: AtomicU32::new(0),
            rules_by_head: IndexMap::new(),
            rules_by_body: IndexMap::new(),
            queries: SmallVec::new(),
            entries: SmallVec::new(),
        }
    }

    /// Declare a new predicate
    pub fn declare_predicate(
        &mut self,
        name: impl Into<String>,
        params: impl IntoIterator<Item = oxiz_core::SortId>,
    ) -> PredId {
        let name = name.into();
        if let Some(&id) = self.pred_by_name.get(&name) {
            return id;
        }

        let id = PredId(self.next_pred_id.fetch_add(1, Ordering::Relaxed));
        let pred = Predicate {
            id,
            name: name.clone(),
            params: params.into_iter().collect(),
        };

        self.pred_by_name.insert(name, id);
        self.predicates.push(pred);
        id
    }

    /// Get a predicate by ID
    #[must_use]
    pub fn get_predicate(&self, id: PredId) -> Option<&Predicate> {
        self.predicates.get(id.0 as usize)
    }

    /// Get a predicate by name
    #[must_use]
    pub fn get_predicate_by_name(&self, name: &str) -> Option<&Predicate> {
        self.pred_by_name
            .get(name)
            .and_then(|&id| self.get_predicate(id))
    }

    /// Get a predicate ID by name
    #[must_use]
    pub fn get_predicate_id(&self, name: &str) -> Option<PredId> {
        self.pred_by_name.get(name).copied()
    }

    /// Add a rule to the system
    pub fn add_rule(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        body: RuleBody,
        head: RuleHead,
        name: Option<String>,
    ) -> RuleId {
        let id = RuleId(self.next_rule_id.fetch_add(1, Ordering::Relaxed));

        // Track queries and entries
        if head.is_query() {
            self.queries.push(id);
        }
        if body.is_init() {
            self.entries.push(id);
        }

        // Index by head predicate
        if let Some(pred_id) = head.as_predicate().map(|a| a.pred) {
            self.rules_by_head.entry(pred_id).or_default().push(id);
        }

        // Index by body predicates
        for app in &body.predicates {
            self.rules_by_body.entry(app.pred).or_default().push(id);
        }

        let rule = Rule {
            id,
            vars: vars.into_iter().collect(),
            body,
            head,
            name,
        };

        self.rules.push(rule);
        id
    }

    /// Add a simple init rule: `constraint => P(args)`
    pub fn add_init_rule(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        constraint: TermId,
        head_pred: PredId,
        head_args: impl IntoIterator<Item = TermId>,
    ) -> RuleId {
        let body = RuleBody::init(constraint);
        let head = RuleHead::Predicate(PredicateApp::new(head_pred, head_args));
        self.add_rule(vars, body, head, None)
    }

    /// Add a transition rule: `P1(args1) /\ ... /\ constraint => P(args)`
    pub fn add_transition_rule(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        body_preds: impl IntoIterator<Item = PredicateApp>,
        constraint: TermId,
        head_pred: PredId,
        head_args: impl IntoIterator<Item = TermId>,
    ) -> RuleId {
        let body = RuleBody::new(body_preds, constraint);
        let head = RuleHead::Predicate(PredicateApp::new(head_pred, head_args));
        self.add_rule(vars, body, head, None)
    }

    /// Add a query rule: `P(args) /\ constraint => false`
    pub fn add_query(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        body_preds: impl IntoIterator<Item = PredicateApp>,
        constraint: TermId,
    ) -> RuleId {
        let body = RuleBody::new(body_preds, constraint);
        self.add_rule(vars, body, RuleHead::Query, None)
    }

    /// Get a rule by ID
    #[must_use]
    pub fn get_rule(&self, id: RuleId) -> Option<&Rule> {
        self.rules.get(id.0 as usize)
    }

    /// Get all rules
    pub fn rules(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter()
    }

    /// Get all predicates
    pub fn predicates(&self) -> impl Iterator<Item = &Predicate> {
        self.predicates.iter()
    }

    /// Get query rules
    pub fn queries(&self) -> impl Iterator<Item = &Rule> {
        self.queries.iter().filter_map(|&id| self.get_rule(id))
    }

    /// Get entry rules (init rules)
    pub fn entries(&self) -> impl Iterator<Item = &Rule> {
        self.entries.iter().filter_map(|&id| self.get_rule(id))
    }

    /// Get rules with a given head predicate
    pub fn rules_by_head(&self, pred: PredId) -> impl Iterator<Item = &Rule> {
        self.rules_by_head
            .get(&pred)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|&id| self.get_rule(id))
    }

    /// Get rules that use a predicate in the body
    pub fn rules_using(&self, pred: PredId) -> impl Iterator<Item = &Rule> {
        self.rules_by_body
            .get(&pred)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|&id| self.get_rule(id))
    }

    /// Get the number of predicates
    #[must_use]
    pub fn num_predicates(&self) -> usize {
        self.predicates.len()
    }

    /// Get the number of rules
    #[must_use]
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Check if the system is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Get predicates in topological order (if acyclic)
    pub fn topological_order(&self) -> Option<Vec<PredId>> {
        let mut in_degree: FxHashMap<PredId, usize> = FxHashMap::default();
        let mut result = Vec::new();

        // Initialize in-degrees
        for pred in &self.predicates {
            in_degree.insert(pred.id, 0);
        }

        // Count dependencies
        for rule in &self.rules {
            if let Some(head_pred) = rule.head_predicate() {
                for body_pred in rule.body_predicates() {
                    if body_pred != head_pred {
                        *in_degree.entry(head_pred).or_default() += 1;
                    }
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<PredId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();

        while let Some(pred) = queue.pop() {
            result.push(pred);

            for rule in self.rules_by_body.get(&pred).into_iter().flatten() {
                if let Some(head_pred) = self.get_rule(*rule).and_then(|r| r.head_predicate())
                    && let Some(deg) = in_degree.get_mut(&head_pred)
                {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push(head_pred);
                    }
                }
            }
        }

        if result.len() == self.predicates.len() {
            Some(result)
        } else {
            None // Cycle detected
        }
    }
}

/// Builder for constructing CHC systems conveniently
pub struct ChcBuilder<'a> {
    system: ChcSystem,
    terms: &'a mut TermManager,
}

impl<'a> ChcBuilder<'a> {
    /// Create a new CHC builder
    pub fn new(terms: &'a mut TermManager) -> Self {
        Self {
            system: ChcSystem::new(),
            terms,
        }
    }

    /// Declare a predicate
    pub fn declare_pred(
        &mut self,
        name: impl Into<String>,
        params: impl IntoIterator<Item = oxiz_core::SortId>,
    ) -> PredId {
        self.system.declare_predicate(name, params)
    }

    /// Get the term manager
    pub fn terms(&mut self) -> &mut TermManager {
        self.terms
    }

    /// Build the CHC system
    pub fn build(self) -> ChcSystem {
        self.system
    }

    /// Get mutable access to the system
    pub fn system_mut(&mut self) -> &mut ChcSystem {
        &mut self.system
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chc_system_creation() {
        let terms = TermManager::new();
        let mut system = ChcSystem::new();

        // Declare predicates
        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let err = system.declare_predicate("Err", []);

        assert_eq!(system.num_predicates(), 2);
        assert_eq!(system.get_predicate(inv).unwrap().name, "Inv");
        assert_eq!(system.get_predicate(err).unwrap().arity(), 0);
    }

    #[test]
    fn test_chc_rules() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);

        // Init rule: x = 0 => Inv(x)
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let init_constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            init_constraint,
            inv,
            [x],
        );

        // Transition rule: Inv(x) /\ x' = x + 1 => Inv(x')
        let x_prime = terms.mk_var("x'", terms.sorts.int_sort);
        let one = terms.mk_int(1);
        let x_plus_one = terms.mk_add([x, one]);
        let trans_constraint = terms.mk_eq(x_prime, x_plus_one);

        system.add_transition_rule(
            [
                ("x".to_string(), terms.sorts.int_sort),
                ("x'".to_string(), terms.sorts.int_sort),
            ],
            [PredicateApp::new(inv, [x])],
            trans_constraint,
            inv,
            [x_prime],
        );

        // Query: Inv(x) /\ x < 0 => false
        let neg_constraint = terms.mk_lt(x, zero);
        system.add_query(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(inv, [x])],
            neg_constraint,
        );

        assert_eq!(system.num_rules(), 3);
        assert_eq!(system.entries().count(), 1);
        assert_eq!(system.queries().count(), 1);
    }

    #[test]
    fn test_rule_indexing() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let p = system.declare_predicate("P", [terms.sorts.int_sort]);
        let q = system.declare_predicate("Q", [terms.sorts.int_sort]);

        let x = terms.mk_var("x", terms.sorts.int_sort);
        let constraint = terms.mk_true();

        // P(x) => Q(x)
        system.add_transition_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(p, [x])],
            constraint,
            q,
            [x],
        );

        // Rules with head Q
        assert_eq!(system.rules_by_head(q).count(), 1);
        assert_eq!(system.rules_by_head(p).count(), 0);

        // Rules using P in body
        assert_eq!(system.rules_using(p).count(), 1);
        assert_eq!(system.rules_using(q).count(), 0);
    }

    #[test]
    fn test_topological_order() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let p1 = system.declare_predicate("P1", [terms.sorts.int_sort]);
        let p2 = system.declare_predicate("P2", [terms.sorts.int_sort]);
        let p3 = system.declare_predicate("P3", [terms.sorts.int_sort]);

        let x = terms.mk_var("x", terms.sorts.int_sort);
        let constraint = terms.mk_true();

        // P1 => P2, P2 => P3 (acyclic)
        system.add_transition_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(p1, [x])],
            constraint,
            p2,
            [x],
        );
        system.add_transition_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(p2, [x])],
            constraint,
            p3,
            [x],
        );

        let order = system.topological_order();
        assert!(order.is_some());

        let order = order.unwrap();
        let p1_pos = order.iter().position(|&id| id == p1).unwrap();
        let p2_pos = order.iter().position(|&id| id == p2).unwrap();
        let p3_pos = order.iter().position(|&id| id == p3).unwrap();

        // P1 should come before P2, P2 before P3
        assert!(p1_pos < p2_pos);
        assert!(p2_pos < p3_pos);
    }
}
