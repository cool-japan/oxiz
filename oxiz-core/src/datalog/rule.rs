//! Datalog rule representation
//!
//! Rules are Horn clauses of the form: head :- body1, body2, ..., bodyn
//! This module provides AST types for rules, atoms, and terms.

use lasso::Spur;
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::relation::RelationId;
use super::tuple::Value;

/// Unique identifier for a rule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId(pub u64);

impl RuleId {
    /// Create a new rule ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get raw value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// A variable in a Datalog rule
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable(pub Spur);

impl Variable {
    /// Create a new variable
    pub fn new(name: Spur) -> Self {
        Self(name)
    }

    /// Get variable name
    pub fn name(&self) -> Spur {
        self.0
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{:?}", self.0)
    }
}

/// A term in an atom (can be variable or constant)
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A variable
    Var(Variable),
    /// A constant value
    Const(Value),
    /// Wildcard (don't care)
    Wildcard,
    /// Aggregate function
    Aggregate(AggregateKind, Box<Term>),
}

impl Term {
    /// Create a variable term
    pub fn var(name: Spur) -> Self {
        Term::Var(Variable::new(name))
    }

    /// Create a constant term
    pub fn constant(value: Value) -> Self {
        Term::Const(value)
    }

    /// Create a wildcard term
    pub fn wildcard() -> Self {
        Term::Wildcard
    }

    /// Check if this is a variable
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// Check if this is a constant
    pub fn is_const(&self) -> bool {
        matches!(self, Term::Const(_))
    }

    /// Check if this is a wildcard
    pub fn is_wildcard(&self) -> bool {
        matches!(self, Term::Wildcard)
    }

    /// Get as variable
    pub fn as_var(&self) -> Option<Variable> {
        match self {
            Term::Var(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as constant
    pub fn as_const(&self) -> Option<&Value> {
        match self {
            Term::Const(v) => Some(v),
            _ => None,
        }
    }

    /// Collect variables
    pub fn variables(&self) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut HashSet<Variable>) {
        match self {
            Term::Var(v) => {
                vars.insert(*v);
            }
            Term::Aggregate(_, inner) => {
                inner.collect_variables(vars);
            }
            _ => {}
        }
    }
}

/// Aggregate function kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateKind {
    /// Count
    Count,
    /// Sum
    Sum,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Average
    Avg,
}

impl fmt::Display for AggregateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateKind::Count => write!(f, "count"),
            AggregateKind::Sum => write!(f, "sum"),
            AggregateKind::Min => write!(f, "min"),
            AggregateKind::Max => write!(f, "max"),
            AggregateKind::Avg => write!(f, "avg"),
        }
    }
}

/// Kind of atom in a rule
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomKind {
    /// Positive literal
    Positive,
    /// Negated literal (negation-as-failure)
    Negated,
    /// Built-in comparison
    Comparison,
    /// Arithmetic expression
    Arithmetic,
    /// Aggregation
    Aggregate,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonOp::Eq => write!(f, "="),
            ComparisonOp::Ne => write!(f, "!="),
            ComparisonOp::Lt => write!(f, "<"),
            ComparisonOp::Le => write!(f, "<="),
            ComparisonOp::Gt => write!(f, ">"),
            ComparisonOp::Ge => write!(f, ">="),
        }
    }
}

/// Arithmetic operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Modulo
    Mod,
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithOp::Add => write!(f, "+"),
            ArithOp::Sub => write!(f, "-"),
            ArithOp::Mul => write!(f, "*"),
            ArithOp::Div => write!(f, "/"),
            ArithOp::Mod => write!(f, "%"),
        }
    }
}

/// An atom (predicate application) in a rule
#[derive(Debug, Clone)]
pub struct Atom {
    /// Atom kind
    kind: AtomKind,
    /// Relation name (for positive/negated atoms)
    relation: Option<Spur>,
    /// Relation ID (resolved during compilation)
    relation_id: Option<RelationId>,
    /// Arguments
    terms: Vec<Term>,
    /// Comparison operator (for comparisons)
    comparison_op: Option<ComparisonOp>,
    /// Arithmetic operator (for arithmetic)
    arith_op: Option<ArithOp>,
    /// Aggregate kind
    aggregate_kind: Option<AggregateKind>,
    /// Group-by variables (for aggregates)
    group_by: Vec<Variable>,
}

impl Atom {
    /// Create a positive atom
    pub fn positive(relation: Spur, terms: Vec<Term>) -> Self {
        Self {
            kind: AtomKind::Positive,
            relation: Some(relation),
            relation_id: None,
            terms,
            comparison_op: None,
            arith_op: None,
            aggregate_kind: None,
            group_by: Vec::new(),
        }
    }

    /// Create a negated atom
    pub fn negated(relation: Spur, terms: Vec<Term>) -> Self {
        Self {
            kind: AtomKind::Negated,
            relation: Some(relation),
            relation_id: None,
            terms,
            comparison_op: None,
            arith_op: None,
            aggregate_kind: None,
            group_by: Vec::new(),
        }
    }

    /// Create a comparison atom
    pub fn comparison(left: Term, op: ComparisonOp, right: Term) -> Self {
        Self {
            kind: AtomKind::Comparison,
            relation: None,
            relation_id: None,
            terms: vec![left, right],
            comparison_op: Some(op),
            arith_op: None,
            aggregate_kind: None,
            group_by: Vec::new(),
        }
    }

    /// Create an arithmetic atom (result = left op right)
    pub fn arithmetic(result: Term, left: Term, op: ArithOp, right: Term) -> Self {
        Self {
            kind: AtomKind::Arithmetic,
            relation: None,
            relation_id: None,
            terms: vec![result, left, right],
            comparison_op: None,
            arith_op: Some(op),
            aggregate_kind: None,
            group_by: Vec::new(),
        }
    }

    /// Create an aggregate atom
    pub fn aggregate(
        result: Term,
        agg_kind: AggregateKind,
        relation: Spur,
        agg_var: Term,
        group_by: Vec<Variable>,
    ) -> Self {
        Self {
            kind: AtomKind::Aggregate,
            relation: Some(relation),
            relation_id: None,
            terms: vec![result, agg_var],
            comparison_op: None,
            arith_op: None,
            aggregate_kind: Some(agg_kind),
            group_by,
        }
    }

    /// Get atom kind
    pub fn kind(&self) -> AtomKind {
        self.kind
    }

    /// Get relation name
    pub fn relation(&self) -> Option<Spur> {
        self.relation
    }

    /// Get relation ID
    pub fn relation_id(&self) -> Option<RelationId> {
        self.relation_id
    }

    /// Set relation ID (during resolution)
    pub fn set_relation_id(&mut self, id: RelationId) {
        self.relation_id = Some(id);
    }

    /// Get terms
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// Get mutable terms
    pub fn terms_mut(&mut self) -> &mut Vec<Term> {
        &mut self.terms
    }

    /// Get arity
    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    /// Get comparison operator
    pub fn comparison_op(&self) -> Option<ComparisonOp> {
        self.comparison_op
    }

    /// Get arithmetic operator
    pub fn arith_op(&self) -> Option<ArithOp> {
        self.arith_op
    }

    /// Get aggregate kind
    pub fn aggregate_kind(&self) -> Option<AggregateKind> {
        self.aggregate_kind
    }

    /// Get group-by variables
    pub fn group_by(&self) -> &[Variable] {
        &self.group_by
    }

    /// Collect all variables in atom
    pub fn variables(&self) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        for term in &self.terms {
            for v in term.variables() {
                vars.insert(v);
            }
        }
        for v in &self.group_by {
            vars.insert(*v);
        }
        vars
    }

    /// Check if atom is safe (all variables appear in positive body atom)
    pub fn is_ground(&self) -> bool {
        self.terms.iter().all(|t| t.is_const())
    }
}

/// A Datalog rule
#[derive(Debug, Clone)]
pub struct Rule {
    /// Rule ID
    id: RuleId,
    /// Head atom
    head: Atom,
    /// Body atoms
    body: Vec<Atom>,
    /// Rule priority (for stratification)
    priority: i32,
    /// Whether rule is recursive
    is_recursive: bool,
}

impl Rule {
    /// Create a new rule
    pub fn new(id: RuleId, head: Atom, body: Vec<Atom>) -> Self {
        let head_rel = head.relation();
        let is_recursive = body
            .iter()
            .any(|atom| atom.relation() == head_rel && atom.kind() == AtomKind::Positive);

        Self {
            id,
            head,
            body,
            priority: 0,
            is_recursive,
        }
    }

    /// Create a fact (rule with empty body)
    pub fn fact(id: RuleId, head: Atom) -> Self {
        Self {
            id,
            head,
            body: Vec::new(),
            priority: 0,
            is_recursive: false,
        }
    }

    /// Get rule ID
    pub fn id(&self) -> RuleId {
        self.id
    }

    /// Get head atom
    pub fn head(&self) -> &Atom {
        &self.head
    }

    /// Get mutable head atom
    pub fn head_mut(&mut self) -> &mut Atom {
        &mut self.head
    }

    /// Get body atoms
    pub fn body(&self) -> &[Atom] {
        &self.body
    }

    /// Get mutable body atoms
    pub fn body_mut(&mut self) -> &mut Vec<Atom> {
        &mut self.body
    }

    /// Check if this is a fact
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }

    /// Check if this is recursive
    pub fn is_recursive(&self) -> bool {
        self.is_recursive
    }

    /// Get priority
    pub fn priority(&self) -> i32 {
        self.priority
    }

    /// Set priority
    pub fn set_priority(&mut self, priority: i32) {
        self.priority = priority;
    }

    /// Get all variables in rule
    pub fn variables(&self) -> HashSet<Variable> {
        let mut vars = self.head.variables();
        for atom in &self.body {
            vars.extend(atom.variables());
        }
        vars
    }

    /// Get head variables
    pub fn head_variables(&self) -> HashSet<Variable> {
        self.head.variables()
    }

    /// Get body variables
    pub fn body_variables(&self) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        for atom in &self.body {
            vars.extend(atom.variables());
        }
        vars
    }

    /// Check if rule is safe (all head vars appear in positive body)
    pub fn is_safe(&self) -> bool {
        let head_vars = self.head_variables();
        let positive_body_vars: HashSet<_> = self
            .body
            .iter()
            .filter(|a| a.kind() == AtomKind::Positive)
            .flat_map(|a| a.variables())
            .collect();

        head_vars.is_subset(&positive_body_vars)
    }

    /// Check if rule is range-restricted
    pub fn is_range_restricted(&self) -> bool {
        self.is_safe()
    }

    /// Get positive body atoms
    pub fn positive_body(&self) -> impl Iterator<Item = &Atom> {
        self.body.iter().filter(|a| a.kind() == AtomKind::Positive)
    }

    /// Get negated body atoms
    pub fn negated_body(&self) -> impl Iterator<Item = &Atom> {
        self.body.iter().filter(|a| a.kind() == AtomKind::Negated)
    }

    /// Get comparison atoms
    pub fn comparisons(&self) -> impl Iterator<Item = &Atom> {
        self.body
            .iter()
            .filter(|a| a.kind() == AtomKind::Comparison)
    }

    /// Get relation dependencies
    pub fn dependencies(&self) -> HashSet<Spur> {
        self.body.iter().filter_map(|a| a.relation()).collect()
    }

    /// Get negation dependencies
    pub fn negation_dependencies(&self) -> HashSet<Spur> {
        self.body
            .iter()
            .filter(|a| a.kind() == AtomKind::Negated)
            .filter_map(|a| a.relation())
            .collect()
    }
}

/// Variable binding during rule evaluation
#[derive(Debug, Clone)]
pub struct Binding {
    /// Variable to value mapping
    values: HashMap<Variable, Value>,
}

impl Binding {
    /// Create an empty binding
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Bind a variable to a value
    pub fn bind(&mut self, var: Variable, value: Value) {
        self.values.insert(var, value);
    }

    /// Get binding for variable
    pub fn get(&self, var: Variable) -> Option<&Value> {
        self.values.get(&var)
    }

    /// Check if variable is bound
    pub fn is_bound(&self, var: Variable) -> bool {
        self.values.contains_key(&var)
    }

    /// Get all bound variables
    pub fn variables(&self) -> impl Iterator<Item = Variable> + '_ {
        self.values.keys().copied()
    }

    /// Merge with another binding (fails if conflicting)
    pub fn merge(&self, other: &Binding) -> Option<Binding> {
        let mut result = self.clone();
        for (var, value) in &other.values {
            if let Some(existing) = result.values.get(var) {
                if existing != value {
                    return None; // Conflict
                }
            } else {
                result.values.insert(*var, value.clone());
            }
        }
        Some(result)
    }

    /// Apply binding to a term
    pub fn apply(&self, term: &Term) -> Option<Value> {
        match term {
            Term::Var(v) => self.get(*v).cloned(),
            Term::Const(c) => Some(c.clone()),
            Term::Wildcard => None,
            Term::Aggregate(_, _) => None, // Handled separately
        }
    }

    /// Clear all bindings
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

impl Default for Binding {
    fn default() -> Self {
        Self::new()
    }
}

/// Rule builder for fluent API
pub struct RuleBuilder {
    id: RuleId,
    head: Option<Atom>,
    body: Vec<Atom>,
    interner: lasso::ThreadedRodeo,
}

impl RuleBuilder {
    /// Create a new rule builder
    pub fn new(id: RuleId) -> Self {
        Self {
            id,
            head: None,
            body: Vec::new(),
            interner: lasso::ThreadedRodeo::default(),
        }
    }

    /// Set head atom
    pub fn head(mut self, relation: &str, terms: Vec<Term>) -> Self {
        let rel_name = self.interner.get_or_intern(relation);
        self.head = Some(Atom::positive(rel_name, terms));
        self
    }

    /// Add positive body atom
    pub fn positive(mut self, relation: &str, terms: Vec<Term>) -> Self {
        let rel_name = self.interner.get_or_intern(relation);
        self.body.push(Atom::positive(rel_name, terms));
        self
    }

    /// Add negated body atom
    pub fn negated(mut self, relation: &str, terms: Vec<Term>) -> Self {
        let rel_name = self.interner.get_or_intern(relation);
        self.body.push(Atom::negated(rel_name, terms));
        self
    }

    /// Add comparison
    pub fn compare(mut self, left: Term, op: ComparisonOp, right: Term) -> Self {
        self.body.push(Atom::comparison(left, op, right));
        self
    }

    /// Build the rule
    pub fn build(self) -> Option<Rule> {
        self.head.map(|head| Rule::new(self.id, head, self.body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Term {
        let interner = lasso::ThreadedRodeo::default();
        Term::var(interner.get_or_intern(name))
    }

    fn const_i64(n: i64) -> Term {
        Term::constant(Value::Int64(n))
    }

    #[test]
    fn test_term_types() {
        let interner = lasso::ThreadedRodeo::default();
        let v = Term::var(interner.get_or_intern("x"));
        let c = Term::constant(Value::Int64(42));
        let w = Term::wildcard();

        assert!(v.is_var());
        assert!(c.is_const());
        assert!(w.is_wildcard());
    }

    #[test]
    fn test_atom_creation() {
        let interner = lasso::ThreadedRodeo::default();
        let rel = interner.get_or_intern("edge");
        let x = Term::var(interner.get_or_intern("x"));
        let y = Term::var(interner.get_or_intern("y"));

        let atom = Atom::positive(rel, vec![x, y]);
        assert_eq!(atom.kind(), AtomKind::Positive);
        assert_eq!(atom.arity(), 2);
    }

    #[test]
    fn test_comparison_atom() {
        let interner = lasso::ThreadedRodeo::default();
        let x = Term::var(interner.get_or_intern("x"));
        let c = Term::constant(Value::Int64(10));

        let atom = Atom::comparison(x, ComparisonOp::Lt, c);
        assert_eq!(atom.kind(), AtomKind::Comparison);
        assert_eq!(atom.comparison_op(), Some(ComparisonOp::Lt));
    }

    #[test]
    fn test_rule_creation() {
        let interner = lasso::ThreadedRodeo::default();

        let path = interner.get_or_intern("path");
        let edge = interner.get_or_intern("edge");
        let x = Term::var(interner.get_or_intern("x"));
        let y = Term::var(interner.get_or_intern("y"));

        // path(x, y) :- edge(x, y)
        let head = Atom::positive(path, vec![x.clone(), y.clone()]);
        let body = vec![Atom::positive(edge, vec![x, y])];

        let rule = Rule::new(RuleId::new(1), head, body);
        assert!(!rule.is_fact());
        assert!(rule.is_safe());
    }

    #[test]
    fn test_recursive_rule() {
        let interner = lasso::ThreadedRodeo::default();

        let path = interner.get_or_intern("path");
        let edge = interner.get_or_intern("edge");
        let x = Term::var(interner.get_or_intern("x"));
        let y = Term::var(interner.get_or_intern("y"));
        let z = Term::var(interner.get_or_intern("z"));

        // path(x, z) :- path(x, y), edge(y, z)
        let head = Atom::positive(path, vec![x.clone(), z.clone()]);
        let body = vec![
            Atom::positive(path, vec![x, y.clone()]),
            Atom::positive(edge, vec![y, z]),
        ];

        let rule = Rule::new(RuleId::new(1), head, body);
        assert!(rule.is_recursive());
    }

    #[test]
    fn test_binding_merge() {
        let interner = lasso::ThreadedRodeo::default();
        let x = Variable::new(interner.get_or_intern("x"));
        let y = Variable::new(interner.get_or_intern("y"));

        let mut b1 = Binding::new();
        b1.bind(x, Value::Int64(1));

        let mut b2 = Binding::new();
        b2.bind(y, Value::Int64(2));

        let merged = b1.merge(&b2);
        assert!(merged.is_some());
        let merged = merged.unwrap();
        assert_eq!(merged.get(x), Some(&Value::Int64(1)));
        assert_eq!(merged.get(y), Some(&Value::Int64(2)));
    }

    #[test]
    fn test_binding_conflict() {
        let interner = lasso::ThreadedRodeo::default();
        let x = Variable::new(interner.get_or_intern("x"));

        let mut b1 = Binding::new();
        b1.bind(x, Value::Int64(1));

        let mut b2 = Binding::new();
        b2.bind(x, Value::Int64(2));

        assert!(b1.merge(&b2).is_none());
    }

    #[test]
    fn test_unsafe_rule() {
        let interner = lasso::ThreadedRodeo::default();

        let rel = interner.get_or_intern("rel");
        let x = Term::var(interner.get_or_intern("x"));
        let y = Term::var(interner.get_or_intern("y"));

        // rel(x, y) :- y > 0 -- unsafe because x not bound
        let head = Atom::positive(rel, vec![x, y.clone()]);
        let body = vec![Atom::comparison(
            y,
            ComparisonOp::Gt,
            Term::constant(Value::Int64(0)),
        )];

        let rule = Rule::new(RuleId::new(1), head, body);
        assert!(!rule.is_safe());
    }

    #[test]
    fn test_rule_builder() {
        let interner = lasso::ThreadedRodeo::default();
        let x = Term::var(interner.get_or_intern("x"));

        let rule = RuleBuilder::new(RuleId::new(1))
            .head("result", vec![x.clone()])
            .positive("source", vec![x])
            .build();

        assert!(rule.is_some());
        let rule = rule.unwrap();
        assert!(!rule.is_fact());
    }
}
