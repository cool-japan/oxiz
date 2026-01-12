//! Constraint Logic Programming (CLP) integration for Datalog
//!
//! Extends Datalog with constraint solving capabilities for numeric
//! and symbolic constraints.

use num_rational::BigRational;
use std::collections::HashMap;

use super::rule::Variable;
use super::tuple::Value;

/// Kind of constraint
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Equality: x = y or x = c
    Equal,
    /// Disequality: x != y or x != c
    NotEqual,
    /// Less than: x < y or x < c
    LessThan,
    /// Less than or equal: x <= y or x <= c
    LessEqual,
    /// Greater than: x > y or x > c
    GreaterThan,
    /// Greater than or equal: x >= y or x >= c
    GreaterEqual,
    /// Membership: x in {c1, c2, ...}
    Member,
    /// Not member: x not in {c1, c2, ...}
    NotMember,
    /// Linear arithmetic: a1*x1 + a2*x2 + ... = c
    Linear,
}

/// A constraint in CLP
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Kind of constraint
    kind: ConstraintKind,
    /// Variables involved
    variables: Vec<Variable>,
    /// Coefficients (for linear constraints)
    coefficients: Vec<BigRational>,
    /// Constant term
    constant: Option<Value>,
    /// Set of values (for member/not member)
    value_set: Vec<Value>,
}

impl Constraint {
    /// Create an equality constraint
    pub fn equal(x: Variable, y: Variable) -> Self {
        Self {
            kind: ConstraintKind::Equal,
            variables: vec![x, y],
            coefficients: Vec::new(),
            constant: None,
            value_set: Vec::new(),
        }
    }

    /// Create an equality with constant
    pub fn equal_const(x: Variable, c: Value) -> Self {
        Self {
            kind: ConstraintKind::Equal,
            variables: vec![x],
            coefficients: Vec::new(),
            constant: Some(c),
            value_set: Vec::new(),
        }
    }

    /// Create a disequality constraint
    pub fn not_equal(x: Variable, y: Variable) -> Self {
        Self {
            kind: ConstraintKind::NotEqual,
            variables: vec![x, y],
            coefficients: Vec::new(),
            constant: None,
            value_set: Vec::new(),
        }
    }

    /// Create a less-than constraint
    pub fn less_than(x: Variable, y: Variable) -> Self {
        Self {
            kind: ConstraintKind::LessThan,
            variables: vec![x, y],
            coefficients: Vec::new(),
            constant: None,
            value_set: Vec::new(),
        }
    }

    /// Create a less-equal constraint
    pub fn less_equal(x: Variable, y: Variable) -> Self {
        Self {
            kind: ConstraintKind::LessEqual,
            variables: vec![x, y],
            coefficients: Vec::new(),
            constant: None,
            value_set: Vec::new(),
        }
    }

    /// Create a membership constraint
    pub fn member(x: Variable, values: Vec<Value>) -> Self {
        Self {
            kind: ConstraintKind::Member,
            variables: vec![x],
            coefficients: Vec::new(),
            constant: None,
            value_set: values,
        }
    }

    /// Create a linear constraint: sum(coeffs * vars) = constant
    pub fn linear(vars: Vec<Variable>, coeffs: Vec<BigRational>, constant: BigRational) -> Self {
        Self {
            kind: ConstraintKind::Linear,
            variables: vars,
            coefficients: coeffs,
            constant: Some(Value::Rational(constant)),
            value_set: Vec::new(),
        }
    }

    /// Get constraint kind
    pub fn kind(&self) -> &ConstraintKind {
        &self.kind
    }

    /// Get variables
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[BigRational] {
        &self.coefficients
    }

    /// Get constant
    pub fn constant(&self) -> Option<&Value> {
        self.constant.as_ref()
    }

    /// Get value set
    pub fn value_set(&self) -> &[Value] {
        &self.value_set
    }

    /// Check if constraint involves variable
    pub fn involves(&self, var: Variable) -> bool {
        self.variables.contains(&var)
    }
}

/// Constraint store for CLP solver
#[derive(Debug)]
pub struct ConstraintStore {
    /// All constraints
    constraints: Vec<Constraint>,
    /// Constraints indexed by variable
    var_constraints: HashMap<Variable, Vec<usize>>,
    /// Variable domains
    domains: HashMap<Variable, Domain>,
}

/// Domain of a variable
#[derive(Debug, Clone)]
pub enum Domain {
    /// Unrestricted
    Any,
    /// Boolean domain
    Bool,
    /// Integer domain with bounds
    Integer { min: Option<i64>, max: Option<i64> },
    /// Rational domain with bounds
    Rational {
        min: Option<BigRational>,
        max: Option<BigRational>,
    },
    /// Finite set of values
    Finite(Vec<Value>),
    /// Empty (unsatisfiable)
    Empty,
}

impl Domain {
    /// Create an integer domain
    pub fn integer() -> Self {
        Domain::Integer {
            min: None,
            max: None,
        }
    }

    /// Create a bounded integer domain
    pub fn integer_range(min: i64, max: i64) -> Self {
        Domain::Integer {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Create a finite domain
    pub fn finite(values: Vec<Value>) -> Self {
        if values.is_empty() {
            Domain::Empty
        } else {
            Domain::Finite(values)
        }
    }

    /// Check if domain is empty
    pub fn is_empty(&self) -> bool {
        matches!(self, Domain::Empty)
    }

    /// Check if domain is singleton
    pub fn is_singleton(&self) -> bool {
        match self {
            Domain::Finite(v) => v.len() == 1,
            _ => false,
        }
    }

    /// Get singleton value
    pub fn singleton_value(&self) -> Option<&Value> {
        match self {
            Domain::Finite(v) if v.len() == 1 => v.first(),
            _ => None,
        }
    }

    /// Intersect with another domain
    pub fn intersect(&self, other: &Domain) -> Domain {
        match (self, other) {
            (Domain::Empty, _) | (_, Domain::Empty) => Domain::Empty,
            (Domain::Any, d) | (d, Domain::Any) => d.clone(),
            (Domain::Finite(a), Domain::Finite(b)) => {
                let intersection: Vec<_> = a.iter().filter(|v| b.contains(v)).cloned().collect();
                Domain::finite(intersection)
            }
            (
                Domain::Integer {
                    min: min1,
                    max: max1,
                },
                Domain::Integer {
                    min: min2,
                    max: max2,
                },
            ) => {
                let new_min = match (min1, min2) {
                    (Some(a), Some(b)) => Some(*a.max(b)),
                    (Some(a), None) => Some(*a),
                    (None, Some(b)) => Some(*b),
                    (None, None) => None,
                };
                let new_max = match (max1, max2) {
                    (Some(a), Some(b)) => Some(*a.min(b)),
                    (Some(a), None) => Some(*a),
                    (None, Some(b)) => Some(*b),
                    (None, None) => None,
                };
                if let (Some(min), Some(max)) = (new_min, new_max)
                    && min > max
                {
                    return Domain::Empty;
                }
                Domain::Integer {
                    min: new_min,
                    max: new_max,
                }
            }
            _ => Domain::Any, // Simplified
        }
    }

    /// Remove a value from domain
    pub fn remove(&mut self, value: &Value) {
        if let Domain::Finite(values) = self {
            values.retain(|v| v != value);
            if values.is_empty() {
                *self = Domain::Empty;
            }
        }
    }
}

impl ConstraintStore {
    /// Create a new constraint store
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            var_constraints: HashMap::new(),
            domains: HashMap::new(),
        }
    }

    /// Add a constraint
    pub fn add(&mut self, constraint: Constraint) {
        let idx = self.constraints.len();
        for var in constraint.variables() {
            self.var_constraints.entry(*var).or_default().push(idx);
        }
        self.constraints.push(constraint);
    }

    /// Set domain for variable
    pub fn set_domain(&mut self, var: Variable, domain: Domain) {
        self.domains.insert(var, domain);
    }

    /// Get domain for variable
    pub fn domain(&self, var: Variable) -> Option<&Domain> {
        self.domains.get(&var)
    }

    /// Get constraints involving variable
    pub fn constraints_for(&self, var: Variable) -> Vec<&Constraint> {
        self.var_constraints
            .get(&var)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| self.constraints.get(i))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.var_constraints.clear();
        self.domains.clear();
    }
}

impl Default for ConstraintStore {
    fn default() -> Self {
        Self::new()
    }
}

/// CLP solver result
#[derive(Debug)]
pub enum ClpResult {
    /// Satisfiable with variable assignment
    Sat(HashMap<Variable, Value>),
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout or resource limit)
    Unknown,
}

/// CLP solver
#[derive(Debug)]
pub struct ClpSolver {
    /// Constraint store
    store: ConstraintStore,
    /// Current assignment
    assignment: HashMap<Variable, Value>,
    /// Propagation queue
    prop_queue: Vec<Variable>,
    /// Solver configuration
    config: ClpConfig,
}

/// CLP solver configuration
#[derive(Debug, Clone)]
pub struct ClpConfig {
    /// Maximum propagation iterations
    pub max_propagations: usize,
    /// Enable arc consistency
    pub arc_consistency: bool,
    /// Enable bound propagation
    pub bound_propagation: bool,
}

impl Default for ClpConfig {
    fn default() -> Self {
        Self {
            max_propagations: 10000,
            arc_consistency: true,
            bound_propagation: true,
        }
    }
}

impl ClpSolver {
    /// Create a new CLP solver
    pub fn new() -> Self {
        Self {
            store: ConstraintStore::new(),
            assignment: HashMap::new(),
            prop_queue: Vec::new(),
            config: ClpConfig::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: ClpConfig) -> Self {
        Self {
            store: ConstraintStore::new(),
            assignment: HashMap::new(),
            prop_queue: Vec::new(),
            config,
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        // Queue variables for propagation
        for var in constraint.variables() {
            if !self.prop_queue.contains(var) {
                self.prop_queue.push(*var);
            }
        }
        self.store.add(constraint);
    }

    /// Set variable domain
    pub fn set_domain(&mut self, var: Variable, domain: Domain) {
        self.store.set_domain(var, domain);
        if !self.prop_queue.contains(&var) {
            self.prop_queue.push(var);
        }
    }

    /// Assign a value to a variable
    pub fn assign(&mut self, var: Variable, value: Value) {
        self.assignment.insert(var, value);
        if !self.prop_queue.contains(&var) {
            self.prop_queue.push(var);
        }
    }

    /// Get current assignment
    pub fn get_assignment(&self, var: Variable) -> Option<&Value> {
        self.assignment.get(&var)
    }

    /// Solve the constraint system
    pub fn solve(&mut self) -> ClpResult {
        // Propagate constraints
        if !self.propagate() {
            return ClpResult::Unsat;
        }

        // Check if all variables are assigned
        let unassigned: Vec<_> = self
            .store
            .domains
            .keys()
            .filter(|v| !self.assignment.contains_key(v))
            .copied()
            .collect();

        if unassigned.is_empty() {
            // All assigned - verify constraints
            if self.verify_all() {
                return ClpResult::Sat(self.assignment.clone());
            } else {
                return ClpResult::Unsat;
            }
        }

        // Try to assign unassigned variables from singleton domains
        for var in &unassigned {
            if let Some(domain) = self.store.domain(*var)
                && let Some(value) = domain.singleton_value()
            {
                self.assignment.insert(*var, value.clone());
            }
        }

        // Re-check
        if self.verify_all() {
            ClpResult::Sat(self.assignment.clone())
        } else {
            ClpResult::Unknown
        }
    }

    /// Propagate constraints
    fn propagate(&mut self) -> bool {
        let mut iterations = 0;

        while !self.prop_queue.is_empty() && iterations < self.config.max_propagations {
            iterations += 1;

            let var = self
                .prop_queue
                .pop()
                .unwrap_or_else(|| panic!("Queue should not be empty at this point"));

            // Propagate constraints involving this variable
            let constraint_indices: Vec<_> = self
                .store
                .var_constraints
                .get(&var)
                .cloned()
                .unwrap_or_default();

            for idx in constraint_indices {
                if let Some(constraint) = self.store.constraints.get(idx)
                    && !self.propagate_constraint(constraint.clone())
                {
                    return false;
                }
            }
        }

        true
    }

    /// Propagate a single constraint
    fn propagate_constraint(&mut self, constraint: Constraint) -> bool {
        match constraint.kind() {
            ConstraintKind::Equal => self.propagate_equality(&constraint),
            ConstraintKind::NotEqual => self.propagate_disequality(&constraint),
            ConstraintKind::Member => self.propagate_membership(&constraint),
            _ => true, // Other constraints handled differently
        }
    }

    /// Propagate equality constraint
    fn propagate_equality(&mut self, constraint: &Constraint) -> bool {
        let vars = constraint.variables();

        if vars.len() == 1 {
            // x = c
            let var = vars[0];
            if let Some(constant) = constraint.constant() {
                if let Some(existing) = self.assignment.get(&var) {
                    return existing == constant;
                }
                self.assignment.insert(var, constant.clone());
            }
        } else if vars.len() == 2 {
            // x = y
            let x = vars[0];
            let y = vars[1];

            match (
                self.assignment.get(&x).cloned(),
                self.assignment.get(&y).cloned(),
            ) {
                (Some(vx), Some(vy)) => return vx == vy,
                (Some(vx), None) => {
                    self.assignment.insert(y, vx);
                }
                (None, Some(vy)) => {
                    self.assignment.insert(x, vy);
                }
                (None, None) => {}
            }
        }

        true
    }

    /// Propagate disequality constraint
    fn propagate_disequality(&mut self, constraint: &Constraint) -> bool {
        let vars = constraint.variables();

        if vars.len() == 2 {
            let x = vars[0];
            let y = vars[1];

            if let (Some(vx), Some(vy)) = (self.assignment.get(&x), self.assignment.get(&y)) {
                return vx != vy;
            }
        }

        true
    }

    /// Propagate membership constraint
    fn propagate_membership(&mut self, constraint: &Constraint) -> bool {
        let vars = constraint.variables();

        if vars.len() == 1 {
            let var = vars[0];
            let values = constraint.value_set();

            if let Some(assigned) = self.assignment.get(&var) {
                return values.contains(assigned);
            }

            // Restrict domain
            let new_domain = Domain::finite(values.to_vec());
            if let Some(existing) = self.store.domains.get(&var) {
                let intersected = existing.intersect(&new_domain);
                if intersected.is_empty() {
                    return false;
                }
                self.store.set_domain(var, intersected);
            } else {
                self.store.set_domain(var, new_domain);
            }
        }

        true
    }

    /// Verify all constraints
    fn verify_all(&self) -> bool {
        for constraint in &self.store.constraints {
            if !self.verify_constraint(constraint) {
                return false;
            }
        }
        true
    }

    /// Verify a single constraint
    fn verify_constraint(&self, constraint: &Constraint) -> bool {
        let vars = constraint.variables();

        // Get assigned values
        let values: Vec<_> = vars.iter().filter_map(|v| self.assignment.get(v)).collect();

        // If not all variables assigned, consider satisfied
        if values.len() != vars.len() {
            return true;
        }

        match constraint.kind() {
            ConstraintKind::Equal => {
                if values.len() == 1 {
                    constraint.constant().is_none_or(|c| values[0] == c)
                } else {
                    values.windows(2).all(|w| w[0] == w[1])
                }
            }
            ConstraintKind::NotEqual => {
                if values.len() == 2 {
                    values[0] != values[1]
                } else {
                    true
                }
            }
            ConstraintKind::LessThan => {
                if values.len() == 2 {
                    values[0] < values[1]
                } else {
                    true
                }
            }
            ConstraintKind::LessEqual => {
                if values.len() == 2 {
                    values[0] <= values[1]
                } else {
                    true
                }
            }
            ConstraintKind::GreaterThan => {
                if values.len() == 2 {
                    values[0] > values[1]
                } else {
                    true
                }
            }
            ConstraintKind::GreaterEqual => {
                if values.len() == 2 {
                    values[0] >= values[1]
                } else {
                    true
                }
            }
            ConstraintKind::Member => values
                .first()
                .is_none_or(|v| constraint.value_set().contains(v)),
            ConstraintKind::NotMember => values
                .first()
                .is_none_or(|v| !constraint.value_set().contains(v)),
            ConstraintKind::Linear => {
                // Simplified linear constraint verification
                true
            }
        }
    }

    /// Reset solver state
    pub fn reset(&mut self) {
        self.assignment.clear();
        self.prop_queue.clear();
    }

    /// Clear all constraints and state
    pub fn clear(&mut self) {
        self.store.clear();
        self.assignment.clear();
        self.prop_queue.clear();
    }
}

impl Default for ClpSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::ThreadedRodeo;

    #[test]
    fn test_equality_constraint() {
        let interner = ThreadedRodeo::default();
        let x = Variable::new(interner.get_or_intern("x"));

        let mut solver = ClpSolver::new();
        solver.add_constraint(Constraint::equal_const(x, Value::Int64(42)));

        let result = solver.solve();
        match result {
            ClpResult::Sat(assignment) => {
                assert_eq!(assignment.get(&x), Some(&Value::Int64(42)));
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_variable_equality() {
        let interner = ThreadedRodeo::default();
        let x = Variable::new(interner.get_or_intern("x"));
        let y = Variable::new(interner.get_or_intern("y"));

        let mut solver = ClpSolver::new();
        solver.add_constraint(Constraint::equal(x, y));
        solver.assign(x, Value::Int64(10));

        let result = solver.solve();
        match result {
            ClpResult::Sat(assignment) => {
                assert_eq!(assignment.get(&x), assignment.get(&y));
            }
            _ => panic!("Expected SAT"),
        }
    }

    #[test]
    fn test_disequality_unsat() {
        let interner = ThreadedRodeo::default();
        let x = Variable::new(interner.get_or_intern("x"));
        let y = Variable::new(interner.get_or_intern("y"));

        let mut solver = ClpSolver::new();
        solver.add_constraint(Constraint::equal(x, y));
        solver.add_constraint(Constraint::not_equal(x, y));
        solver.assign(x, Value::Int64(5));

        let result = solver.solve();
        assert!(matches!(result, ClpResult::Unsat));
    }

    #[test]
    fn test_membership_constraint() {
        let interner = ThreadedRodeo::default();
        let x = Variable::new(interner.get_or_intern("x"));

        let mut solver = ClpSolver::new();
        solver.add_constraint(Constraint::member(
            x,
            vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)],
        ));
        solver.assign(x, Value::Int64(2));

        let result = solver.solve();
        assert!(matches!(result, ClpResult::Sat(_)));
    }

    #[test]
    fn test_domain_intersection() {
        let d1 = Domain::finite(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]);
        let d2 = Domain::finite(vec![Value::Int64(2), Value::Int64(3), Value::Int64(4)]);

        let intersected = d1.intersect(&d2);
        if let Domain::Finite(values) = intersected {
            assert_eq!(values.len(), 2);
            assert!(values.contains(&Value::Int64(2)));
            assert!(values.contains(&Value::Int64(3)));
        } else {
            panic!("Expected finite domain");
        }
    }

    #[test]
    fn test_domain_empty() {
        let d1 = Domain::finite(vec![Value::Int64(1)]);
        let d2 = Domain::finite(vec![Value::Int64(2)]);

        let intersected = d1.intersect(&d2);
        assert!(intersected.is_empty());
    }
}
