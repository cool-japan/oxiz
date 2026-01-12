//! Query processing for Datalog
//!
//! Provides query parsing, optimization, and execution.

use lasso::Spur;
use std::collections::HashSet;

use super::relation::RelationId;
use super::rule::{Atom, Binding, ComparisonOp, Term, Variable};
use super::schema::ColumnId;
use super::tuple::{Tuple, Value};

/// A Datalog query
#[derive(Debug, Clone)]
pub struct Query {
    /// Query pattern (head)
    pattern: Vec<Term>,
    /// Query body (conditions)
    body: Vec<Atom>,
    /// Relation being queried
    relation: Option<Spur>,
    /// Limit on results
    limit: Option<usize>,
    /// Offset for pagination
    offset: usize,
    /// Order by columns
    order_by: Vec<(ColumnId, bool)>, // (column, ascending)
}

impl Query {
    /// Create a new query for a relation
    pub fn new(relation: Spur, pattern: Vec<Term>) -> Self {
        Self {
            pattern,
            body: Vec::new(),
            relation: Some(relation),
            limit: None,
            offset: 0,
            order_by: Vec::new(),
        }
    }

    /// Create a query with body conditions
    pub fn with_body(pattern: Vec<Term>, body: Vec<Atom>) -> Self {
        Self {
            pattern,
            body,
            relation: None,
            limit: None,
            offset: 0,
            order_by: Vec::new(),
        }
    }

    /// Set result limit
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set offset
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    /// Add ordering
    pub fn order_by(mut self, col: ColumnId, ascending: bool) -> Self {
        self.order_by.push((col, ascending));
        self
    }

    /// Get pattern
    pub fn pattern(&self) -> &[Term] {
        &self.pattern
    }

    /// Get body
    pub fn body(&self) -> &[Atom] {
        &self.body
    }

    /// Get relation
    pub fn relation(&self) -> Option<Spur> {
        self.relation
    }

    /// Get limit
    pub fn get_limit(&self) -> Option<usize> {
        self.limit
    }

    /// Get offset
    pub fn get_offset(&self) -> usize {
        self.offset
    }

    /// Get ordering
    pub fn ordering(&self) -> &[(ColumnId, bool)] {
        &self.order_by
    }

    /// Get variables in query
    pub fn variables(&self) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        for term in &self.pattern {
            if let Term::Var(v) = term {
                vars.insert(*v);
            }
        }
        for atom in &self.body {
            vars.extend(atom.variables());
        }
        vars
    }

    /// Check if query is ground (no variables)
    pub fn is_ground(&self) -> bool {
        self.pattern.iter().all(|t| matches!(t, Term::Const(_)))
            && self.body.iter().all(|a| a.is_ground())
    }
}

/// Query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Plan nodes
    nodes: Vec<PlanNode>,
    /// Root node index
    root: usize,
    /// Estimated cost
    estimated_cost: f64,
    /// Statistics from planning
    stats: PlanStats,
}

/// Statistics collected during planning
#[derive(Debug, Clone, Default)]
pub struct PlanStats {
    /// Number of relations accessed
    pub relations_accessed: usize,
    /// Number of indexes used
    pub indexes_used: usize,
    /// Estimated result size
    pub estimated_rows: usize,
}

/// A node in the query plan
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Table scan
    Scan {
        relation: RelationId,
        filter: Option<Box<FilterExpr>>,
    },
    /// Index lookup
    IndexLookup {
        relation: RelationId,
        index: usize,
        key: Vec<Value>,
    },
    /// Filter operation
    Filter { input: usize, predicate: FilterExpr },
    /// Projection
    Project {
        input: usize,
        columns: Vec<ColumnId>,
    },
    /// Join
    Join {
        left: usize,
        right: usize,
        join_type: JoinType,
        condition: JoinCondition,
    },
    /// Union
    Union { inputs: Vec<usize> },
    /// Sort
    Sort {
        input: usize,
        keys: Vec<(ColumnId, bool)>,
    },
    /// Limit
    Limit {
        input: usize,
        limit: usize,
        offset: usize,
    },
    /// Aggregation
    Aggregate {
        input: usize,
        group_by: Vec<ColumnId>,
        aggregates: Vec<AggregateExpr>,
    },
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join
    Inner,
    /// Left outer join
    LeftOuter,
    /// Right outer join
    RightOuter,
    /// Full outer join
    FullOuter,
    /// Semi join (exists)
    Semi,
    /// Anti join (not exists)
    Anti,
}

/// Join condition
#[derive(Debug, Clone)]
pub struct JoinCondition {
    /// Left columns
    pub left_cols: Vec<ColumnId>,
    /// Right columns
    pub right_cols: Vec<ColumnId>,
}

/// Filter expression
#[derive(Debug, Clone)]
pub enum FilterExpr {
    /// Comparison
    Compare {
        left: ColumnExpr,
        op: ComparisonOp,
        right: ColumnExpr,
    },
    /// Conjunction
    And(Box<FilterExpr>, Box<FilterExpr>),
    /// Disjunction
    Or(Box<FilterExpr>, Box<FilterExpr>),
    /// Negation
    Not(Box<FilterExpr>),
    /// Is null check
    IsNull(ColumnExpr),
    /// Is not null check
    IsNotNull(ColumnExpr),
    /// In list
    In(ColumnExpr, Vec<Value>),
}

/// Column expression
#[derive(Debug, Clone)]
pub enum ColumnExpr {
    /// Column reference
    Column(ColumnId),
    /// Constant value
    Const(Value),
    /// Variable binding
    Var(Variable),
}

/// Aggregate expression
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// Aggregate function
    pub function: AggregateFunc,
    /// Input column
    pub input: ColumnId,
    /// Output alias
    pub alias: Option<String>,
}

/// Aggregate functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    /// Count
    Count,
    /// Sum
    Sum,
    /// Average
    Avg,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Count distinct
    CountDistinct,
}

impl QueryPlan {
    /// Create a new query plan
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            estimated_cost: 0.0,
            stats: PlanStats::default(),
        }
    }

    /// Add a plan node and return its index
    pub fn add_node(&mut self, node: PlanNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Set root node
    pub fn set_root(&mut self, idx: usize) {
        self.root = idx;
    }

    /// Get root node index
    pub fn root(&self) -> usize {
        self.root
    }

    /// Get node by index
    pub fn node(&self, idx: usize) -> Option<&PlanNode> {
        self.nodes.get(idx)
    }

    /// Get estimated cost
    pub fn cost(&self) -> f64 {
        self.estimated_cost
    }

    /// Set estimated cost
    pub fn set_cost(&mut self, cost: f64) {
        self.estimated_cost = cost;
    }

    /// Get statistics
    pub fn stats(&self) -> &PlanStats {
        &self.stats
    }

    /// Check if plan uses indexes
    pub fn uses_indexes(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n, PlanNode::IndexLookup { .. }))
    }
}

impl Default for QueryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of query execution
#[derive(Debug)]
pub struct QueryResult {
    /// Result tuples
    tuples: Vec<Tuple>,
    /// Column names for display
    columns: Vec<String>,
    /// Execution statistics
    stats: QueryExecStats,
}

/// Query execution statistics
#[derive(Debug, Default)]
pub struct QueryExecStats {
    /// Rows examined
    pub rows_examined: usize,
    /// Rows returned
    pub rows_returned: usize,
    /// Execution time in microseconds
    pub time_us: u64,
    /// Indexes used
    pub indexes_used: usize,
}

impl QueryResult {
    /// Create a new query result
    pub fn new(tuples: Vec<Tuple>, columns: Vec<String>) -> Self {
        let rows_returned = tuples.len();
        Self {
            tuples,
            columns,
            stats: QueryExecStats {
                rows_returned,
                ..Default::default()
            },
        }
    }

    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            tuples: Vec::new(),
            columns: Vec::new(),
            stats: QueryExecStats::default(),
        }
    }

    /// Get tuples
    pub fn tuples(&self) -> &[Tuple] {
        &self.tuples
    }

    /// Get columns
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        self.tuples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Get execution statistics
    pub fn stats(&self) -> &QueryExecStats {
        &self.stats
    }

    /// Iterate over tuples
    pub fn iter(&self) -> impl Iterator<Item = &Tuple> {
        self.tuples.iter()
    }

    /// Convert to bindings (for variable patterns)
    pub fn to_bindings(&self, pattern: &[Term]) -> Vec<Binding> {
        self.tuples
            .iter()
            .map(|tuple| {
                let mut binding = Binding::new();
                for (term, value) in pattern.iter().zip(tuple.values()) {
                    if let Term::Var(v) = term {
                        binding.bind(*v, value.clone());
                    }
                }
                binding
            })
            .collect()
    }
}

impl IntoIterator for QueryResult {
    type Item = Tuple;
    type IntoIter = std::vec::IntoIter<Tuple>;

    fn into_iter(self) -> Self::IntoIter {
        self.tuples.into_iter()
    }
}

/// Query optimizer
#[derive(Debug)]
pub struct QueryOptimizer {
    /// Cost model
    cost_model: CostModel,
    /// Enable index optimization
    use_indexes: bool,
    /// Enable join reordering
    reorder_joins: bool,
}

/// Cost model for query optimization
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Cost per row scan
    pub scan_cost: f64,
    /// Cost per index lookup
    pub index_lookup_cost: f64,
    /// Cost per hash join probe
    pub hash_join_cost: f64,
    /// Cost per sort operation
    pub sort_cost: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            scan_cost: 1.0,
            index_lookup_cost: 0.1,
            hash_join_cost: 0.5,
            sort_cost: 2.0,
        }
    }
}

impl QueryOptimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        Self {
            cost_model: CostModel::default(),
            use_indexes: true,
            reorder_joins: true,
        }
    }

    /// Create optimizer with custom cost model
    pub fn with_cost_model(cost_model: CostModel) -> Self {
        Self {
            cost_model,
            use_indexes: true,
            reorder_joins: true,
        }
    }

    /// Optimize a query
    pub fn optimize(&self, query: &Query) -> QueryPlan {
        let mut plan = QueryPlan::new();

        // Simple optimization: create a scan and filter
        if let Some(_rel) = query.relation() {
            // Add table scan (would resolve relation ID in real impl)
            let scan_idx = plan.add_node(PlanNode::Scan {
                relation: RelationId::new(0), // Placeholder
                filter: None,
            });
            plan.set_root(scan_idx);
        }

        // Add limit if specified
        if let Some(limit) = query.get_limit() {
            let input = plan.root();
            let limit_idx = plan.add_node(PlanNode::Limit {
                input,
                limit,
                offset: query.get_offset(),
            });
            plan.set_root(limit_idx);
        }

        // Add sort if ordering specified
        if !query.ordering().is_empty() {
            let input = plan.root();
            let sort_idx = plan.add_node(PlanNode::Sort {
                input,
                keys: query.ordering().to_vec(),
            });
            plan.set_root(sort_idx);
        }

        plan
    }

    /// Estimate cost of a plan
    pub fn estimate_cost(&self, plan: &QueryPlan) -> f64 {
        let mut total_cost = 0.0;

        for node in &plan.nodes {
            total_cost += match node {
                PlanNode::Scan { .. } => self.cost_model.scan_cost * 1000.0, // Assume 1000 rows
                PlanNode::IndexLookup { .. } => self.cost_model.index_lookup_cost,
                PlanNode::Filter { .. } => self.cost_model.scan_cost * 0.1,
                PlanNode::Join { .. } => self.cost_model.hash_join_cost * 100.0,
                PlanNode::Sort { .. } => self.cost_model.sort_cost * 100.0,
                PlanNode::Limit { .. } => 0.0,
                _ => self.cost_model.scan_cost,
            };
        }

        total_cost
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Query builder for fluent API
pub struct QueryBuilder {
    query: Query,
    interner: lasso::ThreadedRodeo,
}

impl QueryBuilder {
    /// Create a new query builder
    pub fn select(relation: &str) -> Self {
        let interner = lasso::ThreadedRodeo::default();
        let rel = interner.get_or_intern(relation);
        Self {
            query: Query::new(rel, Vec::new()),
            interner,
        }
    }

    /// Add a variable to select
    pub fn var(mut self, name: &str) -> Self {
        let v = Term::var(self.interner.get_or_intern(name));
        self.query.pattern.push(v);
        self
    }

    /// Add a constant to select
    pub fn constant(mut self, value: Value) -> Self {
        self.query.pattern.push(Term::constant(value));
        self
    }

    /// Add a wildcard
    pub fn wildcard(mut self) -> Self {
        self.query.pattern.push(Term::wildcard());
        self
    }

    /// Set limit
    pub fn limit(mut self, n: usize) -> Self {
        self.query = self.query.limit(n);
        self
    }

    /// Set offset
    pub fn offset(mut self, n: usize) -> Self {
        self.query = self.query.offset(n);
        self
    }

    /// Build the query
    pub fn build(self) -> Query {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_creation() {
        let interner = lasso::ThreadedRodeo::default();
        let rel = interner.get_or_intern("edge");
        let x = Term::var(interner.get_or_intern("x"));
        let y = Term::var(interner.get_or_intern("y"));

        let query = Query::new(rel, vec![x, y]);
        assert_eq!(query.pattern().len(), 2);
        assert!(query.relation().is_some());
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::select("edge")
            .var("x")
            .var("y")
            .limit(10)
            .build();

        assert_eq!(query.pattern().len(), 2);
        assert_eq!(query.get_limit(), Some(10));
    }

    #[test]
    fn test_query_variables() {
        let interner = lasso::ThreadedRodeo::default();
        let rel = interner.get_or_intern("rel");
        let x = Term::var(interner.get_or_intern("x"));
        let c = Term::constant(Value::Int64(42));

        let query = Query::new(rel, vec![x, c]);
        let vars = query.variables();
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn test_query_result() {
        use crate::datalog::tuple::TupleBuilder;

        let tuples = vec![
            TupleBuilder::new().push_i64(1).push_i64(2).build(),
            TupleBuilder::new().push_i64(3).push_i64(4).build(),
        ];

        let result = QueryResult::new(tuples, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(result.len(), 2);
        assert_eq!(result.columns().len(), 2);
    }

    #[test]
    fn test_query_optimizer() {
        let interner = lasso::ThreadedRodeo::default();
        let rel = interner.get_or_intern("edge");

        let query = Query::new(rel, vec![Term::wildcard(), Term::wildcard()]).limit(100);

        let optimizer = QueryOptimizer::new();
        let plan = optimizer.optimize(&query);

        assert!(plan.nodes.len() > 0);
    }

    #[test]
    fn test_plan_node_creation() {
        let mut plan = QueryPlan::new();

        let scan = plan.add_node(PlanNode::Scan {
            relation: RelationId::new(1),
            filter: None,
        });

        let limit = plan.add_node(PlanNode::Limit {
            input: scan,
            limit: 10,
            offset: 0,
        });

        plan.set_root(limit);
        assert_eq!(plan.root(), 1);
    }
}
