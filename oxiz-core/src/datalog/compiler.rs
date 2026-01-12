//! Rule compilation for Datalog
//!
//! Compiles Datalog rules into efficient execution plans.

use std::collections::{HashMap, HashSet};

use super::index::IndexId;
use super::relation::RelationId;
use super::rule::{ArithOp, Atom, AtomKind, ComparisonOp, Rule, RuleId, Term, Variable};
use super::schema::ColumnId;

/// Compiled rule ready for execution
#[derive(Debug)]
pub struct CompiledRule {
    /// Original rule ID
    rule_id: RuleId,
    /// Execution plan
    plan: ExecutionPlan,
    /// Variable positions in output
    output_vars: Vec<Variable>,
    /// Is recursive
    is_recursive: bool,
    /// Compilation statistics
    stats: CompilationStats,
}

/// Statistics from compilation
#[derive(Debug, Default)]
pub struct CompilationStats {
    /// Number of joins
    pub join_count: usize,
    /// Number of filters
    pub filter_count: usize,
    /// Index lookups used
    pub index_lookups: usize,
    /// Compilation time in microseconds
    pub compile_time_us: u64,
}

impl CompiledRule {
    /// Get rule ID
    pub fn rule_id(&self) -> RuleId {
        self.rule_id
    }

    /// Get execution plan
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Get output variables
    pub fn output_vars(&self) -> &[Variable] {
        &self.output_vars
    }

    /// Check if recursive
    pub fn is_recursive(&self) -> bool {
        self.is_recursive
    }

    /// Get compilation stats
    pub fn stats(&self) -> &CompilationStats {
        &self.stats
    }
}

/// Execution plan for a rule
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Operations in execution order
    operations: Vec<Operation>,
    /// Estimated cost
    cost: f64,
}

impl ExecutionPlan {
    /// Create a new execution plan
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            cost: 0.0,
        }
    }

    /// Add an operation
    pub fn add_operation(&mut self, op: Operation) {
        self.operations.push(op);
    }

    /// Get operations
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Get estimated cost
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Set cost
    pub fn set_cost(&mut self, cost: f64) {
        self.cost = cost;
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// An operation in the execution plan
#[derive(Debug, Clone)]
pub enum Operation {
    /// Scan a relation
    Scan {
        relation: RelationId,
        output_vars: Vec<Variable>,
    },
    /// Scan delta (for semi-naive)
    ScanDelta {
        relation: RelationId,
        output_vars: Vec<Variable>,
    },
    /// Index lookup
    IndexLookup {
        relation: RelationId,
        index: IndexId,
        key_vars: Vec<Variable>,
        output_vars: Vec<Variable>,
    },
    /// Hash join
    HashJoin {
        left_vars: Vec<Variable>,
        right_vars: Vec<Variable>,
        join_vars: Vec<Variable>,
    },
    /// Nested loop join
    NestedLoopJoin {
        outer_vars: Vec<Variable>,
        inner_vars: Vec<Variable>,
    },
    /// Filter with comparison
    Filter {
        left: FilterOperand,
        op: ComparisonOp,
        right: FilterOperand,
    },
    /// Arithmetic computation
    Compute {
        result: Variable,
        left: FilterOperand,
        op: ArithOp,
        right: FilterOperand,
    },
    /// Project to output variables
    Project { vars: Vec<Variable> },
    /// Deduplicate results
    Deduplicate,
    /// Check negation (anti-join)
    AntiJoin {
        relation: RelationId,
        join_vars: Vec<Variable>,
    },
}

/// Operand in a filter or computation
#[derive(Debug, Clone)]
pub enum FilterOperand {
    /// Variable reference
    Var(Variable),
    /// Constant value
    Const(i64),
    /// Column reference
    Column(ColumnId),
}

/// Rule compiler
#[derive(Debug)]
pub struct RuleCompiler {
    /// Available indexes per relation
    relation_indexes: HashMap<RelationId, Vec<IndexInfo>>,
    /// Cost model
    cost_model: CompilerCostModel,
    /// Enable optimizations
    optimize: bool,
}

/// Index information for optimization
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Index ID
    pub id: IndexId,
    /// Indexed columns
    pub columns: Vec<ColumnId>,
    /// Is unique
    pub is_unique: bool,
}

/// Cost model for compilation
#[derive(Debug, Clone)]
pub struct CompilerCostModel {
    /// Cost per tuple scan
    pub scan_cost: f64,
    /// Cost per index lookup
    pub index_cost: f64,
    /// Cost per hash join probe
    pub hash_join_cost: f64,
    /// Cost per nested loop iteration
    pub nested_loop_cost: f64,
}

impl Default for CompilerCostModel {
    fn default() -> Self {
        Self {
            scan_cost: 1.0,
            index_cost: 0.1,
            hash_join_cost: 0.5,
            nested_loop_cost: 10.0,
        }
    }
}

impl RuleCompiler {
    /// Create a new compiler
    pub fn new() -> Self {
        Self {
            relation_indexes: HashMap::new(),
            cost_model: CompilerCostModel::default(),
            optimize: true,
        }
    }

    /// Register an index for a relation
    pub fn register_index(&mut self, relation: RelationId, info: IndexInfo) {
        self.relation_indexes
            .entry(relation)
            .or_default()
            .push(info);
    }

    /// Compile a rule
    pub fn compile(&self, rule: &Rule) -> CompiledRule {
        let start = std::time::Instant::now();

        let mut plan = ExecutionPlan::new();
        let mut stats = CompilationStats::default();

        if rule.is_fact() {
            // Facts don't need execution plan
            return CompiledRule {
                rule_id: rule.id(),
                plan,
                output_vars: Vec::new(),
                is_recursive: false,
                stats,
            };
        }

        // Collect output variables from head
        let output_vars: Vec<_> = rule
            .head()
            .terms()
            .iter()
            .filter_map(|t| t.as_var())
            .collect();

        // Order body atoms for optimal execution
        let ordered_atoms = self.order_atoms(rule.body());

        // Generate operations for each body atom
        let mut bound_vars: HashSet<Variable> = HashSet::new();

        for atom in &ordered_atoms {
            match atom.kind() {
                AtomKind::Positive => {
                    self.compile_positive_atom(atom, &bound_vars, &mut plan, &mut stats);
                    // Add atom's variables to bound set
                    for var in atom.variables() {
                        bound_vars.insert(var);
                    }
                }
                AtomKind::Negated => {
                    self.compile_negated_atom(atom, &bound_vars, &mut plan, &mut stats);
                }
                AtomKind::Comparison => {
                    self.compile_comparison_atom(atom, &mut plan, &mut stats);
                }
                AtomKind::Arithmetic => {
                    self.compile_arithmetic_atom(atom, &mut plan, &mut stats);
                }
                AtomKind::Aggregate => {
                    // Aggregates handled specially
                }
            }
        }

        // Add projection to output vars
        if !output_vars.is_empty() {
            plan.add_operation(Operation::Project {
                vars: output_vars.clone(),
            });
        }

        // Add deduplication
        plan.add_operation(Operation::Deduplicate);

        stats.compile_time_us = start.elapsed().as_micros() as u64;

        CompiledRule {
            rule_id: rule.id(),
            plan,
            output_vars,
            is_recursive: rule.is_recursive(),
            stats,
        }
    }

    /// Order atoms for optimal evaluation
    fn order_atoms<'a>(&self, atoms: &'a [Atom]) -> Vec<&'a Atom> {
        // Simple ordering: positive first, then comparisons, then negated
        let positive: Vec<_> = atoms
            .iter()
            .filter(|a| a.kind() == AtomKind::Positive)
            .collect();
        let comparisons: Vec<_> = atoms
            .iter()
            .filter(|a| a.kind() == AtomKind::Comparison)
            .collect();
        let arithmetic: Vec<_> = atoms
            .iter()
            .filter(|a| a.kind() == AtomKind::Arithmetic)
            .collect();
        let negated: Vec<_> = atoms
            .iter()
            .filter(|a| a.kind() == AtomKind::Negated)
            .collect();

        // Simple heuristic: smaller relations first
        // In real impl, would use statistics

        let mut result = Vec::with_capacity(atoms.len());
        result.extend(positive);
        result.extend(arithmetic);
        result.extend(comparisons);
        result.extend(negated);
        result
    }

    /// Compile a positive atom
    fn compile_positive_atom(
        &self,
        atom: &Atom,
        bound_vars: &HashSet<Variable>,
        plan: &mut ExecutionPlan,
        stats: &mut CompilationStats,
    ) {
        let rel_id = match atom.relation_id() {
            Some(id) => id,
            None => return,
        };

        // Collect bound and unbound variables
        let atom_vars: Vec<_> = atom.terms().iter().filter_map(|t| t.as_var()).collect();

        let bound_in_atom: Vec<_> = atom_vars
            .iter()
            .filter(|v| bound_vars.contains(v))
            .copied()
            .collect();

        if bound_vars.is_empty() {
            // First atom - do a scan
            plan.add_operation(Operation::Scan {
                relation: rel_id,
                output_vars: atom_vars,
            });
        } else if !bound_in_atom.is_empty() {
            // Can use join on bound variables
            plan.add_operation(Operation::HashJoin {
                left_vars: bound_in_atom.clone(),
                right_vars: bound_in_atom,
                join_vars: atom_vars,
            });
            stats.join_count += 1;
        } else {
            // No shared variables - cartesian product (nested loop)
            plan.add_operation(Operation::NestedLoopJoin {
                outer_vars: bound_vars.iter().copied().collect(),
                inner_vars: atom_vars,
            });
            stats.join_count += 1;
        }
    }

    /// Compile a negated atom
    fn compile_negated_atom(
        &self,
        atom: &Atom,
        _bound_vars: &HashSet<Variable>,
        plan: &mut ExecutionPlan,
        _stats: &mut CompilationStats,
    ) {
        let rel_id = match atom.relation_id() {
            Some(id) => id,
            None => return,
        };

        let join_vars: Vec<_> = atom.terms().iter().filter_map(|t| t.as_var()).collect();

        plan.add_operation(Operation::AntiJoin {
            relation: rel_id,
            join_vars,
        });
    }

    /// Compile a comparison atom
    fn compile_comparison_atom(
        &self,
        atom: &Atom,
        plan: &mut ExecutionPlan,
        stats: &mut CompilationStats,
    ) {
        let op = match atom.comparison_op() {
            Some(op) => op,
            None => return,
        };

        let terms = atom.terms();
        if terms.len() != 2 {
            return;
        }

        let left = self.term_to_operand(&terms[0]);
        let right = self.term_to_operand(&terms[1]);

        plan.add_operation(Operation::Filter { left, op, right });
        stats.filter_count += 1;
    }

    /// Compile an arithmetic atom
    fn compile_arithmetic_atom(
        &self,
        atom: &Atom,
        plan: &mut ExecutionPlan,
        _stats: &mut CompilationStats,
    ) {
        let op = match atom.arith_op() {
            Some(op) => op,
            None => return,
        };

        let terms = atom.terms();
        if terms.len() != 3 {
            return;
        }

        let result = match terms[0].as_var() {
            Some(v) => v,
            None => return,
        };

        let left = self.term_to_operand(&terms[1]);
        let right = self.term_to_operand(&terms[2]);

        plan.add_operation(Operation::Compute {
            result,
            left,
            op,
            right,
        });
    }

    /// Convert a term to filter operand
    fn term_to_operand(&self, term: &Term) -> FilterOperand {
        match term {
            Term::Var(v) => FilterOperand::Var(*v),
            Term::Const(c) => {
                match c.as_i64() {
                    Some(n) => FilterOperand::Const(n),
                    None => FilterOperand::Const(0), // Fallback
                }
            }
            _ => FilterOperand::Const(0),
        }
    }

    /// Estimate cost of a plan
    pub fn estimate_cost(&self, plan: &ExecutionPlan) -> f64 {
        let mut cost = 0.0;
        for op in plan.operations() {
            cost += match op {
                Operation::Scan { .. } => self.cost_model.scan_cost * 1000.0,
                Operation::ScanDelta { .. } => self.cost_model.scan_cost * 100.0,
                Operation::IndexLookup { .. } => self.cost_model.index_cost,
                Operation::HashJoin { .. } => self.cost_model.hash_join_cost * 100.0,
                Operation::NestedLoopJoin { .. } => self.cost_model.nested_loop_cost * 1000.0,
                Operation::Filter { .. } => 0.1,
                Operation::Compute { .. } => 0.01,
                Operation::Project { .. } => 0.01,
                Operation::Deduplicate => 10.0,
                Operation::AntiJoin { .. } => self.cost_model.hash_join_cost * 100.0,
            };
        }
        cost
    }
}

impl Default for RuleCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch compiler for multiple rules
#[derive(Debug)]
pub struct BatchCompiler {
    /// Individual rule compiler
    compiler: RuleCompiler,
    /// Compiled rules cache
    cache: HashMap<RuleId, CompiledRule>,
}

impl BatchCompiler {
    /// Create a new batch compiler
    pub fn new() -> Self {
        Self {
            compiler: RuleCompiler::new(),
            cache: HashMap::new(),
        }
    }

    /// Compile all rules
    pub fn compile_all(&mut self, rules: &[Rule]) -> Vec<&CompiledRule> {
        for rule in rules {
            if !self.cache.contains_key(&rule.id()) {
                let compiled = self.compiler.compile(rule);
                self.cache.insert(rule.id(), compiled);
            }
        }

        rules
            .iter()
            .filter_map(|r| self.cache.get(&r.id()))
            .collect()
    }

    /// Get compiled rule
    pub fn get(&self, id: RuleId) -> Option<&CompiledRule> {
        self.cache.get(&id)
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for BatchCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datalog::tuple::Value;

    #[test]
    fn test_compile_simple_rule() {
        let interner = lasso::ThreadedRodeo::default();

        let edge = interner.get_or_intern("edge");
        let path = interner.get_or_intern("path");
        let x = Term::var(interner.get_or_intern("x"));
        let y = Term::var(interner.get_or_intern("y"));

        // path(x, y) :- edge(x, y)
        let head = Atom::positive(path, vec![x.clone(), y.clone()]);
        let body = vec![Atom::positive(edge, vec![x, y])];

        let mut rule = Rule::new(RuleId::new(1), head, body);
        rule.head_mut().set_relation_id(RelationId::new(1));
        rule.body_mut()[0].set_relation_id(RelationId::new(2));

        let compiler = RuleCompiler::new();
        let compiled = compiler.compile(&rule);

        assert!(!compiled.is_recursive());
        assert_eq!(compiled.output_vars().len(), 2);
    }

    #[test]
    fn test_compile_recursive_rule() {
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

        let mut rule = Rule::new(RuleId::new(1), head, body);
        rule.head_mut().set_relation_id(RelationId::new(1));
        rule.body_mut()[0].set_relation_id(RelationId::new(1));
        rule.body_mut()[1].set_relation_id(RelationId::new(2));

        let compiler = RuleCompiler::new();
        let compiled = compiler.compile(&rule);

        assert!(compiled.is_recursive());
    }

    #[test]
    fn test_compile_rule_with_comparison() {
        let interner = lasso::ThreadedRodeo::default();

        let num = interner.get_or_intern("num");
        let big = interner.get_or_intern("big");
        let x = Term::var(interner.get_or_intern("x"));

        // big(x) :- num(x), x > 10
        let head = Atom::positive(big, vec![x.clone()]);
        let body = vec![
            Atom::positive(num, vec![x.clone()]),
            Atom::comparison(x, ComparisonOp::Gt, Term::constant(Value::Int64(10))),
        ];

        let mut rule = Rule::new(RuleId::new(1), head, body);
        rule.head_mut().set_relation_id(RelationId::new(1));
        rule.body_mut()[0].set_relation_id(RelationId::new(2));

        let compiler = RuleCompiler::new();
        let compiled = compiler.compile(&rule);

        assert_eq!(compiled.stats().filter_count, 1);
    }

    #[test]
    fn test_batch_compiler() {
        let interner = lasso::ThreadedRodeo::default();

        let rel = interner.get_or_intern("rel");
        let x = Term::var(interner.get_or_intern("x"));

        let head = Atom::positive(rel, vec![x.clone()]);
        let body = vec![Atom::positive(rel, vec![x])];

        let mut rule1 = Rule::new(RuleId::new(1), head.clone(), body.clone());
        rule1.head_mut().set_relation_id(RelationId::new(1));
        rule1.body_mut()[0].set_relation_id(RelationId::new(1));

        let mut rule2 = Rule::new(RuleId::new(2), head, body);
        rule2.head_mut().set_relation_id(RelationId::new(1));
        rule2.body_mut()[0].set_relation_id(RelationId::new(1));

        let mut batch = BatchCompiler::new();
        let compiled = batch.compile_all(&[rule1, rule2]);

        assert_eq!(compiled.len(), 2);
    }
}
