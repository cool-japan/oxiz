//! Datalog Relations Engine
//!
//! A complete implementation of a Datalog evaluation engine with support for:
//! - Extensible Database (EDB) and Intensional Database (IDB) relations
//! - Semi-naive evaluation with stratification
//! - Efficient join algorithms (hash join, merge join)
//! - Indexed relation storage
//! - Constraint Logic Programming (CLP) integration
//! - Query compilation and optimization

// Allow dead code as this is a library module with public APIs for future use
#![allow(dead_code)]

mod clp;
mod compiler;
mod engine;
mod execution;
mod index;
mod query;
mod relation;
mod rule;
mod schema;
mod tuple;

pub use clp::{ClpSolver, Constraint, ConstraintKind};
pub use compiler::{CompiledRule, ExecutionPlan, RuleCompiler};
pub use engine::{DatalogEngine, EngineConfig, EvaluationResult};
pub use execution::{ExecutionContext, ExecutionStats};
pub use index::{Index, IndexId, IndexKind};
pub use query::{Query, QueryPlan, QueryResult};
pub use relation::{Relation, RelationId, RelationKind};
pub use rule::{Atom, AtomKind, Binding, Rule, RuleId, Term as RuleTerm, Variable};
pub use schema::{Column, ColumnId, Schema};
pub use tuple::{Tuple, TupleBuilder, TupleRef};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Basic sanity test that modules compile
        let schema = Schema::new("test".to_string());
        assert_eq!(schema.name(), "test");
    }
}
