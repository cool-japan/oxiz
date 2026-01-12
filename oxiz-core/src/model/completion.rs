//! Model Completion
//!
//! Completes partial models with default values for undefined terms.

use super::{Model, ValueFactory};
use crate::ast::{TermId, TermKind, TermManager};
use crate::sort::SortId;
use std::collections::HashSet;

/// Configuration for model completion
#[derive(Debug, Clone)]
pub struct ModelCompletionConfig {
    /// Use minimal values (0, false, etc.)
    pub use_minimal: bool,
    /// Complete function applications
    pub complete_functions: bool,
    /// Complete array defaults
    pub complete_arrays: bool,
}

impl Default for ModelCompletionConfig {
    fn default() -> Self {
        Self {
            use_minimal: true,
            complete_functions: true,
            complete_arrays: true,
        }
    }
}

/// Model completion utility
#[derive(Debug)]
pub struct ModelCompletion {
    config: ModelCompletionConfig,
    factory: ValueFactory,
    completed: HashSet<TermId>,
}

impl ModelCompletion {
    /// Create a new model completion utility
    pub fn new() -> Self {
        Self {
            config: ModelCompletionConfig::default(),
            factory: ValueFactory::new(),
            completed: HashSet::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ModelCompletionConfig) -> Self {
        Self {
            config,
            factory: ValueFactory::new(),
            completed: HashSet::new(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ModelCompletionConfig {
        &self.config
    }

    /// Complete a model by assigning default values to undefined terms
    pub fn complete(&mut self, model: &mut Model, terms: &[TermId], manager: &TermManager) {
        self.completed.clear();

        for &term in terms {
            if !model.has(term) && !self.completed.contains(&term) {
                self.complete_term(model, term, manager);
            }
        }
    }

    /// Complete a single term with a default value
    fn complete_term(&mut self, model: &mut Model, term: TermId, manager: &TermManager) {
        if self.completed.contains(&term) || model.has(term) {
            return;
        }

        self.completed.insert(term);

        // Determine sort from term structure
        if let Some(t) = manager.get(term) {
            let sort = self.infer_sort(&t.kind);
            let value = self.factory.default_value(sort);
            model.assign(term, value);
        }
    }

    /// Infer sort from term kind
    fn infer_sort(&self, kind: &TermKind) -> SortId {
        match kind {
            TermKind::True
            | TermKind::False
            | TermKind::Not(_)
            | TermKind::And(_)
            | TermKind::Or(_)
            | TermKind::Xor(_, _)
            | TermKind::Implies(_, _)
            | TermKind::Eq(_, _)
            | TermKind::Distinct(_)
            | TermKind::Lt(_, _)
            | TermKind::Le(_, _)
            | TermKind::Gt(_, _)
            | TermKind::Ge(_, _) => SortId(0), // Bool

            TermKind::IntConst(_)
            | TermKind::Neg(_)
            | TermKind::Add(_)
            | TermKind::Sub(_, _)
            | TermKind::Mul(_)
            | TermKind::Div(_, _)
            | TermKind::Mod(_, _) => SortId(1), // Int

            TermKind::RealConst(_) => SortId(2), // Real

            TermKind::BitVecConst { .. }
            | TermKind::BvNot(_)
            | TermKind::BvAnd(_, _)
            | TermKind::BvOr(_, _)
            | TermKind::BvXor(_, _)
            | TermKind::BvAdd(_, _)
            | TermKind::BvSub(_, _)
            | TermKind::BvMul(_, _) => SortId(5), // BitVec

            _ => SortId(100), // Uninterpreted/Unknown
        }
    }

    /// Complete all variables in a formula
    pub fn complete_variables(&mut self, model: &mut Model, root: TermId, manager: &TermManager) {
        let mut visited = HashSet::new();
        let mut worklist = vec![root];

        while let Some(term) = worklist.pop() {
            if visited.contains(&term) {
                continue;
            }
            visited.insert(term);

            if let Some(t) = manager.get(term) {
                match &t.kind {
                    TermKind::Var(_) => {
                        if !model.has(term) {
                            self.complete_term(model, term, manager);
                        }
                    }
                    _ => {
                        // Add children to worklist based on term kind
                        self.add_children(&t.kind, &mut worklist);
                    }
                }
            }
        }
    }

    /// Add children of a term kind to the worklist
    fn add_children(&self, kind: &TermKind, worklist: &mut Vec<TermId>) {
        match kind {
            TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => {
                worklist.push(*a);
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::Distinct(args) => {
                for arg in args.iter() {
                    worklist.push(*arg);
                }
            }
            TermKind::Xor(a, b)
            | TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::BvAnd(a, b)
            | TermKind::BvOr(a, b)
            | TermKind::BvXor(a, b)
            | TermKind::BvAdd(a, b)
            | TermKind::BvSub(a, b)
            | TermKind::BvMul(a, b)
            | TermKind::BvConcat(a, b) => {
                worklist.push(*a);
                worklist.push(*b);
            }
            TermKind::Ite(a, b, c) => {
                worklist.push(*a);
                worklist.push(*b);
                worklist.push(*c);
            }
            _ => {}
        }
    }

    /// Reset completion state
    pub fn reset(&mut self) {
        self.completed.clear();
    }

    /// Number of terms completed
    pub fn num_completed(&self) -> usize {
        self.completed.len()
    }
}

impl Default for ModelCompletion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_config() {
        let config = ModelCompletionConfig::default();
        assert!(config.use_minimal);
        assert!(config.complete_functions);
        assert!(config.complete_arrays);
    }

    #[test]
    fn test_completion_creation() {
        let completion = ModelCompletion::new();
        assert_eq!(completion.num_completed(), 0);
    }

    #[test]
    fn test_completion_reset() {
        let mut completion = ModelCompletion::new();
        completion.completed.insert(TermId::from(1u32));
        assert_eq!(completion.num_completed(), 1);

        completion.reset();
        assert_eq!(completion.num_completed(), 0);
    }
}
