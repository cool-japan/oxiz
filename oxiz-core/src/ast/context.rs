//! Context for incremental solving with push/pop support
//!
//! The context maintains a stack of assertion scopes, allowing
//! incremental solving with backtracking via push/pop operations.

use crate::ast::TermId;

/// A context for incremental solving
#[derive(Debug, Clone)]
pub struct Context {
    /// Stack of assertion scopes
    scopes: Vec<Scope>,
    /// All assertions in the current context (flattened view)
    assertions: Vec<TermId>,
}

/// A scope represents a level in the assertion stack
#[derive(Debug, Clone)]
struct Scope {
    /// The number of assertions at the start of this scope
    assertion_base: usize,
}

impl Context {
    /// Create a new empty context
    #[must_use]
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope { assertion_base: 0 }],
            assertions: Vec::new(),
        }
    }

    /// Add an assertion to the current scope
    pub fn assert(&mut self, term: TermId) {
        self.assertions.push(term);
    }

    /// Add multiple assertions to the current scope
    pub fn assert_many(&mut self, terms: impl IntoIterator<Item = TermId>) {
        self.assertions.extend(terms);
    }

    /// Push a new assertion scope onto the stack
    pub fn push(&mut self) {
        let assertion_base = self.assertions.len();
        self.scopes.push(Scope { assertion_base });
    }

    /// Pop the most recent assertion scope from the stack
    ///
    /// Returns `true` if a scope was popped, `false` if already at base level
    pub fn pop(&mut self) -> bool {
        if self.scopes.len() <= 1 {
            // Cannot pop the base scope
            return false;
        }

        let scope = self
            .scopes
            .pop()
            .expect("scopes has elements after length check");
        self.assertions.truncate(scope.assertion_base);
        true
    }

    /// Get the current number of scopes
    #[must_use]
    pub fn num_scopes(&self) -> usize {
        self.scopes.len()
    }

    /// Get all assertions in the current context
    #[must_use]
    pub fn assertions(&self) -> &[TermId] {
        &self.assertions
    }

    /// Get the number of assertions in the current context
    #[must_use]
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Check if the context is empty (no assertions)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assertions.is_empty()
    }

    /// Reset the context to the initial state
    pub fn reset(&mut self) {
        self.scopes.clear();
        self.scopes.push(Scope { assertion_base: 0 });
        self.assertions.clear();
    }

    /// Get assertions added in the current scope
    #[must_use]
    pub fn current_scope_assertions(&self) -> &[TermId] {
        let current_scope = self
            .scopes
            .last()
            .expect("scopes always has at least base scope");
        &self.assertions[current_scope.assertion_base..]
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// A named assertion for unsat core tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedAssertion {
    /// The assertion term
    pub term: TermId,
    /// Optional name for the assertion
    pub name: Option<String>,
}

impl NamedAssertion {
    /// Create a new named assertion
    #[must_use]
    pub fn new(term: TermId, name: Option<String>) -> Self {
        Self { term, name }
    }

    /// Create an unnamed assertion
    #[must_use]
    pub fn unnamed(term: TermId) -> Self {
        Self { term, name: None }
    }

    /// Create a named assertion
    #[must_use]
    pub fn named(term: TermId, name: impl Into<String>) -> Self {
        Self {
            term,
            name: Some(name.into()),
        }
    }
}

/// Extended context with named assertions for unsat core support
#[derive(Debug, Clone)]
pub struct NamedContext {
    /// Stack of assertion scopes
    scopes: Vec<Scope>,
    /// Named assertions in the current context
    assertions: Vec<NamedAssertion>,
}

impl NamedContext {
    /// Create a new empty named context
    #[must_use]
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope { assertion_base: 0 }],
            assertions: Vec::new(),
        }
    }

    /// Add a named assertion to the current scope
    pub fn assert_named(&mut self, term: TermId, name: Option<String>) {
        self.assertions.push(NamedAssertion::new(term, name));
    }

    /// Add an unnamed assertion to the current scope
    pub fn assert(&mut self, term: TermId) {
        self.assertions.push(NamedAssertion::unnamed(term));
    }

    /// Push a new assertion scope onto the stack
    pub fn push(&mut self) {
        let assertion_base = self.assertions.len();
        self.scopes.push(Scope { assertion_base });
    }

    /// Pop the most recent assertion scope from the stack
    ///
    /// Returns `true` if a scope was popped, `false` if already at base level
    pub fn pop(&mut self) -> bool {
        if self.scopes.len() <= 1 {
            return false;
        }

        let scope = self
            .scopes
            .pop()
            .expect("scopes has elements after length check");
        self.assertions.truncate(scope.assertion_base);
        true
    }

    /// Get the current number of scopes
    #[must_use]
    pub fn num_scopes(&self) -> usize {
        self.scopes.len()
    }

    /// Get all assertions in the current context
    #[must_use]
    pub fn assertions(&self) -> &[NamedAssertion] {
        &self.assertions
    }

    /// Get all assertion terms (without names)
    #[must_use]
    pub fn assertion_terms(&self) -> Vec<TermId> {
        self.assertions.iter().map(|a| a.term).collect()
    }

    /// Get the number of assertions in the current context
    #[must_use]
    pub fn num_assertions(&self) -> usize {
        self.assertions.len()
    }

    /// Check if the context is empty (no assertions)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assertions.is_empty()
    }

    /// Reset the context to the initial state
    pub fn reset(&mut self) {
        self.scopes.clear();
        self.scopes.push(Scope { assertion_base: 0 });
        self.assertions.clear();
    }

    /// Get assertions by name
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Vec<TermId> {
        self.assertions
            .iter()
            .filter_map(|a| {
                if a.name.as_deref() == Some(name) {
                    Some(a.term)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for NamedContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_context() {
        let ctx = Context::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.num_scopes(), 1);
        assert_eq!(ctx.num_assertions(), 0);
    }

    #[test]
    fn test_assert() {
        let mut ctx = Context::new();

        ctx.assert(TermId(1));
        ctx.assert(TermId(2));

        assert!(!ctx.is_empty());
        assert_eq!(ctx.num_assertions(), 2);
        assert_eq!(ctx.assertions(), &[TermId(1), TermId(2)]);
    }

    #[test]
    fn test_push_pop() {
        let mut ctx = Context::new();

        // Base level: add assertions 1, 2
        ctx.assert(TermId(1));
        ctx.assert(TermId(2));
        assert_eq!(ctx.num_scopes(), 1);

        // Push scope 1: add assertion 3
        ctx.push();
        ctx.assert(TermId(3));
        assert_eq!(ctx.num_scopes(), 2);
        assert_eq!(ctx.num_assertions(), 3);

        // Push scope 2: add assertion 4
        ctx.push();
        ctx.assert(TermId(4));
        assert_eq!(ctx.num_scopes(), 3);
        assert_eq!(ctx.num_assertions(), 4);

        // Pop scope 2
        assert!(ctx.pop());
        assert_eq!(ctx.num_scopes(), 2);
        assert_eq!(ctx.num_assertions(), 3);
        assert_eq!(ctx.assertions(), &[TermId(1), TermId(2), TermId(3)]);

        // Pop scope 1
        assert!(ctx.pop());
        assert_eq!(ctx.num_scopes(), 1);
        assert_eq!(ctx.num_assertions(), 2);
        assert_eq!(ctx.assertions(), &[TermId(1), TermId(2)]);

        // Cannot pop base scope
        assert!(!ctx.pop());
        assert_eq!(ctx.num_scopes(), 1);
        assert_eq!(ctx.num_assertions(), 2);
    }

    #[test]
    fn test_reset() {
        let mut ctx = Context::new();

        ctx.assert(TermId(1));
        ctx.push();
        ctx.assert(TermId(2));

        ctx.reset();

        assert!(ctx.is_empty());
        assert_eq!(ctx.num_scopes(), 1);
        assert_eq!(ctx.num_assertions(), 0);
    }

    #[test]
    fn test_current_scope_assertions() {
        let mut ctx = Context::new();

        ctx.assert(TermId(1));
        ctx.push();
        ctx.assert(TermId(2));
        ctx.assert(TermId(3));

        let current = ctx.current_scope_assertions();
        assert_eq!(current, &[TermId(2), TermId(3)]);
    }

    #[test]
    fn test_named_context() {
        let mut ctx = NamedContext::new();

        ctx.assert_named(TermId(1), Some("a1".to_string()));
        ctx.assert(TermId(2));
        ctx.assert_named(TermId(3), Some("a3".to_string()));

        assert_eq!(ctx.num_assertions(), 3);

        let by_name = ctx.get_by_name("a1");
        assert_eq!(by_name, vec![TermId(1)]);

        let by_name = ctx.get_by_name("a3");
        assert_eq!(by_name, vec![TermId(3)]);
    }

    #[test]
    fn test_named_context_push_pop() {
        let mut ctx = NamedContext::new();

        ctx.assert_named(TermId(1), Some("base".to_string()));
        ctx.push();
        ctx.assert_named(TermId(2), Some("scope1".to_string()));
        ctx.push();
        ctx.assert_named(TermId(3), Some("scope2".to_string()));

        assert_eq!(ctx.num_assertions(), 3);

        ctx.pop();
        assert_eq!(ctx.num_assertions(), 2);
        assert!(ctx.get_by_name("scope2").is_empty());

        ctx.pop();
        assert_eq!(ctx.num_assertions(), 1);
        assert!(ctx.get_by_name("scope1").is_empty());
    }

    #[test]
    fn test_assert_many() {
        let mut ctx = Context::new();

        ctx.assert_many(vec![TermId(1), TermId(2), TermId(3)]);

        assert_eq!(ctx.num_assertions(), 3);
        assert_eq!(ctx.assertions(), &[TermId(1), TermId(2), TermId(3)]);
    }

    #[test]
    fn test_named_assertion_helpers() {
        let unnamed = NamedAssertion::unnamed(TermId(1));
        assert_eq!(unnamed.name, None);

        let named = NamedAssertion::named(TermId(2), "test");
        assert_eq!(named.name, Some("test".to_string()));
    }
}
