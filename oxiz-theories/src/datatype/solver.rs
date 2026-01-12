//! Datatype Theory Solver for Algebraic Data Types.

use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// A field in a constructor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Field sort (as string for now)
    pub sort: String,
}

/// A selector function for extracting a field from a constructor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selector {
    /// Selector name
    pub name: String,
    /// Index of the field in the constructor
    pub field_index: usize,
    /// The constructor this selector applies to
    pub constructor: String,
}

/// A constructor for a datatype
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constructor {
    /// Constructor name
    pub name: String,
    /// Constructor fields
    pub fields: Vec<Field>,
    /// Constructor tag (unique within the datatype)
    pub tag: u32,
}

impl Constructor {
    /// Create a new constructor
    #[must_use]
    pub fn new(name: impl Into<String>, tag: u32) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            tag,
        }
    }

    /// Add a field to the constructor
    pub fn with_field(mut self, name: impl Into<String>, sort: impl Into<String>) -> Self {
        self.fields.push(Field {
            name: name.into(),
            sort: sort.into(),
        });
        self
    }

    /// Check if this is a nullary constructor (no fields)
    #[must_use]
    pub fn is_nullary(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get the number of fields
    #[must_use]
    pub fn arity(&self) -> usize {
        self.fields.len()
    }
}

/// A datatype sort declaration
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatatypeSort {
    /// Sort name
    pub name: String,
    /// Number of type parameters
    pub arity: u32,
}

/// A datatype declaration
#[derive(Debug, Clone)]
pub struct DatatypeDecl {
    /// The datatype sort
    pub sort: DatatypeSort,
    /// Constructors
    pub constructors: Vec<Constructor>,
    /// Is this datatype recursive?
    pub is_recursive: bool,
}

impl DatatypeDecl {
    /// Create a new datatype declaration
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            sort: DatatypeSort {
                name: name.into(),
                arity: 0,
            },
            constructors: Vec::new(),
            is_recursive: false,
        }
    }

    /// Create a parametric datatype
    #[must_use]
    pub fn parametric(name: impl Into<String>, arity: u32) -> Self {
        Self {
            sort: DatatypeSort {
                name: name.into(),
                arity,
            },
            constructors: Vec::new(),
            is_recursive: false,
        }
    }

    /// Add a constructor
    pub fn with_constructor(mut self, constructor: Constructor) -> Self {
        self.constructors.push(constructor);
        self
    }

    /// Mark as recursive
    pub fn recursive(mut self) -> Self {
        self.is_recursive = true;
        self
    }

    /// Get a constructor by name
    #[must_use]
    pub fn get_constructor(&self, name: &str) -> Option<&Constructor> {
        self.constructors.iter().find(|c| c.name == name)
    }

    /// Get a constructor by tag
    #[must_use]
    pub fn get_constructor_by_tag(&self, tag: u32) -> Option<&Constructor> {
        self.constructors.iter().find(|c| c.tag == tag)
    }

    /// Create selectors for all constructors
    #[must_use]
    pub fn selectors(&self) -> Vec<Selector> {
        let mut selectors = Vec::new();
        for cons in &self.constructors {
            for (idx, field) in cons.fields.iter().enumerate() {
                selectors.push(Selector {
                    name: field.name.clone(),
                    field_index: idx,
                    constructor: cons.name.clone(),
                });
            }
        }
        selectors
    }

    /// Create recognizer function names for all constructors
    #[must_use]
    pub fn recognizers(&self) -> Vec<String> {
        self.constructors
            .iter()
            .map(|c| format!("is-{}", c.name))
            .collect()
    }
}

/// A datatype value (term tagged with constructor info)
#[derive(Debug, Clone)]
struct DatatypeValue {
    /// The term
    #[allow(dead_code)]
    term: TermId,
    /// The datatype
    datatype: String,
    /// Known constructor (if determined)
    constructor: Option<u32>,
    /// Field values (if known)
    fields: SmallVec<[Option<TermId>; 4]>,
}

/// Equality or disequality constraint
#[derive(Debug, Clone)]
struct DtConstraint {
    /// Left-hand side term
    lhs: TermId,
    /// Right-hand side term
    rhs: TermId,
    /// True for equality, false for disequality
    is_eq: bool,
    /// Reason term
    reason: TermId,
}

/// Datatype Theory Solver
#[derive(Debug)]
pub struct DatatypeSolver {
    /// Registered datatypes
    datatypes: FxHashMap<String, DatatypeDecl>,
    /// Term to datatype value mapping
    term_info: FxHashMap<TermId, DatatypeValue>,
    /// Pending constraints
    constraints: Vec<DtConstraint>,
    /// Constructor applications: term -> (constructor_name, arguments)
    constructor_apps: FxHashMap<TermId, (String, Vec<TermId>)>,
    /// Selector applications: term -> (selector_name, argument)
    selector_apps: FxHashMap<TermId, (String, TermId)>,
    /// Recognizer applications: term -> (recognizer_name, argument)
    recognizer_apps: FxHashMap<TermId, (String, TermId)>,
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
}

/// State for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_constraints: usize,
}

impl Default for DatatypeSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DatatypeSolver {
    /// Create a new Datatype solver
    #[must_use]
    pub fn new() -> Self {
        Self {
            datatypes: FxHashMap::default(),
            term_info: FxHashMap::default(),
            constraints: Vec::new(),
            constructor_apps: FxHashMap::default(),
            selector_apps: FxHashMap::default(),
            recognizer_apps: FxHashMap::default(),
            context_stack: Vec::new(),
        }
    }

    /// Register a datatype
    pub fn register_datatype(&mut self, decl: DatatypeDecl) {
        self.datatypes.insert(decl.sort.name.clone(), decl);
    }

    /// Get a registered datatype
    #[must_use]
    pub fn get_datatype(&self, name: &str) -> Option<&DatatypeDecl> {
        self.datatypes.get(name)
    }

    /// Register a term as a datatype value
    pub fn register_term(&mut self, term: TermId, datatype: &str) {
        self.term_info.entry(term).or_insert_with(|| DatatypeValue {
            term,
            datatype: datatype.to_string(),
            constructor: None,
            fields: SmallVec::new(),
        });
    }

    /// Register a constructor application
    pub fn register_constructor(&mut self, result: TermId, constructor: &str, args: Vec<TermId>) {
        // Extract info from datatype without holding borrow
        let dt_info = self.find_datatype_for_constructor(constructor).map(|dt| {
            let sort_name = dt.sort.name.clone();
            let cons_tag = dt.get_constructor(constructor).map(|c| c.tag);
            (sort_name, cons_tag)
        });

        if let Some((sort_name, cons_tag)) = dt_info {
            self.register_term(result, &sort_name);
            if let Some(tag) = cons_tag
                && let Some(info) = self.term_info.get_mut(&result)
            {
                info.constructor = Some(tag);
                info.fields = args.iter().map(|&t| Some(t)).collect();
            }
        }
        self.constructor_apps
            .insert(result, (constructor.to_string(), args));
    }

    /// Register a selector application
    pub fn register_selector(&mut self, result: TermId, selector: &str, arg: TermId) {
        self.selector_apps
            .insert(result, (selector.to_string(), arg));
    }

    /// Register a recognizer application
    pub fn register_recognizer(&mut self, result: TermId, recognizer: &str, arg: TermId) {
        self.recognizer_apps
            .insert(result, (recognizer.to_string(), arg));
    }

    /// Find the datatype that defines a constructor
    fn find_datatype_for_constructor(&self, constructor: &str) -> Option<&DatatypeDecl> {
        self.datatypes
            .values()
            .find(|dt| dt.get_constructor(constructor).is_some())
    }

    /// Assert equality: a = b
    pub fn assert_eq(&mut self, a: TermId, b: TermId, reason: TermId) {
        self.constraints.push(DtConstraint {
            lhs: a,
            rhs: b,
            is_eq: true,
            reason,
        });
    }

    /// Assert disequality: a != b
    pub fn assert_neq(&mut self, a: TermId, b: TermId, reason: TermId) {
        self.constraints.push(DtConstraint {
            lhs: a,
            rhs: b,
            is_eq: false,
            reason,
        });
    }

    /// Assert that a term has a specific constructor
    pub fn assert_constructor(&mut self, term: TermId, constructor: &str) {
        // Extract constructor tag without holding borrow
        let cons_tag = self
            .find_datatype_for_constructor(constructor)
            .and_then(|dt| dt.get_constructor(constructor))
            .map(|c| c.tag);

        if let Some(tag) = cons_tag
            && let Some(info) = self.term_info.get_mut(&term)
        {
            info.constructor = Some(tag);
        }
    }

    /// Check for conflicts
    fn check_constraints(&self) -> Option<Vec<TermId>> {
        // Check distinctness: different constructors => different values
        for constraint in &self.constraints {
            if constraint.is_eq {
                // Check if lhs and rhs have different constructors
                let lhs_info = self.term_info.get(&constraint.lhs);
                let rhs_info = self.term_info.get(&constraint.rhs);

                if let (Some(lhs), Some(rhs)) = (lhs_info, rhs_info)
                    && lhs.datatype == rhs.datatype
                    && let (Some(lhs_cons), Some(rhs_cons)) = (lhs.constructor, rhs.constructor)
                    && lhs_cons != rhs_cons
                {
                    // Different constructors cannot be equal
                    return Some(vec![constraint.reason]);
                }
            } else {
                // Disequality constraint
                // Check if lhs and rhs are the same constructor application with same args
                if let (Some((lhs_cons, lhs_args)), Some((rhs_cons, rhs_args))) = (
                    self.constructor_apps.get(&constraint.lhs),
                    self.constructor_apps.get(&constraint.rhs),
                ) && lhs_cons == rhs_cons
                    && lhs_args == rhs_args
                {
                    // Same constructor with same arguments must be equal
                    return Some(vec![constraint.reason]);
                }
            }
        }

        // Check for injectivity conflicts
        // If cons(a1, ..., an) = cons(b1, ..., bn) then ai = bi for all i
        for constraint in &self.constraints {
            if constraint.is_eq
                && let (Some((lhs_cons, lhs_args)), Some((rhs_cons, rhs_args))) = (
                    self.constructor_apps.get(&constraint.lhs),
                    self.constructor_apps.get(&constraint.rhs),
                )
                && lhs_cons == rhs_cons
            {
                // Same constructor, check if any arguments are known to be different
                for (la, ra) in lhs_args.iter().zip(rhs_args.iter()) {
                    // Check if we have a disequality constraint between these
                    for neq in &self.constraints {
                        if !neq.is_eq
                            && ((neq.lhs == *la && neq.rhs == *ra)
                                || (neq.lhs == *ra && neq.rhs == *la))
                        {
                            // Arguments differ, but constructors are equal - conflict
                            return Some(vec![constraint.reason, neq.reason]);
                        }
                    }
                }
            }
        }

        None
    }

    /// Generate theory propagations
    fn propagate(&self) -> Vec<(TermId, Vec<TermId>)> {
        let mut propagations = Vec::new();

        // Selector axioms: if we know a term is a specific constructor,
        // we can propagate field values
        for (term, (cons_name, args)) in &self.constructor_apps {
            // For each selector that applies to this constructor,
            // the selector applied to this term equals the corresponding argument
            for (sel_result, (sel_name, sel_arg)) in &self.selector_apps {
                if sel_arg == term {
                    // Find which field this selector extracts
                    if let Some(dt) = self.find_datatype_for_constructor(cons_name)
                        && let Some(cons) = dt.get_constructor(cons_name)
                    {
                        for (idx, field) in cons.fields.iter().enumerate() {
                            if &field.name == sel_name
                                && let Some(&arg) = args.get(idx)
                            {
                                // Propagate: sel_result = args[idx]
                                propagations.push((*sel_result, vec![*term, arg]));
                            }
                        }
                    }
                }
            }
        }

        // Recognizer axioms
        for (term, (cons_name, _)) in &self.constructor_apps {
            for (rec_result, (rec_name, rec_arg)) in &self.recognizer_apps {
                if rec_arg == term {
                    let expected_cons = rec_name.strip_prefix("is-").unwrap_or(rec_name);
                    if expected_cons == cons_name {
                        // Recognizer should be true
                        propagations.push((*rec_result, vec![*term]));
                    }
                }
            }
        }

        propagations
    }
}

impl Theory for DatatypeSolver {
    fn id(&self) -> TheoryId {
        TheoryId::Datatype
    }

    fn name(&self) -> &str {
        "Datatype"
    }

    fn can_handle(&self, _term: TermId) -> bool {
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // Handle recognizer assertions
        if let Some((rec_name, arg)) = self.recognizer_apps.get(&term).cloned() {
            let cons_name = rec_name.strip_prefix("is-").unwrap_or(&rec_name);
            self.assert_constructor(arg, cons_name);
        }
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, _term: TermId) -> Result<TheoryResult> {
        // Handle negated recognizer assertions
        // This means the term is NOT of that constructor
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        if let Some(conflict) = self.check_constraints() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        let propagations = self.propagate();
        if !propagations.is_empty() {
            return Ok(TheoryResult::Propagate(propagations));
        }

        Ok(TheoryResult::Sat)
    }

    fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_constraints: self.constraints.len(),
        });
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            self.constraints.truncate(state.num_constraints);
        }
    }

    fn reset(&mut self) {
        self.datatypes.clear();
        self.term_info.clear();
        self.constraints.clear();
        self.constructor_apps.clear();
        self.selector_apps.clear();
        self.recognizer_apps.clear();
        self.context_stack.clear();
    }
}

// Standard datatype builders for common types

impl DatatypeDecl {
    /// Create a Unit type (single nullary constructor)
    #[must_use]
    pub fn unit() -> Self {
        Self::new("Unit").with_constructor(Constructor::new("unit", 0))
    }

    /// Create a Boolean type (two nullary constructors)
    #[must_use]
    pub fn boolean() -> Self {
        Self::new("Bool")
            .with_constructor(Constructor::new("true", 0))
            .with_constructor(Constructor::new("false", 1))
    }

    /// Create an Option type (None | Some(value))
    #[must_use]
    pub fn option(element_sort: &str) -> Self {
        Self::parametric("Option", 1)
            .with_constructor(Constructor::new("None", 0))
            .with_constructor(Constructor::new("Some", 1).with_field("value", element_sort))
    }

    /// Create a Pair type
    #[must_use]
    pub fn pair(first_sort: &str, second_sort: &str) -> Self {
        Self::parametric("Pair", 2).with_constructor(
            Constructor::new("mkpair", 0)
                .with_field("first", first_sort)
                .with_field("second", second_sort),
        )
    }

    /// Create a List type (nil | cons(head, tail))
    #[must_use]
    pub fn list(element_sort: &str) -> Self {
        Self::parametric("List", 1)
            .recursive()
            .with_constructor(Constructor::new("nil", 0))
            .with_constructor(
                Constructor::new("cons", 1)
                    .with_field("head", element_sort)
                    .with_field("tail", "List"),
            )
    }

    /// Create a Tree type (leaf(value) | node(left, right))
    #[must_use]
    pub fn tree(element_sort: &str) -> Self {
        Self::parametric("Tree", 1)
            .recursive()
            .with_constructor(Constructor::new("leaf", 0).with_field("value", element_sort))
            .with_constructor(
                Constructor::new("node", 1)
                    .with_field("left", "Tree")
                    .with_field("right", "Tree"),
            )
    }

    /// Create a simple enumeration
    #[must_use]
    pub fn enumeration(name: &str, values: &[&str]) -> Self {
        let mut decl = Self::new(name);
        for (tag, &value) in values.iter().enumerate() {
            decl = decl.with_constructor(Constructor::new(value, tag as u32));
        }
        decl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructor_builder() {
        let cons = Constructor::new("cons", 0)
            .with_field("head", "Int")
            .with_field("tail", "List");

        assert_eq!(cons.name, "cons");
        assert_eq!(cons.arity(), 2);
        assert!(!cons.is_nullary());
        assert_eq!(cons.fields[0].name, "head");
        assert_eq!(cons.fields[1].name, "tail");
    }

    #[test]
    fn test_datatype_list() {
        let list = DatatypeDecl::list("Int");

        assert_eq!(list.sort.name, "List");
        assert_eq!(list.sort.arity, 1);
        assert!(list.is_recursive);
        assert_eq!(list.constructors.len(), 2);

        assert!(list.get_constructor("nil").unwrap().is_nullary());
        assert_eq!(list.get_constructor("cons").unwrap().arity(), 2);
    }

    #[test]
    fn test_datatype_tree() {
        let tree = DatatypeDecl::tree("Int");

        assert_eq!(tree.sort.name, "Tree");
        assert!(tree.is_recursive);
        assert_eq!(tree.constructors.len(), 2);

        assert_eq!(tree.get_constructor("leaf").unwrap().arity(), 1);
        assert_eq!(tree.get_constructor("node").unwrap().arity(), 2);
    }

    #[test]
    fn test_enumeration() {
        let color = DatatypeDecl::enumeration("Color", &["Red", "Green", "Blue"]);

        assert_eq!(color.sort.name, "Color");
        assert_eq!(color.constructors.len(), 3);
        assert!(color.get_constructor("Red").unwrap().is_nullary());
        assert!(color.get_constructor("Green").unwrap().is_nullary());
        assert!(color.get_constructor("Blue").unwrap().is_nullary());
    }

    #[test]
    fn test_selectors() {
        let pair = DatatypeDecl::pair("Int", "Bool");
        let selectors = pair.selectors();

        assert_eq!(selectors.len(), 2);
        assert!(selectors.iter().any(|s| s.name == "first"));
        assert!(selectors.iter().any(|s| s.name == "second"));
    }

    #[test]
    fn test_recognizers() {
        let list = DatatypeDecl::list("Int");
        let recognizers = list.recognizers();

        assert_eq!(recognizers.len(), 2);
        assert!(recognizers.contains(&"is-nil".to_string()));
        assert!(recognizers.contains(&"is-cons".to_string()));
    }

    #[test]
    fn test_solver_register_datatype() {
        let mut solver = DatatypeSolver::new();
        let list = DatatypeDecl::list("Int");
        solver.register_datatype(list);

        assert!(solver.get_datatype("List").is_some());
    }

    #[test]
    fn test_solver_constructor_distinctness() {
        let mut solver = DatatypeSolver::new();
        solver.register_datatype(DatatypeDecl::list("Int"));

        let nil = TermId::new(1);
        let cons = TermId::new(2);
        let head = TermId::new(3);
        let tail = TermId::new(4);
        let reason = TermId::new(100);

        solver.register_constructor(nil, "nil", vec![]);
        solver.register_constructor(cons, "cons", vec![head, tail]);

        // Assert nil = cons (should cause conflict)
        solver.assert_eq(nil, cons, reason);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_same_constructor_no_conflict() {
        let mut solver = DatatypeSolver::new();
        solver.register_datatype(DatatypeDecl::list("Int"));

        let nil1 = TermId::new(1);
        let nil2 = TermId::new(2);
        let reason = TermId::new(100);

        solver.register_constructor(nil1, "nil", vec![]);
        solver.register_constructor(nil2, "nil", vec![]);

        // Assert nil1 = nil2 (same constructor, should be OK)
        solver.assert_eq(nil1, nil2, reason);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_injectivity_conflict() {
        let mut solver = DatatypeSolver::new();
        solver.register_datatype(DatatypeDecl::pair("Int", "Int"));

        let pair1 = TermId::new(1);
        let pair2 = TermId::new(2);
        let a = TermId::new(3);
        let b = TermId::new(4);
        let c = TermId::new(5);
        let reason_eq = TermId::new(100);
        let reason_neq = TermId::new(101);

        solver.register_constructor(pair1, "mkpair", vec![a, b]);
        solver.register_constructor(pair2, "mkpair", vec![c, b]);

        // Assert pair1 = pair2
        solver.assert_eq(pair1, pair2, reason_eq);

        // Assert a != c (conflict with injectivity)
        solver.assert_neq(a, c, reason_neq);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_push_pop() {
        let mut solver = DatatypeSolver::new();
        solver.register_datatype(DatatypeDecl::list("Int"));

        let nil = TermId::new(1);
        let cons = TermId::new(2);
        let head = TermId::new(3);
        let tail = TermId::new(4);
        let reason = TermId::new(100);

        solver.register_constructor(nil, "nil", vec![]);
        solver.register_constructor(cons, "cons", vec![head, tail]);

        solver.push();

        solver.assert_eq(nil, cons, reason);
        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));

        solver.pop();

        // After pop, should be SAT again
        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_option_datatype() {
        let option = DatatypeDecl::option("Int");

        assert_eq!(option.sort.name, "Option");
        assert_eq!(option.constructors.len(), 2);
        assert!(option.get_constructor("None").unwrap().is_nullary());
        assert_eq!(option.get_constructor("Some").unwrap().arity(), 1);
    }
}
