//! Comprehensive integration tests for Algebraic Datatype Theory (QF_DT)
//!
//! These tests verify the correctness of the complete Datatype implementation including:
//! - Simple enumerations (Color, Direction, etc.)
//! - Recursive datatypes (Lists, Trees)
//! - Constructor testers (is-nil, is-cons, is-leaf, is-node)
//! - Selectors (head, tail, left, right, value)
//! - Parameterized datatypes
//! - Multiple constructors with fields
//! - Pattern matching and equality
//! - Satisfiability checking and conflict detection

use oxiz_core::ast::TermId;
use oxiz_theories::datatype::{Constructor, DatatypeDecl, DatatypeSolver};
use oxiz_theories::{Theory, TheoryCheckResult as TheoryResult};

/// Helper to create unique term IDs for testing
fn fresh_term(id: u32) -> TermId {
    TermId::new(id)
}

#[test]
fn test_simple_enumeration_color() {
    let mut solver = DatatypeSolver::new();

    // Define Color enumeration: Red | Green | Blue
    let color = DatatypeDecl::enumeration("Color", &["Red", "Green", "Blue"]);
    solver.register_datatype(color);

    // Create color values
    let red = fresh_term(1);
    let green = fresh_term(2);
    let blue = fresh_term(3);

    solver.register_constructor(red, "Red", vec![]);
    solver.register_constructor(green, "Green", vec![]);
    solver.register_constructor(blue, "Blue", vec![]);

    // Check that the solver is consistent
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Test distinctness: Red != Green
    solver.assert_neq(red, green, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Test conflict: Red = Blue should conflict with Red != Blue
    let another_red = fresh_term(4);
    solver.register_constructor(another_red, "Red", vec![]);
    solver.assert_eq(red, blue, fresh_term(101));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_simple_enumeration_direction() {
    let mut solver = DatatypeSolver::new();

    // Define Direction enumeration: North | South | East | West
    let direction = DatatypeDecl::enumeration("Direction", &["North", "South", "East", "West"]);
    solver.register_datatype(direction);

    let north = fresh_term(1);
    let south = fresh_term(2);
    let east = fresh_term(3);
    let west = fresh_term(4);

    solver.register_constructor(north, "North", vec![]);
    solver.register_constructor(south, "South", vec![]);
    solver.register_constructor(east, "East", vec![]);
    solver.register_constructor(west, "West", vec![]);

    // All directions should be satisfiable
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Test that North != South
    solver.assert_eq(north, south, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_list_recursive_datatype() {
    let mut solver = DatatypeSolver::new();

    // Define List: nil | cons(head: Int, tail: List)
    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    let nil = fresh_term(1);
    let cons1 = fresh_term(2);
    let head1 = fresh_term(10);
    let tail1 = fresh_term(11);

    solver.register_constructor(nil, "nil", vec![]);
    solver.register_constructor(cons1, "cons", vec![head1, tail1]);

    // nil and cons should be distinct
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Test distinctness: nil != cons
    solver.assert_eq(nil, cons1, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_list_constructor_equality() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    // Create two cons cells with same head and tail
    let head = fresh_term(10);
    let tail = fresh_term(11);
    let cons1 = fresh_term(1);
    let cons2 = fresh_term(2);

    solver.register_constructor(cons1, "cons", vec![head, tail]);
    solver.register_constructor(cons2, "cons", vec![head, tail]);

    // cons(h, t) = cons(h, t) should be satisfiable
    solver.assert_eq(cons1, cons2, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

#[test]
fn test_list_injectivity() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    // Create two cons cells with different heads
    let head1 = fresh_term(10);
    let head2 = fresh_term(11);
    let tail = fresh_term(12);
    let cons1 = fresh_term(1);
    let cons2 = fresh_term(2);

    solver.register_constructor(cons1, "cons", vec![head1, tail]);
    solver.register_constructor(cons2, "cons", vec![head2, tail]);

    // Assert cons1 = cons2
    solver.assert_eq(cons1, cons2, fresh_term(100));

    // Assert head1 != head2 (should cause conflict by injectivity)
    solver.assert_neq(head1, head2, fresh_term(101));

    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_tree_recursive_datatype() {
    let mut solver = DatatypeSolver::new();

    // Define Tree: leaf(value: Int) | node(left: Tree, right: Tree)
    let tree = DatatypeDecl::tree("Int");
    solver.register_datatype(tree);

    let leaf1 = fresh_term(1);
    let value1 = fresh_term(10);
    let node1 = fresh_term(2);
    let left1 = fresh_term(20);
    let right1 = fresh_term(21);

    solver.register_constructor(leaf1, "leaf", vec![value1]);
    solver.register_constructor(node1, "node", vec![left1, right1]);

    // leaf and node should be distinct
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Test distinctness: leaf != node
    solver.assert_eq(leaf1, node1, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_tree_multiple_leaves() {
    let mut solver = DatatypeSolver::new();

    let tree = DatatypeDecl::tree("Int");
    solver.register_datatype(tree);

    // Create multiple leaves with different values
    let leaf1 = fresh_term(1);
    let leaf2 = fresh_term(2);
    let leaf3 = fresh_term(3);
    let val1 = fresh_term(10);
    let val2 = fresh_term(11);
    let val3 = fresh_term(12);

    solver.register_constructor(leaf1, "leaf", vec![val1]);
    solver.register_constructor(leaf2, "leaf", vec![val2]);
    solver.register_constructor(leaf3, "leaf", vec![val3]);

    // All leaves should coexist
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // leaf(v1) = leaf(v2) with v1 != v2 should conflict
    solver.assert_eq(leaf1, leaf2, fresh_term(100));
    solver.assert_neq(val1, val2, fresh_term(101));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_constructor_tester_is_nil() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    let nil = fresh_term(1);
    solver.register_constructor(nil, "nil", vec![]);

    // Register recognizer: (is-nil nil)
    let is_nil_result = fresh_term(10);
    solver.register_recognizer(is_nil_result, "is-nil", nil);

    // Assert that is-nil is true
    let result = solver.assert_true(is_nil_result);
    assert!(result.is_ok());

    let check_result = solver.check().expect("Check should succeed");
    assert!(matches!(
        check_result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));
}

#[test]
fn test_constructor_tester_is_cons() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    let cons = fresh_term(1);
    let head = fresh_term(10);
    let tail = fresh_term(11);
    solver.register_constructor(cons, "cons", vec![head, tail]);

    // Register recognizer: (is-cons cons)
    let is_cons_result = fresh_term(20);
    solver.register_recognizer(is_cons_result, "is-cons", cons);

    // Assert that is-cons is true
    let result = solver.assert_true(is_cons_result);
    assert!(result.is_ok());

    let check_result = solver.check().expect("Check should succeed");
    assert!(matches!(
        check_result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));
}

#[test]
fn test_selector_head() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    let cons = fresh_term(1);
    let head_val = fresh_term(10);
    let tail_val = fresh_term(11);
    solver.register_constructor(cons, "cons", vec![head_val, tail_val]);

    // Register selector: (head cons)
    let head_result = fresh_term(20);
    solver.register_selector(head_result, "head", cons);

    // Check should propagate: head_result = head_val
    let check_result = solver.check().expect("Check should succeed");
    assert!(matches!(
        check_result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));
}

#[test]
fn test_selector_tail() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    let cons = fresh_term(1);
    let head_val = fresh_term(10);
    let tail_val = fresh_term(11);
    solver.register_constructor(cons, "cons", vec![head_val, tail_val]);

    // Register selector: (tail cons)
    let tail_result = fresh_term(20);
    solver.register_selector(tail_result, "tail", cons);

    // Check should propagate: tail_result = tail_val
    let check_result = solver.check().expect("Check should succeed");
    assert!(matches!(
        check_result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));
}

#[test]
fn test_parameterized_option_type() {
    let mut solver = DatatypeSolver::new();

    // Define Option: None | Some(value: T)
    let option = DatatypeDecl::option("Int");
    solver.register_datatype(option);

    let none = fresh_term(1);
    let some = fresh_term(2);
    let value = fresh_term(10);

    solver.register_constructor(none, "None", vec![]);
    solver.register_constructor(some, "Some", vec![value]);

    // None and Some should be distinct
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    solver.assert_eq(none, some, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_parameterized_pair_type() {
    let mut solver = DatatypeSolver::new();

    // Define Pair: mkpair(first: T1, second: T2)
    let pair = DatatypeDecl::pair("Int", "Bool");
    solver.register_datatype(pair);

    let pair1 = fresh_term(1);
    let pair2 = fresh_term(2);
    let first1 = fresh_term(10);
    let second1 = fresh_term(11);
    let first2 = fresh_term(12);
    let second2 = fresh_term(13);

    solver.register_constructor(pair1, "mkpair", vec![first1, second1]);
    solver.register_constructor(pair2, "mkpair", vec![first2, second2]);

    // Two different pairs should be satisfiable
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Assert pair1 = pair2, first1 != first2 (conflict)
    solver.assert_eq(pair1, pair2, fresh_term(100));
    solver.assert_neq(first1, first2, fresh_term(101));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_complex_nested_structure() {
    let mut solver = DatatypeSolver::new();

    // List of lists
    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    // Create: cons(h1, cons(h2, nil))
    let nil = fresh_term(1);
    let inner_cons = fresh_term(2);
    let outer_cons = fresh_term(3);
    let h1 = fresh_term(10);
    let h2 = fresh_term(11);

    solver.register_constructor(nil, "nil", vec![]);
    solver.register_constructor(inner_cons, "cons", vec![h2, nil]);
    solver.register_constructor(outer_cons, "cons", vec![h1, inner_cons]);

    // This structure should be satisfiable
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Test selector on nested structure
    let tail_result = fresh_term(20);
    solver.register_selector(tail_result, "tail", outer_cons);

    // tail(outer_cons) should equal inner_cons
    let check_result = solver.check().expect("Check should succeed");
    assert!(matches!(
        check_result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));
}

#[test]
fn test_multiple_constructors_with_fields() {
    let mut solver = DatatypeSolver::new();

    // Define Shape: Circle(radius: Int) | Rectangle(width: Int, height: Int) | Triangle(base: Int, height: Int)
    let shape = DatatypeDecl::new("Shape")
        .with_constructor(Constructor::new("Circle", 0).with_field("radius", "Int"))
        .with_constructor(
            Constructor::new("Rectangle", 1)
                .with_field("width", "Int")
                .with_field("height", "Int"),
        )
        .with_constructor(
            Constructor::new("Triangle", 2)
                .with_field("base", "Int")
                .with_field("height", "Int"),
        );

    solver.register_datatype(shape);

    let circle = fresh_term(1);
    let rectangle = fresh_term(2);
    let triangle = fresh_term(3);

    solver.register_constructor(circle, "Circle", vec![fresh_term(10)]);
    solver.register_constructor(rectangle, "Rectangle", vec![fresh_term(11), fresh_term(12)]);
    solver.register_constructor(triangle, "Triangle", vec![fresh_term(13), fresh_term(14)]);

    // All shapes should coexist
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Circle != Rectangle
    solver.assert_eq(circle, rectangle, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_push_pop_with_assertions() {
    let mut solver = DatatypeSolver::new();

    let list = DatatypeDecl::list("Int");
    solver.register_datatype(list);

    let nil = fresh_term(1);
    let cons = fresh_term(2);
    solver.register_constructor(nil, "nil", vec![]);
    solver.register_constructor(cons, "cons", vec![fresh_term(10), fresh_term(11)]);

    // Initially satisfiable
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Push and add conflicting assertion
    solver.push();
    solver.assert_eq(nil, cons, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));

    // Pop should restore satisfiability
    solver.pop();
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

#[test]
fn test_pattern_matching_simulation() {
    let mut solver = DatatypeSolver::new();

    // Simulate pattern matching on Option type
    let option = DatatypeDecl::option("Int");
    solver.register_datatype(option);

    let opt = fresh_term(1);
    let value = fresh_term(10);

    // Case 1: opt is Some(value)
    solver.push();
    solver.register_constructor(opt, "Some", vec![value]);

    let is_some = fresh_term(20);
    solver.register_recognizer(is_some, "is-Some", opt);
    let _ = solver.assert_true(is_some);

    let result = solver.check().expect("Check should succeed");
    assert!(matches!(
        result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));

    solver.pop();

    // Case 2: opt is None
    solver.push();
    solver.register_constructor(opt, "None", vec![]);

    let is_none = fresh_term(21);
    solver.register_recognizer(is_none, "is-None", opt);
    let _ = solver.assert_true(is_none);

    let result = solver.check().expect("Check should succeed");
    assert!(matches!(
        result,
        TheoryResult::Sat | TheoryResult::Propagate(_)
    ));

    solver.pop();
}

#[test]
fn test_datatype_equality_transitivity() {
    let mut solver = DatatypeSolver::new();

    let color = DatatypeDecl::enumeration("Color", &["Red", "Green", "Blue"]);
    solver.register_datatype(color);

    let c1 = fresh_term(1);
    let c2 = fresh_term(2);
    let c3 = fresh_term(3);

    solver.register_constructor(c1, "Red", vec![]);
    solver.register_constructor(c2, "Red", vec![]);
    solver.register_constructor(c3, "Green", vec![]);

    // c1 = c2 (both Red)
    solver.assert_eq(c1, c2, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // c2 = c3 (Red = Green, should conflict)
    solver.assert_eq(c2, c3, fresh_term(101));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

#[test]
fn test_unit_datatype() {
    let mut solver = DatatypeSolver::new();

    // Unit type has only one constructor: unit
    let unit_type = DatatypeDecl::unit();
    solver.register_datatype(unit_type);

    let u1 = fresh_term(1);
    let u2 = fresh_term(2);

    solver.register_constructor(u1, "unit", vec![]);
    solver.register_constructor(u2, "unit", vec![]);

    // All unit values are equal
    solver.assert_eq(u1, u2, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

#[test]
fn test_boolean_datatype() {
    let mut solver = DatatypeSolver::new();

    // Boolean type: true | false
    let bool_type = DatatypeDecl::boolean();
    solver.register_datatype(bool_type);

    let t = fresh_term(1);
    let f = fresh_term(2);

    solver.register_constructor(t, "true", vec![]);
    solver.register_constructor(f, "false", vec![]);

    // true != false
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    solver.assert_eq(t, f, fresh_term(100));
    let result = solver.check().expect("Check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}
