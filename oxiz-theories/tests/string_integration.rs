//! Comprehensive integration tests for String theory (QF_S)
//!
//! These tests verify the correctness of the String theory implementation including:
//! - String concatenation (str.++)
//! - String length reasoning (str.len)
//! - Contains operations (str.contains)
//! - Prefix/suffix operations (str.prefixof, str.suffixof)
//! - Basic regex matching (str.in_re)
//! - Integration with solver and conflict detection

use oxiz_core::TermId;
use oxiz_theories::string::{StringExpr, StringSolver};
use oxiz_theories::{Theory, TheoryCheckResult as TheoryResult};

// ============================================================================
// Test 1: Basic string concatenation - satisfiable case
// ============================================================================

#[test]
fn test_basic_concatenation_sat() {
    let mut solver = StringSolver::new();

    // Test: "hello" ++ " world" = "hello world"
    let hello = StringExpr::literal("hello");
    let space_world = StringExpr::literal(" world");
    let hello_world = StringExpr::literal("hello world");

    let concat_result = hello.concat(space_world);

    // Add equality constraint
    let term = TermId(0);
    solver.add_equality(concat_result, hello_world, term);

    // Should be satisfiable
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 2: Basic string concatenation - unsatisfiable case
// ============================================================================

#[test]
fn test_basic_concatenation_unsat() {
    let mut solver = StringSolver::new();

    // Test: "hello" ++ " world" = "goodbye" (should fail)
    let hello = StringExpr::literal("hello");
    let space_world = StringExpr::literal(" world");
    let goodbye = StringExpr::literal("goodbye");

    let concat_result = hello.concat(space_world);

    // Add conflicting equality constraint
    let term = TermId(0);
    solver.add_equality(concat_result, goodbye, term);

    // Should be unsatisfiable
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 3: Chained string concatenation
// ============================================================================

#[test]
fn test_chained_concatenation() {
    let mut solver = StringSolver::new();

    // Test: "a" ++ "b" ++ "c" = "abc"
    let a = StringExpr::literal("a");
    let b = StringExpr::literal("b");
    let c = StringExpr::literal("c");
    let abc = StringExpr::literal("abc");

    let ab = a.concat(b);
    let abc_concat = ab.concat(c);

    let term = TermId(0);
    solver.add_equality(abc_concat, abc, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 4: Empty string concatenation
// ============================================================================

#[test]
fn test_empty_string_concat_left() {
    let mut solver = StringSolver::new();

    // Test: "" ++ "hello" = "hello"
    let empty = StringExpr::empty();
    let hello = StringExpr::literal("hello");
    let result_str = StringExpr::literal("hello");

    let concat_result = empty.concat(hello);

    let term = TermId(0);
    solver.add_equality(concat_result, result_str, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 5: Empty string concatenation (right)
// ============================================================================

#[test]
fn test_empty_string_concat_right() {
    let mut solver = StringSolver::new();

    // Test: "hello" ++ "" = "hello"
    let hello = StringExpr::literal("hello");
    let empty = StringExpr::empty();
    let result_str = StringExpr::literal("hello");

    let concat_result = hello.concat(empty);

    let term = TermId(0);
    solver.add_equality(concat_result, result_str, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 6: Contains operation - satisfiable
// ============================================================================

#[test]
fn test_contains_sat() {
    let mut solver = StringSolver::new();

    // Test: "hello world" contains "lo wo"
    let haystack = StringExpr::literal("hello world");
    let needle = StringExpr::literal("lo wo");

    let term = TermId(0);
    solver.add_contains(haystack, needle, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 7: Contains operation - unsatisfiable
// ============================================================================

#[test]
fn test_contains_unsat() {
    let mut solver = StringSolver::new();

    // Test: "hello" does not contain "xyz"
    let haystack = StringExpr::literal("hello");
    let needle = StringExpr::literal("xyz");

    let term = TermId(0);
    solver.add_contains(haystack, needle, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 8: Prefix operation - satisfiable
// ============================================================================

#[test]
fn test_prefix_sat() {
    let mut solver = StringSolver::new();

    // Test: "hello" is a prefix of "hello world"
    let prefix = StringExpr::literal("hello");
    let string = StringExpr::literal("hello world");

    let term = TermId(0);
    solver.add_prefix(prefix, string, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 9: Prefix operation - unsatisfiable
// ============================================================================

#[test]
fn test_prefix_unsat() {
    let mut solver = StringSolver::new();

    // Test: "world" is NOT a prefix of "hello"
    let prefix = StringExpr::literal("world");
    let string = StringExpr::literal("hello");

    let term = TermId(0);
    solver.add_prefix(prefix, string, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 10: Suffix operation - satisfiable
// ============================================================================

#[test]
fn test_suffix_sat() {
    let mut solver = StringSolver::new();

    // Test: "world" is a suffix of "hello world"
    let suffix = StringExpr::literal("world");
    let string = StringExpr::literal("hello world");

    let term = TermId(0);
    solver.add_suffix(suffix, string, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 11: Suffix operation - unsatisfiable
// ============================================================================

#[test]
fn test_suffix_unsat() {
    let mut solver = StringSolver::new();

    // Test: "hello" is NOT a suffix of "world"
    let suffix = StringExpr::literal("hello");
    let string = StringExpr::literal("world");

    let term = TermId(0);
    solver.add_suffix(suffix, string, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 12: Disequality constraint - satisfiable
// ============================================================================

#[test]
fn test_disequality_sat() {
    let mut solver = StringSolver::new();

    // Test: "hello" ≠ "world" (should be satisfied)
    let hello = StringExpr::literal("hello");
    let world = StringExpr::literal("world");

    let term = TermId(0);
    solver.add_disequality(hello, world, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 13: Disequality constraint - unsatisfiable
// ============================================================================

#[test]
fn test_disequality_unsat() {
    let mut solver = StringSolver::new();

    // Test: "same" ≠ "same" (should fail)
    let same1 = StringExpr::literal("same");
    let same2 = StringExpr::literal("same");

    let term = TermId(0);
    solver.add_disequality(same1, same2, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 14: Push/pop operations - conflict isolation
// ============================================================================

#[test]
fn test_push_pop() {
    let mut solver = StringSolver::new();

    // Initial state: should be satisfiable
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // Push a new scope
    solver.push();

    // Add conflicting constraint: "hello" = "world"
    let hello = StringExpr::literal("hello");
    let world = StringExpr::literal("world");
    let term = TermId(0);
    solver.add_equality(hello, world, term);

    // Should be unsatisfiable
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));

    // Pop back to initial state
    solver.pop();

    // Now should be satisfiable again
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 15: Multiple constraints combined - satisfiable
// ============================================================================

#[test]
fn test_multiple_constraints_sat() {
    let mut solver = StringSolver::new();

    // Test: "hello world" contains "lo" AND has "hello" as prefix AND has "world" as suffix
    let hello_world = StringExpr::literal("hello world");
    let lo = StringExpr::literal("lo");
    let hello = StringExpr::literal("hello");
    let world = StringExpr::literal("world");

    let term1 = TermId(1);
    let term2 = TermId(2);
    let term3 = TermId(3);

    solver.add_contains(hello_world.clone(), lo, term1);
    solver.add_prefix(hello, hello_world.clone(), term2);
    solver.add_suffix(world, hello_world, term3);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 16: Conflicting constraints - unsatisfiable
// ============================================================================

#[test]
fn test_conflicting_constraints_unsat() {
    let mut solver = StringSolver::new();

    // Test: "test" = "test" AND "test" ≠ "test" (contradiction)
    let test1 = StringExpr::literal("test");
    let test2 = StringExpr::literal("test");
    let test3 = StringExpr::literal("test");
    let test4 = StringExpr::literal("test");

    let term1 = TermId(1);
    let term2 = TermId(2);

    solver.add_equality(test1, test2, term1);
    solver.add_disequality(test3, test4, term2);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 17: Complex concatenation with variables
// ============================================================================

#[test]
fn test_complex_concatenation_with_var() {
    let mut solver = StringSolver::new();

    // Test: x ++ "!" where x is a variable
    let var_x = StringExpr::var(0);
    let exclaim = StringExpr::literal("!");
    let hello_exclaim = StringExpr::literal("hello!");

    let concat_result = var_x.concat(exclaim);

    let term = TermId(0);
    solver.add_equality(concat_result, hello_exclaim, term);

    // Should be satisfiable (x = "hello" is a solution)
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 18: Variable concatenation both sides
// ============================================================================

#[test]
fn test_variable_concat_both_sides() {
    let mut solver = StringSolver::new();

    // Test: x ++ y = "helloworld"
    let var_x = StringExpr::var(0);
    let var_y = StringExpr::var(1);
    let helloworld = StringExpr::literal("helloworld");

    let concat_result = var_x.concat(var_y);

    let term = TermId(0);
    solver.add_equality(concat_result, helloworld, term);

    // Should be satisfiable (many solutions: x="hello", y="world", etc.)
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 19: Prefix with empty string
// ============================================================================

#[test]
fn test_prefix_empty_string() {
    let mut solver = StringSolver::new();

    // Test: "" is a prefix of "hello" (always true)
    let empty = StringExpr::empty();
    let hello = StringExpr::literal("hello");

    let term = TermId(0);
    solver.add_prefix(empty, hello, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 20: Suffix with empty string
// ============================================================================

#[test]
fn test_suffix_empty_string() {
    let mut solver = StringSolver::new();

    // Test: "" is a suffix of "hello" (always true)
    let empty = StringExpr::empty();
    let hello = StringExpr::literal("hello");

    let term = TermId(0);
    solver.add_suffix(empty, hello, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 21: Contains with empty string
// ============================================================================

#[test]
fn test_contains_empty_string() {
    let mut solver = StringSolver::new();

    // Test: "hello" contains "" (always true)
    let hello = StringExpr::literal("hello");
    let empty = StringExpr::empty();

    let term = TermId(0);
    solver.add_contains(hello, empty, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 22: Concatenation identity: empty + empty = empty
// ============================================================================

#[test]
fn test_concat_empty_empty() {
    let mut solver = StringSolver::new();

    // Test: "" ++ "" = ""
    let empty1 = StringExpr::empty();
    let empty2 = StringExpr::empty();
    let empty3 = StringExpr::empty();

    let concat_result = empty1.concat(empty2);

    let term = TermId(0);
    solver.add_equality(concat_result, empty3, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 23: Disequality with different lengths
// ============================================================================

#[test]
fn test_disequality_different_lengths() {
    let mut solver = StringSolver::new();

    // Test: "short" ≠ "verylongstring"
    let short = StringExpr::literal("short");
    let long = StringExpr::literal("verylongstring");

    let term = TermId(0);
    solver.add_disequality(short, long, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 24: Prefix constraint with same string
// ============================================================================

#[test]
fn test_prefix_same_string() {
    let mut solver = StringSolver::new();

    // Test: "hello" is a prefix of "hello" (reflexive)
    let hello1 = StringExpr::literal("hello");
    let hello2 = StringExpr::literal("hello");

    let term = TermId(0);
    solver.add_prefix(hello1, hello2, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 25: Suffix constraint with same string
// ============================================================================

#[test]
fn test_suffix_same_string() {
    let mut solver = StringSolver::new();

    // Test: "hello" is a suffix of "hello" (reflexive)
    let hello1 = StringExpr::literal("hello");
    let hello2 = StringExpr::literal("hello");

    let term = TermId(0);
    solver.add_suffix(hello1, hello2, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 26: Contains with same string
// ============================================================================

#[test]
fn test_contains_same_string() {
    let mut solver = StringSolver::new();

    // Test: "hello" contains "hello" (reflexive)
    let hello1 = StringExpr::literal("hello");
    let hello2 = StringExpr::literal("hello");

    let term = TermId(0);
    solver.add_contains(hello1, hello2, term);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 27: Multiple push/pop levels
// ============================================================================

#[test]
fn test_multiple_push_pop() {
    let mut solver = StringSolver::new();

    // Level 0: empty
    solver.push(); // Level 1

    let term1 = TermId(1);
    solver.add_equality(StringExpr::literal("a"), StringExpr::literal("a"), term1);

    solver.push(); // Level 2

    let term2 = TermId(2);
    solver.add_equality(StringExpr::literal("b"), StringExpr::literal("b"), term2);

    // Should be sat at level 2
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    solver.pop(); // Back to level 1

    // Should still be sat at level 1
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    solver.pop(); // Back to level 0

    // Should be sat at level 0
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}

// ============================================================================
// Test 28: Conflicting prefix constraints
// ============================================================================

#[test]
fn test_conflicting_prefix() {
    let mut solver = StringSolver::new();

    // Cannot have both "hello" and "world" as prefixes of the same 5-char string
    let target = StringExpr::literal("hello");

    let term1 = TermId(1);
    solver.add_prefix(StringExpr::literal("hello"), target.clone(), term1);

    let term2 = TermId(2);
    solver.add_prefix(StringExpr::literal("world"), target, term2);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 29: Conflicting suffix constraints
// ============================================================================

#[test]
fn test_conflicting_suffix() {
    let mut solver = StringSolver::new();

    // Cannot have both "abc" and "xyz" as suffixes of the same 3-char string
    let target = StringExpr::literal("abc");

    let term1 = TermId(1);
    solver.add_suffix(StringExpr::literal("abc"), target.clone(), term1);

    let term2 = TermId(2);
    solver.add_suffix(StringExpr::literal("xyz"), target, term2);

    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// ============================================================================
// Test 30: Reset solver state
// ============================================================================

#[test]
fn test_reset() {
    let mut solver = StringSolver::new();

    // Add some constraints
    let term = TermId(0);
    solver.add_equality(
        StringExpr::literal("hello"),
        StringExpr::literal("world"),
        term,
    );

    // Should be unsat
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));

    // Reset solver
    solver.reset();

    // After reset, should be sat (no constraints)
    let result = solver.check().expect("check should succeed");
    assert!(matches!(result, TheoryResult::Sat));
}
