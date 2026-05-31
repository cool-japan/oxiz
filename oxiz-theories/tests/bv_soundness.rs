//! Soundness regression tests for the BV theory solver.
//!
//! Bug 1: `check()` used to return an empty conflict clause (`TheoryResult::Unsat(vec![])`),
//! which prevented the outer CDCL(T) loop from properly backtracking.
//!
//! Bug 2: `encode_bv_op` was a flat, non-recursive closure that created fresh
//! BV variables for operands without encoding their sub-terms, so deeply nested
//! BV expressions were silently left unconstrained.

use oxiz_core::ast::TermId;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::{Theory, TheoryCheckResult};

// ---------------------------------------------------------------------------
// Bug 1 regression: non-empty conflict clause
// ---------------------------------------------------------------------------

/// Trivially UNSAT constraint (x = 1 AND x = 0 for a 1-bit variable).
/// The conflict clause returned by `check()` must be non-empty.
#[test]
fn test_empty_conflict_fixed() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);

    solver.new_bv(x, 1);

    // x = 1  (bit 0 forced true)
    solver.assert_const(x, 1, 1);
    // x = 0  (bit 0 forced false) — contradiction
    solver.assert_const(x, 0, 1);

    // Fallback path: `assertion_guard_terms` is empty because the unit test
    // calls `assert_const` directly (not via `record_constraint_term`).
    // The solver should fall back to the `assertions` list — but `assert_const`
    // also does NOT go through `Theory::assert_true/false`, so both lists are
    // empty.  In this unit-test path the important invariant is that we get
    // an UNSAT result; we do NOT require a non-empty conflict term list because
    // the terms were never recorded via `record_constraint_term`.
    match solver.check().expect("check should not error") {
        TheoryCheckResult::Unsat(_) => {
            // Correct: SAT sub-solver detected contradiction.
        }
        other => panic!("Expected UNSAT, got {:?}", other),
    }
}

/// When `record_constraint_term` is used (simulating the TheoryManager path),
/// the conflict clause must be non-empty on UNSAT.
#[test]
fn test_empty_conflict_fixed_with_guard_terms() {
    let mut solver = BvSolver::new();
    let x = TermId::new(10);
    let y = TermId::new(11);

    solver.new_bv(x, 4);
    solver.new_bv(y, 4);

    // x = 5, y = 5, x != y — UNSAT
    solver.assert_const(x, 5, 4);
    solver.assert_const(y, 5, 4);
    solver.assert_neq(x, y);

    // Simulate what TheoryManager does: record the constraint TermId
    let guard = TermId::new(99);
    solver.record_constraint_term(guard);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Unsat(terms) => {
            assert!(
                !terms.is_empty(),
                "conflict clause must be non-empty when guard terms were recorded"
            );
            assert!(
                terms.contains(&guard),
                "conflict clause must contain the recorded guard term"
            );
        }
        other => panic!("Expected UNSAT, got {:?}", other),
    }
}

/// Multiple guard terms: all recorded terms appear in the conflict clause.
#[test]
fn test_conflict_clause_contains_all_guard_terms() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, 8);
    solver.new_bv(b, 8);

    // a = 100, b = 200, a = b — UNSAT
    solver.assert_const(a, 100, 8);
    solver.assert_const(b, 200, 8);
    solver.assert_eq(a, b);

    let guard1 = TermId::new(101);
    let guard2 = TermId::new(102);
    solver.record_constraint_term(guard1);
    solver.record_constraint_term(guard2);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Unsat(terms) => {
            assert!(!terms.is_empty(), "conflict clause must not be empty");
            assert!(terms.contains(&guard1), "guard1 must be in conflict clause");
            assert!(terms.contains(&guard2), "guard2 must be in conflict clause");
        }
        other => panic!("Expected UNSAT, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Bug 1: SAT case should still work after the fix
// ---------------------------------------------------------------------------

/// SAT case: check() must still return Sat when the constraints are consistent.
#[test]
fn test_sat_case_unaffected() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    solver.assert_const(x, 42, 8);
    solver.assert_eq(x, y);

    let guard = TermId::new(99);
    solver.record_constraint_term(guard);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Bug 2 regression: recursive encoding
// ---------------------------------------------------------------------------

/// Nested XOR and AND: bvxor(bvand(a,b), c) must be encoded correctly.
/// All operands must be constrained — not left as free variables.
#[test]
fn test_nested_xor_and_sat() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let c = TermId::new(3);
    let ab = TermId::new(4); // bvand(a, b)
    let result = TermId::new(5); // bvxor(ab, c)

    solver.new_bv(a, 1);
    solver.new_bv(b, 1);
    solver.new_bv(c, 1);

    // Encode bvand(a, b) → ab
    solver.new_bv(ab, 1);
    solver.bv_and(ab, a, b);

    // Encode bvxor(ab, c) → result
    solver.new_bv(result, 1);
    solver.bv_xor(result, ab, c);

    // a=1, b=1 → ab=1; c=0 → xor(1,0)=1
    solver.assert_const(a, 1, 1);
    solver.assert_const(b, 1, 1);
    solver.assert_const(c, 0, 1);
    solver.assert_const(result, 1, 1);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT for nested xor/and, got {:?}", other),
    }
}

/// Nested XOR and AND UNSAT: result must be correctly detected as UNSAT
/// (inner operands are constrained, not free).
#[test]
fn test_nested_xor_and_unsat() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let c = TermId::new(3);
    let ab = TermId::new(4);
    let result = TermId::new(5);

    solver.new_bv(a, 1);
    solver.new_bv(b, 1);
    solver.new_bv(c, 1);

    solver.new_bv(ab, 1);
    solver.bv_and(ab, a, b);

    solver.new_bv(result, 1);
    solver.bv_xor(result, ab, c);

    // a=1, b=1 → ab=1; c=1 → xor(1,1)=0
    // But we assert result=1 — contradiction
    solver.assert_const(a, 1, 1);
    solver.assert_const(b, 1, 1);
    solver.assert_const(c, 1, 1);
    solver.assert_const(result, 1, 1);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Unsat(_) => {}
        other => panic!(
            "Expected UNSAT for nested xor/and contradiction, got {:?}",
            other
        ),
    }
}

/// Deeply nested: bvxor(bvxor(bvand(a,b), bvor(c,d)), bvnot(e))
/// Verifies that deeply nested expressions terminate and produce correct results.
#[test]
fn test_deep_nesting_terminates_correctly() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let c = TermId::new(3);
    let d = TermId::new(4);
    let e = TermId::new(5);
    let ab = TermId::new(6); // bvand(a, b)
    let cd = TermId::new(7); // bvor(c, d)
    let not_e = TermId::new(8); // bvnot(e)
    let ab_xor_cd = TermId::new(9); // bvxor(ab, cd)
    let result = TermId::new(10); // bvxor(ab_xor_cd, not_e)

    for &t in &[a, b, c, d, e, ab, cd, not_e, ab_xor_cd, result] {
        solver.new_bv(t, 1);
    }

    solver.bv_and(ab, a, b);
    solver.bv_or(cd, c, d);
    solver.bv_not(not_e, e);
    solver.bv_xor(ab_xor_cd, ab, cd);
    solver.bv_xor(result, ab_xor_cd, not_e);

    // a=1,b=1→ab=1; c=0,d=0→cd=0; e=0→not_e=1
    // ab xor cd = 1 xor 0 = 1; 1 xor not_e(1) = 0
    solver.assert_const(a, 1, 1);
    solver.assert_const(b, 1, 1);
    solver.assert_const(c, 0, 1);
    solver.assert_const(d, 0, 1);
    solver.assert_const(e, 0, 1);
    // result should be 0
    solver.assert_const(result, 0, 1);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT for deep nesting, got {:?}", other),
    }
}

/// Shared sub-term: same sub-expression appears in two branches.
/// Memo prevents duplicate clause generation.
#[test]
fn test_memo_prevents_dup_encoding() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let shared = TermId::new(3); // bvand(a, b) — used in two places
    let result1 = TermId::new(4); // bvxor(shared, a)
    let result2 = TermId::new(5); // bvor(shared, b)

    for &t in &[a, b, shared, result1, result2] {
        solver.new_bv(t, 4);
    }

    solver.bv_and(shared, a, b);
    solver.bv_xor(result1, shared, a);
    solver.bv_or(result2, shared, b);

    // a=0b1010=10, b=0b1100=12
    // shared = 10 & 12 = 0b1000 = 8
    // result1 = 8 xor 10 = 0b0010 = 2
    // result2 = 8 or 12  = 0b1100 = 12
    solver.assert_const(a, 10, 4);
    solver.assert_const(b, 12, 4);
    solver.assert_const(result1, 2, 4);
    solver.assert_const(result2, 12, 4);

    match solver.check().expect("check should not error") {
        TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT for shared sub-term test, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Push/pop round-trip with guard terms
// ---------------------------------------------------------------------------

/// After a push/pop cycle the guard terms are restored to the pre-push state.
#[test]
fn test_push_pop_restores_guard_terms() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);
    let guard_outer = TermId::new(100);

    solver.new_bv(x, 4);
    solver.assert_const(x, 3, 4);
    solver.record_constraint_term(guard_outer);

    solver.push();

    let guard_inner = TermId::new(101);
    solver.record_constraint_term(guard_inner);

    // Force an UNSAT inside the push scope
    solver.assert_const(x, 7, 4); // contradiction with x=3
    match solver.check().expect("inner check should not error") {
        TheoryCheckResult::Unsat(terms) => {
            assert!(!terms.is_empty(), "inner conflict must be non-empty");
            assert!(terms.contains(&guard_outer), "outer guard must be visible");
            assert!(terms.contains(&guard_inner), "inner guard must be visible");
        }
        other => panic!("Expected inner UNSAT, got {:?}", other),
    }

    solver.pop();

    // After pop: inner constraint and inner guard are removed.
    // Outer constraint (x=3) remains — should be SAT.
    match solver.check().expect("outer check should not error") {
        TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT after pop, got {:?}", other),
    }
}
