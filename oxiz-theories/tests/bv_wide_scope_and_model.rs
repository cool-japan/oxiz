//! Unit-level regressions for the wide-bit-vector and scope-rollback fixes in
//! `oxiz-theories`'s BV solver (cargo-formal findings U-Z11 and U-Z10).
//!
//! These exercise `BvSolver` directly, below the SMT-LIB script layer that
//! `oxiz-solver/tests/bv_wide_soundness.rs` and
//! `oxiz-solver/tests/bv_scope_rollback.rs` drive, so a future regression is
//! localised to this crate rather than only showing up as a wrong verdict.

use num_bigint::BigUint;
use oxiz_core::ast::TermId;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::{Theory, TheoryCheckResult};

/// `push` / `pop` must retract a circuit node created inside the scope.
///
/// This is U-Z10 at its smallest: `x` is bit-blasted and pinned to `1` *inside*
/// a pushed scope, so `sat.pop()` deletes both its defining and its pinning
/// clauses. Before the fix `term_to_bv` kept the entry, so `get_bv(x)` still
/// answered `Some` after the pop and the encoder's idempotence guard
/// (`oxiz-solver`'s `theory_bv_encode::encode_bv_term_recursive`, `if
/// bv.get_bv(tid).is_some() { continue; }`) then refused to rebuild it — leaving
/// a completely unconstrained bit-vector that made unsatisfiable formulas
/// answer `sat`. After the fix the term is unknown again and gets re-encoded.
#[test]
fn pop_retracts_a_circuit_node_created_inside_the_scope() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);

    solver.push();
    solver.new_bv(x, 8);
    assert!(solver.assert_const(x, 1, 8));
    assert!(
        solver.get_bv(x).is_some(),
        "the circuit exists inside the scope"
    );
    solver.pop();

    assert!(
        solver.get_bv(x).is_none(),
        "0.3.3/0.3.4 kept this cache entry after pop, leaving an unconstrained \
         bit-vector the encoder would never rebuild (U-Z10)"
    );
}

/// A node created *below* the pushed scope must survive the pop: the rollback
/// must retract exactly what `sat.pop()` deleted, no more.
#[test]
fn pop_keeps_a_circuit_node_created_below_the_scope() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.push();
    solver.new_bv(y, 8);
    solver.pop();

    assert!(solver.get_bv(x).is_some(), "x predates the push");
    assert!(solver.get_bv(y).is_none(), "y was created inside the scope");
}

/// Nested scopes unwind one level at a time, and `reset` clears everything.
#[test]
fn nested_pops_unwind_one_level_at_a_time() {
    let mut solver = BvSolver::new();
    let (a, b, c) = (TermId::new(1), TermId::new(2), TermId::new(3));

    solver.new_bv(a, 8);
    solver.push();
    solver.new_bv(b, 8);
    solver.push();
    solver.new_bv(c, 8);

    solver.pop();
    assert!(solver.get_bv(c).is_none());
    assert!(solver.get_bv(b).is_some());

    solver.pop();
    assert!(solver.get_bv(b).is_none());
    assert!(solver.get_bv(a).is_some());

    solver.reset();
    assert!(solver.get_bv(a).is_none(), "reset clears the caches");
}

/// The comparison memo is rolled back too, not just `term_to_bv`.
///
/// `assert_ult` memoises its comparison variable in `ult_cache` and encodes
/// `a <u b` into it once. Before the fix that entry survived a pop whose
/// `sat.pop()` had deleted the variable's defining clauses, so the *next*
/// `assert_ult` on the same pair hit the memo, found a now-**free** variable and
/// asserted it — which is trivially satisfiable whatever `a` and `b` hold. Here
/// `a` and `b` are pinned to 5 and 3 at the base level, so `a <u b` is false and
/// the second assertion must be refuted.
///
/// Measured: OxiZ 0.3.3/0.3.4 answered **`Sat`**; after the fix, `Unsat`.
#[test]
fn pop_retracts_the_ult_memo_so_a_later_conflict_is_still_found() {
    let mut solver = BvSolver::new();
    let (a, b) = (TermId::new(1), TermId::new(2));
    solver.new_bv(a, 8);
    solver.new_bv(b, 8);
    // Pinned at the base level, so the pop below cannot retract them.
    assert!(solver.assert_const(a, 5, 8));
    assert!(solver.assert_const(b, 3, 8));

    solver.push();
    // Creates the `ult_cache` entry for (a, b) and encodes it *inside* the
    // scope, so `sat.pop()` deletes exactly those defining clauses.
    assert!(solver.assert_ult(a, b));
    solver.pop();

    assert!(solver.assert_ult(a, b));
    assert!(
        matches!(solver.check(), Ok(TheoryCheckResult::Unsat(_))),
        "5 <u 3 is false; a stale ult_cache entry made the re-asserted \
         comparison a free variable, so 0.3.3/0.3.4 answered Sat (U-Z10)"
    );
}

/// `bvsub` of equal wide operands is zero — the U-Z11 shift bug at unit level.
///
/// `bv_sub` encodes `a - b` as `a + (~b + 1)` and the `+1` went through
/// `encode_add_const`, whose `((constant >> i) & 1)` read bit `i >= 64` of the
/// `u64` constant as bit `i % 64`. So `1` was blasted as `1 + 2^64 + …` and the
/// difference of two equal 128-bit values came out non-zero: 0.3.3/0.3.4
/// answered `Unsat` to this satisfiable constraint set (and panicked with
/// "attempt to shift right with overflow" in debug builds).
#[test]
fn wide_sub_of_equal_operands_is_zero_at_128_bits() {
    for width in [65u32, 96, 128] {
        let mut solver = BvSolver::new();
        let (a, b, d, zero) = (
            TermId::new(1),
            TermId::new(2),
            TermId::new(3),
            TermId::new(4),
        );
        solver.new_bv(a, width);
        solver.new_bv(b, width);
        assert!(solver.bv_sub(d, a, b), "encode a - b at {width}");
        assert!(solver.assert_eq(a, b));
        assert!(solver.assert_const(zero, 0, width));
        assert!(solver.assert_eq(d, zero));
        assert!(
            matches!(solver.check(), Ok(TheoryCheckResult::Sat)),
            "a = b implies a - b = 0 at width {width}; 0.3.3/0.3.4 answered Unsat"
        );
    }
}

/// Control for the row above: the negation of the same identity must still be
/// refuted, so the fix did not simply make the circuit weaker.
///
/// This one was **already correct** on 0.3.3/0.3.4 — the corrupted constant
/// perturbs the difference by `2^64`, which does not make `a - b = 1` reachable
/// when `a = b`, so the shapes that contradict in the low 64 bits were refuted
/// before the fix too (`p2-oxiz-034/report.md` §6.2). It is here to pin that
/// direction, not as a regression witness.
#[test]
fn wide_sub_of_equal_operands_is_not_one_at_128_bits() {
    for width in [65u32, 96, 128] {
        let mut solver = BvSolver::new();
        let (a, b, d, one) = (
            TermId::new(1),
            TermId::new(2),
            TermId::new(3),
            TermId::new(4),
        );
        solver.new_bv(a, width);
        solver.new_bv(b, width);
        assert!(solver.bv_sub(d, a, b));
        assert!(solver.assert_eq(a, b));
        assert!(solver.assert_const(one, 1, width));
        assert!(solver.assert_eq(d, one));
        assert!(
            matches!(solver.check(), Ok(TheoryCheckResult::Unsat(_))),
            "a = b makes a - b = 1 impossible at width {width}"
        );
    }
}

/// `Theory::get_model` must not merge two wide terms whose 64-bit folds collide.
///
/// Its grouping key used to be `(u64, u32)`, built with `value |= 1u64 << i`
/// over **every** bit of the term. In debug builds that panicked for `i >= 64`
/// ("attempt to shift left with overflow"); in release builds Rust masks the
/// shift amount, so bit `i` was ORed into bit `i % 64` and two *different* wide
/// values could fold onto one key. Equal keys are exactly what makes two terms
/// share a representative here, so the collision reports unequal terms as
/// having the same value — the hazard `extract_model_equalities` already
/// documents and avoids with a `BigUint` key. This function now uses the same
/// key.
///
/// `1` and `2^64` are the minimal witness: bit 64 folds onto bit 0, so both
/// hashed to `1`. Measured on 0.3.3/0.3.4: release builds **merged** them;
/// debug builds panicked.
#[test]
fn get_model_does_not_merge_wide_terms_whose_64_bit_folds_collide() {
    let mut solver = BvSolver::new();
    let (a, b) = (TermId::new(1), TermId::new(2));
    solver.new_bv(a, 128);
    solver.new_bv(b, 128);
    assert!(solver.assert_const(a, 1, 128));
    let two_pow_64 = BigUint::from(1u8) << 64u32;
    assert!(solver.assert_const_big(b, &two_pow_64, 128));
    assert!(matches!(solver.check(), Ok(TheoryCheckResult::Sat)));

    // Every entry is `(term, representative-of-its-value-group)`. `a` and `b`
    // hold different values, so neither may appear as the other's
    // representative.
    for (term, representative) in solver.get_model() {
        if term == a {
            assert_ne!(
                representative, b,
                "a = 1 and b = 2^64 differ; the model must not merge them"
            );
        }
        if term == b {
            assert_ne!(
                representative, a,
                "a = 1 and b = 2^64 differ; the model must not merge them"
            );
        }
    }
}
