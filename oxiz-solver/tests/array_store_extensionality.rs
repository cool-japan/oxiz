//! Regression tests for the *store-only* array false-`sat` family.
//!
//! A formula can be an array problem without ever reading an array. The
//! store-commutativity shape
//!
//! ```text
//! (not (= (store (store a 1 x) 2 y) (store (store a 2 y) 1 x)))
//! ```
//!
//! is unsatisfiable — writing two distinct indices in either order yields the
//! same array — but deciding it needs extensionality, which manufactures the
//! read the formula itself never performs.
//!
//! `Solver::has_array_ops` is the switch that lets the lazy array-axiom
//! refinement loop in `check_core` run at all, and it was set from exactly two
//! places: the encoder's `Select`/`Store` *Boolean* arm (unreachable for an
//! array-sorted operand) and `track_theory_vars`' `Select` arm. `Store` was in
//! that walk's explicit no-op list. So a formula that stores but never selects
//! left the flag `false`, `instantiate_array_axioms` was never called, the
//! array disequality stayed a free Boolean for the SAT core, and `unsat` came
//! back `sat` in milliseconds.
//!
//! The fix gives `track_theory_vars` a `Store` arm that sets the flag (descent
//! is unchanged — a store's operands are not arithmetic variables of the
//! constraint being registered). Extensionality then supplies a witness index
//! `k` for the disequal pair and read-over-write reduces both chains at `k`.
//!
//! Every formula below is written **in-house**: minimal shapes reproduced from
//! the diagnosis of the QF_AUFLIA `storecomm` family, never a copy of an
//! upstream benchmark file.
//!
//! Each unsatisfiable case is paired with a satisfiable control that differs
//! by one index or one value. The controls matter as much as the `unsat`s:
//! turning the refinement loop on for store-only inputs must not cost a
//! genuine `sat`.

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .expect("script should parse and run")
}

// ---------------------------------------------------------------------
// Store-only formulas: no `select` anywhere in the input.
// ---------------------------------------------------------------------

/// The minimal repro: two writes at distinct constant indices, swapped.
///
/// Before the fix this answered `sat` with no search at all.
#[test]
fn store_commutativity_two_indices_is_unsat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (not (= (store (store a 1 x) 2 y)
                         (store (store a 2 y) 1 x))))
         (check-sat)");
    assert_eq!(out, vec!["unsat"], "store commutativity at 1 != 2");
}

/// Control: the same shape with the *values* swapped instead of preserved is
/// genuinely satisfiable (`x != y` makes the two arrays differ at index 1).
/// This is what fails if the new flag makes the refinement loop over-eager
/// and starts reporting `unsat`/`unknown` where a model exists.
#[test]
fn store_commutativity_with_swapped_values_is_sat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (not (= x y)))
         (assert (not (= (store (store a 1 x) 2 y)
                         (store (store a 1 y) 2 x))))
         (check-sat)");
    assert_eq!(out, vec!["sat"], "the two chains really do differ at 1");
}

/// Writes at *symbolic* indices known to be distinct. The disequality has to
/// travel through arithmetic before the read-over-write case split can close,
/// so this exercises the array and arithmetic solvers together rather than
/// constant-index matching alone.
#[test]
fn store_commutativity_symbolic_distinct_indices_is_unsat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun i () Int)
         (declare-fun j () Int)
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (not (= i j)))
         (assert (not (= (store (store a i x) j y)
                         (store (store a j y) i x))))
         (check-sat)");
    assert_eq!(out, vec!["unsat"], "distinct symbolic indices commute");
}

/// Control for the symbolic case: drop the `i != j` premise and a model
/// exists (take `i = j`, `x != y`; then the two chains disagree at that
/// index).
#[test]
fn store_commutativity_without_distinctness_is_sat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun i () Int)
         (declare-fun j () Int)
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (not (= (store (store a i x) j y)
                         (store (store a j y) i x))))
         (check-sat)");
    assert_eq!(out, vec!["sat"], "i = j with x != y is a model");
}

/// The benchmark family's actual shape: each intermediate array is a named
/// constant tied to a store by an asserted equality, so the store chain is
/// reachable only through those aliases. This is the form the QF_AUFLIA
/// `storecomm_*_np_*` files use, and it exercises the alias-aware
/// read-over-write path in `array_axioms::build_read_over_write`.
#[test]
fn aliased_store_chain_commutativity_is_unsat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun b1 () (Array Int Int))
         (declare-fun b2 () (Array Int Int))
         (declare-fun c1 () (Array Int Int))
         (declare-fun c2 () (Array Int Int))
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (= b1 (store a 1 x)))
         (assert (= b2 (store b1 2 y)))
         (assert (= c1 (store a 2 y)))
         (assert (= c2 (store c1 1 x)))
         (assert (not (= b2 c2)))
         (check-sat)");
    assert_eq!(out, vec!["unsat"], "aliased store chains commute");
}

/// Control for the aliased shape: one chain writes `y` where the other writes
/// `x` at index 2, so with `x != y` the arrays genuinely differ.
#[test]
fn aliased_store_chain_with_different_value_is_sat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun b1 () (Array Int Int))
         (declare-fun b2 () (Array Int Int))
         (declare-fun c1 () (Array Int Int))
         (declare-fun c2 () (Array Int Int))
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (not (= x y)))
         (assert (= b1 (store a 1 x)))
         (assert (= b2 (store b1 2 y)))
         (assert (= c1 (store a 2 x)))
         (assert (= c2 (store c1 1 x)))
         (assert (not (= b2 c2)))
         (check-sat)");
    assert_eq!(out, vec!["sat"], "the chains differ at index 2");
}

/// Idempotent write: storing the same value twice at the same index is the
/// same array as storing it once. No arithmetic case split is needed, only
/// extensionality plus a single read-over-write hit.
#[test]
fn duplicate_store_at_same_index_is_unsat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun x () Int)
         (assert (not (= (store (store a 1 x) 1 x)
                         (store a 1 x))))
         (check-sat)");
    assert_eq!(out, vec!["unsat"], "a duplicate write changes nothing");
}

/// A later write at the same index overwrites the earlier one.
#[test]
fn shadowed_store_is_unsat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (not (= (store (store a 1 x) 1 y)
                         (store a 1 y))))
         (check-sat)");
    assert_eq!(out, vec!["unsat"], "the second write shadows the first");
}

// ---------------------------------------------------------------------
// The flag itself: a store must switch array reasoning on.
// ---------------------------------------------------------------------

/// A store-only problem that is *satisfiable* still has to come back `sat`
/// promptly — enabling the refinement loop for stores must not turn every
/// store-only input into a search. Two writes at indices the input leaves
/// free, with no disequality to refute.
#[test]
fn store_only_satisfiable_problem_stays_sat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun b () (Array Int Int))
         (declare-fun i () Int)
         (declare-fun x () Int)
         (assert (= b (store a i x)))
         (assert (>= i 0))
         (check-sat)");
    assert_eq!(out, vec!["sat"], "nothing here is contradictory");
}

/// Mixed store/select: the reading side already worked before the fix (the
/// `Select` arm set the flag), so this pins that the new `Store` arm did not
/// disturb it.
#[test]
fn read_over_write_through_a_store_chain_is_unsat() {
    let out = run("(set-logic QF_AUFLIA)
         (declare-fun a () (Array Int Int))
         (declare-fun x () Int)
         (declare-fun y () Int)
         (assert (= (select (store (store a 1 x) 2 y) 1) 5))
         (assert (not (= x 5)))
         (check-sat)");
    assert_eq!(out, vec!["unsat"], "reading index 1 must yield x");
}
