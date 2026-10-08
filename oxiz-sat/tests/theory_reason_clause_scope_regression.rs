//! Regression test for `#P2b-21`: the clause `Solver::add_theory_reason_clause`
//! installs to justify a theory propagation was registered in **neither**
//! `learned_clause_ids` nor the open assertion level, so it survived both
//! `forget_learned_since` and `pop`.
//!
//! # The shape of the defect
//!
//! `oxiz-sat/src/solver/learn.rs`'s `add_theory_reason_clause` turns a theory
//! propagation `l₁ ∧ … ∧ lₙ ⊨ p` into the clause `p ∨ ¬l₁ ∨ … ∨ ¬lₙ` and puts
//! it in the database with `ClauseDatabase::add_learned`. It was the one live
//! clause-installing site in the crate that registered the result nowhere —
//! the last remaining instance of `#P2b-4`'s literal shape, left open when
//! `#P2b-19` fixed `learn_clause`.
//!
//! It was *argued* sound: such a clause is a theory tautology over formula
//! atoms, and a tautology holds at every assertion level, so retracting the
//! scope that happened to expose it changes nothing. The argument is only as
//! strong as the theory's own reasons, and it had never been tested. A theory
//! whose propagation depends on constraints that live in the pushed scope —
//! which is what a scoped theory solver *is* — hands back a reason that is not
//! a tautology of the base level, and the surviving clause then over-constrains
//! everything asserted after the `pop`: a **false proof**, the same failure mode
//! as `#P2b-19`.
//!
//! # The discriminating instance
//!
//! `ScopedImplicationTheory` below propagates `p` from reason `[q]` — i.e. the
//! clause `(p ∨ ¬q)` — but only while its `scope_open` flag is set, modelling a
//! theory constraint asserted inside the pushed scope. The script is:
//!
//! ```text
//! push
//!   (q)                      ← the scope's only clause
//!   solve_with_theory        → Sat, and the theory propagates p, installing (p ∨ ¬q)
//! pop                        ← (q) goes away; (p ∨ ¬q) must go with it
//! (¬p) (q)
//!   solve                    → Sat: p = false, q = true
//! ```
//!
//! With the reason clause retained, that last solve answers `Unsat` instead:
//! `(¬p) ∧ (q) ∧ (p ∨ ¬q)` has no model. The second solve deliberately uses the
//! plain Boolean `solve()`, with no theory attached at all, so the only thing
//! that can produce the wrong verdict is the stale clause sitting in the SAT
//! clause database.
//!
//! Measured: on the pre-fix tree this file's two tests fail with
//! `num_learned_clauses() == 1` after the `pop` (expected 0) and the verdict
//! `Unsat` (expected `Sat`). After the fix both pass.
//!
//! # The fix
//!
//! `add_theory_reason_clause` calls `register_learned_at_assertion_level`, the
//! same helper `learn_clause`'s three arms use since `#P2b-19` — the
//! conservative choice of the two the TODO entry lists, costing only precision
//! (a genuinely level-independent reason clause is re-derived by the next
//! propagation that needs it) and buying scope-correctness unconditionally.

use oxiz_sat::{Lit, Solver, SolverResult, TheoryCallback, TheoryCheckResult, Var};

/// `p`, the literal the theory propagates.
fn p_lit() -> Lit {
    Lit::pos(Var::new(0))
}

/// `q`, the literal the theory blames for it.
fn q_lit() -> Lit {
    Lit::pos(Var::new(1))
}

/// A theory that propagates `p` whenever it sees `q`, *while the scope that
/// justifies the implication is open*.
///
/// The `scope_open` flag is the whole point: a real theory solver's
/// propagations are consequences of the constraints currently asserted into it,
/// and those are scoped. Once the caller pops the scope, this theory no longer
/// believes `q → p` — but the SAT-level reason clause that recorded the belief
/// is not the theory's to retract.
struct ScopedImplicationTheory {
    /// Whether the scope holding the constraint `q → p` is still open.
    scope_open: bool,
    /// Whether `p` has already been propagated since the last backtrack.
    propagated: bool,
    /// How many propagations this theory has handed back in total.
    propagations: usize,
}

impl ScopedImplicationTheory {
    /// A theory with its scope open and nothing propagated yet.
    fn new() -> Self {
        Self {
            scope_open: true,
            propagated: false,
            propagations: 0,
        }
    }
}

impl TheoryCallback for ScopedImplicationTheory {
    fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult {
        if self.scope_open && !self.propagated && lit == q_lit() {
            self.propagated = true;
            self.propagations += 1;
            return TheoryCheckResult::Propagated(vec![(p_lit(), [q_lit()].into_iter().collect())]);
        }
        TheoryCheckResult::Sat
    }

    fn final_check(&mut self) -> TheoryCheckResult {
        TheoryCheckResult::Sat
    }

    fn on_backtrack(&mut self, _level: u32) {
        self.propagated = false;
    }
}

/// Build the solver, run the scoped solve that installs the theory reason
/// clause, and return it together with the theory (so the caller can assert the
/// propagation really happened) — stopping just before the `pop`.
fn solver_after_scoped_theory_propagation() -> (Solver, ScopedImplicationTheory) {
    let mut solver = Solver::new();
    solver.new_var(); // p
    solver.new_var(); // q

    let learned_before = solver.num_learned_clauses();
    assert_eq!(
        learned_before, 0,
        "a fresh solver holds no learned clauses; the counts below are differences \
         against this"
    );

    solver.push();
    // The scope's only clause: `q`.
    assert!(
        solver.add_clause_dimacs(&[2]),
        "`(q)` must be accepted inside the pushed scope"
    );

    let mut theory = ScopedImplicationTheory::new();
    let verdict = solver.solve_with_theory(&mut theory);
    assert_eq!(
        verdict,
        SolverResult::Sat,
        "`(q)` with `q → p` is satisfied by `p = q = true`"
    );
    assert!(
        theory.propagations > 0,
        "the theory must actually have propagated, or `add_theory_reason_clause` \
         was never reached and this test proves nothing"
    );
    assert_eq!(
        solver.num_learned_clauses(),
        1,
        "the single clause in the database's learned half must be the theory \
         reason clause `(p ∨ ¬q)`: this instance has no conflicts, so nothing \
         else can have been learned"
    );

    (solver, theory)
}

/// Structural half: the theory reason clause is gone from the clause database
/// after the `pop` that retracts the scope which produced it.
#[test]
fn a_theory_reason_clause_does_not_survive_the_pop_that_retracts_its_scope() {
    let (mut solver, _theory) = solver_after_scoped_theory_propagation();

    solver.pop();

    assert_eq!(
        solver.num_learned_clauses(),
        0,
        "`Solver::pop` must retract the theory reason clause `(p ∨ ¬q)` together \
         with the scope clause `(q)` that exposed it; it survived because \
         `add_theory_reason_clause` registered it in neither `learned_clause_ids` \
         nor `assertion_clause_ids` (#P2b-21)"
    );
}

/// Semantic half, and the one that would be a false proof in the field: with
/// the reason clause retained, a goal that is plainly satisfiable comes back
/// `Unsat`.
///
/// The second solve uses the plain Boolean [`Solver::solve`] with no theory
/// attached, so the stale clause in the SAT clause database is the only thing
/// that can refute `(¬p) ∧ (q)`.
#[test]
fn a_stale_theory_reason_clause_does_not_refute_a_satisfiable_goal() {
    // The theory is deliberately dropped here rather than carried into the
    // second solve: once the scope is popped it no longer believes `q → p`, and
    // leaving it out entirely is the strongest form of that — nothing but the
    // clause database can decide the verdict below.
    let (mut solver, _theory) = solver_after_scoped_theory_propagation();

    solver.pop();

    assert!(
        solver.add_clause_dimacs(&[-1]),
        "`(¬p)` must be accepted after the pop"
    );
    assert!(
        solver.add_clause_dimacs(&[2]),
        "`(q)` must be accepted after the pop"
    );

    let verdict = solver.solve();
    assert_eq!(
        verdict,
        SolverResult::Sat,
        "`(¬p) ∧ (q)` is satisfied by `p = false, q = true`; answering `Unsat` \
         means the theory reason clause `(p ∨ ¬q)` outlived the scope that \
         justified it (#P2b-21)"
    );
}

/// The control: the clause is registered at the level that was open when it was
/// derived, not deleted by *any* `pop`.
///
/// A nested scope opened and closed on top of it must leave it alone — which is
/// what distinguishes "registered at the open assertion level" from the cruder
/// alternative of discarding every reason clause on every `pop`, and what rules
/// out a fix implemented by simply never installing the clause (the helper
/// already pins that it *is* installed).
#[test]
fn a_theory_reason_clause_outlives_a_nested_scope_popped_above_it() {
    let (mut solver, _theory) = solver_after_scoped_theory_propagation();

    // A second scope, opened and closed above the one that produced the clause.
    solver.push();
    solver.pop();
    assert_eq!(
        solver.num_learned_clauses(),
        1,
        "popping a scope opened *after* the reason clause was derived must not \
         touch it: it belongs to the still-open level below"
    );

    // And the level it does belong to still retracts it.
    solver.pop();
    assert_eq!(
        solver.num_learned_clauses(),
        0,
        "popping the scope the clause was derived in retracts it"
    );
}
