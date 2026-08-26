//! Regression for issue #42: `solve_with_theory`'s inner theory-check loop used
//! to clone the *entire* trail (`self.trail.assignments().to_vec()`) on every
//! iteration and only then slice `[safe_start..]` — an O(trail_len^2) copy over
//! a search. The fix reads `self.trail.assignments()[safe_start..]` in place
//! (no clone at all), so each iteration's work is proportional to what's
//! actually new.
//!
//! This test pins the *delivery contract* the `theory_processed` clamps are
//! supposed to guarantee, independent of how the suffix is materialized:
//!
//!   (a) a literal is never delivered to `TheoryCallback::on_assignment` twice
//!       within the same "quiescent stretch" (the run of assignments between
//!       one `on_backtrack` and the next) — i.e. the clamp never re-sends a
//!       still-valid trail entry it already sent since the last unwind;
//!   (b) every literal on the final trail was delivered to the theory at some
//!       point (verified by comparing the theory's last-recorded polarity for
//!       each variable against the solved model).
//!
//! The mock theory never fabricates a conflict of its own (an invented
//! "conflict" clause that is not an actual consequence of the formula would
//! make the solver learn something unsound and the assertions below
//! meaningless) — it only ever answers `Sat`, purely observing whatever
//! genuine Boolean conflicts/backtracks the CDCL core produces on its own.
//! Two instances exercise both directions:
//!   - a small unsatisfiable pigeonhole instance (3 pigeons, 2 holes), which
//!     forces multiple real conflicts and backtracks before the search
//!     concludes `Unsat`, checked against (a);
//!   - a small satisfiable instance, checked against both (a) and (b).

use oxiz_sat::{Lit, Solver, SolverResult, TheoryCallback, TheoryCheckResult, Var};
use std::collections::HashMap;

/// A theory that never reports a conflict or propagation of its own — it only
/// records what it is told, so any conflict/backtrack activity observed comes
/// entirely from genuine Boolean CDCL, keeping the solve itself sound.
struct RecordingTheory {
    // var -> polarity (true = positive literal) delivered since the last
    // on_backtrack call (i.e. within the current quiescent stretch).
    delivered_since_backtrack: HashMap<Var, bool>,
    // var -> polarity last delivered, ever (used to check the final trail).
    last_polarity: HashMap<Var, bool>,
    violations: Vec<String>,
    backtracks_seen: u32,
}

impl RecordingTheory {
    fn new() -> Self {
        Self {
            delivered_since_backtrack: HashMap::new(),
            last_polarity: HashMap::new(),
            violations: Vec::new(),
            backtracks_seen: 0,
        }
    }
}

impl TheoryCallback for RecordingTheory {
    fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult {
        if self.delivered_since_backtrack.contains_key(&lit.var()) {
            self.violations.push(format!(
                "var {:?} delivered twice within one quiescent stretch \
                 (no intervening backtrack)",
                lit.var()
            ));
        }
        self.delivered_since_backtrack
            .insert(lit.var(), lit.is_pos());
        self.last_polarity.insert(lit.var(), lit.is_pos());
        TheoryCheckResult::Sat
    }

    fn final_check(&mut self) -> TheoryCheckResult {
        TheoryCheckResult::Sat
    }

    fn on_backtrack(&mut self, _level: u32) {
        self.backtracks_seen += 1;
        // A backtrack unwinds some suffix of the trail; the clamps in
        // `solve_with_theory` are keyed off a trail *index* (`boundary`), not
        // decision level, and chronological backtracking can re-append
        // literals that survive the rollback. The one contract this test
        // pins is per quiescent stretch (between backtracks), so start that
        // stretch's bookkeeping over rather than trying to predict exactly
        // which variables `boundary` kept valid.
        self.delivered_since_backtrack.clear();
    }
}

#[test]
fn theory_delivery_has_no_duplicates_within_a_quiescent_stretch_unsat() {
    // Pigeonhole: 3 pigeons, 2 holes -- unsatisfiable, and small enough that
    // CDCL still needs multiple conflicts/backtracks to prove it.
    // p[i][j] = pigeon i is in hole j, i in 0..3, j in 0..2.
    let mut solver = Solver::new();
    let p: Vec<Vec<Var>> = (0..3)
        .map(|_| vec![solver.new_var(), solver.new_var()])
        .collect();

    // Every pigeon is in some hole.
    for row in &p {
        solver.add_clause([Lit::pos(row[0]), Lit::pos(row[1])]);
    }
    // No hole holds two pigeons: every unordered pair of pigeon rows, for
    // both holes.
    for (i, row_i) in p.iter().enumerate() {
        for row_k in p.iter().skip(i + 1) {
            for (&hole_i, &hole_k) in row_i.iter().zip(row_k.iter()) {
                solver.add_clause([Lit::neg(hole_i), Lit::neg(hole_k)]);
            }
        }
    }

    let mut theory = RecordingTheory::new();
    let result = solver.solve_with_theory(&mut theory);

    assert_eq!(
        result,
        SolverResult::Unsat,
        "pigeonhole 3-into-2 is unsatisfiable"
    );
    assert!(
        theory.violations.is_empty(),
        "delivery-contract violations: {:?}",
        theory.violations
    );
    assert!(
        theory.backtracks_seen > 0,
        "pigeonhole should force at least one genuine CDCL backtrack"
    );
}

#[test]
fn theory_delivery_matches_final_model_sat() {
    let mut solver = Solver::new();
    let vars: Vec<Var> = (0..6).map(|_| solver.new_var()).collect();

    solver.add_clause([Lit::pos(vars[0]), Lit::pos(vars[1]), Lit::pos(vars[2])]);
    solver.add_clause([Lit::neg(vars[0]), Lit::pos(vars[3])]);
    solver.add_clause([Lit::pos(vars[3]), Lit::pos(vars[4]), Lit::pos(vars[5])]);
    solver.add_clause([Lit::neg(vars[4]), Lit::neg(vars[5])]);

    let mut theory = RecordingTheory::new();
    let result = solver.solve_with_theory(&mut theory);

    assert_eq!(result, SolverResult::Sat);
    assert!(
        theory.violations.is_empty(),
        "delivery-contract violations: {:?}",
        theory.violations
    );

    // Every variable's final model value must match the theory's last
    // recorded polarity for it -- i.e. the final trail was fully delivered
    // to the theory, not silently skipped by an over-eager clamp.
    for &v in &vars {
        let model_positive = solver.model_value(v).is_true();
        let delivered_positive = theory.last_polarity.get(&v).copied();
        assert_eq!(
            delivered_positive,
            Some(model_positive),
            "var {v:?}: theory's last-delivered polarity does not match the final model"
        );
    }
}
