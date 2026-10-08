//! Regression tests for the duplicate-deletion guard in `oxiz_sat::Solver::pop`
//! (`oxiz-sat/src/solver/mod.rs`, inside the `assertion_clause_ids` removal
//! loop), added alongside the `#P2b-19` fix and until now verified by
//! inspection only.
//!
//! # What the guard is for
//!
//! `pop` deletes every clause id the retracted assertion level registered.
//! Since `Solver::learn_clause` started registering there (the `#P2b-19` fix —
//! without it a learned clause outlived the scope that entailed it, a false
//! proof), that list routinely names clauses **some other mechanism already
//! removed**: `reduce_clause_database` and the on-the-fly `check_subsumption`
//! in `learn_clause` both delete learned clauses mid-solve without pruning the
//! level's id list, and so does `forget_learned_since`.
//!
//! `ClauseDatabase::remove` and `Solver::purge_binary_edges` are idempotent on
//! an already-deleted clause, so the database itself is fine either way. The
//! *proof log* is not: `drat_delete` / `lrat_delete` are unconditional writes,
//! so a second visit emits a second deletion record for a clause the proof has
//! already retracted, and the log then disagrees with the database it claims to
//! describe. The guard skips any id whose clause is already gone.
//!
//! # How the first test discriminates
//!
//! A duplicate deletion is invisible to a proof *checker*:
//! `oxiz_proof::lrat_check::check_lrat_proof`'s deletion arm is
//! `active.remove(&id)`, and removing an absent key is a silent no-op. So the
//! load-bearing assertion here is a direct parse of the emitted LRAT text — every
//! id named on a `d` record, with no id allowed to appear twice — not the
//! checker's verdict.
//!
//! The scenario is built so the guard has something to skip: a satisfiable
//! planted random 3-SAT instance is asserted inside a pushed scope with
//! `clause_deletion_threshold` lowered to 50, which makes
//! `reduce_clause_database` run repeatedly during the solve and delete learned
//! clauses the open level has registered. The test asserts that precondition
//! (`deleted_clauses > 0` before the `pop`) rather than assuming it, so a future
//! change that stops exercising the guard fails loudly instead of passing
//! vacuously.
//!
//! Measured on the fixed tree (2026-09-14, `nvars = 90`, seed 7): 187 conflicts,
//! 187 learned clauses, 114 of them deleted by `reduce_clause_database` during
//! the solve, 554 deletion ids emitted overall, **0 duplicates**. With the
//! guard's `continue` deleted from `Solver::pop`, the same run emits 668 ids of
//! which **114 are duplicates** — exactly the clauses the reduction had already
//! retracted.
//!
//! # Why the emitted proof is not checker-verifiable (a separate, recorded gap)
//!
//! `check_lrat_proof` is run anyway, and it rejects. The cause is **not** in the
//! `pop` path, and it is recorded as `#P2b-23` in `TODO.md` rather than fixed
//! here. Two independent mechanisms, both read from the source:
//!
//! * `Solver::lrat_emit_empty_from` (`oxiz-sat/src/solver/lrat_trace.rs`) does
//!   `self.lrat.take()` the moment an `Unsat` is concluded, deliberately — so a
//!   trace whose *first* verdict is `Unsat` is closed before any later `pop`
//!   could record anything at all (the second test below pins exactly that).
//! * `LratWriter` hands out original and derived clause ids from one monotone
//!   `next_id` counter, while `check_lrat_proof` numbers the original formula
//!   `1..=n` in the order given. An incremental caller that adds original
//!   clauses *after* the search has started learning therefore gets ids the
//!   checker cannot reproduce, so its hint chains dangle.
//!
//! Between them, no script shape exists in which a `pop` emits deletion records
//! *and* the resulting trace concludes with a checkable empty clause. That is a
//! statement about incremental LRAT support, not about `pop`.
//!
//! Temp files use `std::env::temp_dir()`, are uniquely named per run and are
//! removed afterwards.

use oxiz_proof::lrat_check::check_lrat_proof;
use oxiz_sat::{Solver, SolverConfig, SolverResult};
use std::collections::HashSet;
use std::path::PathBuf;

/// A fresh, uniquely-named path under `std::env::temp_dir()` for one test's
/// LRAT output file.
fn unique_lrat_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "oxiz_sat_pop_deletion_{tag}_{}_{nanos}.lrat",
        std::process::id()
    ))
}

/// A 64-bit linear congruential generator, so the instances below are
/// reproducible byte for byte without a `rand` dev-dependency.
struct Lcg(u64);

impl Lcg {
    /// The next raw draw (high bits, which is where an LCG's randomness is).
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// A draw in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A random 3-SAT instance with a **planted** satisfying assignment: every
/// clause is repaired if the plant falsifies it, so the instance is satisfiable
/// by construction while still sitting near the ratio-4.3 hard region and
/// producing hundreds of conflicts.
///
/// Satisfiability is a property of the formula, so this precondition cannot rot
/// as the solver changes.
fn planted_3sat(seed: u64, vars: usize, clauses: usize) -> Vec<Vec<i32>> {
    let mut rng = Lcg(seed);
    let plant: Vec<bool> = (0..vars).map(|_| rng.below(2) == 1).collect();
    let literal = |v: usize, positive: bool| {
        let n = v as i32 + 1;
        if positive { n } else { -n }
    };
    let mut out = Vec::new();
    while out.len() < clauses {
        let vs = [
            rng.below(vars as u64) as usize,
            rng.below(vars as u64) as usize,
            rng.below(vars as u64) as usize,
        ];
        if vs[0] == vs[1] || vs[1] == vs[2] || vs[0] == vs[2] {
            continue;
        }
        let mut clause = Vec::with_capacity(3);
        let mut satisfied = false;
        for &v in &vs {
            let positive = rng.below(2) == 1;
            satisfied |= positive == plant[v];
            clause.push(literal(v, positive));
        }
        if !satisfied {
            // Repair the clause so the plant satisfies it.
            clause[0] = literal(vs[0], plant[vs[0]]);
        }
        out.push(clause);
    }
    out
}

/// The pigeonhole-principle instance: `pigeons` items into `holes` slots. UNSAT
/// whenever `pigeons > holes`, and it forces real conflict-driven learning.
/// Variables are numbered from `first_var` so it can share a solver with other
/// constraints.
fn pigeonhole(first_var: usize, pigeons: usize, holes: usize) -> Vec<Vec<i32>> {
    let var = |p: usize, h: usize| (first_var + p * holes + h + 1) as i32;
    let mut out = Vec::new();
    for p in 0..pigeons {
        out.push((0..holes).map(|h| var(p, h)).collect());
    }
    for h in 0..holes {
        for p1 in 0..pigeons {
            for p2 in (p1 + 1)..pigeons {
                out.push(vec![-var(p1, h), -var(p2, h)]);
            }
        }
    }
    out
}

/// Every clause id named on an LRAT deletion record, in emission order.
///
/// An LRAT deletion line is `<id> d <deleted ids…> 0`; an addition line is
/// `<id> <literals…> 0 <hints…> 0` and is skipped here.
fn deleted_ids(lrat_text: &str) -> Vec<i64> {
    let mut ids = Vec::new();
    for line in lrat_text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 2 || tokens[1] != "d" {
            continue;
        }
        for token in &tokens[2..] {
            match token.parse::<i64>() {
                Ok(0) => break,
                Ok(id) => ids.push(id),
                Err(_) => {}
            }
        }
    }
    ids
}

/// The first id `ids` names twice, if any.
fn first_duplicate(ids: &[i64]) -> Option<i64> {
    let mut seen = HashSet::new();
    ids.iter().copied().find(|id| !seen.insert(*id))
}

/// The load-bearing test: a `pop` that retracts a scope whose learned clauses
/// were partly deleted mid-solve emits **one** LRAT deletion record per clause,
/// never two.
///
/// See this file's header for the measured with/without-guard numbers and for
/// why the parse — not `check_lrat_proof` — is what detects the defect.
#[test]
fn a_pop_emits_no_second_lrat_deletion_for_an_already_deleted_clause() {
    const VARS: usize = 90;
    const BASE_CLAUSES: usize = 20;

    let path = unique_lrat_path("no_duplicate");
    let config = SolverConfig {
        // Low enough that `reduce_clause_database` runs several times inside the
        // pushed scope and deletes learned clauses the level has registered.
        clause_deletion_threshold: 50,
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    solver
        .enable_lrat_proof(&path)
        .expect("enable_lrat_proof must succeed before any add_clause");
    for _ in 0..VARS {
        solver.new_var();
    }

    let all = planted_3sat(7, VARS, (VARS as f64 * 4.3) as usize);
    let (base, scoped) = all.split_at(BASE_CLAUSES);
    for clause in base {
        solver.add_clause_dimacs(clause);
    }

    solver.push();
    for clause in scoped {
        solver.add_clause_dimacs(clause);
    }

    let under_push = solver.solve();
    assert_eq!(
        under_push,
        SolverResult::Sat,
        "the planted instance is satisfiable by construction, and the trace must \
         stay open past this verdict: an `Unsat` closes the LRAT writer outright"
    );
    let deleted_during_solve = solver.stats().deleted_clauses;
    assert!(
        solver.stats().conflicts > 0,
        "the scoped instance must actually make the solver learn"
    );
    // The precondition the guard exists for. Asserted, not assumed: without
    // clauses deleted *during* the solve the `pop` has no already-deleted id to
    // skip and this test would pass with the guard removed.
    assert!(
        deleted_during_solve > 0,
        "no clause was deleted mid-solve, so `pop` has nothing to skip and this \
         test no longer exercises the guard; raise the instance size or lower \
         `clause_deletion_threshold`"
    );
    assert!(
        solver.lrat_proof_enabled(),
        "the LRAT writer must still be open when the `pop` runs, or its deletion \
         records are never emitted and the assertion below is vacuous"
    );

    solver.pop();

    // A second verdict, so the trace is a complete script rather than a
    // half-finished one. `[1]` and `[-1]` contradict outright.
    solver.add_clause_dimacs(&[1]);
    solver.add_clause_dimacs(&[-1]);
    let after_pop = solver.solve();
    assert_eq!(
        after_pop,
        SolverResult::Unsat,
        "`(1) ∧ (¬1)` is unsatisfiable whatever else survived the pop"
    );

    solver.disable_lrat_proof(); // flushes buffered output to disk
    let lrat_text = std::fs::read_to_string(&path).expect("read emitted LRAT proof");
    let _ = std::fs::remove_file(&path);

    let ids = deleted_ids(&lrat_text);
    assert!(
        ids.len() as u64 > deleted_during_solve,
        "the `pop` itself must have emitted deletion records on top of the {} the \
         solve emitted; got {} ids in total",
        deleted_during_solve,
        ids.len()
    );
    assert!(
        ids.len() >= scoped.len(),
        "the `pop` retracts all {} scoped original clauses, so at least that many \
         deletion ids must appear; got {}",
        scoped.len(),
        ids.len()
    );
    assert_eq!(
        first_duplicate(&ids),
        None,
        "clause id deleted twice in the LRAT trace: `Solver::pop` re-deleted a \
         clause `reduce_clause_database` had already removed. {} ids emitted, {} \
         clauses deleted during the solve",
        ids.len(),
        deleted_during_solve
    );

    // Part (a) of the task, and a recorded finding rather than a fixed defect:
    // the trace cannot verify, for reasons outside the `pop` path (see this
    // file's header and `#P2b-23`). Pinned so that a future incremental-LRAT
    // fix makes this line fail and prompts the header to be rewritten.
    let originals: Vec<Vec<i32>> = all.clone();
    let report = check_lrat_proof(&originals, &lrat_text);
    assert!(
        !report.verified,
        "incremental LRAT is expected NOT to verify today (#P2b-23): the writer \
         closes at the first `Unsat` and original ids added after learning began \
         are not the `1..=n` the checker assumes. If this now verifies, the gap \
         is closed — update `#P2b-23` and this test"
    );
}

/// The script shape the task names literally — base clauses, `push`, an
/// unsatisfiable pigeonhole, `solve` (`Unsat`), `pop`, more clauses, `solve`
/// again — pinning both the verdict sequence and *why* it contributes no
/// deletion records: `lrat_emit_empty_from` closes the writer the instant the
/// first `Unsat` is concluded.
///
/// Without that mechanism spelled out in a test, the duplicate-deletion
/// assertion above could be written in this shape and pass for the wrong reason
/// (an empty proof trivially has no duplicates).
#[test]
fn an_unsat_under_push_closes_the_trace_before_the_pop_can_record_anything() {
    const PIGEONS: usize = 5;
    const HOLES: usize = 4;
    // Two base variables plus the pigeonhole grid.
    const BASE_VARS: usize = 2;

    let path = unique_lrat_path("unsat_under_push");
    let mut solver = Solver::new();
    solver
        .enable_lrat_proof(&path)
        .expect("enable_lrat_proof must succeed before any add_clause");
    for _ in 0..(BASE_VARS + PIGEONS * HOLES) {
        solver.new_var();
    }

    // Base: satisfiable on its own (`1 ∨ 2`).
    solver.add_clause_dimacs(&[1, 2]);

    solver.push();
    let scoped = pigeonhole(BASE_VARS, PIGEONS, HOLES);
    for clause in &scoped {
        solver.add_clause_dimacs(clause);
    }
    let under_push = solver.solve();
    assert_eq!(
        under_push,
        SolverResult::Unsat,
        "{PIGEONS} pigeons do not fit in {HOLES} holes"
    );
    assert!(
        solver.stats().conflicts > 0,
        "the pigeonhole instance must be refuted by conflict-driven learning, not \
         by a level-0 contradiction inside `add_clause`"
    );
    // The mechanism: concluding the proof closes the writer.
    assert!(
        !solver.lrat_proof_enabled(),
        "`lrat_emit_empty_from` takes and closes the writer when it concludes an \
         `Unsat` proof, so nothing this solver does afterwards is traced"
    );

    solver.pop();
    solver.add_clause_dimacs(&[-1]);
    let after_pop = solver.solve();
    assert_eq!(
        after_pop,
        SolverResult::Sat,
        "after the `pop` only `(1 ∨ 2) ∧ (¬1)` remains, which `2` satisfies — the \
         `#P2b-19` verdict-sequence property: retracting a scope must not leave \
         the goal over-constrained"
    );

    solver.disable_lrat_proof();
    let lrat_text = std::fs::read_to_string(&path).expect("read emitted LRAT proof");
    let _ = std::fs::remove_file(&path);

    // Whatever the closed trace does contain, no id may be retracted twice.
    let ids = deleted_ids(&lrat_text);
    assert_eq!(
        first_duplicate(&ids),
        None,
        "clause id deleted twice in the LRAT trace of the unsat-under-push shape"
    );

    // And the part of the trace that *was* written is a genuine, verifiable
    // refutation of `base ∧ scoped` — the originals are exactly the clauses
    // added before the first `solve`, in order, so the checker's `1..=n`
    // numbering matches here (unlike the incremental shape above).
    let mut originals = vec![vec![1, 2]];
    originals.extend(scoped.iter().cloned());
    let report = check_lrat_proof(&originals, &lrat_text);
    assert!(
        report.verified,
        "the first solve's LRAT refutation must verify. It can, unlike the \
         incremental shape in the other test, because every original clause here \
         was added *before* the first `solve` — so `LratWriter`'s monotone id \
         counter handed them exactly the `1..=n` that `check_lrat_proof` assumes, \
         and nothing was appended to the trace after it concluded. A failure here \
         is therefore about the hint chains themselves (an inprocessing pass that \
         stopped gating itself off under proof tracing, or a change in how \
         `lrat_build_hint_chain` walks antecedents), not about incremental LRAT \
         (#P2b-23). Checker said: {:?}",
        report.failure
    );
}
