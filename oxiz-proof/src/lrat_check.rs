//! Pure-Rust LRAT (Linear RAT) proof checker.
//!
//! An LRAT proof augments a plain DRAT proof with, for every added clause, an
//! explicit **hint chain**: the ids of clauses that — when unit-propagated in
//! the given order after the new clause's literals are assumed false — reach
//! a conflict. Because the hints are given rather than searched for, checking
//! an LRAT proof only ever needs *forward* unit propagation guided by the
//! hint list, never the backward RAT search a plain DRAT checker performs.
//! That is what makes an LRAT checker small enough to trust: this module is
//! the whole thing, no external tool required.
//!
//! # Format
//!
//! Text LRAT, one record per line (matches what `oxiz-sat`'s `LratWriter`
//! emits — not linked here, as this crate does not unconditionally depend on
//! `oxiz-sat` — and what the reference `lrat-check` tool reads):
//!
//! - Addition: `<id> <lit>… 0 <hint>… 0` — `id` is the clause's own id,
//!   `<lit>…` its literals (DIMACS convention, terminated by a literal `0`),
//!   `<hint>…` the RUP chain (clause ids, also terminated by `0`; may be
//!   empty, in which case only the mandatory terminating `0` appears).
//! - Deletion: `<id> d <clause-id>… 0` — the leading `id` is cosmetic (the id
//!   of the most-recently-added clause, per the LRAT convention); what
//!   matters is the `d` marker and the list of now-inactive clause ids.
//!
//! Original (input-formula) clauses are not part of the LRAT stream itself —
//! they are supplied separately, numbered `1..=N` in the order given, exactly
//! as a checker reading an accompanying DIMACS CNF file would number the
//! lines of that file.
//!
//! # Checking algorithm
//!
//! For each addition line, in order:
//!
//! 1. Assume every literal of the new clause is **false** (the negation of
//!    the clause being added).
//! 2. Replay the hint chain: each hint's clause must, under the assumption
//!    built so far, be either already satisfied (a literal already true —
//!    skipped as a harmless no-op) or have at most one literal not yet
//!    assigned. An unassigned literal is a propagation: it is assigned true
//!    and the walk continues. A hint clause with every literal false is the
//!    conflict the chain was building toward — checking that addition line
//!    succeeds immediately (any hints left unprocessed after that point are
//!    simply unused, which is allowed).
//! 3. If the hint chain runs out without ever reaching a fully-false clause,
//!    or a hint clause has more than one unassigned literal, or a hint
//!    references an id that is not currently active (never added, or since
//!    deleted), the line fails to verify and the whole proof is rejected.
//!
//! A clause whose own literals contain both a variable and its negation
//! (a tautology) trivially verifies: assuming its negation is self-
//! contradictory before any hint is even consulted.
//!
//! Deletion lines remove their listed ids from the active set; a later hint
//! referencing a deleted id fails to verify, matching real LRAT semantics
//! (`lrat-check` rejects the same).
//!
//! The proof as a whole verifies UNSAT once *any* clause with zero literals
//! is active — either an original clause supplied directly as `[]`, or one
//! derived by a verified addition line.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

/// Outcome of checking one LRAT proof against its original clause set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LratCheckReport {
    /// `true` iff the proof verified: every addition line's hint chain
    /// checked out and the stream produced (or was given) an empty clause.
    pub verified: bool,
    /// Number of addition lines whose RUP check succeeded before the report
    /// was finalized (a rejected proof's count stops at the failing line).
    pub additions_checked: usize,
    /// Number of deletion lines processed.
    pub deletions_applied: usize,
    /// `None` when `verified` is `true`; otherwise a human-readable reason a
    /// reviewer or test assertion can match against.
    pub failure: Option<String>,
}

impl LratCheckReport {
    fn reject(
        reason: impl Into<String>,
        additions_checked: usize,
        deletions_applied: usize,
    ) -> Self {
        Self {
            verified: false,
            additions_checked,
            deletions_applied,
            failure: Some(reason.into()),
        }
    }

    fn accept(additions_checked: usize, deletions_applied: usize) -> Self {
        Self {
            verified: true,
            additions_checked,
            deletions_applied,
            failure: None,
        }
    }
}

/// Errors from the file-based entry point ([`check_lrat_files`]) that occur
/// before any proof checking can even begin.
#[derive(Debug)]
pub enum LratCheckIoError {
    /// Reading the CNF or LRAT file failed.
    Io(io::Error),
    /// The CNF text was not well-formed DIMACS: a clause token was neither a
    /// literal nor the `0` terminator. Surfaced as an error rather than
    /// silently dropped — see `parse_dimacs_clauses` (a private function, not
    /// part of this crate's public API) for why silently skipping a bad
    /// token is unsafe for a checker (it would under-count the clause's
    /// literals, potentially turning a real original clause into one the
    /// checker treats as easier to satisfy than it actually is).
    MalformedCnf(String),
}

impl fmt::Display for LratCheckIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error reading proof input: {e}"),
            Self::MalformedCnf(reason) => write!(f, "malformed CNF input: {reason}"),
        }
    }
}

impl std::error::Error for LratCheckIoError {}

impl From<io::Error> for LratCheckIoError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Truth value of a literal under a partial assignment, keyed by variable.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LitStatus {
    True,
    False,
    Unassigned,
}

/// A partial assignment over DIMACS variables (positive `i32`s), recording
/// each assigned variable's truth value.
struct Assignment(HashMap<i32, bool>);

impl Assignment {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn status(&self, lit: i32) -> LitStatus {
        match self.0.get(&lit.abs()) {
            None => LitStatus::Unassigned,
            Some(&var_is_true) => {
                if (lit > 0) == var_is_true {
                    LitStatus::True
                } else {
                    LitStatus::False
                }
            }
        }
    }

    /// Assign `lit` true. Returns `false` if `lit`'s variable was already
    /// assigned to the *opposite* value (a genuine contradiction, distinct
    /// from re-affirming an already-consistent value).
    fn assign_true(&mut self, lit: i32) -> bool {
        let var = lit.abs();
        let want = lit > 0;
        match self.0.get(&var) {
            Some(&have) => have == want,
            None => {
                self.0.insert(var, want);
                true
            }
        }
    }
}

/// Verify one addition line's hint chain: does assuming `clause`'s negation
/// and replaying `hints` in order (against `active`) reach a clause every one
/// of whose literals is false?
fn rup_check(active: &HashMap<i64, Vec<i32>>, clause: &[i32], hints: &[i64]) -> Result<(), String> {
    let mut assigned = Assignment::new();

    // Assume the new clause's negation: every one of its own literals is
    // false. A clause that contains both `x` and `-x` makes this assumption
    // self-contradictory immediately — trivially RUP, no hints needed.
    for &lit in clause {
        if !assigned.assign_true(-lit) {
            return Ok(());
        }
    }

    for &hint_id in hints {
        let Some(hint_clause) = active.get(&hint_id) else {
            return Err(format!(
                "hint {hint_id} does not reference a currently active clause \
                 (never added, or deleted before this point)"
            ));
        };

        let mut satisfied = false;
        let mut unassigned_lit: Option<i32> = None;
        let mut has_multiple_unassigned = false;

        for &lit in hint_clause {
            match assigned.status(lit) {
                LitStatus::True => {
                    satisfied = true;
                    break;
                }
                LitStatus::False => {}
                LitStatus::Unassigned => {
                    if unassigned_lit.is_some() {
                        has_multiple_unassigned = true;
                    } else {
                        unassigned_lit = Some(lit);
                    }
                }
            }
        }

        if satisfied {
            // Already true under the current assignment: a harmless no-op
            // hint (this admits redundant hints a solver-side hint builder
            // may include without needing to prove minimality).
            continue;
        }

        match unassigned_lit {
            None => {
                // Every literal false: this hint clause is the conflict.
                // Any hints after this one are simply unused.
                return Ok(());
            }
            Some(lit) if !has_multiple_unassigned => {
                // Exactly one literal left undetermined: it must be true for
                // the clause to hold, so propagate it.
                let ok = assigned.assign_true(lit);
                debug_assert!(
                    ok,
                    "a literal reported Unassigned by `status` cannot already \
                     disagree with its own forced value"
                );
            }
            _ => {
                return Err(format!(
                    "hint {hint_id} is not unit under the assignment built so far \
                     (more than one literal still undetermined)"
                ));
            }
        }
    }

    Err("hint chain exhausted without deriving a conflict".to_string())
}

/// One parsed LRAT record.
enum LratLine {
    Addition {
        id: i64,
        lits: Vec<i32>,
        hints: Vec<i64>,
    },
    Deletion {
        ids: Vec<i64>,
    },
}

/// Parse one non-empty, non-comment LRAT text line.
fn parse_lrat_line(line: &str, line_no: usize) -> Result<LratLine, String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut idx = 0;

    let _id: i64 = tokens
        .first()
        .ok_or_else(|| format!("line {line_no}: empty record"))?
        .parse()
        .map_err(|_| format!("line {line_no}: leading id is not an integer"))?;
    idx += 1;

    if tokens.get(idx) == Some(&"d") {
        idx += 1;
        let mut ids = Vec::new();
        loop {
            let tok = tokens
                .get(idx)
                .ok_or_else(|| format!("line {line_no}: deletion line missing terminating 0"))?;
            idx += 1;
            let v: i64 = tok.parse().map_err(|_| {
                format!("line {line_no}: non-integer token {tok:?} in deletion ids")
            })?;
            if v == 0 {
                break;
            }
            ids.push(v);
        }
        return Ok(LratLine::Deletion { ids });
    }

    let mut lits = Vec::new();
    loop {
        let tok = tokens
            .get(idx)
            .ok_or_else(|| format!("line {line_no}: addition line missing literal terminator 0"))?;
        idx += 1;
        let v: i32 = tok
            .parse()
            .map_err(|_| format!("line {line_no}: non-integer token {tok:?} in literals"))?;
        if v == 0 {
            break;
        }
        lits.push(v);
    }

    let mut hints = Vec::new();
    loop {
        let tok = tokens
            .get(idx)
            .ok_or_else(|| format!("line {line_no}: addition line missing hint terminator 0"))?;
        idx += 1;
        let v: i64 = tok
            .parse()
            .map_err(|_| format!("line {line_no}: non-integer token {tok:?} in hints"))?;
        if v == 0 {
            break;
        }
        hints.push(v);
    }

    Ok(LratLine::Addition {
        id: {
            // Re-derive from the first token rather than reusing `_id` so a
            // malformed leading id (non-fatal above, since the leading id on
            // an addition line is itself the value we want) still surfaces
            // as the actual assigned id for the new clause.
            tokens[0]
                .parse()
                .map_err(|_| format!("line {line_no}: leading id is not an integer"))?
        },
        lits,
        hints,
    })
}

/// Check an LRAT text proof against its original clause set.
///
/// `original_clauses` are numbered `1..=original_clauses.len()` in the order
/// given (matching a DIMACS CNF file's clause-line order); `lrat_text` is the
/// proof body, one record per line, as described in the module documentation.
///
/// Never panics: every malformed-input path (bad token, out-of-range hint,
/// unterminated record, exhausted hint chain) is reported through
/// [`LratCheckReport::failure`] rather than a panic or an `unwrap`.
#[must_use]
pub fn check_lrat_proof(original_clauses: &[Vec<i32>], lrat_text: &str) -> LratCheckReport {
    let mut active: HashMap<i64, Vec<i32>> = HashMap::new();
    for (i, clause) in original_clauses.iter().enumerate() {
        let id = (i + 1) as i64;
        if clause.is_empty() {
            // An empty clause given directly in the input formula is, on its
            // own, a complete proof of UNSAT — no LRAT lines need to run.
            return LratCheckReport::accept(0, 0);
        }
        active.insert(id, clause.clone());
    }

    let mut additions_checked = 0usize;
    let mut deletions_applied = 0usize;

    for (i, raw_line) in lrat_text.lines().enumerate() {
        let line_no = i + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let parsed = match parse_lrat_line(line, line_no) {
            Ok(p) => p,
            Err(reason) => {
                return LratCheckReport::reject(reason, additions_checked, deletions_applied);
            }
        };

        match parsed {
            LratLine::Deletion { ids } => {
                for id in ids {
                    active.remove(&id);
                }
                deletions_applied += 1;
            }
            LratLine::Addition { id, lits, hints } => {
                if let Err(reason) = rup_check(&active, &lits, &hints) {
                    return LratCheckReport::reject(
                        format!(
                            "line {line_no}: addition of clause {id} failed to verify: {reason}"
                        ),
                        additions_checked,
                        deletions_applied,
                    );
                }
                let is_empty = lits.is_empty();
                active.insert(id, lits);
                additions_checked += 1;
                if is_empty {
                    return LratCheckReport::accept(additions_checked, deletions_applied);
                }
            }
        }
    }

    LratCheckReport::reject(
        "proof stream ended without ever deriving (or being given) the empty clause",
        additions_checked,
        deletions_applied,
    )
}

/// Parse a minimal DIMACS CNF file: skip `c` comment lines and the `p cnf …`
/// header, read every remaining clause (whitespace-separated integers,
/// terminated by a literal `0`) into a `Vec<i32>`.
///
/// Two failure-shaped inputs get careful handling, both because a checker
/// that reads originals from this output is only as trustworthy as this
/// parse:
///
/// - Some SATLIB-era generators end the clause section with a bare `%` line
///   before trailing junk (a repeated literal count, a filename, …). Reading
///   past it — as `oxiz-sat`'s own `dimacs.rs` reader does not — would parse
///   that junk as more clause tokens; a lone digit trailer like `0\n` reads
///   as a spurious **empty** original clause, and an empty original clause
///   makes *any* LRAT stream vacuously verify (the empty clause is already
///   present, no derivation needed) regardless of what the real formula
///   says. Stopping at `%`, mirroring `oxiz-sat/src/dimacs.rs`, closes that
///   hole.
/// - A token that is neither a valid `i32` literal nor absent used to be
///   silently skipped. Dropping a stray token out of a clause *shortens*
///   that clause, which only ever makes it easier to satisfy — silently
///   strengthening the formula the checker verifies against relative to what
///   the input file actually says. Every token must parse; one that does not
///   is now a hard error.
fn parse_dimacs_clauses(text: &str) -> Result<Vec<Vec<i32>>, String> {
    let mut clauses = Vec::new();
    let mut current: Vec<i32> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') || line.starts_with('p') {
            continue;
        }
        if line.starts_with('%') {
            break;
        }
        for tok in line.split_whitespace() {
            let v: i32 = tok
                .parse()
                .map_err(|_| format!("token {tok:?} is not a valid DIMACS literal"))?;
            if v == 0 {
                clauses.push(std::mem::take(&mut current));
            } else {
                current.push(v);
            }
        }
    }
    // A trailing clause with no terminating 0 on the final line is still
    // meaningful content; DIMACS requires the terminator, but there is no
    // reason to silently drop a well-intentioned final clause a strict
    // writer always terminates anyway.
    if !current.is_empty() {
        clauses.push(current);
    }
    Ok(clauses)
}

/// [`check_lrat_proof`], reading the original formula and the LRAT proof
/// from files (a minimal DIMACS CNF reader and a plain LRAT text reader —
/// this crate never shells out to an external tool).
pub fn check_lrat_files(
    cnf_path: impl AsRef<Path>,
    lrat_path: impl AsRef<Path>,
) -> Result<LratCheckReport, LratCheckIoError> {
    let cnf_text = fs::read_to_string(cnf_path)?;
    let lrat_text = fs::read_to_string(lrat_path)?;
    let clauses = parse_dimacs_clauses(&cnf_text).map_err(LratCheckIoError::MalformedCnf)?;
    Ok(check_lrat_proof(&clauses, &lrat_text))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (a ∨ b), (¬a ∨ b), (a ∨ ¬b), (¬a ∨ ¬b) is UNSAT: resolving the first
    /// two pairs on `a` gives `b` and `¬b`, resolving those gives the empty
    /// clause. Hand-built as a well-formed LRAT proof (clause ids 1..4 are
    /// the originals; 5 and 6 are the two intermediate resolvents; 7 is the
    /// empty clause).
    fn tiny_unsat_instance() -> (Vec<Vec<i32>>, &'static str) {
        let clauses = vec![vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]];
        let lrat = "5 2 0 1 2 0\n6 -2 0 3 4 0\n7 0 5 6 0\n";
        (clauses, lrat)
    }

    #[test]
    fn test_pr26_proof_lrat_check_accepts_valid_proof() {
        let (clauses, lrat) = tiny_unsat_instance();
        let report = check_lrat_proof(&clauses, lrat);
        assert!(report.verified, "failure: {:?}", report.failure);
        assert_eq!(report.additions_checked, 3);
    }

    #[test]
    fn test_pr26_proof_lrat_check_rejects_bad_hint_reference() {
        let (clauses, _) = tiny_unsat_instance();
        // Hint 99 was never added.
        let lrat = "5 2 0 1 99 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(!report.verified);
        assert!(
            report
                .failure
                .unwrap()
                .contains("does not reference a currently active clause")
        );
    }

    #[test]
    fn test_pr26_proof_lrat_check_rejects_hint_that_is_not_unit() {
        let (clauses, _) = tiny_unsat_instance();
        // Clause 1 is (a ∨ b): neither literal is discharged by assuming
        // nothing, so it is not unit — this hint chain cannot derive `b`.
        let lrat = "5 2 0 1 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(!report.verified);
    }

    #[test]
    fn test_pr26_proof_lrat_check_rejects_use_after_delete() {
        let (clauses, _) = tiny_unsat_instance();
        // Delete clause 1 before it is used as hint 1's target.
        let lrat = "0 d 1 0\n5 2 0 1 2 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(!report.verified);
        assert_eq!(report.deletions_applied, 1);
    }

    #[test]
    fn test_pr26_proof_lrat_check_rejects_missing_empty_clause() {
        let (clauses, _) = tiny_unsat_instance();
        // Only derives the two intermediate units, never the empty clause.
        let lrat = "5 2 0 1 2 0\n6 -2 0 3 4 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(!report.verified);
        assert!(report.failure.unwrap().contains("without ever deriving"));
    }

    #[test]
    fn test_pr26_proof_lrat_check_accepts_original_empty_clause() {
        let clauses = vec![vec![1, 2], vec![]];
        let report = check_lrat_proof(&clauses, "");
        assert!(report.verified);
        assert_eq!(report.additions_checked, 0);
    }

    #[test]
    fn test_pr26_proof_lrat_check_accepts_tautology_addition_mid_proof() {
        let (clauses, _) = tiny_unsat_instance();
        // Interpose an unrelated tautology, (a ∨ ¬a), with no hints, between
        // the two genuine derivation steps. Its own addition line must
        // verify trivially (its negation is self-contradictory before any
        // hint is even consulted) without disturbing the rest of the proof,
        // which still needs to go on to derive the empty clause.
        let lrat = "5 2 0 1 2 0\n9 1 -1 0 0\n6 -2 0 3 4 0\n7 0 5 6 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(report.verified, "failure: {:?}", report.failure);
        assert_eq!(report.additions_checked, 4);
    }

    #[test]
    fn test_pr26_proof_lrat_check_rejects_garbage_text() {
        let (clauses, _) = tiny_unsat_instance();
        let report = check_lrat_proof(&clauses, "not a valid lrat line at all\n");
        assert!(!report.verified);
    }

    #[test]
    fn test_pr26_proof_lrat_check_tolerates_redundant_already_satisfied_hint() {
        let (clauses, _) = tiny_unsat_instance();
        // Hint 1 ((a ∨ b)) is already satisfied once `b` is assumed false and
        // `a`... actually simplest: reuse hint 1 twice, the second use is a
        // harmless no-op once it is already satisfied by an earlier
        // propagation (here, clause 1 is not yet satisfied, so exercise the
        // "skip already-true hint" path with a duplicate.
        let lrat = "5 2 0 1 1 2 0\n6 -2 0 3 4 0\n7 0 5 6 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(report.verified, "failure: {:?}", report.failure);
    }

    #[test]
    fn test_pr26_proof_lrat_check_files_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "oxiz_proof_lrat_check_files_{}_{}",
            std::process::id(),
            oxiz_time::SystemTime::now()
                .duration_since(oxiz_time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        let cnf_path = dir.join("tiny.cnf");
        let lrat_path = dir.join("tiny.lrat");

        fs::write(&cnf_path, "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n").expect("write cnf");
        fs::write(&lrat_path, "5 2 0 1 2 0\n6 -2 0 3 4 0\n7 0 5 6 0\n").expect("write lrat");

        let report = check_lrat_files(&cnf_path, &lrat_path).expect("check files");
        assert!(report.verified, "failure: {:?}", report.failure);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pr26_proof_lrat_check_deleting_an_unrelated_clause_does_not_break_a_valid_proof() {
        let (mut clauses, _) = tiny_unsat_instance();
        // A 5th original clause, (c), that the derivation never touches.
        clauses.push(vec![3]);
        // Delete it up front; the rest of the derivation is unaffected.
        let lrat = "0 d 5 0\n6 2 0 1 2 0\n7 -2 0 3 4 0\n8 0 6 7 0\n";
        let report = check_lrat_proof(&clauses, lrat);
        assert!(report.verified, "failure: {:?}", report.failure);
        assert_eq!(report.deletions_applied, 1);
        assert_eq!(report.additions_checked, 3);
    }

    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxiz_proof_lrat_check_{tag}_{}_{}",
            std::process::id(),
            oxiz_time::SystemTime::now()
                .duration_since(oxiz_time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    /// Gatekeeper SK-2: a SATLIB-style CNF trailer (`%` followed by a lone
    /// `0` and, in the wild, further junk like a repeated clause count or a
    /// filename) must not be read as more clause content. Before the fix,
    /// the trailing `0` line parsed as an empty *original* clause, and
    /// `check_lrat_proof` treats any empty original clause as a complete
    /// proof on its own — so a genuinely non-trivial formula with this
    /// trailer would verify against literally any LRAT text, including an
    /// empty one that derives nothing. The real formula here is UNSAT but
    /// requires a real derivation; an empty proof must be rejected.
    #[test]
    fn test_pr26_proof_lrat_check_satlib_percent_trailer_is_not_read_as_an_empty_clause() {
        let dir = unique_temp_dir("satlib_trailer");
        fs::create_dir_all(&dir).expect("create temp dir");
        let cnf_path = dir.join("satlib.cnf");
        let lrat_path = dir.join("empty.lrat");

        fs::write(
            &cnf_path,
            "p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n%\n0\n",
        )
        .expect("write cnf");
        fs::write(&lrat_path, "").expect("write empty lrat");

        let report = check_lrat_files(&cnf_path, &lrat_path).expect("parses cleanly");
        assert!(
            !report.verified,
            "an empty proof must not verify a formula that was never actually refuted"
        );

        // Same CNF, but with the real hand-built derivation from
        // `tiny_unsat_instance`: must verify, confirming the trailer did not
        // also corrupt the four genuine original clauses.
        let real_lrat_path = dir.join("real.lrat");
        fs::write(&real_lrat_path, "5 2 0 1 2 0\n6 -2 0 3 4 0\n7 0 5 6 0\n")
            .expect("write real lrat");
        let report = check_lrat_files(&cnf_path, &real_lrat_path).expect("parses cleanly");
        assert!(report.verified, "failure: {:?}", report.failure);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Gatekeeper SK-2: a clause token that is not a valid DIMACS literal
    /// must be a hard parse error, not a silently-dropped token. Dropping it
    /// would shorten the clause (strictly easier to satisfy than the input
    /// actually says), letting the checker accept a proof that does not
    /// actually refute the real formula.
    #[test]
    fn test_pr26_proof_lrat_check_unparseable_cnf_token_is_an_error() {
        let dir = unique_temp_dir("bad_token");
        fs::create_dir_all(&dir).expect("create temp dir");
        let cnf_path = dir.join("bad.cnf");
        let lrat_path = dir.join("whatever.lrat");
        fs::write(&cnf_path, "p cnf 2 1\n1 elephant 0\n").expect("write cnf");
        fs::write(&lrat_path, "").expect("write lrat");

        let result = check_lrat_files(&cnf_path, &lrat_path);
        match result {
            Err(LratCheckIoError::MalformedCnf(reason)) => {
                assert!(reason.contains("elephant"), "reason: {reason}");
            }
            other => panic!("expected LratCheckIoError::MalformedCnf, got {other:?}"),
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
