//! What the embedded solver keeps for ever, what it scopes, and how a check
//! reads the scope: the root-scoped bit-blasting of re-fix pass 12
//! (`#P2b-46` (f), decision (45)).
//!
//! # The two kinds of clause, and why only one of them is scoped
//!
//! Every clause this solver hands its embedded SAT engine is one of two kinds.
//!
//! * A **definition** — a gate of a bit-blasted circuit (`bvadd`'s adder, a
//!   multiplexer for an `ite`, the comparator behind `bvult`), the truth
//!   variable of a Boolean node (`eq_cache`, `ult_cache`, `slt_cache`,
//!   `bool_node`), the bits of a constant term, the selector clause of a
//!   memoised disjunction.  Each defines fresh variables as a function of
//!   variables that already exist, or states a fact that holds of the term it
//!   encodes whatever the rest of the problem says (a constant's bits).  So a
//!   definition is a **conservative extension of every clause set**: any
//!   assignment to the variables it does not define extends — uniquely, by
//!   evaluating the gate — to one that satisfies it.  It is therefore valid at
//!   every scope, and it is installed at the root with
//!   [`oxiz_sat::Solver::add_clause_at_root`] and never retracted (until
//!   `reset`).
//! * An **assertion** — "this atom holds under the current assignment of the
//!   enclosing search": `a = b`, `a ≠ b`, `a <u b`, a constant pinned onto a
//!   variable, an outer Boolean value pinned onto a node, a lemma of the
//!   bit-vector / EUF exchange.  Each is a single **literal** over a defined
//!   variable (the equality gate, the comparison gate, a bit, the node, the
//!   selector of the lemma's clause), recorded on `active` together with the
//!   term that is to blame for it, and scoped: `push` records the length of
//!   `active` and `pop` truncates it.  No assertion is ever a clause.
//!
//! # A check is one solve under the scope's literals as assumptions
//!
//! [`BvSolver::solve_scope`] hands every active literal (and every pinned
//! outer Boolean value) to [`oxiz_sat::Solver::solve_with_assumptions`].  By
//! the conservative-extension property the definitions add no constraint on
//! the atoms' operands, so the question answered is exactly "is the
//! conjunction of the currently asserted atoms satisfiable over bit-vectors" —
//! for the current scope, and independently of anything asserted and popped
//! before: a popped assertion is simply no longer among the assumptions, and
//! the gate it was asserted through stays behind as a definition.
//!
//! # Why the U-Z10 rollback no longer applies to definitions
//!
//! U-Z10 was a memo entry (`term_to_bv`, `ult_cache`, `eq_cache`, `bool_node`)
//! that outlived the clauses defining it: `sat.pop()` deleted the clauses, the
//! entry survived, and the encoder's idempotence guard handed the next check
//! an unconstrained bit-vector.  The four undo journals existed to retract the
//! entry with its clauses.  A root definition is never deleted, so the entry
//! and its clauses live exactly as long as each other and there is nothing to
//! retract; the journals are gone.  The price they used to charge — a fresh
//! circuit, with fresh SAT variables, every time a popped term was asserted
//! again — was the `O(num_vars)` leak of `#P2b-46` (f): the embedded variable
//! count grew with the number of outer backtracks, not with the formula.
//!
//! # Learned clauses outlive every check, and why that is sound
//!
//! The per-check cleanup this solver used to run — roll the trail back,
//! forget the check's learned clauses, and re-verify an `Unsat` from scratch —
//! existed because an assertion used to be a *bare level-0 unit*: a trail
//! assignment with no clause behind it, which conflict analysis treats as an
//! unconditional fact, so a clause learned under it silently depended on it
//! and became unsound the moment the unit was rolled back.  No assertion is a
//! unit any more.  Assumptions are *decisions* at levels above 0, which
//! conflict analysis resolves through like any other decision, and the only
//! level-0 facts are root definitions and learned units — all permanent.  So
//! every learned clause is a consequence of the permanent clause set alone,
//! valid under every later assumption set, and the cleanup is dropped: the
//! embedded engine keeps what it learns, check after check.
//!
//! `#P2b-19`/`#P2b-21`'s assertion-level registration — a learned clause, or a
//! theory reason clause, is registered in the innermost open assertion level
//! so that `pop` retracts it with the scope that entailed it — is consistent
//! with this: the embedded engine never opens an assertion level now, so every
//! registration lands at level 0, which is exactly the level a clause entailed
//! by the permanent clause set belongs to.
//!
//! # The explanation of an `Unsat` is the core, not the scope
//!
//! `solve_with_assumptions` returns the subset of the assumptions its
//! refutation used.  Each is mapped back to the term that asserted it (the
//! `blame` recorded by [`BvSolver::record_constraint_term`], or the pinned
//! outer atom); their conjunction is unsatisfiable together with definitions
//! that hold in every bit-vector model, so its negation is a valid theory
//! lemma.  The old explanation — every recorded constraint term and every pin
//! — was sound too, but blamed the whole scope, so the enclosing CDCL(T)
//! search learned one clause per refuted *combination* of bit-vector atoms:
//! `#P2b-59`'s repro spent 6,300 of its 8,294 outer conflicts in one
//! refinement round doing exactly that.  When a core literal carries no blame
//! (an assertion made through the public API without a recorded term), the
//! whole-scope explanation is used, so a caller that records nothing is never
//! handed an explanation that omits its hypotheses.

use super::{BvSolver, ComparisonKey};
use oxiz_core::ast::TermId;
use oxiz_sat::{Lit, SolverResult, Var};
use smallvec::SmallVec;

/// Whether `model` assigns `lit` true (an unassigned or unknown variable does
/// not count).
fn literal_holds(model: &[oxiz_sat::LBool], lit: Lit) -> bool {
    model.get(lit.var().index()).is_some_and(|value| {
        if lit.is_pos() {
            value.is_true()
        } else {
            value.is_false()
        }
    })
}

/// One assertion of the current scope: the literal a check assumes, and the
/// term whose assertion made it (see [`BvSolver::record_constraint_term`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Asserted {
    pub(super) lit: Lit,
    pub(super) blame: Option<TermId>,
}

impl BvSolver {
    /// Retract every assertion — every scope, every pin, every recorded
    /// constraint term — and keep every definition, every learned clause and
    /// the last model.
    ///
    /// The soft counterpart of `Theory::reset`, for an owning solver that
    /// starts a new search over the *same* terms (a refinement round, an MBQI
    /// round): the circuits it encoded are definitions, valid in the new
    /// search exactly as in the old one, so re-encoding them from scratch is
    /// pure cost — on a 64-store replay script it was most of the check.
    /// Nothing asserted survives: the new search starts with an empty scope,
    /// as after `reset`.
    pub fn retract_assertions(&mut self) {
        self.assertions.clear();
        self.context_stack.clear();
        self.assertion_guard_terms.clear();
        self.active.clear();
        self.blame_from = 0;
        self.outer_bool.clear();
        self.outer_bool_journal.clear();
        self.shared_equalities.clear();
        self.equality_notifications.clear();
    }

    /// Install a definitional clause at the root; see the module docs.
    pub(super) fn define(&mut self, lits: impl IntoIterator<Item = Lit>) {
        // A new clause may be one the last model does not satisfy.
        self.model_is_current = false;
        let _ = self.sat.add_clause_at_root(lits);
    }

    /// A variable defined to be the constant `value` (one per polarity per
    /// solver), for the constant inputs of adders and the like.
    pub(super) fn const_var(&mut self, value: bool) -> Var {
        let slot = if value {
            self.const_true
        } else {
            self.const_false
        };
        if let Some(var) = slot {
            return var;
        }
        let var = self.sat.new_var();
        self.define([if value { Lit::pos(var) } else { Lit::neg(var) }]);
        if value {
            self.const_true = Some(var);
        } else {
            self.const_false = Some(var);
        }
        var
    }

    /// Assert `lit` in the current scope (blame recorded later by
    /// [`Self::record_constraint_term`]).
    pub(super) fn assert_lit(&mut self, lit: Lit) {
        self.active.push(Asserted { lit, blame: None });
    }

    /// The selector literal of the disjunction `lits`, defined once per
    /// distinct disjunction by the root clause `¬s ∨ lits`.
    pub(super) fn clause_selector(&mut self, lits: &[Lit]) -> Lit {
        let mut key: SmallVec<[Lit; 4]> = lits.iter().copied().collect();
        key.sort_unstable_by_key(|lit| lit.code());
        key.dedup();
        if let Some(&selector) = self.clause_selectors.get(&key) {
            return Lit::pos(selector);
        }
        let selector = self.sat.new_var();
        let mut clause: SmallVec<[Lit; 8]> = key.iter().copied().collect();
        clause.push(Lit::neg(selector));
        self.define(clause);
        self.clause_selectors.insert(key, selector);
        Lit::pos(selector)
    }

    /// The signed less-than gate `a <s b`, defined once per ordered pair.
    pub(super) fn slt_gate(&mut self, a: TermId, b: TermId) -> Option<Var> {
        let key = ComparisonKey { a, b };
        if let Some(&var) = self.slt_cache.get(&key) {
            return Some(var);
        }
        let (va, vb) = self.binop_bits(a, b)?;
        let width = va.width as usize;
        if width == 0 {
            return None;
        }
        let sign_a = va.bits[width - 1];
        let sign_b = vb.bits[width - 1];
        let diff_sign = self.sat.new_var();
        self.encode_xor(diff_sign, sign_a, sign_b);
        let result = self.sat.new_var();
        // Signs differ: a < b iff a is negative.
        self.define([Lit::neg(diff_sign), Lit::neg(sign_a), Lit::pos(result)]);
        self.define([Lit::neg(diff_sign), Lit::pos(sign_a), Lit::neg(result)]);
        // Signs agree: the unsigned comparison decides.
        let ult = self.ult_gate(a, b)?;
        self.define([Lit::pos(diff_sign), Lit::neg(ult), Lit::pos(result)]);
        self.define([Lit::pos(diff_sign), Lit::pos(ult), Lit::neg(result)]);
        self.slt_cache.insert(key, result);
        Some(result)
    }

    /// The unsigned less-than gate `a <u b`, defined once per ordered pair.
    pub(super) fn ult_gate(&mut self, a: TermId, b: TermId) -> Option<Var> {
        let key = ComparisonKey { a, b };
        if let Some(&var) = self.ult_cache.get(&key) {
            return Some(var);
        }
        let (va, vb) = self.binop_bits(a, b)?;
        let var = self.sat.new_var();
        self.encode_ult_result(&va.bits, &vb.bits, var);
        // Asymmetry, a theorem of the order: never both `a < b` and `b < a`.
        if let Some(&reverse) = self.ult_cache.get(&ComparisonKey { a: b, b: a }) {
            self.define([Lit::neg(var), Lit::neg(reverse)]);
        }
        self.ult_cache.insert(key, var);
        Some(var)
    }

    /// The outer Boolean values currently pinned onto a live boolean node, as
    /// `(term, literal)`, in term order (so the assumption order, and with it
    /// the embedded search, is deterministic).
    pub(super) fn current_pins(&self) -> Vec<(TermId, Lit)> {
        let mut pins: Vec<(TermId, Lit)> = self
            .outer_bool
            .iter()
            .filter_map(|(&term, &value)| {
                let &var = self.bool_node.get(&term)?;
                Some((term, if value { Lit::pos(var) } else { Lit::neg(var) }))
            })
            .collect();
        pins.sort_unstable_by_key(|&(term, _)| term.raw());
        pins
    }

    /// One embedded solve under the scope's assumptions: every active
    /// assertion, then every pin.  Arms the budget, charges the conflicts,
    /// snapshots the model on `Sat`, and returns the core on `Unsat`.
    pub(super) fn solve_scope(&mut self) -> (SolverResult, Option<Vec<Lit>>) {
        // Whatever was asserted without a recorded term stays unblamed from
        // here on (see `record_constraint_term`).
        self.blame_from = self.active.len();
        let mut assumptions: Vec<Lit> = self.active.iter().map(|entry| entry.lit).collect();
        assumptions.extend(self.current_pins().into_iter().map(|(_, lit)| lit));
        // The last model already answers the question when no clause has been
        // added since it was found and it satisfies every assumption: it is a
        // model of the whole clause set (a learned clause since is implied by
        // that set, so it holds there too) and of the scope, so `Sat` is
        // exact and the snapshot stays the model the check reports.  This is
        // the common case once the round's circuits exist — a re-asserted
        // atom whose literal the current model already makes true — and it
        // costs one lookup per assumption instead of a search that assigns
        // every variable of every circuit defined so far.
        if self.model_is_current
            && assumptions
                .iter()
                .all(|&lit| literal_holds(&self.last_sat_model, lit))
        {
            // A variable minted since the model was found (a free leaf, a
            // circuit's fresh bit before its first clause) occurs in no
            // clause — any clause would have cleared `model_is_current` — so
            // every value of it extends the model: the snapshot is widened to
            // it (unassigned, read as `0`), which is what lets
            // `snapshot_covers` tell such a variable from one the snapshot has
            // never seen.
            let vars = self.sat.num_vars();
            if self.last_sat_model.len() < vars {
                self.last_sat_model.resize(vars, oxiz_sat::LBool::Undef);
            }
            return (SolverResult::Sat, None);
        }
        let allowance = self.remaining_conflict_budget();
        self.apply_budget_to_embedded(allowance);
        let conflicts_before = self.sat.stats().conflicts;
        let (result, core) = self.sat.solve_with_assumptions(&assumptions);
        self.charge_embedded_conflicts(conflicts_before);
        if result == SolverResult::Sat {
            self.last_sat_model = self.sat.model().to_vec();
            self.model_is_current = true;
        }
        (result, core)
    }

    /// Whether every bit of `term`'s circuit existed when the model snapshot
    /// of the last `Sat` check was taken (`false` for a term with no circuit).
    ///
    /// A circuit defined *after* that check — a definition installed while
    /// encoding an atom that was then never asserted (an operand the encoder
    /// could model beside one it could not), or one built for a partition
    /// candidate — has no value in the snapshot, and
    /// [`Self::get_value_big`] then reads its bits as `0` whatever they are
    /// defined to be: a literal `#b1` read back as `#b0`.  A caller that
    /// compares circuit values (the bit-vector / EUF exchange of
    /// `oxiz-solver`) must refresh the snapshot with a check first; reading
    /// such a value is how the stale `#b1 = #b0` of the recheck-12 wrong
    /// `unsat` was manufactured.
    ///
    /// The test is on the variable's *index*, not on the snapshot holding a
    /// defined value: a variable the last `Sat` search left unassigned is a
    /// don't-care of that model (every clause is satisfied without it), and
    /// reading it as `0` is a value of the model, while a variable minted
    /// after the snapshot is not in it at all.  (Testing for a defined value
    /// made a free leaf look stale after every refresh, and the exchange
    /// re-checked until its round bound turned the verdict into `unknown`.)
    #[must_use]
    pub fn snapshot_covers(&self, term: TermId) -> bool {
        let known = self.last_sat_model.len();
        self.term_to_bv
            .get(&term)
            .is_some_and(|bv| bv.bits.iter().all(|var| var.index() < known))
    }

    /// The terms to blame for a refutation whose failed assumptions are
    /// `core`: each literal's recorded blame, or its pinned atom.  `None` when
    /// some literal carries no blame (or the core is empty), in which case the
    /// caller falls back to the whole scope.
    pub(super) fn explain_core(&self, core: &[Lit]) -> Option<Vec<TermId>> {
        if core.is_empty() {
            return None;
        }
        let pins = self.current_pins();
        let mut terms: Vec<TermId> = Vec::with_capacity(core.len());
        for lit in core {
            // Any active entry that asserted this literal justifies it; prefer
            // one with a recorded blame.
            let blame = match self
                .active
                .iter()
                .filter(|entry| entry.lit == *lit)
                .find_map(|entry| entry.blame)
            {
                Some(term) => term,
                None => pins
                    .iter()
                    .find(|(_, pin)| pin == lit)
                    .map(|&(term, _)| term)?,
            };
            if !terms.contains(&blame) {
                terms.push(blame);
            }
        }
        Some(terms)
    }
}
