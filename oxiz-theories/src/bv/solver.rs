//! BitVector Theory Solver

use crate::config::BvConfig;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::theory::{EqualityNotification, Theory, TheoryId, TheoryResult};
use num_bigint::BigUint;
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use oxiz_sat::{LBool, Lit, Solver as SatSolver, SolverConfig as SatConfig, SolverResult, Var};
use smallvec::SmallVec;

/// Boolean nodes: `ite` selectors, the outer-assignment pins and the conflict
/// hypotheses they contribute (`#P2b-24`, `#P2b-25`).
mod bool_node;
/// Resource budgets for the embedded bit-blasting SAT solver (U-Z12).
mod budget;
/// Nelson-Oppen combination: the `TheoryCombination` implementation.
mod combination;
/// Division / remainder encodings (`bvudiv`, `bvurem`, `bvsdiv`, `bvsrem`).
mod division;
/// Barrel-shifter encodings (`bvshl`, `bvlshr`, `bvashr`).
mod shifts;

/// A bit vector variable (sequence of SAT variables)
#[derive(Debug, Clone)]
pub struct BvVar {
    /// SAT variables for each bit (LSB first)
    bits: SmallVec<[Var; 32]>,
    /// Width in bits
    width: u32,
}

/// Bit `index` of `constant`, read as the bit of an arbitrarily wide
/// bit-vector whose low 64 bits are `constant` and whose higher bits are `0`.
///
/// A `u64` has no bit at index 64 or above, so the answer there is `false`.
/// Saying that *totally* is the point: OxiZ 0.3.3/0.3.4 wrote
/// `((constant >> i) & 1) == 1` inside `encode_add_const`, where `i` runs over
/// the bit-vector width. Rust's `>>` on a `u64` uses only the low 6 bits of the
/// shift amount, so bit 64 of `constant = 1` read back as `1` in release builds
/// (and panicked with "attempt to shift right with overflow" in debug ones).
/// Every call site passes `constant = 1` as the `+1` of a two's-complement
/// negation, so every `bvsub`/`bvneg` circuit wider than 64 bits was blasted
/// against `1 + 2^64 + 2^128 + …` instead of `1` — a different formula, which
/// fabricated both wrong `sat` answers and wrong `unsat` *proofs*
/// (`oxiz-solver/tests/bv_wide_soundness.rs`, the `wide_*_above_64` group).
#[inline]
#[must_use]
fn const_bit_of(constant: u64, index: usize) -> bool {
    match u32::try_from(index) {
        Ok(i) if i < u64::BITS => (constant >> i) & 1 == 1,
        _ => false,
    }
}

/// Comparison tracking for conflict detection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ComparisonKey {
    a: TermId,
    b: TermId,
}

/// Saved lengths of every retractable buffer, recorded by `push` so `pop`
/// restores exactly the state the enclosing decision level had.
#[derive(Debug, Clone, Copy)]
struct ContextMark {
    /// Length of `assertions`.
    assertions_len: usize,
    /// Length of `assertion_guard_terms`.
    guard_terms_len: usize,
    /// Length of `outer_bool_journal`.
    outer_bool_len: usize,
    /// Length of `term_to_bv_journal`.
    term_to_bv_len: usize,
    /// Length of `ult_cache_journal`.
    ult_cache_len: usize,
    /// Length of `eq_cache_journal`.
    eq_cache_len: usize,
    /// Length of `bool_node_journal`.
    bool_node_len: usize,
    /// Length of `pinned_terms`.
    pinned_len: usize,
    /// Length of `opaque_leaves`.
    opaque_len: usize,
}

/// BitVector Theory Solver using bit-blasting
#[derive(Debug)]
pub struct BvSolver {
    /// Embedded SAT solver
    sat: SatSolver,
    /// Term to BV variable mapping
    term_to_bv: FxHashMap<TermId, BvVar>,
    /// Pending assertions
    assertions: Vec<(TermId, bool)>,
    /// Context stack: one [`ContextMark`] per open push, restored by `pop`.
    context_stack: Vec<ContextMark>,
    /// Configuration
    config: BvConfig,
    /// Track unsigned less-than comparisons for conflict detection
    /// Maps (a, b) -> SAT variable representing a < b
    ult_cache: FxHashMap<ComparisonKey, Var>,
    /// The truth variable of the bit equality `a = b` per unordered operand
    /// pair (`a.raw() <= b.raw()`), built once by `bool_bv_eq` and reused by
    /// every later Bool `=` node and every lemma of the bit-vector / EUF
    /// exchange that mentions the pair.
    ///
    /// Without it each lemma of `assert_any` re-encoded its pair
    /// disequalities from scratch — three fresh variables and nine clauses
    /// per bit per pair, every round — so the embedded instance
    /// grew with the round count (76 variables / 171 clauses at round 1 of
    /// a width-3 QF_ABV script, 16,689 / 50,354 at round 500) and the
    /// full-assignment search each round runs grew with it: 4 ms per round
    /// at the start, 429 ms at the end, 52–60 s to reach `MAX_LEMMAS`.  The
    /// pairs a loop can mention are bounded by the candidates, so with the
    /// memo the instance stops growing after the first few rounds and a
    /// give-up costs seconds (`#P2b-29`'s open remainder).
    eq_cache: FxHashMap<ComparisonKey, Var>,
    /// Shared equalities derived by BV theory for Nelson-Oppen combination.
    /// BV is a finite domain theory, so equalities are extracted from the
    /// current model/assignment using model-based combination.
    shared_equalities: Vec<EqualityNotification>,
    /// Pending equality notifications received from other theories
    equality_notifications: Vec<EqualityNotification>,
    /// Constraint-level TermIds recorded by the theory manager for conflict reporting.
    /// On UNSAT, all recorded terms form a sound (superset) conflict explanation.
    assertion_guard_terms: Vec<TermId>,
    /// Snapshot of the embedded SAT model captured at the most recent SAT
    /// `check()`, taken *before* `backtrack_to_root()` discards the live trail.
    /// Without this snapshot, `get_value` would read an all-`Undef` (→ 0) trail
    /// after backtracking, producing degenerate counterexample models.
    last_sat_model: Vec<LBool>,
    /// Cache mapping a Bool-sorted term to the single SAT variable encoding its
    /// truth value, used when bit-blasting `ite` conditions and the boolean
    /// connectives that build them (so `not(c)` stays the negation of `c`).
    bool_node: FxHashMap<TermId, Var>,
    /// Truth values the *outer* CDCL(T) search has fixed for Bool-sorted terms,
    /// so that a bit-blasted `ite` selector agrees with the enclosing solver
    /// instead of floating free. See [`Self::assert_bool_value`].
    outer_bool: FxHashMap<TermId, bool>,
    /// Undo journal for `outer_bool`: `(term, value it had before)`, replayed
    /// in reverse by [`Theory::pop`] so the link is retracted with the decision
    /// level that established it.
    outer_bool_journal: Vec<(TermId, Option<bool>)>,
    /// Undo journal for `term_to_bv`: every term whose circuit was *created*
    /// (not merely looked up) since the enclosing push.
    ///
    /// The invariant these journals restore is the one the solver silently
    /// assumed and did not maintain: *a term has a `term_to_bv` /
    /// `ult_cache` / `eq_cache` / `bool_node` entry **iff** the clauses
    /// defining that entry are live in the embedded SAT solver*. `sat.pop()` deletes exactly the
    /// clauses added since the matching `push`, so a cache entry that survived
    /// the pop handed the next `check()` a completely unconstrained bit-vector,
    /// and the idempotence guard in `oxiz-solver`'s
    /// `theory_bv_encode::encode_bv_term_recursive` then refused to re-encode
    /// it. That is finding U-Z10: wrong `sat` on QF_BV with Boolean structure
    /// over BV atoms (4.8 % of random unsat width-8 formulas).
    term_to_bv_journal: Vec<TermId>,
    /// Undo journal for `ult_cache`; see [`Self::term_to_bv_journal`].
    ult_cache_journal: Vec<ComparisonKey>,
    /// Undo journal for `eq_cache`; see [`Self::term_to_bv_journal`].  An
    /// equality variable is built after both operand circuits exist, so it
    /// is journalled at the same or a deeper scope than either operand and
    /// is retracted with whichever of them goes first.
    eq_cache_journal: Vec<ComparisonKey>,
    /// Undo journal for `bool_node`; see [`Self::term_to_bv_journal`].
    bool_node_journal: Vec<TermId>,
    /// Every outer Boolean atom whose truth value is *currently pinned* into
    /// a live boolean node by a unit clause (see [`Self::pin_bool_var`]), in
    /// pin order, truncated by [`Theory::pop`] to the length `push` recorded.
    ///
    /// These are conflict hypotheses.  A pin is an assertion the embedded SAT
    /// solver may resolve against exactly like an `assert_eq` unit, so an
    /// `Unsat` it contributes to is `Unsat` *under that atom's value*; an
    /// explanation that omitted it made the owning CDCL(T) core learn a clause
    /// stronger than the theory derived — a **false proof**.  `(= x (ite c a
    /// b))` with `c` pinned `true` and `x = b` asserted is unsatisfiable only
    /// while `c` holds, and the conflict clause used to say it was
    /// unsatisfiable outright.  See [`Self::collect_conflict_terms`] and the
    /// `#P2b-25` entry in `TODO.md`.
    pinned_terms: Vec<TermId>,
    /// The set behind [`Self::pinned_terms`], so a re-pin of an already
    /// blamed atom is not journalled twice.
    pinned_set: FxHashSet<TermId>,
    /// Every bit-vector-sorted term whose circuit is a *free* bit-vector
    /// standing in for a value this theory knows nothing about — an
    /// uninterpreted application, an array `select` — in creation order,
    /// truncated by [`Theory::pop`] to the length `push` recorded, exactly
    /// like `term_to_bv_journal` (the leaf's circuit goes with the same pop).
    ///
    /// These are the terms the *EUF* layer may prove equal by congruence
    /// (`f(a) = f(b)` from `a = b`) while this circuit still holds two
    /// unrelated free vectors for them; `oxiz-solver`'s theory manager reads
    /// the list back through [`Self::opaque_leaves()`], interns every entry
    /// into congruence closure, and asserts the bit-equality of any two that
    /// congruence puts in one class (`#P2b-29`).  Without that crossing,
    /// `(= a b) ∧ (distinct (bvadd (f a) #x01) (bvadd (f b) #x01))` answered
    /// `sat`.
    opaque_leaves: Vec<TermId>,
    /// The set behind [`Self::opaque_leaves()`].
    opaque_leaf_set: FxHashSet<TermId>,
    /// Total conflict allowance for every embedded `solve()` until the next
    /// [`Self::set_budget`], or `None` for unbounded.  See the `budget` module.
    budget_max_conflicts: Option<u64>,
    /// Wall-clock deadline shared by every embedded `solve()`, or `None`.
    /// Shared rather than re-derived per probe, so `(set-option :timeout N)`
    /// bounds the whole check instead of granting `N` ms to each of the
    /// hundreds of probes a search makes.  See the `budget` module.
    budget_deadline: Option<oxiz_time::Instant>,
    /// Embedded SAT conflicts charged against `budget_max_conflicts` so far.
    ///
    /// The accumulator lives here, not in the embedded solver's statistics,
    /// because [`Theory::reset`] calls `sat.reset()` — which zeroes those
    /// statistics — and the owning solver resets this theory once per check
    /// *and* once per repair round.  Keeping the running total here is what
    /// makes the allowance a total rather than a per-round re-arm.
    conflicts_spent: u64,
}

impl Default for BvSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BvSolver {
    /// Create a new BitVector solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(BvConfig::default())
    }

    /// Create a new BitVector solver with custom configuration
    #[must_use]
    pub fn with_config(config: BvConfig) -> Self {
        Self {
            sat: SatSolver::with_config(Self::embedded_sat_config()),
            term_to_bv: FxHashMap::default(),
            assertions: Vec::new(),
            context_stack: Vec::new(),
            config,
            ult_cache: FxHashMap::default(),
            eq_cache: FxHashMap::default(),
            shared_equalities: Vec::new(),
            equality_notifications: Vec::new(),
            assertion_guard_terms: Vec::new(),
            last_sat_model: Vec::new(),
            bool_node: FxHashMap::default(),
            outer_bool: FxHashMap::default(),
            outer_bool_journal: Vec::new(),
            term_to_bv_journal: Vec::new(),
            ult_cache_journal: Vec::new(),
            eq_cache_journal: Vec::new(),
            bool_node_journal: Vec::new(),
            pinned_terms: Vec::new(),
            pinned_set: FxHashSet::default(),
            opaque_leaves: Vec::new(),
            opaque_leaf_set: FxHashSet::default(),
            budget_max_conflicts: None,
            budget_deadline: None,
            conflicts_spent: 0,
        }
    }

    /// SAT-solver configuration for the embedded bit-blasting engine.
    ///
    /// `BvSolver::check()` drives the SAT solver *incrementally*: it asserts
    /// clauses, runs a full `solve()`, then discards that probe's search
    /// residue so the next probe sees only the honestly-asserted clauses. The
    /// residue cleanup relies on two contracts — `restore_to_trail_size`
    /// (roll the trail back to the committed prefix) and `forget_learned_since`
    /// (drop exactly the clauses this probe *learned*). The second contract
    /// only covers clauses registered in the SAT solver's learned-clause list.
    ///
    /// Any search feature that injects *other* clauses into the database during
    /// `solve()` therefore breaks the contract: those clauses are invisible to
    /// `forget_learned_since`, survive both the per-probe cleanup and the
    /// enclosing `pop()`, and leak into later probes. Because the bit-vector
    /// unit constraints installed by `assert_const`/`assert_eq` sit on the
    /// trail as bare level-0 decisions, such a leaked clause can implicitly
    /// depend on a since-retracted assignment and spuriously force `Unsat` —
    /// e.g. turning the genuinely satisfiable `a = x*3 ∧ a ≠ x ∧ a = 7` into a
    /// false `Unsat` once an earlier probe has run.
    ///
    /// The two offenders are **lazy hyper-binary resolution** (adds derived
    /// binary clauses mid-search) and **inprocessing** (adds/rewrites clauses
    /// between search rounds). Both are pure performance heuristics — disabling
    /// them costs only speed, never soundness or completeness — so the embedded
    /// solver turns them off to keep the incremental cleanup contract exact.
    ///
    /// Chronological backtracking, by contrast, is left **on** (the workspace
    /// default): it never adds a clause to the database, so it does not touch
    /// the cleanup contract above.  It was briefly disabled here while
    /// `oxiz-sat` recorded a learned clause's asserting literal at the
    /// post-backtrack decision level instead of at its true implication level —
    /// which pinned unit lemmas inside a decision level and let conflict
    /// analysis emit clauses stronger than resolution derives, refuting
    /// satisfiable bit-blasted circuits such as `x <=u (x bvxor x)`.  That is
    /// fixed in the SAT engine itself (see `Solver::assert_learned_clause` and
    /// `Trail::backtrack_to_with_callback`), so the embedded solver no longer
    /// needs to opt out.
    fn embedded_sat_config() -> SatConfig {
        SatConfig {
            enable_lazy_hyper_binary: false,
            enable_inprocessing: false,
            ..SatConfig::default()
        }
    }

    /// Record a constraint-level TermId that the theory manager is about to assert.
    ///
    /// The theory manager calls this before each `assert_const` / `assert_eq` /
    /// `assert_ult` etc. so that `check()` can return a non-empty conflict clause
    /// on UNSAT.  The term must be a constraint term that is registered in the
    /// theory manager's `term_to_var` map, so that `terms_to_conflict_clause`
    /// can convert it to a SAT literal.
    pub fn record_constraint_term(&mut self, term: TermId) {
        // Deduplicate: only add if not already present
        if !self.assertion_guard_terms.contains(&term) {
            self.assertion_guard_terms.push(term);
        }
    }

    /// Every hypothesis the embedded SAT solver may have resolved against, as
    /// the conflict explanation: the recorded constraint terms **and** the
    /// outer Boolean atoms currently pinned into the circuit
    /// ([`Self::pinned_terms`]).
    ///
    /// A sound superset: the `Unsat` is caused by the conjunction of
    /// everything asserted since the last push/reset, and a clause that
    /// blames more than the minimal core is merely weaker, never wrong.  What
    /// is *not* sound is blaming less.  Until `#P2b-25` this returned the
    /// constraint terms alone, while `assert_bool_value` had installed the
    /// values of `ite` selectors and comparison nodes as level-scoped unit
    /// clauses the same solver resolved through; a conflict that rested on
    /// such a unit was handed back as if it rested on the constraints only,
    /// and the CDCL(T) core learned a lemma that the theory never derived.
    /// Measured in the 0.3.4 tree: a satisfiable width-63 script answered
    /// `unsat` (`oxiz-solver/tests/bv_ite_selfcheck_regressions.rs`).
    fn collect_conflict_terms(&self) -> Vec<TermId> {
        let mut terms = self.assertion_guard_terms.clone();
        for &term in &self.pinned_terms {
            if !terms.contains(&term) {
                terms.push(term);
            }
        }
        terms
    }

    /// Create a new bit vector variable
    ///
    /// A *creation* (as opposed to a cache hit) is journalled so that
    /// [`Theory::pop`] retracts it together with the clauses `sat.pop()`
    /// deletes; see the `term_to_bv_journal` field.
    pub fn new_bv(&mut self, term: TermId, width: u32) -> &BvVar {
        if !self.term_to_bv.contains_key(&term) {
            self.term_to_bv_journal.push(term);
        }
        self.term_to_bv.entry(term).or_insert_with(|| {
            let bits: SmallVec<[Var; 32]> = (0..width).map(|_| self.sat.new_var()).collect();
            BvVar { bits, width }
        })
    }

    /// Get the BV variable for a term
    #[must_use]
    pub fn get_bv(&self, term: TermId) -> Option<&BvVar> {
        self.term_to_bv.get(&term)
    }

    /// Create the free bit-vector standing in for an *opaque leaf* — a
    /// bit-vector-sorted term that is not a bit-vector operation (an
    /// uninterpreted application, an array `select`, …) — and record it on
    /// [`Self::opaque_leaves()`] so the owning theory manager can share the
    /// equalities congruence closure derives for it (`#P2b-29`).
    ///
    /// Same journalling as [`Self::new_bv`]: the record is retracted by the
    /// `pop` that retracts the circuit.
    pub fn new_opaque_leaf(&mut self, term: TermId, width: u32) -> &BvVar {
        if self.opaque_leaf_set.insert(term) {
            self.opaque_leaves.push(term);
        }
        self.new_bv(term, width)
    }

    /// The opaque leaves currently holding a live free circuit, in creation
    /// order (see [`Self::new_opaque_leaf`]).
    #[must_use]
    pub fn opaque_leaves(&self) -> &[TermId] {
        &self.opaque_leaves
    }

    /// Every Bool-sorted term that currently has a boolean node in this
    /// circuit (an `ite` selector, or a connective / comparison underneath
    /// one), in no particular order.  Paired with [`Self::bool_value`] this
    /// is how a model can publish the value the circuit chose for a Boolean
    /// the enclosing search never assigned (`#P2b-27`).
    pub fn bool_node_terms(&self) -> impl Iterator<Item = TermId> + '_ {
        self.bool_node.keys().copied()
    }

    /// Every term that currently has a bit-blasted circuit, in no
    /// particular order.  Paired with [`Self::get_value_big`] this is how a
    /// model can publish the value the circuit chose for a bit-vector
    /// variable the owning solver never tracked as a theory variable — one
    /// that occurs only as the argument of an uninterpreted function, whose
    /// circuit the equality exchange built (`#P2b-29`).
    pub fn circuit_terms(&self) -> impl Iterator<Item = TermId> + '_ {
        self.term_to_bv.keys().copied()
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &BvConfig {
        &self.config
    }

    /// The bit-vectors of two operands, if both are bit-blasted **and** share a
    /// width.
    ///
    /// Every binary bit-level encoding goes through this: a caller that hands
    /// over operands of *different* widths (which the term builder does not
    /// currently reject for `(bvadd x8 y16)`) has asked for a circuit that does
    /// not exist, and the honest answer is "not encodable" — not an
    /// `assert_eq!` that aborts the process, and not a circuit wired from the
    /// bits that happen to line up.
    fn binop_bits(&self, a: TermId, b: TermId) -> Option<(BvVar, BvVar)> {
        let va = self.term_to_bv.get(&a)?.clone();
        let vb = self.term_to_bv.get(&b)?.clone();
        (va.width == vb.width).then_some((va, vb))
    }

    /// Assert equality: a = b
    ///
    /// Returns `false` — asserting nothing — when either operand has not been
    /// bit-blasted or the two have different widths.
    #[must_use]
    pub fn assert_eq(&mut self, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            for i in 0..va.width as usize {
                // a[i] <=> b[i]
                // (a[i] => b[i]) and (b[i] => a[i])
                // (~a[i] or b[i]) and (~b[i] or a[i])
                self.sat
                    .add_clause([Lit::neg(va.bits[i]), Lit::pos(vb.bits[i])]);
                self.sat
                    .add_clause([Lit::neg(vb.bits[i]), Lit::pos(va.bits[i])]);
            }
            return true;
        }
        false
    }

    /// Assert disequality: a != b
    ///
    /// Returns `false` — asserting nothing — when either operand has not been
    /// bit-blasted or the two have different widths.
    #[must_use]
    pub fn assert_neq(&mut self, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            // At least one bit must differ
            // Introduce auxiliary variables for XOR of each bit pair
            let mut diff_lits: SmallVec<[Lit; 32]> = SmallVec::new();

            for i in 0..va.width as usize {
                // diff[i] = a[i] XOR b[i]
                let diff = self.sat.new_var();
                diff_lits.push(Lit::pos(diff));

                let ai = va.bits[i];
                let bi = vb.bits[i];

                // diff <=> (a XOR b)
                // diff => (a or b) and (~a or ~b)
                // ~diff => (~a or b) and (a or ~b)
                self.sat
                    .add_clause([Lit::neg(diff), Lit::pos(ai), Lit::pos(bi)]);
                self.sat
                    .add_clause([Lit::neg(diff), Lit::neg(ai), Lit::neg(bi)]);
                self.sat
                    .add_clause([Lit::pos(diff), Lit::neg(ai), Lit::pos(bi)]);
                self.sat
                    .add_clause([Lit::pos(diff), Lit::pos(ai), Lit::neg(bi)]);
            }

            // At least one diff bit must be true
            self.sat.add_clause(diff_lits);
            return true;
        }
        false
    }

    /// Assert unsigned less than: a < b
    ///
    /// Returns `false` — asserting nothing — when either operand has not been
    /// bit-blasted or the two have different widths.
    #[must_use]
    pub fn assert_ult(&mut self, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            // Get or create comparison result variable for a < b
            let key_ab = ComparisonKey { a, b };
            let ult_ab = if let Some(&var) = self.ult_cache.get(&key_ab) {
                var
            } else {
                let var = self.sat.new_var();
                self.encode_ult_result(&va.bits, &vb.bits, var);
                self.ult_cache.insert(key_ab.clone(), var);
                self.ult_cache_journal.push(key_ab.clone());
                var
            };

            // Assert that a < b is true
            self.sat.add_clause([Lit::pos(ult_ab)]);

            // Check for conflict with b < a
            let key_ba = ComparisonKey { a: b, b: a };
            if let Some(&ult_ba) = self.ult_cache.get(&key_ba) {
                // If both a < b and b < a are asserted, we have a conflict
                // Add clause: NOT(a < b) OR NOT(b < a)
                // Since we already asserted a < b, this will make b < a false
                self.sat.add_clause([Lit::neg(ult_ab), Lit::neg(ult_ba)]);
            }

            // Also check for conflict with a <= b and b <= a
            // If a < b, then NOT(a = b), so we ensure anti-symmetry
            return true;
        }
        false
    }

    /// Assert unsigned less than or equal: a <= b
    ///
    /// `ule(a, b)` is equivalent to `NOT(ult(b, a))`: encode the unsigned
    /// comparison `b < a` into a fresh SAT variable and assert its negation.
    ///
    /// Returns `false` — asserting nothing — when either operand has not been
    /// bit-blasted or the two have different widths.
    #[must_use]
    pub fn assert_ule(&mut self, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            // Encode b < a (unsigned) into `ult_ba`.
            let ult_ba = self.sat.new_var();
            self.encode_ult_result(&vb.bits, &va.bits, ult_ba);

            // Assert NOT(b < a), which is exactly a <= b.
            self.sat.add_clause([Lit::neg(ult_ba)]);
            return true;
        }
        false
    }

    /// Assert a constant value for a bit vector whose width is at most 64.
    ///
    /// `value` supplies the low 64 bits; every higher bit of a wider vector is
    /// pinned to `0`, which is only the intended meaning when the constant
    /// really does fit in a `u64`.  A caller holding a *wider* constant must
    /// use [`Self::assert_const_big`] (or [`Self::assert_const_limbs`]):
    /// truncating it to a `u64` here would pin the wrong bits and admit
    /// assignments the constant forbids — the shape that answered `sat` for
    /// `x = 2^64 ∧ x <u 1` at width 128.
    ///
    /// Returns `false` when `term` already has a bit-vector of a *different*
    /// width, in which case nothing is pinned: the caller asked for a constant
    /// the existing circuit cannot represent, and silently pinning the bits
    /// that happen to exist would constrain a different value.
    pub fn assert_const(&mut self, term: TermId, value: u64, width: u32) -> bool {
        self.assert_const_limbs(term, &[value], width)
    }

    /// Assert an arbitrary-width constant value for a bit vector.
    ///
    /// Every bit of `value` below `width` is pinned, so this is correct for
    /// bit-vectors wider than 64 bits.  Bits of `value` at or above `width` are
    /// ignored (the literal is read modulo `2^width`, exactly as SMT-LIB reads
    /// an out-of-range numeral).  See [`Self::assert_const`] for the return
    /// value.
    pub fn assert_const_big(&mut self, term: TermId, value: &BigUint, width: u32) -> bool {
        let limbs: SmallVec<[u64; 2]> = value.iter_u64_digits().collect();
        self.assert_const_limbs(term, &limbs, width)
    }

    /// Assert an arbitrary-width constant given as little-endian 64-bit limbs
    /// (`limbs[0]` holds bits 0..63, `limbs[1]` bits 64..127, and so on).
    ///
    /// This is the primitive both [`Self::assert_const`] and
    /// [`Self::assert_const_big`] delegate to; it exists so a caller holding a
    /// `BigInt`-backed literal can pass `value.iter_u64_digits()` directly
    /// without first deciding on a sign-carrying big-integer type.  A limb
    /// beyond the end of the slice reads as `0`, which is the value of that bit
    /// in the little-endian encoding — not a fallback.  See
    /// [`Self::assert_const`] for the return value.
    pub fn assert_const_limbs(&mut self, term: TermId, limbs: &[u64], width: u32) -> bool {
        let bv = self.new_bv(term, width).clone();
        if bv.width != width {
            return false;
        }

        for (i, &bit_var) in bv.bits.iter().enumerate() {
            let bit = limbs.get(i / 64).map_or(0, |limb| (limb >> (i % 64)) & 1);
            if bit == 1 {
                self.sat.add_clause([Lit::pos(bit_var)]);
            } else {
                self.sat.add_clause([Lit::neg(bit_var)]);
            }
        }
        true
    }

    /// Concatenate two bit vectors: result = high ++ low
    /// result[0..low.width-1] = low, result[low.width..low.width+high.width-1] = high
    ///
    /// Returns `false` — encoding nothing — when either operand has not been
    /// bit-blasted, or when `result` already denotes a bit-vector whose width
    /// is not the sum of the operand widths.
    pub fn concat(&mut self, result: TermId, high: TermId, low: TermId) -> bool {
        if let (Some(h), Some(l)) = (
            self.term_to_bv.get(&high).cloned(),
            self.term_to_bv.get(&low).cloned(),
        ) {
            let result_width = h.width + l.width;
            let r = self.new_bv(result, result_width).clone();
            if r.width != result_width {
                return false;
            }

            // Copy low bits
            for i in 0..l.width as usize {
                self.encode_bit_eq(r.bits[i], l.bits[i]);
            }

            // Copy high bits
            for i in 0..h.width as usize {
                self.encode_bit_eq(r.bits[l.width as usize + i], h.bits[i]);
            }
            return true;
        }
        false
    }

    /// Extract a bit range from a bit vector: result = bv\[high:low\]
    /// Extract bits from position `low` to `high` (inclusive)
    ///
    /// Returns `false` — encoding nothing — when `bv` has not been bit-blasted
    /// or the range `[low, high]` does not lie inside it.  An out-of-range
    /// extraction is a malformed term, not a reason to abort the process.
    pub fn extract(&mut self, result: TermId, bv: TermId, high: u32, low: u32) -> bool {
        if let Some(v) = self.term_to_bv.get(&bv).cloned() {
            if high < low || high >= v.width {
                return false;
            }

            let result_width = high - low + 1;
            let r = self.new_bv(result, result_width).clone();
            if r.width != result_width {
                return false;
            }

            for i in 0..result_width {
                let src_idx = (low + i) as usize;
                self.encode_bit_eq(r.bits[i as usize], v.bits[src_idx]);
            }
            return true;
        }
        false
    }

    /// Bit-blast a BV-sorted `ite(cond, then, else)`: a fresh result BV whose
    /// every bit is `cond ? then[i] : else[i]`.
    ///
    /// `cond` is encoded to a single truth variable via [`Self::encode_bool_node`]
    /// (so boolean structure such as `not(c)` is respected); `then` and `else`
    /// must already be bit-blasted to equal-width BVs.
    ///
    /// Returns `false` — encoding nothing — when either branch is missing,
    /// the branches differ in width, or the condition is outside the
    /// fragment `encode_bool_node` models.  It used to return `()` and fail
    /// silently, and the encoder in `oxiz-solver` then went on as if the
    /// `ite` had a circuit: the parent operation's `new_bv` handed the term
    /// a *free* bit-vector, which is the shape behind the width-64 `ite`
    /// self-check failures cargo-formal reported (`#P2b-24`).
    #[must_use]
    pub fn bv_ite(
        &mut self,
        result: TermId,
        cond: TermId,
        then_t: TermId,
        else_t: TermId,
        manager: &oxiz_core::ast::TermManager,
    ) -> bool {
        let Some(sel) = self.encode_bool_node(cond, manager) else {
            return false;
        };
        let (vt, ve) = match (
            self.term_to_bv.get(&then_t).cloned(),
            self.term_to_bv.get(&else_t).cloned(),
        ) {
            (Some(vt), Some(ve)) if vt.width == ve.width => (vt, ve),
            _ => return false,
        };
        let Some(r) = self.result_bits(result, vt.width) else {
            return false;
        };
        for i in 0..vt.width as usize {
            self.encode_mux(r.bits[i], sel, vt.bits[i], ve.bits[i]);
        }
        true
    }

    /// The result bit-vector of a unary/binary operation at `width`, or `None`
    /// when `result` already denotes a bit-vector of a different width.
    fn result_bits(&mut self, result: TermId, width: u32) -> Option<BvVar> {
        let r = self.new_bv(result, width).clone();
        (r.width == width).then_some(r)
    }

    /// Bitwise NOT: result = ~a
    ///
    /// Returns `false` — encoding nothing — when `a` has not been bit-blasted
    /// or `result` already has a different width.
    pub fn bv_not(&mut self, result: TermId, a: TermId) -> bool {
        if let Some(va) = self.term_to_bv.get(&a).cloned() {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            for i in 0..va.width as usize {
                // r[i] = ~a[i]
                self.encode_not(r.bits[i], va.bits[i]);
            }
            return true;
        }
        false
    }

    /// Bitwise AND: result = a & b
    ///
    /// Returns `false` — encoding nothing — when an operand has not been
    /// bit-blasted, the two operands have different widths, or `result`
    /// already has a different width.
    pub fn bv_and(&mut self, result: TermId, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            for i in 0..va.width as usize {
                self.encode_and(r.bits[i], va.bits[i], vb.bits[i]);
            }
            return true;
        }
        false
    }

    /// Bitwise OR: result = a | b
    ///
    /// Returns `false` — encoding nothing — when an operand has not been
    /// bit-blasted, the two operands have different widths, or `result`
    /// already has a different width.
    pub fn bv_or(&mut self, result: TermId, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            for i in 0..va.width as usize {
                self.encode_or(r.bits[i], va.bits[i], vb.bits[i]);
            }
            return true;
        }
        false
    }

    /// Bitwise XOR: result = a ^ b
    ///
    /// Returns `false` — encoding nothing — when an operand has not been
    /// bit-blasted, the two operands have different widths, or `result`
    /// already has a different width.
    pub fn bv_xor(&mut self, result: TermId, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            for i in 0..va.width as usize {
                self.encode_xor(r.bits[i], va.bits[i], vb.bits[i]);
            }
            return true;
        }
        false
    }

    /// Negation (two's complement): result = -a = ~a + 1
    ///
    /// Returns `false` — encoding nothing — when `a` has not been bit-blasted
    /// or `result` already has a different width.
    pub fn bv_neg(&mut self, result: TermId, a: TermId) -> bool {
        if let Some(va) = self.term_to_bv.get(&a).cloned() {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            // First compute ~a
            let mut not_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &va.bits {
                let not_bit = self.sat.new_var();
                self.encode_not(not_bit, bit);
                not_bits.push(not_bit);
            }

            // Then add 1 using a ripple-carry adder
            self.encode_add_const(&r.bits, &not_bits, 1);
            return true;
        }
        false
    }

    /// Addition: result = a + b
    ///
    /// Returns `false` — encoding nothing — when an operand has not been
    /// bit-blasted, the two operands have different widths, or `result`
    /// already has a different width.
    pub fn bv_add(&mut self, result: TermId, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            self.encode_adder(&r.bits, &va.bits, &vb.bits);
            return true;
        }
        false
    }

    /// Subtraction: result = a - b = a + (-b)
    ///
    /// Returns `false` — encoding nothing — when an operand has not been
    /// bit-blasted, the two operands have different widths, or `result`
    /// already has a different width.
    pub fn bv_sub(&mut self, result: TermId, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };

            // Compute -b (two's complement)
            let mut neg_b: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &vb.bits {
                let not_bit = self.sat.new_var();
                self.encode_not(not_bit, bit);
                neg_b.push(not_bit);
            }

            // Create temp variables for -b
            let mut neg_b_with_one: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..va.width {
                neg_b_with_one.push(self.sat.new_var());
            }
            self.encode_add_const(&neg_b_with_one, &neg_b, 1);

            // Add a + (-b)
            self.encode_adder(&r.bits, &va.bits, &neg_b_with_one);
            return true;
        }
        false
    }

    /// Multiplication: result = a * b (using shift-and-add)
    ///
    /// Returns `false` — encoding nothing — when an operand has not been
    /// bit-blasted, the two operands have different widths, or `result`
    /// already has a different width.
    pub fn bv_mul(&mut self, result: TermId, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };
            self.encode_mul(&r.bits, &va.bits, &vb.bits);
            return true;
        }
        false
    }

    /// Left shift by a compile-time constant: result = a << shift_amount.
    ///
    /// Encodes each result bit as a direct wire from the source bit `shift_amount`
    /// positions below, or as a constant-0 for the low `shift_amount` bits.
    /// Used to constant-fold `bvmul(x, 2^k)` without the expensive multiplier.
    ///
    /// Returns `false` — encoding nothing — when `a` has not been bit-blasted,
    /// `a` is not `width` bits wide, or `result` already has a different width.
    pub fn bv_shl_const(&mut self, result: TermId, a: TermId, shift: u32, width: u32) -> bool {
        if let Some(va) = self.term_to_bv.get(&a).cloned() {
            if va.width != width {
                return false;
            }
            let Some(r) = self.result_bits(result, width) else {
                return false;
            };
            for k in 0..width as usize {
                if shift >= width || k < shift as usize {
                    self.sat.add_clause([Lit::neg(r.bits[k])]);
                } else {
                    self.encode_bit_eq(r.bits[k], va.bits[k - shift as usize]);
                }
            }
            return true;
        }
        false
    }

    /// Signed less than: a < b (two's complement)
    ///
    /// Returns `false` — asserting nothing — when either operand has not been
    /// bit-blasted, the two have different widths, or the width is zero (a
    /// zero-width vector has no sign bit).
    #[must_use]
    pub fn assert_slt(&mut self, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let width = va.width as usize;
            if width == 0 {
                return false;
            }

            // For signed comparison:
            // If sign bits differ: a < b iff a is negative (a[n-1] = 1)
            // If sign bits same: compare as unsigned

            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            // diff_sign = sign_a XOR sign_b
            let diff_sign = self.sat.new_var();
            self.encode_xor(diff_sign, sign_a, sign_b);

            // If signs differ, result = sign_a
            // If signs same, result = unsigned comparison of remaining bits

            // Create result variable
            let result = self.sat.new_var();

            // Case 1: diff_sign => result = sign_a
            // diff_sign => (sign_a <=> result)
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::neg(sign_a), Lit::pos(result)]);
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::pos(sign_a), Lit::neg(result)]);

            // Case 2: ~diff_sign => result = ult(a, b)
            // We need to compute unsigned less than and assert it when signs are equal
            let ult_result = self.sat.new_var();
            self.encode_ult_result(&va.bits, &vb.bits, ult_result);

            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::neg(ult_result), Lit::pos(result)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::pos(ult_result), Lit::neg(result)]);

            // Assert that result is true
            self.sat.add_clause([Lit::pos(result)]);
            return true;
        }
        false
    }

    /// Signed less than or equal: a <= b
    ///
    /// Returns `false` — asserting nothing — when either operand has not been
    /// bit-blasted, the two have different widths, or the width is zero (a
    /// zero-width vector has no sign bit).
    #[must_use]
    pub fn assert_sle(&mut self, a: TermId, b: TermId) -> bool {
        if let Some((va, vb)) = self.binop_bits(a, b) {
            let width = va.width as usize;
            if width == 0 {
                return false;
            }

            // a <= b is equivalent to NOT(b < a)
            // Create temporary variables for checking b < a
            let slt_ba = self.sat.new_var();

            // Encode b < a into slt_ba
            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            let diff_sign = self.sat.new_var();
            self.encode_xor(diff_sign, sign_b, sign_a);

            // If signs differ, b < a iff sign_b = 1
            // If signs same, b < a iff ult(b, a)
            let ult_result = self.sat.new_var();
            self.encode_ult_result(&vb.bits, &va.bits, ult_result);

            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::neg(sign_b), Lit::pos(slt_ba)]);
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::pos(sign_b), Lit::neg(slt_ba)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::neg(ult_result), Lit::pos(slt_ba)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::pos(ult_result), Lit::neg(slt_ba)]);

            // Assert NOT(slt_ba) which means a <= b
            self.sat.add_clause([Lit::neg(slt_ba)]);
            return true;
        }
        false
    }

    // ===== Helper encoding functions =====

    /// Encode bit equality: a <=> b
    fn encode_bit_eq(&mut self, a: Var, b: Var) {
        self.sat.add_clause([Lit::neg(a), Lit::pos(b)]);
        self.sat.add_clause([Lit::pos(a), Lit::neg(b)]);
    }

    /// Encode NOT gate: out = ~in
    fn encode_not(&mut self, out: Var, input: Var) {
        self.sat.add_clause([Lit::pos(out), Lit::pos(input)]);
        self.sat.add_clause([Lit::neg(out), Lit::neg(input)]);
    }

    /// Encode AND gate: out = a & b
    fn encode_and(&mut self, out: Var, a: Var, b: Var) {
        // out <=> (a AND b)
        // out => a, out => b, (a AND b) => out
        self.sat.add_clause([Lit::neg(out), Lit::pos(a)]);
        self.sat.add_clause([Lit::neg(out), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
    }

    /// Encode OR gate: out = a | b
    fn encode_or(&mut self, out: Var, a: Var, b: Var) {
        // out <=> (a OR b)
        self.sat
            .add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
        self.sat.add_clause([Lit::pos(out), Lit::neg(a)]);
        self.sat.add_clause([Lit::pos(out), Lit::neg(b)]);
    }

    /// Encode XOR gate: out = a ^ b
    fn encode_xor(&mut self, out: Var, a: Var, b: Var) {
        // out <=> (a XOR b)
        self.sat
            .add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
        self.sat
            .add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
    }

    /// Encode multiplexer: out = sel ? if_true : if_false
    fn encode_mux(&mut self, out: Var, sel: Var, if_true: Var, if_false: Var) {
        // out = (sel AND if_true) OR (~sel AND if_false)
        self.sat
            .add_clause([Lit::neg(sel), Lit::neg(if_true), Lit::pos(out)]);
        self.sat
            .add_clause([Lit::neg(sel), Lit::pos(if_true), Lit::neg(out)]);
        self.sat
            .add_clause([Lit::pos(sel), Lit::neg(if_false), Lit::pos(out)]);
        self.sat
            .add_clause([Lit::pos(sel), Lit::pos(if_false), Lit::neg(out)]);
    }

    /// Encode full adder: (sum, carry_out) = a + b + carry_in
    fn encode_full_adder(&mut self, sum: Var, carry_out: Var, a: Var, b: Var, carry_in: Var) {
        // sum = a XOR b XOR carry_in
        let xor_ab = self.sat.new_var();
        self.encode_xor(xor_ab, a, b);
        self.encode_xor(sum, xor_ab, carry_in);

        // carry_out = (a AND b) OR (carry_in AND (a XOR b))
        let and_ab = self.sat.new_var();
        self.encode_and(and_ab, a, b);

        let and_cin_xor = self.sat.new_var();
        self.encode_and(and_cin_xor, carry_in, xor_ab);

        self.encode_or(carry_out, and_ab, and_cin_xor);
    }

    /// Encode ripple-carry adder: result = a + b
    fn encode_adder(&mut self, result: &[Var], a: &[Var], b: &[Var]) {
        // Discard the carry-out: width-only wrapping addition.
        let _ = self.encode_adder_carry(result, a, b);
    }

    /// Encode a ripple-carry adder `result = a + b` and return the final
    /// carry-out variable (true iff the unsigned sum overflows `width` bits).
    ///
    /// Callers that must forbid wrap-around (e.g. the division/remainder
    /// equation `a = q*b + r`) constrain the returned carry-out to 0.
    fn encode_adder_carry(&mut self, result: &[Var], a: &[Var], b: &[Var]) -> Var {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());

        let width = result.len();
        let mut carry = self.sat.new_var();
        self.sat.add_clause([Lit::neg(carry)]); // Initial carry = 0

        for i in 0..width {
            let next_carry = self.sat.new_var();
            self.encode_full_adder(result[i], next_carry, a[i], b[i], carry);
            carry = next_carry;
        }

        carry
    }

    /// Encode addition with constant: result = a + const
    ///
    /// `constant` is a `u64`, so every bit of it at index 64 and above is `0`;
    /// [`const_bit_of`] says so totally, for any `result.len()` the SMT-LIB
    /// parser can produce (widths up to 65536).
    fn encode_add_const(&mut self, result: &[Var], a: &[Var], constant: u64) {
        assert_eq!(result.len(), a.len());

        let width = result.len();
        let mut carry = self.sat.new_var();
        self.sat.add_clause([Lit::neg(carry)]); // Initial carry = 0

        for i in 0..width {
            let const_bit = const_bit_of(constant, i);
            let next_carry = self.sat.new_var(); // Overflow carry ignored for last iteration

            if const_bit {
                // Half adder with constant 1
                let one = self.sat.new_var();
                self.sat.add_clause([Lit::pos(one)]);
                self.encode_full_adder(result[i], next_carry, a[i], one, carry);
            } else {
                // Half adder with constant 0
                let zero = self.sat.new_var();
                self.sat.add_clause([Lit::neg(zero)]);
                self.encode_full_adder(result[i], next_carry, a[i], zero, carry);
            }

            carry = next_carry;
        }
    }

    /// Encode unsigned less than and store result in a variable
    /// Encode unsigned less-than: result ⇔ (a < b)
    /// Uses LSB-to-MSB comparison: higher bits override lower bits.
    fn encode_ult_result(&mut self, a_bits: &[Var], b_bits: &[Var], result: Var) {
        let width = a_bits.len();
        if width == 0 {
            // Empty bitvectors: 0 < 0 is false
            self.sat.add_clause([Lit::neg(result)]);
            return;
        }

        // Compare from LSB to MSB
        // lt_i represents "a < b considering only bits 0..i"
        // Higher indexed bits (more significant) override lower bits
        // Recurrence: lt_next = (~a[i] & b[i]) | ((a[i] = b[i]) & lt_prev)
        //
        // Meaning:
        // - If a[i] < b[i], then a < b (current bit overrides lower bits)
        // - If a[i] > b[i], then a > b (current bit overrides lower bits)
        // - If a[i] = b[i], result depends on lower bits (lt_prev)

        // Start with LSB (bit 0)
        // lt_0 = ~a[0] & b[0]
        let mut lt_prev = self.sat.new_var();
        self.encode_and_not_a(lt_prev, a_bits[0], b_bits[0]);

        // Process bits from 1 to MSB
        for i in 1..width {
            let ai = a_bits[i];
            let bi = b_bits[i];

            // lt_at_i = ~ai & bi (a < b at this specific bit)
            let lt_at_i = self.sat.new_var();
            self.encode_and_not_a(lt_at_i, ai, bi);

            // eq_i = (ai ⇔ bi) (bits are equal)
            let eq_i = self.sat.new_var();
            self.encode_xnor(eq_i, ai, bi);

            // carry_prev = eq_i & lt_prev (propagate from lower bits)
            let carry_prev = self.sat.new_var();
            self.encode_and(carry_prev, eq_i, lt_prev);

            // lt_next = lt_at_i | carry_prev
            let lt_next = self.sat.new_var();
            self.encode_or(lt_next, lt_at_i, carry_prev);

            lt_prev = lt_next;
        }

        self.encode_bit_eq(result, lt_prev);
    }

    /// Encode out = ~a & b (AND with first input negated)
    fn encode_and_not_a(&mut self, out: Var, a: Var, b: Var) {
        // out ⇔ (~a & b)
        // out → ~a: ~out | ~a
        self.sat.add_clause([Lit::neg(out), Lit::neg(a)]);
        // out → b: ~out | b
        self.sat.add_clause([Lit::neg(out), Lit::pos(b)]);
        // (~a & b) → out: a | ~b | out
        self.sat
            .add_clause([Lit::pos(a), Lit::neg(b), Lit::pos(out)]);
    }

    /// Encode out = (a ⇔ b) (XNOR gate)
    fn encode_xnor(&mut self, out: Var, a: Var, b: Var) {
        // out ⇔ (a ⇔ b)
        // out is true when a = b
        // Clauses:
        // ~out | ~a | b    (out & a → b)
        // ~out | a | ~b    (out & ~a → ~b)
        // out | ~a | ~b    (~out → a ≠ b, i.e., ~a & ~b → out, or a | b → ~out)
        // out | a | b      (~out → a ≠ b, i.e., a & b → out, or ~a | ~b → ~out)
        self.sat
            .add_clause([Lit::neg(out), Lit::neg(a), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::neg(out), Lit::pos(a), Lit::neg(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::pos(a), Lit::pos(b)]);
    }

    // ===== Additional helper encoding functions =====

    /// Encode: out = 1 iff all bits in the list are 0
    fn encode_all_zero(&mut self, out: Var, bits: &[Var]) {
        if bits.is_empty() {
            self.sat.add_clause([Lit::pos(out)]);
            return;
        }

        // out = AND(~bits[i] for all i)
        // out => ~bits[i] for all i
        for &bit in bits {
            self.sat.add_clause([Lit::neg(out), Lit::neg(bit)]);
        }

        // (~bits[0] AND ... AND ~bits[n-1]) => out
        let mut clause: SmallVec<[Lit; 32]> = SmallVec::new();
        clause.push(Lit::pos(out));
        for &bit in bits {
            clause.push(Lit::pos(bit));
        }
        self.sat.add_clause(clause);
    }

    /// Encode two's complement negation: result = -a
    fn encode_two_complement(&mut self, result: &[Var], a: &[Var]) {
        assert_eq!(result.len(), a.len());

        // ~a
        let mut not_a: SmallVec<[Var; 32]> = SmallVec::new();
        for &bit in a {
            let not_bit = self.sat.new_var();
            self.encode_not(not_bit, bit);
            not_a.push(not_bit);
        }

        // ~a + 1
        self.encode_add_const(result, &not_a, 1);
    }

    /// Encode multiplication using symmetric schoolbook method: result = a * b
    /// This encoding is symmetric with respect to a and b, allowing solving for either operand.
    /// Uses Wallace tree-style carry propagation with proper column tracking.
    fn encode_mul(&mut self, result: &[Var], a: &[Var], b: &[Var]) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());

        let width = result.len();

        // Create partial products: columns[k] contains all bits that contribute to result[k]
        // Initially, columns[k] = { a[i] AND b[j] | i + j = k }
        let mut columns: Vec<Vec<Var>> = vec![Vec::new(); width];

        for (i, &a_bit) in a.iter().enumerate().take(width) {
            for (j, &b_bit) in b.iter().enumerate().take(width) {
                let sum_pos = i + j;
                if sum_pos < width {
                    let pp = self.sat.new_var();
                    self.encode_and(pp, a_bit, b_bit);
                    columns[sum_pos].push(pp);
                }
            }
        }

        // Use carry-save reduction to reduce each column to at most 2 bits
        // Then do a final ripple-carry addition
        self.reduce_columns_and_add(result, &mut columns);
    }

    /// Reduce columns using 3:2 compressors until each column has at most 2 bits,
    /// then use a final ripple-carry adder to produce the result.
    fn reduce_columns_and_add(&mut self, result: &[Var], columns: &mut Vec<Vec<Var>>) {
        let width = columns.len();

        // Repeatedly reduce columns using 3:2 compressors
        // Each full adder takes 3 bits from column k and produces:
        //   - 1 sum bit in column k
        //   - 1 carry bit in column k+1
        loop {
            let max_height = columns.iter().map(|c| c.len()).max().unwrap_or(0);
            if max_height <= 2 {
                break;
            }

            let mut new_columns: Vec<Vec<Var>> = vec![Vec::new(); width];

            for k in 0..width {
                let bits = &columns[k];
                let mut i = 0;

                while i + 2 < bits.len() {
                    // Full adder: sum stays in column k, carry goes to column k+1
                    let sum = self.sat.new_var();
                    let carry = self.sat.new_var();
                    self.encode_full_adder_bit(sum, carry, bits[i], bits[i + 1], bits[i + 2]);
                    new_columns[k].push(sum);
                    if k + 1 < width {
                        new_columns[k + 1].push(carry);
                    }
                    i += 3;
                }

                // Pass through remaining bits (0, 1, or 2)
                for &bit in &bits[i..] {
                    new_columns[k].push(bit);
                }
            }

            *columns = new_columns;
        }

        // Now each column has at most 2 bits
        // Create two operands for final addition
        let mut operand_a: SmallVec<[Var; 32]> = SmallVec::new();
        let mut operand_b: SmallVec<[Var; 32]> = SmallVec::new();

        for column in columns.iter().take(width) {
            match column.len() {
                0 => {
                    let zero = self.sat.new_var();
                    self.sat.add_clause([Lit::neg(zero)]);
                    operand_a.push(zero);
                    let zero2 = self.sat.new_var();
                    self.sat.add_clause([Lit::neg(zero2)]);
                    operand_b.push(zero2);
                }
                1 => {
                    operand_a.push(column[0]);
                    let zero = self.sat.new_var();
                    self.sat.add_clause([Lit::neg(zero)]);
                    operand_b.push(zero);
                }
                2 => {
                    operand_a.push(column[0]);
                    operand_b.push(column[1]);
                }
                _ => unreachable!("Column should have at most 2 bits after reduction"),
            }
        }

        // Final ripple-carry addition
        self.encode_adder(result, &operand_a, &operand_b);
    }

    /// Full adder for single bits: sum = a XOR b XOR cin, cout = (a AND b) OR (cin AND (a XOR b))
    fn encode_full_adder_bit(&mut self, sum: Var, cout: Var, a: Var, b: Var, cin: Var) {
        // a XOR b
        let a_xor_b = self.sat.new_var();
        self.encode_xor(a_xor_b, a, b);

        // sum = a_xor_b XOR cin
        self.encode_xor(sum, a_xor_b, cin);

        // a AND b
        let a_and_b = self.sat.new_var();
        self.encode_and(a_and_b, a, b);

        // cin AND (a XOR b)
        let cin_and_axorb = self.sat.new_var();
        self.encode_and(cin_and_axorb, cin, a_xor_b);

        // cout = (a AND b) OR (cin AND (a XOR b))
        self.encode_or(cout, a_and_b, cin_and_axorb);
    }

    /// Encode full multiplication: result = a * b with double-width result
    /// result has length 2*width, a and b have length width
    /// result[0..width-1] = low bits, result[width..2*width-1] = high bits
    /// Uses Wallace tree-style carry propagation with proper column tracking.
    fn encode_mul_full(&mut self, result: &[Var], a: &[Var], b: &[Var]) {
        let width = a.len();
        assert_eq!(b.len(), width);
        assert_eq!(result.len(), 2 * width);

        let double_width = 2 * width;

        // Create partial products: columns[k] contains all bits that contribute to result[k]
        let mut columns: Vec<Vec<Var>> = vec![Vec::new(); double_width];

        for (i, &a_bit) in a.iter().enumerate().take(width) {
            for (j, &b_bit) in b.iter().enumerate().take(width) {
                let sum_pos = i + j;
                let pp = self.sat.new_var();
                self.encode_and(pp, a_bit, b_bit);
                columns[sum_pos].push(pp);
            }
        }

        // Use carry-save reduction and final addition
        self.reduce_columns_and_add(result, &mut columns);
    }

    /// Get the value of a bit vector from the model, as a `u64`.
    ///
    /// Returns `None` for bit-vectors wider than 64 bits: a `u64` cannot
    /// represent their value, and `1u64 << i` for `i >= 64` would panic in
    /// debug builds (shift amount >= bit width) or silently wrap to a wrong
    /// bit (release builds mask the shift amount modulo 64) rather than
    /// error out. Callers needing the full-width value should use
    /// [`Self::get_value_big`] instead; callers of this method already
    /// treat `None` as "value unavailable from this solver" and fall back
    /// accordingly (e.g. `oxiz-solver`'s model builder falls back to the
    /// arithmetic theory's value, then to a default of `0`).
    #[must_use]
    pub fn get_value(&self, term: TermId) -> Option<u64> {
        let bv = self.term_to_bv.get(&term)?;
        if bv.bits.len() > u64::BITS as usize {
            return None;
        }

        let mut value = 0u64;
        for (i, &var) in bv.bits.iter().enumerate() {
            if self.read_model_bit(var) {
                value |= 1 << i;
            }
        }
        Some(value)
    }

    /// Get the value of a bit vector from the model as an arbitrary-width
    /// [`BigUint`], correctly supporting widths beyond 64 bits (unlike
    /// [`Self::get_value`]).
    #[must_use]
    pub fn get_value_big(&self, term: TermId) -> Option<BigUint> {
        let bv = self.term_to_bv.get(&term)?;
        let mut value = BigUint::ZERO;
        for (i, &var) in bv.bits.iter().enumerate() {
            if self.read_model_bit(var) {
                value.set_bit(i as u64, true);
            }
        }
        Some(value)
    }

    /// Truth value the model assigns to a Bool-sorted term that was encoded
    /// into this solver by [`Self::encode_bool_node`] — the `ite` selectors and
    /// the comparisons underneath them.
    ///
    /// Returns `None` for a term that was never encoded as a boolean node, so
    /// callers can tell "the model says false" apart from "this solver has no
    /// opinion". Used by `oxiz-solver`'s debug-only model-validity net to
    /// resolve a bit-blasted `ite` to the branch the model actually selected.
    #[must_use]
    pub fn bool_value(&self, term: TermId) -> Option<bool> {
        let &var = self.bool_node.get(&term)?;
        Some(self.read_model_bit(var))
    }

    /// Read a single SAT variable's boolean value from the model.
    ///
    /// Prefers the snapshot captured at the last SAT check: the live trail
    /// has been backtracked to root and would read all-`Undef` (→ 0). Falls
    /// back to the live model only when no snapshot exists (e.g. direct
    /// unit-test usage that reads before any backtrack).
    fn read_model_bit(&self, var: Var) -> bool {
        let idx = var.index();
        if let Some(v) = self.last_sat_model.get(idx)
            && v.is_defined()
        {
            return v.is_true();
        }
        self.sat.model().get(idx).is_some_and(|v| v.is_true())
    }
}

impl Theory for BvSolver {
    fn id(&self) -> TheoryId {
        TheoryId::BV
    }

    fn name(&self) -> &str {
        "BV"
    }

    fn can_handle(&self, _term: TermId) -> bool {
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        self.assertions.push((term, true));
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        self.assertions.push((term, false));
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // `BvSolver::check()` is driven incrementally by the theory manager:
        // assert more clauses, then `check()` again.  Each `check()` runs a full
        // `solve()`, but the embedded SAT solver does NOT reset its persisted
        // search state on entry, so without the cleanup below a single probe can
        // leave two kinds of unsound residue that poison the next probe and turn
        // a genuinely-SATISFIABLE formula into a false `Unsat`:
        //
        //   1. The satisfying *model* itself.  `solve()` returns with the model
        //      on the trail; some assignments (even a branch `Decision`) land at
        //      decision level 0.  A model value chosen arbitrarily for one probe
        //      then contradicts a constant asserted before the next probe.
        //      Fixed by `restore_to_trail_size`, rolling the trail back to the
        //      committed (asserted) prefix captured here.
        //
        //   2. Clauses *learned* during the solve.  `assert_const` / `assert_eq`
        //      install their unit constraints as level-0 trail assignments with
        //      `reason = Decision` and no backing clause, so a clause learned
        //      while such a literal is on the trail implicitly depends on it;
        //      once the trail is rolled back that learned clause is missing a
        //      hypothesis and can spuriously force `Unsat`.  Fixed by
        //      `forget_learned_since`, dropping exactly this probe's learned
        //      clauses (the asserted clauses remain as the sound core).
        let committed_trail = self.sat.trail_size();
        let learned_before = self.sat.learned_clause_count();

        // Resource budget (U-Z12).  This `solve()` is the call that used to run
        // unbounded no matter what `(set-option :timeout N)` or
        // `(set-option :max-conflicts N)` said: it happens inside the enclosing
        // CDCL(T) search's `on_assignment` callback, so every budget poll in
        // that layer is outside it.  `first_solve_allowance` keeps a quarter of
        // the remaining conflict allowance back for the re-verification below;
        // see the `budget` module for why the allowance is a total rather than
        // a per-probe grant.
        let first_allowance = self.first_solve_allowance();
        self.apply_budget_to_embedded(first_allowance);
        let conflicts_before = self.sat.stats().conflicts;
        let mut solve_result = self.sat.solve();
        self.charge_embedded_conflicts(conflicts_before);

        // Defensive re-verification of an `Unsat` verdict.
        //
        // Audit regression (theories-bv): the SAME unsound-learned-clause
        // hazard documented above for *cross-probe* contamination can also
        // corrupt THIS probe's own verdict, within a single `solve()` call:
        // conflict analysis resolves through the bare, clause-less level-0
        // decision literals that `assert_const`/`assert_eq` install (see
        // `Solver::forget_learned_since`'s doc comment), and an internal
        // restart can expose a learned clause that implicitly -- and
        // unsoundly -- depended on one of them. This has been observed to
        // turn a genuinely SATISFIABLE bit-blasted formula (e.g. an
        // inverse `bvudiv` constraint with a free divisor) into a `solve()`
        // call that reports `Unsat` on its FIRST attempt, even though
        // discarding this probe's learned clauses and solving again -- on
        // nothing but the original, honestly-asserted clauses -- finds a
        // model. Clause learning is sound only if every learned clause is
        // logically entailed by the original clauses; discarding learned
        // clauses can therefore only WEAKEN the formula (never strengthen
        // it), so retrying after `forget_learned_since` can never turn a
        // truly UNSAT formula into a false `Sat` -- it can only correct a
        // false `Unsat` back to the true `Sat`, or confirm the `Unsat`.
        if matches!(solve_result, SolverResult::Unsat) {
            self.sat.restore_to_trail_size(committed_trail);
            self.sat.forget_learned_since(learned_before);
            // The re-verification runs on whatever allowance the first solve
            // left.  If that is nothing, this `solve()` stops at its first
            // budget poll and returns `Unknown`, which overwrites the `Unsat`
            // — and that is the intended, honest outcome, not a lost verdict:
            // the whole reason this block exists is that a first-solve `Unsat`
            // may rest on a learned clause that resolved through a clause-less
            // level-0 decision, so an `Unsat` that was never re-verified is
            // exactly the answer this solver refuses to trust.  Reporting
            // `Unknown` costs precision; keeping the unverified `Unsat` would
            // cost soundness.
            let reverify_allowance = self.remaining_conflict_budget();
            self.apply_budget_to_embedded(reverify_allowance);
            let conflicts_before = self.sat.stats().conflicts;
            solve_result = self.sat.solve();
            self.charge_embedded_conflicts(conflicts_before);
        }

        let result = match solve_result {
            SolverResult::Sat => {
                // Snapshot the satisfying assignment BEFORE rolling the trail
                // back — the rollback discards the model, so `get_value` must
                // consult this captured copy to recover real values.
                self.last_sat_model = self.sat.model().to_vec();
                Ok(TheoryResult::Sat)
            }
            SolverResult::Unsat => {
                // Return all constraint-level terms recorded via
                // `record_constraint_term` as the conflict explanation.
                // This is a sound (superset) conflict clause: the UNSAT is
                // caused by the conjunction of all asserted constraints.
                // If no guard terms were recorded (e.g. in unit tests that
                // call the solver directly), fall back to the assertions list.
                let conflict = if !self.assertion_guard_terms.is_empty() {
                    self.collect_conflict_terms()
                } else {
                    // Fallback: use terms from the assertions list, still
                    // together with the pinned atoms (see
                    // `collect_conflict_terms`).
                    let mut conflict: Vec<TermId> =
                        self.assertions.iter().map(|(t, _)| *t).collect();
                    for &term in &self.pinned_terms {
                        if !conflict.contains(&term) {
                            conflict.push(term);
                        }
                    }
                    conflict
                };
                Ok(TheoryResult::Unsat(conflict))
            }
            SolverResult::Unknown => Ok(TheoryResult::Unknown),
        };

        // Discard this probe's search residue (see the two points above) so the
        // next incremental `check()` starts from only the asserted constraints.
        self.sat.restore_to_trail_size(committed_trail);
        self.sat.forget_learned_since(learned_before);

        result
    }

    fn push(&mut self) {
        self.context_stack.push(ContextMark {
            assertions_len: self.assertions.len(),
            guard_terms_len: self.assertion_guard_terms.len(),
            outer_bool_len: self.outer_bool_journal.len(),
            term_to_bv_len: self.term_to_bv_journal.len(),
            ult_cache_len: self.ult_cache_journal.len(),
            eq_cache_len: self.eq_cache_journal.len(),
            bool_node_len: self.bool_node_journal.len(),
            pinned_len: self.pinned_terms.len(),
            opaque_len: self.opaque_leaves.len(),
        });
        self.sat.push();
    }

    fn pop(&mut self) {
        if let Some(mark) = self.context_stack.pop() {
            self.assertions.truncate(mark.assertions_len);
            self.assertion_guard_terms.truncate(mark.guard_terms_len);
            // Undo the outer-boolean links in reverse so a term fixed at
            // several levels is restored to the value of the surviving one.
            while self.outer_bool_journal.len() > mark.outer_bool_len {
                if let Some((term, previous)) = self.outer_bool_journal.pop() {
                    match previous {
                        Some(value) => self.outer_bool.insert(term, value),
                        None => self.outer_bool.remove(&term),
                    };
                }
            }
            // Retract every circuit node created above this mark, *before*
            // `sat.pop()` runs: that call deletes the clauses which define and
            // pin those nodes, so a surviving cache entry would hand the next
            // `check()` an unconstrained bit-vector and the encoder's
            // idempotence guard would never rebuild it (U-Z10).  Rebuilding can
            // only add constraints relative to the broken behaviour, so this
            // can turn a wrong `sat` into `unsat`, never a true `sat` into
            // `unsat`.
            while self.term_to_bv_journal.len() > mark.term_to_bv_len {
                if let Some(term) = self.term_to_bv_journal.pop() {
                    self.term_to_bv.remove(&term);
                }
            }
            while self.ult_cache_journal.len() > mark.ult_cache_len {
                if let Some(key) = self.ult_cache_journal.pop() {
                    self.ult_cache.remove(&key);
                }
            }
            while self.eq_cache_journal.len() > mark.eq_cache_len {
                if let Some(key) = self.eq_cache_journal.pop() {
                    self.eq_cache.remove(&key);
                }
            }
            while self.bool_node_journal.len() > mark.bool_node_len {
                if let Some(term) = self.bool_node_journal.pop() {
                    self.bool_node.remove(&term);
                }
            }
            // The pins installed inside this scope go with the unit clauses
            // `sat.pop()` retracts; a hypothesis that is no longer asserted
            // must not be blamed by a later conflict.
            while self.pinned_terms.len() > mark.pinned_len {
                if let Some(term) = self.pinned_terms.pop() {
                    self.pinned_set.remove(&term);
                }
            }
            // An opaque leaf's record goes with its circuit (same mark).
            while self.opaque_leaves.len() > mark.opaque_len {
                if let Some(term) = self.opaque_leaves.pop() {
                    self.opaque_leaf_set.remove(&term);
                }
            }
            self.sat.pop();
        }
    }

    fn reset(&mut self) {
        // NOTE: the three budget fields (`budget_max_conflicts`,
        // `budget_deadline`, `conflicts_spent`) are deliberately NOT cleared
        // here, and this block's "clear everything" shape is exactly why the
        // omission needs saying out loud.
        //
        // `sat.reset()` on the next line zeroes the embedded solver's
        // `SolverStats`, and `oxiz-solver`'s `Solver::rebase_theory_state`
        // calls this method once per `(check-sat)` *and* again on every repair
        // round inside one.  A budget that lived in those statistics would
        // therefore re-arm in full several times per check, granting the
        // bit-blaster many times the conflicts the caller asked for.
        // `conflicts_spent` is the running total that survives, so the
        // allowance stays a total across the whole check; only
        // `BvSolver::set_budget` re-arms it.
        self.sat.reset();
        self.term_to_bv.clear();
        self.assertions.clear();
        self.context_stack.clear();
        self.ult_cache.clear();
        self.eq_cache.clear();
        self.shared_equalities.clear();
        self.equality_notifications.clear();
        self.assertion_guard_terms.clear();
        self.last_sat_model.clear();
        self.bool_node.clear();
        self.outer_bool.clear();
        self.outer_bool_journal.clear();
        self.term_to_bv_journal.clear();
        self.ult_cache_journal.clear();
        self.eq_cache_journal.clear();
        self.bool_node_journal.clear();
        self.pinned_terms.clear();
        self.pinned_set.clear();
        self.opaque_leaves.clear();
        self.opaque_leaf_set.clear();
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        // Read the SAT model and construct bit-vector value assignments.
        // For each BV variable term, read its bit values from the SAT model
        // and group terms by their concrete value. For each group, the
        // representative (first term) serves as the "value term", and all
        // other terms in the group map to that representative.
        //
        // Additionally, each term maps to itself as a self-assignment to
        // record its participation in the model.
        //
        // Keyed by `BigUint` rather than `u64`, for the reason
        // [`Self::extract_model_equalities`] already documents: bit-vectors
        // wider than 64 bits are fully supported by the bit-blaster, so a `u64`
        // key needs `1u64 << i` for `i >= 64`, which panics in debug builds and
        // in release ones ORs bit `i` into bit `i % 64` — silently folding two
        // *different* wide values onto one key. Since equal keys are what make
        // two terms share a representative here, such a collision would report
        // unequal terms as having the same value. Same convention, same reason,
        // in both functions.
        let model = self.sat.model();
        let mut value_to_terms: FxHashMap<(BigUint, u32), Vec<TermId>> = FxHashMap::default();

        for (&term, bv_var) in &self.term_to_bv {
            let mut value = BigUint::ZERO;
            for (i, &var) in bv_var.bits.iter().enumerate() {
                if model.get(var.index()).is_some_and(|v| v.is_true()) {
                    value.set_bit(i as u64, true);
                }
            }
            // Key by (value, width) so terms of different widths stay separate
            value_to_terms
                .entry((value, bv_var.width))
                .or_default()
                .push(term);
        }

        let mut assignments = Vec::new();
        for terms in value_to_terms.values() {
            if terms.is_empty() {
                continue;
            }
            // The first term acts as the representative "value term" for this group.
            let representative = terms[0];
            for &term in terms {
                assignments.push((term, representative));
            }
        }
        assignments
    }
}

impl BvSolver {
    /// Extract equalities from the current BV model.
    ///
    /// BV is a finite-domain theory, so we use model-based combination:
    /// if two distinct BV terms evaluate to the same bit-vector value in
    /// the current SAT model, we derive an equality between them.
    fn extract_model_equalities(&mut self) {
        // Collect (term, value) pairs from the current model.
        //
        // Keyed by `BigUint` rather than `u64`: bit-vectors wider than 64
        // bits are fully supported by the bit-blaster (each bit is just
        // another SAT variable), so a `u64` key would require `1u64 << i`
        // for `i >= 64`, which panics in debug builds and silently wraps
        // (masking the shift amount mod 64, corrupting the computed value
        // and potentially deriving a bogus equality) in release builds.
        let model = self.sat.model();
        let mut value_map: FxHashMap<BigUint, Vec<TermId>> = FxHashMap::default();

        for (&term, bv_var) in &self.term_to_bv {
            let mut value = BigUint::ZERO;
            for (i, &var) in bv_var.bits.iter().enumerate() {
                if model.get(var.index()).is_some_and(|v| v.is_true()) {
                    value.set_bit(i as u64, true);
                }
            }
            value_map.entry(value).or_default().push(term);
        }

        // For each group of terms with the same value, derive pairwise equalities
        self.shared_equalities.clear();
        for terms in value_map.values() {
            if terms.len() >= 2 {
                // Only propagate the first pair to avoid quadratic blowup
                self.shared_equalities.push(EqualityNotification {
                    lhs: terms[0],
                    rhs: terms[1],
                    reason: None,
                });
            }
        }
    }
}

/// Unit tests for this module.
#[cfg(test)]
mod tests;
