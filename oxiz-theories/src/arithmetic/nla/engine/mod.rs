// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The lemma cascade: a bounded search over the linear relaxation.
//!
//! [`NlaEngine`] is the driver that the passive layers were built for. It owns
//! a [`LiaSolver`] holding the linear relaxation of one goal, and repeatedly
//! asks it for an integer model. Because the relaxation has forgotten what
//! multiplication means, that model may assign a product variable a value
//! unrelated to its factors; the engine's whole job is to close that gap, by
//! strengthening the relaxation until either it becomes infeasible (the goal is
//! refuted) or a model happens to respect every monic (the goal is satisfied).
//!
//! # The node loop
//!
//! One call to [`NlaEngine::node`] runs at most `max_rounds` iterations of:
//!
//! 1. **Budget.** Out of nodes or too deep ⇒ [`NodeOutcome::Unknown`].
//! 2. **Solve.** [`LiaSolver::check_balanced`] answers with an integer model,
//!    a proof of integer-infeasibility, or a resource limit. Infeasible is
//!    [`NodeOutcome::Refuted`]; a resource limit is `Unknown`, never a verdict.
//! 3. **Check the monics.** Every monic is evaluated in exact [`BigInt`]
//!    arithmetic against the model. All consistent ⇒ [`NodeOutcome::Sat`] with
//!    that model as the witness.
//! 4. **Make progress.** The first stage that achieves anything wins and the
//!    loop restarts: interval propagation over the monics, then McCormick /
//!    tangent cuts for violated degree-two monics, then a case split.
//! 5. **Stuck.** No stage could do anything ⇒ `Unknown`.
//!
//! # Why the split is exhaustive
//!
//! A case split is only sound if its cases cover every integer point. Both
//! forms used here do:
//!
//! * The **sign split** on `x` is `x <= -1 ∨ x = 0 ∨ x >= 1`. Over `Z` there is
//!   nothing between `-1` and `0`, or between `0` and `1`, so this is a
//!   tautology — and it is the split that matters, because almost every lemma
//!   about a product is conditioned on the signs of its factors.
//! * The **value split** on `v` at the model value `k` is `v <= k ∨ v >= k+1`,
//!   which is a tautology over `Z` for integral `k` (and `k` is integral, since
//!   it comes from an integer-feasible model).
//!
//! A child answering `Sat` wins immediately. `Refuted` requires *every* child
//! to be refuted; if any child answered `Unknown` the parent must too, which is
//! what `saw_unknown` tracks. Getting this backwards — treating an unexplored
//! case as refuted — is the single most dangerous mistake available here, so it
//! is a `bool` threaded through one loop rather than an early return.
//!
//! # Exact products
//!
//! Consistency is checked in [`BigInt`], not in the `Rational64` the LP works
//! in. A degree-three monic over values near `2^31` overflows an `i64` product
//! while remaining a perfectly ordinary model, and a wrapped comparison would
//! answer "consistent" for a model that is nothing of the sort — the one place
//! where an overflow could manufacture a wrong `Sat` rather than merely lose
//! precision.

use super::super::lia::LiaSolver;
use super::super::simplex::{LinExpr, VarId};
use super::int_root;
use super::lemmas::{self, Box2, Lemma, LemmaScope};
use super::linearize::{LinAtom, LinAtomKind, Linearization, Monic};
use super::monomial_bounds::{self, Interval, PropOutcome};
use super::{NlaConfig, checked_add_r64, checked_mul_r64, checked_recip_r64};
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::{One, Zero};

mod branch;

/// Reason tag carried by every constraint the engine asserts.
///
/// The engine does not participate in the SAT solver's conflict analysis: it is
/// invoked on a conjunction of ground assertions and answers about that
/// conjunction as a whole, so a per-assertion antecedent would have no consumer.
/// One marker tag keeps that honest rather than borrowing an input constraint's
/// reason and implying a provenance that was never computed.
const NLA_REASON: u32 = u32::MAX - 1;

/// What one node of the search established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NodeOutcome {
    /// An integer model was found that respects every monic. The map assigns
    /// every variable of the relaxation, product variables included.
    Sat(FxHashMap<VarId, Rational64>),
    /// The constraints in scope at this node are unsatisfiable over `Z`.
    Refuted,
    /// Neither could be established within the budget. Never a verdict.
    Unknown,
}

/// The nonlinear search over one linearised goal.
///
/// Borrows the [`Linearization`] it was built from for its whole life, so the
/// monic list and the variable mapping are read straight out of it rather than
/// duplicated.
pub(crate) struct NlaEngine<'a> {
    /// The linearised goal: atoms, monics, and the variable/term mapping.
    lin: &'a Linearization,
    /// The linear relaxation, as an integer solver. Every variable of the
    /// linearisation is registered as an integer variable.
    lia: LiaSolver,
    /// `var_map[i]` is the [`LiaSolver`] variable standing for linearisation
    /// variable `i`. The two agree in practice (both allocate from zero, in
    /// order) but the indirection costs nothing and removes the assumption.
    var_map: Vec<VarId>,
    /// Index into `lin.monics` for each product variable, so a variable seen in
    /// a conflict can be traced back to the monic that defines it.
    monic_of_product: FxHashMap<VarId, usize>,
    /// Case splits consumed so far, against [`NlaConfig::max_nodes`].
    nodes_used: usize,
    /// Cut lemmas emitted so far, against [`NlaConfig::max_tangent_cuts`].
    tangent_cuts_used: usize,
    /// Budgets and switches.
    config: &'a NlaConfig,
}

impl<'a> NlaEngine<'a> {
    /// Build an engine for `lin`, asserting its atoms into a fresh solver.
    ///
    /// Returns `None` when an atom cannot be represented — which, being a
    /// *dropped constraint*, would weaken the problem, and a weakened problem
    /// may not be reported `sat`. Rather than track a second incompleteness
    /// flag alongside [`Linearization::incomplete`], construction simply fails
    /// and the caller degrades to `Unknown`.
    pub(crate) fn new(lin: &'a Linearization, config: &'a NlaConfig) -> Option<Self> {
        let mut lia = LiaSolver::with_configs(config.lia_config(), config.simplex_config());

        // Every variable of the relaxation is an integer variable. The engine
        // is a *nonlinear integer* procedure: a Real-sorted variable would make
        // the branch splits below unsound (`x <= -1 ∨ x = 0 ∨ x >= 1` is not a
        // tautology over the reals), so a linearisation carrying one is
        // declined outright rather than handled approximately.
        if lin.int_vars.len() != lin.num_vars as usize {
            return None;
        }
        let mut var_map = Vec::with_capacity(lin.num_vars as usize);
        for _ in 0..lin.num_vars {
            var_map.push(lia.new_var());
        }

        let mut monic_of_product = FxHashMap::default();
        for (index, monic) in lin.monics.iter().enumerate() {
            let product = *var_map.get(monic.product as usize)?;
            monic_of_product.insert(product, index);
        }

        let mut engine = Self {
            lin,
            lia,
            var_map,
            monic_of_product,
            nodes_used: 0,
            tangent_cuts_used: 0,
            config,
        };

        for atom in &lin.atoms {
            engine.assert_atom(atom)?;
        }
        Some(engine)
    }

    /// Run the search from the root.
    pub(crate) fn solve(&mut self) -> NodeOutcome {
        let entry_depth = self.lia.scope_depth();
        let outcome = self.node(0);
        debug_assert_eq!(
            self.lia.scope_depth(),
            entry_depth,
            "the search must leave the solver at its entry scope depth"
        );
        outcome
    }

    // --- assertion ----------------------------------------------------------

    /// Translate a linearisation variable to its solver variable.
    pub(crate) fn solver_var(&self, v: VarId) -> Option<VarId> {
        self.var_map.get(v as usize).copied()
    }

    /// Rewrite a [`LinExpr`] over linearisation variables into one over solver
    /// variables. `None` when a variable is out of range, which can only happen
    /// if a lemma was built over a variable the linearisation never allocated.
    fn map_expr(&self, expr: &LinExpr) -> Option<LinExpr> {
        let mut out = LinExpr::constant(expr.constant);
        for (v, c) in &expr.terms {
            out.add_term(self.solver_var(*v)?, *c);
        }
        Some(out)
    }

    /// Assert one linear atom into the solver at the current scope.
    fn assert_atom(&mut self, atom: &LinAtom) -> Option<()> {
        let expr = self.map_expr(&atom.expr)?;
        // A single-variable atom is a *bound*, and asserting it as one matters:
        // `add_le` installs a slack row, which constrains the LP identically
        // but leaves the variable itself looking unbounded to every consumer
        // that reads bounds. Interval propagation over the monics reads bounds,
        // so `x * x = 2` asserted as a row gives the propagator nothing at all
        // to work with and the goal becomes unrefutable.
        if let Some(()) = self.assert_as_bound(&expr, atom.kind) {
            return Some(());
        }
        match atom.kind {
            LinAtomKind::Le => self.lia.add_le(expr, NLA_REASON),
            LinAtomKind::Ge => self.lia.add_ge(expr, NLA_REASON),
            LinAtomKind::Eq => self.lia.add_eq(expr, NLA_REASON),
            // Over the integers every variable of this relaxation is integral,
            // so `e < 0` is `e <= -1`; `linearize` already tightens most strict
            // atoms this way, and doing it again here covers the ones whose
            // tightening overflowed. `add_le` on the tightened form is a
            // *stronger* assertion than the strict one only in the sense that
            // it excludes non-integer points, which no model of this problem
            // has.
            LinAtomKind::Lt => {
                let tightened = branch::shift_constant(&expr, Rational64::one())?;
                self.lia.add_le(tightened, NLA_REASON);
            }
            LinAtomKind::Gt => {
                let tightened = branch::shift_constant(&expr, -Rational64::one())?;
                self.lia.add_ge(tightened, NLA_REASON);
            }
        }
        Some(())
    }

    /// Assert `c*v + k ⋈ 0` as a bound on `v`, when the expression really is a
    /// single variable. Returns `None` when it is not, leaving the caller to
    /// assert it as a general constraint.
    ///
    /// The bound is `v ⋈ -k/c`, with the relation flipped when `c` is negative.
    /// Both the division and the strict-to-integer tightening are checked; an
    /// unrepresentable result falls back to `None` and so to the row form,
    /// which is weaker for propagation but never wrong.
    fn assert_as_bound(&mut self, expr: &LinExpr, kind: LinAtomKind) -> Option<()> {
        let [(v, c)] = expr.terms.as_slice() else {
            return None;
        };
        let (v, c) = (*v, *c);
        if c.is_zero() {
            return None;
        }
        // value = -constant / c
        let value = checked_mul_r64(
            super::checked_neg_r64(expr.constant)?,
            checked_recip_r64(c)?,
        )?;
        let negative = c < Rational64::zero();

        // `Le`/`Ge` flip when dividing by a negative coefficient; `Eq` does not.
        let effective = match (kind, negative) {
            (LinAtomKind::Eq, _) => LinAtomKind::Eq,
            (LinAtomKind::Le, false) | (LinAtomKind::Ge, true) => LinAtomKind::Le,
            (LinAtomKind::Ge, false) | (LinAtomKind::Le, true) => LinAtomKind::Ge,
            (LinAtomKind::Lt, false) | (LinAtomKind::Gt, true) => LinAtomKind::Lt,
            (LinAtomKind::Gt, false) | (LinAtomKind::Lt, true) => LinAtomKind::Gt,
        };

        match effective {
            LinAtomKind::Eq => {
                self.lia.tighten_lower(v, value, NLA_REASON);
                self.lia.tighten_upper(v, value, NLA_REASON);
            }
            LinAtomKind::Le => {
                self.lia.tighten_upper(v, value, NLA_REASON);
            }
            LinAtomKind::Ge => {
                self.lia.tighten_lower(v, value, NLA_REASON);
            }
            // Every variable here is integral, so `v < r` is `v <= ceil(r) - 1`
            // and `v > r` is `v >= floor(r) + 1`.
            LinAtomKind::Lt => {
                let tightened = checked_add_r64(value.ceil(), -Rational64::one())?;
                self.lia.tighten_upper(v, tightened, NLA_REASON);
            }
            LinAtomKind::Gt => {
                let tightened = checked_add_r64(value.floor(), Rational64::one())?;
                self.lia.tighten_lower(v, tightened, NLA_REASON);
            }
        }
        Some(())
    }

    /// Assert a lemma's atoms. A lemma that cannot be represented is skipped
    /// entirely — including any atoms already asserted from it, which is
    /// harmless because every atom of a lemma is individually valid in the
    /// scope it is asserted in; a partially asserted lemma is simply a weaker
    /// lemma.
    ///
    /// Returns whether anything was asserted.
    fn assert_lemma(&mut self, lemma: &Lemma) -> bool {
        let mut asserted = false;
        for atom in &lemma.atoms {
            if self.assert_atom(atom).is_some() {
                asserted = true;
            }
        }
        asserted
    }

    // --- the node loop ------------------------------------------------------

    /// Explore the constraint set currently in scope.
    ///
    /// `depth` counts case splits above this node; the caller has already
    /// pushed the scope holding this node's branch constraints, and is
    /// responsible for popping it.
    fn node(&mut self, depth: usize) -> NodeOutcome {
        for _ in 0..self.config.max_rounds {
            if depth >= self.config.max_depth || self.nodes_used >= self.config.max_nodes {
                return NodeOutcome::Unknown;
            }

            let model = match self.lia.check_balanced() {
                // Proven integer-infeasible: an LP infeasibility closure over
                // the input atoms, the monic-derived bounds, valid lemmas and
                // the branch atoms in scope. Every one of those is a
                // consequence over `Z` of the input, so the refutation is too.
                Ok(None) => return NodeOutcome::Refuted,
                // A resource limit inside the LIA solver. Not a verdict.
                Err(_) => return NodeOutcome::Unknown,
                Ok(Some(model)) => model,
            };

            let violated = self.violated_monics(&model);
            if violated.is_empty() {
                return NodeOutcome::Sat(model);
            }

            // (d1) Interval propagation: cheapest, and the only stage that can
            // refute without a split.
            match self.propagate_bounds_to_fixpoint() {
                // Bounds moved; re-solve under them.
                PropOutcome::Progress => continue,
                // An empty interval means the bounds in scope are already
                // contradictory. The tightened bounds are in the solver, so the
                // next `check_balanced` sees the contradiction and refutes;
                // answering `Refuted` here directly would bypass the LP, which
                // is the component that actually holds the proof.
                PropOutcome::Conflict => continue,
                // Nothing more to derive, or a coefficient that could not be
                // represented. Either way the next stage gets its turn.
                PropOutcome::Fixpoint | PropOutcome::Overflow => {}
            }

            // (d2) Linear relaxations of the violated products.
            if self.emit_cuts(&violated, &model) {
                continue;
            }

            // (d3) Case split.
            return self.branch(&violated, &model, depth);
        }
        NodeOutcome::Unknown
    }

    // --- monic consistency --------------------------------------------------

    /// The monics the model violates, as indices into `lin.monics`.
    ///
    /// A monic whose model values cannot be read at all counts as violated: an
    /// unreadable value is not a demonstration of consistency, and treating it
    /// as one is how a `Sat` gets manufactured.
    fn violated_monics(&self, model: &FxHashMap<VarId, Rational64>) -> Vec<usize> {
        let mut out = Vec::new();
        for (index, monic) in self.lin.monics.iter().enumerate() {
            if !self.monic_holds(monic, model).unwrap_or(false) {
                out.push(index);
            }
        }
        out
    }

    /// Whether `model` satisfies `product = prod factor^exponent`, computed in
    /// exact [`BigInt`] arithmetic. `None` when a value is missing or is not an
    /// integer.
    fn monic_holds(&self, monic: &Monic, model: &FxHashMap<VarId, Rational64>) -> Option<bool> {
        let product = self.model_int(monic.product, model)?;
        let mut expected = BigInt::from(1);
        for (factor, exponent) in &monic.factors {
            let base = self.model_int(*factor, model)?;
            for _ in 0..*exponent {
                expected *= &base;
            }
        }
        Some(product == expected)
    }

    /// The model value of a linearisation variable as an exact integer.
    fn model_int(&self, v: VarId, model: &FxHashMap<VarId, Rational64>) -> Option<BigInt> {
        let solver_var = self.solver_var(v)?;
        let value = model.get(&solver_var)?;
        if !value.is_integer() {
            return None;
        }
        Some(BigInt::from(value.to_integer()))
    }

    // --- (d1) interval propagation ------------------------------------------

    /// Run monic interval propagation until nothing more moves, and assert
    /// every derived bound that improves on what the solver already holds.
    ///
    /// The step budget is `2 * |monics| + 8`: enough for a couple of sweeps
    /// over every monic plus slack for a short propagation chain, and bounded
    /// so that a cycle of ever-tinier tightenings cannot spin forever.
    ///
    /// Returns [`PropOutcome::Progress`] when a bound was asserted,
    /// [`PropOutcome::Conflict`] when an interval came out empty,
    /// [`PropOutcome::Overflow`] when a coefficient could not be represented
    /// and nothing was asserted, and [`PropOutcome::Fixpoint`] otherwise.
    fn propagate_bounds_to_fixpoint(&mut self) -> PropOutcome {
        // Make row-implied bounds visible first. A constraint such as
        // `x + y <= 10` bounds `x` above without ever writing a bound on `x`,
        // and the interval layer reads bounds, not rows — so without this the
        // box is far weaker than what the LP already knows, and a McCormick
        // envelope over it could not be built at all.
        self.lia.propagate_lp_bounds();

        let mut intervals = self.read_intervals();
        let budget = self.lin.monics.len().saturating_mul(2).saturating_add(8);
        let mut conflict = false;
        let mut overflowed = false;
        let mut moved = false;

        'sweeps: for _ in 0..budget {
            let mut sweep_moved = false;
            for monic in &self.lin.monics {
                let Some(mut product) = intervals.get(&monic.product).cloned() else {
                    continue;
                };
                let mut factors: Vec<(Interval, u32)> = Vec::with_capacity(monic.factors.len());
                for (factor, exponent) in &monic.factors {
                    let iv = intervals.get(factor).cloned().unwrap_or_default();
                    factors.push((iv, *exponent));
                }

                let mut outcome = monomial_bounds::propagate_monic(&mut product, &mut factors);

                // `propagate_monic` declines to invert a factor raised to a
                // power. For the single-power shape `v = x^e` that inversion is
                // exact over `Z`, and it is the only thing that ever bounds `x`
                // in a goal like `x * x = 2`; see `int_root`.
                if let [(_, e)] = monic.factors.as_slice()
                    && let Some(slot) = factors.first_mut()
                {
                    let derived = int_root::power_backward(&product, *e);
                    match slot.0.tighten(&derived) {
                        PropOutcome::Conflict => outcome = PropOutcome::Conflict,
                        PropOutcome::Progress if outcome != PropOutcome::Conflict => {
                            outcome = PropOutcome::Progress;
                        }
                        _ => {}
                    }
                }

                // `propagate_monic` documents that `Overflow` means "partially
                // applied, then gave up" — the arguments may already hold
                // bounds derived before the failure. Those bounds came from
                // exact computations and are sound, so they are kept.
                intervals.insert(monic.product, product);
                for ((factor, _), (iv, _)) in monic.factors.iter().zip(factors) {
                    intervals.insert(*factor, iv);
                }

                match outcome {
                    PropOutcome::Progress => sweep_moved = true,
                    PropOutcome::Conflict => {
                        conflict = true;
                        break 'sweeps;
                    }
                    PropOutcome::Overflow => overflowed = true,
                    PropOutcome::Fixpoint => {}
                }
            }
            if !sweep_moved {
                break;
            }
            moved = true;
        }

        // Write back whatever survived. A conflict is written back too: the
        // crossing pair of bounds is exactly what makes the next LP check
        // refute, and that check is where the refutation is actually proved.
        let asserted = self.assert_intervals(&intervals);

        if conflict {
            PropOutcome::Conflict
        } else if asserted {
            PropOutcome::Progress
        } else if overflowed && !moved {
            PropOutcome::Overflow
        } else {
            PropOutcome::Fixpoint
        }
    }

    /// Snapshot the solver's current box as intervals over linearisation
    /// variables.
    fn read_intervals(&self) -> FxHashMap<VarId, Interval> {
        let mut out = FxHashMap::default();
        for v in 0..self.lin.num_vars {
            let Some(solver_var) = self.solver_var(v) else {
                continue;
            };
            let mut iv = Interval::unbounded();
            if let Some((value, reason)) = self.lia.bound_lower(solver_var) {
                iv.lo = Some(monomial_bounds::Bound::tagged(value, reason));
            }
            if let Some((value, reason)) = self.lia.bound_upper(solver_var) {
                iv.hi = Some(monomial_bounds::Bound::tagged(value, reason));
            }
            out.insert(v, iv);
        }
        out
    }

    /// Assert every interval endpoint that is strictly tighter than what the
    /// solver already holds. Returns whether anything was asserted.
    ///
    /// Endpoints are written as *bounds*, not as the equivalent single-variable
    /// constraints. The two are interchangeable for feasibility and not at all
    /// interchangeable here: `add_ge` on `x - 3` installs a slack row, which
    /// constrains the LP correctly but leaves `x` looking unbounded to the next
    /// round's [`Self::read_intervals`]. Propagation would then re-derive the
    /// same bound forever instead of reaching a fixpoint.
    ///
    /// [`LiaSolver::tighten_lower`] and [`LiaSolver::tighten_upper`] write only
    /// strict improvements, which is what keeps a sequence of raw bound writes
    /// equivalent to intersecting them.
    fn assert_intervals(&mut self, intervals: &FxHashMap<VarId, Interval>) -> bool {
        let mut asserted = false;
        // Deterministic order: iteration over a hash map is not stable across
        // runs, and the assertion order changes the LP's pivot sequence and so
        // the search's shape. A budgeted search whose shape varies would give
        // different verdicts for the same input.
        for v in 0..self.lin.num_vars {
            let Some(iv) = intervals.get(&v) else {
                continue;
            };
            let Some(solver_var) = self.solver_var(v) else {
                continue;
            };
            if let Some(lo) = &iv.lo
                && self.lia.tighten_lower(solver_var, lo.value, NLA_REASON)
            {
                asserted = true;
            }
            if let Some(hi) = &iv.hi
                && self.lia.tighten_upper(solver_var, hi.value, NLA_REASON)
            {
                asserted = true;
            }
        }
        asserted
    }

    // --- (d2) cuts ----------------------------------------------------------

    /// Emit linear relaxations of the violated degree-two monics.
    ///
    /// A relaxation needs *both* envelopes to pin a product down, and which
    /// constructor supplies which differs by shape:
    ///
    /// * For a *square* `v = x²`, the tangent `v >= 2a·x - a²` at the model's
    ///   `x` is the lower envelope. It is [`LemmaScope::Global`] — it is
    ///   `(x - a)² >= 0` — so it is valid at every scope and asserted
    ///   unconditionally. On its own it is not enough: it only ever pushes `v`
    ///   *up*, so a model that sets `v` too high is never cut off. The matching
    ///   upper envelope is the secant `(xu - x)(x - xl) >= 0` over the current
    ///   box, which [`lemmas::mccormick`] produces when handed `x` for both
    ///   factors — its repeated-variable merging turns the bilinear envelopes
    ///   into the square's tangent-and-secant pair.
    /// * For a *bilinear* `v = x·y` whose factors are both boxed, the four
    ///   McCormick envelopes of that box.
    ///
    /// Everything derived from a box depends on the bounds in scope, so it is
    /// asserted at the current scope and goes away with it.
    ///
    /// Returns whether anything was asserted.
    fn emit_cuts(&mut self, violated: &[usize], model: &FxHashMap<VarId, Rational64>) -> bool {
        let mut asserted = false;
        for &index in violated {
            if self.tangent_cuts_used >= self.config.max_tangent_cuts {
                break;
            }
            let Some(monic) = self.lin.monics.get(index) else {
                continue;
            };
            if monic.degree() != 2 {
                continue;
            }
            let mut pending: Vec<Lemma> = Vec::new();
            match monic.factors.as_slice() {
                // v = x^2: tangent below, secant above.
                [(x, 2)] => {
                    if let Some(a) = self.model_value(*x, model)
                        && let Some(lemma) = lemmas::square_tangent(monic.product, *x, a)
                    {
                        pending.push(lemma);
                    }
                    if let Some(b) = self.factor_box(*x, *x)
                        && let Some(lemma) =
                            lemmas::mccormick(monic.product, *x, *x, &b, &[NLA_REASON])
                    {
                        pending.push(lemma);
                    }
                }
                // v = x*y: the four envelopes of the box.
                [(x, 1), (y, 1)] => {
                    if let Some(b) = self.factor_box(*x, *y)
                        && let Some(lemma) =
                            lemmas::mccormick(monic.product, *x, *y, &b, &[NLA_REASON])
                    {
                        pending.push(lemma);
                    }
                }
                _ => continue,
            }

            for lemma in &pending {
                debug_assert!(
                    lemma.scope == LemmaScope::Global || !lemma.premises.is_empty(),
                    "a branch-local lemma must record the premises it leans on"
                );
                if self.assert_lemma(lemma) {
                    self.tangent_cuts_used += 1;
                    asserted = true;
                }
            }
        }
        asserted
    }

    /// The model value of a linearisation variable, or `None` when unassigned.
    fn model_value(&self, v: VarId, model: &FxHashMap<VarId, Rational64>) -> Option<Rational64> {
        let solver_var = self.solver_var(v)?;
        model.get(&solver_var).copied()
    }

    /// The box `[xl, xu] × [yl, yu]` the solver currently holds for `x` and
    /// `y`, or `None` when either is unbounded on either side (McCormick needs
    /// a finite box).
    fn factor_box(&self, x: VarId, y: VarId) -> Option<Box2> {
        let sx = self.solver_var(x)?;
        let sy = self.solver_var(y)?;
        Some(Box2 {
            xl: self.lia.bound_lower(sx)?.0,
            xu: self.lia.bound_upper(sx)?.0,
            yl: self.lia.bound_lower(sy)?.0,
            yu: self.lia.bound_upper(sy)?.0,
        })
    }
}

#[cfg(test)]
mod tests;
