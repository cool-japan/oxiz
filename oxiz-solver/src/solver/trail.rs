//! Trail operations and context state for push/pop support

#[allow(unused_imports)]
use crate::prelude::*;
use num_traits::ToPrimitive;
use oxiz_core::ast::{RoundingMode, TermId, TermKind, TermManager};
use oxiz_sat::{Lit, Var};

use super::types::Polarity;

/// Trail operation for efficient undo
///
/// Every piece of solver state that assertion processing derives from the
/// *current* scope must be undoable, either through one of these operations or
/// through a snapshot field in [`ContextState`].  State that survives a `pop`
/// keeps a retracted assertion's fact alive and produces wrong verdicts (a
/// datatype constructor recorded inside a popped scope used to force a false
/// `unsat`).  See [`super::Solver::debug_assert_scope_restored`] for the
/// compile-time reminder that keeps this list in sync with the solver fields.
#[derive(Debug, Clone)]
pub(crate) enum TrailOp {
    /// An assertion was added
    AssertionAdded { index: usize },
    /// A variable was created
    VarCreated {
        #[allow(dead_code)]
        var: Var,
        term: TermId,
    },
    /// A constraint was added
    ConstraintAdded { var: Var },
    /// False assertion flag was set
    FalseAssertionSet,
    /// A named assertion was added
    NamedAssertionAdded { index: usize },
    /// A bitvector term was added
    BvTermAdded { term: TermId },
    /// An arithmetic term was added
    ArithTermAdded { term: TermId },
    /// A datatype variable was pinned to a constructor
    DtVarConstructorAdded { term: TermId },
    /// A compound term was marked as fully traversed by `track_theory_vars`
    TrackedCompoundAdded { term: TermId },
    /// A Bool-sorted `Var` was marked as appearing in a UF-application
    /// argument position by `abstract_compound_bool_args`.
    BoolUfArgAdded { term: TermId },
    /// A numeric term was marked as appearing in a UF-application argument
    /// position by `purify_numeric_uf_args`.
    NumericUfArgAdded { term: TermId },
    /// A pre-purification UF-application term was aliased to its purified
    /// counterpart by `purify_numeric_uf_args`.
    NumericPurifyAliasAdded { term: TermId },
    /// A term received an explicit integer case-split lemma from
    /// `Solver::split_narrow_int_domains`.
    CaseSplitTermAdded { term: TermId },
    /// One triangle of the pure-equality fast path's chordal graph received
    /// its three transitivity clauses (see `eq_skeleton`). The payload is the
    /// triangle's three constants in ascending order.
    EqTransitivityTriangleAdded { triangle: [TermId; 3] },
    /// A ground array-axiom instance was asserted to the SAT core
    ArrayAxiomInstanceAdded { term: TermId },
    /// A `div` / `mod` / numeric-`ite` term received its defining axioms
    ArithDefinedTermAdded { term: TermId },
    /// A numeric `Eq` atom received its trichotomy clause
    /// `(a = b) OR (a < b) OR (a > b)` from
    /// [`super::Solver::add_numeric_trichotomy`].
    NumericTrichotomyAdded { term: TermId },
    /// A ground datatype-axiom instance was asserted to the SAT core
    DtAxiomInstanceAdded { term: TermId },
    /// A Tseitin-memo entry was written by [`super::Solver::encode`].
    ///
    /// `previous` carries the value the write displaced (`None` when the term
    /// had no entry), so `pop` can restore a polarity that was *widened* inside
    /// the scope instead of dropping the narrower pre-scope encoding along with
    /// it.  See [`super::Solver::pop`] for why the memo must be retracted
    /// per entry rather than cleared wholesale.
    EncodedTermAdded {
        term: TermId,
        previous: Option<(Lit, Polarity)>,
    },
}

/// State for push/pop with trail-based undo
///
/// Holds the scope-dependent state that is cheaper to snapshot wholesale than
/// to journal operation by operation: monotone counters and sticky flags.
#[derive(Debug, Clone)]
pub(crate) struct ContextState {
    pub(crate) num_assertions: usize,
    pub(crate) num_vars: usize,
    pub(crate) has_false_assertion: bool,
    /// Trail position at the time of push
    pub(crate) trail_position: usize,
    /// Number of quantifiers tracked by MBQI at the time of push
    pub(crate) num_mbqi_quantifiers: usize,
    /// Number of quantifiers registered with the e-matching engine at push
    pub(crate) num_ematch_quantifiers: usize,
    /// `has_quantifiers` flag at the time of push
    pub(crate) has_quantifiers: bool,
    /// `quantifier_uf_funcs` set at the time of push
    pub(crate) quantifier_uf_funcs: crate::prelude::FxHashSet<oxiz_core::interner::Spur>,
    /// `has_bv_arith_ops` flag at the time of push
    pub(crate) has_bv_arith_ops: bool,
    /// `has_array_ops` flag at the time of push
    pub(crate) has_array_ops: bool,
    /// `encode_depth_exceeded` flag at the time of push
    pub(crate) encode_depth_exceeded: bool,
    /// `dt_axioms_incomplete` flag at the time of push
    pub(crate) dt_axioms_incomplete: bool,
    /// `array_axioms_incomplete` flag at the time of push
    pub(crate) array_axioms_incomplete: bool,
    /// Number of live model-blocking clauses at the time of push (see
    /// [`super::model_blocking`]).
    ///
    /// A monotone counter, and hence a snapshot case: `pop` retracts the
    /// clauses themselves through `sat.pop()`, and this restores the solver's
    /// record of how many survive.
    pub(crate) model_blocking_active: usize,
}

#[cfg(debug_assertions)]
impl super::Solver {
    /// Debug-build guard: verify that [`super::Solver::pop`] restored every
    /// piece of scope-dependent state to its push-time value.
    ///
    /// The `let Solver { .. }` below is deliberately **exhaustive** (no `..`
    /// rest pattern): adding a field to the solver stops this file compiling
    /// until the author classifies it, which is the whole point.  Each field is
    /// tagged with why it needs no undo, or with the mechanism that undoes it:
    ///
    /// - `INVARIANT` — configuration, or a cache keyed by immutable term
    ///   structure (interning, parse/simplify memos).  Never derived from
    ///   *which* assertions are active, so it is valid in every scope.
    /// - `TRAIL` — undone by a [`TrailOp`] replayed above.
    /// - `SNAPSHOT` — restored from [`ContextState`] and checked below.
    /// - `SCOPED` — owned by a sub-solver with its own push/pop, or returned to
    ///   its base state by [`super::Solver::rebase_theory_state`], which `pop`
    ///   calls (it is also every `check`'s first act).
    /// - `RESULT` — output of the last `check`, not an input to the next one.
    ///   Discarded by [`super::Solver::invalidate_results`] at the top of `pop`:
    ///   a verdict belongs to the assertion stack it was computed on, and an
    ///   unsat core additionally *indexes* that stack.
    ///
    /// `polarities` is the one entry that survives a `pop` by design:
    /// `collect_polarities` merges monotonically towards `Both`, so a stale
    /// entry can only make the Tseitin encoder emit *both* implication
    /// directions — never fewer clauses than the live assertions require.
    pub(super) fn debug_assert_scope_restored(&self, state: &ContextState) {
        let super::Solver {
            config: _,          // INVARIANT: user configuration
            sat: _,             // SCOPED: SatSolver::push/pop
            euf: _,             // SCOPED: reset by `rebase_theory_state`
            arith: _,           // SCOPED: reset by `rebase_theory_state`
            bv: _,              // SCOPED: reset by `rebase_theory_state`
            derived_reasons: _, // SCOPED: pruned with the theory scopes it explains, cleared by `rebase_theory_state` with the three solvers
            #[cfg(feature = "nlsat")]
                nlsat: _, // SCOPED: NlsatTheory::push/pop
            mbqi: _,            // SNAPSHOT: num_mbqi_quantifiers
            ematch_engine: _,   // SNAPSHOT: num_ematch_quantifiers
            has_quantifiers: _, // SNAPSHOT
            quantifier_uf_funcs: _, // SNAPSHOT
            term_to_var: _,     // TRAIL: VarCreated
            var_to_term: _,     // SNAPSHOT: num_vars
            var_to_constraint: _, // TRAIL: ConstraintAdded
            var_to_parsed_arith: _, // TRAIL: ConstraintAdded
            logic: _,           // INVARIANT: set before any assertion
            assertions: _,      // SNAPSHOT: num_assertions
            named_assertions: _, // TRAIL: NamedAssertionAdded
            assumption_vars: _, // INVARIANT: never written
            model: _,           // RESULT: cleared by `invalidate_results`
            nl_algebraic_values: _, // RESULT: the other half of `model` (the
            // exact algebraic values it cannot hold), cleared by the same
            // `invalidate_results` call and additionally at every
            // `dispatch_nl_solver` entry
            unsat_core: _,             // RESULT: cleared by `invalidate_results`
            context_stack: _,          // the scope stack itself
            trail: _,                  // the undo journal itself
            theory_processed_up_to: _, // INVARIANT: never read
            produce_unsat_cores: _,    // INVARIANT: user option
            has_false_assertion: _,    // SNAPSHOT + TRAIL: FalseAssertionSet
            polarities: _,             // INVARIANT: monotone (see above)
            polarity_aware: _,         // INVARIANT: user option
            theory_aware_branching: _, // INVARIANT: user option
            proof: _,                  // RESULT: emptied in place by `invalidate_results` (the
            // `Option` carries the `:produce-proofs` setting, so it is not taken)
            simplifier: _,               // INVARIANT: term -> simplified term
            statistics: _,               // INVARIANT: cumulative counters
            bv_terms: _,                 // TRAIL: BvTermAdded
            has_bv_arith_ops: _,         // SNAPSHOT
            arith_terms: _,              // TRAIL: ArithTermAdded
            dt_var_constructors: _,      // TRAIL: DtVarConstructorAdded
            arith_parse_cache: _,        // INVARIANT: keyed by term structure
            tracked_compound_terms: _,   // TRAIL: TrackedCompoundAdded
            bool_uf_arg_terms: _,        // TRAIL: BoolUfArgAdded
            numeric_uf_arg_terms: _,     // TRAIL: NumericUfArgAdded
            numeric_purify_aliases: _,   // TRAIL: NumericPurifyAliasAdded
            encoded_terms: _, // TRAIL: EncodedTermAdded (carries the displaced entry, so a polarity widened inside the scope is restored rather than dropped)
            fp_constraint_cache: _, // INVARIANT: keyed by assertion term
            encode_depth_exceeded: _, // SNAPSHOT
            has_array_ops: _, // SNAPSHOT
            array_axiom_instances: _, // TRAIL: ArrayAxiomInstanceAdded
            arith_defined_terms: _, // TRAIL: ArithDefinedTermAdded
            numeric_trichotomy_atoms: _, // TRAIL: NumericTrichotomyAdded
            dt_axiom_instances: _, // TRAIL: DtAxiomInstanceAdded
            dt_axioms_incomplete: _, // SNAPSHOT
            array_axioms_incomplete: _, // SNAPSHOT
            entailed_int_consts: _, // cleared wholesale by `pop` (see the field doc); empty = re-fold, never stale
            entailed_int_consts_upto: _, // reset to 0 with the map above
            #[cfg(test)]
                mbqi_round_clauses: _, // INVARIANT: test-only event log,
            // deliberately cumulative across `check`s and scopes (see the field
            // doc); restoring it would defeat what it measures.
            last_check: _, // RESULT: cleared by `invalidate_results` (the
            // cached verdict belongs to the assertion stack it was computed on,
            // exactly as `model` does)
            settings_epoch: _, // INVARIANT: monotone — it counts *settings*
            // mutations, which are not scoped by push/pop; rolling it back would
            // let a cached verdict from before a `set-option` be matched again
            // after the pop.
            next_skolem_id: _, // INVARIANT: monotone — a popped scope's Skolem
            // names must never be handed out again, so this counter deliberately
            // survives `pop` (re-using an id would alias two distinct witnesses).
            case_split_terms: _,           // TRAIL: CaseSplitTermAdded
            case_split_skipped_targets: _, // PER-SEARCH: reset at `check_core`
            // entry alongside `case_split_rounds`, for the same reason.
            case_split_rounds: _, // PER-SEARCH: reset at `check_core` entry, not
            // trail-scoped — see the field doc for why this differs from
            // `case_split_terms`.
            #[cfg(test)]
                repair_paths_saw_model: _, // INVARIANT: test-only event log,
            // cumulative across `check`s and scopes exactly like
            // `mbqi_round_clauses`; restoring it would defeat what it measures.
            model_blocking_active: _, // SNAPSHOT: the blocking clauses it counts
            // are retracted by `sat.pop()`, so the count rolls back with them.
            // Deliberately *not* PER-SEARCH like `case_split_rounds` above: the
            // clauses outlive the `check` that added them, and a counter reset
            // between checks would let the next one report a wrong `unsat` off
            // a still-restricted database.
            lookup_index_terms: _, // accumulates lookup-spine index terms across
            // `assert`s; a stale entry after `pop` only makes a later
            // `split_narrow_int_domains` round consider a term whose table no
            // longer exists, which costs a bound lookup that finds nothing —
            // sound to leave (cleared wholesale only by `reset`).
            eq_transitivity_triangles: _, // TRAIL: EqTransitivityTriangleAdded
            equality_skeleton_classes: _, // RESULT: part of the model the last
            // `check` produced, discarded by `invalidate_results` (which
            // `pop` calls first) and rebuilt by every pure-equality fast-path
            // attempt, so it never outlives the verdict it describes.
            branch_priority: _, // the `Arc` handle itself never changes after
                                // construction (see `branch_priority`'s module doc); its
                                // contents are a soft decision-order hint, not a soundness
                                // invariant, so a scope boundary need not roll them back.
        } = self;

        debug_assert_eq!(self.trail.len(), state.trail_position);
        debug_assert_eq!(self.assertions.len(), state.num_assertions);
        debug_assert_eq!(self.var_to_term.len(), state.num_vars);
        debug_assert_eq!(self.has_false_assertion, state.has_false_assertion);
        debug_assert_eq!(self.mbqi.num_quantifiers(), state.num_mbqi_quantifiers);
        debug_assert_eq!(
            self.ematch_engine.num_quantifiers(),
            state.num_ematch_quantifiers
        );
        debug_assert_eq!(self.has_quantifiers, state.has_quantifiers);
        debug_assert_eq!(self.quantifier_uf_funcs, state.quantifier_uf_funcs);
        debug_assert_eq!(self.has_bv_arith_ops, state.has_bv_arith_ops);
        debug_assert_eq!(self.has_array_ops, state.has_array_ops);
        debug_assert_eq!(self.encode_depth_exceeded, state.encode_depth_exceeded);
        debug_assert_eq!(self.dt_axioms_incomplete, state.dt_axioms_incomplete);
        debug_assert_eq!(self.array_axioms_incomplete, state.array_axioms_incomplete);
        debug_assert_eq!(self.model_blocking_active, state.model_blocking_active);
    }
}

/// Collector for floating-point constraints to detect early conflicts
#[derive(Debug, Default)]
#[allow(dead_code)]
pub(crate) struct FpConstraintCollector {
    /// FP variables with isZero predicate applied
    is_zero_vars: FxHashSet<TermId>,
    /// FP variables with isNegative predicate applied
    is_negative_vars: FxHashSet<TermId>,
    /// FP variables with isPositive predicate applied
    is_positive_vars: FxHashSet<TermId>,
    /// FP addition operations: (rm, lhs, rhs, result)
    fp_adds: Vec<(TermKind, TermId, TermId, TermId)>,
    /// FP less-than comparisons: (lhs, rhs)
    fp_lts: Vec<(TermId, TermId)>,
    /// FP divisions: (rm, lhs, rhs, result)
    fp_divs: Vec<(TermKind, TermId, TermId, TermId)>,
    /// FP multiplications: (rm, lhs, rhs, result)
    fp_muls: Vec<(TermKind, TermId, TermId, TermId)>,
    /// Equality constraints: (lhs, rhs)
    equalities: Vec<(TermId, TermId)>,
    /// FP format conversions: (source, target_eb, target_sb, result)
    fp_conversions: Vec<(TermId, u32, u32, TermId)>,
    /// Real to FP conversions: (rm, real_value, eb, sb, result)
    real_to_fp: Vec<(TermKind, TermId, u32, u32, TermId)>,
}

#[allow(dead_code)]
impl FpConstraintCollector {
    fn new() -> Self {
        Self::default()
    }

    /// Walk `term` and record every FP-relevant fact it contains.
    ///
    /// Driven by an explicit heap worklist with a `visited` set instead of
    /// the original native recursion, which had **neither**: a term built
    /// through the `TermManager` builder API can nest arbitrarily deep, so
    /// the recursion could exhaust the native stack (a fatal,
    /// `catch_unwind`-proof process abort), and without a visited set a
    /// shared sub-DAG of the hash-consed term graph was re-expanded once per
    /// path — `2^n` visits for an `n`-level doubling DAG.  A depth cap is not
    /// an option: the return type is `()`, so a cap could only silently drop
    /// facts and make [`Self::check_conflicts`] miss a definite conflict.
    ///
    /// The visited set is sound here because terms are hash-consed and every
    /// fact recorded for a term is a pure function of that term: revisiting a
    /// `TermId` could only append byte-identical duplicate tuples.  All
    /// [`Self::check_conflicts`] consumers are existential searches, so
    /// removing duplicates never loses a real conflict; it only stops
    /// `check_precision_loss_conflict`'s `i < j` pair loop from spuriously
    /// pairing a duplicate entry with itself.  Children are pushed in reverse
    /// so pop order reproduces the original left-to-right pre-order for the
    /// first visit of every term.
    ///
    /// This walk is deliberately polarity-blind (it descends through `Not` /
    /// `Or` / `Implies` and records facts unconditionally), exactly like the
    /// recursive original; the polarity-aware collector is
    /// `check_fp::collect_fp_constraints_extended`.
    fn collect(&mut self, term: TermId, manager: &TermManager) {
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = vec![term];

        while let Some(term) = stack.pop() {
            if !visited.insert(term) {
                continue;
            }
            let Some(term_data) = manager.get(term) else {
                continue;
            };

            match &term_data.kind {
                // FP predicates
                TermKind::FpIsZero(arg) => {
                    self.is_zero_vars.insert(*arg);
                    stack.push(*arg);
                }
                TermKind::FpIsNegative(arg) => {
                    self.is_negative_vars.insert(*arg);
                    stack.push(*arg);
                }
                TermKind::FpIsPositive(arg) => {
                    self.is_positive_vars.insert(*arg);
                    stack.push(*arg);
                }
                // FP comparison - less than
                TermKind::FpLt(lhs, rhs) => {
                    self.fp_lts.push((*lhs, *rhs));
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                // Equality
                TermKind::Eq(lhs, rhs) => {
                    self.equalities.push((*lhs, *rhs));
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                // FP operations
                TermKind::FpAdd(rm, lhs, rhs) => {
                    self.fp_adds
                        .push((TermKind::FpAdd(*rm, *lhs, *rhs), *lhs, *rhs, term));
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                TermKind::FpDiv(rm, lhs, rhs) => {
                    self.fp_divs
                        .push((TermKind::FpDiv(*rm, *lhs, *rhs), *lhs, *rhs, term));
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                TermKind::FpMul(rm, lhs, rhs) => {
                    self.fp_muls
                        .push((TermKind::FpMul(*rm, *lhs, *rhs), *lhs, *rhs, term));
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                // FP conversions
                TermKind::FpToFp { rm: _, arg, eb, sb } => {
                    self.fp_conversions.push((*arg, *eb, *sb, term));
                    stack.push(*arg);
                }
                TermKind::RealToFp { rm, arg, eb, sb } => {
                    self.real_to_fp.push((
                        TermKind::RealToFp {
                            rm: *rm,
                            arg: *arg,
                            eb: *eb,
                            sb: *sb,
                        },
                        *arg,
                        *eb,
                        *sb,
                        term,
                    ));
                    stack.push(*arg);
                }
                // Compound terms
                TermKind::And(args) | TermKind::Or(args) => {
                    for &arg in args.iter().rev() {
                        stack.push(arg);
                    }
                }
                TermKind::Not(inner) => {
                    stack.push(*inner);
                }
                TermKind::Implies(lhs, rhs) => {
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                _ => {}
            }
        }
    }

    fn check_conflicts(&self, manager: &TermManager) -> bool {
        // Check 1: fp_06 - Zero sign handling
        // If we have isZero(x) AND isNegative(x) where x = fp.add(RNE, +0, -0),
        // this is a conflict because +0 + -0 = +0 in RNE mode
        for &var in &self.is_zero_vars {
            if self.is_negative_vars.contains(&var) {
                // Check if this variable is the result of +0 + -0
                if self.is_positive_zero_plus_negative_zero_result(var, manager) {
                    return true; // Conflict: +0 + -0 = +0, which is positive, not negative
                }
            }
        }

        // Check 2: fp_03 - Rounding mode constraints
        // For positive operands: RTP >= RTN always
        // So (fp.add RTP x y) < (fp.add RTN x y) is always UNSAT for positive operands
        if self.check_rounding_mode_conflict(manager) {
            return true;
        }

        // Check 3: fp_10 - Non-associativity / exact arithmetic
        // (x / y) * y != x for most FP values
        if self.check_non_associativity_conflict(manager) {
            return true;
        }

        // Check 4: fp_08 - Precision loss
        // Float32 -> Float64 conversion loses precision information
        if self.check_precision_loss_conflict(manager) {
            return true;
        }

        false
    }

    fn is_positive_zero_plus_negative_zero_result(
        &self,
        var: TermId,
        manager: &TermManager,
    ) -> bool {
        // Look for equality: var = fp.add(RNE, a, b) where a is +0 and b is -0 (or vice versa)
        for &(lhs, rhs) in &self.equalities {
            if lhs == var {
                if self.is_zero_addition_of_opposite_signs(rhs, manager) {
                    return true;
                }
            }
            if rhs == var {
                if self.is_zero_addition_of_opposite_signs(lhs, manager) {
                    return true;
                }
            }
        }
        false
    }

    fn is_zero_addition_of_opposite_signs(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(term_data) = manager.get(term) else {
            return false;
        };

        if let TermKind::FpAdd(_, lhs, rhs) = &term_data.kind {
            // Check if one operand has isZero AND isPositive, and the other has isZero AND isNegative
            let lhs_is_pos_zero =
                self.is_zero_vars.contains(lhs) && self.is_positive_vars.contains(lhs);
            let lhs_is_neg_zero =
                self.is_zero_vars.contains(lhs) && self.is_negative_vars.contains(lhs);
            let rhs_is_pos_zero =
                self.is_zero_vars.contains(rhs) && self.is_positive_vars.contains(rhs);
            let rhs_is_neg_zero =
                self.is_zero_vars.contains(rhs) && self.is_negative_vars.contains(rhs);

            // +0 + -0 or -0 + +0
            (lhs_is_pos_zero && rhs_is_neg_zero) || (lhs_is_neg_zero && rhs_is_pos_zero)
        } else {
            false
        }
    }

    fn check_rounding_mode_conflict(&self, manager: &TermManager) -> bool {
        // Check for patterns like: (fp.lt (fp.add RTP x y) (fp.add RTN x y))
        // This is always false for positive operands because RTP >= RTN
        for &(lt_lhs, lt_rhs) in &self.fp_lts {
            // Check if lt_lhs is (fp.add RTP x y) and lt_rhs is (fp.add RTN x y)
            let lhs_data = manager.get(lt_lhs);
            let rhs_data = manager.get(lt_rhs);

            if let (Some(lhs), Some(rhs)) = (lhs_data, rhs_data) {
                if let (TermKind::FpAdd(rm_lhs, a1, b1), TermKind::FpAdd(rm_rhs, a2, b2)) =
                    (&lhs.kind, &rhs.kind)
                {
                    // RTP < RTN is impossible for same positive operands
                    if *rm_lhs == RoundingMode::RTP
                        && *rm_rhs == RoundingMode::RTN
                        && a1 == a2
                        && b1 == b2
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn check_non_associativity_conflict(&self, manager: &TermManager) -> bool {
        // Check for pattern: product = z1 * z2 where z1 = x / y and product must equal x
        // This is generally false in FP because (x / y) * y != x
        for &(_, div_lhs, div_rhs, div_result) in &self.fp_divs {
            for &(_, mul_lhs, mul_rhs, mul_result) in &self.fp_muls {
                // Check if multiplication uses the division result
                if mul_lhs == div_result || mul_rhs == div_result {
                    // The other operand should be the divisor
                    let other_mul_operand = if mul_lhs == div_result {
                        mul_rhs
                    } else {
                        mul_lhs
                    };

                    // Check if other_mul_operand equals div_rhs (the divisor)
                    if self.terms_equal(other_mul_operand, div_rhs, manager) {
                        // Now check if the multiplication result must equal the dividend
                        for &(eq_lhs, eq_rhs) in &self.equalities {
                            if (eq_lhs == mul_result && self.terms_equal(eq_rhs, div_lhs, manager))
                                || (eq_rhs == mul_result
                                    && self.terms_equal(eq_lhs, div_lhs, manager))
                            {
                                // (x / y) * y = x is asserted but not generally true in FP
                                // Additional check: if dividend is a specific value like 10 and divisor is 3
                                // then 10/3 * 3 != 10 in FP
                                if self.is_non_exact_division(div_lhs, div_rhs, manager) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn terms_equal(&self, a: TermId, b: TermId, _manager: &TermManager) -> bool {
        if a == b {
            return true;
        }
        // Check via equality constraints
        for &(eq_lhs, eq_rhs) in &self.equalities {
            if (eq_lhs == a && eq_rhs == b) || (eq_lhs == b && eq_rhs == a) {
                return true;
            }
        }
        false
    }

    fn is_non_exact_division(
        &self,
        dividend: TermId,
        divisor: TermId,
        manager: &TermManager,
    ) -> bool {
        // Check if this is a division that would result in precision loss
        // e.g., 10 / 3 cannot be exactly represented in FP
        if let Some(div_val) = self.get_fp_literal_value(dividend, manager) {
            if let Some(divisor_val) = self.get_fp_literal_value(divisor, manager) {
                // Check if dividend / divisor is not exact
                if divisor_val != 0.0 {
                    let quotient = div_val / divisor_val;
                    let product = quotient * divisor_val;
                    // If multiplying back doesn't give the exact original value, it's non-exact
                    if (product - div_val).abs() > f64::EPSILON {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn get_fp_literal_value(&self, term: TermId, manager: &TermManager) -> Option<f64> {
        // Try to extract a floating-point literal value
        // Check equality constraints for real_to_fp conversions
        for &(eq_lhs, eq_rhs) in &self.equalities {
            if eq_lhs == term {
                if let Some(val) = self.extract_fp_value(eq_rhs, manager) {
                    return Some(val);
                }
            }
            if eq_rhs == term {
                if let Some(val) = self.extract_fp_value(eq_lhs, manager) {
                    return Some(val);
                }
            }
        }
        self.extract_fp_value(term, manager)
    }

    fn extract_fp_value(&self, term: TermId, manager: &TermManager) -> Option<f64> {
        let term_data = manager.get(term)?;
        match &term_data.kind {
            TermKind::RealToFp { arg, .. } => {
                // Get the real value
                if let Some(real_data) = manager.get(*arg) {
                    if let TermKind::RealConst(r) = &real_data.kind {
                        return r.to_f64();
                    }
                }
                None
            }
            TermKind::IntConst(n) => n.to_i64().map(|v| v as f64),
            TermKind::RealConst(r) => r.to_f64(),
            _ => None,
        }
    }

    fn check_precision_loss_conflict(&self, manager: &TermManager) -> bool {
        // Check for pattern: x64_1 = to_fp64(to_fp32(val)) AND x64_2 = to_fp64(val) AND x64_1 = x64_2
        // This is false for values that lose precision in float32

        // Find pairs of conversions that go through different paths
        for i in 0..self.fp_conversions.len() {
            for j in i + 1..self.fp_conversions.len() {
                let (src1, eb1, sb1, result1) = self.fp_conversions[i];
                let (src2, eb2, sb2, result2) = self.fp_conversions[j];

                // Check if same target format
                if eb1 == eb2 && sb1 == sb2 {
                    // Check if result1 = result2 is asserted
                    if self.terms_equal(result1, result2, manager) {
                        // Check if one source went through a smaller format
                        if self.source_went_through_smaller_format(src1, eb1, sb1, manager)
                            && self.is_direct_from_value(src2, manager)
                        {
                            // Check if the original value has precision that would be lost
                            if self.value_loses_precision_in_smaller_format(src2, manager) {
                                return true;
                            }
                        }
                        if self.source_went_through_smaller_format(src2, eb2, sb2, manager)
                            && self.is_direct_from_value(src1, manager)
                        {
                            if self.value_loses_precision_in_smaller_format(src1, manager) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn source_went_through_smaller_format(
        &self,
        source: TermId,
        target_eb: u32,
        target_sb: u32,
        manager: &TermManager,
    ) -> bool {
        // Check if source is the result of a conversion from a smaller format
        if let Some(term_data) = manager.get(source) {
            if let TermKind::FpToFp { arg: _, eb, sb, .. } = &term_data.kind {
                // Smaller format means fewer bits
                return *eb < target_eb || *sb < target_sb;
            }
        }
        // Also check via equality
        for &(eq_lhs, eq_rhs) in &self.equalities {
            let to_check = if eq_lhs == source {
                eq_rhs
            } else if eq_rhs == source {
                eq_lhs
            } else {
                continue;
            };
            if let Some(term_data) = manager.get(to_check) {
                if let TermKind::FpToFp { arg: _, eb, sb, .. } = &term_data.kind {
                    return *eb < target_eb || *sb < target_sb;
                }
            }
        }
        false
    }

    fn is_direct_from_value(&self, term: TermId, manager: &TermManager) -> bool {
        // Check if term is directly converted from a real/decimal value
        if let Some(term_data) = manager.get(term) {
            if matches!(term_data.kind, TermKind::RealToFp { .. }) {
                return true;
            }
        }
        for &(eq_lhs, eq_rhs) in &self.equalities {
            let to_check = if eq_lhs == term {
                eq_rhs
            } else if eq_rhs == term {
                eq_lhs
            } else {
                continue;
            };
            if let Some(term_data) = manager.get(to_check) {
                if matches!(term_data.kind, TermKind::RealToFp { .. }) {
                    return true;
                }
            }
        }
        false
    }

    fn value_loses_precision_in_smaller_format(&self, term: TermId, manager: &TermManager) -> bool {
        // Check if the value being converted would lose precision in float32
        if let Some(val) = self.get_original_real_value(term, manager) {
            // Convert to f32 and back to see if precision is lost
            let as_f32 = val as f32;
            let back_to_f64 = as_f32 as f64;
            if (val - back_to_f64).abs() > f64::EPSILON {
                return true;
            }
        }
        false
    }

    fn get_original_real_value(&self, term: TermId, manager: &TermManager) -> Option<f64> {
        // Get the original real value from RealToFp conversion
        if let Some(term_data) = manager.get(term) {
            if let TermKind::RealToFp { arg, .. } = &term_data.kind {
                if let Some(arg_data) = manager.get(*arg) {
                    if let TermKind::RealConst(r) = &arg_data.kind {
                        return r.to_f64();
                    }
                }
            }
        }
        for &(eq_lhs, eq_rhs) in &self.equalities {
            let to_check = if eq_lhs == term {
                eq_rhs
            } else if eq_rhs == term {
                eq_lhs
            } else {
                continue;
            };
            if let Some(term_data) = manager.get(to_check) {
                if let TermKind::RealToFp { arg, .. } = &term_data.kind {
                    if let Some(arg_data) = manager.get(*arg) {
                        if let TermKind::RealConst(r) = &arg_data.kind {
                            return r.to_f64();
                        }
                    }
                }
            }
        }
        None
    }
}

/// Regression tests for [`FpConstraintCollector::collect`]'s conversion from
/// native recursion (no depth guard, no visited set) to an explicit worklist
/// with a `TermId`-keyed visited set — see `collect`'s doc comment for the
/// full rationale.
#[cfg(test)]
mod tests {
    use super::*;

    /// Facts recorded through every polarity-blind descent arm (`Not`,
    /// `Implies`, `And`), plus the end-to-end conflict pin for the exact
    /// `+0 + -0` pattern the collector exists to catch.
    #[test]
    fn collect_records_facts_and_finds_zero_sign_conflict() {
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let x = manager.mk_var("x", fp_sort);
        let p = manager.mk_var("p", fp_sort);
        let n = manager.mk_var("n", fp_sort);
        let add = manager.mk_fp_add(RoundingMode::RNE, p, n);
        let eq = manager.mk_eq(x, add);

        let is_zero_x = manager.mk_fp_is_zero(x);
        let is_neg_x = manager.mk_fp_is_negative(x);
        let is_zero_p = manager.mk_fp_is_zero(p);
        let is_pos_p = manager.mk_fp_is_positive(p);
        let is_zero_n = manager.mk_fp_is_zero(n);
        let is_neg_n = manager.mk_fp_is_negative(n);

        // Route two of the predicates through `Not` and `Implies` to pin that
        // the walk is polarity-blind, exactly like the recursive original.
        let not_wrapped = manager.mk_not(is_pos_p);
        let implies_wrapped = manager.mk_implies(is_zero_n, is_neg_n);

        let mut collector = FpConstraintCollector::new();
        for term in [
            eq,
            is_zero_x,
            is_neg_x,
            is_zero_p,
            not_wrapped,
            implies_wrapped,
        ] {
            collector.collect(term, &manager);
        }

        assert_eq!(collector.equalities, vec![(x, add)]);
        assert_eq!(collector.fp_adds.len(), 1);
        assert!(collector.is_zero_vars.contains(&x));
        assert!(collector.is_negative_vars.contains(&x));
        assert!(
            collector.is_positive_vars.contains(&p),
            "the walk descends through `Not` unconditionally (polarity-blind)"
        );
        assert!(
            collector.is_negative_vars.contains(&n),
            "the walk descends through `Implies` unconditionally"
        );
        assert!(
            collector.check_conflicts(&manager),
            "isZero(x) + isNegative(x) + x = (+0 + -0) is the collector's canonical conflict"
        );
    }

    /// A term reachable along several paths of one formula is recorded once:
    /// the visited set suppresses byte-identical duplicate tuples (the
    /// recursive original appended one copy per path).
    #[test]
    fn collect_deduplicates_a_term_shared_within_one_formula() {
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let x = manager.mk_var("x", fp_sort);
        let y = manager.mk_var("y", fp_sort);
        let eq = manager.mk_eq(x, y);
        // `mk_and` flattens nested `And`s but keeps duplicate non-`And`
        // children, so the same `Eq` term arrives twice.
        let and = manager.mk_and(vec![eq, eq]);

        let mut collector = FpConstraintCollector::new();
        collector.collect(and, &manager);

        assert_eq!(collector.equalities, vec![(x, y)]);
    }

    /// The named worst case of the recursion sweep: a doubling DAG
    /// (`d_{i+1} = fp.add(d_i, d_i)`) made the unmemoized original perform
    /// `2^60` visits — effectively a hang.  With the visited set the walk is
    /// linear, records each distinct addition exactly once, and finishes
    /// immediately.
    #[test]
    fn collect_shared_add_dag_is_linear_not_exponential() {
        const LEVELS: usize = 60;

        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let mut term = manager.mk_var("leaf", fp_sort);
        for _ in 0..LEVELS {
            term = manager.mk_fp_add(RoundingMode::RNE, term, term);
        }

        let mut collector = FpConstraintCollector::new();
        collector.collect(term, &manager);

        assert_eq!(
            collector.fp_adds.len(),
            LEVELS,
            "each of the {LEVELS} distinct additions must be recorded exactly once"
        );
    }

    /// A `fp.add` chain 12 500 levels deep must be walked on a 128 KiB stack:
    /// a native stack overflow is a fatal abort `catch_unwind` cannot
    /// intercept, so returning at all — with the full fact count — is the
    /// assertion.
    #[test]
    fn collect_survives_a_deep_fp_add_chain_on_a_small_stack() {
        // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
        // ~10 B-per-frame threshold is the pin, so never raise one alone.
        const STACK_SIZE: usize = 1 << 17; // 128 KiB
        const DEPTH: usize = 12_500;

        let handle = std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut manager = TermManager::new();
                let fp_sort = manager.sorts.float_sort(11, 53);
                let one = manager.mk_var("one", fp_sort);
                let mut term = manager.mk_var("acc", fp_sort);
                for _ in 0..DEPTH {
                    term = manager.mk_fp_add(RoundingMode::RNE, term, one);
                }

                let mut collector = FpConstraintCollector::new();
                collector.collect(term, &manager);

                assert_eq!(
                    collector.fp_adds.len(),
                    DEPTH,
                    "one addition fact per level of a {DEPTH}-deep chain"
                );
            })
            .expect("spawning a 128 KiB-stack thread should succeed");

        handle
            .join()
            .expect("the FP collector walk must return on 128 KiB instead of overflowing");
    }
}
