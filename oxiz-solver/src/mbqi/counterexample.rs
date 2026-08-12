//! Counter-example Generation and Refinement
//!
//! This module implements counterexample generation for MBQI. A counterexample
//! is an assignment to quantified variables that falsifies the quantifier body
//! under the current model.
//!
//! For a universal quantifier ∀x.φ(x), a counterexample is an assignment σ such
//! that ¬φ(σ(x)) holds in the current model.
//!
//! # Strategy
//!
//! 1. **Model Evaluation**: Evaluate quantifier body under candidate assignments
//! 2. **Satisfiability Checking**: Use auxiliary solver to find counterexamples
//! 3. **Conflict Analysis**: When no counterexamples exist, analyze why
//! 4. **Refinement**: Use counterexamples to refine the search space

#[allow(unused_imports)]
use crate::prelude::*;
use core::fmt;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;
#[cfg(feature = "std")]
use oxiz_time::{Duration, Instant};
use smallvec::SmallVec;

use super::model_completion::CompletedModel;
use super::{Instantiation, InstantiationReason, QuantifiedFormula};

/// Compute SMT-LIB Euclidean division and remainder for integers.
///
/// Returns `(q, r)` such that `a = b*q + r` and `0 <= r < |b|`.  The caller
/// must ensure `b != 0`.  Rust's truncated `/` and `%` take the sign of the
/// dividend, so this floor-adjusts to produce a non-negative remainder,
/// matching SMT-LIB `div`/`mod` (e.g. `(-7) div 2 = -4`, `(-7) mod 2 = 1`).
fn euclidean_div_rem(a: &BigInt, b: &BigInt) -> (BigInt, BigInt) {
    use num_traits::Zero;
    let q_trunc = a / b;
    let r_trunc = a % b;
    if r_trunc < BigInt::zero() {
        // Remainder is negative; shift it into [0, |b|) and adjust the quotient
        // so the identity a = b*q + r still holds.
        if *b > BigInt::zero() {
            (q_trunc - 1, r_trunc + b)
        } else {
            (q_trunc + 1, r_trunc - b)
        }
    } else {
        (q_trunc, r_trunc)
    }
}

/// A counter-example to a quantified formula
#[derive(Debug, Clone)]
pub struct CounterExample {
    /// The quantifier this is a counterexample for
    pub quantifier: TermId,
    /// Assignment to bound variables
    pub assignment: FxHashMap<Spur, TermId>,
    /// Witness terms (the concrete values assigned)
    pub witnesses: Vec<TermId>,
    /// Evaluation of the body under this assignment
    pub body_value: Option<TermId>,
    /// Quality score (higher = better counterexample)
    pub quality: f64,
    /// Generation at which this was found
    pub generation: u32,
}

impl CounterExample {
    /// Create a new counter-example
    pub fn new(
        quantifier: TermId,
        assignment: FxHashMap<Spur, TermId>,
        witnesses: Vec<TermId>,
        generation: u32,
    ) -> Self {
        Self {
            quantifier,
            assignment,
            witnesses,
            body_value: None,
            quality: 1.0,
            generation,
        }
    }

    /// Convert to an instantiation
    pub fn to_instantiation(&self, result: TermId) -> Instantiation {
        Instantiation::with_reason(
            self.quantifier,
            self.assignment.clone(),
            result,
            self.generation,
            InstantiationReason::Conflict,
        )
    }

    /// Calculate quality score based on term complexity
    pub fn calculate_quality(&mut self, manager: &TermManager) {
        let mut total_size = 0;
        let mut num_constants = 0;

        for &witness in &self.witnesses {
            let size = self.term_size(witness, manager);
            total_size += size;

            if self.is_constant(witness, manager) {
                num_constants += 1;
            }
        }

        // Prefer simpler terms (smaller size)
        let size_factor = 1.0 / (1.0 + total_size as f64);
        // Prefer more constants (ground terms)
        let const_factor = 1.0 + (num_constants as f64 / self.witnesses.len().max(1) as f64);

        self.quality = size_factor * const_factor;
    }

    /// Number of distinct subterms reached from `term` through the
    /// structural kinds below (each shared subterm counted once).
    ///
    /// Iterative with an explicit heap stack: the previous helper recursed
    /// once per nesting level and returned a plain `usize` with no error
    /// channel, so a deeply nested witness — witnesses come straight from
    /// user input on the default `check_sat` path — could abort the process
    /// by exhausting the native stack, and a depth cap could only have
    /// reported a silently wrong size (skewing the counterexample quality
    /// score).  `visited` keeps re-expansion of the shared, hash-consed DAG
    /// linear, exactly as before.
    ///
    /// Semantics are preserved exactly: every newly visited term
    /// contributes 1 (including an unresolvable id, whose children are not
    /// explored), repeat visits contribute 0, and only `And`/`Or`/`Not`/
    /// `Neg`/`Eq`/`Lt` descend — the sum is order-independent, so the
    /// traversal order is immaterial.
    fn term_size(&self, term: TermId, manager: &TermManager) -> usize {
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = vec![term];
        let mut size = 0usize;

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            size += 1;

            let Some(t) = manager.get(current) else {
                continue;
            };
            match &t.kind {
                TermKind::And(args) | TermKind::Or(args) => stack.extend(args.iter().copied()),
                TermKind::Not(arg) | TermKind::Neg(arg) => stack.push(*arg),
                TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                    stack.push(*lhs);
                    stack.push(*rhs);
                }
                _ => {}
            }
        }

        size
    }

    fn is_constant(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
        )
    }
}

/// Result of counterexample generation for a single quantifier
#[derive(Debug, Clone)]
pub struct CexGenerationResult {
    /// Counterexamples found
    pub counterexamples: Vec<CounterExample>,
    /// Whether all candidate evaluations resolved to concrete boolean values.
    /// If false, some evaluations produced symbolic residuals, meaning we cannot
    /// be sure the quantifier is satisfied even if no counterexamples were found.
    pub all_evaluations_ground: bool,
}

/// Counter-example generator
#[derive(Debug)]
pub struct CounterExampleGenerator {
    /// Maximum number of counterexamples to generate per quantifier
    max_cex_per_quantifier: usize,
    /// Maximum number of candidates to try per variable
    max_candidates_per_var: usize,
    /// Maximum total search time
    #[cfg(feature = "std")]
    max_search_time: Duration,
    /// Current generation bound for term selection
    generation_bound: u32,
    /// Statistics
    stats: CexStats,
    /// Candidate cache
    candidate_cache: FxHashMap<SortId, Vec<TermId>>,
}

impl CounterExampleGenerator {
    /// Create a new counterexample generator
    pub fn new() -> Self {
        Self {
            max_cex_per_quantifier: 5,
            max_candidates_per_var: 10,
            #[cfg(feature = "std")]
            max_search_time: Duration::from_secs(1),
            generation_bound: 0,
            stats: CexStats::default(),
            candidate_cache: FxHashMap::default(),
        }
    }

    /// Create with custom limits
    #[cfg(feature = "std")]
    pub fn with_limits(max_cex: usize, max_candidates: usize, max_time: Duration) -> Self {
        let mut generator = Self::new();
        generator.max_cex_per_quantifier = max_cex;
        generator.max_candidates_per_var = max_candidates;
        generator.max_search_time = max_time;
        generator
    }

    /// Generate counterexamples for a quantifier.
    ///
    /// Returns a `CexGenerationResult` containing the counterexamples found and
    /// a flag indicating whether all evaluations resolved to concrete booleans.
    pub fn generate(
        &mut self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> CexGenerationResult {
        #[cfg(feature = "std")]
        let start_time = Instant::now();
        let mut counterexamples = Vec::new();
        let mut all_ground = true;
        self.stats.num_searches += 1;

        // Build candidate lists for each bound variable
        let candidates = self.build_candidate_lists(&quantifier.bound_vars, model, manager);

        // Enumerate combinations of candidates
        let combinations = self.enumerate_combinations(
            &candidates,
            self.max_candidates_per_var,
            self.max_cex_per_quantifier * 20, // Generate more combinations than we need
        );

        self.stats.num_combinations_tried += combinations.len();

        // If there are no combinations to try, we cannot verify the quantifier
        // is satisfied -- mark as non-ground.
        if combinations.is_empty() {
            all_ground = false;
        }

        for combo in combinations {
            #[cfg(feature = "std")]
            if start_time.elapsed() > self.max_search_time {
                self.stats.num_timeouts += 1;
                // Timeout means we could not fully verify -- not ground.
                all_ground = false;
                break;
            }

            if counterexamples.len() >= self.max_cex_per_quantifier {
                break;
            }

            // Build assignment from combination
            let mut assignment = FxHashMap::default();
            for (i, &candidate) in combo.iter().enumerate() {
                if let Some(var_name) = quantifier.var_name(i) {
                    assignment.insert(var_name, candidate);
                }
            }

            // Apply substitution and evaluate
            let substituted = self.apply_substitution(quantifier.body, &assignment, manager);
            let evaluated = self.evaluate_under_model(substituted, model, manager);

            // Track whether this evaluation resolved to a concrete boolean
            if !self.is_ground_boolean(evaluated, manager) {
                all_ground = false;
            }

            // Check if this is a counterexample
            if self.is_counterexample(evaluated, quantifier.is_universal, manager) {
                let mut cex =
                    CounterExample::new(quantifier.term, assignment, combo, model.generation);
                cex.body_value = Some(evaluated);
                cex.calculate_quality(manager);
                counterexamples.push(cex);
                self.stats.num_counterexamples_found += 1;
            }
        }

        // Sort by quality (best first)
        counterexamples.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Limit to max
        counterexamples.truncate(self.max_cex_per_quantifier);

        #[cfg(feature = "std")]
        {
            self.stats.total_time += start_time.elapsed();
        }

        CexGenerationResult {
            counterexamples,
            all_evaluations_ground: all_ground,
        }
    }

    /// Build candidate lists for bound variables
    fn build_candidate_lists(
        &mut self,
        bound_vars: &[(Spur, SortId)],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Vec<TermId>> {
        let mut result = Vec::new();

        for &(_var_name, sort) in bound_vars {
            // Check cache first
            if let Some(cached) = self.candidate_cache.get(&sort) {
                result.push(cached.clone());
                continue;
            }

            let mut candidates = Vec::new();

            // Strategy 1: Use values from the universe (for uninterpreted sorts)
            if let Some(universe) = model.universe(sort) {
                candidates.extend_from_slice(universe);
            }

            // Strategy 2: Use values from the model
            for (&term, &value) in &model.assignments {
                if let Some(t) = manager.get(term)
                    && t.sort == sort
                    && !candidates.contains(&value)
                {
                    candidates.push(value);
                }
            }

            // Strategy 3: Add default values based on sort
            self.add_default_candidates(sort, &mut candidates, manager);

            // Limit candidates
            candidates.truncate(self.max_candidates_per_var);

            // Cache for future use
            self.candidate_cache.insert(sort, candidates.clone());

            result.push(candidates);
        }

        result
    }

    /// Add default candidate values for a sort
    fn add_default_candidates(
        &self,
        sort: SortId,
        candidates: &mut Vec<TermId>,
        manager: &mut TermManager,
    ) {
        if sort == manager.sorts.int_sort {
            // Add small integers
            for i in -2..=5 {
                let val = manager.mk_int(BigInt::from(i));
                if !candidates.contains(&val) {
                    candidates.push(val);
                }
            }
        } else if sort == manager.sorts.bool_sort {
            let true_val = manager.mk_true();
            let false_val = manager.mk_false();
            if !candidates.contains(&true_val) {
                candidates.push(true_val);
            }
            if !candidates.contains(&false_val) {
                candidates.push(false_val);
            }
        }
    }

    /// Enumerate combinations of candidates
    fn enumerate_combinations(
        &self,
        candidates: &[Vec<TermId>],
        max_per_dim: usize,
        max_total: usize,
    ) -> Vec<Vec<TermId>> {
        if candidates.is_empty() {
            return vec![vec![]];
        }

        let mut results = Vec::new();
        let mut indices = vec![0usize; candidates.len()];

        loop {
            // Build current combination
            let combo: Vec<TermId> = indices
                .iter()
                .enumerate()
                .filter_map(|(i, &idx)| candidates.get(i).and_then(|c| c.get(idx).copied()))
                .collect();

            if combo.len() == candidates.len() {
                results.push(combo);
            }

            if results.len() >= max_total {
                break;
            }

            // Increment indices (like odometer)
            let mut carry = true;
            for (i, idx) in indices.iter_mut().enumerate() {
                if carry {
                    *idx += 1;
                    let limit = candidates.get(i).map_or(1, |c| c.len().min(max_per_dim));
                    if *idx >= limit {
                        *idx = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                // Overflow - tried all combinations
                break;
            }
        }

        results
    }

    /// Substitute a quantifier's bound variables in `term`, by variable name.
    ///
    /// Delegates to [`utils::substitute`](crate::mbqi::macros::utils::substitute),
    /// the one shared implementation for this crate, which resolves the
    /// name-keyed map against the term's actual free occurrences and hands the
    /// result to [`TermManager::substitute`].
    ///
    /// This used to be a local recursive walk with a memo table and a
    /// `TermKind` whitelist that ended in `_ => term`, so every kind outside
    /// the whitelist was returned **unchanged** -- the whitelist covered 20 kinds, so
    /// `Xor`, `Distinct`, every bit-vector, string, floating-point and
    /// datatype operator, and every binder fell through. A
    /// bound variable sitting anywhere under such a kind therefore survived
    /// into the "ground instance", which is then not an instance at all: the
    /// engine reported a substitution it had not performed. Four
    /// near-identical copies of that walk existed in this module (here, and in
    /// `instantiation`, `counterexample`, `lazy_instantiation`,
    /// `conflict_driven`); they are all now this one call, because a duplicate
    /// that has diverged four times will diverge again.
    ///
    /// The shared routine additionally descends into
    /// `Forall`/`Exists`/`Let`/`Match` bodies, bindings, cases and trigger
    /// patterns with capture-avoiding alpha-renaming, and walks with an
    /// explicit heap stack rather than native recursion.
    ///
    /// [`TermManager::substitute`]: oxiz_core::ast::TermManager::substitute
    fn apply_substitution(
        &self,
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        crate::mbqi::macros::utils::substitute(term, subst, manager)
    }

    /// Evaluate a term under a model
    fn evaluate_under_model(
        &self,
        term: TermId,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> TermId {
        let mut cache = FxHashMap::default();
        self.evaluate_under_model_cached(term, model, manager, &mut cache)
    }

    /// Evaluate `root` under `model`, memoizing resolved subterms in `cache`.
    ///
    /// # Why this is an explicit-stack machine
    ///
    /// This used to be a pair of mutually recursive functions: the term
    /// evaluator, and an inline `Exists` witness search
    /// (`evaluate_exists_inline`) that re-entered the evaluator on every
    /// substituted candidate body.  Evaluation depth therefore consumed
    /// native call stack, and a deeply nested quantifier body — or a chain
    /// of nested existentials — aborted the whole process by stack overflow.
    /// MBQI runs on the default `check_sat` path for any quantified input,
    /// so that was reachable from plain SMT-LIB scripts.  The memo cache
    /// bounds *work* on the shared, hash-consed term DAG (each `TermId` is
    /// resolved at most once per outer call), but it never bounded *stack*.
    ///
    /// The return type has no error channel — a symbolic residual is a
    /// legitimate answer — so a depth cap could only have fabricated a wrong
    /// evaluation.  Instead, both halves of the old recursion now run in one
    /// frame machine: `stack` owns every suspended step as an `EvalFrame`,
    /// the most recently completed subresult travels in `value`, and the
    /// mutual edge (`Exists` candidate body → evaluator) is the
    /// `ExistsAdvance` / `ExistsJudge` frame pair.  Nesting depth costs
    /// heap, never native stack.
    ///
    /// Semantics are preserved exactly, including every short-circuit: a
    /// definite `False` conjunct (dually `True` disjunct) resolves its
    /// connective with the remaining operands left unevaluated; a decided
    /// `Implies` premise or `Ite` condition skips the dead branch; and the
    /// `Exists` odometer stops at the first witness.
    fn evaluate_under_model_cached(
        &self,
        root: TermId,
        model: &CompletedModel,
        manager: &mut TermManager,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        /// A binary operator whose operands are both evaluated (left, then
        /// right) before folding.
        enum BinOp {
            /// `=` — folds via `eval_eq`.
            Eq,
            /// `<` — folds via `eval_lt`.
            Lt,
            /// `<=` — folds via `eval_le`.
            Le,
            /// `>` — folds via `eval_gt`.
            Gt,
            /// `>=` — folds via `eval_ge`.
            Ge,
            /// `-` — folds via `eval_sub`.
            Sub,
            /// `div` — folds via `eval_div`.
            Div,
            /// `mod` — folds via `eval_modulo`.
            Mod,
        }

        /// An n-ary arithmetic operator; every operand is evaluated before
        /// folding.
        enum NaryOp {
            /// `+` — folds via `eval_add`.
            Add,
            /// `*` — folds via `eval_mul`.
            Mul,
        }

        /// The suspended inline evaluation of one `Exists` term: the
        /// candidate odometer of the old `evaluate_exists_inline`, lifted
        /// into a heap frame so evaluating a candidate body can re-enter the
        /// machine without native recursion.
        ///
        /// The verdict rule is unchanged:
        /// - `True` if any candidate gives a `True` body evaluation;
        /// - `False` if ALL candidates give `False` body evaluations
        ///   (provably no witness among the candidates);
        /// - the symbolic body if any evaluation stayed symbolic (cannot
        ///   determine), or if there was nothing to enumerate.
        struct ExistsState {
            /// The `Exists` term itself (the cache key for the verdict).
            term: TermId,
            /// The quantifier body candidates are substituted into.
            body: TermId,
            /// The bound variables, in `candidate_lists` order.
            vars: SmallVec<[(Spur, SortId); 2]>,
            /// Per-variable candidate values.
            candidate_lists: Vec<Vec<TermId>>,
            /// Odometer position (`indices[i]` indexes `candidate_lists[i]`).
            indices: Vec<usize>,
            /// Combinations classified so far.
            combo_count: usize,
            /// Enumeration budget (product of list lengths, capped at 50).
            total_combos: usize,
            /// A candidate body evaluated to `True` (witness found).
            found_true: bool,
            /// A candidate body stayed symbolic.
            found_symbolic: bool,
            /// Every classified candidate body evaluated to `False`.
            all_false: bool,
        }

        /// One suspended evaluation step, owned by the heap `stack`.  A
        /// frame is pushed *below* the `Enter` of the subterm it waits on
        /// and reads that subterm's result from `value` when it resurfaces.
        enum EvalFrame {
            /// Evaluate a term: cache probe, direct model value, then
            /// dispatch on the term kind.
            Enter(TermId),
            /// `Not(arg)`: fold the evaluated argument.
            NotArg {
                /// The `Not` term (cache key).
                term: TermId,
            },
            /// `And(args)`: classify the conjunct just evaluated, then
            /// evaluate `args[next..]` — unless a `False` ends it early.
            AndArgs {
                /// The `And` term (cache key).
                term: TermId,
                /// All conjuncts.
                args: SmallVec<[TermId; 4]>,
                /// Index of the next conjunct to evaluate.
                next: usize,
                /// Every conjunct so far evaluated to `True`.
                all_true: bool,
            },
            /// `Or(args)`: dual of `AndArgs`.
            OrArgs {
                /// The `Or` term (cache key).
                term: TermId,
                /// All disjuncts.
                args: SmallVec<[TermId; 4]>,
                /// Index of the next disjunct to evaluate.
                next: usize,
                /// Every disjunct so far evaluated to `False`.
                all_false: bool,
            },
            /// `Implies(lhs, rhs)`: inspect the evaluated premise and decide
            /// how the conclusion is handled.
            ImpliesLhs {
                /// The `Implies` term (cache key).
                term: TermId,
                /// The unevaluated conclusion.
                rhs: TermId,
            },
            /// `Implies` whose premise stayed symbolic (or unresolvable):
            /// rebuild from both evaluated halves.
            ImpliesRhs {
                /// The `Implies` term (cache key).
                term: TermId,
                /// The evaluated premise.
                lhs_eval: TermId,
            },
            /// The child's value *is* this term's value (an `Implies` with a
            /// `True` premise, or an `Ite` whose condition decided a
            /// branch); records it in the cache under `term`.
            Forward {
                /// The term whose cache entry receives the forwarded value.
                term: TermId,
            },
            /// Binary operator: the left operand's value arrives next.
            BinLhs {
                /// The operator term (cache key).
                term: TermId,
                /// Which operator folds the operands.
                op: BinOp,
                /// The unevaluated right operand.
                rhs: TermId,
            },
            /// Binary operator: the right operand's value arrives next, then
            /// the operator folds.
            BinRhs {
                /// The operator term (cache key).
                term: TermId,
                /// Which operator folds the operands.
                op: BinOp,
                /// The evaluated left operand.
                lhs_eval: TermId,
            },
            /// N-ary operator: collect the operand just evaluated, evaluate
            /// `args[next..]`, then fold.
            NaryArgs {
                /// The operator term (cache key).
                term: TermId,
                /// Which operator folds the operands.
                op: NaryOp,
                /// All operands.
                args: SmallVec<[TermId; 4]>,
                /// Index of the next operand to evaluate.
                next: usize,
                /// Operand values collected so far.
                evaluated: SmallVec<[TermId; 4]>,
            },
            /// `Neg(arg)`: fold the evaluated argument.
            NegArg {
                /// The `Neg` term (cache key).
                term: TermId,
            },
            /// `Ite`: the condition's value arrives next.
            IteCond {
                /// The `Ite` term (cache key).
                term: TermId,
                /// The unevaluated then-branch.
                then_br: TermId,
                /// The unevaluated else-branch.
                else_br: TermId,
            },
            /// `Ite` with a symbolic condition: the then-branch's value
            /// arrives next (both branches are evaluated, as before).
            IteThen {
                /// The `Ite` term (cache key).
                term: TermId,
                /// The evaluated (symbolic) condition.
                cond_eval: TermId,
                /// The unevaluated else-branch.
                else_br: TermId,
            },
            /// `Ite` with a symbolic condition: the else-branch's value
            /// arrives next, then the `ite` is rebuilt.
            IteElse {
                /// The `Ite` term (cache key).
                term: TermId,
                /// The evaluated (symbolic) condition.
                cond_eval: TermId,
                /// The evaluated then-branch.
                then_eval: TermId,
            },
            /// `Apply`: collect the argument just evaluated, evaluate
            /// `args[next..]`, then consult the model's function table.
            ApplyArgs {
                /// The application term (cache key).
                term: TermId,
                /// The application's result sort (for rebuilding).
                sort: SortId,
                /// The applied function symbol.
                func: Spur,
                /// All argument terms.
                args: SmallVec<[TermId; 4]>,
                /// Index of the next argument to evaluate.
                next: usize,
                /// Argument values collected so far.
                evaluated: Vec<TermId>,
            },
            /// `Select`: the index's value arrives next; the array is only
            /// evaluated if the model lookups miss.
            SelectIndex {
                /// The `Select` term (cache key).
                term: TermId,
                /// The original (unevaluated) array operand.
                array: TermId,
            },
            /// `Select` whose model lookups missed: the array's value
            /// arrives next, then the select is rebuilt and probed once
            /// more.
            SelectArray {
                /// The `Select` term (cache key).
                term: TermId,
                /// The evaluated index.
                index_eval: TermId,
            },
            /// `Exists` loop top: emit the next candidate combination, or
            /// finish when the budget is spent.
            ExistsAdvance(Box<ExistsState>),
            /// `Exists` after one candidate body evaluated: classify it,
            /// step the odometer, and loop or finish.
            ExistsJudge(Box<ExistsState>),
        }

        /// Close out an `Exists` search: the exact verdict rule of the old
        /// `evaluate_exists_inline` tail, cached under the `Exists` term.
        fn finish_exists(
            st: &ExistsState,
            manager: &mut TermManager,
            cache: &mut FxHashMap<TermId, TermId>,
        ) -> TermId {
            let result = if st.found_true {
                manager.mk_true()
            } else if st.all_false && !st.found_symbolic && st.combo_count > 0 {
                // All candidates gave False: exists is provably False under
                // this model.
                manager.mk_false()
            } else {
                // Some candidates were symbolic or we had a witness check
                // issue -- return symbolic.
                st.body
            };
            cache.insert(st.term, result);
            result
        }

        /// Fold a fully evaluated application: function-table lookup first,
        /// then rebuild with the evaluated arguments and probe the model for
        /// the rebuilt term.
        fn fold_apply(
            func: Spur,
            evaluated_args: &[TermId],
            sort: SortId,
            model: &CompletedModel,
            manager: &mut TermManager,
        ) -> TermId {
            // Try looking up the function in the model's interpretation table
            if let Some(result) = model.eval_apply(func, evaluated_args) {
                result
            } else {
                // Rebuild with evaluated args and check model for the new term
                let func_name = manager.resolve_str(func).to_string();
                let new_term = manager.mk_apply(&func_name, evaluated_args.iter().copied(), sort);
                model.eval(new_term).unwrap_or(new_term)
            }
        }

        let mut stack: Vec<EvalFrame> = vec![EvalFrame::Enter(root)];
        // The most recently completed subresult; only ever read by a frame
        // popped immediately after the completion that wrote it.
        let mut value: TermId = root;

        while let Some(frame) = stack.pop() {
            match frame {
                EvalFrame::Enter(term) => {
                    if let Some(&cached) = cache.get(&term) {
                        value = cached;
                        continue;
                    }
                    // Check if we have a direct model value
                    if let Some(val) = model.eval(term) {
                        cache.insert(term, val);
                        value = val;
                        continue;
                    }
                    let Some(t) = manager.get(term).cloned() else {
                        // Unknown id: hand it back unchanged (and uncached),
                        // exactly as the recursive version did.
                        value = term;
                        continue;
                    };
                    let term_sort = t.sort;
                    match t.kind {
                        // Constants evaluate to themselves
                        TermKind::True
                        | TermKind::False
                        | TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. }
                        | TermKind::StringLit(_) => {
                            cache.insert(term, term);
                            value = term;
                        }

                        // Boolean connectives
                        TermKind::Not(arg) => {
                            stack.push(EvalFrame::NotArg { term });
                            stack.push(EvalFrame::Enter(arg));
                        }
                        TermKind::And(args) => {
                            if let Some(&first) = args.first() {
                                stack.push(EvalFrame::AndArgs {
                                    term,
                                    args,
                                    next: 1,
                                    all_true: true,
                                });
                                stack.push(EvalFrame::Enter(first));
                            } else {
                                // Empty conjunction: vacuously true.
                                let result = manager.mk_true();
                                cache.insert(term, result);
                                value = result;
                            }
                        }
                        TermKind::Or(args) => {
                            if let Some(&first) = args.first() {
                                stack.push(EvalFrame::OrArgs {
                                    term,
                                    args,
                                    next: 1,
                                    all_false: true,
                                });
                                stack.push(EvalFrame::Enter(first));
                            } else {
                                // Empty disjunction: vacuously false.
                                let result = manager.mk_false();
                                cache.insert(term, result);
                                value = result;
                            }
                        }
                        TermKind::Implies(lhs, rhs) => {
                            stack.push(EvalFrame::ImpliesLhs { term, rhs });
                            stack.push(EvalFrame::Enter(lhs));
                        }

                        // Comparisons
                        TermKind::Eq(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Eq,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Lt(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Lt,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Le(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Le,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Gt(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Gt,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Ge(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Ge,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }

                        // Arithmetic
                        TermKind::Add(args) => {
                            if let Some(&first) = args.first() {
                                stack.push(EvalFrame::NaryArgs {
                                    term,
                                    op: NaryOp::Add,
                                    args,
                                    next: 1,
                                    evaluated: SmallVec::new(),
                                });
                                stack.push(EvalFrame::Enter(first));
                            } else {
                                let result = self.eval_add(&[], manager);
                                cache.insert(term, result);
                                value = result;
                            }
                        }
                        TermKind::Mul(args) => {
                            if let Some(&first) = args.first() {
                                stack.push(EvalFrame::NaryArgs {
                                    term,
                                    op: NaryOp::Mul,
                                    args,
                                    next: 1,
                                    evaluated: SmallVec::new(),
                                });
                                stack.push(EvalFrame::Enter(first));
                            } else {
                                let result = self.eval_mul(&[], manager);
                                cache.insert(term, result);
                                value = result;
                            }
                        }
                        TermKind::Sub(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Sub,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Div(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Div,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Mod(lhs, rhs) => {
                            stack.push(EvalFrame::BinLhs {
                                term,
                                op: BinOp::Mod,
                                rhs,
                            });
                            stack.push(EvalFrame::Enter(lhs));
                        }
                        TermKind::Neg(arg) => {
                            stack.push(EvalFrame::NegArg { term });
                            stack.push(EvalFrame::Enter(arg));
                        }

                        // If-then-else
                        TermKind::Ite(cond, then_br, else_br) => {
                            stack.push(EvalFrame::IteCond {
                                term,
                                then_br,
                                else_br,
                            });
                            stack.push(EvalFrame::Enter(cond));
                        }

                        // Function applications: evaluate args, then look up
                        // in the function table
                        TermKind::Apply { func, args } => {
                            if let Some(&first) = args.first() {
                                stack.push(EvalFrame::ApplyArgs {
                                    term,
                                    sort: term_sort,
                                    func,
                                    args,
                                    next: 1,
                                    evaluated: Vec::new(),
                                });
                                stack.push(EvalFrame::Enter(first));
                            } else {
                                let result = fold_apply(func, &[], term_sort, model, manager);
                                cache.insert(term, result);
                                value = result;
                            }
                        }

                        // Forall: return symbolic -- we only instantiate at
                        // the top level
                        TermKind::Forall { .. } => {
                            cache.insert(term, term);
                            value = term;
                        }

                        // Exists: try to find a witness using default
                        // candidates (see `ExistsState` for the verdict rule).
                        TermKind::Exists { vars, body, .. } => {
                            let candidate_lists =
                                self.build_exists_candidate_lists(&vars, model, manager);
                            if candidate_lists.is_empty()
                                || candidate_lists.iter().any(|c| c.is_empty())
                            {
                                // No candidates → cannot determine; symbolic.
                                cache.insert(term, body);
                                value = body;
                            } else {
                                // Enumerate combinations (simplified for the
                                // single-var case; limit total).
                                let total_combos: usize = candidate_lists
                                    .iter()
                                    .map(|c| c.len())
                                    .product::<usize>()
                                    .min(50);
                                let indices = vec![0usize; candidate_lists.len()];
                                stack.push(EvalFrame::ExistsAdvance(Box::new(ExistsState {
                                    term,
                                    body,
                                    vars,
                                    candidate_lists,
                                    indices,
                                    combo_count: 0,
                                    total_combos,
                                    found_true: false,
                                    found_symbolic: false,
                                    all_false: true,
                                })));
                            }
                        }

                        // Array select: evaluate index, then try multiple
                        // model lookups.
                        //
                        // Key insight: the model stores values keyed by the
                        // *original* term graph (e.g. select(a, 3)), not by
                        // any model-evaluated variant.  If we first evaluate
                        // the array `a` via the model (obtaining some value V)
                        // and then build select(V, 3), the resulting TermId
                        // won't match the model's entry for select(a, 3).
                        // Therefore, we first try looking up
                        // `select(original_array, evaluated_index)` before
                        // falling back.
                        TermKind::Select(array, index) => {
                            stack.push(EvalFrame::SelectIndex { term, array });
                            stack.push(EvalFrame::Enter(index));
                        }

                        // Variables that haven't been substituted -- look up
                        // in model or return as-is
                        TermKind::Var(_) => {
                            let result = model.eval(term).unwrap_or(term);
                            cache.insert(term, result);
                            value = result;
                        }

                        // Anything else: try simplification
                        _ => {
                            let result = manager.simplify(term);
                            cache.insert(term, result);
                            value = result;
                        }
                    }
                }
                EvalFrame::NotArg { term } => {
                    let eval_arg = value;
                    let result = if let Some(arg_t) = manager.get(eval_arg) {
                        match arg_t.kind {
                            TermKind::True => manager.mk_false(),
                            TermKind::False => manager.mk_true(),
                            _ => manager.mk_not(eval_arg),
                        }
                    } else {
                        manager.mk_not(eval_arg)
                    };
                    cache.insert(term, result);
                    value = result;
                }
                EvalFrame::AndArgs {
                    term,
                    args,
                    next,
                    mut all_true,
                } => {
                    // `value` is the evaluation of `args[next - 1]`.
                    let mut decided = false;
                    if let Some(arg_t) = manager.get(value) {
                        match arg_t.kind {
                            TermKind::False => decided = true,
                            TermKind::True => { /* continue */ }
                            _ => all_true = false,
                        }
                    } else {
                        all_true = false;
                    }
                    if decided {
                        // A definite False decides the conjunction; the
                        // remaining conjuncts stay unevaluated (the exact
                        // recursive short-circuit).
                        let false_val = manager.mk_false();
                        cache.insert(term, false_val);
                        value = false_val;
                    } else if let Some(&next_arg) = args.get(next) {
                        stack.push(EvalFrame::AndArgs {
                            term,
                            args,
                            next: next + 1,
                            all_true,
                        });
                        stack.push(EvalFrame::Enter(next_arg));
                    } else if all_true {
                        let result = manager.mk_true();
                        cache.insert(term, result);
                        value = result;
                    } else {
                        // Not fully evaluated -- return symbolic
                        cache.insert(term, term);
                        value = term;
                    }
                }
                EvalFrame::OrArgs {
                    term,
                    args,
                    next,
                    mut all_false,
                } => {
                    let mut decided = false;
                    if let Some(arg_t) = manager.get(value) {
                        match arg_t.kind {
                            TermKind::True => decided = true,
                            TermKind::False => { /* continue */ }
                            _ => all_false = false,
                        }
                    } else {
                        all_false = false;
                    }
                    if decided {
                        // A definite True decides the disjunction; the
                        // remaining disjuncts stay unevaluated.
                        let true_val = manager.mk_true();
                        cache.insert(term, true_val);
                        value = true_val;
                    } else if let Some(&next_arg) = args.get(next) {
                        stack.push(EvalFrame::OrArgs {
                            term,
                            args,
                            next: next + 1,
                            all_false,
                        });
                        stack.push(EvalFrame::Enter(next_arg));
                    } else if all_false {
                        let result = manager.mk_false();
                        cache.insert(term, result);
                        value = result;
                    } else {
                        cache.insert(term, term);
                        value = term;
                    }
                }
                EvalFrame::ImpliesLhs { term, rhs } => {
                    let eval_lhs = value;
                    if let Some(lhs_t) = manager.get(eval_lhs) {
                        match lhs_t.kind {
                            TermKind::False => {
                                let result = manager.mk_true();
                                cache.insert(term, result);
                                value = result;
                            }
                            TermKind::True => {
                                // The conclusion's value is the implication's
                                // value.
                                stack.push(EvalFrame::Forward { term });
                                stack.push(EvalFrame::Enter(rhs));
                            }
                            _ => {
                                stack.push(EvalFrame::ImpliesRhs {
                                    term,
                                    lhs_eval: eval_lhs,
                                });
                                stack.push(EvalFrame::Enter(rhs));
                            }
                        }
                    } else {
                        // Unresolvable premise: rebuild from both halves,
                        // exactly like the symbolic case.
                        stack.push(EvalFrame::ImpliesRhs {
                            term,
                            lhs_eval: eval_lhs,
                        });
                        stack.push(EvalFrame::Enter(rhs));
                    }
                }
                EvalFrame::ImpliesRhs { term, lhs_eval } => {
                    let result = manager.mk_implies(lhs_eval, value);
                    cache.insert(term, result);
                    value = result;
                }
                EvalFrame::Forward { term } => {
                    cache.insert(term, value);
                }
                EvalFrame::BinLhs { term, op, rhs } => {
                    stack.push(EvalFrame::BinRhs {
                        term,
                        op,
                        lhs_eval: value,
                    });
                    stack.push(EvalFrame::Enter(rhs));
                }
                EvalFrame::BinRhs { term, op, lhs_eval } => {
                    let eval_rhs = value;
                    let result = match op {
                        BinOp::Eq => self.eval_eq(lhs_eval, eval_rhs, manager),
                        BinOp::Lt => self.eval_lt(lhs_eval, eval_rhs, manager),
                        BinOp::Le => self.eval_le(lhs_eval, eval_rhs, manager),
                        BinOp::Gt => self.eval_gt(lhs_eval, eval_rhs, manager),
                        BinOp::Ge => self.eval_ge(lhs_eval, eval_rhs, manager),
                        BinOp::Sub => self.eval_sub(lhs_eval, eval_rhs, manager),
                        BinOp::Div => self.eval_div(lhs_eval, eval_rhs, manager),
                        BinOp::Mod => self.eval_modulo(lhs_eval, eval_rhs, manager),
                    };
                    cache.insert(term, result);
                    value = result;
                }
                EvalFrame::NaryArgs {
                    term,
                    op,
                    args,
                    next,
                    mut evaluated,
                } => {
                    evaluated.push(value);
                    if let Some(&next_arg) = args.get(next) {
                        stack.push(EvalFrame::NaryArgs {
                            term,
                            op,
                            args,
                            next: next + 1,
                            evaluated,
                        });
                        stack.push(EvalFrame::Enter(next_arg));
                    } else {
                        let result = match op {
                            NaryOp::Add => self.eval_add(&evaluated, manager),
                            NaryOp::Mul => self.eval_mul(&evaluated, manager),
                        };
                        cache.insert(term, result);
                        value = result;
                    }
                }
                EvalFrame::NegArg { term } => {
                    let result = self.eval_neg(value, manager);
                    cache.insert(term, result);
                    value = result;
                }
                EvalFrame::IteCond {
                    term,
                    then_br,
                    else_br,
                } => {
                    let eval_cond = value;
                    if let Some(cond_t) = manager.get(eval_cond) {
                        match cond_t.kind {
                            TermKind::True => {
                                stack.push(EvalFrame::Forward { term });
                                stack.push(EvalFrame::Enter(then_br));
                            }
                            TermKind::False => {
                                stack.push(EvalFrame::Forward { term });
                                stack.push(EvalFrame::Enter(else_br));
                            }
                            _ => {
                                // Can't determine branch -- evaluate both and
                                // return ite
                                stack.push(EvalFrame::IteThen {
                                    term,
                                    cond_eval: eval_cond,
                                    else_br,
                                });
                                stack.push(EvalFrame::Enter(then_br));
                            }
                        }
                    } else {
                        cache.insert(term, term);
                        value = term;
                    }
                }
                EvalFrame::IteThen {
                    term,
                    cond_eval,
                    else_br,
                } => {
                    stack.push(EvalFrame::IteElse {
                        term,
                        cond_eval,
                        then_eval: value,
                    });
                    stack.push(EvalFrame::Enter(else_br));
                }
                EvalFrame::IteElse {
                    term,
                    cond_eval,
                    then_eval,
                } => {
                    let result = manager.mk_ite(cond_eval, then_eval, value);
                    cache.insert(term, result);
                    value = result;
                }
                EvalFrame::ApplyArgs {
                    term,
                    sort,
                    func,
                    args,
                    next,
                    mut evaluated,
                } => {
                    evaluated.push(value);
                    if let Some(&next_arg) = args.get(next) {
                        stack.push(EvalFrame::ApplyArgs {
                            term,
                            sort,
                            func,
                            args,
                            next: next + 1,
                            evaluated,
                        });
                        stack.push(EvalFrame::Enter(next_arg));
                    } else {
                        let result = fold_apply(func, &evaluated, sort, model, manager);
                        cache.insert(term, result);
                        value = result;
                    }
                }
                EvalFrame::SelectIndex { term, array } => {
                    let eval_index = value;
                    // Try 1: select(original_array, evaluated_index) — this
                    // matches the term graph created during MBQI
                    // instantiation encoding.
                    let select_with_orig_array = manager.mk_select(array, eval_index);
                    if let Some(val) = model.eval(select_with_orig_array) {
                        cache.insert(term, val);
                        value = val;
                    } else if let Some(val) = model.eval(term) {
                        // Try 2: the un-modified original term (before
                        // substitution)
                        cache.insert(term, val);
                        value = val;
                    } else {
                        // Try 3: also evaluate the array in case it resolves
                        stack.push(EvalFrame::SelectArray {
                            term,
                            index_eval: eval_index,
                        });
                        stack.push(EvalFrame::Enter(array));
                    }
                }
                EvalFrame::SelectArray { term, index_eval } => {
                    let new_select = manager.mk_select(value, index_eval);
                    // If even this misses, return the rebuilt select as a
                    // symbolic residual.  This will make
                    // `all_evaluations_ground` false, causing MBQI to return
                    // Unknown (not Satisfied), which triggers blind
                    // instantiation as a fallback.
                    let result = model.eval(new_select).unwrap_or(new_select);
                    cache.insert(term, result);
                    value = result;
                }
                EvalFrame::ExistsAdvance(st) => {
                    if st.combo_count >= st.total_combos {
                        value = finish_exists(&st, manager, cache);
                        continue;
                    }
                    // Build assignment for this combination
                    let mut subst: FxHashMap<Spur, TermId> = FxHashMap::default();
                    for (i, &(var_name, _sort)) in st.vars.iter().enumerate() {
                        if let Some(&candidate) = st.candidate_lists[i].get(st.indices[i]) {
                            subst.insert(var_name, candidate);
                        }
                    }
                    // Apply substitution and evaluate
                    let substituted = self.apply_substitution(st.body, &subst, manager);
                    stack.push(EvalFrame::ExistsJudge(st));
                    stack.push(EvalFrame::Enter(substituted));
                }
                EvalFrame::ExistsJudge(mut st) => {
                    let evaluated = value;
                    let mut witness = false;
                    if let Some(eval_t) = manager.get(evaluated) {
                        match eval_t.kind {
                            TermKind::True => witness = true,
                            TermKind::False => {
                                // this candidate is False, keep checking
                            }
                            _ => {
                                // symbolic
                                st.found_symbolic = true;
                                st.all_false = false;
                            }
                        }
                    } else {
                        st.found_symbolic = true;
                        st.all_false = false;
                    }
                    if witness {
                        // Witness found: stop enumerating (the recursive
                        // version's `break` before the count/odometer step).
                        st.found_true = true;
                        value = finish_exists(&st, manager, cache);
                        continue;
                    }
                    st.combo_count += 1;
                    // Increment indices (odometer)
                    let mut carry = true;
                    for (i, idx) in st.indices.iter_mut().enumerate() {
                        if carry {
                            *idx += 1;
                            let limit = st.candidate_lists.get(i).map_or(1, |c| c.len());
                            if *idx >= limit {
                                *idx = 0;
                            } else {
                                carry = false;
                            }
                        }
                    }
                    if carry {
                        // All combinations tried.
                        value = finish_exists(&st, manager, cache);
                    } else {
                        stack.push(EvalFrame::ExistsAdvance(st));
                    }
                }
            }
        }
        value
    }

    /// Build the per-variable candidate lists for an inline `Exists`
    /// evaluation: universe elements, same-sort model values, then default
    /// candidates for the built-in sorts, truncated to
    /// `max_candidates_per_var` per variable.  (The candidate-selection half
    /// of the old `evaluate_exists_inline`, unchanged.)
    fn build_exists_candidate_lists(
        &self,
        vars: &[(Spur, SortId)],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Vec<TermId>> {
        let mut candidate_lists: Vec<Vec<TermId>> = Vec::new();
        for &(_var_name, sort) in vars {
            let mut cands = Vec::new();
            // Use universe if available
            if let Some(universe) = model.universe(sort) {
                cands.extend_from_slice(universe);
            }
            // Add values from model assignments
            for (&term, &value) in &model.assignments {
                if let Some(t) = manager.get(term)
                    && t.sort == sort
                    && !cands.contains(&value)
                {
                    cands.push(value);
                }
            }
            // Add default candidates for known sorts
            if sort == manager.sorts.int_sort {
                for i in -2i64..=5 {
                    let val = manager.mk_int(BigInt::from(i));
                    if !cands.contains(&val) {
                        cands.push(val);
                    }
                }
            } else if sort == manager.sorts.bool_sort {
                let t = manager.mk_true();
                let f = manager.mk_false();
                if !cands.contains(&t) {
                    cands.push(t);
                }
                if !cands.contains(&f) {
                    cands.push(f);
                }
            }
            cands.truncate(self.max_candidates_per_var);
            candidate_lists.push(cands);
        }
        candidate_lists
    }

    /// Evaluate equality
    fn eval_eq(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        if lhs == rhs {
            return manager.mk_true();
        }

        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            match (&l.kind, &r.kind) {
                (TermKind::IntConst(a), TermKind::IntConst(b)) => {
                    if a == b {
                        manager.mk_true()
                    } else {
                        manager.mk_false()
                    }
                }
                (TermKind::RealConst(a), TermKind::RealConst(b)) => {
                    if a == b {
                        manager.mk_true()
                    } else {
                        manager.mk_false()
                    }
                }
                (TermKind::True, TermKind::True) | (TermKind::False, TermKind::False) => {
                    manager.mk_true()
                }
                (TermKind::True, TermKind::False) | (TermKind::False, TermKind::True) => {
                    manager.mk_false()
                }
                _ => manager.mk_eq(lhs, rhs),
            }
        } else {
            manager.mk_eq(lhs, rhs)
        }
    }

    /// Evaluate less-than
    fn eval_lt(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                if a < b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
            if let (TermKind::RealConst(a), TermKind::RealConst(b)) = (&l.kind, &r.kind) {
                if a < b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
        }

        manager.mk_lt(lhs, rhs)
    }

    /// Evaluate less-than-or-equal
    fn eval_le(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                if a <= b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
            if let (TermKind::RealConst(a), TermKind::RealConst(b)) = (&l.kind, &r.kind) {
                if a <= b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
        }

        manager.mk_le(lhs, rhs)
    }

    /// Evaluate greater-than
    fn eval_gt(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        self.eval_lt(rhs, lhs, manager)
    }

    /// Evaluate greater-than-or-equal
    fn eval_ge(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        self.eval_le(rhs, lhs, manager)
    }

    /// Evaluate addition
    fn eval_add(&self, args: &[TermId], manager: &mut TermManager) -> TermId {
        let mut result = BigInt::from(0);
        let mut all_ints = true;

        for &arg in args {
            if let Some(arg_t) = manager.get(arg) {
                if let TermKind::IntConst(val) = &arg_t.kind {
                    result += val;
                } else {
                    all_ints = false;
                    break;
                }
            } else {
                all_ints = false;
                break;
            }
        }

        if all_ints {
            manager.mk_int(result)
        } else {
            manager.mk_add(args.iter().copied())
        }
    }

    /// Evaluate multiplication
    fn eval_mul(&self, args: &[TermId], manager: &mut TermManager) -> TermId {
        let mut result = BigInt::from(1);
        let mut all_ints = true;

        for &arg in args {
            if let Some(arg_t) = manager.get(arg) {
                if let TermKind::IntConst(val) = &arg_t.kind {
                    result *= val;
                } else {
                    all_ints = false;
                    break;
                }
            } else {
                all_ints = false;
                break;
            }
        }

        if all_ints {
            manager.mk_int(result)
        } else {
            manager.mk_mul(args.iter().copied())
        }
    }

    /// Evaluate subtraction
    fn eval_sub(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                return manager.mk_int(a - b);
            }
        }

        manager.mk_sub(lhs, rhs)
    }

    /// Evaluate integer division using SMT-LIB Euclidean semantics.
    ///
    /// SMT-LIB `div` is Euclidean: `(div a b)` is the unique `q` with
    /// `a = b*q + r` and `0 <= r < |b|`.  This differs from Rust's truncated
    /// `/` for negative operands (e.g. `(div -7 2) = -4`, not `-3`), so we must
    /// floor-adjust.  Mirrors the canonical implementation in
    /// `oxiz-core` `rewrite/arith.rs` / `model/evaluator.rs`.  Division by zero
    /// is left uninterpreted (never folded).
    fn eval_div(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                if *b != BigInt::from(0) {
                    let (q, _r) = euclidean_div_rem(a, b);
                    return manager.mk_int(q);
                }
            }
        }

        manager.mk_div(lhs, rhs)
    }

    /// Evaluate modulo using SMT-LIB Euclidean semantics.
    ///
    /// SMT-LIB `mod` is Euclidean: `(mod a b)` is always in `[0, |b|)`
    /// (e.g. `(mod -7 2) = 1`, not `-1`), unlike Rust's `%` which takes the
    /// sign of the dividend.  Modulo by zero is left uninterpreted.
    fn eval_modulo(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                if *b != BigInt::from(0) {
                    let (_q, r) = euclidean_div_rem(a, b);
                    return manager.mk_int(r);
                }
            }
        }

        manager.mk_mod(lhs, rhs)
    }

    /// Evaluate negation
    fn eval_neg(&self, arg: TermId, manager: &mut TermManager) -> TermId {
        if let Some(arg_t) = manager.get(arg) {
            if let TermKind::IntConst(val) = &arg_t.kind {
                return manager.mk_int(-val);
            }
        }

        manager.mk_neg(arg)
    }

    /// Check if an evaluated term is a counterexample.
    ///
    /// Policy:
    /// - For ∀x.φ(x): only a CONCRETE False means "definitely a counterexample".
    ///   True => definitely not a counterexample (model satisfies this instance).
    ///   Symbolic residual => not a counterexample; the instance is unknown.
    ///   This avoids generating spurious instantiation lemmas for symbolic
    ///   evaluations (e.g. unconstrained select terms), which would create
    ///   unnecessary arithmetic constraints and cause false UNSAT.
    /// - For ∃x.φ(x): only a concrete True means "definitely a witness".
    ///   False or symbolic => not a witness (conservative).
    fn is_counterexample(
        &self,
        evaluated: TermId,
        is_universal: bool,
        manager: &TermManager,
    ) -> bool {
        let Some(eval_t) = manager.get(evaluated) else {
            // Cannot resolve the term at all -- not a counterexample (unknown).
            return false;
        };

        if is_universal {
            // For ∀x.φ(x): only concrete False is a genuine counterexample.
            // Symbolic residuals (neither True nor False) mean we could not
            // evaluate the body under the current model -- do NOT treat these
            // as counterexamples to avoid injecting unnecessary lemmas.
            matches!(eval_t.kind, TermKind::False)
        } else {
            // For ∃x.φ(x): only concrete True counts as a witness.
            matches!(eval_t.kind, TermKind::True)
        }
    }

    /// Check whether an evaluated term resolved to a concrete boolean value
    /// (True or False) as opposed to a symbolic residual.
    fn is_ground_boolean(&self, evaluated: TermId, manager: &TermManager) -> bool {
        let Some(eval_t) = manager.get(evaluated) else {
            return false;
        };
        matches!(eval_t.kind, TermKind::True | TermKind::False)
    }

    /// Set generation bound for candidate selection
    pub fn set_generation_bound(&mut self, bound: u32) {
        self.generation_bound = bound;
    }

    /// Inject extra candidates from outside (e.g. Skolem function applications).
    ///
    /// These are merged into the per-sort candidate cache so they appear in
    /// every subsequent `build_candidate_lists` call.
    pub fn inject_extra_candidates(&mut self, extras: &FxHashMap<SortId, Vec<TermId>>) {
        for (&sort, terms) in extras {
            let entry = self.candidate_cache.entry(sort).or_default();
            for &t in terms {
                if !entry.contains(&t) {
                    entry.push(t);
                }
            }
        }
    }

    /// Clear the candidate cache
    pub fn clear_cache(&mut self) {
        self.candidate_cache.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &CexStats {
        &self.stats
    }
}

impl Default for CounterExampleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Refinement strategy for narrowing the search space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementStrategy {
    /// No refinement
    None,
    /// Block found counterexamples
    BlockCounterexamples,
    /// Learn from conflicts
    ConflictLearning,
    /// Generalize from counterexamples
    Generalization,
}

/// Statistics for counterexample generation
#[derive(Debug, Clone, Default)]
pub struct CexStats {
    /// Number of search attempts
    pub num_searches: usize,
    /// Number of counterexamples found
    pub num_counterexamples_found: usize,
    /// Number of combinations tried
    pub num_combinations_tried: usize,
    /// Number of timeouts
    pub num_timeouts: usize,
    /// Total time spent
    #[cfg(feature = "std")]
    pub total_time: Duration,
}

impl fmt::Display for CexStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Counterexample Statistics:")?;
        writeln!(f, "  Searches: {}", self.num_searches)?;
        writeln!(f, "  CEX found: {}", self.num_counterexamples_found)?;
        writeln!(f, "  Combinations tried: {}", self.num_combinations_tried)?;
        writeln!(f, "  Timeouts: {}", self.num_timeouts)?;
        #[cfg(feature = "std")]
        writeln!(f, "  Total time: {:.2}ms", self.total_time.as_millis())?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "counterexample_tests.rs"]
mod tests;
