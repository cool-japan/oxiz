//! The Craig interpolation engine: node coloring and interpolant computation.

use super::config::{InterpolationAlgorithm, InterpolationConfig};
use super::error::InterpolationError;
use super::parsing::{PivotLocality, parse_clause_literals, resolution_pivot};
use super::partition::{InterpolantColor, InterpolantPartition, Symbol};
use super::term::InterpolantTerm;
use crate::premise::PremiseTracker;
use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::{FxHashMap, FxHashSet};

/// Work item for the iterative post-order walks over the proof DAG.
#[derive(Debug, Clone, Copy)]
enum WalkFrame {
    /// The node's premises have not been scheduled yet.
    Enter(ProofNodeId),
    /// All premises of the node have been processed.
    Exit(ProofNodeId),
}

/// Craig interpolation engine
#[derive(Debug)]
pub struct CraigInterpolator {
    /// Configuration
    config: InterpolationConfig,
    /// Partition of premises
    partition: InterpolantPartition,
    /// Premise tracker
    premise_tracker: PremiseTracker,
    /// Computed colors for proof nodes
    pub(crate) colors: FxHashMap<ProofNodeId, InterpolantColor>,
    /// Computed interpolants for proof nodes
    interpolants: FxHashMap<ProofNodeId, InterpolantTerm>,
    /// Statistics
    stats: InterpolationStats,
    /// Axiom nodes whose color was determined directly from the user's
    /// partition (via premise-tracker lookup), computed once per `extract`.
    pub(crate) direct_axiom_colors: FxHashMap<ProofNodeId, InterpolantColor>,
    /// Vocabulary observed on directly-colored A axioms.
    known_a_symbols: FxHashSet<Symbol>,
    /// Vocabulary observed on directly-colored B axioms.
    known_b_symbols: FxHashSet<Symbol>,
    /// Shared/global vocabulary: `known_a_symbols ∩ known_b_symbols`, unioned
    /// with any symbols explicitly declared shared on the partition.
    pub(crate) global_symbols: FxHashSet<Symbol>,
}

/// Statistics about interpolation computation
#[derive(Debug, Default, Clone)]
pub struct InterpolationStats {
    /// Number of proof nodes processed
    pub nodes_processed: usize,
    /// Number of A-colored nodes
    pub a_nodes: usize,
    /// Number of B-colored nodes
    pub b_nodes: usize,
    /// Number of AB-colored (mixed) nodes
    pub ab_nodes: usize,
    /// Number of resolution steps
    pub resolution_steps: usize,
    /// Number of theory lemmas
    pub theory_lemmas: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Time spent in interpolation (microseconds)
    pub time_us: u64,
}

impl CraigInterpolator {
    /// Create a new interpolator
    #[must_use]
    pub fn new(
        config: InterpolationConfig,
        partition: InterpolantPartition,
        premise_tracker: PremiseTracker,
    ) -> Self {
        Self {
            config,
            partition,
            premise_tracker,
            colors: FxHashMap::default(),
            interpolants: FxHashMap::default(),
            stats: InterpolationStats::default(),
            direct_axiom_colors: FxHashMap::default(),
            known_a_symbols: FxHashSet::default(),
            known_b_symbols: FxHashSet::default(),
            global_symbols: FxHashSet::default(),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_partition(partition: InterpolantPartition) -> Self {
        Self::new(
            InterpolationConfig::default(),
            partition,
            PremiseTracker::new(),
        )
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &InterpolationStats {
        &self.stats
    }

    /// Extract an interpolant from a proof.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolationError::MixedAxiom`] if some axiom's vocabulary
    /// spans both partitions without being explicitly assigned to either by
    /// the caller (this leaf-based interpolation system cannot soundly
    /// decompose such a node).
    pub fn extract(&mut self, proof: &Proof) -> Result<InterpolantTerm, InterpolationError> {
        let start = oxiz_time::Instant::now();

        let root = proof.root().ok_or(InterpolationError::NoRoot)?;

        // Determine the ground-truth A/B coloring and shared vocabulary from
        // the user's explicit premise partition before doing anything else.
        self.precompute_axiom_partition(proof);

        // Phase 1: Compute colors for all nodes (bottom-up)
        self.compute_colors(proof, root)?;

        // Phase 2: Compute interpolants (bottom-up)
        let result = self.compute_interpolant(proof, root)?;

        // Simplify if configured, bounding the simplification recursion by
        // the configured depth (see `InterpolationConfig::max_simplify_depth`)
        // since the interpolant's depth is driven by the size of the UNSAT
        // proof being interpolated, not by anything this crate controls.
        let final_result = if self.config.simplify_interpolants {
            result.simplify_bounded(self.config.max_simplify_depth)
        } else {
            result
        };

        self.stats.time_us = start.elapsed().as_micros() as u64;

        Ok(final_result)
    }

    /// Precompute, from the proof's axioms, which are directly identifiable
    /// as A- or B-premises via the caller-supplied partition (matched by
    /// conclusion text against the premise tracker), and derive the
    /// resulting known vocabulary for each side plus the shared vocabulary.
    pub(crate) fn precompute_axiom_partition(&mut self, proof: &Proof) {
        self.direct_axiom_colors.clear();
        let mut a_symbols = FxHashSet::default();
        let mut b_symbols = FxHashSet::default();

        for node in proof.nodes() {
            let ProofStep::Axiom { conclusion } = &node.step else {
                continue;
            };
            let Some(premise_id) = self.premise_tracker.get_id(conclusion) else {
                continue;
            };
            let in_a = self.partition.is_a_premise(premise_id);
            let in_b = self.partition.is_b_premise(premise_id);
            let color = match (in_a, in_b) {
                (true, false) => InterpolantColor::A,
                (false, true) => InterpolantColor::B,
                (true, true) => InterpolantColor::AB,
                (false, false) => continue,
            };
            self.direct_axiom_colors.insert(node.id, color);

            let mut symbols = FxHashSet::default();
            for lit in parse_clause_literals(conclusion) {
                lit.collect_symbols(&mut symbols);
            }
            match color {
                InterpolantColor::A => a_symbols.extend(symbols),
                InterpolantColor::B => b_symbols.extend(symbols),
                InterpolantColor::AB => {
                    a_symbols.extend(symbols.iter().cloned());
                    b_symbols.extend(symbols);
                }
            }
        }

        self.known_a_symbols = a_symbols;
        self.known_b_symbols = b_symbols;
        self.global_symbols = self
            .known_a_symbols
            .intersection(&self.known_b_symbols)
            .cloned()
            .collect();
        self.global_symbols
            .extend(self.partition.shared_symbols().iter().cloned());
    }

    /// Classify a symbol's locality against the known A/B vocabulary.
    /// Symbols seen on neither side default to A-local, which is harmless:
    /// such a symbol cannot be shared, so it never triggers the global-pivot
    /// case.
    fn classify_symbol(&self, symbol: &Symbol) -> PivotLocality {
        if self.global_symbols.contains(symbol) {
            PivotLocality::Global
        } else if self.known_b_symbols.contains(symbol) {
            PivotLocality::BLocal
        } else {
            PivotLocality::ALocal
        }
    }

    /// Project a clause's literals down to those entirely within the shared
    /// (global) vocabulary -- the only literals a sound interpolant may
    /// mention.
    fn project_clause_to_shared(&self, conclusion: &str) -> Vec<InterpolantTerm> {
        parse_clause_literals(conclusion)
            .into_iter()
            .filter(|lit| {
                let mut symbols = FxHashSet::default();
                lit.collect_symbols(&mut symbols);
                !symbols.is_empty() && symbols.iter().all(|s| self.global_symbols.contains(s))
            })
            .collect()
    }

    /// Compute colors for proof nodes.
    ///
    /// Iterative post-order walk over the proof DAG using an explicit heap
    /// stack. Refutation depth here is driven by problem hardness rather than
    /// by input nesting, so a legitimately hard `--interpolate` run reaches
    /// depths a recursive descent cannot survive. A cycle in the proof is
    /// reported as an error instead of recursing forever.
    fn compute_colors(
        &mut self,
        proof: &Proof,
        node_id: ProofNodeId,
    ) -> Result<InterpolantColor, InterpolationError> {
        let mut stack = vec![WalkFrame::Enter(node_id)];
        let mut in_progress: FxHashSet<ProofNodeId> = FxHashSet::default();

        while let Some(frame) = stack.pop() {
            match frame {
                WalkFrame::Enter(id) => {
                    if self.colors.contains_key(&id) {
                        self.stats.cache_hits += 1;
                        continue;
                    }

                    let node = proof
                        .get_node(id)
                        .ok_or(InterpolationError::NodeNotFound(id))?;

                    self.stats.nodes_processed += 1;

                    match &node.step {
                        ProofStep::Axiom { conclusion } => {
                            let color = self.color_axiom(id, conclusion);
                            self.record_color(id, color);
                        }
                        ProofStep::Inference { premises, rule, .. } => {
                            // Track rule types
                            if rule == "resolution" {
                                self.stats.resolution_steps += 1;
                            } else if rule.starts_with("theory") {
                                self.stats.theory_lemmas += 1;
                            }

                            if !in_progress.insert(id) {
                                return Err(InterpolationError::NodeNotFound(id));
                            }
                            stack.push(WalkFrame::Exit(id));
                            stack.extend(premises.iter().rev().copied().map(WalkFrame::Enter));
                        }
                    }
                }
                WalkFrame::Exit(id) => {
                    in_progress.remove(&id);
                    let node = proof
                        .get_node(id)
                        .ok_or(InterpolationError::NodeNotFound(id))?;
                    let ProofStep::Inference { premises, .. } = &node.step else {
                        continue;
                    };

                    // Inference nodes are colored based on their premises
                    let mut has_a = false;
                    let mut has_b = false;

                    for premise_id in premises {
                        let premise_color = self
                            .colors
                            .get(premise_id)
                            .copied()
                            .ok_or(InterpolationError::NodeNotFound(*premise_id))?;
                        match premise_color {
                            InterpolantColor::A => has_a = true,
                            InterpolantColor::B => has_b = true,
                            InterpolantColor::AB => {
                                has_a = true;
                                has_b = true;
                            }
                        }
                    }

                    let color = if has_a && has_b {
                        InterpolantColor::AB
                    } else if has_b {
                        InterpolantColor::B
                    } else {
                        InterpolantColor::A
                    };
                    self.record_color(id, color);
                }
            }
        }

        self.colors
            .get(&node_id)
            .copied()
            .ok_or(InterpolationError::NodeNotFound(node_id))
    }

    /// Record a node's color and fold it into the per-color statistics.
    fn record_color(&mut self, node_id: ProofNodeId, color: InterpolantColor) {
        match color {
            InterpolantColor::A => self.stats.a_nodes += 1,
            InterpolantColor::B => self.stats.b_nodes += 1,
            InterpolantColor::AB => self.stats.ab_nodes += 1,
        }
        self.colors.insert(node_id, color);
    }

    /// Color an axiom node using the user's explicit partition when
    /// available, falling back to a McMillan-style symbol-membership
    /// heuristic for axioms outside that partition (typically theory lemmas
    /// synthesized during solving rather than original user assertions).
    pub(crate) fn color_axiom(&self, node_id: ProofNodeId, conclusion: &str) -> InterpolantColor {
        if let Some(&color) = self.direct_axiom_colors.get(&node_id) {
            return color;
        }

        let mut symbols = FxHashSet::default();
        for lit in parse_clause_literals(conclusion) {
            lit.collect_symbols(&mut symbols);
        }
        let touches_a = symbols.iter().any(|s| self.known_a_symbols.contains(s));
        let touches_b = symbols.iter().any(|s| self.known_b_symbols.contains(s));
        match (touches_a, touches_b) {
            (true, false) => InterpolantColor::A,
            (false, true) => InterpolantColor::B,
            (true, true) => InterpolantColor::AB,
            (false, false) => InterpolantColor::A,
        }
    }

    /// Compute interpolant for a proof node.
    ///
    /// Iterative post-order walk over the proof DAG (explicit heap stack); see
    /// [`Self::compute_colors`]. Partial results are always memoized for the
    /// duration of the walk, so a shared premise is computed once even when
    /// `config.enable_caching` is off; that flag continues to control only
    /// whether the results outlive the call in `self.interpolants`.
    fn compute_interpolant(
        &mut self,
        proof: &Proof,
        node_id: ProofNodeId,
    ) -> Result<InterpolantTerm, InterpolationError> {
        let mut computed: FxHashMap<ProofNodeId, InterpolantTerm> = FxHashMap::default();
        let mut stack = vec![WalkFrame::Enter(node_id)];
        let mut in_progress: FxHashSet<ProofNodeId> = FxHashSet::default();

        while let Some(frame) = stack.pop() {
            let (id, is_exit) = match frame {
                WalkFrame::Enter(id) => (id, false),
                WalkFrame::Exit(id) => (id, true),
            };

            if !is_exit {
                if computed.contains_key(&id) {
                    continue;
                }
                if let Some(interp) = self.interpolants.get(&id) {
                    computed.insert(id, interp.clone());
                    continue;
                }
            } else {
                in_progress.remove(&id);
            }

            let node = proof
                .get_node(id)
                .ok_or(InterpolationError::NodeNotFound(id))?;
            let color = *self
                .colors
                .get(&id)
                .ok_or(InterpolationError::NoColor(id))?;

            let interpolant = match &node.step {
                ProofStep::Axiom { conclusion } => {
                    self.compute_axiom_interpolant(id, color, conclusion)?
                }
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    ..
                } => {
                    if !is_exit {
                        if !in_progress.insert(id) {
                            return Err(InterpolationError::NodeNotFound(id));
                        }
                        stack.push(WalkFrame::Exit(id));
                        stack.extend(premises.iter().rev().copied().map(WalkFrame::Enter));
                        continue;
                    }

                    // Premise interpolants are all available; gather them along
                    // with the raw conclusion text (needed for pivot detection).
                    let mut premise_interpolants = Vec::with_capacity(premises.len());
                    let mut premise_conclusions: Vec<String> = Vec::with_capacity(premises.len());

                    for p in premises {
                        let interp = computed
                            .get(p)
                            .cloned()
                            .ok_or(InterpolationError::NodeNotFound(*p))?;
                        premise_interpolants.push(interp);
                        let concl = proof
                            .get_node(*p)
                            .map(|n| n.conclusion().to_string())
                            .unwrap_or_default();
                        premise_conclusions.push(concl);
                    }
                    let premise_conclusion_refs: Vec<&str> =
                        premise_conclusions.iter().map(String::as_str).collect();

                    self.compute_inference_interpolant(
                        rule,
                        &premise_interpolants,
                        &premise_conclusion_refs,
                        conclusion,
                        color,
                    )
                }
            };

            if self.config.enable_caching {
                self.interpolants.insert(id, interpolant.clone());
            }
            computed.insert(id, interpolant);
        }

        computed
            .remove(&node_id)
            .ok_or(InterpolationError::NodeNotFound(node_id))
    }

    /// Compute interpolant for an axiom.
    ///
    /// Base cases (standard McMillan/Pudlák symmetric system for the `A`
    /// side, and its negation-dual for Huang):
    /// - A-axiom: disjunction of its shared-vocabulary literals (`false` if
    ///   it has none -- it contributes nothing further).
    /// - B-axiom: `true` (McMillan/Pudlák), or the negation-dual conjunction
    ///   of negated shared literals (Huang).
    /// - Mixed (`AB`) axiom: this leaf-based system cannot soundly assign an
    ///   interpolant without knowing which part of the axiom's internal
    ///   derivation belongs to each side, so this is reported as an explicit
    ///   error rather than a fabricated value.
    pub(crate) fn compute_axiom_interpolant(
        &self,
        node_id: ProofNodeId,
        color: InterpolantColor,
        conclusion: &str,
    ) -> Result<InterpolantTerm, InterpolationError> {
        let shared_literals = self.project_clause_to_shared(conclusion);
        let term = match self.config.algorithm {
            InterpolationAlgorithm::McMillan | InterpolationAlgorithm::Pudlak => match color {
                InterpolantColor::A => {
                    if shared_literals.is_empty() {
                        InterpolantTerm::false_val()
                    } else {
                        InterpolantTerm::or(shared_literals)
                    }
                }
                InterpolantColor::B => InterpolantTerm::true_val(),
                InterpolantColor::AB => return Err(InterpolationError::MixedAxiom(node_id)),
            },
            InterpolationAlgorithm::Huang => match color {
                InterpolantColor::A => InterpolantTerm::false_val(),
                InterpolantColor::B => InterpolantTerm::and(
                    shared_literals
                        .into_iter()
                        .map(InterpolantTerm::not)
                        .collect(),
                ),
                InterpolantColor::AB => return Err(InterpolationError::MixedAxiom(node_id)),
            },
        };
        Ok(term)
    }

    /// Compute interpolant for an inference step
    fn compute_inference_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_conclusions: &[&str],
        _conclusion: &str,
        color: InterpolantColor,
    ) -> InterpolantTerm {
        match self.config.algorithm {
            InterpolationAlgorithm::McMillan => {
                self.mcmillan_interpolant(rule, premise_interpolants, premise_conclusions, color)
            }
            InterpolationAlgorithm::Pudlak => {
                self.pudlak_interpolant(rule, premise_interpolants, premise_conclusions, color)
            }
            InterpolationAlgorithm::Huang => {
                self.huang_interpolant(rule, premise_interpolants, premise_conclusions, color)
            }
        }
    }

    /// McMillan's algorithm (weaker/left-biased interpolants)
    ///
    /// For binary resolution on a detected pivot `x`:
    /// - If `x` is A-local: I = I1 ∨ I2
    /// - Otherwise (`x` is B-local or global/shared): I = I1 ∧ I2
    ///
    /// This is verified sound: e.g. for A = {p}, B = {¬p} with shared pivot
    /// `p`, it yields `I = p` (not the trivial `true`/`false` collapse the
    /// unfixed implementation produced).
    pub(crate) fn mcmillan_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_conclusions: &[&str],
        color: InterpolantColor,
    ) -> InterpolantTerm {
        if let Some(pivot) = resolution_pivot(rule, premise_interpolants, premise_conclusions) {
            let i_pos = premise_interpolants[pivot.positive_index].clone();
            let i_neg = premise_interpolants[1 - pivot.positive_index].clone();
            return match self.classify_symbol(&pivot.symbol) {
                PivotLocality::ALocal => InterpolantTerm::or(vec![i_pos, i_neg]),
                PivotLocality::BLocal | PivotLocality::Global => {
                    InterpolantTerm::and(vec![i_pos, i_neg])
                }
            };
        }
        // Fallback for rules whose combination law this system does not
        // (yet) model precisely -- e.g. multi-premise theory rules, or
        // resolution steps whose pivot could not be recovered from the
        // textual clause representation. This preserves the recursively
        // computed premise content (unlike a blind true/false collapse) but
        // is not certified sound for these residual cases.
        match color {
            InterpolantColor::A => InterpolantTerm::true_val(),
            InterpolantColor::B => InterpolantTerm::false_val(),
            InterpolantColor::AB => InterpolantTerm::or(premise_interpolants.to_vec()),
        }
    }

    /// Pudlák's algorithm (symmetric interpolants)
    ///
    /// For binary resolution on a detected pivot `x`:
    /// - A-local: I = I1 ∨ I2
    /// - B-local: I = I1 ∧ I2
    /// - Global/shared: I = (I1 ∨ x) ∧ (I2 ∨ ¬x), the standard symmetric
    ///   McMillan/Pudlák formula that explicitly reintroduces the pivot.
    fn pudlak_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_conclusions: &[&str],
        color: InterpolantColor,
    ) -> InterpolantTerm {
        if let Some(pivot) = resolution_pivot(rule, premise_interpolants, premise_conclusions) {
            let i_pos = premise_interpolants[pivot.positive_index].clone();
            let i_neg = premise_interpolants[1 - pivot.positive_index].clone();
            return match self.classify_symbol(&pivot.symbol) {
                PivotLocality::ALocal => InterpolantTerm::or(vec![i_pos, i_neg]),
                PivotLocality::BLocal => InterpolantTerm::and(vec![i_pos, i_neg]),
                PivotLocality::Global => {
                    let pos_var = InterpolantTerm::Var(pivot.symbol.clone());
                    let neg_var = InterpolantTerm::not(pos_var.clone());
                    InterpolantTerm::and(vec![
                        InterpolantTerm::or(vec![i_pos, pos_var]),
                        InterpolantTerm::or(vec![i_neg, neg_var]),
                    ])
                }
            };
        }
        let is_equality_chain =
            (rule == "transitivity" && premise_interpolants.len() >= 2) || rule == "congruence";
        if is_equality_chain {
            InterpolantTerm::and(premise_interpolants.to_vec())
        } else {
            match color {
                InterpolantColor::A => InterpolantTerm::true_val(),
                InterpolantColor::B => InterpolantTerm::false_val(),
                InterpolantColor::AB => InterpolantTerm::or(premise_interpolants.to_vec()),
            }
        }
    }

    /// Huang's algorithm (stronger/right-biased interpolants) -- the
    /// negation-dual of McMillan's system (see [`InterpolationAlgorithm::Huang`]).
    ///
    /// For binary resolution on a detected pivot `x`:
    /// - A-local or global: I = I1 ∨ I2
    /// - B-local: I = I1 ∧ I2
    pub(crate) fn huang_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_conclusions: &[&str],
        color: InterpolantColor,
    ) -> InterpolantTerm {
        if let Some(pivot) = resolution_pivot(rule, premise_interpolants, premise_conclusions) {
            let i_pos = premise_interpolants[pivot.positive_index].clone();
            let i_neg = premise_interpolants[1 - pivot.positive_index].clone();
            return match self.classify_symbol(&pivot.symbol) {
                PivotLocality::ALocal | PivotLocality::Global => {
                    InterpolantTerm::or(vec![i_pos, i_neg])
                }
                PivotLocality::BLocal => InterpolantTerm::and(vec![i_pos, i_neg]),
            };
        }
        match color {
            InterpolantColor::A => InterpolantTerm::false_val(),
            InterpolantColor::B => InterpolantTerm::true_val(),
            InterpolantColor::AB => InterpolantTerm::and(premise_interpolants.to_vec()),
        }
    }
}
