//! Generalization algorithms for lemma strengthening
//!
//! Implements various generalization techniques:
//! - MIC (Minimal Inductive Clause): Generalize cubes to minimal inductive clauses
//! - CTI (Counterexample To Induction): Use counterexamples to guide generalization
//! - MBP (Model-Based Projection): Project out irrelevant literals
//!
//! Reference: Z3's `muz/spacer/spacer_generalizers.cpp`

use crate::chc::{ChcSystem, PredId};
use crate::smt::SmtSolver;
use oxiz_core::{TermId, TermKind, TermManager};
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, trace};

/// Strategy for literal elimination in MIC
#[derive(Debug, Clone, Copy)]
enum LiteralEliminationStrategy {
    /// Try to eliminate literals in sequential order
    Sequential,
    /// Try to eliminate literals in reverse order
    ReverseSequential,
    /// Try to eliminate larger terms first
    LargestFirst,
    /// Cost-based: eliminate high-cost (complex) literals first
    CostBased,
    /// Frequency-based: eliminate less frequent variables first
    FrequencyBased,
    /// Adaptive: combine multiple strategies dynamically
    Adaptive,
}

/// Errors from generalization
#[derive(Error, Debug)]
pub enum GeneralizationError {
    /// SMT error during generalization
    #[error("SMT error: {0}")]
    Smt(String),
    /// Invalid cube (empty or trivial)
    #[error("invalid cube: {0}")]
    InvalidCube(String),
}

/// Result of generalization
#[derive(Debug, Clone)]
pub struct GeneralizationResult {
    /// The generalized lemma (cube)
    pub lemma: SmallVec<[TermId; 8]>,
    /// Literals dropped during generalization
    pub dropped: SmallVec<[TermId; 4]>,
    /// Whether the result is inductive
    pub is_inductive: bool,
    /// Number of SMT queries used
    pub num_queries: u32,
}

impl GeneralizationResult {
    /// Create a new generalization result
    pub fn new(lemma: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            lemma: lemma.into_iter().collect(),
            dropped: SmallVec::new(),
            is_inductive: false,
            num_queries: 0,
        }
    }

    /// Convert to a single formula (conjunction of literals)
    pub fn to_formula(&self, terms: &mut TermManager) -> TermId {
        if self.lemma.is_empty() {
            terms.mk_true()
        } else if self.lemma.len() == 1 {
            self.lemma[0]
        } else {
            terms.mk_and(self.lemma.iter().copied())
        }
    }
}

/// Generalizer for creating inductive lemmas
pub struct Generalizer<'a> {
    /// Term manager
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Number of queries performed
    num_queries: u32,
}

impl<'a> Generalizer<'a> {
    /// Create a new generalizer
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self {
            terms,
            system,
            num_queries: 0,
        }
    }

    /// Generalize a cube using MIC (Minimal Inductive Clause)
    ///
    /// Given a cube (conjunction of literals) that blocks a bad state,
    /// compute a minimal subset that is still inductive.
    ///
    /// Algorithm:
    /// 1. Start with the full cube
    /// 2. Try to drop each literal one at a time
    /// 3. Check if the remaining cube is still inductive
    /// 4. Keep dropping until no more literals can be removed
    ///
    /// Enhanced with multiple elimination strategies
    pub fn mic(
        &mut self,
        pred: PredId,
        cube: &[TermId],
        level: u32,
        frame_formula: TermId,
    ) -> Result<GeneralizationResult, GeneralizationError> {
        debug!(
            "MIC generalization for predicate {:?} with {} literals",
            pred,
            cube.len()
        );

        if cube.is_empty() {
            return Err(GeneralizationError::InvalidCube(
                "Cannot generalize empty cube".to_string(),
            ));
        }

        // Try different elimination strategies and pick the best result
        let strategies = [
            LiteralEliminationStrategy::Sequential,
            LiteralEliminationStrategy::ReverseSequential,
            LiteralEliminationStrategy::LargestFirst,
            LiteralEliminationStrategy::CostBased,
            LiteralEliminationStrategy::FrequencyBased,
            LiteralEliminationStrategy::Adaptive,
        ];

        let mut best_result = None;
        let mut best_size = cube.len();

        for strategy in &strategies {
            match self.mic_with_strategy(pred, cube, level, frame_formula, *strategy) {
                Ok(result) => {
                    if result.lemma.len() < best_size {
                        best_size = result.lemma.len();
                        best_result = Some(result);
                    }
                }
                Err(e) => {
                    debug!("MIC strategy {:?} failed: {}", strategy, e);
                }
            }
        }

        best_result.ok_or_else(|| {
            GeneralizationError::InvalidCube("All MIC strategies failed".to_string())
        })
    }

    /// MIC with a specific elimination strategy
    fn mic_with_strategy(
        &mut self,
        pred: PredId,
        cube: &[TermId],
        level: u32,
        frame_formula: TermId,
        strategy: LiteralEliminationStrategy,
    ) -> Result<GeneralizationResult, GeneralizationError> {
        let mut result = GeneralizationResult::new(cube.iter().copied());
        let mut current_cube = cube.to_vec();

        // Order literals according to strategy
        let indices = self.order_literals_for_elimination(&current_cube, strategy);

        for &idx in &indices {
            if idx >= current_cube.len() {
                continue;
            }

            // Find the current position of this literal
            let actual_idx = current_cube
                .iter()
                .position(|&lit| lit == cube[idx])
                .unwrap_or(idx);

            if actual_idx >= current_cube.len() {
                continue;
            }

            // Remove literal temporarily
            let removed_lit = current_cube.remove(actual_idx);

            // Build candidate lemma from remaining literals
            let candidate = if current_cube.is_empty() {
                self.terms.mk_true()
            } else if current_cube.len() == 1 {
                current_cube[0]
            } else {
                self.terms.mk_and(current_cube.clone())
            };

            // Check if candidate is inductive
            let mut smt = SmtSolver::new(self.terms, self.system);
            self.num_queries += 1;

            let is_inductive = match smt.is_lemma_inductive(pred, candidate, level, frame_formula) {
                Ok(inductive) => inductive,
                Err(e) => {
                    debug!("SMT error during MIC check: {}", e);
                    false
                }
            };

            if is_inductive {
                // Successfully dropped the literal
                result.dropped.push(removed_lit);
                trace!(
                    "MIC: successfully dropped literal with strategy {:?}",
                    strategy
                );
            } else {
                // Need to keep this literal
                current_cube.insert(actual_idx, removed_lit);
            }
        }

        result.lemma = current_cube.into_iter().collect();
        result.is_inductive = true;
        result.num_queries = self.num_queries;

        debug!(
            "MIC with {:?}: reduced from {} to {} literals",
            strategy,
            cube.len(),
            result.lemma.len()
        );

        Ok(result)
    }

    /// Order literals for elimination based on strategy
    fn order_literals_for_elimination(
        &self,
        cube: &[TermId],
        strategy: LiteralEliminationStrategy,
    ) -> Vec<usize> {
        match strategy {
            LiteralEliminationStrategy::Sequential => (0..cube.len()).collect(),
            LiteralEliminationStrategy::ReverseSequential => (0..cube.len()).rev().collect(),
            LiteralEliminationStrategy::LargestFirst => {
                // Order by term size (larger terms first)
                let mut indices: Vec<usize> = (0..cube.len()).collect();
                indices.sort_by_key(|&i| std::cmp::Reverse(self.term_size(cube[i])));
                indices
            }
            LiteralEliminationStrategy::CostBased => {
                // Order by computational cost (higher cost first)
                let mut indices: Vec<usize> = (0..cube.len()).collect();
                indices.sort_by_key(|&i| std::cmp::Reverse(self.term_cost(cube[i])));
                indices
            }
            LiteralEliminationStrategy::FrequencyBased => {
                // Order by variable frequency (less frequent first)
                let var_freq = self.compute_variable_frequency(cube);
                let mut indices: Vec<usize> = (0..cube.len()).collect();
                indices.sort_by_key(|&i| self.literal_frequency_score(cube[i], &var_freq));
                indices
            }
            LiteralEliminationStrategy::Adaptive => {
                // Adaptive: use heuristics based on cube properties
                // For small cubes: use sequential
                // For medium cubes: use cost-based
                // For large cubes: use frequency-based
                if cube.len() < 5 {
                    (0..cube.len()).collect()
                } else if cube.len() < 15 {
                    let mut indices: Vec<usize> = (0..cube.len()).collect();
                    indices.sort_by_key(|&i| std::cmp::Reverse(self.term_cost(cube[i])));
                    indices
                } else {
                    let var_freq = self.compute_variable_frequency(cube);
                    let mut indices: Vec<usize> = (0..cube.len()).collect();
                    indices.sort_by_key(|&i| self.literal_frequency_score(cube[i], &var_freq));
                    indices
                }
            }
        }
    }

    /// Estimate the size of a term (for prioritization).
    ///
    /// Iterative post-order walk with a memo table. The recursive version
    /// had neither a depth bound nor a memo: it overflowed the stack on
    /// deeply nested input and re-expanded a shared DAG once per path
    /// (`2^n` work for `n` doublings). Because the result is a plain
    /// `usize` with no error channel, a depth cap could only have produced
    /// a silently wrong size, which then mis-orders literal elimination.
    ///
    /// The weights are unchanged: every node counts `1`, and only the kinds
    /// the old `match` enumerated are descended into — anything else
    /// (variables, constants, and operators outside that set) counts as a
    /// single node exactly as before.
    fn term_size(&self, term: TermId) -> usize {
        self.accumulate(term, |_| 1)
    }

    /// Estimate the computational cost of a term.
    /// Higher cost means more expensive to check.
    ///
    /// Same iterative post-order walk and the same per-kind weights as the
    /// recursive version it replaces (`Mul`/`Div`/`Mod` = 10, `Add`/`Sub` =
    /// 5, comparisons = 4, `And`/`Or` = 3, `Not` = 2, everything else = 1);
    /// see [`Self::term_size`] for why recursion had to go.
    fn term_cost(&self, term: TermId) -> usize {
        use oxiz_core::TermKind;

        self.accumulate(term, |kind| match kind {
            TermKind::Mul(_) | TermKind::Div(_, _) | TermKind::Mod(_, _) => 10,
            TermKind::Add(_) | TermKind::Sub(_, _) => 5,
            TermKind::Eq(_, _)
            | TermKind::Le(_, _)
            | TermKind::Lt(_, _)
            | TermKind::Ge(_, _)
            | TermKind::Gt(_, _) => 4,
            TermKind::And(_) | TermKind::Or(_) => 3,
            TermKind::Not(_) => 2,
            _ => 1,
        })
    }

    /// Sum `weight(kind)` over the term skeleton both [`Self::term_size`]
    /// and [`Self::term_cost`] walk, using an explicit stack and a memo.
    ///
    /// A term absent from the manager weighs `1`, matching the old
    /// `let Some(t) = … else { return 1 }` guard in both functions.
    fn accumulate(&self, term: TermId, weight: impl Fn(&oxiz_core::TermKind) -> usize) -> usize {
        let mut memo: rustc_hash::FxHashMap<TermId, usize> = rustc_hash::FxHashMap::default();
        let mut stack: Vec<(TermId, bool)> = vec![(term, false)];

        while let Some((current, expanded)) = stack.pop() {
            if memo.contains_key(&current) {
                continue;
            }
            let Some(kind) = self.terms.get(current).map(|t| &t.kind) else {
                memo.insert(current, 1);
                continue;
            };
            let children = Self::measured_children(kind);

            if expanded {
                let total = children
                    .iter()
                    .map(|child| memo.get(child).copied().unwrap_or(0))
                    .sum::<usize>()
                    .saturating_add(weight(kind));
                memo.insert(current, total);
            } else {
                stack.push((current, true));
                for child in children {
                    if !memo.contains_key(&child) {
                        stack.push((child, false));
                    }
                }
            }
        }

        memo.get(&term).copied().unwrap_or(1)
    }

    /// The subterms the size/cost estimates descend into.
    ///
    /// Deliberately *not* every child: this reproduces the exact kind set
    /// the recursive estimates enumerated, so the numbers they feed into
    /// the elimination-order heuristics are unchanged.
    fn measured_children(kind: &oxiz_core::TermKind) -> Vec<TermId> {
        use oxiz_core::TermKind;

        match kind {
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args.to_vec(),
            TermKind::Not(arg) => vec![*arg],
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b) => vec![*a, *b],
            _ => Vec::new(),
        }
    }

    /// Compute variable frequency in a cube
    /// Returns a map from variable to its frequency count
    fn compute_variable_frequency(&self, cube: &[TermId]) -> rustc_hash::FxHashMap<TermId, usize> {
        let mut freq = rustc_hash::FxHashMap::default();

        for &lit in cube {
            let vars = Self::collect_vars(self.terms, lit);
            for var in vars {
                *freq.entry(var).or_insert(0) += 1;
            }
        }

        freq
    }

    /// Compute a frequency score for a literal
    /// Lower score means less frequent variables (prioritize for elimination)
    fn literal_frequency_score(
        &self,
        lit: TermId,
        var_freq: &rustc_hash::FxHashMap<TermId, usize>,
    ) -> usize {
        let vars = Self::collect_vars(self.terms, lit);
        if vars.is_empty() {
            return 0;
        }

        // Average frequency of variables in this literal
        let total_freq: usize = vars.iter().filter_map(|v| var_freq.get(v)).sum();
        total_freq / vars.len().max(1)
    }

    /// Simple down-closure: try to drop literals without inductiveness check
    ///
    /// This is faster than MIC but may produce weaker lemmas.
    /// Useful as a pre-processing step before MIC.
    pub fn down_closure(
        &mut self,
        cube: &[TermId],
        must_block: TermId,
    ) -> Result<GeneralizationResult, GeneralizationError> {
        debug!("Down-closure generalization with {} literals", cube.len());

        if cube.is_empty() {
            return Err(GeneralizationError::InvalidCube(
                "Cannot generalize empty cube".to_string(),
            ));
        }

        let mut result = GeneralizationResult::new(cube.iter().copied());
        let mut current_cube = cube.to_vec();
        let mut i = 0;

        // Try to drop each literal
        while i < current_cube.len() {
            let removed_lit = current_cube.remove(i);

            // Build candidate from remaining literals
            let candidate = if current_cube.is_empty() {
                self.terms.mk_true()
            } else if current_cube.len() == 1 {
                current_cube[0]
            } else {
                self.terms.mk_and(current_cube.clone())
            };

            // Check if candidate still blocks the bad state
            let mut smt = SmtSolver::new(self.terms, self.system);
            self.num_queries += 1;

            let still_blocks = smt.is_blocked_by(candidate, must_block).unwrap_or_default();

            if still_blocks {
                // Successfully dropped the literal
                result.dropped.push(removed_lit);
            } else {
                // Need to keep this literal
                current_cube.insert(i, removed_lit);
                i += 1;
            }
        }

        result.lemma = current_cube.into_iter().collect();
        result.num_queries = self.num_queries;

        debug!(
            "Down-closure: reduced from {} to {} literals",
            cube.len(),
            result.lemma.len()
        );

        Ok(result)
    }

    /// Extract cube (conjunction of literals) from a formula
    ///
    /// Decomposes a formula into a vector of literals (atoms and negated atoms).
    /// Only handles conjunctions; returns error for other forms.
    pub fn extract_cube(terms: &TermManager, formula: TermId) -> Vec<TermId> {
        let mut cube = Vec::new();

        if let Some(term) = terms.get(formula) {
            match &term.kind {
                TermKind::And(args) => {
                    // Conjunction: collect all conjuncts
                    for &arg in args.iter() {
                        Self::collect_literals(terms, arg, &mut cube);
                    }
                }
                _ => {
                    // Single literal or complex formula
                    Self::collect_literals(terms, formula, &mut cube);
                }
            }
        }

        cube
    }

    /// Collect literals from a formula into a cube.
    ///
    /// The `And` tree is flattened with an explicit heap stack
    /// ([`crate::walk::flatten_conjuncts`]); the previous recursive form
    /// descended one native frame per `And` level, with no bound and no
    /// error channel (`-> ()`), so deeply nested cube input aborted the
    /// process. The per-literal rules are unchanged: `true` conjuncts are
    /// skipped, a `false` conjunct resets the cube to itself, and any other
    /// literal is appended unless already present.
    fn collect_literals(terms: &TermManager, formula: TermId, cube: &mut Vec<TermId>) {
        for literal in crate::walk::flatten_conjuncts(terms, formula) {
            let Some(term) = terms.get(literal) else {
                // Not a term of this manager: ignored, as before.
                continue;
            };
            match &term.kind {
                // Skip true literals.
                TermKind::True => {}
                // False makes the whole cube unsatisfiable.
                TermKind::False => {
                    cube.clear();
                    cube.push(literal);
                }
                // Atomic literal or negation.
                _ => {
                    if !cube.contains(&literal) {
                        cube.push(literal);
                    }
                }
            }
        }
    }

    /// Expand a cube by adding relevant constraints
    ///
    /// Uses unsat cores to identify which constraints are actually needed.
    pub fn expand_cube(
        &mut self,
        _pred: PredId,
        cube: &[TermId],
        constraints: &[TermId],
    ) -> Result<GeneralizationResult, GeneralizationError> {
        debug!(
            "Expanding cube with {} literals and {} additional constraints",
            cube.len(),
            constraints.len()
        );

        if constraints.is_empty() {
            // No constraints to add
            return Ok(GeneralizationResult::new(cube.iter().copied()));
        }

        // Enhanced implementation: Try to add constraints that strengthen the cube
        // without making it too specific

        let mut result = GeneralizationResult::new(cube.iter().copied());
        let mut current_cube = cube.to_vec();

        // Try to add each constraint one at a time
        for &constraint in constraints {
            // Skip if this constraint is already in the cube
            if current_cube.contains(&constraint) {
                continue;
            }

            // Check if adding this constraint is beneficial
            // Heuristic: Add if it doesn't make the formula too strong
            // A full implementation would:
            // 1. Check if the constraint is relevant (shares variables with cube)
            // 2. Use SMT to check if it's not redundant
            // 3. Ensure it doesn't make the lemma too specific

            // Simple heuristic: Check if the constraint shares variables with the cube
            if Self::shares_variables(self.terms, constraint, cube) {
                // Add the constraint
                current_cube.push(constraint);
                trace!("Added constraint to cube");
            }
        }

        result.lemma = current_cube.into_iter().collect();
        result.num_queries = self.num_queries;

        debug!(
            "Cube expansion: grew from {} to {} literals",
            cube.len(),
            result.lemma.len()
        );

        Ok(result)
    }

    /// Check if a constraint shares variables with any literal in the cube
    fn shares_variables(terms: &TermManager, constraint: TermId, cube: &[TermId]) -> bool {
        let constraint_vars = Self::collect_vars(terms, constraint);
        if constraint_vars.is_empty() {
            return false;
        }

        for &lit in cube {
            let lit_vars = Self::collect_vars(terms, lit);
            // Check if there's any overlap
            if constraint_vars.iter().any(|v| lit_vars.contains(v)) {
                return true;
            }
        }

        false
    }

    /// Collect the distinct variables occurring in a term.
    ///
    /// Iterative walk with a visited set (see [`crate::walk`]). The old
    /// recursive helper passed a `HashSet` that looked like memoization but
    /// was the *output* set — it never pruned traversal — so a shared DAG
    /// was re-expanded exponentially, and nesting depth was unbounded.
    /// It also stopped at any operator outside a short enumeration
    /// (`_ => {}`), missing variables under `Ite`, `Implies`, `Apply`,
    /// `Select`/`Store` and every bitvector/string operation, which made the
    /// frequency heuristics score literals on incomplete variable sets.
    fn collect_vars(terms: &TermManager, term: TermId) -> Vec<TermId> {
        crate::walk::collect_vars(terms, term)
    }

    /// Get the number of SMT queries performed
    pub fn num_queries(&self) -> u32 {
        self.num_queries
    }

    /// Reset query counter
    pub fn reset_queries(&mut self) {
        self.num_queries = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stack size and nesting depth shared by the deep-recursion tests below.
    ///
    /// The two are scaled together on purpose: what these tests actually pin
    /// is the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). A natively recursive estimator needs far more than
    /// that per frame and still overflows, so the regression keeps every bit
    /// of its detection power. The pair used to be 1 MiB / 50_000 -- the same
    /// 21 bytes. `mk_and` flattens its arguments (so `acc = mk_and([acc,
    /// atom])` never actually nests, and is quadratic to boot), so the deep
    /// `And` chain below is built with `TermManager::intern_term` directly,
    /// which neither flattens nor re-interns already-built prefixes (the
    /// `mk_add` chain in `term_size_survives_deep_nesting` is unaffected --
    /// `mk_add` never flattens). Never raise `DEEP_DEPTH` without raising
    /// `DEEP_STACK` by the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

    #[test]
    fn test_generalizer_creation() {
        let mut terms = TermManager::new();
        let system = ChcSystem::new();
        let _gen = Generalizer::new(&mut terms, &system);
    }

    #[test]
    fn test_extract_cube_single() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);

        let cube = Generalizer::extract_cube(&terms, x);
        assert_eq!(cube.len(), 1);
        assert_eq!(cube[0], x);
    }

    #[test]
    fn test_extract_cube_conjunction() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);
        let z = terms.mk_var("z", terms.sorts.int_sort);

        let conj = terms.mk_and(vec![x, y, z]);

        let cube = Generalizer::extract_cube(&terms, conj);
        assert_eq!(cube.len(), 3);
        assert!(cube.contains(&x));
        assert!(cube.contains(&y));
        assert!(cube.contains(&z));
    }

    #[test]
    fn test_generalization_result() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);

        let result = GeneralizationResult::new(vec![x, y]);
        assert_eq!(result.lemma.len(), 2);
        assert!(!result.is_inductive);

        let formula = result.to_formula(&mut terms);
        // Should be a conjunction
        if let Some(term) = terms.get(formula) {
            assert!(matches!(term.kind, TermKind::And(_)));
        }
    }

    #[test]
    fn test_generalization_result_single() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);

        let result = GeneralizationResult::new(vec![x]);
        let formula = result.to_formula(&mut terms);
        assert_eq!(formula, x);
    }

    #[test]
    fn test_generalization_result_empty() {
        let mut terms = TermManager::new();
        let result = GeneralizationResult::new(Vec::<TermId>::new());
        let formula = result.to_formula(&mut terms);
        assert_eq!(formula, terms.mk_true());
    }

    /// Size and cost must keep their exact previous values, and must be
    /// computed without native recursion or DAG re-expansion.
    #[test]
    fn term_size_and_cost_are_pinned_and_dag_linear() {
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let one = terms.mk_int(1);
        // `x + 1`: Add(1) + x(1) + 1(1) = 3 nodes; cost 5 + 1 + 1 = 7.
        let sum = terms.mk_add([x, one]);
        let zero = terms.mk_int(0);
        // `x + 1 >= 0`: 1 + 3 + 1 = 5 nodes; cost 4 + 7 + 1 = 12.
        let atom = terms.mk_ge(sum, zero);

        let system = ChcSystem::new();
        let generalizer = Generalizer::new(&mut terms, &system);
        assert_eq!(
            generalizer.term_size(atom),
            5,
            "term_size must be unchanged"
        );
        assert_eq!(
            generalizer.term_cost(atom),
            12,
            "term_cost must be unchanged"
        );

        // A 60-deep doubling DAG has 2^60 tree paths: without the memo this
        // never finishes.
        let mut terms = TermManager::new();
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let one = terms.mk_int(1);
        let mut shared = terms.mk_add([x, one]);
        for _ in 0..60 {
            shared = terms.mk_add([shared, shared]);
        }
        let system = ChcSystem::new();
        let generalizer = Generalizer::new(&mut terms, &system);
        assert!(generalizer.term_size(shared) > 0);
        assert!(generalizer.term_cost(shared) > 0);
    }

    /// Neither estimate may overflow the stack on deeply nested input.
    #[test]
    fn term_size_survives_deep_nesting() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut terms = TermManager::new();
                let mut current = terms.mk_var("x", terms.sorts.int_sort);
                for i in 0..DEEP_DEPTH {
                    let k = terms.mk_int(i);
                    current = terms.mk_add([current, k]);
                }
                let system = ChcSystem::new();
                let generalizer = Generalizer::new(&mut terms, &system);
                assert!(generalizer.term_size(current) > DEEP_DEPTH as usize);
                assert!(generalizer.term_cost(current) > DEEP_DEPTH as usize);
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep term_size must return");
    }

    /// Variables under operators the old enumeration skipped must be found.
    #[test]
    fn collect_vars_sees_through_ite() {
        let mut terms = TermManager::new();
        let int_sort = terms.sorts.int_sort;
        let bool_sort = terms.sorts.bool_sort;
        let x = terms.mk_var("x", int_sort);
        let y = terms.mk_var("y", int_sort);
        let cond = terms.mk_var("c", bool_sort);
        let ite = terms.mk_ite(cond, x, y);
        let vars = Generalizer::collect_vars(&terms, ite);
        assert!(vars.contains(&x) && vars.contains(&y) && vars.contains(&cond));
    }

    /// A deeply nested cube must be extracted without overflowing, and the
    /// literal-ordering/dedup rules must be preserved.
    #[test]
    fn extract_cube_survives_deep_nesting_and_pins_rules() {
        let mut terms = TermManager::new();
        let int_sort = terms.sorts.int_sort;
        let x = terms.mk_var("x", int_sort);
        let zero = terms.mk_int(0);
        let a = terms.mk_ge(x, zero);
        let b = terms.mk_lt(x, zero);
        let truth = terms.mk_true();
        // `a /\ true /\ b /\ a` -> [a, b] (true skipped, duplicate dropped).
        let cube_term = terms.mk_and([a, truth, b, a]);
        assert_eq!(Generalizer::extract_cube(&terms, cube_term), vec![a, b]);

        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut terms = TermManager::new();
                let bool_sort = terms.sorts.bool_sort;
                let int_sort = terms.sorts.int_sort;
                let zero = terms.mk_int(0);
                let first = terms.mk_var("v0", int_sort);
                let mut formula = terms.mk_ge(first, zero);
                // Interned directly (not via `mk_and`, which would flatten
                // this back into one wide `And`) so the tree really is
                // `DEEP_DEPTH` deep.
                for i in 1..DEEP_DEPTH {
                    let v = terms.mk_var(&format!("v{i}"), int_sort);
                    let atom = terms.mk_ge(v, zero);
                    formula =
                        terms.intern_term(TermKind::And(vec![formula, atom].into()), bool_sort);
                }
                assert!(Generalizer::extract_cube(&terms, formula).len() >= DEEP_DEPTH as usize);
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep cube extraction must return");
    }
}
