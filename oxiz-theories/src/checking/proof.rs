//! Proof Checker
//!
//! Validates proof steps and integrates with DRAT/Alethe proof generation.

use super::{CheckResult, CheckerStats, Literal};
use crate::prelude::FxHashMap;
#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_time::Instant;
use std::collections::{HashMap, HashSet};

/// Kind of proof step
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofStepKind {
    /// Axiom (input assertion)
    Axiom,
    /// Resolution between two clauses
    Resolution {
        /// The pivot literal
        pivot: TermId,
        /// First clause ID
        clause1: usize,
        /// Second clause ID
        clause2: usize,
    },
    /// Theory lemma
    TheoryLemma(String),
    /// Unit propagation
    UnitPropagation {
        /// The unit literal
        unit: Literal,
        /// Antecedent clause ID
        antecedent: usize,
    },
    /// Assumption (for proof by contradiction)
    Assumption,
    /// Contradiction (empty clause derivation)
    Contradiction,
    /// Symmetry: a = b => b = a
    Symmetry(TermId, TermId),
    /// Transitivity: a = b, b = c => a = c
    Transitivity(TermId, TermId, TermId),
    /// Congruence: f(a) = f(b) from a = b
    Congruence {
        /// The function being applied
        function: TermId,
        /// Arguments for first application
        args1: Vec<TermId>,
        /// Arguments for second application
        args2: Vec<TermId>,
    },
    /// Instantiation of quantifier
    Instantiation {
        /// The quantifier being instantiated
        quantifier: TermId,
        /// Variable substitutions
        substitution: Vec<(TermId, TermId)>,
    },
    /// Skolemization
    Skolemization {
        /// Original quantified formula
        original: TermId,
        /// Skolemized result
        skolemized: TermId,
    },
}

/// A proof step
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Step ID
    pub id: usize,
    /// Kind of step
    pub kind: ProofStepKind,
    /// Result clause
    pub clause: Vec<Literal>,
    /// Antecedent step IDs
    pub antecedents: Vec<usize>,
}

impl ProofStep {
    /// Create a new proof step
    pub fn new(id: usize, kind: ProofStepKind, clause: Vec<Literal>) -> Self {
        Self {
            id,
            kind,
            clause,
            antecedents: Vec::new(),
        }
    }

    /// Create with antecedents
    pub fn with_antecedents(
        id: usize,
        kind: ProofStepKind,
        clause: Vec<Literal>,
        antecedents: Vec<usize>,
    ) -> Self {
        Self {
            id,
            kind,
            clause,
            antecedents,
        }
    }
}

/// Outcome of beginning exploration of a step in [`ProofChecker::validate_step`].
enum StepEntry {
    /// The step's verdict is already known without exploring its
    /// antecedents (already validated, unknown, or a cyclic back-edge).
    Immediate(CheckResult),
    /// The step needs its antecedents checked; the caller should push this
    /// frame and resume the exploration loop.
    Explore(StepFrame),
}

/// One level of [`ProofChecker::validate_step`]'s explicit exploration
/// stack, equivalent to one still-open recursive call.
struct StepFrame {
    /// The step this frame is validating.
    step_id: usize,
    /// A clone of the step's data (owned so it can be read while `self` is
    /// mutably borrowed for `check_step`/stats bookkeeping elsewhere).
    step: ProofStep,
    /// Index of the next antecedent to resolve.
    next_antecedent: usize,
    /// When this frame started, for the `check_time_us` stat.
    start: Instant,
}

/// Proof checker that validates proof steps
#[derive(Debug)]
pub struct ProofChecker {
    /// Steps in the proof
    steps: HashMap<usize, ProofStep>,
    /// Validated steps
    validated: HashSet<usize>,
    /// Statistics
    stats: CheckerStats,
    /// Whether to check thoroughly
    thorough: bool,
}

impl ProofChecker {
    /// Create a new proof checker
    pub fn new() -> Self {
        Self {
            steps: HashMap::new(),
            validated: HashSet::new(),
            stats: CheckerStats::default(),
            thorough: true,
        }
    }

    /// Create with thorough checking disabled
    pub fn quick() -> Self {
        Self {
            steps: HashMap::new(),
            validated: HashSet::new(),
            stats: CheckerStats::default(),
            thorough: false,
        }
    }

    /// Add a proof step
    pub fn add_step(&mut self, step: ProofStep) {
        self.steps.insert(step.id, step);
    }

    /// Validate a proof step against the term manager that owns its terms.
    ///
    /// The manager is required to decode the structure of the terms referenced
    /// by equality/quantifier rules; without it those rules could only be
    /// rubber-stamped. Antecedent steps are validated first (iteratively; see
    /// below), and any step's verdict is exactly what a direct recursive
    /// implementation would have produced: propagate an antecedent's
    /// non-valid result upward as-is (skipping this step's own check and
    /// stats update, matching the antecedent-cascade behaviour), and only
    /// mark a step validated once its own `check_step` succeeds.
    ///
    /// # Why iterative
    ///
    /// A direct recursive implementation ("first validate antecedents, then
    /// check this step") both overflows the native stack on a deep proof and
    /// hangs forever on a cyclic antecedent graph (a step that transitively
    /// depends on itself), since nothing marks a step "in progress" before
    /// recursing into it. This walks the antecedent graph with an explicit
    /// stack instead, and `enter_step` marks each step's exploration
    /// with `in_progress`: revisiting one is a back-edge, reported as a
    /// malformed proof (`CheckResult::Invalid`) rather than looped on
    /// forever. A cyclic proof is unsound by construction (its "premises"
    /// include its own conclusion), so rejecting it is the correct verdict,
    /// not merely a safe one.
    pub fn validate_step(&mut self, step_id: usize, manager: &mut TermManager) -> CheckResult {
        // Steps currently being explored on the current path (as opposed to
        // already validated or not yet visited). Revisiting one of these
        // means the antecedent graph has a cycle back to an ancestor.
        let mut in_progress: HashSet<usize> = HashSet::new();

        // `top` is the frame currently being resolved, owned directly rather
        // than peeked from `stack` -- `stack` holds only the *suspended*
        // ancestors waiting for `top` (or one of its descendants) to finish,
        // one entry per still-open recursive call.
        let mut top = match self.enter_step(step_id, &mut in_progress) {
            StepEntry::Immediate(result) => return result,
            StepEntry::Explore(frame) => frame,
        };
        let mut stack: Vec<StepFrame> = Vec::new();

        loop {
            let next_antecedent = top.step.antecedents.get(top.next_antecedent).copied();
            top.next_antecedent += 1;

            match next_antecedent {
                Some(ant_id) => match self.enter_step(ant_id, &mut in_progress) {
                    StepEntry::Immediate(result) if !result.is_valid() => {
                        // Propagate the antecedent's non-valid verdict as
                        // this step's own verdict, exactly like a
                        // direct-recursion early `return result`: no
                        // check_step call and no stats update for the step
                        // being unwound here -- and every still-open
                        // ancestor below it made that identical recursive
                        // call and so unwinds the same way in turn.
                        in_progress.remove(&top.step_id);
                        return Self::unwind_invalid(&mut stack, &mut in_progress, result);
                    }
                    StepEntry::Immediate(_valid) => {
                        // The antecedent resolved to Valid; keep resuming
                        // `top` at its next antecedent (index already
                        // advanced above).
                    }
                    StepEntry::Explore(frame) => {
                        stack.push(top);
                        top = frame;
                    }
                },
                None => {
                    // Every antecedent validated; check this step itself.
                    let result = self.check_step(&top.step, manager);
                    if result.is_valid() {
                        self.validated.insert(top.step_id);
                    }
                    let elapsed = top.start.elapsed();
                    self.stats.check_time_us += elapsed.as_micros() as u64;
                    in_progress.remove(&top.step_id);

                    if !result.is_valid() {
                        return Self::unwind_invalid(&mut stack, &mut in_progress, result);
                    }
                    top = match stack.pop() {
                        Some(parent) => parent,
                        None => return result,
                    };
                }
            }
        }
    }

    /// Drain every still-suspended ancestor frame in `stack`, removing each
    /// from `in_progress` without ever running its `check_step` (mirroring a
    /// chain of direct-recursion early `return result` calls unwinding
    /// through each waiting caller in turn), then hand back `result` once
    /// none are left -- the empty case is not a failure, it is the answer
    /// for the original `validate_step` call.
    fn unwind_invalid(
        stack: &mut Vec<StepFrame>,
        in_progress: &mut HashSet<usize>,
        result: CheckResult,
    ) -> CheckResult {
        while let Some(frame) = stack.pop() {
            in_progress.remove(&frame.step_id);
        }
        result
    }

    /// Begin exploring `id` during [`Self::validate_step`]: resolve it
    /// immediately if possible (already validated, unknown, or a cyclic
    /// back-edge), otherwise stake out `in_progress` and hand back a frame
    /// for the caller to push.
    fn enter_step(&self, id: usize, in_progress: &mut HashSet<usize>) -> StepEntry {
        if self.validated.contains(&id) {
            return StepEntry::Immediate(CheckResult::Valid);
        }
        let step = match self.steps.get(&id) {
            Some(s) => s.clone(),
            None => {
                return StepEntry::Immediate(CheckResult::Invalid(format!("Unknown step: {}", id)));
            }
        };
        if !in_progress.insert(id) {
            // `id` is already on the exploration path: some antecedent of
            // `id`, transitively, names `id` itself. A proof that requires
            // its own conclusion as a premise is malformed, not merely
            // unverifiable -- report it as such instead of recursing
            // forever.
            return StepEntry::Immediate(CheckResult::Invalid(format!(
                "Cyclic proof: step {} depends on itself through its antecedents",
                id
            )));
        }
        StepEntry::Explore(StepFrame {
            step_id: id,
            step,
            next_antecedent: 0,
            start: Instant::now(),
        })
    }

    /// Check a single proof step
    fn check_step(&self, step: &ProofStep, manager: &mut TermManager) -> CheckResult {
        match &step.kind {
            ProofStepKind::Axiom => {
                // Axioms are always valid (they're input)
                CheckResult::Valid
            }

            ProofStepKind::Assumption => {
                // Assumptions are valid (for proof by contradiction)
                CheckResult::Valid
            }

            ProofStepKind::Resolution {
                pivot,
                clause1,
                clause2,
            } => self.check_resolution(*pivot, *clause1, *clause2, &step.clause),

            ProofStepKind::UnitPropagation { unit, antecedent } => {
                self.check_unit_propagation(*unit, *antecedent, &step.clause)
            }

            ProofStepKind::TheoryLemma(theory) => {
                // Theory lemmas are trusted (checked by theory-specific checker)
                if self.thorough {
                    CheckResult::Unknown(format!("Theory lemma from {}", theory))
                } else {
                    CheckResult::Valid
                }
            }

            ProofStepKind::Contradiction => {
                // Contradiction step should have empty clause
                if step.clause.is_empty() {
                    CheckResult::Valid
                } else {
                    CheckResult::Invalid("Contradiction with non-empty clause".to_string())
                }
            }

            ProofStepKind::Symmetry(a, b) => self.check_symmetry(*a, *b, step, manager),

            ProofStepKind::Transitivity(a, b, c) => {
                self.check_transitivity(*a, *b, *c, step, manager)
            }

            ProofStepKind::Congruence {
                function,
                args1,
                args2,
            } => self.check_congruence(*function, args1, args2, step, manager),

            ProofStepKind::Instantiation {
                quantifier,
                substitution,
            } => self.check_instantiation(*quantifier, substitution, step, manager),

            ProofStepKind::Skolemization {
                original,
                skolemized,
            } => self.check_skolemization(*original, *skolemized, manager),
        }
    }

    /// Decode a positive equality literal into its two sides, if it is one.
    fn as_equality(manager: &TermManager, lit: &Literal) -> Option<(TermId, TermId)> {
        if !lit.positive {
            return None;
        }
        match manager.get(lit.term).map(|t| t.kind.clone()) {
            Some(TermKind::Eq(l, r)) => Some((l, r)),
            _ => None,
        }
    }

    /// Decode a clause that is a single positive equality literal.
    fn single_equality(manager: &TermManager, clause: &[Literal]) -> Option<(TermId, TermId)> {
        if clause.len() != 1 {
            return None;
        }
        Self::as_equality(manager, &clause[0])
    }

    /// Test whether an equality `(x, y)` equates the unordered pair `{a, b}`.
    fn equates(pair: (TermId, TermId), a: TermId, b: TermId) -> bool {
        (pair.0 == a && pair.1 == b) || (pair.0 == b && pair.1 == a)
    }

    /// Collect the equalities established by a step's antecedent clauses.
    fn antecedent_equalities(
        &self,
        manager: &TermManager,
        step: &ProofStep,
    ) -> Vec<(TermId, TermId)> {
        step.antecedents
            .iter()
            .filter_map(|id| self.steps.get(id))
            .filter_map(|s| Self::single_equality(manager, &s.clause))
            .collect()
    }

    /// Check resolution step
    fn check_resolution(
        &self,
        pivot: TermId,
        clause1_id: usize,
        clause2_id: usize,
        result: &[Literal],
    ) -> CheckResult {
        let clause1 = match self.steps.get(&clause1_id) {
            Some(s) => &s.clause,
            None => return CheckResult::Invalid(format!("Missing clause {}", clause1_id)),
        };

        let clause2 = match self.steps.get(&clause2_id) {
            Some(s) => &s.clause,
            None => return CheckResult::Invalid(format!("Missing clause {}", clause2_id)),
        };

        // Check that pivot appears positive in one and negative in other
        let has_pos1 = clause1.iter().any(|l| l.term == pivot && l.positive);
        let has_neg1 = clause1.iter().any(|l| l.term == pivot && !l.positive);
        let has_pos2 = clause2.iter().any(|l| l.term == pivot && l.positive);
        let has_neg2 = clause2.iter().any(|l| l.term == pivot && !l.positive);

        if !((has_pos1 && has_neg2) || (has_neg1 && has_pos2)) {
            return CheckResult::Invalid("Pivot not complementary in clauses".to_string());
        }

        // Check that result is union minus pivot literals
        let expected: HashSet<_> = clause1
            .iter()
            .chain(clause2.iter())
            .filter(|l| l.term != pivot)
            .collect();

        let actual: HashSet<_> = result.iter().collect();

        if expected != actual {
            return CheckResult::Invalid("Resolution result incorrect".to_string());
        }

        CheckResult::Valid
    }

    /// Check unit propagation.
    ///
    /// Structurally verifies that the derived clause is exactly the propagated
    /// unit and that the unit occurs in the antecedent clause. The semantic
    /// side-condition (every other antecedent literal is falsified by the
    /// current trail) cannot be certified without a trail model, so a
    /// well-formed step is reported as `Unknown` rather than silently accepted.
    fn check_unit_propagation(
        &self,
        unit: Literal,
        antecedent: usize,
        result: &[Literal],
    ) -> CheckResult {
        let ante = match self.steps.get(&antecedent) {
            Some(s) => s,
            None => {
                return CheckResult::Invalid(format!(
                    "Unit propagation references missing antecedent {}",
                    antecedent
                ));
            }
        };
        if !ante.clause.contains(&unit) {
            return CheckResult::Invalid(
                "Propagated unit does not occur in the antecedent clause".to_string(),
            );
        }
        if result.len() != 1 || result[0] != unit {
            return CheckResult::Invalid(
                "Unit propagation result clause is not the propagated unit".to_string(),
            );
        }
        CheckResult::Unknown(
            "Unit propagation is well-formed but the trail condition is unchecked".to_string(),
        )
    }

    /// Check symmetry: from `a = b` conclude the equality of `a` and `b`.
    ///
    /// Because `mk_eq` canonicalizes operand order, `a = b` and `b = a` intern
    /// to the same term, so the verifiable content is that the conclusion is an
    /// equality over exactly the pair `{a, b}` (and, when the premise is
    /// supplied as an antecedent, that it too equates `{a, b}`).
    fn check_symmetry(
        &self,
        a: TermId,
        b: TermId,
        step: &ProofStep,
        manager: &TermManager,
    ) -> CheckResult {
        let concl = match Self::single_equality(manager, &step.clause) {
            Some(eq) => eq,
            None => {
                return CheckResult::Invalid(
                    "Symmetry conclusion is not a single equality literal".to_string(),
                );
            }
        };
        if !Self::equates(concl, a, b) {
            return CheckResult::Invalid(
                "Symmetry conclusion does not equate the stated operands".to_string(),
            );
        }
        let premises = self.antecedent_equalities(manager, step);
        if premises.is_empty() {
            // Sound inference shape, but the premise a=b was not attached.
            return CheckResult::Unknown(
                "Symmetry conclusion well-formed but premise a=b not supplied".to_string(),
            );
        }
        if premises.iter().any(|p| Self::equates(*p, a, b)) {
            CheckResult::Valid
        } else {
            CheckResult::Invalid("Symmetry premise does not establish a=b".to_string())
        }
    }

    /// Check transitivity: from `a = b` and `b = c` conclude `a = c`.
    fn check_transitivity(
        &self,
        a: TermId,
        b: TermId,
        c: TermId,
        step: &ProofStep,
        manager: &TermManager,
    ) -> CheckResult {
        let concl = match Self::single_equality(manager, &step.clause) {
            Some(eq) => eq,
            None => {
                return CheckResult::Invalid(
                    "Transitivity conclusion is not a single equality literal".to_string(),
                );
            }
        };
        if !Self::equates(concl, a, c) {
            return CheckResult::Invalid(
                "Transitivity conclusion does not equate the chain endpoints a and c".to_string(),
            );
        }
        let premises = self.antecedent_equalities(manager, step);
        if premises.is_empty() {
            return CheckResult::Unknown(
                "Transitivity endpoints well-formed but premises a=b, b=c not supplied".to_string(),
            );
        }
        let has_ab = premises.iter().any(|p| Self::equates(*p, a, b));
        let has_bc = premises.iter().any(|p| Self::equates(*p, b, c));
        if has_ab && has_bc {
            CheckResult::Valid
        } else {
            CheckResult::Invalid(
                "Transitivity premises do not chain a=b and b=c through the middle term"
                    .to_string(),
            )
        }
    }

    /// Check congruence: from `args1[i] = args2[i]` conclude
    /// `f(args1) = f(args2)`.
    fn check_congruence(
        &self,
        _function: TermId,
        args1: &[TermId],
        args2: &[TermId],
        step: &ProofStep,
        manager: &TermManager,
    ) -> CheckResult {
        if args1.len() != args2.len() {
            return CheckResult::Invalid("Congruence argument lists differ in length".to_string());
        }
        let concl = match Self::single_equality(manager, &step.clause) {
            Some(eq) => eq,
            None => {
                return CheckResult::Invalid(
                    "Congruence conclusion is not a single equality literal".to_string(),
                );
            }
        };
        // Both sides must be applications of the same function symbol, one over
        // args1 and the other over args2 (either operand order, since equality
        // is canonicalized).
        if !(Self::is_application_of(manager, concl.0, args1, concl.1, args2)
            || Self::is_application_of(manager, concl.1, args1, concl.0, args2))
        {
            return CheckResult::Invalid(
                "Congruence conclusion is not f(args1) = f(args2)".to_string(),
            );
        }
        // Every argument position that actually differs must be justified by a
        // premise equality; positions that are syntactically identical need no
        // premise.
        let premises = self.antecedent_equalities(manager, step);
        for (l, r) in args1.iter().zip(args2.iter()) {
            if l == r {
                continue;
            }
            if !premises.iter().any(|p| Self::equates(*p, *l, *r)) {
                return CheckResult::Unknown(format!(
                    "Congruence lacks a premise for differing argument {:?} = {:?}",
                    l, r
                ));
            }
        }
        CheckResult::Valid
    }

    /// Check that `side_x` is `f(exp_x)` and `side_y` is `f(exp_y)` for a
    /// common function symbol `f`.
    fn is_application_of(
        manager: &TermManager,
        side_x: TermId,
        exp_x: &[TermId],
        side_y: TermId,
        exp_y: &[TermId],
    ) -> bool {
        let kx = manager.get(side_x).map(|t| t.kind.clone());
        let ky = manager.get(side_y).map(|t| t.kind.clone());
        match (kx, ky) {
            (
                Some(TermKind::Apply { func: fx, args: ax }),
                Some(TermKind::Apply { func: fy, args: ay }),
            ) => fx == fy && ax.as_slice() == exp_x && ay.as_slice() == exp_y,
            _ => false,
        }
    }

    /// Check quantifier instantiation: `result` should contain `body[subst]`.
    fn check_instantiation(
        &self,
        quantifier: TermId,
        substitution: &[(TermId, TermId)],
        step: &ProofStep,
        manager: &mut TermManager,
    ) -> CheckResult {
        let body = match manager.get(quantifier).map(|t| t.kind.clone()) {
            Some(TermKind::Forall { body, .. }) | Some(TermKind::Exists { body, .. }) => body,
            _ => {
                return CheckResult::Invalid(
                    "Instantiation target is not a quantified formula".to_string(),
                );
            }
        };
        let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (var, replacement) in substitution {
            map.insert(*var, *replacement);
        }
        let instance = manager.substitute(body, &map);
        if step.clause.iter().any(|l| l.term == instance) {
            CheckResult::Valid
        } else {
            CheckResult::Unknown(
                "Instantiation instance not found verbatim in the result clause".to_string(),
            )
        }
    }

    /// Check Skolemization.
    ///
    /// Confirms the original is a quantified formula; verifying that the
    /// Skolemized form introduces genuinely fresh Skolem symbols requires
    /// context this checker does not track, so a well-formed step stays
    /// `Unknown` rather than being accepted.
    fn check_skolemization(
        &self,
        original: TermId,
        _skolemized: TermId,
        manager: &TermManager,
    ) -> CheckResult {
        match manager.get(original).map(|t| t.kind.clone()) {
            Some(TermKind::Exists { .. }) | Some(TermKind::Forall { .. }) => CheckResult::Unknown(
                "Skolemization target is a quantifier but freshness is unchecked".to_string(),
            ),
            _ => {
                CheckResult::Invalid("Skolemization target is not a quantified formula".to_string())
            }
        }
    }

    /// Check if proof derives empty clause
    pub fn check_proof(&mut self, manager: &mut TermManager) -> CheckResult {
        // Find contradiction step
        for (&id, step) in &self.steps.clone() {
            if matches!(step.kind, ProofStepKind::Contradiction) {
                return self.validate_step(id, manager);
            }
        }

        CheckResult::Unknown("No contradiction found".to_string())
    }

    /// Get number of steps
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get number of validated steps
    pub fn num_validated(&self) -> usize {
        self.validated.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &CheckerStats {
        &self.stats
    }

    /// Reset the checker
    pub fn reset(&mut self) {
        self.steps.clear();
        self.validated.clear();
        self.stats = CheckerStats::default();
    }
}

impl Default for ProofChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three distinct uninterpreted integer constants a, b, c for equality
    /// reasoning, returned alongside the manager that owns them.
    fn abc() -> (TermManager, TermId, TermId, TermId) {
        let mut m = TermManager::new();
        let int = m.sorts.int_sort;
        let a = m.mk_var("a", int);
        let b = m.mk_var("b", int);
        let c = m.mk_var("c", int);
        (m, a, b, c)
    }

    #[test]
    fn test_proof_step_creation() {
        let t1 = TermId::from(1u32);
        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        assert_eq!(step.id, 0);
        assert_eq!(step.clause.len(), 1);
        assert!(step.antecedents.is_empty());
    }

    #[test]
    fn test_proof_step_with_antecedents() {
        let t1 = TermId::from(1u32);
        let step = ProofStep::with_antecedents(
            2,
            ProofStepKind::TheoryLemma("arith".to_string()),
            vec![Literal::pos(t1)],
            vec![0, 1],
        );
        assert_eq!(step.id, 2);
        assert_eq!(step.antecedents, vec![0, 1]);
    }

    #[test]
    fn test_proof_checker_creation() {
        let checker = ProofChecker::new();
        assert_eq!(checker.num_steps(), 0);
        assert_eq!(checker.num_validated(), 0);
        assert!(checker.thorough);
    }

    #[test]
    fn test_proof_checker_quick() {
        let checker = ProofChecker::quick();
        assert!(!checker.thorough);
    }

    #[test]
    fn test_add_and_validate_axiom() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        checker.add_step(step);

        let result = checker.validate_step(0, &mut m);
        assert!(result.is_valid());
        assert_eq!(checker.num_validated(), 1);
    }

    #[test]
    fn test_validate_unknown_step() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let result = checker.validate_step(999, &mut m);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_contradiction_step() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let step = ProofStep::new(0, ProofStepKind::Contradiction, vec![]);
        checker.add_step(step);

        let result = checker.validate_step(0, &mut m);
        assert!(result.is_valid());
    }

    #[test]
    fn test_invalid_contradiction() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        let step = ProofStep::new(0, ProofStepKind::Contradiction, vec![Literal::pos(t1)]);
        checker.add_step(step);

        let result = checker.validate_step(0, &mut m);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_resolution() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        let t3 = TermId::from(3u32);

        // Clause 1: t1 OR t2
        let step1 = ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1), Literal::pos(t2)],
        );

        // Clause 2: NOT t1 OR t3
        let step2 = ProofStep::new(
            1,
            ProofStepKind::Axiom,
            vec![Literal::neg(t1), Literal::pos(t3)],
        );

        // Resolution on t1 should give: t2 OR t3
        let step3 = ProofStep::with_antecedents(
            2,
            ProofStepKind::Resolution {
                pivot: t1,
                clause1: 0,
                clause2: 1,
            },
            vec![Literal::pos(t2), Literal::pos(t3)],
            vec![0, 1],
        );

        checker.add_step(step1);
        checker.add_step(step2);
        checker.add_step(step3);

        let result = checker.validate_step(2, &mut m);
        assert!(result.is_valid());
    }

    #[test]
    fn test_check_proof() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step1 = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        let step2 = ProofStep::with_antecedents(1, ProofStepKind::Contradiction, vec![], vec![0]);

        checker.add_step(step1);
        checker.add_step(step2);

        let result = checker.check_proof(&mut m);
        assert!(result.is_valid());
    }

    #[test]
    fn test_no_contradiction() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        checker.add_step(step);

        let result = checker.check_proof(&mut m);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_reset() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        checker.add_step(step);
        let _ = checker.validate_step(0, &mut m);

        assert_eq!(checker.num_steps(), 1);
        assert_eq!(checker.num_validated(), 1);

        checker.reset();
        assert_eq!(checker.num_steps(), 0);
        assert_eq!(checker.num_validated(), 0);
    }

    #[test]
    fn test_theory_lemma() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::quick();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(
            0,
            ProofStepKind::TheoryLemma("arith".to_string()),
            vec![Literal::pos(t1)],
        );
        checker.add_step(step);

        let result = checker.validate_step(0, &mut m);
        assert!(result.is_valid());
    }

    // -----------------------------------------------------------------------
    // Real per-rule verification (equality congruence-closure rules).
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetry_valid_and_rejects_bogus_conclusion() {
        let (mut m, a, b, c) = abc();
        let eq_ab = m.mk_eq(a, b);
        let eq_ac = m.mk_eq(a, c);

        // Premise a=b (step 0), symmetric conclusion b=a === a=b (step 1).
        let mut checker = ProofChecker::new();
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_ab)],
        ));
        checker.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Symmetry(a, b),
            vec![Literal::pos(eq_ab)],
            vec![0],
        ));
        assert!(checker.validate_step(1, &mut m).is_valid());

        // A symmetry step whose conclusion equates the wrong terms is rejected.
        let mut bad = ProofChecker::new();
        bad.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_ab)],
        ));
        bad.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Symmetry(a, b),
            vec![Literal::pos(eq_ac)],
            vec![0],
        ));
        assert!(bad.validate_step(1, &mut m).is_invalid());
    }

    #[test]
    fn test_transitivity_chain_valid_and_rejects_broken_chain() {
        let (mut m, a, b, c) = abc();
        let eq_ab = m.mk_eq(a, b);
        let eq_bc = m.mk_eq(b, c);
        let eq_ac = m.mk_eq(a, c);

        let mut checker = ProofChecker::new();
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_ab)],
        ));
        checker.add_step(ProofStep::new(
            1,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_bc)],
        ));
        checker.add_step(ProofStep::with_antecedents(
            2,
            ProofStepKind::Transitivity(a, b, c),
            vec![Literal::pos(eq_ac)],
            vec![0, 1],
        ));
        assert!(checker.validate_step(2, &mut m).is_valid());

        // Conclusion a=c but premises only give a=b (b=c missing) -> invalid.
        let mut broken = ProofChecker::new();
        broken.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_ab)],
        ));
        broken.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Transitivity(a, b, c),
            vec![Literal::pos(eq_ac)],
            vec![0],
        ));
        assert!(broken.validate_step(1, &mut m).is_invalid());

        // Conclusion does not equate the endpoints a and c -> invalid.
        let mut wrong = ProofChecker::new();
        wrong.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_ab)],
        ));
        wrong.add_step(ProofStep::new(
            1,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_bc)],
        ));
        wrong.add_step(ProofStep::with_antecedents(
            2,
            ProofStepKind::Transitivity(a, b, c),
            vec![Literal::pos(eq_ab)],
            vec![0, 1],
        ));
        assert!(wrong.validate_step(2, &mut m).is_invalid());
    }

    #[test]
    fn test_congruence_valid_and_rejects_missing_premise() {
        let mut m = TermManager::new();
        let int = m.sorts.int_sort;
        let a1 = m.mk_var("a1", int);
        let a2 = m.mk_var("a2", int);
        let fa1 = m.mk_apply("f", [a1], int);
        let fa2 = m.mk_apply("f", [a2], int);
        let eq_args = m.mk_eq(a1, a2);
        let eq_apps = m.mk_eq(fa1, fa2);

        // With premise a1=a2, congruence f(a1)=f(a2) is valid.
        let mut checker = ProofChecker::new();
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_args)],
        ));
        checker.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Congruence {
                function: fa1,
                args1: vec![a1],
                args2: vec![a2],
            },
            vec![Literal::pos(eq_apps)],
            vec![0],
        ));
        assert!(checker.validate_step(1, &mut m).is_valid());

        // Without the premise, the differing argument is unjustified -> Unknown.
        let mut checker2 = ProofChecker::new();
        checker2.add_step(ProofStep::new(
            5,
            ProofStepKind::Congruence {
                function: fa1,
                args1: vec![a1],
                args2: vec![a2],
            },
            vec![Literal::pos(eq_apps)],
        ));
        let r = checker2.validate_step(5, &mut m);
        assert!(
            !r.is_valid() && !r.is_invalid(),
            "expected Unknown, got {r:?}"
        );

        // A congruence whose conclusion is not f(_)=f(_) is invalid.
        let mut checker3 = ProofChecker::new();
        checker3.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(eq_args)],
        ));
        checker3.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Congruence {
                function: fa1,
                args1: vec![a1],
                args2: vec![a2],
            },
            vec![Literal::pos(eq_args)],
            vec![0],
        ));
        assert!(checker3.validate_step(1, &mut m).is_invalid());
    }

    #[test]
    fn test_unit_propagation_structural_checks() {
        let mut m = TermManager::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        // antecedent (t1 OR t2); propagate t1 -> well-formed but Unknown.
        let mut checker = ProofChecker::new();
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1), Literal::pos(t2)],
        ));
        checker.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::UnitPropagation {
                unit: Literal::pos(t1),
                antecedent: 0,
            },
            vec![Literal::pos(t1)],
            vec![0],
        ));
        let r = checker.validate_step(1, &mut m);
        assert!(
            !r.is_valid() && !r.is_invalid(),
            "expected Unknown, got {r:?}"
        );

        // Propagated unit absent from the antecedent -> invalid.
        let mut bad = ProofChecker::new();
        bad.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t2)],
        ));
        bad.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::UnitPropagation {
                unit: Literal::pos(t1),
                antecedent: 0,
            },
            vec![Literal::pos(t1)],
            vec![0],
        ));
        assert!(bad.validate_step(1, &mut m).is_invalid());
    }

    #[test]
    fn test_instantiation_and_skolemization_targets() {
        let mut m = TermManager::new();
        let int = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int);
        // body: p(x)
        let body = m.mk_apply("p", [x], bool_sort);
        let quant = m.mk_forall([("x", int)], body);
        let t = m.mk_var("t", int);
        // instance = body[x := t] = p(t).
        let instance = {
            let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
            map.insert(x, t);
            m.substitute(body, &map)
        };

        let mut checker = ProofChecker::new();
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Instantiation {
                quantifier: quant,
                substitution: vec![(x, t)],
            },
            vec![Literal::pos(instance)],
        ));
        assert!(checker.validate_step(0, &mut m).is_valid());

        // Instantiation of a non-quantifier target is invalid.
        let mut bad = ProofChecker::new();
        bad.add_step(ProofStep::new(
            0,
            ProofStepKind::Instantiation {
                quantifier: x,
                substitution: vec![(x, t)],
            },
            vec![Literal::pos(instance)],
        ));
        assert!(bad.validate_step(0, &mut m).is_invalid());

        // Skolemization of a non-quantifier target is invalid.
        let mut sk = ProofChecker::new();
        sk.add_step(ProofStep::new(
            0,
            ProofStepKind::Skolemization {
                original: x,
                skolemized: t,
            },
            vec![Literal::pos(t)],
        ));
        assert!(sk.validate_step(0, &mut m).is_invalid());
    }

    // -----------------------------------------------------------------------
    // `validate_step` cycle-safety regression tests (audit: `validated.insert`
    // runs only after antecedents are validated, with no in-progress marker,
    // so a cyclic antecedent edge recursed forever).
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_step_self_loop_reports_cyclic_invalid_not_a_hang() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        // Step 0 lists itself as its own antecedent.
        checker.add_step(ProofStep::with_antecedents(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
            vec![0],
        ));

        let result = checker.validate_step(0, &mut m);
        assert!(
            result.is_invalid(),
            "a step that depends on itself must be rejected, got {result:?}"
        );
        assert_eq!(
            checker.num_validated(),
            0,
            "a step whose own cycle fails validation must not be marked validated"
        );
    }

    #[test]
    fn test_validate_step_mutual_cycle_reports_cyclic_invalid_not_a_hang() {
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        // Step 0 depends on step 1, which depends back on step 0.
        checker.add_step(ProofStep::with_antecedents(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
            vec![1],
        ));
        checker.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Axiom,
            vec![Literal::pos(t2)],
            vec![0],
        ));

        let result = checker.validate_step(0, &mut m);
        assert!(
            result.is_invalid(),
            "a 2-cycle between antecedents must be rejected, got {result:?}"
        );
        assert_eq!(checker.num_validated(), 0);
    }

    #[test]
    fn test_validate_step_antecedent_failure_cascades_without_checking_self() {
        // Pins the exact cascading semantics the iterative rewrite must
        // preserve: when an antecedent fails, the *dependent* step's own
        // check_step must never run (and it must not be marked validated),
        // even though that step's own rule (Axiom) would trivially succeed.
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Contradiction,
            vec![Literal::pos(t1)], // non-empty clause -> Invalid
        ));
        checker.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
            vec![0],
        ));

        let result = checker.validate_step(1, &mut m);
        assert!(result.is_invalid());
        assert_eq!(
            checker.num_validated(),
            0,
            "step 1 must not be validated when its antecedent fails"
        );
    }

    #[test]
    fn test_validate_step_diamond_dependency_shared_antecedent() {
        // A (step 0) is a shared antecedent of B (step 1) and C (step 2);
        // D (step 3) depends on both. Not a cycle -- a DAG -- and must
        // validate cleanly, with every step validated exactly once.
        let mut m = TermManager::new();
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        checker.add_step(ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
        ));
        checker.add_step(ProofStep::with_antecedents(
            1,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
            vec![0],
        ));
        checker.add_step(ProofStep::with_antecedents(
            2,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
            vec![0],
        ));
        checker.add_step(ProofStep::with_antecedents(
            3,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1)],
            vec![1, 2],
        ));

        let result = checker.validate_step(3, &mut m);
        assert!(result.is_valid());
        assert_eq!(checker.num_validated(), 4);
    }

    #[test]
    fn test_validate_step_deep_chain_small_stack() {
        // Build (iteratively) a long chain of steps, each depending on the
        // previous one, and validate the deepest one from inside a thread
        // with a deliberately small (128 KiB) stack. A stack overflow aborts
        // the whole process, so "the thread returned at all" is itself part
        // of the assertion. The stack size and `depth` are scaled together and
        // only their ratio (~21 bytes per level) matters -- never raise one
        // without the other.
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut m = TermManager::new();
                let mut checker = ProofChecker::new();
                let t1 = TermId::from(1u32);
                let depth: usize = 6_250;
                checker.add_step(ProofStep::new(
                    0,
                    ProofStepKind::Axiom,
                    vec![Literal::pos(t1)],
                ));
                for i in 1..=depth {
                    checker.add_step(ProofStep::with_antecedents(
                        i,
                        ProofStepKind::Axiom,
                        vec![Literal::pos(t1)],
                        vec![i - 1],
                    ));
                }
                let result = checker.validate_step(depth, &mut m);
                assert!(result.is_valid());
                assert_eq!(checker.num_validated(), depth + 1);
            })
            .expect("spawning a thread with an explicit stack size must succeed");
        handle
            .join()
            .expect("a deep but acyclic antecedent chain must not overflow a 128 KiB stack");
    }
}
