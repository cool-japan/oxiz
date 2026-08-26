//! Existential quantifier handling for CHCs.
//!
//! This module provides support for CHC rules with existential quantifiers,
//! which are common in verification problems involving non-deterministic choice
//! or abstraction.
//!
//! ## Existentially Quantified CHCs
//!
//! Standard form: `forall X. (body => exists Y. head(X, Y))`
//!
//! Existential variables appear only in the head of rules and represent
//! non-deterministic values or abstracted program variables.
//!
//! ## Handling Strategy
//!
//! 1. **Skolemization**: Convert existentials to fresh constants/functions
//! 2. **Projection**: Project out existential variables when learning lemmas
//! 3. **Witness extraction**: Find concrete values for existentials in counterexamples
//!
//! Reference: Existential handling in Z3's Spacer

use crate::chc::{PredId, PredicateApp, Rule, RuleHead};
use crate::pdr::SpacerError;
use oxiz_core::{SortId, TermId, TermKind, TermManager};
use smallvec::SmallVec;
use std::collections::HashMap;
use thiserror::Error;

/// Errors related to existential quantifier handling
#[derive(Error, Debug)]
pub enum ExistentialError {
    /// Unsupported existential pattern
    #[error("unsupported existential pattern: {0}")]
    Unsupported(String),
    /// Skolemization failed
    #[error("skolemization failed: {0}")]
    SkolemizationFailed(String),
    /// Projection failed
    #[error("projection failed: {0}")]
    ProjectionFailed(String),
    /// Spacer error
    #[error("spacer error: {0}")]
    Spacer(#[from] SpacerError),
}

/// Result type for existential operations
pub type ExistentialResult<T> = Result<T, ExistentialError>;

/// Information about existential variables in a rule
#[derive(Debug, Clone)]
pub struct ExistentialInfo {
    /// Variables that are existentially quantified
    pub existential_vars: SmallVec<[(String, SortId); 4]>,
    /// Variables that are universally quantified
    pub universal_vars: SmallVec<[(String, SortId); 4]>,
    /// Whether this rule has any existentials
    pub has_existentials: bool,
}

impl ExistentialInfo {
    /// Analyze a rule for existential variables.
    ///
    /// Existentials are variables that appear in the head's predicate
    /// arguments but are not among the rule's declared universal
    /// variables (`rule.vars`). Actually walking the head's argument
    /// terms (rather than just comparing argument *counts*) requires
    /// resolving each argument's `TermId` back to a variable name/sort,
    /// hence the `terms` parameter.
    ///
    /// This used to leave `existential_vars` permanently empty (`SmallVec::new()`,
    /// never pushed to) and only derive `has_existentials` from a crude
    /// `arg_count > declared_var_count` heuristic -- so any caller that
    /// fed `existential_vars` into [`ExistentialProjector::project`] or
    /// [`SkolemContext::skolemize`] would silently skolemize/project
    /// *nothing*, even on a rule `has_existentials` correctly flagged as
    /// having them.
    pub fn analyze(rule: &Rule, terms: &TermManager) -> Self {
        // Start with all declared universal variables
        let universal_vars: SmallVec<[(String, SortId); 4]> = rule.vars.clone();

        // Collect all variable names that are universal (declared)
        let universal_names: rustc_hash::FxHashSet<&str> =
            rule.vars.iter().map(|(name, _)| name.as_str()).collect();

        // For existentials, walk the head predicate application's
        // argument terms: any argument that is itself a plain variable
        // not among the declared universals is existential.
        let mut existential_vars: SmallVec<[(String, SortId); 4]> = SmallVec::new();
        if let crate::chc::RuleHead::Predicate(app) = &rule.head {
            let mut seen: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
            for &arg in &app.args {
                let Some(term) = terms.get(arg) else {
                    continue;
                };
                let TermKind::Var(name_spur) = &term.kind else {
                    // A compound (non-variable) head argument cannot itself
                    // be *the* existentially-bound variable; any existentials
                    // nested inside it are out of scope for this per-argument
                    // scan.
                    continue;
                };
                let name = terms.resolve_str(*name_spur);
                if !universal_names.contains(name) && seen.insert(name.to_string()) {
                    existential_vars.push((name.to_string(), term.sort));
                }
            }
        }

        let has_existentials = !existential_vars.is_empty();

        Self {
            existential_vars,
            universal_vars,
            has_existentials,
        }
    }

    /// Get the number of existential variables
    pub fn num_existentials(&self) -> usize {
        self.existential_vars.len()
    }
}

/// Skolemization context for existential variables
pub struct SkolemContext {
    /// Mapping from existential variables to Skolem functions/constants
    skolem_map: HashMap<String, TermId>,
    /// Fresh counter for Skolem names
    fresh_counter: u32,
}

impl SkolemContext {
    /// Create a new Skolemization context
    pub fn new() -> Self {
        Self {
            skolem_map: HashMap::new(),
            fresh_counter: 0,
        }
    }

    /// Skolemize an existential variable
    ///
    /// For `exists Y. phi(X, Y)` with free variables X, we create:
    /// - A Skolem constant if X is empty: `sk_Y`
    /// - A Skolem function otherwise: `sk_Y(X)`
    pub fn skolemize(
        &mut self,
        terms: &mut TermManager,
        var_name: &str,
        var_sort: SortId,
        free_vars: &[(String, SortId)],
    ) -> ExistentialResult<TermId> {
        // Check if already skolemized
        if let Some(&skolem) = self.skolem_map.get(var_name) {
            return Ok(skolem);
        }

        // Create Skolem term
        // For simplicity, we always create a Skolem constant
        // A full implementation would create Skolem functions for dependent variables
        let sk_name = if free_vars.is_empty() {
            self.fresh_skolem_name(var_name)
        } else {
            // Include dependencies in the name for uniqueness
            let dep_names: Vec<&str> = free_vars.iter().map(|(n, _)| n.as_str()).collect();
            format!(
                "{}_{}",
                self.fresh_skolem_name(var_name),
                dep_names.join("_")
            )
        };

        let skolem = terms.mk_var(&sk_name, var_sort);

        // Cache the Skolem term
        self.skolem_map.insert(var_name.to_string(), skolem);

        Ok(skolem)
    }

    /// Get fresh Skolem name
    fn fresh_skolem_name(&mut self, base: &str) -> String {
        let name = format!("sk_{}_{}", base, self.fresh_counter);
        self.fresh_counter += 1;
        name
    }
}

impl Default for SkolemContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Existential variable projector
///
/// Projects existential variables out of formulas using quantifier elimination
/// or approximation techniques.
pub struct ExistentialProjector;

impl ExistentialProjector {
    /// Project out existential variables from a formula
    ///
    /// Given a formula `phi(X, Y)` where Y are existential variables,
    /// compute an over-approximation `psi(X)` such that:
    /// - `phi(X, Y)` implies `psi(X)` for all Y
    /// - `psi` contains only variables from X
    pub fn project(
        terms: &mut TermManager,
        formula: TermId,
        existential_vars: &[(String, SortId)],
    ) -> ExistentialResult<TermId> {
        // If no existential variables, return formula as-is
        if existential_vars.is_empty() {
            return Ok(formula);
        }

        // Strategy 1: Syntactic projection (sound over-approximation)
        // If formula is a conjunction, drop all conjuncts containing existentials
        // and keep the rest. This is a sound over-approximation.
        Self::syntactic_projection(terms, formula, existential_vars)
    }

    /// Syntactic projection: drop literals containing existential variables
    ///
    /// This is a sound over-approximation - the result may be weaker than
    /// necessary but is guaranteed to be an over-approximation.
    ///
    /// The disjunction case descends into every disjunct, so the recursion
    /// depth used to equal the `Or`-nesting depth of parsed input; it is an
    /// explicit work stack now. The old code also swallowed a failed
    /// sub-projection with `unwrap_or_else(|_| mk_true())`, turning an
    /// error into a silent `true`; errors propagate with `?` instead.
    fn syntactic_projection(
        terms: &mut TermManager,
        formula: TermId,
        existential_vars: &[(String, SortId)],
    ) -> ExistentialResult<TermId> {
        use oxiz_core::TermKind;

        /// One step of the explicit projection stack.
        enum Work {
            /// Project this subformula.
            Eval(TermId),
            /// Combine the top `n` results into a disjunction.
            Disjoin(usize),
        }

        let mut work: Vec<Work> = vec![Work::Eval(formula)];
        let mut values: Vec<TermId> = Vec::new();

        while let Some(item) = work.pop() {
            match item {
                Work::Eval(current) => {
                    let Some(kind) = terms.get(current).map(|t| t.kind.clone()) else {
                        // Unknown subformula: `true` over-approximates anything.
                        let top = terms.mk_true();
                        values.push(top);
                        continue;
                    };

                    match kind {
                        TermKind::And(args) => {
                            // Keep only conjuncts free of existentials.
                            let projected: Vec<TermId> = args
                                .iter()
                                .copied()
                                .filter(|&arg| {
                                    !Self::contains_existential(arg, terms, existential_vars)
                                })
                                .collect();
                            let combined = match projected.as_slice() {
                                [] => terms.mk_true(),
                                [only] => *only,
                                _ => terms.mk_and(projected),
                            };
                            values.push(combined);
                        }
                        TermKind::Or(args) => {
                            work.push(Work::Disjoin(args.len()));
                            for &arg in args.iter().rev() {
                                work.push(Work::Eval(arg));
                            }
                        }
                        _ => {
                            // Atomic formula: keep it unless it mentions an
                            // existential, in which case project it to `true`.
                            if Self::contains_existential(current, terms, existential_vars) {
                                let top = terms.mk_true();
                                values.push(top);
                            } else {
                                values.push(current);
                            }
                        }
                    }
                }
                Work::Disjoin(count) => {
                    let args = values.split_off(values.len().saturating_sub(count));
                    let combined = terms.mk_or(args);
                    values.push(combined);
                }
            }
        }

        // Exactly one value is produced for the root; the fallback covers no
        // structurally reachable case, and `true` remains a sound answer.
        Ok(values.pop().unwrap_or_else(|| terms.mk_true()))
    }

    /// Check if a term contains any existential variables
    #[allow(dead_code)]
    fn contains_existential(
        term: TermId,
        terms: &TermManager,
        existential_vars: &[(String, SortId)],
    ) -> bool {
        // Enhanced implementation: traverse term AST to check for existential variables
        use std::collections::HashSet;

        // Build set of existential variable names for fast lookup
        let existential_names: HashSet<&str> = existential_vars
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();

        // Check if term contains any existential variable
        Self::contains_existential_rec(term, terms, &existential_names)
    }

    /// Helper for checking existential occurrence.
    ///
    /// This detector is **conservative in one direction only**: answering
    /// "yes, it may contain an existential" merely projects a conjunct away
    /// (a sound over-approximation), whereas answering "no" for a term that
    /// *does* mention an existential variable keeps that variable in the
    /// projected formula — an unsound result, because Spacer then treats a
    /// formula still quantified over Y as if it were over X alone.
    ///
    /// Two defects made the old recursive version answer "no" wrongly:
    ///
    /// * its `match` ended in `_ => false`, so an existential occurring
    ///   under any kind it did not enumerate — `Ite`, `Implies`, `Xor`,
    ///   `Distinct`, `Neg`, `Select`/`Store`, `Apply`, every bitvector and
    ///   string operation, a quantifier body, a `Let` body — was reported
    ///   as absent;
    /// * a dangling `TermId` also answered `false`, i.e. "definitely no
    ///   existential", when nothing at all is known about it.
    ///
    /// Both are fixed here: descent is via
    /// [`oxiz_core::ast::traversal::get_children`] (exhaustive over
    /// `TermKind`, so a new variant is a compile error there rather than a
    /// silent `false` here), and an unknown id answers `true` — the
    /// conservative direction. Descending into quantifier bodies without
    /// tracking shadowing is likewise conservative: a *bound* variable that
    /// happens to share a name is reported as an occurrence, which can only
    /// over-project.
    ///
    /// The walk is iterative with a `visited` set (see [`crate::walk`]): the
    /// old form both overflowed the stack on deeply nested input and
    /// re-expanded shared subterms exponentially.
    fn contains_existential_rec(
        term: TermId,
        terms: &TermManager,
        existential_names: &std::collections::HashSet<&str>,
    ) -> bool {
        use oxiz_core::TermKind;

        crate::walk::any_node(terms, term, |_, kind| match kind {
            Some(TermKind::Var(name_spur)) => {
                existential_names.contains(terms.resolve_str(*name_spur))
            }
            Some(_) => false,
            // Unknown term: assume the worst rather than claim it is clean.
            None => true,
        })
    }

    /// Compute model-based projection
    ///
    /// Given a model M for `phi(X, Y)`, project out Y to get a formula
    /// over X that is implied by the model.
    pub fn mbp(
        terms: &mut TermManager,
        formula: TermId,
        model: &HashMap<TermId, TermId>,
        existential_vars: &[TermId],
    ) -> ExistentialResult<TermId> {
        // Model-based projection substitutes existential variables with their values
        // from the model, then simplifies the result

        // Create a substitution from existential vars to their model values
        let mut subst = HashMap::new();
        for &var in existential_vars {
            if let Some(&value) = model.get(&var) {
                subst.insert(var, value);
            }
        }

        // Apply substitution to the formula
        let projected = Self::apply_substitution(terms, formula, &subst);

        // Simplify the result by evaluating any ground literals
        let simplified = Self::simplify_ground(terms, projected);

        Ok(simplified)
    }

    /// Apply a substitution to a term.
    ///
    /// Delegates to [`TermManager::substitute`], which rewrites *every*
    /// [`oxiz_core::TermKind`] variant (its `match` has no catch-all arm)
    /// with an explicit heap stack and a memo table.
    ///
    /// The previous hand-written walk rebuilt only `And`/`Or`/`Not`/`Eq`/
    /// `Le`/`Lt` and returned the term untouched under a `_ => term`
    /// fallthrough. Model-based projection therefore left existential
    /// variables *unsubstituted* whenever they occurred under any other
    /// operator — `Add`, `Sub`, `Mul`, `Ge`, `Gt`, `Ite`, `Implies`,
    /// `Select`/`Store`, `Apply`, a bitvector operation — so `mbp` returned
    /// a formula still mentioning the variables it was asked to eliminate.
    /// It also had no memo, so a shared DAG was re-expanded into fresh
    /// nodes exponentially, and no depth bound, so deep input overflowed
    /// the stack.
    fn apply_substitution(
        terms: &mut TermManager,
        term: TermId,
        subst: &HashMap<TermId, TermId>,
    ) -> TermId {
        let mapping: rustc_hash::FxHashMap<TermId, TermId> =
            subst.iter().map(|(&k, &v)| (k, v)).collect();
        terms.substitute(term, &mapping)
    }

    /// Simplify ground (variable-free) formulas.
    ///
    /// Bottom-up constant folding over the Boolean/comparison skeleton of
    /// `term`. The set of kinds descended into is exactly the set the prior
    /// recursive version descended into (`Not`, `And`, `Or`, `Eq`, `Lt`,
    /// `Le`, `Gt`, `Ge`); every other kind is a leaf that is returned
    /// unchanged, which is an *identity* rewrite and therefore semantics
    /// preserving, not a silent default. The folding rules are likewise
    /// carried over verbatim.
    ///
    /// What changes is only the mechanism: an explicit heap stack with a
    /// memo table keyed on [`TermId`] replaces native recursion, so a
    /// formula whose `And`/`Or` nesting comes from parsed input can no
    /// longer overflow the process stack, and a shared DAG is folded once
    /// per distinct node instead of once per path (the old form rebuilt
    /// fresh nodes exponentially).
    fn simplify_ground(terms: &mut TermManager, term: TermId) -> TermId {
        let mut memo: rustc_hash::FxHashMap<TermId, TermId> = rustc_hash::FxHashMap::default();
        let mut stack: Vec<(TermId, bool)> = vec![(term, false)];

        while let Some((current, expanded)) = stack.pop() {
            if memo.contains_key(&current) {
                continue;
            }
            let Some(kind) = terms.get(current).map(|t| t.kind.clone()) else {
                memo.insert(current, current);
                continue;
            };
            let Some(children) = Self::simplify_children(&kind) else {
                // Leaf for this rewrite: constants, variables, and every
                // operator the fold has no rule for, all map to themselves.
                memo.insert(current, current);
                continue;
            };

            if expanded {
                let folded: Vec<TermId> = children
                    .iter()
                    .map(|child| memo.get(child).copied().unwrap_or(*child))
                    .collect();
                let result = Self::fold_simplified(terms, &kind, &folded);
                memo.insert(current, result);
            } else {
                stack.push((current, true));
                for child in children {
                    if !memo.contains_key(&child) {
                        stack.push((child, false));
                    }
                }
            }
        }

        memo.get(&term).copied().unwrap_or(term)
    }

    /// The subterms [`Self::simplify_ground`] recurses into, or `None` when
    /// the kind is a leaf of that rewrite.
    fn simplify_children(kind: &oxiz_core::TermKind) -> Option<Vec<TermId>> {
        use oxiz_core::TermKind;

        match kind {
            TermKind::Not(arg) => Some(vec![*arg]),
            TermKind::And(args) | TermKind::Or(args) => Some(args.to_vec()),
            TermKind::Eq(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b) => Some(vec![*a, *b]),
            _ => None,
        }
    }

    /// Rebuild one node of [`Self::simplify_ground`] from its already
    /// simplified children, applying the same constant-folding rules the
    /// recursive implementation applied.
    fn fold_simplified(
        terms: &mut TermManager,
        kind: &oxiz_core::TermKind,
        folded: &[TermId],
    ) -> TermId {
        use oxiz_core::TermKind;

        match kind {
            TermKind::Not(_) => {
                let Some(&arg) = folded.first() else {
                    return terms.mk_true();
                };
                match terms.get(arg).map(|t| &t.kind) {
                    Some(TermKind::True) => terms.mk_false(),
                    Some(TermKind::False) => terms.mk_true(),
                    _ => terms.mk_not(arg),
                }
            }
            TermKind::And(_) => {
                if folded
                    .iter()
                    .any(|&arg| matches!(terms.get(arg).map(|t| &t.kind), Some(TermKind::False)))
                {
                    return terms.mk_false();
                }
                let kept: Vec<TermId> = folded
                    .iter()
                    .copied()
                    .filter(|&arg| !matches!(terms.get(arg).map(|t| &t.kind), Some(TermKind::True)))
                    .collect();
                match kept.as_slice() {
                    [] => terms.mk_true(),
                    [only] => *only,
                    _ => terms.mk_and(kept),
                }
            }
            TermKind::Or(_) => {
                if folded
                    .iter()
                    .any(|&arg| matches!(terms.get(arg).map(|t| &t.kind), Some(TermKind::True)))
                {
                    return terms.mk_true();
                }
                let kept: Vec<TermId> = folded
                    .iter()
                    .copied()
                    .filter(|&arg| {
                        !matches!(terms.get(arg).map(|t| &t.kind), Some(TermKind::False))
                    })
                    .collect();
                match kept.as_slice() {
                    [] => terms.mk_false(),
                    [only] => *only,
                    _ => terms.mk_or(kept),
                }
            }
            TermKind::Eq(_, _)
            | TermKind::Lt(_, _)
            | TermKind::Le(_, _)
            | TermKind::Gt(_, _)
            | TermKind::Ge(_, _) => {
                let (Some(&lhs), Some(&rhs)) = (folded.first(), folded.get(1)) else {
                    return terms.mk_true();
                };
                if let Some(value) = Self::fold_comparison(terms, kind, lhs, rhs) {
                    return terms.mk_bool(value);
                }
                match kind {
                    TermKind::Eq(_, _) => terms.mk_eq(lhs, rhs),
                    TermKind::Lt(_, _) => terms.mk_lt(lhs, rhs),
                    TermKind::Le(_, _) => terms.mk_le(lhs, rhs),
                    TermKind::Gt(_, _) => terms.mk_gt(lhs, rhs),
                    // The outer `match` restricts this arm to the five
                    // comparison kinds, and the four above are handled.
                    _ => terms.mk_ge(lhs, rhs),
                }
            }
            // `simplify_children` returns `None` for every other kind, so
            // no other kind ever reaches this rebuild step.
            _ => folded.first().copied().unwrap_or_else(|| terms.mk_true()),
        }
    }

    /// Evaluate a comparison whose operands folded to constants, or `None`
    /// when it cannot be decided syntactically.
    fn fold_comparison(
        terms: &TermManager,
        kind: &oxiz_core::TermKind,
        lhs: TermId,
        rhs: TermId,
    ) -> Option<bool> {
        use oxiz_core::TermKind;

        let lhs_kind = &terms.get(lhs)?.kind;
        let rhs_kind = &terms.get(rhs)?.kind;

        if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (lhs_kind, rhs_kind) {
            return Some(match kind {
                TermKind::Eq(_, _) => a == b,
                TermKind::Lt(_, _) => a < b,
                TermKind::Le(_, _) => a <= b,
                TermKind::Gt(_, _) => a > b,
                TermKind::Ge(_, _) => a >= b,
                _ => return None,
            });
        }

        // Boolean equality between the two constants.
        if matches!(kind, TermKind::Eq(_, _)) {
            return match (lhs_kind, rhs_kind) {
                (TermKind::True, TermKind::True) | (TermKind::False, TermKind::False) => Some(true),
                (TermKind::True, TermKind::False) | (TermKind::False, TermKind::True) => {
                    Some(false)
                }
                _ => None,
            };
        }

        None
    }
}

/// Witness extraction for existential variables
///
/// Extracts concrete values (witnesses) for existential variables from
/// counterexamples or models.
pub struct WitnessExtractor;

impl WitnessExtractor {
    /// Extract witnesses for existential variables from a model.
    ///
    /// Matching a witness value to the variable it belongs to requires
    /// inspecting each model-key term's [`TermKind::Var`] name, which in turn
    /// needs the owning [`TermManager`]. This method therefore delegates to
    /// [`WitnessExtractor::extract_witnesses_with_terms`]. A previous version
    /// assigned an arbitrary (hash-ordered) model entry to every existential
    /// variable, producing wrong values under wrong names — an unsound result
    /// that this delegation removes.
    pub fn extract_witnesses(
        terms: &TermManager,
        model: &HashMap<TermId, TermId>,
        existential_vars: &[(String, SortId)],
    ) -> HashMap<String, TermId> {
        Self::extract_witnesses_with_terms(terms, model, existential_vars)
    }

    /// Extract witnesses with term manager access for better name matching
    pub fn extract_witnesses_with_terms(
        terms: &TermManager,
        model: &HashMap<TermId, TermId>,
        existential_vars: &[(String, SortId)],
    ) -> HashMap<String, TermId> {
        use oxiz_core::TermKind;

        let mut witnesses = HashMap::new();

        for (var_name, _sort) in existential_vars {
            // Search for the variable term in the model
            for (&term_id, &value) in model {
                if let Some(term) = terms.get(term_id) {
                    // Check if this is a variable with the matching name
                    if let TermKind::Var(name_spur) = &term.kind {
                        // Resolve the Spur to a string for comparison
                        if terms.resolve_str(*name_spur) == var_name {
                            witnesses.insert(var_name.clone(), value);
                            break;
                        }
                    }
                }
            }
        }

        witnesses
    }
}

/// Existential quantifier handler
pub struct ExistentialHandler {
    /// Skolem context
    skolem_ctx: SkolemContext,
    /// Cache of analyzed rules
    rule_cache: HashMap<usize, ExistentialInfo>,
}

impl ExistentialHandler {
    /// Create a new existential handler
    pub fn new() -> Self {
        Self {
            skolem_ctx: SkolemContext::new(),
            rule_cache: HashMap::new(),
        }
    }

    /// Analyze a rule for existentials
    pub fn analyze_rule(
        &mut self,
        rule_id: usize,
        rule: &Rule,
        terms: &TermManager,
    ) -> &ExistentialInfo {
        self.rule_cache
            .entry(rule_id)
            .or_insert_with(|| ExistentialInfo::analyze(rule, terms))
    }

    /// Preprocess a rule by eliminating existentials
    ///
    /// This function:
    /// 1. Identifies existential variables in the rule
    /// 2. Skolemizes them using fresh Skolem constants/functions
    /// 3. Returns a transformed rule with existentials replaced by Skolem terms
    pub fn preprocess_rule(
        &mut self,
        terms: &mut TermManager,
        _pred: PredId,
        rule: &Rule,
    ) -> ExistentialResult<Rule> {
        // Step 1: Analyze rule for existential variables
        // Clone the info to avoid borrow checker issues
        let info = self
            .analyze_rule(rule.id.raw() as usize, rule, terms)
            .clone();

        // If no existentials, return rule unchanged
        if !info.has_existentials || info.existential_vars.is_empty() {
            return Ok(rule.clone());
        }

        // Step 2: Skolemize existential variables
        // The universal variables are the free variables for Skolemization
        let mut skolem_substitution: HashMap<String, TermId> = HashMap::new();

        for (ex_var_name, ex_var_sort) in &info.existential_vars {
            let skolem_term = self.skolem_ctx.skolemize(
                terms,
                ex_var_name,
                *ex_var_sort,
                &info.universal_vars,
            )?;
            skolem_substitution.insert(ex_var_name.clone(), skolem_term);
        }

        // Step 3: Transform the rule by replacing every existential's
        // occurrence in the head with its Skolem term, and adding the
        // Skolem variables to the universal quantifiers.
        //
        // Existentials appear only in the head (per this module's own
        // documented contract), so only the head's predicate-application
        // arguments need rewriting -- the body never references them.
        let mut new_vars = rule.vars.clone();
        for (ex_var_name, ex_var_sort) in &info.existential_vars {
            if let Some(&skolem_term) = skolem_substitution.get(ex_var_name)
                && let Some(term) = terms.get(skolem_term)
                && let TermKind::Var(spur) = &term.kind
            {
                let name = terms.resolve_str(*spur);
                new_vars.push((name.to_string(), *ex_var_sort));
            }
        }

        let new_head = match &rule.head {
            RuleHead::Predicate(app) => {
                let new_args: SmallVec<[TermId; 4]> = app
                    .args
                    .iter()
                    .map(|&arg| {
                        // Replace an argument only if it's itself a plain
                        // variable matching one of the existentials this
                        // rule was just Skolemized for; every other
                        // argument (a universal variable, or any compound
                        // term) passes through unchanged.
                        let Some(term) = terms.get(arg) else {
                            return arg;
                        };
                        let TermKind::Var(name_spur) = &term.kind else {
                            return arg;
                        };
                        let name = terms.resolve_str(*name_spur);
                        skolem_substitution.get(name).copied().unwrap_or(arg)
                    })
                    .collect();
                RuleHead::Predicate(PredicateApp {
                    pred: app.pred,
                    args: new_args,
                })
            }
            RuleHead::Query => RuleHead::Query,
        };

        let transformed_rule = Rule {
            id: rule.id,
            vars: new_vars,
            body: rule.body.clone(),
            head: new_head,
            name: rule.name.clone(),
        };

        Ok(transformed_rule)
    }

    /// Get Skolem context
    pub fn skolem_context(&self) -> &SkolemContext {
        &self.skolem_ctx
    }

    /// Get Skolem context (mutable)
    pub fn skolem_context_mut(&mut self) -> &mut SkolemContext {
        &mut self.skolem_ctx
    }
}

impl Default for ExistentialHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stack size and nesting depth shared by the deep-recursion tests below.
    ///
    /// The two are scaled together on purpose: what these tests actually pin
    /// is the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). A natively recursive projection needs far more than
    /// that per frame and still overflows, so the regression keeps every bit
    /// of its detection power. The pair used to be 1 MiB / 50_000 -- the same
    /// 21 bytes. `mk_and`/`mk_or` flatten their arguments (so `acc =
    /// mk_or([acc, lit])` never actually nests, and is quadratic to boot),
    /// so the deep terms below are built with `TermManager::intern_term`
    /// directly, which neither flattens nor re-interns already-built
    /// prefixes. Never raise `DEEP_DEPTH` without raising `DEEP_STACK` by
    /// the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

    #[test]
    fn test_skolem_context_fresh_names() {
        let mut ctx = SkolemContext::new();

        let name1 = ctx.fresh_skolem_name("x");
        let name2 = ctx.fresh_skolem_name("x");
        let name3 = ctx.fresh_skolem_name("y");

        assert_eq!(name1, "sk_x_0");
        assert_eq!(name2, "sk_x_1");
        assert_eq!(name3, "sk_y_2");
    }

    #[test]
    fn test_existential_info_no_existentials() {
        let info = ExistentialInfo {
            existential_vars: SmallVec::new(),
            universal_vars: vec![("x".to_string(), SortId(0))].into(),
            has_existentials: false,
        };

        assert_eq!(info.num_existentials(), 0);
        assert!(!info.has_existentials);
    }

    #[test]
    fn test_existential_info_with_existentials() {
        let info = ExistentialInfo {
            existential_vars: vec![("y1".to_string(), SortId(1)), ("y2".to_string(), SortId(1))]
                .into(),
            universal_vars: vec![("x".to_string(), SortId(0))].into(),
            has_existentials: true,
        };

        assert_eq!(info.num_existentials(), 2);
        assert!(info.has_existentials);
    }

    // -----------------------------------------------------------------------
    // Regression tests for the `sweep-backend-misc` triage sweep:
    // `ExistentialInfo::analyze` used to leave `existential_vars` always
    // empty (an unpopulated `SmallVec::new()`), so it was structurally
    // impossible for `ExistentialHandler::preprocess_rule` to ever
    // actually Skolemize anything, regardless of `has_existentials`.
    // -----------------------------------------------------------------------

    /// Build a rule whose head references one declared universal
    /// variable (`univ_x`) and one variable that is *not* declared in
    /// `vars` (`exist_y`) -- exactly the shape of a genuine existential.
    fn build_existential_rule() -> (
        TermManager,
        crate::chc::ChcSystem,
        crate::chc::RuleId,
        TermId,
        TermId,
    ) {
        let mut terms = TermManager::new();
        let mut system = crate::chc::ChcSystem::new();
        let pred =
            system.declare_predicate("ExistInv", [terms.sorts.int_sort, terms.sorts.int_sort]);

        let x = terms.mk_var("univ_x", terms.sorts.int_sort);
        let y = terms.mk_var("exist_y", terms.sorts.int_sort);
        let true_term = terms.mk_true();

        let rule_id = system.add_init_rule(
            [("univ_x".to_string(), terms.sorts.int_sort)],
            true_term,
            pred,
            [x, y],
        );

        (terms, system, rule_id, x, y)
    }

    #[test]
    fn test_analyze_populates_existential_vars_from_head() {
        let (terms, system, rule_id, _x, _y) = build_existential_rule();
        let rule = system.get_rule(rule_id).expect("rule must exist");

        let info = ExistentialInfo::analyze(rule, &terms);

        assert!(
            info.has_existentials,
            "a head argument absent from `vars` must be flagged existential"
        );
        assert_eq!(
            info.num_existentials(),
            1,
            "exactly one head argument (exist_y) is undeclared"
        );
        assert_eq!(info.existential_vars[0].0, "exist_y");

        // The declared universal `univ_x` must NOT be misclassified as
        // existential.
        assert!(
            !info
                .existential_vars
                .iter()
                .any(|(name, _)| name == "univ_x")
        );
    }

    #[test]
    fn test_preprocess_rule_applies_skolem_substitution_to_head() {
        let (mut terms, system, rule_id, x, y) = build_existential_rule();
        let rule = system.get_rule(rule_id).expect("rule must exist").clone();
        let pred = rule
            .head
            .as_predicate()
            .expect("head is a predicate application")
            .pred;

        let mut handler = ExistentialHandler::new();
        let transformed = handler
            .preprocess_rule(&mut terms, pred, &rule)
            .expect("preprocessing should not error");

        let RuleHead::Predicate(app) = &transformed.head else {
            panic!("transformed head must still be a predicate application");
        };

        // The universal argument (x) must be left untouched...
        assert_eq!(
            app.args[0], x,
            "a declared universal variable must not be rewritten"
        );
        // ...but the existential argument (y) must have been replaced by
        // a genuinely different (Skolem) term -- this is the exact
        // substitution the old code's own comment admitted never
        // happened ("we would also need to apply the substitution to
        // the rule body and head constraints/arguments").
        assert_ne!(
            app.args[1], y,
            "the existential argument must be replaced by its Skolem term, \
             not left as the original (unbound, out-of-scope) variable"
        );

        let skolem_term = terms.get(app.args[1]).expect("skolem term must exist");
        let TermKind::Var(spur) = &skolem_term.kind else {
            panic!("skolem term must be a variable");
        };
        let skolem_name = terms.resolve_str(*spur).to_string();
        assert!(
            skolem_name.starts_with("sk_exist_y"),
            "expected a Skolem name derived from exist_y, got {skolem_name:?}"
        );

        // The Skolem variable must be declared in the transformed rule's
        // universal variables so the rule remains well-formed (every
        // variable referenced in head/body is declared).
        assert!(
            transformed
                .vars
                .iter()
                .any(|(name, _)| *name == skolem_name),
            "the Skolem variable must be added to the transformed rule's \
             declared variables"
        );
    }

    // -----------------------------------------------------------------------
    // Unbounded-recursion / conservativeness regressions.
    // -----------------------------------------------------------------------

    /// `contains_existential` must answer "yes" for an existential hidden
    /// under an operator the old enumeration did not descend into. Answering
    /// "no" there let the projector keep a conjunct still quantified over Y.
    #[test]
    fn contains_existential_sees_through_every_operator() {
        let mut terms = TermManager::new();
        let int_sort = terms.sorts.int_sort;
        let bool_sort = terms.sorts.bool_sort;
        let y = terms.mk_var("y", int_sort);
        let x = terms.mk_var("x", int_sort);
        let zero = terms.mk_int(0);
        let existentials = [("y".to_string(), int_sort)];

        // `Ite(c, y, 0)`: old code returned `false` (no `Ite` arm).
        let cond = terms.mk_var("c", bool_sort);
        let ite = terms.mk_ite(cond, y, zero);
        assert!(
            ExistentialProjector::contains_existential(ite, &terms, &existentials),
            "existential under Ite must be detected"
        );

        // `Implies(x = 0, y = 0)`.
        let x_eq = terms.mk_eq(x, zero);
        let y_eq = terms.mk_eq(y, zero);
        let implies = terms.mk_implies(x_eq, y_eq);
        assert!(
            ExistentialProjector::contains_existential(implies, &terms, &existentials),
            "existential under Implies must be detected"
        );

        // Negative polarity: a formula over `x` only must answer "no".
        assert!(
            !ExistentialProjector::contains_existential(x_eq, &terms, &existentials),
            "a formula free of existentials must not be reported as containing one"
        );
    }

    /// Deep `Or` nesting must not overflow the stack during projection.
    #[test]
    fn syntactic_projection_survives_deep_nesting() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut terms = TermManager::new();
                let bool_sort = terms.sorts.bool_sort;
                let int_sort = terms.sorts.int_sort;
                let x = terms.mk_var("x", int_sort);
                let zero = terms.mk_int(0);
                let atom = terms.mk_eq(x, zero);
                let mut formula = atom;
                // Interned directly (not via `mk_or`, which would flatten
                // this back into one wide `Or`) so the tree really is
                // `DEEP_DEPTH` deep.
                for i in 0..DEEP_DEPTH {
                    let lit = terms.mk_var(&format!("v{i}"), bool_sort);
                    formula = terms.intern_term(
                        oxiz_core::TermKind::Or(vec![formula, lit].into()),
                        bool_sort,
                    );
                }
                let existentials = [("y".to_string(), int_sort)];
                let projected = ExistentialProjector::project(&mut terms, formula, &existentials);
                assert!(projected.is_ok(), "deep projection must return");
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep projection must not overflow");
    }

    /// `mbp` must substitute existential variables wherever they occur, not
    /// only under the six operators the old walk rebuilt, and must survive
    /// deep nesting.
    #[test]
    fn mbp_substitutes_under_arithmetic_and_survives_depth() {
        let mut terms = TermManager::new();
        let int_sort = terms.sorts.int_sort;
        let x = terms.mk_var("x", int_sort);
        let y = terms.mk_var("y", int_sort);
        let five = terms.mk_int(5);
        // `x + y >= 0` -- `y` sits under `Add`, which the old walk skipped.
        let sum = terms.mk_add([x, y]);
        let zero = terms.mk_int(0);
        let formula = terms.mk_ge(sum, zero);

        let mut model = HashMap::new();
        model.insert(y, five);
        let projected = ExistentialProjector::mbp(&mut terms, formula, &model, &[y])
            .expect("mbp should succeed");

        let names = std::collections::HashSet::from(["y"]);
        assert!(
            !ExistentialProjector::contains_existential_rec(projected, &terms, &names),
            "`y` must be gone from the projected formula"
        );
    }

    /// Ground folding must produce the same answers as before and must not
    /// recurse natively.
    #[test]
    fn simplify_ground_folds_constants_and_survives_depth() {
        let mut terms = TermManager::new();
        let one = terms.mk_int(1);
        let two = terms.mk_int(2);
        let lt = terms.mk_lt(one, two);
        let folded = ExistentialProjector::simplify_ground(&mut terms, lt);
        assert_eq!(
            terms.get(folded).map(|t| t.kind.clone()),
            Some(oxiz_core::TermKind::True),
            "1 < 2 must fold to true"
        );

        let ge = terms.mk_ge(one, two);
        let folded = ExistentialProjector::simplify_ground(&mut terms, ge);
        assert_eq!(
            terms.get(folded).map(|t| t.kind.clone()),
            Some(oxiz_core::TermKind::False),
            "1 >= 2 must fold to false"
        );

        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut terms = TermManager::new();
                let bool_sort = terms.sorts.bool_sort;
                let mut formula = terms.mk_var("b0", bool_sort);
                // Interned directly (not via `mk_and`, which would flatten
                // this back into one wide `And`) so the tree really is
                // `DEEP_DEPTH` deep.
                for i in 1..DEEP_DEPTH {
                    let lit = terms.mk_var(&format!("b{i}"), bool_sort);
                    formula = terms.intern_term(
                        oxiz_core::TermKind::And(vec![formula, lit].into()),
                        bool_sort,
                    );
                }
                let folded = ExistentialProjector::simplify_ground(&mut terms, formula);
                assert!(terms.get(folded).is_some(), "deep fold must return a term");
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep simplify_ground must return");
    }
}
