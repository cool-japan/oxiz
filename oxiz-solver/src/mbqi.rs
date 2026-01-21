//! Model-Based Quantifier Instantiation (MBQI)
//!
//! This module implements MBQI for handling universally quantified formulas.
//! The algorithm works by:
//!
//! 1. Getting a partial model M from the main solver
//! 2. For each universal quantifier ∀x.φ(x): specialize φ with M
//! 3. Search for a counterexample x' where ¬φ(x') holds under M
//! 4. If found: add instantiation φ(x') as a lemma
//! 5. Repeat until no counterexamples are found
//!
//! # References
//!
//! - Ge, Y., & de Moura, L. (2009). Complete instantiation for quantified formulas
//!   in satisfiability modulo theories.

use lasso::Spur;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// A quantified formula tracked by MBQI
#[derive(Debug, Clone)]
pub struct QuantifiedFormula {
    /// The original quantified term
    pub term: TermId,
    /// Bound variables (name, sort)
    pub bound_vars: SmallVec<[(Spur, SortId); 2]>,
    /// The body of the quantifier
    pub body: TermId,
    /// Whether this is universal (true) or existential (false)
    pub universal: bool,
    /// Number of times this quantifier has been instantiated
    pub instantiation_count: usize,
    /// Maximum allowed instantiations
    pub max_instantiations: usize,
}

impl QuantifiedFormula {
    /// Create a new tracked quantified formula
    pub fn new(
        term: TermId,
        bound_vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        universal: bool,
    ) -> Self {
        Self {
            term,
            bound_vars,
            body,
            universal,
            instantiation_count: 0,
            max_instantiations: 100,
        }
    }

    /// Check if we can instantiate more
    pub fn can_instantiate(&self) -> bool {
        self.instantiation_count < self.max_instantiations
    }
}

/// An instantiation of a quantified formula
#[derive(Debug, Clone)]
pub struct Instantiation {
    /// The quantifier that was instantiated
    pub quantifier: TermId,
    /// The substitution used (variable name -> term)
    pub substitution: FxHashMap<Spur, TermId>,
    /// The resulting ground term (body with substitution applied)
    pub result: TermId,
}

/// Result of MBQI check
#[derive(Debug, Clone)]
pub enum MBQIResult {
    /// No quantifiers to process
    NoQuantifiers,
    /// All quantifiers satisfied under the model
    Satisfied,
    /// Found new instantiations to add
    NewInstantiations(Vec<Instantiation>),
    /// Found a conflict (quantifier cannot be satisfied)
    Conflict(Vec<TermId>),
    /// Reached instantiation limit
    InstantiationLimit,
}

/// Model-Based Quantifier Instantiation solver
#[derive(Debug)]
pub struct MBQISolver {
    /// Tracked quantified formulas
    quantifiers: Vec<QuantifiedFormula>,
    /// Generated instantiations (for deduplication)
    generated_instantiations: FxHashSet<(TermId, Vec<(Spur, TermId)>)>,
    /// Candidate terms by sort (for instantiation)
    candidates_by_sort: FxHashMap<SortId, Vec<TermId>>,
    /// Maximum total instantiations
    max_total_instantiations: usize,
    /// Current total instantiation count
    total_instantiation_count: usize,
    /// Whether MBQI is enabled
    enabled: bool,
}

impl Default for MBQISolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MBQISolver {
    /// Create a new MBQI solver
    pub fn new() -> Self {
        Self {
            quantifiers: Vec::new(),
            generated_instantiations: FxHashSet::default(),
            candidates_by_sort: FxHashMap::default(),
            max_total_instantiations: 10000,
            total_instantiation_count: 0,
            enabled: true,
        }
    }

    /// Create with custom instantiation limit
    pub fn with_limit(max_total: usize) -> Self {
        let mut solver = Self::new();
        solver.max_total_instantiations = max_total;
        solver
    }

    /// Enable or disable MBQI
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if MBQI is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.quantifiers.clear();
        self.generated_instantiations.clear();
        self.candidates_by_sort.clear();
        self.total_instantiation_count = 0;
    }

    /// Add a quantified formula to track
    pub fn add_quantifier(&mut self, term: TermId, manager: &TermManager) {
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Forall { vars, body, .. } => {
                self.quantifiers
                    .push(QuantifiedFormula::new(term, vars.clone(), *body, true));
            }
            TermKind::Exists { vars, body, .. } => {
                // Existentials can be handled by negating and treating as universal
                // ∃x.φ(x) ≡ ¬∀x.¬φ(x)
                // For now, just track them
                self.quantifiers
                    .push(QuantifiedFormula::new(term, vars.clone(), *body, false));
            }
            _ => {}
        }
    }

    /// Register a candidate term for instantiation
    pub fn add_candidate(&mut self, term: TermId, sort: SortId) {
        self.candidates_by_sort.entry(sort).or_default().push(term);
    }

    /// Collect ground terms from a formula for use as instantiation candidates
    pub fn collect_ground_terms(&mut self, term: TermId, manager: &TermManager) {
        self.collect_ground_terms_rec(term, manager, &mut FxHashSet::default());
    }

    fn collect_ground_terms_rec(
        &mut self,
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return;
        };

        // Check if this is a ground term (no quantified variables)
        // For simplicity, we treat all non-quantifier terms as potential candidates
        match &t.kind {
            TermKind::Var(_) => {
                // Variables are candidates
                self.add_candidate(term, t.sort);
            }
            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                self.add_candidate(term, t.sort);
            }
            TermKind::Apply { args, .. } => {
                // Function applications might be good candidates
                self.add_candidate(term, t.sort);
                for &arg in args {
                    self.collect_ground_terms_rec(arg, manager, visited);
                }
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args {
                    self.collect_ground_terms_rec(arg, manager, visited);
                }
            }
            TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs)
            | TermKind::Mod(lhs, rhs)
            | TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs) => {
                self.collect_ground_terms_rec(*lhs, manager, visited);
                self.collect_ground_terms_rec(*rhs, manager, visited);
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.collect_ground_terms_rec(*arg, manager, visited);
            }
            TermKind::Ite(cond, then_br, else_br) => {
                self.collect_ground_terms_rec(*cond, manager, visited);
                self.collect_ground_terms_rec(*then_br, manager, visited);
                self.collect_ground_terms_rec(*else_br, manager, visited);
            }
            TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                // Don't descend into quantifier bodies for candidate collection
                // (those terms are not ground)
                let _ = body;
            }
            _ => {}
        }
    }

    /// Get candidates for a given sort
    pub fn get_candidates(&self, sort: SortId) -> &[TermId] {
        self.candidates_by_sort
            .get(&sort)
            .map_or(&[], |v| v.as_slice())
    }

    /// Check if we've reached the instantiation limit
    pub fn at_limit(&self) -> bool {
        self.total_instantiation_count >= self.max_total_instantiations
    }

    /// Perform MBQI with the given model
    ///
    /// This is the main entry point for MBQI. Given a partial model from the
    /// main solver, it tries to find counterexamples to universal quantifiers
    /// and generates instantiation lemmas.
    pub fn check_with_model(
        &mut self,
        model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> MBQIResult {
        if !self.enabled {
            return MBQIResult::NoQuantifiers;
        }

        if self.quantifiers.is_empty() {
            return MBQIResult::NoQuantifiers;
        }

        if self.at_limit() {
            return MBQIResult::InstantiationLimit;
        }

        let mut new_instantiations = Vec::new();

        // Process each universal quantifier
        for i in 0..self.quantifiers.len() {
            if !self.quantifiers[i].can_instantiate() {
                continue;
            }

            if !self.quantifiers[i].universal {
                // Skip existential quantifiers for now
                // (they require different handling)
                continue;
            }

            // Try to find counterexamples
            let instantiations = self.find_counterexamples(i, model, manager);

            for inst in instantiations {
                if self.at_limit() {
                    break;
                }

                // Check for duplicates
                let mut key_vec: Vec<_> = inst.substitution.iter().map(|(&k, &v)| (k, v)).collect();
                key_vec.sort_by_key(|(k, _)| *k);
                let key = (inst.quantifier, key_vec);

                if self.generated_instantiations.contains(&key) {
                    continue;
                }

                self.generated_instantiations.insert(key);
                self.quantifiers[i].instantiation_count += 1;
                self.total_instantiation_count += 1;
                new_instantiations.push(inst);
            }
        }

        if new_instantiations.is_empty() {
            MBQIResult::Satisfied
        } else {
            MBQIResult::NewInstantiations(new_instantiations)
        }
    }

    /// Find counterexamples for a quantifier under the given model
    fn find_counterexamples(
        &self,
        quantifier_idx: usize,
        model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let quant = &self.quantifiers[quantifier_idx];
        let mut results = Vec::new();

        // Strategy 1: Try candidates from the term pool
        let candidates = self.build_candidate_lists(&quant.bound_vars, manager);

        // Generate combinations of candidates
        let combinations = self.enumerate_combinations(&candidates, 10); // Limit combinations

        for combo in combinations {
            // Build substitution
            let mut subst: FxHashMap<Spur, TermId> = FxHashMap::default();
            for (i, &candidate) in combo.iter().enumerate() {
                if i < quant.bound_vars.len() {
                    subst.insert(quant.bound_vars[i].0, candidate);
                }
            }

            // Apply substitution to get ground instance
            let ground_body = self.apply_substitution(quant.body, &subst, manager);

            // Evaluate under the model
            let evaluated = self.evaluate_under_model(ground_body, model, manager);

            // Check if this is a counterexample (evaluates to false for ∀x.φ(x))
            if let Some(t) = manager.get(evaluated)
                && matches!(t.kind, TermKind::False)
            {
                results.push(Instantiation {
                    quantifier: quant.term,
                    substitution: subst,
                    result: ground_body,
                });

                // Limit counterexamples per quantifier per round
                if results.len() >= 5 {
                    break;
                }
            }
        }

        // Strategy 2: Use model values if no candidates found counterexamples
        if results.is_empty() {
            // Try to use values from the model for variables of matching sort
            let model_instantiation = self.instantiate_from_model(quantifier_idx, model, manager);
            if let Some(inst) = model_instantiation {
                results.push(inst);
            }
        }

        results
    }

    /// Build candidate lists for each bound variable
    fn build_candidate_lists(
        &self,
        bound_vars: &[(Spur, SortId)],
        manager: &mut TermManager,
    ) -> Vec<Vec<TermId>> {
        let mut result = Vec::new();

        for &(_name, sort) in bound_vars {
            let mut candidates = Vec::new();

            // Add candidates from the pool
            if let Some(pool) = self.candidates_by_sort.get(&sort) {
                candidates.extend(pool.iter().copied().take(10)); // Limit per sort
            }

            // Add default values based on sort
            if sort == manager.sorts.int_sort {
                // Add some default integers
                for i in 0..3 {
                    let int_id = manager.mk_int(i);
                    if !candidates.contains(&int_id) {
                        candidates.push(int_id);
                    }
                }
            } else if sort == manager.sorts.bool_sort {
                let true_id = manager.mk_true();
                let false_id = manager.mk_false();
                if !candidates.contains(&true_id) {
                    candidates.push(true_id);
                }
                if !candidates.contains(&false_id) {
                    candidates.push(false_id);
                }
            }

            result.push(candidates);
        }

        result
    }

    /// Enumerate combinations of candidates (limited)
    fn enumerate_combinations(
        &self,
        candidates: &[Vec<TermId>],
        max_per_dim: usize,
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

            // Limit total combinations
            if results.len() >= 100 {
                break;
            }

            // Increment indices
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
                // Overflow - we've tried all combinations
                break;
            }
        }

        results
    }

    /// Apply a substitution to a term
    #[allow(clippy::only_used_in_recursion)]
    fn apply_substitution(
        &self,
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        // Use the term manager's substitute method if available
        // For now, implement a simple recursive substitution
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match &t.kind {
            TermKind::Var(name) => {
                // Check if this variable should be substituted
                if let Some(&replacement) = subst.get(name) {
                    replacement
                } else {
                    term
                }
            }
            TermKind::Not(arg) => {
                let new_arg = self.apply_substitution(*arg, subst, manager);
                manager.mk_not(new_arg)
            }
            TermKind::And(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_substitution(a, subst, manager))
                    .collect();
                manager.mk_and(new_args)
            }
            TermKind::Or(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_substitution(a, subst, manager))
                    .collect();
                manager.mk_or(new_args)
            }
            TermKind::Eq(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_eq(new_lhs, new_rhs)
            }
            TermKind::Lt(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_lt(new_lhs, new_rhs)
            }
            TermKind::Le(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_le(new_lhs, new_rhs)
            }
            TermKind::Gt(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_gt(new_lhs, new_rhs)
            }
            TermKind::Ge(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_ge(new_lhs, new_rhs)
            }
            TermKind::Add(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution(a, subst, manager))
                    .collect();
                manager.mk_add(new_args)
            }
            TermKind::Sub(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_sub(new_lhs, new_rhs)
            }
            TermKind::Mul(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution(a, subst, manager))
                    .collect();
                manager.mk_mul(new_args)
            }
            TermKind::Neg(arg) => {
                let new_arg = self.apply_substitution(*arg, subst, manager);
                manager.mk_neg(new_arg)
            }
            TermKind::Implies(lhs, rhs) => {
                let new_lhs = self.apply_substitution(*lhs, subst, manager);
                let new_rhs = self.apply_substitution(*rhs, subst, manager);
                manager.mk_implies(new_lhs, new_rhs)
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let new_cond = self.apply_substitution(*cond, subst, manager);
                let new_then = self.apply_substitution(*then_br, subst, manager);
                let new_else = self.apply_substitution(*else_br, subst, manager);
                manager.mk_ite(new_cond, new_then, new_else)
            }
            TermKind::Apply { func, args } => {
                let func_name = manager.resolve_str(*func).to_string();
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution(a, subst, manager))
                    .collect();
                manager.mk_apply(&func_name, new_args, t.sort)
            }
            // Constants and other terms don't need substitution
            _ => term,
        }
    }

    /// Evaluate a term under a model
    #[allow(clippy::only_used_in_recursion)]
    fn evaluate_under_model(
        &self,
        term: TermId,
        model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        // Check if we have a direct model value
        if let Some(&val) = model.get(&term) {
            return val;
        }

        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match &t.kind {
            TermKind::True | TermKind::False | TermKind::IntConst(_) | TermKind::RealConst(_) => {
                // Constants evaluate to themselves
                term
            }
            TermKind::Var(_) => {
                // Variables: look up in model
                model.get(&term).copied().unwrap_or(term)
            }
            TermKind::Not(arg) => {
                let eval_arg = self.evaluate_under_model(*arg, model, manager);
                if let Some(arg_t) = manager.get(eval_arg) {
                    match arg_t.kind {
                        TermKind::True => return manager.mk_false(),
                        TermKind::False => return manager.mk_true(),
                        _ => {}
                    }
                }
                manager.mk_not(eval_arg)
            }
            TermKind::And(args) => {
                let mut all_true = true;
                for &arg in args {
                    let eval_arg = self.evaluate_under_model(arg, model, manager);
                    if let Some(arg_t) = manager.get(eval_arg) {
                        match arg_t.kind {
                            TermKind::False => return manager.mk_false(),
                            TermKind::True => continue,
                            _ => all_true = false,
                        }
                    } else {
                        all_true = false;
                    }
                }
                if all_true { manager.mk_true() } else { term }
            }
            TermKind::Or(args) => {
                let mut all_false = true;
                for &arg in args {
                    let eval_arg = self.evaluate_under_model(arg, model, manager);
                    if let Some(arg_t) = manager.get(eval_arg) {
                        match arg_t.kind {
                            TermKind::True => return manager.mk_true(),
                            TermKind::False => continue,
                            _ => all_false = false,
                        }
                    } else {
                        all_false = false;
                    }
                }
                if all_false { manager.mk_false() } else { term }
            }
            TermKind::Eq(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model(*lhs, model, manager);
                let eval_rhs = self.evaluate_under_model(*rhs, model, manager);

                // Simple equality check for constants
                if eval_lhs == eval_rhs {
                    return manager.mk_true();
                }

                // Check integer constants
                let lhs_t = manager.get(eval_lhs).cloned();
                let rhs_t = manager.get(eval_rhs).cloned();

                if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
                    match (&l.kind, &r.kind) {
                        (TermKind::IntConst(a), TermKind::IntConst(b)) => {
                            if a == b {
                                return manager.mk_true();
                            } else {
                                return manager.mk_false();
                            }
                        }
                        (TermKind::True, TermKind::True) | (TermKind::False, TermKind::False) => {
                            return manager.mk_true();
                        }
                        (TermKind::True, TermKind::False) | (TermKind::False, TermKind::True) => {
                            return manager.mk_false();
                        }
                        _ => {}
                    }
                }

                term
            }
            TermKind::Lt(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model(*lhs, model, manager);
                let eval_rhs = self.evaluate_under_model(*rhs, model, manager);

                let lhs_t = manager.get(eval_lhs).cloned();
                let rhs_t = manager.get(eval_rhs).cloned();

                if let (Some(l), Some(r)) = (lhs_t, rhs_t)
                    && let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind)
                {
                    if a < b {
                        return manager.mk_true();
                    } else {
                        return manager.mk_false();
                    }
                }

                term
            }
            TermKind::Le(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model(*lhs, model, manager);
                let eval_rhs = self.evaluate_under_model(*rhs, model, manager);

                let lhs_t = manager.get(eval_lhs).cloned();
                let rhs_t = manager.get(eval_rhs).cloned();

                if let (Some(l), Some(r)) = (lhs_t, rhs_t)
                    && let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind)
                {
                    if a <= b {
                        return manager.mk_true();
                    } else {
                        return manager.mk_false();
                    }
                }

                term
            }
            TermKind::Gt(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model(*lhs, model, manager);
                let eval_rhs = self.evaluate_under_model(*rhs, model, manager);

                let lhs_t = manager.get(eval_lhs).cloned();
                let rhs_t = manager.get(eval_rhs).cloned();

                if let (Some(l), Some(r)) = (lhs_t, rhs_t)
                    && let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind)
                {
                    if a > b {
                        return manager.mk_true();
                    } else {
                        return manager.mk_false();
                    }
                }

                term
            }
            TermKind::Ge(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model(*lhs, model, manager);
                let eval_rhs = self.evaluate_under_model(*rhs, model, manager);

                let lhs_t = manager.get(eval_lhs).cloned();
                let rhs_t = manager.get(eval_rhs).cloned();

                if let (Some(l), Some(r)) = (lhs_t, rhs_t)
                    && let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind)
                {
                    if a >= b {
                        return manager.mk_true();
                    } else {
                        return manager.mk_false();
                    }
                }

                term
            }
            // Add more cases as needed
            _ => {
                // For complex terms, try to simplify
                manager.simplify(term)
            }
        }
    }

    /// Try to instantiate from model values
    fn instantiate_from_model(
        &self,
        quantifier_idx: usize,
        model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> Option<Instantiation> {
        let quant = &self.quantifiers[quantifier_idx];
        let mut subst: FxHashMap<Spur, TermId> = FxHashMap::default();

        // Try to find model values for each bound variable's sort
        for &(name, sort) in &quant.bound_vars {
            // Look for any model value with matching sort
            let mut found = None;
            for (&term, &_value) in model {
                if let Some(t) = manager.get(term)
                    && t.sort == sort
                {
                    found = Some(term);
                    break;
                }
            }

            // If no model value found, use a default
            let candidate = match found {
                Some(t) => t,
                None => {
                    if sort == manager.sorts.int_sort {
                        manager.mk_int(0)
                    } else if sort == manager.sorts.bool_sort {
                        manager.mk_true()
                    } else {
                        // Return a dummy for unknown sorts
                        manager.mk_true()
                    }
                }
            };

            subst.insert(name, candidate);
        }

        // Apply substitution
        let ground_body = self.apply_substitution(quant.body, &subst, manager);

        Some(Instantiation {
            quantifier: quant.term,
            substitution: subst,
            result: ground_body,
        })
    }

    /// Get statistics about MBQI
    pub fn stats(&self) -> MBQIStats {
        MBQIStats {
            num_quantifiers: self.quantifiers.len(),
            total_instantiations: self.total_instantiation_count,
            max_instantiations: self.max_total_instantiations,
            unique_instantiations: self.generated_instantiations.len(),
        }
    }
}

/// Statistics about MBQI
#[derive(Debug, Clone)]
pub struct MBQIStats {
    /// Number of tracked quantifiers
    pub num_quantifiers: usize,
    /// Total instantiations generated
    pub total_instantiations: usize,
    /// Maximum allowed instantiations
    pub max_instantiations: usize,
    /// Unique instantiations (after deduplication)
    pub unique_instantiations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbqi_new() {
        let mbqi = MBQISolver::new();
        assert!(mbqi.is_enabled());
        assert_eq!(mbqi.quantifiers.len(), 0);
    }

    #[test]
    fn test_mbqi_disable() {
        let mut mbqi = MBQISolver::new();
        mbqi.set_enabled(false);
        assert!(!mbqi.is_enabled());

        let model = FxHashMap::default();
        let mut manager = TermManager::new();
        let result = mbqi.check_with_model(&model, &mut manager);
        assert!(matches!(result, MBQIResult::NoQuantifiers));
    }

    #[test]
    fn test_mbqi_no_quantifiers() {
        let mut mbqi = MBQISolver::new();
        let model = FxHashMap::default();
        let mut manager = TermManager::new();

        let result = mbqi.check_with_model(&model, &mut manager);
        assert!(matches!(result, MBQIResult::NoQuantifiers));
    }

    #[test]
    fn test_mbqi_add_candidate() {
        let mut mbqi = MBQISolver::new();
        let manager = TermManager::new();

        let sort = manager.sorts.int_sort;
        mbqi.add_candidate(TermId::new(1), sort);
        mbqi.add_candidate(TermId::new(2), sort);

        let candidates = mbqi.get_candidates(sort);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_mbqi_stats() {
        let mbqi = MBQISolver::new();
        let stats = mbqi.stats();

        assert_eq!(stats.num_quantifiers, 0);
        assert_eq!(stats.total_instantiations, 0);
    }

    #[test]
    fn test_mbqi_clear() {
        let mut mbqi = MBQISolver::new();
        let manager = TermManager::new();

        mbqi.add_candidate(TermId::new(1), manager.sorts.int_sort);
        mbqi.total_instantiation_count = 5;

        mbqi.clear();

        assert_eq!(mbqi.quantifiers.len(), 0);
        assert_eq!(mbqi.total_instantiation_count, 0);
    }

    #[test]
    fn test_mbqi_with_limit() {
        let mbqi = MBQISolver::with_limit(100);
        assert_eq!(mbqi.max_total_instantiations, 100);
    }

    #[test]
    fn test_enumerate_combinations_empty() {
        let mbqi = MBQISolver::new();
        let candidates: Vec<Vec<TermId>> = vec![];
        let combos = mbqi.enumerate_combinations(&candidates, 10);
        assert_eq!(combos.len(), 1);
        assert!(combos[0].is_empty());
    }

    #[test]
    fn test_enumerate_combinations_single() {
        let mbqi = MBQISolver::new();
        let candidates = vec![vec![TermId::new(1), TermId::new(2)]];
        let combos = mbqi.enumerate_combinations(&candidates, 10);
        assert_eq!(combos.len(), 2);
    }

    #[test]
    fn test_enumerate_combinations_multiple() {
        let mbqi = MBQISolver::new();
        let candidates = vec![
            vec![TermId::new(1), TermId::new(2)],
            vec![TermId::new(3), TermId::new(4)],
        ];
        let combos = mbqi.enumerate_combinations(&candidates, 10);
        // 2 * 2 = 4 combinations
        assert_eq!(combos.len(), 4);
    }
}
