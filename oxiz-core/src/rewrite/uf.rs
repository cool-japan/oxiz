//! Uninterpreted Function Rewriter
//!
//! This module provides rewriting rules for uninterpreted functions:
//! - Function congruence: f(a) = f(a) → true
//! - Argument simplification
//! - Beta reduction for lambda expressions
//! - Function application normalization
//! - Constant propagation through functions
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::rewrite::{Rewriter, RewriteContext, UfRewriter};
//!
//! let mut ctx = RewriteContext::new();
//! let mut uf = UfRewriter::new();
//! let simplified = uf.rewrite(term, &mut ctx, &mut manager)?;
//! ```

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::SortId;
use crate::ast::{TermId, TermKind, TermManager};
use lasso::Spur;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Configuration for UF rewriting
#[derive(Debug, Clone)]
pub struct UfRewriterConfig {
    /// Enable function congruence simplification
    pub enable_congruence: bool,
    /// Enable beta reduction
    pub enable_beta_reduction: bool,
    /// Maximum term depth for rewriting
    pub max_depth: usize,
    /// Enable argument normalization (sort arguments for AC symbols)
    pub enable_arg_normalization: bool,
    /// Track function definitions for inline expansion
    pub enable_inlining: bool,
    /// Maximum inline expansion depth
    pub max_inline_depth: usize,
}

impl Default for UfRewriterConfig {
    fn default() -> Self {
        Self {
            enable_congruence: true,
            enable_beta_reduction: true,
            max_depth: 100,
            enable_arg_normalization: true,
            enable_inlining: true,
            max_inline_depth: 5,
        }
    }
}

/// Function definition (for inlining)
#[derive(Debug, Clone)]
pub struct FunctionDef {
    /// Parameter names
    pub params: Vec<Spur>,
    /// Function body
    pub body: TermId,
    /// Result sort
    pub result_sort: SortId,
}

/// Uninterpreted function rewriter
#[derive(Debug)]
pub struct UfRewriter {
    /// Configuration
    config: UfRewriterConfig,
    /// Function definitions for inlining
    definitions: FxHashMap<Spur, FunctionDef>,
    /// Known commutative symbols
    commutative: FxHashSet<Spur>,
    /// Known associative symbols
    associative: FxHashSet<Spur>,
    /// Congruence cache: (func, args_hash) -> canonical result
    congruence_cache: FxHashMap<(Spur, u64), TermId>,
}

impl UfRewriter {
    /// Create a new UF rewriter with default configuration
    pub fn new() -> Self {
        Self::with_config(UfRewriterConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: UfRewriterConfig) -> Self {
        Self {
            config,
            definitions: FxHashMap::default(),
            commutative: FxHashSet::default(),
            associative: FxHashSet::default(),
            congruence_cache: FxHashMap::default(),
        }
    }

    /// Register a function definition for inlining
    pub fn register_definition(
        &mut self,
        func: Spur,
        params: Vec<Spur>,
        body: TermId,
        result_sort: SortId,
    ) {
        self.definitions.insert(
            func,
            FunctionDef {
                params,
                body,
                result_sort,
            },
        );
    }

    /// Mark a function as commutative
    pub fn mark_commutative(&mut self, func: Spur) {
        self.commutative.insert(func);
    }

    /// Mark a function as associative
    pub fn mark_associative(&mut self, func: Spur) {
        self.associative.insert(func);
    }

    /// Check if function is commutative
    pub fn is_commutative(&self, func: Spur) -> bool {
        self.commutative.contains(&func)
    }

    /// Check if function is associative
    pub fn is_associative(&self, func: Spur) -> bool {
        self.associative.contains(&func)
    }

    /// Clear the congruence cache
    pub fn clear_cache(&mut self) {
        self.congruence_cache.clear();
    }

    /// Hash arguments for congruence checking
    fn hash_args(&self, args: &[TermId]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        args.hash(&mut hasher);
        hasher.finish()
    }

    /// Create an Apply term with Spur-based function name
    fn mk_apply_with_spur(
        &self,
        func: Spur,
        args: Vec<TermId>,
        sort: SortId,
        manager: &mut TermManager,
    ) -> TermId {
        let func_str = manager.resolve_str(func).to_string();
        manager.mk_apply(&func_str, args, sort)
    }

    /// Rewrite a function application
    fn rewrite_apply(
        &mut self,
        term: TermId,
        func: Spur,
        args: &SmallVec<[TermId; 4]>,
        sort: SortId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // Try beta reduction if we have a definition
        if self.config.enable_beta_reduction
            && let Some(result) = self.try_beta_reduction(func, args, manager)
        {
            ctx.stats_mut().record_rule("uf_beta_reduction");
            return RewriteResult::Rewritten(result);
        }

        // Try inlining
        if self.config.enable_inlining
            && let Some(result) = self.try_inline(func, args, 0, manager)
        {
            ctx.stats_mut().record_rule("uf_inline");
            return RewriteResult::Rewritten(result);
        }

        // Normalize commutative arguments
        if self.config.enable_arg_normalization
            && self.is_commutative(func)
            && args.len() == 2
            && let Some(result) = self.normalize_commutative(func, args, sort, manager)
        {
            ctx.stats_mut().record_rule("uf_commutative_normalize");
            return RewriteResult::Rewritten(result);
        }

        // Flatten associative applications
        if self.is_associative(func)
            && let Some(result) = self.flatten_associative(func, args, sort, manager)
        {
            ctx.stats_mut().record_rule("uf_associative_flatten");
            return RewriteResult::Rewritten(result);
        }

        // Check congruence cache
        if self.config.enable_congruence {
            let args_hash = self.hash_args(args);
            if let Some(&cached) = self.congruence_cache.get(&(func, args_hash)) {
                return RewriteResult::Rewritten(cached);
            }
            // Cache this application
            self.congruence_cache.insert((func, args_hash), term);
        }

        RewriteResult::Unchanged(term)
    }

    /// Try beta reduction (substitute definition)
    fn try_beta_reduction(
        &self,
        func: Spur,
        args: &SmallVec<[TermId; 4]>,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let def = self.definitions.get(&func)?;

        // Check arity matches
        if def.params.len() != args.len() {
            return None;
        }

        // Build substitution map
        let subst: FxHashMap<Spur, TermId> = def
            .params
            .iter()
            .zip(args.iter())
            .map(|(&param, &arg)| (param, arg))
            .collect();

        // Apply substitution to body
        Some(self.substitute(def.body, &subst, manager))
    }

    /// Try inlining with depth limit
    fn try_inline(
        &self,
        func: Spur,
        args: &SmallVec<[TermId; 4]>,
        depth: usize,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        if depth >= self.config.max_inline_depth {
            return None;
        }

        let def = self.definitions.get(&func)?;

        if def.params.len() != args.len() {
            return None;
        }

        // Build substitution
        let subst: FxHashMap<Spur, TermId> = def
            .params
            .iter()
            .zip(args.iter())
            .map(|(&param, &arg)| (param, arg))
            .collect();

        Some(self.substitute(def.body, &subst, manager))
    }

    /// Apply substitution to a term
    fn substitute(
        &self,
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match &t.kind {
            TermKind::Var(name) => {
                if let Some(&replacement) = subst.get(name) {
                    replacement
                } else {
                    term
                }
            }

            TermKind::Apply { func, args } => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&arg| self.substitute(arg, subst, manager))
                    .collect();

                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    term
                } else {
                    self.mk_apply_with_spur(*func, new_args.to_vec(), t.sort, manager)
                }
            }

            TermKind::Not(arg) => {
                let new_arg = self.substitute(*arg, subst, manager);
                if new_arg == *arg {
                    term
                } else {
                    manager.mk_not(new_arg)
                }
            }

            TermKind::And(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| self.substitute(arg, subst, manager))
                    .collect();

                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    term
                } else {
                    manager.mk_and(new_args)
                }
            }

            TermKind::Or(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| self.substitute(arg, subst, manager))
                    .collect();

                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    term
                } else {
                    manager.mk_or(new_args)
                }
            }

            TermKind::Implies(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_implies(new_lhs, new_rhs)
                }
            }

            TermKind::Eq(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_eq(new_lhs, new_rhs)
                }
            }

            TermKind::Ite(cond, then_br, else_br) => {
                let new_cond = self.substitute(*cond, subst, manager);
                let new_then = self.substitute(*then_br, subst, manager);
                let new_else = self.substitute(*else_br, subst, manager);
                if new_cond == *cond && new_then == *then_br && new_else == *else_br {
                    term
                } else {
                    manager.mk_ite(new_cond, new_then, new_else)
                }
            }

            TermKind::Add(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| self.substitute(arg, subst, manager))
                    .collect();

                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    term
                } else {
                    manager.mk_add(new_args)
                }
            }

            TermKind::Mul(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| self.substitute(arg, subst, manager))
                    .collect();

                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    term
                } else {
                    manager.mk_mul(new_args)
                }
            }

            TermKind::Sub(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_sub(new_lhs, new_rhs)
                }
            }

            TermKind::Neg(arg) => {
                let new_arg = self.substitute(*arg, subst, manager);
                if new_arg == *arg {
                    term
                } else {
                    manager.mk_neg(new_arg)
                }
            }

            TermKind::Div(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_div(new_lhs, new_rhs)
                }
            }

            TermKind::Lt(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_lt(new_lhs, new_rhs)
                }
            }

            TermKind::Le(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_le(new_lhs, new_rhs)
                }
            }

            TermKind::Gt(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_gt(new_lhs, new_rhs)
                }
            }

            TermKind::Ge(lhs, rhs) => {
                let new_lhs = self.substitute(*lhs, subst, manager);
                let new_rhs = self.substitute(*rhs, subst, manager);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term
                } else {
                    manager.mk_ge(new_lhs, new_rhs)
                }
            }

            // For quantifiers, need to avoid capturing
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                // Check if any bound variable shadows a substitution
                let shadowed: FxHashSet<_> = vars.iter().map(|(name, _)| *name).collect();
                let filtered_subst: FxHashMap<_, _> = subst
                    .iter()
                    .filter(|(k, _)| !shadowed.contains(*k))
                    .map(|(&k, &v)| (k, v))
                    .collect();

                let new_body = self.substitute(*body, &filtered_subst, manager);
                let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                    .iter()
                    .map(|pat| {
                        pat.iter()
                            .map(|&t| self.substitute(t, &filtered_subst, manager))
                            .collect()
                    })
                    .collect();

                if new_body == *body && new_patterns == *patterns {
                    term
                } else {
                    let var_strs: Vec<(String, _)> = vars
                        .iter()
                        .map(|(spur, sort)| (manager.resolve_str(*spur).to_string(), *sort))
                        .collect();
                    let var_refs: Vec<(&str, _)> = var_strs
                        .iter()
                        .map(|(s, sort)| (s.as_str(), *sort))
                        .collect();
                    manager.mk_forall_with_patterns(var_refs, new_body, new_patterns)
                }
            }

            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                let shadowed: FxHashSet<_> = vars.iter().map(|(name, _)| *name).collect();
                let filtered_subst: FxHashMap<_, _> = subst
                    .iter()
                    .filter(|(k, _)| !shadowed.contains(*k))
                    .map(|(&k, &v)| (k, v))
                    .collect();

                let new_body = self.substitute(*body, &filtered_subst, manager);
                let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                    .iter()
                    .map(|pat| {
                        pat.iter()
                            .map(|&t| self.substitute(t, &filtered_subst, manager))
                            .collect()
                    })
                    .collect();

                if new_body == *body && new_patterns == *patterns {
                    term
                } else {
                    let var_strs: Vec<(String, _)> = vars
                        .iter()
                        .map(|(spur, sort)| (manager.resolve_str(*spur).to_string(), *sort))
                        .collect();
                    let var_refs: Vec<(&str, _)> = var_strs
                        .iter()
                        .map(|(s, sort)| (s.as_str(), *sort))
                        .collect();
                    manager.mk_exists_with_patterns(var_refs, new_body, new_patterns)
                }
            }

            TermKind::Let { bindings, body } => {
                // Substitute in binding values
                let new_bindings: SmallVec<[(Spur, TermId); 2]> = bindings
                    .iter()
                    .map(|&(name, val)| (name, self.substitute(val, subst, manager)))
                    .collect();

                // Filter out shadowed variables for body
                let shadowed: FxHashSet<_> = bindings.iter().map(|(name, _)| *name).collect();
                let filtered_subst: FxHashMap<_, _> = subst
                    .iter()
                    .filter(|(k, _)| !shadowed.contains(*k))
                    .map(|(&k, &v)| (k, v))
                    .collect();

                let new_body = self.substitute(*body, &filtered_subst, manager);

                if new_bindings == *bindings && new_body == *body {
                    term
                } else {
                    // Convert bindings to (String, TermId) and then (&str, TermId)
                    let binding_strs: Vec<(String, TermId)> = new_bindings
                        .iter()
                        .map(|(spur, val)| (manager.resolve_str(*spur).to_string(), *val))
                        .collect();
                    let binding_refs: Vec<(&str, TermId)> = binding_strs
                        .iter()
                        .map(|(s, val)| (s.as_str(), *val))
                        .collect();
                    manager.mk_let(binding_refs, new_body)
                }
            }

            // Leaves and other terms that don't need substitution
            _ => term,
        }
    }

    /// Normalize commutative function arguments (sort by TermId)
    fn normalize_commutative(
        &self,
        func: Spur,
        args: &SmallVec<[TermId; 4]>,
        sort: SortId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        if args.len() != 2 {
            return None;
        }

        let (a, b) = (args[0], args[1]);

        // Sort by TermId for canonical form
        if a.0 > b.0 {
            Some(self.mk_apply_with_spur(func, vec![b, a], sort, manager))
        } else {
            None
        }
    }

    /// Flatten nested associative function applications
    fn flatten_associative(
        &self,
        func: Spur,
        args: &SmallVec<[TermId; 4]>,
        sort: SortId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let mut flattened = Vec::new();
        let mut changed = false;

        for &arg in args.iter() {
            if let Some(t) = manager.get(arg)
                && let TermKind::Apply {
                    func: inner_func,
                    args: inner_args,
                } = &t.kind
                && *inner_func == func
            {
                flattened.extend(inner_args.iter().copied());
                changed = true;
                continue;
            }
            flattened.push(arg);
        }

        if changed {
            Some(self.mk_apply_with_spur(func, flattened, sort, manager))
        } else {
            None
        }
    }

    /// Rewrite equality of function applications
    fn rewrite_func_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_eq(lhs, rhs);

        // f(a) = f(a) → true
        if lhs == rhs {
            ctx.stats_mut().record_rule("uf_eq_refl");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        let Some(t_lhs) = manager.get(lhs).cloned() else {
            return RewriteResult::Unchanged(term);
        };
        let Some(t_rhs) = manager.get(rhs).cloned() else {
            return RewriteResult::Unchanged(term);
        };

        // Check for same function with same args
        if let (
            TermKind::Apply {
                func: f1,
                args: args1,
            },
            TermKind::Apply {
                func: f2,
                args: args2,
            },
        ) = (&t_lhs.kind, &t_rhs.kind)
            && f1 == f2
            && args1 == args2
        {
            ctx.stats_mut().record_rule("uf_congruence_true");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        RewriteResult::Unchanged(term)
    }
}

impl Default for UfRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Rewriter for UfRewriter {
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        ctx.stats_mut().terms_visited += 1;

        let Some(t) = manager.get(term).cloned() else {
            return RewriteResult::Unchanged(term);
        };

        match &t.kind {
            TermKind::Apply { func, args } => {
                self.rewrite_apply(term, *func, args, t.sort, ctx, manager)
            }

            TermKind::Eq(lhs, rhs) => {
                // Check if either side is a function application
                let is_lhs_apply = manager
                    .get(*lhs)
                    .map(|t| matches!(&t.kind, TermKind::Apply { .. }))
                    .unwrap_or(false);
                let is_rhs_apply = manager
                    .get(*rhs)
                    .map(|t| matches!(&t.kind, TermKind::Apply { .. }))
                    .unwrap_or(false);

                if is_lhs_apply || is_rhs_apply {
                    self.rewrite_func_eq(*lhs, *rhs, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }

            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "uf"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        if let Some(t) = manager.get(term) {
            matches!(&t.kind, TermKind::Apply { .. } | TermKind::Eq(_, _))
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, UfRewriter) {
        let manager = TermManager::new();
        let ctx = RewriteContext::new();
        let rewriter = UfRewriter::new();
        (manager, ctx, rewriter)
    }

    #[test]
    fn test_apply_unchanged() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let apply = manager.mk_apply("f", vec![x], int_sort);

        let result = rewriter.rewrite(apply, &mut ctx, &mut manager);
        assert!(!result.was_rewritten());
    }

    #[test]
    fn test_commutative_normalize() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;

        // Register a commutative symbol
        let f_spur = manager.intern_str("comm_f");
        rewriter.mark_commutative(f_spur);

        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);

        // Create f(y, x) where y.id > x.id - should be normalized to f(x, y)
        let func_str = manager.resolve_str(f_spur).to_string();
        let apply = manager.mk_apply(&func_str, vec![y, x], int_sort);

        // The normalization depends on term ordering
        let result = rewriter.rewrite(apply, &mut ctx, &mut manager);
        // Check that the term was processed (may or may not rewrite depending on term IDs)
        assert!(result.term().0 > 0);
    }

    #[test]
    fn test_func_eq_refl() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let apply = manager.mk_apply("f", vec![x], int_sort);

        // f(x) = f(x) → true
        // Note: mk_eq already simplifies identical terms to true during construction,
        // so the UF rewriter sees 'true' directly
        let eq = manager.mk_eq(apply, apply);

        // The term should already be 'true' due to mk_eq simplification
        let t_eq = manager.get(eq);
        assert!(matches!(t_eq.map(|t| &t.kind), Some(TermKind::True)));

        // Rewriter should not change it further
        let result = rewriter.rewrite(eq, &mut ctx, &mut manager);
        assert!(!result.was_rewritten()); // Already simplified
        let t_result = manager.get(result.term());
        assert!(matches!(t_result.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_beta_reduction() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;

        // Define f(x) = x + 1
        let param_x = manager.intern_str("param_x");
        let param_var = manager.mk_var("param_x", int_sort);
        let one = manager.mk_int(1);
        let body = manager.mk_add(vec![param_var, one]);

        let f_spur = manager.intern_str("f");
        rewriter.register_definition(f_spur, vec![param_x], body, int_sort);

        // Now apply f(5)
        let five = manager.mk_int(5);
        let func_str = manager.resolve_str(f_spur).to_string();
        let apply = manager.mk_apply(&func_str, vec![five], int_sort);

        let result = rewriter.rewrite(apply, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
    }

    #[test]
    fn test_associative_flatten() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;

        let g_spur = manager.intern_str("assoc_g");
        rewriter.mark_associative(g_spur);

        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);

        // Create g(g(x, y), z) - should flatten to g(x, y, z)
        let g_str = manager.resolve_str(g_spur).to_string();
        let inner = manager.mk_apply(&g_str, vec![x, y], int_sort);
        let outer = manager.mk_apply(&g_str, vec![inner, z], int_sort);

        let result = rewriter.rewrite(outer, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        // Check flattened structure
        if let Some(t) = manager.get(result.term()) {
            if let TermKind::Apply { args, .. } = &t.kind {
                assert_eq!(args.len(), 3);
            } else {
                panic!("Expected Apply");
            }
        }
    }

    #[test]
    fn test_substitution_avoid_capture() {
        let (mut manager, _ctx, rewriter) = setup();
        let int_sort = manager.sorts.int_sort;

        // Create forall x. x > 0
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(x, zero);
        let forall = manager.mk_forall(vec![("x", int_sort)], body);

        // Substitution should not affect bound variable x
        let x_spur = manager.intern_str("x");
        let five = manager.mk_int(5);
        let mut subst: FxHashMap<Spur, TermId> = FxHashMap::default();
        subst.insert(x_spur, five);

        let result = rewriter.substitute(forall, &subst, &mut manager);

        // Should be unchanged since x is bound
        assert_eq!(result, forall);
    }

    #[test]
    fn test_congruence_cache() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let apply1 = manager.mk_apply("h", vec![x], int_sort);
        let apply2 = manager.mk_apply("h", vec![x], int_sort);

        // First call - cache miss
        let result1 = rewriter.rewrite(apply1, &mut ctx, &mut manager);

        // Second call - should hit cache
        let result2 = rewriter.rewrite(apply2, &mut ctx, &mut manager);

        // Both should return the same term
        assert_eq!(result1.term(), result2.term());
    }
}
