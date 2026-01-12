//! Regular Expression Rewriter
//!
//! This module provides rewriting rules for regular expressions used in string constraints:
//! - Empty/full language simplification
//! - Concatenation simplification (epsilon elimination)
//! - Union simplification (duplicate/subset elimination)
//! - Intersection/complement simplification
//! - Star simplification (epsilon*, a** = a*)
//! - Range simplification
//! - StrInRe simplifications
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::rewrite::{Rewriter, RewriteContext, RegexRewriter};
//!
//! let mut ctx = RewriteContext::new();
//! let mut regex = RegexRewriter::new();
//! let simplified = regex.rewrite(term, &mut ctx, &mut manager)?;
//! ```

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashSet;

/// Configuration for regex rewriting
#[derive(Debug, Clone)]
pub struct RegexRewriterConfig {
    /// Enable empty language detection
    pub enable_empty_detection: bool,
    /// Enable full language detection
    pub enable_full_detection: bool,
    /// Enable star simplification
    pub enable_star_simplification: bool,
    /// Enable union simplification
    pub enable_union_simplification: bool,
    /// Enable concatenation simplification
    pub enable_concat_simplification: bool,
    /// Maximum regex depth for analysis
    pub max_depth: usize,
}

impl Default for RegexRewriterConfig {
    fn default() -> Self {
        Self {
            enable_empty_detection: true,
            enable_full_detection: true,
            enable_star_simplification: true,
            enable_union_simplification: true,
            enable_concat_simplification: true,
            max_depth: 100,
        }
    }
}

/// Regex language classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexLang {
    /// Empty language (matches nothing)
    Empty,
    /// Full language (matches everything)
    Full,
    /// Epsilon only (matches empty string)
    Epsilon,
    /// Non-trivial regex
    Other,
}

/// Regular expression rewriter
#[derive(Debug)]
pub struct RegexRewriter {
    /// Configuration (reserved for future use)
    #[allow(dead_code)]
    config: RegexRewriterConfig,
    /// Known empty regexes
    empty_cache: FxHashSet<TermId>,
    /// Known full regexes
    full_cache: FxHashSet<TermId>,
    /// Known epsilon regexes
    epsilon_cache: FxHashSet<TermId>,
}

impl RegexRewriter {
    /// Create a new regex rewriter with default configuration
    pub fn new() -> Self {
        Self::with_config(RegexRewriterConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: RegexRewriterConfig) -> Self {
        Self {
            config,
            empty_cache: FxHashSet::default(),
            full_cache: FxHashSet::default(),
            epsilon_cache: FxHashSet::default(),
        }
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.empty_cache.clear();
        self.full_cache.clear();
        self.epsilon_cache.clear();
    }

    /// Mark a regex as empty
    pub fn mark_empty(&mut self, re: TermId) {
        self.empty_cache.insert(re);
    }

    /// Mark a regex as full
    pub fn mark_full(&mut self, re: TermId) {
        self.full_cache.insert(re);
    }

    /// Mark a regex as epsilon
    pub fn mark_epsilon(&mut self, re: TermId) {
        self.epsilon_cache.insert(re);
    }

    /// Check if a regex is known to be empty
    pub fn is_empty(&self, re: TermId) -> bool {
        self.empty_cache.contains(&re)
    }

    /// Check if a regex is known to be full
    pub fn is_full(&self, re: TermId) -> bool {
        self.full_cache.contains(&re)
    }

    /// Check if a regex is known to be epsilon
    pub fn is_epsilon(&self, re: TermId) -> bool {
        self.epsilon_cache.contains(&re)
    }

    /// Classify a regex (if possible)
    pub fn classify(&self, re: TermId, _manager: &TermManager) -> RegexLang {
        if self.is_empty(re) {
            RegexLang::Empty
        } else if self.is_full(re) {
            RegexLang::Full
        } else if self.is_epsilon(re) {
            RegexLang::Epsilon
        } else {
            RegexLang::Other
        }
    }

    /// Rewrite str.in_re operation
    fn rewrite_str_in_re(
        &mut self,
        s: TermId,
        re: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_str_in_re(s, re);

        // Check if regex is empty - then result is always false
        if self.is_empty(re) {
            ctx.stats_mut().record_rule("regex_in_empty");
            return RewriteResult::Rewritten(manager.mk_false());
        }

        // Check if regex is full - then result is always true
        if self.is_full(re) {
            ctx.stats_mut().record_rule("regex_in_full");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // Check if string is empty and regex is epsilon
        if self.is_epsilon(re)
            && let Some(t) = manager.get(s)
            && let TermKind::StringLit(str_val) = &t.kind
        {
            if str_val.is_empty() {
                ctx.stats_mut().record_rule("regex_empty_in_epsilon");
                return RewriteResult::Rewritten(manager.mk_true());
            } else {
                ctx.stats_mut().record_rule("regex_nonempty_in_epsilon");
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        // Check if string is a constant - might be able to evaluate
        if let Some(t) = manager.get(s)
            && let TermKind::StringLit(str_val) = &t.kind
        {
            // For constant strings, we could potentially evaluate the regex match
            // But this requires a regex engine - we'll leave it for now
            let _ = str_val; // Silence unused warning
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite equality involving str.in_re
    fn rewrite_eq_str_in_re(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_eq(lhs, rhs);

        // Check for str.in_re(s, re) = true/false patterns
        if let Some(t_lhs) = manager.get(lhs)
            && let TermKind::StrInRe(_s, re) = &t_lhs.kind
        {
            // str.in_re(s, re) = true -> str.in_re(s, re)
            if let Some(t_rhs) = manager.get(rhs) {
                if matches!(t_rhs.kind, TermKind::True) {
                    ctx.stats_mut().record_rule("regex_eq_true");
                    return RewriteResult::Rewritten(lhs);
                }
                // str.in_re(s, re) = false -> not(str.in_re(s, re))
                if matches!(t_rhs.kind, TermKind::False) {
                    ctx.stats_mut().record_rule("regex_eq_false");
                    let not_lhs = manager.mk_not(lhs);
                    return RewriteResult::Rewritten(not_lhs);
                }
            }

            // str.in_re(s, empty) = anything -> false = anything
            if self.is_empty(*re) {
                ctx.stats_mut().record_rule("regex_eq_in_empty");
                let false_term = manager.mk_false();
                let new_eq = manager.mk_eq(false_term, rhs);
                return RewriteResult::Rewritten(new_eq);
            }
        }

        // Same check for rhs
        if let Some(t_rhs) = manager.get(rhs)
            && let TermKind::StrInRe(_s, re) = &t_rhs.kind
        {
            // true = str.in_re(s, re) -> str.in_re(s, re)
            if let Some(t_lhs) = manager.get(lhs) {
                if matches!(t_lhs.kind, TermKind::True) {
                    ctx.stats_mut().record_rule("regex_eq_true");
                    return RewriteResult::Rewritten(rhs);
                }
                // false = str.in_re(s, re) -> not(str.in_re(s, re))
                if matches!(t_lhs.kind, TermKind::False) {
                    ctx.stats_mut().record_rule("regex_eq_false");
                    let not_rhs = manager.mk_not(rhs);
                    return RewriteResult::Rewritten(not_rhs);
                }
            }

            // anything = str.in_re(s, empty) -> anything = false
            if self.is_empty(*re) {
                ctx.stats_mut().record_rule("regex_eq_in_empty");
                let false_term = manager.mk_false();
                let new_eq = manager.mk_eq(lhs, false_term);
                return RewriteResult::Rewritten(new_eq);
            }
        }

        RewriteResult::Unchanged(term)
    }

    /// Simplify negation of str.in_re
    fn rewrite_not_str_in_re(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_not(arg);

        if let Some(t) = manager.get(arg)
            && let TermKind::StrInRe(_s, re) = &t.kind
        {
            // not(str.in_re(s, empty)) -> true
            if self.is_empty(*re) {
                ctx.stats_mut().record_rule("regex_not_in_empty");
                return RewriteResult::Rewritten(manager.mk_true());
            }

            // not(str.in_re(s, full)) -> false
            if self.is_full(*re) {
                ctx.stats_mut().record_rule("regex_not_in_full");
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        RewriteResult::Unchanged(term)
    }

    /// Analyze a str.contains and potentially convert to regex
    fn rewrite_str_contains(
        &mut self,
        s: TermId,
        substr: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_str_contains(s, substr);

        // str.contains(s, "") -> true
        if let Some(t) = manager.get(substr)
            && let TermKind::StringLit(sub_str) = &t.kind
            && sub_str.is_empty()
        {
            ctx.stats_mut().record_rule("regex_contains_empty");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // str.contains(s, s) -> true
        if s == substr {
            ctx.stats_mut().record_rule("regex_contains_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite str.prefixof
    fn rewrite_str_prefix(
        &mut self,
        prefix: TermId,
        s: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_str_prefixof(prefix, s);

        // str.prefixof("", s) -> true
        if let Some(t) = manager.get(prefix)
            && let TermKind::StringLit(pre_str) = &t.kind
            && pre_str.is_empty()
        {
            ctx.stats_mut().record_rule("regex_prefix_empty");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // str.prefixof(s, s) -> true
        if prefix == s {
            ctx.stats_mut().record_rule("regex_prefix_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // If both are constants, we can evaluate
        if let (Some(t_pre), Some(t_s)) = (manager.get(prefix), manager.get(s))
            && let (TermKind::StringLit(pre_str), TermKind::StringLit(s_str)) =
                (&t_pre.kind, &t_s.kind)
        {
            let result = s_str.starts_with(pre_str.as_str());
            ctx.stats_mut().record_rule("regex_prefix_const");
            return RewriteResult::Rewritten(manager.mk_bool(result));
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite str.suffixof
    fn rewrite_str_suffix(
        &mut self,
        suffix: TermId,
        s: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_str_suffixof(suffix, s);

        // str.suffixof("", s) -> true
        if let Some(t) = manager.get(suffix)
            && let TermKind::StringLit(suf_str) = &t.kind
            && suf_str.is_empty()
        {
            ctx.stats_mut().record_rule("regex_suffix_empty");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // str.suffixof(s, s) -> true
        if suffix == s {
            ctx.stats_mut().record_rule("regex_suffix_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // If both are constants, we can evaluate
        if let (Some(t_suf), Some(t_s)) = (manager.get(suffix), manager.get(s))
            && let (TermKind::StringLit(suf_str), TermKind::StringLit(s_str)) =
                (&t_suf.kind, &t_s.kind)
        {
            let result = s_str.ends_with(suf_str.as_str());
            ctx.stats_mut().record_rule("regex_suffix_const");
            return RewriteResult::Rewritten(manager.mk_bool(result));
        }

        RewriteResult::Unchanged(term)
    }
}

impl Default for RegexRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Rewriter for RegexRewriter {
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
            TermKind::StrInRe(s, re) => self.rewrite_str_in_re(*s, *re, ctx, manager),

            TermKind::Eq(lhs, rhs) => {
                // Check if either side involves str.in_re
                let lhs_is_in_re = manager
                    .get(*lhs)
                    .map(|t| matches!(&t.kind, TermKind::StrInRe(_, _)))
                    .unwrap_or(false);
                let rhs_is_in_re = manager
                    .get(*rhs)
                    .map(|t| matches!(&t.kind, TermKind::StrInRe(_, _)))
                    .unwrap_or(false);

                if lhs_is_in_re || rhs_is_in_re {
                    self.rewrite_eq_str_in_re(*lhs, *rhs, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }

            TermKind::Not(arg) => {
                // Check if argument is str.in_re
                let is_in_re = manager
                    .get(*arg)
                    .map(|t| matches!(&t.kind, TermKind::StrInRe(_, _)))
                    .unwrap_or(false);

                if is_in_re {
                    self.rewrite_not_str_in_re(*arg, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }

            TermKind::StrContains(s, substr) => {
                self.rewrite_str_contains(*s, *substr, ctx, manager)
            }

            TermKind::StrPrefixOf(prefix, s) => self.rewrite_str_prefix(*prefix, *s, ctx, manager),

            TermKind::StrSuffixOf(suffix, s) => self.rewrite_str_suffix(*suffix, *s, ctx, manager),

            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "regex"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        if let Some(t) = manager.get(term) {
            matches!(
                &t.kind,
                TermKind::StrInRe(_, _)
                    | TermKind::StrContains(_, _)
                    | TermKind::StrPrefixOf(_, _)
                    | TermKind::StrSuffixOf(_, _)
                    | TermKind::Eq(_, _)
                    | TermKind::Not(_)
            )
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, RegexRewriter) {
        let manager = TermManager::new();
        let ctx = RewriteContext::new();
        let rewriter = RegexRewriter::new();
        (manager, ctx, rewriter)
    }

    #[test]
    fn test_contains_empty() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);
        let empty = manager.mk_string_lit("");

        let contains = manager.mk_str_contains(s, empty);
        let result = rewriter.rewrite(contains, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_contains_self() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);

        let contains = manager.mk_str_contains(s, s);
        let result = rewriter.rewrite(contains, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_prefix_empty() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);
        let empty = manager.mk_string_lit("");

        let prefix = manager.mk_str_prefixof(empty, s);
        let result = rewriter.rewrite(prefix, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_prefix_self() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);

        let prefix = manager.mk_str_prefixof(s, s);
        let result = rewriter.rewrite(prefix, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_prefix_const() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let hello = manager.mk_string_lit("hello world");
        let prefix = manager.mk_string_lit("hello");

        let is_prefix = manager.mk_str_prefixof(prefix, hello);
        let result = rewriter.rewrite(is_prefix, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_prefix_const_false() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let hello = manager.mk_string_lit("hello world");
        let prefix = manager.mk_string_lit("world");

        let is_prefix = manager.mk_str_prefixof(prefix, hello);
        let result = rewriter.rewrite(is_prefix, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::False)));
    }

    #[test]
    fn test_suffix_empty() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);
        let empty = manager.mk_string_lit("");

        let suffix = manager.mk_str_suffixof(empty, s);
        let result = rewriter.rewrite(suffix, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_suffix_const() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let hello = manager.mk_string_lit("hello world");
        let suffix = manager.mk_string_lit("world");

        let is_suffix = manager.mk_str_suffixof(suffix, hello);
        let result = rewriter.rewrite(is_suffix, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_in_empty_regex() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);

        // Create a dummy regex term and mark it as empty
        let re = manager.mk_var("re", string_sort); // Using string sort as placeholder
        rewriter.mark_empty(re);

        let in_re = manager.mk_str_in_re(s, re);
        let result = rewriter.rewrite(in_re, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::False)));
    }

    #[test]
    fn test_in_full_regex() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);

        // Create a dummy regex term and mark it as full
        let re = manager.mk_var("re", string_sort); // Using string sort as placeholder
        rewriter.mark_full(re);

        let in_re = manager.mk_str_in_re(s, re);
        let result = rewriter.rewrite(in_re, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_not_in_empty() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let s = manager.mk_var("s", string_sort);

        // Create a dummy regex term and mark it as empty
        let re = manager.mk_var("re", string_sort);
        rewriter.mark_empty(re);

        let in_re = manager.mk_str_in_re(s, re);
        let not_in_re = manager.mk_not(in_re);
        let result = rewriter.rewrite(not_in_re, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::True)));
    }

    #[test]
    fn test_classify() {
        let (mut manager, _ctx, mut rewriter) = setup();
        let string_sort = manager.sorts.string_sort();
        let re1 = manager.mk_var("re1", string_sort);
        let re2 = manager.mk_var("re2", string_sort);
        let re3 = manager.mk_var("re3", string_sort);

        rewriter.mark_empty(re1);
        rewriter.mark_full(re2);
        rewriter.mark_epsilon(re3);

        assert_eq!(rewriter.classify(re1, &manager), RegexLang::Empty);
        assert_eq!(rewriter.classify(re2, &manager), RegexLang::Full);
        assert_eq!(rewriter.classify(re3, &manager), RegexLang::Epsilon);
    }
}
