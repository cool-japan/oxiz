//! Pattern matching for terms
//!
//! This module provides pattern matching capabilities for terms, enabling
//! powerful pattern-based transformations and analysis.

use super::{TermId, TermKind, TermManager};
use crate::sort::SortId;
use rustc_hash::FxHashMap;

/// A pattern for matching terms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pattern {
    /// Match any term and bind it to a variable
    Wildcard(String),
    /// Match a specific boolean constant
    BoolConst(bool),
    /// Match a specific integer constant
    IntConst(num_bigint::BigInt),
    /// Match any term of a specific sort
    SortedWildcard(String, SortId),
    /// Match a NOT term
    Not(Box<Pattern>),
    /// Match an AND term
    And(Vec<Pattern>),
    /// Match an OR term
    Or(Vec<Pattern>),
    /// Match an IMPLIES term
    Implies(Box<Pattern>, Box<Pattern>),
    /// Match an ITE term
    Ite(Box<Pattern>, Box<Pattern>, Box<Pattern>),
    /// Match an EQ term
    Eq(Box<Pattern>, Box<Pattern>),
    /// Match an ADD term
    Add(Vec<Pattern>),
    /// Match a MUL term
    Mul(Vec<Pattern>),
    /// Match a SUB term
    Sub(Box<Pattern>, Box<Pattern>),
    /// Match a LT term
    Lt(Box<Pattern>, Box<Pattern>),
    /// Match a LE term
    Le(Box<Pattern>, Box<Pattern>),
    /// Match a function application with specific function name
    Apply(String, Vec<Pattern>),
}

/// Result of pattern matching - maps pattern variables to matched terms
pub type Substitution = FxHashMap<String, TermId>;

/// Try to match a term against a pattern
pub fn match_pattern(
    pattern: &Pattern,
    term_id: TermId,
    manager: &TermManager,
) -> Option<Substitution> {
    let mut subst = Substitution::default();
    if match_pattern_rec(pattern, term_id, manager, &mut subst) {
        Some(subst)
    } else {
        None
    }
}

/// Recursive pattern matching helper
fn match_pattern_rec(
    pattern: &Pattern,
    term_id: TermId,
    manager: &TermManager,
    subst: &mut Substitution,
) -> bool {
    match pattern {
        Pattern::Wildcard(var) => {
            // Check if this variable is already bound
            if let Some(&bound_term) = subst.get(var) {
                // Must match the same term
                bound_term == term_id
            } else {
                // Bind the variable
                subst.insert(var.clone(), term_id);
                true
            }
        }

        Pattern::BoolConst(val) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            matches!(
                (&term.kind, val),
                (TermKind::True, true) | (TermKind::False, false)
            )
        }

        Pattern::IntConst(expected) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::IntConst(actual) = &term.kind {
                actual == expected
            } else {
                false
            }
        }

        Pattern::SortedWildcard(var, sort) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if term.sort == *sort {
                if let Some(&bound_term) = subst.get(var) {
                    bound_term == term_id
                } else {
                    subst.insert(var.clone(), term_id);
                    true
                }
            } else {
                false
            }
        }

        Pattern::Not(inner_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Not(inner_term) = term.kind {
                match_pattern_rec(inner_pat, inner_term, manager, subst)
            } else {
                false
            }
        }

        Pattern::And(patterns) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::And(args) = &term.kind {
                if patterns.len() != args.len() {
                    return false;
                }
                for (pat, &arg) in patterns.iter().zip(args.iter()) {
                    if !match_pattern_rec(pat, arg, manager, subst) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        Pattern::Or(patterns) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Or(args) = &term.kind {
                if patterns.len() != args.len() {
                    return false;
                }
                for (pat, &arg) in patterns.iter().zip(args.iter()) {
                    if !match_pattern_rec(pat, arg, manager, subst) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        Pattern::Implies(p_pat, q_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Implies(p, q) = term.kind {
                match_pattern_rec(p_pat, p, manager, subst)
                    && match_pattern_rec(q_pat, q, manager, subst)
            } else {
                false
            }
        }

        Pattern::Ite(c_pat, t_pat, e_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Ite(c, t, e) = term.kind {
                match_pattern_rec(c_pat, c, manager, subst)
                    && match_pattern_rec(t_pat, t, manager, subst)
                    && match_pattern_rec(e_pat, e, manager, subst)
            } else {
                false
            }
        }

        Pattern::Eq(lhs_pat, rhs_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Eq(lhs, rhs) = term.kind {
                match_pattern_rec(lhs_pat, lhs, manager, subst)
                    && match_pattern_rec(rhs_pat, rhs, manager, subst)
            } else {
                false
            }
        }

        Pattern::Add(patterns) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Add(args) = &term.kind {
                if patterns.len() != args.len() {
                    return false;
                }
                for (pat, &arg) in patterns.iter().zip(args.iter()) {
                    if !match_pattern_rec(pat, arg, manager, subst) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        Pattern::Mul(patterns) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Mul(args) = &term.kind {
                if patterns.len() != args.len() {
                    return false;
                }
                for (pat, &arg) in patterns.iter().zip(args.iter()) {
                    if !match_pattern_rec(pat, arg, manager, subst) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        Pattern::Sub(lhs_pat, rhs_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Sub(lhs, rhs) = term.kind {
                match_pattern_rec(lhs_pat, lhs, manager, subst)
                    && match_pattern_rec(rhs_pat, rhs, manager, subst)
            } else {
                false
            }
        }

        Pattern::Lt(lhs_pat, rhs_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Lt(lhs, rhs) = term.kind {
                match_pattern_rec(lhs_pat, lhs, manager, subst)
                    && match_pattern_rec(rhs_pat, rhs, manager, subst)
            } else {
                false
            }
        }

        Pattern::Le(lhs_pat, rhs_pat) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Le(lhs, rhs) = term.kind {
                match_pattern_rec(lhs_pat, lhs, manager, subst)
                    && match_pattern_rec(rhs_pat, rhs, manager, subst)
            } else {
                false
            }
        }

        Pattern::Apply(func_name, patterns) => {
            let Some(term) = manager.get(term_id) else {
                return false;
            };
            if let TermKind::Apply { func, args } = &term.kind {
                let actual_name = manager.resolve_str(*func);
                if actual_name != func_name {
                    return false;
                }
                if patterns.len() != args.len() {
                    return false;
                }
                for (pat, &arg) in patterns.iter().zip(args.iter()) {
                    if !match_pattern_rec(pat, arg, manager, subst) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
    }
}

/// Build a term from a pattern using a substitution
pub fn instantiate_pattern(
    pattern: &Pattern,
    subst: &Substitution,
    manager: &mut TermManager,
) -> Option<TermId> {
    match pattern {
        Pattern::Wildcard(var) => subst.get(var).copied(),

        Pattern::BoolConst(val) => Some(manager.mk_bool(*val)),

        Pattern::IntConst(n) => Some(manager.mk_int(n.clone())),

        Pattern::SortedWildcard(var, _sort) => subst.get(var).copied(),

        Pattern::Not(inner) => {
            let inner_term = instantiate_pattern(inner, subst, manager)?;
            Some(manager.mk_not(inner_term))
        }

        Pattern::And(patterns) => {
            let terms: Option<Vec<TermId>> = patterns
                .iter()
                .map(|p| instantiate_pattern(p, subst, manager))
                .collect();
            terms.map(|t| manager.mk_and(t))
        }

        Pattern::Or(patterns) => {
            let terms: Option<Vec<TermId>> = patterns
                .iter()
                .map(|p| instantiate_pattern(p, subst, manager))
                .collect();
            terms.map(|t| manager.mk_or(t))
        }

        Pattern::Implies(p, q) => {
            let p_term = instantiate_pattern(p, subst, manager)?;
            let q_term = instantiate_pattern(q, subst, manager)?;
            Some(manager.mk_implies(p_term, q_term))
        }

        Pattern::Ite(c, t, e) => {
            let c_term = instantiate_pattern(c, subst, manager)?;
            let t_term = instantiate_pattern(t, subst, manager)?;
            let e_term = instantiate_pattern(e, subst, manager)?;
            Some(manager.mk_ite(c_term, t_term, e_term))
        }

        Pattern::Eq(lhs, rhs) => {
            let lhs_term = instantiate_pattern(lhs, subst, manager)?;
            let rhs_term = instantiate_pattern(rhs, subst, manager)?;
            Some(manager.mk_eq(lhs_term, rhs_term))
        }

        Pattern::Add(patterns) => {
            let terms: Option<Vec<TermId>> = patterns
                .iter()
                .map(|p| instantiate_pattern(p, subst, manager))
                .collect();
            terms.map(|t| manager.mk_add(t))
        }

        Pattern::Mul(patterns) => {
            let terms: Option<Vec<TermId>> = patterns
                .iter()
                .map(|p| instantiate_pattern(p, subst, manager))
                .collect();
            terms.map(|t| manager.mk_mul(t))
        }

        Pattern::Sub(lhs, rhs) => {
            let lhs_term = instantiate_pattern(lhs, subst, manager)?;
            let rhs_term = instantiate_pattern(rhs, subst, manager)?;
            Some(manager.mk_sub(lhs_term, rhs_term))
        }

        Pattern::Lt(lhs, rhs) => {
            let lhs_term = instantiate_pattern(lhs, subst, manager)?;
            let rhs_term = instantiate_pattern(rhs, subst, manager)?;
            Some(manager.mk_lt(lhs_term, rhs_term))
        }

        Pattern::Le(lhs, rhs) => {
            let lhs_term = instantiate_pattern(lhs, subst, manager)?;
            let rhs_term = instantiate_pattern(rhs, subst, manager)?;
            Some(manager.mk_le(lhs_term, rhs_term))
        }

        Pattern::Apply(func_name, patterns) => {
            let terms: Option<Vec<TermId>> = patterns
                .iter()
                .map(|p| instantiate_pattern(p, subst, manager))
                .collect();
            terms.and_then(|args| {
                // We need to infer the result sort somehow
                // For now, just use the first argument's sort as a placeholder
                // This is a limitation - in a real implementation we'd need type information
                if let Some(&first_arg) = args.first() {
                    let first_term = manager.get(first_arg)?;
                    Some(manager.mk_apply(func_name, args, first_term.sort))
                } else {
                    None
                }
            })
        }
    }
}

/// A pattern-based rewrite rule
pub struct PatternRewriteRule {
    name: String,
    lhs: Pattern,
    rhs: Pattern,
}

impl PatternRewriteRule {
    /// Create a new pattern-based rewrite rule
    pub fn new(name: impl Into<String>, lhs: Pattern, rhs: Pattern) -> Self {
        Self {
            name: name.into(),
            lhs,
            rhs,
        }
    }
}

impl super::rewriting::RewriteRule for PatternRewriteRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        // Try to match the LHS pattern
        let subst = match_pattern(&self.lhs, term_id, manager)?;

        // Instantiate the RHS pattern with the substitution
        instantiate_pattern(&self.rhs, &subst, manager)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wildcard_pattern() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let pattern = Pattern::Wildcard("a".to_string());

        let subst = match_pattern(&pattern, x, &manager).unwrap();
        assert_eq!(subst.get("a"), Some(&x));
    }

    #[test]
    fn test_bool_const_pattern() {
        let manager = TermManager::new();

        let t = manager.mk_true();
        let f = manager.mk_false();

        let true_pattern = Pattern::BoolConst(true);
        let false_pattern = Pattern::BoolConst(false);

        assert!(match_pattern(&true_pattern, t, &manager).is_some());
        assert!(match_pattern(&false_pattern, f, &manager).is_some());
        assert!(match_pattern(&true_pattern, f, &manager).is_none());
    }

    #[test]
    fn test_int_const_pattern() {
        let mut manager = TermManager::new();

        let five = manager.mk_int(5);
        let pattern = Pattern::IntConst(num_bigint::BigInt::from(5));

        assert!(match_pattern(&pattern, five, &manager).is_some());

        let wrong_pattern = Pattern::IntConst(num_bigint::BigInt::from(10));
        assert!(match_pattern(&wrong_pattern, five, &manager).is_none());
    }

    #[test]
    fn test_not_pattern() {
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let not_p_kind = TermKind::Not(p);
        let not_p = manager.intern(not_p_kind, manager.sorts.bool_sort);

        let pattern = Pattern::Not(Box::new(Pattern::Wildcard("x".to_string())));

        let subst = match_pattern(&pattern, not_p, &manager).unwrap();
        assert_eq!(subst.get("x"), Some(&p));
    }

    #[test]
    fn test_and_pattern() {
        let mut manager = TermManager::new();

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let and_pq = manager.mk_and([p, q]);

        let pattern = Pattern::And(vec![
            Pattern::Wildcard("a".to_string()),
            Pattern::Wildcard("b".to_string()),
        ]);

        let subst = match_pattern(&pattern, and_pq, &manager).unwrap();
        assert_eq!(subst.get("a"), Some(&p));
        assert_eq!(subst.get("b"), Some(&q));
    }

    #[test]
    fn test_eq_pattern() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let eq = manager.mk_eq(x, five);

        let pattern = Pattern::Eq(
            Box::new(Pattern::Wildcard("var".to_string())),
            Box::new(Pattern::IntConst(num_bigint::BigInt::from(5))),
        );

        let subst = match_pattern(&pattern, eq, &manager).unwrap();
        assert_eq!(subst.get("var"), Some(&x));
    }

    #[test]
    fn test_pattern_instantiation() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let mut subst = Substitution::default();
        subst.insert("a".to_string(), x);

        let pattern = Pattern::Not(Box::new(Pattern::Wildcard("a".to_string())));

        let result = instantiate_pattern(&pattern, &subst, &mut manager).unwrap();
        if let Some(result_term) = manager.get(result) {
            assert!(matches!(result_term.kind, TermKind::Not(_)));
        }
    }

    #[test]
    fn test_pattern_rewrite_rule() {
        use super::super::rewriting::RewriteRule;

        let mut manager = TermManager::new();

        // Rule: (not (not ?x)) → ?x
        let lhs = Pattern::Not(Box::new(Pattern::Not(Box::new(Pattern::Wildcard(
            "x".to_string(),
        )))));
        let rhs = Pattern::Wildcard("x".to_string());
        let rule = PatternRewriteRule::new("double_neg", lhs, rhs);

        // Create (not (not p))
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let not_p_kind = TermKind::Not(p);
        let not_p = manager.intern(not_p_kind, manager.sorts.bool_sort);
        let not_not_p_kind = TermKind::Not(not_p);
        let not_not_p = manager.intern(not_not_p_kind, manager.sorts.bool_sort);

        let result = rule.apply(not_not_p, &mut manager);
        assert_eq!(result, Some(p));
    }

    #[test]
    fn test_multiple_wildcards() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y]);

        // Pattern: (+ ?a ?b)
        let pattern = Pattern::Add(vec![
            Pattern::Wildcard("a".to_string()),
            Pattern::Wildcard("b".to_string()),
        ]);

        let subst = match_pattern(&pattern, sum, &manager).unwrap();
        assert_eq!(subst.len(), 2);
        assert_eq!(subst.get("a"), Some(&x));
        assert_eq!(subst.get("b"), Some(&y));
    }

    #[test]
    fn test_same_wildcard_twice() {
        let mut manager = TermManager::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let sum = manager.mk_add([x, x]); // x + x

        // Pattern: (+ ?a ?a) - should match when both args are the same
        let pattern = Pattern::Add(vec![
            Pattern::Wildcard("a".to_string()),
            Pattern::Wildcard("a".to_string()),
        ]);

        let subst = match_pattern(&pattern, sum, &manager);
        assert!(subst.is_some());
        assert_eq!(subst.unwrap().get("a"), Some(&x));

        // Try with different arguments
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let sum_xy = manager.mk_add([x, y]);

        let subst2 = match_pattern(&pattern, sum_xy, &manager);
        assert!(subst2.is_none()); // Should not match because x != y
    }
}
