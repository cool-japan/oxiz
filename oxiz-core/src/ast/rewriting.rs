//! Term rewriting system
//!
//! This module implements a term rewriting system for transforming terms
//! according to user-defined or built-in rewrite rules.

use super::{TermId, TermKind, TermManager};

/// A rewrite rule that transforms a term
pub trait RewriteRule: Send + Sync {
    /// Try to apply the rule to a term
    /// Returns Some(new_term) if the rule applies, None otherwise
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId>;

    /// Get a name for this rule (for debugging)
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// A collection of rewrite rules
#[derive(Default)]
pub struct RewriteRuleSet {
    rules: Vec<Box<dyn RewriteRule>>,
}

impl RewriteRuleSet {
    /// Create a new empty rule set
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to the set
    pub fn add_rule(&mut self, rule: Box<dyn RewriteRule>) {
        self.rules.push(rule);
    }

    /// Apply all rules to a term until fixpoint
    pub fn apply_fixpoint(&self, term_id: TermId, manager: &mut TermManager) -> TermId {
        let mut current = term_id;
        let mut changed = true;
        let max_iterations = 100; // Prevent infinite loops
        let mut iterations = 0;

        while changed && iterations < max_iterations {
            changed = false;
            iterations += 1;

            for rule in &self.rules {
                if let Some(new_term) = rule.apply(current, manager)
                    && new_term != current
                {
                    current = new_term;
                    changed = true;
                }
            }
        }

        current
    }

    /// Apply rules once (bottom-up) to all subterms
    pub fn apply_bottom_up(&self, term_id: TermId, manager: &mut TermManager) -> TermId {
        use super::traversal::collect_subterms;
        use rustc_hash::FxHashMap;

        // Collect all subterms in post-order
        let subterms = collect_subterms(term_id, manager);
        let mut cache: FxHashMap<TermId, TermId> = FxHashMap::default();

        // Apply rules to each subterm
        for &subterm_id in &subterms {
            for rule in &self.rules {
                if let Some(new_term) = rule.apply(subterm_id, manager)
                    && new_term != subterm_id
                {
                    cache.insert(subterm_id, new_term);
                    break;
                }
            }
        }

        // Return the transformed term
        cache.get(&term_id).copied().unwrap_or(term_id)
    }
}

/// Double negation elimination: ¬¬P → P
#[derive(Debug, Default)]
pub struct DoubleNegationRule;

impl RewriteRule for DoubleNegationRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;
        if let TermKind::Not(inner) = &term.kind
            && let Some(inner_term) = manager.get(*inner)
            && let TermKind::Not(inner_inner) = inner_term.kind
        {
            return Some(inner_inner);
        }
        None
    }

    fn name(&self) -> &str {
        "double_negation"
    }
}

/// De Morgan's laws
#[derive(Debug, Default)]
pub struct DeMorganRule;

impl RewriteRule for DeMorganRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        // ¬(P ∧ Q) → ¬P ∨ ¬Q
        // ¬(P ∨ Q) → ¬P ∧ ¬Q
        if let TermKind::Not(inner) = &term.kind {
            let inner_id = *inner;
            if let Some(inner_term) = manager.get(inner_id) {
                match &inner_term.kind {
                    TermKind::And(args) => {
                        let args_copy = args.clone();
                        let negated_args: Vec<TermId> =
                            args_copy.iter().map(|&arg| manager.mk_not(arg)).collect();
                        return Some(manager.mk_or(negated_args));
                    }
                    TermKind::Or(args) => {
                        let args_copy = args.clone();
                        let negated_args: Vec<TermId> =
                            args_copy.iter().map(|&arg| manager.mk_not(arg)).collect();
                        return Some(manager.mk_and(negated_args));
                    }
                    _ => {}
                }
            }
        }

        None
    }

    fn name(&self) -> &str {
        "de_morgan"
    }
}

/// Implication elimination: P → Q becomes ¬P ∨ Q
#[derive(Debug, Default)]
pub struct ImplicationEliminationRule;

impl RewriteRule for ImplicationEliminationRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;
        if let TermKind::Implies(p, q) = term.kind {
            let not_p = manager.mk_not(p);
            return Some(manager.mk_or([not_p, q]));
        }
        None
    }

    fn name(&self) -> &str {
        "implication_elimination"
    }
}

/// Arithmetic identity rules: x + 0 → x, x * 1 → x, etc.
#[derive(Debug, Default)]
pub struct ArithmeticIdentityRule;

impl RewriteRule for ArithmeticIdentityRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        use num_bigint::BigInt;

        let term = manager.get(term_id)?;

        match &term.kind {
            // x + 0 → x
            TermKind::Add(args) => {
                let zero = BigInt::from(0);
                let non_zero: Vec<TermId> = args
                    .iter()
                    .filter(|&&arg| {
                        if let Some(arg_term) = manager.get(arg)
                            && let TermKind::IntConst(n) = &arg_term.kind
                        {
                            return *n != zero;
                        }
                        true
                    })
                    .copied()
                    .collect();

                if non_zero.len() != args.len() {
                    return match non_zero.len() {
                        0 => Some(manager.mk_int(0)),
                        1 => Some(non_zero[0]),
                        _ => Some(manager.mk_add(non_zero)),
                    };
                }
            }

            // x * 0 → 0
            TermKind::Mul(args) => {
                let zero = BigInt::from(0);
                let one = BigInt::from(1);

                // Check if any argument is 0
                for &arg in args.iter() {
                    if let Some(arg_term) = manager.get(arg)
                        && let TermKind::IntConst(n) = &arg_term.kind
                        && *n == zero
                    {
                        return Some(manager.mk_int(0));
                    }
                }

                // Filter out 1s
                let non_one: Vec<TermId> = args
                    .iter()
                    .filter(|&&arg| {
                        if let Some(arg_term) = manager.get(arg)
                            && let TermKind::IntConst(n) = &arg_term.kind
                        {
                            return *n != one;
                        }
                        true
                    })
                    .copied()
                    .collect();

                if non_one.len() != args.len() {
                    return match non_one.len() {
                        0 => Some(manager.mk_int(1)),
                        1 => Some(non_one[0]),
                        _ => Some(manager.mk_mul(non_one)),
                    };
                }
            }

            _ => {}
        }

        None
    }

    fn name(&self) -> &str {
        "arithmetic_identity"
    }
}

/// Constant folding rule for arithmetic operations
#[derive(Debug, Default)]
pub struct ArithmeticConstantFoldingRule;

impl RewriteRule for ArithmeticConstantFoldingRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        use num_bigint::BigInt;

        let term = manager.get(term_id)?;

        match &term.kind {
            // Fold addition of constants
            TermKind::Add(args) => {
                let mut sum = BigInt::from(0);
                let mut non_const_args = Vec::new();

                for &arg in args.iter() {
                    if let Some(arg_term) = manager.get(arg) {
                        if let TermKind::IntConst(n) = &arg_term.kind {
                            sum += n;
                        } else {
                            non_const_args.push(arg);
                        }
                    } else {
                        non_const_args.push(arg);
                    }
                }

                // If we folded any constants, create a new term
                if non_const_args.len() < args.len() {
                    if sum != BigInt::from(0) || non_const_args.is_empty() {
                        non_const_args.push(manager.mk_int(sum));
                    }

                    return match non_const_args.len() {
                        0 => unreachable!(),
                        1 => Some(non_const_args[0]),
                        _ => Some(manager.mk_add(non_const_args)),
                    };
                }
            }

            // Fold multiplication of constants
            TermKind::Mul(args) => {
                let mut product = BigInt::from(1);
                let mut non_const_args = Vec::new();

                for &arg in args.iter() {
                    if let Some(arg_term) = manager.get(arg) {
                        if let TermKind::IntConst(n) = &arg_term.kind {
                            product *= n;
                        } else {
                            non_const_args.push(arg);
                        }
                    } else {
                        non_const_args.push(arg);
                    }
                }

                if non_const_args.len() < args.len() {
                    if product != BigInt::from(1) || non_const_args.is_empty() {
                        non_const_args.push(manager.mk_int(product));
                    }

                    return match non_const_args.len() {
                        0 => unreachable!(),
                        1 => Some(non_const_args[0]),
                        _ => Some(manager.mk_mul(non_const_args)),
                    };
                }
            }

            // Fold subtraction of constants
            TermKind::Sub(a, b) => {
                if let (Some(a_term), Some(b_term)) = (manager.get(*a), manager.get(*b))
                    && let (TermKind::IntConst(a_val), TermKind::IntConst(b_val)) =
                        (&a_term.kind, &b_term.kind)
                {
                    return Some(manager.mk_int(a_val - b_val));
                }
            }

            _ => {}
        }

        None
    }

    fn name(&self) -> &str {
        "arithmetic_constant_folding"
    }
}

/// Boolean absorption rules: P ∧ (P ∨ Q) → P
#[derive(Debug, Default)]
pub struct BooleanAbsorptionRule;

impl RewriteRule for BooleanAbsorptionRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        match &term.kind {
            // P ∧ (P ∨ Q) → P
            TermKind::And(args) => {
                for &arg in args.iter() {
                    if let Some(arg_term) = manager.get(arg)
                        && let TermKind::Or(or_args) = &arg_term.kind
                    {
                        // Check if any Or arg is in And args
                        for &or_arg in or_args.iter() {
                            if args.contains(&or_arg) {
                                // Found absorption pattern
                                return Some(or_arg);
                            }
                        }
                    }
                }
            }

            // P ∨ (P ∧ Q) → P
            TermKind::Or(args) => {
                for &arg in args.iter() {
                    if let Some(arg_term) = manager.get(arg)
                        && let TermKind::And(and_args) = &arg_term.kind
                    {
                        for &and_arg in and_args.iter() {
                            if args.contains(&and_arg) {
                                return Some(and_arg);
                            }
                        }
                    }
                }
            }

            _ => {}
        }

        None
    }

    fn name(&self) -> &str {
        "boolean_absorption"
    }
}

/// ITE (if-then-else) simplification rules
#[derive(Debug, Default)]
pub struct IteSimplificationRule;

impl RewriteRule for IteSimplificationRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        if let TermKind::Ite(cond, then_branch, else_branch) = &term.kind {
            // ITE(true, t, e) → t
            if *cond == manager.mk_true() {
                return Some(*then_branch);
            }

            // ITE(false, t, e) → e
            if *cond == manager.mk_false() {
                return Some(*else_branch);
            }

            // ITE(c, t, t) → t
            if then_branch == else_branch {
                return Some(*then_branch);
            }

            // ITE(c, true, false) → c
            if *then_branch == manager.mk_true() && *else_branch == manager.mk_false() {
                return Some(*cond);
            }

            // ITE(c, false, true) → ¬c
            if *then_branch == manager.mk_false() && *else_branch == manager.mk_true() {
                return Some(manager.mk_not(*cond));
            }
        }

        None
    }

    fn name(&self) -> &str {
        "ite_simplification"
    }
}

/// Equality and comparison simplification rules
#[derive(Debug, Default)]
pub struct EqualitySimplificationRule;

impl RewriteRule for EqualitySimplificationRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        match &term.kind {
            // x = x → true
            TermKind::Eq(lhs, rhs) if lhs == rhs => Some(manager.mk_true()),

            // x ≠ x → false (using negation of equality)
            TermKind::Not(inner) => {
                if let Some(inner_term) = manager.get(*inner)
                    && let TermKind::Eq(lhs, rhs) = inner_term.kind
                    && lhs == rhs
                {
                    return Some(manager.mk_false());
                }
                None
            }

            // x < x → false
            TermKind::Lt(lhs, rhs) if lhs == rhs => Some(manager.mk_false()),

            // x ≤ x → true
            TermKind::Le(lhs, rhs) if lhs == rhs => Some(manager.mk_true()),

            // x > x → false
            TermKind::Gt(lhs, rhs) if lhs == rhs => Some(manager.mk_false()),

            // x ≥ x → true
            TermKind::Ge(lhs, rhs) if lhs == rhs => Some(manager.mk_true()),

            _ => None,
        }
    }

    fn name(&self) -> &str {
        "equality_simplification"
    }
}

/// Comparison normalization rule - converts > and ≥ to < and ≤
#[derive(Debug, Default)]
pub struct ComparisonNormalizationRule;

impl RewriteRule for ComparisonNormalizationRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        match &term.kind {
            // x > y → y < x
            TermKind::Gt(lhs, rhs) => Some(manager.mk_lt(*rhs, *lhs)),

            // x ≥ y → y ≤ x
            TermKind::Ge(lhs, rhs) => Some(manager.mk_le(*rhs, *lhs)),

            _ => None,
        }
    }

    fn name(&self) -> &str {
        "comparison_normalization"
    }
}

/// Arithmetic distributivity rule: a * (b + c) → a*b + a*c
#[derive(Debug, Default)]
pub struct DistributivityRule;

impl RewriteRule for DistributivityRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        if let TermKind::Mul(args) = &term.kind {
            // Only apply for binary multiplication
            if args.len() == 2 {
                let lhs = args[0];
                let rhs = args[1];

                // Check if rhs is an addition
                if let Some(rhs_term) = manager.get(rhs)
                    && let TermKind::Add(add_args) = &rhs_term.kind
                {
                    // a * (b + c + ...) → a*b + a*c + ...
                    let add_args_clone = add_args.clone();
                    let distributed: Vec<_> = add_args_clone
                        .iter()
                        .map(|&arg| manager.mk_mul([lhs, arg]))
                        .collect();
                    return Some(manager.mk_add(distributed));
                }

                // Check if lhs is an addition
                if let Some(lhs_term) = manager.get(lhs)
                    && let TermKind::Add(add_args) = &lhs_term.kind
                {
                    // (a + b + ...) * c → a*c + b*c + ...
                    let add_args_clone = add_args.clone();
                    let distributed: Vec<_> = add_args_clone
                        .iter()
                        .map(|&arg| manager.mk_mul([arg, rhs]))
                        .collect();
                    return Some(manager.mk_add(distributed));
                }
            }
        }

        None
    }

    fn name(&self) -> &str {
        "distributivity"
    }
}

/// Boolean constant propagation rule
#[derive(Debug, Default)]
pub struct BooleanPropagationRule;

impl RewriteRule for BooleanPropagationRule {
    fn apply(&self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        let term = manager.get(term_id)?;

        match &term.kind {
            // true ∧ x → x,  x ∧ true → x
            // false ∧ x → false,  x ∧ false → false
            TermKind::And(args) => {
                let true_id = manager.mk_true();
                let false_id = manager.mk_false();

                // Check for false
                if args.contains(&false_id) {
                    return Some(false_id);
                }

                // Filter out true
                let filtered: Vec<_> = args.iter().copied().filter(|&arg| arg != true_id).collect();

                match filtered.len() {
                    0 => Some(true_id),
                    1 => Some(filtered[0]),
                    _ if filtered.len() < args.len() => Some(manager.mk_and(filtered)),
                    _ => None,
                }
            }

            // true ∨ x → true,  x ∨ true → true
            // false ∨ x → x,  x ∨ false → x
            TermKind::Or(args) => {
                let true_id = manager.mk_true();
                let false_id = manager.mk_false();

                // Check for true
                if args.contains(&true_id) {
                    return Some(true_id);
                }

                // Filter out false
                let filtered: Vec<_> = args
                    .iter()
                    .copied()
                    .filter(|&arg| arg != false_id)
                    .collect();

                match filtered.len() {
                    0 => Some(false_id),
                    1 => Some(filtered[0]),
                    _ if filtered.len() < args.len() => Some(manager.mk_or(filtered)),
                    _ => None,
                }
            }

            _ => None,
        }
    }

    fn name(&self) -> &str {
        "boolean_propagation"
    }
}

/// Create a standard rule set with common rewrite rules
#[must_use]
pub fn standard_rules() -> RewriteRuleSet {
    let mut ruleset = RewriteRuleSet::new();
    ruleset.add_rule(Box::new(DoubleNegationRule));
    ruleset.add_rule(Box::new(DeMorganRule));
    ruleset.add_rule(Box::new(ImplicationEliminationRule));
    ruleset.add_rule(Box::new(ArithmeticIdentityRule));
    ruleset.add_rule(Box::new(ArithmeticConstantFoldingRule));
    ruleset.add_rule(Box::new(BooleanAbsorptionRule));
    ruleset.add_rule(Box::new(IteSimplificationRule));
    ruleset.add_rule(Box::new(EqualitySimplificationRule));
    ruleset.add_rule(Box::new(ComparisonNormalizationRule));
    ruleset.add_rule(Box::new(BooleanPropagationRule));
    ruleset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_negation() {
        let mut manager = TermManager::new();
        let rule = DoubleNegationRule;

        // Create ¬¬p where p is a variable
        // (can't use boolean constants as mk_not simplifies them)
        let p = manager.mk_var("p", manager.sorts.bool_sort);

        // Manually construct ¬¬p to test the rule
        let not_p_kind = TermKind::Not(p);
        let not_p = manager.intern(not_p_kind, manager.sorts.bool_sort);
        let not_not_p_kind = TermKind::Not(not_p);
        let not_not_p = manager.intern(not_not_p_kind, manager.sorts.bool_sort);

        let result = rule.apply(not_not_p, &mut manager);
        assert_eq!(result, Some(p));
    }

    #[test]
    fn test_de_morgan_and() {
        let mut manager = TermManager::new();
        let rule = DeMorganRule;

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let and = manager.mk_and([p, q]);
        let not_and = manager.mk_not(and);

        // ¬(p ∧ q) → ¬p ∨ ¬q
        if let Some(result) = rule.apply(not_and, &mut manager) {
            if let Some(result_term) = manager.get(result) {
                assert!(matches!(result_term.kind, TermKind::Or(_)));
            }
        } else {
            panic!("De Morgan's law should apply");
        }
    }

    #[test]
    fn test_implication_elimination() {
        let mut manager = TermManager::new();
        let rule = ImplicationEliminationRule;

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let implies = manager.mk_implies(p, q);

        // p → q becomes ¬p ∨ q
        if let Some(result) = rule.apply(implies, &mut manager) {
            if let Some(result_term) = manager.get(result) {
                assert!(matches!(result_term.kind, TermKind::Or(_)));
            }
        } else {
            panic!("Implication elimination should apply");
        }
    }

    #[test]
    fn test_arithmetic_identity() {
        let mut manager = TermManager::new();
        let rule = ArithmeticIdentityRule;

        // x + 0 → x
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let sum = manager.mk_add([x, zero]);

        let result = rule.apply(sum, &mut manager);
        assert_eq!(result, Some(x));

        // x * 1 → x
        let one = manager.mk_int(1);
        let product = manager.mk_mul([x, one]);

        let result = rule.apply(product, &mut manager);
        assert_eq!(result, Some(x));
    }

    #[test]
    fn test_constant_folding() {
        let mut manager = TermManager::new();
        let rule = ArithmeticConstantFoldingRule;

        // (+ 1 2 3) → 6
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let sum = manager.mk_add([one, two, three]);

        if let Some(result) = rule.apply(sum, &mut manager) {
            if let Some(result_term) = manager.get(result) {
                if let TermKind::IntConst(n) = &result_term.kind {
                    assert_eq!(*n, num_bigint::BigInt::from(6));
                } else {
                    panic!("Expected integer constant");
                }
            }
        } else {
            panic!("Constant folding should apply");
        }
    }

    #[test]
    fn test_ite_simplification() {
        let mut manager = TermManager::new();
        let rule = IteSimplificationRule;

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let q = manager.mk_var("q", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // ITE(true, p, q) → p
        let ite_kind = TermKind::Ite(t, p, q);
        let ite = manager.intern(ite_kind, manager.sorts.bool_sort);
        let result = rule.apply(ite, &mut manager);
        assert_eq!(result, Some(p));

        // ITE(c, t, t) → t
        let ite_kind = TermKind::Ite(p, t, t);
        let ite = manager.intern(ite_kind, manager.sorts.bool_sort);
        let result = rule.apply(ite, &mut manager);
        assert_eq!(result, Some(t));

        // ITE(c, true, false) → c
        let ite_kind = TermKind::Ite(p, t, f);
        let ite = manager.intern(ite_kind, manager.sorts.bool_sort);
        let result = rule.apply(ite, &mut manager);
        assert_eq!(result, Some(p));
    }

    #[test]
    fn test_standard_rules() {
        let mut manager = TermManager::new();
        let rules = standard_rules();

        // Test that standard rules work
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let not_p = manager.mk_not(p);
        let not_not_p = manager.mk_not(not_p);

        let result = rules.apply_fixpoint(not_not_p, &mut manager);
        assert_eq!(result, p);
    }

    #[test]
    fn test_equality_simplification() {
        let mut manager = TermManager::new();
        let rule = EqualitySimplificationRule;

        let x = manager.mk_var("x", manager.sorts.int_sort);

        // x = x → true (use intern to bypass mk_eq simplifications)
        let eq = manager.intern(TermKind::Eq(x, x), manager.sorts.bool_sort);
        let result = rule.apply(eq, &mut manager);
        assert_eq!(result, Some(manager.mk_true()));

        // x < x → false
        let lt = manager.intern(TermKind::Lt(x, x), manager.sorts.bool_sort);
        let result = rule.apply(lt, &mut manager);
        assert_eq!(result, Some(manager.mk_false()));

        // x ≤ x → true
        let le = manager.intern(TermKind::Le(x, x), manager.sorts.bool_sort);
        let result = rule.apply(le, &mut manager);
        assert_eq!(result, Some(manager.mk_true()));
    }

    #[test]
    fn test_comparison_normalization() {
        let mut manager = TermManager::new();
        let rule = ComparisonNormalizationRule;

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // x > y → y < x
        let gt = manager.mk_gt(x, y);
        let result = rule.apply(gt, &mut manager);
        if let Some(normalized) = result {
            if let Some(term) = manager.get(normalized) {
                assert!(matches!(term.kind, TermKind::Lt(..)));
            }
        }

        // x ≥ y → y ≤ x
        let ge = manager.mk_ge(x, y);
        let result = rule.apply(ge, &mut manager);
        if let Some(normalized) = result {
            if let Some(term) = manager.get(normalized) {
                assert!(matches!(term.kind, TermKind::Le(..)));
            }
        }
    }

    #[test]
    fn test_distributivity() {
        let mut manager = TermManager::new();
        let rule = DistributivityRule;

        let a = manager.mk_var("a", manager.sorts.int_sort);
        let b = manager.mk_var("b", manager.sorts.int_sort);
        let c = manager.mk_var("c", manager.sorts.int_sort);

        // a * (b + c) → a*b + a*c
        let sum = manager.mk_add([b, c]);
        let mul = manager.mk_mul([a, sum]);

        let result = rule.apply(mul, &mut manager);
        assert!(result.is_some());

        if let Some(distributed) = result {
            if let Some(term) = manager.get(distributed) {
                assert!(matches!(term.kind, TermKind::Add(_)));
            }
        }
    }

    #[test]
    fn test_boolean_propagation() {
        let mut manager = TermManager::new();
        let rule = BooleanPropagationRule;

        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // true ∧ p → p (use intern to bypass mk_and simplifications)
        let and_true = manager.intern(
            TermKind::And(smallvec::smallvec![t, p]),
            manager.sorts.bool_sort,
        );
        let result = rule.apply(and_true, &mut manager);
        assert_eq!(result, Some(p));

        // false ∧ p → false
        let and_false = manager.intern(
            TermKind::And(smallvec::smallvec![f, p]),
            manager.sorts.bool_sort,
        );
        let result = rule.apply(and_false, &mut manager);
        assert_eq!(result, Some(f));

        // true ∨ p → true
        let or_true = manager.intern(
            TermKind::Or(smallvec::smallvec![t, p]),
            manager.sorts.bool_sort,
        );
        let result = rule.apply(or_true, &mut manager);
        assert_eq!(result, Some(t));

        // false ∨ p → p
        let or_false = manager.intern(
            TermKind::Or(smallvec::smallvec![f, p]),
            manager.sorts.bool_sort,
        );
        let result = rule.apply(or_false, &mut manager);
        assert_eq!(result, Some(p));
    }
}
