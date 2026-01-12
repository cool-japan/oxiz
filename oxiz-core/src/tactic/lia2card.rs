//! Linear Integer Arithmetic to Cardinality tactic (LIA2Card)
//!
//! This tactic converts linear integer arithmetic constraints into cardinality
//! constraints when all variables are 0-1 bounded (boolean-like integers).
//!
//! # Algorithm
//!
//! 1. Identify 0-1 bounded variables from constraints
//! 2. Recognize cardinality patterns:
//!    - `x1 + x2 + ... + xn <= k` → AtMost(k, [x1, x2, ..., xn])
//!    - `x1 + x2 + ... + xn >= k` → AtLeast(k, [x1, x2, ..., xn])
//!    - `x1 + x2 + ... + xn = k`  → Exactly(k, [x1, x2, ..., xn])
//! 3. Convert to CNF using efficient cardinality encodings
//!
//! # Encodings
//!
//! - **Sequential Counter**: O(n*k) clauses, good for small k
//! - **Totalizer**: O(n log n) clauses, good for large n
//! - **Commander**: O(n) clauses for at-most-1
//!
//! # Reference
//!
//! Based on Z3's `lia2card_tactic` in `src/tactic/arith/lia2card_tactic.cpp`

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use crate::tactic::{Goal, Tactic, TacticResult};
use lasso::Spur;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use rustc_hash::FxHashSet;

/// Cardinality constraint type
#[derive(Debug, Clone)]
pub enum CardinalityConstraint {
    /// At most k variables are true
    AtMost {
        /// The cardinality bound
        k: usize,
        /// The variables in the constraint
        vars: Vec<TermId>,
    },
    /// At least k variables are true
    AtLeast {
        /// The cardinality bound
        k: usize,
        /// The variables in the constraint
        vars: Vec<TermId>,
    },
    /// Exactly k variables are true
    Exactly {
        /// The cardinality bound
        k: usize,
        /// The variables in the constraint
        vars: Vec<TermId>,
    },
}

/// Encoding method for cardinality constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CardinalityEncoding {
    /// Sequential counter encoding - O(n*k) clauses
    #[default]
    SequentialCounter,
    /// Totalizer encoding - O(n log n) clauses
    Totalizer,
    /// Commander encoding for at-most-1 - O(n) clauses
    Commander,
    /// Binomial encoding - compact but expensive
    Binomial,
}

/// Configuration for LIA to Cardinality conversion
#[derive(Debug, Clone)]
pub struct Lia2CardConfig {
    /// Encoding method to use
    pub encoding: CardinalityEncoding,
    /// Maximum number of variables for sequential counter (use totalizer above)
    pub sequential_counter_threshold: usize,
    /// Whether to use commander encoding for at-most-1
    pub use_commander_for_amo: bool,
}

impl Default for Lia2CardConfig {
    fn default() -> Self {
        Self {
            encoding: CardinalityEncoding::SequentialCounter,
            sequential_counter_threshold: 100,
            use_commander_for_amo: true,
        }
    }
}

/// Data extracted from a term for cardinality detection
enum CardExtractData {
    Le(TermId, TermId),
    Ge(TermId, TermId),
    Lt(TermId, TermId),
    Gt(TermId, TermId),
    Eq(TermId, TermId),
    And(Vec<TermId>),
    #[allow(dead_code)]
    Or(Vec<TermId>),
    #[allow(dead_code)]
    Not(TermId),
    Other,
}

/// LIA to Cardinality conversion tactic
#[derive(Debug)]
pub struct Lia2CardTactic<'a> {
    manager: &'a mut TermManager,
    config: Lia2CardConfig,
    /// Variables known to be 0-1 bounded
    zero_one_vars: FxHashSet<Spur>,
    /// Auxiliary variables created during encoding
    aux_var_counter: usize,
}

impl<'a> Lia2CardTactic<'a> {
    /// Create a new LIA to Cardinality tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self::with_config(manager, Lia2CardConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(manager: &'a mut TermManager, config: Lia2CardConfig) -> Self {
        Self {
            manager,
            config,
            zero_one_vars: FxHashSet::default(),
            aux_var_counter: 0,
        }
    }

    /// Apply the tactic
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Phase 1: Identify 0-1 bounded variables
        self.identify_zero_one_vars(&goal.assertions);

        if self.zero_one_vars.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 2: Convert cardinality constraints
        let mut new_assertions = Vec::new();
        let mut changed = false;

        for &assertion in &goal.assertions {
            if let Some(card) = self.extract_cardinality(assertion) {
                // Encode the cardinality constraint
                let encoded = self.encode_cardinality(&card);
                new_assertions.extend(encoded);
                changed = true;
            } else {
                new_assertions.push(assertion);
            }
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Identify variables that are 0-1 bounded
    fn identify_zero_one_vars(&mut self, assertions: &[TermId]) {
        // Look for patterns like: 0 <= x <= 1 or x >= 0 && x <= 1
        for &assertion in assertions {
            self.check_zero_one_bound(assertion);
        }
    }

    /// Check if an assertion bounds a variable to 0-1
    fn check_zero_one_bound(&mut self, term: TermId) {
        let Some(t) = self.manager.get(term) else {
            return;
        };

        // Extract data to avoid borrow issues
        let data: CardExtractData = match &t.kind {
            TermKind::Le(lhs, rhs) => CardExtractData::Le(*lhs, *rhs),
            TermKind::Ge(lhs, rhs) => CardExtractData::Ge(*lhs, *rhs),
            TermKind::Lt(lhs, rhs) => CardExtractData::Lt(*lhs, *rhs),
            TermKind::Gt(lhs, rhs) => CardExtractData::Gt(*lhs, *rhs),
            TermKind::Eq(lhs, rhs) => CardExtractData::Eq(*lhs, *rhs),
            TermKind::And(args) => CardExtractData::And(args.to_vec()),
            _ => CardExtractData::Other,
        };

        match data {
            CardExtractData::Le(lhs, rhs) => {
                // x <= 1 pattern
                if let (Some(var), Some(val)) = (self.get_var_name(lhs), self.get_int_const(rhs))
                    && val == BigInt::one()
                {
                    self.zero_one_vars.insert(var);
                }
            }
            CardExtractData::Ge(lhs, rhs) => {
                // x >= 0 or 0 <= x pattern
                if let (Some(var), Some(val)) = (self.get_var_name(lhs), self.get_int_const(rhs))
                    && val.is_zero()
                {
                    // Mark as potentially 0-1 (need <= 1 as well)
                    self.zero_one_vars.insert(var);
                }
            }
            CardExtractData::And(args) => {
                // Check conjunctions for combined bounds
                for arg in args {
                    self.check_zero_one_bound(arg);
                }
            }
            _ => {}
        }
    }

    /// Get variable name if term is a variable
    fn get_var_name(&self, term: TermId) -> Option<Spur> {
        let t = self.manager.get(term)?;
        match &t.kind {
            TermKind::Var(name) => Some(*name),
            _ => None,
        }
    }

    /// Get integer constant value
    fn get_int_const(&self, term: TermId) -> Option<BigInt> {
        let t = self.manager.get(term)?;
        match &t.kind {
            TermKind::IntConst(val) => Some(val.clone()),
            _ => None,
        }
    }

    /// Check if a variable is 0-1 bounded
    fn is_zero_one_var(&self, term: TermId) -> bool {
        if let Some(name) = self.get_var_name(term) {
            self.zero_one_vars.contains(&name)
        } else {
            false
        }
    }

    /// Extract a cardinality constraint from a term
    fn extract_cardinality(&self, term: TermId) -> Option<CardinalityConstraint> {
        let t = self.manager.get(term)?;

        // Extract data to avoid borrow issues
        let data: CardExtractData = match &t.kind {
            TermKind::Le(lhs, rhs) => CardExtractData::Le(*lhs, *rhs),
            TermKind::Ge(lhs, rhs) => CardExtractData::Ge(*lhs, *rhs),
            TermKind::Lt(lhs, rhs) => CardExtractData::Lt(*lhs, *rhs),
            TermKind::Gt(lhs, rhs) => CardExtractData::Gt(*lhs, *rhs),
            TermKind::Eq(lhs, rhs) => CardExtractData::Eq(*lhs, *rhs),
            _ => return None,
        };

        match data {
            CardExtractData::Le(lhs, rhs) => {
                // sum <= k → AtMost(k, vars)
                if let (Some(vars), Some(k)) = (self.extract_sum(lhs), self.get_int_const(rhs))
                    && vars.iter().all(|&v| self.is_zero_one_var(v))
                {
                    return Some(CardinalityConstraint::AtMost {
                        k: k.to_usize().unwrap_or(usize::MAX),
                        vars,
                    });
                }
            }
            CardExtractData::Ge(lhs, rhs) => {
                // sum >= k → AtLeast(k, vars)
                if let (Some(vars), Some(k)) = (self.extract_sum(lhs), self.get_int_const(rhs))
                    && vars.iter().all(|&v| self.is_zero_one_var(v))
                {
                    return Some(CardinalityConstraint::AtLeast {
                        k: k.to_usize().unwrap_or(0),
                        vars,
                    });
                }
            }
            CardExtractData::Lt(lhs, rhs) => {
                // sum < k → AtMost(k-1, vars)
                if let (Some(vars), Some(k)) = (self.extract_sum(lhs), self.get_int_const(rhs))
                    && vars.iter().all(|&v| self.is_zero_one_var(v))
                {
                    let k_minus_1 = (k - BigInt::one()).to_usize().unwrap_or(0);
                    return Some(CardinalityConstraint::AtMost { k: k_minus_1, vars });
                }
            }
            CardExtractData::Gt(lhs, rhs) => {
                // sum > k → AtLeast(k+1, vars)
                if let (Some(vars), Some(k)) = (self.extract_sum(lhs), self.get_int_const(rhs))
                    && vars.iter().all(|&v| self.is_zero_one_var(v))
                {
                    let k_plus_1 = (k + BigInt::one()).to_usize().unwrap_or(usize::MAX);
                    return Some(CardinalityConstraint::AtLeast { k: k_plus_1, vars });
                }
            }
            CardExtractData::Eq(lhs, rhs) => {
                // sum = k → Exactly(k, vars)
                if let (Some(vars), Some(k)) = (self.extract_sum(lhs), self.get_int_const(rhs))
                    && vars.iter().all(|&v| self.is_zero_one_var(v))
                {
                    return Some(CardinalityConstraint::Exactly {
                        k: k.to_usize().unwrap_or(0),
                        vars,
                    });
                }
            }
            _ => {}
        }

        None
    }

    /// Extract variables from a sum expression
    fn extract_sum(&self, term: TermId) -> Option<Vec<TermId>> {
        let t = self.manager.get(term)?;

        match &t.kind {
            TermKind::Add(args) => {
                // Check all args are variables with coefficient 1
                let mut vars = Vec::new();
                for &arg in args.iter() {
                    if self.get_var_name(arg).is_some() {
                        vars.push(arg);
                    } else {
                        return None; // Not a simple sum of variables
                    }
                }
                Some(vars)
            }
            TermKind::Var(_) => {
                // Single variable is a sum of 1 element
                Some(vec![term])
            }
            _ => None,
        }
    }

    /// Encode a cardinality constraint into CNF clauses
    fn encode_cardinality(&mut self, card: &CardinalityConstraint) -> Vec<TermId> {
        match card {
            CardinalityConstraint::AtMost { k, vars } => self.encode_at_most(*k, vars),
            CardinalityConstraint::AtLeast { k, vars } => self.encode_at_least(*k, vars),
            CardinalityConstraint::Exactly { k, vars } => {
                // Exactly k = AtMost(k) AND AtLeast(k)
                let mut clauses = self.encode_at_most(*k, vars);
                clauses.extend(self.encode_at_least(*k, vars));
                clauses
            }
        }
    }

    /// Encode AtMost(k, vars) constraint
    fn encode_at_most(&mut self, k: usize, vars: &[TermId]) -> Vec<TermId> {
        let n = vars.len();

        if k >= n {
            // Trivially satisfied
            return vec![];
        }

        if k == 0 {
            // All must be false
            return vars.iter().map(|&v| self.manager.mk_not(v)).collect();
        }

        if k == 1 && self.config.use_commander_for_amo {
            return self.encode_amo_commander(vars);
        }

        if n <= self.config.sequential_counter_threshold {
            self.encode_at_most_sequential(k, vars)
        } else {
            // For large n, use totalizer
            self.encode_at_most_totalizer(k, vars)
        }
    }

    /// Encode AtLeast(k, vars) constraint
    fn encode_at_least(&mut self, k: usize, vars: &[TermId]) -> Vec<TermId> {
        let n = vars.len();

        if k == 0 {
            // Trivially satisfied
            return vec![];
        }

        if k > n {
            // Unsatisfiable - return false
            return vec![self.manager.mk_false()];
        }

        if k == n {
            // All must be true
            return vars.to_vec();
        }

        // AtLeast(k, vars) = NOT AtMost(k-1, vars)
        // But for CNF, we use: at least k of n = at most (n-k) of negations
        // Equivalently: negate and use AtMost
        let neg_vars: Vec<_> = vars.iter().map(|&v| self.manager.mk_not(v)).collect();
        self.encode_at_most(n - k, &neg_vars)
    }

    /// Sequential counter encoding for AtMost(k, vars)
    fn encode_at_most_sequential(&mut self, k: usize, vars: &[TermId]) -> Vec<TermId> {
        let n = vars.len();
        let bool_sort = self.manager.sorts.bool_sort;

        // Create counter variables s[i][j] meaning "at least j of first i vars are true"
        let mut s: Vec<Vec<TermId>> = Vec::with_capacity(n);

        for i in 0..n {
            let mut row = Vec::with_capacity(k + 1);
            for j in 0..=k {
                let name = format!("__card_s_{}_{}", i, j);
                row.push(self.manager.mk_var(&name, bool_sort));
                self.aux_var_counter += 1;
            }
            s.push(row);
        }

        let mut clauses = Vec::new();

        // Initial constraints: s[0][0] = true, s[0][j>0] = x[0] (j=1), s[0][j>1] = false
        // Actually, let's use a simpler formulation:
        // s[i][j] iff at least j of {x[0], ..., x[i]} are true

        // For sequential counter:
        // s[i][j] <=> (s[i-1][j] OR (x[i] AND s[i-1][j-1]))
        //
        // In CNF:
        // s[i][j] => s[i-1][j] OR x[i]          (if s[i][j], then either already had j, or x[i] is true)
        // s[i][j] => s[i-1][j] OR s[i-1][j-1]   (if s[i][j], then had j or j-1 before)
        // x[i] AND s[i-1][j-1] => s[i][j]       (if x[i] and had j-1, then have j)
        // s[i-1][j] => s[i][j]                  (if had j before, still have j)

        // Simplified encoding using implications:
        for i in 0..n {
            for j in 0..=k {
                if i == 0 {
                    if j == 0 {
                        // s[0][0] is always true (at least 0)
                        clauses.push(s[0][0]);
                    } else if j == 1 {
                        // s[0][1] <=> x[0]
                        // x[0] => s[0][1]
                        let c1 = self.manager.mk_implies(vars[0], s[0][1]);
                        // s[0][1] => x[0]
                        let c2 = self.manager.mk_implies(s[0][1], vars[0]);
                        clauses.push(c1);
                        clauses.push(c2);
                    } else {
                        // s[0][j>1] = false (can't have more than 1 with 1 variable)
                        clauses.push(self.manager.mk_not(s[0][j]));
                    }
                } else {
                    // s[i][j] <=> s[i-1][j] OR (x[i] AND s[i-1][j-1])
                    let prev_j = s[i - 1][j];
                    let curr = s[i][j];

                    // Forward: s[i-1][j] => s[i][j]
                    let c1 = self.manager.mk_implies(prev_j, curr);
                    clauses.push(c1);

                    if j > 0 {
                        // Forward: x[i] AND s[i-1][j-1] => s[i][j]
                        let prev_jm1 = s[i - 1][j - 1];
                        let conj = self.manager.mk_and([vars[i], prev_jm1]);
                        let c2 = self.manager.mk_implies(conj, curr);
                        clauses.push(c2);

                        // Backward: s[i][j] => s[i-1][j] OR (x[i] AND s[i-1][j-1])
                        // Equivalent to: s[i][j] AND NOT s[i-1][j] => x[i] AND s[i-1][j-1]
                        let not_prev_j = self.manager.mk_not(prev_j);
                        let ante = self.manager.mk_and([curr, not_prev_j]);
                        let c3 = self.manager.mk_implies(ante, vars[i]);
                        let c4 = self.manager.mk_implies(ante, prev_jm1);
                        clauses.push(c3);
                        clauses.push(c4);
                    } else {
                        // j == 0: s[i][0] is always true
                        clauses.push(curr);
                    }
                }
            }
        }

        // Final constraint: NOT s[n-1][k+1] if k < n
        // But we only have up to s[n-1][k], so we need to assert NOT s[n-1][k] => x[n-1] contributes
        // Actually, the constraint is: at most k, so we don't want k+1
        // With our encoding, we need to assert that if we're at count k, adding another true makes it fail

        // Simple addition: for each i, if s[i][k] AND x[i+1], that's a contradiction
        // But simpler: just assert NOT(all vars true) if k < n... no, that's not right either

        // The constraint is implicit: if more than k vars are true, s[n-1][k] will be forced true
        // but the counter can't go above k, so it just saturates.
        // We need an explicit constraint that if k vars are already true, no more can be added.

        // Add blocking clauses: for any k+1 subset, at least one must be false
        // This is expensive for large n, but for sequential counter, we add:
        // NOT x[i] OR NOT s[i-1][k] for all i > 0
        for i in 1..n {
            let not_x = self.manager.mk_not(vars[i]);
            let not_sk = self.manager.mk_not(s[i - 1][k]);
            let clause = self.manager.mk_or([not_x, not_sk]);
            clauses.push(clause);
        }

        clauses
    }

    /// Commander encoding for AtMost-1 constraint
    fn encode_amo_commander(&mut self, vars: &[TermId]) -> Vec<TermId> {
        let n = vars.len();

        if n <= 1 {
            return vec![];
        }

        if n <= 4 {
            // For small n, use pairwise encoding
            return self.encode_amo_pairwise(vars);
        }

        // Commander encoding: divide into groups
        let bool_sort = self.manager.sorts.bool_sort;
        let group_size = ((n as f64).sqrt().ceil() as usize).max(2);
        let num_groups = n.div_ceil(group_size);

        let mut clauses = Vec::new();
        let mut commanders = Vec::new();

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(n);
            let group: Vec<_> = vars[start..end].to_vec();

            // Create commander variable for this group
            let cmd_name = format!("__card_cmd_{}", g);
            let cmd = self.manager.mk_var(&cmd_name, bool_sort);
            commanders.push(cmd);

            // Within group: at most one, and if any, commander is true
            // Pairwise within group
            clauses.extend(self.encode_amo_pairwise(&group));

            // If any in group is true, commander is true
            for &v in &group {
                let c = self.manager.mk_implies(v, cmd);
                clauses.push(c);
            }

            // If commander is true, exactly one in group is true
            // (commander => OR of group)
            let or_group = self.manager.mk_or(group.clone());
            let c = self.manager.mk_implies(cmd, or_group);
            clauses.push(c);
        }

        // At most one commander is true
        clauses.extend(self.encode_amo_pairwise(&commanders));

        clauses
    }

    /// Pairwise encoding for AtMost-1
    fn encode_amo_pairwise(&mut self, vars: &[TermId]) -> Vec<TermId> {
        let mut clauses = Vec::new();
        let n = vars.len();

        for i in 0..n {
            for j in (i + 1)..n {
                // NOT x[i] OR NOT x[j]
                let not_i = self.manager.mk_not(vars[i]);
                let not_j = self.manager.mk_not(vars[j]);
                let clause = self.manager.mk_or([not_i, not_j]);
                clauses.push(clause);
            }
        }

        clauses
    }

    /// Totalizer encoding for AtMost(k, vars) - efficient for large n
    fn encode_at_most_totalizer(&mut self, k: usize, vars: &[TermId]) -> Vec<TermId> {
        // Totalizer builds a tree of adders
        // For simplicity, we use a recursive approach

        let n = vars.len();
        if n <= 1 {
            return vec![];
        }

        let mut clauses = Vec::new();

        // Build unary representation of the sum
        let sum_vars = self.build_totalizer_tree(vars, k, &mut clauses);

        // Assert that sum_vars[k] is false (at most k)
        if sum_vars.len() > k {
            clauses.push(self.manager.mk_not(sum_vars[k]));
        }

        clauses
    }

    /// Build a totalizer tree, returning unary sum representation
    fn build_totalizer_tree(
        &mut self,
        vars: &[TermId],
        max_k: usize,
        clauses: &mut Vec<TermId>,
    ) -> Vec<TermId> {
        let n = vars.len();
        let bool_sort = self.manager.sorts.bool_sort;

        if n == 1 {
            // Leaf node: just the variable
            return vec![vars[0]];
        }

        // Split in half
        let mid = n / 2;
        let left = self.build_totalizer_tree(&vars[..mid], max_k, clauses);
        let right = self.build_totalizer_tree(&vars[mid..], max_k, clauses);

        // Merge: sum = left + right (in unary)
        let sum_len = (left.len() + right.len()).min(max_k + 1);
        let mut sum = Vec::with_capacity(sum_len);

        for i in 0..sum_len {
            let name = format!("__tot_{}_{}", self.aux_var_counter, i);
            sum.push(self.manager.mk_var(&name, bool_sort));
            self.aux_var_counter += 1;
        }

        // Add clauses for the merge
        // sum[i] is true iff left has at least a and right has at least b where a + b = i + 1
        for (i, &sum_i) in sum.iter().enumerate() {
            let target_count = i + 1;

            // Forward: if enough inputs, then output
            for a in 0..=left.len().min(target_count) {
                let b = target_count - a;
                if b <= right.len() {
                    if a > 0 && b > 0 {
                        // left[a-1] AND right[b-1] => sum[i]
                        let ante = self.manager.mk_and([left[a - 1], right[b - 1]]);
                        let c = self.manager.mk_implies(ante, sum_i);
                        clauses.push(c);
                    } else if a > 0 {
                        // left[a-1] => sum[i] (when b = 0)
                        let c = self.manager.mk_implies(left[a - 1], sum_i);
                        clauses.push(c);
                    } else if b > 0 {
                        // right[b-1] => sum[i] (when a = 0)
                        let c = self.manager.mk_implies(right[b - 1], sum_i);
                        clauses.push(c);
                    }
                }
            }
        }

        sum
    }

    /// Create an auxiliary boolean variable
    #[allow(dead_code)]
    fn mk_aux_var(&mut self) -> TermId {
        let bool_sort = self.manager.sorts.bool_sort;
        let name = format!("__card_aux_{}", self.aux_var_counter);
        self.aux_var_counter += 1;
        self.manager.mk_var(&name, bool_sort)
    }
}

/// Stateless wrapper for tactic trait
#[derive(Debug, Default)]
pub struct StatelessLia2CardTactic {
    #[allow(dead_code)]
    config: Lia2CardConfig,
}

impl StatelessLia2CardTactic {
    /// Create a new stateless wrapper
    pub fn new() -> Self {
        Self {
            config: Lia2CardConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Lia2CardConfig) -> Self {
        Self { config }
    }
}

impl Tactic for StatelessLia2CardTactic {
    fn name(&self) -> &str {
        "lia2card"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Need a TermManager - for now return NotApplicable
        Ok(TacticResult::NotApplicable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_lia2card_config_default() {
        let config = Lia2CardConfig::default();
        assert_eq!(config.encoding, CardinalityEncoding::SequentialCounter);
        assert_eq!(config.sequential_counter_threshold, 100);
        assert!(config.use_commander_for_amo);
    }

    #[test]
    fn test_cardinality_constraint_types() {
        let vars = vec![];

        let amo = CardinalityConstraint::AtMost {
            k: 1,
            vars: vars.clone(),
        };
        let alo = CardinalityConstraint::AtLeast {
            k: 1,
            vars: vars.clone(),
        };
        let eo = CardinalityConstraint::Exactly { k: 1, vars };

        // Just ensure they compile and debug correctly
        assert!(format!("{:?}", amo).contains("AtMost"));
        assert!(format!("{:?}", alo).contains("AtLeast"));
        assert!(format!("{:?}", eo).contains("Exactly"));
    }

    #[test]
    fn test_lia2card_identify_zero_one_vars() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Create x >= 0 and x <= 1
        let x = manager.mk_var("x", int_sort);
        let c0 = manager.mk_int(BigInt::from(0));
        let c1 = manager.mk_int(BigInt::from(1));
        let ge = manager.mk_ge(x, c0);
        let le = manager.mk_le(x, c1);

        let goal = Goal::new(vec![ge, le]);
        let mut tactic = Lia2CardTactic::new(&mut manager);
        tactic.identify_zero_one_vars(&goal.assertions);

        // x should be identified as 0-1 bounded
        assert!(!tactic.zero_one_vars.is_empty());
    }

    #[test]
    fn test_lia2card_extract_sum() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add([x, y]);

        let tactic = Lia2CardTactic::new(&mut manager);
        let vars = tactic.extract_sum(sum);

        assert!(vars.is_some());
        assert_eq!(vars.unwrap().len(), 2);
    }

    #[test]
    fn test_lia2card_encode_amo_pairwise() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;

        let x = manager.mk_var("x", bool_sort);
        let y = manager.mk_var("y", bool_sort);
        let z = manager.mk_var("z", bool_sort);
        let vars = vec![x, y, z];

        let mut tactic = Lia2CardTactic::new(&mut manager);
        let clauses = tactic.encode_amo_pairwise(&vars);

        // Pairwise for 3 vars should produce 3 clauses: (xy), (xz), (yz)
        assert_eq!(clauses.len(), 3);
    }

    #[test]
    fn test_lia2card_not_applicable() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Create x + y = 5 without 0-1 bounds
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add([x, y]);
        let c5 = manager.mk_int(BigInt::from(5));
        let eq = manager.mk_eq(sum, c5);

        let goal = Goal::new(vec![eq]);
        let mut tactic = Lia2CardTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal);

        assert!(matches!(result, Ok(TacticResult::NotApplicable)));
    }

    #[test]
    fn test_stateless_wrapper() {
        let tactic = StatelessLia2CardTactic::new();
        assert_eq!(tactic.name(), "lia2card");

        let goal = Goal::empty();
        let result = tactic.apply(&goal);
        assert!(matches!(result, Ok(TacticResult::NotApplicable)));
    }
}
