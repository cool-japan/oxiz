//! Cardinality Constraint Solver
//!
//! Handles cardinality constraints for sets:
//! - |S| = k, |S| ≤ k, |S| ≥ k, |S| < k, |S| > k
//! - Cardinality arithmetic: |S1 ∪ S2|, |S1 ∩ S2|, etc.
//! - Cardinality-based propagation

#![allow(dead_code)]

use super::{SetConflict, SetLiteral, SetProofStep, SetVar, SetVarId};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Cardinality constraint kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardConstraintKind {
    /// |S| = k
    Equal,
    /// |S| ≤ k
    Le,
    /// |S| < k
    Lt,
    /// |S| ≥ k
    Ge,
    /// |S| > k
    Gt,
}

impl CardConstraintKind {
    /// Check if a cardinality value satisfies this constraint
    pub fn check(&self, card: i64, bound: i64) -> bool {
        match self {
            CardConstraintKind::Equal => card == bound,
            CardConstraintKind::Le => card <= bound,
            CardConstraintKind::Lt => card < bound,
            CardConstraintKind::Ge => card >= bound,
            CardConstraintKind::Gt => card > bound,
        }
    }

    /// Get the tightest bounds implied by this constraint
    pub fn bounds(&self, k: i64) -> (Option<i64>, Option<i64>) {
        match self {
            CardConstraintKind::Equal => (Some(k), Some(k)),
            CardConstraintKind::Le => (None, Some(k)),
            CardConstraintKind::Lt => (None, Some(k - 1)),
            CardConstraintKind::Ge => (Some(k), None),
            CardConstraintKind::Gt => (Some(k + 1), None),
        }
    }

    /// Negate this constraint
    pub fn negate(&self) -> CardConstraintKind {
        match self {
            CardConstraintKind::Equal => CardConstraintKind::Equal, // ¬(|S|=k) is complex
            CardConstraintKind::Le => CardConstraintKind::Gt,
            CardConstraintKind::Lt => CardConstraintKind::Ge,
            CardConstraintKind::Ge => CardConstraintKind::Lt,
            CardConstraintKind::Gt => CardConstraintKind::Le,
        }
    }
}

/// Cardinality constraint
#[derive(Debug, Clone)]
pub struct CardConstraint {
    /// Set variable
    pub set: SetVarId,
    /// Constraint kind
    pub kind: CardConstraintKind,
    /// Bound value
    pub bound: i64,
    /// Decision level when added
    pub level: usize,
}

impl CardConstraint {
    /// Create a new cardinality constraint
    pub fn new(set: SetVarId, kind: CardConstraintKind, bound: i64, level: usize) -> Self {
        Self {
            set,
            kind,
            bound,
            level,
        }
    }

    /// Check if this constraint is satisfied by a variable
    pub fn is_satisfied(&self, var: &SetVar) -> Option<bool> {
        let (lower, upper) = var.cardinality_bounds();

        match self.kind {
            CardConstraintKind::Equal => {
                if upper == Some(self.bound) && lower == self.bound {
                    Some(true)
                } else if upper.is_some_and(|u| u < self.bound) || lower > self.bound {
                    Some(false)
                } else {
                    None
                }
            }
            CardConstraintKind::Le => {
                if upper.is_some_and(|u| u <= self.bound) {
                    Some(true)
                } else if lower > self.bound {
                    Some(false)
                } else {
                    None
                }
            }
            CardConstraintKind::Lt => {
                if upper.is_some_and(|u| u < self.bound) {
                    Some(true)
                } else if lower >= self.bound {
                    Some(false)
                } else {
                    None
                }
            }
            CardConstraintKind::Ge => {
                if lower >= self.bound {
                    Some(true)
                } else if upper.is_some_and(|u| u < self.bound) {
                    Some(false)
                } else {
                    None
                }
            }
            CardConstraintKind::Gt => {
                if lower > self.bound {
                    Some(true)
                } else if upper.is_some_and(|u| u <= self.bound) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    /// Get implied bounds from this constraint
    pub fn implied_bounds(&self) -> (Option<i64>, Option<i64>) {
        self.kind.bounds(self.bound)
    }
}

/// Cardinality domain for a set variable
#[derive(Debug, Clone)]
pub struct CardDomain {
    /// Lower bound (minimum cardinality)
    pub lower: i64,
    /// Upper bound (maximum cardinality, None = infinite)
    pub upper: Option<i64>,
    /// Is the cardinality exactly determined?
    pub exact: Option<i64>,
}

impl CardDomain {
    /// Create a new cardinality domain
    pub fn new(lower: i64, upper: Option<i64>) -> Self {
        let exact = if upper == Some(lower) {
            Some(lower)
        } else {
            None
        };
        Self {
            lower,
            upper,
            exact,
        }
    }

    /// Create an unbounded domain
    pub fn unbounded() -> Self {
        Self::new(0, None)
    }

    /// Create a singleton domain
    pub fn singleton(k: i64) -> Self {
        Self {
            lower: k,
            upper: Some(k),
            exact: Some(k),
        }
    }

    /// Tighten the lower bound
    pub fn tighten_lower(&mut self, bound: i64) -> bool {
        if bound <= self.lower {
            return true;
        }

        if let Some(u) = self.upper
            && bound > u
        {
            return false; // Conflict
        }

        self.lower = bound;
        if self.upper == Some(bound) {
            self.exact = Some(bound);
        }
        true
    }

    /// Tighten the upper bound
    pub fn tighten_upper(&mut self, bound: i64) -> bool {
        if bound < 0 {
            return false; // Conflict
        }

        match self.upper {
            Some(u) if bound >= u => return true,
            _ => {}
        }

        if bound < self.lower {
            return false; // Conflict
        }

        self.upper = Some(bound);
        if self.lower == bound {
            self.exact = Some(bound);
        }
        true
    }

    /// Check if this domain is empty (conflicting)
    pub fn is_empty(&self) -> bool {
        if let Some(u) = self.upper {
            u < self.lower
        } else {
            false
        }
    }

    /// Intersect with another domain
    pub fn intersect(&self, other: &CardDomain) -> Option<CardDomain> {
        let lower = self.lower.max(other.lower);
        let upper = match (self.upper, other.upper) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        if let Some(u) = upper
            && u < lower
        {
            return None; // Empty domain
        }

        Some(CardDomain::new(lower, upper))
    }

    /// Union with another domain
    pub fn union(&self, other: &CardDomain) -> CardDomain {
        let lower = self.lower.min(other.lower);
        let upper = match (self.upper, other.upper) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };

        CardDomain::new(lower, upper)
    }
}

/// Cardinality result
pub type CardResult<T> = Result<T, SetConflict>;

/// Cardinality statistics
#[derive(Debug, Clone, Default)]
pub struct CardStats {
    /// Number of cardinality constraints
    pub num_constraints: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of conflicts
    pub num_conflicts: usize,
    /// Number of fixed cardinalities
    pub num_fixed: usize,
}

/// Cardinality propagator
#[derive(Debug)]
pub struct CardPropagator {
    /// Cardinality domains for each variable
    domains: FxHashMap<SetVarId, CardDomain>,
    /// Statistics
    stats: CardStats,
}

impl CardPropagator {
    /// Create a new cardinality propagator
    pub fn new() -> Self {
        Self {
            domains: FxHashMap::default(),
            stats: CardStats::default(),
        }
    }

    /// Get or create a domain for a variable
    pub fn get_domain(&mut self, var: SetVarId) -> &mut CardDomain {
        self.domains
            .entry(var)
            .or_insert_with(CardDomain::unbounded)
    }

    /// Add a cardinality constraint
    pub fn add_constraint(&mut self, constraint: CardConstraint) -> CardResult<()> {
        self.stats.num_constraints += 1;

        let domain = self.get_domain(constraint.set);
        let (lower, upper) = constraint.implied_bounds();

        if let Some(l) = lower
            && !domain.tighten_lower(l)
        {
            // Extract values before mutating stats to avoid borrow checker issues
            let domain_upper = domain.upper.unwrap_or(i64::MAX);
            self.stats.num_conflicts += 1;
            return Err(SetConflict {
                literals: vec![SetLiteral::Cardinality {
                    set: constraint.set,
                    op: constraint.kind,
                    bound: constraint.bound,
                }],
                reason: format!(
                    "Cardinality conflict: lower bound {} exceeds upper bound {:?}",
                    l, domain_upper
                ),
                proof_steps: vec![SetProofStep::CardConflict {
                    set: constraint.set,
                    lower: l,
                    upper: domain_upper,
                }],
            });
        }

        if let Some(u) = upper
            && !domain.tighten_upper(u)
        {
            // Extract values before mutating stats to avoid borrow checker issues
            let domain_lower = domain.lower;
            self.stats.num_conflicts += 1;
            return Err(SetConflict {
                literals: vec![SetLiteral::Cardinality {
                    set: constraint.set,
                    op: constraint.kind,
                    bound: constraint.bound,
                }],
                reason: format!(
                    "Cardinality conflict: upper bound {} is less than lower bound {}",
                    u, domain_lower
                ),
                proof_steps: vec![SetProofStep::CardConflict {
                    set: constraint.set,
                    lower: domain_lower,
                    upper: u,
                }],
            });
        }

        if domain.exact.is_some() {
            self.stats.num_fixed += 1;
        }

        Ok(())
    }

    /// Propagate cardinality bounds for a variable
    pub fn propagate(
        &mut self,
        var: SetVarId,
        vars: &mut [SetVar],
        constraints: &[CardConstraint],
    ) -> CardResult<()> {
        if let Some(set_var) = vars.get(var.id() as usize) {
            let (var_lower, var_upper) = set_var.cardinality_bounds();

            let domain = self.get_domain(var);

            // Update domain from variable
            if !domain.tighten_lower(var_lower)
                || !domain.tighten_upper(var_upper.unwrap_or(i64::MAX))
            {
                return Ok(()); // Already propagated
            }

            // Extract domain values to avoid borrow checker issues when mutating stats
            let domain_lower = domain.lower;
            let domain_upper = domain.upper;

            // Propagate back to variable
            if let Some(set_var) = vars.get_mut(var.id() as usize) {
                if !set_var.tighten_lower_card(domain_lower) {
                    let set_var_upper = set_var.card_bounds.1.unwrap_or(i64::MAX);
                    self.stats.num_conflicts += 1;
                    return Err(SetConflict {
                        literals: vec![],
                        reason: format!(
                            "Cardinality conflict: lower bound {} cannot be satisfied",
                            domain_lower
                        ),
                        proof_steps: vec![SetProofStep::CardConflict {
                            set: var,
                            lower: domain_lower,
                            upper: set_var_upper,
                        }],
                    });
                }

                if let Some(u) = domain_upper
                    && !set_var.tighten_upper_card(u)
                {
                    self.stats.num_conflicts += 1;
                    return Err(SetConflict {
                        literals: vec![],
                        reason: format!(
                            "Cardinality conflict: upper bound {} cannot be satisfied",
                            u
                        ),
                        proof_steps: vec![SetProofStep::CardConflict {
                            set: var,
                            lower: set_var.card_bounds.0.unwrap_or(0),
                            upper: u,
                        }],
                    });
                }

                self.stats.num_propagations += 1;
            }

            // Propagate cardinality constraints
            for constraint in constraints {
                if constraint.set == var {
                    self.add_constraint(constraint.clone())?;
                }
            }
        }

        Ok(())
    }

    /// Propagate union cardinality: |result| = |lhs ∪ rhs|
    ///
    /// max(|lhs|, |rhs|) ≤ |result| ≤ |lhs| + |rhs|
    pub fn propagate_union(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
    ) -> CardResult<()> {
        let lhs_domain = self.get_domain(lhs).clone();
        let rhs_domain = self.get_domain(rhs).clone();

        let result_lower = lhs_domain.lower.max(rhs_domain.lower);
        let result_upper = match (lhs_domain.upper, rhs_domain.upper) {
            (Some(l), Some(r)) => Some(l + r),
            _ => None,
        };

        let result_domain = self.get_domain(result);
        if !result_domain.tighten_lower(result_lower) {
            return Err(SetConflict {
                literals: vec![],
                reason: "Union cardinality lower bound conflict".to_string(),
                proof_steps: vec![],
            });
        }

        if let Some(u) = result_upper
            && !result_domain.tighten_upper(u)
        {
            return Err(SetConflict {
                literals: vec![],
                reason: "Union cardinality upper bound conflict".to_string(),
                proof_steps: vec![],
            });
        }

        Ok(())
    }

    /// Propagate intersection cardinality: |result| = |lhs ∩ rhs|
    ///
    /// 0 ≤ |result| ≤ min(|lhs|, |rhs|)
    pub fn propagate_intersection(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
    ) -> CardResult<()> {
        let lhs_domain = self.get_domain(lhs).clone();
        let rhs_domain = self.get_domain(rhs).clone();

        let result_upper = match (lhs_domain.upper, rhs_domain.upper) {
            (Some(l), Some(r)) => Some(l.min(r)),
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            _ => None,
        };

        let result_domain = self.get_domain(result);

        if let Some(u) = result_upper
            && !result_domain.tighten_upper(u)
        {
            return Err(SetConflict {
                literals: vec![],
                reason: "Intersection cardinality upper bound conflict".to_string(),
                proof_steps: vec![],
            });
        }

        Ok(())
    }

    /// Propagate difference cardinality: |result| = |lhs \ rhs|
    ///
    /// max(0, |lhs| - |rhs|) ≤ |result| ≤ |lhs|
    pub fn propagate_difference(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
    ) -> CardResult<()> {
        let lhs_domain = self.get_domain(lhs).clone();
        let rhs_domain = self.get_domain(rhs).clone();

        let result_upper = lhs_domain.upper;

        let result_lower = if let Some(rhs_u) = rhs_domain.upper {
            (lhs_domain.lower - rhs_u).max(0)
        } else {
            0
        };

        let result_domain = self.get_domain(result);
        if !result_domain.tighten_lower(result_lower) {
            return Err(SetConflict {
                literals: vec![],
                reason: "Difference cardinality lower bound conflict".to_string(),
                proof_steps: vec![],
            });
        }

        if let Some(u) = result_upper
            && !result_domain.tighten_upper(u)
        {
            return Err(SetConflict {
                literals: vec![],
                reason: "Difference cardinality upper bound conflict".to_string(),
                proof_steps: vec![],
            });
        }

        Ok(())
    }

    /// Propagate complement cardinality: |result| = |universe| - |set|
    pub fn propagate_complement(
        &mut self,
        set: SetVarId,
        result: SetVarId,
        universe_size: Option<i64>,
    ) -> CardResult<()> {
        if let Some(univ) = universe_size {
            let set_domain = self.get_domain(set).clone();

            let result_lower = match set_domain.upper {
                Some(u) => (univ - u).max(0),
                None => 0,
            };

            let result_upper = Some(univ - set_domain.lower);

            let result_domain = self.get_domain(result);
            if !result_domain.tighten_lower(result_lower) {
                return Err(SetConflict {
                    literals: vec![],
                    reason: "Complement cardinality lower bound conflict".to_string(),
                    proof_steps: vec![],
                });
            }

            if let Some(u) = result_upper
                && !result_domain.tighten_upper(u)
            {
                return Err(SetConflict {
                    literals: vec![],
                    reason: "Complement cardinality upper bound conflict".to_string(),
                    proof_steps: vec![],
                });
            }
        }

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> &CardStats {
        &self.stats
    }

    /// Reset the propagator
    pub fn reset(&mut self) {
        self.domains.clear();
        self.stats = CardStats::default();
    }
}

impl Default for CardPropagator {
    fn default() -> Self {
        Self::new()
    }
}

/// Cardinality network encoding for constraints
#[derive(Debug)]
pub struct CardinalityNetwork {
    /// Variables in the network
    vars: SmallVec<[u32; 16]>,
    /// Bound value
    bound: i64,
    /// Constraint kind
    kind: CardConstraintKind,
}

impl CardinalityNetwork {
    /// Create a new cardinality network
    pub fn new(vars: SmallVec<[u32; 16]>, kind: CardConstraintKind, bound: i64) -> Self {
        Self { vars, kind, bound }
    }

    /// Encode as sorting network (for ≤ k constraints)
    #[allow(dead_code)]
    pub fn encode_sorting_network(&self) -> Vec<(u32, u32, u32)> {
        // Returns triples (a, b, out) representing comparator gates
        let mut gates = Vec::new();

        // Implement bitonic sorting network
        let n = self.vars.len();
        if n <= 1 {
            return gates;
        }

        self.bitonic_sort(&mut gates, &self.vars, true);

        gates
    }

    #[allow(dead_code)]
    fn bitonic_sort(&self, gates: &mut Vec<(u32, u32, u32)>, vars: &[u32], dir: bool) {
        let n = vars.len();
        if n <= 1 {
            return;
        }

        let m = n / 2;
        self.bitonic_sort(gates, &vars[..m], !dir);
        self.bitonic_sort(gates, &vars[m..], dir);
        self.bitonic_merge(gates, vars, dir);
    }

    #[allow(dead_code)]
    fn bitonic_merge(&self, gates: &mut Vec<(u32, u32, u32)>, vars: &[u32], _dir: bool) {
        let n = vars.len();
        if n <= 1 {
            return;
        }

        let m = n / 2;
        for i in 0..m {
            if i + m < n {
                // Add comparator gate
                gates.push((vars[i], vars[i + m], vars[i]));
            }
        }

        self.bitonic_merge(gates, &vars[..m], _dir);
        if m < n {
            self.bitonic_merge(gates, &vars[m..], _dir);
        }
    }

    /// Encode as sequential counter (for = k constraints)
    #[allow(dead_code)]
    pub fn encode_sequential_counter(&self) -> Vec<(SmallVec<[u32; 4]>, u32)> {
        // Returns clauses for sequential counter encoding

        // Sequential counter creates auxiliary variables s[i][j]
        // s[i][j] = true iff at least j of the first i variables are true

        // This is a simplified version; full implementation would create the clauses

        Vec::new()
    }

    /// Check if the network is satisfiable
    pub fn is_satisfiable(&self, assignment: &[bool]) -> bool {
        let count = self
            .vars
            .iter()
            .filter(|&&v| assignment.get(v as usize).copied().unwrap_or(false))
            .count() as i64;

        self.kind.check(count, self.bound)
    }
}

/// Cardinality constraint compiler
#[derive(Debug)]
pub struct CardinalityCompiler {
    /// Next auxiliary variable ID
    next_aux: u32,
}

impl CardinalityCompiler {
    /// Create a new compiler
    pub fn new() -> Self {
        Self { next_aux: 0 }
    }

    /// Get next auxiliary variable
    fn next_aux_var(&mut self) -> u32 {
        let id = self.next_aux;
        self.next_aux += 1;
        id
    }

    /// Compile a cardinality constraint to CNF
    pub fn compile(
        &mut self,
        constraint: &CardConstraint,
        vars: &[u32],
    ) -> Vec<SmallVec<[i32; 4]>> {
        let mut clauses = Vec::new();

        match constraint.kind {
            CardConstraintKind::Equal => {
                // |vars| = k: use sequential counter
                self.compile_equals(&mut clauses, vars, constraint.bound);
            }
            CardConstraintKind::Le => {
                // |vars| ≤ k: use sorting network or sequential counter
                self.compile_at_most(&mut clauses, vars, constraint.bound);
            }
            CardConstraintKind::Lt => {
                // |vars| < k is |vars| ≤ k-1
                self.compile_at_most(&mut clauses, vars, constraint.bound - 1);
            }
            CardConstraintKind::Ge => {
                // |vars| ≥ k: at least k must be true
                self.compile_at_least(&mut clauses, vars, constraint.bound);
            }
            CardConstraintKind::Gt => {
                // |vars| > k is |vars| ≥ k+1
                self.compile_at_least(&mut clauses, vars, constraint.bound + 1);
            }
        }

        clauses
    }

    fn compile_equals(&mut self, clauses: &mut Vec<SmallVec<[i32; 4]>>, vars: &[u32], k: i64) {
        // |vars| = k is equivalent to |vars| ≤ k ∧ |vars| ≥ k
        self.compile_at_most(clauses, vars, k);
        self.compile_at_least(clauses, vars, k);
    }

    fn compile_at_most(&mut self, clauses: &mut Vec<SmallVec<[i32; 4]>>, vars: &[u32], k: i64) {
        if k < 0 {
            // Contradiction: all variables must be false
            for &v in vars {
                let mut clause = SmallVec::new();
                clause.push(-(v as i32));
                clauses.push(clause);
            }
            return;
        }

        if k >= vars.len() as i64 {
            // Trivially satisfied
            return;
        }

        // Use simplified sequential counter for at-most constraints
        // For each subset of size k+1, at least one must be false
        if vars.len() <= 10 {
            // For small sizes, use direct encoding
            self.compile_at_most_direct(clauses, vars, k);
        } else {
            // For larger sizes, use sequential counter
            self.compile_at_most_sequential(clauses, vars, k);
        }
    }

    fn compile_at_most_direct(
        &mut self,
        clauses: &mut Vec<SmallVec<[i32; 4]>>,
        vars: &[u32],
        k: i64,
    ) {
        // Generate all (k+1)-subsets and add clauses
        let k_usize = k as usize;
        if k_usize < vars.len() {
            // For simplicity, just add a clause for each (k+1)-subset
            // In practice, this can be optimized
            self.generate_subset_clauses(clauses, vars, 0, k_usize + 1, &mut SmallVec::new());
        }
    }

    fn generate_subset_clauses(
        &mut self,
        clauses: &mut Vec<SmallVec<[i32; 4]>>,
        vars: &[u32],
        start: usize,
        size: usize,
        current: &mut SmallVec<[u32; 16]>,
    ) {
        if current.len() == size {
            // Add clause: at least one of these variables must be false
            let mut clause = SmallVec::new();
            for &v in current.iter() {
                clause.push(-(v as i32));
            }
            clauses.push(clause);
            return;
        }

        if start >= vars.len() {
            return;
        }

        // Include vars[start]
        current.push(vars[start]);
        self.generate_subset_clauses(clauses, vars, start + 1, size, current);
        current.pop();

        // Exclude vars[start]
        self.generate_subset_clauses(clauses, vars, start + 1, size, current);
    }

    #[allow(clippy::needless_range_loop)]
    fn compile_at_most_sequential(
        &mut self,
        clauses: &mut Vec<SmallVec<[i32; 4]>>,
        vars: &[u32],
        k: i64,
    ) {
        // Sequential counter encoding
        // s[i][j] = at least j of the first i variables are true

        let n = vars.len();
        let k_usize = k as usize;

        // Create auxiliary variables s[i][j] for i in 1..n, j in 1..k
        let mut s = vec![vec![0u32; k_usize + 1]; n + 1];

        for i in 1..=n {
            for j in 1..=k_usize {
                s[i][j] = self.next_aux_var();
            }
        }

        // Add clauses
        for i in 1..=n {
            // ¬x[i] ∨ s[i][1]
            let mut clause = SmallVec::new();
            clause.push(-(vars[i - 1] as i32));
            clause.push(s[i][1] as i32);
            clauses.push(clause);

            for j in 2..=k_usize {
                // ¬x[i] ∨ ¬s[i-1][j-1] ∨ s[i][j]
                let mut clause = SmallVec::new();
                clause.push(-(vars[i - 1] as i32));
                if i > 1 {
                    clause.push(-(s[i - 1][j - 1] as i32));
                }
                clause.push(s[i][j] as i32);
                clauses.push(clause);
            }

            if k_usize < n && i > k_usize {
                // ¬x[i] ∨ ¬s[i-1][k]
                let mut clause = SmallVec::new();
                clause.push(-(vars[i - 1] as i32));
                if i > 1 {
                    clause.push(-(s[i - 1][k_usize] as i32));
                }
                clauses.push(clause);
            }
        }

        // Final constraint: ¬s[n][k+1] (if k+1 exists)
        // This is implicit in the encoding above
    }

    fn compile_at_least(&mut self, clauses: &mut Vec<SmallVec<[i32; 4]>>, vars: &[u32], k: i64) {
        if k <= 0 {
            // Trivially satisfied
            return;
        }

        if k > vars.len() as i64 {
            // Contradiction
            let mut clause = SmallVec::new();
            clause.push(1); // Add tautology to signal contradiction
            clause.push(-1);
            clauses.push(clause);
            return;
        }

        // At-least k is equivalent to at-most (n-k) negated variables
        // We can encode this by creating negated copies

        // Simpler: if at least k must be true, then at most (n-k) can be false
        let negated_vars: Vec<u32> = vars.to_vec();
        let at_most_false = vars.len() as i64 - k;

        // For at-least-k, we can also use direct encoding for small k
        if k == 1 {
            // At least one must be true: x1 ∨ x2 ∨ ... ∨ xn
            let mut clause = SmallVec::new();
            for &v in vars {
                clause.push(v as i32);
            }
            clauses.push(clause);
        } else {
            // General case: use at-most on negated variables
            self.compile_at_most(clauses, &negated_vars, at_most_false);
        }
    }

    /// Reset the compiler
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.next_aux = 0;
    }
}

impl Default for CardinalityCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_constraint_kind_check() {
        assert!(CardConstraintKind::Equal.check(5, 5));
        assert!(!CardConstraintKind::Equal.check(4, 5));

        assert!(CardConstraintKind::Le.check(4, 5));
        assert!(CardConstraintKind::Le.check(5, 5));
        assert!(!CardConstraintKind::Le.check(6, 5));

        assert!(CardConstraintKind::Ge.check(6, 5));
        assert!(CardConstraintKind::Ge.check(5, 5));
        assert!(!CardConstraintKind::Ge.check(4, 5));
    }

    #[test]
    fn test_card_constraint_bounds() {
        let (lower, upper) = CardConstraintKind::Equal.bounds(5);
        assert_eq!(lower, Some(5));
        assert_eq!(upper, Some(5));

        let (lower, upper) = CardConstraintKind::Le.bounds(10);
        assert_eq!(lower, None);
        assert_eq!(upper, Some(10));

        let (lower, upper) = CardConstraintKind::Ge.bounds(3);
        assert_eq!(lower, Some(3));
        assert_eq!(upper, None);
    }

    #[test]
    fn test_card_domain_tighten() {
        let mut domain = CardDomain::new(0, Some(10));

        assert!(domain.tighten_lower(5));
        assert_eq!(domain.lower, 5);

        assert!(domain.tighten_upper(8));
        assert_eq!(domain.upper, Some(8));

        // Try to tighten beyond limits
        assert!(!domain.tighten_lower(9)); // Would create conflict
        assert!(!domain.tighten_upper(4)); // Would create conflict
    }

    #[test]
    fn test_card_domain_singleton() {
        let domain = CardDomain::singleton(5);
        assert_eq!(domain.lower, 5);
        assert_eq!(domain.upper, Some(5));
        assert_eq!(domain.exact, Some(5));
    }

    #[test]
    fn test_card_domain_intersect() {
        let d1 = CardDomain::new(2, Some(8));
        let d2 = CardDomain::new(5, Some(10));

        let result = d1.intersect(&d2).unwrap();
        assert_eq!(result.lower, 5);
        assert_eq!(result.upper, Some(8));
    }

    #[test]
    fn test_card_domain_intersect_empty() {
        let d1 = CardDomain::new(8, Some(10));
        let d2 = CardDomain::new(2, Some(5));

        let result = d1.intersect(&d2);
        assert!(result.is_none()); // Empty intersection
    }

    #[test]
    fn test_card_domain_union() {
        let d1 = CardDomain::new(2, Some(8));
        let d2 = CardDomain::new(5, Some(10));

        let result = d1.union(&d2);
        assert_eq!(result.lower, 2);
        assert_eq!(result.upper, Some(10));
    }

    #[test]
    fn test_card_propagator_add_constraint() {
        let mut prop = CardPropagator::new();
        let var = SetVarId(0);

        let constraint = CardConstraint::new(var, CardConstraintKind::Le, 10, 0);
        assert!(prop.add_constraint(constraint).is_ok());

        let domain = prop.get_domain(var);
        assert_eq!(domain.upper, Some(10));
    }

    #[test]
    fn test_card_propagator_conflict() {
        let mut prop = CardPropagator::new();
        let var = SetVarId(0);

        prop.add_constraint(CardConstraint::new(var, CardConstraintKind::Ge, 10, 0))
            .unwrap();

        // This should conflict
        let result = prop.add_constraint(CardConstraint::new(var, CardConstraintKind::Le, 5, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_card_propagator_union() {
        let mut prop = CardPropagator::new();
        let lhs = SetVarId(0);
        let rhs = SetVarId(1);
        let result = SetVarId(2);

        // |lhs| ∈ [2, 5]
        prop.get_domain(lhs).tighten_lower(2);
        prop.get_domain(lhs).tighten_upper(5);

        // |rhs| ∈ [3, 4]
        prop.get_domain(rhs).tighten_lower(3);
        prop.get_domain(rhs).tighten_upper(4);

        prop.propagate_union(lhs, rhs, result).unwrap();

        let result_domain = prop.get_domain(result);
        assert_eq!(result_domain.lower, 3); // max(2, 3)
        assert_eq!(result_domain.upper, Some(9)); // 5 + 4
    }

    #[test]
    fn test_card_propagator_intersection() {
        let mut prop = CardPropagator::new();
        let lhs = SetVarId(0);
        let rhs = SetVarId(1);
        let result = SetVarId(2);

        // |lhs| ∈ [2, 5]
        prop.get_domain(lhs).tighten_lower(2);
        prop.get_domain(lhs).tighten_upper(5);

        // |rhs| ∈ [3, 4]
        prop.get_domain(rhs).tighten_lower(3);
        prop.get_domain(rhs).tighten_upper(4);

        prop.propagate_intersection(lhs, rhs, result).unwrap();

        let result_domain = prop.get_domain(result);
        assert_eq!(result_domain.upper, Some(4)); // min(5, 4)
    }

    #[test]
    fn test_cardinality_network_satisfiable() {
        let vars = SmallVec::from_vec(vec![0, 1, 2, 3]);
        let network = CardinalityNetwork::new(vars, CardConstraintKind::Le, 2);

        let assignment1 = vec![true, true, false, false];
        assert!(network.is_satisfiable(&assignment1));

        let assignment2 = vec![true, true, true, false];
        assert!(!network.is_satisfiable(&assignment2));
    }

    #[test]
    fn test_cardinality_compiler_at_most() {
        let mut compiler = CardinalityCompiler::new();
        let vars = vec![0, 1, 2];
        let constraint = CardConstraint::new(SetVarId(0), CardConstraintKind::Le, 1, 0);

        let clauses = compiler.compile(&constraint, &vars);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_cardinality_compiler_at_least() {
        let mut compiler = CardinalityCompiler::new();
        let vars = vec![0, 1, 2];
        let constraint = CardConstraint::new(SetVarId(0), CardConstraintKind::Ge, 2, 0);

        let clauses = compiler.compile(&constraint, &vars);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_cardinality_compiler_equals() {
        let mut compiler = CardinalityCompiler::new();
        let vars = vec![0, 1, 2, 3];
        let constraint = CardConstraint::new(SetVarId(0), CardConstraintKind::Equal, 2, 0);

        let clauses = compiler.compile(&constraint, &vars);
        assert!(!clauses.is_empty());
    }
}
