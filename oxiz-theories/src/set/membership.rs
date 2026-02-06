//! Membership Constraint Handling
//!
//! Handles membership constraints (x ∈ S) and propagation

#![allow(dead_code)]

use super::{SetConflict, SetLiteral, SetProofStep, SetVar, SetVarId};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Membership variable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemberVar {
    /// Element identifier
    pub element: u32,
    /// Set variable
    pub set: SetVarId,
}

impl MemberVar {
    /// Create a new membership variable
    pub fn new(element: u32, set: SetVarId) -> Self {
        Self { element, set }
    }
}

/// Membership domain (whether element is in set)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberDomain {
    /// Definitely in the set
    True,
    /// Definitely not in the set
    False,
    /// Unknown
    Unknown,
}

impl MemberDomain {
    /// Check if this domain is determined
    pub fn is_determined(&self) -> bool {
        matches!(self, MemberDomain::True | MemberDomain::False)
    }

    /// Get the boolean value if determined
    pub fn value(&self) -> Option<bool> {
        match self {
            MemberDomain::True => Some(true),
            MemberDomain::False => Some(false),
            MemberDomain::Unknown => None,
        }
    }

    /// Intersect with another domain
    pub fn intersect(&self, other: &MemberDomain) -> Option<MemberDomain> {
        match (self, other) {
            (MemberDomain::True, MemberDomain::False) => None,
            (MemberDomain::False, MemberDomain::True) => None,
            (MemberDomain::True, _) => Some(MemberDomain::True),
            (_, MemberDomain::True) => Some(MemberDomain::True),
            (MemberDomain::False, _) => Some(MemberDomain::False),
            (_, MemberDomain::False) => Some(MemberDomain::False),
            _ => Some(MemberDomain::Unknown),
        }
    }
}

/// Membership constraint
#[derive(Debug, Clone)]
pub struct MemberConstraint {
    /// Element
    pub element: u32,
    /// Set variable
    pub set: SetVarId,
    /// Is positive (true = ∈, false = ∉)
    pub sign: bool,
    /// Decision level when added
    pub level: usize,
}

impl MemberConstraint {
    /// Create a new membership constraint
    pub fn new(element: u32, set: SetVarId, sign: bool, level: usize) -> Self {
        Self {
            element,
            set,
            sign,
            level,
        }
    }

    /// Check if this constraint is satisfied by a variable
    pub fn is_satisfied(&self, var: &SetVar) -> Option<bool> {
        match var.contains(self.element) {
            Some(true) => Some(self.sign),
            Some(false) => Some(!self.sign),
            None => None,
        }
    }
}

/// Membership result
pub type MemberResult<T> = Result<T, SetConflict>;

/// Membership statistics
#[derive(Debug, Clone, Default)]
pub struct MemberStats {
    /// Number of membership constraints
    pub num_constraints: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of conflicts
    pub num_conflicts: usize,
    /// Number of fixed memberships
    pub num_fixed: usize,
}

/// Membership propagator
#[derive(Debug)]
pub struct MemberPropagator {
    /// Membership domains
    domains: FxHashMap<MemberVar, MemberDomain>,
    /// Elements to sets mapping (for propagation)
    element_to_sets: FxHashMap<u32, SmallVec<[SetVarId; 4]>>,
    /// Statistics
    stats: MemberStats,
    /// Watch lists for membership propagation
    watch_lists: FxHashMap<SetVarId, SmallVec<[u32; 16]>>,
}

impl MemberPropagator {
    /// Create a new membership propagator
    pub fn new() -> Self {
        Self {
            domains: FxHashMap::default(),
            element_to_sets: FxHashMap::default(),
            stats: MemberStats::default(),
            watch_lists: FxHashMap::default(),
        }
    }

    /// Get or create a domain for a membership variable
    pub fn get_domain(&mut self, var: MemberVar) -> MemberDomain {
        *self.domains.get(&var).unwrap_or(&MemberDomain::Unknown)
    }

    /// Set the domain for a membership variable
    pub fn set_domain(&mut self, var: MemberVar, domain: MemberDomain) -> MemberResult<()> {
        let current = self.get_domain(var);

        if let Some(result) = current.intersect(&domain) {
            if result.is_determined() && current != result {
                self.stats.num_fixed += 1;
            }
            self.domains.insert(var, result);
            Ok(())
        } else {
            self.stats.num_conflicts += 1;
            Err(SetConflict {
                literals: vec![
                    SetLiteral::Member {
                        element: var.element,
                        set: var.set,
                        sign: true,
                    },
                    SetLiteral::Member {
                        element: var.element,
                        set: var.set,
                        sign: false,
                    },
                ],
                reason: format!(
                    "Membership conflict: element {} cannot be both in and not in set",
                    var.element
                ),
                proof_steps: vec![SetProofStep::Assume(SetLiteral::Member {
                    element: var.element,
                    set: var.set,
                    sign: true,
                })],
            })
        }
    }

    /// Add a membership constraint
    pub fn add_constraint(&mut self, constraint: MemberConstraint) -> MemberResult<()> {
        self.stats.num_constraints += 1;

        let var = MemberVar::new(constraint.element, constraint.set);
        let domain = if constraint.sign {
            MemberDomain::True
        } else {
            MemberDomain::False
        };

        self.set_domain(var, domain)?;

        // Update watch lists
        self.element_to_sets
            .entry(constraint.element)
            .or_default()
            .push(constraint.set);

        self.watch_lists
            .entry(constraint.set)
            .or_default()
            .push(constraint.element);

        Ok(())
    }

    /// Propagate membership constraints for a variable
    pub fn propagate(&mut self, var: SetVarId, vars: &mut [SetVar]) -> MemberResult<()> {
        if let Some(set_var) = vars.get(var.id() as usize) {
            // Propagate must_members
            for &elem in &set_var.must_members.clone() {
                let member_var = MemberVar::new(elem, var);
                self.set_domain(member_var, MemberDomain::True)?;
                self.stats.num_propagations += 1;
            }

            // Propagate must_not_members
            for &elem in &set_var.must_not_members.clone() {
                let member_var = MemberVar::new(elem, var);
                self.set_domain(member_var, MemberDomain::False)?;
                self.stats.num_propagations += 1;
            }

            // Backward propagation: update set_var from domains
            for (&member_var, &domain) in &self.domains {
                if member_var.set == var
                    && let Some(set_var_mut) = vars.get_mut(var.id() as usize)
                {
                    match domain {
                        MemberDomain::True => {
                            if !set_var_mut.add_must_member(member_var.element) {
                                return Err(SetConflict {
                                    literals: vec![],
                                    reason: format!(
                                        "Membership propagation conflict: element {} already excluded",
                                        member_var.element
                                    ),
                                    proof_steps: vec![],
                                });
                            }
                        }
                        MemberDomain::False => {
                            if !set_var_mut.add_must_not_member(member_var.element) {
                                return Err(SetConflict {
                                    literals: vec![],
                                    reason: format!(
                                        "Membership propagation conflict: element {} already included",
                                        member_var.element
                                    ),
                                    proof_steps: vec![],
                                });
                            }
                        }
                        MemberDomain::Unknown => {}
                    }
                }
            }
        }

        Ok(())
    }

    /// Propagate subset membership: if S1 ⊆ S2 and x ∈ S1, then x ∈ S2
    pub fn propagate_subset(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        vars: &[SetVar],
    ) -> MemberResult<()> {
        if let Some(lhs_var) = vars.get(lhs.id() as usize) {
            for &elem in &lhs_var.must_members {
                let rhs_member = MemberVar::new(elem, rhs);
                self.set_domain(rhs_member, MemberDomain::True)?;
                self.stats.num_propagations += 1;
            }
        }

        if let Some(rhs_var) = vars.get(rhs.id() as usize) {
            for &elem in &rhs_var.must_not_members {
                let lhs_member = MemberVar::new(elem, lhs);
                self.set_domain(lhs_member, MemberDomain::False)?;
                self.stats.num_propagations += 1;
            }
        }

        Ok(())
    }

    /// Propagate union membership: x ∈ (S1 ∪ S2) ⟺ x ∈ S1 ∨ x ∈ S2
    pub fn propagate_union(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
        vars: &[SetVar],
    ) -> MemberResult<()> {
        // Collect elements to check from both vars and domains
        let mut elements = FxHashSet::default();

        if let Some(lhs_var) = vars.get(lhs.id() as usize) {
            elements.extend(&lhs_var.must_members);
            elements.extend(&lhs_var.must_not_members);
        }

        if let Some(rhs_var) = vars.get(rhs.id() as usize) {
            elements.extend(&rhs_var.must_members);
            elements.extend(&rhs_var.must_not_members);
        }

        if let Some(result_var) = vars.get(result.id() as usize) {
            elements.extend(&result_var.must_members);
            elements.extend(&result_var.must_not_members);
        }

        // Also collect elements from the domain map for these sets
        for member_var in self.domains.keys() {
            if member_var.set == lhs || member_var.set == rhs || member_var.set == result {
                elements.insert(member_var.element);
            }
        }

        for &elem in &elements {
            let lhs_member = MemberVar::new(elem, lhs);
            let rhs_member = MemberVar::new(elem, rhs);
            let result_member = MemberVar::new(elem, result);

            let lhs_domain = self.get_domain(lhs_member);
            let rhs_domain = self.get_domain(rhs_member);
            let result_domain = self.get_domain(result_member);

            // x ∈ lhs ⟹ x ∈ result
            if lhs_domain == MemberDomain::True {
                self.set_domain(result_member, MemberDomain::True)?;
            }

            // x ∈ rhs ⟹ x ∈ result
            if rhs_domain == MemberDomain::True {
                self.set_domain(result_member, MemberDomain::True)?;
            }

            // x ∉ lhs ∧ x ∉ rhs ⟹ x ∉ result
            if lhs_domain == MemberDomain::False && rhs_domain == MemberDomain::False {
                self.set_domain(result_member, MemberDomain::False)?;
            }

            // x ∉ result ⟹ x ∉ lhs ∧ x ∉ rhs
            if result_domain == MemberDomain::False {
                self.set_domain(lhs_member, MemberDomain::False)?;
                self.set_domain(rhs_member, MemberDomain::False)?;
            }
        }

        Ok(())
    }

    /// Propagate intersection membership: x ∈ (S1 ∩ S2) ⟺ x ∈ S1 ∧ x ∈ S2
    pub fn propagate_intersection(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
        vars: &[SetVar],
    ) -> MemberResult<()> {
        // Collect elements to check from both vars and domains
        let mut elements = FxHashSet::default();

        if let Some(lhs_var) = vars.get(lhs.id() as usize) {
            elements.extend(&lhs_var.must_members);
            elements.extend(&lhs_var.must_not_members);
        }

        if let Some(rhs_var) = vars.get(rhs.id() as usize) {
            elements.extend(&rhs_var.must_members);
            elements.extend(&rhs_var.must_not_members);
        }

        if let Some(result_var) = vars.get(result.id() as usize) {
            elements.extend(&result_var.must_members);
            elements.extend(&result_var.must_not_members);
        }

        // Also collect elements from the domain map for these sets
        for member_var in self.domains.keys() {
            if member_var.set == lhs || member_var.set == rhs || member_var.set == result {
                elements.insert(member_var.element);
            }
        }

        for &elem in &elements {
            let lhs_member = MemberVar::new(elem, lhs);
            let rhs_member = MemberVar::new(elem, rhs);
            let result_member = MemberVar::new(elem, result);

            let lhs_domain = self.get_domain(lhs_member);
            let rhs_domain = self.get_domain(rhs_member);
            let result_domain = self.get_domain(result_member);

            // x ∈ lhs ∧ x ∈ rhs ⟹ x ∈ result
            if lhs_domain == MemberDomain::True && rhs_domain == MemberDomain::True {
                self.set_domain(result_member, MemberDomain::True)?;
            }

            // x ∉ lhs ⟹ x ∉ result
            if lhs_domain == MemberDomain::False {
                self.set_domain(result_member, MemberDomain::False)?;
            }

            // x ∉ rhs ⟹ x ∉ result
            if rhs_domain == MemberDomain::False {
                self.set_domain(result_member, MemberDomain::False)?;
            }

            // x ∈ result ⟹ x ∈ lhs ∧ x ∈ rhs
            if result_domain == MemberDomain::True {
                self.set_domain(lhs_member, MemberDomain::True)?;
                self.set_domain(rhs_member, MemberDomain::True)?;
            }
        }

        Ok(())
    }

    /// Propagate difference membership: x ∈ (S1 \ S2) ⟺ x ∈ S1 ∧ x ∉ S2
    pub fn propagate_difference(
        &mut self,
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
        vars: &[SetVar],
    ) -> MemberResult<()> {
        // Collect elements to check from both vars and domains
        let mut elements = FxHashSet::default();

        if let Some(lhs_var) = vars.get(lhs.id() as usize) {
            elements.extend(&lhs_var.must_members);
            elements.extend(&lhs_var.must_not_members);
        }

        if let Some(rhs_var) = vars.get(rhs.id() as usize) {
            elements.extend(&rhs_var.must_members);
            elements.extend(&rhs_var.must_not_members);
        }

        if let Some(result_var) = vars.get(result.id() as usize) {
            elements.extend(&result_var.must_members);
            elements.extend(&result_var.must_not_members);
        }

        // Also collect elements from the domain map for these sets
        for member_var in self.domains.keys() {
            if member_var.set == lhs || member_var.set == rhs || member_var.set == result {
                elements.insert(member_var.element);
            }
        }

        for &elem in &elements {
            let lhs_member = MemberVar::new(elem, lhs);
            let rhs_member = MemberVar::new(elem, rhs);
            let result_member = MemberVar::new(elem, result);

            let lhs_domain = self.get_domain(lhs_member);
            let rhs_domain = self.get_domain(rhs_member);
            let result_domain = self.get_domain(result_member);

            // x ∈ lhs ∧ x ∉ rhs ⟹ x ∈ result
            if lhs_domain == MemberDomain::True && rhs_domain == MemberDomain::False {
                self.set_domain(result_member, MemberDomain::True)?;
            }

            // x ∉ lhs ⟹ x ∉ result
            if lhs_domain == MemberDomain::False {
                self.set_domain(result_member, MemberDomain::False)?;
            }

            // x ∈ rhs ⟹ x ∉ result
            if rhs_domain == MemberDomain::True {
                self.set_domain(result_member, MemberDomain::False)?;
            }

            // x ∈ result ⟹ x ∈ lhs
            if result_domain == MemberDomain::True {
                self.set_domain(lhs_member, MemberDomain::True)?;
                self.set_domain(rhs_member, MemberDomain::False)?;
            }
        }

        Ok(())
    }

    /// Propagate complement membership: x ∈ ¬S ⟺ x ∉ S
    pub fn propagate_complement(
        &mut self,
        set: SetVarId,
        result: SetVarId,
        universe: Option<&FxHashSet<u32>>,
        vars: &[SetVar],
    ) -> MemberResult<()> {
        let mut elements = FxHashSet::default();

        if let Some(set_var) = vars.get(set.id() as usize) {
            elements.extend(&set_var.must_members);
            elements.extend(&set_var.must_not_members);
        }

        if let Some(result_var) = vars.get(result.id() as usize) {
            elements.extend(&result_var.must_members);
            elements.extend(&result_var.must_not_members);
        }

        if let Some(univ) = universe {
            elements.extend(univ);
        }

        for &elem in &elements {
            let set_member = MemberVar::new(elem, set);
            let result_member = MemberVar::new(elem, result);

            let set_domain = self.get_domain(set_member);
            let result_domain = self.get_domain(result_member);

            // x ∈ set ⟹ x ∉ result
            if set_domain == MemberDomain::True {
                self.set_domain(result_member, MemberDomain::False)?;
            }

            // x ∉ set ⟹ x ∈ result (only if in universe)
            if set_domain == MemberDomain::False && universe.is_none_or(|u| u.contains(&elem)) {
                self.set_domain(result_member, MemberDomain::True)?;
            }

            // x ∈ result ⟹ x ∉ set
            if result_domain == MemberDomain::True {
                self.set_domain(set_member, MemberDomain::False)?;
            }

            // x ∉ result ⟹ x ∈ set (only if in universe)
            if result_domain == MemberDomain::False && universe.is_none_or(|u| u.contains(&elem)) {
                self.set_domain(set_member, MemberDomain::True)?;
            }
        }

        Ok(())
    }

    /// Get all elements that must be in a set
    pub fn get_must_members(&self, set: SetVarId) -> FxHashSet<u32> {
        self.domains
            .iter()
            .filter_map(|(var, domain)| {
                if var.set == set && *domain == MemberDomain::True {
                    Some(var.element)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all elements that must not be in a set
    pub fn get_must_not_members(&self, set: SetVarId) -> FxHashSet<u32> {
        self.domains
            .iter()
            .filter_map(|(var, domain)| {
                if var.set == set && *domain == MemberDomain::False {
                    Some(var.element)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &MemberStats {
        &self.stats
    }

    /// Reset the propagator
    pub fn reset(&mut self) {
        self.domains.clear();
        self.element_to_sets.clear();
        self.watch_lists.clear();
        self.stats = MemberStats::default();
    }
}

impl Default for MemberPropagator {
    fn default() -> Self {
        Self::new()
    }
}

/// Membership inference engine
#[derive(Debug)]
pub struct MembershipInference {
    /// Known memberships
    known: FxHashMap<MemberVar, bool>,
    /// Inference rules
    rules: Vec<InferenceRule>,
}

impl MembershipInference {
    /// Create a new inference engine
    pub fn new() -> Self {
        Self {
            known: FxHashMap::default(),
            rules: Vec::new(),
        }
    }

    /// Add a known membership fact
    pub fn add_fact(&mut self, element: u32, set: SetVarId, member: bool) {
        let var = MemberVar::new(element, set);
        self.known.insert(var, member);
    }

    /// Add an inference rule
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    /// Run inference to deduce new memberships
    pub fn infer(&mut self) -> FxHashMap<MemberVar, bool> {
        let mut inferred = FxHashMap::default();
        let mut changed = true;

        while changed {
            changed = false;

            for rule in &self.rules {
                if let Some((var, value)) = rule.apply(&self.known)
                    && let std::collections::hash_map::Entry::Vacant(e) = self.known.entry(var)
                {
                    e.insert(value);
                    inferred.insert(var, value);
                    changed = true;
                }
            }
        }

        inferred
    }

    /// Check if a membership is known
    #[allow(dead_code)]
    pub fn is_known(&self, element: u32, set: SetVarId) -> Option<bool> {
        let var = MemberVar::new(element, set);
        self.known.get(&var).copied()
    }

    /// Get all known memberships for a set
    #[allow(dead_code)]
    pub fn get_members(&self, set: SetVarId) -> FxHashSet<u32> {
        self.known
            .iter()
            .filter_map(|(var, &value)| {
                if var.set == set && value {
                    Some(var.element)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all known non-members for a set
    #[allow(dead_code)]
    pub fn get_non_members(&self, set: SetVarId) -> FxHashSet<u32> {
        self.known
            .iter()
            .filter_map(|(var, &value)| {
                if var.set == set && !value {
                    Some(var.element)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for MembershipInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Inference rule for membership
#[derive(Debug, Clone)]
pub enum InferenceRule {
    /// If x ∈ S1 and S1 ⊆ S2, then x ∈ S2
    SubsetTransfer {
        subset: SetVarId,
        superset: SetVarId,
    },
    /// If x ∈ S1 or x ∈ S2, then x ∈ S1 ∪ S2
    UnionIntro {
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
    },
    /// If x ∈ S1 and x ∈ S2, then x ∈ S1 ∩ S2
    IntersectionIntro {
        lhs: SetVarId,
        rhs: SetVarId,
        result: SetVarId,
    },
}

impl InferenceRule {
    /// Apply this rule to known facts
    pub fn apply(&self, known: &FxHashMap<MemberVar, bool>) -> Option<(MemberVar, bool)> {
        match self {
            InferenceRule::SubsetTransfer { subset, superset } => {
                // Find elements in subset and transfer to superset
                for (var, &value) in known {
                    if var.set == *subset && value {
                        let super_var = MemberVar::new(var.element, *superset);
                        if !known.contains_key(&super_var) {
                            return Some((super_var, true));
                        }
                    }
                }
                None
            }
            InferenceRule::UnionIntro { lhs, rhs, result } => {
                // If x ∈ lhs or x ∈ rhs, then x ∈ result
                for (var, &value) in known {
                    if value && (var.set == *lhs || var.set == *rhs) {
                        let result_var = MemberVar::new(var.element, *result);
                        if !known.contains_key(&result_var) {
                            return Some((result_var, true));
                        }
                    }
                }
                None
            }
            InferenceRule::IntersectionIntro { lhs, rhs, result } => {
                // If x ∈ lhs and x ∈ rhs, then x ∈ result
                for (lhs_var, &lhs_value) in known {
                    if lhs_var.set == *lhs && lhs_value {
                        let rhs_var = MemberVar::new(lhs_var.element, *rhs);
                        if let Some(&true) = known.get(&rhs_var) {
                            let result_var = MemberVar::new(lhs_var.element, *result);
                            if !known.contains_key(&result_var) {
                                return Some((result_var, true));
                            }
                        }
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_member_domain() {
        let d1 = MemberDomain::True;
        let d2 = MemberDomain::False;
        let d3 = MemberDomain::Unknown;

        assert!(d1.is_determined());
        assert!(d2.is_determined());
        assert!(!d3.is_determined());

        assert_eq!(d1.value(), Some(true));
        assert_eq!(d2.value(), Some(false));
        assert_eq!(d3.value(), None);
    }

    #[test]
    fn test_member_domain_intersect() {
        assert_eq!(
            MemberDomain::True.intersect(&MemberDomain::True),
            Some(MemberDomain::True)
        );
        assert_eq!(
            MemberDomain::True.intersect(&MemberDomain::Unknown),
            Some(MemberDomain::True)
        );
        assert_eq!(MemberDomain::True.intersect(&MemberDomain::False), None);

        assert_eq!(
            MemberDomain::Unknown.intersect(&MemberDomain::Unknown),
            Some(MemberDomain::Unknown)
        );
    }

    #[test]
    fn test_member_propagator_add_constraint() {
        let mut prop = MemberPropagator::new();

        let constraint = MemberConstraint::new(42, SetVarId(0), true, 0);
        assert!(prop.add_constraint(constraint).is_ok());

        let var = MemberVar::new(42, SetVarId(0));
        assert_eq!(prop.get_domain(var), MemberDomain::True);
    }

    #[test]
    fn test_member_propagator_conflict() {
        let mut prop = MemberPropagator::new();

        prop.add_constraint(MemberConstraint::new(42, SetVarId(0), true, 0))
            .unwrap();

        let result = prop.add_constraint(MemberConstraint::new(42, SetVarId(0), false, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_member_propagator_union() {
        let mut prop = MemberPropagator::new();
        let lhs = SetVarId(0);
        let rhs = SetVarId(1);
        let result = SetVarId(2);

        // 42 ∈ lhs
        prop.add_constraint(MemberConstraint::new(42, lhs, true, 0))
            .unwrap();

        let vars = vec![
            SetVar::new(lhs, "lhs".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(rhs, "rhs".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(
                result,
                "result".to_string(),
                super::super::SetSort::IntSet,
                0,
            ),
        ];

        prop.propagate_union(lhs, rhs, result, &vars).unwrap();

        // 42 should be in result
        let result_var = MemberVar::new(42, result);
        assert_eq!(prop.get_domain(result_var), MemberDomain::True);
    }

    #[test]
    fn test_member_propagator_intersection() {
        let mut prop = MemberPropagator::new();
        let lhs = SetVarId(0);
        let rhs = SetVarId(1);
        let result = SetVarId(2);

        // 42 ∈ lhs
        prop.add_constraint(MemberConstraint::new(42, lhs, true, 0))
            .unwrap();
        // 42 ∈ rhs
        prop.add_constraint(MemberConstraint::new(42, rhs, true, 0))
            .unwrap();

        let vars = vec![
            SetVar::new(lhs, "lhs".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(rhs, "rhs".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(
                result,
                "result".to_string(),
                super::super::SetSort::IntSet,
                0,
            ),
        ];

        prop.propagate_intersection(lhs, rhs, result, &vars)
            .unwrap();

        // 42 should be in result
        let result_var = MemberVar::new(42, result);
        assert_eq!(prop.get_domain(result_var), MemberDomain::True);
    }

    #[test]
    fn test_member_propagator_difference() {
        let mut prop = MemberPropagator::new();
        let lhs = SetVarId(0);
        let rhs = SetVarId(1);
        let result = SetVarId(2);

        // 42 ∈ lhs
        prop.add_constraint(MemberConstraint::new(42, lhs, true, 0))
            .unwrap();
        // 42 ∉ rhs
        prop.add_constraint(MemberConstraint::new(42, rhs, false, 0))
            .unwrap();

        let vars = vec![
            SetVar::new(lhs, "lhs".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(rhs, "rhs".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(
                result,
                "result".to_string(),
                super::super::SetSort::IntSet,
                0,
            ),
        ];

        prop.propagate_difference(lhs, rhs, result, &vars).unwrap();

        // 42 should be in result
        let result_var = MemberVar::new(42, result);
        assert_eq!(prop.get_domain(result_var), MemberDomain::True);
    }

    #[test]
    fn test_member_propagator_complement() {
        let mut prop = MemberPropagator::new();
        let set = SetVarId(0);
        let result = SetVarId(1);

        // 42 ∈ set
        prop.add_constraint(MemberConstraint::new(42, set, true, 0))
            .unwrap();

        let mut universe = FxHashSet::default();
        universe.insert(42);
        universe.insert(43);

        let vars = vec![
            SetVar::new(set, "set".to_string(), super::super::SetSort::IntSet, 0),
            SetVar::new(
                result,
                "result".to_string(),
                super::super::SetSort::IntSet,
                0,
            ),
        ];

        prop.propagate_complement(set, result, Some(&universe), &vars)
            .unwrap();

        // 42 should not be in result
        let result_var_42 = MemberVar::new(42, result);
        assert_eq!(prop.get_domain(result_var_42), MemberDomain::False);
    }

    #[test]
    fn test_membership_inference() {
        let mut inference = MembershipInference::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        inference.add_fact(42, s1, true);
        inference.add_rule(InferenceRule::SubsetTransfer {
            subset: s1,
            superset: s2,
        });

        let inferred = inference.infer();

        // 42 should be inferred to be in s2
        assert!(inferred.contains_key(&MemberVar::new(42, s2)));
        assert_eq!(inferred.get(&MemberVar::new(42, s2)), Some(&true));
    }

    #[test]
    fn test_membership_inference_union() {
        let mut inference = MembershipInference::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        inference.add_fact(42, s1, true);
        inference.add_rule(InferenceRule::UnionIntro {
            lhs: s1,
            rhs: s2,
            result: s3,
        });

        let inferred = inference.infer();

        // 42 should be inferred to be in s3
        assert!(inferred.contains_key(&MemberVar::new(42, s3)));
    }

    #[test]
    fn test_membership_inference_intersection() {
        let mut inference = MembershipInference::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        inference.add_fact(42, s1, true);
        inference.add_fact(42, s2, true);
        inference.add_rule(InferenceRule::IntersectionIntro {
            lhs: s1,
            rhs: s2,
            result: s3,
        });

        let inferred = inference.infer();

        // 42 should be inferred to be in s3
        assert!(inferred.contains_key(&MemberVar::new(42, s3)));
    }

    #[test]
    fn test_get_must_members() {
        let mut prop = MemberPropagator::new();
        let set = SetVarId(0);

        prop.add_constraint(MemberConstraint::new(1, set, true, 0))
            .unwrap();
        prop.add_constraint(MemberConstraint::new(2, set, true, 0))
            .unwrap();
        prop.add_constraint(MemberConstraint::new(3, set, false, 0))
            .unwrap();

        let must_members = prop.get_must_members(set);
        assert_eq!(must_members.len(), 2);
        assert!(must_members.contains(&1));
        assert!(must_members.contains(&2));
        assert!(!must_members.contains(&3));
    }

    #[test]
    fn test_get_must_not_members() {
        let mut prop = MemberPropagator::new();
        let set = SetVarId(0);

        prop.add_constraint(MemberConstraint::new(1, set, true, 0))
            .unwrap();
        prop.add_constraint(MemberConstraint::new(2, set, false, 0))
            .unwrap();
        prop.add_constraint(MemberConstraint::new(3, set, false, 0))
            .unwrap();

        let must_not_members = prop.get_must_not_members(set);
        assert_eq!(must_not_members.len(), 2);
        assert!(!must_not_members.contains(&1));
        assert!(must_not_members.contains(&2));
        assert!(must_not_members.contains(&3));
    }
}
