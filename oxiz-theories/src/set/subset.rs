//! Subset Relation Reasoning
//!
//! Handles subset constraints (S1 ⊆ S2) and transitive closure

use super::{SetConflict, SetLiteral, SetProofStep, SetVar, SetVarId};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;

/// Subset constraint
#[derive(Debug, Clone)]
pub struct SubsetConstraint {
    /// Left-hand side (subset)
    pub lhs: SetVarId,
    /// Right-hand side (superset)
    pub rhs: SetVarId,
    /// Is positive (true = ⊆, false = ⊈)
    pub sign: bool,
    /// Decision level when added
    pub level: usize,
}

impl SubsetConstraint {
    /// Create a new subset constraint
    pub fn new(lhs: SetVarId, rhs: SetVarId, sign: bool, level: usize) -> Self {
        Self {
            lhs,
            rhs,
            sign,
            level,
        }
    }

    /// Check if this constraint is satisfied
    pub fn is_satisfied(&self, lhs_var: &SetVar, rhs_var: &SetVar) -> Option<bool> {
        if !self.sign {
            // S1 ⊈ S2: need to find an element in S1 but not in S2
            for &elem in &lhs_var.must_members {
                if rhs_var.must_not_members.contains(&elem) {
                    return Some(true); // Constraint is satisfied (S1 ⊈ S2)
                }
            }
            return None; // Unknown
        }

        // S1 ⊆ S2: all elements in S1 must be in S2
        for &elem in &lhs_var.must_members {
            if rhs_var.must_not_members.contains(&elem) {
                return Some(false); // Violation
            }
        }

        // If lhs is fully determined and all its elements are in rhs
        if let Some(may_members) = &lhs_var.may_members {
            let all_in_rhs = lhs_var
                .must_members
                .iter()
                .all(|e| rhs_var.must_members.contains(e));
            if all_in_rhs
                && may_members
                    .iter()
                    .all(|e| !rhs_var.must_not_members.contains(e))
            {
                return Some(true);
            }
        }

        None // Unknown
    }
}

/// Subset domain (whether S1 ⊆ S2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsetDomain {
    /// Definitely S1 ⊆ S2
    True,
    /// Definitely S1 ⊈ S2
    False,
    /// Unknown
    Unknown,
}

impl SubsetDomain {
    /// Check if this domain is determined
    pub fn is_determined(&self) -> bool {
        matches!(self, SubsetDomain::True | SubsetDomain::False)
    }
}

/// Subset relation graph for transitive closure
#[derive(Debug, Clone)]
pub struct SubsetGraph {
    /// Adjacency list: S1 -> {S2 | S1 ⊆ S2}
    successors: FxHashMap<SetVarId, FxHashSet<SetVarId>>,
    /// Reverse adjacency list: S2 -> {S1 | S1 ⊆ S2}
    predecessors: FxHashMap<SetVarId, FxHashSet<SetVarId>>,
    /// Transitive closure cache
    closure: FxHashMap<SetVarId, FxHashSet<SetVarId>>,
    /// Is the closure up-to-date?
    closure_valid: bool,
}

impl SubsetGraph {
    /// Create a new subset graph
    pub fn new() -> Self {
        Self {
            successors: FxHashMap::default(),
            predecessors: FxHashMap::default(),
            closure: FxHashMap::default(),
            closure_valid: false,
        }
    }

    /// Add a subset relation: lhs ⊆ rhs
    pub fn add_edge(&mut self, lhs: SetVarId, rhs: SetVarId) -> bool {
        // Check for cycles (would mean lhs = rhs)
        let cycle_detected = self.has_path(rhs, lhs);

        let added = self.successors.entry(lhs).or_default().insert(rhs);

        if added {
            self.predecessors.entry(rhs).or_default().insert(lhs);
            self.closure_valid = false;
        }

        cycle_detected // Return true if this edge creates a cycle
    }

    /// Check if there's a path from start to end
    pub fn has_path(&self, start: SetVarId, end: SetVarId) -> bool {
        if start == end {
            return true;
        }

        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if current == end {
                return true;
            }

            if let Some(succs) = self.successors.get(&current) {
                for &succ in succs {
                    if !visited.contains(&succ) {
                        visited.insert(succ);
                        queue.push_back(succ);
                    }
                }
            }
        }

        false
    }

    /// Compute transitive closure
    pub fn compute_closure(&mut self) {
        if self.closure_valid {
            return;
        }

        self.closure.clear();

        // For each node, compute all reachable nodes
        for &node in self.successors.keys() {
            let reachable = self.compute_reachable(node);
            self.closure.insert(node, reachable);
        }

        self.closure_valid = true;
    }

    fn compute_reachable(&self, start: SetVarId) -> FxHashSet<SetVarId> {
        let mut reachable = FxHashSet::default();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        reachable.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(succs) = self.successors.get(&current) {
                for &succ in succs {
                    if !reachable.contains(&succ) {
                        reachable.insert(succ);
                        queue.push_back(succ);
                    }
                }
            }
        }

        reachable
    }

    /// Get all sets that this set is a subset of
    pub fn get_supersets(&self, set: SetVarId) -> Option<&FxHashSet<SetVarId>> {
        self.successors.get(&set)
    }

    /// Get all sets that are subsets of this set
    pub fn get_subsets(&self, set: SetVarId) -> Option<&FxHashSet<SetVarId>> {
        self.predecessors.get(&set)
    }

    /// Get the transitive closure for a set
    pub fn get_closure(&self, set: SetVarId) -> Option<&FxHashSet<SetVarId>> {
        self.closure.get(&set)
    }

    /// Find strongly connected components (equivalence classes)
    pub fn find_scc(&self) -> Vec<Vec<SetVarId>> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = FxHashMap::default();
        let mut lowlinks = FxHashMap::default();
        let mut on_stack = FxHashSet::default();
        let mut sccs = Vec::new();

        for &v in self.successors.keys() {
            if !indices.contains_key(&v) {
                self.strongconnect(
                    v,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut sccs,
                );
            }
        }

        sccs
    }

    #[allow(clippy::too_many_arguments)]
    fn strongconnect(
        &self,
        v: SetVarId,
        index: &mut usize,
        stack: &mut Vec<SetVarId>,
        indices: &mut FxHashMap<SetVarId, usize>,
        lowlinks: &mut FxHashMap<SetVarId, usize>,
        on_stack: &mut FxHashSet<SetVarId>,
        sccs: &mut Vec<Vec<SetVarId>>,
    ) {
        indices.insert(v, *index);
        lowlinks.insert(v, *index);
        *index += 1;
        stack.push(v);
        on_stack.insert(v);

        if let Some(succs) = self.successors.get(&v) {
            for &w in succs {
                if !indices.contains_key(&w) {
                    self.strongconnect(w, index, stack, indices, lowlinks, on_stack, sccs);
                    let w_lowlink = lowlinks[&w];
                    let v_lowlink = lowlinks
                        .get_mut(&v)
                        .expect("v must be in lowlinks after insertion at line 248");
                    *v_lowlink = (*v_lowlink).min(w_lowlink);
                } else if on_stack.contains(&w) {
                    let w_index = indices[&w];
                    let v_lowlink = lowlinks
                        .get_mut(&v)
                        .expect("v must be in lowlinks after insertion at line 248");
                    *v_lowlink = (*v_lowlink).min(w_index);
                }
            }
        }

        if lowlinks[&v] == indices[&v] {
            let mut scc = Vec::new();
            loop {
                let w = stack
                    .pop()
                    .expect("Stack cannot be empty in SCC extraction: v was pushed at line 250");
                on_stack.remove(&w);
                scc.push(w);
                if w == v {
                    break;
                }
            }
            if scc.len() > 1
                || (scc.len() == 1
                    && self
                        .successors
                        .get(&scc[0])
                        .is_some_and(|s| s.contains(&scc[0])))
            {
                sccs.push(scc);
            }
        }
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.successors.clear();
        self.predecessors.clear();
        self.closure.clear();
        self.closure_valid = false;
    }
}

impl Default for SubsetGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Subset result
pub type SubsetResult<T> = Result<T, SetConflict>;

/// Subset statistics
#[derive(Debug, Clone, Default)]
pub struct SubsetStats {
    /// Number of subset constraints
    pub num_constraints: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of transitive closures computed
    pub num_closures: usize,
    /// Number of equivalences found
    pub num_equivalences: usize,
}

/// Subset propagator
#[derive(Debug)]
pub struct SubsetPropagator {
    /// Subset graph
    graph: SubsetGraph,
    /// Subset domains
    domains: FxHashMap<(SetVarId, SetVarId), SubsetDomain>,
    /// Statistics
    stats: SubsetStats,
}

impl SubsetPropagator {
    /// Create a new subset propagator
    pub fn new() -> Self {
        Self {
            graph: SubsetGraph::new(),
            domains: FxHashMap::default(),
            stats: SubsetStats::default(),
        }
    }

    /// Get the subset domain
    pub fn get_domain(&self, lhs: SetVarId, rhs: SetVarId) -> SubsetDomain {
        *self
            .domains
            .get(&(lhs, rhs))
            .unwrap_or(&SubsetDomain::Unknown)
    }

    /// Add a subset constraint
    pub fn add_constraint(&mut self, constraint: SubsetConstraint) -> SubsetResult<()> {
        self.stats.num_constraints += 1;

        if constraint.sign {
            // S1 ⊆ S2
            let cycle = self.graph.add_edge(constraint.lhs, constraint.rhs);
            if cycle {
                // S1 ⊆ S2 and S2 ⊆ S1, so S1 = S2
                self.stats.num_equivalences += 1;
            }

            self.domains
                .insert((constraint.lhs, constraint.rhs), SubsetDomain::True);
        } else {
            // S1 ⊈ S2
            self.domains
                .insert((constraint.lhs, constraint.rhs), SubsetDomain::False);

            // Check for conflict with existing subset relation
            if self.graph.has_path(constraint.lhs, constraint.rhs) {
                return Err(SetConflict {
                    literals: vec![
                        SetLiteral::Subset {
                            lhs: constraint.lhs,
                            rhs: constraint.rhs,
                            sign: true,
                        },
                        SetLiteral::Subset {
                            lhs: constraint.lhs,
                            rhs: constraint.rhs,
                            sign: false,
                        },
                    ],
                    reason: "Subset conflict: cannot have both S1 ⊆ S2 and S1 ⊈ S2".to_string(),
                    proof_steps: vec![],
                });
            }
        }

        Ok(())
    }

    /// Propagate subset constraints
    pub fn propagate(
        &mut self,
        var: SetVarId,
        vars: &mut [SetVar],
        constraints: &[SubsetConstraint],
    ) -> SubsetResult<()> {
        // Compute transitive closure
        self.graph.compute_closure();
        self.stats.num_closures += 1;

        // Propagate memberships based on subset relations
        let supersets_to_process: Vec<_> = self
            .graph
            .get_supersets(var)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();

        for superset in supersets_to_process {
            self.propagate_membership(var, superset, vars)?;
        }

        // Check for equivalences (SCCs)
        let sccs = self.graph.find_scc();
        for scc in sccs {
            if scc.len() > 1 {
                // All sets in this SCC are equivalent
                self.propagate_equivalence(&scc, vars)?;
            }
        }

        // Propagate explicit constraints
        for constraint in constraints {
            if (constraint.lhs == var || constraint.rhs == var)
                && let (Some(lhs_var), Some(rhs_var)) = (
                    vars.get(constraint.lhs.id() as usize),
                    vars.get(constraint.rhs.id() as usize),
                )
                && let Some(satisfied) = constraint.is_satisfied(lhs_var, rhs_var)
                && !satisfied
            {
                return Err(SetConflict {
                    literals: vec![],
                    reason: "Subset constraint violated".to_string(),
                    proof_steps: vec![],
                });
            }
        }

        Ok(())
    }

    /// Propagate membership from subset to superset
    fn propagate_membership(
        &mut self,
        subset: SetVarId,
        superset: SetVarId,
        vars: &mut [SetVar],
    ) -> SubsetResult<()> {
        // Get elements from subset
        let subset_must: SmallVec<[u32; 16]> =
            if let Some(subset_var) = vars.get(subset.id() as usize) {
                subset_var.must_members.iter().copied().collect()
            } else {
                return Ok(());
            };

        // Propagate to superset
        if let Some(superset_var) = vars.get_mut(superset.id() as usize) {
            for elem in subset_must {
                if superset_var.must_not_members.contains(&elem) {
                    return Err(SetConflict {
                        literals: vec![
                            SetLiteral::Member {
                                element: elem,
                                set: subset,
                                sign: true,
                            },
                            SetLiteral::Member {
                                element: elem,
                                set: superset,
                                sign: false,
                            },
                            SetLiteral::Subset {
                                lhs: subset,
                                rhs: superset,
                                sign: true,
                            },
                        ],
                        reason: format!(
                            "Subset membership conflict: element {} is in subset but not in superset",
                            elem
                        ),
                        proof_steps: vec![SetProofStep::SubsetProp {
                            from: subset,
                            mid: subset,
                            to: superset,
                        }],
                    });
                }

                superset_var.add_must_member(elem);
                self.stats.num_propagations += 1;
            }
        }

        // Get elements that must not be in superset
        let superset_must_not: SmallVec<[u32; 16]> =
            if let Some(superset_var) = vars.get(superset.id() as usize) {
                superset_var.must_not_members.iter().copied().collect()
            } else {
                return Ok(());
            };

        // Propagate to subset (contrapositive)
        if let Some(subset_var) = vars.get_mut(subset.id() as usize) {
            for elem in superset_must_not {
                if subset_var.must_members.contains(&elem) {
                    return Err(SetConflict {
                        literals: vec![],
                        reason: format!(
                            "Subset membership conflict: element {} cannot be in subset",
                            elem
                        ),
                        proof_steps: vec![],
                    });
                }

                subset_var.add_must_not_member(elem);
                self.stats.num_propagations += 1;
            }
        }

        Ok(())
    }

    /// Propagate equivalence for a strongly connected component
    fn propagate_equivalence(&mut self, scc: &[SetVarId], vars: &mut [SetVar]) -> SubsetResult<()> {
        if scc.is_empty() {
            return Ok(());
        }

        // Collect all must_members and must_not_members from the SCC
        let mut all_must = FxHashSet::default();
        let mut all_must_not = FxHashSet::default();

        for &set_id in scc {
            if let Some(var) = vars.get(set_id.id() as usize) {
                all_must.extend(&var.must_members);
                all_must_not.extend(&var.must_not_members);
            }
        }

        // Check for conflicts
        for &elem in &all_must {
            if all_must_not.contains(&elem) {
                return Err(SetConflict {
                    literals: vec![],
                    reason: format!(
                        "Equivalence conflict: element {} is both in and not in equivalent sets",
                        elem
                    ),
                    proof_steps: vec![],
                });
            }
        }

        // Propagate to all sets in the SCC
        for &set_id in scc {
            if let Some(var) = vars.get_mut(set_id.id() as usize) {
                for &elem in &all_must {
                    var.add_must_member(elem);
                }
                for &elem in &all_must_not {
                    var.add_must_not_member(elem);
                }
            }
        }

        Ok(())
    }

    /// Get the subset graph
    pub fn graph(&self) -> &SubsetGraph {
        &self.graph
    }

    /// Get statistics
    pub fn stats(&self) -> &SubsetStats {
        &self.stats
    }

    /// Reset the propagator
    pub fn reset(&mut self) {
        self.graph.clear();
        self.domains.clear();
        self.stats = SubsetStats::default();
    }
}

impl Default for SubsetPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::SetSort;
    use super::*;

    #[test]
    fn test_subset_graph_add_edge() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        let cycle = graph.add_edge(s1, s2);
        assert!(!cycle);

        assert!(graph.has_path(s1, s2));
        assert!(!graph.has_path(s2, s1));
    }

    #[test]
    fn test_subset_graph_cycle() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        graph.add_edge(s1, s2);
        let cycle = graph.add_edge(s2, s1);
        assert!(cycle); // Creates a cycle, meaning s1 = s2
    }

    #[test]
    fn test_subset_graph_transitive() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        graph.add_edge(s1, s2);
        graph.add_edge(s2, s3);

        assert!(graph.has_path(s1, s3)); // Transitive
    }

    #[test]
    fn test_subset_graph_closure() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        graph.add_edge(s1, s2);
        graph.add_edge(s2, s3);
        graph.compute_closure();

        let closure = graph.get_closure(s1).unwrap();
        assert!(closure.contains(&s1));
        assert!(closure.contains(&s2));
        assert!(closure.contains(&s3));
    }

    #[test]
    fn test_subset_graph_scc() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        graph.add_edge(s1, s2);
        graph.add_edge(s2, s3);
        graph.add_edge(s3, s1);

        let sccs = graph.find_scc();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_subset_propagator_add_constraint() {
        let mut prop = SubsetPropagator::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        let constraint = SubsetConstraint::new(s1, s2, true, 0);
        assert!(prop.add_constraint(constraint).is_ok());

        assert_eq!(prop.get_domain(s1, s2), SubsetDomain::True);
    }

    #[test]
    fn test_subset_propagator_conflict() {
        let mut prop = SubsetPropagator::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        prop.add_constraint(SubsetConstraint::new(s1, s2, true, 0))
            .unwrap();

        let result = prop.add_constraint(SubsetConstraint::new(s1, s2, false, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_subset_propagator_membership() {
        let mut prop = SubsetPropagator::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        prop.add_constraint(SubsetConstraint::new(s1, s2, true, 0))
            .unwrap();

        let mut vars = vec![
            SetVar::new(s1, "S1".to_string(), SetSort::IntSet, 0),
            SetVar::new(s2, "S2".to_string(), SetSort::IntSet, 0),
        ];

        // Add element 42 to S1
        vars[0].add_must_member(42);

        prop.propagate_membership(s1, s2, &mut vars).unwrap();

        // Element 42 should now be in S2
        assert!(vars[1].must_members.contains(&42));
    }

    #[test]
    fn test_subset_propagator_equivalence() {
        let mut prop = SubsetPropagator::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        // Create a cycle: S1 ⊆ S2 ⊆ S3 ⊆ S1
        prop.add_constraint(SubsetConstraint::new(s1, s2, true, 0))
            .unwrap();
        prop.add_constraint(SubsetConstraint::new(s2, s3, true, 0))
            .unwrap();
        prop.add_constraint(SubsetConstraint::new(s3, s1, true, 0))
            .unwrap();

        let mut vars = vec![
            SetVar::new(s1, "S1".to_string(), SetSort::IntSet, 0),
            SetVar::new(s2, "S2".to_string(), SetSort::IntSet, 0),
            SetVar::new(s3, "S3".to_string(), SetSort::IntSet, 0),
        ];

        vars[0].add_must_member(42);

        let scc = vec![s1, s2, s3];
        prop.propagate_equivalence(&scc, &mut vars).unwrap();

        // All sets should have element 42
        assert!(vars[0].must_members.contains(&42));
        assert!(vars[1].must_members.contains(&42));
        assert!(vars[2].must_members.contains(&42));
    }

    #[test]
    fn test_subset_constraint_is_satisfied() {
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        let constraint = SubsetConstraint::new(s1, s2, true, 0);

        let mut lhs_var = SetVar::new(s1, "S1".to_string(), SetSort::IntSet, 0);
        let mut rhs_var = SetVar::new(s2, "S2".to_string(), SetSort::IntSet, 0);

        lhs_var.add_must_member(1);
        rhs_var.add_must_member(1);
        rhs_var.add_must_member(2);

        // S1 = {1} ⊆ S2 = {1, 2} should be satisfiable
        let result = constraint.is_satisfied(&lhs_var, &rhs_var);
        // It's unknown because we don't know if S1 is fully determined
        assert!(result.is_none());

        // Add a may_members to make S1 fully determined
        let mut may = FxHashSet::default();
        may.insert(1);
        lhs_var.may_members = Some(may);

        let result = constraint.is_satisfied(&lhs_var, &rhs_var);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn test_subset_constraint_unsatisfied() {
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);

        let constraint = SubsetConstraint::new(s1, s2, true, 0);

        let mut lhs_var = SetVar::new(s1, "S1".to_string(), SetSort::IntSet, 0);
        let mut rhs_var = SetVar::new(s2, "S2".to_string(), SetSort::IntSet, 0);

        lhs_var.add_must_member(1);
        rhs_var.add_must_not_member(1);

        // S1 has element 1, S2 doesn't allow element 1
        let result = constraint.is_satisfied(&lhs_var, &rhs_var);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn test_subset_graph_supersets() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        graph.add_edge(s1, s2);
        graph.add_edge(s1, s3);

        let supersets = graph.get_supersets(s1).unwrap();
        assert_eq!(supersets.len(), 2);
        assert!(supersets.contains(&s2));
        assert!(supersets.contains(&s3));
    }

    #[test]
    fn test_subset_graph_subsets() {
        let mut graph = SubsetGraph::new();
        let s1 = SetVarId(0);
        let s2 = SetVarId(1);
        let s3 = SetVarId(2);

        graph.add_edge(s1, s3);
        graph.add_edge(s2, s3);

        let subsets = graph.get_subsets(s3).unwrap();
        assert_eq!(subsets.len(), 2);
        assert!(subsets.contains(&s1));
        assert!(subsets.contains(&s2));
    }
}
