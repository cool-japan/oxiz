/// Community detection for SAT formulas using graph-based clustering.
///
/// This module implements Variable Incidence Graph (VIG) construction and community
/// detection algorithms to partition variables into communities. This improves:
/// - Cache locality by keeping related variables together
/// - Branching decisions by prioritizing variables in the same community
/// - Clause database management by partitioning based on communities
/// - Parallel solving by distributing communities across threads
///
/// The implementation uses the Louvain algorithm for community detection, which is
/// fast and produces high-quality partitions with high modularity.
use crate::clause::Clause;
use std::collections::HashMap;

/// Variable Incidence Graph (VIG).
///
/// In the VIG, each variable is a node, and edges connect variables that appear
/// together in clauses. Edge weights represent the number of clauses shared.
#[derive(Debug, Clone)]
pub struct VariableIncidenceGraph {
    /// Number of variables
    num_vars: usize,
    /// Adjacency list: var -> (neighbor, weight)
    adjacency: Vec<HashMap<usize, f64>>,
    /// Total edge weight in the graph
    total_weight: f64,
}

impl VariableIncidenceGraph {
    /// Creates a new VIG from a set of clauses.
    pub fn from_clauses(num_vars: usize, clauses: &[Clause]) -> Self {
        let mut adjacency = vec![HashMap::new(); num_vars];
        let mut total_weight = 0.0;

        for clause in clauses {
            let vars: Vec<usize> = clause.lits.iter().map(|lit| lit.var().index()).collect();

            // Add edges between all pairs of variables in this clause
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let (u, v) = (vars[i], vars[j]);
                    if u < num_vars && v < num_vars {
                        *adjacency[u].entry(v).or_insert(0.0) += 1.0;
                        *adjacency[v].entry(u).or_insert(0.0) += 1.0;
                        total_weight += 2.0;
                    }
                }
            }
        }

        Self {
            num_vars,
            adjacency,
            total_weight,
        }
    }

    /// Returns the number of variables in the graph.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the neighbors of a variable with their edge weights.
    pub fn neighbors(&self, var: usize) -> &HashMap<usize, f64> {
        &self.adjacency[var]
    }

    /// Returns the degree (sum of edge weights) of a variable.
    pub fn degree(&self, var: usize) -> f64 {
        self.adjacency[var].values().sum()
    }

    /// Returns the total edge weight in the graph.
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }
}

/// Community structure representing a partition of variables.
#[derive(Debug, Clone)]
pub struct Communities {
    /// Assignment of each variable to a community ID
    assignment: Vec<usize>,
    /// Number of distinct communities
    num_communities: usize,
    /// Modularity score of the partition (quality metric)
    modularity: f64,
}

impl Communities {
    /// Creates a new community structure with each variable in its own community.
    pub fn new(num_vars: usize) -> Self {
        Self {
            assignment: (0..num_vars).collect(),
            num_communities: num_vars,
            modularity: 0.0,
        }
    }

    /// Returns the community ID of a variable.
    pub fn community(&self, var: usize) -> usize {
        self.assignment[var]
    }

    /// Returns the number of communities.
    pub fn num_communities(&self) -> usize {
        self.num_communities
    }

    /// Returns the modularity score.
    pub fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Assigns a variable to a community.
    fn assign(&mut self, var: usize, community: usize) {
        self.assignment[var] = community;
    }

    /// Renumbers communities to be contiguous starting from 0.
    fn renumber_communities(&mut self) {
        let mut community_map = HashMap::new();
        let mut next_id = 0;

        for var in 0..self.assignment.len() {
            let old_id = self.assignment[var];
            let new_id = *community_map.entry(old_id).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            self.assignment[var] = new_id;
        }

        self.num_communities = next_id;
    }

    /// Returns variables in each community.
    pub fn get_communities(&self) -> Vec<Vec<usize>> {
        let mut communities = vec![Vec::new(); self.num_communities];
        for (var, &comm_id) in self.assignment.iter().enumerate() {
            if comm_id < self.num_communities {
                communities[comm_id].push(var);
            }
        }
        communities
    }
}

/// Louvain community detection algorithm.
///
/// This is a greedy optimization method that maximizes modularity. It operates in two phases:
/// 1. Local moving: Each variable is moved to the community that gives the maximum increase
///    in modularity.
/// 2. Aggregation: Build a new graph where nodes are communities.
///
/// These phases are repeated until modularity stops improving.
pub struct LouvainDetector {
    /// Maximum number of iterations
    max_iterations: usize,
    /// Minimum modularity improvement to continue
    min_improvement: f64,
}

impl Default for LouvainDetector {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            min_improvement: 1e-6,
        }
    }
}

impl LouvainDetector {
    /// Creates a new Louvain detector with custom parameters.
    pub fn new(max_iterations: usize, min_improvement: f64) -> Self {
        Self {
            max_iterations,
            min_improvement,
        }
    }

    /// Detects communities in the variable incidence graph.
    pub fn detect(&self, vig: &VariableIncidenceGraph) -> Communities {
        let mut communities = Communities::new(vig.num_vars());

        // Phase 1: Local optimization
        for _ in 0..self.max_iterations {
            let old_modularity = communities.modularity;
            let improved = self.local_moving(vig, &mut communities);

            if !improved || (communities.modularity - old_modularity) < self.min_improvement {
                break;
            }
        }

        communities.renumber_communities();
        communities
    }

    /// Phase 1: Local moving phase.
    ///
    /// Each variable is moved to the neighbor community that maximizes modularity gain.
    fn local_moving(&self, vig: &VariableIncidenceGraph, communities: &mut Communities) -> bool {
        let mut improved = false;
        let m2 = vig.total_weight();

        // Pre-compute community degrees
        let mut comm_degrees: Vec<f64> = vec![0.0; vig.num_vars()];
        for var in 0..vig.num_vars() {
            let comm = communities.community(var);
            comm_degrees[comm] += vig.degree(var);
        }

        // Try to improve each variable's community assignment
        for var in 0..vig.num_vars() {
            let current_comm = communities.community(var);
            let var_degree = vig.degree(var);

            // Calculate modularity gain for moving to each neighbor community
            let mut best_comm = current_comm;
            let mut best_gain = 0.0;

            // Count edges to neighboring communities
            let mut neighbor_comms: HashMap<usize, f64> = HashMap::new();
            for (&neighbor, &weight) in vig.neighbors(var).iter() {
                let neighbor_comm = communities.community(neighbor);
                *neighbor_comms.entry(neighbor_comm).or_insert(0.0) += weight;
            }

            for (&comm, &edge_weight) in neighbor_comms.iter() {
                if comm == current_comm {
                    continue;
                }

                // Modularity gain calculation
                let sigma_tot = comm_degrees[comm];
                let k_i = var_degree;
                let k_i_in = edge_weight;

                let gain = (k_i_in / m2) - (sigma_tot * k_i / (m2 * m2));

                if gain > best_gain {
                    best_gain = gain;
                    best_comm = comm;
                }
            }

            // Move variable to best community
            if best_comm != current_comm && best_gain > 0.0 {
                comm_degrees[current_comm] -= var_degree;
                comm_degrees[best_comm] += var_degree;
                communities.assign(var, best_comm);
                improved = true;
            }
        }

        // Update modularity
        communities.modularity = self.calculate_modularity(vig, communities);
        improved
    }

    /// Calculates the modularity of a community partition.
    ///
    /// Modularity Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
    /// where A_ij is edge weight, k_i is degree, m is total weight, c_i is community.
    fn calculate_modularity(&self, vig: &VariableIncidenceGraph, communities: &Communities) -> f64 {
        let m2 = vig.total_weight();
        if m2 == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;

        for i in 0..vig.num_vars() {
            let comm_i = communities.community(i);
            let deg_i = vig.degree(i);

            for (&j, &weight) in vig.neighbors(i).iter() {
                let comm_j = communities.community(j);

                if comm_i == comm_j {
                    let deg_j = vig.degree(j);
                    modularity += weight - (deg_i * deg_j / m2);
                }
            }
        }

        modularity / m2
    }
}

/// Community-aware variable ordering.
///
/// Orders variables by community membership to improve cache locality.
pub struct CommunityOrdering {
    /// Community structure
    communities: Communities,
    /// Variable ordering (sorted by community)
    ordering: Vec<usize>,
}

impl CommunityOrdering {
    /// Creates a new community-aware ordering.
    pub fn new(communities: Communities) -> Self {
        let mut ordering = Vec::with_capacity(communities.assignment.len());

        // Group variables by community
        let comm_groups = communities.get_communities();

        for group in comm_groups {
            ordering.extend(group);
        }

        Self {
            communities,
            ordering,
        }
    }

    /// Returns the ordered list of variables.
    pub fn ordering(&self) -> &[usize] {
        &self.ordering
    }

    /// Returns the community of a variable.
    pub fn community(&self, var: usize) -> usize {
        self.communities.community(var)
    }

    /// Returns variables in the same community as the given variable.
    pub fn same_community(&self, var: usize) -> Vec<usize> {
        let target_comm = self.communities.community(var);
        (0..self.communities.assignment.len())
            .filter(|&v| self.communities.community(v) == target_comm)
            .collect()
    }
}

/// Statistics for community detection.
#[derive(Debug, Clone, Default)]
pub struct CommunityStats {
    /// Number of communities detected
    pub num_communities: usize,
    /// Modularity score
    pub modularity: f64,
    /// Average community size
    pub avg_community_size: f64,
    /// Largest community size
    pub max_community_size: usize,
    /// Smallest community size
    pub min_community_size: usize,
    /// Number of variables
    pub num_vars: usize,
}

impl CommunityStats {
    /// Creates statistics from a community structure.
    pub fn from_communities(communities: &Communities) -> Self {
        let groups = communities.get_communities();
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();

        let avg_size = if !sizes.is_empty() {
            sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
        } else {
            0.0
        };

        Self {
            num_communities: communities.num_communities(),
            modularity: communities.modularity(),
            avg_community_size: avg_size,
            max_community_size: sizes.iter().copied().max().unwrap_or(0),
            min_community_size: sizes.iter().copied().min().unwrap_or(0),
            num_vars: communities.assignment.len(),
        }
    }

    /// Displays the statistics.
    pub fn display(&self) -> String {
        format!(
            "Community Detection Statistics:\n\
             - Variables: {}\n\
             - Communities: {}\n\
             - Modularity: {:.4}\n\
             - Avg Size: {:.2}\n\
             - Size Range: [{}, {}]",
            self.num_vars,
            self.num_communities,
            self.modularity,
            self.avg_community_size,
            self.min_community_size,
            self.max_community_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::{Lit, Var};

    fn make_lit(var: usize, sign: bool) -> Lit {
        let v = Var::new(var as u32);
        if sign { Lit::pos(v) } else { Lit::neg(v) }
    }

    #[test]
    fn test_vig_creation() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(1, false), make_lit(2, false)]),
            Clause::original(vec![make_lit(0, false), make_lit(2, false)]),
        ];

        let vig = VariableIncidenceGraph::from_clauses(3, &clauses);

        assert_eq!(vig.num_vars(), 3);
        assert!(vig.degree(0) > 0.0);
        assert!(vig.degree(1) > 0.0);
        assert!(vig.degree(2) > 0.0);
    }

    #[test]
    fn test_vig_edges() {
        let clauses = vec![Clause::original(vec![
            make_lit(0, false),
            make_lit(1, false),
        ])];

        let vig = VariableIncidenceGraph::from_clauses(2, &clauses);

        assert_eq!(vig.neighbors(0).get(&1), Some(&1.0));
        assert_eq!(vig.neighbors(1).get(&0), Some(&1.0));
    }

    #[test]
    fn test_communities_creation() {
        let communities = Communities::new(5);

        assert_eq!(communities.num_communities(), 5);
        assert_eq!(communities.community(0), 0);
        assert_eq!(communities.community(4), 4);
    }

    #[test]
    fn test_communities_renumber() {
        let mut communities = Communities::new(5);
        communities.assign(0, 10);
        communities.assign(1, 10);
        communities.assign(2, 20);
        communities.assign(3, 20);
        communities.assign(4, 30);

        communities.renumber_communities();

        assert_eq!(communities.num_communities(), 3);
        assert_eq!(communities.community(0), communities.community(1));
        assert_eq!(communities.community(2), communities.community(3));
    }

    #[test]
    fn test_louvain_simple() {
        // Create a simple graph: 0-1-2-3, 4-5
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(1, false), make_lit(2, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
            Clause::original(vec![make_lit(4, false), make_lit(5, false)]),
        ];

        let vig = VariableIncidenceGraph::from_clauses(6, &clauses);
        let detector = LouvainDetector::default();
        let communities = detector.detect(&vig);

        // Should detect 2 communities
        assert!(communities.num_communities() <= 3);

        // 0,1,2,3 should be in one community, 4,5 in another
        let comm_01 = communities.community(0);
        let comm_45 = communities.community(4);
        assert_ne!(comm_01, comm_45);
    }

    #[test]
    fn test_community_ordering() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
        ];

        let vig = VariableIncidenceGraph::from_clauses(4, &clauses);
        let detector = LouvainDetector::default();
        let communities = detector.detect(&vig);
        let ordering = CommunityOrdering::new(communities);

        assert_eq!(ordering.ordering().len(), 4);
    }

    #[test]
    fn test_same_community() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
        ];

        let vig = VariableIncidenceGraph::from_clauses(4, &clauses);
        let detector = LouvainDetector::default();
        let communities = detector.detect(&vig);
        let ordering = CommunityOrdering::new(communities);

        let same = ordering.same_community(0);
        assert!(same.contains(&0));
    }

    #[test]
    fn test_community_stats() {
        let mut communities = Communities::new(10);
        // Assign all variables to two communities
        communities.assign(0, 0);
        communities.assign(1, 0);
        communities.assign(2, 0);
        communities.assign(3, 0);
        communities.assign(4, 0);
        communities.assign(5, 1);
        communities.assign(6, 1);
        communities.assign(7, 1);
        communities.assign(8, 1);
        communities.assign(9, 1);
        communities.renumber_communities();

        let stats = CommunityStats::from_communities(&communities);

        assert_eq!(stats.num_communities, 2);
        assert_eq!(stats.num_vars, 10);
        assert_eq!(stats.avg_community_size, 5.0);
        assert_eq!(stats.min_community_size, 5);
        assert_eq!(stats.max_community_size, 5);
    }

    #[test]
    fn test_modularity_calculation() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
        ];

        let vig = VariableIncidenceGraph::from_clauses(4, &clauses);
        let detector = LouvainDetector::default();
        let communities = detector.detect(&vig);

        // Modularity should be non-negative
        assert!(communities.modularity() >= 0.0);
    }
}
