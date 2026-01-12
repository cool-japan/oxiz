/// Community-based clause database partitioning for improved cache locality.
///
/// This module integrates community detection with clause database management to partition
/// clauses based on their variable communities. This improves cache locality and reduces
/// memory access overhead during solving.
use crate::clause::Clause;
use crate::community::{Communities, CommunityOrdering, LouvainDetector, VariableIncidenceGraph};
use std::collections::HashMap;

/// Partitioned clause database organized by variable communities.
#[derive(Debug, Clone)]
pub struct CommunityPartition {
    /// Clauses grouped by their primary community (community with most variables)
    partitions: Vec<Vec<usize>>,
    /// Community structure
    communities: Communities,
    /// Number of communities
    num_communities: usize,
}

impl CommunityPartition {
    /// Creates a new community partition from clauses.
    pub fn from_clauses(num_vars: usize, clauses: &[Clause]) -> Self {
        // Build VIG and detect communities
        let vig = VariableIncidenceGraph::from_clauses(num_vars, clauses);
        let detector = LouvainDetector::default();
        let communities = detector.detect(&vig);
        let num_communities = communities.num_communities();

        // Partition clauses by their dominant community
        let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); num_communities];

        for (idx, clause) in clauses.iter().enumerate() {
            if let Some(primary_comm) = Self::get_primary_community(clause, &communities)
                && primary_comm < num_communities
            {
                partitions[primary_comm].push(idx);
            }
        }

        Self {
            partitions,
            communities,
            num_communities,
        }
    }

    /// Gets the primary community for a clause (community with most variables in clause).
    fn get_primary_community(clause: &Clause, communities: &Communities) -> Option<usize> {
        if clause.lits.is_empty() {
            return None;
        }

        let mut comm_counts: HashMap<usize, usize> = HashMap::new();

        for lit in &clause.lits {
            let var = lit.var().index();
            let comm = communities.community(var);
            *comm_counts.entry(comm).or_insert(0) += 1;
        }

        comm_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(comm, _)| comm)
    }

    /// Returns the clause indices in a specific community partition.
    pub fn get_partition(&self, community: usize) -> &[usize] {
        if community < self.partitions.len() {
            &self.partitions[community]
        } else {
            &[]
        }
    }

    /// Returns the number of communities.
    pub fn num_communities(&self) -> usize {
        self.num_communities
    }

    /// Returns the community ID for a variable.
    pub fn community(&self, var: usize) -> usize {
        self.communities.community(var)
    }

    /// Returns all partition sizes.
    pub fn partition_sizes(&self) -> Vec<usize> {
        self.partitions.iter().map(|p| p.len()).collect()
    }

    /// Returns the total number of clauses across all partitions.
    pub fn total_clauses(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    /// Returns the communities structure.
    pub fn communities(&self) -> &Communities {
        &self.communities
    }

    /// Creates a community-aware variable ordering.
    pub fn create_ordering(&self) -> CommunityOrdering {
        CommunityOrdering::new(self.communities.clone())
    }
}

/// Statistics for community-based partitioning.
#[derive(Debug, Clone, Default)]
pub struct PartitionStats {
    /// Number of communities
    pub num_communities: usize,
    /// Total clauses partitioned
    pub total_clauses: usize,
    /// Partition sizes
    pub partition_sizes: Vec<usize>,
    /// Average partition size
    pub avg_partition_size: f64,
    /// Largest partition size
    pub max_partition_size: usize,
    /// Smallest partition size
    pub min_partition_size: usize,
    /// Modularity of the partition
    pub modularity: f64,
}

impl PartitionStats {
    /// Creates statistics from a community partition.
    pub fn from_partition(partition: &CommunityPartition) -> Self {
        let sizes = partition.partition_sizes();
        let total = partition.total_clauses();
        let avg = if !sizes.is_empty() {
            total as f64 / sizes.len() as f64
        } else {
            0.0
        };

        Self {
            num_communities: partition.num_communities(),
            total_clauses: total,
            partition_sizes: sizes.clone(),
            avg_partition_size: avg,
            max_partition_size: sizes.iter().copied().max().unwrap_or(0),
            min_partition_size: sizes.iter().copied().min().unwrap_or(0),
            modularity: partition.communities().modularity(),
        }
    }

    /// Displays the statistics.
    pub fn display(&self) -> String {
        format!(
            "Community Partition Statistics:\n\
             - Communities: {}\n\
             - Total Clauses: {}\n\
             - Avg Partition Size: {:.2}\n\
             - Size Range: [{}, {}]\n\
             - Modularity: {:.4}",
            self.num_communities,
            self.total_clauses,
            self.avg_partition_size,
            self.min_partition_size,
            self.max_partition_size,
            self.modularity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::community::{LouvainDetector, VariableIncidenceGraph};
    use crate::literal::{Lit, Var};

    fn make_lit(var: usize, sign: bool) -> Lit {
        let v = Var::new(var as u32);
        if sign { Lit::pos(v) } else { Lit::neg(v) }
    }

    #[test]
    fn test_partition_creation() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(0, false), make_lit(1, true)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
            Clause::original(vec![make_lit(2, true), make_lit(3, false)]),
        ];

        let partition = CommunityPartition::from_clauses(4, &clauses);

        assert!(partition.num_communities() > 0);
        assert_eq!(partition.total_clauses(), 4);
    }

    #[test]
    fn test_get_partition() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
        ];

        let partition = CommunityPartition::from_clauses(4, &clauses);

        for i in 0..partition.num_communities() {
            let part = partition.get_partition(i);
            assert!(part.len() <= 2);
        }
    }

    #[test]
    fn test_primary_community() {
        // Create a simple VIG to get communities
        let test_clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(0, true)]),
            Clause::original(vec![make_lit(1, false)]),
        ];

        let vig = VariableIncidenceGraph::from_clauses(2, &test_clauses);
        let detector = LouvainDetector::default();
        let communities = detector.detect(&vig);

        // Now test with a clause where variable 0 appears twice
        let clause = Clause::original(vec![
            make_lit(0, false),
            make_lit(0, true),
            make_lit(1, false),
        ]);

        let primary = CommunityPartition::get_primary_community(&clause, &communities);
        assert!(primary.is_some());
        // The primary community should be var 0's community since it appears twice
        assert_eq!(primary, Some(communities.community(0)));
    }

    #[test]
    fn test_partition_sizes() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
            Clause::original(vec![make_lit(4, false), make_lit(5, false)]),
        ];

        let partition = CommunityPartition::from_clauses(6, &clauses);
        let sizes = partition.partition_sizes();

        assert_eq!(sizes.iter().sum::<usize>(), 3);
    }

    #[test]
    fn test_create_ordering() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
        ];

        let partition = CommunityPartition::from_clauses(4, &clauses);
        let ordering = partition.create_ordering();

        assert_eq!(ordering.ordering().len(), 4);
    }

    #[test]
    fn test_partition_stats() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
            Clause::original(vec![make_lit(4, false), make_lit(5, false)]),
        ];

        let partition = CommunityPartition::from_clauses(6, &clauses);
        let stats = PartitionStats::from_partition(&partition);

        assert_eq!(stats.total_clauses, 3);
        assert!(stats.avg_partition_size > 0.0);
        assert!(stats.modularity >= 0.0);
    }

    #[test]
    fn test_empty_partition() {
        let partition = CommunityPartition::from_clauses(5, &[]);

        assert_eq!(partition.total_clauses(), 0);
        assert_eq!(partition.num_communities(), 5);
    }

    #[test]
    fn test_single_clause_partition() {
        let clauses = vec![Clause::original(vec![
            make_lit(0, false),
            make_lit(1, false),
        ])];

        let partition = CommunityPartition::from_clauses(2, &clauses);

        assert_eq!(partition.total_clauses(), 1);
    }

    #[test]
    fn test_community_lookup() {
        let clauses = vec![
            Clause::original(vec![make_lit(0, false), make_lit(1, false)]),
            Clause::original(vec![make_lit(2, false), make_lit(3, false)]),
        ];

        let partition = CommunityPartition::from_clauses(4, &clauses);

        // All variables should be assigned to some community
        for var in 0..4 {
            let comm = partition.community(var);
            assert!(comm < partition.num_communities());
        }
    }
}
