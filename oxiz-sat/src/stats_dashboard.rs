//! Statistics Dashboard
//!
//! Comprehensive statistics aggregation and display for the SAT solver.
//! This module provides a unified view of all solver statistics with
//! formatted output for analysis and debugging.

use crate::solver::SolverStats;
use std::fmt;
use std::time::Duration;

/// Comprehensive statistics dashboard
#[derive(Debug, Clone)]
pub struct StatsDashboard {
    /// Solver statistics
    solver_stats: SolverStats,
    /// Total wall clock time
    wall_time: Duration,
    /// CPU time (if available)
    cpu_time: Option<Duration>,
    /// Peak memory usage in bytes
    peak_memory: Option<usize>,
}

impl StatsDashboard {
    /// Create a new statistics dashboard
    #[must_use]
    pub fn new(solver_stats: SolverStats, wall_time: Duration) -> Self {
        Self {
            solver_stats,
            wall_time,
            cpu_time: None,
            peak_memory: None,
        }
    }

    /// Set CPU time
    pub fn with_cpu_time(mut self, cpu_time: Duration) -> Self {
        self.cpu_time = Some(cpu_time);
        self
    }

    /// Set peak memory usage
    pub fn with_peak_memory(mut self, bytes: usize) -> Self {
        self.peak_memory = Some(bytes);
        self
    }

    /// Get solver statistics
    #[must_use]
    pub const fn solver_stats(&self) -> &SolverStats {
        &self.solver_stats
    }

    /// Get wall time
    #[must_use]
    pub const fn wall_time(&self) -> Duration {
        self.wall_time
    }

    /// Format human-readable size
    fn format_size(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Format duration in human-readable form
    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();
        let millis = duration.subsec_millis();

        if secs >= 60 {
            let mins = secs / 60;
            let secs = secs % 60;
            format!("{}m {}.{:03}s", mins, secs, millis)
        } else {
            format!("{}.{:03}s", secs, millis)
        }
    }

    /// Get propagations per second
    #[must_use]
    pub fn props_per_second(&self) -> f64 {
        let secs = self.wall_time.as_secs_f64();
        if secs > 0.0 {
            self.solver_stats.propagations as f64 / secs
        } else {
            0.0
        }
    }

    /// Get conflicts per second
    #[must_use]
    pub fn conflicts_per_second(&self) -> f64 {
        let secs = self.wall_time.as_secs_f64();
        if secs > 0.0 {
            self.solver_stats.conflicts as f64 / secs
        } else {
            0.0
        }
    }

    /// Get decisions per second
    #[must_use]
    pub fn decisions_per_second(&self) -> f64 {
        let secs = self.wall_time.as_secs_f64();
        if secs > 0.0 {
            self.solver_stats.decisions as f64 / secs
        } else {
            0.0
        }
    }

    /// Display compact summary
    pub fn display_compact(&self) -> String {
        format!(
            "Time: {} | Conflicts: {} ({:.0}/s) | Decisions: {} | Propagations: {} ({:.0}/s)",
            Self::format_duration(self.wall_time),
            self.solver_stats.conflicts,
            self.conflicts_per_second(),
            self.solver_stats.decisions,
            self.solver_stats.propagations,
            self.props_per_second()
        )
    }

    /// Display detailed statistics
    pub fn display_detailed(&self) -> String {
        let mut output = String::new();

        output.push_str("═══════════════════════════════════════════════════════════════\n");
        output.push_str("                    SOLVER STATISTICS                          \n");
        output.push_str("═══════════════════════════════════════════════════════════════\n\n");

        // Timing information
        output.push_str("TIMING:\n");
        output.push_str(&format!(
            "  Wall Time:          {}\n",
            Self::format_duration(self.wall_time)
        ));
        if let Some(cpu_time) = self.cpu_time {
            output.push_str(&format!(
                "  CPU Time:           {}\n",
                Self::format_duration(cpu_time)
            ));
        }
        if let Some(peak_mem) = self.peak_memory {
            output.push_str(&format!(
                "  Peak Memory:        {}\n",
                Self::format_size(peak_mem)
            ));
        }
        output.push('\n');

        // Search statistics
        output.push_str("SEARCH:\n");
        output.push_str(&format!(
            "  Conflicts:          {} ({:.0}/s)\n",
            self.solver_stats.conflicts,
            self.conflicts_per_second()
        ));
        output.push_str(&format!(
            "  Decisions:          {} ({:.0}/s)\n",
            self.solver_stats.decisions,
            self.decisions_per_second()
        ));
        output.push_str(&format!(
            "  Propagations:       {} ({:.0}/s)\n",
            self.solver_stats.propagations,
            self.props_per_second()
        ));
        output.push_str(&format!(
            "  Restarts:           {}\n",
            self.solver_stats.restarts
        ));
        if self.solver_stats.conflicts > 0 {
            output.push_str(&format!(
                "  Dec/Conflict:       {:.2}\n",
                self.solver_stats.avg_decisions_per_conflict()
            ));
        }
        output.push('\n');

        // Clause learning
        output.push_str("CLAUSE LEARNING:\n");
        output.push_str(&format!(
            "  Learned Clauses:    {}\n",
            self.solver_stats.learned_clauses
        ));
        output.push_str(&format!(
            "  Binary Clauses:     {}\n",
            self.solver_stats.binary_clauses
        ));
        output.push_str(&format!(
            "  Unit Clauses:       {}\n",
            self.solver_stats.unit_clauses
        ));
        output.push_str(&format!(
            "  Deleted Clauses:    {}\n",
            self.solver_stats.deleted_clauses
        ));
        if self.solver_stats.learned_clauses > 0 {
            output.push_str(&format!(
                "  Average LBD:        {:.2}\n",
                self.solver_stats.avg_lbd()
            ));
        }
        output.push('\n');

        // Minimization
        if self.solver_stats.minimizations > 0 {
            output.push_str("MINIMIZATION:\n");
            output.push_str(&format!(
                "  Minimizations:      {}\n",
                self.solver_stats.minimizations
            ));
            output.push_str(&format!(
                "  Literals Removed:   {}\n",
                self.solver_stats.literals_removed
            ));
            if self.solver_stats.minimizations > 0 {
                output.push_str(&format!(
                    "  Lits/Minimization:  {:.2}\n",
                    self.solver_stats.literals_removed as f64
                        / self.solver_stats.minimizations as f64
                ));
            }
            output.push('\n');
        }

        // Backtracking
        if self.solver_stats.chrono_backtracks > 0 || self.solver_stats.non_chrono_backtracks > 0 {
            output.push_str("BACKTRACKING:\n");
            output.push_str(&format!(
                "  Chronological:      {}\n",
                self.solver_stats.chrono_backtracks
            ));
            output.push_str(&format!(
                "  Non-chronological:  {}\n",
                self.solver_stats.non_chrono_backtracks
            ));
            let total =
                self.solver_stats.chrono_backtracks + self.solver_stats.non_chrono_backtracks;
            if total > 0 {
                let chrono_pct = 100.0 * self.solver_stats.chrono_backtracks as f64 / total as f64;
                output.push_str(&format!("  Chrono %:           {:.1}%\n", chrono_pct));
            }
            output.push('\n');
        }

        output.push_str("═══════════════════════════════════════════════════════════════\n");

        output
    }
}

impl fmt::Display for StatsDashboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_detailed())
    }
}

/// Statistics aggregator for multiple solver runs
#[derive(Debug, Default)]
pub struct StatsAggregator {
    /// Individual run statistics
    runs: Vec<StatsDashboard>,
}

impl StatsAggregator {
    /// Create a new aggregator
    #[must_use]
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    /// Add a run's statistics
    pub fn add_run(&mut self, stats: StatsDashboard) {
        self.runs.push(stats);
    }

    /// Get number of runs
    #[must_use]
    pub fn num_runs(&self) -> usize {
        self.runs.len()
    }

    /// Get average conflicts across all runs
    #[must_use]
    pub fn avg_conflicts(&self) -> f64 {
        if self.runs.is_empty() {
            return 0.0;
        }
        let total: u64 = self.runs.iter().map(|r| r.solver_stats().conflicts).sum();
        total as f64 / self.runs.len() as f64
    }

    /// Get average wall time across all runs
    #[must_use]
    pub fn avg_wall_time(&self) -> Duration {
        if self.runs.is_empty() {
            return Duration::ZERO;
        }
        let total_nanos: u128 = self.runs.iter().map(|r| r.wall_time().as_nanos()).sum();
        Duration::from_nanos((total_nanos / self.runs.len() as u128) as u64)
    }

    /// Get total conflicts across all runs
    #[must_use]
    pub fn total_conflicts(&self) -> u64 {
        self.runs.iter().map(|r| r.solver_stats().conflicts).sum()
    }

    /// Get total wall time across all runs
    #[must_use]
    pub fn total_wall_time(&self) -> Duration {
        let total_nanos: u128 = self.runs.iter().map(|r| r.wall_time().as_nanos()).sum();
        Duration::from_nanos(total_nanos as u64)
    }

    /// Display aggregate summary
    pub fn display_summary(&self) -> String {
        if self.runs.is_empty() {
            return "No runs recorded\n".to_string();
        }

        let mut output = String::new();

        output.push_str("═══════════════════════════════════════════════════════════════\n");
        output.push_str("                   AGGREGATE STATISTICS                        \n");
        output.push_str("═══════════════════════════════════════════════════════════════\n\n");

        output.push_str(&format!("Total Runs:         {}\n", self.num_runs()));
        output.push_str(&format!(
            "Average Conflicts:  {:.0}\n",
            self.avg_conflicts()
        ));
        output.push_str(&format!("Total Conflicts:    {}\n", self.total_conflicts()));
        output.push_str(&format!(
            "Average Time:       {}\n",
            StatsDashboard::format_duration(self.avg_wall_time())
        ));
        output.push_str(&format!(
            "Total Time:         {}\n",
            StatsDashboard::format_duration(self.total_wall_time())
        ));

        output.push_str("\n═══════════════════════════════════════════════════════════════\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_dashboard_creation() {
        let solver_stats = SolverStats::default();
        let dashboard = StatsDashboard::new(solver_stats, Duration::from_secs(1));

        assert_eq!(dashboard.wall_time(), Duration::from_secs(1));
        assert_eq!(dashboard.solver_stats().conflicts, 0);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(StatsDashboard::format_size(512), "512 B");
        assert_eq!(StatsDashboard::format_size(2048), "2.00 KB");
        assert_eq!(StatsDashboard::format_size(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(
            StatsDashboard::format_size(3 * 1024 * 1024 * 1024),
            "3.00 GB"
        );
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(
            StatsDashboard::format_duration(Duration::from_millis(500)),
            "0.500s"
        );
        assert_eq!(
            StatsDashboard::format_duration(Duration::from_secs(30)),
            "30.000s"
        );
        assert!(StatsDashboard::format_duration(Duration::from_secs(90)).contains("1m"));
    }

    #[test]
    fn test_rates_calculation() {
        let solver_stats = SolverStats {
            conflicts: 1000,
            decisions: 5000,
            propagations: 10000,
            ..Default::default()
        };

        let dashboard = StatsDashboard::new(solver_stats, Duration::from_secs(1));

        assert_eq!(dashboard.conflicts_per_second(), 1000.0);
        assert_eq!(dashboard.decisions_per_second(), 5000.0);
        assert_eq!(dashboard.props_per_second(), 10000.0);
    }

    #[test]
    fn test_compact_display() {
        let solver_stats = SolverStats::default();
        let dashboard = StatsDashboard::new(solver_stats, Duration::from_secs(1));

        let compact = dashboard.display_compact();
        assert!(compact.contains("Time:"));
        assert!(compact.contains("Conflicts:"));
        assert!(compact.contains("Decisions:"));
    }

    #[test]
    fn test_detailed_display() {
        let solver_stats = SolverStats::default();
        let dashboard = StatsDashboard::new(solver_stats, Duration::from_secs(1));

        let detailed = dashboard.display_detailed();
        assert!(detailed.contains("SOLVER STATISTICS"));
        assert!(detailed.contains("TIMING:"));
        assert!(detailed.contains("SEARCH:"));
    }

    #[test]
    fn test_stats_aggregator() {
        let mut aggregator = StatsAggregator::new();
        assert_eq!(aggregator.num_runs(), 0);

        let stats1 = SolverStats {
            conflicts: 100,
            ..Default::default()
        };
        let dashboard1 = StatsDashboard::new(stats1, Duration::from_secs(1));
        aggregator.add_run(dashboard1);

        let stats2 = SolverStats {
            conflicts: 200,
            ..Default::default()
        };
        let dashboard2 = StatsDashboard::new(stats2, Duration::from_secs(2));
        aggregator.add_run(dashboard2);

        assert_eq!(aggregator.num_runs(), 2);
        assert_eq!(aggregator.avg_conflicts(), 150.0);
        assert_eq!(aggregator.total_conflicts(), 300);
    }

    #[test]
    fn test_aggregator_summary() {
        let aggregator = StatsAggregator::new();
        let summary = aggregator.display_summary();
        assert!(summary.contains("No runs recorded"));

        let mut aggregator = StatsAggregator::new();
        let stats = SolverStats::default();
        let dashboard = StatsDashboard::new(stats, Duration::from_secs(1));
        aggregator.add_run(dashboard);

        let summary = aggregator.display_summary();
        assert!(summary.contains("AGGREGATE STATISTICS"));
        assert!(summary.contains("Total Runs:"));
    }
}
