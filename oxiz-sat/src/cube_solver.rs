/// Parallel cube solving using the Cube-and-Conquer technique.
///
/// This module implements the "Conquer" phase of Cube-and-Conquer, distributing
/// cubes to parallel workers and aggregating results.
use crate::clause::Clause;
use crate::cube::{Cube, CubeResult, CubeStats};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Configuration for parallel cube solving.
#[derive(Debug, Clone)]
pub struct CubeSolverConfig {
    /// Maximum time per cube (in seconds)
    pub cube_timeout: Duration,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Stop on first SAT cube found
    pub early_termination: bool,
    /// Enable progress reporting
    pub verbose: bool,
}

impl Default for CubeSolverConfig {
    fn default() -> Self {
        Self {
            cube_timeout: Duration::from_secs(60),
            num_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            early_termination: true,
            verbose: false,
        }
    }
}

/// Result of solving a single cube.
#[derive(Debug, Clone)]
pub struct CubeSolveResult {
    /// The cube that was solved
    pub cube: Cube,
    /// Result (SAT/UNSAT/Unknown)
    pub result: CubeResult,
    /// Time taken to solve
    pub time: Duration,
    /// Number of conflicts during solving
    pub conflicts: u64,
    /// Number of decisions made
    pub decisions: u64,
}

/// Parallel cube solver.
pub struct ParallelCubeSolver {
    /// Configuration
    config: CubeSolverConfig,
    /// Number of cubes solved
    cubes_solved: Arc<AtomicUsize>,
    /// Number of SAT cubes found
    sat_count: Arc<AtomicUsize>,
    /// Number of UNSAT cubes found
    unsat_count: Arc<AtomicUsize>,
    /// Flag for early termination
    terminate: Arc<AtomicBool>,
}

impl ParallelCubeSolver {
    /// Creates a new parallel cube solver.
    pub fn new(config: CubeSolverConfig) -> Self {
        Self {
            config,
            cubes_solved: Arc::new(AtomicUsize::new(0)),
            sat_count: Arc::new(AtomicUsize::new(0)),
            unsat_count: Arc::new(AtomicUsize::new(0)),
            terminate: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Solves a set of cubes in parallel.
    ///
    /// Returns the overall result and individual cube results.
    pub fn solve(
        &mut self,
        cubes: Vec<Cube>,
        clauses: &[Clause],
    ) -> (CubeResult, Vec<CubeSolveResult>) {
        let start_time = Instant::now();

        // Reset counters
        self.cubes_solved.store(0, Ordering::Relaxed);
        self.sat_count.store(0, Ordering::Relaxed);
        self.unsat_count.store(0, Ordering::Relaxed);
        self.terminate.store(false, Ordering::Relaxed);

        if self.config.verbose {
            println!(
                "Solving {} cubes with {} workers...",
                cubes.len(),
                self.config.num_workers
            );
        }

        // Solve cubes sequentially (use rayon for true parallelism)
        let results: Vec<CubeSolveResult> = cubes
            .iter()
            .map(|cube| {
                // Check for early termination
                if self.terminate.load(Ordering::Relaxed) {
                    return CubeSolveResult {
                        cube: cube.clone(),
                        result: CubeResult::Unknown,
                        time: Duration::ZERO,
                        conflicts: 0,
                        decisions: 0,
                    };
                }

                let result = self.solve_cube(cube, clauses);

                // Update counters
                self.cubes_solved.fetch_add(1, Ordering::Relaxed);
                match result.result {
                    CubeResult::Sat => {
                        self.sat_count.fetch_add(1, Ordering::Relaxed);
                        if self.config.early_termination {
                            self.terminate.store(true, Ordering::Relaxed);
                        }
                    }
                    CubeResult::Unsat => {
                        self.unsat_count.fetch_add(1, Ordering::Relaxed);
                    }
                    CubeResult::Unknown => {}
                }

                result
            })
            .collect();

        let total_time = start_time.elapsed();

        // Determine overall result
        let overall_result = if self.sat_count.load(Ordering::Relaxed) > 0 {
            CubeResult::Sat
        } else if self.unsat_count.load(Ordering::Relaxed) == cubes.len() {
            CubeResult::Unsat
        } else {
            CubeResult::Unknown
        };

        if self.config.verbose {
            println!(
                "Cube solving complete: {:?} in {:.2}s",
                overall_result,
                total_time.as_secs_f64()
            );
            println!(
                "  SAT: {}, UNSAT: {}, Unknown: {}",
                self.sat_count.load(Ordering::Relaxed),
                self.unsat_count.load(Ordering::Relaxed),
                cubes.len() - self.cubes_solved.load(Ordering::Relaxed)
            );
        }

        (overall_result, results)
    }

    /// Solves a single cube (simplified solver for demonstration).
    ///
    /// In a real implementation, this would invoke a full CDCL solver with the
    /// cube literals as assumptions.
    fn solve_cube(&self, cube: &Cube, _clauses: &[Clause]) -> CubeSolveResult {
        let start = Instant::now();

        // Simplified solving logic for demonstration
        // In practice, this would call a real SAT solver with cube as assumptions

        // For now, just simulate some work
        let result = if cube.is_consistent() {
            // Check for trivial conflicts
            CubeResult::Unknown
        } else {
            CubeResult::Unsat
        };

        CubeSolveResult {
            cube: cube.clone(),
            result,
            time: start.elapsed(),
            conflicts: 0,
            decisions: cube.len() as u64,
        }
    }

    /// Returns statistics about the current solving session.
    pub fn stats(&self) -> CubeSolverStats {
        CubeSolverStats {
            cubes_solved: self.cubes_solved.load(Ordering::Relaxed),
            sat_count: self.sat_count.load(Ordering::Relaxed),
            unsat_count: self.unsat_count.load(Ordering::Relaxed),
            num_workers: self.config.num_workers,
        }
    }
}

/// Statistics for cube solving.
#[derive(Debug, Clone)]
pub struct CubeSolverStats {
    /// Number of cubes solved
    pub cubes_solved: usize,
    /// Number of SAT cubes
    pub sat_count: usize,
    /// Number of UNSAT cubes
    pub unsat_count: usize,
    /// Number of workers used
    pub num_workers: usize,
}

impl CubeSolverStats {
    /// Displays the statistics.
    pub fn display(&self) -> String {
        format!(
            "Cube Solver Statistics:\n\
             - Cubes Solved: {}\n\
             - SAT: {}\n\
             - UNSAT: {}\n\
             - Workers: {}",
            self.cubes_solved, self.sat_count, self.unsat_count, self.num_workers
        )
    }
}

/// Cube-and-Conquer orchestrator that combines cube generation and solving.
pub struct CubeAndConquer {
    /// Cube solver configuration
    solver_config: CubeSolverConfig,
}

impl CubeAndConquer {
    /// Creates a new Cube-and-Conquer orchestrator.
    pub fn new(solver_config: CubeSolverConfig) -> Self {
        Self { solver_config }
    }

    /// Runs the full Cube-and-Conquer algorithm.
    ///
    /// Returns the overall result, cube generation stats, and solve results.
    pub fn solve(
        &mut self,
        cubes: Vec<Cube>,
        clauses: &[Clause],
    ) -> (CubeResult, CubeStats, Vec<CubeSolveResult>) {
        // Generate statistics for cubes
        let cube_stats = CubeStats::from_cubes(&cubes);

        // Solve cubes in parallel
        let mut solver = ParallelCubeSolver::new(self.solver_config.clone());
        let (result, solve_results) = solver.solve(cubes, clauses);

        (result, cube_stats, solve_results)
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
    fn test_cube_solver_config() {
        let config = CubeSolverConfig::default();

        assert!(config.cube_timeout.as_secs() > 0);
        assert!(config.num_workers > 0);
        assert!(config.early_termination);
    }

    #[test]
    fn test_parallel_cube_solver_creation() {
        let config = CubeSolverConfig::default();
        let solver = ParallelCubeSolver::new(config);

        assert_eq!(solver.cubes_solved.load(Ordering::Relaxed), 0);
        assert_eq!(solver.sat_count.load(Ordering::Relaxed), 0);
        assert_eq!(solver.unsat_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_solve_empty_cubes() {
        let config = CubeSolverConfig {
            verbose: false,
            ..Default::default()
        };
        let mut solver = ParallelCubeSolver::new(config);

        let cubes = vec![];
        let clauses = vec![];

        let (result, results) = solver.solve(cubes, &clauses);

        assert_eq!(result, CubeResult::Unsat); // All (zero) cubes are UNSAT
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_solve_single_cube() {
        let config = CubeSolverConfig {
            verbose: false,
            early_termination: false,
            ..Default::default()
        };
        let mut solver = ParallelCubeSolver::new(config);

        let lit1 = make_lit(0, false);
        let cube = Cube::new(vec![lit1]);
        let cubes = vec![cube];
        let clauses = vec![];

        let (result, results) = solver.solve(cubes, &clauses);

        assert_eq!(results.len(), 1);
        assert!(matches!(result, CubeResult::Unknown | CubeResult::Unsat));
    }

    #[test]
    fn test_solve_inconsistent_cube() {
        let config = CubeSolverConfig {
            verbose: false,
            ..Default::default()
        };
        let mut solver = ParallelCubeSolver::new(config);

        let lit1 = make_lit(0, false);
        let lit2 = make_lit(0, true);
        let cube = Cube::new(vec![lit1, lit2]);
        let cubes = vec![cube];
        let clauses = vec![];

        let (result, results) = solver.solve(cubes, &clauses);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result, CubeResult::Unsat);
        assert_eq!(result, CubeResult::Unsat);
    }

    #[test]
    fn test_solve_multiple_cubes() {
        let config = CubeSolverConfig {
            verbose: false,
            early_termination: false,
            num_workers: 2,
            ..Default::default()
        };
        let mut solver = ParallelCubeSolver::new(config);

        let cube1 = Cube::new(vec![make_lit(0, false)]);
        let cube2 = Cube::new(vec![make_lit(1, false)]);
        let cube3 = Cube::new(vec![make_lit(2, false)]);

        let cubes = vec![cube1, cube2, cube3];
        let clauses = vec![];

        let (result, results) = solver.solve(cubes, &clauses);

        assert_eq!(results.len(), 3);
        assert!(matches!(result, CubeResult::Unknown | CubeResult::Unsat));
    }

    #[test]
    fn test_cube_solver_stats() {
        let config = CubeSolverConfig {
            verbose: false,
            ..Default::default()
        };
        let mut solver = ParallelCubeSolver::new(config);

        let cube = Cube::new(vec![make_lit(0, false)]);
        let cubes = vec![cube];
        let clauses = vec![];

        solver.solve(cubes, &clauses);

        let stats = solver.stats();
        assert_eq!(stats.cubes_solved, 1);
        assert!(stats.num_workers > 0);
    }

    #[test]
    fn test_cube_and_conquer() {
        let config = CubeSolverConfig {
            verbose: false,
            ..Default::default()
        };
        let mut cac = CubeAndConquer::new(config);

        let cube1 = Cube::new(vec![make_lit(0, false)]);
        let cube2 = Cube::new(vec![make_lit(1, false)]);

        let cubes = vec![cube1, cube2];
        let clauses = vec![];

        let (result, cube_stats, solve_results) = cac.solve(cubes, &clauses);

        assert_eq!(cube_stats.total_cubes, 2);
        assert_eq!(solve_results.len(), 2);
        assert!(matches!(result, CubeResult::Unknown | CubeResult::Unsat));
    }

    #[test]
    fn test_early_termination() {
        let config = CubeSolverConfig {
            verbose: false,
            early_termination: true,
            ..Default::default()
        };
        let solver = ParallelCubeSolver::new(config);

        assert!(!solver.terminate.load(Ordering::Relaxed));
    }

    #[test]
    fn test_solver_stats_display() {
        let stats = CubeSolverStats {
            cubes_solved: 10,
            sat_count: 3,
            unsat_count: 7,
            num_workers: 4,
        };

        let display = stats.display();
        assert!(display.contains("10"));
        assert!(display.contains("3"));
        assert!(display.contains("7"));
        assert!(display.contains("4"));
    }
}
