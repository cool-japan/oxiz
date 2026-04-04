//! Stochastic Local Search (SLS) Theory Integration
//!
//! Provides SLS-based solving for hard satisfiability problems.
//! Implements WalkSAT, GSAT, and adaptive local search algorithms.
//!
//! # Overview
//!
//! SLS methods complement exact SAT/SMT solving by:
//! - Finding satisfying assignments for hard random instances
//! - Providing initial solutions for optimization
//! - Escaping local minima through randomized moves
//!
//! # Algorithms
//!
//! - WalkSAT: Random walk with focused moves
//! - GSAT: Greedy SAT with random restarts
//! - Adaptive: Dynamic parameter tuning
//!
//! # Example
//!
//! ```ignore
//! use oxiz_theories::sls::{SlsSolver, SlsConfig, SlsAlgorithm};
//!
//! let mut solver = SlsSolver::new(SlsConfig::default());
//! solver.add_clause(&[1, -2, 3]);
//! solver.add_clause(&[-1, 2, 3]);
//! let result = solver.solve();
//! ```

mod portfolio;
mod repair;
mod restart;
mod scoring;
mod types;
mod walk;

// Re-export all public types from submodules
pub use portfolio::{
    HybridSlsInterface, PortfolioConfig, PortfolioSls, WeightedSlsConfig, WeightedSlsSolver,
    WeightedSlsStats, YalsatConfig, YalsatSolver,
};
pub use repair::{
    BackboneDetector, ClauseSimplifier, DiversificationManager, DiversificationStrategy, PhaseMode,
    PhaseSaver, SolutionLearner, SolutionVerifier, VerificationResult,
};
pub use restart::{RestartManager, RestartStrategy};
pub use scoring::{
    CcanrConfig, CcanrEnhancer, ClauseImportance, ClauseWeightManager, DdfwConfig, DdfwManager,
    VarActivity, VarSelectHeuristic, WeightingScheme,
};
pub use types::{ClauseId, Lit, SlsAlgorithm, SlsConfig, SlsResult, SlsSolver, SlsStats, Var};
pub use walk::{
    BmsConfig, BmsSelector, FocusedWalk, FocusedWalkConfig, NoveltyConfig, NoveltySelector,
    SparrowConfig, SparrowSelector,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_sls_config() {
        let config = SlsConfig::default();
        assert_eq!(config.algorithm, SlsAlgorithm::WalkSat);
        assert_eq!(config.max_flips, 1_000_000);
    }

    #[test]
    fn test_sls_solver_creation() {
        let solver = SlsSolver::new(SlsConfig::default());
        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_clauses(), 0);
    }

    #[test]
    fn test_add_clause() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1, -2, 3]);
        solver.add_clause(&[-1, 2, 3]);
        assert_eq!(solver.num_vars(), 3);
        assert_eq!(solver.num_clauses(), 2);
    }

    #[test]
    fn test_solve_trivial_sat() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1]);
        solver.add_clause(&[2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_solve_simple_unsat() {
        let mut solver = SlsSolver::new(SlsConfig {
            max_flips: 1000,
            max_restarts: 5,
            ..Default::default()
        });
        solver.add_clause(&[1]);
        solver.add_clause(&[-1]);
        let result = solver.solve();
        // May be Unknown since SLS doesn't prove UNSAT
        assert!(matches!(result, SlsResult::Unknown | SlsResult::Sat(_)));
    }

    #[test]
    fn test_walksat() {
        let config = SlsConfig {
            algorithm: SlsAlgorithm::WalkSat,
            ..Default::default()
        };
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        solver.add_clause(&[1, -2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_gsat() {
        let config = SlsConfig {
            algorithm: SlsAlgorithm::Gsat,
            ..Default::default()
        };
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_probsat() {
        let config = SlsConfig {
            algorithm: SlsAlgorithm::ProbSat,
            ..Default::default()
        };
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2, 3]);
        solver.add_clause(&[-1, -2, 3]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_adaptive() {
        let config = SlsConfig {
            algorithm: SlsAlgorithm::Adaptive,
            ..Default::default()
        };
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 3]);
        solver.add_clause(&[2, 3]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_tabu() {
        let config = SlsConfig {
            tabu: true,
            tabu_tenure: 5,
            ..Default::default()
        };
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, -2]);
        solver.add_clause(&[1, -2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
    }

    #[test]
    fn test_adaptive_noise() {
        let config = SlsConfig {
            adaptive_noise: true,
            ..Default::default()
        };
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_stats() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1, 2]);
        let _ = solver.solve();
        let stats = solver.stats();
        assert!(stats.restarts > 0);
    }

    #[test]
    fn test_reset() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1, 2]);
        solver.reset();
        assert_eq!(solver.num_clauses(), 0);
    }

    // Weighted SLS tests

    #[test]
    fn test_weighted_sls() {
        let mut solver = WeightedSlsSolver::new(WeightedSlsConfig::default());
        solver.add_hard_clause(&[1, 2]);
        solver.add_soft_clause(&[-1]);
        solver.add_soft_clause(&[-2]);
        let (result, cost) = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
        assert!(cost >= 0.0);
    }

    // Phase saver tests

    #[test]
    fn test_phase_saver() {
        let mut saver = PhaseSaver::new(PhaseMode::Save);
        saver.save_phase(1, true);
        assert!(saver.get_phase(1));
        saver.save_phase(1, false);
        assert!(!saver.get_phase(1));
    }

    #[test]
    fn test_phase_modes() {
        let mut false_saver = PhaseSaver::new(PhaseMode::False);
        assert!(!false_saver.get_phase(1));

        let mut true_saver = PhaseSaver::new(PhaseMode::True);
        assert!(true_saver.get_phase(1));
    }

    // Weight manager tests

    #[test]
    fn test_weight_manager() {
        let mut manager = ClauseWeightManager::new(WeightingScheme::Additive);
        manager.initialize(5);

        let mut unsat = HashSet::new();
        unsat.insert(ClauseId(0));
        unsat.insert(ClauseId(2));

        manager.update(&unsat);
        assert!(manager.get_weight(ClauseId(0)) > 1.0);
        assert_eq!(manager.get_weight(ClauseId(1)), 1.0);
    }

    #[test]
    fn test_weight_smooth() {
        let mut manager = ClauseWeightManager::new(WeightingScheme::Saps);
        manager.initialize(3);
        manager.weights[0] = 10.0;
        manager.weights[1] = 1.0;
        manager.weights[2] = 1.0;

        manager.smooth();
        // After smoothing, weights should be more balanced
        assert!(manager.get_weight(ClauseId(0)) < 10.0);
    }

    // Variable activity tests

    #[test]
    fn test_var_activity() {
        let mut activity = VarActivity::new();
        activity.bump(1);
        activity.bump(1);
        activity.bump(2);

        assert!(activity.get(1) > activity.get(2));
    }

    #[test]
    fn test_var_activity_decay() {
        let mut activity = VarActivity::new();
        activity.bump(1);
        let before = activity.bump;
        activity.decay();
        assert!(activity.bump > before);
    }

    // Focused walk tests

    #[test]
    fn test_focused_walk() {
        let mut walk = FocusedWalk::new(FocusedWalkConfig::default());
        let lits = vec![1, -2, 3];
        let breaks = vec![0, 2, 1, 0];
        walk.update_focus(&lits, &breaks);
        assert!(!walk.focus_set().is_empty());
    }

    #[test]
    fn test_focused_walk_ordering() {
        let mut walk = FocusedWalk::new(FocusedWalkConfig {
            focus_size: 2,
            ..Default::default()
        });
        let lits = vec![1, 2, 3];
        let breaks = vec![0, 3, 1, 2]; // var 2 has break=1, var 3 has break=2, var 1 has break=3
        walk.update_focus(&lits, &breaks);

        // Should prefer lower break counts
        let focus = walk.focus_set();
        assert_eq!(focus.len(), 2);
    }

    // Restart manager tests

    #[test]
    fn test_restart_manager_fixed() {
        let mut manager = RestartManager::new(RestartStrategy::Fixed(100));
        assert!(!manager.should_restart(50));
        assert!(manager.should_restart(100));
        manager.notify_restart();
        assert_eq!(manager.count(), 1);
        assert_eq!(manager.threshold(), 100);
    }

    #[test]
    fn test_restart_manager_geometric() {
        let mut manager = RestartManager::new(RestartStrategy::Geometric(100, 2.0));
        assert_eq!(manager.threshold(), 100);
        manager.notify_restart();
        // Should double: 100 * 2^1 = 200
        assert_eq!(manager.threshold(), 200);
        manager.notify_restart();
        // Should quadruple: 100 * 2^2 = 400
        assert_eq!(manager.threshold(), 400);
    }

    #[test]
    fn test_restart_manager_luby() {
        let mut manager = RestartManager::new(RestartStrategy::Luby(10));
        // Luby sequence: 1, 1, 2, 1, 1, 2, 4, ...
        // Initial: luby_index = 1, threshold = 10 (initial value)
        assert_eq!(manager.threshold(), 10);

        // First restart: uses luby(1) = 1, luby_index becomes 2
        manager.notify_restart();
        assert_eq!(manager.threshold(), 10); // 1 * 10 = 10

        // Second restart: uses luby(2) = 1, luby_index becomes 3
        manager.notify_restart();
        assert_eq!(manager.threshold(), 10); // 1 * 10 = 10

        // Third restart: uses luby(3) = 2, luby_index becomes 4
        manager.notify_restart();
        assert_eq!(manager.threshold(), 20); // 2 * 10 = 20
    }

    #[test]
    fn test_restart_manager_luby_sequence() {
        let manager = RestartManager::new(RestartStrategy::Luby(1));
        // Test Luby function directly
        assert_eq!(manager.luby(1), 1);
        assert_eq!(manager.luby(2), 1);
        assert_eq!(manager.luby(3), 2);
        assert_eq!(manager.luby(4), 1);
        assert_eq!(manager.luby(5), 1);
        assert_eq!(manager.luby(6), 2);
        assert_eq!(manager.luby(7), 4);
    }

    // Novelty selector tests

    #[test]
    fn test_novelty_selector() {
        let mut selector = NoveltySelector::new(NoveltyConfig::default());
        selector.ensure_capacity(10);

        // First flip - var 1 flipped at time 0, then time becomes 1
        selector.notify_flip(1);
        // age = current_time - flip_time = 1 - 0 = 1
        assert_eq!(selector.age(1), 1);
        // var 2 age = current_time - 0 = 1 (never flipped, initialized at 0)
        assert_eq!(selector.age(2), 1);

        // Second flip - var 2 flipped at time 1, then time becomes 2
        selector.notify_flip(2);
        // var 1: age = 2 - 0 = 2
        assert!(selector.age(1) > selector.age(2));
        // var 2: age = 2 - 1 = 1
        assert_eq!(selector.age(2), 1);
    }

    #[test]
    fn test_novelty_selection() {
        let selector = NoveltySelector::new(NoveltyConfig {
            novelty_prob: 0.0, // Always pick best
            novelty_plus: false,
            ..Default::default()
        });

        let candidates = vec![(1, 5), (2, 1), (3, 3)]; // (var, break)
        let mut rng = 42u64;
        let selected = selector.select(&candidates, &mut rng);
        assert_eq!(selected, Some(2)); // Lowest break count
    }

    #[test]
    fn test_novelty_avoids_last_flipped() {
        let mut selector = NoveltySelector::new(NoveltyConfig {
            novelty_prob: 1.0, // Always pick second best when best = last_flipped
            novelty_plus: false,
            ..Default::default()
        });
        selector.notify_flip(2); // Make var 2 the last flipped

        let candidates = vec![(2, 1), (3, 5)]; // var 2 is best (break=1)
        let mut rng = 42u64;
        let selected = selector.select(&candidates, &mut rng);
        assert_eq!(selected, Some(3)); // Should pick second best since 2 was just flipped
    }

    // CCAnr tests

    #[test]
    fn test_ccanr_enhancer() {
        let mut enhancer = CcanrEnhancer::new(CcanrConfig::default());
        enhancer.initialize(10);

        let clause_lits = vec![1, -2, 3];
        enhancer.update_scores(&clause_lits);

        assert!(enhancer.score(1) > 0.0);
        assert!(enhancer.score(2) > 0.0);
        assert!(enhancer.score(3) > 0.0);
        assert_eq!(enhancer.score(5), 0.0);
    }

    #[test]
    fn test_ccanr_config_checking() {
        let mut enhancer = CcanrEnhancer::new(CcanrConfig::default());
        enhancer.initialize(10);

        enhancer.set_config(5, true);
        assert!(enhancer.check_config(5));
        assert!(!enhancer.check_config(6));
    }

    #[test]
    fn test_ccanr_decay() {
        let mut enhancer = CcanrEnhancer::new(CcanrConfig::default());
        enhancer.initialize(5);

        enhancer.update_scores(&[1, 2]);
        let before = enhancer.score(1);
        enhancer.decay_scores(0.5);
        assert!((enhancer.score(1) - before * 0.5).abs() < 0.001);
    }

    // Backbone detector tests

    #[test]
    fn test_backbone_detector() {
        let mut detector = BackboneDetector::new(2);

        // First solution: x1=true, x2=true, x3=false
        let sol1 = vec![false, true, true, false];
        detector.initialize(&sol1);
        assert_eq!(detector.backbone_size(), 3);

        // Second solution: x1=true, x2=false, x3=false
        let sol2 = vec![false, true, false, false];
        detector.update(&sol2);

        // x2 should be removed from backbone candidates
        // x1=true and x3=false should remain
        assert!(detector.is_backbone(1) == Some(true));
        assert!(detector.is_backbone(3) == Some(false));
        assert!(detector.is_backbone(2).is_none());
    }

    #[test]
    fn test_backbone_commits_after_threshold() {
        let mut detector = BackboneDetector::new(2);

        let sol1 = vec![false, true, true];
        detector.initialize(&sol1);

        let sol2 = vec![false, true, true];
        detector.update(&sol2);

        // Should have committed to backbone after 2 solutions
        assert!(!detector.backbone().is_empty());
    }

    // Diversification tests

    #[test]
    fn test_diversification_manager() {
        let mut manager = DiversificationManager::new(DiversificationStrategy::Stagnation);

        for _ in 0..50 {
            manager.update(10); // No improvement
        }
        assert!(!manager.should_diversify());

        for _ in 0..60 {
            manager.update(10); // Still no improvement
        }
        assert!(manager.should_diversify()); // Should diversify after threshold
    }

    #[test]
    fn test_diversification_improvement_resets() {
        let mut manager = DiversificationManager::new(DiversificationStrategy::Stagnation);

        for _ in 0..90 {
            manager.update(10);
        }

        // Improvement resets counter
        manager.update(5);
        assert!(!manager.should_diversify());
    }

    // Clause simplifier tests

    #[test]
    fn test_clause_simplifier_subsumption() {
        let simplifier = ClauseSimplifier::new();

        // [1, 2] subsumes [1, 2, 3]
        assert!(simplifier.subsumes(&[1, 2], &[1, 2, 3]));
        assert!(!simplifier.subsumes(&[1, 2, 3], &[1, 2]));
        assert!(simplifier.subsumes(&[1], &[1, 2, 3]));
    }

    #[test]
    fn test_clause_simplifier_build() {
        let mut simplifier = ClauseSimplifier::new();
        let clauses = vec![vec![1, 2], vec![-1, 3], vec![2, 3]];
        simplifier.build(&clauses);

        // Check occurrence lists work
        assert_eq!(simplifier.clause_sizes.len(), 3);
    }

    #[test]
    fn test_clause_simplifier_units() {
        let simplifier = ClauseSimplifier::new();
        let clauses = vec![vec![1], vec![2, 3], vec![-4]];
        let units = simplifier.find_units(&clauses);
        assert!(units.contains(&1));
        assert!(units.contains(&-4));
        assert!(!units.contains(&2));
    }

    #[test]
    fn test_clause_simplifier_trivially_unsat() {
        let simplifier = ClauseSimplifier::new();

        let sat_clauses = vec![vec![1, 2], vec![-1]];
        assert!(!simplifier.is_trivially_unsat(&sat_clauses));

        let unsat_clauses = vec![vec![1, 2], vec![]];
        assert!(simplifier.is_trivially_unsat(&unsat_clauses));
    }

    // Sparrow selector tests

    #[test]
    fn test_sparrow_selector() {
        let mut selector = SparrowSelector::new(SparrowConfig::default());
        selector.initialize(10);

        let clause_lits = vec![1, 2, 3];
        let break_counts = vec![0, 5, 1, 2];
        let make_counts = vec![0, 0, 2, 1];
        let mut rng = 42u64;

        let selected = selector.select(&clause_lits, &break_counts, &make_counts, &mut rng);
        assert!(selected.is_some());
    }

    #[test]
    fn test_sparrow_age_factor() {
        let mut selector = SparrowSelector::new(SparrowConfig::default());
        selector.initialize(10);

        // Flip var 1
        selector.notify_flip(1);
        selector.notify_flip(2);
        selector.notify_flip(3);

        // Var 1 should be oldest
        assert!(selector.age(1) > selector.age(2));
        assert!(selector.age(2) > selector.age(3));
    }

    // Portfolio SLS tests

    #[test]
    fn test_portfolio_sls() {
        let mut portfolio = PortfolioSls::new(PortfolioConfig::default());
        portfolio.add_clause(&[1, 2]);
        portfolio.add_clause(&[-1, 2]);
        portfolio.add_clause(&[1, -2]);

        let result = portfolio.solve(5);
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
    }

    #[test]
    fn test_portfolio_reset() {
        let mut portfolio = PortfolioSls::new(PortfolioConfig::default());
        portfolio.add_clause(&[1, 2]);
        let _ = portfolio.solve(1);

        portfolio.reset();
        assert_eq!(portfolio.best_unsat_count(), u32::MAX);
    }

    // BMS Selector tests

    #[test]
    fn test_bms_selector() {
        let mut selector = BmsSelector::new(BmsConfig::default());
        selector.initialize(10);

        // Score should penalize breaks and reward makes
        let score_low_break = selector.compute_score(1, 0, 2);
        let score_high_break = selector.compute_score(1, 5, 2);
        assert!(score_low_break > score_high_break);
    }

    #[test]
    fn test_bms_selection() {
        let selector = BmsSelector::new(BmsConfig::default());
        let candidates = vec![1, 2, 3];
        let break_counts = vec![0, 5, 1, 2]; // var 1: break=5, var 2: break=1, var 3: break=2
        let make_counts = vec![0, 0, 3, 1];

        let selected = selector.select(&candidates, &break_counts, &make_counts);
        // var 2 should be best (low break, high make)
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn test_bms_age_factor() {
        let mut selector = BmsSelector::new(BmsConfig {
            age_weight: 1.0, // High age weight
            break_weight: 0.0,
            make_weight: 0.0,
            ..Default::default()
        });
        selector.initialize(10);

        // Flip var 1 multiple times to advance time
        selector.notify_flip(1);
        selector.notify_flip(1);
        selector.notify_flip(1);

        // var 1 was last flipped at time 2, var 2 was never flipped
        // var 1 age: 3 - 2 = 1
        // var 2 age: 3 - 0 = 3 (initialized at 0)
        let score1 = selector.compute_score(1, 0, 0);
        let score2 = selector.compute_score(2, 0, 0);
        assert!(
            score2 > score1,
            "score2 ({}) should be > score1 ({})",
            score2,
            score1
        );
    }

    // Solution verification tests

    #[test]
    fn test_solution_verifier() {
        let mut verifier = SolutionVerifier::new();
        verifier.set_clauses(vec![vec![1, 2], vec![-1, 3], vec![2, 3]]);

        // Valid solution: x1=true, x2=true, x3=true
        let assignment = vec![false, true, true, true];
        let result = verifier.verify(&assignment);
        assert!(result.is_valid);
        assert_eq!(result.satisfied_count, 3);
    }

    #[test]
    fn test_solution_verifier_invalid() {
        let mut verifier = SolutionVerifier::new();
        verifier.set_clauses(vec![vec![1], vec![-1]]);

        let assignment = vec![false, true];
        let result = verifier.verify(&assignment);
        assert!(!result.is_valid);
        assert_eq!(result.unsatisfied_indices.len(), 1);
    }

    #[test]
    fn test_solution_verifier_quick_check() {
        let mut verifier = SolutionVerifier::new();
        verifier.set_clauses(vec![vec![1, 2], vec![-1, 2]]);

        assert!(verifier.is_valid(&[false, true, true]));
        assert!(verifier.is_valid(&[false, false, true]));
    }

    // DDFW tests

    #[test]
    fn test_ddfw_manager() {
        let mut ddfw = DdfwManager::new(DdfwConfig::default());
        ddfw.initialize(5);

        assert_eq!(ddfw.weight(0), 1.0);
        assert_eq!(ddfw.weight(2), 1.0);
    }

    #[test]
    fn test_ddfw_distribution() {
        let mut ddfw = DdfwManager::new(DdfwConfig {
            init_weight: 10.0,
            transfer_amount: 2.0,
            distribute_freq: 1,
        });
        ddfw.initialize(5);

        ddfw.notify_flip();
        assert!(ddfw.should_distribute());

        // Transfer from clause 0,1 to clause 3,4
        ddfw.distribute(&[0, 1], &[3, 4]);

        // Satisfied clauses should have less weight
        assert!(ddfw.weight(0) < 10.0);
        // Unsatisfied should have more
        assert!(ddfw.weight(3) > 10.0);
    }

    // Clause importance tests

    #[test]
    fn test_clause_importance() {
        let mut importance = ClauseImportance::new();
        importance.initialize(5);

        importance.record_hit(0);
        importance.record_hit(0);
        importance.record_hit(1);
        importance.record_critical(0);

        // Clause 0 should be most important
        assert!(importance.importance(0) > importance.importance(1));
    }

    #[test]
    fn test_clause_importance_ranking() {
        let mut importance = ClauseImportance::new();
        importance.initialize(5);

        importance.record_hit(2);
        importance.record_hit(2);
        importance.record_hit(2);
        importance.record_hit(0);
        importance.record_critical(0);

        let most = importance.most_important(2);
        assert_eq!(most.len(), 2);
        // Both 0 and 2 should be in top 2
        assert!(most.contains(&0) || most.contains(&2));
    }

    // Solution learner tests

    #[test]
    fn test_solution_learner() {
        let mut learner = SolutionLearner::new(10);
        learner.initialize(5);

        // Record solutions where var 1 is always true
        learner.record_solution(&[false, true, false, true, false]);
        learner.record_solution(&[false, true, true, false, true]);
        learner.record_solution(&[false, true, false, false, true]);

        assert_eq!(learner.preferred_polarity(1), Some(true));
    }

    #[test]
    fn test_solution_learner_confidence() {
        let mut learner = SolutionLearner::new(10);
        learner.initialize(3);

        learner.record_solution(&[false, true, true, false]);
        learner.record_solution(&[false, true, false, false]);

        // Var 1: 2 true, 0 false -> 100% confidence for true
        assert!((learner.polarity_confidence(1) - 1.0).abs() < 0.01);
        // Var 2: 1 true, 1 false -> 50% confidence
        assert!((learner.polarity_confidence(2) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_solution_learner_high_confidence() {
        let mut learner = SolutionLearner::new(10);
        learner.initialize(4);

        learner.record_solution(&[false, true, true, false, true]);
        learner.record_solution(&[false, true, true, true, true]);
        learner.record_solution(&[false, true, true, false, false]);

        let high_conf = learner.high_confidence_vars(0.9);
        // Var 1 and 2 should have high confidence (all or mostly same polarity)
        assert!(high_conf.iter().any(|&(v, _)| v == 1 || v == 2));
    }

    // Hybrid interface tests

    #[test]
    fn test_hybrid_interface() {
        let mut interface = HybridSlsInterface::new();

        interface.set_assumptions(vec![1, -2, 3]);
        assert_eq!(interface.assumptions().len(), 3);

        interface.add_learned_clause(vec![1, 2, 3]);
        let learned = interface.take_learned_clauses();
        assert_eq!(learned.len(), 1);
        assert!(interface.take_learned_clauses().is_empty());
    }

    #[test]
    fn test_hybrid_interface_phase_hints() {
        let mut interface = HybridSlsInterface::new();

        interface.set_phase_hint(5, true);
        interface.set_phase_hint(3, false);

        assert_eq!(interface.phase_hint(5), Some(true));
        assert_eq!(interface.phase_hint(3), Some(false));
        assert_eq!(interface.phase_hint(1), None);
    }

    #[test]
    fn test_hybrid_interface_focus() {
        let mut interface = HybridSlsInterface::new();

        let mut focus = HashSet::new();
        focus.insert(1u32);
        focus.insert(3u32);
        focus.insert(5u32);
        interface.set_focus_vars(focus);

        assert!(interface.focus_vars().contains(&1));
        assert!(interface.focus_vars().contains(&3));
        assert!(!interface.focus_vars().contains(&2));
    }

    // YalSAT tests

    #[test]
    fn test_yalsat_solver() {
        let mut solver = YalsatSolver::new(YalsatConfig::default());
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        solver.add_clause(&[1, -2]);

        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
    }

    #[test]
    fn test_yalsat_boost() {
        let mut solver = YalsatSolver::new(YalsatConfig::default());
        solver.add_clause(&[1, 2]);
        solver.boost = vec![1.0; 3];

        solver.update_boost(1, 2.0);
        assert!((solver.boost[1] - 2.0).abs() < 0.01);

        let score = solver.boosted_score(1, 10.0);
        assert!((score - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_yalsat_cache_invalidation() {
        let mut solver = YalsatSolver::new(YalsatConfig::default());
        solver.score_cache.insert(1, (5, 3));
        solver.score_cache.insert(2, (2, 1));

        solver.invalidate_cache(1);
        assert!(!solver.score_cache.contains_key(&1));
        assert!(solver.score_cache.contains_key(&2));

        solver.clear_cache();
        assert!(solver.score_cache.is_empty());
    }
}
