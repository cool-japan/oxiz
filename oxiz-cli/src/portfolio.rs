//! Portfolio solving with parallel strategy execution
//!
//! This module implements parallel portfolio solving where multiple solver strategies
//! are executed concurrently, and the first one to find a solution wins.
//!
//! Strategies include:
//! - CDCL (Conflict-Driven Clause Learning)
//! - DPLL (Davis-Putnam-Logemann-Loveland)
//! - Local Search
//! - Simplification-heavy approach
//! - Theory-specialized solvers

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use oxiz_solver::Context;

use crate::Args;

/// Strategy configuration for portfolio solving
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Name of the strategy
    pub name: &'static str,
    /// Description of the strategy
    #[allow(dead_code)]
    pub description: &'static str,
    /// Options to set for this strategy
    pub options: Vec<(&'static str, &'static str)>,
}

impl StrategyConfig {
    /// Apply this strategy configuration to a context
    pub fn apply(&self, ctx: &mut Context) {
        for (key, value) in &self.options {
            ctx.set_option(key, value);
        }
    }
}

/// Result from a portfolio solver strategy
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    /// Name of the strategy that found the result
    pub strategy_name: String,
    /// The output from the solver
    pub output: Vec<String>,
    /// Time taken in milliseconds
    pub time_ms: u128,
}

/// Get default portfolio strategies
pub fn get_default_strategies() -> Vec<StrategyConfig> {
    vec![
        StrategyConfig {
            name: "CDCL-Aggressive",
            description: "Fast CDCL with frequent restarts",
            options: vec![
                ("strategy", "cdcl"),
                ("simplify", "true"),
                ("restarts", "frequent"),
                ("branching", "vsids"),
                ("clause-learning", "aggressive"),
            ],
        },
        StrategyConfig {
            name: "CDCL-Stable",
            description: "CDCL with moderate restarts",
            options: vec![
                ("strategy", "cdcl"),
                ("simplify", "true"),
                ("restarts", "moderate"),
                ("branching", "vmtf"),
                ("clause-learning", "moderate"),
            ],
        },
        StrategyConfig {
            name: "DPLL-Lookahead",
            description: "DPLL with lookahead heuristics",
            options: vec![
                ("strategy", "dpll"),
                ("simplify", "true"),
                ("lookahead", "true"),
                ("branching", "moms"),
            ],
        },
        StrategyConfig {
            name: "LocalSearch",
            description: "Local search for large instances",
            options: vec![
                ("strategy", "local-search"),
                ("simplify", "false"),
                ("max-tries", "1000000"),
                ("noise", "0.1"),
            ],
        },
        StrategyConfig {
            name: "Simplify-Heavy",
            description: "Heavy preprocessing and simplification",
            options: vec![
                ("strategy", "cdcl"),
                ("simplify", "true"),
                ("preprocessing", "aggressive"),
                ("elimination", "true"),
                ("subsumption", "true"),
                ("vivification", "true"),
            ],
        },
    ]
}

/// Run portfolio solving with multiple strategies in parallel
pub fn solve_portfolio(
    script: &str,
    args: &Args,
    logic: Option<&str>,
    _base_ctx: &Context,
    timeout_secs: u64,
) -> Result<PortfolioResult, String> {
    let strategies = get_default_strategies();
    let script = Arc::new(script.to_string());
    let (tx, rx): (Sender<PortfolioResult>, Receiver<PortfolioResult>) = channel();
    let solved = Arc::new(AtomicBool::new(false));
    let start_time = Instant::now();

    let mut handles = Vec::new();

    // Spawn a thread for each strategy
    for strategy in strategies {
        let tx = tx.clone();
        let script = Arc::clone(&script);
        let solved = Arc::clone(&solved);
        let logic = logic.map(|s| s.to_string());
        let args_clone = args.clone();

        let handle = thread::spawn(move || {
            // Create a new context for this strategy
            let mut ctx = Context::new();

            // Set logic if provided
            if let Some(ref logic_str) = logic {
                ctx.set_logic(logic_str);
            }

            // Apply strategy-specific configuration first
            strategy.apply(&mut ctx);

            // Apply additional args-based options (resource limits, etc.)
            // But skip strategy-related options since we already applied our portfolio strategy
            let mut modified_args = args_clone.clone();
            modified_args.strategy = None; // Don't override our portfolio strategy
            crate::apply_solver_options(&mut ctx, &modified_args);

            let strategy_start = Instant::now();

            // Try to solve with this strategy
            match ctx.execute_script(&script) {
                Ok(output) => {
                    // Check if we're the first to finish
                    if !solved.swap(true, Ordering::SeqCst) {
                        let result = PortfolioResult {
                            strategy_name: strategy.name.to_string(),
                            output,
                            time_ms: strategy_start.elapsed().as_millis(),
                        };
                        let _ = tx.send(result);
                    }
                }
                Err(_) => {
                    // Strategy failed or timed out, ignore
                }
            }
        });

        handles.push(handle);
    }

    // Drop the original sender so the channel closes when all threads finish
    drop(tx);

    // Wait for the first result or timeout
    let timeout = if timeout_secs > 0 {
        Duration::from_secs(timeout_secs)
    } else {
        Duration::from_secs(300) // Default 5 minute timeout
    };

    // Note: threads will naturally finish or be interrupted when the program ends
    // We don't force-kill them to avoid unsafe operations

    if let Ok(result) = rx.recv_timeout(timeout) {
        // Mark as solved to stop other threads
        solved.store(true, Ordering::SeqCst);
        Ok(result)
    } else if start_time.elapsed() >= timeout {
        Err("Portfolio solving timed out".to_string())
    } else {
        Err("All strategies failed".to_string())
    }
}

/// Run portfolio solving with custom strategies
#[allow(dead_code)]
pub fn solve_portfolio_custom(
    script: &str,
    strategies: Vec<StrategyConfig>,
    args: &Args,
    logic: Option<&str>,
    _base_ctx: &Context,
    timeout_secs: u64,
) -> Result<PortfolioResult, String> {
    if strategies.is_empty() {
        return Err("No strategies provided".to_string());
    }

    let script = Arc::new(script.to_string());
    let (tx, rx): (Sender<PortfolioResult>, Receiver<PortfolioResult>) = channel();
    let solved = Arc::new(AtomicBool::new(false));
    let start_time = Instant::now();

    let mut handles = Vec::new();

    for strategy in strategies {
        let tx = tx.clone();
        let script = Arc::clone(&script);
        let solved = Arc::clone(&solved);
        let logic = logic.map(|s| s.to_string());
        let args_clone = args.clone();

        let handle = thread::spawn(move || {
            let mut ctx = Context::new();

            if let Some(ref logic_str) = logic {
                ctx.set_logic(logic_str);
            }

            strategy.apply(&mut ctx);

            let mut modified_args = args_clone.clone();
            modified_args.strategy = None;
            crate::apply_solver_options(&mut ctx, &modified_args);

            let strategy_start = Instant::now();

            if let Ok(output) = ctx.execute_script(&script) {
                if !solved.swap(true, Ordering::SeqCst) {
                    let result = PortfolioResult {
                        strategy_name: strategy.name.to_string(),
                        output,
                        time_ms: strategy_start.elapsed().as_millis(),
                    };
                    let _ = tx.send(result);
                }
            }
        });

        handles.push(handle);
    }

    drop(tx);

    let timeout = if timeout_secs > 0 {
        Duration::from_secs(timeout_secs)
    } else {
        Duration::from_secs(300)
    };

    if let Ok(result) = rx.recv_timeout(timeout) {
        solved.store(true, Ordering::SeqCst);
        Ok(result)
    } else if start_time.elapsed() >= timeout {
        Err("Portfolio solving timed out".to_string())
    } else {
        Err("All strategies failed".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_config() {
        let strategies = get_default_strategies();
        assert!(!strategies.is_empty());
        assert!(strategies.len() >= 3);

        // Check that each strategy has a name and options
        for strategy in strategies {
            assert!(!strategy.name.is_empty());
            assert!(!strategy.options.is_empty());
        }
    }

    #[test]
    fn test_strategy_apply() {
        let mut ctx = Context::new();
        let strategy = StrategyConfig {
            name: "test",
            description: "test strategy",
            options: vec![("strategy", "cdcl"), ("simplify", "true")],
        };

        strategy.apply(&mut ctx);
        // The options should be set (we can't directly verify without Context API)
    }
}
