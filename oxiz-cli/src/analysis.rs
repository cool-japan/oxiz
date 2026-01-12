//! Query complexity analysis and problem classification module
//!
//! This module provides tools for analyzing SMT-LIB2 queries to determine their complexity,
//! classify the problem type, and provide solver recommendations. It includes functionality
//! for automatic tuning based on problem characteristics and syntax validation.

use oxiz_solver::Context;
use serde::Serialize;
use std::collections::HashMap;

/// Query complexity analysis results
#[derive(Debug, Clone, Serialize)]
pub struct ComplexityAnalysis {
    pub declarations: usize,
    pub assertions: usize,
    pub commands: usize,
    pub max_nesting_depth: usize,
    pub avg_nesting_depth: f64,
    pub quantifiers: usize,
    pub operators: HashMap<String, usize>,
    pub theories: Vec<String>,
    pub estimated_difficulty: String,
    pub recommended_strategy: String,
    pub recommended_timeout: u64,
}

/// Problem classification based on SMT-LIB2 logic and structure
#[derive(Debug, Clone, Serialize)]
pub struct ProblemClassification {
    pub logic: String,
    pub is_quantifier_free: bool,
    pub primary_theory: String,
    pub complexity_class: String,
    pub solver_recommendations: Vec<String>,
}

/// Analyze SMT-LIB2 query complexity
pub fn analyze_query_complexity(script: &str) -> ComplexityAnalysis {
    let mut declarations = 0;
    let mut assertions = 0;
    let mut commands = 0;
    let mut max_depth = 0;
    let mut total_depth = 0;
    let mut depth_count = 0;
    let mut quantifiers = 0;
    let mut operators: HashMap<String, usize> = HashMap::new();
    let mut theories = Vec::new();

    // Track nesting depth
    let mut current_depth: usize = 0;
    let mut in_comment = false;
    let mut in_string = false;

    // Parse script for analysis
    for line in script.lines() {
        let trimmed = line.trim();

        // Skip comments
        if trimmed.starts_with(';') || in_comment {
            in_comment = trimmed.contains(';') && !trimmed.contains('\n');
            continue;
        }

        // Count declarations
        if trimmed.contains("declare-const")
            || trimmed.contains("declare-fun")
            || trimmed.contains("declare-sort")
        {
            declarations += 1;
            commands += 1;
        }

        // Count assertions
        if trimmed.contains("(assert") {
            assertions += 1;
            commands += 1;
        }

        // Count other commands
        if trimmed.contains("(check-sat")
            || trimmed.contains("(get-model")
            || trimmed.contains("(push")
            || trimmed.contains("(pop")
        {
            commands += 1;
        }

        // Count quantifiers
        if trimmed.contains("forall") || trimmed.contains("exists") {
            quantifiers += 1;
        }

        // Detect theories
        if (trimmed.contains("Int")
            || trimmed.contains("(+")
            || trimmed.contains("(- ")
            || trimmed.contains("(*")
            || trimmed.contains("(<")
            || trimmed.contains("(>")
            || trimmed.contains("(<=")
            || trimmed.contains("(>="))
            && !theories.contains(&"Arithmetic".to_string())
        {
            theories.push("Arithmetic".to_string());
        }
        if (trimmed.contains("BitVec") || trimmed.contains("bv"))
            && !theories.contains(&"BitVectors".to_string())
        {
            theories.push("BitVectors".to_string());
        }
        if (trimmed.contains("Array") || trimmed.contains("select") || trimmed.contains("store"))
            && !theories.contains(&"Arrays".to_string())
        {
            theories.push("Arrays".to_string());
        }

        // Track nesting depth
        for ch in trimmed.chars() {
            match ch {
                '"' if !in_comment => in_string = !in_string,
                '(' if !in_string && !in_comment => {
                    current_depth += 1;
                    if current_depth > max_depth {
                        max_depth = current_depth;
                    }
                    total_depth += current_depth;
                    depth_count += 1;
                }
                ')' if !in_string && !in_comment => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        // Count operators
        for op in &[
            "+", "-", "*", "/", "=", "<", ">", "<=", ">=", "and", "or", "not", "=>", "ite",
        ] {
            if trimmed.contains(op) {
                *operators.entry(op.to_string()).or_insert(0) += 1;
            }
        }
    }

    let avg_depth = if depth_count > 0 {
        total_depth as f64 / depth_count as f64
    } else {
        0.0
    };

    // Estimate difficulty
    let difficulty_score = (assertions as f64 * 1.0)
        + (max_depth as f64 * 2.0)
        + (quantifiers as f64 * 10.0)
        + (theories.len() as f64 * 3.0);

    let estimated_difficulty = if difficulty_score < 10.0 {
        "Easy".to_string()
    } else if difficulty_score < 50.0 {
        "Medium".to_string()
    } else if difficulty_score < 200.0 {
        "Hard".to_string()
    } else {
        "Very Hard".to_string()
    };

    // Recommend strategy
    let recommended_strategy = if quantifiers > 0 || theories.len() > 2 {
        "portfolio".to_string()
    } else if assertions < 10 && max_depth < 5 {
        "fast".to_string()
    } else {
        "balanced".to_string()
    };

    // Recommend timeout
    let recommended_timeout = if difficulty_score < 10.0 {
        10
    } else if difficulty_score < 50.0 {
        60
    } else if difficulty_score < 200.0 {
        300
    } else {
        600
    };

    ComplexityAnalysis {
        declarations,
        assertions,
        commands,
        max_nesting_depth: max_depth,
        avg_nesting_depth: avg_depth,
        quantifiers,
        operators,
        theories,
        estimated_difficulty,
        recommended_strategy,
        recommended_timeout,
    }
}

/// Classify the SMT problem and provide recommendations
pub fn classify_problem(_script: &str, analysis: &ComplexityAnalysis) -> ProblemClassification {
    let is_qf = analysis.quantifiers == 0;

    // Determine primary theory
    let primary_theory = if analysis.theories.is_empty() {
        "Propositional".to_string()
    } else if analysis.theories.len() == 1 {
        analysis.theories[0].clone()
    } else {
        "Combination".to_string()
    };

    // Determine logic
    let logic = if is_qf {
        if primary_theory == "Arithmetic" {
            "QF_LIA/QF_NIA".to_string()
        } else if primary_theory == "BitVectors" {
            "QF_BV".to_string()
        } else if primary_theory == "Arrays" {
            "QF_AX".to_string()
        } else if primary_theory == "Combination" {
            "QF_AUFLIA".to_string()
        } else {
            "QF_UF".to_string()
        }
    } else {
        "ALL".to_string()
    };

    // Determine complexity class
    let complexity_class = if is_qf && primary_theory != "Combination" {
        "NP-complete".to_string()
    } else if is_qf {
        "NP-hard".to_string()
    } else {
        "Undecidable".to_string()
    };

    // Generate solver recommendations
    let mut recommendations = Vec::new();

    if is_qf {
        recommendations.push("Use CDCL-based solver for good performance".to_string());
    } else {
        recommendations.push("Use portfolio strategy for quantified formulas".to_string());
    }

    if analysis.max_nesting_depth > 10 {
        recommendations.push("Enable simplification to reduce formula depth".to_string());
    }

    if primary_theory == "Arithmetic" {
        recommendations
            .push("Enable arithmetic optimizations (--theory-opt lia:fastpath)".to_string());
    } else if primary_theory == "BitVectors" {
        recommendations.push("Enable bit-blasting (--theory-opt bv:bitblast)".to_string());
    }

    if analysis.assertions > 100 {
        recommendations.push("Consider parallel solving (--parallel)".to_string());
    }

    if analysis.estimated_difficulty == "Very Hard" {
        recommendations.push(format!(
            "Set timeout to {}s or higher (--timeout {})",
            analysis.recommended_timeout, analysis.recommended_timeout
        ));
    }

    ProblemClassification {
        logic,
        is_quantifier_free: is_qf,
        primary_theory,
        complexity_class,
        solver_recommendations: recommendations,
    }
}

/// Apply automatic tuning based on problem characteristics
pub fn apply_auto_tune(
    ctx: &mut Context,
    analysis: &ComplexityAnalysis,
    classification: &ProblemClassification,
) {
    // Apply recommended strategy
    ctx.set_option("strategy", &analysis.recommended_strategy);

    // Enable simplification for complex problems
    if analysis.max_nesting_depth > 10 || analysis.assertions > 50 {
        ctx.set_option("simplify", "true");
    }

    // Set timeout based on difficulty
    if analysis.recommended_timeout > 0 {
        ctx.set_option(
            "timeout",
            &format!("{}", analysis.recommended_timeout * 1000),
        ); // Convert to ms
    }

    // Apply theory-specific optimizations
    if classification.primary_theory == "Arithmetic" {
        ctx.set_option("arith-solver", "simplex");
    } else if classification.primary_theory == "BitVectors" {
        ctx.set_option("bv-solver", "bitblast");
    }

    // Use portfolio for quantified or complex problems
    if !classification.is_quantifier_free || analysis.estimated_difficulty == "Very Hard" {
        ctx.set_option("strategy", "portfolio");
    }

    // Enable proof generation for small problems
    if analysis.assertions < 20 && analysis.max_nesting_depth < 5 {
        ctx.set_option("produce-proofs", "true");
    }
}

/// Validate SMT-LIB2 script syntax without solving
pub fn validate_script(script: &str) -> Result<String, String> {
    // Basic syntax validation
    let mut paren_count = 0;
    let mut line = 1;
    let mut col = 0;
    let mut in_string = false;
    let mut in_comment = false;

    for ch in script.chars() {
        match ch {
            '\n' => {
                line += 1;
                col = 0;
                in_comment = false;
            }
            ';' if !in_string => {
                in_comment = true;
            }
            '"' if !in_comment => {
                in_string = !in_string;
            }
            '(' if !in_string && !in_comment => {
                paren_count += 1;
            }
            ')' if !in_string && !in_comment => {
                paren_count -= 1;
                if paren_count < 0 {
                    return Err(format!(
                        "Unmatched closing parenthesis at line {}, column {}",
                        line, col
                    ));
                }
            }
            _ => {}
        }
        col += 1;
    }

    if paren_count > 0 {
        return Err(format!(
            "Unclosed parentheses: {} opening parentheses without matching closing",
            paren_count
        ));
    }

    if in_string {
        return Err("Unclosed string literal".to_string());
    }

    Ok("Syntax validation passed".to_string())
}
