//! Dependency tracking between assertions
//!
//! This module provides functionality to track dependencies between SMT-LIB2 assertions,
//! helping users understand the structure and relationships in their problems.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Represents a dependency graph for assertions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// Map from assertion index to the set of symbols it depends on
    pub assertion_symbols: HashMap<usize, HashSet<String>>,
    /// Map from symbol to the set of assertion indices that use it
    pub symbol_assertions: HashMap<String, HashSet<usize>>,
    /// Map from assertion index to its expression
    pub assertions: HashMap<usize, String>,
    /// All symbols declared in the problem
    pub declared_symbols: HashSet<String>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self {
            assertion_symbols: HashMap::new(),
            symbol_assertions: HashMap::new(),
            assertions: HashMap::new(),
            declared_symbols: HashSet::new(),
        }
    }

    /// Add a symbol declaration
    pub fn add_declaration(&mut self, symbol: String) {
        self.declared_symbols.insert(symbol);
    }

    /// Add an assertion with its index and extract dependencies
    pub fn add_assertion(&mut self, index: usize, expr: &str) {
        self.assertions.insert(index, expr.to_string());
        let symbols = Self::extract_symbols(expr);

        for symbol in &symbols {
            self.symbol_assertions
                .entry(symbol.clone())
                .or_default()
                .insert(index);
        }

        self.assertion_symbols.insert(index, symbols);
    }

    /// Extract symbols from an SMT-LIB2 expression
    fn extract_symbols(expr: &str) -> HashSet<String> {
        let mut symbols = HashSet::new();
        let mut current_token = String::new();
        let mut in_string = false;

        for ch in expr.chars() {
            if ch == '"' {
                in_string = !in_string;
                current_token.clear();
            } else if in_string {
                continue;
            } else if ch.is_whitespace() || ch == '(' || ch == ')' {
                if !current_token.is_empty() {
                    // Skip operators and keywords
                    if !Self::is_operator_or_keyword(&current_token) {
                        symbols.insert(current_token.clone());
                    }
                    current_token.clear();
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() && !Self::is_operator_or_keyword(&current_token) {
            symbols.insert(current_token);
        }

        symbols
    }

    /// Check if a token is an operator or keyword
    fn is_operator_or_keyword(token: &str) -> bool {
        matches!(
            token,
            "=" | "+"
                | "-"
                | "*"
                | "/"
                | "div"
                | "mod"
                | "<"
                | "<="
                | ">"
                | ">="
                | "and"
                | "or"
                | "not"
                | "=>"
                | "ite"
                | "let"
                | "forall"
                | "exists"
                | "true"
                | "false"
                | "assert"
                | "declare-const"
                | "declare-fun"
                | "define-fun"
                | "check-sat"
                | "get-model"
                | "get-proof"
                | "push"
                | "pop"
                | "Int"
                | "Bool"
                | "Real"
                | "Array"
                | "BitVec"
                | "bvadd"
                | "bvsub"
                | "bvmul"
                | "bvand"
                | "bvor"
                | "bvnot"
                | "select"
                | "store"
                | "concat"
                | "extract"
        ) || token.parse::<i64>().is_ok()
            || token.parse::<f64>().is_ok()
    }

    /// Find assertions that depend on a given symbol
    #[allow(dead_code)]
    pub fn find_dependent_assertions(&self, symbol: &str) -> Vec<usize> {
        self.symbol_assertions
            .get(symbol)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Find symbols used by a given assertion
    pub fn find_assertion_symbols(&self, index: usize) -> Vec<String> {
        self.assertion_symbols
            .get(&index)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Find assertions that share symbols with a given assertion
    pub fn find_related_assertions(&self, index: usize) -> Vec<usize> {
        let mut related = HashSet::new();

        if let Some(symbols) = self.assertion_symbols.get(&index) {
            for symbol in symbols {
                if let Some(assertions) = self.symbol_assertions.get(symbol) {
                    for &assertion_index in assertions {
                        if assertion_index != index {
                            related.insert(assertion_index);
                        }
                    }
                }
            }
        }

        related.into_iter().collect()
    }

    /// Get statistics about the dependency graph
    pub fn statistics(&self) -> DependencyStatistics {
        let mut symbol_usage: Vec<usize> = self
            .symbol_assertions
            .values()
            .map(|set| set.len())
            .collect();
        symbol_usage.sort_unstable();

        let mut assertion_complexity: Vec<usize> = self
            .assertion_symbols
            .values()
            .map(|set| set.len())
            .collect();
        assertion_complexity.sort_unstable();

        DependencyStatistics {
            total_assertions: self.assertions.len(),
            total_symbols: self.symbol_assertions.len(),
            declared_symbols: self.declared_symbols.len(),
            undeclared_symbols: self
                .symbol_assertions
                .keys()
                .filter(|s| !self.declared_symbols.contains(*s))
                .count(),
            min_symbol_usage: symbol_usage.first().copied().unwrap_or(0),
            max_symbol_usage: symbol_usage.last().copied().unwrap_or(0),
            avg_symbol_usage: if symbol_usage.is_empty() {
                0.0
            } else {
                symbol_usage.iter().sum::<usize>() as f64 / symbol_usage.len() as f64
            },
            min_assertion_complexity: assertion_complexity.first().copied().unwrap_or(0),
            max_assertion_complexity: assertion_complexity.last().copied().unwrap_or(0),
            avg_assertion_complexity: if assertion_complexity.is_empty() {
                0.0
            } else {
                assertion_complexity.iter().sum::<usize>() as f64
                    / assertion_complexity.len() as f64
            },
        }
    }

    /// Find highly connected symbols (symbols used by many assertions)
    pub fn find_hub_symbols(&self, threshold: usize) -> Vec<(String, usize)> {
        let mut hubs: Vec<(String, usize)> = self
            .symbol_assertions
            .iter()
            .filter(|(_, assertions)| assertions.len() >= threshold)
            .map(|(symbol, assertions)| (symbol.clone(), assertions.len()))
            .collect();

        hubs.sort_by(|a, b| b.1.cmp(&a.1));
        hubs
    }

    /// Find isolated assertions (assertions with few symbol connections)
    #[allow(dead_code)]
    pub fn find_isolated_assertions(&self, threshold: usize) -> Vec<usize> {
        self.assertion_symbols
            .iter()
            .filter(|(_, symbols)| symbols.len() <= threshold)
            .map(|(index, _)| *index)
            .collect()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyStatistics {
    /// Total number of assertions
    pub total_assertions: usize,
    /// Total number of symbols used
    pub total_symbols: usize,
    /// Number of declared symbols
    pub declared_symbols: usize,
    /// Number of undeclared symbols (used but not declared)
    pub undeclared_symbols: usize,
    /// Minimum number of assertions using any symbol
    pub min_symbol_usage: usize,
    /// Maximum number of assertions using any symbol
    pub max_symbol_usage: usize,
    /// Average number of assertions per symbol
    pub avg_symbol_usage: f64,
    /// Minimum number of symbols in any assertion
    pub min_assertion_complexity: usize,
    /// Maximum number of symbols in any assertion
    pub max_assertion_complexity: usize,
    /// Average number of symbols per assertion
    pub avg_assertion_complexity: f64,
}

/// Analyze dependencies in an SMT-LIB2 script
pub fn analyze_dependencies(script: &str) -> DependencyGraph {
    let mut graph = DependencyGraph::new();
    let mut assertion_index = 0;

    // Simple parser for SMT-LIB2 commands
    let mut in_command = false;
    let mut current_command = String::new();
    let mut paren_depth = 0;

    for ch in script.chars() {
        if ch == '(' {
            paren_depth += 1;
            in_command = true;
            current_command.push(ch);
        } else if ch == ')' {
            paren_depth -= 1;
            current_command.push(ch);

            if paren_depth == 0 && in_command {
                // Process completed command
                let trimmed = current_command.trim();

                if trimmed.starts_with("(declare-const") || trimmed.starts_with("(declare-fun") {
                    if let Some(symbol) = extract_declaration_symbol(trimmed) {
                        graph.add_declaration(symbol);
                    }
                } else if trimmed.starts_with("(assert") {
                    graph.add_assertion(assertion_index, trimmed);
                    assertion_index += 1;
                }

                current_command.clear();
                in_command = false;
            }
        } else if in_command {
            current_command.push(ch);
        }
    }

    graph
}

/// Extract symbol name from a declaration command
fn extract_declaration_symbol(decl: &str) -> Option<String> {
    let decl = decl.trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<&str> = decl.split_whitespace().collect();

    if parts.len() >= 2 && (parts[0] == "declare-const" || parts[0] == "declare-fun") {
        Some(parts[1].to_string())
    } else {
        None
    }
}

/// Format dependency graph as a human-readable string
pub fn format_dependency_graph(graph: &DependencyGraph, detailed: bool) -> String {
    let mut output = String::new();
    let stats = graph.statistics();

    output.push_str("=== Dependency Analysis ===\n\n");

    output.push_str(&format!("Assertions: {}\n", stats.total_assertions));
    output.push_str(&format!(
        "Symbols: {} ({} declared, {} undeclared)\n\n",
        stats.total_symbols, stats.declared_symbols, stats.undeclared_symbols
    ));

    output.push_str("Symbol Usage:\n");
    output.push_str(&format!("  Min: {} assertions\n", stats.min_symbol_usage));
    output.push_str(&format!("  Max: {} assertions\n", stats.max_symbol_usage));
    output.push_str(&format!(
        "  Avg: {:.2} assertions\n\n",
        stats.avg_symbol_usage
    ));

    output.push_str("Assertion Complexity:\n");
    output.push_str(&format!(
        "  Min: {} symbols\n",
        stats.min_assertion_complexity
    ));
    output.push_str(&format!(
        "  Max: {} symbols\n",
        stats.max_assertion_complexity
    ));
    output.push_str(&format!(
        "  Avg: {:.2} symbols\n\n",
        stats.avg_assertion_complexity
    ));

    // Find hub symbols (used by many assertions)
    let hubs = graph.find_hub_symbols(3);
    if !hubs.is_empty() {
        output.push_str("Highly Connected Symbols (hubs):\n");
        for (symbol, count) in hubs.iter().take(10) {
            output.push_str(&format!("  {} (used in {} assertions)\n", symbol, count));
        }
        output.push('\n');
    }

    if detailed {
        output.push_str("=== Detailed Dependencies ===\n\n");

        for i in 0..stats.total_assertions {
            let symbols = graph.find_assertion_symbols(i);
            let related = graph.find_related_assertions(i);

            output.push_str(&format!("Assertion {}:\n", i));
            output.push_str(&format!("  Symbols: {:?}\n", symbols));
            if !related.is_empty() {
                output.push_str(&format!("  Related assertions: {:?}\n", related));
            }
            output.push('\n');
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_graph_creation() {
        let mut graph = DependencyGraph::new();
        graph.add_declaration("x".to_string());
        graph.add_declaration("y".to_string());
        graph.add_assertion(0, "(assert (= x 5))");
        graph.add_assertion(1, "(assert (> y 10))");

        assert_eq!(graph.declared_symbols.len(), 2);
        assert_eq!(graph.assertions.len(), 2);
    }

    #[test]
    fn test_symbol_extraction() {
        let expr = "(and (= x 5) (> y 10))";
        let symbols = DependencyGraph::extract_symbols(expr);

        assert!(symbols.contains("x"));
        assert!(symbols.contains("y"));
        assert!(!symbols.contains("and"));
        assert!(!symbols.contains("="));
    }

    #[test]
    fn test_dependency_analysis() {
        let script = r#"
            (declare-const x Int)
            (declare-const y Int)
            (assert (= x 5))
            (assert (> y 10))
            (assert (= (+ x y) 15))
        "#;

        let graph = analyze_dependencies(script);

        assert_eq!(graph.declared_symbols.len(), 2);
        assert_eq!(graph.assertions.len(), 3);

        // Check that x is used in assertions 0 and 2
        let x_assertions = graph.find_dependent_assertions("x");
        assert!(x_assertions.contains(&0));
        assert!(x_assertions.contains(&2));
    }

    #[test]
    fn test_related_assertions() {
        let script = r#"
            (declare-const x Int)
            (declare-const y Int)
            (assert (= x 5))
            (assert (> y 10))
            (assert (= (+ x y) 15))
        "#;

        let graph = analyze_dependencies(script);

        // Assertion 0 and 2 both use x, so they should be related
        let related = graph.find_related_assertions(0);
        assert!(related.contains(&2));
    }

    #[test]
    fn test_hub_symbols() {
        let mut graph = DependencyGraph::new();
        graph.add_declaration("x".to_string());

        // x is used in 4 assertions
        for i in 0..4 {
            graph.add_assertion(i, &format!("(assert (= x {}))", i));
        }

        let hubs = graph.find_hub_symbols(3);
        assert_eq!(hubs.len(), 1);
        assert_eq!(hubs[0].0, "x");
        assert_eq!(hubs[0].1, 4);
    }

    #[test]
    fn test_statistics() {
        let script = r#"
            (declare-const x Int)
            (declare-const y Int)
            (assert (= x 5))
            (assert (> y 10))
        "#;

        let graph = analyze_dependencies(script);
        let stats = graph.statistics();

        assert_eq!(stats.total_assertions, 2);
        assert_eq!(stats.declared_symbols, 2);
    }
}
