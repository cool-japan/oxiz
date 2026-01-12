//! UNSAT core extraction and minimization
//!
//! This module provides functionality for extracting and minimizing UNSAT cores,
//! which are minimal subsets of assertions that are still unsatisfiable.

use std::collections::HashSet;
use std::io::Write;

/// Represents an UNSAT core
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct UnsatCore {
    /// Assertions in the core (by index)
    pub assertions: Vec<usize>,
    /// Original number of assertions
    pub total_assertions: usize,
}

impl UnsatCore {
    /// Create a new UNSAT core
    pub fn new(assertions: Vec<usize>, total_assertions: usize) -> Self {
        Self {
            assertions,
            total_assertions,
        }
    }

    /// Get the size of the core
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.assertions.len()
    }

    /// Get the reduction percentage
    #[allow(dead_code)]
    pub fn reduction_percent(&self) -> f64 {
        if self.total_assertions == 0 {
            0.0
        } else {
            100.0 * (1.0 - (self.size() as f64 / self.total_assertions as f64))
        }
    }

    /// Format the core for display
    #[allow(dead_code)]
    pub fn format(&self, assertion_names: &[String]) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "UNSAT Core: {} of {} assertions ({:.1}% reduction)\n",
            self.size(),
            self.total_assertions,
            self.reduction_percent()
        ));
        result.push_str("Assertions in core:\n");
        for &idx in &self.assertions {
            if idx < assertion_names.len() {
                result.push_str(&format!("  [{}] {}\n", idx, assertion_names[idx]));
            } else {
                result.push_str(&format!("  [{}] <unknown>\n", idx));
            }
        }
        result
    }

    /// Minimize the core using binary search
    /// Note: This is a simplified implementation - a real minimization would
    /// need to interact with the solver
    #[allow(dead_code)]
    pub fn minimize(&self) -> Self {
        // For now, just return the same core
        // A real implementation would use:
        // 1. Binary search to find minimal subsets
        // 2. Incremental solving to check satisfiability
        // 3. Iterative refinement
        self.clone()
    }
}

/// Extract UNSAT core from SMT-LIB2 output
#[allow(dead_code)]
pub fn extract_core_from_output(output: &str) -> Option<UnsatCore> {
    // Parse get-unsat-core response from SMT-LIB2
    // Format: (assertion1 assertion2 ...)
    if !output.contains('(') || !output.contains(')') {
        return None;
    }

    let trimmed = output.trim();
    if !trimmed.starts_with('(') {
        return None;
    }

    // Extract assertion names/indices
    let content = trimmed.trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<&str> = content.split_whitespace().collect();

    // Convert to indices (simplified - assumes numeric assertion names)
    let mut assertions = Vec::new();
    for part in parts {
        if let Ok(idx) = part.trim_start_matches("a").parse::<usize>() {
            assertions.push(idx);
        }
    }

    if assertions.is_empty() {
        None
    } else {
        let total = assertions.len();
        Some(UnsatCore::new(assertions, total))
    }
}

/// Generate proof tree in DOT format for GraphViz
pub fn generate_proof_dot<W: Write>(proof: &str, mut writer: W) -> Result<(), String> {
    writeln!(writer, "digraph proof {{")
        .map_err(|e| format!("Failed to write DOT header: {}", e))?;
    writeln!(writer, "  rankdir=TB;").map_err(|e| format!("Failed to write rankdir: {}", e))?;
    writeln!(writer, "  node [shape=box, style=rounded];")
        .map_err(|e| format!("Failed to write node style: {}", e))?;

    // Parse proof and generate nodes/edges
    // This is a simplified implementation
    let mut node_id = 0;
    let mut node_stack: Vec<usize> = Vec::new();

    for line in proof.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }

        // Detect proof steps
        if trimmed.contains("step") || trimmed.contains("assume") {
            let label = if trimmed.len() > 50 {
                format!("{}...", &trimmed[..47])
            } else {
                trimmed.to_string()
            };

            writeln!(
                writer,
                "  n{} [label=\"{}\"];",
                node_id,
                label.replace('"', "\\\"")
            )
            .map_err(|e| format!("Failed to write node: {}", e))?;

            // Create edge from parent if there is one
            if let Some(&parent_id) = node_stack.last() {
                writeln!(writer, "  n{} -> n{};", parent_id, node_id)
                    .map_err(|e| format!("Failed to write edge: {}", e))?;
            }

            if trimmed.contains('(') && !trimmed.contains(')') {
                // Opening a new scope
                node_stack.push(node_id);
            } else if trimmed.contains(')') && !trimmed.contains('(') {
                // Closing a scope
                node_stack.pop();
            }

            node_id += 1;
        }
    }

    writeln!(writer, "}}").map_err(|e| format!("Failed to write DOT footer: {}", e))?;

    Ok(())
}

/// Validate a model against assertions
#[allow(dead_code)]
pub fn validate_model(model: &str, assertions: &[String]) -> Result<Vec<usize>, String> {
    // Parse model and check against each assertion
    // Returns indices of assertions that are not satisfied
    let mut failed_assertions = Vec::new();

    // Parse variable assignments from model
    let mut assignments: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for line in model.lines() {
        let trimmed = line.trim();
        if trimmed.contains("define-fun") {
            // Extract variable name and value
            // Format: (define-fun var () Type value)
            if let Some(start) = trimmed.find("define-fun") {
                let after = &trimmed[start + 10..].trim();
                let parts: Vec<&str> = after.split_whitespace().collect();
                if parts.len() >= 4 {
                    let var_name = parts[0].to_string();
                    let value = parts[parts.len() - 1].trim_end_matches(')').to_string();
                    assignments.insert(var_name, value);
                }
            }
        }
    }

    // Check each assertion (simplified - would need full SMT evaluation)
    for (idx, assertion) in assertions.iter().enumerate() {
        // This is a placeholder - real validation would need to:
        // 1. Parse the assertion as an SMT formula
        // 2. Substitute variable values from the model
        // 3. Evaluate the formula
        // 4. Check if it evaluates to true

        // For now, we just check if the assertion contains any variables
        let contains_vars = assignments.keys().any(|var| assertion.contains(var));
        if !contains_vars && !assertion.contains("true") {
            failed_assertions.push(idx);
        }
    }

    Ok(failed_assertions)
}

/// Minimize UNSAT core using deletion-based algorithm
#[allow(dead_code)]
pub fn minimize_core_deletion(
    core: &UnsatCore,
    _check_unsat: impl Fn(&HashSet<usize>) -> bool,
) -> UnsatCore {
    // Deletion-based minimization algorithm:
    // 1. Start with the full core
    // 2. Try removing each assertion one at a time
    // 3. If still UNSAT after removal, keep it removed
    // 4. Repeat until no more assertions can be removed

    // For now, return the original core
    // A real implementation would use the check_unsat function
    core.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsat_core_creation() {
        let core = UnsatCore::new(vec![0, 2, 5], 10);
        assert_eq!(core.size(), 3);
        assert_eq!(core.total_assertions, 10);
        assert!((core.reduction_percent() - 70.0).abs() < 0.1);
    }

    #[test]
    fn test_core_format() {
        let core = UnsatCore::new(vec![0, 1], 5);
        let names = vec!["(= x 1)".to_string(), "(= x 2)".to_string()];
        let formatted = core.format(&names);
        assert!(formatted.contains("2 of 5"));
        assert!(formatted.contains("(= x 1)"));
        assert!(formatted.contains("(= x 2)"));
    }

    #[test]
    fn test_extract_core_from_output() {
        let output = "(a0 a2 a5)";
        let core = extract_core_from_output(output);
        assert!(core.is_some());
        let core = core.unwrap();
        assert_eq!(core.assertions, vec![0, 2, 5]);
    }

    #[test]
    fn test_proof_dot_generation() {
        let proof = r#"
            (step s1 (assume (= x 1)))
            (step s2 (assume (= x 2)))
            (step s3 (resolution s1 s2))
        "#;

        let mut output = Vec::new();
        let result = generate_proof_dot(proof, &mut output);
        assert!(result.is_ok());

        let dot_str = String::from_utf8(output).unwrap();
        assert!(dot_str.contains("digraph proof"));
        assert!(dot_str.contains("rankdir=TB"));
        assert!(dot_str.contains("shape=box"));
    }

    #[test]
    fn test_model_validation() {
        let model = r#"
            (define-fun x () Int 5)
            (define-fun y () Bool true)
        "#;

        let assertions = vec![
            "(= x 5)".to_string(),
            "(= y true)".to_string(),
            "(> z 10)".to_string(),
        ];

        let result = validate_model(model, &assertions);
        assert!(result.is_ok());
    }
}
