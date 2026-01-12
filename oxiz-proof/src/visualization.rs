//! Proof visualization utilities.
//!
//! This module provides tools for visualizing proof trees in various formats,
//! including DOT (Graphviz), ASCII art, and structured text.

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::io::{self, Write};

/// Visualization format for proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    /// DOT format for Graphviz.
    Dot,
    /// ASCII tree format.
    AsciiTree,
    /// Indented text format.
    IndentedText,
    /// JSON format.
    Json,
}

/// Proof visualizer.
#[derive(Debug)]
pub struct ProofVisualizer {
    /// Maximum depth to visualize (None = unlimited).
    max_depth: Option<usize>,
    /// Whether to show node IDs.
    show_ids: bool,
    /// Whether to show full conclusions (or truncate).
    show_full_conclusions: bool,
    /// Maximum conclusion length (if not showing full).
    max_conclusion_length: usize,
}

impl ProofVisualizer {
    /// Create a new proof visualizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            show_ids: true,
            show_full_conclusions: false,
            max_conclusion_length: 40,
        }
    }

    /// Set the maximum depth to visualize.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set whether to show node IDs.
    pub fn with_show_ids(mut self, show: bool) -> Self {
        self.show_ids = show;
        self
    }

    /// Set whether to show full conclusions.
    pub fn with_full_conclusions(mut self, show: bool) -> Self {
        self.show_full_conclusions = show;
        self
    }

    /// Visualize a proof in the specified format.
    pub fn visualize<W: Write>(
        &self,
        proof: &Proof,
        format: VisualizationFormat,
        writer: &mut W,
    ) -> io::Result<()> {
        match format {
            VisualizationFormat::Dot => self.visualize_dot(proof, writer),
            VisualizationFormat::AsciiTree => self.visualize_ascii_tree(proof, writer),
            VisualizationFormat::IndentedText => self.visualize_indented(proof, writer),
            VisualizationFormat::Json => self.visualize_json(proof, writer),
        }
    }

    /// Visualize proof as DOT format for Graphviz.
    fn visualize_dot<W: Write>(&self, proof: &Proof, writer: &mut W) -> io::Result<()> {
        writeln!(writer, "digraph Proof {{")?;
        writeln!(writer, "  rankdir=BT;")?;
        writeln!(writer, "  node [shape=box];")?;

        // Write nodes
        let mut visited = HashSet::new();
        if let Some(root) = proof.root() {
            self.write_dot_nodes(proof, root, writer, &mut visited, 0)?;
        }

        writeln!(writer, "}}")?;
        Ok(())
    }

    fn write_dot_nodes<W: Write>(
        &self,
        proof: &Proof,
        node_id: ProofNodeId,
        writer: &mut W,
        visited: &mut HashSet<ProofNodeId>,
        depth: usize,
    ) -> io::Result<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        if let Some(max_depth) = self.max_depth
            && depth >= max_depth
        {
            return Ok(());
        }

        visited.insert(node_id);

        if let Some(node) = proof.get_node(node_id) {
            let label = self.format_node_label(node);
            let color = match &node.step {
                ProofStep::Axiom { .. } => "lightblue",
                ProofStep::Inference { .. } => "lightgreen",
            };

            writeln!(
                writer,
                "  {} [label=\"{}\", fillcolor={}, style=filled];",
                node_id.0, label, color
            )?;

            // Write edges to premises
            if let ProofStep::Inference { premises, .. } = &node.step {
                for &premise_id in premises {
                    writeln!(writer, "  {} -> {};", premise_id.0, node_id.0)?;
                    self.write_dot_nodes(proof, premise_id, writer, visited, depth + 1)?;
                }
            }
        }

        Ok(())
    }

    /// Visualize proof as ASCII tree.
    fn visualize_ascii_tree<W: Write>(&self, proof: &Proof, writer: &mut W) -> io::Result<()> {
        if let Some(root) = proof.root_node() {
            self.write_ascii_node(proof, root, writer, "", true, 0)?;
        }
        Ok(())
    }

    fn write_ascii_node<W: Write>(
        &self,
        proof: &Proof,
        node: &ProofNode,
        writer: &mut W,
        prefix: &str,
        is_last: bool,
        depth: usize,
    ) -> io::Result<()> {
        if let Some(max_depth) = self.max_depth
            && depth >= max_depth
        {
            return Ok(());
        }

        let connector = if is_last { "└─" } else { "├─" };
        let label = self.format_node_label(node);

        writeln!(writer, "{}{} {}", prefix, connector, label)?;

        if let ProofStep::Inference { premises, .. } = &node.step {
            let new_prefix = format!("{}{}  ", prefix, if is_last { " " } else { "│" });

            for (i, &premise_id) in premises.iter().enumerate() {
                if let Some(premise_node) = proof.get_node(premise_id) {
                    let is_last_premise = i == premises.len() - 1;
                    self.write_ascii_node(
                        proof,
                        premise_node,
                        writer,
                        &new_prefix,
                        is_last_premise,
                        depth + 1,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Visualize proof as indented text.
    fn visualize_indented<W: Write>(&self, proof: &Proof, writer: &mut W) -> io::Result<()> {
        if let Some(root) = proof.root_node() {
            self.write_indented_node(proof, root, writer, 0, 0)?;
        }
        Ok(())
    }

    fn write_indented_node<W: Write>(
        &self,
        proof: &Proof,
        node: &ProofNode,
        writer: &mut W,
        indent: usize,
        depth: usize,
    ) -> io::Result<()> {
        if let Some(max_depth) = self.max_depth
            && depth >= max_depth
        {
            return Ok(());
        }

        let indent_str = "  ".repeat(indent);
        let label = self.format_node_label(node);

        writeln!(writer, "{}{}", indent_str, label)?;

        if let ProofStep::Inference { premises, .. } = &node.step {
            for &premise_id in premises {
                if let Some(premise_node) = proof.get_node(premise_id) {
                    self.write_indented_node(proof, premise_node, writer, indent + 1, depth + 1)?;
                }
            }
        }

        Ok(())
    }

    /// Visualize proof as JSON.
    fn visualize_json<W: Write>(&self, proof: &Proof, writer: &mut W) -> io::Result<()> {
        writeln!(writer, "{{")?;
        writeln!(writer, "  \"type\": \"proof\",")?;
        writeln!(writer, "  \"node_count\": {},", proof.node_count())?;
        writeln!(writer, "  \"depth\": {},", proof.depth())?;
        writeln!(writer, "  \"root\": {{")?;

        if let Some(root) = proof.root_node() {
            self.write_json_node(proof, root, writer, 2, 0)?;
        }

        writeln!(writer, "  }}")?;
        writeln!(writer, "}}")?;
        Ok(())
    }

    fn write_json_node<W: Write>(
        &self,
        proof: &Proof,
        node: &ProofNode,
        writer: &mut W,
        indent: usize,
        depth: usize,
    ) -> io::Result<()> {
        if let Some(max_depth) = self.max_depth
            && depth >= max_depth
        {
            return Ok(());
        }

        let indent_str = "  ".repeat(indent);

        if self.show_ids {
            writeln!(writer, "{}\"id\": \"{}\",", indent_str, node.id)?;
        }

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                writeln!(writer, "{}\"type\": \"axiom\",", indent_str)?;
                writeln!(
                    writer,
                    "{}\"conclusion\": \"{}\"",
                    indent_str,
                    escape_json(conclusion)
                )?;
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                writeln!(writer, "{}\"type\": \"inference\",", indent_str)?;
                writeln!(writer, "{}\"rule\": \"{}\",", indent_str, escape_json(rule))?;
                writeln!(
                    writer,
                    "{}\"conclusion\": \"{}\",",
                    indent_str,
                    escape_json(conclusion)
                )?;

                if !premises.is_empty() {
                    writeln!(writer, "{}\"premises\": [", indent_str)?;
                    for (i, &premise_id) in premises.iter().enumerate() {
                        if let Some(premise_node) = proof.get_node(premise_id) {
                            writeln!(writer, "{}  {{", indent_str)?;
                            self.write_json_node(
                                proof,
                                premise_node,
                                writer,
                                indent + 2,
                                depth + 1,
                            )?;
                            if i < premises.len() - 1 {
                                writeln!(writer, "{}  }},", indent_str)?;
                            } else {
                                writeln!(writer, "{}  }}", indent_str)?;
                            }
                        }
                    }
                    writeln!(writer, "{}]", indent_str)?;
                }
            }
        }

        Ok(())
    }

    /// Format a node label for display.
    fn format_node_label(&self, node: &ProofNode) -> String {
        let mut label = String::new();

        if self.show_ids {
            let _ = write!(label, "{}: ", node.id);
        }

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                let _ = write!(label, "axiom ");
                label.push_str(&self.format_conclusion(conclusion));
            }
            ProofStep::Inference {
                rule, conclusion, ..
            } => {
                let _ = write!(label, "{} ", rule);
                label.push_str(&self.format_conclusion(conclusion));
            }
        }

        label
    }

    /// Format a conclusion, possibly truncating it.
    fn format_conclusion(&self, conclusion: &str) -> String {
        if self.show_full_conclusions || conclusion.len() <= self.max_conclusion_length {
            conclusion.to_string()
        } else {
            format!("{}...", &conclusion[..self.max_conclusion_length])
        }
    }
}

impl Default for ProofVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_new() {
        let viz = ProofVisualizer::new();
        assert!(viz.show_ids);
        assert!(!viz.show_full_conclusions);
        assert_eq!(viz.max_conclusion_length, 40);
        assert!(viz.max_depth.is_none());
    }

    #[test]
    fn test_visualizer_with_options() {
        let viz = ProofVisualizer::new()
            .with_max_depth(5)
            .with_show_ids(false)
            .with_full_conclusions(true);

        assert_eq!(viz.max_depth, Some(5));
        assert!(!viz.show_ids);
        assert!(viz.show_full_conclusions);
    }

    #[test]
    fn test_visualize_dot() {
        let mut proof = Proof::new();
        proof.add_axiom("test");
        let viz = ProofVisualizer::new();

        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::Dot, &mut output)
            .unwrap();

        let dot = String::from_utf8(output).unwrap();
        assert!(dot.contains("digraph Proof"));
        assert!(dot.contains("axiom"));
        assert!(dot.contains("test"));
    }

    #[test]
    fn test_visualize_ascii_tree() {
        let mut proof = Proof::new();
        let p = proof.add_axiom("p");
        let q = proof.add_axiom("q");
        let _and_node = proof.add_inference("and", vec![p, q], "(and p q)");

        let viz = ProofVisualizer::new();
        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::AsciiTree, &mut output)
            .unwrap();

        let tree = String::from_utf8(output).unwrap();
        assert!(tree.contains("and"));
        assert!(tree.contains("axiom"));
    }

    #[test]
    fn test_visualize_indented() {
        let mut proof = Proof::new();
        proof.add_axiom("test");
        let viz = ProofVisualizer::new();

        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::IndentedText, &mut output)
            .unwrap();

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("axiom"));
        assert!(text.contains("test"));
    }

    #[test]
    fn test_visualize_json() {
        let mut proof = Proof::new();
        proof.add_axiom("test");
        let viz = ProofVisualizer::new();

        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::Json, &mut output)
            .unwrap();

        let json = String::from_utf8(output).unwrap();
        assert!(json.contains("\"type\": \"proof\""));
        assert!(json.contains("\"type\": \"axiom\""));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json("path\\to\\file"), "path\\\\to\\\\file");
    }

    #[test]
    fn test_visualize_with_max_depth() {
        let mut proof = Proof::new();
        let p = proof.add_axiom("p");
        let q = proof.add_axiom("q");
        let r = proof.add_axiom("r");
        let and1 = proof.add_inference("and", vec![q, r], "(and q r)");
        let _and2 = proof.add_inference("and", vec![p, and1], "(and p (and q r))");

        let viz = ProofVisualizer::new().with_max_depth(1);
        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::IndentedText, &mut output)
            .unwrap();

        let text = String::from_utf8(output).unwrap();
        // Should only show root and its immediate children
        assert!(text.contains("and"));
    }

    #[test]
    fn test_format_conclusion_truncate() {
        let viz = ProofVisualizer::new();

        let short = "short";
        assert_eq!(viz.format_conclusion(short), "short");

        let long = "a".repeat(50);
        let formatted = viz.format_conclusion(&long);
        assert!(formatted.ends_with("..."));
        assert!(formatted.len() < long.len());
    }

    #[test]
    fn test_format_conclusion_full() {
        let viz = ProofVisualizer::new().with_full_conclusions(true);

        let long = "a".repeat(50);
        let formatted = viz.format_conclusion(&long);
        assert_eq!(formatted, long);
        assert!(!formatted.contains("..."));
    }
}
