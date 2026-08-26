//! Proof visualization utilities.
//!
//! This module provides tools for visualizing proof trees in various formats,
//! including DOT (Graphviz), ASCII art, and structured text.
//!
//! # Cost of each format
//!
//! Every renderer here walks the proof with an explicit heap stack, so none of
//! them is bounded by the call stack. What they differ in is how many bytes
//! they emit and how much each pending work item retains, as a function of the
//! proof's depth `d`:
//!
//! | Format | Output bytes | Retained per pending frame |
//! |---|---|---|
//! | [`VisualizationFormat::Dot`] | Theta(nodes + edges) | O(1) |
//! | [`VisualizationFormat::IndentedText`] | Theta(d^2) | O(1) |
//! | [`VisualizationFormat::Json`] | Theta(d^2) | O(1) |
//! | [`VisualizationFormat::AsciiTree`] | Theta(d^2) | O(d) |
//!
//! The right-hand column is per *frame*; the live heap is that times the stack
//! height. Three of the walks stack one frame per node whose turn has not come
//! yet, so their height is the number of un-rendered siblings along the current
//! path — Theta(1) on a chain, Theta(d) on a binary tree of depth `d`, O(nodes)
//! in general. [`VisualizationFormat::Json`] also stacks the closing `}` and
//! `]` of every ancestor still open, so its height is Theta(d) in every shape.
//!
//! `AsciiTree` is the one format whose live heap is superlinear in the depth:
//! on a branching proof it holds Theta(d) frames each owning its own O(d)
//! prefix, i.e. Theta(d^2) live. The other three hold O(1) per frame in every
//! shape, so `Json` is Theta(d) live and the other two are O(nodes).
//!
//! The quadratic *output* of the three tree formats is inherent to
//! indent-by-depth rendering: a node `d` levels down prefixes every line it
//! emits with `2*d` characters. That is output volume only — a caller
//! streaming to a file or a pipe never holds it — and the only way to bound it
//! is to cap the indent, which changes the rendered format. The cost is
//! documented here rather than silently changed.
//!
//! `AsciiTree`'s extra live cost is not a plain indent and cannot be
//! reconstructed from the depth: its prefix records, for each ancestor,
//! whether that ancestor was a last child (`" "` vs `"\u{2502}"`), so each
//! stacked frame owns its own grown copy. `Json` used to be quadratic-live for
//! a different and avoidable reason — its delimiters were pre-rendered
//! `String`s stacked below the child, which made even a *chain* quadratic; see
//! `JsonFrame`.
//!
//! [`VisualizationFormat::Dot`] additionally keeps a `visited` set of every
//! node it has emitted, so it is Theta(nodes) live regardless of shape — that
//! set is what lets it render the DAG once instead of re-expanding it.
//!
//! On a heavily *shared* DAG the three tree formats repeat each shared premise
//! once per path reaching it, which is exponential in the worst case.
//! [`VisualizationFormat::Dot`] renders the DAG itself and does not.

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

/// A structural delimiter line emitted by the iterative JSON writer.
///
/// Only the delimiter *token* is stored; the indentation is rendered at pop
/// time from the frame's indent level (see [`JsonFrame::Delim`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonDelim {
    /// `[`-array terminator closing a `"premises"` list.
    CloseBracket,
    /// Premise object terminator followed by another premise.
    CloseBraceComma,
    /// Premise object terminator for the last premise in a list.
    CloseBrace,
    /// Premise object opener.
    OpenBrace,
}

impl JsonDelim {
    /// The delimiter token, without any indentation.
    fn token(self) -> &'static str {
        match self {
            Self::CloseBracket => "]",
            Self::CloseBraceComma => "},",
            Self::CloseBrace => "}",
            Self::OpenBrace => "{",
        }
    }
}

/// Work item for the iterative JSON writer.
///
/// Every variant is `Copy` and fixed-size on purpose: a frame must not own an
/// indent-bearing `String`. The delimiters used to be materialized eagerly
/// with the full indent baked in and pushed *below* the child frame, so a
/// chain of depth d kept three O(d)-length strings alive per ancestor level —
/// Theta(d^2) live heap (~14 GB at depth 60,000), which no output sink could
/// avoid. Storing `(indent, kind)` makes the retained stack Theta(d).
#[derive(Debug, Clone, Copy)]
enum JsonFrame {
    /// Render the node's object body at the given indent/depth.
    Node {
        /// Node to render.
        id: ProofNodeId,
        /// Indentation level for the node's keys.
        indent: usize,
        /// Distance from the visualization root (for `max_depth`).
        depth: usize,
    },
    /// Emit a structural line (brace, bracket, separator) at `indent`.
    Delim {
        /// Indentation level for the delimiter token.
        indent: usize,
        /// Which delimiter to emit.
        kind: JsonDelim,
    },
}

/// A `JsonFrame` that owned a `String` would reintroduce the Theta(depth^2)
/// live heap described above, and an owned field is exactly what makes a frame
/// need dropping. Pinned here so the regression cannot compile.
const _: () = assert!(!std::mem::needs_drop::<JsonFrame>());

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

    /// Emit the DOT node/edge lines for the sub-DAG rooted at `node_id`.
    ///
    /// Iterative (explicit heap stack): proof DAGs are routinely far deeper than
    /// the call stack can accommodate, and the `max_depth` cap is `None` by
    /// default so it cannot be relied on as a bound.
    fn write_dot_nodes<W: Write>(
        &self,
        proof: &Proof,
        node_id: ProofNodeId,
        writer: &mut W,
        visited: &mut HashSet<ProofNodeId>,
        depth: usize,
    ) -> io::Result<()> {
        // (node, node it is a premise of, depth)
        let mut stack: Vec<(ProofNodeId, Option<ProofNodeId>, usize)> =
            vec![(node_id, None, depth)];

        while let Some((current, parent, current_depth)) = stack.pop() {
            // The edge is written before the child is expanded, and regardless
            // of whether the child turns out to be already-visited or capped.
            if let Some(parent_id) = parent {
                writeln!(writer, "  {} -> {};", current.0, parent_id.0)?;
            }

            if visited.contains(&current) {
                continue;
            }
            if let Some(max_depth) = self.max_depth
                && current_depth >= max_depth
            {
                continue;
            }

            visited.insert(current);

            let Some(node) = proof.get_node(current) else {
                continue;
            };

            let label = escape_dot_label(&self.format_node_label(node));
            let color = match &node.step {
                ProofStep::Axiom { .. } => "lightblue",
                ProofStep::Inference { .. } => "lightgreen",
            };

            writeln!(
                writer,
                "  {} [label=\"{}\", fillcolor={}, style=filled];",
                current.0, label, color
            )?;

            // Write edges to premises, leftmost first (hence reversed pushes)
            if let ProofStep::Inference { premises, .. } = &node.step {
                stack.extend(
                    premises
                        .iter()
                        .rev()
                        .map(|&premise_id| (premise_id, Some(current), current_depth + 1)),
                );
            }
        }

        Ok(())
    }

    /// Visualize proof as ASCII tree.
    ///
    /// This is a *tree* rendering: a premise shared by several inferences is
    /// printed once under each of them, exactly as the recursive version did.
    /// On a heavily shared proof DAG that output is exponentially large — use
    /// [`VisualizationFormat::Dot`], which renders the DAG itself, instead.
    ///
    /// This is also the most expensive format on a *deep* proof: Theta(depth^2)
    /// output bytes always, and Theta(depth^2) live heap once the proof
    /// branches (see the module-level cost table). Bounding either one would
    /// change the rendered tree, so the cost is documented rather than capped.
    fn visualize_ascii_tree<W: Write>(&self, proof: &Proof, writer: &mut W) -> io::Result<()> {
        if let Some(root) = proof.root() {
            self.write_ascii_node(proof, root, writer, String::new(), true, 0)?;
        }
        Ok(())
    }

    /// Iterative (explicit heap stack) ASCII-tree rendering.
    ///
    /// Each stacked frame owns the box-drawing prefix for its own line, grown
    /// by two characters per level. A chain pops and pushes one frame at a time
    /// so the stack stays a single O(d) prefix, but a branching proof of depth
    /// `d` keeps Theta(d) frames alive at once and so holds Theta(d^2) bytes.
    ///
    /// Unlike the JSON writer's indents, this prefix is *not* a function of the
    /// depth alone — it records for every ancestor whether that ancestor was a
    /// last child — so it cannot be re-derived at pop time from an integer.
    /// Deliberately left as-is: the alternative is to stack the ancestor
    /// `is_last` bits (a bitset, O(d) total rather than O(d) per frame) and
    /// rebuild the prefix per line, trading live heap for a slower render.
    fn write_ascii_node<W: Write>(
        &self,
        proof: &Proof,
        node_id: ProofNodeId,
        writer: &mut W,
        prefix: String,
        is_last: bool,
        depth: usize,
    ) -> io::Result<()> {
        // (node, prefix for that node's own line, is_last, depth)
        let mut stack: Vec<(ProofNodeId, String, bool, usize)> =
            vec![(node_id, prefix, is_last, depth)];

        while let Some((current, current_prefix, current_is_last, current_depth)) = stack.pop() {
            if let Some(max_depth) = self.max_depth
                && current_depth >= max_depth
            {
                continue;
            }

            let Some(node) = proof.get_node(current) else {
                continue;
            };

            let connector = if current_is_last { "└─" } else { "├─" };
            let label = self.format_node_label(node);

            writeln!(writer, "{}{} {}", current_prefix, connector, label)?;

            if let ProofStep::Inference { premises, .. } = &node.step {
                let new_prefix = format!(
                    "{}{}  ",
                    current_prefix,
                    if current_is_last { " " } else { "│" }
                );

                let last_index = premises.len().saturating_sub(1);
                stack.extend(premises.iter().enumerate().rev().map(|(i, &premise_id)| {
                    (
                        premise_id,
                        new_prefix.clone(),
                        i == last_index,
                        current_depth + 1,
                    )
                }));
            }
        }

        Ok(())
    }

    /// Visualize proof as indented text.
    ///
    /// Like the ASCII tree, this is a tree rendering and repeats shared premises.
    fn visualize_indented<W: Write>(&self, proof: &Proof, writer: &mut W) -> io::Result<()> {
        if let Some(root) = proof.root() {
            self.write_indented_node(proof, root, writer, 0, 0)?;
        }
        Ok(())
    }

    /// Iterative (explicit heap stack) indented-text rendering.
    ///
    /// A frame carries only its indent *level*, so it is O(1) however deep it
    /// sits — but the indent itself is re-rendered per line, which makes the
    /// output Theta(depth^2) bytes. That is volume a streaming caller never
    /// retains, and capping the indent would change the format, so it stands.
    fn write_indented_node<W: Write>(
        &self,
        proof: &Proof,
        node_id: ProofNodeId,
        writer: &mut W,
        indent: usize,
        depth: usize,
    ) -> io::Result<()> {
        // (node, indent level, depth)
        let mut stack: Vec<(ProofNodeId, usize, usize)> = vec![(node_id, indent, depth)];

        while let Some((current, current_indent, current_depth)) = stack.pop() {
            if let Some(max_depth) = self.max_depth
                && current_depth >= max_depth
            {
                continue;
            }

            let Some(node) = proof.get_node(current) else {
                continue;
            };

            let indent_str = "  ".repeat(current_indent);
            let label = self.format_node_label(node);

            writeln!(writer, "{}{}", indent_str, label)?;

            if let ProofStep::Inference { premises, .. } = &node.step {
                stack.extend(
                    premises
                        .iter()
                        .rev()
                        .map(|&premise_id| (premise_id, current_indent + 1, current_depth + 1)),
                );
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

        if let Some(root) = proof.root() {
            self.write_json_node(proof, root, writer, 2, 0)?;
        }

        writeln!(writer, "  }}")?;
        writeln!(writer, "}}")?;
        Ok(())
    }

    /// Iterative (explicit heap stack) JSON rendering.
    ///
    /// Depth truncation and dangling premises are both reported *in* the JSON
    /// rather than by silently emitting a partial object: a node cut off by
    /// `max_depth` becomes `{"truncated": true}`, and the trailing-comma
    /// bookkeeping counts only the premises actually emitted, so the output is
    /// always parseable.
    ///
    /// The stack keeps the closing `}` and `]` of every ancestor still open, so
    /// it is Theta(depth) frames deep even on a chain. Each frame is O(1) —
    /// a delimiter carries an indent *level*, not a rendered indent string (see
    /// [`JsonDelim`]) — so the live heap is Theta(depth), not Theta(depth^2).
    /// The output remains Theta(depth^2) bytes, as for every indent-by-depth
    /// format here.
    fn write_json_node<W: Write>(
        &self,
        proof: &Proof,
        node_id: ProofNodeId,
        writer: &mut W,
        indent: usize,
        depth: usize,
    ) -> io::Result<()> {
        let mut stack = vec![JsonFrame::Node {
            id: node_id,
            indent,
            depth,
        }];

        while let Some(frame) = stack.pop() {
            let (current, current_indent, current_depth) = match frame {
                JsonFrame::Delim { indent, kind } => {
                    // The indent is rendered here, not stored in the frame.
                    writeln!(writer, "{}{}", "  ".repeat(indent), kind.token())?;
                    continue;
                }
                JsonFrame::Node { id, indent, depth } => (id, indent, depth),
            };

            let indent_str = "  ".repeat(current_indent);

            if let Some(max_depth) = self.max_depth
                && current_depth >= max_depth
            {
                writeln!(writer, "{}\"truncated\": true", indent_str)?;
                continue;
            }

            let Some(node) = proof.get_node(current) else {
                writeln!(writer, "{}\"missing\": true", indent_str)?;
                continue;
            };

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

                    // Only premises that resolve to a node are emitted, and the
                    // comma bookkeeping is over that emitted set.
                    let present: Vec<ProofNodeId> = premises
                        .iter()
                        .copied()
                        .filter(|&id| proof.get_node(id).is_some())
                        .collect();

                    if !present.is_empty() {
                        writeln!(writer, "{}\"premises\": [", indent_str)?;
                        let last_index = present.len().saturating_sub(1);
                        stack.push(JsonFrame::Delim {
                            indent: current_indent,
                            kind: JsonDelim::CloseBracket,
                        });
                        for (i, &premise_id) in present.iter().enumerate().rev() {
                            let close = if i < last_index {
                                JsonDelim::CloseBraceComma
                            } else {
                                JsonDelim::CloseBrace
                            };
                            stack.push(JsonFrame::Delim {
                                indent: current_indent + 1,
                                kind: close,
                            });
                            stack.push(JsonFrame::Node {
                                id: premise_id,
                                indent: current_indent + 2,
                                depth: current_depth + 1,
                            });
                            stack.push(JsonFrame::Delim {
                                indent: current_indent + 1,
                                kind: JsonDelim::OpenBrace,
                            });
                        }
                    }
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

/// Escape a string for use inside a double-quoted DOT (Graphviz) label.
///
/// A DOT quoted string ends at the first unescaped `"`, and `\` is itself
/// the escape introducer that would otherwise change the meaning of the
/// character after it. `write_dot_nodes` embeds free-form proof text (a
/// conclusion or a rule name, neither validated against DOT's grammar)
/// directly into `label="..."`; without this, a `"` in that text ended the
/// attribute early and corrupted the rest of the graph description (and any
/// premises/edges written after it).
fn escape_dot_label(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
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
            .expect("test operation should succeed");

        let dot = String::from_utf8(output).expect("test operation should succeed");
        assert!(dot.contains("digraph Proof"));
        assert!(dot.contains("axiom"));
        assert!(dot.contains("test"));
    }

    /// Direct unit test of the DOT label escaper: `\` and `"` are escaped
    /// (they are DOT's own escape introducer and quoted-string terminator);
    /// non-ASCII and control characters are left alone since they are not
    /// special to DOT's quoted-string grammar.
    #[test]
    fn test_escape_dot_label() {
        assert_eq!(escape_dot_label("hello"), "hello");
        assert_eq!(escape_dot_label("a\"b"), "a\\\"b");
        assert_eq!(escape_dot_label("a\\b"), "a\\\\b");
        assert_eq!(escape_dot_label("caf\u{e9}"), "caf\u{e9}");
        assert_eq!(escape_dot_label("a\u{0}b"), "a\u{0}b");
    }

    /// Regression: `write_dot_nodes` used to interpolate the node label
    /// (built from the free-form conclusion/rule text) straight into
    /// `label="..."` with no escaping at all, so a `"` in a conclusion ended
    /// the DOT attribute early and corrupted the rest of the graph
    /// (dangling text, unbalanced brackets, or a truncated file depending on
    /// what followed). Every `label="..."` attribute must now be a properly
    /// terminated, balanced quoted string even when the conclusion contains
    /// a quote and a backslash.
    #[test]
    fn test_visualize_dot_conclusion_with_quote_and_backslash_is_escaped() {
        let mut proof = Proof::new();
        proof.add_axiom("has \"quote\" and \\backslash");
        let viz = ProofVisualizer::new();

        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::Dot, &mut output)
            .expect("test operation should succeed");
        let dot = String::from_utf8(output).expect("test operation should succeed");

        assert!(
            dot.contains(r#"\"quote\""#),
            "the quote must be escaped, not raw, in: {dot}"
        );
        assert!(
            dot.contains(r"\\backslash"),
            "the backslash must be escaped, not raw, in: {dot}"
        );

        // Every `label="..."` attribute must terminate at an *unescaped*
        // quote before the line ends (i.e. escaping did not leave a bare
        // `"` that ends the attribute early).
        for line in dot.lines().filter(|l| l.contains("label=\"")) {
            let after_label = line
                .split_once("label=\"")
                .expect("filtered line contains label=\"")
                .1;
            let mut chars = after_label.chars();
            let mut terminated = false;
            while let Some(c) = chars.next() {
                if c == '\\' {
                    chars.next(); // escaped character: does not terminate.
                    continue;
                }
                if c == '"' {
                    terminated = true;
                    break;
                }
            }
            assert!(
                terminated,
                "label attribute never reaches an unescaped closing quote in: {line}"
            );
        }
    }

    /// Control: a conclusion with no special characters renders in the DOT
    /// label completely unchanged.
    #[test]
    fn test_visualize_dot_plain_conclusion_unchanged() {
        let mut proof = Proof::new();
        proof.add_axiom("plain conclusion with no special chars");
        let viz = ProofVisualizer::new();

        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::Dot, &mut output)
            .expect("test operation should succeed");
        let dot = String::from_utf8(output).expect("test operation should succeed");

        assert!(dot.contains("plain conclusion with no special chars"));
        assert!(!dot.contains('\\'), "no character here needed escaping");
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
            .expect("test operation should succeed");

        let tree = String::from_utf8(output).expect("test operation should succeed");
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
            .expect("test operation should succeed");

        let text = String::from_utf8(output).expect("test operation should succeed");
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
            .expect("test operation should succeed");

        let json = String::from_utf8(output).expect("test operation should succeed");
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
        // Non-ASCII passes through raw (valid inside a JSON string as-is);
        // a control character below 0x20 is not one of the five characters
        // `escape_json` special-cases, so it also passes through unescaped
        // here -- JSON's grammar technically requires *some* escape for
        // control characters, but that gap is pre-existing and out of scope
        // for this fix (this test exists to pin current behaviour, not
        // claim strict JSON conformance).
        assert_eq!(escape_json("caf\u{e9}"), "caf\u{e9}");
    }

    /// Confirmation (not a fix): `visualize_json` already routes `rule` and
    /// `conclusion` through `escape_json` (unlike the DOT path, which did
    /// not route its label through any escaper at all). A `"` in a
    /// conclusion must not break the surrounding `"conclusion": "..."` JSON
    /// string value.
    #[test]
    fn test_visualize_json_conclusion_with_quote_is_escaped() {
        let mut proof = Proof::new();
        proof.add_axiom("has \"quote\" inside");
        let viz = ProofVisualizer::new();

        let mut output = Vec::new();
        viz.visualize(&proof, VisualizationFormat::Json, &mut output)
            .expect("test operation should succeed");
        let json = String::from_utf8(output).expect("test operation should succeed");

        assert!(
            json.contains(r#"has \"quote\" inside"#),
            "the quote must be escaped, not raw, in: {json}"
        );
        assert!(
            !json.contains("\"conclusion\": \"has \"quote\""),
            "a raw quote must not end the conclusion string value early in: {json}"
        );
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
            .expect("test operation should succeed");

        let text = String::from_utf8(output).expect("test operation should succeed");
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

    /// Render `proof` as JSON into a `String`, for the byte-exact goldens.
    fn render_json(proof: &Proof, viz: &ProofVisualizer) -> String {
        let mut out = Vec::new();
        viz.visualize(proof, VisualizationFormat::Json, &mut out)
            .expect("test operation should succeed");
        String::from_utf8(out).expect("json output should be utf-8")
    }

    /// A three-level proof with a two-premise node at two different indents,
    /// so every delimiter the writer can emit (`{`, `}`, `},`, `]`) appears at
    /// more than one indent level.
    fn nested_proof() -> Proof {
        let mut proof = Proof::new();
        let a = proof.add_axiom("a");
        let b = proof.add_axiom("b");
        let c = proof.add_axiom("c");
        let ab = proof.add_inference("and", vec![a, b], "(and a b)");
        let abc = proof.add_inference("and", vec![ab, c], "(and (and a b) c)");
        let _root = proof.add_inference("not", vec![abc], "(not (and (and a b) c))");
        proof
    }

    /// Byte-exact pin of the JSON rendering.
    ///
    /// The delimiter frames used to be `String`s with the indent already baked
    /// in; they now carry an indent *level* rendered at pop time. The whole
    /// point of that change is that it is invisible in the output, and a
    /// `contains`-style assertion would not notice an off-by-one in the indent
    /// of a `}` or a `]`. Hence the full text.
    #[test]
    fn test_visualize_json_nested_output_is_byte_exact() {
        let expected = r#"{
  "type": "proof",
  "node_count": 6,
  "depth": 3,
  "root": {
    "id": "p5",
    "type": "inference",
    "rule": "not",
    "conclusion": "(not (and (and a b) c))",
    "premises": [
      {
        "id": "p4",
        "type": "inference",
        "rule": "and",
        "conclusion": "(and (and a b) c)",
        "premises": [
          {
            "id": "p3",
            "type": "inference",
            "rule": "and",
            "conclusion": "(and a b)",
            "premises": [
              {
                "id": "p0",
                "type": "axiom",
                "conclusion": "a"
              },
              {
                "id": "p1",
                "type": "axiom",
                "conclusion": "b"
              }
            ]
          },
          {
            "id": "p2",
            "type": "axiom",
            "conclusion": "c"
          }
        ]
      }
    ]
  }
}
"#;
        assert_eq!(
            render_json(&nested_proof(), &ProofVisualizer::new()),
            expected
        );
    }

    /// Byte-exact pin of the `max_depth` path: the truncation marker replaces
    /// the node body but the surrounding braces and bracket are still emitted,
    /// at the indents the untruncated node would have used.
    #[test]
    fn test_visualize_json_truncated_output_is_byte_exact() {
        let mut proof = Proof::new();
        let mut current = proof.add_axiom("p0");
        for level in 1..=5 {
            current = proof.add_inference("step", vec![current], format!("p{level}"));
        }

        let expected = r#"{
  "type": "proof",
  "node_count": 6,
  "depth": 5,
  "root": {
    "id": "p5",
    "type": "inference",
    "rule": "step",
    "conclusion": "p5",
    "premises": [
      {
        "id": "p4",
        "type": "inference",
        "rule": "step",
        "conclusion": "p4",
        "premises": [
          {
            "truncated": true
          }
        ]
      }
    ]
  }
}
"#;
        assert_eq!(
            render_json(&proof, &ProofVisualizer::new().with_max_depth(2)),
            expected
        );
    }

    /// Byte-exact pin of an inference with no premises at all: no `"premises"`
    /// key and therefore none of the delimiter frames.
    ///
    /// Note the trailing comma after `"conclusion"`, which makes this
    /// particular output *not* valid JSON. That is a pre-existing defect of the
    /// zero-premise case (the conclusion line is written with a comma on the
    /// assumption that a `"premises"` key follows); it is pinned here as
    /// current behaviour, not endorsed.
    #[test]
    fn test_visualize_json_inference_without_premises_is_byte_exact() {
        let mut proof = Proof::new();
        let _ = proof.add_inference("assume", vec![], "nothing");

        let expected = r#"{
  "type": "proof",
  "node_count": 1,
  "depth": 1,
  "root": {
    "id": "p0",
    "type": "inference",
    "rule": "assume",
    "conclusion": "nothing",
  }
}
"#;
        assert_eq!(render_json(&proof, &ProofVisualizer::new()), expected);
    }

    /// Byte-exact pin of the dangling-premise path: ids that resolve to no node
    /// are dropped, and the comma bookkeeping runs over the surviving premises
    /// only, so the single survivor gets a plain `}` rather than a `},`.
    #[test]
    fn test_visualize_json_dangling_premises_are_omitted_byte_exact() {
        let mut proof = Proof::new();
        let k = proof.add_axiom("k");
        let _ = proof.add_inference("mix", vec![ProofNodeId(999), k, ProofNodeId(998)], "mixed");

        let expected = r#"{
  "type": "proof",
  "node_count": 2,
  "depth": 1,
  "root": {
    "id": "p1",
    "type": "inference",
    "rule": "mix",
    "conclusion": "mixed",
    "premises": [
      {
        "id": "p0",
        "type": "axiom",
        "conclusion": "k"
      }
    ]
  }
}
"#;
        assert_eq!(render_json(&proof, &ProofVisualizer::new()), expected);
    }

    /// A `Write` sink that keeps counters instead of bytes.
    ///
    /// The deep-rendering test below must not collect its output into a
    /// `String` or a `Vec`: the JSON output is Theta(depth^2) bytes by design,
    /// so buffering it would measure output volume rather than the live heap
    /// the test is actually about (and would cost gigabytes).
    #[derive(Debug, Default)]
    struct CountingSink {
        /// Total bytes written.
        bytes: usize,
        /// Number of `\n` bytes written.
        lines: usize,
    }

    impl Write for CountingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes += buf.len();
            self.lines += buf.iter().filter(|&&b| b == b'\n').count();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A single-premise chain of `depth` inferences over one axiom, with
    /// fixed-width conclusions so the rendered size is a closed form of the
    /// depth alone.
    fn fixed_width_chain(depth: usize) -> Proof {
        let mut proof = Proof::new();
        let mut current = proof.add_axiom("c000000");
        for level in 1..=depth {
            current = proof.add_inference("step", vec![current], format!("c{level:06}"));
        }
        proof
    }

    /// Exact (bytes, lines) of `fixed_width_chain(depth)` rendered as JSON with
    /// ids suppressed.
    ///
    /// Derived from the format, not from a recording, so it is sensitive to the
    /// indent level of every line — including the delimiter lines, whose indent
    /// is now computed at pop time. The `14 * indent` term is what a uniform
    /// off-by-one in a delimiter's indent would perturb.
    fn expected_chain_json_size(depth: usize) -> (usize, usize) {
        // Header emitted by `visualize_json` before the root node body.
        let header = format!(
            "{{\n  \"type\": \"proof\",\n  \"node_count\": {},\n  \"depth\": {},\n  \"root\": {{\n",
            depth + 1,
            depth
        )
        .len();
        // Closing `  }` and `}`.
        let footer = "  }\n}\n".len();

        let mut bytes = header + footer;
        let mut lines = 5 + 2;

        for level in 0..depth {
            // Inference node: 5 lines at this indent (`type`, `rule`,
            // `conclusion`, `"premises": [`, `]`) and 2 at one deeper (`{`,
            // `}`), for 14 indent characters per level plus 82 fixed bytes.
            let indent = 2 + 2 * level;
            bytes += 14 * indent + 86;
            lines += 7;
        }

        // Axiom leaf: `type` and `conclusion` only, no trailing comma.
        let leaf_indent = 2 + 2 * depth;
        bytes += 4 * leaf_indent + 41;
        lines += 2;

        (bytes, lines)
    }

    /// Cross-check of [`expected_chain_json_size`] against real renderings, so
    /// the deep test below is pinned to the format rather than to arithmetic
    /// that could drift with it.
    #[test]
    fn test_chain_json_size_formula_matches_rendering() {
        for depth in 1..=6 {
            let proof = fixed_width_chain(depth);
            let rendered = render_json(&proof, &ProofVisualizer::new().with_show_ids(false));
            assert_eq!(
                (rendered.len(), rendered.lines().count()),
                expected_chain_json_size(depth),
                "size formula disagrees with the rendering at depth {depth}:\n{rendered}"
            );
        }
    }

    /// Regression: the JSON writer must retain only Theta(depth), not
    /// Theta(depth^2).
    ///
    /// The closing `]` / `}` / `},` and the opening `{` used to be built as
    /// fully-indented `String`s and pushed *below* the child frame, so a chain
    /// of depth d kept three O(d)-length strings alive per ancestor level —
    /// about 6*d^2 bytes, or ~15 GB at depth 60,000. That is live heap no sink
    /// can avoid, and it is what drove a 32 GB test process.
    ///
    /// The depth here is deliberately moderate. The *output* is inherently
    /// Theta(depth^2) (about 1.4 GB at 10,000, streamed to the counting sink
    /// and discarded), so a depth of 60,000 would take tens of gigabytes of
    /// formatting work no matter how little is retained — the quadratic that
    /// remains is documented at module level, not tested. What this test pins
    /// is that the render completes, and that every line — delimiters included
    /// — lands at exactly the indent the format calls for even 10,000 levels
    /// down. The absence of an owning frame is pinned separately, at compile
    /// time, by the `needs_drop` assertion on `JsonFrame`.
    #[test]
    fn test_visualize_json_deep_chain_is_linear_in_live_heap() {
        const DEPTH: usize = 10_000;

        let proof = fixed_width_chain(DEPTH);
        let mut sink = CountingSink::default();
        ProofVisualizer::new()
            .with_show_ids(false)
            .visualize(&proof, VisualizationFormat::Json, &mut sink)
            .expect("test operation should succeed");

        assert_eq!(
            (sink.bytes, sink.lines),
            expected_chain_json_size(DEPTH),
            "deep rendering does not match the format's closed-form size"
        );
    }
}
