//! Solver State Visualization.
//!
//! Provides `SolverStateSnapshot` for capturing solver state at a point in time,
//! human-readable text dumps, and DOT graph format for implication graphs.
//!
//! ## References
//!
//! - Z3's `smt/smt_model_reporter.cpp`

#[allow(unused_imports)]
use crate::prelude::*;

/// A variable assignment in the solver state.
#[derive(Debug, Clone)]
pub struct VarAssignment {
    /// Variable identifier (index or name).
    pub var_id: u32,
    /// Display name for the variable.
    pub name: String,
    /// Boolean assignment (true/false).
    pub bool_value: Option<bool>,
    /// Theory value (e.g., "3", "1/2", "#b0101").
    pub theory_value: Option<String>,
    /// Decision level at which this assignment was made.
    pub decision_level: u32,
}

/// A decision on the trail.
#[derive(Debug, Clone)]
pub struct TrailDecision {
    /// Variable that was decided.
    pub var_id: u32,
    /// Value assigned.
    pub value: bool,
    /// Decision level.
    pub level: u32,
    /// Whether this was a propagation (vs. a decision).
    pub is_propagation: bool,
    /// Reason clause index (if propagation).
    pub reason_clause: Option<u32>,
}

/// An active conflict.
#[derive(Debug, Clone)]
pub struct ActiveConflict {
    /// Conflicting clause index.
    pub clause_id: u32,
    /// Literals in the conflicting clause.
    pub literals: Vec<i64>,
    /// Description of the conflict.
    pub description: String,
}

/// Simplified theory solver state.
#[derive(Debug, Clone)]
pub struct TheorySolverState {
    /// Theory name (e.g., "EUF", "LRA", "BV").
    pub name: String,
    /// Number of tracked terms.
    pub num_terms: usize,
    /// Number of equalities.
    pub num_equalities: usize,
    /// Number of disequalities.
    pub num_disequalities: usize,
    /// Whether the theory is consistent.
    pub is_consistent: bool,
    /// Additional info lines.
    pub info: Vec<String>,
}

/// Statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct StatsSnapshot {
    /// Number of decisions made.
    pub decisions: u64,
    /// Number of conflicts encountered.
    pub conflicts: u64,
    /// Number of propagations performed.
    pub propagations: u64,
    /// Number of restarts.
    pub restarts: u64,
    /// Number of learned clauses.
    pub learned_clauses: u64,
    /// Number of theory propagations.
    pub theory_propagations: u64,
    /// Number of theory conflicts.
    pub theory_conflicts: u64,
}

/// A snapshot of the solver state at a point in time.
#[derive(Debug, Clone)]
pub struct SolverStateSnapshot {
    /// Timestamp or label for this snapshot.
    pub label: String,
    /// Variable assignments (bool + theory).
    pub assignments: Vec<VarAssignment>,
    /// Decision trail with levels.
    pub trail: Vec<TrailDecision>,
    /// Active conflicts (if any).
    pub conflicts: Vec<ActiveConflict>,
    /// Theory solver states (simplified).
    pub theory_states: Vec<TheorySolverState>,
    /// Statistics snapshot.
    pub statistics: StatsSnapshot,
}

impl SolverStateSnapshot {
    /// Create a new empty snapshot with the given label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            assignments: Vec::new(),
            trail: Vec::new(),
            conflicts: Vec::new(),
            theory_states: Vec::new(),
            statistics: StatsSnapshot::default(),
        }
    }

    /// Add a variable assignment.
    pub fn add_assignment(&mut self, assignment: VarAssignment) {
        self.assignments.push(assignment);
    }

    /// Add a trail decision.
    pub fn add_trail_entry(&mut self, entry: TrailDecision) {
        self.trail.push(entry);
    }

    /// Add an active conflict.
    pub fn add_conflict(&mut self, conflict: ActiveConflict) {
        self.conflicts.push(conflict);
    }

    /// Add a theory solver state.
    pub fn add_theory_state(&mut self, state: TheorySolverState) {
        self.theory_states.push(state);
    }

    /// Set the statistics snapshot.
    pub fn set_statistics(&mut self, stats: StatsSnapshot) {
        self.statistics = stats;
    }

    /// Format the snapshot as human-readable text.
    pub fn format_state_text(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("=== Solver State: {} ===\n\n", self.label));

        // Statistics
        out.push_str("--- Statistics ---\n");
        out.push_str(&format!("  Decisions:           {}\n", self.statistics.decisions));
        out.push_str(&format!("  Conflicts:           {}\n", self.statistics.conflicts));
        out.push_str(&format!("  Propagations:        {}\n", self.statistics.propagations));
        out.push_str(&format!("  Restarts:            {}\n", self.statistics.restarts));
        out.push_str(&format!("  Learned clauses:     {}\n", self.statistics.learned_clauses));
        out.push_str(&format!(
            "  Theory propagations: {}\n",
            self.statistics.theory_propagations
        ));
        out.push_str(&format!(
            "  Theory conflicts:    {}\n",
            self.statistics.theory_conflicts
        ));
        out.push('\n');

        // Assignments
        out.push_str(&format!("--- Assignments ({}) ---\n", self.assignments.len()));
        for a in &self.assignments {
            let bool_str = a
                .bool_value
                .map_or_else(|| "?".to_string(), |v| format!("{}", v));
            let theory_str = a
                .theory_value
                .as_deref()
                .unwrap_or("-");
            out.push_str(&format!(
                "  {} (v{}): bool={}, theory={}, level={}\n",
                a.name, a.var_id, bool_str, theory_str, a.decision_level
            ));
        }
        out.push('\n');

        // Trail
        out.push_str(&format!("--- Trail ({} entries) ---\n", self.trail.len()));
        for (i, t) in self.trail.iter().enumerate() {
            let kind = if t.is_propagation { "prop" } else { "decide" };
            let reason = t
                .reason_clause
                .map_or_else(|| "-".to_string(), |c| format!("clause #{}", c));
            out.push_str(&format!(
                "  [{}] v{} = {} ({}, level={}, reason={})\n",
                i, t.var_id, t.value, kind, t.level, reason
            ));
        }
        out.push('\n');

        // Conflicts
        if !self.conflicts.is_empty() {
            out.push_str(&format!("--- Active Conflicts ({}) ---\n", self.conflicts.len()));
            for c in &self.conflicts {
                out.push_str(&format!(
                    "  Clause #{}: {:?} -- {}\n",
                    c.clause_id, c.literals, c.description
                ));
            }
            out.push('\n');
        }

        // Theory states
        if !self.theory_states.is_empty() {
            out.push_str(&format!(
                "--- Theory States ({}) ---\n",
                self.theory_states.len()
            ));
            for ts in &self.theory_states {
                let status = if ts.is_consistent {
                    "consistent"
                } else {
                    "INCONSISTENT"
                };
                out.push_str(&format!(
                    "  {}: terms={}, eq={}, diseq={}, status={}\n",
                    ts.name, ts.num_terms, ts.num_equalities, ts.num_disequalities, status
                ));
                for info in &ts.info {
                    out.push_str(&format!("    {}\n", info));
                }
            }
        }

        out
    }

    /// Format the implication graph as DOT format.
    ///
    /// This generates a DOT graph suitable for rendering with Graphviz.
    pub fn format_state_dot(&self) -> String {
        let mut dot = ImplicationGraphDot::new("solver_state");

        // Add decision nodes
        for t in &self.trail {
            let label = format!(
                "v{} = {}\\nlevel={}",
                t.var_id, t.value, t.level
            );
            if t.is_propagation {
                dot.add_propagation_node(t.var_id, &label);
            } else {
                dot.add_decision_node(t.var_id, &label);
            }
        }

        // Add edges from reason clauses
        for t in &self.trail {
            if let Some(clause_id) = t.reason_clause {
                // The reason clause implies this propagation.
                // We add an edge from a synthetic clause node to this variable.
                let clause_node_id = 100_000 + clause_id;
                dot.add_clause_node(clause_node_id, &format!("clause #{}", clause_id));
                dot.add_edge(clause_node_id, t.var_id, "reason");
            }
        }

        // Add conflict nodes
        for c in &self.conflicts {
            let conflict_node_id = 200_000 + c.clause_id;
            dot.add_conflict_node(conflict_node_id, &format!("CONFLICT\\n{}", c.description));

            // Add edges from involved literals to conflict
            for &lit in &c.literals {
                let var_id = lit.unsigned_abs() as u32;
                dot.add_edge(var_id, conflict_node_id, "conflict");
            }
        }

        dot.to_dot()
    }
}

/// Node kind in the DOT graph.
#[derive(Debug, Clone)]
enum DotNodeKind {
    /// A decision node.
    Decision,
    /// A propagation node.
    Propagation,
    /// A clause node.
    Clause,
    /// A conflict node.
    Conflict,
}

/// A node in the DOT graph.
#[derive(Debug, Clone)]
struct DotNode {
    /// Node ID.
    id: u32,
    /// Label.
    label: String,
    /// Kind.
    kind: DotNodeKind,
}

/// An edge in the DOT graph.
#[derive(Debug, Clone)]
struct DotEdge {
    /// Source node ID.
    from: u32,
    /// Target node ID.
    to: u32,
    /// Label.
    label: String,
}

/// Generates DOT format for implication graphs used in conflict analysis.
#[derive(Debug)]
pub struct ImplicationGraphDot {
    /// Graph name.
    name: String,
    /// Nodes.
    nodes: Vec<DotNode>,
    /// Edges.
    edges: Vec<DotEdge>,
}

impl ImplicationGraphDot {
    /// Create a new DOT graph generator.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a decision node.
    pub fn add_decision_node(&mut self, id: u32, label: &str) {
        self.nodes.push(DotNode {
            id,
            label: label.to_string(),
            kind: DotNodeKind::Decision,
        });
    }

    /// Add a propagation node.
    pub fn add_propagation_node(&mut self, id: u32, label: &str) {
        self.nodes.push(DotNode {
            id,
            label: label.to_string(),
            kind: DotNodeKind::Propagation,
        });
    }

    /// Add a clause node.
    pub fn add_clause_node(&mut self, id: u32, label: &str) {
        self.nodes.push(DotNode {
            id,
            label: label.to_string(),
            kind: DotNodeKind::Clause,
        });
    }

    /// Add a conflict node.
    pub fn add_conflict_node(&mut self, id: u32, label: &str) {
        self.nodes.push(DotNode {
            id,
            label: label.to_string(),
            kind: DotNodeKind::Conflict,
        });
    }

    /// Add an edge.
    pub fn add_edge(&mut self, from: u32, to: u32, label: &str) {
        self.edges.push(DotEdge {
            from,
            to,
            label: label.to_string(),
        });
    }

    /// Generate the DOT format string.
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("digraph {} {{\n", self.name));
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [fontname=\"Helvetica\"];\n\n");

        for node in &self.nodes {
            let (shape, color, style) = match node.kind {
                DotNodeKind::Decision => ("box", "blue", "filled"),
                DotNodeKind::Propagation => ("ellipse", "green", "filled"),
                DotNodeKind::Clause => ("diamond", "gray", "filled"),
                DotNodeKind::Conflict => ("octagon", "red", "filled"),
            };
            out.push_str(&format!(
                "  n{} [label=\"{}\", shape={}, color={}, style={}, fillcolor=\"{}30\"];\n",
                node.id, node.label, shape, color, style, color
            ));
        }

        out.push('\n');
        for edge in &self.edges {
            if edge.label.is_empty() {
                out.push_str(&format!("  n{} -> n{};\n", edge.from, edge.to));
            } else {
                out.push_str(&format!(
                    "  n{} -> n{} [label=\"{}\"];\n",
                    edge.from, edge.to, edge.label
                ));
            }
        }

        out.push_str("}\n");
        out
    }

    /// Build a DOT graph from implication graph data.
    ///
    /// Takes a list of (literal, level, antecedents, is_decision) tuples
    /// and a conflict clause to build a complete implication graph.
    pub fn from_implication_data(
        implications: &[(i64, u32, Vec<i64>, bool)],
        conflict_clause: &[i64],
    ) -> Self {
        let mut dot = Self::new("implication_graph");

        // Add nodes
        for &(lit, level, ref _antes, is_decision) in implications {
            let var_id = lit.unsigned_abs() as u32;
            let polarity = if lit > 0 { "T" } else { "F" };
            let label = format!("x{} = {}\\nlevel={}", var_id, polarity, level);
            if is_decision {
                dot.add_decision_node(var_id, &label);
            } else {
                dot.add_propagation_node(var_id, &label);
            }
        }

        // Add edges from antecedents
        for &(lit, _level, ref antes, _is_decision) in implications {
            let to = lit.unsigned_abs() as u32;
            for &ante in antes {
                let from = ante.unsigned_abs() as u32;
                dot.add_edge(from, to, "");
            }
        }

        // Add conflict node
        if !conflict_clause.is_empty() {
            let conflict_id = 999_999;
            dot.add_conflict_node(conflict_id, "CONFLICT");
            for &lit in conflict_clause {
                let var_id = lit.unsigned_abs() as u32;
                dot.add_edge(var_id, conflict_id, "");
            }
        }

        dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_snapshot() {
        let snap = SolverStateSnapshot::new("test");
        assert_eq!(snap.label, "test");
        assert!(snap.assignments.is_empty());
        assert!(snap.trail.is_empty());
        assert!(snap.conflicts.is_empty());
        assert!(snap.theory_states.is_empty());
    }

    #[test]
    fn test_snapshot_add_assignment() {
        let mut snap = SolverStateSnapshot::new("assign_test");
        snap.add_assignment(VarAssignment {
            var_id: 1,
            name: "x".to_string(),
            bool_value: Some(true),
            theory_value: None,
            decision_level: 0,
        });
        snap.add_assignment(VarAssignment {
            var_id: 2,
            name: "y".to_string(),
            bool_value: Some(false),
            theory_value: Some("42".to_string()),
            decision_level: 1,
        });
        assert_eq!(snap.assignments.len(), 2);
        assert_eq!(snap.assignments[0].name, "x");
        assert_eq!(snap.assignments[1].theory_value.as_deref(), Some("42"));
    }

    #[test]
    fn test_snapshot_format_text() {
        let mut snap = SolverStateSnapshot::new("text_test");
        snap.set_statistics(StatsSnapshot {
            decisions: 10,
            conflicts: 3,
            propagations: 50,
            restarts: 1,
            learned_clauses: 3,
            theory_propagations: 5,
            theory_conflicts: 1,
        });
        snap.add_assignment(VarAssignment {
            var_id: 1,
            name: "p".to_string(),
            bool_value: Some(true),
            theory_value: None,
            decision_level: 0,
        });
        snap.add_trail_entry(TrailDecision {
            var_id: 1,
            value: true,
            level: 0,
            is_propagation: false,
            reason_clause: None,
        });

        let text = snap.format_state_text();
        assert!(text.contains("Solver State: text_test"));
        assert!(text.contains("Decisions:           10"));
        assert!(text.contains("p (v1): bool=true"));
        assert!(text.contains("v1 = true (decide"));
    }

    #[test]
    fn test_snapshot_format_text_with_conflict() {
        let mut snap = SolverStateSnapshot::new("conflict_test");
        snap.add_conflict(ActiveConflict {
            clause_id: 7,
            literals: vec![1, -2, 3],
            description: "Theory conflict in EUF".to_string(),
        });

        let text = snap.format_state_text();
        assert!(text.contains("Active Conflicts"));
        assert!(text.contains("Clause #7"));
        assert!(text.contains("Theory conflict in EUF"));
    }

    #[test]
    fn test_snapshot_format_dot() {
        let mut snap = SolverStateSnapshot::new("dot_test");
        snap.add_trail_entry(TrailDecision {
            var_id: 1,
            value: true,
            level: 0,
            is_propagation: false,
            reason_clause: None,
        });
        snap.add_trail_entry(TrailDecision {
            var_id: 2,
            value: false,
            level: 0,
            is_propagation: true,
            reason_clause: Some(5),
        });

        let dot = snap.format_state_dot();
        assert!(dot.contains("digraph solver_state"));
        assert!(dot.contains("n1"));
        assert!(dot.contains("n2"));
        assert!(dot.contains("shape=box")); // decision
        assert!(dot.contains("shape=ellipse")); // propagation
    }

    #[test]
    fn test_implication_graph_dot_empty() {
        let dot = ImplicationGraphDot::new("empty");
        let output = dot.to_dot();
        assert!(output.contains("digraph empty"));
        assert!(output.contains('}'));
    }

    #[test]
    fn test_implication_graph_dot_from_data() {
        let implications = vec![
            (1_i64, 0_u32, vec![], true),        // x1=T at level 0, decision
            (2, 0, vec![1], false),               // x2=T at level 0, propagated from x1
            (-3, 1, vec![], true),                // x3=F at level 1, decision
            (4, 1, vec![2, -3], false),           // x4=T at level 1, propagated from x2, x3
        ];
        let conflict_clause = vec![4, -2];

        let dot = ImplicationGraphDot::from_implication_data(&implications, &conflict_clause);
        let output = dot.to_dot();

        assert!(output.contains("digraph implication_graph"));
        assert!(output.contains("CONFLICT"));
        // Should have decision nodes and propagation nodes
        assert!(output.contains("shape=box"));    // decisions
        assert!(output.contains("shape=ellipse")); // propagations
        assert!(output.contains("shape=octagon")); // conflict
    }

    #[test]
    fn test_theory_state_display() {
        let mut snap = SolverStateSnapshot::new("theory_test");
        snap.add_theory_state(TheorySolverState {
            name: "EUF".to_string(),
            num_terms: 15,
            num_equalities: 3,
            num_disequalities: 2,
            is_consistent: true,
            info: vec!["Congruence classes: 5".to_string()],
        });
        snap.add_theory_state(TheorySolverState {
            name: "LRA".to_string(),
            num_terms: 8,
            num_equalities: 1,
            num_disequalities: 0,
            is_consistent: false,
            info: vec![],
        });

        let text = snap.format_state_text();
        assert!(text.contains("EUF: terms=15"));
        assert!(text.contains("consistent"));
        assert!(text.contains("LRA: terms=8"));
        assert!(text.contains("INCONSISTENT"));
        assert!(text.contains("Congruence classes: 5"));
    }

    #[test]
    fn test_dot_edge_labels() {
        let mut dot = ImplicationGraphDot::new("edges");
        dot.add_decision_node(1, "x1");
        dot.add_propagation_node(2, "x2");
        dot.add_edge(1, 2, "unit");
        dot.add_edge(2, 1, "");

        let output = dot.to_dot();
        assert!(output.contains("n1 -> n2 [label=\"unit\"]"));
        assert!(output.contains("n2 -> n1;"));
    }
}
