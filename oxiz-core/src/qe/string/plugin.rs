//! String Theory Quantifier Elimination Plugin.
//!
//! Eliminates quantifiers over string variables using automata-based
//! decision procedures and length constraints.
//!
//! ## Strategy
//!
//! For `∃x:String. φ(x)`:
//! 1. Extract length constraints and word equations
//! 2. Build automaton representing solutions
//! 3. Check if automaton is non-empty
//! 4. Eliminate quantifier based on automaton properties
//!
//! ## References
//!
//! - "Solving String Constraints with Regex-Dependent Functions" (Lin & Barceló, 2016)
//! - Z3's `qe/qe_arith.cpp` (adapted for strings)

use crate::Term;
use rustc_hash::{FxHashMap, FxHashSet};

/// Variable identifier.
pub type VarId = usize;

/// String constraint type.
#[derive(Debug, Clone)]
pub enum StringConstraint {
    /// x = y
    Equality(VarId, VarId),
    /// x = "const"
    ConstantEquality(VarId, String),
    /// x = concat(y, z)
    Concatenation(VarId, VarId, VarId),
    /// contains(x, "pattern")
    Contains(VarId, String),
    /// length(x) op k
    Length(VarId, LengthOp, i64),
    /// x matches regex
    RegexMatch(VarId, String),
}

/// Length comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthOp {
    /// ==
    Equal,
    /// <
    Less,
    /// <=
    LessEq,
    /// >
    Greater,
    /// >=
    GreaterEq,
}

/// Configuration for string QE.
#[derive(Debug, Clone)]
pub struct StringQeConfig {
    /// Enable automata-based elimination.
    pub enable_automata: bool,
    /// Enable length-based elimination.
    pub enable_length: bool,
    /// Maximum automaton size.
    pub max_automaton_states: usize,
}

impl Default for StringQeConfig {
    fn default() -> Self {
        Self {
            enable_automata: true,
            enable_length: true,
            max_automaton_states: 10_000,
        }
    }
}

/// Statistics for string QE.
#[derive(Debug, Clone, Default)]
pub struct StringQeStats {
    /// Variables eliminated.
    pub vars_eliminated: u64,
    /// Automata constructed.
    pub automata_constructed: u64,
    /// Length constraints solved.
    pub length_constraints_solved: u64,
    /// Concatenations eliminated.
    pub concatenations_eliminated: u64,
}

/// String QE plugin.
#[derive(Debug)]
pub struct StringQePlugin {
    /// Known string constraints.
    constraints: Vec<StringConstraint>,
    /// Variable dependencies.
    dependencies: FxHashMap<VarId, FxHashSet<VarId>>,
    /// Configuration.
    config: StringQeConfig,
    /// Statistics.
    stats: StringQeStats,
}

impl StringQePlugin {
    /// Create a new string QE plugin.
    pub fn new(config: StringQeConfig) -> Self {
        Self {
            constraints: Vec::new(),
            dependencies: FxHashMap::default(),
            config,
            stats: StringQeStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(StringQeConfig::default())
    }

    /// Add a string constraint.
    pub fn add_constraint(&mut self, constraint: StringConstraint) {
        // Track dependencies
        match &constraint {
            StringConstraint::Equality(x, y) => {
                self.dependencies.entry(*x).or_default().insert(*y);
                self.dependencies.entry(*y).or_default().insert(*x);
            }
            StringConstraint::Concatenation(x, y, z) => {
                self.dependencies.entry(*x).or_default().insert(*y);
                self.dependencies.entry(*x).or_default().insert(*z);
            }
            _ => {}
        }

        self.constraints.push(constraint);
    }

    /// Eliminate quantifier over a string variable.
    pub fn eliminate(&mut self, var: VarId) -> Option<Term> {
        // Collect constraints mentioning var
        let relevant = self.collect_relevant_constraints(var);

        if relevant.is_empty() {
            // Unconstrained variable - always satisfiable
            self.stats.vars_eliminated += 1;
            return Some(self.create_true());
        }

        // Cache config flags to avoid borrow issues
        let enable_length = self.config.enable_length;
        let enable_automata = self.config.enable_automata;

        // Try length-based elimination first (simpler)
        if enable_length && let Some(result) = self.eliminate_by_length(var, &relevant) {
            self.stats.vars_eliminated += 1;
            self.stats.length_constraints_solved += 1;
            return Some(result);
        }

        // Try automata-based elimination
        if enable_automata && let Some(result) = self.eliminate_by_automaton(var, &relevant) {
            self.stats.vars_eliminated += 1;
            self.stats.automata_constructed += 1;
            return Some(result);
        }

        None
    }

    /// Collect constraints relevant to a variable.
    fn collect_relevant_constraints(&self, var: VarId) -> Vec<&StringConstraint> {
        self.constraints
            .iter()
            .filter(|c| self.mentions_var(c, var))
            .collect()
    }

    /// Check if constraint mentions a variable.
    fn mentions_var(&self, constraint: &StringConstraint, var: VarId) -> bool {
        match constraint {
            StringConstraint::Equality(x, y) => *x == var || *y == var,
            StringConstraint::ConstantEquality(x, _) => *x == var,
            StringConstraint::Concatenation(x, y, z) => *x == var || *y == var || *z == var,
            StringConstraint::Contains(x, _) => *x == var,
            StringConstraint::Length(x, _, _) => *x == var,
            StringConstraint::RegexMatch(x, _) => *x == var,
        }
    }

    /// Eliminate using length constraints.
    fn eliminate_by_length(&self, _var: VarId, constraints: &[&StringConstraint]) -> Option<Term> {
        // Check if all constraints are length-based
        let all_length = constraints
            .iter()
            .all(|c| matches!(c, StringConstraint::Length(..)));

        if !all_length {
            return None;
        }

        // Simplified: would solve length constraints and check satisfiability
        Some(self.create_true())
    }

    /// Eliminate using automaton construction.
    fn eliminate_by_automaton(
        &self,
        _var: VarId,
        _constraints: &[&StringConstraint],
    ) -> Option<Term> {
        // Simplified: would build automaton and check non-emptiness
        Some(self.create_true())
    }

    /// Create a "true" term (placeholder).
    fn create_true(&self) -> Term {
        unimplemented!("placeholder term")
    }

    /// Get statistics.
    pub fn stats(&self) -> &StringQeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = StringQeStats::default();
    }
}

impl Default for StringQePlugin {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_creation() {
        let plugin = StringQePlugin::default_config();
        assert_eq!(plugin.stats().vars_eliminated, 0);
    }

    #[test]
    fn test_add_constraint() {
        let mut plugin = StringQePlugin::default_config();
        plugin.add_constraint(StringConstraint::Equality(0, 1));
        plugin.add_constraint(StringConstraint::Length(0, LengthOp::Equal, 5));

        assert_eq!(plugin.constraints.len(), 2);
    }

    #[test]
    fn test_mentions_var() {
        let plugin = StringQePlugin::default_config();

        let eq = StringConstraint::Equality(0, 1);
        assert!(plugin.mentions_var(&eq, 0));
        assert!(plugin.mentions_var(&eq, 1));
        assert!(!plugin.mentions_var(&eq, 2));
    }

    #[test]
    fn test_collect_relevant() {
        let mut plugin = StringQePlugin::default_config();
        plugin.add_constraint(StringConstraint::Equality(0, 1));
        plugin.add_constraint(StringConstraint::Length(0, LengthOp::Equal, 5));
        plugin.add_constraint(StringConstraint::Length(2, LengthOp::Equal, 3));

        let relevant = plugin.collect_relevant_constraints(0);
        assert_eq!(relevant.len(), 2); // Equality and Length for var 0
    }

    #[test]
    fn test_dependencies() {
        let mut plugin = StringQePlugin::default_config();
        plugin.add_constraint(StringConstraint::Equality(0, 1));

        assert!(plugin.dependencies.contains_key(&0));
        assert!(plugin.dependencies.contains_key(&1));
    }

    #[test]
    fn test_stats() {
        let mut plugin = StringQePlugin::default_config();
        plugin.stats.vars_eliminated = 10;

        assert_eq!(plugin.stats().vars_eliminated, 10);

        plugin.reset_stats();
        assert_eq!(plugin.stats().vars_eliminated, 0);
    }
}
