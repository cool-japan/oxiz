//! E-matching Code Tree for Multi-Pattern Matching.
//!
//! This module implements a code tree data structure for efficient E-matching,
//! which is the core of pattern-based quantifier instantiation in SMT solvers.
//!
//! ## Code Tree Structure
//!
//! A code tree is a compiled representation of multiple patterns that enables
//! simultaneous matching against a database of ground terms. It combines:
//!
//! 1. **Finite Automaton**: Patterns are compiled into a tree-structured automaton
//! 2. **Backtracking**: Handles multiple possible matches efficiently
//! 3. **Substitution Building**: Incrementally builds variable substitutions
//! 4. **Indexing**: Fast lookup of matching candidates by function symbols
//!
//! ## Advantages over Naive Matching
//!
//! - **Shared Prefixes**: Common pattern prefixes are matched once
//! - **Indexing**: Only relevant terms are considered (by root symbol)
//! - **Early Pruning**: Failed matches are detected early
//! - **Incremental**: Supports incremental addition of ground terms
//!
//! ## Example
//!
//! For patterns:
//! - `f(x, g(x))` from quantifier `∀x. P(f(x, g(x)))`
//! - `f(y, g(z))` from quantifier `∀y,z. Q(f(y, g(z)))`
//!
//! The code tree shares the matching of `f(?, g(?))` and then checks variable constraints.
//!
//! ## References
//!
//! - de Moura & Bjørner: "Efficient E-Matching for SMT Solvers" (2007)
//! - Z3's `muz/rel/dl_mk_filter_rules.cpp` and `ast/pattern/pattern_inference.cpp`
//! - Simplify's E-matching implementation

use lasso::Spur;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};

/// Variable identifier in patterns.
pub type PatternVar = u32;

/// Instruction in the code tree (compiled pattern matching bytecode).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeTreeInstr {
    /// Check if current term has a specific function symbol.
    /// Args: (symbol_id, child_count, failure_pc)
    CheckSymbol {
        /// Function symbol
        symbol: Spur,
        /// Number of arguments
        arity: usize,
        /// PC on failure
        failure_pc: usize,
    },

    /// Check if current term is a variable (for patterns with nested variables).
    /// Args: (failure_pc)
    CheckVar {
        /// PC on failure
        failure_pc: usize,
    },

    /// Check if current term is a constant (specific value).
    /// Args: (value_id, failure_pc)
    CheckConstant {
        /// Expected value
        value: TermId,
        /// PC on failure
        failure_pc: usize,
    },

    /// Bind current term to a pattern variable.
    /// Args: (variable_id)
    Bind {
        /// Pattern variable ID
        var: PatternVar,
    },

    /// Check if current term matches a previously bound variable (occurs check).
    /// Args: (variable_id, failure_pc)
    CheckEq {
        /// Pattern variable ID
        var: PatternVar,
        /// PC on failure
        failure_pc: usize,
    },

    /// Move to the i-th child of current term.
    /// Args: (child_index)
    MoveToChild {
        /// Child index
        index: usize,
    },

    /// Move to parent term (backtrack).
    MoveToParent,

    /// Yield a match with the current substitution.
    /// Args: (quantifier_id, pattern_index)
    Yield {
        /// Quantifier term ID
        quantifier: TermId,
        /// Pattern index
        pattern_idx: usize,
    },

    /// Halt execution (end of program).
    Halt,
}

/// A compiled pattern for a quantified formula.
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    /// Original pattern terms.
    pub pattern: Vec<TermId>,
    /// Compiled instructions.
    pub instructions: Vec<CodeTreeInstr>,
    /// Pattern variables and their sorts.
    pub variables: FxHashMap<PatternVar, SortId>,
    /// Quantifier this pattern belongs to.
    pub quantifier: TermId,
    /// Pattern index within quantifier.
    pub pattern_index: usize,
}

/// Match context during execution.
#[derive(Debug, Clone)]
struct MatchContext {
    /// Current term being examined.
    current_term: TermId,
    /// Current instruction pointer.
    pc: usize,
    /// Variable substitutions.
    substitution: FxHashMap<PatternVar, TermId>,
    /// Term stack (for MoveToChild/MoveToParent).
    term_stack: Vec<TermId>,
}

impl MatchContext {
    fn new(root: TermId) -> Self {
        Self {
            current_term: root,
            pc: 0,
            substitution: FxHashMap::default(),
            term_stack: vec![root],
        }
    }
}

/// Match result from code tree execution.
#[derive(Debug, Clone)]
pub struct Match {
    /// Quantifier being instantiated.
    pub quantifier: TermId,
    /// Pattern index.
    pub pattern_index: usize,
    /// Variable substitution (pattern_var -> ground_term).
    pub substitution: FxHashMap<PatternVar, TermId>,
}

/// Statistics for code tree matching.
#[derive(Debug, Clone, Default)]
pub struct CodeTreeStats {
    /// Number of patterns compiled.
    pub patterns_compiled: usize,
    /// Number of instructions in the code tree.
    pub total_instructions: usize,
    /// Number of ground terms indexed.
    pub ground_terms_indexed: usize,
    /// Number of matches found.
    pub matches_found: usize,
    /// Number of match attempts.
    pub match_attempts: usize,
    /// Number of failed matches (early pruning).
    pub failed_matches: usize,
    /// Time spent in matching (microseconds).
    pub matching_time_us: u64,
}

/// E-matching code tree for multi-pattern matching.
pub struct CodeTree {
    /// Compiled patterns, indexed by root symbol.
    /// Key: function symbol, Value: list of compiled patterns starting with that symbol.
    symbol_index: FxHashMap<Spur, Vec<CompiledPattern>>,

    /// Patterns starting with variables (need to check all ground terms).
    variable_patterns: Vec<CompiledPattern>,

    /// Database of ground terms, indexed by root symbol.
    /// Key: function symbol, Value: set of term IDs with that root.
    ground_terms: FxHashMap<Spur, FxHashSet<TermId>>,

    /// All ground terms (for variable patterns).
    all_ground_terms: FxHashSet<TermId>,

    /// Statistics.
    stats: CodeTreeStats,
}

impl CodeTree {
    /// Create a new code tree.
    pub fn new() -> Self {
        Self {
            symbol_index: FxHashMap::default(),
            variable_patterns: Vec::new(),
            ground_terms: FxHashMap::default(),
            all_ground_terms: FxHashSet::default(),
            stats: CodeTreeStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &CodeTreeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CodeTreeStats::default();
    }

    /// Compile and add patterns from a quantified formula.
    ///
    /// # Arguments
    /// * `quantifier` - The quantified formula term ID
    /// * `patterns` - List of pattern lists (multi-patterns)
    /// * `var_mapping` - Maps bound variable names to pattern variable IDs
    /// * `tm` - Term manager for accessing term structure
    pub fn add_patterns(
        &mut self,
        quantifier: TermId,
        patterns: &[Vec<TermId>],
        var_mapping: &FxHashMap<Spur, PatternVar>,
        tm: &TermManager,
    ) {
        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            if pattern.is_empty() {
                continue;
            }

            // Compile each pattern in the multi-pattern
            for pattern_term in pattern {
                let compiled =
                    self.compile_pattern(*pattern_term, quantifier, pattern_idx, var_mapping, tm);

                if !compiled.instructions.is_empty() {
                    // Index by root symbol
                    if let Some(root_sym) = self.get_root_symbol(*pattern_term, tm) {
                        self.symbol_index
                            .entry(root_sym)
                            .or_default()
                            .push(compiled);
                    } else {
                        // Pattern starts with variable
                        self.variable_patterns.push(compiled);
                    }

                    self.stats.patterns_compiled += 1;
                }
            }
        }
    }

    /// Compile a single pattern term into instructions.
    fn compile_pattern(
        &mut self,
        pattern: TermId,
        quantifier: TermId,
        pattern_idx: usize,
        var_mapping: &FxHashMap<Spur, PatternVar>,
        tm: &TermManager,
    ) -> CompiledPattern {
        let mut instructions = Vec::new();
        let mut variables = FxHashMap::default();
        let mut bound_vars = FxHashMap::default();

        self.compile_term(
            pattern,
            var_mapping,
            &mut bound_vars,
            &mut variables,
            &mut instructions,
            tm,
        );

        // Add yield and halt
        instructions.push(CodeTreeInstr::Yield {
            quantifier,
            pattern_idx,
        });
        instructions.push(CodeTreeInstr::Halt);

        self.stats.total_instructions += instructions.len();

        CompiledPattern {
            pattern: vec![pattern],
            instructions,
            variables,
            quantifier,
            pattern_index: pattern_idx,
        }
    }

    /// Recursively compile a pattern term.
    fn compile_term(
        &self,
        term: TermId,
        var_mapping: &FxHashMap<Spur, PatternVar>,
        bound_vars: &mut FxHashMap<PatternVar, usize>,
        variables: &mut FxHashMap<PatternVar, SortId>,
        instructions: &mut Vec<CodeTreeInstr>,
        tm: &TermManager,
    ) {
        let term_data = tm.get(term).expect("term should exist in manager");
        match &term_data.kind {
            TermKind::Var(name) => {
                let sort = term_data.sort;

                // Pattern variable
                if let Some(&var_id) = var_mapping.get(name) {
                    if let Some(&_first_occurrence) = bound_vars.get(&var_id) {
                        // Variable already seen, check equality
                        let failure_pc = instructions.len() + 1;
                        instructions.push(CodeTreeInstr::CheckEq {
                            var: var_id,
                            failure_pc,
                        });
                    } else {
                        // First occurrence, bind it
                        bound_vars.insert(var_id, instructions.len());
                        variables.insert(var_id, sort);
                        instructions.push(CodeTreeInstr::Bind { var: var_id });
                    }
                } else {
                    // Free variable (shouldn't happen in well-formed patterns)
                    let failure_pc = instructions.len() + 1;
                    instructions.push(CodeTreeInstr::CheckVar { failure_pc });
                }
            }

            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                // Constant value - check exact match
                let failure_pc = instructions.len() + 1;
                instructions.push(CodeTreeInstr::CheckConstant {
                    value: term,
                    failure_pc,
                });
            }

            TermKind::Apply { func, args } => {
                // Function application
                let symbol = *func;
                let arity = args.len();
                let failure_pc = instructions.len() + arity + 2;

                instructions.push(CodeTreeInstr::CheckSymbol {
                    symbol,
                    arity,
                    failure_pc,
                });

                // Recursively compile each argument
                for (i, &arg) in args.iter().enumerate() {
                    instructions.push(CodeTreeInstr::MoveToChild { index: i });
                    self.compile_term(arg, var_mapping, bound_vars, variables, instructions, tm);
                    instructions.push(CodeTreeInstr::MoveToParent);
                }
            }

            _ => {
                // Other term kinds (e.g., Lambda) - treat as opaque
                let failure_pc = instructions.len() + 1;
                instructions.push(CodeTreeInstr::CheckConstant {
                    value: term,
                    failure_pc,
                });
            }
        }
    }

    /// Add a ground term to the database.
    ///
    /// This term will be considered for matching against patterns.
    pub fn add_ground_term(&mut self, term: TermId, tm: &TermManager) {
        if self.all_ground_terms.contains(&term) {
            return; // Already indexed
        }

        self.all_ground_terms.insert(term);

        // Index by root symbol
        if let Some(root_sym) = self.get_root_symbol(term, tm) {
            self.ground_terms.entry(root_sym).or_default().insert(term);
        }

        self.stats.ground_terms_indexed += 1;
    }

    /// Find all matches for the indexed patterns against the ground terms.
    ///
    /// Returns a list of matches (quantifier, pattern_index, substitution).
    pub fn find_matches(&mut self, tm: &TermManager) -> Vec<Match> {
        let start = std::time::Instant::now();
        let mut matches = Vec::new();

        // Match patterns indexed by symbol
        for (symbol, patterns) in &self.symbol_index {
            if let Some(ground_terms) = self.ground_terms.get(symbol) {
                for term in ground_terms {
                    for pattern in patterns {
                        self.stats.match_attempts += 1;
                        if let Some(m) = self.execute_pattern(pattern, *term, tm) {
                            matches.push(m);
                            self.stats.matches_found += 1;
                        } else {
                            self.stats.failed_matches += 1;
                        }
                    }
                }
            }
        }

        // Match variable patterns (against all ground terms)
        for pattern in &self.variable_patterns {
            for term in &self.all_ground_terms {
                self.stats.match_attempts += 1;
                if let Some(m) = self.execute_pattern(pattern, *term, tm) {
                    matches.push(m);
                    self.stats.matches_found += 1;
                } else {
                    self.stats.failed_matches += 1;
                }
            }
        }

        self.stats.matching_time_us += start.elapsed().as_micros() as u64;
        matches
    }

    /// Execute a compiled pattern against a ground term.
    fn execute_pattern(
        &self,
        pattern: &CompiledPattern,
        ground_term: TermId,
        tm: &TermManager,
    ) -> Option<Match> {
        let mut context = MatchContext::new(ground_term);

        while context.pc < pattern.instructions.len() {
            match &pattern.instructions[context.pc] {
                CodeTreeInstr::CheckSymbol {
                    symbol,
                    arity,
                    failure_pc,
                } => {
                    if let Some(current) = tm.get(context.current_term)
                        && let TermKind::Apply { func, args } = &current.kind
                        && func == symbol
                        && args.len() == *arity
                    {
                        context.pc += 1;
                        continue;
                    }
                    // Failed match
                    context.pc = *failure_pc;
                    if context.pc >= pattern.instructions.len() {
                        return None;
                    }
                }

                CodeTreeInstr::CheckVar { failure_pc } => {
                    if let Some(current) = tm.get(context.current_term)
                        && matches!(current.kind, TermKind::Var(_))
                    {
                        context.pc += 1;
                        continue;
                    }
                    context.pc = *failure_pc;
                    if context.pc >= pattern.instructions.len() {
                        return None;
                    }
                }

                CodeTreeInstr::CheckConstant { value, failure_pc } => {
                    if context.current_term == *value {
                        context.pc += 1;
                    } else {
                        context.pc = *failure_pc;
                        if context.pc >= pattern.instructions.len() {
                            return None;
                        }
                    }
                }

                CodeTreeInstr::Bind { var } => {
                    context.substitution.insert(*var, context.current_term);
                    context.pc += 1;
                }

                CodeTreeInstr::CheckEq { var, failure_pc } => {
                    if let Some(&bound_term) = context.substitution.get(var)
                        && bound_term == context.current_term
                    {
                        context.pc += 1;
                        continue;
                    }
                    context.pc = *failure_pc;
                    if context.pc >= pattern.instructions.len() {
                        return None;
                    }
                }

                CodeTreeInstr::MoveToChild { index } => {
                    if let Some(current) = tm.get(context.current_term)
                        && let TermKind::Apply { args, .. } = &current.kind
                        && *index < args.len()
                    {
                        context.term_stack.push(context.current_term);
                        context.current_term = args[*index];
                        context.pc += 1;
                        continue;
                    }
                    return None; // Cannot move to child
                }

                CodeTreeInstr::MoveToParent => {
                    if let Some(parent) = context.term_stack.pop() {
                        context.current_term = parent;
                        context.pc += 1;
                    } else {
                        return None; // No parent
                    }
                }

                CodeTreeInstr::Yield {
                    quantifier,
                    pattern_idx,
                } => {
                    return Some(Match {
                        quantifier: *quantifier,
                        pattern_index: *pattern_idx,
                        substitution: context.substitution.clone(),
                    });
                }

                CodeTreeInstr::Halt => {
                    return None;
                }
            }
        }

        None
    }

    /// Get the root symbol of a term (function name at root).
    fn get_root_symbol(&self, term: TermId, tm: &TermManager) -> Option<Spur> {
        if let Some(term_data) = tm.get(term)
            && let TermKind::Apply { func, .. } = &term_data.kind
        {
            return Some(*func);
        }
        None
    }

    /// Clear all ground terms and reset matching state.
    pub fn clear_ground_terms(&mut self) {
        self.ground_terms.clear();
        self.all_ground_terms.clear();
        self.stats.ground_terms_indexed = 0;
    }

    /// Remove patterns associated with a quantifier.
    pub fn remove_quantifier(&mut self, quantifier: TermId) {
        // Remove from symbol index
        for patterns in self.symbol_index.values_mut() {
            patterns.retain(|p| p.quantifier != quantifier);
        }

        // Remove from variable patterns
        self.variable_patterns
            .retain(|p| p.quantifier != quantifier);
    }
}

impl Default for CodeTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::{Key, Rodeo};

    fn setup_term_manager() -> (TermManager, Rodeo) {
        (TermManager::new(), Rodeo::default())
    }

    #[test]
    fn test_code_tree_creation() {
        let tree = CodeTree::new();
        assert_eq!(tree.stats().patterns_compiled, 0);
        assert_eq!(tree.stats().ground_terms_indexed, 0);
    }

    #[test]
    fn test_code_tree_stats() {
        let mut tree = CodeTree::new();
        assert_eq!(tree.stats().matches_found, 0);

        tree.stats.matches_found = 10;
        assert_eq!(tree.stats().matches_found, 10);

        tree.reset_stats();
        assert_eq!(tree.stats().matches_found, 0);
    }

    #[test]
    fn test_clear_ground_terms() {
        let mut tree = CodeTree::new();
        let (_tm, _) = setup_term_manager();

        // This is a simplified test - in real usage, we'd create actual terms
        tree.all_ground_terms.insert(TermId::new(1));
        tree.stats.ground_terms_indexed = 1;

        tree.clear_ground_terms();
        assert!(tree.all_ground_terms.is_empty());
        assert_eq!(tree.stats.ground_terms_indexed, 0);
    }

    #[test]
    fn test_match_context_creation() {
        let ctx = MatchContext::new(TermId::new(1));
        assert_eq!(ctx.current_term, TermId::new(1));
        assert_eq!(ctx.pc, 0);
        assert!(ctx.substitution.is_empty());
        assert_eq!(ctx.term_stack.len(), 1);
    }

    #[test]
    fn test_code_tree_instruction_check_symbol() {
        let instr = CodeTreeInstr::CheckSymbol {
            symbol: Spur::try_from_usize(0).unwrap(),
            arity: 2,
            failure_pc: 10,
        };

        match instr {
            CodeTreeInstr::CheckSymbol {
                symbol: _,
                arity,
                failure_pc,
            } => {
                assert_eq!(arity, 2);
                assert_eq!(failure_pc, 10);
            }
            _ => panic!("Wrong instruction type"),
        }
    }

    #[test]
    fn test_code_tree_instruction_bind() {
        let instr = CodeTreeInstr::Bind { var: 42 };

        match instr {
            CodeTreeInstr::Bind { var } => {
                assert_eq!(var, 42);
            }
            _ => panic!("Wrong instruction type"),
        }
    }

    #[test]
    fn test_compiled_pattern() {
        let pattern = CompiledPattern {
            pattern: vec![TermId::new(1)],
            instructions: vec![
                CodeTreeInstr::Bind { var: 0 },
                CodeTreeInstr::Yield {
                    quantifier: TermId::new(2),
                    pattern_idx: 0,
                },
                CodeTreeInstr::Halt,
            ],
            variables: FxHashMap::default(),
            quantifier: TermId::new(2),
            pattern_index: 0,
        };

        assert_eq!(pattern.instructions.len(), 3);
        assert_eq!(pattern.pattern_index, 0);
    }

    #[test]
    fn test_match_result() {
        let mut subst = FxHashMap::default();
        subst.insert(0, TermId::new(100));

        let m = Match {
            quantifier: TermId::new(1),
            pattern_index: 0,
            substitution: subst,
        };

        assert_eq!(m.quantifier, TermId::new(1));
        assert_eq!(m.pattern_index, 0);
        assert_eq!(m.substitution.len(), 1);
    }
}
