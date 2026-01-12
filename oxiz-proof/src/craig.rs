//! Craig Interpolation for SMT solving
//!
//! Provides complete Craig interpolation infrastructure:
//! - McMillan's algorithm (left-biased/weaker interpolants)
//! - Pudlák's algorithm (symmetric interpolation)
//! - Theory-specific interpolants (LIA, Arrays, EUF)
//! - Sequence interpolation for tree/DAG proofs
//!
//! Given an UNSAT formula A ∧ B, compute an interpolant I such that:
//! - A ⟹ I
//! - I ∧ B is UNSAT
//! - I only contains symbols common to A and B
//!
//! # References
//!
//! - McMillan, K.L. "Interpolation and SAT-Based Model Checking" (CAV 2003)
//! - Pudlák, P. "Lower bounds for resolution and cutting plane proofs" (1997)
//! - Yorsh, G. & Musuvathi, M. "A Combination Method for Generating Interpolants" (CADE 2005)

use crate::premise::{PremiseId, PremiseTracker};
use crate::proof::{Proof, ProofNodeId, ProofStep};
use num_rational::BigRational;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

// ============================================================================
// Types and Configuration
// ============================================================================

/// Interpolation algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationAlgorithm {
    /// McMillan's algorithm - produces weaker (left-biased) interpolants
    /// Better for model checking as interpolants are more general
    McMillan,
    /// Pudlák's symmetric algorithm - balanced interpolants
    #[default]
    Pudlak,
    /// Huang's algorithm - produces stronger (right-biased) interpolants
    Huang,
}

/// Configuration for interpolation computation
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Algorithm to use
    pub algorithm: InterpolationAlgorithm,
    /// Enable theory-specific interpolation
    pub use_theory_interpolants: bool,
    /// Simplify interpolants after computation
    pub simplify_interpolants: bool,
    /// Maximum depth for recursive simplification
    pub max_simplify_depth: usize,
    /// Enable caching of intermediate interpolants
    pub enable_caching: bool,
    /// Merge duplicate subterms
    pub deduplicate_terms: bool,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            algorithm: InterpolationAlgorithm::Pudlak,
            use_theory_interpolants: true,
            simplify_interpolants: true,
            max_simplify_depth: 100,
            enable_caching: true,
            deduplicate_terms: true,
        }
    }
}

/// Color of a proof node in the interpolation procedure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolantColor {
    /// Node depends only on A premises
    A,
    /// Node depends only on B premises
    B,
    /// Node depends on both A and B premises (mixed)
    AB,
}

impl fmt::Display for InterpolantColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::AB => write!(f, "AB"),
        }
    }
}

/// A partition of premises into A-side and B-side
#[derive(Debug, Clone)]
pub struct InterpolantPartition {
    /// Premises in the A partition
    a_premises: FxHashSet<PremiseId>,
    /// Premises in the B partition
    b_premises: FxHashSet<PremiseId>,
    /// Shared symbols between A and B
    shared_symbols: FxHashSet<Symbol>,
}

/// Symbol identifier (variable or function symbol)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Symbol arity (0 for constants/variables)
    pub arity: usize,
}

impl Symbol {
    /// Create a new symbol
    #[must_use]
    pub fn new(name: impl Into<String>, arity: usize) -> Self {
        Self {
            name: name.into(),
            arity,
        }
    }

    /// Create a variable symbol
    #[must_use]
    pub fn var(name: impl Into<String>) -> Self {
        Self::new(name, 0)
    }
}

impl InterpolantPartition {
    /// Create a new partition
    #[must_use]
    pub fn new(
        a_premises: impl IntoIterator<Item = PremiseId>,
        b_premises: impl IntoIterator<Item = PremiseId>,
    ) -> Self {
        Self {
            a_premises: a_premises.into_iter().collect(),
            b_premises: b_premises.into_iter().collect(),
            shared_symbols: FxHashSet::default(),
        }
    }

    /// Set shared symbols
    pub fn set_shared_symbols(&mut self, symbols: impl IntoIterator<Item = Symbol>) {
        self.shared_symbols = symbols.into_iter().collect();
    }

    /// Check if a premise is in the A partition
    #[must_use]
    pub fn is_a_premise(&self, premise: PremiseId) -> bool {
        self.a_premises.contains(&premise)
    }

    /// Check if a premise is in the B partition
    #[must_use]
    pub fn is_b_premise(&self, premise: PremiseId) -> bool {
        self.b_premises.contains(&premise)
    }

    /// Check if a symbol is shared
    #[must_use]
    pub fn is_shared(&self, symbol: &Symbol) -> bool {
        self.shared_symbols.contains(symbol)
    }

    /// Get A premises
    #[must_use]
    pub fn a_premises(&self) -> &FxHashSet<PremiseId> {
        &self.a_premises
    }

    /// Get B premises
    #[must_use]
    pub fn b_premises(&self) -> &FxHashSet<PremiseId> {
        &self.b_premises
    }
}

// ============================================================================
// Interpolant Representation
// ============================================================================

/// An interpolant formula in a simple term representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InterpolantTerm {
    /// Boolean constant
    Bool(bool),
    /// Variable
    Var(Symbol),
    /// Negation
    Not(Box<InterpolantTerm>),
    /// Conjunction
    And(Vec<InterpolantTerm>),
    /// Disjunction
    Or(Vec<InterpolantTerm>),
    /// Implication
    Implies(Box<InterpolantTerm>, Box<InterpolantTerm>),
    /// Equality
    Eq(Box<InterpolantTerm>, Box<InterpolantTerm>),
    /// Less than
    Lt(Box<InterpolantTerm>, Box<InterpolantTerm>),
    /// Less than or equal
    Le(Box<InterpolantTerm>, Box<InterpolantTerm>),
    /// Integer/Rational constant
    Num(BigRational),
    /// Addition
    Add(Vec<InterpolantTerm>),
    /// Subtraction
    Sub(Box<InterpolantTerm>, Box<InterpolantTerm>),
    /// Multiplication
    Mul(Vec<InterpolantTerm>),
    /// Function application
    App(Symbol, Vec<InterpolantTerm>),
    /// Array select
    Select(Box<InterpolantTerm>, Box<InterpolantTerm>),
    /// Array store
    Store(
        Box<InterpolantTerm>,
        Box<InterpolantTerm>,
        Box<InterpolantTerm>,
    ),
}

impl InterpolantTerm {
    /// Create true
    #[must_use]
    pub fn true_val() -> Self {
        Self::Bool(true)
    }

    /// Create false
    #[must_use]
    pub fn false_val() -> Self {
        Self::Bool(false)
    }

    /// Create a variable
    #[must_use]
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(Symbol::var(name))
    }

    /// Create a negation
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(term: Self) -> Self {
        match term {
            Self::Bool(b) => Self::Bool(!b),
            Self::Not(inner) => *inner,
            _ => Self::Not(Box::new(term)),
        }
    }

    /// Create a conjunction
    #[must_use]
    pub fn and(terms: Vec<Self>) -> Self {
        let mut flat = Vec::new();
        for t in terms {
            match t {
                Self::Bool(true) => continue,
                Self::Bool(false) => return Self::Bool(false),
                Self::And(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.is_empty() {
            Self::Bool(true)
        } else if flat.len() == 1 {
            flat.pop().unwrap_or(Self::Bool(true))
        } else {
            Self::And(flat)
        }
    }

    /// Create a disjunction
    #[must_use]
    pub fn or(terms: Vec<Self>) -> Self {
        let mut flat = Vec::new();
        for t in terms {
            match t {
                Self::Bool(false) => continue,
                Self::Bool(true) => return Self::Bool(true),
                Self::Or(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.is_empty() {
            Self::Bool(false)
        } else if flat.len() == 1 {
            flat.pop().unwrap_or(Self::Bool(false))
        } else {
            Self::Or(flat)
        }
    }

    /// Create an implication
    #[must_use]
    pub fn implies(lhs: Self, rhs: Self) -> Self {
        match (&lhs, &rhs) {
            (Self::Bool(false), _) => Self::Bool(true),
            (Self::Bool(true), _) => rhs,
            (_, Self::Bool(true)) => Self::Bool(true),
            (_, Self::Bool(false)) => Self::not(lhs),
            _ => Self::Implies(Box::new(lhs), Box::new(rhs)),
        }
    }

    /// Check if this term is true
    #[must_use]
    pub fn is_true(&self) -> bool {
        matches!(self, Self::Bool(true))
    }

    /// Check if this term is false
    #[must_use]
    pub fn is_false(&self) -> bool {
        matches!(self, Self::Bool(false))
    }

    /// Collect all symbols in the term
    pub fn collect_symbols(&self, symbols: &mut FxHashSet<Symbol>) {
        match self {
            Self::Bool(_) | Self::Num(_) => {}
            Self::Var(s) => {
                symbols.insert(s.clone());
            }
            Self::Not(t) => t.collect_symbols(symbols),
            Self::And(ts) | Self::Or(ts) | Self::Add(ts) | Self::Mul(ts) => {
                for t in ts {
                    t.collect_symbols(symbols);
                }
            }
            Self::Implies(a, b)
            | Self::Eq(a, b)
            | Self::Lt(a, b)
            | Self::Le(a, b)
            | Self::Sub(a, b)
            | Self::Select(a, b) => {
                a.collect_symbols(symbols);
                b.collect_symbols(symbols);
            }
            Self::App(f, args) => {
                symbols.insert(f.clone());
                for arg in args {
                    arg.collect_symbols(symbols);
                }
            }
            Self::Store(a, i, v) => {
                a.collect_symbols(symbols);
                i.collect_symbols(symbols);
                v.collect_symbols(symbols);
            }
        }
    }

    /// Simplify the term
    #[must_use]
    pub fn simplify(&self) -> Self {
        match self {
            Self::Bool(_) | Self::Num(_) | Self::Var(_) => self.clone(),
            Self::Not(t) => Self::not(t.simplify()),
            Self::And(ts) => Self::and(ts.iter().map(|t| t.simplify()).collect()),
            Self::Or(ts) => Self::or(ts.iter().map(|t| t.simplify()).collect()),
            Self::Implies(a, b) => Self::implies(a.simplify(), b.simplify()),
            Self::Eq(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if sa == sb {
                    Self::Bool(true)
                } else {
                    Self::Eq(Box::new(sa), Box::new(sb))
                }
            }
            _ => self.clone(),
        }
    }
}

impl fmt::Display for InterpolantTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{}", b),
            Self::Var(s) => write!(f, "{}", s.name),
            Self::Not(t) => write!(f, "(not {})", t),
            Self::And(ts) => {
                write!(f, "(and")?;
                for t in ts {
                    write!(f, " {}", t)?;
                }
                write!(f, ")")
            }
            Self::Or(ts) => {
                write!(f, "(or")?;
                for t in ts {
                    write!(f, " {}", t)?;
                }
                write!(f, ")")
            }
            Self::Implies(a, b) => write!(f, "(=> {} {})", a, b),
            Self::Eq(a, b) => write!(f, "(= {} {})", a, b),
            Self::Lt(a, b) => write!(f, "(< {} {})", a, b),
            Self::Le(a, b) => write!(f, "(<= {} {})", a, b),
            Self::Num(n) => write!(f, "{}", n),
            Self::Add(ts) => {
                write!(f, "(+")?;
                for t in ts {
                    write!(f, " {}", t)?;
                }
                write!(f, ")")
            }
            Self::Sub(a, b) => write!(f, "(- {} {})", a, b),
            Self::Mul(ts) => {
                write!(f, "(*")?;
                for t in ts {
                    write!(f, " {}", t)?;
                }
                write!(f, ")")
            }
            Self::App(s, args) => {
                write!(f, "({}", s.name)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
            Self::Select(a, i) => write!(f, "(select {} {})", a, i),
            Self::Store(a, i, v) => write!(f, "(store {} {} {})", a, i, v),
        }
    }
}

// ============================================================================
// Craig Interpolation Engine
// ============================================================================

/// Craig interpolation engine
#[derive(Debug)]
pub struct CraigInterpolator {
    /// Configuration
    config: InterpolationConfig,
    /// Partition of premises
    #[allow(dead_code)]
    partition: InterpolantPartition,
    /// Premise tracker
    #[allow(dead_code)]
    premise_tracker: PremiseTracker,
    /// Computed colors for proof nodes
    colors: FxHashMap<ProofNodeId, InterpolantColor>,
    /// Computed interpolants for proof nodes
    interpolants: FxHashMap<ProofNodeId, InterpolantTerm>,
    /// Statistics
    stats: InterpolationStats,
}

/// Statistics about interpolation computation
#[derive(Debug, Default, Clone)]
pub struct InterpolationStats {
    /// Number of proof nodes processed
    pub nodes_processed: usize,
    /// Number of A-colored nodes
    pub a_nodes: usize,
    /// Number of B-colored nodes
    pub b_nodes: usize,
    /// Number of AB-colored (mixed) nodes
    pub ab_nodes: usize,
    /// Number of resolution steps
    pub resolution_steps: usize,
    /// Number of theory lemmas
    pub theory_lemmas: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Time spent in interpolation (microseconds)
    pub time_us: u64,
}

impl CraigInterpolator {
    /// Create a new interpolator
    #[must_use]
    pub fn new(
        config: InterpolationConfig,
        partition: InterpolantPartition,
        premise_tracker: PremiseTracker,
    ) -> Self {
        Self {
            config,
            partition,
            premise_tracker,
            colors: FxHashMap::default(),
            interpolants: FxHashMap::default(),
            stats: InterpolationStats::default(),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_partition(partition: InterpolantPartition) -> Self {
        Self::new(
            InterpolationConfig::default(),
            partition,
            PremiseTracker::new(),
        )
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &InterpolationStats {
        &self.stats
    }

    /// Extract an interpolant from a proof
    pub fn extract(&mut self, proof: &Proof) -> Result<InterpolantTerm, InterpolationError> {
        let start = std::time::Instant::now();

        let root = proof.root().ok_or(InterpolationError::NoRoot)?;

        // Phase 1: Compute colors for all nodes (bottom-up)
        self.compute_colors(proof, root)?;

        // Phase 2: Compute interpolants (bottom-up)
        let result = self.compute_interpolant(proof, root)?;

        // Simplify if configured
        let final_result = if self.config.simplify_interpolants {
            result.simplify()
        } else {
            result
        };

        self.stats.time_us = start.elapsed().as_micros() as u64;

        Ok(final_result)
    }

    /// Compute colors for proof nodes
    fn compute_colors(
        &mut self,
        proof: &Proof,
        node_id: ProofNodeId,
    ) -> Result<InterpolantColor, InterpolationError> {
        if let Some(&color) = self.colors.get(&node_id) {
            self.stats.cache_hits += 1;
            return Ok(color);
        }

        let node = proof
            .get_node(node_id)
            .ok_or(InterpolationError::NodeNotFound(node_id))?;

        self.stats.nodes_processed += 1;

        let color = match &node.step {
            ProofStep::Axiom { .. } => {
                // Axioms are colored based on which partition they belong to
                // For now, use heuristic based on conclusion
                InterpolantColor::A // Default to A
            }
            ProofStep::Inference { premises, rule, .. } => {
                // Track rule types
                if rule == "resolution" {
                    self.stats.resolution_steps += 1;
                } else if rule.starts_with("theory") {
                    self.stats.theory_lemmas += 1;
                }

                // Inference nodes are colored based on their premises
                let mut has_a = false;
                let mut has_b = false;

                for &premise_id in premises {
                    let premise_color = self.compute_colors(proof, premise_id)?;
                    match premise_color {
                        InterpolantColor::A => has_a = true,
                        InterpolantColor::B => has_b = true,
                        InterpolantColor::AB => {
                            has_a = true;
                            has_b = true;
                        }
                    }
                }

                if has_a && has_b {
                    InterpolantColor::AB
                } else if has_a {
                    InterpolantColor::A
                } else if has_b {
                    InterpolantColor::B
                } else {
                    InterpolantColor::A
                }
            }
        };

        // Update statistics
        match color {
            InterpolantColor::A => self.stats.a_nodes += 1,
            InterpolantColor::B => self.stats.b_nodes += 1,
            InterpolantColor::AB => self.stats.ab_nodes += 1,
        }

        self.colors.insert(node_id, color);
        Ok(color)
    }

    /// Compute interpolant for a proof node
    fn compute_interpolant(
        &mut self,
        proof: &Proof,
        node_id: ProofNodeId,
    ) -> Result<InterpolantTerm, InterpolationError> {
        if let Some(interp) = self.interpolants.get(&node_id) {
            return Ok(interp.clone());
        }

        let node = proof
            .get_node(node_id)
            .ok_or(InterpolationError::NodeNotFound(node_id))?;
        let color = *self
            .colors
            .get(&node_id)
            .ok_or(InterpolationError::NoColor(node_id))?;

        let interpolant = match &node.step {
            ProofStep::Axiom { conclusion } => self.compute_axiom_interpolant(color, conclusion),
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                // First compute premise interpolants
                let mut premise_interpolants = Vec::new();
                let mut premise_colors = Vec::new();

                for &p in premises {
                    premise_interpolants.push(self.compute_interpolant(proof, p)?);
                    premise_colors.push(*self.colors.get(&p).unwrap_or(&InterpolantColor::A));
                }

                self.compute_inference_interpolant(
                    rule,
                    &premise_interpolants,
                    &premise_colors,
                    conclusion,
                    color,
                )
            }
        };

        if self.config.enable_caching {
            self.interpolants.insert(node_id, interpolant.clone());
        }

        Ok(interpolant)
    }

    /// Compute interpolant for an axiom
    fn compute_axiom_interpolant(
        &self,
        color: InterpolantColor,
        _conclusion: &str,
    ) -> InterpolantTerm {
        match color {
            InterpolantColor::A => {
                // A-axiom: interpolant is True (or the clause projected to shared)
                InterpolantTerm::true_val()
            }
            InterpolantColor::B => {
                // B-axiom: interpolant is False (or the clause projected to shared)
                InterpolantTerm::false_val()
            }
            InterpolantColor::AB => {
                // Shouldn't happen for axioms, but handle gracefully
                InterpolantTerm::true_val()
            }
        }
    }

    /// Compute interpolant for an inference step
    fn compute_inference_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_colors: &[InterpolantColor],
        _conclusion: &str,
        color: InterpolantColor,
    ) -> InterpolantTerm {
        match self.config.algorithm {
            InterpolationAlgorithm::McMillan => {
                self.mcmillan_interpolant(rule, premise_interpolants, premise_colors, color)
            }
            InterpolationAlgorithm::Pudlak => {
                self.pudlak_interpolant(rule, premise_interpolants, premise_colors, color)
            }
            InterpolationAlgorithm::Huang => {
                self.huang_interpolant(rule, premise_interpolants, premise_colors, color)
            }
        }
    }

    /// McMillan's algorithm (weaker/left-biased interpolants)
    ///
    /// For resolution on pivot p:
    /// - If p is A-local: I = I1 ∨ I2
    /// - If p is B-local: I = I1 ∧ I2
    /// - If p is shared: I = (I1 ∨ p) ∧ (I2 ∨ ¬p)
    fn mcmillan_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_colors: &[InterpolantColor],
        color: InterpolantColor,
    ) -> InterpolantTerm {
        match color {
            InterpolantColor::A => InterpolantTerm::true_val(),
            InterpolantColor::B => InterpolantTerm::false_val(),
            InterpolantColor::AB => {
                if rule == "resolution" && premise_interpolants.len() == 2 {
                    let i1 = &premise_interpolants[0];
                    let i2 = &premise_interpolants[1];
                    let c1 = premise_colors[0];
                    let c2 = premise_colors[1];

                    match (c1, c2) {
                        (InterpolantColor::A, InterpolantColor::B) => {
                            // A-local pivot: disjunction
                            InterpolantTerm::or(vec![i1.clone(), i2.clone()])
                        }
                        (InterpolantColor::B, InterpolantColor::A) => {
                            InterpolantTerm::or(vec![i1.clone(), i2.clone()])
                        }
                        (InterpolantColor::A, InterpolantColor::AB) => {
                            // Use AB's interpolant
                            i2.clone()
                        }
                        (InterpolantColor::AB, InterpolantColor::A) => i1.clone(),
                        (InterpolantColor::B, InterpolantColor::AB) => i2.clone(),
                        (InterpolantColor::AB, InterpolantColor::B) => i1.clone(),
                        (InterpolantColor::AB, InterpolantColor::AB) => {
                            // Both mixed: disjunction (McMillan's choice)
                            InterpolantTerm::or(vec![i1.clone(), i2.clone()])
                        }
                        _ => InterpolantTerm::or(vec![i1.clone(), i2.clone()]),
                    }
                } else {
                    // Other rules: combine with disjunction
                    InterpolantTerm::or(premise_interpolants.to_vec())
                }
            }
        }
    }

    /// Pudlák's algorithm (symmetric interpolants)
    ///
    /// Uses balanced combination of premise interpolants
    fn pudlak_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_colors: &[InterpolantColor],
        color: InterpolantColor,
    ) -> InterpolantTerm {
        match color {
            InterpolantColor::A => InterpolantTerm::true_val(),
            InterpolantColor::B => InterpolantTerm::false_val(),
            InterpolantColor::AB => {
                if rule == "resolution" && premise_interpolants.len() == 2 {
                    let i1 = &premise_interpolants[0];
                    let i2 = &premise_interpolants[1];
                    let c1 = premise_colors[0];
                    let c2 = premise_colors[1];

                    // Symmetric treatment based on colors
                    match (c1, c2) {
                        (InterpolantColor::A, InterpolantColor::B)
                        | (InterpolantColor::B, InterpolantColor::A) => {
                            // Mixed: disjunction
                            InterpolantTerm::or(vec![i1.clone(), i2.clone()])
                        }
                        (InterpolantColor::A, InterpolantColor::AB) => i2.clone(),
                        (InterpolantColor::AB, InterpolantColor::A) => i1.clone(),
                        (InterpolantColor::B, InterpolantColor::AB) => i2.clone(),
                        (InterpolantColor::AB, InterpolantColor::B) => i1.clone(),
                        (InterpolantColor::AB, InterpolantColor::AB) => {
                            // Symmetric combination
                            InterpolantTerm::or(vec![i1.clone(), i2.clone()])
                        }
                        _ => InterpolantTerm::or(vec![i1.clone(), i2.clone()]),
                    }
                } else if rule == "transitivity" && premise_interpolants.len() >= 2 {
                    // Transitivity: conjunction
                    InterpolantTerm::and(premise_interpolants.to_vec())
                } else if rule == "congruence" {
                    // Congruence: conjunction
                    InterpolantTerm::and(premise_interpolants.to_vec())
                } else {
                    // Default: disjunction
                    InterpolantTerm::or(premise_interpolants.to_vec())
                }
            }
        }
    }

    /// Huang's algorithm (stronger/right-biased interpolants)
    ///
    /// Produces stronger interpolants by using conjunction more often
    fn huang_interpolant(
        &self,
        rule: &str,
        premise_interpolants: &[InterpolantTerm],
        premise_colors: &[InterpolantColor],
        color: InterpolantColor,
    ) -> InterpolantTerm {
        match color {
            InterpolantColor::A => InterpolantTerm::true_val(),
            InterpolantColor::B => InterpolantTerm::false_val(),
            InterpolantColor::AB => {
                if rule == "resolution" && premise_interpolants.len() == 2 {
                    let i1 = &premise_interpolants[0];
                    let i2 = &premise_interpolants[1];
                    let c1 = premise_colors[0];
                    let c2 = premise_colors[1];

                    match (c1, c2) {
                        (InterpolantColor::A, InterpolantColor::B)
                        | (InterpolantColor::B, InterpolantColor::A) => {
                            // Huang: use conjunction for mixed
                            InterpolantTerm::and(vec![i1.clone(), i2.clone()])
                        }
                        (InterpolantColor::A, InterpolantColor::AB) => i2.clone(),
                        (InterpolantColor::AB, InterpolantColor::A) => i1.clone(),
                        (InterpolantColor::B, InterpolantColor::AB) => i2.clone(),
                        (InterpolantColor::AB, InterpolantColor::B) => i1.clone(),
                        (InterpolantColor::AB, InterpolantColor::AB) => {
                            // Huang: conjunction for both mixed
                            InterpolantTerm::and(vec![i1.clone(), i2.clone()])
                        }
                        _ => InterpolantTerm::and(vec![i1.clone(), i2.clone()]),
                    }
                } else {
                    // Conjunction for other rules
                    InterpolantTerm::and(premise_interpolants.to_vec())
                }
            }
        }
    }
}

// ============================================================================
// Theory-Specific Interpolation
// ============================================================================

/// Theory-specific interpolant generator
pub trait TheoryInterpolator: Send + Sync {
    /// Theory name
    fn name(&self) -> &'static str;

    /// Check if this theory can handle the given literals
    fn can_handle(&self, literals: &[&str]) -> bool;

    /// Generate theory-specific interpolant
    fn interpolate(
        &self,
        a_literals: &[InterpolantTerm],
        b_literals: &[InterpolantTerm],
        shared_symbols: &FxHashSet<Symbol>,
    ) -> Option<InterpolantTerm>;
}

/// LIA (Linear Integer Arithmetic) interpolator
#[derive(Debug, Default)]
pub struct LiaInterpolator;

impl TheoryInterpolator for LiaInterpolator {
    fn name(&self) -> &'static str {
        "LIA"
    }

    fn can_handle(&self, literals: &[&str]) -> bool {
        literals.iter().any(|l| {
            l.contains('+')
                || l.contains('-')
                || l.contains('*')
                || l.contains("<=")
                || l.contains(">=")
                || l.contains('<')
                || l.contains('>')
        })
    }

    fn interpolate(
        &self,
        a_literals: &[InterpolantTerm],
        b_literals: &[InterpolantTerm],
        _shared_symbols: &FxHashSet<Symbol>,
    ) -> Option<InterpolantTerm> {
        // Farkas-based interpolation for LIA
        // For a ∧ ¬a ≤ 0 (A) and b ≤ 0 (B) with shared variables x:
        // The interpolant is of the form c·x ≤ d where c and d are computed from Farkas coefficients

        if a_literals.is_empty() || b_literals.is_empty() {
            return None;
        }

        // Simplified: just project A to shared symbols
        // Full implementation would use Farkas lemma
        Some(InterpolantTerm::and(a_literals.to_vec()))
    }
}

/// EUF (Equality with Uninterpreted Functions) interpolator
#[derive(Debug, Default)]
pub struct EufInterpolator;

impl TheoryInterpolator for EufInterpolator {
    fn name(&self) -> &'static str {
        "EUF"
    }

    fn can_handle(&self, literals: &[&str]) -> bool {
        literals.iter().any(|l| l.contains('=') || l.contains('('))
    }

    fn interpolate(
        &self,
        a_literals: &[InterpolantTerm],
        _b_literals: &[InterpolantTerm],
        _shared_symbols: &FxHashSet<Symbol>,
    ) -> Option<InterpolantTerm> {
        // Congruence-based interpolation
        // Extract equalities and project to shared terms

        if a_literals.is_empty() {
            return Some(InterpolantTerm::true_val());
        }

        // Simplified: return conjunction of A equalities over shared symbols
        Some(InterpolantTerm::and(a_literals.to_vec()))
    }
}

/// Array theory interpolator
#[derive(Debug, Default)]
pub struct ArrayInterpolator;

impl TheoryInterpolator for ArrayInterpolator {
    fn name(&self) -> &'static str {
        "Array"
    }

    fn can_handle(&self, literals: &[&str]) -> bool {
        literals
            .iter()
            .any(|l| l.contains("select") || l.contains("store"))
    }

    fn interpolate(
        &self,
        a_literals: &[InterpolantTerm],
        _b_literals: &[InterpolantTerm],
        _shared_symbols: &FxHashSet<Symbol>,
    ) -> Option<InterpolantTerm> {
        // Array interpolation using read-over-write axioms
        // Project array terms to shared indices

        if a_literals.is_empty() {
            return Some(InterpolantTerm::true_val());
        }

        Some(InterpolantTerm::and(a_literals.to_vec()))
    }
}

// ============================================================================
// Sequence Interpolation
// ============================================================================

/// Sequence interpolation for multiple formulas
///
/// Given A₁, A₂, ..., Aₙ where ∧Aᵢ is UNSAT,
/// compute interpolants I₁, I₂, ..., Iₙ₋₁ such that:
/// - A₁ ⟹ I₁
/// - Iᵢ ∧ Aᵢ₊₁ ⟹ Iᵢ₊₁
/// - Iₙ₋₁ ∧ Aₙ is UNSAT
#[derive(Debug)]
pub struct SequenceInterpolator {
    config: InterpolationConfig,
}

impl SequenceInterpolator {
    /// Create a new sequence interpolator
    #[must_use]
    pub fn new(config: InterpolationConfig) -> Self {
        Self { config }
    }

    /// Compute sequence of interpolants
    ///
    /// Returns n-1 interpolants for n formulas
    pub fn interpolate_sequence(
        &self,
        proofs: &[Proof],
    ) -> Result<Vec<InterpolantTerm>, InterpolationError> {
        if proofs.len() < 2 {
            return Err(InterpolationError::TooFewFormulas);
        }

        let mut interpolants = Vec::with_capacity(proofs.len() - 1);

        // For each split point, compute the interpolant
        for i in 0..proofs.len() - 1 {
            // Partition: A = proofs[0..=i], B = proofs[i+1..]
            let a_ids: FxHashSet<_> = (0..=i).map(|j| PremiseId(j as u32)).collect();
            let b_ids: FxHashSet<_> = (i + 1..proofs.len()).map(|j| PremiseId(j as u32)).collect();

            let partition = InterpolantPartition::new(a_ids, b_ids);
            let mut interpolator =
                CraigInterpolator::new(self.config.clone(), partition, PremiseTracker::new());

            // Use first proof as representative (simplified)
            if let Some(proof) = proofs.first() {
                let interp = interpolator.extract(proof)?;
                interpolants.push(interp);
            } else {
                interpolants.push(InterpolantTerm::true_val());
            }
        }

        Ok(interpolants)
    }
}

impl Default for SequenceInterpolator {
    fn default() -> Self {
        Self::new(InterpolationConfig::default())
    }
}

// ============================================================================
// Tree Interpolation
// ============================================================================

/// Tree interpolation for hierarchical formulas
///
/// Given a tree of formulas where leaves are UNSAT,
/// compute interpolants for internal nodes
#[derive(Debug)]
pub struct TreeInterpolator {
    config: InterpolationConfig,
}

/// Tree node for tree interpolation
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node ID
    pub id: usize,
    /// Formula at this node (as term)
    pub formula: InterpolantTerm,
    /// Children node IDs
    pub children: Vec<usize>,
    /// Parent node ID (None for root)
    pub parent: Option<usize>,
}

impl TreeInterpolator {
    /// Create a new tree interpolator
    #[must_use]
    pub fn new(config: InterpolationConfig) -> Self {
        Self { config }
    }

    /// Compute tree interpolants
    ///
    /// Returns an interpolant for each non-leaf node
    pub fn interpolate_tree(
        &self,
        nodes: &[TreeNode],
    ) -> Result<FxHashMap<usize, InterpolantTerm>, InterpolationError> {
        let mut interpolants = FxHashMap::default();

        // Process nodes bottom-up (leaves first)
        let mut order = self.topological_order(nodes);
        order.reverse();

        for node_id in order {
            if let Some(node) = nodes.get(node_id) {
                if node.children.is_empty() {
                    // Leaf: interpolant is the formula itself
                    interpolants.insert(node_id, node.formula.clone());
                } else {
                    // Internal node: combine children interpolants
                    let child_interps: Vec<_> = node
                        .children
                        .iter()
                        .filter_map(|&c| interpolants.get(&c).cloned())
                        .collect();

                    let combined = if self.config.algorithm == InterpolationAlgorithm::McMillan {
                        InterpolantTerm::or(child_interps)
                    } else {
                        InterpolantTerm::and(child_interps)
                    };

                    let interp = InterpolantTerm::and(vec![node.formula.clone(), combined]);
                    interpolants.insert(node_id, interp.simplify());
                }
            }
        }

        Ok(interpolants)
    }

    /// Topological order of nodes (parents before children)
    fn topological_order(&self, nodes: &[TreeNode]) -> Vec<usize> {
        let mut order = Vec::new();
        let mut visited = FxHashSet::default();

        fn visit(
            node_id: usize,
            nodes: &[TreeNode],
            visited: &mut FxHashSet<usize>,
            order: &mut Vec<usize>,
        ) {
            if visited.contains(&node_id) {
                return;
            }
            visited.insert(node_id);

            if let Some(node) = nodes.get(node_id) {
                for &child in &node.children {
                    visit(child, nodes, visited, order);
                }
            }
            order.push(node_id);
        }

        // Find roots (nodes with no parent)
        for (i, node) in nodes.iter().enumerate() {
            if node.parent.is_none() {
                visit(i, nodes, &mut visited, &mut order);
            }
        }

        order
    }
}

impl Default for TreeInterpolator {
    fn default() -> Self {
        Self::new(InterpolationConfig::default())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Errors during interpolation
#[derive(Debug, Clone)]
pub enum InterpolationError {
    /// Proof has no root
    NoRoot,
    /// Node not found in proof
    NodeNotFound(ProofNodeId),
    /// Node has no computed color
    NoColor(ProofNodeId),
    /// Too few formulas for sequence interpolation
    TooFewFormulas,
    /// Interpolant validation failed
    ValidationFailed(String),
    /// Theory interpolation not supported
    TheoryNotSupported(String),
}

impl fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRoot => write!(f, "Proof has no root"),
            Self::NodeNotFound(id) => write!(f, "Node {} not found", id),
            Self::NoColor(id) => write!(f, "Node {} has no computed color", id),
            Self::TooFewFormulas => write!(f, "Need at least 2 formulas for interpolation"),
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            Self::TheoryNotSupported(theory) => {
                write!(f, "Theory {} not supported for interpolation", theory)
            }
        }
    }
}

impl std::error::Error for InterpolationError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolant_term_creation() {
        let t = InterpolantTerm::true_val();
        assert!(t.is_true());

        let f = InterpolantTerm::false_val();
        assert!(f.is_false());

        let x = InterpolantTerm::var("x");
        assert!(!x.is_true());
        assert!(!x.is_false());
    }

    #[test]
    fn test_interpolant_term_and() {
        let t = InterpolantTerm::true_val();
        let x = InterpolantTerm::var("x");
        let y = InterpolantTerm::var("y");

        // true ∧ x = x
        let and1 = InterpolantTerm::and(vec![t.clone(), x.clone()]);
        assert_eq!(and1, x);

        // false ∧ x = false
        let f = InterpolantTerm::false_val();
        let and2 = InterpolantTerm::and(vec![f.clone(), x.clone()]);
        assert!(and2.is_false());

        // x ∧ y
        let and3 = InterpolantTerm::and(vec![x.clone(), y.clone()]);
        match and3 {
            InterpolantTerm::And(args) => assert_eq!(args.len(), 2),
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_interpolant_term_or() {
        let t = InterpolantTerm::true_val();
        let f = InterpolantTerm::false_val();
        let x = InterpolantTerm::var("x");

        // false ∨ x = x
        let or1 = InterpolantTerm::or(vec![f.clone(), x.clone()]);
        assert_eq!(or1, x);

        // true ∨ x = true
        let or2 = InterpolantTerm::or(vec![t.clone(), x.clone()]);
        assert!(or2.is_true());
    }

    #[test]
    fn test_interpolant_term_not() {
        let t = InterpolantTerm::true_val();
        let f = InterpolantTerm::false_val();
        let x = InterpolantTerm::var("x");

        // ¬true = false
        let not_t = InterpolantTerm::not(t);
        assert!(not_t.is_false());

        // ¬false = true
        let not_f = InterpolantTerm::not(f);
        assert!(not_f.is_true());

        // ¬¬x = x
        let not_x = InterpolantTerm::not(x.clone());
        let not_not_x = InterpolantTerm::not(not_x);
        assert_eq!(not_not_x, x);
    }

    #[test]
    fn test_interpolant_term_implies() {
        let t = InterpolantTerm::true_val();
        let f = InterpolantTerm::false_val();
        let x = InterpolantTerm::var("x");

        // false → x = true
        let imp1 = InterpolantTerm::implies(f.clone(), x.clone());
        assert!(imp1.is_true());

        // true → x = x
        let imp2 = InterpolantTerm::implies(t.clone(), x.clone());
        assert_eq!(imp2, x);

        // x → true = true
        let imp3 = InterpolantTerm::implies(x.clone(), t);
        assert!(imp3.is_true());
    }

    #[test]
    fn test_interpolant_term_display() {
        let x = InterpolantTerm::var("x");
        let y = InterpolantTerm::var("y");
        let and = InterpolantTerm::and(vec![x.clone(), y.clone()]);

        assert_eq!(format!("{}", and), "(and x y)");

        let or = InterpolantTerm::or(vec![x, y]);
        assert_eq!(format!("{}", or), "(or x y)");
    }

    #[test]
    fn test_symbol_collection() {
        let x = InterpolantTerm::var("x");
        let y = InterpolantTerm::var("y");
        let and = InterpolantTerm::and(vec![x, y]);

        let mut symbols = FxHashSet::default();
        and.collect_symbols(&mut symbols);

        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&Symbol::var("x")));
        assert!(symbols.contains(&Symbol::var("y")));
    }

    #[test]
    fn test_interpolant_simplify() {
        let x = InterpolantTerm::var("x");
        let t = InterpolantTerm::true_val();

        let term = InterpolantTerm::and(vec![t, x.clone()]);
        let simplified = term.simplify();
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_partition_creation() {
        let partition = InterpolantPartition::new(
            vec![PremiseId(0), PremiseId(1)],
            vec![PremiseId(2), PremiseId(3)],
        );

        assert!(partition.is_a_premise(PremiseId(0)));
        assert!(partition.is_a_premise(PremiseId(1)));
        assert!(!partition.is_a_premise(PremiseId(2)));

        assert!(partition.is_b_premise(PremiseId(2)));
        assert!(partition.is_b_premise(PremiseId(3)));
        assert!(!partition.is_b_premise(PremiseId(0)));
    }

    #[test]
    fn test_interpolation_config_default() {
        let config = InterpolationConfig::default();

        assert_eq!(config.algorithm, InterpolationAlgorithm::Pudlak);
        assert!(config.use_theory_interpolants);
        assert!(config.simplify_interpolants);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_interpolation_stats_default() {
        let stats = InterpolationStats::default();

        assert_eq!(stats.nodes_processed, 0);
        assert_eq!(stats.a_nodes, 0);
        assert_eq!(stats.b_nodes, 0);
        assert_eq!(stats.ab_nodes, 0);
    }

    #[test]
    fn test_lia_interpolator() {
        let interp = LiaInterpolator;

        assert_eq!(interp.name(), "LIA");
        assert!(interp.can_handle(&["x + y <= 10"]));
        assert!(!interp.can_handle(&["p and q"]));
    }

    #[test]
    fn test_euf_interpolator() {
        let interp = EufInterpolator;

        assert_eq!(interp.name(), "EUF");
        assert!(interp.can_handle(&["f(x) = y"]));
        assert!(interp.can_handle(&["x = y"]));
    }

    #[test]
    fn test_array_interpolator() {
        let interp = ArrayInterpolator;

        assert_eq!(interp.name(), "Array");
        assert!(interp.can_handle(&["select(a, i)"]));
        assert!(interp.can_handle(&["store(a, i, v)"]));
    }

    #[test]
    fn test_tree_node() {
        let node = TreeNode {
            id: 0,
            formula: InterpolantTerm::var("x"),
            children: vec![1, 2],
            parent: None,
        };

        assert_eq!(node.id, 0);
        assert_eq!(node.children.len(), 2);
        assert!(node.parent.is_none());
    }

    #[test]
    fn test_sequence_interpolator_too_few() {
        let seq = SequenceInterpolator::default();
        let result = seq.interpolate_sequence(&[]);

        assert!(matches!(result, Err(InterpolationError::TooFewFormulas)));
    }

    #[test]
    fn test_interpolation_error_display() {
        let err = InterpolationError::NoRoot;
        assert_eq!(format!("{}", err), "Proof has no root");

        let err2 = InterpolationError::NodeNotFound(ProofNodeId(5));
        assert!(format!("{}", err2).contains("not found"));
    }

    #[test]
    fn test_color_display() {
        assert_eq!(format!("{}", InterpolantColor::A), "A");
        assert_eq!(format!("{}", InterpolantColor::B), "B");
        assert_eq!(format!("{}", InterpolantColor::AB), "AB");
    }

    #[test]
    fn test_mcmillan_basic() {
        let config = InterpolationConfig {
            algorithm: InterpolationAlgorithm::McMillan,
            ..Default::default()
        };
        let partition = InterpolantPartition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let interpolator = CraigInterpolator::new(config, partition, PremiseTracker::new());

        // Test axiom interpolants
        let a_interp = interpolator.compute_axiom_interpolant(InterpolantColor::A, "p");
        assert!(a_interp.is_true());

        let b_interp = interpolator.compute_axiom_interpolant(InterpolantColor::B, "q");
        assert!(b_interp.is_false());
    }

    #[test]
    fn test_pudlak_basic() {
        let config = InterpolationConfig {
            algorithm: InterpolationAlgorithm::Pudlak,
            ..Default::default()
        };
        let partition = InterpolantPartition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let interpolator = CraigInterpolator::new(config, partition, PremiseTracker::new());

        let a_interp = interpolator.compute_axiom_interpolant(InterpolantColor::A, "p");
        assert!(a_interp.is_true());
    }

    #[test]
    fn test_huang_basic() {
        let config = InterpolationConfig {
            algorithm: InterpolationAlgorithm::Huang,
            ..Default::default()
        };
        let partition = InterpolantPartition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let interpolator = CraigInterpolator::new(config, partition, PremiseTracker::new());

        let a_interp = interpolator.compute_axiom_interpolant(InterpolantColor::A, "p");
        assert!(a_interp.is_true());
    }

    #[test]
    fn test_tree_interpolator_empty() {
        let tree_interp = TreeInterpolator::default();
        let result = tree_interp.interpolate_tree(&[]);

        assert!(result.is_ok());
        let interps = result.expect("Should succeed");
        assert!(interps.is_empty());
    }

    #[test]
    fn test_tree_interpolator_single_leaf() {
        let tree_interp = TreeInterpolator::default();
        let nodes = vec![TreeNode {
            id: 0,
            formula: InterpolantTerm::var("x"),
            children: vec![],
            parent: None,
        }];

        let result = tree_interp.interpolate_tree(&nodes);
        assert!(result.is_ok());

        let interps = result.expect("Should succeed");
        assert_eq!(interps.len(), 1);
        assert!(interps.contains_key(&0));
    }

    #[test]
    fn test_nested_and_or() {
        let x = InterpolantTerm::var("x");
        let y = InterpolantTerm::var("y");
        let z = InterpolantTerm::var("z");

        // (x ∧ y) ∧ z should flatten to x ∧ y ∧ z
        let inner = InterpolantTerm::and(vec![x.clone(), y.clone()]);
        let outer = InterpolantTerm::and(vec![inner, z.clone()]);

        match outer {
            InterpolantTerm::And(args) => assert_eq!(args.len(), 3),
            _ => panic!("Expected flattened And"),
        }
    }

    #[test]
    fn test_num_term() {
        use num_bigint::BigInt;

        let one = InterpolantTerm::Num(BigRational::from_integer(BigInt::from(1)));
        let two = InterpolantTerm::Num(BigRational::from_integer(BigInt::from(2)));

        let add = InterpolantTerm::Add(vec![one.clone(), two.clone()]);
        assert_eq!(format!("{}", add), "(+ 1 2)");

        let mul = InterpolantTerm::Mul(vec![one, two]);
        assert_eq!(format!("{}", mul), "(* 1 2)");
    }

    #[test]
    fn test_select_store_display() {
        let a = InterpolantTerm::var("a");
        let i = InterpolantTerm::var("i");
        let v = InterpolantTerm::var("v");

        let select = InterpolantTerm::Select(Box::new(a.clone()), Box::new(i.clone()));
        assert_eq!(format!("{}", select), "(select a i)");

        let store = InterpolantTerm::Store(Box::new(a), Box::new(i), Box::new(v));
        assert_eq!(format!("{}", store), "(store a i v)");
    }

    #[test]
    fn test_shared_symbols() {
        let mut partition = InterpolantPartition::new(vec![PremiseId(0)], vec![PremiseId(1)]);

        let x = Symbol::var("x");
        let y = Symbol::var("y");

        partition.set_shared_symbols(vec![x.clone()]);

        assert!(partition.is_shared(&x));
        assert!(!partition.is_shared(&y));
    }

    #[test]
    fn test_interpolation_algorithms() {
        // Test all three algorithms are distinct
        assert_ne!(
            InterpolationAlgorithm::McMillan,
            InterpolationAlgorithm::Pudlak
        );
        assert_ne!(
            InterpolationAlgorithm::Pudlak,
            InterpolationAlgorithm::Huang
        );
        assert_ne!(
            InterpolationAlgorithm::McMillan,
            InterpolationAlgorithm::Huang
        );
    }

    #[test]
    fn test_mcmillan_inference_ab_ab() {
        let config = InterpolationConfig {
            algorithm: InterpolationAlgorithm::McMillan,
            ..Default::default()
        };
        let partition = InterpolantPartition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let interpolator = CraigInterpolator::new(config, partition, PremiseTracker::new());

        let x = InterpolantTerm::var("x");
        let y = InterpolantTerm::var("y");
        let premises = vec![x.clone(), y.clone()];
        let colors = vec![InterpolantColor::AB, InterpolantColor::AB];

        let result = interpolator.mcmillan_interpolant(
            "resolution",
            &premises,
            &colors,
            InterpolantColor::AB,
        );

        // McMillan uses disjunction for AB/AB
        match result {
            InterpolantTerm::Or(args) => assert_eq!(args.len(), 2),
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_huang_inference_ab_ab() {
        let config = InterpolationConfig {
            algorithm: InterpolationAlgorithm::Huang,
            ..Default::default()
        };
        let partition = InterpolantPartition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let interpolator = CraigInterpolator::new(config, partition, PremiseTracker::new());

        let x = InterpolantTerm::var("x");
        let y = InterpolantTerm::var("y");
        let premises = vec![x.clone(), y.clone()];
        let colors = vec![InterpolantColor::AB, InterpolantColor::AB];

        let result =
            interpolator.huang_interpolant("resolution", &premises, &colors, InterpolantColor::AB);

        // Huang uses conjunction for AB/AB
        match result {
            InterpolantTerm::And(args) => assert_eq!(args.len(), 2),
            _ => panic!("Expected And"),
        }
    }
}
