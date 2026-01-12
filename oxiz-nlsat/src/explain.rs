//! Conflict explanation for NLSAT.
//!
//! This module provides explanation generation for conflicts in the NLSAT solver.
//! When a conflict is detected (either propositional or theory-level), we need
//! to generate a learned clause that explains why the conflict occurred.
//!
//! The key components are:
//! - **Resolution-based explanation**: Standard CDCL resolution for propositional conflicts
//! - **Theory explanation**: CAD-based explanation for polynomial constraint conflicts
//! - **Projection**: Computing projections for CAD explanation
//!
//! Reference: Z3's `nlsat/nlsat_explain.cpp`

use crate::types::{Atom, AtomKind, BoolVar, IneqAtom, Lbool, Literal, NULL_BOOL_VAR};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashSet;
use std::collections::HashMap;

/// Configuration for explanation generation.
#[derive(Debug, Clone)]
pub struct ExplainConfig {
    /// Whether to minimize explanations.
    pub minimize: bool,
    /// Whether to use projection for theory explanations.
    pub use_projection: bool,
    /// Maximum projection depth.
    pub max_projection_depth: usize,
}

impl Default for ExplainConfig {
    fn default() -> Self {
        Self {
            minimize: true,
            use_projection: true,
            max_projection_depth: 10,
        }
    }
}

/// An explanation for a conflict or implication.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// The literals that form the explanation (as a clause).
    /// The explanation says: if all these literals are false, then conflict.
    pub literals: Vec<Literal>,
    /// The type of explanation.
    pub kind: ExplanationKind,
}

impl Explanation {
    /// Create a new explanation.
    pub fn new(literals: Vec<Literal>, kind: ExplanationKind) -> Self {
        Self { literals, kind }
    }

    /// Create an empty explanation (for level 0 conflicts).
    pub fn empty() -> Self {
        Self {
            literals: Vec::new(),
            kind: ExplanationKind::Empty,
        }
    }

    /// Check if the explanation is empty.
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Get the literals.
    pub fn as_clause(&self) -> &[Literal] {
        &self.literals
    }
}

/// The kind of explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplanationKind {
    /// Empty explanation (conflict at level 0).
    Empty,
    /// Resolution-based explanation (propositional).
    Resolution,
    /// Theory explanation (polynomial constraints).
    Theory,
    /// Combined explanation.
    Combined,
}

/// Context for generating explanations.
pub struct ExplainContext<'a> {
    /// Configuration.
    config: ExplainConfig,
    /// The atoms in the problem.
    atoms: &'a [Atom],
    /// Current boolean assignment.
    bool_values: &'a [Lbool],
    /// Current arithmetic assignment.
    arith_values: &'a [Option<BigRational>],
    /// Seen variables during explanation.
    seen: FxHashSet<BoolVar>,
    /// The projection set (polynomials for CAD projection).
    projection_set: Vec<Polynomial>,
    /// Cache for polynomial signs.
    sign_cache: HashMap<usize, i8>,
}

impl<'a> ExplainContext<'a> {
    /// Create a new explanation context.
    pub fn new(
        config: ExplainConfig,
        atoms: &'a [Atom],
        bool_values: &'a [Lbool],
        arith_values: &'a [Option<BigRational>],
    ) -> Self {
        Self {
            config,
            atoms,
            bool_values,
            arith_values,
            seen: FxHashSet::default(),
            projection_set: Vec::new(),
            sign_cache: HashMap::new(),
        }
    }

    /// Clear the context for reuse.
    pub fn clear(&mut self) {
        self.seen.clear();
        self.projection_set.clear();
        self.sign_cache.clear();
    }

    /// Generate an explanation for a theory conflict.
    ///
    /// A theory conflict occurs when the current assignment of boolean and
    /// arithmetic variables makes some polynomial constraint unsatisfiable.
    pub fn explain_theory_conflict(
        &mut self,
        conflicting_atom_id: usize,
        conflicting_value: bool,
    ) -> Explanation {
        let mut explanation = Vec::new();

        // Get the conflicting atom
        let atom = match self.atoms.get(conflicting_atom_id) {
            Some(a) => a,
            None => return Explanation::empty(),
        };

        // Add the conflicting literal
        let bool_var = atom.bool_var();
        if bool_var != NULL_BOOL_VAR {
            let lit = if conflicting_value {
                Literal::negative(bool_var)
            } else {
                Literal::positive(bool_var)
            };
            explanation.push(lit);
        }

        // For theory conflicts, we need to find all assignments that led to this conflict
        match atom {
            Atom::Ineq(ineq) => {
                self.explain_ineq_conflict(ineq, &mut explanation);
            }
            Atom::Root(root) => {
                // For root atoms, explain based on the polynomial's roots
                self.add_to_projection(&root.poly);
                self.explain_root_conflict(root.var, &mut explanation);
            }
        }

        // Minimize explanation if configured
        if self.config.minimize {
            self.minimize_explanation(&mut explanation);
        }

        Explanation::new(explanation, ExplanationKind::Theory)
    }

    /// Explain an inequality atom conflict.
    fn explain_ineq_conflict(&mut self, ineq: &IneqAtom, explanation: &mut Vec<Literal>) {
        // For each factor in the inequality, find assignments that constrain its sign
        for factor in &ineq.factors {
            self.add_to_projection(&factor.poly);

            // Find all atoms that share variables with this polynomial
            for var in factor.poly.vars() {
                self.explain_var_constraint(var, explanation);
            }
        }
    }

    /// Explain constraints on a variable.
    fn explain_var_constraint(&mut self, var: Var, explanation: &mut Vec<Literal>) {
        // Find all atoms that constrain this variable
        for atom in self.atoms.iter() {
            let bool_var = atom.bool_var();
            if bool_var == NULL_BOOL_VAR {
                continue;
            }

            // Check if this atom involves the variable
            let involves_var = match atom {
                Atom::Ineq(ineq) => ineq.factors.iter().any(|f| f.poly.vars().contains(&var)),
                Atom::Root(root) => root.var == var || root.poly.vars().contains(&var),
            };

            if !involves_var {
                continue;
            }

            // Check if this atom is assigned
            let value = self
                .bool_values
                .get(bool_var as usize)
                .copied()
                .unwrap_or(Lbool::Undef);
            if value.is_undef() {
                continue;
            }

            // Add the negation of the assigned value to explanation
            let lit = if value.is_true() {
                Literal::negative(bool_var)
            } else {
                Literal::positive(bool_var)
            };

            if !self.seen.contains(&bool_var) {
                self.seen.insert(bool_var);
                explanation.push(lit);
            }
        }
    }

    /// Explain a root atom conflict.
    fn explain_root_conflict(&mut self, var: Var, explanation: &mut Vec<Literal>) {
        // Find all atoms that constrain this variable
        self.explain_var_constraint(var, explanation);
    }

    /// Add a polynomial to the projection set.
    fn add_to_projection(&mut self, poly: &Polynomial) {
        // Check if already in projection set
        for p in &self.projection_set {
            if p == poly {
                return;
            }
        }
        self.projection_set.push(poly.clone());
    }

    /// Minimize an explanation by removing redundant literals.
    fn minimize_explanation(&self, explanation: &mut Vec<Literal>) {
        // Simple minimization: remove duplicates
        explanation.sort_by_key(|l| l.index());
        explanation.dedup();

        // Sophisticated minimization: recursive redundancy elimination
        // Try to remove each literal and check if it's truly necessary
        let mut i = 0;
        while i < explanation.len() {
            let lit = explanation[i];

            // Try removing this literal
            explanation.remove(i);

            // Check if the explanation is still valid without this literal
            // A literal is redundant if its corresponding atom's constraint
            // is already implied by the other literals in the explanation
            if self.is_literal_redundant(lit, explanation) {
                // Keep it removed (redundant)
                continue;
            } else {
                // Not redundant, put it back
                explanation.insert(i, lit);
                i += 1;
            }
        }
    }

    /// Check if a literal is redundant given the current explanation.
    /// A literal is redundant if the constraint it represents is already
    /// implied by the other literals in the explanation.
    fn is_literal_redundant(&self, lit: Literal, explanation: &[Literal]) -> bool {
        // Get the atom corresponding to this literal
        let var = lit.var();
        let is_pos = lit.is_positive();

        // Check if this variable appears elsewhere in the explanation
        // If it does with the same polarity, it's redundant
        for &other_lit in explanation {
            if other_lit.var() == var && other_lit.is_positive() == is_pos {
                return true;
            }
        }

        // For theory atoms, check if the constraint is subsumed
        // by other constraints in the explanation
        if let Some(atom) = self.atoms.get(var as usize) {
            match atom {
                Atom::Ineq(ineq) => {
                    // Check if this inequality is subsumed by others
                    self.is_ineq_subsumed(ineq, is_pos, explanation)
                }
                Atom::Root(_) => {
                    // Root atoms are typically not redundant
                    false
                }
            }
        } else {
            // Unknown atom, assume not redundant
            false
        }
    }

    /// Check if an inequality atom is subsumed by other atoms in the explanation.
    fn is_ineq_subsumed(
        &self,
        ineq: &IneqAtom,
        _is_positive: bool,
        explanation: &[Literal],
    ) -> bool {
        // Check if any other literal in the explanation involves the same factors
        for &lit in explanation {
            if let Some(Atom::Ineq(other_ineq)) = self.atoms.get(lit.var() as usize) {
                // If the factors are identical, check if the constraints subsume
                if ineq.factors.len() == other_ineq.factors.len() {
                    let mut all_match = true;
                    for (f1, f2) in ineq.factors.iter().zip(other_ineq.factors.iter()) {
                        if !polynomials_equal(&f1.poly, &f2.poly) || f1.is_even != f2.is_even {
                            all_match = false;
                            break;
                        }
                    }

                    if all_match && ineq.kind == other_ineq.kind {
                        // Same factors and same kind - will be caught by duplicate check
                        return false;
                    }
                }
            }
        }

        false
    }

    /// Compute the projection of the projection set onto a variable.
    ///
    /// This removes the variable from consideration, adding resultants and
    /// discriminants to handle the variable's possible values.
    pub fn project(&mut self, var: Var) -> Vec<Polynomial> {
        if !self.config.use_projection {
            return Vec::new();
        }

        let mut result = Vec::new();

        // Collect polynomials involving this variable
        let relevant: Vec<&Polynomial> = self
            .projection_set
            .iter()
            .filter(|p| p.vars().contains(&var))
            .collect();

        // For each polynomial, compute its derivative (discriminant involves this)
        for poly in &relevant {
            // Derivative with respect to var
            let derivative = poly.derivative(var);
            if !derivative.is_zero() {
                result.push(derivative);
            }

            // Leading coefficient (when viewing as polynomial in var)
            if let Some(lc) = leading_coefficient(poly, var)
                && !lc.is_constant()
            {
                result.push(lc);
            }
        }

        // For each pair of polynomials, compute resultant
        for i in 0..relevant.len() {
            for j in (i + 1)..relevant.len() {
                if let Some(res) = resultant(relevant[i], relevant[j], var)
                    && !res.is_zero()
                    && !res.is_constant()
                {
                    result.push(res);
                }
            }
        }

        result
    }

    /// Compute the sign of a polynomial at the current assignment.
    pub fn poly_sign(&mut self, poly: &Polynomial) -> Option<i8> {
        // Check cache first (using polynomial identity, not hash)
        // For now, compute directly

        // Build assignment map
        let mut assignment = rustc_hash::FxHashMap::default();
        for (i, val) in self.arith_values.iter().enumerate() {
            if let Some(v) = val {
                assignment.insert(i as Var, v.clone());
            }
        }

        // Check if all variables are assigned
        for var in poly.vars() {
            if !assignment.contains_key(&var) {
                return None;
            }
        }

        // Evaluate
        let value = poly.eval(&assignment);
        Some(if value.is_zero() {
            0
        } else if value.is_positive() {
            1
        } else {
            -1
        })
    }
}

/// Check if two polynomials are equal.
fn polynomials_equal(p1: &Polynomial, p2: &Polynomial) -> bool {
    let t1 = p1.terms();
    let t2 = p2.terms();

    if t1.len() != t2.len() {
        return false;
    }

    for (term1, term2) in t1.iter().zip(t2.iter()) {
        if term1.monomial != term2.monomial || term1.coeff != term2.coeff {
            return false;
        }
    }

    true
}

/// Compute the leading coefficient of a polynomial with respect to a variable.
fn leading_coefficient(poly: &Polynomial, var: Var) -> Option<Polynomial> {
    let degree = poly.degree(var);
    if degree == 0 {
        return None;
    }

    // Extract the leading coefficient by setting var = 0 in p / var^degree
    // This is a simplified version - proper implementation would extract coefficient
    Some(poly.clone())
}

/// Compute the resultant of two polynomials with respect to a variable.
///
/// The resultant is a polynomial in the remaining variables that is zero
/// iff the two polynomials have a common root in the given variable.
fn resultant(p: &Polynomial, q: &Polynomial, var: Var) -> Option<Polynomial> {
    let deg_p = p.degree(var);
    let deg_q = q.degree(var);

    if deg_p == 0 || deg_q == 0 {
        return None;
    }

    // For linear polynomials, resultant is straightforward
    if deg_p == 1 && deg_q == 1 {
        // p = a*x + b, q = c*x + d
        // resultant = a*d - b*c
        // This is a placeholder - proper implementation would extract coefficients
        return Some(Polynomial::zero());
    }

    // For higher degrees, use Sylvester matrix determinant
    // This is a placeholder - proper implementation would compute the full resultant
    Some(Polynomial::zero())
}

/// Implication graph for conflict analysis.
pub struct ImplicationGraph {
    /// Nodes: (literal, level, reason clause ID)
    nodes: Vec<(Literal, u32, Option<u32>)>,
    /// Edges: (from node, to node)
    edges: Vec<(usize, usize)>,
}

impl ImplicationGraph {
    /// Create a new implication graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, lit: Literal, level: u32, reason: Option<u32>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push((lit, level, reason));
        idx
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }

    /// Find all literals at a given level.
    pub fn literals_at_level(&self, level: u32) -> Vec<Literal> {
        self.nodes
            .iter()
            .filter(|(_, l, _)| *l == level)
            .map(|(lit, _, _)| *lit)
            .collect()
    }

    /// Find the first UIP (Unique Implication Point) at a level.
    ///
    /// The first UIP is the node closest to the conflict that dominates
    /// all paths from the decision to the conflict.
    pub fn find_first_uip(&self, level: u32) -> Option<Literal> {
        // Find all nodes at this level
        let level_nodes: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, (_, l, _))| *l == level)
            .collect();

        if level_nodes.is_empty() {
            return None;
        }

        // The first UIP is typically the last decision or propagation at this level
        // that all conflict paths pass through
        level_nodes.last().map(|(_, (lit, _, _))| *lit)
    }
}

impl Default for ImplicationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Sign condition for a polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignCondition {
    /// Polynomial is zero.
    Zero,
    /// Polynomial is positive.
    Positive,
    /// Polynomial is negative.
    Negative,
    /// Polynomial is non-negative (>= 0).
    NonNegative,
    /// Polynomial is non-positive (<= 0).
    NonPositive,
    /// Polynomial is non-zero (!= 0).
    NonZero,
}

impl SignCondition {
    /// Check if a sign satisfies this condition.
    pub fn satisfied_by(&self, sign: i8) -> bool {
        match self {
            SignCondition::Zero => sign == 0,
            SignCondition::Positive => sign > 0,
            SignCondition::Negative => sign < 0,
            SignCondition::NonNegative => sign >= 0,
            SignCondition::NonPositive => sign <= 0,
            SignCondition::NonZero => sign != 0,
        }
    }

    /// Get the condition from an atom kind and polarity.
    pub fn from_atom_kind(kind: AtomKind, positive: bool) -> Self {
        match (kind, positive) {
            (AtomKind::Eq, true) => SignCondition::Zero,
            (AtomKind::Eq, false) => SignCondition::NonZero,
            (AtomKind::Lt, true) => SignCondition::Negative,
            (AtomKind::Lt, false) => SignCondition::NonNegative,
            (AtomKind::Gt, true) => SignCondition::Positive,
            (AtomKind::Gt, false) => SignCondition::NonPositive,
            _ => SignCondition::Zero, // Default for root atoms
        }
    }
}

/// A cell in the CAD (Cylindrical Algebraic Decomposition).
///
/// A cell is a connected region of R^n where all polynomials in the
/// projection set have constant sign.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Sample point in the cell.
    pub sample: Vec<BigRational>,
    /// Signs of polynomials in this cell.
    pub signs: Vec<i8>,
    /// Lower bounds for each variable (None = -infinity).
    pub lower: Vec<Option<BigRational>>,
    /// Upper bounds for each variable (None = +infinity).
    pub upper: Vec<Option<BigRational>>,
}

impl Cell {
    /// Create a new cell with the given sample point.
    pub fn new(sample: Vec<BigRational>) -> Self {
        let n = sample.len();
        Self {
            sample,
            signs: Vec::new(),
            lower: vec![None; n],
            upper: vec![None; n],
        }
    }

    /// Check if a point is in this cell.
    pub fn contains(&self, point: &[BigRational]) -> bool {
        if point.len() != self.sample.len() {
            return false;
        }

        for (i, p) in point.iter().enumerate() {
            if let Some(lo) = &self.lower[i]
                && p < lo
            {
                return false;
            }
            if let Some(hi) = &self.upper[i]
                && p > hi
            {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explanation_empty() {
        let exp = Explanation::empty();
        assert!(exp.is_empty());
        assert_eq!(exp.kind, ExplanationKind::Empty);
    }

    #[test]
    fn test_explanation_new() {
        let lits = vec![Literal::positive(0), Literal::negative(1)];
        let exp = Explanation::new(lits.clone(), ExplanationKind::Resolution);
        assert_eq!(exp.as_clause(), &lits);
        assert_eq!(exp.kind, ExplanationKind::Resolution);
    }

    #[test]
    fn test_sign_condition() {
        assert!(SignCondition::Zero.satisfied_by(0));
        assert!(!SignCondition::Zero.satisfied_by(1));
        assert!(!SignCondition::Zero.satisfied_by(-1));

        assert!(SignCondition::Positive.satisfied_by(1));
        assert!(!SignCondition::Positive.satisfied_by(0));
        assert!(!SignCondition::Positive.satisfied_by(-1));

        assert!(SignCondition::Negative.satisfied_by(-1));
        assert!(!SignCondition::Negative.satisfied_by(0));
        assert!(!SignCondition::Negative.satisfied_by(1));

        assert!(SignCondition::NonNegative.satisfied_by(0));
        assert!(SignCondition::NonNegative.satisfied_by(1));
        assert!(!SignCondition::NonNegative.satisfied_by(-1));

        assert!(SignCondition::NonPositive.satisfied_by(0));
        assert!(SignCondition::NonPositive.satisfied_by(-1));
        assert!(!SignCondition::NonPositive.satisfied_by(1));

        assert!(SignCondition::NonZero.satisfied_by(1));
        assert!(SignCondition::NonZero.satisfied_by(-1));
        assert!(!SignCondition::NonZero.satisfied_by(0));
    }

    #[test]
    fn test_implication_graph() {
        let mut graph = ImplicationGraph::new();

        let n0 = graph.add_node(Literal::positive(0), 1, None);
        let n1 = graph.add_node(Literal::positive(1), 1, Some(0));
        let n2 = graph.add_node(Literal::positive(2), 2, Some(1));

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);

        let lits_level1 = graph.literals_at_level(1);
        assert_eq!(lits_level1.len(), 2);

        let lits_level2 = graph.literals_at_level(2);
        assert_eq!(lits_level2.len(), 1);

        let uip = graph.find_first_uip(1);
        assert!(uip.is_some());
    }

    #[test]
    fn test_cell() {
        use num_bigint::BigInt;

        let sample = vec![
            BigRational::from_integer(BigInt::from(1)),
            BigRational::from_integer(BigInt::from(2)),
        ];
        let mut cell = Cell::new(sample.clone());

        assert!(cell.contains(&sample));

        // Set bounds
        cell.lower[0] = Some(BigRational::from_integer(BigInt::from(0)));
        cell.upper[0] = Some(BigRational::from_integer(BigInt::from(2)));

        assert!(cell.contains(&sample));

        let outside = vec![
            BigRational::from_integer(BigInt::from(-1)),
            BigRational::from_integer(BigInt::from(2)),
        ];
        assert!(!cell.contains(&outside));
    }
}
