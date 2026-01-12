//! Extended resolution with extension variables
//!
//! This module implements extended resolution, which allows introducing
//! new variables (extension variables) and definitions during solving
//! to simplify formulas and enable more powerful reasoning.

use crate::literal::{Lit, Var};
use std::collections::HashMap;

/// Types of extension variable definitions
#[derive(Debug, Clone)]
pub enum ExtensionType {
    /// AND: z ↔ (x ∧ y)
    And(Lit, Lit),
    /// OR: z ↔ (x ∨ y)
    Or(Lit, Lit),
    /// XOR: z ↔ (x ⊕ y)
    Xor(Lit, Lit),
    /// ITE: z ↔ (if c then t else e)
    Ite {
        /// Condition literal
        cond: Lit,
        /// Then branch literal
        then_lit: Lit,
        /// Else branch literal
        else_lit: Lit,
    },
    /// Equivalence: z ↔ (x ↔ y)
    Equiv(Lit, Lit),
    /// General definition from conjunction/disjunction
    General {
        /// Positive literals in the definition
        positive: Vec<Lit>,
        /// Negative literals in the definition
        negative: Vec<Lit>,
    },
}

/// Extension variable definition
#[derive(Debug, Clone)]
pub struct Extension {
    /// The extension variable
    pub var: Var,
    /// The definition type
    pub def: ExtensionType,
}

impl Extension {
    /// Create a new extension variable definition
    pub fn new(var: Var, def: ExtensionType) -> Self {
        Self { var, def }
    }

    /// Get the CNF clauses representing this extension
    pub fn to_cnf(&self) -> Vec<Vec<Lit>> {
        let z = Lit::pos(self.var);
        let nz = Lit::neg(self.var);

        match &self.def {
            ExtensionType::And(x, y) => {
                // z → (x ∧ y): ¬z ∨ x, ¬z ∨ y
                // (x ∧ y) → z: ¬x ∨ ¬y ∨ z
                vec![vec![nz, *x], vec![nz, *y], vec![x.negate(), y.negate(), z]]
            }
            ExtensionType::Or(x, y) => {
                // z → (x ∨ y): ¬z ∨ x ∨ y
                // (x ∨ y) → z: (¬x ∨ z) ∧ (¬y ∨ z)
                vec![vec![nz, *x, *y], vec![x.negate(), z], vec![y.negate(), z]]
            }
            ExtensionType::Xor(x, y) => {
                // z ↔ (x ⊕ y)
                // z → (x ⊕ y): (¬z ∨ ¬x ∨ ¬y) ∧ (¬z ∨ x ∨ y)
                // (x ⊕ y) → z: (x ∨ ¬y ∨ z) ∧ (¬x ∨ y ∨ z)
                vec![
                    vec![nz, x.negate(), y.negate()],
                    vec![nz, *x, *y],
                    vec![*x, y.negate(), z],
                    vec![x.negate(), *y, z],
                ]
            }
            ExtensionType::Ite {
                cond,
                then_lit,
                else_lit,
            } => {
                // z ↔ (c ? t : e) ≡ z ↔ ((c ∧ t) ∨ (¬c ∧ e))
                // (c ∧ t) → z: ¬c ∨ ¬t ∨ z
                // (¬c ∧ e) → z: c ∨ ¬e ∨ z
                // z ∧ c → t: ¬z ∨ ¬c ∨ t
                // z ∧ ¬c → e: ¬z ∨ c ∨ e
                vec![
                    vec![cond.negate(), then_lit.negate(), z],
                    vec![*cond, else_lit.negate(), z],
                    vec![nz, cond.negate(), *then_lit],
                    vec![nz, *cond, *else_lit],
                ]
            }
            ExtensionType::Equiv(x, y) => {
                // z ↔ (x ↔ y)
                // Same as XOR negation
                // z → (x ↔ y): (¬z ∨ ¬x ∨ y) ∧ (¬z ∨ x ∨ ¬y)
                // (x ↔ y) → z: (x ∨ y ∨ z) ∧ (¬x ∨ ¬y ∨ z)
                vec![
                    vec![nz, x.negate(), *y],
                    vec![nz, *x, y.negate()],
                    vec![*x, *y, z],
                    vec![x.negate(), y.negate(), z],
                ]
            }
            ExtensionType::General { positive, negative } => {
                // z → (p1 ∨ ... ∨ pk ∨ ¬n1 ∨ ... ∨ ¬nm)
                // Reverse implications
                let mut clauses = Vec::new();

                // Forward implication: z → definition
                let mut forward = vec![nz];
                forward.extend(positive.iter().copied());
                for &lit in negative {
                    forward.push(lit.negate());
                }
                clauses.push(forward);

                // Backward implications: each literal → z
                for &lit in positive {
                    clauses.push(vec![lit.negate(), z]);
                }
                for &lit in negative {
                    clauses.push(vec![lit, z]);
                }

                clauses
            }
        }
    }
}

/// Extended resolution manager
pub struct ExtendedResolution {
    /// Extension variable definitions
    extensions: HashMap<Var, Extension>,
    /// Next variable ID for extension variables
    next_var: u32,
    /// Base number of original variables
    base_num_vars: u32,
}

impl ExtendedResolution {
    /// Create a new extended resolution manager
    pub fn new(num_vars: u32) -> Self {
        Self {
            extensions: HashMap::new(),
            next_var: num_vars,
            base_num_vars: num_vars,
        }
    }

    /// Add an extension variable
    pub fn add_extension(&mut self, def: ExtensionType) -> Var {
        let var = Var(self.next_var);
        self.next_var += 1;
        self.extensions.insert(var, Extension::new(var, def));
        var
    }

    /// Add an AND extension: z ↔ (x ∧ y)
    pub fn add_and(&mut self, x: Lit, y: Lit) -> Var {
        self.add_extension(ExtensionType::And(x, y))
    }

    /// Add an OR extension: z ↔ (x ∨ y)
    pub fn add_or(&mut self, x: Lit, y: Lit) -> Var {
        self.add_extension(ExtensionType::Or(x, y))
    }

    /// Add an XOR extension: z ↔ (x ⊕ y)
    pub fn add_xor(&mut self, x: Lit, y: Lit) -> Var {
        self.add_extension(ExtensionType::Xor(x, y))
    }

    /// Add an ITE extension: z ↔ (if c then t else e)
    pub fn add_ite(&mut self, cond: Lit, then_lit: Lit, else_lit: Lit) -> Var {
        self.add_extension(ExtensionType::Ite {
            cond,
            then_lit,
            else_lit,
        })
    }

    /// Add an EQUIV extension: z ↔ (x ↔ y)
    pub fn add_equiv(&mut self, x: Lit, y: Lit) -> Var {
        self.add_extension(ExtensionType::Equiv(x, y))
    }

    /// Get all CNF clauses for all extensions
    pub fn get_all_cnf(&self) -> Vec<Vec<Lit>> {
        let mut clauses = Vec::new();
        for ext in self.extensions.values() {
            clauses.extend(ext.to_cnf());
        }
        clauses
    }

    /// Get extension definition for a variable
    pub fn get_extension(&self, var: Var) -> Option<&Extension> {
        self.extensions.get(&var)
    }

    /// Check if a variable is an extension variable
    pub fn is_extension(&self, var: Var) -> bool {
        var.0 >= self.base_num_vars
    }

    /// Get the current number of variables (including extensions)
    pub fn num_vars(&self) -> u32 {
        self.next_var
    }

    /// Get the number of extension variables
    pub fn num_extensions(&self) -> usize {
        self.extensions.len()
    }

    /// Get all extension variables
    pub fn get_extensions(&self) -> Vec<Var> {
        let mut vars: Vec<Var> = self.extensions.keys().copied().collect();
        vars.sort_by_key(|v| v.0);
        vars
    }

    /// Tseitin transformation for a formula
    /// Returns the top-level variable representing the formula
    pub fn tseitin_and(&mut self, lits: &[Lit]) -> Var {
        if lits.is_empty() {
            // Empty AND is true, but we need a variable for it
            // Create a unit clause [z] to force it true
            return self.add_extension(ExtensionType::General {
                positive: vec![],
                negative: vec![],
            });
        }
        if lits.len() == 1 {
            return lits[0].var();
        }

        // Build a balanced tree of AND gates
        let mid = lits.len() / 2;
        let left = self.tseitin_and(&lits[..mid]);
        let right = self.tseitin_and(&lits[mid..]);
        self.add_and(Lit::pos(left), Lit::pos(right))
    }

    /// Tseitin transformation for OR
    pub fn tseitin_or(&mut self, lits: &[Lit]) -> Var {
        if lits.is_empty() {
            // Empty OR is false
            return self.add_extension(ExtensionType::General {
                positive: vec![],
                negative: vec![],
            });
        }
        if lits.len() == 1 {
            return lits[0].var();
        }

        // Build a balanced tree of OR gates
        let mid = lits.len() / 2;
        let left = self.tseitin_or(&lits[..mid]);
        let right = self.tseitin_or(&lits[mid..]);
        self.add_or(Lit::pos(left), Lit::pos(right))
    }
}

/// Clause substitution using extension variables
pub struct ClauseSubstitution {
    /// Map from literal pairs to extension variables
    substitutions: HashMap<(Lit, Lit), Var>,
}

impl ClauseSubstitution {
    /// Create a new clause substitution helper
    pub fn new() -> Self {
        Self {
            substitutions: HashMap::new(),
        }
    }

    /// Record a substitution
    pub fn add(&mut self, x: Lit, y: Lit, z: Var) {
        self.substitutions.insert((x, y), z);
        self.substitutions.insert((y, x), z);
    }

    /// Get substitution for a literal pair
    pub fn get(&self, x: Lit, y: Lit) -> Option<Var> {
        self.substitutions.get(&(x, y)).copied()
    }
}

impl Default for ClauseSubstitution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_extension() {
        let x = Lit::pos(Var(0));
        let y = Lit::pos(Var(1));
        let z = Var(2);

        let ext = Extension::new(z, ExtensionType::And(x, y));
        let cnf = ext.to_cnf();

        assert_eq!(cnf.len(), 3);
    }

    #[test]
    fn test_or_extension() {
        let x = Lit::pos(Var(0));
        let y = Lit::pos(Var(1));
        let z = Var(2);

        let ext = Extension::new(z, ExtensionType::Or(x, y));
        let cnf = ext.to_cnf();

        assert_eq!(cnf.len(), 3);
    }

    #[test]
    fn test_xor_extension() {
        let x = Lit::pos(Var(0));
        let y = Lit::pos(Var(1));
        let z = Var(2);

        let ext = Extension::new(z, ExtensionType::Xor(x, y));
        let cnf = ext.to_cnf();

        assert_eq!(cnf.len(), 4);
    }

    #[test]
    fn test_ite_extension() {
        let c = Lit::pos(Var(0));
        let t = Lit::pos(Var(1));
        let e = Lit::pos(Var(2));
        let z = Var(3);

        let ext = Extension::new(
            z,
            ExtensionType::Ite {
                cond: c,
                then_lit: t,
                else_lit: e,
            },
        );
        let cnf = ext.to_cnf();

        assert_eq!(cnf.len(), 4);
    }

    #[test]
    fn test_extended_resolution_manager() {
        let mut er = ExtendedResolution::new(10);

        let x = Lit::pos(Var(0));
        let y = Lit::pos(Var(1));

        let z = er.add_and(x, y);
        assert!(er.is_extension(z));
        assert_eq!(er.num_extensions(), 1);

        let w = er.add_or(x, y);
        assert!(er.is_extension(w));
        assert_eq!(er.num_extensions(), 2);
    }

    #[test]
    fn test_tseitin_and() {
        let mut er = ExtendedResolution::new(10);

        let lits = vec![
            Lit::pos(Var(0)),
            Lit::pos(Var(1)),
            Lit::pos(Var(2)),
            Lit::pos(Var(3)),
        ];

        let top = er.tseitin_and(&lits);
        assert!(er.is_extension(top));

        // Should create a tree structure
        assert!(er.num_extensions() >= 1);
    }

    #[test]
    fn test_clause_substitution() {
        let mut subst = ClauseSubstitution::new();

        let x = Lit::pos(Var(0));
        let y = Lit::pos(Var(1));
        let z = Var(2);

        subst.add(x, y, z);

        assert_eq!(subst.get(x, y), Some(z));
        assert_eq!(subst.get(y, x), Some(z));
    }
}
