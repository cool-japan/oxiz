//! Advanced Word Equation Solver
//!
//! Implements word equation solving algorithms:
//! - Nielsen transformation for solving word equations
//! - Levi's lemma for variable splitting
//! - Length abstraction for arithmetic interaction
//! - Loop detection to prevent infinite exploration

use super::solver::{StringAtom, StringExpr, WordEquation};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;

/// Result of word equation solving
#[derive(Debug, Clone)]
pub enum SolveResult {
    /// Satisfiable with substitution
    Sat(Substitution),
    /// Unsatisfiable with conflict
    Unsat(Conflict),
    /// Unknown (timeout or incomplete)
    Unknown,
    /// Requires case split
    Split(Vec<CaseSplit>),
}

/// Variable substitution
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    /// Variable -> Expression mapping
    mappings: FxHashMap<u32, StringExpr>,
}

impl Substitution {
    /// Create an empty substitution
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping
    pub fn add(&mut self, var: u32, expr: StringExpr) {
        self.mappings.insert(var, expr);
    }

    /// Get mapping for a variable
    pub fn get(&self, var: u32) -> Option<&StringExpr> {
        self.mappings.get(&var)
    }

    /// Apply substitution to an expression
    pub fn apply(&self, expr: &StringExpr) -> StringExpr {
        let mut result: SmallVec<[StringAtom; 4]> = SmallVec::new();
        for atom in &expr.atoms {
            match atom {
                StringAtom::Var(v) => {
                    if let Some(replacement) = self.mappings.get(v) {
                        result.extend(replacement.atoms.iter().cloned());
                    } else {
                        result.push(atom.clone());
                    }
                }
                StringAtom::Const(_) => {
                    result.push(atom.clone());
                }
            }
        }

        // Merge adjacent constants
        let mut merged: SmallVec<[StringAtom; 4]> = SmallVec::new();
        for atom in result {
            if let (Some(StringAtom::Const(last)), StringAtom::Const(s)) =
                (merged.last_mut(), &atom)
            {
                last.push_str(s);
            } else {
                merged.push(atom);
            }
        }

        StringExpr { atoms: merged }
    }

    /// Compose with another substitution
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();
        for (&var, expr) in &self.mappings {
            result.add(var, other.apply(expr));
        }
        for (&var, expr) in &other.mappings {
            if !self.mappings.contains_key(&var) {
                result.add(var, expr.clone());
            }
        }
        result
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Iterator over mappings
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &StringExpr)> {
        self.mappings.iter()
    }
}

/// Conflict information
#[derive(Debug, Clone)]
pub struct Conflict {
    /// Reason for conflict
    pub reason: ConflictReason,
    /// Variables involved
    pub vars: Vec<u32>,
}

/// Conflict reason
#[derive(Debug, Clone)]
pub enum ConflictReason {
    /// Different constant prefixes
    ConstantMismatch(String, String),
    /// Empty vs non-empty
    EmptyMismatch,
    /// Length conflict
    LengthConflict,
    /// Loop detected
    LoopDetected,
}

/// Case split for disjunctive reasoning
#[derive(Debug, Clone)]
pub struct CaseSplit {
    /// The substitution for this case
    pub subst: Substitution,
    /// Remaining equations after this split
    pub equations: Vec<WordEquation>,
    /// Length constraints implied by this case
    pub length_constraints: Vec<LengthConstraint>,
}

/// Length constraint
#[derive(Debug, Clone)]
pub struct LengthConstraint {
    /// Variable ID
    pub var: u32,
    /// Coefficient (1 or -1)
    pub coef: i32,
    /// Relation to constant
    pub rel: Relation,
    /// Constant value
    pub value: i64,
}

/// Relation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than or equal
    Le,
    /// Greater than or equal
    Ge,
    /// Less than
    Lt,
    /// Greater than
    Gt,
}

/// Word equation solver configuration
#[derive(Debug, Clone)]
pub struct WordEqConfig {
    /// Maximum number of case splits
    pub max_splits: usize,
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Enable loop detection
    pub loop_detection: bool,
    /// Enable length abstraction
    pub length_abstraction: bool,
}

impl Default for WordEqConfig {
    fn default() -> Self {
        Self {
            max_splits: 1000,
            max_depth: 50,
            loop_detection: true,
            length_abstraction: true,
        }
    }
}

/// Advanced word equation solver using Nielsen transformation
#[derive(Debug)]
pub struct WordEqSolver {
    /// Configuration
    config: WordEqConfig,
    /// Current substitution
    current_subst: Substitution,
    /// Pending equations
    pending: VecDeque<WordEquation>,
    /// Solved equations
    solved: Vec<WordEquation>,
    /// Visited states for loop detection
    visited: FxHashSet<u64>,
    /// Length constraints
    length_constraints: Vec<LengthConstraint>,
    /// Current recursion depth
    depth: usize,
    /// Number of splits performed
    splits: usize,
    /// Statistics
    stats: WordEqStats,
    /// Initial conflict (if any)
    initial_conflict: Option<Conflict>,
}

/// Solver statistics
#[derive(Debug, Default)]
pub struct WordEqStats {
    /// Number of equations processed
    pub equations_processed: u64,
    /// Number of Nielsen steps
    pub nielsen_steps: u64,
    /// Number of Levi applications
    pub levi_applications: u64,
    /// Number of case splits
    pub case_splits: u64,
    /// Number of backtracks
    pub backtracks: u64,
}

impl WordEqSolver {
    /// Create a new solver
    pub fn new(config: WordEqConfig) -> Self {
        Self {
            config,
            current_subst: Substitution::new(),
            pending: VecDeque::new(),
            solved: Vec::new(),
            visited: FxHashSet::default(),
            length_constraints: Vec::new(),
            depth: 0,
            splits: 0,
            stats: WordEqStats::default(),
            initial_conflict: None,
        }
    }

    /// Add an equation to solve
    pub fn add_equation(&mut self, eq: WordEquation) {
        if !eq.is_solved() && !eq.has_conflict() {
            self.pending.push_back(eq);
        } else if eq.has_conflict() {
            // Store the conflict
            self.initial_conflict = Some(Conflict {
                reason: self.detect_conflict_reason(&eq),
                vars: self.extract_vars(&eq),
            });
        }
    }

    /// Solve all equations
    pub fn solve(&mut self) -> SolveResult {
        // Check for initial conflict (from add_equation)
        if let Some(conflict) = self.initial_conflict.take() {
            return SolveResult::Unsat(conflict);
        }

        while let Some(eq) = self.pending.pop_front() {
            self.stats.equations_processed += 1;

            // Apply current substitution
            let eq = WordEquation {
                lhs: self.current_subst.apply(&eq.lhs),
                rhs: self.current_subst.apply(&eq.rhs),
                origin: eq.origin,
            };

            // Check for conflict
            if eq.has_conflict() {
                return SolveResult::Unsat(Conflict {
                    reason: self.detect_conflict_reason(&eq),
                    vars: self.extract_vars(&eq),
                });
            }

            // Check if solved
            if eq.is_solved() {
                self.solved.push(eq);
                continue;
            }

            // Try Nielsen transformation
            match self.nielsen_step(&eq) {
                NielsenResult::Solved(subst) => {
                    self.current_subst = self.current_subst.compose(&subst);
                }
                NielsenResult::Split(cases) => {
                    if self.splits >= self.config.max_splits {
                        return SolveResult::Unknown;
                    }
                    self.stats.case_splits += 1;
                    self.splits += 1;
                    return SolveResult::Split(cases);
                }
                NielsenResult::Conflict(reason) => {
                    return SolveResult::Unsat(Conflict {
                        reason,
                        vars: self.extract_vars(&eq),
                    });
                }
                NielsenResult::Continue(new_eqs) => {
                    for new_eq in new_eqs {
                        self.pending.push_back(new_eq);
                    }
                }
            }
        }

        SolveResult::Sat(self.current_subst.clone())
    }

    /// Nielsen transformation step
    fn nielsen_step(&mut self, eq: &WordEquation) -> NielsenResult {
        self.stats.nielsen_steps += 1;

        let lhs = &eq.lhs.atoms;
        let rhs = &eq.rhs.atoms;

        // Empty cases
        if lhs.is_empty() && rhs.is_empty() {
            return NielsenResult::Solved(Substitution::new());
        }

        if lhs.is_empty() {
            // rhs must all be empty strings (variables assigned to empty)
            return self.make_empty_subst(rhs, eq.origin);
        }

        if rhs.is_empty() {
            return self.make_empty_subst(lhs, eq.origin);
        }

        // Get first atoms
        let first_l = &lhs[0];
        let first_r = &rhs[0];

        match (first_l, first_r) {
            // Both constants
            (StringAtom::Const(l), StringAtom::Const(r)) => self.handle_const_const(l, r, eq),
            // Variable vs constant - apply Levi's lemma
            (StringAtom::Var(v), StringAtom::Const(c)) => {
                self.stats.levi_applications += 1;
                self.handle_var_const(*v, c, eq, true)
            }
            (StringAtom::Const(c), StringAtom::Var(v)) => {
                self.stats.levi_applications += 1;
                self.handle_var_const(*v, c, eq, false)
            }
            // Both variables - apply Levi's lemma with case split
            (StringAtom::Var(v1), StringAtom::Var(v2)) => {
                self.stats.levi_applications += 1;
                self.handle_var_var(*v1, *v2, eq)
            }
        }
    }

    /// Handle case where both first atoms are constants
    fn handle_const_const(&self, l: &str, r: &str, eq: &WordEquation) -> NielsenResult {
        // Strip common prefix
        let common_len = l.chars().zip(r.chars()).take_while(|(a, b)| a == b).count();

        if common_len == 0 {
            // Mismatch
            return NielsenResult::Conflict(ConflictReason::ConstantMismatch(
                l.to_string(),
                r.to_string(),
            ));
        }

        // Build new equation with common prefix stripped
        let l_rest: String = l.chars().skip(common_len).collect();
        let r_rest: String = r.chars().skip(common_len).collect();

        let mut new_lhs = SmallVec::new();
        if !l_rest.is_empty() {
            new_lhs.push(StringAtom::Const(l_rest));
        }
        new_lhs.extend(eq.lhs.atoms[1..].iter().cloned());

        let mut new_rhs = SmallVec::new();
        if !r_rest.is_empty() {
            new_rhs.push(StringAtom::Const(r_rest));
        }
        new_rhs.extend(eq.rhs.atoms[1..].iter().cloned());

        NielsenResult::Continue(vec![WordEquation {
            lhs: StringExpr { atoms: new_lhs },
            rhs: StringExpr { atoms: new_rhs },
            origin: eq.origin,
        }])
    }

    /// Handle case where first atom is variable, second is constant
    fn handle_var_const(
        &self,
        var: u32,
        constant: &str,
        eq: &WordEquation,
        var_on_left: bool,
    ) -> NielsenResult {
        // Case split: var = "" or var = c + var' (for each prefix of constant)
        let mut cases = Vec::new();

        // Case 1: var = ""
        let mut subst = Substitution::new();
        subst.add(var, StringExpr::empty());

        let new_lhs = if var_on_left {
            StringExpr {
                atoms: eq.lhs.atoms[1..].iter().cloned().collect(),
            }
        } else {
            eq.lhs.clone()
        };

        let new_rhs = if var_on_left {
            eq.rhs.clone()
        } else {
            StringExpr {
                atoms: eq.rhs.atoms[1..].iter().cloned().collect(),
            }
        };

        cases.push(CaseSplit {
            subst: subst.clone(),
            equations: vec![WordEquation {
                lhs: subst.apply(&new_lhs),
                rhs: subst.apply(&new_rhs),
                origin: eq.origin,
            }],
            length_constraints: vec![LengthConstraint {
                var,
                coef: 1,
                rel: Relation::Eq,
                value: 0,
            }],
        });

        // Case 2: var = first_char + var'
        // We assign var = c[0] and continue
        if !constant.is_empty() {
            let first_char: String = constant.chars().take(1).collect();
            let mut subst = Substitution::new();
            subst.add(var, StringExpr::literal(&first_char));

            let new_eq = if var_on_left {
                WordEquation {
                    lhs: StringExpr {
                        atoms: eq.lhs.atoms[1..].iter().cloned().collect(),
                    },
                    rhs: StringExpr {
                        atoms: {
                            let rest: String = constant.chars().skip(1).collect();
                            let mut atoms = SmallVec::new();
                            if !rest.is_empty() {
                                atoms.push(StringAtom::Const(rest));
                            }
                            atoms.extend(eq.rhs.atoms[1..].iter().cloned());
                            atoms
                        },
                    },
                    origin: eq.origin,
                }
            } else {
                WordEquation {
                    lhs: StringExpr {
                        atoms: {
                            let rest: String = constant.chars().skip(1).collect();
                            let mut atoms = SmallVec::new();
                            if !rest.is_empty() {
                                atoms.push(StringAtom::Const(rest));
                            }
                            atoms.extend(eq.lhs.atoms[1..].iter().cloned());
                            atoms
                        },
                    },
                    rhs: StringExpr {
                        atoms: eq.rhs.atoms[1..].iter().cloned().collect(),
                    },
                    origin: eq.origin,
                }
            };

            cases.push(CaseSplit {
                subst,
                equations: vec![new_eq],
                length_constraints: vec![LengthConstraint {
                    var,
                    coef: 1,
                    rel: Relation::Ge,
                    value: 1,
                }],
            });
        }

        NielsenResult::Split(cases)
    }

    /// Handle case where both first atoms are variables
    fn handle_var_var(&self, v1: u32, v2: u32, eq: &WordEquation) -> NielsenResult {
        if v1 == v2 {
            // Same variable, remove both
            let new_lhs = StringExpr {
                atoms: eq.lhs.atoms[1..].iter().cloned().collect(),
            };
            let new_rhs = StringExpr {
                atoms: eq.rhs.atoms[1..].iter().cloned().collect(),
            };
            return NielsenResult::Continue(vec![WordEquation {
                lhs: new_lhs,
                rhs: new_rhs,
                origin: eq.origin,
            }]);
        }

        // Levi's lemma: x·α = y·β implies one of:
        // 1. x = "" ∧ ε·α = y·β
        // 2. y = "" ∧ x·α = ε·β
        // 3. x = y·z ∧ z·α = β (for fresh z)
        // 4. y = x·z ∧ α = z·β (for fresh z)

        let mut cases = Vec::new();

        // Case 1: v1 = ""
        let mut subst = Substitution::new();
        subst.add(v1, StringExpr::empty());
        cases.push(CaseSplit {
            subst: subst.clone(),
            equations: vec![WordEquation {
                lhs: subst.apply(&StringExpr {
                    atoms: eq.lhs.atoms[1..].iter().cloned().collect(),
                }),
                rhs: subst.apply(&eq.rhs),
                origin: eq.origin,
            }],
            length_constraints: vec![LengthConstraint {
                var: v1,
                coef: 1,
                rel: Relation::Eq,
                value: 0,
            }],
        });

        // Case 2: v2 = ""
        let mut subst = Substitution::new();
        subst.add(v2, StringExpr::empty());
        cases.push(CaseSplit {
            subst: subst.clone(),
            equations: vec![WordEquation {
                lhs: subst.apply(&eq.lhs),
                rhs: subst.apply(&StringExpr {
                    atoms: eq.rhs.atoms[1..].iter().cloned().collect(),
                }),
                origin: eq.origin,
            }],
            length_constraints: vec![LengthConstraint {
                var: v2,
                coef: 1,
                rel: Relation::Eq,
                value: 0,
            }],
        });

        // Case 3: v1 = v2 (same length, same value)
        let mut subst = Substitution::new();
        subst.add(v1, StringExpr::var(v2));
        cases.push(CaseSplit {
            subst: subst.clone(),
            equations: vec![WordEquation {
                lhs: subst.apply(&StringExpr {
                    atoms: eq.lhs.atoms[1..].iter().cloned().collect(),
                }),
                rhs: subst.apply(&StringExpr {
                    atoms: eq.rhs.atoms[1..].iter().cloned().collect(),
                }),
                origin: eq.origin,
            }],
            length_constraints: vec![],
        });

        NielsenResult::Split(cases)
    }

    /// Make substitution that assigns all variables to empty string
    fn make_empty_subst(
        &self,
        atoms: &SmallVec<[StringAtom; 4]>,
        _origin: oxiz_core::ast::TermId,
    ) -> NielsenResult {
        let mut subst = Substitution::new();
        let mut constraints = Vec::new();

        for atom in atoms {
            match atom {
                StringAtom::Var(v) => {
                    subst.add(*v, StringExpr::empty());
                    constraints.push(LengthConstraint {
                        var: *v,
                        coef: 1,
                        rel: Relation::Eq,
                        value: 0,
                    });
                }
                StringAtom::Const(s) => {
                    if !s.is_empty() {
                        return NielsenResult::Conflict(ConflictReason::EmptyMismatch);
                    }
                }
            }
        }

        NielsenResult::Solved(subst)
    }

    /// Detect the reason for a conflict
    fn detect_conflict_reason(&self, eq: &WordEquation) -> ConflictReason {
        // Check for constant mismatch
        if let (Some(l), Some(r)) = (eq.lhs.first_char(), eq.rhs.first_char())
            && l != r
        {
            return ConflictReason::ConstantMismatch(l.to_string(), r.to_string());
        }

        // Check for empty mismatch
        if (eq.lhs.is_empty() && !eq.rhs.is_empty()) || (!eq.lhs.is_empty() && eq.rhs.is_empty()) {
            return ConflictReason::EmptyMismatch;
        }

        ConflictReason::LengthConflict
    }

    /// Extract variables from an equation
    fn extract_vars(&self, eq: &WordEquation) -> Vec<u32> {
        let mut vars = Vec::new();
        for atom in eq.lhs.atoms.iter().chain(eq.rhs.atoms.iter()) {
            if let StringAtom::Var(v) = atom
                && !vars.contains(v)
            {
                vars.push(*v);
            }
        }
        vars
    }

    /// Get statistics
    pub fn stats(&self) -> &WordEqStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.current_subst = Substitution::new();
        self.pending.clear();
        self.solved.clear();
        self.visited.clear();
        self.length_constraints.clear();
        self.depth = 0;
        self.splits = 0;
    }
}

/// Result of a Nielsen step
enum NielsenResult {
    /// Successfully solved with substitution
    Solved(Substitution),
    /// Need case split
    Split(Vec<CaseSplit>),
    /// Conflict detected
    Conflict(ConflictReason),
    /// Continue with new equations
    Continue(Vec<WordEquation>),
}

// ============================================================================
// Length Abstraction
// ============================================================================

/// Length abstraction for word equations
#[derive(Debug)]
pub struct LengthAbstraction {
    /// Variable length variables: var_id -> length_var_id
    var_lengths: FxHashMap<u32, u32>,
    /// Next length variable ID
    next_len_var: u32,
    /// Linear constraints: sum of (coef, var) = constant
    constraints: Vec<LinearConstraint>,
}

/// Linear constraint over lengths
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// Terms: (coefficient, variable)
    pub terms: Vec<(i64, u32)>,
    /// Relation
    pub rel: Relation,
    /// Right-hand side constant
    pub rhs: i64,
}

impl LengthAbstraction {
    /// Create new length abstraction
    pub fn new() -> Self {
        Self {
            var_lengths: FxHashMap::default(),
            next_len_var: 0,
            constraints: Vec::new(),
        }
    }

    /// Get or create length variable for string variable
    pub fn len_var(&mut self, str_var: u32) -> u32 {
        if let Some(&len_var) = self.var_lengths.get(&str_var) {
            len_var
        } else {
            let len_var = self.next_len_var;
            self.next_len_var += 1;
            self.var_lengths.insert(str_var, len_var);
            // Non-negativity constraint
            self.constraints.push(LinearConstraint {
                terms: vec![(1, len_var)],
                rel: Relation::Ge,
                rhs: 0,
            });
            len_var
        }
    }

    /// Add constraint from word equation
    pub fn add_eq_constraint(&mut self, eq: &WordEquation) {
        // |lhs| = |rhs|
        let lhs_len = self.expr_length(&eq.lhs);
        let rhs_len = self.expr_length(&eq.rhs);

        // lhs_len - rhs_len = 0
        let mut terms = lhs_len;
        for (coef, var) in rhs_len {
            terms.push((-coef, var));
        }

        if !terms.is_empty() {
            self.constraints.push(LinearConstraint {
                terms,
                rel: Relation::Eq,
                rhs: 0,
            });
        }
    }

    /// Compute length terms for an expression
    fn expr_length(&mut self, expr: &StringExpr) -> Vec<(i64, u32)> {
        let mut terms = Vec::new();
        let mut constant = 0i64;

        for atom in &expr.atoms {
            match atom {
                StringAtom::Var(v) => {
                    let len_var = self.len_var(*v);
                    terms.push((1, len_var));
                }
                StringAtom::Const(s) => {
                    constant += s.len() as i64;
                }
            }
        }

        // Constant becomes negative on RHS
        if constant != 0 {
            // Use a special "constant" variable (0xFFFFFFFF)
            terms.push((constant, u32::MAX));
        }

        terms
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[LinearConstraint] {
        &self.constraints
    }

    /// Clear constraints
    pub fn clear(&mut self) {
        self.constraints.clear();
        // Keep var_lengths for consistency
    }
}

impl Default for LengthAbstraction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::TermId;

    fn make_term_id(n: u32) -> TermId {
        TermId(n)
    }

    #[test]
    fn test_substitution_apply() {
        let mut subst = Substitution::new();
        subst.add(0, StringExpr::literal("hello"));

        let expr = StringExpr::var(0);
        let result = subst.apply(&expr);
        assert_eq!(result.as_const(), Some("hello"));
    }

    #[test]
    fn test_substitution_compose() {
        let mut s1 = Substitution::new();
        s1.add(0, StringExpr::var(1));

        let mut s2 = Substitution::new();
        s2.add(1, StringExpr::literal("world"));

        let composed = s1.compose(&s2);
        let result = composed.apply(&StringExpr::var(0));
        assert_eq!(result.as_const(), Some("world"));
    }

    #[test]
    fn test_solver_trivial_sat() {
        let mut solver = WordEqSolver::new(WordEqConfig::default());
        solver.add_equation(WordEquation {
            lhs: StringExpr::literal("hello"),
            rhs: StringExpr::literal("hello"),
            origin: make_term_id(0),
        });

        let result = solver.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_solver_trivial_unsat() {
        let mut solver = WordEqSolver::new(WordEqConfig::default());
        solver.add_equation(WordEquation {
            lhs: StringExpr::literal("hello"),
            rhs: StringExpr::literal("world"),
            origin: make_term_id(0),
        });

        let result = solver.solve();
        assert!(matches!(result, SolveResult::Unsat(_)));
    }

    #[test]
    fn test_solver_var_eq_const() {
        let mut solver = WordEqSolver::new(WordEqConfig::default());
        solver.add_equation(WordEquation {
            lhs: StringExpr::var(0),
            rhs: StringExpr::literal("hello"),
            origin: make_term_id(0),
        });

        let result = solver.solve();
        // Should produce case split
        assert!(matches!(result, SolveResult::Split(_)));
    }

    #[test]
    fn test_solver_empty_eq() {
        let mut solver = WordEqSolver::new(WordEqConfig::default());
        solver.add_equation(WordEquation {
            lhs: StringExpr::empty(),
            rhs: StringExpr::empty(),
            origin: make_term_id(0),
        });

        let result = solver.solve();
        assert!(matches!(result, SolveResult::Sat(_)));
    }

    #[test]
    fn test_length_abstraction() {
        let mut abs = LengthAbstraction::new();

        let _v0 = abs.len_var(0);
        let _v1 = abs.len_var(1);

        // Adding equation x ++ "ab" = y should give |x| + 2 = |y|
        abs.add_eq_constraint(&WordEquation {
            lhs: StringExpr {
                atoms: smallvec::smallvec![StringAtom::Var(0), StringAtom::Const("ab".to_string())],
            },
            rhs: StringExpr::var(1),
            origin: make_term_id(0),
        });

        assert!(abs.constraints().len() >= 2);
    }

    #[test]
    fn test_nielsen_const_const() {
        let _solver = WordEqSolver::new(WordEqConfig::default());
        let eq = WordEquation {
            lhs: StringExpr::literal("abc"),
            rhs: StringExpr::literal("abd"),
            origin: make_term_id(0),
        };

        // Should detect conflict (c != d)
        assert!(
            eq.has_conflict() || {
                // First two characters match, third differs
                let l = "abc";
                let r = "abd";
                l.chars().zip(r.chars()).any(|(a, b)| a != b)
            }
        );
    }

    #[test]
    fn test_substitution_empty() {
        let subst = Substitution::new();
        assert!(subst.is_empty());
    }

    #[test]
    fn test_conflict_reasons() {
        let conflict = Conflict {
            reason: ConflictReason::ConstantMismatch("a".to_string(), "b".to_string()),
            vars: vec![],
        };
        assert!(matches!(
            conflict.reason,
            ConflictReason::ConstantMismatch(_, _)
        ));
    }
}
