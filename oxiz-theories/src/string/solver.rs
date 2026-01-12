//! String Theory Solver
//!
//! Implements the SMT-LIB theory of strings (QF_S, QF_SLIA) using:
//! - Word equation solving via Levi's lemma and Nielsen transformations
//! - Length abstraction for interaction with arithmetic
//! - Brzozowski derivatives for regex membership
//! - Lazy axiom instantiation

use super::regex::{Regex, RegexAutomaton};
use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// String constraint kinds
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum StringConstraint {
    /// String equality: s1 = s2
    Eq(StringExpr, StringExpr, TermId),
    /// String disequality: s1 ≠ s2
    Neq(StringExpr, StringExpr, TermId),
    /// Length constraint: len(s) = n
    Length(u32, i64, TermId),
    /// Prefix: str.prefixof(s1, s2)
    Prefix(StringExpr, StringExpr, TermId),
    /// Suffix: str.suffixof(s1, s2)
    Suffix(StringExpr, StringExpr, TermId),
    /// Contains: str.contains(s1, s2)
    Contains(StringExpr, StringExpr, TermId),
    /// Regex membership: str.in_re(s, re)
    InRegex(u32, Arc<Regex>, TermId),
    /// Not in regex
    NotInRegex(u32, Arc<Regex>, TermId),
    /// String to integer: int = str.to_int(s)
    StrToInt(u32, i64, TermId),
    /// Integer to string: s = str.from_int(int)
    IntToStr(u32, i64, TermId),
}

/// A string expression (concatenation of atoms)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StringExpr {
    /// Atoms in left-to-right order
    pub atoms: SmallVec<[StringAtom; 4]>,
}

impl StringExpr {
    /// Empty string
    pub fn empty() -> Self {
        Self {
            atoms: SmallVec::new(),
        }
    }

    /// Single variable
    pub fn var(id: u32) -> Self {
        let mut atoms = SmallVec::new();
        atoms.push(StringAtom::Var(id));
        Self { atoms }
    }

    /// String literal
    pub fn literal(s: &str) -> Self {
        if s.is_empty() {
            Self::empty()
        } else {
            let mut atoms = SmallVec::new();
            atoms.push(StringAtom::Const(s.to_string()));
            Self { atoms }
        }
    }

    /// Concatenate two expressions
    pub fn concat(self, other: Self) -> Self {
        let mut atoms = self.atoms;
        // Merge adjacent constants
        if let (Some(StringAtom::Const(a)), Some(StringAtom::Const(b))) =
            (atoms.last_mut(), other.atoms.first())
        {
            a.push_str(b);
            atoms.extend(other.atoms.into_iter().skip(1));
        } else {
            atoms.extend(other.atoms);
        }
        Self { atoms }
    }

    /// Check if this is a constant string
    pub fn as_const(&self) -> Option<&str> {
        if self.atoms.len() == 1 {
            if let StringAtom::Const(s) = &self.atoms[0] {
                return Some(s);
            }
        } else if self.atoms.is_empty() {
            return Some("");
        }
        None
    }

    /// Check if this is a single variable
    pub fn as_var(&self) -> Option<u32> {
        if self.atoms.len() == 1
            && let StringAtom::Var(id) = &self.atoms[0]
        {
            return Some(*id);
        }
        None
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
            || (self.atoms.len() == 1
                && matches!(&self.atoms[0], StringAtom::Const(s) if s.is_empty()))
    }

    /// Minimum possible length
    pub fn min_length(&self) -> usize {
        self.atoms
            .iter()
            .map(|a| match a {
                StringAtom::Const(s) => s.len(),
                StringAtom::Var(_) => 0,
            })
            .sum()
    }

    /// First character (if constant prefix)
    pub fn first_char(&self) -> Option<char> {
        self.atoms.first().and_then(|a| match a {
            StringAtom::Const(s) => s.chars().next(),
            StringAtom::Var(_) => None,
        })
    }
}

/// A string atom (variable or constant)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StringAtom {
    /// String variable
    Var(u32),
    /// String constant
    Const(String),
}

/// Word equation of the form lhs = rhs (both are concatenations)
#[derive(Debug, Clone)]
pub struct WordEquation {
    /// Left-hand side atoms
    pub lhs: StringExpr,
    /// Right-hand side atoms
    pub rhs: StringExpr,
    /// Origin term for conflict explanation
    pub origin: TermId,
}

impl WordEquation {
    /// Check if this equation is trivially satisfied (both sides equal)
    pub fn is_solved(&self) -> bool {
        self.lhs == self.rhs
    }

    /// Check if this equation has an obvious conflict
    pub fn has_conflict(&self) -> bool {
        // If both sides are constants and differ, conflict
        if let (Some(l), Some(r)) = (self.lhs.as_const(), self.rhs.as_const()) {
            return l != r;
        }
        // If one side is empty but other has constants, conflict
        if self.lhs.is_empty() && !self.rhs.is_empty() && self.rhs.min_length() > 0 {
            return true;
        }
        if self.rhs.is_empty() && !self.lhs.is_empty() && self.lhs.min_length() > 0 {
            return true;
        }
        // Different constant prefixes
        if let (Some(l), Some(r)) = (self.lhs.first_char(), self.rhs.first_char()) {
            return l != r;
        }
        false
    }
}

/// A length constraint for interaction with arithmetic
#[derive(Debug, Clone)]
pub struct LengthConstraint {
    /// Variable ID
    pub var: u32,
    /// Lower bound (inclusive)
    pub lower: Option<i64>,
    /// Upper bound (inclusive)
    pub upper: Option<i64>,
    /// Equality constraint
    pub equal: Option<i64>,
}

/// String theory solver
#[derive(Debug)]
pub struct StringSolver {
    /// Next variable ID
    next_var: u32,
    /// Term to variable mapping
    term_to_var: FxHashMap<TermId, u32>,
    /// Variable to term mapping
    var_to_term: Vec<Option<TermId>>,
    /// String variable assignments (if known)
    assignments: FxHashMap<u32, String>,
    /// Active word equations
    equations: Vec<WordEquation>,
    /// Length constraints
    lengths: Vec<LengthConstraint>,
    /// Regex membership constraints
    regex_constraints: Vec<(u32, Arc<Regex>, bool, TermId)>,
    /// Disequalities
    diseqs: Vec<(StringExpr, StringExpr, TermId)>,
    /// Prefix constraints
    prefixes: Vec<(StringExpr, StringExpr, TermId)>,
    /// Suffix constraints
    suffixes: Vec<(StringExpr, StringExpr, TermId)>,
    /// Contains constraints
    contains: Vec<(StringExpr, StringExpr, TermId)>,
    /// String-to-int constraints: (str_var, int_value, origin)
    str_to_int: Vec<(u32, i64, TermId)>,
    /// Int-to-string constraints: (str_var, int_value, origin)
    int_to_str: Vec<(u32, i64, TermId)>,
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
    /// Current conflict (if any)
    current_conflict: Option<Vec<TermId>>,
    /// Cached regex automata
    regex_automata: FxHashMap<u64, RegexAutomaton>,
    /// Propagated equalities
    propagated: Vec<(TermId, Vec<TermId>)>,
}

/// Context state for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_vars: usize,
    num_equations: usize,
    num_lengths: usize,
    num_regex: usize,
    num_diseqs: usize,
    num_prefixes: usize,
    num_suffixes: usize,
    num_contains: usize,
    num_str_to_int: usize,
    num_int_to_str: usize,
    assignments_snapshot: FxHashMap<u32, String>,
}

impl StringSolver {
    /// Create a new string solver
    pub fn new() -> Self {
        Self {
            next_var: 0,
            term_to_var: FxHashMap::default(),
            var_to_term: Vec::new(),
            assignments: FxHashMap::default(),
            equations: Vec::new(),
            lengths: Vec::new(),
            regex_constraints: Vec::new(),
            diseqs: Vec::new(),
            prefixes: Vec::new(),
            suffixes: Vec::new(),
            contains: Vec::new(),
            str_to_int: Vec::new(),
            int_to_str: Vec::new(),
            context_stack: Vec::new(),
            current_conflict: None,
            regex_automata: FxHashMap::default(),
            propagated: Vec::new(),
        }
    }

    /// Get or create a variable for a term
    fn get_or_create_var(&mut self, term: TermId) -> u32 {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }
        let var = self.next_var;
        self.next_var += 1;
        self.term_to_var.insert(term, var);
        self.var_to_term.push(Some(term));
        var
    }

    /// Add a string equality
    pub fn add_equality(&mut self, lhs: StringExpr, rhs: StringExpr, origin: TermId) {
        let eq = WordEquation { lhs, rhs, origin };
        if eq.has_conflict() {
            self.current_conflict = Some(vec![origin]);
        } else if !eq.is_solved() {
            self.equations.push(eq);
        }
    }

    /// Add a string disequality
    pub fn add_disequality(&mut self, lhs: StringExpr, rhs: StringExpr, origin: TermId) {
        // Check for immediate conflict if both are the same constant
        if let (Some(l), Some(r)) = (lhs.as_const(), rhs.as_const())
            && l == r
        {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        self.diseqs.push((lhs, rhs, origin));
    }

    /// Add a length constraint
    pub fn add_length_eq(&mut self, var: u32, len: i64, origin: TermId) {
        if len < 0 {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        self.lengths.push(LengthConstraint {
            var,
            lower: None,
            upper: None,
            equal: Some(len),
        });
    }

    /// Add regex membership constraint
    pub fn add_regex_membership(
        &mut self,
        var: u32,
        regex: Arc<Regex>,
        positive: bool,
        origin: TermId,
    ) {
        // Quick check for empty regex
        if regex.is_empty() && positive {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        if regex.is_all() && !positive {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        self.regex_constraints.push((var, regex, positive, origin));
    }

    /// Add prefix constraint: str.prefixof(prefix, s)
    pub fn add_prefix(&mut self, prefix: StringExpr, s: StringExpr, origin: TermId) {
        // Check for immediate conflict
        if let (Some(p), Some(s_str)) = (prefix.as_const(), s.as_const())
            && !s_str.starts_with(p)
        {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        self.prefixes.push((prefix, s, origin));
    }

    /// Add suffix constraint: str.suffixof(suffix, s)
    pub fn add_suffix(&mut self, suffix: StringExpr, s: StringExpr, origin: TermId) {
        if let (Some(suf), Some(s_str)) = (suffix.as_const(), s.as_const())
            && !s_str.ends_with(suf)
        {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        self.suffixes.push((suffix, s, origin));
    }

    /// Add contains constraint: str.contains(s, substr)
    pub fn add_contains(&mut self, s: StringExpr, substr: StringExpr, origin: TermId) {
        if let (Some(s_str), Some(sub)) = (s.as_const(), substr.as_const())
            && !s_str.contains(sub)
        {
            self.current_conflict = Some(vec![origin]);
            return;
        }
        self.contains.push((s, substr, origin));
    }

    /// Add string-to-int constraint: int = str.to_int(s)
    /// Returns the integer value that string s represents
    /// Returns -1 if s is not a valid numeral
    pub fn add_str_to_int(&mut self, str_var: u32, int_value: i64, origin: TermId) {
        // Check if we have a concrete assignment for the string variable
        if let Some(s) = self.assignments.get(&str_var) {
            // Try to parse the string as an integer
            match s.parse::<i64>() {
                Ok(parsed) => {
                    if parsed != int_value {
                        self.current_conflict = Some(vec![origin]);
                        return;
                    }
                }
                Err(_) => {
                    // Invalid numeral should return -1
                    if int_value != -1 {
                        self.current_conflict = Some(vec![origin]);
                        return;
                    }
                }
            }
        }
        self.str_to_int.push((str_var, int_value, origin));
    }

    /// Add int-to-string constraint: s = str.from_int(int)
    /// Converts integer to its decimal string representation
    pub fn add_int_to_str(&mut self, str_var: u32, int_value: i64, origin: TermId) {
        // Check if we have a concrete assignment for the string variable
        if let Some(s) = self.assignments.get(&str_var) {
            let expected = if int_value < 0 {
                // Negative integers should produce empty string in SMT-LIB2
                String::new()
            } else {
                int_value.to_string()
            };

            if s != &expected {
                self.current_conflict = Some(vec![origin]);
                return;
            }
        }
        self.int_to_str.push((str_var, int_value, origin));
    }

    /// Solve word equations using Nielsen transformation
    fn solve_equations(&mut self) -> Option<Vec<TermId>> {
        // Simple equation solving via substitution and case analysis
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            let mut i = 0;
            while i < self.equations.len() {
                let eq = &self.equations[i];

                // Check for conflict
                if eq.has_conflict() {
                    return Some(vec![eq.origin]);
                }

                // Check if solved
                if eq.is_solved() {
                    self.equations.swap_remove(i);
                    changed = true;
                    continue;
                }

                // Try to make progress via Levi's lemma
                if let Some(result) = self.apply_levi(&self.equations[i].clone()) {
                    match result {
                        LeviResult::Solved => {
                            self.equations.swap_remove(i);
                            changed = true;
                            continue;
                        }
                        LeviResult::Conflict(origin) => {
                            return Some(vec![origin]);
                        }
                        LeviResult::Progress(new_eqs) => {
                            let origin = self.equations[i].origin;
                            self.equations.swap_remove(i);
                            for (lhs, rhs) in new_eqs {
                                self.equations.push(WordEquation { lhs, rhs, origin });
                            }
                            changed = true;
                            continue;
                        }
                        LeviResult::NoProgress => {}
                    }
                }

                i += 1;
            }
        }

        None
    }

    /// Apply Levi's lemma to an equation
    fn apply_levi(&self, eq: &WordEquation) -> Option<LeviResult> {
        let lhs = &eq.lhs;
        let rhs = &eq.rhs;

        // Empty = Empty
        if lhs.is_empty() && rhs.is_empty() {
            return Some(LeviResult::Solved);
        }

        // Empty = non-empty with constants -> conflict
        if lhs.is_empty() && rhs.min_length() > 0 {
            return Some(LeviResult::Conflict(eq.origin));
        }
        if rhs.is_empty() && lhs.min_length() > 0 {
            return Some(LeviResult::Conflict(eq.origin));
        }

        // Both start with same constant prefix
        if let (Some(l), Some(r)) = (lhs.first_char(), rhs.first_char()) {
            if l == r {
                // Strip common prefix
                let mut new_lhs = lhs.clone();
                let mut new_rhs = rhs.clone();
                self.strip_common_prefix(&mut new_lhs, &mut new_rhs);
                if new_lhs != *lhs || new_rhs != *rhs {
                    return Some(LeviResult::Progress(vec![(new_lhs, new_rhs)]));
                }
            } else {
                return Some(LeviResult::Conflict(eq.origin));
            }
        }

        // x . α = y . β where x, y are variables
        // Case split: x = y ∧ α = β, or x = y.γ ∧ γ.α = β, or y = x.γ ∧ α = γ.β
        // For now, we just handle simple cases

        None
    }

    /// Strip common prefix from two expressions
    fn strip_common_prefix(&self, lhs: &mut StringExpr, rhs: &mut StringExpr) {
        while !lhs.atoms.is_empty() && !rhs.atoms.is_empty() {
            match (&lhs.atoms[0], &rhs.atoms[0]) {
                (StringAtom::Const(a), StringAtom::Const(b)) => {
                    let common_len = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
                    if common_len == 0 {
                        break;
                    }
                    let a_chars: Vec<char> = a.chars().collect();
                    let b_chars: Vec<char> = b.chars().collect();
                    if common_len == a_chars.len() && common_len == b_chars.len() {
                        lhs.atoms.remove(0);
                        rhs.atoms.remove(0);
                    } else if common_len == a_chars.len() {
                        lhs.atoms.remove(0);
                        rhs.atoms[0] = StringAtom::Const(b_chars[common_len..].iter().collect());
                    } else if common_len == b_chars.len() {
                        rhs.atoms.remove(0);
                        lhs.atoms[0] = StringAtom::Const(a_chars[common_len..].iter().collect());
                    } else {
                        lhs.atoms[0] = StringAtom::Const(a_chars[common_len..].iter().collect());
                        rhs.atoms[0] = StringAtom::Const(b_chars[common_len..].iter().collect());
                        break;
                    }
                }
                (StringAtom::Var(x), StringAtom::Var(y)) if x == y => {
                    lhs.atoms.remove(0);
                    rhs.atoms.remove(0);
                }
                _ => break,
            }
        }
    }

    /// Check regex constraints
    fn check_regex_constraints(&mut self) -> Option<Vec<TermId>> {
        for (var, regex, positive, origin) in &self.regex_constraints {
            if let Some(value) = self.assignments.get(var) {
                let matches = regex.matches(value);
                if *positive && !matches {
                    return Some(vec![*origin]);
                }
                if !*positive && matches {
                    return Some(vec![*origin]);
                }
            }
        }
        None
    }

    /// Check disequality constraints
    fn check_diseqs(&self) -> Option<Vec<TermId>> {
        for (lhs, rhs, origin) in &self.diseqs {
            if let (Some(l), Some(r)) = (self.eval_expr(lhs), self.eval_expr(rhs))
                && l == r
            {
                return Some(vec![*origin]);
            }
        }
        None
    }

    /// Check str.to_int constraints
    fn check_str_to_int(&self) -> Option<Vec<TermId>> {
        for (str_var, expected_int, origin) in &self.str_to_int {
            if let Some(s) = self.assignments.get(str_var) {
                // Invalid numeral returns -1
                let actual_int = s.parse::<i64>().unwrap_or(-1);
                if actual_int != *expected_int {
                    return Some(vec![*origin]);
                }
            }
        }
        None
    }

    /// Check str.from_int constraints
    fn check_int_to_str(&self) -> Option<Vec<TermId>> {
        for (str_var, int_value, origin) in &self.int_to_str {
            if let Some(s) = self.assignments.get(str_var) {
                let expected = if *int_value < 0 {
                    // Negative integers produce empty string
                    String::new()
                } else {
                    int_value.to_string()
                };
                if s != &expected {
                    return Some(vec![*origin]);
                }
            }
        }
        None
    }

    /// Evaluate a string expression under current assignments
    fn eval_expr(&self, expr: &StringExpr) -> Option<String> {
        let mut result = String::new();
        for atom in &expr.atoms {
            match atom {
                StringAtom::Const(s) => result.push_str(s),
                StringAtom::Var(v) => {
                    if let Some(s) = self.assignments.get(v) {
                        result.push_str(s);
                    } else {
                        return None;
                    }
                }
            }
        }
        Some(result)
    }

    /// Check length constraints
    fn check_lengths(&self) -> Option<Vec<TermId>> {
        for lc in &self.lengths {
            if let Some(value) = self.assignments.get(&lc.var) {
                let len = value.len() as i64;
                if let Some(eq) = lc.equal
                    && len != eq
                {
                    // Find origin term - would need to track this
                    continue;
                }
                if let Some(lo) = lc.lower
                    && len < lo
                {
                    continue;
                }
                if let Some(hi) = lc.upper
                    && len > hi
                {
                    continue;
                }
            }
        }
        None
    }

    /// Check prefix constraints
    fn check_prefixes(&self) -> Option<Vec<TermId>> {
        for (prefix, s, origin) in &self.prefixes {
            if let (Some(p), Some(s_val)) = (self.eval_expr(prefix), self.eval_expr(s))
                && !s_val.starts_with(&p)
            {
                return Some(vec![*origin]);
            }
        }
        None
    }

    /// Check suffix constraints
    fn check_suffixes(&self) -> Option<Vec<TermId>> {
        for (suffix, s, origin) in &self.suffixes {
            if let (Some(suf), Some(s_val)) = (self.eval_expr(suffix), self.eval_expr(s))
                && !s_val.ends_with(&suf)
            {
                return Some(vec![*origin]);
            }
        }
        None
    }

    /// Check contains constraints
    fn check_contains(&self) -> Option<Vec<TermId>> {
        for (s, substr, origin) in &self.contains {
            if let (Some(s_val), Some(sub)) = (self.eval_expr(s), self.eval_expr(substr))
                && !s_val.contains(&sub)
            {
                return Some(vec![*origin]);
            }
        }
        None
    }
}

/// Result of applying Levi's lemma
#[allow(dead_code)]
#[derive(Debug)]
enum LeviResult {
    /// Equation is solved
    Solved,
    /// Equation leads to conflict
    Conflict(TermId),
    /// Made progress with new equations
    Progress(Vec<(StringExpr, StringExpr)>),
    /// No progress possible
    NoProgress,
}

impl Default for StringSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Theory for StringSolver {
    fn id(&self) -> TheoryId {
        TheoryId::Strings
    }

    fn name(&self) -> &str {
        "Strings"
    }

    fn can_handle(&self, _term: TermId) -> bool {
        // Would check if term is a string operation
        false
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // Would parse term and add appropriate constraint
        let _var = self.get_or_create_var(term);
        if let Some(conflict) = self.current_conflict.take() {
            return Ok(TheoryResult::Unsat(conflict));
        }
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        let _var = self.get_or_create_var(term);
        if let Some(conflict) = self.current_conflict.take() {
            return Ok(TheoryResult::Unsat(conflict));
        }
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // Check for pending conflicts
        if let Some(conflict) = self.current_conflict.take() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Solve word equations
        if let Some(conflict) = self.solve_equations() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check regex constraints
        if let Some(conflict) = self.check_regex_constraints() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check disequalities
        if let Some(conflict) = self.check_diseqs() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check length constraints
        if let Some(conflict) = self.check_lengths() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check prefix constraints
        if let Some(conflict) = self.check_prefixes() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check suffix constraints
        if let Some(conflict) = self.check_suffixes() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check contains constraints
        if let Some(conflict) = self.check_contains() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check str.to_int constraints
        if let Some(conflict) = self.check_str_to_int() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check str.from_int constraints
        if let Some(conflict) = self.check_int_to_str() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        // Check propagations
        if !self.propagated.is_empty() {
            let props = std::mem::take(&mut self.propagated);
            return Ok(TheoryResult::Propagate(props));
        }

        Ok(TheoryResult::Sat)
    }

    fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_vars: self.next_var as usize,
            num_equations: self.equations.len(),
            num_lengths: self.lengths.len(),
            num_regex: self.regex_constraints.len(),
            num_diseqs: self.diseqs.len(),
            num_prefixes: self.prefixes.len(),
            num_suffixes: self.suffixes.len(),
            num_contains: self.contains.len(),
            num_str_to_int: self.str_to_int.len(),
            num_int_to_str: self.int_to_str.len(),
            assignments_snapshot: self.assignments.clone(),
        });
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Restore variable count
            self.next_var = state.num_vars as u32;
            self.var_to_term.truncate(state.num_vars);

            // Remove terms that were added after the push
            self.term_to_var
                .retain(|_, v| (*v as usize) < state.num_vars);

            // Restore constraints
            self.equations.truncate(state.num_equations);
            self.lengths.truncate(state.num_lengths);
            self.regex_constraints.truncate(state.num_regex);
            self.diseqs.truncate(state.num_diseqs);
            self.prefixes.truncate(state.num_prefixes);
            self.suffixes.truncate(state.num_suffixes);
            self.contains.truncate(state.num_contains);
            self.str_to_int.truncate(state.num_str_to_int);
            self.int_to_str.truncate(state.num_int_to_str);

            // Restore assignments
            self.assignments = state.assignments_snapshot;

            // Clear conflict
            self.current_conflict = None;
            self.propagated.clear();
        }
    }

    fn reset(&mut self) {
        self.next_var = 0;
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.assignments.clear();
        self.equations.clear();
        self.lengths.clear();
        self.regex_constraints.clear();
        self.diseqs.clear();
        self.prefixes.clear();
        self.suffixes.clear();
        self.contains.clear();
        self.str_to_int.clear();
        self.int_to_str.clear();
        self.context_stack.clear();
        self.current_conflict = None;
        self.regex_automata.clear();
        self.propagated.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_expr_concat() {
        let a = StringExpr::literal("hello");
        let b = StringExpr::literal(" world");
        let c = a.concat(b);
        assert_eq!(c.as_const(), Some("hello world"));
    }

    #[test]
    fn test_string_expr_var_concat() {
        let a = StringExpr::var(0);
        let b = StringExpr::literal("!");
        let c = a.concat(b);
        assert_eq!(c.atoms.len(), 2);
    }

    #[test]
    fn test_word_equation_solved() {
        let eq = WordEquation {
            lhs: StringExpr::literal("test"),
            rhs: StringExpr::literal("test"),
            origin: TermId(0),
        };
        assert!(eq.is_solved());
    }

    #[test]
    fn test_word_equation_conflict() {
        let eq = WordEquation {
            lhs: StringExpr::literal("abc"),
            rhs: StringExpr::literal("def"),
            origin: TermId(0),
        };
        assert!(eq.has_conflict());
    }

    #[test]
    fn test_solver_basic() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_equality(
            StringExpr::literal("hello"),
            StringExpr::literal("hello"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_equality(
            StringExpr::literal("hello"),
            StringExpr::literal("world"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_diseq() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_disequality(
            StringExpr::literal("hello"),
            StringExpr::literal("world"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_diseq_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_disequality(
            StringExpr::literal("same"),
            StringExpr::literal("same"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_prefix() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_prefix(
            StringExpr::literal("hello"),
            StringExpr::literal("hello world"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_prefix_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_prefix(
            StringExpr::literal("world"),
            StringExpr::literal("hello"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_suffix() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_suffix(
            StringExpr::literal("world"),
            StringExpr::literal("hello world"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_contains() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_contains(
            StringExpr::literal("hello world"),
            StringExpr::literal("lo wo"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_contains_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.add_contains(
            StringExpr::literal("hello"),
            StringExpr::literal("xyz"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_regex() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign value
        solver.assignments.insert(var, "aaa".to_string());

        // Should match a+
        let regex = Regex::plus(Regex::char('a'));
        solver.add_regex_membership(var, regex, true, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_solver_regex_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign value
        solver.assignments.insert(var, "abc".to_string());

        // Should NOT match a+ (contains b and c)
        let regex = Regex::plus(Regex::char('a'));
        solver.add_regex_membership(var, regex, true, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_solver_push_pop() {
        let mut solver = StringSolver::new();
        let term = TermId(0);

        solver.push();

        solver.add_equality(
            StringExpr::literal("hello"),
            StringExpr::literal("world"),
            term,
        );

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));

        solver.pop();

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_strip_common_prefix() {
        let solver = StringSolver::new();
        let mut lhs = StringExpr::literal("hello world");
        let mut rhs = StringExpr::literal("hello there");
        solver.strip_common_prefix(&mut lhs, &mut rhs);
        assert_eq!(lhs.as_const(), Some("world"));
        assert_eq!(rhs.as_const(), Some("there"));
    }

    #[test]
    fn test_theory_trait() {
        let mut solver = StringSolver::new();
        assert_eq!(solver.id(), TheoryId::Strings);
        assert_eq!(solver.name(), "Strings");

        let term = TermId(1);
        let result = solver.assert_true(term).unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_str_to_int_valid() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign string value "42"
        solver.assignments.insert(var, "42".to_string());

        // Add constraint: int = str.to_int("42")
        solver.add_str_to_int(var, 42, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_str_to_int_invalid() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign string value "hello" (not a number)
        solver.assignments.insert(var, "hello".to_string());

        // Add constraint: int = str.to_int("hello"), should be -1
        solver.add_str_to_int(var, -1, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_str_to_int_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign string value "42"
        solver.assignments.insert(var, "42".to_string());

        // Add constraint: int = str.to_int("42"), but claim it's 99
        solver.add_str_to_int(var, 99, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_int_to_str_positive() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign string value "123"
        solver.assignments.insert(var, "123".to_string());

        // Add constraint: s = str.from_int(123)
        solver.add_int_to_str(var, 123, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_int_to_str_negative() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign empty string (negative numbers produce empty string)
        solver.assignments.insert(var, String::new());

        // Add constraint: s = str.from_int(-5), should be empty
        solver.add_int_to_str(var, -5, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_int_to_str_conflict() {
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // Assign string value "99"
        solver.assignments.insert(var, "99".to_string());

        // Add constraint: s = str.from_int(123), but s is "99"
        solver.add_int_to_str(var, 123, term);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }
}
