//! String Theory Solver
//!
//! Implements the SMT-LIB theory of strings (QF_S, QF_SLIA) using:
//! - Word equation solving via Levi's lemma and Nielsen transformations
//! - Length abstraction for interaction with arithmetic
//! - Brzozowski derivatives for regex membership
//! - Lazy axiom instantiation

use super::regex::{Regex, RegexAutomaton};
use super::regex_membership::{VarModel, solve_membership};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::theory::{EqualityNotification, Theory, TheoryCombination, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use smallvec::SmallVec;

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

    /// Return a copy of this expression with every occurrence of `Var(var)`
    /// replaced by the constant `value`, merging any adjacent constants that
    /// result (an empty `value` simply drops the variable atom).
    fn substituted(&self, var: u32, value: &str) -> StringExpr {
        let mut atoms: SmallVec<[StringAtom; 4]> = SmallVec::new();
        for atom in &self.atoms {
            let next = match atom {
                StringAtom::Var(v) if *v == var => {
                    if value.is_empty() {
                        continue;
                    }
                    StringAtom::Const(value.to_string())
                }
                other => other.clone(),
            };
            if let (Some(StringAtom::Const(prev)), StringAtom::Const(cur)) =
                (atoms.last_mut(), &next)
            {
                prev.push_str(cur);
            } else {
                atoms.push(next);
            }
        }
        StringExpr { atoms }
    }
}

/// Outcome of [`StringSolver::solve_equations`].
#[derive(Debug)]
enum SolveOutcome {
    /// A definite unsatisfiable equation was found (with its conflict origins).
    Conflict(Vec<TermId>),
    /// Every active equation was discharged (system consistent so far).
    Resolved,
    /// Equations remain that this incomplete procedure cannot decide; the
    /// caller must report `Unknown`, never `Sat`.
    Unresolved,
}

/// Result of a single [`StringSolver::try_resolve_equation`] step.
#[derive(Debug)]
enum ResolveStep {
    /// The equation was resolved and removed.
    Resolved,
    /// No sound resolution rule applied.
    NoProgress,
}

/// Whether `expr` is non-empty and consists solely of variables that each
/// occur exactly once across the whole equation system (per `occ`) AND appear
/// in no other constraint (per `constrained`). Such "free" variables can be
/// assigned an arbitrary witness without affecting any other equation *or*
/// constraint.
///
/// The `constrained` guard is essential for soundness: `occ` only counts
/// occurrences across word equations, so a variable that occurs once in the
/// equations but also carries a length/regex/prefix/suffix/contains/str<->int
/// constraint would otherwise be treated as free. Committing an arbitrary
/// witness (see [`StringSolver::resolve_free_side`]) for such a variable can
/// violate that other constraint even when a different word-equation solution
/// satisfies everything, producing a false `Unsat`. Excluding constrained
/// variables keeps the equation unresolved (-> honest `Unknown`) instead.
fn is_all_free_once(
    expr: &StringExpr,
    occ: &FxHashMap<u32, usize>,
    constrained: &FxHashSet<u32>,
) -> bool {
    if expr.atoms.is_empty() {
        return false;
    }
    expr.atoms.iter().all(|a| match a {
        StringAtom::Var(v) => occ.get(v).copied().unwrap_or(0) == 1 && !constrained.contains(v),
        StringAtom::Const(_) => false,
    })
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
    /// Origin term that introduced this constraint (for conflict explanations)
    pub origin: TermId,
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
    /// Shared equalities derived by the string theory for Nelson-Oppen combination.
    /// These include length equalities (s = t implies len(s) = len(t)) and
    /// decomposition equalities from word equation solving.
    shared_equalities: Vec<EqualityNotification>,
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
    /// Length of `shared_equalities` at push time, so `pop` can truncate
    /// away equalities derived (via `notify_equality`) inside the popped
    /// scope. Without this, a Nelson-Oppen shared equality that only held
    /// because of an assumption made in the popped scope survives the pop
    /// and keeps being reported to other theories as if it still held.
    num_shared_equalities: usize,
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
            shared_equalities: Vec::new(),
        }
    }

    /// Get or create the string-theory variable that represents `term`.
    ///
    /// This is the entry point a solver uses to register a `String`-sorted
    /// term (e.g. the first operand of `str.in_re`) before attaching
    /// constraints such as [`Self::add_regex_membership`].
    pub fn get_or_create_var(&mut self, term: TermId) -> u32 {
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
            origin,
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

    /// Solve word equations by normalization + sound substitution.
    ///
    /// Repeatedly, for each equation: strip matching constant prefix/suffix,
    /// detect trivial conflicts/solutions, and apply the substitution rules in
    /// [`Self::try_resolve_equation`] (each of which is sound: it only fires
    /// when a variable's value is *forced*, or when a side consists solely of
    /// variables that occur nowhere else and can therefore be assigned freely).
    ///
    /// Returns:
    /// - [`SolveOutcome::Conflict`] on a definite unsatisfiable equation,
    /// - [`SolveOutcome::Resolved`] when every equation was discharged,
    /// - [`SolveOutcome::Unresolved`] when equations remain that this
    ///   (deliberately incomplete) procedure cannot decide. The caller MUST
    ///   translate `Unresolved` into `Unknown` rather than claiming `Sat`.
    fn solve_equations(&mut self) -> SolveOutcome {
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            let mut i = 0;
            while i < self.equations.len() {
                // Normalize by stripping the matching constant prefix/suffix.
                let mut lhs = self.equations[i].lhs.clone();
                let mut rhs = self.equations[i].rhs.clone();
                self.strip_common_prefix(&mut lhs, &mut rhs);
                self.strip_common_suffix(&mut lhs, &mut rhs);
                if lhs != self.equations[i].lhs || rhs != self.equations[i].rhs {
                    self.equations[i].lhs = lhs;
                    self.equations[i].rhs = rhs;
                    changed = true;
                }

                // Definite conflict?
                if self.equations[i].has_conflict() {
                    return SolveOutcome::Conflict(vec![self.equations[i].origin]);
                }

                // Trivially solved?
                if self.equations[i].is_solved() {
                    self.equations.swap_remove(i);
                    changed = true;
                    continue;
                }

                match self.try_resolve_equation(i) {
                    ResolveStep::Resolved => {
                        changed = true;
                        continue;
                    }
                    ResolveStep::NoProgress => {}
                }

                i += 1;
            }
        }

        if self.equations.is_empty() {
            SolveOutcome::Resolved
        } else {
            SolveOutcome::Unresolved
        }
    }

    /// Attempt one sound resolution step on the equation at index `i`.
    ///
    /// On success the equation is removed (and any implied assignment is
    /// substituted into the remaining equations). Only fires when correctness
    /// is guaranteed:
    /// - Rule A: a single-variable side whose partner evaluates to a concrete
    ///   ground string forces that variable's value.
    /// - Rule B/C: a side made up solely of variables that each occur exactly
    ///   once across the whole system can absorb a ground partner (or two such
    ///   sides can both be emptied); occur-once variables appear nowhere else,
    ///   so the chosen assignment cannot invalidate any other equation.
    fn try_resolve_equation(&mut self, i: usize) -> ResolveStep {
        let lhs = self.equations[i].lhs.clone();
        let rhs = self.equations[i].rhs.clone();

        // Rule A: single-variable side determined by a ground partner.
        if let Some(x) = lhs.as_var()
            && let Some(c) = self.eval_expr(&rhs)
        {
            self.equations.swap_remove(i);
            self.substitute_all(x, &c);
            return ResolveStep::Resolved;
        }
        if let Some(x) = rhs.as_var()
            && let Some(c) = self.eval_expr(&lhs)
        {
            self.equations.swap_remove(i);
            self.substitute_all(x, &c);
            return ResolveStep::Resolved;
        }

        // Rule B/C: free (occur-once) variable side(s). A variable is only
        // "free" if it occurs once across the equations AND participates in no
        // other constraint, otherwise the arbitrary witness committed below
        // could contradict that constraint (see `constrained_vars`).
        let occ = self.var_occurrences();
        let constrained = self.constrained_vars();
        let lhs_free_once = is_all_free_once(&lhs, &occ, &constrained);
        let rhs_free_once = is_all_free_once(&rhs, &occ, &constrained);

        if lhs_free_once && let Some(g) = self.eval_expr(&rhs) {
            self.resolve_free_side(i, &lhs, &g);
            return ResolveStep::Resolved;
        }
        if rhs_free_once && let Some(g) = self.eval_expr(&lhs) {
            self.resolve_free_side(i, &rhs, &g);
            return ResolveStep::Resolved;
        }
        if lhs_free_once && rhs_free_once {
            let vars: Vec<u32> = lhs
                .atoms
                .iter()
                .chain(rhs.atoms.iter())
                .filter_map(|a| match a {
                    StringAtom::Var(v) => Some(*v),
                    StringAtom::Const(_) => None,
                })
                .collect();
            self.equations.swap_remove(i);
            for v in vars {
                self.substitute_all(v, "");
            }
            return ResolveStep::Resolved;
        }

        ResolveStep::NoProgress
    }

    /// Resolve an equation whose `side` is entirely occur-once variables and
    /// whose partner evaluates to the concrete string `ground`: assign the
    /// first variable the whole ground value and the rest the empty string,
    /// then substitute. Sound because these variables occur nowhere else.
    fn resolve_free_side(&mut self, i: usize, side: &StringExpr, ground: &str) {
        let vars: Vec<u32> = side
            .atoms
            .iter()
            .filter_map(|a| match a {
                StringAtom::Var(v) => Some(*v),
                StringAtom::Const(_) => None,
            })
            .collect();
        self.equations.swap_remove(i);
        for (k, v) in vars.iter().enumerate() {
            if k == 0 {
                self.substitute_all(*v, ground);
            } else {
                self.substitute_all(*v, "");
            }
        }
    }

    /// Count how many times each variable occurs across all active equations
    /// (both sides). Used to identify occur-once ("free") variables that can
    /// be assigned without affecting any other equation.
    fn var_occurrences(&self) -> FxHashMap<u32, usize> {
        let mut counts: FxHashMap<u32, usize> = FxHashMap::default();
        for eq in &self.equations {
            for atom in eq.lhs.atoms.iter().chain(eq.rhs.atoms.iter()) {
                if let StringAtom::Var(v) = atom {
                    *counts.entry(*v).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    /// Collect every variable that participates in a constraint *other than*
    /// the word equations: length, regex membership, disequality, prefix,
    /// suffix, contains, and str<->int constraints.
    ///
    /// These variables must be excluded from the occur-once "free variable"
    /// resolution rules ([`Self::resolve_free_side`] and the both-sides-empty
    /// case in [`Self::try_resolve_equation`]), which commit an *arbitrary*
    /// witness. That witness is only sound when the variable is genuinely
    /// unconstrained: if it also appears here, the arbitrary choice could
    /// violate the other constraint even though a different word-equation
    /// solution would satisfy everything, giving a false `Unsat`.
    fn constrained_vars(&self) -> FxHashSet<u32> {
        let mut set: FxHashSet<u32> = FxHashSet::default();
        for l in &self.lengths {
            set.insert(l.var);
        }
        for (v, _regex, _positive, _origin) in &self.regex_constraints {
            set.insert(*v);
        }
        for (v, _value, _origin) in &self.str_to_int {
            set.insert(*v);
        }
        for (v, _value, _origin) in &self.int_to_str {
            set.insert(*v);
        }
        for (lhs, rhs, _origin) in self
            .diseqs
            .iter()
            .chain(self.prefixes.iter())
            .chain(self.suffixes.iter())
            .chain(self.contains.iter())
        {
            for atom in lhs.atoms.iter().chain(rhs.atoms.iter()) {
                if let StringAtom::Var(v) = atom {
                    set.insert(*v);
                }
            }
        }
        set
    }

    /// Record `var = value` and substitute it into every active equation.
    fn substitute_all(&mut self, var: u32, value: &str) {
        self.assignments.insert(var, value.to_string());
        for eq in &mut self.equations {
            eq.lhs = eq.lhs.substituted(var, value);
            eq.rhs = eq.rhs.substituted(var, value);
        }
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

    /// Strip common suffix from two expressions (mirror of
    /// [`Self::strip_common_prefix`], working from the end).
    fn strip_common_suffix(&self, lhs: &mut StringExpr, rhs: &mut StringExpr) {
        while !lhs.atoms.is_empty() && !rhs.atoms.is_empty() {
            let li = lhs.atoms.len() - 1;
            let ri = rhs.atoms.len() - 1;
            match (&lhs.atoms[li], &rhs.atoms[ri]) {
                (StringAtom::Const(a), StringAtom::Const(b)) => {
                    let a_chars: Vec<char> = a.chars().collect();
                    let b_chars: Vec<char> = b.chars().collect();
                    let common = a_chars
                        .iter()
                        .rev()
                        .zip(b_chars.iter().rev())
                        .take_while(|(x, y)| x == y)
                        .count();
                    if common == 0 {
                        break;
                    }
                    let a_full = common == a_chars.len();
                    let b_full = common == b_chars.len();
                    if a_full && b_full {
                        lhs.atoms.pop();
                        rhs.atoms.pop();
                    } else if a_full {
                        lhs.atoms.pop();
                        rhs.atoms[ri] =
                            StringAtom::Const(b_chars[..b_chars.len() - common].iter().collect());
                    } else if b_full {
                        rhs.atoms.pop();
                        lhs.atoms[li] =
                            StringAtom::Const(a_chars[..a_chars.len() - common].iter().collect());
                    } else {
                        lhs.atoms[li] =
                            StringAtom::Const(a_chars[..a_chars.len() - common].iter().collect());
                        rhs.atoms[ri] =
                            StringAtom::Const(b_chars[..b_chars.len() - common].iter().collect());
                        break;
                    }
                }
                (StringAtom::Var(x), StringAtom::Var(y)) if x == y => {
                    lhs.atoms.pop();
                    rhs.atoms.pop();
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
                    let s = self.assignments.get(v)?;
                    result.push_str(s);
                }
            }
        }
        Some(result)
    }

    /// Check length constraints
    fn check_lengths(&self) -> Option<Vec<TermId>> {
        for lc in &self.lengths {
            if let Some(value) = self.assignments.get(&lc.var) {
                // Count Unicode scalar values, not UTF-8 bytes, so the length
                // matches SMT-LIB string semantics (str.len counts characters).
                let len = value.chars().count() as i64;
                if let Some(eq) = lc.equal
                    && len != eq
                {
                    // Assigned string violates the asserted length equality:
                    // report the origin of the length constraint as the conflict.
                    return Some(vec![lc.origin]);
                }
                if let Some(lo) = lc.lower
                    && len < lo
                {
                    return Some(vec![lc.origin]);
                }
                if let Some(hi) = lc.upper
                    && len > hi
                {
                    return Some(vec![lc.origin]);
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

        // Solve word equations. A definite conflict is Unsat immediately; an
        // `Unresolved` outcome is remembered so we can honestly report
        // `Unknown` at the end rather than claiming `Sat`.
        let outcome = self.solve_equations();
        if let SolveOutcome::Conflict(conflict) = &outcome {
            return Ok(TheoryResult::Unsat(conflict.clone()));
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
            let props = core::mem::take(&mut self.propagated);
            return Ok(TheoryResult::Propagate(props));
        }

        // Honesty gate. `solve_equations` now resolves determinable word
        // equations (via prefix/suffix stripping + sound substitution), so a
        // consistent case such as `x ++ "!" = "hello!"` empties the equation
        // list and stays `Sat`. Anything it could NOT resolve remains in
        // `self.equations`; reporting `Sat` there would be a silent wrong
        // result, so we report `Unknown` for the unhandled fragment. Likewise
        // a regex membership constraint on a still-unassigned variable is
        // undecided by this theory and must not be silently accepted as `Sat`.
        if matches!(outcome, SolveOutcome::Unresolved) {
            return Ok(TheoryResult::Unknown);
        }
        // Actively construct a satisfying string value for variables that carry
        // only regex-membership (and length) constraints: search the product
        // automaton for the shortest accepted word. A provably empty language
        // is a real conflict (unsatisfiable / empty intersection); anything the
        // search cannot decide is left unassigned and falls through to the
        // honest `Unknown` gate below.
        if let Some(conflict) = self.construct_regex_models() {
            return Ok(TheoryResult::Unsat(conflict));
        }
        for (var, _regex, _positive, _origin) in &self.regex_constraints {
            if !self.assignments.contains_key(var) {
                return Ok(TheoryResult::Unknown);
            }
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
            num_shared_equalities: self.shared_equalities.len(),
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

            // Restore shared equalities: drop any Nelson-Oppen equalities
            // derived inside the popped scope so they stop being reported
            // to other theories once that scope's assumptions are gone.
            self.shared_equalities.truncate(state.num_shared_equalities);

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
        self.shared_equalities.clear();
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        // Build a model from the string solver's variable assignments.
        // For each string variable with a known assignment, group variables
        // that share the same concrete string value and map them to a
        // representative term (the first term found with that value).
        //
        // Variables without concrete assignments are also included in the model:
        // each unassigned variable maps to itself as a representative.

        let mut value_to_repr: FxHashMap<&str, TermId> = FxHashMap::default();
        let mut assignments = Vec::new();

        // First pass: collect representatives for each string value
        for (&var, val) in &self.assignments {
            if let Some(Some(term)) = self.var_to_term.get(var as usize) {
                value_to_repr.entry(val.as_str()).or_insert(*term);
            }
        }

        // Second pass: map each assigned variable to its value representative
        for (&var, val) in &self.assignments {
            if let Some(Some(term)) = self.var_to_term.get(var as usize)
                && let Some(&repr) = value_to_repr.get(val.as_str())
            {
                assignments.push((*term, repr));
            }
        }

        // Third pass: include unassigned variables as self-mappings
        for (&term, &var) in &self.term_to_var {
            if !self.assignments.contains_key(&var) {
                assignments.push((term, term));
            }
        }

        assignments
    }
}

impl StringSolver {
    /// Variables in a constraint other than regex membership or length (word
    /// equations, disequalities, prefix/suffix/contains, str<->int). Such a
    /// variable must not be assigned an arbitrary regex witness, so it is left
    /// for the honest-`Unknown` path; length is folded into the word search.
    fn regex_entangled_vars(&self) -> FxHashSet<u32> {
        let mut set: FxHashSet<u32> = FxHashSet::default();
        for eq in &self.equations {
            for atom in eq.lhs.atoms.iter().chain(eq.rhs.atoms.iter()) {
                if let StringAtom::Var(v) = atom {
                    set.insert(*v);
                }
            }
        }
        for (lhs, rhs, _origin) in self
            .diseqs
            .iter()
            .chain(self.prefixes.iter())
            .chain(self.suffixes.iter())
            .chain(self.contains.iter())
        {
            for atom in lhs.atoms.iter().chain(rhs.atoms.iter()) {
                if let StringAtom::Var(v) = atom {
                    set.insert(*v);
                }
            }
        }
        for (v, _value, _origin) in &self.str_to_int {
            set.insert(*v);
        }
        for (v, _value, _origin) in &self.int_to_str {
            set.insert(*v);
        }
        set
    }

    /// Combine the length constraints on `var` into `(lower, upper, origins)`.
    fn length_bounds(&self, var: u32) -> (usize, Option<usize>, Vec<TermId>) {
        let mut lo: i64 = 0;
        let mut hi: Option<i64> = None;
        let mut origins: Vec<TermId> = Vec::new();
        for l in &self.lengths {
            if l.var != var {
                continue;
            }
            origins.push(l.origin);
            if let Some(e) = l.equal {
                lo = lo.max(e);
                hi = Some(hi.map_or(e, |h| h.min(e)));
            }
            if let Some(low) = l.lower {
                lo = lo.max(low);
            }
            if let Some(up) = l.upper {
                hi = Some(hi.map_or(up, |h| h.min(up)));
            }
        }
        let lo = lo.max(0) as usize;
        let hi = hi.map(|h| if h < 0 { 0 } else { h as usize });
        (lo, hi, origins)
    }

    /// Build a satisfying string value for every variable constrained *only* by
    /// regex membership (and length) via a shortest-accepted-word search over
    /// the intersection of its positive memberships and complemented negatives.
    /// Returns `Some(conflict)` when a variable's combined language is provably
    /// empty (unsatisfiable memberships / empty intersection); undecidable
    /// variables are left unassigned for the honest-`Unknown` gate.
    fn construct_regex_models(&mut self) -> Option<Vec<TermId>> {
        let entangled = self.regex_entangled_vars();

        // Distinct, unassigned, non-entangled variables carrying a membership.
        let mut targets: Vec<u32> = Vec::new();
        for (v, _regex, _positive, _origin) in &self.regex_constraints {
            if !self.assignments.contains_key(v) && !entangled.contains(v) && !targets.contains(v) {
                targets.push(*v);
            }
        }

        for v in targets {
            let memberships: Vec<_> = self
                .regex_constraints
                .iter()
                .filter(|(var, ..)| *var == v)
                .map(|(_, regex, positive, origin)| (*positive, regex.clone(), *origin))
                .collect();
            let (lo, hi, len_origins) = self.length_bounds(v);
            match solve_membership(&memberships, lo, hi, &len_origins) {
                VarModel::Assign(word) => self.substitute_all(v, &word),
                VarModel::Conflict(conflict) => return Some(conflict),
                VarModel::Undecided => { /* leave unassigned -> honest Unknown */ }
            }
        }
        None
    }

    /// Get all string variable assignments.
    /// Returns (term, string_value) for variables with known assignments.
    pub fn get_assignments(&self) -> Vec<(TermId, String)> {
        let mut result = Vec::new();
        for (&var, val) in &self.assignments {
            if let Some(Some(term)) = self.var_to_term.get(var as usize) {
                result.push((*term, val.clone()));
            }
        }
        result
    }

    /// Get all interned string terms.
    pub fn get_interned_terms(&self) -> Vec<TermId> {
        self.term_to_var.keys().copied().collect()
    }

    /// Check if a term is a string variable.
    pub fn is_string_term(&self, term: TermId) -> bool {
        self.term_to_var.contains_key(&term)
    }
}

impl TheoryCombination for StringSolver {
    fn notify_equality(&mut self, eq: EqualityNotification) -> bool {
        let lhs_var = self.term_to_var.get(&eq.lhs).copied();
        let rhs_var = self.term_to_var.get(&eq.rhs).copied();

        match (lhs_var, rhs_var) {
            (Some(lhs_v), Some(rhs_v)) => {
                // Both terms are string variables.
                // When s = t, add a word equation and propagate len(s) = len(t).
                let lhs_expr = StringExpr::var(lhs_v);
                let rhs_expr = StringExpr::var(rhs_v);
                let origin = eq.reason.unwrap_or(eq.lhs);

                self.add_equality(lhs_expr, rhs_expr, origin);

                // Check if this immediately caused a conflict
                if self.current_conflict.is_some() {
                    return false;
                }

                // Propagate length equality: len(s) = len(t)
                // If both variables have known assignments, check consistency
                if let (Some(s_val), Some(t_val)) =
                    (self.assignments.get(&lhs_v), self.assignments.get(&rhs_v))
                    && s_val != t_val
                {
                    self.current_conflict = Some(vec![origin]);
                    return false;
                }

                // Derive shared equalities: when s = t, propagate len(s) = len(t)
                // as a shared equality for the arithmetic theory.
                self.shared_equalities.push(EqualityNotification {
                    lhs: eq.lhs,
                    rhs: eq.rhs,
                    reason: eq.reason,
                });

                true
            }
            (Some(_), None) | (None, Some(_)) => {
                // One term is a string variable, the other is foreign.
                // Accept and record for later processing.
                true
            }
            (None, None) => {
                // Neither term is relevant to the string theory
                false
            }
        }
    }

    fn get_shared_equalities(&self) -> Vec<EqualityNotification> {
        self.shared_equalities.clone()
    }

    fn is_relevant(&self, term: TermId) -> bool {
        self.term_to_var.contains_key(&term)
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_length_eq_violation_is_unsat() {
        // Regression (audit theories-p3): a string variable assigned "abc" with
        // len(s) = 2 asserted must be UNSAT, not silently dropped to SAT.
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let origin = TermId(77);
        let var = solver.get_or_create_var(term);

        solver.assignments.insert(var, "abc".to_string());
        solver.add_length_eq(var, 2, origin);

        let result = solver.check().expect("check must not error");
        match result {
            TheoryResult::Unsat(reasons) => {
                assert!(
                    reasons.contains(&origin),
                    "conflict must cite the length-constraint origin"
                );
            }
            other => panic!("expected Unsat for len(\"abc\") = 2, got {other:?}"),
        }
    }

    #[test]
    fn test_length_eq_satisfied_is_sat() {
        // A matching length must remain SAT.
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        solver.assignments.insert(var, "abc".to_string());
        solver.add_length_eq(var, 3, TermId(88));

        let result = solver.check().expect("check must not error");
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_length_eq_counts_chars_not_bytes() {
        // Multi-byte characters count as one each (SMT-LIB str.len semantics).
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);

        // "café" is 4 characters but 5 UTF-8 bytes.
        solver.assignments.insert(var, "café".to_string());
        solver.add_length_eq(var, 4, TermId(99));

        let result = solver.check().expect("check must not error");
        assert!(
            matches!(result, TheoryResult::Sat),
            "length must count characters, not bytes"
        );
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

        let result = solver.check().expect("test operation should succeed");
        assert!(matches!(result, TheoryResult::Unsat(_)));

        solver.pop();

        let result = solver.check().expect("test operation should succeed");
        assert!(matches!(result, TheoryResult::Sat));
    }

    // Audit regression (theories-string): `pop()` restored every piece of
    // solver state EXCEPT `shared_equalities`, so a Nelson-Oppen equality
    // derived (via `notify_equality`) inside a pushed scope survived a
    // `pop()` and kept being reported to other theories via
    // `get_shared_equalities()` even after the scope that produced it was
    // retracted.
    #[test]
    fn audit_pop_restores_shared_equalities() {
        let mut solver = StringSolver::new();

        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);
        let d = TermId(4);

        // A shared equality established BEFORE the push must survive pop.
        solver.get_or_create_var(a);
        solver.get_or_create_var(b);
        assert!(solver.notify_equality(EqualityNotification {
            lhs: a,
            rhs: b,
            reason: None,
        }));
        assert_eq!(solver.get_shared_equalities().len(), 1);

        solver.push();

        // A second shared equality established INSIDE the pushed scope
        // must NOT survive pop.
        solver.get_or_create_var(c);
        solver.get_or_create_var(d);
        assert!(solver.notify_equality(EqualityNotification {
            lhs: c,
            rhs: d,
            reason: None,
        }));
        assert_eq!(
            solver.get_shared_equalities().len(),
            2,
            "both equalities should be visible inside the pushed scope"
        );

        solver.pop();

        let remaining = solver.get_shared_equalities();
        assert_eq!(
            remaining.len(),
            1,
            "the equality derived inside the popped scope must be discarded"
        );
        assert_eq!(remaining[0].lhs, a);
        assert_eq!(remaining[0].rhs, b);
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
        let result = solver
            .assert_true(term)
            .expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
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

        let result = solver.check().expect("test operation should succeed");
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    // ------------------------------------------------------------------
    // Honesty (theories-string): `check()` must NOT report `Sat` while word
    // equations or regex constraints on unassigned variables remain
    // unresolved. Determinable equations are resolved (staying Sat/Unsat);
    // genuinely-undecided ones yield `Unknown`.
    // ------------------------------------------------------------------

    #[test]
    fn audit_resolvable_var_suffix_equation_is_sat() {
        // x ++ "!" = "hello!"  is determinable (x = "hello"); must stay Sat
        // and record the forced assignment.
        let mut solver = StringSolver::new();
        let concat = StringExpr::var(0).concat(StringExpr::literal("!"));
        solver.add_equality(concat, StringExpr::literal("hello!"), TermId(0));

        let result = solver.check().expect("check must not error");
        assert!(
            matches!(result, TheoryResult::Sat),
            "determinable word equation must remain Sat, got {result:?}"
        );
        assert_eq!(
            solver.assignments.get(&0).map(String::as_str),
            Some("hello")
        );
    }

    #[test]
    fn audit_free_vars_equal_constant_is_sat() {
        // x ++ y = "helloworld" is satisfiable (x, y occur once); must be Sat.
        let mut solver = StringSolver::new();
        let concat = StringExpr::var(0).concat(StringExpr::var(1));
        solver.add_equality(concat, StringExpr::literal("helloworld"), TermId(0));

        let result = solver.check().expect("check must not error");
        assert!(
            matches!(result, TheoryResult::Sat),
            "free-variable concatenation equal to a constant must be Sat, got {result:?}"
        );
    }

    #[test]
    fn audit_free_var_with_length_constraint_is_not_false_unsat() {
        // Regression: (str.++ x y) = "helloworld" AND (str.len x) = 5.
        // The true answer is SAT (x = "hello", y = "world"). The occur-once
        // "free variable" rule must NOT eagerly commit an arbitrary witness
        // (x = "helloworld", y = "") here, because `x` also carries a length
        // constraint that the arbitrary witness (len 10 != 5) would violate,
        // producing a false Unsat. With the fix `x` is excluded from the
        // free-variable rule, so the equation stays unresolved and `check()`
        // returns Unknown -- never Unsat.
        let mut solver = StringSolver::new();
        let concat = StringExpr::var(0).concat(StringExpr::var(1));
        solver.add_equality(concat, StringExpr::literal("helloworld"), TermId(0));
        solver.add_length_eq(0, 5, TermId(1));

        let result = solver.check().expect("check must not error");
        assert!(
            !matches!(result, TheoryResult::Unsat(_)),
            "satisfiable constraint set must not be reported Unsat; got {result:?}"
        );
        assert!(
            matches!(result, TheoryResult::Unknown),
            "incomplete procedure must report Unknown for this fragment; got {result:?}"
        );
        // The arbitrary witness must not have been committed.
        assert_ne!(
            solver.assignments.get(&0).map(String::as_str),
            Some("helloworld"),
            "must not commit the length-violating arbitrary witness for x"
        );
    }

    #[test]
    fn audit_free_var_no_constraint_still_resolves() {
        // Control: with NO extra constraint on the occur-once variables the
        // free-variable rule still fires and the constant split stays Sat.
        let mut solver = StringSolver::new();
        let concat = StringExpr::var(0).concat(StringExpr::var(1));
        solver.add_equality(concat, StringExpr::literal("helloworld"), TermId(0));

        let result = solver.check().expect("check must not error");
        assert!(
            matches!(result, TheoryResult::Sat),
            "unconstrained free-variable split must remain Sat; got {result:?}"
        );
    }

    #[test]
    fn audit_unresolved_word_equation_is_unknown() {
        // x ++ "a" = "b" ++ y : neither side is a lone variable nor an
        // all-free-variable side, and there is no constant prefix/suffix to
        // strip. This incomplete procedure cannot decide it, so `check()`
        // must return Unknown rather than a silent Sat.
        let mut solver = StringSolver::new();
        let lhs = StringExpr::var(0).concat(StringExpr::literal("a"));
        let rhs = StringExpr::literal("b").concat(StringExpr::var(1));
        solver.add_equality(lhs, rhs, TermId(0));

        let result = solver.check().expect("check must not error");
        assert!(
            matches!(result, TheoryResult::Unknown),
            "an undecided word equation must yield Unknown, not Sat; got {result:?}"
        );
    }

    #[test]
    fn audit_resolved_regex_membership_unchanged() {
        // Once the variable is assigned, a regex membership is decided as
        // before (here: satisfied => Sat).
        let mut solver = StringSolver::new();
        let term = TermId(0);
        let var = solver.get_or_create_var(term);
        solver.assignments.insert(var, "aaa".to_string());
        let regex = Regex::plus(Regex::char('a'));
        solver.add_regex_membership(var, regex, true, term);

        let result = solver.check().expect("check must not error");
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn audit_strip_common_suffix_reduces_equation() {
        let solver = StringSolver::new();
        let mut lhs = StringExpr::var(0).concat(StringExpr::literal("world"));
        let mut rhs = StringExpr::literal("hello").concat(StringExpr::literal("world"));
        solver.strip_common_suffix(&mut lhs, &mut rhs);
        assert_eq!(lhs.as_var(), Some(0));
        assert_eq!(rhs.as_const(), Some("hello"));
    }
}
