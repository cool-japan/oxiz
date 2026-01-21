//! Sequence Operations for String Theory
//!
//! Implements SMT-LIB sequence operations for strings:
//! - seq.nth: Get character at index
//! - seq.extract: Get substring
//! - seq.replace: Replace first occurrence
//! - seq.replace_all: Replace all occurrences
//! - seq.at: Get single character
//! - seq.unit: Create singleton sequence
//! - seq.indexof: Find first occurrence
//! - seq.last_indexof: Find last occurrence

use rustc_hash::FxHashMap;

/// Sequence operation result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqResult {
    /// Concrete string result
    String(String),
    /// Concrete integer result
    Integer(i64),
    /// Symbolic expression
    Symbolic(SeqExpr),
    /// Error/undefined
    Undefined,
}

/// Symbolic sequence expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeqExpr {
    /// String variable
    Var(u32),
    /// String literal
    Literal(String),
    /// Concatenation
    Concat(Vec<SeqExpr>),
    /// Extract substring: extract(s, start, len)
    Extract(Box<SeqExpr>, Box<IntExpr>, Box<IntExpr>),
    /// Replace first: replace(s, from, to)
    Replace(Box<SeqExpr>, Box<SeqExpr>, Box<SeqExpr>),
    /// Replace all: replace_all(s, from, to)
    ReplaceAll(Box<SeqExpr>, Box<SeqExpr>, Box<SeqExpr>),
    /// Replace regex: replace_re(s, regex, replacement)
    ReplaceRe(Box<SeqExpr>, RegexId, Box<SeqExpr>),
    /// Character at index: at(s, i)
    At(Box<SeqExpr>, Box<IntExpr>),
    /// Single character sequence
    Unit(Box<IntExpr>),
    /// Reverse
    Reverse(Box<SeqExpr>),
}

/// Symbolic integer expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntExpr {
    /// Integer variable
    Var(u32),
    /// Integer literal
    Literal(i64),
    /// Length of string
    Length(Box<SeqExpr>),
    /// Index of substring: indexof(s, pattern, start)
    IndexOf(Box<SeqExpr>, Box<SeqExpr>, Box<IntExpr>),
    /// Last index of substring
    LastIndexOf(Box<SeqExpr>, Box<SeqExpr>, Box<IntExpr>),
    /// Character to code: to_code(s)
    ToCode(Box<SeqExpr>),
    /// String to int: to_int(s)
    ToInt(Box<SeqExpr>),
    /// Addition
    Add(Vec<IntExpr>),
    /// Subtraction
    Sub(Box<IntExpr>, Box<IntExpr>),
}

/// Regex identifier for symbolic expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegexId(pub u32);

/// Sequence operations evaluator
#[derive(Debug)]
pub struct SeqEvaluator {
    /// String variable assignments
    string_vars: FxHashMap<u32, String>,
    /// Integer variable assignments
    int_vars: FxHashMap<u32, i64>,
    /// Evaluation cache for expressions
    cache: FxHashMap<u64, SeqResult>,
}

impl SeqEvaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            string_vars: FxHashMap::default(),
            int_vars: FxHashMap::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Set a string variable value
    pub fn set_string(&mut self, var: u32, value: String) {
        self.string_vars.insert(var, value);
    }

    /// Set an integer variable value
    pub fn set_int(&mut self, var: u32, value: i64) {
        self.int_vars.insert(var, value);
    }

    /// Get string value
    pub fn get_string(&self, var: u32) -> Option<&String> {
        self.string_vars.get(&var)
    }

    /// Get integer value
    pub fn get_int(&self, var: u32) -> Option<i64> {
        self.int_vars.get(&var).copied()
    }

    /// Evaluate a sequence expression
    pub fn eval_seq(&self, expr: &SeqExpr) -> SeqResult {
        match expr {
            SeqExpr::Var(v) => {
                if let Some(s) = self.string_vars.get(v) {
                    SeqResult::String(s.clone())
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
            SeqExpr::Literal(s) => SeqResult::String(s.clone()),
            SeqExpr::Concat(parts) => {
                let mut result = String::new();
                for part in parts {
                    match self.eval_seq(part) {
                        SeqResult::String(s) => result.push_str(&s),
                        _ => return SeqResult::Symbolic(expr.clone()),
                    }
                }
                SeqResult::String(result)
            }
            SeqExpr::Extract(s, start, len) => {
                if let (SeqResult::String(s), Some(start), Some(len)) =
                    (self.eval_seq(s), self.eval_int(start), self.eval_int(len))
                {
                    let start = start.max(0) as usize;
                    let len = len.max(0) as usize;
                    if start >= s.len() {
                        SeqResult::String(String::new())
                    } else {
                        let end = (start + len).min(s.len());
                        SeqResult::String(s[start..end].to_string())
                    }
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
            SeqExpr::Replace(s, from, to) => {
                if let (SeqResult::String(s), SeqResult::String(from), SeqResult::String(to)) =
                    (self.eval_seq(s), self.eval_seq(from), self.eval_seq(to))
                {
                    SeqResult::String(s.replacen(&from, &to, 1))
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
            SeqExpr::ReplaceAll(s, from, to) => {
                if let (SeqResult::String(s), SeqResult::String(from), SeqResult::String(to)) =
                    (self.eval_seq(s), self.eval_seq(from), self.eval_seq(to))
                {
                    SeqResult::String(s.replace(&from, &to))
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
            SeqExpr::ReplaceRe(_, _, _) => {
                // Regex replacement requires regex engine
                SeqResult::Symbolic(expr.clone())
            }
            SeqExpr::At(s, i) => {
                if let (SeqResult::String(s), Some(i)) = (self.eval_seq(s), self.eval_int(i)) {
                    if i >= 0
                        && (i as usize) < s.len()
                        && let Some(c) = s.chars().nth(i as usize)
                    {
                        return SeqResult::String(c.to_string());
                    }
                    SeqResult::String(String::new())
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
            SeqExpr::Unit(code) => {
                if let Some(code) = self.eval_int(code) {
                    if (0..=0x10FFFF).contains(&code)
                        && let Some(c) = char::from_u32(code as u32)
                    {
                        return SeqResult::String(c.to_string());
                    }
                    SeqResult::String(String::new())
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
            SeqExpr::Reverse(s) => {
                if let SeqResult::String(s) = self.eval_seq(s) {
                    SeqResult::String(s.chars().rev().collect())
                } else {
                    SeqResult::Symbolic(expr.clone())
                }
            }
        }
    }

    /// Evaluate an integer expression
    pub fn eval_int(&self, expr: &IntExpr) -> Option<i64> {
        match expr {
            IntExpr::Var(v) => self.int_vars.get(v).copied(),
            IntExpr::Literal(n) => Some(*n),
            IntExpr::Length(s) => {
                if let SeqResult::String(s) = self.eval_seq(s) {
                    Some(s.len() as i64)
                } else {
                    None
                }
            }
            IntExpr::IndexOf(haystack, needle, start) => {
                if let (SeqResult::String(h), SeqResult::String(n), Some(start)) = (
                    self.eval_seq(haystack),
                    self.eval_seq(needle),
                    self.eval_int(start),
                ) {
                    let start = start.max(0) as usize;
                    if start >= h.len() {
                        return Some(-1);
                    }
                    if n.is_empty() {
                        return Some(start as i64);
                    }
                    h[start..].find(&n).map(|i| (i + start) as i64).or(Some(-1))
                } else {
                    None
                }
            }
            IntExpr::LastIndexOf(haystack, needle, start) => {
                if let (SeqResult::String(h), SeqResult::String(n), Some(start)) = (
                    self.eval_seq(haystack),
                    self.eval_seq(needle),
                    self.eval_int(start),
                ) {
                    let start = start.max(0) as usize;
                    if n.is_empty() {
                        return Some(start.min(h.len()) as i64);
                    }
                    let search_end = start.min(h.len());
                    if search_end < n.len() {
                        return Some(-1);
                    }
                    h[..search_end].rfind(&n).map(|i| i as i64).or(Some(-1))
                } else {
                    None
                }
            }
            IntExpr::ToCode(s) => {
                if let SeqResult::String(s) = self.eval_seq(s) {
                    if s.len() == 1 {
                        s.chars().next().map(|c| c as i64)
                    } else {
                        Some(-1)
                    }
                } else {
                    None
                }
            }
            IntExpr::ToInt(s) => {
                if let SeqResult::String(s) = self.eval_seq(s) {
                    s.parse::<i64>().ok().or(Some(-1))
                } else {
                    None
                }
            }
            IntExpr::Add(terms) => {
                let mut sum = 0i64;
                for term in terms {
                    sum = sum.saturating_add(self.eval_int(term)?);
                }
                Some(sum)
            }
            IntExpr::Sub(a, b) => {
                let a = self.eval_int(a)?;
                let b = self.eval_int(b)?;
                Some(a.saturating_sub(b))
            }
        }
    }

    /// Clear all assignments
    pub fn clear(&mut self) {
        self.string_vars.clear();
        self.int_vars.clear();
        self.cache.clear();
    }
}

impl Default for SeqEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sequence Constraint Generation
// ============================================================================

/// Constraint generated from sequence operations
#[derive(Debug, Clone)]
pub enum SeqConstraint {
    /// String equality
    StringEq(SeqExpr, SeqExpr),
    /// Integer equality
    IntEq(IntExpr, IntExpr),
    /// Integer inequality
    IntLe(IntExpr, IntExpr),
    /// Integer less than
    IntLt(IntExpr, IntExpr),
    /// Non-negative constraint
    NonNeg(IntExpr),
    /// And of constraints
    And(Vec<SeqConstraint>),
    /// Or of constraints
    Or(Vec<SeqConstraint>),
    /// Implication
    Implies(Box<SeqConstraint>, Box<SeqConstraint>),
}

/// Constraint generator for sequence operations
#[derive(Debug)]
pub struct SeqConstraintGen {
    /// Generated constraints
    constraints: Vec<SeqConstraint>,
    /// Next fresh variable ID
    next_var: u32,
    /// Variable bounds (var -> (lower, upper))
    #[allow(dead_code)]
    bounds: FxHashMap<u32, (Option<i64>, Option<i64>)>,
}

impl SeqConstraintGen {
    /// Create a new constraint generator
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            next_var: 0,
            bounds: FxHashMap::default(),
        }
    }

    /// Create a fresh string variable
    pub fn fresh_string_var(&mut self) -> u32 {
        let var = self.next_var;
        self.next_var += 1;
        var
    }

    /// Create a fresh integer variable
    pub fn fresh_int_var(&mut self) -> u32 {
        let var = self.next_var;
        self.next_var += 1;
        var
    }

    /// Add a constraint
    pub fn add(&mut self, constraint: SeqConstraint) {
        self.constraints.push(constraint);
    }

    /// Generate constraints for extract operation
    /// extract(s, start, len) = result
    pub fn gen_extract(&mut self, s: &SeqExpr, start: &IntExpr, len: &IntExpr, result: &SeqExpr) {
        // Preconditions: start >= 0, len >= 0
        self.add(SeqConstraint::NonNeg(start.clone()));
        self.add(SeqConstraint::NonNeg(len.clone()));

        // Length constraint: |result| <= len
        let result_len = IntExpr::Length(Box::new(result.clone()));
        self.add(SeqConstraint::IntLe(result_len.clone(), len.clone()));

        // Length constraint: |result| <= |s| - start
        let s_len = IntExpr::Length(Box::new(s.clone()));
        let avail = IntExpr::Sub(Box::new(s_len.clone()), Box::new(start.clone()));
        self.add(SeqConstraint::IntLe(result_len.clone(), avail));

        // If start < |s|, then |result| >= min(len, |s| - start)
        // This is a complex constraint that requires case analysis
    }

    /// Generate constraints for indexof operation
    /// indexof(s, pattern, start) = result
    pub fn gen_indexof(
        &mut self,
        s: &SeqExpr,
        pattern: &SeqExpr,
        start: &IntExpr,
        result_var: u32,
    ) {
        let result = IntExpr::Var(result_var);

        // Preconditions
        self.add(SeqConstraint::NonNeg(start.clone()));

        // Result is either -1 or a valid index
        // result >= -1
        self.add(SeqConstraint::IntLe(IntExpr::Literal(-1), result.clone()));

        // If result != -1:
        // - result >= start
        // - result + |pattern| <= |s|
        // - s[result..result+|pattern|] = pattern
        let s_len = IntExpr::Length(Box::new(s.clone()));
        let pattern_len = IntExpr::Length(Box::new(pattern.clone()));

        // result + |pattern| <= |s| when result >= 0
        let end = IntExpr::Add(vec![result.clone(), pattern_len.clone()]);
        self.add(SeqConstraint::Or(vec![
            SeqConstraint::IntEq(result.clone(), IntExpr::Literal(-1)),
            SeqConstraint::IntLe(end, s_len),
        ]));
    }

    /// Generate constraints for replace operation
    /// replace(s, from, to) = result
    pub fn gen_replace(&mut self, s: &SeqExpr, from: &SeqExpr, to: &SeqExpr, result: &SeqExpr) {
        let from_len = IntExpr::Length(Box::new(from.clone()));
        let to_len = IntExpr::Length(Box::new(to.clone()));
        let s_len = IntExpr::Length(Box::new(s.clone()));
        let result_len = IntExpr::Length(Box::new(result.clone()));

        // Case 1: from not found -> result = s
        // Case 2: from found at index i -> result = s[0..i] ++ to ++ s[i+|from|..]
        // |result| = |s| - |from| + |to| (if found) or |result| = |s| (if not found)

        let found_len = IntExpr::Add(vec![
            IntExpr::Sub(Box::new(s_len.clone()), Box::new(from_len.clone())),
            to_len.clone(),
        ]);

        self.add(SeqConstraint::Or(vec![
            SeqConstraint::IntEq(result_len.clone(), s_len),
            SeqConstraint::IntEq(result_len, found_len),
        ]));
    }

    /// Get all generated constraints
    pub fn constraints(&self) -> &[SeqConstraint] {
        &self.constraints
    }

    /// Take all constraints
    pub fn take_constraints(&mut self) -> Vec<SeqConstraint> {
        std::mem::take(&mut self.constraints)
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraints.clear();
    }
}

impl Default for SeqConstraintGen {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// String Builder for Model Generation
// ============================================================================

/// String builder for generating concrete values
#[derive(Debug)]
pub struct StringBuilder {
    /// Character alphabet for random generation
    alphabet: Vec<char>,
    /// Random seed
    seed: u64,
}

impl StringBuilder {
    /// Create a new string builder with default alphabet
    pub fn new() -> Self {
        Self {
            alphabet: ('a'..='z').collect(),
            seed: 42,
        }
    }

    /// Create with custom alphabet
    pub fn with_alphabet(alphabet: Vec<char>) -> Self {
        Self { alphabet, seed: 42 }
    }

    /// Set random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Generate a random string of given length
    pub fn random_string(&mut self, len: usize) -> String {
        if self.alphabet.is_empty() {
            return String::new();
        }

        let mut result = String::with_capacity(len);
        for _ in 0..len {
            let idx = self.random() % self.alphabet.len();
            result.push(self.alphabet[idx]);
        }
        result
    }

    /// Generate string containing a pattern
    pub fn string_containing(&mut self, pattern: &str, min_len: usize) -> String {
        if pattern.is_empty() {
            return self.random_string(min_len);
        }

        let total_len = min_len.max(pattern.len());
        let prefix_len = if total_len > pattern.len() {
            self.random() % (total_len - pattern.len() + 1)
        } else {
            0
        };
        let suffix_len = total_len.saturating_sub(prefix_len + pattern.len());

        let mut result = self.random_string(prefix_len);
        result.push_str(pattern);
        result.push_str(&self.random_string(suffix_len));
        result
    }

    /// Generate string with given prefix
    pub fn string_with_prefix(&mut self, prefix: &str, total_len: usize) -> String {
        let suffix_len = total_len.saturating_sub(prefix.len());
        let mut result = prefix.to_string();
        result.push_str(&self.random_string(suffix_len));
        result
    }

    /// Generate string with given suffix
    pub fn string_with_suffix(&mut self, suffix: &str, total_len: usize) -> String {
        let prefix_len = total_len.saturating_sub(suffix.len());
        let mut result = self.random_string(prefix_len);
        result.push_str(suffix);
        result
    }

    /// Generate string NOT containing a pattern (best effort)
    pub fn string_avoiding(&mut self, pattern: &str, len: usize) -> Option<String> {
        if pattern.is_empty() {
            return None; // All strings contain empty pattern
        }

        // Try a few times to generate a string not containing pattern
        for _ in 0..100 {
            let s = self.random_string(len);
            if !s.contains(pattern) {
                return Some(s);
            }
        }

        // Fallback: construct a string that definitely doesn't contain pattern
        // Use a different character set
        let mut result = String::with_capacity(len);
        let c = if pattern.contains('_') { '-' } else { '_' };
        for _ in 0..len {
            result.push(c);
        }
        if result.contains(pattern) {
            None
        } else {
            Some(result)
        }
    }

    /// Simple xorshift random
    fn random(&mut self) -> usize {
        let mut x = self.seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed = x;
        x as usize
    }
}

impl Default for StringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sequence Rewriter
// ============================================================================

/// Rewriter for sequence expressions
#[derive(Debug)]
pub struct SeqRewriter {
    /// Rewrite rules cache
    cache: FxHashMap<u64, SeqExpr>,
}

impl SeqRewriter {
    /// Create a new rewriter
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Simplify a sequence expression
    #[allow(clippy::only_used_in_recursion)]
    pub fn simplify(&mut self, expr: SeqExpr) -> SeqExpr {
        match expr {
            SeqExpr::Concat(parts) => {
                let simplified: Vec<_> = parts.into_iter().map(|p| self.simplify(p)).collect();

                // Merge adjacent literals
                let mut result = Vec::new();
                let mut current_lit = String::new();

                for part in simplified {
                    match part {
                        SeqExpr::Literal(s) => current_lit.push_str(&s),
                        other => {
                            if !current_lit.is_empty() {
                                result.push(SeqExpr::Literal(std::mem::take(&mut current_lit)));
                            }
                            // Skip empty concats
                            if !matches!(&other, SeqExpr::Concat(v) if v.is_empty()) {
                                result.push(other);
                            }
                        }
                    }
                }

                if !current_lit.is_empty() {
                    result.push(SeqExpr::Literal(current_lit));
                }

                match result.len() {
                    0 => SeqExpr::Literal(String::new()),
                    1 => result
                        .pop()
                        .unwrap_or_else(|| SeqExpr::Literal(String::new())),
                    _ => SeqExpr::Concat(result),
                }
            }
            SeqExpr::Extract(s, start, len) => {
                let s = self.simplify(*s);

                // If s is a literal and start/len are literals, compute directly
                if let SeqExpr::Literal(s_str) = &s
                    && let (IntExpr::Literal(start_val), IntExpr::Literal(len_val)) =
                        (&*start, &*len)
                {
                    let start_idx = (*start_val).max(0) as usize;
                    let len_val = (*len_val).max(0) as usize;
                    if start_idx >= s_str.len() {
                        return SeqExpr::Literal(String::new());
                    }
                    let end_idx = (start_idx + len_val).min(s_str.len());
                    return SeqExpr::Literal(s_str[start_idx..end_idx].to_string());
                }

                SeqExpr::Extract(Box::new(s), start, len)
            }
            SeqExpr::Replace(s, from, to) => {
                let s = self.simplify(*s);
                let from = self.simplify(*from);
                let to = self.simplify(*to);

                // If all are literals, compute directly
                if let (
                    SeqExpr::Literal(s_str),
                    SeqExpr::Literal(from_str),
                    SeqExpr::Literal(to_str),
                ) = (&s, &from, &to)
                {
                    return SeqExpr::Literal(s_str.replacen(from_str, to_str, 1));
                }

                // Replace with empty from is identity
                if matches!(&from, SeqExpr::Literal(f) if f.is_empty()) {
                    return s;
                }

                SeqExpr::Replace(Box::new(s), Box::new(from), Box::new(to))
            }
            SeqExpr::ReplaceAll(s, from, to) => {
                let s = self.simplify(*s);
                let from = self.simplify(*from);
                let to = self.simplify(*to);

                if let (
                    SeqExpr::Literal(s_str),
                    SeqExpr::Literal(from_str),
                    SeqExpr::Literal(to_str),
                ) = (&s, &from, &to)
                {
                    return SeqExpr::Literal(s_str.replace(from_str, to_str));
                }

                if matches!(&from, SeqExpr::Literal(f) if f.is_empty()) {
                    return s;
                }

                SeqExpr::ReplaceAll(Box::new(s), Box::new(from), Box::new(to))
            }
            SeqExpr::At(s, i) => {
                let s = self.simplify(*s);

                if let (SeqExpr::Literal(s_str), IntExpr::Literal(i_val)) = (&s, &*i) {
                    if *i_val >= 0
                        && (*i_val as usize) < s_str.len()
                        && let Some(c) = s_str.chars().nth(*i_val as usize)
                    {
                        return SeqExpr::Literal(c.to_string());
                    }
                    return SeqExpr::Literal(String::new());
                }

                SeqExpr::At(Box::new(s), i)
            }
            SeqExpr::Unit(code) => {
                if let IntExpr::Literal(c) = &*code {
                    if *c >= 0
                        && *c <= 0x10FFFF
                        && let Some(ch) = char::from_u32(*c as u32)
                    {
                        return SeqExpr::Literal(ch.to_string());
                    }
                    return SeqExpr::Literal(String::new());
                }
                SeqExpr::Unit(code)
            }
            SeqExpr::Reverse(s) => {
                let s = self.simplify(*s);

                if let SeqExpr::Literal(s_str) = &s {
                    return SeqExpr::Literal(s_str.chars().rev().collect());
                }

                // Reverse of reverse is identity
                if let SeqExpr::Reverse(inner) = s {
                    return *inner;
                }

                SeqExpr::Reverse(Box::new(s))
            }
            other => other,
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Default for SeqRewriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_literal() {
        let eval = SeqEvaluator::new();
        let result = eval.eval_seq(&SeqExpr::Literal("hello".to_string()));
        assert_eq!(result, SeqResult::String("hello".to_string()));
    }

    #[test]
    fn test_eval_concat() {
        let eval = SeqEvaluator::new();
        let expr = SeqExpr::Concat(vec![
            SeqExpr::Literal("hello".to_string()),
            SeqExpr::Literal(" world".to_string()),
        ]);
        let result = eval.eval_seq(&expr);
        assert_eq!(result, SeqResult::String("hello world".to_string()));
    }

    #[test]
    fn test_eval_extract() {
        let eval = SeqEvaluator::new();
        let expr = SeqExpr::Extract(
            Box::new(SeqExpr::Literal("hello".to_string())),
            Box::new(IntExpr::Literal(1)),
            Box::new(IntExpr::Literal(3)),
        );
        let result = eval.eval_seq(&expr);
        assert_eq!(result, SeqResult::String("ell".to_string()));
    }

    #[test]
    fn test_eval_replace() {
        let eval = SeqEvaluator::new();
        let expr = SeqExpr::Replace(
            Box::new(SeqExpr::Literal("hello hello".to_string())),
            Box::new(SeqExpr::Literal("hello".to_string())),
            Box::new(SeqExpr::Literal("world".to_string())),
        );
        let result = eval.eval_seq(&expr);
        assert_eq!(result, SeqResult::String("world hello".to_string()));
    }

    #[test]
    fn test_eval_replace_all() {
        let eval = SeqEvaluator::new();
        let expr = SeqExpr::ReplaceAll(
            Box::new(SeqExpr::Literal("hello hello".to_string())),
            Box::new(SeqExpr::Literal("hello".to_string())),
            Box::new(SeqExpr::Literal("world".to_string())),
        );
        let result = eval.eval_seq(&expr);
        assert_eq!(result, SeqResult::String("world world".to_string()));
    }

    #[test]
    fn test_eval_at() {
        let eval = SeqEvaluator::new();
        let expr = SeqExpr::At(
            Box::new(SeqExpr::Literal("hello".to_string())),
            Box::new(IntExpr::Literal(1)),
        );
        let result = eval.eval_seq(&expr);
        assert_eq!(result, SeqResult::String("e".to_string()));
    }

    #[test]
    fn test_eval_reverse() {
        let eval = SeqEvaluator::new();
        let expr = SeqExpr::Reverse(Box::new(SeqExpr::Literal("hello".to_string())));
        let result = eval.eval_seq(&expr);
        assert_eq!(result, SeqResult::String("olleh".to_string()));
    }

    #[test]
    fn test_eval_length() {
        let eval = SeqEvaluator::new();
        let result = eval.eval_int(&IntExpr::Length(Box::new(SeqExpr::Literal(
            "hello".to_string(),
        ))));
        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_eval_indexof() {
        let eval = SeqEvaluator::new();
        let result = eval.eval_int(&IntExpr::IndexOf(
            Box::new(SeqExpr::Literal("hello world".to_string())),
            Box::new(SeqExpr::Literal("world".to_string())),
            Box::new(IntExpr::Literal(0)),
        ));
        assert_eq!(result, Some(6));
    }

    #[test]
    fn test_eval_indexof_not_found() {
        let eval = SeqEvaluator::new();
        let result = eval.eval_int(&IntExpr::IndexOf(
            Box::new(SeqExpr::Literal("hello".to_string())),
            Box::new(SeqExpr::Literal("world".to_string())),
            Box::new(IntExpr::Literal(0)),
        ));
        assert_eq!(result, Some(-1));
    }

    #[test]
    fn test_eval_to_code() {
        let eval = SeqEvaluator::new();
        let result = eval.eval_int(&IntExpr::ToCode(Box::new(SeqExpr::Literal(
            "A".to_string(),
        ))));
        assert_eq!(result, Some(65));
    }

    #[test]
    fn test_eval_unit() {
        let eval = SeqEvaluator::new();
        let result = eval.eval_seq(&SeqExpr::Unit(Box::new(IntExpr::Literal(65))));
        assert_eq!(result, SeqResult::String("A".to_string()));
    }

    #[test]
    fn test_string_builder_random() {
        let mut builder = StringBuilder::new();
        let s = builder.random_string(10);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_string_builder_containing() {
        let mut builder = StringBuilder::new();
        let s = builder.string_containing("test", 20);
        assert!(s.contains("test"));
        assert!(s.len() >= 20);
    }

    #[test]
    fn test_string_builder_prefix() {
        let mut builder = StringBuilder::new();
        let s = builder.string_with_prefix("hello", 10);
        assert!(s.starts_with("hello"));
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_string_builder_suffix() {
        let mut builder = StringBuilder::new();
        let s = builder.string_with_suffix("world", 10);
        assert!(s.ends_with("world"));
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_rewriter_concat() {
        let mut rewriter = SeqRewriter::new();
        let expr = SeqExpr::Concat(vec![
            SeqExpr::Literal("hello".to_string()),
            SeqExpr::Literal(" ".to_string()),
            SeqExpr::Literal("world".to_string()),
        ]);
        let result = rewriter.simplify(expr);
        assert_eq!(result, SeqExpr::Literal("hello world".to_string()));
    }

    #[test]
    fn test_rewriter_extract() {
        let mut rewriter = SeqRewriter::new();
        let expr = SeqExpr::Extract(
            Box::new(SeqExpr::Literal("hello".to_string())),
            Box::new(IntExpr::Literal(1)),
            Box::new(IntExpr::Literal(3)),
        );
        let result = rewriter.simplify(expr);
        assert_eq!(result, SeqExpr::Literal("ell".to_string()));
    }

    #[test]
    fn test_rewriter_reverse_reverse() {
        let mut rewriter = SeqRewriter::new();
        let expr = SeqExpr::Reverse(Box::new(SeqExpr::Reverse(Box::new(SeqExpr::Var(1)))));
        let result = rewriter.simplify(expr);
        assert_eq!(result, SeqExpr::Var(1));
    }

    #[test]
    fn test_constraint_gen() {
        let mut cgen = SeqConstraintGen::new();
        let var_id = cgen.fresh_int_var();
        cgen.gen_indexof(
            &SeqExpr::Var(0),
            &SeqExpr::Literal("test".to_string()),
            &IntExpr::Literal(0),
            var_id,
        );
        assert!(!cgen.constraints().is_empty());
    }
}
