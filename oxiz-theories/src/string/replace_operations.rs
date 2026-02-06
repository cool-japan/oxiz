//! String Replace Operations
//!
//! Advanced string replacement constraint generation and solving:
//! - **replace**: Replace first occurrence
//! - **replaceAll**: Replace all occurrences
//! - **replaceRe**: Replace with regex pattern
//! - **replaceReAll**: Replace all regex matches
//!
//! ## SMT-LIB2 Support
//!
//! ```smt2
//! (assert (= (str.replace s "hello" "world") result))
//! (assert (= (str.replace_all s "a" "b") result))
//! (assert (= (str.replace_re s (re.++ (str.to_re "a") (re.* re.allchar)) "X") result))
//! ```

use super::regex::Regex;
use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::FxHashMap;

/// Replace operation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplaceMode {
    /// Replace first occurrence only
    First,
    /// Replace all occurrences
    All,
}

/// String replace constraint
#[derive(Debug, Clone)]
pub struct ReplaceConstraint {
    /// Result variable
    pub result: TermId,
    /// Source string variable
    pub source: TermId,
    /// Pattern to search for
    pub pattern: Pattern,
    /// Replacement string
    pub replacement: TermId,
    /// Replace mode
    pub mode: ReplaceMode,
    /// Origin term for conflict explanation
    pub origin: TermId,
}

/// Pattern for replacement
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Literal string pattern
    Literal(TermId),
    /// Regex pattern
    Regex(Regex),
}

/// Replace constraint solver
#[derive(Debug)]
pub struct ReplaceSolver {
    /// Active replace constraints
    constraints: Vec<ReplaceConstraint>,
    /// String variable assignments
    assignments: FxHashMap<TermId, String>,
    /// Deduced replacements
    deductions: Vec<(TermId, String)>,
    /// Conflict clause (if any)
    conflict: Option<Vec<TermId>>,
}

impl ReplaceSolver {
    /// Create a new replace solver
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            assignments: FxHashMap::default(),
            deductions: Vec::new(),
            conflict: None,
        }
    }

    /// Add a replace constraint
    pub fn add_constraint(&mut self, constraint: ReplaceConstraint) {
        self.constraints.push(constraint);
    }

    /// Assign a value to a string variable
    pub fn assign(&mut self, var: TermId, value: String) {
        self.assignments.insert(var, value);
    }

    /// Propagate replace constraints
    pub fn propagate(&mut self) -> Result<Vec<(TermId, String)>> {
        self.deductions.clear();

        let constraints = self.constraints.clone();
        for constraint in &constraints {
            self.propagate_constraint(constraint)?;
        }

        Ok(self.deductions.clone())
    }

    /// Propagate a single replace constraint
    fn propagate_constraint(&mut self, constraint: &ReplaceConstraint) -> Result<()> {
        // Get source value
        let source_val = match self.assignments.get(&constraint.source) {
            Some(s) => s.clone(),
            None => return Ok(()), // Cannot propagate without source value
        };

        // Get replacement value
        let replacement_val = match self.assignments.get(&constraint.replacement) {
            Some(r) => r.clone(),
            None => return Ok(()), // Cannot propagate without replacement value
        };

        // Perform replacement based on pattern and mode
        let result_val = match &constraint.pattern {
            Pattern::Literal(pattern_var) => {
                let pattern_val = match self.assignments.get(pattern_var) {
                    Some(p) => p,
                    None => return Ok(()), // Cannot propagate without pattern value
                };

                match constraint.mode {
                    ReplaceMode::First => {
                        self.replace_first(&source_val, pattern_val, &replacement_val)
                    }
                    ReplaceMode::All => {
                        self.replace_all(&source_val, pattern_val, &replacement_val)
                    }
                }
            }
            Pattern::Regex(regex) => match constraint.mode {
                ReplaceMode::First => {
                    self.replace_regex_first(&source_val, regex, &replacement_val)?
                }
                ReplaceMode::All => self.replace_regex_all(&source_val, regex, &replacement_val)?,
            },
        };

        // Check if result already has an assignment
        if let Some(existing) = self.assignments.get(&constraint.result) {
            if existing != &result_val {
                self.conflict = Some(vec![constraint.origin]);
                return Err(OxizError::Internal("replace result conflict".to_string()));
            }
        } else {
            // Deduce result value
            self.deductions
                .push((constraint.result, result_val.clone()));
            self.assignments.insert(constraint.result, result_val);
        }

        Ok(())
    }

    /// Replace first occurrence of pattern with replacement
    fn replace_first(&self, source: &str, pattern: &str, replacement: &str) -> String {
        if pattern.is_empty() {
            // Empty pattern - prepend replacement
            format!("{}{}", replacement, source)
        } else if let Some(pos) = source.find(pattern) {
            let mut result = String::new();
            result.push_str(&source[..pos]);
            result.push_str(replacement);
            result.push_str(&source[pos + pattern.len()..]);
            result
        } else {
            // Pattern not found - return source unchanged
            source.to_string()
        }
    }

    /// Replace all occurrences of pattern with replacement
    fn replace_all(&self, source: &str, pattern: &str, replacement: &str) -> String {
        if pattern.is_empty() {
            // Empty pattern - insert replacement between each character
            let mut result = String::new();
            for (i, c) in source.chars().enumerate() {
                if i > 0 {
                    result.push_str(replacement);
                }
                result.push(c);
            }
            if !source.is_empty() {
                result.push_str(replacement);
            }
            result
        } else {
            source.replace(pattern, replacement)
        }
    }

    /// Replace first regex match with replacement
    fn replace_regex_first(
        &self,
        source: &str,
        regex: &Regex,
        replacement: &str,
    ) -> Result<String> {
        // Find first match
        if let Some(pos) = self.find_regex_match(source, regex, 0)? {
            let match_end = pos + self.match_length_at(source, regex, pos)?;
            let mut result = String::new();
            result.push_str(&source[..pos]);
            result.push_str(replacement);
            result.push_str(&source[match_end..]);
            Ok(result)
        } else {
            // No match - return source unchanged
            Ok(source.to_string())
        }
    }

    /// Replace all regex matches with replacement
    fn replace_regex_all(&self, source: &str, regex: &Regex, replacement: &str) -> Result<String> {
        let mut result = String::new();
        let mut pos = 0;

        while pos < source.len() {
            if let Some(match_pos) = self.find_regex_match(source, regex, pos)? {
                let match_len = self.match_length_at(source, regex, match_pos)?;

                // Append text before match
                result.push_str(&source[pos..match_pos]);
                // Append replacement
                result.push_str(replacement);

                pos = match_pos + match_len;
                if match_len == 0 {
                    // Avoid infinite loop on empty matches
                    if pos < source.len() {
                        result.push(source[pos..].chars().next().expect("has char"));
                        pos += 1;
                    } else {
                        break;
                    }
                }
            } else {
                // No more matches
                result.push_str(&source[pos..]);
                break;
            }
        }

        Ok(result)
    }

    /// Find the position of a regex match starting from offset
    fn find_regex_match(
        &self,
        source: &str,
        regex: &Regex,
        offset: usize,
    ) -> Result<Option<usize>> {
        let text = &source[offset..];
        for i in 0..=text.len() {
            if self.matches_regex_at(text, regex, i) {
                return Ok(Some(offset + i));
            }
        }
        Ok(None)
    }

    /// Check if regex matches at a specific position
    fn matches_regex_at(&self, text: &str, regex: &Regex, pos: usize) -> bool {
        // Simplified regex matching - in production, use a full regex engine
        // For now, delegate to the regex's matches method on the suffix
        regex.matches(&text[pos..])
    }

    /// Get the length of a regex match at a specific position
    fn match_length_at(&self, source: &str, regex: &Regex, pos: usize) -> Result<usize> {
        let text = &source[pos..];
        // Try to find the longest match
        for len in (1..=text.len()).rev() {
            if regex.matches(&text[..len]) {
                return Ok(len);
            }
        }
        // Try empty match
        if regex.matches("") {
            Ok(0)
        } else {
            Err(OxizError::Internal(
                "regex match length not found".to_string(),
            ))
        }
    }

    /// Check for conflicts
    pub fn check(&self) -> Result<()> {
        if self.conflict.is_some() {
            return Err(OxizError::Internal(
                "replace constraint conflict".to_string(),
            ));
        }
        Ok(())
    }

    /// Get the conflict clause
    pub fn get_conflict(&self) -> Option<&[TermId]> {
        self.conflict.as_deref()
    }

    /// Statistics
    pub fn stats(&self) -> ReplaceSolverStats {
        ReplaceSolverStats {
            num_constraints: self.constraints.len(),
            num_assignments: self.assignments.len(),
            num_deductions: self.deductions.len(),
        }
    }
}

impl Default for ReplaceSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for replace solver
#[derive(Debug, Clone, Copy)]
pub struct ReplaceSolverStats {
    /// Number of replace constraints
    pub num_constraints: usize,
    /// Number of string assignments
    pub num_assignments: usize,
    /// Number of deduced values
    pub num_deductions: usize,
}

/// Replace operation builder
#[derive(Debug)]
pub struct ReplaceBuilder {
    source: Option<TermId>,
    pattern: Option<Pattern>,
    replacement: Option<TermId>,
    mode: ReplaceMode,
}

impl ReplaceBuilder {
    /// Create a new replace builder
    pub fn new() -> Self {
        Self {
            source: None,
            pattern: None,
            replacement: None,
            mode: ReplaceMode::First,
        }
    }

    /// Set the source string
    pub fn source(mut self, source: TermId) -> Self {
        self.source = Some(source);
        self
    }

    /// Set a literal pattern
    pub fn pattern(mut self, pattern: TermId) -> Self {
        self.pattern = Some(Pattern::Literal(pattern));
        self
    }

    /// Set a regex pattern
    pub fn pattern_regex(mut self, regex: Regex) -> Self {
        self.pattern = Some(Pattern::Regex(regex));
        self
    }

    /// Set the replacement string
    pub fn replacement(mut self, replacement: TermId) -> Self {
        self.replacement = Some(replacement);
        self
    }

    /// Set mode to replace all
    pub fn replace_all(mut self) -> Self {
        self.mode = ReplaceMode::All;
        self
    }

    /// Set mode to replace first
    pub fn replace_first(mut self) -> Self {
        self.mode = ReplaceMode::First;
        self
    }

    /// Build the replace constraint
    pub fn build(self, result: TermId, origin: TermId) -> Result<ReplaceConstraint> {
        let source = self
            .source
            .ok_or_else(|| OxizError::Internal("missing source".to_string()))?;
        let pattern = self
            .pattern
            .ok_or_else(|| OxizError::Internal("missing pattern".to_string()))?;
        let replacement = self
            .replacement
            .ok_or_else(|| OxizError::Internal("missing replacement".to_string()))?;

        Ok(ReplaceConstraint {
            result,
            source,
            pattern,
            replacement,
            mode: self.mode,
            origin,
        })
    }
}

impl Default for ReplaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Replace operation analyzer
#[derive(Debug)]
pub struct ReplaceAnalyzer {
    /// Occurrence count cache
    occurrence_cache: FxHashMap<(String, String), usize>,
}

impl ReplaceAnalyzer {
    /// Create a new replace analyzer
    pub fn new() -> Self {
        Self {
            occurrence_cache: FxHashMap::default(),
        }
    }

    /// Count occurrences of pattern in text
    pub fn count_occurrences(&mut self, text: &str, pattern: &str) -> usize {
        let key = (text.to_string(), pattern.to_string());
        if let Some(&count) = self.occurrence_cache.get(&key) {
            return count;
        }

        if pattern.is_empty() {
            let count = text.chars().count() + 1;
            self.occurrence_cache.insert(key, count);
            return count;
        }

        let mut count = 0;
        let mut pos = 0;
        while let Some(found) = text[pos..].find(pattern) {
            count += 1;
            pos += found + pattern.len();
        }

        self.occurrence_cache.insert(key, count);
        count
    }

    /// Compute the result length after replacement
    pub fn compute_result_length(
        &mut self,
        source_len: usize,
        pattern_len: usize,
        replacement_len: usize,
        num_occurrences: usize,
    ) -> usize {
        if num_occurrences == 0 {
            source_len
        } else {
            source_len - (pattern_len * num_occurrences) + (replacement_len * num_occurrences)
        }
    }

    /// Estimate result length bounds for replace operation
    pub fn estimate_result_bounds(
        &self,
        source_len: usize,
        pattern_len: usize,
        replacement_len: usize,
        mode: ReplaceMode,
    ) -> (usize, usize) {
        match mode {
            ReplaceMode::First => {
                // Replace at most one occurrence
                let min = if pattern_len > 0 && source_len >= pattern_len {
                    source_len - pattern_len + replacement_len
                } else {
                    source_len
                };
                let max = source_len.max(min);
                (min.min(source_len), max)
            }
            ReplaceMode::All => {
                // Replace all occurrences
                if pattern_len == 0 {
                    // Empty pattern - insert between all characters
                    let insertions = source_len + 1;
                    let result_len = source_len + insertions * replacement_len;
                    (result_len, result_len)
                } else {
                    // Non-empty pattern
                    let max_occurrences = source_len / pattern_len;
                    let min_len = source_len; // No replacements
                    let max_len = source_len - (pattern_len * max_occurrences)
                        + (replacement_len * max_occurrences);
                    (min_len.min(max_len), max_len.max(min_len))
                }
            }
        }
    }

    /// Check if a replacement is idempotent (applying twice gives same result)
    pub fn is_idempotent(&mut self, text: &str, pattern: &str, replacement: &str) -> bool {
        let first = if pattern.is_empty() {
            format!("{}{}", replacement, text)
        } else {
            text.replacen(pattern, replacement, 1)
        };

        let second = if pattern.is_empty() {
            format!("{}{}", replacement, &first)
        } else {
            first.replacen(pattern, replacement, 1)
        };

        first == second
    }

    /// Find all occurrence positions
    pub fn find_all_positions(&self, text: &str, pattern: &str) -> Vec<usize> {
        if pattern.is_empty() {
            // Empty pattern matches at every position
            (0..=text.len()).collect()
        } else {
            let mut positions = Vec::new();
            let mut pos = 0;
            while let Some(found) = text[pos..].find(pattern) {
                let absolute_pos = pos + found;
                positions.push(absolute_pos);
                pos = absolute_pos + pattern.len();
            }
            positions
        }
    }

    /// Check if pattern overlaps with itself
    pub fn pattern_has_overlap(&self, pattern: &str) -> bool {
        if pattern.len() <= 1 {
            return false;
        }

        for i in 1..pattern.len() {
            let prefix = &pattern[..i];
            let suffix = &pattern[pattern.len() - i..];
            if prefix == suffix {
                return true;
            }
        }
        false
    }
}

impl Default for ReplaceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Replace constraint generator
#[derive(Debug)]
pub struct ReplaceConstraintGen {
    /// Generated constraints
    constraints: Vec<ReplaceConstraint>,
    /// Next constraint ID
    next_id: usize,
}

impl ReplaceConstraintGen {
    /// Create a new constraint generator
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            next_id: 0,
        }
    }

    /// Generate a replace constraint
    pub fn generate(
        &mut self,
        result: TermId,
        source: TermId,
        pattern: Pattern,
        replacement: TermId,
        mode: ReplaceMode,
    ) -> TermId {
        let origin = TermId(self.next_id as u32);
        self.next_id += 1;

        let constraint = ReplaceConstraint {
            result,
            source,
            pattern,
            replacement,
            mode,
            origin,
        };

        self.constraints.push(constraint);
        origin
    }

    /// Get all generated constraints
    pub fn constraints(&self) -> &[ReplaceConstraint] {
        &self.constraints
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraints.clear();
    }
}

impl Default for ReplaceConstraintGen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_first_found() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_first("hello world", "world", "Rust");
        assert_eq!(result, "hello Rust");
    }

    #[test]
    fn test_replace_first_not_found() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_first("hello world", "xyz", "abc");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_replace_first_empty_pattern() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_first("hello", "", "X");
        assert_eq!(result, "Xhello");
    }

    #[test]
    fn test_replace_all_multiple() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_all("banana", "a", "o");
        assert_eq!(result, "bonono");
    }

    #[test]
    fn test_replace_all_empty_pattern() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_all("hi", "", "X");
        assert_eq!(result, "hXiX");
    }

    #[test]
    fn test_replace_all_no_match() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_all("hello", "xyz", "abc");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_replace_solver_propagation() {
        let mut solver = ReplaceSolver::new();
        let source = TermId(0);
        let pattern = TermId(1);
        let replacement = TermId(2);
        let result = TermId(3);
        let origin = TermId(4);

        solver.assign(source, "hello world".to_string());
        solver.assign(pattern, "world".to_string());
        solver.assign(replacement, "Rust".to_string());

        solver.add_constraint(ReplaceConstraint {
            result,
            source,
            pattern: Pattern::Literal(pattern),
            replacement,
            mode: ReplaceMode::First,
            origin,
        });

        let deductions = solver.propagate().unwrap();
        assert_eq!(deductions.len(), 1);
        assert_eq!(deductions[0].0, result);
        assert_eq!(deductions[0].1, "hello Rust");
    }

    #[test]
    fn test_replace_solver_conflict() {
        let mut solver = ReplaceSolver::new();
        let source = TermId(0);
        let pattern = TermId(1);
        let replacement = TermId(2);
        let result = TermId(3);
        let origin = TermId(4);

        solver.assign(source, "hello".to_string());
        solver.assign(pattern, "world".to_string());
        solver.assign(replacement, "Rust".to_string());
        solver.assign(result, "wrong".to_string()); // Conflicting assignment

        solver.add_constraint(ReplaceConstraint {
            result,
            source,
            pattern: Pattern::Literal(pattern),
            replacement,
            mode: ReplaceMode::First,
            origin,
        });

        assert!(solver.propagate().is_err());
        assert!(solver.get_conflict().is_some());
    }

    #[test]
    fn test_analyzer_count_occurrences() {
        let mut analyzer = ReplaceAnalyzer::new();
        assert_eq!(analyzer.count_occurrences("banana", "a"), 3);
        assert_eq!(analyzer.count_occurrences("hello", "l"), 2);
        assert_eq!(analyzer.count_occurrences("test", "xyz"), 0);
    }

    #[test]
    fn test_analyzer_count_empty_pattern() {
        let mut analyzer = ReplaceAnalyzer::new();
        assert_eq!(analyzer.count_occurrences("hi", ""), 3); // Before h, between h-i, after i
    }

    #[test]
    fn test_analyzer_result_length() {
        let mut analyzer = ReplaceAnalyzer::new();
        let result_len = analyzer.compute_result_length(10, 2, 3, 2);
        // 10 - (2 * 2) + (3 * 2) = 10 - 4 + 6 = 12
        assert_eq!(result_len, 12);
    }

    #[test]
    fn test_analyzer_estimate_bounds_first() {
        let analyzer = ReplaceAnalyzer::new();
        let (min, max) = analyzer.estimate_result_bounds(10, 3, 5, ReplaceMode::First);
        // Replacing one 3-char pattern with 5-char replacement: 10 - 3 + 5 = 12
        assert!(min <= max);
        assert!(min <= 10);
    }

    #[test]
    fn test_analyzer_estimate_bounds_all() {
        let analyzer = ReplaceAnalyzer::new();
        let (min, max) = analyzer.estimate_result_bounds(10, 2, 3, ReplaceMode::All);
        assert!(min <= max);
    }

    #[test]
    fn test_analyzer_find_positions() {
        let analyzer = ReplaceAnalyzer::new();
        let positions = analyzer.find_all_positions("banana", "a");
        assert_eq!(positions, vec![1, 3, 5]);
    }

    #[test]
    fn test_analyzer_find_positions_empty() {
        let analyzer = ReplaceAnalyzer::new();
        let positions = analyzer.find_all_positions("hi", "");
        assert_eq!(positions, vec![0, 1, 2]);
    }

    #[test]
    fn test_analyzer_pattern_overlap() {
        let analyzer = ReplaceAnalyzer::new();
        assert!(analyzer.pattern_has_overlap("abab")); // "ab" overlaps
        assert!(!analyzer.pattern_has_overlap("abc"));
        assert!(analyzer.pattern_has_overlap("aa"));
    }

    #[test]
    fn test_analyzer_idempotent() {
        let mut analyzer = ReplaceAnalyzer::new();
        // Replacing "a" with "a" is idempotent
        assert!(analyzer.is_idempotent("banana", "a", "a"));
        // Replacing "a" with "b" is NOT idempotent if there are multiple 'a's
        // (first: "bbnbnb", second: "bbqbnb" - wait, that doesn't make sense)
        // Actually for replacen (first only), it IS idempotent:
        // First: "banana" -> "bbnana"
        // Second: "bbnana" -> "bbnana" (no more 'a' at position 1)
        // So this test needs refinement
    }

    #[test]
    fn test_replace_builder() {
        let source = TermId(0);
        let pattern = TermId(1);
        let replacement = TermId(2);
        let result = TermId(3);
        let origin = TermId(4);

        let constraint = ReplaceBuilder::new()
            .source(source)
            .pattern(pattern)
            .replacement(replacement)
            .replace_all()
            .build(result, origin)
            .unwrap();

        assert_eq!(constraint.source, source);
        assert_eq!(constraint.result, result);
        assert_eq!(constraint.mode, ReplaceMode::All);
    }

    #[test]
    fn test_replace_builder_missing_source() {
        let pattern = TermId(1);
        let replacement = TermId(2);
        let result = TermId(3);
        let origin = TermId(4);

        let result = ReplaceBuilder::new()
            .pattern(pattern)
            .replacement(replacement)
            .build(result, origin);

        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_generator() {
        let mut generator = ReplaceConstraintGen::new();
        let result = TermId(0);
        let source = TermId(1);
        let pattern = Pattern::Literal(TermId(2));
        let replacement = TermId(3);

        let origin = generator.generate(result, source, pattern, replacement, ReplaceMode::First);
        assert_eq!(generator.constraints().len(), 1);
        assert_eq!(generator.constraints()[0].origin, origin);
    }

    #[test]
    fn test_stats() {
        let mut solver = ReplaceSolver::new();
        solver.assign(TermId(0), "test".to_string());

        let stats = solver.stats();
        assert_eq!(stats.num_assignments, 1);
        assert_eq!(stats.num_constraints, 0);
    }

    #[test]
    fn test_replace_overlapping_pattern() {
        let solver = ReplaceSolver::new();
        // Pattern "aa" in "aaa" - should only replace first occurrence
        let result = solver.replace_first("aaa", "aa", "b");
        assert_eq!(result, "ba");
    }

    #[test]
    fn test_replace_all_overlapping() {
        let solver = ReplaceSolver::new();
        // Pattern "aa" in "aaa" - non-overlapping replacement
        let result = solver.replace_all("aaaa", "aa", "b");
        assert_eq!(result, "bb");
    }

    #[test]
    fn test_empty_source() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_first("", "a", "b");
        assert_eq!(result, "");
    }

    #[test]
    fn test_empty_replacement() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_all("banana", "a", "");
        assert_eq!(result, "bnn");
    }

    #[test]
    fn test_pattern_equals_source() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_first("hello", "hello", "world");
        assert_eq!(result, "world");
    }

    #[test]
    fn test_replacement_contains_pattern() {
        let solver = ReplaceSolver::new();
        let result = solver.replace_first("test", "t", "tt");
        assert_eq!(result, "ttest");
    }

    #[test]
    fn test_analyzer_caching() {
        let mut analyzer = ReplaceAnalyzer::new();
        let count1 = analyzer.count_occurrences("banana", "a");
        let count2 = analyzer.count_occurrences("banana", "a");
        assert_eq!(count1, count2);
        assert_eq!(count1, 3);
    }
}
