//! String theory implementation
//!
//! Implements the theory of strings with operations and constraints.
//! Supports SMT-LIB2 string operations including:
//! - String concatenation, length, substring
//! - Character access and manipulation
//! - String comparison and containment
//! - Regular expression matching
//!
//! Reference: SMT-LIB2 String Theory specification

use crate::ast::{TermId, TermKind, TermManager};
use crate::sort::SortManager;
use rustc_hash::{FxHashMap, FxHashSet};

/// String theory axioms and lemmas
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StringAxiom {
    /// Length of concatenation: len(s1 ++ s2) = len(s1) + len(s2)
    LengthConcat {
        /// First string
        s1: TermId,
        /// Second string
        s2: TermId,
    },
    /// Length of empty string: len("") = 0
    LengthEmpty,
    /// Length is non-negative: len(s) >= 0
    LengthNonNegative {
        /// String term
        s: TermId,
    },
    /// Substring axiom: 0 <= i <= len(s) ∧ 0 <= n ⟹ len(substr(s,i,n)) <= n
    SubstringLength {
        /// String term
        s: TermId,
        /// Start index
        i: TermId,
        /// Length
        n: TermId,
    },
    /// Concatenation associativity: (s1 ++ s2) ++ s3 = s1 ++ (s2 ++ s3)
    ConcatAssoc {
        /// First string
        s1: TermId,
        /// Second string
        s2: TermId,
        /// Third string
        s3: TermId,
    },
    /// Empty string identity: "" ++ s = s ∧ s ++ "" = s
    ConcatIdentity {
        /// String term
        s: TermId,
    },
    /// Contains transitivity: contains(s, t) ∧ contains(t, u) ⟹ contains(s, u)
    ContainsTransitive {
        /// Outer string
        s: TermId,
        /// Middle string
        t: TermId,
        /// Inner string
        u: TermId,
    },
    /// Prefix implies contains: prefixof(s, t) ⟹ contains(t, s)
    PrefixContains {
        /// Prefix
        s: TermId,
        /// String
        t: TermId,
    },
    /// Suffix implies contains: suffixof(s, t) ⟹ contains(t, s)
    SuffixContains {
        /// Suffix
        s: TermId,
        /// String
        t: TermId,
    },
    /// IndexOf bounds: 0 <= indexof(s, t, i) <= len(s)
    IndexOfBounds {
        /// String to search in
        s: TermId,
        /// String to search for
        t: TermId,
        /// Start offset
        i: TermId,
    },
    /// At character bounds: 0 <= i < len(s) ⟹ len(at(s, i)) = 1
    AtBounds {
        /// String term
        s: TermId,
        /// Index
        i: TermId,
    },
}

/// String theory reasoning engine
#[derive(Debug, Clone)]
pub struct StringTheory {
    /// Tracked string terms
    strings: FxHashSet<TermId>,
    /// Concatenation terms: maps (s1, s2) to concat term
    concats: FxHashMap<(TermId, TermId), TermId>,
    /// Length terms: maps string to length term
    lengths: FxHashMap<TermId, TermId>,
    /// Substring terms: maps (string, start, len) to substring term
    substrings: FxHashMap<(TermId, TermId, TermId), TermId>,
    /// Contains terms: maps (string, substring) to contains term
    contains: FxHashMap<(TermId, TermId), TermId>,
    /// Pending axiom instantiations
    pending_axioms: Vec<StringAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<StringAxiom>,
    /// String constants (literal values)
    constants: FxHashSet<String>,
}

impl StringTheory {
    /// Create a new string theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            strings: FxHashSet::default(),
            concats: FxHashMap::default(),
            lengths: FxHashMap::default(),
            substrings: FxHashMap::default(),
            contains: FxHashMap::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            constants: FxHashSet::default(),
        }
    }

    /// Register a string term
    pub fn register_string(&mut self, string: TermId) {
        self.strings.insert(string);
    }

    /// Register a string constant
    pub fn register_constant(&mut self, value: String) {
        self.constants.insert(value);
    }

    /// Register a concatenation term
    pub fn register_concat(&mut self, concat: TermId, s1: TermId, s2: TermId) {
        self.concats.insert((s1, s2), concat);
        self.strings.insert(concat);
    }

    /// Register a length term
    pub fn register_length(&mut self, length: TermId, string: TermId) {
        self.lengths.insert(string, length);
    }

    /// Register a substring term
    pub fn register_substring(
        &mut self,
        substr: TermId,
        string: TermId,
        start: TermId,
        len: TermId,
    ) {
        self.substrings.insert((string, start, len), substr);
        self.strings.insert(substr);
    }

    /// Register a contains term
    pub fn register_contains(&mut self, contains: TermId, string: TermId, substring: TermId) {
        self.contains.insert((string, substring), contains);
    }

    /// Add a term to the theory and extract string operations
    pub fn add_term(&mut self, term: TermId, manager: &TermManager, sort_manager: &SortManager) {
        if let Some(t) = manager.get(term) {
            // Check if this is a string sort
            let is_string = if let Some(sort) = sort_manager.get(t.sort) {
                sort.is_string()
            } else {
                false
            };

            match &t.kind {
                TermKind::StringLit(s) => {
                    self.register_string(term);
                    self.register_constant(s.clone());
                    // Generate length axiom for empty string
                    if s.is_empty() {
                        self.generate_length_empty_axiom();
                    }
                }
                TermKind::StrConcat(s1, s2) => {
                    self.register_concat(term, *s1, *s2);
                    self.generate_concat_axioms(*s1, *s2);
                }
                TermKind::StrLen(s) => {
                    self.register_length(term, *s);
                    self.generate_length_axioms(*s);
                }
                TermKind::StrSubstr(s, start, len) => {
                    self.register_substring(term, *s, *start, *len);
                    self.generate_substring_axioms(*s, *start, *len);
                }
                TermKind::StrContains(s, sub) => {
                    self.register_contains(term, *s, *sub);
                    self.generate_contains_axioms(*s, *sub);
                }
                TermKind::StrPrefixOf(prefix, s) => {
                    self.generate_prefix_axioms(*prefix, *s);
                }
                TermKind::StrSuffixOf(suffix, s) => {
                    self.generate_suffix_axioms(*suffix, *s);
                }
                TermKind::StrIndexOf(s, sub, offset) => {
                    self.generate_indexof_axioms(*s, *sub, *offset);
                }
                TermKind::StrAt(s, i) => {
                    self.generate_at_axioms(*s, *i);
                }
                TermKind::Var(_) if is_string => {
                    self.register_string(term);
                }
                _ => {}
            }
        }
    }

    /// Generate axioms for concatenation
    fn generate_concat_axioms(&mut self, s1: TermId, s2: TermId) {
        // Length axiom: len(s1 ++ s2) = len(s1) + len(s2)
        let axiom = StringAxiom::LengthConcat { s1, s2 };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }

        // Identity axiom if one is empty
        let identity1 = StringAxiom::ConcatIdentity { s: s1 };
        if self.instantiated.insert(identity1.clone()) {
            self.pending_axioms.push(identity1);
        }

        let identity2 = StringAxiom::ConcatIdentity { s: s2 };
        if self.instantiated.insert(identity2.clone()) {
            self.pending_axioms.push(identity2);
        }
    }

    /// Generate axioms for length
    fn generate_length_axioms(&mut self, s: TermId) {
        // Non-negative axiom
        let axiom = StringAxiom::LengthNonNegative { s };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Generate axiom for empty string length
    fn generate_length_empty_axiom(&mut self) {
        let axiom = StringAxiom::LengthEmpty;
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Generate axioms for substring
    fn generate_substring_axioms(&mut self, s: TermId, i: TermId, n: TermId) {
        let axiom = StringAxiom::SubstringLength { s, i, n };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Generate axioms for contains
    fn generate_contains_axioms(&mut self, s: TermId, sub: TermId) {
        // Check for transitivity with other contains
        for (s2, u) in self.contains.keys() {
            if *s2 == sub {
                let axiom = StringAxiom::ContainsTransitive { s, t: sub, u: *u };
                if self.instantiated.insert(axiom.clone()) {
                    self.pending_axioms.push(axiom);
                }
            }
        }
    }

    /// Generate axioms for prefix
    fn generate_prefix_axioms(&mut self, prefix: TermId, s: TermId) {
        let axiom = StringAxiom::PrefixContains { s: prefix, t: s };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Generate axioms for suffix
    fn generate_suffix_axioms(&mut self, suffix: TermId, s: TermId) {
        let axiom = StringAxiom::SuffixContains { s: suffix, t: s };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Generate axioms for indexof
    fn generate_indexof_axioms(&mut self, s: TermId, t: TermId, i: TermId) {
        let axiom = StringAxiom::IndexOfBounds { s, t, i };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Generate axioms for at (character access)
    fn generate_at_axioms(&mut self, s: TermId, i: TermId) {
        let axiom = StringAxiom::AtBounds { s, i };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Convert a string axiom to a term
    #[allow(clippy::too_many_lines)]
    pub fn axiom_to_term(&self, axiom: &StringAxiom, manager: &mut TermManager) -> TermId {
        match axiom {
            StringAxiom::LengthConcat { s1, s2 } => {
                // len(s1 ++ s2) = len(s1) + len(s2)
                let concat = manager.mk_str_concat(*s1, *s2);
                let len_concat = manager.mk_str_len(concat);
                let len_s1 = manager.mk_str_len(*s1);
                let len_s2 = manager.mk_str_len(*s2);
                let sum = manager.mk_add([len_s1, len_s2]);
                manager.mk_eq(len_concat, sum)
            }
            StringAxiom::LengthEmpty => {
                // len("") = 0
                let empty = manager.mk_string_lit("");
                let len_empty = manager.mk_str_len(empty);
                let zero = manager.mk_int(0);
                manager.mk_eq(len_empty, zero)
            }
            StringAxiom::LengthNonNegative { s } => {
                // len(s) >= 0
                let len_s = manager.mk_str_len(*s);
                let zero = manager.mk_int(0);
                manager.mk_geq(len_s, zero)
            }
            StringAxiom::SubstringLength { s, i, n } => {
                // 0 <= i <= len(s) ∧ 0 <= n ⟹ len(substr(s,i,n)) <= n
                let zero = manager.mk_int(0);
                let len_s = manager.mk_str_len(*s);
                let substr = manager.mk_str_substr(*s, *i, *n);
                let len_substr = manager.mk_str_len(substr);

                // Premise: 0 <= i <= len(s) ∧ 0 <= n
                let i_geq_0 = manager.mk_geq(*i, zero);
                let i_leq_len = manager.mk_leq(*i, len_s);
                let n_geq_0 = manager.mk_geq(*n, zero);
                let premise = manager.mk_and([i_geq_0, i_leq_len, n_geq_0]);

                // Conclusion: len(substr(s,i,n)) <= n
                let conclusion = manager.mk_leq(len_substr, *n);

                manager.mk_implies(premise, conclusion)
            }
            StringAxiom::ConcatAssoc { s1, s2, s3 } => {
                // (s1 ++ s2) ++ s3 = s1 ++ (s2 ++ s3)
                let concat12 = manager.mk_str_concat(*s1, *s2);
                let left = manager.mk_str_concat(concat12, *s3);
                let concat23 = manager.mk_str_concat(*s2, *s3);
                let right = manager.mk_str_concat(*s1, concat23);
                manager.mk_eq(left, right)
            }
            StringAxiom::ConcatIdentity { s } => {
                // "" ++ s = s ∧ s ++ "" = s
                let empty = manager.mk_string_lit("");
                let left_concat = manager.mk_str_concat(empty, *s);
                let left_eq = manager.mk_eq(left_concat, *s);
                let right_concat = manager.mk_str_concat(*s, empty);
                let right_eq = manager.mk_eq(right_concat, *s);
                manager.mk_and([left_eq, right_eq])
            }
            StringAxiom::ContainsTransitive { s, t, u } => {
                // contains(s, t) ∧ contains(t, u) ⟹ contains(s, u)
                let contains_st = manager.mk_str_contains(*s, *t);
                let contains_tu = manager.mk_str_contains(*t, *u);
                let premise = manager.mk_and([contains_st, contains_tu]);
                let conclusion = manager.mk_str_contains(*s, *u);
                manager.mk_implies(premise, conclusion)
            }
            StringAxiom::PrefixContains { s, t } => {
                // prefixof(s, t) ⟹ contains(t, s)
                let prefix = manager.mk_str_prefixof(*s, *t);
                let contains = manager.mk_str_contains(*t, *s);
                manager.mk_implies(prefix, contains)
            }
            StringAxiom::SuffixContains { s, t } => {
                // suffixof(s, t) ⟹ contains(t, s)
                let suffix = manager.mk_str_suffixof(*s, *t);
                let contains = manager.mk_str_contains(*t, *s);
                manager.mk_implies(suffix, contains)
            }
            StringAxiom::IndexOfBounds { s, t, i } => {
                // 0 <= indexof(s, t, i) <= len(s)
                let zero = manager.mk_int(0);
                let indexof = manager.mk_str_indexof(*s, *t, *i);
                let len_s = manager.mk_str_len(*s);
                let lower = manager.mk_geq(indexof, zero);
                let upper = manager.mk_leq(indexof, len_s);
                manager.mk_and([lower, upper])
            }
            StringAxiom::AtBounds { s, i } => {
                // 0 <= i < len(s) ⟹ len(at(s, i)) = 1
                let zero = manager.mk_int(0);
                let one = manager.mk_int(1);
                let len_s = manager.mk_str_len(*s);
                let at = manager.mk_str_at(*s, *i);
                let len_at = manager.mk_str_len(at);

                // Premise: 0 <= i < len(s)
                let i_geq_0 = manager.mk_geq(*i, zero);
                let i_lt_len = manager.mk_lt(*i, len_s);
                let premise = manager.mk_and([i_geq_0, i_lt_len]);

                // Conclusion: len(at(s, i)) = 1
                let conclusion = manager.mk_eq(len_at, one);

                manager.mk_implies(premise, conclusion)
            }
        }
    }

    /// Get all pending axioms
    #[must_use]
    pub fn get_pending_axioms(&self) -> &[StringAxiom] {
        &self.pending_axioms
    }

    /// Clear pending axioms
    pub fn clear_pending(&mut self) {
        self.pending_axioms.clear();
    }

    /// Get all pending axioms as terms and clear the queue
    pub fn propagate(&mut self, manager: &mut TermManager) -> Vec<TermId> {
        let axioms: Vec<_> = self
            .pending_axioms
            .iter()
            .map(|axiom| self.axiom_to_term(axiom, manager))
            .collect();

        self.pending_axioms.clear();
        axioms
    }

    /// Get statistics about the theory
    #[must_use]
    pub fn statistics(&self) -> StringTheoryStats {
        StringTheoryStats {
            num_strings: self.strings.len(),
            num_concats: self.concats.len(),
            num_lengths: self.lengths.len(),
            num_substrings: self.substrings.len(),
            num_contains: self.contains.len(),
            num_constants: self.constants.len(),
            num_axioms_instantiated: self.instantiated.len(),
            num_pending_axioms: self.pending_axioms.len(),
        }
    }

    /// Reset the theory state
    pub fn reset(&mut self) {
        self.strings.clear();
        self.concats.clear();
        self.lengths.clear();
        self.substrings.clear();
        self.contains.clear();
        self.constants.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
    }
}

/// Statistics about string theory reasoning
#[derive(Debug, Default, Clone)]
pub struct StringTheoryStats {
    /// Number of string terms
    pub num_strings: usize,
    /// Number of concatenation operations
    pub num_concats: usize,
    /// Number of length operations
    pub num_lengths: usize,
    /// Number of substring operations
    pub num_substrings: usize,
    /// Number of contains operations
    pub num_contains: usize,
    /// Number of string constants
    pub num_constants: usize,
    /// Number of axioms instantiated
    pub num_axioms_instantiated: usize,
    /// Number of pending axioms
    pub num_pending_axioms: usize,
}

impl Default for StringTheory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_empty_theory() {
        let theory = StringTheory::new();
        assert_eq!(theory.strings.len(), 0);
        assert_eq!(theory.concats.len(), 0);
    }

    #[test]
    fn test_register_string() {
        let mut theory = StringTheory::new();
        let string = TermId(1);

        theory.register_string(string);
        assert_eq!(theory.strings.len(), 1);
        assert!(theory.strings.contains(&string));
    }

    #[test]
    fn test_register_constant() {
        let mut theory = StringTheory::new();
        theory.register_constant("hello".to_string());
        assert_eq!(theory.constants.len(), 1);
        assert!(theory.constants.contains("hello"));
    }

    #[test]
    fn test_register_concat() {
        let mut theory = StringTheory::new();
        let s1 = TermId(1);
        let s2 = TermId(2);
        let concat = TermId(3);

        theory.register_concat(concat, s1, s2);
        assert_eq!(theory.concats.len(), 1);
        assert_eq!(theory.concats.get(&(s1, s2)), Some(&concat));
    }

    #[test]
    fn test_length_axiom() {
        let mut manager = TermManager::new();
        let mut theory = StringTheory::new();

        let s1 = manager.mk_string_lit("hello");
        let s2 = manager.mk_string_lit("world");

        theory.generate_concat_axioms(s1, s2);

        assert!(!theory.pending_axioms.is_empty());
        assert!(
            theory
                .pending_axioms
                .iter()
                .any(|ax| matches!(ax, StringAxiom::LengthConcat { .. }))
        );
    }

    #[test]
    fn test_propagate() {
        let mut manager = TermManager::new();
        let mut theory = StringTheory::new();

        let s = manager.mk_string_lit("test");
        theory.generate_length_axioms(s);

        let axioms = theory.propagate(&mut manager);
        assert!(!axioms.is_empty());
        assert!(theory.pending_axioms.is_empty());
    }

    #[test]
    fn test_statistics() {
        let mut theory = StringTheory::new();

        theory.register_string(TermId(1));
        theory.register_concat(TermId(2), TermId(1), TermId(3));
        theory.register_constant("test".to_string());

        let stats = theory.statistics();
        assert_eq!(stats.num_strings, 2); // concat also registers as string
        assert_eq!(stats.num_concats, 1);
        assert_eq!(stats.num_constants, 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = StringTheory::new();

        theory.register_string(TermId(1));
        theory.register_constant("test".to_string());

        theory.reset();

        assert!(theory.strings.is_empty());
        assert!(theory.constants.is_empty());
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = StringTheory::new();

        let s = TermId(1);
        let axiom = StringAxiom::LengthNonNegative { s };

        // Add the same axiom twice
        if theory.instantiated.insert(axiom.clone()) {
            theory.pending_axioms.push(axiom.clone());
        }

        if theory.instantiated.insert(axiom) {
            theory
                .pending_axioms
                .push(StringAxiom::LengthNonNegative { s });
        }

        // Should only have one copy
        assert_eq!(theory.pending_axioms.len(), 1);
        assert_eq!(theory.instantiated.len(), 1);
    }
}
