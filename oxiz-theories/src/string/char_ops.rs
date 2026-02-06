//! Character-level String Operations
//!
//! This module implements fine-grained character operations for string theory solving:
//! - Character-at-position (`str.at`, `str.nth`)
//! - Character code operations (`str.to_code`, `str.from_code`)
//! - Character range constraints
//! - Character class membership
//! - ASCII/Unicode character classification

use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Represents a character at a specific position in a string
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharAt {
    /// The string variable
    pub string: TermId,
    /// The position (index) in the string
    pub position: TermId,
    /// The resulting character
    pub character: TermId,
}

/// Character code point (Unicode scalar value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CodePoint(pub u32);

impl CodePoint {
    /// Create a new code point, validating it's a valid Unicode scalar
    pub fn new(value: u32) -> Option<Self> {
        if value <= 0x10FFFF && !(0xD800..=0xDFFF).contains(&value) {
            Some(CodePoint(value))
        } else {
            None
        }
    }

    /// ASCII range check
    pub fn is_ascii(&self) -> bool {
        self.0 <= 0x7F
    }

    /// Check if alphanumeric
    pub fn is_alphanumeric(&self) -> bool {
        matches!(self.0, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A)
    }

    /// Check if digit
    pub fn is_digit(&self) -> bool {
        matches!(self.0, 0x30..=0x39)
    }

    /// Check if alphabetic
    pub fn is_alphabetic(&self) -> bool {
        matches!(self.0, 0x41..=0x5A | 0x61..=0x7A)
    }

    /// Check if whitespace
    pub fn is_whitespace(&self) -> bool {
        matches!(self.0, 0x20 | 0x09 | 0x0A | 0x0D)
    }

    /// Convert to character if valid
    pub fn to_char(&self) -> Option<char> {
        char::from_u32(self.0)
    }
}

/// Character class for pattern matching
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharClass {
    /// Single character
    Single(CodePoint),
    /// Range of characters [start, end] inclusive
    Range(CodePoint, CodePoint),
    /// Set of discrete characters
    Set(FxHashSet<CodePoint>),
    /// Predefined class (digit, alpha, whitespace, etc.)
    Predefined(PredefinedClass),
    /// Negation of a class
    Negated(Box<CharClass>),
    /// Union of multiple classes
    Union(Vec<CharClass>),
    /// Intersection of multiple classes
    Intersection(Vec<CharClass>),
}

/// Predefined character classes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredefinedClass {
    /// Digits [0-9]
    Digit,
    /// Letters [A-Za-z]
    Alpha,
    /// Alphanumeric [A-Za-z0-9]
    Alnum,
    /// Whitespace [ \t\n\r]
    Whitespace,
    /// ASCII printable [32-126]
    Print,
    /// ASCII control characters [0-31, 127]
    Control,
    /// Any character (.)
    Any,
}

impl CharClass {
    /// Check if a code point matches this character class
    pub fn matches(&self, cp: CodePoint) -> bool {
        match self {
            CharClass::Single(c) => *c == cp,
            CharClass::Range(start, end) => cp >= *start && cp <= *end,
            CharClass::Set(set) => set.contains(&cp),
            CharClass::Predefined(cls) => cls.matches(cp),
            CharClass::Negated(inner) => !inner.matches(cp),
            CharClass::Union(classes) => classes.iter().any(|c| c.matches(cp)),
            CharClass::Intersection(classes) => classes.iter().all(|c| c.matches(cp)),
        }
    }

    /// Compute the complement of this character class
    pub fn complement(&self) -> CharClass {
        match self {
            CharClass::Negated(inner) => (**inner).clone(),
            other => CharClass::Negated(Box::new(other.clone())),
        }
    }

    /// Simplify the character class by normalizing structure
    pub fn simplify(&self) -> CharClass {
        match self {
            CharClass::Negated(inner) => {
                if let CharClass::Negated(inner2) = inner.as_ref() {
                    // Double negation
                    inner2.simplify()
                } else {
                    CharClass::Negated(Box::new(inner.simplify()))
                }
            }
            CharClass::Union(classes) => {
                let simplified: Vec<_> = classes.iter().map(|c| c.simplify()).collect();
                // Flatten nested unions
                let mut flat = Vec::new();
                for cls in simplified {
                    if let CharClass::Union(inner) = cls {
                        flat.extend(inner);
                    } else {
                        flat.push(cls);
                    }
                }
                if flat.len() == 1 {
                    flat.into_iter().next().expect("exactly one element")
                } else {
                    CharClass::Union(flat)
                }
            }
            CharClass::Intersection(classes) => {
                let simplified: Vec<_> = classes.iter().map(|c| c.simplify()).collect();
                // Flatten nested intersections
                let mut flat = Vec::new();
                for cls in simplified {
                    if let CharClass::Intersection(inner) = cls {
                        flat.extend(inner);
                    } else {
                        flat.push(cls);
                    }
                }
                if flat.len() == 1 {
                    flat.into_iter().next().expect("exactly one element")
                } else {
                    CharClass::Intersection(flat)
                }
            }
            other => other.clone(),
        }
    }
}

impl PredefinedClass {
    /// Check if a code point matches this predefined class
    pub fn matches(&self, cp: CodePoint) -> bool {
        match self {
            PredefinedClass::Digit => cp.is_digit(),
            PredefinedClass::Alpha => cp.is_alphabetic(),
            PredefinedClass::Alnum => cp.is_alphanumeric(),
            PredefinedClass::Whitespace => cp.is_whitespace(),
            PredefinedClass::Print => matches!(cp.0, 32..=126),
            PredefinedClass::Control => cp.0 <= 31 || cp.0 == 127,
            PredefinedClass::Any => true,
        }
    }
}

/// Character-level constraint solver
#[derive(Debug)]
pub struct CharOpSolver {
    /// Character-at constraints: (char, string, position)
    char_at_constraints: Vec<CharAt>,
    /// Character code conversions: char -> code
    char_to_code: FxHashMap<TermId, TermId>,
    /// Code to character conversions: code -> char
    code_to_char: FxHashMap<TermId, TermId>,
    /// Character class memberships: (char, class_id)
    class_memberships: Vec<(TermId, usize)>,
    /// Character class definitions
    char_classes: Vec<CharClass>,
    /// Deduced character values
    char_values: FxHashMap<TermId, CodePoint>,
    /// Deduced string lengths
    string_lengths: FxHashMap<TermId, usize>,
    /// Propagation queue
    propagation_queue: VecDeque<TermId>,
}

impl CharOpSolver {
    /// Create a new character operation solver
    pub fn new() -> Self {
        Self {
            char_at_constraints: Vec::new(),
            char_to_code: FxHashMap::default(),
            code_to_char: FxHashMap::default(),
            class_memberships: Vec::new(),
            char_classes: Vec::new(),
            char_values: FxHashMap::default(),
            string_lengths: FxHashMap::default(),
            propagation_queue: VecDeque::new(),
        }
    }

    /// Add a character-at constraint: char = string\[position\]
    pub fn add_char_at(&mut self, constraint: CharAt) -> Result<()> {
        self.char_at_constraints.push(constraint);
        self.propagation_queue.push_back(constraint.character);
        Ok(())
    }

    /// Add a character-to-code conversion: code = to_code(char)
    pub fn add_char_to_code(&mut self, char: TermId, code: TermId) -> Result<()> {
        if let Some(&existing) = self.char_to_code.get(&char)
            && existing != code
        {
            return Err(OxizError::Internal(
                "conflicting char-to-code conversions".to_string(),
            ));
        }
        self.char_to_code.insert(char, code);
        self.code_to_char.insert(code, char);
        self.propagation_queue.push_back(char);
        Ok(())
    }

    /// Add a character class membership constraint
    pub fn add_class_membership(&mut self, char: TermId, class: CharClass) -> Result<()> {
        let class_id = self.char_classes.len();
        self.char_classes.push(class);
        self.class_memberships.push((char, class_id));
        self.propagation_queue.push_back(char);
        Ok(())
    }

    /// Set a known character value
    pub fn set_char_value(&mut self, char: TermId, value: CodePoint) -> Result<()> {
        if let Some(&existing) = self.char_values.get(&char)
            && existing != value
        {
            return Err(OxizError::Internal(
                "conflicting character values".to_string(),
            ));
        }
        self.char_values.insert(char, value);
        self.propagation_queue.push_back(char);
        Ok(())
    }

    /// Set a known string length
    pub fn set_string_length(&mut self, string: TermId, length: usize) {
        self.string_lengths.insert(string, length);
    }

    /// Propagate character constraints
    pub fn propagate(&mut self) -> Result<Vec<(TermId, CodePoint)>> {
        let mut deductions = Vec::new();

        while let Some(term) = self.propagation_queue.pop_front() {
            // Propagate through char-to-code conversions
            if let Some(&code_term) = self.char_to_code.get(&term)
                && let Some(&char_value) = self.char_values.get(&term)
            {
                // We know the character, deduce the code
                deductions.push((code_term, char_value));
            }

            // Propagate through code-to-char conversions
            if let Some(&char_term) = self.code_to_char.get(&term) {
                // If term is a code and we have its value, deduce the character
                if let Some(&code_value) = self.char_values.get(&term)
                    && let Some(_c) = code_value.to_char()
                {
                    self.char_values.insert(char_term, code_value);
                    deductions.push((char_term, code_value));
                }
            }

            // Check character class memberships
            for &(char, class_id) in &self.class_memberships {
                if char == term
                    && let Some(&value) = self.char_values.get(&char)
                {
                    let class = &self.char_classes[class_id];
                    if !class.matches(value) {
                        return Err(OxizError::Internal(
                            "character does not match required class".to_string(),
                        ));
                    }
                }
            }
        }

        Ok(deductions)
    }

    /// Check for conflicts in character-at constraints
    pub fn check_conflicts(&self) -> Result<()> {
        for _constraint in &self.char_at_constraints {
            // Check if we have concrete values
            // (This is simplified - in practice, we'd check position bounds)
            // In a full implementation, we'd check position term bounds
        }
        Ok(())
    }

    /// Get all deduced character values
    pub fn get_char_values(&self) -> &FxHashMap<TermId, CodePoint> {
        &self.char_values
    }

    /// Statistics
    pub fn stats(&self) -> CharOpStats {
        CharOpStats {
            num_char_at: self.char_at_constraints.len(),
            num_conversions: self.char_to_code.len(),
            num_class_checks: self.class_memberships.len(),
            num_deduced: self.char_values.len(),
        }
    }
}

/// Statistics for character operations
#[derive(Debug, Clone, Copy)]
pub struct CharOpStats {
    /// Number of char-at constraints
    pub num_char_at: usize,
    /// Number of char/code conversions
    pub num_conversions: usize,
    /// Number of class membership checks
    pub num_class_checks: usize,
    /// Number of deduced character values
    pub num_deduced: usize,
}

impl Default for CharOpSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_point_creation() {
        assert!(CodePoint::new(b'A' as u32).is_some());
        assert!(CodePoint::new(0x10FFFF).is_some());
        assert!(CodePoint::new(0x110000).is_none()); // Out of range
        assert!(CodePoint::new(0xD800).is_none()); // Surrogate
    }

    #[test]
    fn test_code_point_properties() {
        let a = CodePoint::new(b'A' as u32).expect("valid");
        assert!(a.is_ascii());
        assert!(a.is_alphabetic());
        assert!(a.is_alphanumeric());
        assert!(!a.is_digit());

        let nine = CodePoint::new(b'9' as u32).expect("valid");
        assert!(nine.is_digit());
        assert!(nine.is_alphanumeric());
    }

    #[test]
    fn test_char_class_single() {
        let cls = CharClass::Single(CodePoint::new(b'A' as u32).expect("valid"));
        assert!(cls.matches(CodePoint::new(b'A' as u32).expect("valid")));
        assert!(!cls.matches(CodePoint::new(b'B' as u32).expect("valid")));
    }

    #[test]
    fn test_char_class_range() {
        let cls = CharClass::Range(
            CodePoint::new(b'A' as u32).expect("valid"),
            CodePoint::new(b'Z' as u32).expect("valid"),
        );
        assert!(cls.matches(CodePoint::new(b'A' as u32).expect("valid")));
        assert!(cls.matches(CodePoint::new(b'M' as u32).expect("valid")));
        assert!(cls.matches(CodePoint::new(b'Z' as u32).expect("valid")));
        assert!(!cls.matches(CodePoint::new(b'a' as u32).expect("valid")));
    }

    #[test]
    fn test_char_class_predefined() {
        let digit = CharClass::Predefined(PredefinedClass::Digit);
        assert!(digit.matches(CodePoint::new(b'5' as u32).expect("valid")));
        assert!(!digit.matches(CodePoint::new(b'A' as u32).expect("valid")));

        let alpha = CharClass::Predefined(PredefinedClass::Alpha);
        assert!(alpha.matches(CodePoint::new(b'A' as u32).expect("valid")));
        assert!(!alpha.matches(CodePoint::new(b'5' as u32).expect("valid")));
    }

    #[test]
    fn test_char_class_negation() {
        let not_digit = CharClass::Negated(Box::new(CharClass::Predefined(PredefinedClass::Digit)));
        assert!(!not_digit.matches(CodePoint::new(b'5' as u32).expect("valid")));
        assert!(not_digit.matches(CodePoint::new(b'A' as u32).expect("valid")));
    }

    #[test]
    fn test_char_class_simplify() {
        let double_neg = CharClass::Negated(Box::new(CharClass::Negated(Box::new(
            CharClass::Single(CodePoint::new(b'A' as u32).expect("valid")),
        ))));
        let simplified = double_neg.simplify();
        assert!(
            matches!(simplified, CharClass::Single(_)),
            "double negation should simplify"
        );
    }
}
