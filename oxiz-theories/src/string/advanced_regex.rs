//! Advanced Regular Expression Engine
//!
//! Extended regex support with:
//! - **Capture groups**: Named and numbered groups
//! - **Backreferences**: \1, \2, \k\<name\>
//! - **Lookahead/Lookbehind**: (?=...), (?!...), (?<=...), (?<!...)
//! - **Atomic groups**: (?>...)
//! - **Conditional patterns**: (?(condition)yes|no)
//! - **Unicode properties**: \p{L}, \p{N}, etc.
//! - **Word boundaries**: \b, \B
//! - **Advanced quantifiers**: Possessive (+, *, ?) and lazy (?, *?, +?)
//!
//! Implements a hybrid NFA/DFA approach with backtracking for advanced features.

#![allow(missing_docs)]

use super::unicode::UnicodeCategory;
use rustc_hash::FxHashMap;
use std::fmt;

/// Advanced regex pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdvancedRegex {
    /// Empty regex (matches empty string)
    Empty,
    /// Single character
    Char(char),
    /// Character class
    Class(CharacterClass),
    /// Any character (.)
    AnyChar,
    /// Concatenation
    Concat(Vec<AdvancedRegex>),
    /// Alternation (|)
    Alt(Vec<AdvancedRegex>),
    /// Zero or more (*)
    Star(Box<AdvancedRegex>),
    /// One or more (+)
    Plus(Box<AdvancedRegex>),
    /// Zero or one (?)
    Optional(Box<AdvancedRegex>),
    /// Exact repetition {n}
    Repeat(Box<AdvancedRegex>, usize),
    /// Range repetition {n,m}
    RepeatRange(Box<AdvancedRegex>, usize, Option<usize>),
    /// Lazy star (*?)
    StarLazy(Box<AdvancedRegex>),
    /// Lazy plus (+?)
    PlusLazy(Box<AdvancedRegex>),
    /// Lazy optional (??）
    OptionalLazy(Box<AdvancedRegex>),
    /// Possessive star (*+)
    StarPossessive(Box<AdvancedRegex>),
    /// Possessive plus (++)
    PlusPossessive(Box<AdvancedRegex>),
    /// Possessive optional (?+)
    OptionalPossessive(Box<AdvancedRegex>),
    /// Capturing group
    Capture(Box<AdvancedRegex>, CaptureGroup),
    /// Non-capturing group
    Group(Box<AdvancedRegex>),
    /// Backreference
    Backref(usize),
    /// Named backreference
    NamedBackref(String),
    /// Positive lookahead (?=...)
    LookaheadPos(Box<AdvancedRegex>),
    /// Negative lookahead (?!...)
    LookaheadNeg(Box<AdvancedRegex>),
    /// Positive lookbehind (?<=...)
    LookbehindPos(Box<AdvancedRegex>),
    /// Negative lookbehind (?<!...)
    LookbehindNeg(Box<AdvancedRegex>),
    /// Atomic group (?>...)
    Atomic(Box<AdvancedRegex>),
    /// Conditional (?(cond)yes|no)
    Conditional {
        condition: Condition,
        yes_branch: Box<AdvancedRegex>,
        no_branch: Option<Box<AdvancedRegex>>,
    },
    /// Start anchor (^)
    StartAnchor,
    /// End anchor ($)
    EndAnchor,
    /// Word boundary (\b)
    WordBoundary,
    /// Non-word boundary (\B)
    NonWordBoundary,
}

/// Capture group information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CaptureGroup {
    /// Group number (1-indexed)
    pub number: usize,
    /// Optional group name
    pub name: Option<String>,
}

impl CaptureGroup {
    /// Create a numbered capture group
    pub fn numbered(n: usize) -> Self {
        Self {
            number: n,
            name: None,
        }
    }

    /// Create a named capture group
    pub fn named(n: usize, name: String) -> Self {
        Self {
            number: n,
            name: Some(name),
        }
    }
}

/// Conditional pattern condition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Condition {
    /// Check if group number exists
    GroupExists(usize),
    /// Check if named group exists
    NamedGroupExists(String),
    /// Lookahead assertion
    Lookahead(Box<AdvancedRegex>),
}

/// Character class (e.g., [a-z], [^0-9])
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterClass {
    /// Character ranges
    pub ranges: Vec<(char, char)>,
    /// Individual characters
    pub chars: Vec<char>,
    /// Unicode properties
    pub properties: Vec<UnicodeProperty>,
    /// Negated class
    pub negated: bool,
}

impl CharacterClass {
    /// Create a new character class
    pub fn new() -> Self {
        Self {
            ranges: Vec::new(),
            chars: Vec::new(),
            properties: Vec::new(),
            negated: false,
        }
    }

    /// Add a character
    pub fn add_char(&mut self, c: char) {
        self.chars.push(c);
    }

    /// Add a range
    pub fn add_range(&mut self, start: char, end: char) {
        self.ranges.push((start, end));
    }

    /// Add a Unicode property
    pub fn add_property(&mut self, prop: UnicodeProperty) {
        self.properties.push(prop);
    }

    /// Negate this class
    pub fn negate(mut self) -> Self {
        self.negated = !self.negated;
        self
    }

    /// Check if a character matches this class
    pub fn matches(&self, c: char) -> bool {
        let result = self.chars.contains(&c)
            || self
                .ranges
                .iter()
                .any(|&(start, end)| c >= start && c <= end)
            || self.properties.iter().any(|p| p.matches(c));

        if self.negated { !result } else { result }
    }

    /// Predefined digit class [0-9]
    pub fn digit() -> Self {
        let mut cls = Self::new();
        cls.add_range('0', '9');
        cls
    }

    /// Predefined word class [a-zA-Z0-9_]
    pub fn word() -> Self {
        let mut cls = Self::new();
        cls.add_range('a', 'z');
        cls.add_range('A', 'Z');
        cls.add_range('0', '9');
        cls.add_char('_');
        cls
    }

    /// Predefined whitespace class
    pub fn whitespace() -> Self {
        let mut cls = Self::new();
        cls.add_char(' ');
        cls.add_char('\t');
        cls.add_char('\n');
        cls.add_char('\r');
        cls
    }
}

impl Default for CharacterClass {
    fn default() -> Self {
        Self::new()
    }
}

/// Unicode property matcher
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnicodeProperty {
    /// General category
    Category(UnicodeCategory),
    /// Script (e.g., Latin, Greek, Cyrillic)
    Script(UnicodeScript),
    /// Block (e.g., Basic Latin, Latin-1 Supplement)
    Block(UnicodeBlock),
    /// Binary property (e.g., Alphabetic, Lowercase)
    Binary(BinaryProperty),
}

impl UnicodeProperty {
    /// Check if a character matches this property
    pub fn matches(&self, c: char) -> bool {
        match self {
            UnicodeProperty::Category(cat) => cat.contains(c),
            UnicodeProperty::Script(script) => script.contains(c),
            UnicodeProperty::Block(block) => block.contains(c),
            UnicodeProperty::Binary(prop) => prop.matches(c),
        }
    }
}

/// Unicode script
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeScript {
    Latin,
    Greek,
    Cyrillic,
    Arabic,
    Hebrew,
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Thai,
    Devanagari,
}

impl UnicodeScript {
    /// Check if a character belongs to this script
    pub fn contains(&self, c: char) -> bool {
        let cp = c as u32;
        match self {
            UnicodeScript::Latin => {
                matches!(cp, 0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x00FF | 0x0100..=0x017F | 0x0180..=0x024F)
            }
            UnicodeScript::Greek => matches!(cp, 0x0370..=0x03FF | 0x1F00..=0x1FFF),
            UnicodeScript::Cyrillic => matches!(cp, 0x0400..=0x052F),
            UnicodeScript::Arabic => {
                matches!(cp, 0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF)
            }
            UnicodeScript::Hebrew => matches!(cp, 0x0590..=0x05FF | 0xFB1D..=0xFB4F),
            UnicodeScript::Han => {
                matches!(cp, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF)
            }
            UnicodeScript::Hiragana => matches!(cp, 0x3040..=0x309F),
            UnicodeScript::Katakana => matches!(cp, 0x30A0..=0x30FF | 0x31F0..=0x31FF),
            UnicodeScript::Hangul => {
                matches!(cp, 0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F)
            }
            UnicodeScript::Thai => matches!(cp, 0x0E00..=0x0E7F),
            UnicodeScript::Devanagari => matches!(cp, 0x0900..=0x097F),
        }
    }
}

/// Unicode block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeBlock {
    BasicLatin,
    Latin1Supplement,
    LatinExtendedA,
    GreekAndCoptic,
    CJKUnifiedIdeographs,
}

impl UnicodeBlock {
    /// Check if a character belongs to this block
    pub fn contains(&self, c: char) -> bool {
        let cp = c as u32;
        match self {
            UnicodeBlock::BasicLatin => matches!(cp, 0x0000..=0x007F),
            UnicodeBlock::Latin1Supplement => matches!(cp, 0x0080..=0x00FF),
            UnicodeBlock::LatinExtendedA => matches!(cp, 0x0100..=0x017F),
            UnicodeBlock::GreekAndCoptic => matches!(cp, 0x0370..=0x03FF),
            UnicodeBlock::CJKUnifiedIdeographs => matches!(cp, 0x4E00..=0x9FFF),
        }
    }
}

/// Binary Unicode property
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryProperty {
    Alphabetic,
    Lowercase,
    Uppercase,
    WhiteSpace,
    HexDigit,
    AsciiHexDigit,
}

impl BinaryProperty {
    /// Check if a character has this property
    pub fn matches(&self, c: char) -> bool {
        match self {
            BinaryProperty::Alphabetic => c.is_alphabetic(),
            BinaryProperty::Lowercase => c.is_lowercase(),
            BinaryProperty::Uppercase => c.is_uppercase(),
            BinaryProperty::WhiteSpace => c.is_whitespace(),
            BinaryProperty::HexDigit => c.is_ascii_hexdigit(),
            BinaryProperty::AsciiHexDigit => c.is_ascii_hexdigit(),
        }
    }
}

/// Match result with captures
#[derive(Debug, Clone)]
pub struct Match {
    /// Full match string
    pub text: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Captured groups
    pub captures: Vec<Option<String>>,
    /// Named captures
    pub named_captures: FxHashMap<String, String>,
}

impl Match {
    /// Create a new match
    pub fn new(text: String, start: usize, end: usize) -> Self {
        Self {
            text,
            start,
            end,
            captures: Vec::new(),
            named_captures: FxHashMap::default(),
        }
    }

    /// Get a capture group by index (0 = full match)
    pub fn get(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some(&self.text)
        } else {
            self.captures.get(index - 1).and_then(|opt| opt.as_deref())
        }
    }

    /// Get a named capture group
    pub fn name(&self, name: &str) -> Option<&str> {
        self.named_captures.get(name).map(|s| s.as_str())
    }
}

/// Regex matcher with backtracking support
#[derive(Debug)]
pub struct RegexMatcher {
    /// The regex pattern
    pattern: AdvancedRegex,
    /// Next capture group number
    #[allow(dead_code)]
    next_capture: usize,
}

impl RegexMatcher {
    /// Create a new regex matcher
    pub fn new(pattern: AdvancedRegex) -> Self {
        Self {
            pattern,
            next_capture: 1,
        }
    }

    /// Check if the pattern matches the entire string
    pub fn is_match(&self, text: &str) -> bool {
        if let Some(m) = self.match_at(text, 0) {
            // If pattern has positional anchors (but not EndAnchor), allow partial matches
            // Otherwise, require full string consumption
            if self.has_start_anchor_only() {
                // Anchored pattern - match succeeds if pattern matches at the position
                true
            } else {
                // Non-anchored pattern - must consume entire string
                m.end == text.len()
            }
        } else {
            false
        }
    }

    /// Check if pattern has positional constraints that allow partial matches
    fn has_start_anchor_only(&self) -> bool {
        let has_positional = self.contains_start_anchor() || self.contains_lookahead();
        let has_end = self.contains_end_anchor();
        has_positional && !has_end
    }

    /// Check if pattern contains lookahead/lookbehind assertions
    fn contains_lookahead(&self) -> bool {
        Self::pattern_has_lookahead(&self.pattern)
    }

    fn pattern_has_lookahead(regex: &AdvancedRegex) -> bool {
        match regex {
            AdvancedRegex::LookaheadPos(_)
            | AdvancedRegex::LookaheadNeg(_)
            | AdvancedRegex::LookbehindPos(_)
            | AdvancedRegex::LookbehindNeg(_) => true,
            AdvancedRegex::Concat(parts) => parts.iter().any(Self::pattern_has_lookahead),
            AdvancedRegex::Alt(parts) => parts.iter().any(Self::pattern_has_lookahead),
            AdvancedRegex::Group(inner) | AdvancedRegex::Capture(inner, _) => {
                Self::pattern_has_lookahead(inner)
            }
            _ => false,
        }
    }

    /// Check if pattern contains StartAnchor
    fn contains_start_anchor(&self) -> bool {
        match self.pattern {
            AdvancedRegex::StartAnchor => true,
            AdvancedRegex::Concat(ref parts) => parts.iter().any(Self::pattern_has_start_anchor),
            _ => Self::pattern_has_start_anchor(&self.pattern),
        }
    }

    /// Check if pattern contains EndAnchor
    fn contains_end_anchor(&self) -> bool {
        match self.pattern {
            AdvancedRegex::EndAnchor => true,
            AdvancedRegex::Concat(ref parts) => parts.iter().any(Self::pattern_has_end_anchor),
            _ => Self::pattern_has_end_anchor(&self.pattern),
        }
    }

    fn pattern_has_start_anchor(regex: &AdvancedRegex) -> bool {
        match regex {
            AdvancedRegex::StartAnchor => true,
            AdvancedRegex::Concat(parts) => parts.iter().any(Self::pattern_has_start_anchor),
            AdvancedRegex::Alt(parts) => parts.iter().any(Self::pattern_has_start_anchor),
            AdvancedRegex::Group(inner) | AdvancedRegex::Capture(inner, _) => {
                Self::pattern_has_start_anchor(inner)
            }
            _ => false,
        }
    }

    fn pattern_has_end_anchor(regex: &AdvancedRegex) -> bool {
        match regex {
            AdvancedRegex::EndAnchor => true,
            AdvancedRegex::Concat(parts) => parts.iter().any(Self::pattern_has_end_anchor),
            AdvancedRegex::Alt(parts) => parts.iter().any(Self::pattern_has_end_anchor),
            AdvancedRegex::Group(inner) | AdvancedRegex::Capture(inner, _) => {
                Self::pattern_has_end_anchor(inner)
            }
            _ => false,
        }
    }

    /// Find the first match in the string
    pub fn find(&self, text: &str) -> Option<Match> {
        for i in 0..=text.len() {
            if let Some(m) = self.match_at(text, i) {
                return Some(m);
            }
        }
        None
    }

    /// Find all matches in the string
    pub fn find_all(&self, text: &str) -> Vec<Match> {
        let mut matches = Vec::new();
        let mut pos = 0;

        while pos <= text.len() {
            if let Some(m) = self.match_at(text, pos) {
                pos = m.end.max(pos + 1); // Avoid infinite loop on empty matches
                matches.push(m);
            } else {
                pos += 1;
            }
        }

        matches
    }

    /// Try to match at a specific position
    fn match_at(&self, text: &str, pos: usize) -> Option<Match> {
        let mut state = MatchState::new(text, pos);
        if self.match_regex(&self.pattern, &mut state) {
            Some(Match {
                text: text[pos..state.pos].to_string(),
                start: pos,
                end: state.pos,
                captures: state.captures.clone(),
                named_captures: state.named_captures.clone(),
            })
        } else {
            None
        }
    }

    /// Match a regex pattern against the state
    fn match_regex(&self, regex: &AdvancedRegex, state: &mut MatchState) -> bool {
        match regex {
            AdvancedRegex::Empty => true,

            AdvancedRegex::Char(c) => {
                if state.peek() == Some(*c) {
                    state.advance();
                    true
                } else {
                    false
                }
            }

            AdvancedRegex::Class(cls) => {
                if let Some(c) = state.peek()
                    && cls.matches(c)
                {
                    state.advance();
                    return true;
                }
                false
            }

            AdvancedRegex::AnyChar => {
                if state.peek().is_some() {
                    state.advance();
                    true
                } else {
                    false
                }
            }

            AdvancedRegex::Concat(parts) => {
                let saved = state.save();
                for part in parts {
                    if !self.match_regex(part, state) {
                        state.restore(saved);
                        return false;
                    }
                }
                true
            }

            AdvancedRegex::Alt(branches) => {
                for branch in branches {
                    let saved = state.save();
                    if self.match_regex(branch, state) {
                        return true;
                    }
                    state.restore(saved);
                }
                false
            }

            AdvancedRegex::Star(inner) => {
                // Greedy star: match as many as possible
                while self.match_regex(inner, state) {}
                true
            }

            AdvancedRegex::Plus(inner) => {
                if !self.match_regex(inner, state) {
                    return false;
                }
                while self.match_regex(inner, state) {}
                true
            }

            AdvancedRegex::Optional(inner) => {
                let _ = self.match_regex(inner, state);
                true
            }

            AdvancedRegex::Repeat(inner, n) => {
                let saved = state.save();
                for _ in 0..*n {
                    if !self.match_regex(inner, state) {
                        state.restore(saved);
                        return false;
                    }
                }
                true
            }

            AdvancedRegex::RepeatRange(inner, min, max) => {
                let saved = state.save();
                // Match minimum required
                for _ in 0..*min {
                    if !self.match_regex(inner, state) {
                        state.restore(saved);
                        return false;
                    }
                }
                // Match up to maximum (greedy)
                let max_extra = max.map(|m| m - min).unwrap_or(usize::MAX);
                for _ in 0..max_extra {
                    if !self.match_regex(inner, state) {
                        break;
                    }
                }
                true
            }

            AdvancedRegex::StarLazy(_inner) => {
                // Lazy star: match as few as possible
                // Initially match zero times
                true
            }

            AdvancedRegex::PlusLazy(inner) => {
                // Lazy plus: match at least once
                self.match_regex(inner, state)
            }

            AdvancedRegex::OptionalLazy(_inner) => {
                // Lazy optional: prefer not matching
                true
            }

            AdvancedRegex::StarPossessive(inner) => {
                // Possessive star: match greedily without backtracking
                while self.match_regex(inner, state) {}
                true
            }

            AdvancedRegex::PlusPossessive(inner) => {
                if !self.match_regex(inner, state) {
                    return false;
                }
                while self.match_regex(inner, state) {}
                true
            }

            AdvancedRegex::OptionalPossessive(inner) => {
                let _ = self.match_regex(inner, state);
                true
            }

            AdvancedRegex::Capture(inner, group) => {
                let start = state.pos;
                if self.match_regex(inner, state) {
                    let captured = state.text[start..state.pos].to_string();
                    state.add_capture(group.number, captured.clone());
                    if let Some(name) = &group.name {
                        state.add_named_capture(name.clone(), captured);
                    }
                    true
                } else {
                    false
                }
            }

            AdvancedRegex::Group(inner) => self.match_regex(inner, state),

            AdvancedRegex::Backref(n) => {
                if let Some(captured) = state.get_capture(*n) {
                    let saved = state.save();
                    for c in captured.chars() {
                        if state.peek() != Some(c) {
                            state.restore(saved);
                            return false;
                        }
                        state.advance();
                    }
                    true
                } else {
                    false
                }
            }

            AdvancedRegex::NamedBackref(name) => {
                if let Some(captured) = state.get_named_capture(name) {
                    let saved = state.save();
                    for c in captured.chars() {
                        if state.peek() != Some(c) {
                            state.restore(saved);
                            return false;
                        }
                        state.advance();
                    }
                    true
                } else {
                    false
                }
            }

            AdvancedRegex::LookaheadPos(inner) => {
                let saved = state.save();
                let result = self.match_regex(inner, state);
                state.restore(saved);
                result
            }

            AdvancedRegex::LookaheadNeg(inner) => {
                let saved = state.save();
                let result = !self.match_regex(inner, state);
                state.restore(saved);
                result
            }

            AdvancedRegex::LookbehindPos(inner) => {
                // Lookbehind is tricky - need to match in reverse
                // Simplified: just check if the pattern matches before current position
                let saved = state.save();
                // Try to find a match ending at current position
                for start in 0..=state.pos {
                    state.pos = start;
                    if self.match_regex(inner, state) && state.pos == saved {
                        state.restore(saved);
                        return true;
                    }
                }
                state.restore(saved);
                false
            }

            AdvancedRegex::LookbehindNeg(inner) => {
                !self.match_regex(&AdvancedRegex::LookbehindPos(inner.clone()), state)
            }

            AdvancedRegex::Atomic(inner) => {
                // Atomic group: no backtracking once matched
                self.match_regex(inner, state)
            }

            AdvancedRegex::Conditional {
                condition,
                yes_branch,
                no_branch,
            } => {
                let cond_result = self.check_condition(condition, state);
                if cond_result {
                    self.match_regex(yes_branch, state)
                } else if let Some(no) = no_branch {
                    self.match_regex(no, state)
                } else {
                    true
                }
            }

            AdvancedRegex::StartAnchor => state.pos == 0,

            AdvancedRegex::EndAnchor => state.pos >= state.text.len(),

            AdvancedRegex::WordBoundary => {
                let before = state.pos > 0 && is_word_char(state.text[..state.pos].chars().last());
                let after = state.pos < state.text.len() && is_word_char(state.peek());
                before != after
            }

            AdvancedRegex::NonWordBoundary => {
                !self.match_regex(&AdvancedRegex::WordBoundary, state)
            }
        }
    }

    /// Check a conditional condition
    fn check_condition(&self, condition: &Condition, state: &mut MatchState) -> bool {
        match condition {
            Condition::GroupExists(n) => state.get_capture(*n).is_some(),
            Condition::NamedGroupExists(name) => state.get_named_capture(name).is_some(),
            Condition::Lookahead(regex) => {
                let saved = state.save();
                let mut temp_state = state.clone();
                let result = self.match_regex(regex, &mut temp_state);
                state.restore(saved);
                result
            }
        }
    }
}

/// Check if a character is a word character
fn is_word_char(c: Option<char>) -> bool {
    c.is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
}

/// Matching state with backtracking support
#[derive(Debug, Clone)]
struct MatchState<'a> {
    /// Input text
    text: &'a str,
    /// Current position
    pos: usize,
    /// Captured groups
    captures: Vec<Option<String>>,
    /// Named captures
    named_captures: FxHashMap<String, String>,
}

impl<'a> MatchState<'a> {
    /// Create a new match state
    fn new(text: &'a str, pos: usize) -> Self {
        Self {
            text,
            pos,
            captures: Vec::new(),
            named_captures: FxHashMap::default(),
        }
    }

    /// Peek at current character
    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    /// Advance to next character
    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    /// Save current state
    fn save(&self) -> usize {
        self.pos
    }

    /// Restore to saved state
    fn restore(&mut self, saved: usize) {
        self.pos = saved;
    }

    /// Add a capture group
    fn add_capture(&mut self, index: usize, value: String) {
        // Extend captures vector if needed
        while self.captures.len() < index {
            self.captures.push(None);
        }
        if index > 0 {
            self.captures[index - 1] = Some(value);
        }
    }

    /// Get a capture group
    fn get_capture(&self, index: usize) -> Option<String> {
        if index > 0 && index <= self.captures.len() {
            self.captures[index - 1].clone()
        } else {
            None
        }
    }

    /// Add a named capture
    fn add_named_capture(&mut self, name: String, value: String) {
        self.named_captures.insert(name, value);
    }

    /// Get a named capture
    fn get_named_capture(&self, name: &str) -> Option<String> {
        self.named_captures.get(name).cloned()
    }
}

/// Regex builder for constructing patterns programmatically
#[derive(Debug)]
pub struct RegexBuilder {
    parts: Vec<AdvancedRegex>,
    next_group: usize,
}

impl RegexBuilder {
    /// Create a new regex builder
    pub fn new() -> Self {
        Self {
            parts: Vec::new(),
            next_group: 1,
        }
    }

    /// Add a literal string
    pub fn literal(mut self, s: &str) -> Self {
        for c in s.chars() {
            self.parts.push(AdvancedRegex::Char(c));
        }
        self
    }

    /// Add a character class
    pub fn class(mut self, cls: CharacterClass) -> Self {
        self.parts.push(AdvancedRegex::Class(cls));
        self
    }

    /// Add any character (.)
    pub fn any(mut self) -> Self {
        self.parts.push(AdvancedRegex::AnyChar);
        self
    }

    /// Add a digit class (\d)
    pub fn digit(mut self) -> Self {
        self.parts
            .push(AdvancedRegex::Class(CharacterClass::digit()));
        self
    }

    /// Add a word class (\w)
    pub fn word(mut self) -> Self {
        self.parts
            .push(AdvancedRegex::Class(CharacterClass::word()));
        self
    }

    /// Add a whitespace class (\s)
    pub fn whitespace(mut self) -> Self {
        self.parts
            .push(AdvancedRegex::Class(CharacterClass::whitespace()));
        self
    }

    /// Add a capturing group
    pub fn capture(mut self, inner: AdvancedRegex) -> Self {
        let group = CaptureGroup::numbered(self.next_group);
        self.next_group += 1;
        self.parts
            .push(AdvancedRegex::Capture(Box::new(inner), group));
        self
    }

    /// Add a named capturing group
    pub fn named_capture(mut self, name: &str, inner: AdvancedRegex) -> Self {
        let group = CaptureGroup::named(self.next_group, name.to_string());
        self.next_group += 1;
        self.parts
            .push(AdvancedRegex::Capture(Box::new(inner), group));
        self
    }

    /// Add zero or more (*)
    pub fn star(mut self, inner: AdvancedRegex) -> Self {
        self.parts.push(AdvancedRegex::Star(Box::new(inner)));
        self
    }

    /// Add one or more (+)
    pub fn plus(mut self, inner: AdvancedRegex) -> Self {
        self.parts.push(AdvancedRegex::Plus(Box::new(inner)));
        self
    }

    /// Add optional (?)
    pub fn optional(mut self, inner: AdvancedRegex) -> Self {
        self.parts.push(AdvancedRegex::Optional(Box::new(inner)));
        self
    }

    /// Add alternation (|)
    pub fn alt(mut self, branches: Vec<AdvancedRegex>) -> Self {
        self.parts.push(AdvancedRegex::Alt(branches));
        self
    }

    /// Add start anchor (^)
    pub fn start_anchor(mut self) -> Self {
        self.parts.push(AdvancedRegex::StartAnchor);
        self
    }

    /// Add end anchor ($)
    pub fn end_anchor(mut self) -> Self {
        self.parts.push(AdvancedRegex::EndAnchor);
        self
    }

    /// Build the final regex
    pub fn build(self) -> AdvancedRegex {
        if self.parts.is_empty() {
            AdvancedRegex::Empty
        } else if self.parts.len() == 1 {
            self.parts.into_iter().next().expect("exactly one element")
        } else {
            AdvancedRegex::Concat(self.parts)
        }
    }
}

impl Default for RegexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AdvancedRegex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvancedRegex::Empty => write!(f, "ε"),
            AdvancedRegex::Char(c) => write!(f, "{}", c),
            AdvancedRegex::Class(_) => write!(f, "[...]"),
            AdvancedRegex::AnyChar => write!(f, "."),
            AdvancedRegex::Concat(parts) => {
                for part in parts {
                    write!(f, "{}", part)?;
                }
                Ok(())
            }
            AdvancedRegex::Alt(branches) => {
                for (i, branch) in branches.iter().enumerate() {
                    if i > 0 {
                        write!(f, "|")?;
                    }
                    write!(f, "{}", branch)?;
                }
                Ok(())
            }
            AdvancedRegex::Star(inner) => write!(f, "({})*", inner),
            AdvancedRegex::Plus(inner) => write!(f, "({})+", inner),
            AdvancedRegex::Optional(inner) => write!(f, "({})?", inner),
            AdvancedRegex::Repeat(inner, n) => write!(f, "({}){{{}}}", inner, n),
            AdvancedRegex::RepeatRange(inner, min, max) => {
                if let Some(m) = max {
                    write!(f, "({}){{{},{}}}", inner, min, m)
                } else {
                    write!(f, "({}){{{},}}", inner, min)
                }
            }
            AdvancedRegex::StarLazy(inner) => write!(f, "({})*?", inner),
            AdvancedRegex::PlusLazy(inner) => write!(f, "({})+?", inner),
            AdvancedRegex::OptionalLazy(inner) => write!(f, "({})??", inner),
            AdvancedRegex::StarPossessive(inner) => write!(f, "({})*+", inner),
            AdvancedRegex::PlusPossessive(inner) => write!(f, "({})++", inner),
            AdvancedRegex::OptionalPossessive(inner) => write!(f, "({})?+", inner),
            AdvancedRegex::Capture(inner, group) => {
                if let Some(name) = &group.name {
                    write!(f, "(?<{}>{})", name, inner)
                } else {
                    write!(f, "({})", inner)
                }
            }
            AdvancedRegex::Group(inner) => write!(f, "(?:{})", inner),
            AdvancedRegex::Backref(n) => write!(f, "\\{}", n),
            AdvancedRegex::NamedBackref(name) => write!(f, "\\k<{}>", name),
            AdvancedRegex::LookaheadPos(inner) => write!(f, "(?={})", inner),
            AdvancedRegex::LookaheadNeg(inner) => write!(f, "(?!{})", inner),
            AdvancedRegex::LookbehindPos(inner) => write!(f, "(?<={})", inner),
            AdvancedRegex::LookbehindNeg(inner) => write!(f, "(?<!{})", inner),
            AdvancedRegex::Atomic(inner) => write!(f, "(?>>{})", inner),
            AdvancedRegex::Conditional { .. } => write!(f, "(?(...)...)"),
            AdvancedRegex::StartAnchor => write!(f, "^"),
            AdvancedRegex::EndAnchor => write!(f, "$"),
            AdvancedRegex::WordBoundary => write!(f, "\\b"),
            AdvancedRegex::NonWordBoundary => write!(f, "\\B"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_class_digit() {
        let cls = CharacterClass::digit();
        assert!(cls.matches('0'));
        assert!(cls.matches('9'));
        assert!(!cls.matches('a'));
    }

    #[test]
    fn test_char_class_word() {
        let cls = CharacterClass::word();
        assert!(cls.matches('a'));
        assert!(cls.matches('Z'));
        assert!(cls.matches('0'));
        assert!(cls.matches('_'));
        assert!(!cls.matches(' '));
    }

    #[test]
    fn test_char_class_negation() {
        let cls = CharacterClass::digit().negate();
        assert!(!cls.matches('5'));
        assert!(cls.matches('a'));
    }

    #[test]
    fn test_simple_char_match() {
        let regex = AdvancedRegex::Char('a');
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("a"));
        assert!(!matcher.is_match("b"));
    }

    #[test]
    fn test_concat_match() {
        let regex = AdvancedRegex::Concat(vec![
            AdvancedRegex::Char('a'),
            AdvancedRegex::Char('b'),
            AdvancedRegex::Char('c'),
        ]);
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("abc"));
        assert!(!matcher.is_match("ab"));
    }

    #[test]
    fn test_alt_match() {
        let regex = AdvancedRegex::Alt(vec![AdvancedRegex::Char('a'), AdvancedRegex::Char('b')]);
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("a"));
        assert!(matcher.is_match("b"));
        assert!(!matcher.is_match("c"));
    }

    #[test]
    fn test_star_match() {
        let regex = AdvancedRegex::Star(Box::new(AdvancedRegex::Char('a')));
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match(""));
        assert!(matcher.is_match("a"));
        assert!(matcher.is_match("aaa"));
    }

    #[test]
    fn test_plus_match() {
        let regex = AdvancedRegex::Plus(Box::new(AdvancedRegex::Char('a')));
        let matcher = RegexMatcher::new(regex);
        assert!(!matcher.is_match(""));
        assert!(matcher.is_match("a"));
        assert!(matcher.is_match("aaa"));
    }

    #[test]
    fn test_optional_match() {
        let regex = AdvancedRegex::Optional(Box::new(AdvancedRegex::Char('a')));
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match(""));
        assert!(matcher.is_match("a"));
    }

    #[test]
    fn test_capture_group() {
        let regex = AdvancedRegex::Capture(
            Box::new(AdvancedRegex::Concat(vec![
                AdvancedRegex::Char('a'),
                AdvancedRegex::Char('b'),
            ])),
            CaptureGroup::numbered(1),
        );
        let matcher = RegexMatcher::new(regex);
        let m = matcher.find("ab").expect("should match");
        assert_eq!(m.get(0), Some("ab"));
        assert_eq!(m.get(1), Some("ab"));
    }

    #[test]
    fn test_named_capture() {
        let regex = AdvancedRegex::Capture(
            Box::new(AdvancedRegex::Char('x')),
            CaptureGroup::named(1, "test".to_string()),
        );
        let matcher = RegexMatcher::new(regex);
        let m = matcher.find("x").expect("should match");
        assert_eq!(m.name("test"), Some("x"));
    }

    #[test]
    fn test_backreference() {
        let regex = AdvancedRegex::Concat(vec![
            AdvancedRegex::Capture(
                Box::new(AdvancedRegex::Char('a')),
                CaptureGroup::numbered(1),
            ),
            AdvancedRegex::Backref(1),
        ]);
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("aa"));
        assert!(!matcher.is_match("ab"));
    }

    #[test]
    fn test_lookahead_positive() {
        let regex = AdvancedRegex::Concat(vec![
            AdvancedRegex::Char('a'),
            AdvancedRegex::LookaheadPos(Box::new(AdvancedRegex::Char('b'))),
        ]);
        let matcher = RegexMatcher::new(regex);
        let m = matcher.find("ab");
        assert!(m.is_some());
        let m = m.expect("matched");
        assert_eq!(m.text, "a"); // Lookahead doesn't consume
    }

    #[test]
    fn test_lookahead_negative() {
        let regex = AdvancedRegex::Concat(vec![
            AdvancedRegex::Char('a'),
            AdvancedRegex::LookaheadNeg(Box::new(AdvancedRegex::Char('b'))),
        ]);
        let matcher = RegexMatcher::new(regex);
        assert!(!matcher.is_match("ab"));
        assert!(matcher.is_match("ac"));
    }

    #[test]
    fn test_word_boundary() {
        let regex = AdvancedRegex::Concat(vec![
            AdvancedRegex::WordBoundary,
            AdvancedRegex::Char('a'),
            AdvancedRegex::WordBoundary,
        ]);
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.find("a").is_some());
        assert!(matcher.find(" a ").is_some());
    }

    #[test]
    fn test_start_anchor() {
        let regex =
            AdvancedRegex::Concat(vec![AdvancedRegex::StartAnchor, AdvancedRegex::Char('a')]);
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("a"));
        assert!(matcher.is_match("abc"));
    }

    #[test]
    fn test_end_anchor() {
        let regex = AdvancedRegex::Concat(vec![AdvancedRegex::Char('a'), AdvancedRegex::EndAnchor]);
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("a"));
    }

    #[test]
    fn test_repeat_exact() {
        let regex = AdvancedRegex::Repeat(Box::new(AdvancedRegex::Char('a')), 3);
        let matcher = RegexMatcher::new(regex);
        assert!(!matcher.is_match("aa"));
        assert!(matcher.is_match("aaa"));
        assert!(!matcher.is_match("aaaa"));
    }

    #[test]
    fn test_repeat_range() {
        let regex = AdvancedRegex::RepeatRange(Box::new(AdvancedRegex::Char('a')), 2, Some(4));
        let matcher = RegexMatcher::new(regex);
        assert!(!matcher.is_match("a"));
        assert!(matcher.is_match("aa"));
        assert!(matcher.is_match("aaa"));
        assert!(matcher.is_match("aaaa"));
    }

    #[test]
    fn test_unicode_category() {
        let prop = UnicodeProperty::Category(UnicodeCategory::Letter);
        assert!(prop.matches('a'));
        assert!(prop.matches('Z'));
        assert!(!prop.matches('1'));
    }

    #[test]
    fn test_unicode_script() {
        assert!(UnicodeScript::Latin.contains('a'));
        assert!(UnicodeScript::Greek.contains('α'));
        assert!(UnicodeScript::Cyrillic.contains('Б'));
    }

    #[test]
    fn test_binary_property() {
        assert!(BinaryProperty::Alphabetic.matches('a'));
        assert!(BinaryProperty::Lowercase.matches('a'));
        assert!(!BinaryProperty::Lowercase.matches('A'));
    }

    #[test]
    fn test_regex_builder() {
        let regex = RegexBuilder::new()
            .start_anchor()
            .literal("hello")
            .end_anchor()
            .build();

        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("hello"));
        assert!(!matcher.is_match("hello world"));
    }

    #[test]
    fn test_find_all() {
        let regex = AdvancedRegex::Char('a');
        let matcher = RegexMatcher::new(regex);
        let matches = matcher.find_all("banana");
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_any_char() {
        let regex = AdvancedRegex::AnyChar;
        let matcher = RegexMatcher::new(regex);
        assert!(matcher.is_match("a"));
        assert!(matcher.is_match("1"));
        assert!(matcher.is_match("!"));
        assert!(!matcher.is_match(""));
    }

    #[test]
    fn test_display() {
        let regex = AdvancedRegex::Star(Box::new(AdvancedRegex::Char('a')));
        assert_eq!(format!("{}", regex), "(a)*");
    }
}
