//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};

use super::functions::CodePoint;


/// Unicode block (ranges of code points)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeBlock {
    /// Basic Latin (U+0000..U+007F)
    BasicLatin,
    /// Latin-1 Supplement (U+0080..U+00FF)
    Latin1Supplement,
    /// Latin Extended-A (U+0100..U+017F)
    LatinExtendedA,
    /// Latin Extended-B (U+0180..U+024F)
    LatinExtendedB,
    /// Greek and Coptic (U+0370..U+03FF)
    GreekAndCoptic,
    /// Cyrillic (U+0400..U+04FF)
    Cyrillic,
    /// Arabic (U+0600..U+06FF)
    Arabic,
    /// Hebrew (U+0590..U+05FF)
    Hebrew,
    /// Devanagari (U+0900..U+097F)
    Devanagari,
    /// Thai (U+0E00..U+0E7F)
    Thai,
    /// CJK Unified Ideographs (U+4E00..U+9FFF)
    CjkUnifiedIdeographs,
    /// Hiragana (U+3040..U+309F)
    Hiragana,
    /// Katakana (U+30A0..U+30FF)
    Katakana,
    /// Hangul Syllables (U+AC00..U+D7AF)
    HangulSyllables,
    /// Private Use Area (U+E000..U+F8FF)
    PrivateUseArea,
    /// Emoticons (U+1F600..U+1F64F)
    Emoticons,
    /// Other/Unknown block
    Other,
}
impl UnicodeBlock {
    /// Get the Unicode block for a code point
    pub fn from_code_point(cp: CodePoint) -> Self {
        match cp {
            0x0000..=0x007F => UnicodeBlock::BasicLatin,
            0x0080..=0x00FF => UnicodeBlock::Latin1Supplement,
            0x0100..=0x017F => UnicodeBlock::LatinExtendedA,
            0x0180..=0x024F => UnicodeBlock::LatinExtendedB,
            0x0370..=0x03FF => UnicodeBlock::GreekAndCoptic,
            0x0400..=0x04FF => UnicodeBlock::Cyrillic,
            0x0590..=0x05FF => UnicodeBlock::Hebrew,
            0x0600..=0x06FF => UnicodeBlock::Arabic,
            0x0900..=0x097F => UnicodeBlock::Devanagari,
            0x0E00..=0x0E7F => UnicodeBlock::Thai,
            0x3040..=0x309F => UnicodeBlock::Hiragana,
            0x30A0..=0x30FF => UnicodeBlock::Katakana,
            0x4E00..=0x9FFF => UnicodeBlock::CjkUnifiedIdeographs,
            0xAC00..=0xD7AF => UnicodeBlock::HangulSyllables,
            0xE000..=0xF8FF => UnicodeBlock::PrivateUseArea,
            0x1F600..=0x1F64F => UnicodeBlock::Emoticons,
            _ => UnicodeBlock::Other,
        }
    }
    /// Get the range for this block
    pub fn range(&self) -> (CodePoint, CodePoint) {
        match self {
            UnicodeBlock::BasicLatin => (0x0000, 0x007F),
            UnicodeBlock::Latin1Supplement => (0x0080, 0x00FF),
            UnicodeBlock::LatinExtendedA => (0x0100, 0x017F),
            UnicodeBlock::LatinExtendedB => (0x0180, 0x024F),
            UnicodeBlock::GreekAndCoptic => (0x0370, 0x03FF),
            UnicodeBlock::Cyrillic => (0x0400, 0x04FF),
            UnicodeBlock::Hebrew => (0x0590, 0x05FF),
            UnicodeBlock::Arabic => (0x0600, 0x06FF),
            UnicodeBlock::Devanagari => (0x0900, 0x097F),
            UnicodeBlock::Thai => (0x0E00, 0x0E7F),
            UnicodeBlock::Hiragana => (0x3040, 0x309F),
            UnicodeBlock::Katakana => (0x30A0, 0x30FF),
            UnicodeBlock::CjkUnifiedIdeographs => (0x4E00, 0x9FFF),
            UnicodeBlock::HangulSyllables => (0xAC00, 0xD7AF),
            UnicodeBlock::PrivateUseArea => (0xE000, 0xF8FF),
            UnicodeBlock::Emoticons => (0x1F600, 0x1F64F),
            UnicodeBlock::Other => (0, 0x10FFFF),
        }
    }
    /// Check if a code point is in this block
    pub fn contains(&self, cp: CodePoint) -> bool {
        let (low, high) = self.range();
        cp >= low && cp <= high
    }
}
/// Statistics for character solver
#[derive(Debug, Clone, Default)]
pub struct CharStats {
    /// Number of constraints added
    pub constraints_added: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Number of conflicts
    pub conflicts: u64,
    /// Number of decisions
    pub decisions: u64,
}
/// Case folding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaseFoldMode {
    /// Simple case folding (1:1 mapping)
    Simple,
    /// Full case folding (may expand)
    Full,
    /// Turkic case folding (special I/ı handling)
    Turkic,
}
/// A domain of possible character values
#[derive(Debug, Clone)]
pub struct CharDomain {
    /// Included ranges
    ranges: Vec<(CodePoint, CodePoint)>,
    /// Excluded values
    excluded: HashSet<CodePoint>,
    /// Is the domain empty?
    empty: bool,
}
impl CharDomain {
    /// Create a full domain (all valid Unicode)
    pub fn full() -> Self {
        Self {
            ranges: vec![(0, 0x10FFFF)],
            excluded: HashSet::new(),
            empty: false,
        }
    }
    /// Create an empty domain
    pub fn empty() -> Self {
        Self {
            ranges: Vec::new(),
            excluded: HashSet::new(),
            empty: true,
        }
    }
    /// Create a domain from a single value
    pub fn singleton(cp: CodePoint) -> Self {
        Self {
            ranges: vec![(cp, cp)],
            excluded: HashSet::new(),
            empty: false,
        }
    }
    /// Create a domain from a range
    pub fn range(low: CodePoint, high: CodePoint) -> Self {
        if low > high {
            Self::empty()
        } else {
            Self {
                ranges: vec![(low, high)],
                excluded: HashSet::new(),
                empty: false,
            }
        }
    }
    /// Check if the domain is empty
    pub fn is_empty(&self) -> bool {
        self.empty
    }
    /// Check if the domain is a singleton
    pub fn is_singleton(&self) -> Option<CodePoint> {
        if self.ranges.len() == 1 && self.excluded.is_empty() {
            let (low, high) = self.ranges[0];
            if low == high {
                return Some(low);
            }
        }
        None
    }
    /// Check if a value is in the domain
    pub fn contains(&self, cp: CodePoint) -> bool {
        if self.empty || self.excluded.contains(&cp) {
            return false;
        }
        self.ranges.iter().any(|(low, high)| cp >= *low && cp <= *high)
    }
    /// Intersect with another domain
    pub fn intersect(&mut self, other: &CharDomain) {
        if self.empty || other.empty {
            self.empty = true;
            self.ranges.clear();
            return;
        }
        let mut new_ranges = Vec::new();
        for (l1, h1) in &self.ranges {
            for (l2, h2) in &other.ranges {
                let low = (*l1).max(*l2);
                let high = (*h1).min(*h2);
                if low <= high {
                    new_ranges.push((low, high));
                }
            }
        }
        self.ranges = new_ranges;
        self.excluded.extend(other.excluded.iter());
        if self.ranges.is_empty() {
            self.empty = true;
        }
    }
    /// Exclude a value
    pub fn exclude(&mut self, cp: CodePoint) {
        self.excluded.insert(cp);
        self.check_empty();
    }
    /// Restrict to a range
    pub fn restrict_to_range(&mut self, low: CodePoint, high: CodePoint) {
        let range_domain = CharDomain::range(low, high);
        self.intersect(&range_domain);
    }
    /// Get the minimum value in the domain
    pub fn min(&self) -> Option<CodePoint> {
        if self.empty {
            return None;
        }
        for &(low, high) in &self.ranges {
            for cp in low..=high {
                if !self.excluded.contains(&cp) && CharSolver::is_valid_code_point(cp) {
                    return Some(cp);
                }
            }
        }
        None
    }
    /// Get the maximum value in the domain
    pub fn max(&self) -> Option<CodePoint> {
        if self.empty {
            return None;
        }
        for &(low, high) in self.ranges.iter().rev() {
            for cp in (low..=high).rev() {
                if !self.excluded.contains(&cp) && CharSolver::is_valid_code_point(cp) {
                    return Some(cp);
                }
            }
        }
        None
    }
    /// Get the size of the domain (may be very large)
    pub fn size(&self) -> u64 {
        if self.empty {
            return 0;
        }
        let total: u64 = self
            .ranges
            .iter()
            .map(|(l, h)| (*h as u64) - (*l as u64) + 1)
            .sum();
        total.saturating_sub(self.excluded.len() as u64)
    }
    fn check_empty(&mut self) {
        if self.ranges.is_empty() {
            self.empty = true;
            return;
        }
        let total_excluded: usize = self
            .ranges
            .iter()
            .map(|(l, h)| (*l..=*h).filter(|cp| self.excluded.contains(cp)).count())
            .sum();
        let total_values: u64 = self
            .ranges
            .iter()
            .map(|(l, h)| (*h as u64) - (*l as u64) + 1)
            .sum();
        if total_excluded as u64 >= total_values {
            self.empty = true;
        }
    }
}
/// Character width classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CharWidth {
    /// Narrow (most Western characters)
    Narrow,
    /// Wide (CJK ideographs, etc.)
    Wide,
    /// Half-width (half-width CJK)
    Halfwidth,
    /// Full-width (full-width ASCII)
    Fullwidth,
    /// Ambiguous (depends on context)
    Ambiguous,
    /// Neutral (doesn't apply)
    Neutral,
}
impl CharWidth {
    /// Get the width class for a code point
    pub fn from_code_point(cp: CodePoint) -> Self {
        match cp {
            0x0020..=0x007E => CharWidth::Narrow,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x3040..=0x30FF | 0xAC00..=0xD7AF => {
                CharWidth::Wide
            }
            0xFF01..=0xFF5E => CharWidth::Fullwidth,
            0xFF65..=0xFF9F => CharWidth::Halfwidth,
            0x00A0..=0x00FF => CharWidth::Ambiguous,
            0x0000..=0x001F | 0x007F => CharWidth::Neutral,
            _ => CharWidth::Neutral,
        }
    }
    /// Get the display width (1 or 2 cells)
    pub fn cells(&self) -> u8 {
        match self {
            CharWidth::Narrow | CharWidth::Halfwidth | CharWidth::Neutral => 1,
            CharWidth::Wide | CharWidth::Fullwidth => 2,
            CharWidth::Ambiguous => 1,
        }
    }
}
/// Configuration for character solver
#[derive(Debug, Clone)]
pub struct CharConfig {
    /// Maximum code point to consider
    pub max_code_point: CodePoint,
    /// Enable Unicode normalization
    pub normalize_unicode: bool,
    /// Enable case folding
    pub case_folding: bool,
}
/// Character constraint kind
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharConstraint {
    /// Character equals a specific code point
    Eq(CodePoint),
    /// Character does not equal a specific code point
    Ne(CodePoint),
    /// Character is in a range [low, high]
    InRange(CodePoint, CodePoint),
    /// Character is not in a range
    NotInRange(CodePoint, CodePoint),
    /// Character is a digit (0-9)
    IsDigit,
    /// Character is a letter
    IsLetter,
    /// Character is alphanumeric
    IsAlphanumeric,
    /// Character is whitespace
    IsWhitespace,
    /// Character is uppercase
    IsUppercase,
    /// Character is lowercase
    IsLowercase,
    /// Character is ASCII
    IsAscii,
    /// Character has specific Unicode category
    HasCategory(UnicodeCategory),
    /// Two characters are equal
    EqVar(CharVar),
    /// Two characters are not equal
    NeVar(CharVar),
    /// Character is less than another
    LtVar(CharVar),
    /// Character is less than or equal to another
    LeVar(CharVar),
    /// Arithmetic: char1 + offset = char2
    AddOffset(i32, CharVar),
    /// To uppercase of another char
    ToUpper(CharVar),
    /// To lowercase of another char
    ToLower(CharVar),
}
/// Character normalizer (simplified implementation)
#[derive(Debug)]
pub struct CharNormalizer {
    /// Normalization form
    form: NormalizationForm,
    /// Canonical combining class cache
    ccc_cache: HashMap<CodePoint, u8>,
}
impl CharNormalizer {
    /// Create a new normalizer
    pub fn new(form: NormalizationForm) -> Self {
        Self {
            form,
            ccc_cache: HashMap::new(),
        }
    }
    /// Get canonical combining class (simplified)
    pub fn combining_class(&mut self, cp: CodePoint) -> u8 {
        if let Some(&ccc) = self.ccc_cache.get(&cp) {
            return ccc;
        }
        let ccc = match cp {
            0x0300..=0x036F => 230,
            0x0591..=0x05BD => 220,
            0x064B..=0x065F => 30,
            _ => 0,
        };
        self.ccc_cache.insert(cp, ccc);
        ccc
    }
    /// Check if a character is a combining mark
    pub fn is_combining(&self, cp: CodePoint) -> bool {
        matches!(
            cp, 0x0300..= 0x036F | 0x0483..= 0x0489 | 0x0591..= 0x05BD | 0x064B..= 0x065F
        )
    }
    /// Get NFD decomposition (simplified)
    pub fn decompose_nfd(&self, cp: CodePoint) -> Vec<CodePoint> {
        match cp {
            0x00C0 => vec![0x0041, 0x0300],
            0x00C1 => vec![0x0041, 0x0301],
            0x00C2 => vec![0x0041, 0x0302],
            0x00C3 => vec![0x0041, 0x0303],
            0x00C4 => vec![0x0041, 0x0308],
            0x00E0 => vec![0x0061, 0x0300],
            0x00E1 => vec![0x0061, 0x0301],
            0x00E2 => vec![0x0061, 0x0302],
            0x00E9 => vec![0x0065, 0x0301],
            0x00F1 => vec![0x006E, 0x0303],
            _ => vec![cp],
        }
    }
    /// Compose two code points (simplified)
    pub fn compose(&self, cp1: CodePoint, cp2: CodePoint) -> Option<CodePoint> {
        match (cp1, cp2) {
            (0x0041, 0x0300) => Some(0x00C0),
            (0x0041, 0x0301) => Some(0x00C1),
            (0x0041, 0x0302) => Some(0x00C2),
            (0x0041, 0x0303) => Some(0x00C3),
            (0x0041, 0x0308) => Some(0x00C4),
            (0x0061, 0x0300) => Some(0x00E0),
            (0x0061, 0x0301) => Some(0x00E1),
            (0x0061, 0x0302) => Some(0x00E2),
            (0x0065, 0x0301) => Some(0x00E9),
            (0x006E, 0x0303) => Some(0x00F1),
            _ => None,
        }
    }
    /// Normalize a sequence of code points
    pub fn normalize(&mut self, input: &[CodePoint]) -> Vec<CodePoint> {
        match self.form {
            NormalizationForm::Nfd | NormalizationForm::Nfkd => {
                self.normalize_decompose(input)
            }
            NormalizationForm::Nfc | NormalizationForm::Nfkc => {
                self.normalize_compose(input)
            }
        }
    }
    fn normalize_decompose(&self, input: &[CodePoint]) -> Vec<CodePoint> {
        let mut result = Vec::new();
        for &cp in input {
            result.extend(self.decompose_nfd(cp));
        }
        result
    }
    fn normalize_compose(&mut self, input: &[CodePoint]) -> Vec<CodePoint> {
        let decomposed = self.normalize_decompose(input);
        let mut result = Vec::new();
        let mut i = 0;
        while i < decomposed.len() {
            let cp = decomposed[i];
            if i + 1 < decomposed.len()
                && let Some(composed) = self.compose(cp, decomposed[i + 1])
            {
                result.push(composed);
                i += 2;
                continue;
            }
            result.push(cp);
            i += 1;
        }
        result
    }
    /// Get the normalization form
    pub fn form(&self) -> NormalizationForm {
        self.form
    }
}
/// Unicode normalization form
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalizationForm {
    /// Canonical Decomposition (NFD)
    Nfd,
    /// Canonical Composition (NFC)
    Nfc,
    /// Compatibility Decomposition (NFKD)
    Nfkd,
    /// Compatibility Composition (NFKC)
    Nfkc,
}
/// Unicode category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeCategory {
    /// Letter (L)
    Letter,
    /// Mark (M)
    Mark,
    /// Number (N)
    Number,
    /// Punctuation (P)
    Punctuation,
    /// Symbol (S)
    Symbol,
    /// Separator (Z)
    Separator,
    /// Other (C)
    Other,
}
impl UnicodeCategory {
    /// Get category for a code point
    pub fn from_code_point(cp: CodePoint) -> Self {
        if let Some(c) = char::from_u32(cp) {
            if c.is_alphabetic() {
                UnicodeCategory::Letter
            } else if c.is_numeric() {
                UnicodeCategory::Number
            } else if c.is_whitespace() {
                UnicodeCategory::Separator
            } else if c.is_ascii_punctuation() {
                UnicodeCategory::Punctuation
            } else if c.is_control() {
                UnicodeCategory::Other
            } else {
                UnicodeCategory::Symbol
            }
        } else {
            UnicodeCategory::Other
        }
    }
}
/// Result of character solver check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown
    Unknown,
}
/// Character class for regex-like matching
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharClass {
    /// Any single character
    Any,
    /// Literal character
    Literal(CodePoint),
    /// Range of characters
    Range(CodePoint, CodePoint),
    /// Negation of a class
    Negation(Box<CharClass>),
    /// Union of classes
    Union(Vec<CharClass>),
    /// Intersection of classes
    Intersection(Vec<CharClass>),
    /// POSIX class: `[:alpha:]`
    Alpha,
    /// POSIX class: `[:digit:]`
    Digit,
    /// POSIX class: `[:alnum:]`
    Alnum,
    /// POSIX class: `[:space:]`
    Space,
    /// POSIX class: `[:word:]` (letters, digits, underscore)
    Word,
    /// POSIX class: `[:punct:]`
    Punct,
    /// POSIX class: `[:print:]`
    Print,
    /// Unicode category class
    Category(UnicodeCategory),
    /// Unicode block class
    Block(UnicodeBlock),
    /// Unicode script class
    Script(UnicodeScript),
}
impl CharClass {
    /// Check if a code point matches this class
    pub fn matches(&self, cp: CodePoint) -> bool {
        match self {
            CharClass::Any => CharSolver::is_valid_code_point(cp),
            CharClass::Literal(lit) => cp == *lit,
            CharClass::Range(low, high) => cp >= *low && cp <= *high,
            CharClass::Negation(inner) => !inner.matches(cp),
            CharClass::Union(classes) => classes.iter().any(|c| c.matches(cp)),
            CharClass::Intersection(classes) => classes.iter().all(|c| c.matches(cp)),
            CharClass::Alpha => {
                char::from_u32(cp).map(|c| c.is_alphabetic()).unwrap_or(false)
            }
            CharClass::Digit => {
                char::from_u32(cp).map(|c| c.is_ascii_digit()).unwrap_or(false)
            }
            CharClass::Alnum => {
                char::from_u32(cp).map(|c| c.is_alphanumeric()).unwrap_or(false)
            }
            CharClass::Space => {
                char::from_u32(cp).map(|c| c.is_whitespace()).unwrap_or(false)
            }
            CharClass::Word => {
                char::from_u32(cp)
                    .map(|c| c.is_alphanumeric() || c == '_')
                    .unwrap_or(false)
            }
            CharClass::Punct => {
                char::from_u32(cp).map(|c| c.is_ascii_punctuation()).unwrap_or(false)
            }
            CharClass::Print => {
                char::from_u32(cp)
                    .map(|c| !c.is_control() && c != '\u{7F}')
                    .unwrap_or(false)
            }
            CharClass::Category(cat) => UnicodeCategory::from_code_point(cp) == *cat,
            CharClass::Block(block) => block.contains(cp),
            CharClass::Script(script) => UnicodeScript::from_code_point(cp) == *script,
        }
    }
    /// Get all code points matching this class (limited to ASCII for efficiency)
    pub fn enumerate_ascii(&self) -> Vec<CodePoint> {
        (0u32..=127).filter(|&cp| self.matches(cp)).collect()
    }
    /// Create a word character class
    pub fn word() -> Self {
        CharClass::Word
    }
    /// Create a digit character class
    pub fn digit() -> Self {
        CharClass::Digit
    }
    /// Create a whitespace class
    pub fn whitespace() -> Self {
        CharClass::Space
    }
    /// Create a class from a regex-like pattern
    pub fn from_pattern(pattern: &str) -> Option<Self> {
        match pattern {
            r"\d" => Some(CharClass::Digit),
            r"\w" => Some(CharClass::Word),
            r"\s" => Some(CharClass::Space),
            r"\D" => Some(CharClass::Negation(Box::new(CharClass::Digit))),
            r"\W" => Some(CharClass::Negation(Box::new(CharClass::Word))),
            r"\S" => Some(CharClass::Negation(Box::new(CharClass::Space))),
            "." => Some(CharClass::Any),
            _ => None,
        }
    }
}
/// Case folder for case-insensitive comparison
#[derive(Debug)]
pub struct CaseFolder {
    /// Folding mode
    mode: CaseFoldMode,
    /// Cache of folded characters
    cache: HashMap<CodePoint, Vec<CodePoint>>,
}
impl CaseFolder {
    /// Create a new case folder
    pub fn new(mode: CaseFoldMode) -> Self {
        Self {
            mode,
            cache: HashMap::new(),
        }
    }
    /// Fold a single code point
    pub fn fold(&mut self, cp: CodePoint) -> Vec<CodePoint> {
        if let Some(cached) = self.cache.get(&cp) {
            return cached.clone();
        }
        let result = self.compute_fold(cp);
        self.cache.insert(cp, result.clone());
        result
    }
    fn compute_fold(&self, cp: CodePoint) -> Vec<CodePoint> {
        match self.mode {
            CaseFoldMode::Simple => {
                if let Some(lower) = CharSolver::to_lowercase(cp) {
                    vec![lower]
                } else {
                    vec![cp]
                }
            }
            CaseFoldMode::Full => {
                match cp {
                    0x00DF => vec![0x0073, 0x0073],
                    0x0130 => vec![0x0069, 0x0307],
                    0x0149 => vec![0x02BC, 0x006E],
                    _ => {
                        if let Some(lower) = CharSolver::to_lowercase(cp) {
                            vec![lower]
                        } else {
                            vec![cp]
                        }
                    }
                }
            }
            CaseFoldMode::Turkic => {
                match cp {
                    0x0049 => vec![0x0131],
                    0x0130 => vec![0x0069],
                    _ => {
                        if let Some(lower) = CharSolver::to_lowercase(cp) {
                            vec![lower]
                        } else {
                            vec![cp]
                        }
                    }
                }
            }
        }
    }
    /// Compare two strings case-insensitively
    pub fn equals(&mut self, s1: &[CodePoint], s2: &[CodePoint]) -> bool {
        let f1: Vec<CodePoint> = s1.iter().flat_map(|&cp| self.fold(cp)).collect();
        let f2: Vec<CodePoint> = s2.iter().flat_map(|&cp| self.fold(cp)).collect();
        f1 == f2
    }
    /// Get the folding mode
    pub fn mode(&self) -> CaseFoldMode {
        self.mode
    }
}
/// Character theory solver
#[derive(Debug)]
pub struct CharSolver {
    /// Configuration
    config: CharConfig,
    /// Variable assignments
    assignments: HashMap<CharVar, CharValue>,
    /// Constraints per variable
    constraints: HashMap<CharVar, Vec<CharConstraint>>,
    /// All variables
    variables: HashSet<CharVar>,
    /// Next variable ID
    next_var: u32,
    /// Statistics
    stats: CharStats,
    /// Decision stack
    decision_stack: Vec<(CharVar, CharValue)>,
    /// Conflict
    has_conflict: bool,
}
impl CharSolver {
    /// Create a new character solver
    pub fn new() -> Self {
        Self {
            config: CharConfig::default(),
            assignments: HashMap::new(),
            constraints: HashMap::new(),
            variables: HashSet::new(),
            next_var: 0,
            stats: CharStats::default(),
            decision_stack: Vec::new(),
            has_conflict: false,
        }
    }
    /// Create with configuration
    pub fn with_config(config: CharConfig) -> Self {
        Self {
            config,
            assignments: HashMap::new(),
            constraints: HashMap::new(),
            variables: HashSet::new(),
            next_var: 0,
            stats: CharStats::default(),
            decision_stack: Vec::new(),
            has_conflict: false,
        }
    }
    /// Create a new character variable
    pub fn new_var(&mut self) -> CharVar {
        let var = CharVar::new(self.next_var);
        self.next_var += 1;
        self.variables.insert(var);
        var
    }
    /// Add a constraint for a variable
    pub fn add_constraint(&mut self, var: CharVar, constraint: CharConstraint) {
        self.constraints.entry(var).or_default().push(constraint);
        self.stats.constraints_added += 1;
    }
    /// Assign a value to a variable
    pub fn assign(&mut self, var: CharVar, value: CharValue) {
        self.assignments.insert(var, value);
        self.decision_stack.push((var, value));
    }
    /// Get the value of a variable
    pub fn get_value(&self, var: CharVar) -> Option<&CharValue> {
        self.assignments.get(&var)
    }
    /// Check satisfiability
    pub fn check(&mut self) -> CharResult {
        self.has_conflict = false;
        if !self.propagate() {
            self.stats.conflicts += 1;
            return CharResult::Unsat;
        }
        if self.all_assigned() {
            return CharResult::Sat;
        }
        while !self.all_assigned() {
            if let Some(var) = self.pick_unassigned() {
                if let Some(value) = self.find_valid_value(var) {
                    self.assign(var, value);
                    self.stats.decisions += 1;
                    if !self.propagate() {
                        self.stats.conflicts += 1;
                        return CharResult::Unsat;
                    }
                } else {
                    self.stats.conflicts += 1;
                    return CharResult::Unsat;
                }
            } else {
                break;
            }
        }
        CharResult::Sat
    }
    /// Propagate constraints
    fn propagate(&mut self) -> bool {
        let mut changed = true;
        while changed {
            changed = false;
            for var in self.variables.clone() {
                if let Some(value) = self.assignments.get(&var).copied() {
                    if !self.check_constraints(var, value) {
                        self.has_conflict = true;
                        return false;
                    }
                } else {
                    if let Some(inferred) = self.infer_value(var) {
                        self.assignments.insert(var, inferred);
                        self.stats.propagations += 1;
                        changed = true;
                    }
                }
            }
        }
        true
    }
    /// Check if all constraints are satisfied for a variable
    fn check_constraints(&self, var: CharVar, value: CharValue) -> bool {
        let constraints = match self.constraints.get(&var) {
            Some(c) => c,
            None => return true,
        };
        let cp = match value {
            CharValue::Known(cp) => cp,
            CharValue::Unknown => return true,
            CharValue::Invalid => return false,
        };
        for constraint in constraints {
            if !self.check_single_constraint(cp, constraint) {
                return false;
            }
        }
        true
    }
    /// Check a single constraint
    fn check_single_constraint(
        &self,
        cp: CodePoint,
        constraint: &CharConstraint,
    ) -> bool {
        match constraint {
            CharConstraint::Eq(expected) => cp == *expected,
            CharConstraint::Ne(forbidden) => cp != *forbidden,
            CharConstraint::InRange(low, high) => cp >= *low && cp <= *high,
            CharConstraint::NotInRange(low, high) => cp < *low || cp > *high,
            CharConstraint::IsDigit => {
                char::from_u32(cp).map(|c| c.is_ascii_digit()).unwrap_or(false)
            }
            CharConstraint::IsLetter => {
                char::from_u32(cp).map(|c| c.is_alphabetic()).unwrap_or(false)
            }
            CharConstraint::IsAlphanumeric => {
                char::from_u32(cp).map(|c| c.is_alphanumeric()).unwrap_or(false)
            }
            CharConstraint::IsWhitespace => {
                char::from_u32(cp).map(|c| c.is_whitespace()).unwrap_or(false)
            }
            CharConstraint::IsUppercase => {
                char::from_u32(cp).map(|c| c.is_uppercase()).unwrap_or(false)
            }
            CharConstraint::IsLowercase => {
                char::from_u32(cp).map(|c| c.is_lowercase()).unwrap_or(false)
            }
            CharConstraint::IsAscii => cp <= 127,
            CharConstraint::HasCategory(cat) => {
                UnicodeCategory::from_code_point(cp) == *cat
            }
            CharConstraint::EqVar(other) => {
                if let Some(CharValue::Known(other_cp)) = self.assignments.get(other) {
                    cp == *other_cp
                } else {
                    true
                }
            }
            CharConstraint::NeVar(other) => {
                if let Some(CharValue::Known(other_cp)) = self.assignments.get(other) {
                    cp != *other_cp
                } else {
                    true
                }
            }
            CharConstraint::LtVar(other) => {
                if let Some(CharValue::Known(other_cp)) = self.assignments.get(other) {
                    cp < *other_cp
                } else {
                    true
                }
            }
            CharConstraint::LeVar(other) => {
                if let Some(CharValue::Known(other_cp)) = self.assignments.get(other) {
                    cp <= *other_cp
                } else {
                    true
                }
            }
            CharConstraint::AddOffset(offset, result) => {
                let expected = (cp as i64 + *offset as i64) as u32;
                if let Some(CharValue::Known(result_cp)) = self.assignments.get(result) {
                    expected == *result_cp
                } else {
                    true
                }
            }
            CharConstraint::ToUpper(result) => {
                if let Some(c) = char::from_u32(cp) {
                    let upper: Vec<char> = c.to_uppercase().collect();
                    if upper.len() == 1
                        && let Some(CharValue::Known(result_cp)) = self
                            .assignments
                            .get(result)
                    {
                        return upper[0] as u32 == *result_cp;
                    }
                }
                true
            }
            CharConstraint::ToLower(result) => {
                if let Some(c) = char::from_u32(cp) {
                    let lower: Vec<char> = c.to_lowercase().collect();
                    if lower.len() == 1
                        && let Some(CharValue::Known(result_cp)) = self
                            .assignments
                            .get(result)
                    {
                        return lower[0] as u32 == *result_cp;
                    }
                }
                true
            }
        }
    }
    /// Try to infer a value for a variable
    fn infer_value(&self, var: CharVar) -> Option<CharValue> {
        let constraints = self.constraints.get(&var)?;
        for constraint in constraints {
            if let CharConstraint::Eq(cp) = constraint {
                return Some(CharValue::Known(*cp));
            }
            if let CharConstraint::EqVar(other) = constraint
                && let Some(CharValue::Known(cp)) = self.assignments.get(other)
            {
                return Some(CharValue::Known(*cp));
            }
        }
        None
    }
    /// Check if all variables are assigned
    fn all_assigned(&self) -> bool {
        self.variables.iter().all(|v| self.assignments.contains_key(v))
    }
    /// Pick an unassigned variable
    fn pick_unassigned(&self) -> Option<CharVar> {
        self.variables.iter().find(|v| !self.assignments.contains_key(v)).copied()
    }
    /// Find a valid value for a variable
    fn find_valid_value(&self, var: CharVar) -> Option<CharValue> {
        let constraints = self.constraints.get(&var);
        for cp in 0u32..=127 {
            let value = CharValue::Known(cp);
            if constraints
                .map(|cs| cs.iter().all(|c| self.check_single_constraint(cp, c)))
                .unwrap_or(true)
            {
                return Some(value);
            }
        }
        for cp in 128u32..=self.config.max_code_point {
            if char::from_u32(cp).is_none() {
                continue;
            }
            let value = CharValue::Known(cp);
            if constraints
                .map(|cs| cs.iter().all(|c| self.check_single_constraint(cp, c)))
                .unwrap_or(true)
            {
                return Some(value);
            }
        }
        None
    }
    /// Get the model (assignments)
    pub fn get_model(&self) -> &HashMap<CharVar, CharValue> {
        &self.assignments
    }
    /// Reset the solver
    pub fn reset(&mut self) {
        self.assignments.clear();
        self.decision_stack.clear();
        self.has_conflict = false;
    }
    /// Clear all constraints
    pub fn clear(&mut self) {
        self.reset();
        self.constraints.clear();
        self.variables.clear();
        self.next_var = 0;
    }
    /// Get statistics
    pub fn stats(&self) -> &CharStats {
        &self.stats
    }
    /// Number of variables
    pub fn num_vars(&self) -> usize {
        self.variables.len()
    }
    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.values().map(|v| v.len()).sum()
    }
    /// Check if a code point is a valid Unicode scalar value
    pub fn is_valid_code_point(cp: CodePoint) -> bool {
        char::from_u32(cp).is_some()
    }
    /// Convert character to uppercase
    pub fn to_uppercase(cp: CodePoint) -> Option<CodePoint> {
        char::from_u32(cp)
            .and_then(|c| {
                let upper: Vec<char> = c.to_uppercase().collect();
                if upper.len() == 1 { Some(upper[0] as u32) } else { None }
            })
    }
    /// Convert character to lowercase
    pub fn to_lowercase(cp: CodePoint) -> Option<CodePoint> {
        char::from_u32(cp)
            .and_then(|c| {
                let lower: Vec<char> = c.to_lowercase().collect();
                if lower.len() == 1 { Some(lower[0] as u32) } else { None }
            })
    }
    /// Get Unicode category of a character
    pub fn get_category(cp: CodePoint) -> UnicodeCategory {
        UnicodeCategory::from_code_point(cp)
    }
    /// Check if two characters are case-insensitive equal
    pub fn case_insensitive_eq(cp1: CodePoint, cp2: CodePoint) -> bool {
        let upper1 = Self::to_uppercase(cp1);
        let upper2 = Self::to_uppercase(cp2);
        upper1.is_some() && upper1 == upper2
    }
}
/// Advanced character solver with domain propagation
#[derive(Debug)]
pub struct AdvancedCharSolver {
    /// Base solver
    base: CharSolver,
    /// Domains for each variable
    domains: HashMap<CharVar, CharDomain>,
    /// Watch lists: constraint -> variables
    #[allow(dead_code)]
    watches: HashMap<usize, Vec<CharVar>>,
    /// Trail for backtracking
    trail: Vec<TrailEntry>,
    /// Decision level
    decision_level: u32,
}
impl AdvancedCharSolver {
    /// Create a new advanced solver
    pub fn new() -> Self {
        Self {
            base: CharSolver::new(),
            domains: HashMap::new(),
            watches: HashMap::new(),
            trail: Vec::new(),
            decision_level: 0,
        }
    }
    /// Create a new variable with full domain
    pub fn new_var(&mut self) -> CharVar {
        let var = self.base.new_var();
        self.domains.insert(var, CharDomain::full());
        var
    }
    /// Add a constraint
    pub fn add_constraint(&mut self, var: CharVar, constraint: CharConstraint) {
        self.apply_domain_constraint(var, &constraint);
        self.base.add_constraint(var, constraint);
    }
    /// Apply constraint to domain
    fn apply_domain_constraint(&mut self, var: CharVar, constraint: &CharConstraint) {
        let domain = self.domains.entry(var).or_insert_with(CharDomain::full);
        match constraint {
            CharConstraint::Eq(cp) => {
                *domain = CharDomain::singleton(*cp);
            }
            CharConstraint::Ne(cp) => {
                domain.exclude(*cp);
            }
            CharConstraint::InRange(low, high) => {
                domain.restrict_to_range(*low, *high);
            }
            CharConstraint::NotInRange(low, high) => {
                let mut new_domain = CharDomain::empty();
                if *low > 0 {
                    let below = CharDomain::range(0, low.saturating_sub(1));
                    new_domain = below;
                }
                if *high < 0x10FFFF {
                    let above = CharDomain::range(high + 1, 0x10FFFF);
                    if !new_domain.is_empty() {
                        new_domain.ranges.extend(above.ranges);
                    } else {
                        new_domain = above;
                    }
                }
                domain.intersect(&new_domain);
            }
            CharConstraint::IsDigit => {
                domain.restrict_to_range(b'0' as u32, b'9' as u32);
            }
            CharConstraint::IsLetter => {
                let mut letter_domain = CharDomain::range(b'A' as u32, b'Z' as u32);
                letter_domain.ranges.push((b'a' as u32, b'z' as u32));
                letter_domain.ranges.push((0x00C0, 0x024F));
                domain.intersect(&letter_domain);
            }
            CharConstraint::IsAscii => {
                domain.restrict_to_range(0, 127);
            }
            CharConstraint::IsUppercase => {
                domain.restrict_to_range(b'A' as u32, b'Z' as u32);
            }
            CharConstraint::IsLowercase => {
                domain.restrict_to_range(b'a' as u32, b'z' as u32);
            }
            CharConstraint::IsWhitespace => {
                let mut ws_domain = CharDomain::singleton(b' ' as u32);
                ws_domain.ranges.push((b'\t' as u32, b'\r' as u32));
                domain.intersect(&ws_domain);
            }
            _ => {}
        }
    }
    /// Get the domain for a variable
    pub fn get_domain(&self, var: CharVar) -> Option<&CharDomain> {
        self.domains.get(&var)
    }
    /// Check satisfiability with domain propagation
    pub fn check(&mut self) -> CharResult {
        for domain in self.domains.values() {
            if domain.is_empty() {
                return CharResult::Unsat;
            }
        }
        if !self.propagate() {
            return CharResult::Unsat;
        }
        for (&var, domain) in &self.domains {
            if let Some(cp) = domain.min() && self.base.get_value(var).is_none() {
                self.base.assign(var, CharValue::Known(cp));
            }
        }
        self.base.check()
    }
    /// Domain propagation
    fn propagate(&mut self) -> bool {
        let mut changed = true;
        while changed {
            changed = false;
            for domain in self.domains.values() {
                if domain.is_empty() {
                    return false;
                }
            }
            for (&var, constraints) in &self.base.constraints {
                for constraint in constraints {
                    if let CharConstraint::EqVar(other) = constraint
                        && let (Some(d1), Some(d2)) = (
                            self.domains.get(&var).cloned(),
                            self.domains.get(other).cloned(),
                        )
                    {
                        let mut new_d1 = d1.clone();
                        new_d1.intersect(&d2);
                        let mut new_d2 = d2.clone();
                        new_d2.intersect(&d1);
                        if let Some(dom) = self.domains.get_mut(&var)
                            && dom.size() != new_d1.size()
                        {
                            *dom = new_d1;
                            changed = true;
                        }
                        if let Some(dom) = self.domains.get_mut(other)
                            && dom.size() != new_d2.size()
                        {
                            *dom = new_d2;
                            changed = true;
                        }
                    }
                }
            }
        }
        true
    }
    /// Push a decision
    #[allow(dead_code)]
    fn push_decision(&mut self, var: CharVar, value: CodePoint) {
        self.decision_level += 1;
        if let Some(old_domain) = self.domains.get(&var).cloned() {
            self.trail
                .push(TrailEntry {
                    var,
                    old_domain,
                    level: self.decision_level,
                });
        }
        self.domains.insert(var, CharDomain::singleton(value));
    }
    /// Backtrack to a level
    #[allow(dead_code)]
    fn backtrack(&mut self, level: u32) {
        while let Some(entry) = self.trail.last() {
            if entry.level <= level {
                break;
            }
            let entry = self.trail.pop().expect("trail not empty");
            self.domains.insert(entry.var, entry.old_domain);
        }
        self.decision_level = level;
    }
    /// Reset the solver
    pub fn reset(&mut self) {
        self.base.reset();
        self.trail.clear();
        self.decision_level = 0;
        for domain in self.domains.values_mut() {
            *domain = CharDomain::full();
        }
    }
    /// Get statistics
    pub fn stats(&self) -> &CharStats {
        self.base.stats()
    }
}
/// Character variable ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharVar(pub u32);
impl CharVar {
    /// Create a new character variable
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}
/// Unicode script (writing system)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeScript {
    /// Latin script
    Latin,
    /// Greek script
    Greek,
    /// Cyrillic script
    Cyrillic,
    /// Arabic script
    Arabic,
    /// Hebrew script
    Hebrew,
    /// Devanagari script
    Devanagari,
    /// Thai script
    Thai,
    /// Han (CJK) script
    Han,
    /// Hiragana script
    Hiragana,
    /// Katakana script
    Katakana,
    /// Hangul script
    Hangul,
    /// Common (shared across scripts)
    Common,
    /// Unknown script
    Unknown,
}
impl UnicodeScript {
    /// Get the script for a code point
    pub fn from_code_point(cp: CodePoint) -> Self {
        match cp {
            0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x00FF | 0x0100..=0x024F => {
                UnicodeScript::Latin
            }
            0x0370..=0x03FF => UnicodeScript::Greek,
            0x0400..=0x04FF => UnicodeScript::Cyrillic,
            0x0600..=0x06FF => UnicodeScript::Arabic,
            0x0590..=0x05FF => UnicodeScript::Hebrew,
            0x0900..=0x097F => UnicodeScript::Devanagari,
            0x0E00..=0x0E7F => UnicodeScript::Thai,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF => UnicodeScript::Han,
            0x3040..=0x309F => UnicodeScript::Hiragana,
            0x30A0..=0x30FF => UnicodeScript::Katakana,
            0xAC00..=0xD7AF | 0x1100..=0x11FF => UnicodeScript::Hangul,
            0x0020..=0x0040 | 0x005B..=0x0060 | 0x007B..=0x007F => UnicodeScript::Common,
            _ => UnicodeScript::Unknown,
        }
    }
    /// Check if script uses left-to-right direction
    pub fn is_ltr(&self) -> bool {
        !matches!(self, UnicodeScript::Arabic | UnicodeScript::Hebrew)
    }
    /// Check if script is an East Asian script
    pub fn is_east_asian(&self) -> bool {
        matches!(
            self, UnicodeScript::Han | UnicodeScript::Hiragana | UnicodeScript::Katakana
            | UnicodeScript::Hangul
        )
    }
}
/// Character value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharValue {
    /// Known code point
    Known(CodePoint),
    /// Unknown (any valid code point)
    Unknown,
    /// Invalid
    Invalid,
}
impl CharValue {
    /// Check if this is a known value
    pub fn is_known(&self) -> bool {
        matches!(self, CharValue::Known(_))
    }
    /// Get the code point if known
    pub fn code_point(&self) -> Option<CodePoint> {
        match self {
            CharValue::Known(cp) => Some(*cp),
            _ => None,
        }
    }
    /// Convert to char if valid
    pub fn to_char(&self) -> Option<char> {
        match self {
            CharValue::Known(cp) => char::from_u32(*cp),
            _ => None,
        }
    }
}
/// Trail entry for backtracking
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TrailEntry {
    /// Variable
    var: CharVar,
    /// Old domain
    old_domain: CharDomain,
    /// Decision level at which this was pushed
    level: u32,
}
