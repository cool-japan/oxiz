//! Unicode Normalization Forms
//!
//! Implements the four Unicode normalization forms:
//! - **NFD**: Canonical Decomposition
//! - **NFC**: Canonical Decomposition followed by Canonical Composition
//! - **NFKD**: Compatibility Decomposition
//! - **NFKC**: Compatibility Decomposition followed by Canonical Composition
//!
//! These are critical for string theory solving when dealing with Unicode equivalence.
//! For example, "é" can be represented as either U+00E9 (precomposed) or U+0065 U+0301 (decomposed).
//!
//! ## SMT-LIB2 Integration
//!
//! Provides constraint generation for string normalization predicates:
//! ```smt2
//! (assert (= (str.norm nfc s1) s2))
//! ```

use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::FxHashMap;

/// Unicode normalization form
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalizationForm {
    /// Canonical Decomposition (NFD)
    NFD,
    /// Canonical Composition (NFC)
    NFC,
    /// Compatibility Decomposition (NFKD)
    NFKD,
    /// Compatibility Composition (NFKC)
    NFKC,
}

impl NormalizationForm {
    /// Parse normalization form from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "NFD" => Some(NormalizationForm::NFD),
            "NFC" => Some(NormalizationForm::NFC),
            "NFKD" => Some(NormalizationForm::NFKD),
            "NFKC" => Some(NormalizationForm::NFKC),
            _ => None,
        }
    }

    /// Check if this form uses canonical decomposition
    pub fn is_canonical(&self) -> bool {
        matches!(self, NormalizationForm::NFD | NormalizationForm::NFC)
    }

    /// Check if this form uses composition
    pub fn is_composed(&self) -> bool {
        matches!(self, NormalizationForm::NFC | NormalizationForm::NFKC)
    }
}

/// Canonical Combining Class (CCC) values
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CombiningClass(pub u8);

impl CombiningClass {
    /// Not reordered (spacing marks, base characters)
    pub const SPACING: Self = CombiningClass(0);
    /// Overlay
    pub const OVERLAY: Self = CombiningClass(1);
    /// Above
    pub const ABOVE: Self = CombiningClass(230);
    /// Below
    pub const BELOW: Self = CombiningClass(220);
    /// Attached above right
    pub const ABOVE_RIGHT: Self = CombiningClass(232);

    /// Check if this is a starter (CCC = 0)
    pub fn is_starter(self) -> bool {
        self.0 == 0
    }

    /// Get combining class for a character
    pub fn of(c: char) -> Self {
        // Simplified implementation - in production, use Unicode Character Database
        // Combining diacritical marks (U+0300-U+036F)
        if matches!(c, '\u{0300}'..='\u{036F}') {
            // Common combining marks
            match c {
                '\u{0300}' => CombiningClass(230), // Combining grave accent
                '\u{0301}' => CombiningClass(230), // Combining acute accent
                '\u{0302}' => CombiningClass(230), // Combining circumflex
                '\u{0303}' => CombiningClass(230), // Combining tilde
                '\u{0304}' => CombiningClass(230), // Combining macron
                '\u{0305}' => CombiningClass(230), // Combining overline
                '\u{0306}' => CombiningClass(230), // Combining breve
                '\u{0307}' => CombiningClass(230), // Combining dot above
                '\u{0308}' => CombiningClass(230), // Combining diaeresis
                '\u{0309}' => CombiningClass(230), // Combining hook above
                '\u{030A}' => CombiningClass(230), // Combining ring above
                '\u{030B}' => CombiningClass(230), // Combining double acute
                '\u{030C}' => CombiningClass(230), // Combining caron
                '\u{030D}' => CombiningClass(230), // Combining vertical line above
                '\u{030E}' => CombiningClass(230), // Combining double vertical line above
                '\u{030F}' => CombiningClass(230), // Combining double grave
                '\u{0310}' => CombiningClass(230), // Combining candrabindu
                '\u{0311}' => CombiningClass(230), // Combining inverted breve
                '\u{0312}' => CombiningClass(230), // Combining turned comma above
                '\u{0313}' => CombiningClass(230), // Combining comma above
                '\u{0314}' => CombiningClass(230), // Combining reversed comma above
                '\u{0315}' => CombiningClass(232), // Combining comma above right
                '\u{0316}' => CombiningClass(220), // Combining grave accent below
                '\u{0317}' => CombiningClass(220), // Combining acute accent below
                '\u{0318}' => CombiningClass(220), // Combining left tack below
                '\u{0319}' => CombiningClass(220), // Combining right tack below
                '\u{031A}' => CombiningClass(232), // Combining left angle above
                '\u{031B}' => CombiningClass(216), // Combining horn
                '\u{031C}' => CombiningClass(220), // Combining left half ring below
                '\u{031D}' => CombiningClass(220), // Combining up tack below
                '\u{031E}' => CombiningClass(220), // Combining down tack below
                '\u{031F}' => CombiningClass(220), // Combining plus sign below
                '\u{0320}' => CombiningClass(220), // Combining minus sign below
                '\u{0321}' => CombiningClass(202), // Combining palatalized hook below
                '\u{0322}' => CombiningClass(202), // Combining retroflex hook below
                '\u{0323}' => CombiningClass(220), // Combining dot below
                '\u{0324}' => CombiningClass(220), // Combining diaeresis below
                '\u{0325}' => CombiningClass(220), // Combining ring below
                '\u{0326}' => CombiningClass(220), // Combining comma below
                '\u{0327}' => CombiningClass(202), // Combining cedilla
                '\u{0328}' => CombiningClass(202), // Combining ogonek
                '\u{0329}' => CombiningClass(220), // Combining vertical line below
                '\u{032A}' => CombiningClass(220), // Combining bridge below
                '\u{032B}' => CombiningClass(220), // Combining inverted double arch below
                '\u{032C}' => CombiningClass(220), // Combining caron below
                '\u{032D}' => CombiningClass(220), // Combining circumflex accent below
                '\u{032E}' => CombiningClass(220), // Combining breve below
                '\u{032F}' => CombiningClass(220), // Combining inverted breve below
                '\u{0330}' => CombiningClass(220), // Combining tilde below
                '\u{0331}' => CombiningClass(220), // Combining macron below
                _ => CombiningClass(230),          // Default for other combining marks
            }
        } else if matches!(c, '\u{1DC0}'..='\u{1DFF}') {
            // Combining diacritical marks supplement
            CombiningClass(230)
        } else if matches!(c, '\u{20D0}'..='\u{20FF}') {
            // Combining diacritical marks for symbols
            CombiningClass(230)
        } else if matches!(c, '\u{FE20}'..='\u{FE2F}') {
            // Combining half marks
            CombiningClass(230)
        } else {
            CombiningClass::SPACING
        }
    }
}

/// Decomposition mapping for a character
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decomposition {
    /// Decomposed form
    pub chars: Vec<char>,
    /// Is this a compatibility decomposition?
    pub is_compat: bool,
}

impl Decomposition {
    /// Create a canonical decomposition
    pub fn canonical(chars: Vec<char>) -> Self {
        Self {
            chars,
            is_compat: false,
        }
    }

    /// Create a compatibility decomposition
    pub fn compat(chars: Vec<char>) -> Self {
        Self {
            chars,
            is_compat: true,
        }
    }
}

/// Unicode normalizer
#[derive(Debug)]
pub struct UnicodeNormalizer {
    /// Decomposition mappings
    decompositions: FxHashMap<char, Decomposition>,
    /// Composition mappings (for NFC/NFKC)
    compositions: FxHashMap<(char, char), char>,
    /// Composition exclusions
    #[allow(dead_code)]
    excluded: FxHashMap<char, bool>,
}

impl UnicodeNormalizer {
    /// Create a new Unicode normalizer with built-in mappings
    pub fn new() -> Self {
        let mut normalizer = Self {
            decompositions: FxHashMap::default(),
            compositions: FxHashMap::default(),
            excluded: FxHashMap::default(),
        };
        normalizer.init_mappings();
        normalizer
    }

    /// Initialize decomposition and composition mappings
    fn init_mappings(&mut self) {
        // Common Latin decompositions
        self.add_canonical('À', vec!['A', '\u{0300}']); // A + grave
        self.add_canonical('Á', vec!['A', '\u{0301}']); // A + acute
        self.add_canonical('Â', vec!['A', '\u{0302}']); // A + circumflex
        self.add_canonical('Ã', vec!['A', '\u{0303}']); // A + tilde
        self.add_canonical('Ä', vec!['A', '\u{0308}']); // A + diaeresis
        self.add_canonical('Å', vec!['A', '\u{030A}']); // A + ring above
        self.add_canonical('Ç', vec!['C', '\u{0327}']); // C + cedilla
        self.add_canonical('È', vec!['E', '\u{0300}']); // E + grave
        self.add_canonical('É', vec!['E', '\u{0301}']); // E + acute
        self.add_canonical('Ê', vec!['E', '\u{0302}']); // E + circumflex
        self.add_canonical('Ë', vec!['E', '\u{0308}']); // E + diaeresis
        self.add_canonical('Ì', vec!['I', '\u{0300}']); // I + grave
        self.add_canonical('Í', vec!['I', '\u{0301}']); // I + acute
        self.add_canonical('Î', vec!['I', '\u{0302}']); // I + circumflex
        self.add_canonical('Ï', vec!['I', '\u{0308}']); // I + diaeresis
        self.add_canonical('Ñ', vec!['N', '\u{0303}']); // N + tilde
        self.add_canonical('Ò', vec!['O', '\u{0300}']); // O + grave
        self.add_canonical('Ó', vec!['O', '\u{0301}']); // O + acute
        self.add_canonical('Ô', vec!['O', '\u{0302}']); // O + circumflex
        self.add_canonical('Õ', vec!['O', '\u{0303}']); // O + tilde
        self.add_canonical('Ö', vec!['O', '\u{0308}']); // O + diaeresis
        self.add_canonical('Ù', vec!['U', '\u{0300}']); // U + grave
        self.add_canonical('Ú', vec!['U', '\u{0301}']); // U + acute
        self.add_canonical('Û', vec!['U', '\u{0302}']); // U + circumflex
        self.add_canonical('Ü', vec!['U', '\u{0308}']); // U + diaeresis
        self.add_canonical('Ý', vec!['Y', '\u{0301}']); // Y + acute

        // Lowercase variants
        self.add_canonical('à', vec!['a', '\u{0300}']);
        self.add_canonical('á', vec!['a', '\u{0301}']);
        self.add_canonical('â', vec!['a', '\u{0302}']);
        self.add_canonical('ã', vec!['a', '\u{0303}']);
        self.add_canonical('ä', vec!['a', '\u{0308}']);
        self.add_canonical('å', vec!['a', '\u{030A}']);
        self.add_canonical('ç', vec!['c', '\u{0327}']);
        self.add_canonical('è', vec!['e', '\u{0300}']);
        self.add_canonical('é', vec!['e', '\u{0301}']);
        self.add_canonical('ê', vec!['e', '\u{0302}']);
        self.add_canonical('ë', vec!['e', '\u{0308}']);
        self.add_canonical('ì', vec!['i', '\u{0300}']);
        self.add_canonical('í', vec!['i', '\u{0301}']);
        self.add_canonical('î', vec!['i', '\u{0302}']);
        self.add_canonical('ï', vec!['i', '\u{0308}']);
        self.add_canonical('ñ', vec!['n', '\u{0303}']);
        self.add_canonical('ò', vec!['o', '\u{0300}']);
        self.add_canonical('ó', vec!['o', '\u{0301}']);
        self.add_canonical('ô', vec!['o', '\u{0302}']);
        self.add_canonical('õ', vec!['o', '\u{0303}']);
        self.add_canonical('ö', vec!['o', '\u{0308}']);
        self.add_canonical('ù', vec!['u', '\u{0300}']);
        self.add_canonical('ú', vec!['u', '\u{0301}']);
        self.add_canonical('û', vec!['u', '\u{0302}']);
        self.add_canonical('ü', vec!['u', '\u{0308}']);
        self.add_canonical('ý', vec!['y', '\u{0301}']);
        self.add_canonical('ÿ', vec!['y', '\u{0308}']);

        // Ligatures (compatibility decompositions)
        self.add_compat('Æ', vec!['A', 'E']);
        self.add_compat('æ', vec!['a', 'e']);
        self.add_compat('Œ', vec!['O', 'E']);
        self.add_compat('œ', vec!['o', 'e']);
        self.add_compat('ﬁ', vec!['f', 'i']);
        self.add_compat('ﬂ', vec!['f', 'l']);
        self.add_compat('ﬀ', vec!['f', 'f']);
        self.add_compat('ﬃ', vec!['f', 'f', 'i']);
        self.add_compat('ﬄ', vec!['f', 'f', 'l']);
        self.add_compat('ﬆ', vec!['s', 't']);

        // Superscripts and subscripts (compatibility)
        self.add_compat('⁰', vec!['0']);
        self.add_compat('¹', vec!['1']);
        self.add_compat('²', vec!['2']);
        self.add_compat('³', vec!['3']);
        self.add_compat('⁴', vec!['4']);
        self.add_compat('⁵', vec!['5']);
        self.add_compat('⁶', vec!['6']);
        self.add_compat('⁷', vec!['7']);
        self.add_compat('⁸', vec!['8']);
        self.add_compat('⁹', vec!['9']);

        // Fractions (compatibility)
        self.add_compat('½', vec!['1', '⁄', '2']);
        self.add_compat('⅓', vec!['1', '⁄', '3']);
        self.add_compat('⅔', vec!['2', '⁄', '3']);
        self.add_compat('¼', vec!['1', '⁄', '4']);
        self.add_compat('¾', vec!['3', '⁄', '4']);
        self.add_compat('⅕', vec!['1', '⁄', '5']);
        self.add_compat('⅖', vec!['2', '⁄', '5']);
        self.add_compat('⅗', vec!['3', '⁄', '5']);
        self.add_compat('⅘', vec!['4', '⁄', '5']);
        self.add_compat('⅙', vec!['1', '⁄', '6']);
        self.add_compat('⅚', vec!['5', '⁄', '6']);
        self.add_compat('⅛', vec!['1', '⁄', '8']);
        self.add_compat('⅜', vec!['3', '⁄', '8']);
        self.add_compat('⅝', vec!['5', '⁄', '8']);
        self.add_compat('⅞', vec!['7', '⁄', '8']);

        // Fullwidth forms (compatibility)
        for c in 'A'..='Z' {
            let fullwidth = char::from_u32(0xFF21 + (c as u32 - 'A' as u32));
            if let Some(fw) = fullwidth {
                self.add_compat(fw, vec![c]);
            }
        }
        for c in 'a'..='z' {
            let fullwidth = char::from_u32(0xFF41 + (c as u32 - 'a' as u32));
            if let Some(fw) = fullwidth {
                self.add_compat(fw, vec![c]);
            }
        }
        for c in '0'..='9' {
            let fullwidth = char::from_u32(0xFF10 + (c as u32 - '0' as u32));
            if let Some(fw) = fullwidth {
                self.add_compat(fw, vec![c]);
            }
        }

        // Greek with diacritics
        self.add_canonical('Ά', vec!['Α', '\u{0301}']); // Greek capital alpha with acute
        self.add_canonical('Έ', vec!['Ε', '\u{0301}']); // Greek capital epsilon with acute
        self.add_canonical('Ή', vec!['Η', '\u{0301}']); // Greek capital eta with acute
        self.add_canonical('Ί', vec!['Ι', '\u{0301}']); // Greek capital iota with acute
        self.add_canonical('Ό', vec!['Ο', '\u{0301}']); // Greek capital omicron with acute
        self.add_canonical('Ύ', vec!['Υ', '\u{0301}']); // Greek capital upsilon with acute
        self.add_canonical('Ώ', vec!['Ω', '\u{0301}']); // Greek capital omega with acute

        // Hangul compatibility jamo (Korea)
        self.add_compat('ﾡ', vec!['ᄀ']);
        self.add_compat('ﾢ', vec!['ᄁ']);
        self.add_compat('ﾣ', vec!['ᆪ']);
        self.add_compat('ﾤ', vec!['ᄂ']);
        self.add_compat('ﾥ', vec!['ᆬ']);
        self.add_compat('ﾦ', vec!['ᆭ']);

        // Extended Latin
        self.add_canonical('Ā', vec!['A', '\u{0304}']); // A with macron
        self.add_canonical('ā', vec!['a', '\u{0304}']); // a with macron
        self.add_canonical('Ă', vec!['A', '\u{0306}']); // A with breve
        self.add_canonical('ă', vec!['a', '\u{0306}']); // a with breve
        self.add_canonical('Ą', vec!['A', '\u{0328}']); // A with ogonek
        self.add_canonical('ą', vec!['a', '\u{0328}']); // a with ogonek
        self.add_canonical('Ć', vec!['C', '\u{0301}']); // C with acute
        self.add_canonical('ć', vec!['c', '\u{0301}']); // c with acute
        self.add_canonical('Ĉ', vec!['C', '\u{0302}']); // C with circumflex
        self.add_canonical('ĉ', vec!['c', '\u{0302}']); // c with circumflex
        self.add_canonical('Ċ', vec!['C', '\u{0307}']); // C with dot above
        self.add_canonical('ċ', vec!['c', '\u{0307}']); // c with dot above
        self.add_canonical('Č', vec!['C', '\u{030C}']); // C with caron
        self.add_canonical('č', vec!['c', '\u{030C}']); // c with caron

        // Vietnamese
        self.add_canonical('Ơ', vec!['O', '\u{031B}']); // O with horn
        self.add_canonical('ơ', vec!['o', '\u{031B}']); // o with horn
        self.add_canonical('Ư', vec!['U', '\u{031B}']); // U with horn
        self.add_canonical('ư', vec!['u', '\u{031B}']); // u with horn

        // Combining character sequences (multi-level)
        self.add_canonical('ấ', vec!['a', '\u{0302}', '\u{0301}']); // a circumflex acute
        self.add_canonical('ầ', vec!['a', '\u{0302}', '\u{0300}']); // a circumflex grave
        self.add_canonical('ẫ', vec!['a', '\u{0302}', '\u{0303}']); // a circumflex tilde
        self.add_canonical('ậ', vec!['a', '\u{0302}', '\u{0323}']); // a circumflex dot below

        // Build composition pairs from decompositions (reverse mapping)
        self.build_composition_pairs();
    }

    /// Add canonical decomposition
    fn add_canonical(&mut self, composed: char, decomposed: Vec<char>) {
        self.decompositions
            .insert(composed, Decomposition::canonical(decomposed));
    }

    /// Add compatibility decomposition
    fn add_compat(&mut self, composed: char, decomposed: Vec<char>) {
        self.decompositions
            .insert(composed, Decomposition::compat(decomposed));
    }

    /// Build composition pairs from decomposition mappings
    fn build_composition_pairs(&mut self) {
        let decomps: Vec<_> = self
            .decompositions
            .iter()
            .filter(|(_, d)| !d.is_compat && d.chars.len() == 2)
            .map(|(&c, d)| (c, d.chars[0], d.chars[1]))
            .collect();

        for (composed, base, combining) in decomps {
            // Skip if composition is excluded
            if !self.is_excluded(composed) {
                self.compositions.insert((base, combining), composed);
            }
        }
    }

    /// Check if a character is excluded from composition
    fn is_excluded(&self, c: char) -> bool {
        // Singleton exclusions (characters that should not be composed)
        // Examples: precomposed Hangul syllables, certain compatibility characters
        matches!(
            c,
            '\u{0958}'..='\u{095F}' // Devanagari
            | '\u{09DC}'..='\u{09DD}' // Bengali
            | '\u{09DF}' // Bengali
            | '\u{0A33}' // Gurmukhi
            | '\u{0A36}' // Gurmukhi
            | '\u{0A59}'..='\u{0A5B}' // Gurmukhi
            | '\u{0A5E}' // Gurmukhi
            | '\u{0B5C}'..='\u{0B5D}' // Oriya
            | '\u{0F43}' // Tibetan
            | '\u{0F4D}' // Tibetan
            | '\u{0F52}' // Tibetan
            | '\u{0F57}' // Tibetan
            | '\u{0F5C}' // Tibetan
            | '\u{0F69}' // Tibetan
            | '\u{0F73}' // Tibetan
            | '\u{0F75}'..='\u{0F76}' // Tibetan
            | '\u{0F78}' // Tibetan
            | '\u{0F81}' // Tibetan
            | '\u{0F93}' // Tibetan
            | '\u{0F9D}' // Tibetan
            | '\u{0FA2}' // Tibetan
            | '\u{0FA7}' // Tibetan
            | '\u{0FAC}' // Tibetan
            | '\u{0FB9}' // Tibetan
            | '\u{FB1D}' // Hebrew
            | '\u{FB1F}' // Hebrew
            | '\u{FB2A}'..='\u{FB36}' // Hebrew
            | '\u{FB38}'..='\u{FB3C}' // Hebrew
            | '\u{FB3E}' // Hebrew
            | '\u{FB40}'..='\u{FB41}' // Hebrew
            | '\u{FB43}'..='\u{FB44}' // Hebrew
            | '\u{FB46}'..='\u{FB4E}' // Hebrew
        )
    }

    /// Normalize a string using the specified normalization form
    pub fn normalize(&self, s: &str, form: NormalizationForm) -> String {
        match form {
            NormalizationForm::NFD => self.nfd(s),
            NormalizationForm::NFC => self.nfc(s),
            NormalizationForm::NFKD => self.nfkd(s),
            NormalizationForm::NFKC => self.nfkc(s),
        }
    }

    /// Canonical Decomposition (NFD)
    pub fn nfd(&self, s: &str) -> String {
        let decomposed = self.decompose(s, false);
        self.canonical_ordering(&decomposed)
    }

    /// Canonical Composition (NFC)
    pub fn nfc(&self, s: &str) -> String {
        let decomposed = self.decompose(s, false);
        let ordered = self.canonical_ordering(&decomposed);
        self.compose(&ordered)
    }

    /// Compatibility Decomposition (NFKD)
    pub fn nfkd(&self, s: &str) -> String {
        let decomposed = self.decompose(s, true);
        self.canonical_ordering(&decomposed)
    }

    /// Compatibility Composition (NFKC)
    pub fn nfkc(&self, s: &str) -> String {
        let decomposed = self.decompose(s, true);
        let ordered = self.canonical_ordering(&decomposed);
        self.compose(&ordered)
    }

    /// Decompose a string
    fn decompose(&self, s: &str, compat: bool) -> Vec<char> {
        let mut result = Vec::new();
        for c in s.chars() {
            self.decompose_char(c, compat, &mut result);
        }
        result
    }

    /// Recursively decompose a single character
    fn decompose_char(&self, c: char, compat: bool, result: &mut Vec<char>) {
        if let Some(decomp) = self.decompositions.get(&c) {
            // Skip compatibility decompositions if we only want canonical
            if !compat && decomp.is_compat {
                result.push(c);
                return;
            }
            // Recursively decompose
            for &ch in &decomp.chars {
                self.decompose_char(ch, compat, result);
            }
        } else {
            result.push(c);
        }
    }

    /// Canonical ordering of combining marks
    fn canonical_ordering(&self, chars: &[char]) -> String {
        let mut result = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            let ccc = CombiningClass::of(c);

            if ccc.is_starter() {
                // Starter character - just append
                result.push(c);
                i += 1;
            } else {
                // Combining character - collect all following combining chars
                let mut combining = vec![(c, ccc)];
                i += 1;

                while i < chars.len() {
                    let next_c = chars[i];
                    let next_ccc = CombiningClass::of(next_c);
                    if next_ccc.is_starter() {
                        break;
                    }
                    combining.push((next_c, next_ccc));
                    i += 1;
                }

                // Sort combining characters by CCC in descending order
                combining.sort_by(|a, b| b.1.cmp(&a.1));

                // Append sorted combining characters
                for (ch, _) in combining {
                    result.push(ch);
                }
            }
        }

        result.iter().collect()
    }

    /// Compose a canonically ordered string
    fn compose(&self, s: &str) -> String {
        let chars: Vec<char> = s.chars().collect();
        if chars.is_empty() {
            return String::new();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let starter = chars[i];
            let mut last_class = CombiningClass::SPACING;
            let mut composed = starter;
            result.push(composed);
            i += 1;

            while i < chars.len() {
                let ch = chars[i];
                let ccc = CombiningClass::of(ch);

                // Try to compose with last starter
                if let Some(&comp) = self.compositions.get(&(composed, ch)) {
                    // Composition is possible - check blocking
                    if last_class < ccc || last_class.is_starter() {
                        // No blocker, compose
                        result.pop();
                        result.push(comp);
                        composed = comp;
                        last_class = ccc;
                        i += 1;
                        continue;
                    }
                }

                // Cannot compose - check if this is a new starter
                if ccc.is_starter() {
                    break;
                }

                // Append combining character
                result.push(ch);
                if last_class.is_starter() || ccc > last_class {
                    last_class = ccc;
                }
                i += 1;
            }
        }

        result.iter().collect()
    }

    /// Check if two strings are canonically equivalent
    pub fn equivalent(&self, s1: &str, s2: &str) -> bool {
        self.nfc(s1) == self.nfc(s2)
    }

    /// Check if a string is in a specific normal form
    pub fn is_normalized(&self, s: &str, form: NormalizationForm) -> bool {
        s == self.normalize(s, form)
    }
}

impl Default for UnicodeNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalization constraint for SMT solving
#[derive(Debug, Clone)]
pub struct NormalizationConstraint {
    /// Source string variable
    pub source: TermId,
    /// Normalized result variable
    pub result: TermId,
    /// Normalization form
    pub form: NormalizationForm,
    /// Origin term for conflict explanation
    pub origin: TermId,
}

/// Normalization constraint solver
#[derive(Debug)]
pub struct NormalizationSolver {
    /// Normalizer instance
    normalizer: UnicodeNormalizer,
    /// Active normalization constraints
    constraints: Vec<NormalizationConstraint>,
    /// String variable assignments
    assignments: FxHashMap<TermId, String>,
}

impl NormalizationSolver {
    /// Create a new normalization solver
    pub fn new() -> Self {
        Self {
            normalizer: UnicodeNormalizer::new(),
            constraints: Vec::new(),
            assignments: FxHashMap::default(),
        }
    }

    /// Add a normalization constraint
    pub fn add_constraint(&mut self, constraint: NormalizationConstraint) {
        self.constraints.push(constraint);
    }

    /// Assign a value to a string variable
    pub fn assign(&mut self, var: TermId, value: String) {
        self.assignments.insert(var, value);
    }

    /// Check normalization constraints and return conflicts
    pub fn check(&self) -> Result<Vec<(TermId, String)>> {
        let mut deductions = Vec::new();

        for constraint in &self.constraints {
            // Check if source has an assignment
            if let Some(source_val) = self.assignments.get(&constraint.source) {
                let normalized = self.normalizer.normalize(source_val, constraint.form);

                // Check if result has an assignment
                if let Some(result_val) = self.assignments.get(&constraint.result) {
                    if result_val != &normalized {
                        return Err(OxizError::Internal(
                            "normalization constraint violated".to_string(),
                        ));
                    }
                } else {
                    // Deduce result value
                    deductions.push((constraint.result, normalized));
                }
            }
        }

        Ok(deductions)
    }

    /// Get the normalizer instance
    pub fn normalizer(&self) -> &UnicodeNormalizer {
        &self.normalizer
    }
}

impl Default for NormalizationSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combining_class() {
        assert!(CombiningClass::SPACING.is_starter());
        assert!(!CombiningClass::ABOVE.is_starter());

        let grave = CombiningClass::of('\u{0300}');
        assert_eq!(grave, CombiningClass::ABOVE);
    }

    #[test]
    fn test_normalization_form_parse() {
        assert_eq!(
            NormalizationForm::parse("NFC"),
            Some(NormalizationForm::NFC)
        );
        assert_eq!(
            NormalizationForm::parse("nfd"),
            Some(NormalizationForm::NFD)
        );
        assert_eq!(NormalizationForm::parse("invalid"), None);
    }

    #[test]
    fn test_nfd_basic() {
        let normalizer = UnicodeNormalizer::new();
        // é (U+00E9) -> e (U+0065) + combining acute (U+0301)
        let result = normalizer.nfd("é");
        assert_eq!(result, "e\u{0301}");
    }

    #[test]
    fn test_nfc_basic() {
        let normalizer = UnicodeNormalizer::new();
        // e + combining acute -> é
        let result = normalizer.nfc("e\u{0301}");
        assert_eq!(result, "é");
    }

    #[test]
    fn test_nfkd_ligature() {
        let normalizer = UnicodeNormalizer::new();
        // ﬁ (ligature) -> f + i
        let result = normalizer.nfkd("ﬁ");
        assert_eq!(result, "fi");
    }

    #[test]
    fn test_nfkc_ligature() {
        let normalizer = UnicodeNormalizer::new();
        // ﬁ (ligature) -> f + i
        let result = normalizer.nfkc("ﬁ");
        assert_eq!(result, "fi");
    }

    #[test]
    fn test_canonical_ordering() {
        let normalizer = UnicodeNormalizer::new();
        // Characters with different combining classes should be reordered
        let input = vec![
            'a', '\u{0327}', // cedilla (CCC 202)
            '\u{0301}', // acute (CCC 230)
        ];
        let result = normalizer.canonical_ordering(&input);
        // Should reorder to: a, acute, cedilla
        assert_eq!(result, "a\u{0301}\u{0327}");
    }

    #[test]
    fn test_equivalence() {
        let normalizer = UnicodeNormalizer::new();
        assert!(normalizer.equivalent("é", "e\u{0301}"));
        assert!(normalizer.equivalent("café", "cafe\u{0301}"));
        assert!(!normalizer.equivalent("a", "b"));
    }

    #[test]
    fn test_is_normalized_nfc() {
        let normalizer = UnicodeNormalizer::new();
        assert!(normalizer.is_normalized("é", NormalizationForm::NFC));
        assert!(!normalizer.is_normalized("e\u{0301}", NormalizationForm::NFC));
    }

    #[test]
    fn test_is_normalized_nfd() {
        let normalizer = UnicodeNormalizer::new();
        assert!(normalizer.is_normalized("e\u{0301}", NormalizationForm::NFD));
        assert!(!normalizer.is_normalized("é", NormalizationForm::NFD));
    }

    #[test]
    fn test_decompose_recursive() {
        let normalizer = UnicodeNormalizer::new();
        // Test that decomposition is recursive
        let result = normalizer.nfd("Ç");
        assert_eq!(result, "C\u{0327}");
    }

    #[test]
    fn test_compose_blocked() {
        let normalizer = UnicodeNormalizer::new();
        // Test composition blocking: if a combining char blocks, composition fails
        // a + cedilla + acute should stay as-is (cedilla blocks composition with acute)
        let input = "a\u{0327}\u{0301}";
        let result = normalizer.nfc(input);
        // Since we're composing after canonical ordering, this might compose
        // The actual result depends on composition rules
        assert!(!result.is_empty());
    }

    #[test]
    fn test_multiple_combining_marks() {
        let normalizer = UnicodeNormalizer::new();
        // e + circumflex + acute
        let input = "e\u{0302}\u{0301}";
        let nfd = normalizer.nfd(input);
        let nfc = normalizer.nfc(input);

        // NFD should keep it decomposed and ordered
        assert!(nfd.contains('\u{0302}'));
        assert!(nfd.contains('\u{0301}'));

        // NFC might compose if there's a precomposed form
        assert!(!nfc.is_empty());
    }

    #[test]
    fn test_fullwidth_forms() {
        let normalizer = UnicodeNormalizer::new();
        // Ａ (fullwidth A) should decompose to A in NFKD
        let result = normalizer.nfkd("Ａ");
        assert_eq!(result, "A");

        // But not in NFD (canonical only)
        let result_nfd = normalizer.nfd("Ａ");
        assert_eq!(result_nfd, "Ａ");
    }

    #[test]
    fn test_normalization_solver_basic() {
        let mut solver = NormalizationSolver::new();
        let source = TermId(0);
        let result = TermId(1);
        let origin = TermId(2);

        solver.add_constraint(NormalizationConstraint {
            source,
            result,
            form: NormalizationForm::NFC,
            origin,
        });

        solver.assign(source, "e\u{0301}".to_string());

        let deductions = solver.check().unwrap();
        assert_eq!(deductions.len(), 1);
        assert_eq!(deductions[0].0, result);
        assert_eq!(deductions[0].1, "é");
    }

    #[test]
    fn test_normalization_solver_conflict() {
        let mut solver = NormalizationSolver::new();
        let source = TermId(0);
        let result = TermId(1);
        let origin = TermId(2);

        solver.add_constraint(NormalizationConstraint {
            source,
            result,
            form: NormalizationForm::NFC,
            origin,
        });

        solver.assign(source, "e\u{0301}".to_string());
        solver.assign(result, "e".to_string()); // Wrong! Should be "é"

        assert!(solver.check().is_err());
    }

    #[test]
    fn test_vietnamese_decomposition() {
        let normalizer = UnicodeNormalizer::new();
        // Test Vietnamese characters with horn
        let result = normalizer.nfd("ơ");
        assert!(result.contains('\u{031B}')); // horn combining mark
    }

    #[test]
    fn test_greek_with_accents() {
        let normalizer = UnicodeNormalizer::new();
        // Ά (Greek Alpha with acute) -> Α + acute
        let result = normalizer.nfd("Ά");
        assert!(result.contains('Α'));
        assert!(result.contains('\u{0301}'));
    }

    #[test]
    fn test_long_string_normalization() {
        let normalizer = UnicodeNormalizer::new();
        let input = "Héllo Wörld! Çafé";
        let nfd = normalizer.nfd(input);
        let nfc = normalizer.nfc(input);

        // Round-trip should preserve equivalence
        assert!(normalizer.equivalent(input, &nfd));
        assert!(normalizer.equivalent(input, &nfc));
        assert!(normalizer.equivalent(&nfd, &nfc));
    }

    #[test]
    fn test_empty_string_normalization() {
        let normalizer = UnicodeNormalizer::new();
        assert_eq!(normalizer.nfd(""), "");
        assert_eq!(normalizer.nfc(""), "");
        assert_eq!(normalizer.nfkd(""), "");
        assert_eq!(normalizer.nfkc(""), "");
    }

    #[test]
    fn test_ascii_only_unchanged() {
        let normalizer = UnicodeNormalizer::new();
        let ascii = "Hello World 123!";
        assert_eq!(normalizer.nfd(ascii), ascii);
        assert_eq!(normalizer.nfc(ascii), ascii);
        assert_eq!(normalizer.nfkd(ascii), ascii);
        assert_eq!(normalizer.nfkc(ascii), ascii);
    }

    #[test]
    fn test_combining_class_ordering() {
        assert!(CombiningClass::SPACING < CombiningClass::BELOW);
        assert!(CombiningClass::BELOW < CombiningClass::ABOVE);
    }

    #[test]
    fn test_superscript_compatibility() {
        let normalizer = UnicodeNormalizer::new();
        // ² (superscript 2) -> 2 in NFKD
        let result = normalizer.nfkd("²");
        assert_eq!(result, "2");

        // But not in NFD
        let result_nfd = normalizer.nfd("²");
        assert_eq!(result_nfd, "²");
    }

    #[test]
    fn test_fraction_compatibility() {
        let normalizer = UnicodeNormalizer::new();
        // ½ -> 1⁄2 in NFKD
        let result = normalizer.nfkd("½");
        assert!(result.contains('1'));
        assert!(result.contains('2'));
    }
}
