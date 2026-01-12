//! Unicode character class support for regular expressions
//!
//! Provides Unicode General Category support for regex character classes
//! like \p{L} (Letters), \p{N} (Numbers), etc.

/// Unicode General Category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeCategory {
    /// Letter, any kind
    Letter,
    /// Letter, lowercase
    LowercaseLetter,
    /// Letter, uppercase
    UppercaseLetter,
    /// Letter, titlecase
    TitlecaseLetter,
    /// Letter, modifier
    ModifierLetter,
    /// Letter, other
    OtherLetter,

    /// Mark, any kind
    Mark,
    /// Mark, nonspacing
    NonspacingMark,
    /// Mark, spacing combining
    SpacingMark,
    /// Mark, enclosing
    EnclosingMark,

    /// Number, any kind
    Number,
    /// Number, decimal digit
    DecimalNumber,
    /// Number, letter
    LetterNumber,
    /// Number, other
    OtherNumber,

    /// Punctuation, any kind
    Punctuation,
    /// Punctuation, connector
    ConnectorPunctuation,
    /// Punctuation, dash
    DashPunctuation,
    /// Punctuation, open
    OpenPunctuation,
    /// Punctuation, close
    ClosePunctuation,
    /// Punctuation, initial quote
    InitialPunctuation,
    /// Punctuation, final quote
    FinalPunctuation,
    /// Punctuation, other
    OtherPunctuation,

    /// Symbol, any kind
    Symbol,
    /// Symbol, math
    MathSymbol,
    /// Symbol, currency
    CurrencySymbol,
    /// Symbol, modifier
    ModifierSymbol,
    /// Symbol, other
    OtherSymbol,

    /// Separator, any kind
    Separator,
    /// Separator, space
    SpaceSeparator,
    /// Separator, line
    LineSeparator,
    /// Separator, paragraph
    ParagraphSeparator,

    /// Other, any kind
    Other,
    /// Other, control
    Control,
    /// Other, format
    Format,
    /// Other, surrogate
    Surrogate,
    /// Other, private use
    PrivateUse,
    /// Other, not assigned
    Unassigned,
}

impl UnicodeCategory {
    /// Check if a character belongs to this category
    #[must_use]
    pub fn contains(&self, c: char) -> bool {
        use UnicodeCategory::*;

        match self {
            // Letters
            Letter => c.is_alphabetic(),
            LowercaseLetter => c.is_lowercase(),
            UppercaseLetter => c.is_uppercase(),
            TitlecaseLetter => {
                // Titlecase check - characters that are uppercase but used in title position
                // Examples: Dž, Lj, Nj (digraphs)
                matches!(c, '\u{01C5}' | '\u{01C8}' | '\u{01CB}' | '\u{01F2}')
            }
            ModifierLetter => {
                // Modifier letters (Lm) - spacing characters that indicate modifications
                matches!(c as u32,
                    0x02B0..=0x02C1 | 0x02C6..=0x02D1 | 0x02E0..=0x02E4 |
                    0x02EC | 0x02EE | 0x0374 | 0x037A | 0x0559 |
                    0x0640 | 0x06E5..=0x06E6 | 0x07F4..=0x07F5 | 0x07FA |
                    0x081A | 0x0824 | 0x0828 | 0x0971 | 0x0E46 |
                    0x0EC6 | 0x10FC | 0x17D7 | 0x1843 | 0x1AA7 |
                    0x1C78..=0x1C7D | 0x1D2C..=0x1D6A | 0x1D78 |
                    0x1D9B..=0x1DBF | 0x2071 | 0x207F | 0x2090..=0x209C |
                    0x2C7C..=0x2C7D | 0x2D6F | 0x2E2F | 0x3005 |
                    0x3031..=0x3035 | 0x303B | 0x309D..=0x309E |
                    0x30FC..=0x30FE | 0xA015 | 0xA4F8..=0xA4FD |
                    0xA60C | 0xA67F | 0xA69C..=0xA69D | 0xA717..=0xA71F |
                    0xA770 | 0xA788 | 0xA7F8..=0xA7F9 | 0xA9CF |
                    0xA9E6 | 0xAA70 | 0xAADD | 0xAAF3..=0xAAF4 |
                    0xAB5C..=0xAB5F | 0xFF70 | 0xFF9E..=0xFF9F |
                    0x16B40..=0x16B43 | 0x16F93..=0x16F9F |
                    0x16FE0..=0x16FE1 | 0x1E137..=0x1E13D | 0x1E944..=0x1E946
                )
            }
            OtherLetter => {
                // Other letters (Lo) - ideographs, syllables, etc.
                c.is_alphabetic()
                    && !c.is_lowercase()
                    && !c.is_uppercase()
                    && !matches!(c, '\u{01C5}' | '\u{01C8}' | '\u{01CB}' | '\u{01F2}')
            }

            // Marks
            Mark => is_mark(c),
            NonspacingMark => is_nonspacing_mark(c),
            SpacingMark => is_spacing_mark(c),
            EnclosingMark => is_enclosing_mark(c),

            // Numbers
            Number => c.is_numeric(),
            DecimalNumber => c.is_ascii_digit() || is_decimal_number(c),
            LetterNumber => is_letter_number(c),
            OtherNumber => c.is_numeric() && !is_decimal_number(c) && !is_letter_number(c),

            // Punctuation
            Punctuation => is_punctuation(c),
            ConnectorPunctuation => {
                matches!(c, '_' | '\u{203F}'..='\u{2040}' | '\u{2054}' | '\u{FE33}'..='\u{FE34}' | '\u{FE4D}'..='\u{FE4F}' | '\u{FF3F}')
            }
            DashPunctuation => {
                matches!(c, '-' | '\u{058A}' | '\u{05BE}' | '\u{1400}' | '\u{1806}' | '\u{2010}'..='\u{2015}' | '\u{2E17}' | '\u{2E1A}' | '\u{2E3A}'..='\u{2E3B}' | '\u{2E40}' | '\u{301C}' | '\u{3030}' | '\u{30A0}' | '\u{FE31}'..='\u{FE32}' | '\u{FE58}' | '\u{FE63}' | '\u{FF0D}')
            }
            OpenPunctuation => is_open_punctuation(c),
            ClosePunctuation => is_close_punctuation(c),
            InitialPunctuation => matches!(
                c,
                '\u{00AB}' | '\u{2018}' | '\u{201B}'
                    ..='\u{201C}'
                        | '\u{201F}'
                        | '\u{2039}'
                        | '\u{2E02}'
                        | '\u{2E04}'
                        | '\u{2E09}'
                        | '\u{2E0C}'
                        | '\u{2E1C}'
                        | '\u{2E20}'
            ),
            FinalPunctuation => matches!(
                c,
                '\u{00BB}'
                    | '\u{2019}'
                    | '\u{201D}'
                    | '\u{203A}'
                    | '\u{2E03}'
                    | '\u{2E05}'
                    | '\u{2E0A}'
                    | '\u{2E0D}'
                    | '\u{2E1D}'
                    | '\u{2E21}'
            ),
            OtherPunctuation => {
                is_punctuation(c) && !is_open_punctuation(c) && !is_close_punctuation(c)
            }

            // Symbols
            Symbol => is_symbol(c),
            MathSymbol => is_math_symbol(c),
            CurrencySymbol => {
                matches!(c, '$' | '\u{00A2}'..='\u{00A5}' | '\u{058F}' | '\u{060B}' | '\u{09F2}'..='\u{09F3}' | '\u{09FB}' | '\u{0AF1}' | '\u{0BF9}' | '\u{0E3F}' | '\u{17DB}' | '\u{20A0}'..='\u{20BF}' | '\u{A838}' | '\u{FDFC}' | '\u{FE69}' | '\u{FF04}' | '\u{FFE0}'..='\u{FFE1}' | '\u{FFE5}'..='\u{FFE6}')
            }
            ModifierSymbol => is_modifier_symbol(c),
            OtherSymbol => is_symbol(c) && !is_math_symbol(c) && !is_modifier_symbol(c),

            // Separators
            Separator => c.is_whitespace(),
            SpaceSeparator => matches!(
                c,
                ' ' | '\u{00A0}' | '\u{1680}' | '\u{2000}'
                    ..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
            ),
            LineSeparator => c == '\u{2028}',
            ParagraphSeparator => c == '\u{2029}',

            // Other
            Other => is_other(c),
            Control => c.is_control(),
            Format => is_format(c),
            Surrogate => is_surrogate(c),
            PrivateUse => is_private_use(c),
            Unassigned => is_unassigned(c),
        }
    }

    /// Parse a Unicode category from a string (e.g., "L", "Ll", "Letter")
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        use UnicodeCategory::*;

        match s {
            "L" | "Letter" => Some(Letter),
            "Ll" | "LowercaseLetter" | "Lowercase_Letter" => Some(LowercaseLetter),
            "Lu" | "UppercaseLetter" | "Uppercase_Letter" => Some(UppercaseLetter),
            "Lt" | "TitlecaseLetter" | "Titlecase_Letter" => Some(TitlecaseLetter),
            "Lm" | "ModifierLetter" | "Modifier_Letter" => Some(ModifierLetter),
            "Lo" | "OtherLetter" | "Other_Letter" => Some(OtherLetter),

            "M" | "Mark" => Some(Mark),
            "Mn" | "NonspacingMark" | "Nonspacing_Mark" => Some(NonspacingMark),
            "Mc" | "SpacingMark" | "Spacing_Mark" => Some(SpacingMark),
            "Me" | "EnclosingMark" | "Enclosing_Mark" => Some(EnclosingMark),

            "N" | "Number" => Some(Number),
            "Nd" | "DecimalNumber" | "Decimal_Number" => Some(DecimalNumber),
            "Nl" | "LetterNumber" | "Letter_Number" => Some(LetterNumber),
            "No" | "OtherNumber" | "Other_Number" => Some(OtherNumber),

            "P" | "Punctuation" => Some(Punctuation),
            "Pc" | "ConnectorPunctuation" | "Connector_Punctuation" => Some(ConnectorPunctuation),
            "Pd" | "DashPunctuation" | "Dash_Punctuation" => Some(DashPunctuation),
            "Ps" | "OpenPunctuation" | "Open_Punctuation" => Some(OpenPunctuation),
            "Pe" | "ClosePunctuation" | "Close_Punctuation" => Some(ClosePunctuation),
            "Pi" | "InitialPunctuation" | "Initial_Punctuation" => Some(InitialPunctuation),
            "Pf" | "FinalPunctuation" | "Final_Punctuation" => Some(FinalPunctuation),
            "Po" | "OtherPunctuation" | "Other_Punctuation" => Some(OtherPunctuation),

            "S" | "Symbol" => Some(Symbol),
            "Sm" | "MathSymbol" | "Math_Symbol" => Some(MathSymbol),
            "Sc" | "CurrencySymbol" | "Currency_Symbol" => Some(CurrencySymbol),
            "Sk" | "ModifierSymbol" | "Modifier_Symbol" => Some(ModifierSymbol),
            "So" | "OtherSymbol" | "Other_Symbol" => Some(OtherSymbol),

            "Z" | "Separator" => Some(Separator),
            "Zs" | "SpaceSeparator" | "Space_Separator" => Some(SpaceSeparator),
            "Zl" | "LineSeparator" | "Line_Separator" => Some(LineSeparator),
            "Zp" | "ParagraphSeparator" | "Paragraph_Separator" => Some(ParagraphSeparator),

            "C" | "Other" => Some(Other),
            "Cc" | "Control" => Some(Control),
            "Cf" | "Format" => Some(Format),
            "Cs" | "Surrogate" => Some(Surrogate),
            "Co" | "PrivateUse" | "Private_Use" => Some(PrivateUse),
            "Cn" | "Unassigned" => Some(Unassigned),

            _ => None,
        }
    }
}

// Helper functions for Unicode categories

#[inline]
fn is_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF |
        0x20D0..=0x20FF | 0xFE20..=0xFE2F
    )
}

#[inline]
fn is_nonspacing_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x1DC0..=0x1DFF | 0x20D0..=0x20F0 | 0xFE20..=0xFE2F
    )
}

#[inline]
fn is_spacing_mark(c: char) -> bool {
    matches!(c as u32,
        0x0903 | 0x093B | 0x093E..=0x0940 | 0x0949..=0x094C | 0x094E..=0x094F
    )
}

#[inline]
fn is_enclosing_mark(c: char) -> bool {
    matches!(c as u32, 0x0488..=0x0489 | 0x1ABE | 0x20DD..=0x20E0 | 0x20E2..=0x20E4 | 0xA670..=0xA672)
}

#[inline]
fn is_decimal_number(c: char) -> bool {
    matches!(c as u32,
        0x0030..=0x0039 | 0x0660..=0x0669 | 0x06F0..=0x06F9 |
        0x07C0..=0x07C9 | 0x0966..=0x096F | 0x09E6..=0x09EF |
        0x0A66..=0x0A6F | 0x0AE6..=0x0AEF | 0x0B66..=0x0B6F |
        0x0BE6..=0x0BEF | 0x0C66..=0x0C6F | 0x0CE6..=0x0CEF |
        0x0D66..=0x0D6F | 0x0DE6..=0x0DEF | 0x0E50..=0x0E59 |
        0x0ED0..=0x0ED9 | 0x0F20..=0x0F29 | 0x1040..=0x1049 |
        0x1090..=0x1099 | 0x17E0..=0x17E9 | 0x1810..=0x1819 |
        0x1946..=0x194F | 0x19D0..=0x19D9 | 0x1A80..=0x1A89 |
        0x1A90..=0x1A99 | 0x1B50..=0x1B59 | 0x1BB0..=0x1BB9 |
        0x1C40..=0x1C49 | 0x1C50..=0x1C59 | 0xA620..=0xA629 |
        0xA8D0..=0xA8D9 | 0xA900..=0xA909 | 0xA9D0..=0xA9D9 |
        0xA9F0..=0xA9F9 | 0xAA50..=0xAA59 | 0xABF0..=0xABF9 |
        0xFF10..=0xFF19
    )
}

#[inline]
fn is_letter_number(c: char) -> bool {
    matches!(c as u32,
        0x16EE..=0x16F0 | 0x2160..=0x2182 | 0x2185..=0x2188 |
        0x3007 | 0x3021..=0x3029 | 0x3038..=0x303A |
        0xA6E6..=0xA6EF | 0x10140..=0x10174 | 0x10341 |
        0x1034A | 0x103D1..=0x103D5 | 0x12400..=0x1246E
    )
}

#[inline]
fn is_punctuation(c: char) -> bool {
    matches!(
        c,
        '!' | '"'
            | '#'
            | '%'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '_'
            | '{'
            | '}'
            | '\u{00A1}'
            | '\u{00A7}'
            | '\u{00AB}'
            | '\u{00B6}'..='\u{00B7}' | '\u{00BB}' | '\u{00BF}' | '\u{037E}' | '\u{0387}'
    ) || matches!(c as u32, 0x055A..=0x055F | 0x0589..=0x058A | 0x05BE | 0x05C0 | 0x05C3 | 0x05C6)
}

#[inline]
fn is_open_punctuation(c: char) -> bool {
    matches!(
        c,
        '(' | '['
            | '{'
            | '\u{0F3A}'
            | '\u{0F3C}'
            | '\u{169B}'
            | '\u{201A}'
            | '\u{201E}'
            | '\u{2045}'
            | '\u{207D}'
            | '\u{208D}'
            | '\u{2308}'
            | '\u{230A}'
            | '\u{2329}'
            | '\u{2768}'
            | '\u{276A}'
            | '\u{276C}'
            | '\u{276E}'
            | '\u{2770}'
            | '\u{2772}'
            | '\u{2774}'
            | '\u{27C5}'
            | '\u{27E6}'
            | '\u{27E8}'
            | '\u{27EA}'
            | '\u{27EC}'
            | '\u{27EE}'
            | '\u{2983}'
            | '\u{2985}'
            | '\u{2987}'
            | '\u{2989}'
            | '\u{298B}'
            | '\u{298D}'
            | '\u{298F}'
            | '\u{2991}'
            | '\u{2993}'
            | '\u{2995}'
            | '\u{2997}'
            | '\u{29D8}'
            | '\u{29DA}'
            | '\u{29FC}'
            | '\u{2E22}'
            | '\u{2E24}'
            | '\u{2E26}'
            | '\u{2E28}'
            | '\u{2E42}'
            | '\u{3008}'
            | '\u{300A}'
            | '\u{300C}'
            | '\u{300E}'
            | '\u{3010}'
            | '\u{3014}'
            | '\u{3016}'
            | '\u{3018}'
            | '\u{301A}'
            | '\u{301D}'
            | '\u{FD3E}'
            | '\u{FE17}'
            | '\u{FE35}'
            | '\u{FE37}'
            | '\u{FE39}'
            | '\u{FE3B}'
            | '\u{FE3D}'
            | '\u{FE3F}'
            | '\u{FE41}'
            | '\u{FE43}'
            | '\u{FE47}'
            | '\u{FE59}'
            | '\u{FE5B}'
            | '\u{FE5D}'
            | '\u{FF08}'
            | '\u{FF3B}'
            | '\u{FF5B}'
            | '\u{FF5F}'
            | '\u{FF62}'
    )
}

#[inline]
fn is_close_punctuation(c: char) -> bool {
    matches!(
        c,
        ')' | ']'
            | '}'
            | '\u{0F3B}'
            | '\u{0F3D}'
            | '\u{169C}'
            | '\u{2046}'
            | '\u{207E}'
            | '\u{208E}'
            | '\u{2309}'
            | '\u{230B}'
            | '\u{232A}'
            | '\u{2769}'
            | '\u{276B}'
            | '\u{276D}'
            | '\u{276F}'
            | '\u{2771}'
            | '\u{2773}'
            | '\u{2775}'
            | '\u{27C6}'
            | '\u{27E7}'
            | '\u{27E9}'
            | '\u{27EB}'
            | '\u{27ED}'
            | '\u{27EF}'
            | '\u{2984}'
            | '\u{2986}'
            | '\u{2988}'
            | '\u{298A}'
            | '\u{298C}'
            | '\u{298E}'
            | '\u{2990}'
            | '\u{2992}'
            | '\u{2994}'
            | '\u{2996}'
            | '\u{2998}'
            | '\u{29D9}'
            | '\u{29DB}'
            | '\u{29FD}'
            | '\u{2E23}'
            | '\u{2E25}'
            | '\u{2E27}'
            | '\u{2E29}'
            | '\u{3009}'
            | '\u{300B}'
            | '\u{300D}'
            | '\u{300F}'
            | '\u{3011}'
            | '\u{3015}'
            | '\u{3017}'
            | '\u{3019}'
            | '\u{301B}'
            | '\u{301E}'
            ..='\u{301F}'
                | '\u{FD3F}'
                | '\u{FE18}'
                | '\u{FE36}'
                | '\u{FE38}'
                | '\u{FE3A}'
                | '\u{FE3C}'
                | '\u{FE3E}'
                | '\u{FE40}'
                | '\u{FE42}'
                | '\u{FE44}'
                | '\u{FE48}'
                | '\u{FE5A}'
                | '\u{FE5C}'
                | '\u{FE5E}'
                | '\u{FF09}'
                | '\u{FF3D}'
                | '\u{FF5D}'
                | '\u{FF60}'
                | '\u{FF63}'
    )
}

#[inline]
fn is_symbol(c: char) -> bool {
    matches!(c,
        '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~' |
        '\u{00A2}'..='\u{00A6}' | '\u{00A8}'..='\u{00A9}' | '\u{00AC}' |
        '\u{00AE}'..='\u{00B1}' | '\u{00B4}' | '\u{00B8}' | '\u{00D7}' | '\u{00F7}'
    )
}

#[inline]
fn is_math_symbol(c: char) -> bool {
    matches!(
        c,
        '+' | '<' | '=' | '>' | '|' | '~' | '\u{00AC}' | '\u{00B1}' | '\u{00D7}' | '\u{00F7}'
    ) || matches!(c as u32, 0x2200..=0x22FF | 0x2A00..=0x2AFF | 0x27C0..=0x27EF | 0x2980..=0x29FF)
}

#[inline]
fn is_modifier_symbol(c: char) -> bool {
    matches!(
        c,
        '^' | '`' | '\u{00A8}' | '\u{00AF}' | '\u{00B4}' | '\u{00B8}'
    ) || matches!(c as u32, 0x02C2..=0x02C5 | 0x02D2..=0x02DF | 0x02E5..=0x02EB | 0x02ED | 0x02EF..=0x02FF)
}

#[inline]
fn is_format(c: char) -> bool {
    matches!(c as u32,
        0x00AD | 0x0600..=0x0605 | 0x061C | 0x06DD | 0x070F |
        0x180E | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 |
        0x2066..=0x206F | 0xFEFF | 0xFFF9..=0xFFFB |
        0x110BD | 0x1BCA0..=0x1BCA3 | 0x1D173..=0x1D17A |
        0xE0001 | 0xE0020..=0xE007F
    )
}

#[inline]
fn is_surrogate(c: char) -> bool {
    matches!(c as u32, 0xD800..=0xDFFF)
}

#[inline]
fn is_private_use(c: char) -> bool {
    matches!(c as u32,
        0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD
    )
}

#[inline]
fn is_unassigned(c: char) -> bool {
    // Simplified: just check against defined ranges
    !is_assigned(c)
}

#[inline]
fn is_assigned(c: char) -> bool {
    // Simplified check - in real implementation would use Unicode database
    let cp = c as u32;
    cp <= 0x10FFFF && !is_surrogate(c) && !is_private_use(c)
}

#[inline]
fn is_other(c: char) -> bool {
    is_format(c) || is_surrogate(c) || is_private_use(c) || is_unassigned(c) || c.is_control()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letter_category() {
        assert!(UnicodeCategory::Letter.contains('a'));
        assert!(UnicodeCategory::Letter.contains('Z'));
        assert!(UnicodeCategory::Letter.contains('α')); // Greek
        assert!(!UnicodeCategory::Letter.contains('1'));
        assert!(!UnicodeCategory::Letter.contains(' '));
    }

    #[test]
    fn test_number_category() {
        assert!(UnicodeCategory::Number.contains('0'));
        assert!(UnicodeCategory::Number.contains('9'));
        assert!(UnicodeCategory::DecimalNumber.contains('5'));
        assert!(!UnicodeCategory::Number.contains('a'));
    }

    #[test]
    fn test_punctuation_category() {
        assert!(UnicodeCategory::Punctuation.contains('.'));
        assert!(UnicodeCategory::Punctuation.contains(','));
        assert!(UnicodeCategory::Punctuation.contains('!'));
        assert!(!UnicodeCategory::Punctuation.contains('a'));
    }

    #[test]
    fn test_category_parse() {
        assert_eq!(UnicodeCategory::parse("L"), Some(UnicodeCategory::Letter));
        assert_eq!(
            UnicodeCategory::parse("Ll"),
            Some(UnicodeCategory::LowercaseLetter)
        );
        assert_eq!(
            UnicodeCategory::parse("Nd"),
            Some(UnicodeCategory::DecimalNumber)
        );
        assert_eq!(UnicodeCategory::parse("Invalid"), None);
    }

    #[test]
    fn test_separator_category() {
        assert!(UnicodeCategory::Separator.contains(' '));
        assert!(UnicodeCategory::SpaceSeparator.contains(' '));
        assert!(!UnicodeCategory::Separator.contains('a'));
    }
}
