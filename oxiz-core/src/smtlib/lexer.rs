//! SMT-LIB2 Lexer

#[allow(unused_imports)]
use crate::prelude::*;

/// Largest code point in the alphabet of the SMT-LIB 2.6 Unicode Strings
/// theory (`0x2FFFF`, i.e. 196607 — the BMP plus the two next planes).
/// A `\u` escape denoting a larger value is **not** an escape sequence at
/// all, so its backslash stands for itself.
///
/// Reference: Z3's `zstring.h` (`unicode_max_char()`).
pub(crate) const MAX_STRING_CODE_POINT: u32 = 0x2_FFFF;

/// Value of a single ASCII hexadecimal digit byte, or `None`.
fn hex_digit_value(byte: u8) -> Option<u32> {
    match byte {
        b'0'..=b'9' => Some(u32::from(byte - b'0')),
        b'a'..=b'f' => Some(u32::from(byte - b'a') + 10),
        b'A'..=b'F' => Some(u32::from(byte - b'A') + 10),
        _ => None,
    }
}

/// Try to read an SMT-LIB Unicode-Strings escape sequence at the start of
/// `input` (which must begin at a `\`), returning `(code_point, byte_len)`.
///
/// The SMT-LIB 2.6 Unicode Strings theory defines exactly two escape forms
/// inside a string literal:
///
/// * `\ud₃d₂d₁d₀` — exactly four hexadecimal digits;
/// * `\u{d₀}` … `\u{d₄d₃d₂d₁d₀}` — one to five hexadecimal digits in braces.
///
/// Both denote the character whose code point is that hexadecimal number,
/// which must be in the range `0..=0x2FFFF`. **Any other occurrence of `\` is not an
/// escape sequence and stands for itself** — including `\u{}`, a six-digit
/// braced form, a too-large code point, and a `\` followed by anything else.
///
/// Reference: Z3's `zstring.cpp` (`zstring::is_escape_char`), mirrored here
/// digit-for-digit so both accept exactly the same literals.
fn scan_unicode_escape(input: &str) -> Option<(u32, usize)> {
    // Only ASCII bytes are inspected, and every byte of a multi-byte UTF-8
    // sequence is >= 0x80, so byte indexing can never split a character.
    let bytes = input.as_bytes();
    if bytes.first() != Some(&b'\\') || bytes.get(1) != Some(&b'u') {
        return None;
    }

    // Braced form `\u{d...}`; `\u{}` is explicitly excluded (as in Z3, which
    // requires the byte after `{` to differ from `}`).
    if bytes.get(2) == Some(&b'{') && bytes.get(3).is_some_and(|b| *b != b'}') {
        let mut value: u32 = 0;
        // At most five digits fit, so the sixth inspected byte must be `}`
        // for the sequence to be an escape at all.
        for i in 0..6 {
            let byte = *bytes.get(3 + i)?;
            if let Some(digit) = hex_digit_value(byte) {
                value = value * 16 + digit;
            } else if byte == b'}' {
                return (value <= MAX_STRING_CODE_POINT).then_some((value, 4 + i));
            } else {
                return None;
            }
        }
        return None;
    }

    // Unbraced form `\ud₃d₂d₁d₀`: exactly four hexadecimal digits.
    let mut value: u32 = 0;
    for i in 0..4 {
        value = value * 16 + hex_digit_value(*bytes.get(2 + i)?)?;
    }
    (value <= MAX_STRING_CODE_POINT).then_some((value, 6))
}

/// Token kind
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Left parenthesis
    LParen,
    /// Right parenthesis
    RParen,
    /// Symbol (identifier)
    Symbol(String),
    /// Keyword (prefixed with :)
    Keyword(String),
    /// Numeral (integer)
    Numeral(String),
    /// Decimal (floating point)
    Decimal(String),
    /// Hexadecimal (#x...)
    Hexadecimal(String),
    /// Binary (#b...)
    Binary(String),
    /// String literal
    StringLit(String),
    /// End of file
    Eof,
}

/// A token with position information
#[derive(Debug, Clone)]
pub struct Token {
    /// The kind of token
    pub kind: TokenKind,
    /// Start position in input
    pub start: usize,
    /// End position in input
    pub end: usize,
}

/// A lexical error detected while scanning.
///
/// `Lexer::next_token` keeps returning `Some(Token)` even when it hits one
/// of these conditions (unterminated string/quoted-symbol literals used to
/// be accepted silently, consuming the rest of the input as their content;
/// a bare `#` not followed by `x`/`X`/`b`/`B` used to be accepted silently
/// as a one-character `Symbol`, even though `#` cannot start a valid
/// SMT-LIB symbol) so that callers depending on the existing `Option<Token>`
/// token stream keep working unchanged. Callers that want to reject
/// malformed input should check [`Lexer::errors`] after lexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    /// Human-readable description of the problem.
    pub message: String,
    /// Byte offset where the problem was detected.
    pub pos: usize,
}

/// Lexer for SMT-LIB2
pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
    /// Lexical errors accumulated so far. See [`LexError`] for why these
    /// don't abort tokenization outright.
    errors: Vec<LexError>,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            errors: Vec::new(),
        }
    }

    /// Get the current position
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Lexical errors accumulated so far (unterminated string/quoted-symbol
    /// literals, bare `#` tokens, ...). Empty for well-formed input.
    #[must_use]
    pub fn errors(&self) -> &[LexError] {
        &self.errors
    }

    /// Whether any lexical error has been recorded so far.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Peek at the next token without consuming it
    #[must_use]
    pub fn peek(&self) -> Option<Token> {
        let mut lexer = Self {
            input: self.input,
            pos: self.pos,
            errors: Vec::new(),
        };
        lexer.next_token()
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Option<Token> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.input.len() {
            return Some(Token {
                kind: TokenKind::Eof,
                start: self.pos,
                end: self.pos,
            });
        }

        let start = self.pos;
        let remaining = &self.input[self.pos..];
        let first = remaining.chars().next()?;

        let kind = match first {
            '(' => {
                self.pos += 1;
                TokenKind::LParen
            }
            ')' => {
                self.pos += 1;
                TokenKind::RParen
            }
            ':' => {
                self.pos += 1;
                let sym = self.read_symbol_chars();
                TokenKind::Keyword(sym)
            }
            '"' => {
                self.pos += 1;
                let s = self.read_string_lit(start);
                TokenKind::StringLit(s)
            }
            '#' => {
                self.pos += 1;
                if let Some(next) = remaining.chars().nth(1) {
                    match next {
                        'x' | 'X' => {
                            self.pos += 1;
                            let hex = self.read_hex_chars();
                            TokenKind::Hexadecimal(hex)
                        }
                        'b' | 'B' => {
                            self.pos += 1;
                            let bin = self.read_binary_chars();
                            TokenKind::Binary(bin)
                        }
                        _ => {
                            // A bare `#` (not `#x...`/`#b...`) cannot start a
                            // valid SMT-LIB symbol; record it instead of
                            // silently minting a one-character `Symbol("#")`
                            // that would otherwise surface downstream as a
                            // confusing "undefined symbol" rather than a
                            // lex-level error.
                            self.errors.push(LexError {
                                message: "bare '#' is not a valid token (expected #x... or #b...)"
                                    .to_string(),
                                pos: start,
                            });
                            TokenKind::Symbol("#".to_string())
                        }
                    }
                } else {
                    self.errors.push(LexError {
                        message: "unexpected end of input after '#' (expected #x... or #b...)"
                            .to_string(),
                        pos: start,
                    });
                    TokenKind::Symbol("#".to_string())
                }
            }
            '0'..='9' => {
                let num = self.read_numeral();
                // SMT-LIB `<numeral>` grammar: `0 | a non-empty sequence of
                // digits not starting with 0`. A leading zero followed by
                // more digits (e.g. `007`) is not a valid numeral — record
                // it as a lexical error rather than silently accepting a
                // token some scripts may rely on rejecting (and to avoid a
                // C-style octal misreading). This check applies only to the
                // integer part: a decimal's fractional part is grammatically
                // `0*<numeral>` and *does* permit leading zeros (e.g. the
                // `001` in `0.001`), so it is deliberately left unchecked.
                if num.len() > 1 && num.starts_with('0') {
                    self.errors.push(LexError {
                        message: format!(
                            "numeral '{num}' has a leading zero (SMT-LIB numerals must be \
                             '0' or a digit sequence not starting with '0')"
                        ),
                        pos: start,
                    });
                }
                if self.pos < self.input.len() && self.input[self.pos..].starts_with('.') {
                    self.pos += 1;
                    let frac = self.read_numeral();
                    TokenKind::Decimal(format!("{num}.{frac}"))
                } else {
                    TokenKind::Numeral(num)
                }
            }
            '|' => {
                self.pos += 1;
                let sym = self.read_quoted_symbol(start);
                TokenKind::Symbol(sym)
            }
            _ => {
                let sym = self.read_symbol_chars();
                if sym.is_empty() {
                    // `first` can neither start nor continue an SMT-LIB simple
                    // symbol, so `read_symbol_chars` consumed nothing. Handing
                    // back the empty `Symbol("")` it produced would leave
                    // `self.pos` exactly where it was, and every later
                    // `next_token` would mint the same zero-width token at the
                    // same offset for ever — a caller scanning to `Eof` (the
                    // parser's balanced-paren skip, say) never terminates.
                    // Consume exactly the one offending character and record
                    // it, the way the bare `#` case above does.
                    self.pos += first.len_utf8();
                    self.errors.push(LexError {
                        message: format!(
                            "unexpected character U+{:04X} cannot start a token",
                            u32::from(first)
                        ),
                        pos: start,
                    });
                    TokenKind::Symbol(self.input[start..self.pos].to_string())
                } else {
                    TokenKind::Symbol(sym)
                }
            }
        };

        Some(Token {
            kind,
            start,
            end: self.pos,
        })
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.pos < self.input.len() {
                if let Some(c) = self.input[self.pos..].chars().next() {
                    if c.is_whitespace() {
                        self.pos += c.len_utf8();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            // Skip comments
            if self.pos < self.input.len() && self.input[self.pos..].starts_with(';') {
                while self.pos < self.input.len() {
                    if let Some(c) = self.input[self.pos..].chars().next() {
                        self.pos += c.len_utf8();
                        if c == '\n' {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_symbol_chars(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            if let Some(c) = self.input[self.pos..].chars().next() {
                if c.is_alphanumeric()
                    || matches!(
                        c,
                        '+' | '-'
                            | '/'
                            | '*'
                            | '='
                            | '%'
                            | '?'
                            | '!'
                            | '.'
                            | '$'
                            | '_'
                            | '~'
                            | '&'
                            | '^'
                            | '<'
                            | '>'
                            | '@'
                    )
                {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    /// Read a `|...|`-quoted symbol body, starting just past the opening
    /// `|` (i.e. `self.pos` already advanced past it by the caller).
    ///
    /// The `start_of_token` argument is the position of the opening `|`,
    /// used only to anchor an error if the closing `|` is never found.
    fn read_quoted_symbol(&mut self, start_of_token: usize) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            if let Some(c) = self.input[self.pos..].chars().next() {
                self.pos += c.len_utf8();
                if c == '|' {
                    return self.input[start..self.pos - 1].to_string();
                }
            } else {
                break;
            }
        }
        // Ran off the end of input without finding the closing `|`: the
        // rest of the file was silently swallowed as symbol content. Record
        // it rather than pretending this was a well-formed token.
        self.errors.push(LexError {
            message: "unterminated quoted symbol (missing closing '|')".to_string(),
            pos: start_of_token,
        });
        self.input[start..self.pos].to_string()
    }

    /// Read a `"..."`-quoted string literal body, starting just past the
    /// opening `"`. `start_of_token` is the opening `"`'s position, used
    /// only to anchor an error if the closing `"` is never found.
    ///
    /// The returned `String` is the literal's *value*: the doubled-quote
    /// escape `""` is folded to a single `"`, and the two SMT-LIB Unicode
    /// Strings escape forms (`\ud₃d₂d₁d₀` and `\u{d...}`) are decoded to the
    /// single character they denote. Any other backslash stands for itself —
    /// see [`scan_unicode_escape`].
    fn read_string_lit(&mut self, start_of_token: usize) -> String {
        let mut result = String::new();
        let mut terminated = false;
        while self.pos < self.input.len() {
            let Some(c) = self.input[self.pos..].chars().next() else {
                break;
            };
            if c == '"' {
                self.pos += 1;
                // Check for escaped quote
                if self.pos < self.input.len() && self.input[self.pos..].starts_with('"') {
                    result.push('"');
                    self.pos += 1;
                } else {
                    terminated = true;
                    break;
                }
                continue;
            }
            if c == '\\'
                && let Some((code_point, len)) = scan_unicode_escape(&self.input[self.pos..])
            {
                if let Some(decoded) = char::from_u32(code_point) {
                    result.push(decoded);
                } else {
                    // The SMT-LIB alphabet includes the UTF-16 surrogate range
                    // `0xD800..=0xDFFF`, which a Rust `char` (and therefore
                    // OxiZ's string representation) cannot hold. Rejecting the
                    // literal is honest; silently keeping the undecoded text
                    // would make `str.len` and friends answer about a
                    // different string.
                    self.errors.push(LexError {
                        message: format!(
                            "string literal escape denotes surrogate code point U+{code_point:04X}, \
                             which is not representable"
                        ),
                        pos: self.pos,
                    });
                    result.push_str(&self.input[self.pos..self.pos + len]);
                }
                self.pos += len;
                continue;
            }
            self.pos += c.len_utf8();
            result.push(c);
        }
        if !terminated {
            self.errors.push(LexError {
                message: "unterminated string literal (missing closing '\"')".to_string(),
                pos: start_of_token,
            });
        }
        result
    }

    fn read_numeral(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            if let Some(c) = self.input[self.pos..].chars().next() {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn read_hex_chars(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            if let Some(c) = self.input[self.pos..].chars().next() {
                if c.is_ascii_hexdigit() {
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn read_binary_chars(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            if let Some(c) = self.input[self.pos..].chars().next() {
                if c == '0' || c == '1' {
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }
}

impl Iterator for Lexer<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let token = self.next_token()?;
        if matches!(token.kind, TokenKind::Eof) {
            None
        } else {
            Some(token)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("(+ 1 2)");
        assert!(matches!(
            lexer.next_token().expect("should get lparen token").kind,
            TokenKind::LParen
        ));
        assert!(matches!(
            lexer.next_token().expect("should get plus symbol token").kind,
            TokenKind::Symbol(s) if s == "+"
        ));
        assert!(matches!(
            lexer.next_token().expect("should get numeral 1 token").kind,
            TokenKind::Numeral(n) if n == "1"
        ));
        assert!(matches!(
            lexer.next_token().expect("should get numeral 2 token").kind,
            TokenKind::Numeral(n) if n == "2"
        ));
        assert!(matches!(
            lexer.next_token().expect("should get rparen token").kind,
            TokenKind::RParen
        ));
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("; this is a comment\n(test)");
        assert!(matches!(
            lexer
                .next_token()
                .expect("should get lparen token after comment")
                .kind,
            TokenKind::LParen
        ));
        assert!(matches!(
            lexer.next_token().expect("should get test symbol token").kind,
            TokenKind::Symbol(s) if s == "test"
        ));
    }

    #[test]
    fn test_hex_binary() {
        let mut lexer = Lexer::new("#xDEAD #b1010");
        assert!(matches!(
            lexer.next_token().expect("should get hexadecimal token").kind,
            TokenKind::Hexadecimal(h) if h == "DEAD"
        ));
        assert!(matches!(
            lexer.next_token().expect("should get binary token").kind,
            TokenKind::Binary(b) if b == "1010"
        ));
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new("\"hello world\"");
        assert!(matches!(
            lexer.next_token().expect("should get string literal token").kind,
            TokenKind::StringLit(s) if s == "hello world"
        ));
    }

    #[test]
    fn test_keyword() {
        let mut lexer = Lexer::new(":named");
        assert!(matches!(
            lexer.next_token().expect("should get keyword token").kind,
            TokenKind::Keyword(k) if k == "named"
        ));
    }

    /// Lex `input` (which must be a single string literal) and return its
    /// decoded value together with whether any lexical error was recorded.
    fn lex_string_lit(input: &str) -> (String, bool) {
        let mut lexer = Lexer::new(input);
        let token = lexer.next_token().expect("should get a token");
        match token.kind {
            TokenKind::StringLit(s) => (s, lexer.has_errors()),
            other => panic!("expected a string literal token, got {other:?}"),
        }
    }

    #[test]
    fn test_string_escape_braced_form() {
        // `\u{d...}` with 1..=5 hex digits decodes to one code point.
        let (value, errored) = lex_string_lit(r#""\u{e9}""#);
        assert_eq!(value, "\u{e9}");
        assert_eq!(value.chars().count(), 1);
        assert!(!errored);

        let (value, _) = lex_string_lit(r#""\u{7}""#);
        assert_eq!(value.chars().count(), 1);
        let (value, _) = lex_string_lit(r#""\u{1F600}""#);
        assert_eq!(value, "\u{1f600}");
        assert_eq!(value.chars().count(), 1);
    }

    #[test]
    fn test_string_escape_four_digit_form() {
        // The unbraced form requires exactly four hexadecimal digits.
        let (value, errored) = lex_string_lit("\"\\u0041\"");
        assert_eq!(value, "A");
        assert!(!errored);

        // Three digits then a non-hex character is not an escape.
        let (value, _) = lex_string_lit(r#""\u004z""#);
        assert_eq!(value, r"\u004z");
    }

    #[test]
    fn test_string_escape_code_point_boundaries() {
        // 0 and 0x2FFFF are both in range and decode to one character.
        let (value, _) = lex_string_lit(r#""\u{0}""#);
        assert_eq!(value, "\0");
        assert_eq!(value.chars().count(), 1);

        let (value, _) = lex_string_lit(r#""\u{2ffff}""#);
        assert_eq!(value, "\u{2ffff}");
        assert_eq!(value.chars().count(), 1);

        // 0x30000 is one above the SMT-LIB alphabet's maximum, so the
        // sequence is not an escape and stands for itself.
        let (value, errored) = lex_string_lit(r#""\u{30000}""#);
        assert_eq!(value, r"\u{30000}");
        assert_eq!(value.chars().count(), 9);
        assert!(!errored);
    }

    #[test]
    fn test_string_non_escape_backslash_stands_for_itself() {
        for (input, expected) in [
            (r#""\q""#, r"\q"),
            (r#""\u{}""#, r"\u{}"),
            (r#""\u{110000}""#, r"\u{110000}"),
            (r#""\u{000041}""#, r"\u{000041}"),
            (r#""\u00""#, r"\u00"),
            (r#""\\""#, r"\\"),
        ] {
            let (value, errored) = lex_string_lit(input);
            assert_eq!(value, expected, "input {input}");
            assert!(!errored, "input {input} should not be a lexical error");
        }
    }

    #[test]
    fn test_string_doubled_quote_escape() {
        let (value, errored) = lex_string_lit("\"a\"\"b\"");
        assert_eq!(value, "a\"b");
        assert!(!errored);
    }

    #[test]
    fn test_string_mixed_escapes_and_plain_text() {
        let (value, errored) = lex_string_lit("\"a\\u{62}c\\u0064e\"");
        assert_eq!(value, "abcde");
        assert!(!errored);
    }

    /// Drain `input` with a hard step budget, returning the tokens read and
    /// whether any lexical error was recorded.
    ///
    /// The budget is the point: a `next_token` that fails to advance `self.pos`
    /// yields the same zero-width token for ever, and a test that simply loops
    /// until `Eof` would hang the whole test binary instead of failing. This
    /// one fails.
    fn lex_until_eof(input: &str) -> (Vec<TokenKind>, bool) {
        const MAX_STEPS: usize = 1000;

        let mut lexer = Lexer::new(input);
        let mut kinds = Vec::new();
        for _ in 0..MAX_STEPS {
            let Some(token) = lexer.next_token() else {
                break;
            };
            let done = token.kind == TokenKind::Eof;
            kinds.push(token.kind);
            if done {
                return (kinds, lexer.has_errors());
            }
        }
        panic!("lexing {input:?} did not terminate within {MAX_STEPS} tokens");
    }

    #[test]
    fn test_unlexable_character_terminates_with_an_error() {
        // Every one of these is neither alphanumeric nor one of the symbol
        // punctuation characters, so `read_symbol_chars` reads nothing at all.
        // Before the fix the catch-all arm turned that into a `Symbol("")` at
        // an unchanged position, and the token stream never reached `Eof`.
        for input in [
            "\u{0}", ",", "[", "]", "{", "}", "\\", "'", "`", "\u{7f}", "\u{1}",
        ] {
            let (kinds, errored) = lex_until_eof(input);
            assert!(errored, "input {input:?} must record a lexical error");
            assert_eq!(kinds.last(), Some(&TokenKind::Eof), "input {input:?}");
            assert!(kinds.len() >= 2, "input {input:?} must yield a real token");
        }
    }

    #[test]
    fn test_unlexable_character_at_command_head_still_terminates() {
        // The shape that hung the wasm build: an unlexable byte in command-head
        // position, with a well-formed script around it.
        let (kinds, errored) = lex_until_eof("(\u{0}check-sat)(check-sat)");
        assert!(errored, "an unlexable head character must be reported");
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, TokenKind::Symbol(s) if s == "check-sat")),
            "lexing must make progress past the offending character: {kinds:?}"
        );
    }

    #[test]
    fn test_lexing_stays_exact_for_well_formed_input() {
        // Control: the new error path must not fire on ordinary scripts.
        let (_, errored) = lex_until_eof("(set-logic QF_LIA)(assert (= x 1))(check-sat)");
        assert!(
            !errored,
            "a well-formed script must record no lexical error"
        );
    }

    #[test]
    fn test_string_surrogate_escape_is_a_lexical_error() {
        let (_, errored) = lex_string_lit(r#""\ud800""#);
        assert!(errored, "a lone surrogate escape must be reported");
        let (_, errored) = lex_string_lit(r#""\u{dfff}""#);
        assert!(errored, "a lone surrogate escape must be reported");
    }
}
