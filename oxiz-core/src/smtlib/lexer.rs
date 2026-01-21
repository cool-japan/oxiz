//! SMT-LIB2 Lexer

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

/// Lexer for SMT-LIB2
pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer
    #[must_use]
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    /// Get the current position
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Peek at the next token without consuming it
    #[must_use]
    pub fn peek(&self) -> Option<Token> {
        let mut lexer = Self {
            input: self.input,
            pos: self.pos,
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
                let s = self.read_string_lit();
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
                        _ => TokenKind::Symbol("#".to_string()),
                    }
                } else {
                    TokenKind::Symbol("#".to_string())
                }
            }
            '0'..='9' => {
                let num = self.read_numeral();
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
                let sym = self.read_quoted_symbol();
                TokenKind::Symbol(sym)
            }
            _ => {
                let sym = self.read_symbol_chars();
                TokenKind::Symbol(sym)
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
                let c = self.input[self.pos..]
                    .chars()
                    .next()
                    .expect("pos within input bounds");
                if c.is_whitespace() {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            }

            // Skip comments
            if self.pos < self.input.len() && self.input[self.pos..].starts_with(';') {
                while self.pos < self.input.len() {
                    let c = self.input[self.pos..]
                        .chars()
                        .next()
                        .expect("pos within input bounds");
                    self.pos += c.len_utf8();
                    if c == '\n' {
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
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos within input bounds");
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
        }
        self.input[start..self.pos].to_string()
    }

    fn read_quoted_symbol(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos within input bounds");
            self.pos += c.len_utf8();
            if c == '|' {
                return self.input[start..self.pos - 1].to_string();
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn read_string_lit(&mut self) -> String {
        let mut result = String::new();
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos within input bounds");
            self.pos += c.len_utf8();
            if c == '"' {
                // Check for escaped quote
                if self.pos < self.input.len() && self.input[self.pos..].starts_with('"') {
                    result.push('"');
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    fn read_numeral(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos within input bounds");
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn read_hex_chars(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos within input bounds");
            if c.is_ascii_hexdigit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn read_binary_chars(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos within input bounds");
            if c == '0' || c == '1' {
                self.pos += 1;
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
            lexer.next_token().unwrap().kind,
            TokenKind::LParen
        ));
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Symbol(s) if s == "+"
        ));
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Numeral(n) if n == "1"
        ));
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Numeral(n) if n == "2"
        ));
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::RParen
        ));
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("; this is a comment\n(test)");
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::LParen
        ));
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Symbol(s) if s == "test"
        ));
    }

    #[test]
    fn test_hex_binary() {
        let mut lexer = Lexer::new("#xDEAD #b1010");
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Hexadecimal(h) if h == "DEAD"
        ));
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Binary(b) if b == "1010"
        ));
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new("\"hello world\"");
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::StringLit(s) if s == "hello world"
        ));
    }

    #[test]
    fn test_keyword() {
        let mut lexer = Lexer::new(":named");
        assert!(matches!(
            lexer.next_token().unwrap().kind,
            TokenKind::Keyword(k) if k == "named"
        ));
    }
}
