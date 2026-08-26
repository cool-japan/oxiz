//! CHC-COMP format parser
//!
//! Parses Constrained Horn Clauses in SMT-LIB2/CHC-COMP format.
//!
//! Reference: <https://chc-comp.github.io/format.html>

use crate::chc::{ChcSystem, PredId, PredicateApp, RuleBody, RuleHead};
use num_rational::Rational64;
use oxiz_core::ast::TermKind;
use oxiz_core::sort::SortId;
use oxiz_core::{TermId, TermManager};
use std::collections::HashMap;
use thiserror::Error;

/// Token types for SMT-LIB2 lexer
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Left parenthesis
    LParen,
    /// Right parenthesis
    RParen,
    /// Symbol (identifier)
    Symbol(String),
    /// Keyword (starts with :)
    Keyword(String),
    /// String literal
    StringLit(String),
    /// Numeral (non-negative integer)
    Numeral(String),
    /// Decimal (floating point)
    Decimal(String),
}

/// Lexer for SMT-LIB2 format
pub struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    /// Create a new lexer from input string
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Result<Option<Token>, ParseError> {
        // Skip whitespace and comments
        self.skip_whitespace_and_comments()?;

        if self.pos >= self.input.len() {
            return Ok(None);
        }

        let ch = self.input[self.pos];

        match ch {
            '(' => {
                self.pos += 1;
                Ok(Some(Token::LParen))
            }
            ')' => {
                self.pos += 1;
                Ok(Some(Token::RParen))
            }
            '"' => self.read_string(),
            ':' => self.read_keyword(),
            '0'..='9' => self.read_number(),
            _ if Self::is_symbol_char(ch) => self.read_symbol(),
            _ => Err(ParseError::InvalidSyntax(format!(
                "unexpected character: '{}'",
                ch
            ))),
        }
    }

    /// Skip whitespace and comments
    fn skip_whitespace_and_comments(&mut self) -> Result<(), ParseError> {
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];

            if ch.is_whitespace() {
                self.pos += 1;
            } else if ch == ';' {
                // Skip comment until end of line
                while self.pos < self.input.len() && self.input[self.pos] != '\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Read a string literal
    fn read_string(&mut self) -> Result<Option<Token>, ParseError> {
        self.pos += 1; // Skip opening quote

        let mut s = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == '"' {
                self.pos += 1;
                return Ok(Some(Token::StringLit(s)));
            } else if ch == '\\' && self.pos + 1 < self.input.len() {
                // Escape sequence
                self.pos += 1;
                s.push(self.input[self.pos]);
                self.pos += 1;
            } else {
                s.push(ch);
                self.pos += 1;
            }
        }

        Err(ParseError::InvalidSyntax(
            "unterminated string literal".to_string(),
        ))
    }

    /// Read a keyword (starts with :)
    fn read_keyword(&mut self) -> Result<Option<Token>, ParseError> {
        self.pos += 1; // Skip ':'

        let start = self.pos;
        while self.pos < self.input.len() && Self::is_symbol_char(self.input[self.pos]) {
            self.pos += 1;
        }

        let keyword: String = self.input[start..self.pos].iter().collect();
        Ok(Some(Token::Keyword(keyword)))
    }

    /// Read a number (numeral or decimal)
    fn read_number(&mut self) -> Result<Option<Token>, ParseError> {
        let start = self.pos;
        let mut has_dot = false;

        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch.is_ascii_digit() {
                self.pos += 1;
            } else if ch == '.' && !has_dot {
                has_dot = true;
                self.pos += 1;
            } else {
                break;
            }
        }

        let number: String = self.input[start..self.pos].iter().collect();

        if has_dot {
            Ok(Some(Token::Decimal(number)))
        } else {
            Ok(Some(Token::Numeral(number)))
        }
    }

    /// Read a symbol
    fn read_symbol(&mut self) -> Result<Option<Token>, ParseError> {
        let start = self.pos;

        while self.pos < self.input.len() && Self::is_symbol_char(self.input[self.pos]) {
            self.pos += 1;
        }

        let symbol: String = self.input[start..self.pos].iter().collect();
        Ok(Some(Token::Symbol(symbol)))
    }

    /// Check if a character can be part of a symbol
    fn is_symbol_char(ch: char) -> bool {
        ch.is_alphanumeric()
            || ch == '_'
            || ch == '-'
            || ch == '+'
            || ch == '*'
            || ch == '/'
            || ch == '<'
            || ch == '>'
            || ch == '='
            || ch == '!'
            || ch == '?'
            || ch == '.'
    }

    /// Tokenize the entire input
    pub fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();

        while let Some(token) = self.next_token()? {
            tokens.push(token);
        }

        Ok(tokens)
    }
}

/// S-expression representation
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    /// Atom (symbol, keyword, number, or string)
    Atom(Token),
    /// List of S-expressions
    List(Vec<SExpr>),
}

impl SExpr {
    /// Check if this is a list
    pub fn is_list(&self) -> bool {
        matches!(self, SExpr::List(_))
    }

    /// Check if this is an atom
    pub fn is_atom(&self) -> bool {
        matches!(self, SExpr::Atom(_))
    }

    /// Get as a symbol, if it is one
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            SExpr::Atom(Token::Symbol(s)) => Some(s),
            _ => None,
        }
    }

    /// Get as a list, if it is one
    pub fn as_list(&self) -> Option<&[SExpr]> {
        match self {
            SExpr::List(items) => Some(items),
            _ => None,
        }
    }
}

impl Drop for SExpr {
    /// Tear an S-expression down iteratively.
    ///
    /// Now that [`SExprParser::parse_sexpr`] can build an arbitrarily deep
    /// tree without consuming native stack, the compiler-generated drop
    /// glue became the remaining depth limit: dropping a 100k-deep
    /// `SExpr::List` recursed 100k frames and aborted the process. `Drop`
    /// has no error channel at all, so this too has to be an explicit heap
    /// stack rather than a bounded one.
    ///
    /// Each nested list's children are moved into a pending queue and the
    /// husk is dropped with an empty `Vec`, so the derived glue never
    /// descends more than one level.
    fn drop(&mut self) {
        let SExpr::List(items) = self else {
            return;
        };
        if items.is_empty() {
            return;
        }

        let mut pending: Vec<Vec<SExpr>> = vec![std::mem::take(items)];
        while let Some(batch) = pending.pop() {
            for mut item in batch {
                if let SExpr::List(children) = &mut item
                    && !children.is_empty()
                {
                    pending.push(std::mem::take(children));
                }
            }
        }
    }
}

/// Parser for S-expressions
pub struct SExprParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl SExprParser {
    /// Create a new S-expression parser from tokens
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse a single S-expression.
    ///
    /// The nesting depth of the input is directly attacker-controlled
    /// (`((((…))))` in a CHC/SMT file reaching `SExprParser::parse_str`,
    /// `ChcParser::parse`, or `ChcCompBuilder::from_file`), so this is
    /// parsed with an explicit heap stack of partially built lists rather
    /// than one native frame per level. Only the mechanism changes: the
    /// same three error cases are reported, at the same points.
    pub fn parse_sexpr(&mut self) -> Result<SExpr, ParseError> {
        // Stack of the lists currently being built; empty means "not
        // inside any list", i.e. the next completed expression is the
        // result.
        let mut open_lists: Vec<Vec<SExpr>> = Vec::new();

        loop {
            let Some(token) = self.tokens.get(self.pos) else {
                return Err(ParseError::InvalidSyntax(if open_lists.is_empty() {
                    "unexpected end of input".to_string()
                } else {
                    "unclosed parenthesis".to_string()
                }));
            };

            let finished = match token {
                Token::LParen => {
                    self.pos += 1;
                    open_lists.push(Vec::new());
                    continue;
                }
                Token::RParen => {
                    let Some(items) = open_lists.pop() else {
                        return Err(ParseError::InvalidSyntax(
                            "unexpected closing parenthesis".to_string(),
                        ));
                    };
                    self.pos += 1;
                    SExpr::List(items)
                }
                atom => {
                    let expr = SExpr::Atom(atom.clone());
                    self.pos += 1;
                    expr
                }
            };

            // Attach the finished expression to its parent, or return it
            // when there is no enclosing list left.
            match open_lists.last_mut() {
                Some(parent) => parent.push(finished),
                None => return Ok(finished),
            }
        }
    }

    /// Parse all S-expressions in the token stream
    pub fn parse_all(&mut self) -> Result<Vec<SExpr>, ParseError> {
        let mut exprs = Vec::new();

        while self.pos < self.tokens.len() {
            exprs.push(self.parse_sexpr()?);
        }

        Ok(exprs)
    }

    /// Parse from a string (convenience method)
    pub fn parse_str(input: &str) -> Result<Vec<SExpr>, ParseError> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = SExprParser::new(tokens);
        parser.parse_all()
    }
}

/// Errors that can occur during parsing
#[derive(Error, Debug)]
pub enum ParseError {
    /// Invalid syntax
    #[error("invalid syntax: {0}")]
    InvalidSyntax(String),
    /// Undefined symbol
    #[error("undefined symbol: {0}")]
    UndefinedSymbol(String),
    /// Type error
    #[error("type error: {0}")]
    TypeError(String),
    /// Unsupported feature
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// CHC parser state
pub struct ChcParser<'a> {
    /// Term manager
    #[allow(dead_code)]
    terms: &'a mut TermManager,
    /// CHC system being built
    system: ChcSystem,
    /// Predicate name to ID mapping
    predicates: HashMap<String, PredId>,
    /// Variable name to term ID mapping (local to current rule)
    #[allow(dead_code)]
    variables: HashMap<String, TermId>,
    /// Current nesting depth of [`ChcParser::parse_term`].
    term_depth: usize,
}

/// Maximum S-expression nesting `ChcParser::parse_term` will descend into.
///
/// Unlike the s-expression reader, term construction dispatches through a
/// large per-operator `match`, so its native frames are fat; the depth is
/// still attacker-controlled. `parse_term` returns a `Result`, so exceeding
/// the bound is reported as a real parse error — the input is *rejected*,
/// never silently truncated into a different (wrong) term.
///
/// The value is chosen so that parsing at the limit is safe even when the
/// caller runs on a small (1 MiB) worker thread in an unoptimized build,
/// which `parse_term_accepts_nesting_just_under_the_limit` pins; SMT-LIB
/// formulas nest far shallower than this in practice (they are wide
/// conjunctions, not deep chains).
const MAX_TERM_NESTING: usize = 500;

impl<'a> ChcParser<'a> {
    /// Create a new CHC parser
    pub fn new(terms: &'a mut TermManager) -> Self {
        Self {
            terms,
            system: ChcSystem::new(),
            predicates: HashMap::new(),
            variables: HashMap::new(),
            term_depth: 0,
        }
    }

    /// Parse a CHC problem from a string
    pub fn parse(&mut self, input: &str) -> Result<ChcSystem, ParseError> {
        // Full SMT-LIB2 parser implementation
        // 1. Tokenize the input
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;

        // 2. Parse S-expressions
        let mut parser = SExprParser::new(tokens);
        let exprs = parser.parse_all()?;

        // 3. Process each command
        for expr in exprs {
            self.process_command(&expr)?;
        }

        Ok(std::mem::take(&mut self.system))
    }

    /// Process a single SMT-LIB2 command (S-expression)
    fn process_command(&mut self, expr: &SExpr) -> Result<(), ParseError> {
        let items = expr
            .as_list()
            .ok_or_else(|| ParseError::InvalidSyntax("expected command as list".to_string()))?;

        if items.is_empty() {
            return Ok(());
        }

        let cmd = items[0]
            .as_symbol()
            .ok_or_else(|| ParseError::InvalidSyntax("expected command name".to_string()))?;

        match cmd {
            "set-logic" => {
                // (set-logic HORN)
                Ok(())
            }
            "declare-fun" => {
                // (declare-fun P (Int Bool) Bool)
                if items.len() < 4 {
                    return Err(ParseError::InvalidSyntax(
                        "declare-fun requires name, arg sorts, and return sort".to_string(),
                    ));
                }

                let name = items[1].as_symbol().ok_or_else(|| {
                    ParseError::InvalidSyntax("expected predicate name".to_string())
                })?;

                // Parse argument sorts
                let arg_sorts_list = items[2].as_list().ok_or_else(|| {
                    ParseError::InvalidSyntax("expected argument sort list".to_string())
                })?;

                let arg_sorts: Vec<SortId> = arg_sorts_list
                    .iter()
                    .map(|s| {
                        let sort_name = s.as_symbol().ok_or_else(|| {
                            ParseError::InvalidSyntax("expected sort name".to_string())
                        })?;
                        self.parse_sort(sort_name)
                    })
                    .collect::<Result<Vec<_>, ParseError>>()?;

                // Declare predicate
                let pred_id = self.system.declare_predicate(name, arg_sorts);
                self.predicates.insert(name.to_string(), pred_id);

                Ok(())
            }
            "assert" => {
                // (assert formula)
                if items.len() < 2 {
                    return Err(ParseError::InvalidSyntax(
                        "assert requires a formula".to_string(),
                    ));
                }

                let formula = self.parse_term(&items[1])?;
                self.process_assertion(formula)?;

                Ok(())
            }
            "check-sat" => {
                // Ignore check-sat commands in CHC parsing
                Ok(())
            }
            _ => {
                // Unknown command, skip
                Ok(())
            }
        }
    }

    /// Parse a sort name to SortId.
    ///
    /// Unknown/unsupported sort names (e.g. `BitVec`, `Array`, or any
    /// undeclared user sort) are reported as an honest
    /// [`ParseError::UndefinedSymbol`] rather than silently coerced to
    /// `Bool` -- a CHC predicate argument or bound variable typed `Bool`
    /// when it was actually meant to be, say, a `BitVec` sort would
    /// silently corrupt every constraint that touches it.
    fn parse_sort(&self, name: &str) -> Result<SortId, ParseError> {
        match name {
            "Bool" => Ok(self.terms.sorts.bool_sort),
            "Int" => Ok(self.terms.sorts.int_sort),
            "Real" => Ok(self.terms.sorts.real_sort),
            _ => Err(ParseError::UndefinedSymbol(format!(
                "unknown or unsupported sort: {}",
                name
            ))),
        }
    }

    /// Parse a term from an S-expression.
    ///
    /// Descends one native frame per nesting level, so the depth is capped
    /// at [`MAX_TERM_NESTING`] and exceeding it is an honest
    /// [`ParseError`]: the caller learns the input was refused instead of
    /// receiving a truncated term that would silently mean something else.
    fn parse_term(&mut self, expr: &SExpr) -> Result<TermId, ParseError> {
        if self.term_depth >= MAX_TERM_NESTING {
            return Err(ParseError::InvalidSyntax(format!(
                "term nesting exceeds the supported limit of {MAX_TERM_NESTING} levels"
            )));
        }

        self.term_depth += 1;
        let parsed = match expr {
            SExpr::Atom(token) => self.parse_atom(token),
            SExpr::List(items) => self.parse_application(items),
        };
        self.term_depth -= 1;
        parsed
    }

    /// Parse an atomic term (variable, constant, etc.)
    fn parse_atom(&mut self, token: &Token) -> Result<TermId, ParseError> {
        match token {
            Token::Symbol(s) => {
                // Check if it's a known constant
                match s.as_str() {
                    "true" => Ok(self.terms.mk_true()),
                    "false" => Ok(self.terms.mk_false()),
                    _ => {
                        // Treat as variable
                        // Look up in variables map, or create new
                        if let Some(&var) = self.variables.get(s) {
                            Ok(var)
                        } else {
                            // Create new variable with Int sort by default
                            let var = self.terms.mk_var(s, self.terms.sorts.int_sort);
                            self.variables.insert(s.clone(), var);
                            Ok(var)
                        }
                    }
                }
            }
            Token::Numeral(n) => {
                // Parse as integer
                let value = n
                    .parse::<i64>()
                    .map_err(|_| ParseError::TypeError(format!("invalid integer: {}", n)))?;
                Ok(self.terms.mk_int(value))
            }
            Token::Decimal(d) => {
                // Parse a decimal literal (e.g. "3.25", "-0.5") into an
                // exact rational: `whole.frac` -> `(whole * 10^len(frac) +
                // frac) / 10^len(frac)`. This used to discard both parsed
                // components and always return the constant `0`, silently
                // corrupting every decimal literal in a CHC-COMP input.
                let negative = d.starts_with('-');
                let unsigned = d.strip_prefix('-').unwrap_or(d.as_str());
                let parts: Vec<&str> = unsigned.split('.').collect();
                if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
                    return Err(ParseError::TypeError(format!("invalid decimal: {}", d)));
                }
                let (whole_str, frac_str) = (parts[0], parts[1]);
                if !whole_str.bytes().all(|b| b.is_ascii_digit())
                    || !frac_str.bytes().all(|b| b.is_ascii_digit())
                {
                    return Err(ParseError::TypeError(format!("invalid decimal: {}", d)));
                }

                let denom: i64 = 10i64.checked_pow(frac_str.len() as u32).ok_or_else(|| {
                    ParseError::TypeError(format!("decimal literal too precise: {}", d))
                })?;
                let whole: i64 = whole_str.parse().map_err(|_| {
                    ParseError::TypeError(format!("invalid decimal whole part: {}", whole_str))
                })?;
                let frac: i64 = frac_str.parse().map_err(|_| {
                    ParseError::TypeError(format!("invalid decimal fractional part: {}", frac_str))
                })?;
                let magnitude = whole
                    .checked_mul(denom)
                    .and_then(|w| w.checked_add(frac))
                    .ok_or_else(|| {
                        ParseError::TypeError(format!("decimal literal overflow: {}", d))
                    })?;
                let numer = if negative { -magnitude } else { magnitude };

                Ok(self.terms.mk_real(Rational64::new(numer, denom)))
            }
            _ => Err(ParseError::Unsupported(format!(
                "unsupported token type: {:?}",
                token
            ))),
        }
    }

    /// Parse a function/predicate application
    fn parse_application(&mut self, items: &[SExpr]) -> Result<TermId, ParseError> {
        if items.is_empty() {
            return Err(ParseError::InvalidSyntax("empty application".to_string()));
        }

        let func_name = items[0]
            .as_symbol()
            .ok_or_else(|| ParseError::InvalidSyntax("expected function name".to_string()))?;

        match func_name {
            // Logical operators
            "and" => {
                let args: Vec<TermId> = items[1..]
                    .iter()
                    .map(|e| self.parse_term(e))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self.terms.mk_and(args))
            }
            "or" => {
                let args: Vec<TermId> = items[1..]
                    .iter()
                    .map(|e| self.parse_term(e))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self.terms.mk_or(args))
            }
            "not" => {
                if items.len() != 2 {
                    return Err(ParseError::InvalidSyntax(
                        "not requires 1 argument".to_string(),
                    ));
                }
                let arg = self.parse_term(&items[1])?;
                Ok(self.terms.mk_not(arg))
            }
            "=>" | "implies" => {
                if items.len() != 3 {
                    return Err(ParseError::InvalidSyntax(
                        "implies requires 2 arguments".to_string(),
                    ));
                }
                let lhs = self.parse_term(&items[1])?;
                let rhs = self.parse_term(&items[2])?;
                Ok(self.terms.mk_implies(lhs, rhs))
            }
            // Arithmetic operators
            "=" => {
                if items.len() != 3 {
                    return Err(ParseError::InvalidSyntax(
                        "= requires 2 arguments".to_string(),
                    ));
                }
                let lhs = self.parse_term(&items[1])?;
                let rhs = self.parse_term(&items[2])?;
                Ok(self.terms.mk_eq(lhs, rhs))
            }
            "<" => {
                if items.len() != 3 {
                    return Err(ParseError::InvalidSyntax(
                        "< requires 2 arguments".to_string(),
                    ));
                }
                let lhs = self.parse_term(&items[1])?;
                let rhs = self.parse_term(&items[2])?;
                Ok(self.terms.mk_lt(lhs, rhs))
            }
            "<=" => {
                if items.len() != 3 {
                    return Err(ParseError::InvalidSyntax(
                        "<= requires 2 arguments".to_string(),
                    ));
                }
                let lhs = self.parse_term(&items[1])?;
                let rhs = self.parse_term(&items[2])?;
                Ok(self.terms.mk_le(lhs, rhs))
            }
            ">" => {
                if items.len() != 3 {
                    return Err(ParseError::InvalidSyntax(
                        "> requires 2 arguments".to_string(),
                    ));
                }
                let lhs = self.parse_term(&items[1])?;
                let rhs = self.parse_term(&items[2])?;
                Ok(self.terms.mk_gt(lhs, rhs))
            }
            ">=" => {
                if items.len() != 3 {
                    return Err(ParseError::InvalidSyntax(
                        ">= requires 2 arguments".to_string(),
                    ));
                }
                let lhs = self.parse_term(&items[1])?;
                let rhs = self.parse_term(&items[2])?;
                Ok(self.terms.mk_ge(lhs, rhs))
            }
            "+" => {
                let args: Vec<TermId> = items[1..]
                    .iter()
                    .map(|e| self.parse_term(e))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self.terms.mk_add(args))
            }
            "-" => {
                if items.len() == 2 {
                    // Unary minus
                    let arg = self.parse_term(&items[1])?;
                    let zero = self.terms.mk_int(0);
                    Ok(self.terms.mk_sub(zero, arg))
                } else if items.len() == 3 {
                    // Binary minus
                    let lhs = self.parse_term(&items[1])?;
                    let rhs = self.parse_term(&items[2])?;
                    Ok(self.terms.mk_sub(lhs, rhs))
                } else {
                    Err(ParseError::InvalidSyntax(
                        "- requires 1 or 2 arguments".to_string(),
                    ))
                }
            }
            "*" => {
                let args: Vec<TermId> = items[1..]
                    .iter()
                    .map(|e| self.parse_term(e))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self.terms.mk_mul(args))
            }
            // Quantifiers
            "forall" => self.parse_quantifier(items, true),
            "exists" => self.parse_quantifier(items, false),
            // Predicate application
            _ => {
                // Check if it's a declared predicate
                if self.predicates.contains_key(func_name) {
                    let args: Vec<TermId> = items[1..]
                        .iter()
                        .map(|e| self.parse_term(e))
                        .collect::<Result<Vec<_>, _>>()?;
                    // Build a real `TermKind::Apply` so that head/body predicate
                    // extraction (`try_extract_predicate_app`) can recover the
                    // predicate symbol and its argument terms.  Predicates return
                    // Bool in the CHC fragment.
                    let bool_sort = self.terms.sorts.bool_sort;
                    Ok(self.terms.mk_apply(func_name, args, bool_sort))
                } else {
                    Err(ParseError::UndefinedSymbol(func_name.to_string()))
                }
            }
        }
    }

    /// Parse a quantified formula: (forall ((x Int) (y Bool)) body)
    fn parse_quantifier(&mut self, items: &[SExpr], is_forall: bool) -> Result<TermId, ParseError> {
        if items.len() != 3 {
            return Err(ParseError::InvalidSyntax(
                "quantifier requires variable list and body".to_string(),
            ));
        }

        // Parse variable declarations
        let var_list = items[1].as_list().ok_or_else(|| {
            ParseError::InvalidSyntax("expected variable declaration list".to_string())
        })?;

        let mut quantified_vars = Vec::new();
        let old_vars = self.variables.clone();

        for var_decl in var_list {
            let decl_items = var_decl.as_list().ok_or_else(|| {
                ParseError::InvalidSyntax("expected variable declaration".to_string())
            })?;

            if decl_items.len() != 2 {
                return Err(ParseError::InvalidSyntax(
                    "variable declaration must be (name sort)".to_string(),
                ));
            }

            let var_name = decl_items[0]
                .as_symbol()
                .ok_or_else(|| ParseError::InvalidSyntax("expected variable name".to_string()))?;

            let sort_name = decl_items[1]
                .as_symbol()
                .ok_or_else(|| ParseError::InvalidSyntax("expected sort name".to_string()))?;

            let sort = self.parse_sort(sort_name)?;
            let var = self.terms.mk_var(var_name, sort);
            self.variables.insert(var_name.to_string(), var);
            quantified_vars.push((var_name, sort));
        }

        // Parse body
        let body = self.parse_term(&items[2])?;

        // Restore old variable scope
        self.variables = old_vars;

        // Create quantified formula using variable names and sorts
        // mk_forall/mk_exists expect (&str, SortId)
        if is_forall {
            Ok(self.terms.mk_forall(quantified_vars, body))
        } else {
            Ok(self.terms.mk_exists(quantified_vars, body))
        }
    }

    /// Process an assertion (convert to CHC rule).
    ///
    /// Extracts Horn clause structure from the formula and adds it to the CHC system.
    /// Handles three top-level shapes:
    ///   - `(forall ((x Sort) ...) (=> body head))` — universal Horn clause
    ///   - `(=> body head)` — bare implication
    ///   - any other formula — treated as a constraint / query
    fn process_assertion(&mut self, formula: TermId) -> Result<(), ParseError> {
        let Some(term_data) = self.terms.get(formula) else {
            return Err(ParseError::InvalidSyntax(
                "invalid term in assertion".to_string(),
            ));
        };

        match &term_data.kind.clone() {
            // (forall ((x Sort) ...) body)
            TermKind::Forall { vars, body, .. } => {
                let body_id = *body;
                let bound_vars: Vec<(String, SortId)> = vars
                    .iter()
                    .map(|(name_spur, sort)| {
                        (self.terms.resolve_str(*name_spur).to_string(), *sort)
                    })
                    .collect();
                self.process_horn_clause(body_id, bound_vars)
            }
            // (=> body head)
            TermKind::Implies(body_term, head_term) => {
                let (b, h) = (*body_term, *head_term);
                self.process_implication(b, h, Vec::new())
            }
            // Anything else: fact or query
            _ => {
                let body = RuleBody::init(self.terms.mk_true());
                if let Some(pred_app) = self.try_extract_predicate_app(formula) {
                    let head = RuleHead::Predicate(pred_app);
                    self.system.add_rule(Vec::new(), body, head, None);
                } else {
                    let query_body = RuleBody::init(formula);
                    let head = RuleHead::Query;
                    self.system.add_rule(Vec::new(), query_body, head, None);
                }
                Ok(())
            }
        }
    }

    /// Process a possibly-quantified Horn clause body.
    fn process_horn_clause(
        &mut self,
        body: TermId,
        vars: Vec<(String, SortId)>,
    ) -> Result<(), ParseError> {
        let Some(body_term) = self.terms.get(body) else {
            return Err(ParseError::InvalidSyntax(
                "invalid body in Horn clause".to_string(),
            ));
        };

        match body_term.kind.clone() {
            TermKind::Implies(lhs, rhs) => self.process_implication(lhs, rhs, vars),
            _ => {
                let rule_body = RuleBody::init(body);
                let head = RuleHead::Query;
                self.system.add_rule(vars, rule_body, head, None);
                Ok(())
            }
        }
    }

    /// Process `body => head` into a CHC rule.
    fn process_implication(
        &mut self,
        body_term: TermId,
        head_term: TermId,
        vars: Vec<(String, SortId)>,
    ) -> Result<(), ParseError> {
        // Split the body into uninterpreted predicate applications and linear constraints.
        let (body_preds, body_constraint) = self.decompose_conjunction(body_term);

        // Determine the head.
        let head = if let Some(head_data) = self.terms.get(head_term) {
            match head_data.kind.clone() {
                TermKind::False => RuleHead::Query,
                TermKind::Apply { func, args } => {
                    let func_name = self.terms.resolve_str(func).to_string();
                    if let Some(&pred_id) = self.predicates.get(&func_name) {
                        RuleHead::Predicate(PredicateApp::new(pred_id, args.iter().copied()))
                    } else {
                        return Err(ParseError::UndefinedSymbol(func_name));
                    }
                }
                _ => {
                    if let Some(pred_app) = self.try_extract_predicate_app(head_term) {
                        RuleHead::Predicate(pred_app)
                    } else {
                        RuleHead::Query
                    }
                }
            }
        } else {
            RuleHead::Query
        };

        let rule_body = if body_preds.is_empty() {
            RuleBody::init(body_constraint)
        } else {
            RuleBody::new(body_preds, body_constraint)
        };

        self.system.add_rule(vars, rule_body, head, None);
        Ok(())
    }

    /// Flatten an AND-tree into individual conjuncts.
    ///
    /// The `And` nesting depth is whatever the parsed file contains, so the
    /// flattening walks an explicit heap stack
    /// ([`crate::walk::flatten_conjuncts`]) rather than recursing, and does
    /// not re-expand a shared `And` sub-DAG (conjunction is idempotent, so
    /// that only costs exponential time and output size).
    fn collect_conjuncts(&self, term: TermId) -> Vec<TermId> {
        crate::walk::flatten_conjuncts(self.terms, term)
    }

    /// Split a conjunction into predicate applications and theory constraints.
    fn decompose_conjunction(&mut self, term: TermId) -> (Vec<PredicateApp>, TermId) {
        let mut predicates = Vec::new();
        let mut constraints = Vec::new();

        for conjunct in self.collect_conjuncts(term) {
            if let Some(pred_app) = self.try_extract_predicate_app(conjunct) {
                predicates.push(pred_app);
            } else {
                constraints.push(conjunct);
            }
        }

        let constraint = match constraints.len() {
            0 => self.terms.mk_true(),
            1 => constraints[0],
            _ => self.terms.mk_and(constraints),
        };

        (predicates, constraint)
    }

    /// Try to identify `term` as an application of a declared predicate.
    fn try_extract_predicate_app(&self, term: TermId) -> Option<PredicateApp> {
        let term_data = self.terms.get(term)?;
        match &term_data.kind {
            TermKind::Apply { func, args } => {
                let func_name = self.terms.resolve_str(*func).to_string();
                let pred_id = *self.predicates.get(&func_name)?;
                Some(PredicateApp::new(pred_id, args.iter().copied()))
            }
            _ => None,
        }
    }

    /// Get the parsed CHC system
    pub fn system(self) -> ChcSystem {
        self.system
    }

    /// Declare a predicate (helper for programmatic construction)
    pub fn declare_predicate(
        &mut self,
        name: &str,
        arg_sorts: impl IntoIterator<Item = oxiz_core::SortId>,
    ) -> PredId {
        let id = self.system.declare_predicate(name, arg_sorts);
        self.predicates.insert(name.to_string(), id);
        id
    }

    /// Add an init rule (helper for programmatic construction)
    #[allow(clippy::too_many_arguments)]
    pub fn add_init_rule(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        constraint: TermId,
        head_pred: PredId,
        head_args: impl IntoIterator<Item = TermId>,
    ) {
        self.system
            .add_init_rule(vars, constraint, head_pred, head_args);
    }

    /// Add a transition rule (helper for programmatic construction)
    #[allow(clippy::too_many_arguments)]
    pub fn add_transition_rule(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        body_preds: impl IntoIterator<Item = PredicateApp>,
        constraint: TermId,
        head_pred: PredId,
        head_args: impl IntoIterator<Item = TermId>,
    ) {
        self.system
            .add_transition_rule(vars, body_preds, constraint, head_pred, head_args);
    }

    /// Add a query rule (helper for programmatic construction)
    pub fn add_query(
        &mut self,
        vars: impl IntoIterator<Item = (String, oxiz_core::SortId)>,
        body_preds: impl IntoIterator<Item = PredicateApp>,
        constraint: TermId,
    ) {
        self.system.add_query(vars, body_preds, constraint);
    }
}

/// Builder for constructing CHC systems from SMT-LIB2 format
pub struct ChcCompBuilder<'a> {
    parser: ChcParser<'a>,
}

impl<'a> ChcCompBuilder<'a> {
    /// Create a new CHC-COMP builder
    pub fn new(terms: &'a mut TermManager) -> Self {
        Self {
            parser: ChcParser::new(terms),
        }
    }

    /// Parse from a file
    pub fn from_file(&mut self, path: &str) -> Result<(), ParseError> {
        // Read the file contents
        let contents = std::fs::read_to_string(path)
            .map_err(|e| ParseError::Unsupported(format!("Failed to read file {}: {}", path, e)))?;

        // Parse the contents
        self.from_str(&contents)
    }

    /// Parse from a string
    pub fn from_str(&mut self, input: &str) -> Result<(), ParseError> {
        self.parser.parse(input)?;
        Ok(())
    }

    /// Build the CHC system
    pub fn build(self) -> ChcSystem {
        self.parser.system()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let mut terms = TermManager::new();
        let parser = ChcParser::new(&mut terms);
        let system = parser.system();
        assert!(system.is_empty());
    }

    #[test]
    fn test_programmatic_construction() {
        let mut terms = TermManager::new();

        // Create terms first
        let int_sort = terms.sorts.int_sort;
        let x = terms.mk_var("x", int_sort);
        let zero = terms.mk_int(0);
        let constraint = terms.mk_eq(x, zero);

        // Now create parser and use the terms
        let mut parser = ChcParser::new(&mut terms);
        let inv = parser.declare_predicate("Inv", [int_sort]);
        parser.add_init_rule([("x".to_string(), int_sort)], constraint, inv, [x]);

        let system = parser.system();
        assert_eq!(system.num_predicates(), 1);
        assert_eq!(system.num_rules(), 1);
    }

    #[test]
    fn test_full_parse_basic() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let result = parser.parse("(set-logic HORN)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_predicate_declaration() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let input = "(declare-fun P (Int Bool) Bool)";
        let result = parser.parse(input);
        assert!(result.is_ok());

        let system = result.expect("test operation should succeed");
        assert_eq!(system.num_predicates(), 1);
    }

    #[test]
    fn test_parse_arithmetic() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        // Test integer parsing
        let result = parser.parse_atom(&Token::Numeral("42".to_string()));
        assert!(result.is_ok());

        // Test arithmetic expression
        let expr = SExpr::List(vec![
            SExpr::Atom(Token::Symbol("+".to_string())),
            SExpr::Atom(Token::Numeral("1".to_string())),
            SExpr::Atom(Token::Numeral("2".to_string())),
        ]);
        let result = parser.parse_term(&expr);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Regression tests for the `sweep-backend-misc` triage sweep.
    // -----------------------------------------------------------------------

    /// Decimal literals used to always parse as the constant `0`
    /// regardless of their actual text (`parse_atom`'s `Token::Decimal`
    /// arm discarded both parsed components and hardcoded `mk_int(0)`).
    /// This verifies decimals now parse to their true rational value.
    #[test]
    fn test_parse_decimal_literal_exact_value() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let term = parser
            .parse_atom(&Token::Decimal("3.25".to_string()))
            .expect("3.25 should parse");
        let node = parser.terms.get(term).expect("term must exist");
        assert_eq!(node.kind, TermKind::RealConst(Rational64::new(325, 100)));
        assert_eq!(node.sort, parser.terms.sorts.real_sort);
    }

    #[test]
    fn test_parse_decimal_literal_negative() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let term = parser
            .parse_atom(&Token::Decimal("-0.5".to_string()))
            .expect("-0.5 should parse");
        let node = parser.terms.get(term).expect("term must exist");
        assert_eq!(node.kind, TermKind::RealConst(Rational64::new(-1, 2)));
    }

    #[test]
    fn test_parse_decimal_literal_trailing_zeros_preserved_exactly() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        // "3.05" must be 305/100, not 35/10 or 0.
        let term = parser
            .parse_atom(&Token::Decimal("3.05".to_string()))
            .expect("3.05 should parse");
        let node = parser.terms.get(term).expect("term must exist");
        assert_eq!(node.kind, TermKind::RealConst(Rational64::new(305, 100)));
    }

    #[test]
    fn test_parse_decimal_literal_malformed_is_error() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        assert!(
            parser
                .parse_atom(&Token::Decimal("1.2.3".to_string()))
                .is_err()
        );
        assert!(
            parser
                .parse_atom(&Token::Decimal(".5".to_string()))
                .is_err()
        );
        assert!(
            parser
                .parse_atom(&Token::Decimal("5.".to_string()))
                .is_err()
        );
    }

    /// Unknown sort names used to silently resolve to `Bool`. This
    /// verifies they now surface as an honest parse error instead.
    #[test]
    fn test_parse_unknown_sort_is_error_not_silent_bool() {
        let mut terms = TermManager::new();
        let parser = ChcParser::new(&mut terms);

        let err = parser
            .parse_sort("BitVec")
            .expect_err("unknown sort must error, not silently become Bool");
        assert!(matches!(err, ParseError::UndefinedSymbol(_)));
    }

    #[test]
    fn test_parse_unknown_sort_in_declare_fun_propagates_error() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let result = parser.parse("(declare-fun P (BitVec) Bool)");
        assert!(
            result.is_err(),
            "an undeclared/unsupported sort in declare-fun must fail parsing, \
             not silently declare the argument as Bool"
        );
    }

    #[test]
    fn test_parse_known_sorts_still_work() {
        let mut terms = TermManager::new();
        let parser = ChcParser::new(&mut terms);

        assert_eq!(
            parser.parse_sort("Bool").expect("Bool is known"),
            parser.terms.sorts.bool_sort
        );
        assert_eq!(
            parser.parse_sort("Int").expect("Int is known"),
            parser.terms.sorts.int_sort
        );
        assert_eq!(
            parser.parse_sort("Real").expect("Real is known"),
            parser.terms.sorts.real_sort
        );
    }

    #[test]
    fn test_lexer_basic() {
        let mut lexer = Lexer::new("(set-logic HORN)");
        let tokens = lexer.tokenize().expect("test operation should succeed");

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::LParen);
        assert_eq!(tokens[1], Token::Symbol("set-logic".to_string()));
        assert_eq!(tokens[2], Token::Symbol("HORN".to_string()));
        assert_eq!(tokens[3], Token::RParen);
    }

    #[test]
    fn test_lexer_keywords() {
        let mut lexer = Lexer::new(":name :type");
        let tokens = lexer.tokenize().expect("test operation should succeed");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], Token::Keyword("name".to_string()));
        assert_eq!(tokens[1], Token::Keyword("type".to_string()));
    }

    #[test]
    fn test_lexer_numbers() {
        let mut lexer = Lexer::new("42 3.14");
        let tokens = lexer.tokenize().expect("test operation should succeed");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], Token::Numeral("42".to_string()));
        assert_eq!(tokens[1], Token::Decimal("3.14".to_string()));
    }

    #[test]
    fn test_lexer_string() {
        let mut lexer = Lexer::new(r#""hello world""#);
        let tokens = lexer.tokenize().expect("test operation should succeed");

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::StringLit("hello world".to_string()));
    }

    #[test]
    fn test_lexer_comments() {
        let mut lexer = Lexer::new("; this is a comment\n(foo bar)");
        let tokens = lexer.tokenize().expect("test operation should succeed");

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::LParen);
        assert_eq!(tokens[1], Token::Symbol("foo".to_string()));
        assert_eq!(tokens[2], Token::Symbol("bar".to_string()));
        assert_eq!(tokens[3], Token::RParen);
    }

    #[test]
    fn test_sexpr_parser_atom() {
        let exprs = SExprParser::parse_str("foo").expect("test operation should succeed");

        assert_eq!(exprs.len(), 1);
        assert!(exprs[0].is_atom());
        assert_eq!(exprs[0].as_symbol(), Some("foo"));
    }

    #[test]
    fn test_sexpr_parser_list() {
        let exprs = SExprParser::parse_str("(foo bar)").expect("test operation should succeed");

        assert_eq!(exprs.len(), 1);
        assert!(exprs[0].is_list());

        let list = exprs[0].as_list().expect("test operation should succeed");
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_symbol(), Some("foo"));
        assert_eq!(list[1].as_symbol(), Some("bar"));
    }

    #[test]
    fn test_sexpr_parser_nested() {
        let exprs =
            SExprParser::parse_str("(foo (bar baz) qux)").expect("test operation should succeed");

        assert_eq!(exprs.len(), 1);
        let list = exprs[0].as_list().expect("test operation should succeed");
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_symbol(), Some("foo"));
        assert!(list[1].is_list());
        assert_eq!(list[2].as_symbol(), Some("qux"));

        let inner = list[1].as_list().expect("test operation should succeed");
        assert_eq!(inner.len(), 2);
        assert_eq!(inner[0].as_symbol(), Some("bar"));
        assert_eq!(inner[1].as_symbol(), Some("baz"));
    }

    #[test]
    fn test_sexpr_parser_multiple() {
        let exprs = SExprParser::parse_str("(foo) (bar)").expect("test operation should succeed");

        assert_eq!(exprs.len(), 2);
        assert!(exprs[0].is_list());
        assert!(exprs[1].is_list());
    }

    #[test]
    fn test_sexpr_parser_error_unclosed() {
        let result = SExprParser::parse_str("(foo bar");
        assert!(result.is_err());
    }

    #[test]
    fn test_sexpr_parser_error_unexpected_close() {
        let result = SExprParser::parse_str("foo)");
        assert!(result.is_err());
    }

    // ── process_assertion tests ───────────────────────────────────────────────

    /// A query assertion `(assert false)` should produce a query rule.
    #[test]
    fn test_parse_assertion_false_query() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let input = "(set-logic HORN)\n\
                     (declare-fun Inv (Int) Bool)\n\
                     (assert false)";
        let result = parser.parse(input);
        assert!(result.is_ok(), "parse failed: {:?}", result);
        let system = result.expect("parse succeeded");
        // A bare `false` assertion becomes a query rule.
        assert!(
            system.num_rules() > 0,
            "expected at least one rule from (assert false)"
        );
    }

    /// A forward Horn clause: `(assert (forall ((x Int)) (=> (and (= x 0)) (Inv x))))`.
    /// Parses correctly and registers 1 predicate and 1 rule.
    #[test]
    fn test_parse_assertion_forall_horn_clause() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        // This input: x=0 implies Inv(x).  One Horn rule with Inv in the head.
        let input = "(set-logic HORN)\n\
                     (declare-fun Inv (Int) Bool)\n\
                     (assert (forall ((x Int)) (=> (= x 0) (Inv x))))";
        let result = parser.parse(input);
        assert!(result.is_ok(), "parse failed: {:?}", result);
        let system = result.expect("parse succeeded");
        assert_eq!(system.num_predicates(), 1, "should have 1 predicate (Inv)");
        assert_eq!(system.num_rules(), 1, "should have 1 Horn rule");
    }

    /// Parsing two Horn clauses produces 2 rules.
    #[test]
    fn test_parse_assertion_two_clauses() {
        let mut terms = TermManager::new();
        let mut parser = ChcParser::new(&mut terms);

        let input = "(set-logic HORN)\n\
                     (declare-fun Inv (Int) Bool)\n\
                     (assert (forall ((x Int)) (=> (= x 0) (Inv x))))\n\
                     (assert (forall ((x Int)) (=> (Inv x) false)))";
        let result = parser.parse(input);
        assert!(result.is_ok(), "parse failed: {:?}", result);
        let system = result.expect("parse succeeded");
        assert_eq!(system.num_predicates(), 1);
        assert_eq!(system.num_rules(), 2, "should have 2 Horn rules");
    }

    /// 100k nested parentheses must parse without overflowing the stack —
    /// the nesting depth is directly attacker-controlled.
    #[test]
    fn parse_sexpr_survives_deep_nesting() {
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — `parse_sexpr`
        // itself is an explicit-stack scan with no native recursion, so the
        // 100k depth here is not bounded by thread stack size at all; 1 MiB
        // is just a conventional headroom, not a tuned minimum. See
        // TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                const DEPTH: usize = 100_000;
                let mut input = String::with_capacity(DEPTH * 2 + 1);
                input.push_str(&"(".repeat(DEPTH));
                input.push('a');
                input.push_str(&")".repeat(DEPTH));

                let parsed = SExprParser::parse_str(&input).expect("deep s-expr must parse");
                assert_eq!(parsed.len(), 1);

                // Walk down iteratively to confirm the shape, then let
                // `parsed` drop: `SExpr`'s own `Drop` is an explicit
                // iterative teardown (see `impl Drop for SExpr` above), so
                // this test pins both the bounded-recursion parse and the
                // iterative drop at once.
                let mut depth = 0usize;
                let mut node = &parsed[0];
                while let SExpr::List(items) = node {
                    depth += 1;
                    match items.first() {
                        Some(first) => node = first,
                        None => break,
                    }
                }
                assert_eq!(depth, DEPTH, "every level must be preserved");
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep parse must return");
    }

    /// The three error cases of `parse_sexpr` must be reported exactly as
    /// before the explicit-stack rewrite.
    #[test]
    fn parse_sexpr_error_cases_are_unchanged() {
        let err = SExprParser::parse_str("(a b").expect_err("unclosed list must fail");
        assert!(
            err.to_string().contains("unclosed parenthesis"),
            "got: {err}"
        );

        let err = SExprParser::parse_str(")").expect_err("stray `)` must fail");
        assert!(
            err.to_string().contains("unexpected closing parenthesis"),
            "got: {err}"
        );

        let mut parser = SExprParser::new(Vec::new());
        let err = parser.parse_sexpr().expect_err("empty input must fail");
        assert!(
            err.to_string().contains("unexpected end of input"),
            "got: {err}"
        );
    }

    /// Nested lists keep their structure and order.
    #[test]
    fn parse_sexpr_preserves_nested_structure() {
        let parsed = SExprParser::parse_str("(a (b c) d)").expect("must parse");
        assert_eq!(parsed.len(), 1);
        let SExpr::List(items) = &parsed[0] else {
            panic!("expected a list");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].as_symbol(), Some("a"));
        let SExpr::List(inner) = &items[1] else {
            panic!("expected a nested list");
        };
        assert_eq!(inner.len(), 2);
        assert_eq!(inner[0].as_symbol(), Some("b"));
        assert_eq!(inner[1].as_symbol(), Some("c"));
        assert_eq!(items[2].as_symbol(), Some("d"));
    }

    /// A term nested far past [`MAX_TERM_NESTING`] must be *rejected*, not
    /// crash the process and not silently produce a truncated term.
    #[test]
    fn parse_term_rejects_excessive_nesting() {
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — the term
        // parser's own bound (`MAX_TERM_NESTING` = 500) is enforced by
        // deliberately bounded native recursion at ~2 KiB/frame; a 128 KiB
        // stack would overflow before the bound could reject the input.
        // See TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                const DEPTH: usize = 100_000;
                let mut input = String::from("(assert ");
                input.push_str(&"(not ".repeat(DEPTH));
                input.push_str("true");
                input.push_str(&")".repeat(DEPTH));
                input.push(')');

                let mut terms = TermManager::new();
                let mut parser = ChcParser::new(&mut terms);
                let err = parser
                    .parse(&input)
                    .expect_err("a 100k-deep term must be refused");
                assert!(
                    err.to_string().contains("term nesting exceeds"),
                    "got: {err}"
                );
            })
            .expect("thread spawn should succeed");
        handle
            .join()
            .expect("over-deep term parsing must return an error, not abort");
    }

    /// The limit itself must be *reachable* on a small stack: a term just
    /// under [`MAX_TERM_NESTING`] parses successfully inside a 1 MiB
    /// thread, so the bound rejects only inputs that would have crashed.
    #[test]
    fn parse_term_accepts_nesting_just_under_the_limit() {
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — same
        // deliberately bounded native recursion as
        // `parse_term_rejects_excessive_nesting` (`MAX_TERM_NESTING` = 500
        // at ~2 KiB/frame); a 128 KiB stack would overflow. See TODO.md
        // "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let depth = MAX_TERM_NESTING - 10;
                let mut input = String::from("(assert ");
                input.push_str(&"(and ".repeat(depth));
                input.push_str("true");
                input.push_str(&")".repeat(depth));
                input.push(')');

                let mut terms = TermManager::new();
                let mut parser = ChcParser::new(&mut terms);
                assert!(
                    parser.parse(&input).is_ok(),
                    "a term just under the limit must still parse"
                );
            })
            .expect("thread spawn should succeed");
        handle
            .join()
            .expect("parsing at the configured depth limit must not overflow");
    }
}
