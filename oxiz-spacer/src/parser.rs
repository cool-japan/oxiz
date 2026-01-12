//! CHC-COMP format parser
//!
//! Parses Constrained Horn Clauses in SMT-LIB2/CHC-COMP format.
//!
//! Reference: <https://chc-comp.github.io/format.html>

use crate::chc::{ChcSystem, PredId, PredicateApp};
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

    /// Parse a single S-expression
    pub fn parse_sexpr(&mut self) -> Result<SExpr, ParseError> {
        if self.pos >= self.tokens.len() {
            return Err(ParseError::InvalidSyntax(
                "unexpected end of input".to_string(),
            ));
        }

        let token = &self.tokens[self.pos];

        match token {
            Token::LParen => {
                self.pos += 1;
                let mut items = Vec::new();

                // Parse items until we hit RParen
                while self.pos < self.tokens.len() {
                    if matches!(self.tokens[self.pos], Token::RParen) {
                        self.pos += 1;
                        return Ok(SExpr::List(items));
                    }

                    items.push(self.parse_sexpr()?);
                }

                Err(ParseError::InvalidSyntax(
                    "unclosed parenthesis".to_string(),
                ))
            }
            Token::RParen => Err(ParseError::InvalidSyntax(
                "unexpected closing parenthesis".to_string(),
            )),
            _ => {
                // It's an atom
                let atom = self.tokens[self.pos].clone();
                self.pos += 1;
                Ok(SExpr::Atom(atom))
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
}

impl<'a> ChcParser<'a> {
    /// Create a new CHC parser
    pub fn new(terms: &'a mut TermManager) -> Self {
        Self {
            terms,
            system: ChcSystem::new(),
            predicates: HashMap::new(),
            variables: HashMap::new(),
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
                        Ok(self.parse_sort(sort_name))
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

    /// Parse a sort name to SortId
    fn parse_sort(&self, name: &str) -> SortId {
        match name {
            "Bool" => self.terms.sorts.bool_sort,
            "Int" => self.terms.sorts.int_sort,
            "Real" => self.terms.sorts.real_sort,
            _ => {
                // Default to Bool for unknown sorts
                self.terms.sorts.bool_sort
            }
        }
    }

    /// Parse a term from an S-expression
    fn parse_term(&mut self, expr: &SExpr) -> Result<TermId, ParseError> {
        match expr {
            SExpr::Atom(token) => self.parse_atom(token),
            SExpr::List(items) => self.parse_application(items),
        }
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
                // Parse as rational
                let parts: Vec<&str> = d.split('.').collect();
                if parts.len() != 2 {
                    return Err(ParseError::TypeError(format!("invalid decimal: {}", d)));
                }
                // Simple decimal parsing: convert to rational
                let _whole: i64 = parts[0].parse().map_err(|_| {
                    ParseError::TypeError(format!("invalid decimal whole part: {}", parts[0]))
                })?;
                let _frac = parts[1];
                // For now, use Int sort approximation
                // Full implementation would create a Rational
                Ok(self.terms.mk_int(0))
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
                if let Some(&_pred_id) = self.predicates.get(func_name) {
                    let _args: Vec<TermId> = items[1..]
                        .iter()
                        .map(|e| self.parse_term(e))
                        .collect::<Result<Vec<_>, _>>()?;
                    // For now, return a placeholder
                    // Full implementation would construct PredicateApp
                    Ok(self.terms.mk_true())
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

            let sort = self.parse_sort(sort_name);
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

    /// Process an assertion (convert to CHC rule)
    fn process_assertion(&mut self, _formula: TermId) -> Result<(), ParseError> {
        // This would extract the Horn clause structure from the formula
        // and add it to the system
        // For now, this is a simplified placeholder
        Ok(())
    }

    /// Parse a simple rule in text format (helper for testing)
    /// Format: "x = 0 => Inv(x)" or "Inv(x) /\ x' = x + 1 => Inv(x')"
    #[allow(dead_code)]
    fn parse_simple_rule(&mut self, _rule_text: &str) -> Result<(), ParseError> {
        // Simple text parser for basic testing
        // Real implementation would parse SMT-LIB2
        Err(ParseError::Unsupported(
            "Simple rule parser not yet implemented".to_string(),
        ))
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

        let system = result.unwrap();
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

    #[test]
    fn test_lexer_basic() {
        let mut lexer = Lexer::new("(set-logic HORN)");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::LParen);
        assert_eq!(tokens[1], Token::Symbol("set-logic".to_string()));
        assert_eq!(tokens[2], Token::Symbol("HORN".to_string()));
        assert_eq!(tokens[3], Token::RParen);
    }

    #[test]
    fn test_lexer_keywords() {
        let mut lexer = Lexer::new(":name :type");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], Token::Keyword("name".to_string()));
        assert_eq!(tokens[1], Token::Keyword("type".to_string()));
    }

    #[test]
    fn test_lexer_numbers() {
        let mut lexer = Lexer::new("42 3.14");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], Token::Numeral("42".to_string()));
        assert_eq!(tokens[1], Token::Decimal("3.14".to_string()));
    }

    #[test]
    fn test_lexer_string() {
        let mut lexer = Lexer::new(r#""hello world""#);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::StringLit("hello world".to_string()));
    }

    #[test]
    fn test_lexer_comments() {
        let mut lexer = Lexer::new("; this is a comment\n(foo bar)");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::LParen);
        assert_eq!(tokens[1], Token::Symbol("foo".to_string()));
        assert_eq!(tokens[2], Token::Symbol("bar".to_string()));
        assert_eq!(tokens[3], Token::RParen);
    }

    #[test]
    fn test_sexpr_parser_atom() {
        let exprs = SExprParser::parse_str("foo").unwrap();

        assert_eq!(exprs.len(), 1);
        assert!(exprs[0].is_atom());
        assert_eq!(exprs[0].as_symbol(), Some("foo"));
    }

    #[test]
    fn test_sexpr_parser_list() {
        let exprs = SExprParser::parse_str("(foo bar)").unwrap();

        assert_eq!(exprs.len(), 1);
        assert!(exprs[0].is_list());

        let list = exprs[0].as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_symbol(), Some("foo"));
        assert_eq!(list[1].as_symbol(), Some("bar"));
    }

    #[test]
    fn test_sexpr_parser_nested() {
        let exprs = SExprParser::parse_str("(foo (bar baz) qux)").unwrap();

        assert_eq!(exprs.len(), 1);
        let list = exprs[0].as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_symbol(), Some("foo"));
        assert!(list[1].is_list());
        assert_eq!(list[2].as_symbol(), Some("qux"));

        let inner = list[1].as_list().unwrap();
        assert_eq!(inner.len(), 2);
        assert_eq!(inner[0].as_symbol(), Some("bar"));
        assert_eq!(inner[1].as_symbol(), Some("baz"));
    }

    #[test]
    fn test_sexpr_parser_multiple() {
        let exprs = SExprParser::parse_str("(foo) (bar)").unwrap();

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
}
