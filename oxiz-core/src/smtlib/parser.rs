//! SMT-LIB2 Parser
#![allow(clippy::while_let_loop)] // Parser uses explicit loop control

use super::lexer::{Lexer, TokenKind};
use crate::ast::{RoundingMode, TermId, TermManager};
use crate::error::{OxizError, Result};
use crate::sort::SortId;
use num_rational::Rational64;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// SMT-LIB2 attribute value
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Symbol value
    Symbol(String),
    /// Numeral value
    Numeral(String),
    /// String value
    String(String),
    /// Term value (for :pattern, etc.)
    Term(TermId),
    /// S-expression (list of values)
    SExpr(Vec<AttributeValue>),
}

/// SMT-LIB2 attribute (key-value pair)
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Attribute keyword (without leading :)
    pub key: String,
    /// Optional attribute value
    pub value: Option<AttributeValue>,
}

/// SMT-LIB2 command
#[derive(Debug, Clone)]
pub enum Command {
    /// Set logic
    SetLogic(String),
    /// Set option
    SetOption(String, String),
    /// Get option
    GetOption(String),
    /// Declare sort
    DeclareSort(String, u32),
    /// Define sort
    DefineSort(String, Vec<String>, String),
    /// Declare datatype
    DeclareDatatype {
        /// Datatype name
        name: String,
        /// Constructors
        constructors: Vec<(String, Vec<(String, String)>)>,
    },
    /// Declare const
    DeclareConst(String, String),
    /// Declare fun
    DeclareFun(String, Vec<String>, String),
    /// Define fun
    DefineFun(String, Vec<(String, String)>, String, TermId),
    /// Assert
    Assert(TermId),
    /// Check sat
    CheckSat,
    /// Check sat with assumptions
    CheckSatAssuming(Vec<TermId>),
    /// Get model
    GetModel,
    /// Get value
    GetValue(Vec<TermId>),
    /// Get unsat core
    GetUnsatCore,
    /// Get assertions
    GetAssertions,
    /// Get assignment
    GetAssignment,
    /// Get proof
    GetProof,
    /// Push
    Push(u32),
    /// Pop
    Pop(u32),
    /// Reset
    Reset,
    /// Reset assertions (keeps declarations)
    ResetAssertions,
    /// Exit
    Exit,
    /// Echo
    Echo(String),
    /// Get info
    GetInfo(String),
    /// Set info
    SetInfo(String, String),
    /// Simplify (Z3 extension)
    Simplify(TermId),
}

/// Parser state
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    manager: &'a mut TermManager,
    /// Variable bindings (for let expressions)
    bindings: FxHashMap<String, TermId>,
    /// Declared constants
    constants: FxHashMap<String, SortId>,
    /// Declared functions
    #[allow(dead_code)]
    functions: FxHashMap<String, (Vec<SortId>, SortId)>,
    /// Sort aliases from define-sort
    sort_aliases: FxHashMap<String, (Vec<String>, String)>,
    /// Function definitions from define-fun
    function_defs: FxHashMap<String, (Vec<(String, String)>, TermId)>,
    /// Term annotations (term -> attributes)
    annotations: FxHashMap<TermId, Vec<Attribute>>,
    /// Error recovery mode enabled
    #[allow(dead_code)]
    recovery_mode: bool,
    /// Collected errors during parsing
    #[allow(dead_code)]
    errors: Vec<OxizError>,
    /// Datatype constructor names -> (datatype_sort, arity/selector_info)
    /// For nullary constructors (enums), the Vec is empty
    dt_constructors: FxHashMap<String, SortId>,
}

impl<'a> Parser<'a> {
    /// Create a new parser
    pub fn new(input: &'a str, manager: &'a mut TermManager) -> Self {
        Self {
            lexer: Lexer::new(input),
            manager,
            bindings: FxHashMap::default(),
            constants: FxHashMap::default(),
            functions: FxHashMap::default(),
            sort_aliases: FxHashMap::default(),
            function_defs: FxHashMap::default(),
            annotations: FxHashMap::default(),
            recovery_mode: false,
            errors: Vec::new(),
            dt_constructors: FxHashMap::default(),
        }
    }

    /// Create a new parser with error recovery enabled
    #[allow(dead_code)]
    pub fn with_recovery(input: &'a str, manager: &'a mut TermManager) -> Self {
        Self {
            lexer: Lexer::new(input),
            manager,
            bindings: FxHashMap::default(),
            constants: FxHashMap::default(),
            functions: FxHashMap::default(),
            sort_aliases: FxHashMap::default(),
            function_defs: FxHashMap::default(),
            annotations: FxHashMap::default(),
            recovery_mode: true,
            errors: Vec::new(),
            dt_constructors: FxHashMap::default(),
        }
    }

    /// Record an error and optionally continue parsing
    #[allow(dead_code)]
    fn record_error(&mut self, error: OxizError) -> Result<()> {
        if self.recovery_mode {
            self.errors.push(error);
            Ok(())
        } else {
            Err(error)
        }
    }

    /// Get all collected errors
    #[must_use]
    #[allow(dead_code)]
    pub fn get_errors(&self) -> &[OxizError] {
        &self.errors
    }

    /// Check if any errors were collected
    #[must_use]
    #[allow(dead_code)]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Synchronize parser state after an error
    /// Skips tokens until we find a safe synchronization point
    #[allow(dead_code)]
    fn synchronize(&mut self) {
        let mut depth = 1;
        while depth > 0 {
            match self.lexer.next_token().map(|t| t.kind) {
                Some(TokenKind::LParen) => depth += 1,
                Some(TokenKind::RParen) => depth -= 1,
                Some(TokenKind::Eof) | None => break,
                _ => {}
            }
        }
    }

    /// Parse a term
    pub fn parse_term(&mut self) -> Result<TermId> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "unexpected end of input".to_string(),
            })?;

        match token.kind {
            TokenKind::LParen => self.parse_compound_term(),
            TokenKind::Symbol(s) => self.parse_symbol(&s),
            TokenKind::Numeral(n) => {
                let value: i64 = n.parse().map_err(|_| OxizError::ParseError {
                    position: token.start,
                    message: format!("invalid numeral: {n}"),
                })?;
                Ok(self.manager.mk_int(value))
            }
            TokenKind::Hexadecimal(h) => {
                let value = i64::from_str_radix(&h, 16).map_err(|_| OxizError::ParseError {
                    position: token.start,
                    message: format!("invalid hexadecimal: {h}"),
                })?;
                let width = (h.len() * 4) as u32;
                Ok(self.manager.mk_bitvec(value, width))
            }
            TokenKind::Binary(b) => {
                let value = i64::from_str_radix(&b, 2).map_err(|_| OxizError::ParseError {
                    position: token.start,
                    message: format!("invalid binary: {b}"),
                })?;
                let width = b.len() as u32;
                Ok(self.manager.mk_bitvec(value, width))
            }
            TokenKind::Decimal(d) => {
                // Parse decimal literal as Rational64
                let rational =
                    parse_decimal_to_rational(&d).map_err(|e| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid decimal: {d} - {e}"),
                    })?;
                Ok(self.manager.mk_real(rational))
            }
            TokenKind::StringLit(s) => {
                // Parse string literal
                Ok(self.manager.mk_string_lit(&s))
            }
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("unexpected token: {:?}", token.kind),
            }),
        }
    }

    fn parse_symbol(&mut self, s: &str) -> Result<TermId> {
        match s {
            "true" => Ok(self.manager.mk_true()),
            "false" => Ok(self.manager.mk_false()),
            _ => {
                // Check bindings first
                if let Some(&term) = self.bindings.get(s) {
                    return Ok(term);
                }
                // Check if this is a datatype constructor (e.g., Monday, nil, cons, etc.)
                if let Some(&dt_sort) = self.dt_constructors.get(s) {
                    return Ok(self.manager.mk_dt_constructor(s, vec![], dt_sort));
                }
                // Check constants
                if let Some(&sort) = self.constants.get(s) {
                    return Ok(self.manager.mk_var(s, sort));
                }
                // Default to boolean variable
                let sort = self.manager.sorts.bool_sort;
                Ok(self.manager.mk_var(s, sort))
            }
        }
    }

    fn parse_compound_term(&mut self) -> Result<TermId> {
        let op_token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "unexpected end of input".to_string(),
            })?;

        // Handle indexed identifiers that start with `(`: ((_ to_fp 8 24) RNE 1.5)
        if matches!(op_token.kind, TokenKind::LParen) {
            // Expect underscore
            let underscore = self.expect_symbol()?;
            if underscore != "_" {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!("expected '_' in indexed identifier, found '{underscore}'"),
                });
            }

            let name = self.expect_symbol()?;

            // Parse indices (can be numerals or symbols, depending on the operation)
            let mut index_parts = Vec::new();
            loop {
                if let Some(token) = self.lexer.peek() {
                    match &token.kind {
                        TokenKind::RParen => {
                            self.lexer.next_token(); // consume rparen
                            break;
                        }
                        TokenKind::Numeral(n) => {
                            let n = n.clone();
                            self.lexer.next_token();
                            index_parts.push(n);
                        }
                        TokenKind::Symbol(s) => {
                            // For datatype testers like (_ is nil), the constructor name is a symbol
                            let s = s.clone();
                            self.lexer.next_token();
                            index_parts.push(s);
                        }
                        _ => {
                            return Err(OxizError::ParseError {
                                position: token.start,
                                message: format!(
                                    "expected numeral, symbol, or ')' in indexed identifier, found {:?}",
                                    token.kind
                                ),
                            });
                        }
                    }
                } else {
                    break;
                }
            }

            // Handle special indexed operators
            if name == "is" {
                // Handle datatype tester: ((_ is constructor) arg)
                if index_parts.len() != 1 {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "(_ is) requires exactly 1 constructor name, got {}",
                            index_parts.len()
                        ),
                    });
                }
                let constructor_name = &index_parts[0];
                let arg = self.parse_term()?;
                self.expect_rparen()?; // Close the outer application
                return Ok(self.manager.mk_dt_tester(constructor_name, arg));
            }

            // Now parse the arguments and closing paren
            // Build the indexed identifier name
            let indices_str = index_parts.join(" ");
            let func_name = if index_parts.is_empty() {
                format!("(_ {name})")
            } else {
                format!("(_ {name} {indices_str})")
            };

            // Parse arguments
            let args = self.parse_term_list()?;
            let sort = self.manager.sorts.bool_sort; // Default
            return Ok(self.manager.mk_apply(&func_name, args, sort));
        }

        // Handle indexed identifiers: (_ extract 7 4), (_ sign_extend 16), etc.
        if matches!(op_token.kind, TokenKind::Symbol(ref s) if s == "_") {
            let name = self.expect_symbol()?;

            // Parse indices (can be numerals or symbols)
            let mut indices = Vec::new();
            let mut index_parts = Vec::new();
            loop {
                if let Some(token) = self.lexer.peek() {
                    match &token.kind {
                        TokenKind::RParen => {
                            break;
                        }
                        TokenKind::Numeral(n) => {
                            let n = n.clone();
                            self.lexer.next_token();
                            let idx = n.parse::<u32>().map_err(|_| OxizError::ParseError {
                                position: token.start,
                                message: format!("invalid index: {n}"),
                            })?;
                            indices.push(idx);
                            index_parts.push(n);
                        }
                        TokenKind::Symbol(s) => {
                            // For datatype testers and similar constructs
                            let s = s.clone();
                            self.lexer.next_token();
                            index_parts.push(s);
                        }
                        _ => break,
                    }
                } else {
                    break;
                }
            }

            // Handle indexed operations
            match name.as_str() {
                "extract" => {
                    if indices.len() != 2 {
                        return Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!(
                                "extract requires exactly 2 indices, got {}",
                                indices.len()
                            ),
                        });
                    }
                    let arg = self.parse_term()?;
                    self.expect_rparen()?;
                    return Ok(self.manager.mk_bv_extract(indices[0], indices[1], arg));
                }
                "is" => {
                    // Handle datatype tester: (_ is constructor) arg
                    if index_parts.len() != 1 {
                        return Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!(
                                "(_ is) requires exactly 1 constructor name, got {}",
                                index_parts.len()
                            ),
                        });
                    }
                    self.expect_rparen()?; // Close the (_ is X) part
                    let constructor_name = &index_parts[0];
                    let arg = self.parse_term()?;
                    self.expect_rparen()?; // Close the outer application
                    return Ok(self.manager.mk_dt_tester(constructor_name, arg));
                }
                _ => {
                    // For unrecognized indexed identifiers (like to_fp, is, etc.),
                    // treat them as function applications and parse arguments
                    // Expect closing paren for the indexed identifier
                    self.expect_rparen()?;

                    // Build the indexed identifier name
                    let indices_str = index_parts.join(" ");
                    let func_name = if index_parts.is_empty() {
                        format!("(_ {name})")
                    } else {
                        format!("(_ {name} {indices_str})")
                    };

                    // Parse arguments
                    let args = self.parse_term_list()?;
                    let sort = self.manager.sorts.bool_sort; // Default
                    return Ok(self.manager.mk_apply(&func_name, args, sort));
                }
            }
        }

        let op = match &op_token.kind {
            TokenKind::Symbol(s) => s.clone(),
            TokenKind::Keyword(k) => format!(":{k}"),
            _ => {
                return Err(OxizError::ParseError {
                    position: op_token.start,
                    message: format!("expected operator, found {:?}", op_token.kind),
                });
            }
        };

        let result = match op.as_str() {
            "!" => {
                // Annotation: (! term :attr1 val1 :attr2 val2 ...)
                let term = self.parse_term()?;
                let attrs = self.parse_attributes()?;
                self.expect_rparen()?;

                // Store annotations for this term
                if !attrs.is_empty() {
                    self.annotations.insert(term, attrs);
                }

                term
            }
            "not" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_not(arg)
            }
            "and" => {
                let args = self.parse_term_list()?;
                self.manager.mk_and(args)
            }
            "or" => {
                let args = self.parse_term_list()?;
                self.manager.mk_or(args)
            }
            "=>" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_implies(lhs, rhs)
            }
            "xor" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                // XOR = (a and not b) or (not a and b)
                let not_lhs = self.manager.mk_not(lhs);
                let not_rhs = self.manager.mk_not(rhs);
                let and1 = self.manager.mk_and([lhs, not_rhs]);
                let and2 = self.manager.mk_and([not_lhs, rhs]);
                self.manager.mk_or([and1, and2])
            }
            "ite" => {
                let cond = self.parse_term()?;
                let then_branch = self.parse_term()?;
                let else_branch = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_ite(cond, then_branch, else_branch)
            }
            "=" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_eq(lhs, rhs)
            }
            "distinct" => {
                let args = self.parse_term_list()?;
                self.manager.mk_distinct(args)
            }
            "+" => {
                let args = self.parse_term_list()?;
                self.manager.mk_add(args)
            }
            "-" => {
                let first = self.parse_term()?;
                if let Some(token) = self.lexer.peek()
                    && matches!(token.kind, TokenKind::RParen)
                {
                    self.lexer.next_token();
                    // Unary minus - create zero of the appropriate sort
                    let zero = if let Some(term) = self.manager.get(first) {
                        if term.sort == self.manager.sorts.real_sort {
                            self.manager.mk_real(Rational64::new(0, 1))
                        } else {
                            self.manager.mk_int(0)
                        }
                    } else {
                        self.manager.mk_int(0)
                    };
                    return Ok(self.manager.mk_sub(zero, first));
                }
                let second = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_sub(first, second)
            }
            "*" => {
                let args = self.parse_term_list()?;
                self.manager.mk_mul(args)
            }
            "div" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                // For now, treat div as subtraction placeholder
                self.manager.mk_sub(lhs, rhs)
            }
            "mod" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                // For now, treat mod as subtraction placeholder
                self.manager.mk_sub(lhs, rhs)
            }
            "<" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_lt(lhs, rhs)
            }
            "<=" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_le(lhs, rhs)
            }
            ">" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_gt(lhs, rhs)
            }
            ">=" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_ge(lhs, rhs)
            }
            "select" => {
                let array = self.parse_term()?;
                let index = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_select(array, index)
            }
            "store" => {
                let array = self.parse_term()?;
                let index = self.parse_term()?;
                let value = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_store(array, index, value)
            }
            "let" => self.parse_let()?,
            "forall" => self.parse_forall()?,
            "exists" => self.parse_exists()?,
            // BitVector operations
            "bvnot" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_not(arg)
            }
            "bvand" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_and(lhs, rhs)
            }
            "bvor" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_or(lhs, rhs)
            }
            "bvadd" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_add(lhs, rhs)
            }
            "bvsub" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_sub(lhs, rhs)
            }
            "bvmul" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_mul(lhs, rhs)
            }
            "bvult" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ult(lhs, rhs)
            }
            "bvslt" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_slt(lhs, rhs)
            }
            "bvule" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ule(lhs, rhs)
            }
            "bvsle" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_sle(lhs, rhs)
            }
            "bvxor" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_xor(lhs, rhs)
            }
            "bvudiv" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_udiv(lhs, rhs)
            }
            "bvsdiv" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_sdiv(lhs, rhs)
            }
            "bvurem" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_urem(lhs, rhs)
            }
            "bvsrem" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_srem(lhs, rhs)
            }
            "bvshl" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_shl(lhs, rhs)
            }
            "bvlshr" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_lshr(lhs, rhs)
            }
            "bvashr" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ashr(lhs, rhs)
            }
            "concat" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_concat(lhs, rhs)
            }
            // Floating-point arithmetic operations (take rounding mode as first argument)
            "fp.add" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_add(rm, lhs, rhs)
            }
            "fp.sub" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_sub(rm, lhs, rhs)
            }
            "fp.mul" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_mul(rm, lhs, rhs)
            }
            "fp.div" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_div(rm, lhs, rhs)
            }
            "fp.rem" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_rem(lhs, rhs)
            }
            "fp.sqrt" => {
                let rm = self.parse_rounding_mode()?;
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_sqrt(rm, arg)
            }
            "fp.fma" => {
                let rm = self.parse_rounding_mode()?;
                let x = self.parse_term()?;
                let y = self.parse_term()?;
                let z = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_fma(rm, x, y, z)
            }
            "fp.roundToIntegral" => {
                let rm = self.parse_rounding_mode()?;
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_round_to_integral(rm, arg)
            }
            // Floating-point comparisons
            "fp.eq" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_eq(lhs, rhs)
            }
            "fp.lt" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_lt(lhs, rhs)
            }
            "fp.gt" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_gt(lhs, rhs)
            }
            "fp.leq" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_leq(lhs, rhs)
            }
            "fp.geq" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_geq(lhs, rhs)
            }
            // Floating-point predicates
            "fp.isNormal" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_normal(arg)
            }
            "fp.isSubnormal" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_subnormal(arg)
            }
            "fp.isZero" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_zero(arg)
            }
            "fp.isInfinite" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_infinite(arg)
            }
            "fp.isNaN" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_nan(arg)
            }
            "fp.isNegative" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_negative(arg)
            }
            "fp.isPositive" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_positive(arg)
            }
            // Floating-point unary operations
            "fp.abs" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_abs(arg)
            }
            "fp.neg" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_neg(arg)
            }
            // Floating-point binary min/max
            "fp.min" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_min(lhs, rhs)
            }
            "fp.max" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_max(lhs, rhs)
            }
            // String operations
            "str.++" => {
                let mut result = self.parse_term()?;
                loop {
                    if let Some(token) = self.lexer.peek()
                        && matches!(token.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    let next = self.parse_term()?;
                    result = self.manager.mk_str_concat(result, next);
                }
                result
            }
            "str.len" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_len(arg)
            }
            "str.substr" => {
                let s = self.parse_term()?;
                let start = self.parse_term()?;
                let len = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_substr(s, start, len)
            }
            "str.at" => {
                let s = self.parse_term()?;
                let i = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_at(s, i)
            }
            "str.contains" => {
                let s = self.parse_term()?;
                let sub = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_contains(s, sub)
            }
            "str.prefixof" => {
                let prefix = self.parse_term()?;
                let s = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_prefixof(prefix, s)
            }
            "str.suffixof" => {
                let suffix = self.parse_term()?;
                let s = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_suffixof(suffix, s)
            }
            "str.indexof" => {
                let s = self.parse_term()?;
                let sub = self.parse_term()?;
                let offset = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_indexof(s, sub, offset)
            }
            "str.replace" => {
                let s = self.parse_term()?;
                let pattern = self.parse_term()?;
                let replacement = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_replace(s, pattern, replacement)
            }
            "str.replace_all" => {
                let s = self.parse_term()?;
                let pattern = self.parse_term()?;
                let replacement = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_replace_all(s, pattern, replacement)
            }
            "str.to_int" | "str.to.int" => {
                let s = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_to_int(s)
            }
            "int.to_str" | "int.to.str" | "str.from_int" => {
                let n = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_int_to_str(n)
            }
            "str.in_re" | "str.in.re" => {
                let s = self.parse_term()?;
                let re = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_in_re(s, re)
            }
            _ => {
                // Check for defined function
                if let Some((params, body)) = self.function_defs.get(&op).cloned() {
                    // Parse arguments
                    let args = self.parse_term_list()?;

                    if args.len() != params.len() {
                        return Err(OxizError::ParseError {
                            position: 0,
                            message: format!(
                                "wrong number of arguments for {}: expected {}, got {}",
                                op,
                                params.len(),
                                args.len()
                            ),
                        });
                    }

                    // Substitute arguments into the body
                    let mut substitution = FxHashMap::default();
                    for ((param_name, _param_sort), &arg) in params.iter().zip(args.iter()) {
                        // Find the parameter variable in the body
                        let param_sort = self
                            .constants
                            .get(param_name)
                            .copied()
                            .unwrap_or(self.manager.sorts.bool_sort);
                        let param_var = self.manager.mk_var(param_name, param_sort);
                        substitution.insert(param_var, arg);
                    }

                    // Apply substitution to get the result
                    self.manager.substitute(body, &substitution)
                } else {
                    // Regular function application
                    let args = self.parse_term_list()?;
                    let sort = self.manager.sorts.bool_sort; // Default
                    self.manager.mk_apply(&op, args, sort)
                }
            }
        };

        Ok(result)
    }

    fn parse_term_list(&mut self) -> Result<SmallVec<[TermId; 4]>> {
        let mut args = SmallVec::new();
        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }
            args.push(self.parse_term()?);
        }
        Ok(args)
    }

    fn expect_rparen(&mut self) -> Result<()> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected ')', found end of input".to_string(),
            })?;

        if !matches!(token.kind, TokenKind::RParen) {
            return Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected ')', found {:?}", token.kind),
            });
        }
        Ok(())
    }

    fn parse_let(&mut self) -> Result<TermId> {
        // Parse bindings: ((name term) ...)
        self.expect_lparen()?;

        let mut new_bindings: Vec<(String, TermId)> = Vec::new();

        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }

            self.expect_lparen()?;
            let name = self.expect_symbol()?;
            let term = self.parse_term()?;
            self.expect_rparen()?;
            new_bindings.push((name, term));
        }

        // Add bindings to scope
        let old_bindings: Vec<_> = new_bindings
            .iter()
            .filter_map(|(name, _)| self.bindings.get(name).map(|&t| (name.clone(), t)))
            .collect();

        for (name, term) in &new_bindings {
            self.bindings.insert(name.clone(), *term);
        }

        // Parse body
        let body = self.parse_term()?;
        self.expect_rparen()?;

        // Restore old bindings
        for (name, _) in &new_bindings {
            self.bindings.remove(name);
        }
        for (name, term) in old_bindings {
            self.bindings.insert(name, term);
        }

        // Create let term
        let bindings: Vec<_> = new_bindings.iter().map(|(n, t)| (n.as_str(), *t)).collect();
        Ok(self.manager.mk_let(bindings, body))
    }

    fn parse_forall(&mut self) -> Result<TermId> {
        // Parse sorted vars: ((name sort) ...)
        self.expect_lparen()?;
        let vars = self.parse_sorted_vars()?;

        // Parse body
        let body = self.parse_term()?;
        self.expect_rparen()?;

        let var_refs: Vec<_> = vars.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        Ok(self.manager.mk_forall(var_refs, body))
    }

    fn parse_exists(&mut self) -> Result<TermId> {
        // Parse sorted vars: ((name sort) ...)
        self.expect_lparen()?;
        let vars = self.parse_sorted_vars()?;

        // Parse body
        let body = self.parse_term()?;
        self.expect_rparen()?;

        let var_refs: Vec<_> = vars.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        Ok(self.manager.mk_exists(var_refs, body))
    }

    fn parse_sorted_vars(&mut self) -> Result<Vec<(String, SortId)>> {
        let mut vars = Vec::new();
        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }

            self.expect_lparen()?;
            let name = self.expect_symbol()?;
            let sort = self.parse_sort()?;
            self.expect_rparen()?;
            vars.push((name, sort));
        }
        Ok(vars)
    }

    fn parse_sort_name(&mut self, name: &str) -> Result<SortId> {
        match name {
            "Bool" => Ok(self.manager.sorts.bool_sort),
            "Int" => Ok(self.manager.sorts.int_sort),
            "Real" => Ok(self.manager.sorts.real_sort),
            "String" => Ok(self.manager.sorts.string_sort()),
            _ => {
                // Check for sort alias first
                if let Some((params, base_sort)) = self.sort_aliases.get(name).cloned() {
                    // For now, only support 0-arity sort aliases
                    if params.is_empty() {
                        return self.parse_sort_name(&base_sort);
                    }
                }

                // Check for BitVec
                // Note: Proper SMT-LIB2 syntax is `(_ BitVec n)` which requires
                // parsing an indexed identifier. For now, we support simple names
                // like "BitVec32" as a compromise.
                if let Some(width_str) = name.strip_prefix("BitVec") {
                    if let Ok(width) = width_str.parse::<u32>() {
                        if width > 0 && width <= 65536 {
                            // Reasonable bit width limit
                            Ok(self.manager.sorts.bitvec(width))
                        } else {
                            Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("invalid BitVec width: {width} (must be 1-65536)"),
                            })
                        }
                    } else if width_str.is_empty() {
                        // Just "BitVec" without width - use default 32
                        Ok(self.manager.sorts.bitvec(32))
                    } else {
                        Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!("invalid BitVec sort name: {name}"),
                        })
                    }
                } else {
                    // Uninterpreted sort
                    let spur = self.manager.intern_str(name);
                    Ok(self
                        .manager
                        .sorts
                        .intern(crate::sort::SortKind::Uninterpreted(spur)))
                }
            }
        }
    }

    /// Parse an indexed identifier: (_ name index1 index2 ...)
    /// Returns (name, indices)
    fn parse_indexed_identifier(&mut self) -> Result<(String, Vec<u32>)> {
        // Expect LParen (already consumed by caller)
        // Expect underscore symbol
        let underscore = self.expect_symbol()?;
        if underscore != "_" {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("expected '_', found '{underscore}'"),
            });
        }

        // Get the identifier name
        let name = self.expect_symbol()?;

        // Parse indices (numerals)
        let mut indices = Vec::new();
        loop {
            if let Some(token) = self.lexer.peek() {
                match &token.kind {
                    TokenKind::RParen => {
                        self.lexer.next_token(); // consume rparen
                        break;
                    }
                    TokenKind::Numeral(n) => {
                        let n = n.clone();
                        self.lexer.next_token();
                        let idx = n.parse::<u32>().map_err(|_| OxizError::ParseError {
                            position: token.start,
                            message: format!("invalid index: {n}"),
                        })?;
                        indices.push(idx);
                    }
                    _ => {
                        return Err(OxizError::ParseError {
                            position: token.start,
                            message: format!("expected numeral or ')', found {:?}", token.kind),
                        });
                    }
                }
            } else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input in indexed identifier".to_string(),
                });
            }
        }

        Ok((name, indices))
    }

    /// Parse a sort (can be a simple name or indexed identifier)
    fn parse_sort(&mut self) -> Result<SortId> {
        if let Some(token) = self.lexer.peek() {
            match &token.kind {
                TokenKind::Symbol(s) => {
                    let s = s.clone();
                    self.lexer.next_token();
                    self.parse_sort_name(&s)
                }
                TokenKind::LParen => {
                    self.lexer.next_token(); // consume lparen

                    // Check if this is an indexed identifier or a parametric sort like Array
                    let next_token = self.lexer.peek().ok_or_else(|| OxizError::ParseError {
                        position: self.lexer.position(),
                        message: "unexpected end of input in sort".to_string(),
                    })?;

                    if matches!(next_token.kind, TokenKind::Symbol(ref s) if s == "_") {
                        // Indexed identifier: (_ BitVec 32)
                        let (name, indices) = self.parse_indexed_identifier()?;

                        // Handle indexed sorts
                        match name.as_str() {
                            "BitVec" => {
                                if indices.len() != 1 {
                                    return Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "BitVec requires exactly 1 index, got {}",
                                            indices.len()
                                        ),
                                    });
                                }
                                let width = indices[0];
                                if width > 0 && width <= 65536 {
                                    Ok(self.manager.sorts.bitvec(width))
                                } else {
                                    Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "invalid BitVec width: {width} (must be 1-65536)"
                                        ),
                                    })
                                }
                            }
                            "FloatingPoint" => {
                                if indices.len() != 2 {
                                    return Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "FloatingPoint requires exactly 2 indices (eb, sb), got {}",
                                            indices.len()
                                        ),
                                    });
                                }
                                let eb = indices[0]; // exponent bits
                                let sb = indices[1]; // significand bits
                                Ok(self.manager.sorts.float_sort(eb, sb))
                            }
                            _ => Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("unknown indexed sort: {name}"),
                            }),
                        }
                    } else if let TokenKind::Symbol(s) = &next_token.kind {
                        // Parametric sort like (Array Int Int)
                        let sort_name = s.clone();
                        self.lexer.next_token(); // consume the symbol

                        match sort_name.as_str() {
                            "Array" => {
                                // Parse domain and range sorts
                                let domain = self.parse_sort()?;
                                let range = self.parse_sort()?;
                                self.expect_rparen()?;
                                Ok(self.manager.sorts.array(domain, range))
                            }
                            _ => Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("unknown parametric sort: {sort_name}"),
                            }),
                        }
                    } else {
                        Err(OxizError::ParseError {
                            position: next_token.start,
                            message: format!("unexpected token in sort: {:?}", next_token.kind),
                        })
                    }
                }
                _ => Err(OxizError::ParseError {
                    position: token.start,
                    message: format!("expected sort, found {:?}", token.kind),
                }),
            }
        } else {
            Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected sort, found end of input".to_string(),
            })
        }
    }

    /// Convert a SortId to its string representation for Command storage
    fn sort_id_to_string(&self, sort_id: SortId) -> String {
        if let Some(sort) = self.manager.sorts.get(sort_id) {
            match &sort.kind {
                crate::sort::SortKind::Bool => "Bool".to_string(),
                crate::sort::SortKind::Int => "Int".to_string(),
                crate::sort::SortKind::Real => "Real".to_string(),
                crate::sort::SortKind::String => "String".to_string(),
                crate::sort::SortKind::BitVec(w) => format!("(_ BitVec {w})"),
                crate::sort::SortKind::FloatingPoint { eb, sb } => {
                    format!("(_ FloatingPoint {eb} {sb})")
                }
                crate::sort::SortKind::Array { domain, range } => {
                    let domain_str = self.sort_id_to_string(*domain);
                    let range_str = self.sort_id_to_string(*range);
                    format!("(Array {domain_str} {range_str})")
                }
                crate::sort::SortKind::Uninterpreted(spur) => {
                    self.manager.resolve_str(*spur).to_string()
                }
                crate::sort::SortKind::Datatype(spur) => {
                    self.manager.resolve_str(*spur).to_string()
                }
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        }
    }

    fn expect_lparen(&mut self) -> Result<()> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected '(', found end of input".to_string(),
            })?;

        if !matches!(token.kind, TokenKind::LParen) {
            return Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected '(', found {:?}", token.kind),
            });
        }
        Ok(())
    }

    fn expect_symbol(&mut self) -> Result<String> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected symbol, found end of input".to_string(),
            })?;

        match token.kind {
            TokenKind::Symbol(s) => Ok(s),
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected symbol, found {:?}", token.kind),
            }),
        }
    }

    /// Parse a command
    pub fn parse_command(&mut self) -> Result<Option<Command>> {
        let token = match self.lexer.next_token() {
            Some(t) if matches!(t.kind, TokenKind::Eof) => return Ok(None),
            Some(t) => t,
            None => return Ok(None),
        };

        if !matches!(token.kind, TokenKind::LParen) {
            return Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected '(', found {:?}", token.kind),
            });
        }

        let cmd_name = self.expect_symbol()?;

        let cmd = match cmd_name.as_str() {
            "set-logic" => {
                let logic = self.expect_symbol()?;
                self.expect_rparen()?;
                Command::SetLogic(logic)
            }
            "set-option" => {
                let opt = self.expect_keyword()?;
                let val = self.expect_symbol().unwrap_or_default();
                self.expect_rparen()?;
                Command::SetOption(opt, val)
            }
            "declare-const" => {
                let name = self.expect_symbol()?;
                let sort_id = self.parse_sort()?;
                self.expect_rparen()?;
                self.constants.insert(name.clone(), sort_id);
                // For the command, we'll use a simple string representation
                let sort_str = format!(
                    "BitVec{}",
                    self.manager
                        .sorts
                        .get(sort_id)
                        .and_then(|s| s.bitvec_width())
                        .unwrap_or(32)
                );
                Command::DeclareConst(name, sort_str)
            }
            "declare-fun" => {
                let name = self.expect_symbol()?;
                self.expect_lparen()?;
                let mut arg_sorts = Vec::new();
                let mut arg_sort_ids = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    let sort_id = self.parse_sort()?;
                    arg_sort_ids.push(sort_id);
                    arg_sorts.push(self.sort_id_to_string(sort_id));
                }
                let ret_sort_id = self.parse_sort()?;
                let ret_sort = self.sort_id_to_string(ret_sort_id);
                self.expect_rparen()?;

                if arg_sorts.is_empty() {
                    self.constants.insert(name.clone(), ret_sort_id);
                }
                Command::DeclareFun(name, arg_sorts, ret_sort)
            }
            "assert" => {
                let term = self.parse_term()?;
                self.expect_rparen()?;
                Command::Assert(term)
            }
            "check-sat" => {
                self.expect_rparen()?;
                Command::CheckSat
            }
            "get-model" => {
                self.expect_rparen()?;
                Command::GetModel
            }
            "get-value" => {
                self.expect_lparen()?;
                let mut terms = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    terms.push(self.parse_term()?);
                }
                self.expect_rparen()?;
                Command::GetValue(terms)
            }
            "push" => {
                let n = if let Some(t) = self.lexer.peek() {
                    if matches!(t.kind, TokenKind::Numeral(_)) {
                        if let Some(token) = self.lexer.next_token() {
                            if let TokenKind::Numeral(n) = token.kind {
                                n.parse().unwrap_or(1)
                            } else {
                                1
                            }
                        } else {
                            1
                        }
                    } else {
                        1
                    }
                } else {
                    1
                };
                self.expect_rparen()?;
                Command::Push(n)
            }
            "pop" => {
                let n = if let Some(t) = self.lexer.peek() {
                    if matches!(t.kind, TokenKind::Numeral(_)) {
                        if let Some(token) = self.lexer.next_token() {
                            if let TokenKind::Numeral(n) = token.kind {
                                n.parse().unwrap_or(1)
                            } else {
                                1
                            }
                        } else {
                            1
                        }
                    } else {
                        1
                    }
                } else {
                    1
                };
                self.expect_rparen()?;
                Command::Pop(n)
            }
            "reset" => {
                self.expect_rparen()?;
                Command::Reset
            }
            "reset-assertions" => {
                self.expect_rparen()?;
                Command::ResetAssertions
            }
            "get-assertions" => {
                self.expect_rparen()?;
                Command::GetAssertions
            }
            "get-assignment" => {
                self.expect_rparen()?;
                Command::GetAssignment
            }
            "get-proof" => {
                self.expect_rparen()?;
                Command::GetProof
            }
            "get-option" => {
                let opt = self.expect_keyword()?;
                self.expect_rparen()?;
                Command::GetOption(opt)
            }
            "check-sat-assuming" => {
                self.expect_lparen()?;
                let mut assumptions = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    assumptions.push(self.parse_term()?);
                }
                self.expect_rparen()?;
                Command::CheckSatAssuming(assumptions)
            }
            "simplify" => {
                let term = self.parse_term()?;
                self.expect_rparen()?;
                Command::Simplify(term)
            }
            "exit" => {
                self.expect_rparen()?;
                Command::Exit
            }
            "echo" => {
                let msg = self.expect_string()?;
                self.expect_rparen()?;
                Command::Echo(msg)
            }
            "set-info" => {
                let keyword = self.expect_keyword()?;
                let value = self.expect_symbol().or_else(|_| self.expect_string())?;
                self.expect_rparen()?;
                Command::SetInfo(keyword, value)
            }
            "get-info" => {
                let keyword = self.expect_keyword()?;
                self.expect_rparen()?;
                Command::GetInfo(keyword)
            }
            "define-sort" => {
                // (define-sort name (params) sort-expr)
                let name = self.expect_symbol()?;
                self.expect_lparen()?;
                let mut params = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    params.push(self.expect_symbol()?);
                }
                let sort_expr = self.expect_symbol()?;
                self.expect_rparen()?;

                // Register the sort alias for later use
                self.sort_aliases
                    .insert(name.clone(), (params.clone(), sort_expr.clone()));

                Command::DefineSort(name, params, sort_expr)
            }
            "define-fun" => {
                // (define-fun name ((param sort) ...) ret-sort body)
                let name = self.expect_symbol()?;
                self.expect_lparen()?;

                // Parse parameters
                let mut params: Vec<(String, String)> = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    self.expect_lparen()?;
                    let param_name = self.expect_symbol()?;
                    let param_sort = self.expect_symbol()?;
                    self.expect_rparen()?;
                    params.push((param_name, param_sort));
                }

                let ret_sort = self.expect_symbol()?;

                // Add parameters as local bindings for parsing the body
                let old_bindings: Vec<(String, TermId)> = params
                    .iter()
                    .filter_map(|(pname, _)| self.bindings.get(pname).map(|&t| (pname.clone(), t)))
                    .collect();

                // Create placeholder terms for parameters
                for (pname, psort) in &params {
                    let sort_id = self.parse_sort_name(psort)?;
                    let param_term = self.manager.mk_var(pname, sort_id);
                    self.bindings.insert(pname.clone(), param_term);
                }

                // Parse body
                let body = self.parse_term()?;
                self.expect_rparen()?;

                // Restore old bindings
                for (pname, _) in &params {
                    self.bindings.remove(pname);
                }
                for (pname, term) in old_bindings {
                    self.bindings.insert(pname, term);
                }

                // Register the function definition for later use
                self.function_defs
                    .insert(name.clone(), (params.clone(), body));

                Command::DefineFun(name, params, ret_sort, body)
            }
            "declare-datatypes" => {
                // (declare-datatypes ((name1 arity1) (name2 arity2) ...)
                //                    ((constructors1 ...) (constructors2 ...)))
                // For now, we'll support single datatype declarations only
                // Skip the names/arities list
                self.expect_lparen()?;

                // Parse datatype names and arities
                let mut datatype_names = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }

                    self.expect_lparen()?;
                    let dt_name = self.expect_symbol()?;
                    // Skip the arity (we don't use it yet)
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::Numeral(_))
                    {
                        self.lexer.next_token();
                    }
                    self.expect_rparen()?;
                    datatype_names.push(dt_name);
                }

                // Parse constructors list
                self.expect_lparen()?;

                // For simplicity, parse only the first datatype's constructors
                // (multi-datatype support can be added later)
                let mut constructors = Vec::new();

                // Expect opening paren for constructor list
                self.expect_lparen()?;

                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }

                    // Parse constructor
                    self.expect_lparen()?;
                    let ctor_name = self.expect_symbol()?;

                    // Parse selectors
                    let mut selectors = Vec::new();
                    loop {
                        if let Some(t) = self.lexer.peek()
                            && matches!(t.kind, TokenKind::RParen)
                        {
                            self.lexer.next_token();
                            break;
                        }

                        self.expect_lparen()?;
                        let selector_name = self.expect_symbol()?;
                        let selector_sort = self.expect_symbol()?;
                        self.expect_rparen()?;
                        selectors.push((selector_name, selector_sort));
                    }

                    constructors.push((ctor_name, selectors));
                }

                // Close constructor list and outer list
                self.expect_rparen()?;
                self.expect_rparen()?;

                // Use the first datatype name
                let name = datatype_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "UnknownDatatype".to_string());

                // Create the datatype sort and register constructors
                let dt_sort = self.manager.sorts.mk_datatype_sort(&name);
                for (ctor_name, _selectors) in &constructors {
                    self.dt_constructors.insert(ctor_name.clone(), dt_sort);
                }

                Command::DeclareDatatype { name, constructors }
            }
            "declare-datatype" => {
                // (declare-datatype name ((constructor (selector sort) ...) ...))
                let name = self.expect_symbol()?;
                self.expect_lparen()?;

                let mut constructors = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }

                    // Parse constructor
                    self.expect_lparen()?;
                    let ctor_name = self.expect_symbol()?;

                    // Parse selectors
                    let mut selectors = Vec::new();
                    loop {
                        if let Some(t) = self.lexer.peek()
                            && matches!(t.kind, TokenKind::RParen)
                        {
                            self.lexer.next_token();
                            break;
                        }

                        self.expect_lparen()?;
                        let selector_name = self.expect_symbol()?;
                        let selector_sort = self.expect_symbol()?;
                        self.expect_rparen()?;
                        selectors.push((selector_name, selector_sort));
                    }

                    constructors.push((ctor_name, selectors));
                }

                self.expect_rparen()?;

                // Create the datatype sort and register constructors
                let dt_sort = self.manager.sorts.mk_datatype_sort(&name);
                for (ctor_name, _selectors) in &constructors {
                    self.dt_constructors.insert(ctor_name.clone(), dt_sort);
                }

                Command::DeclareDatatype { name, constructors }
            }
            _ => {
                // Skip unknown command
                let mut depth = 1;
                while depth > 0 {
                    match self.lexer.next_token().map(|t| t.kind) {
                        Some(TokenKind::LParen) => depth += 1,
                        Some(TokenKind::RParen) => depth -= 1,
                        Some(TokenKind::Eof) | None => break,
                        _ => {}
                    }
                }
                return self.parse_command();
            }
        };

        Ok(Some(cmd))
    }

    fn expect_keyword(&mut self) -> Result<String> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected keyword, found end of input".to_string(),
            })?;

        match token.kind {
            TokenKind::Keyword(k) => Ok(k),
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected keyword, found {:?}", token.kind),
            }),
        }
    }

    fn expect_string(&mut self) -> Result<String> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected string, found end of input".to_string(),
            })?;

        match token.kind {
            TokenKind::StringLit(s) => Ok(s),
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected string, found {:?}", token.kind),
            }),
        }
    }

    /// Parse an IEEE 754 rounding mode symbol
    fn parse_rounding_mode(&mut self) -> Result<RoundingMode> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected rounding mode, found end of input".to_string(),
            })?;

        match &token.kind {
            TokenKind::Symbol(s) => match s.as_str() {
                "RNE" | "roundNearestTiesToEven" => Ok(RoundingMode::RNE),
                "RNA" | "roundNearestTiesToAway" => Ok(RoundingMode::RNA),
                "RTP" | "roundTowardPositive" => Ok(RoundingMode::RTP),
                "RTN" | "roundTowardNegative" => Ok(RoundingMode::RTN),
                "RTZ" | "roundTowardZero" => Ok(RoundingMode::RTZ),
                _ => Err(OxizError::ParseError {
                    position: token.start,
                    message: format!("unknown rounding mode: {}", s),
                }),
            },
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected rounding mode symbol, found {:?}", token.kind),
            }),
        }
    }

    /// Parse attributes in an annotation
    fn parse_attributes(&mut self) -> Result<Vec<Attribute>> {
        let mut attrs = Vec::new();

        loop {
            // Check if we've reached the closing paren
            if let Some(token) = self.lexer.peek() {
                if matches!(token.kind, TokenKind::RParen) {
                    break;
                }

                // Attributes start with a keyword (e.g., :named, :pattern)
                if let TokenKind::Keyword(key) = &token.kind {
                    let key = key.clone();
                    self.lexer.next_token(); // consume the keyword

                    // Try to parse the attribute value
                    let value = if let Some(next_token) = self.lexer.peek() {
                        match &next_token.kind {
                            // If next is a keyword or rparen, this attribute has no value
                            TokenKind::Keyword(_) | TokenKind::RParen => None,
                            // Otherwise, parse the value
                            _ => Some(self.parse_attribute_value()?),
                        }
                    } else {
                        None
                    };

                    attrs.push(Attribute { key, value });
                } else {
                    return Err(OxizError::ParseError {
                        position: token.start,
                        message: format!("expected keyword in annotation, found {:?}", token.kind),
                    });
                }
            } else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input in annotation".to_string(),
                });
            }
        }

        Ok(attrs)
    }

    /// Parse an attribute value
    fn parse_attribute_value(&mut self) -> Result<AttributeValue> {
        let token = self.lexer.peek().ok_or_else(|| OxizError::ParseError {
            position: self.lexer.position(),
            message: "unexpected end of input in attribute value".to_string(),
        })?;

        match &token.kind {
            TokenKind::Symbol(s) => {
                let s = s.clone();
                self.lexer.next_token();
                Ok(AttributeValue::Symbol(s))
            }
            TokenKind::Numeral(n) => {
                let n = n.clone();
                self.lexer.next_token();
                Ok(AttributeValue::Numeral(n))
            }
            TokenKind::StringLit(s) => {
                let s = s.clone();
                self.lexer.next_token();
                Ok(AttributeValue::String(s))
            }
            TokenKind::LParen => {
                // Could be an S-expression or a term
                // For :pattern, this would be a term list
                self.lexer.next_token(); // consume lparen
                let mut values = Vec::new();

                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }

                    // Try to parse as term first
                    let term = self.parse_term()?;
                    values.push(AttributeValue::Term(term));
                }

                Ok(AttributeValue::SExpr(values))
            }
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("unexpected token in attribute value: {:?}", token.kind),
            }),
        }
    }
}

/// Parse a decimal string to a Rational64
/// Handles decimal literals like "5.5", "3.14159", "0.0", etc.
fn parse_decimal_to_rational(s: &str) -> Result<Rational64> {
    // Split by decimal point
    let parts: Vec<&str> = s.split('.').collect();

    if parts.len() != 2 {
        return Err(OxizError::ParseError {
            position: 0,
            message: format!("invalid decimal format: {s}"),
        });
    }

    let integer_part = parts[0];
    let fractional_part = parts[1];

    // Parse integer part (can be empty for decimals like ".5")
    let integer_value: i64 = if integer_part.is_empty() {
        0
    } else {
        integer_part.parse().map_err(|_| OxizError::ParseError {
            position: 0,
            message: format!("invalid integer part in decimal: {integer_part}"),
        })?
    };

    // Parse fractional part
    let fractional_digits = fractional_part.len();
    let fractional_value: i64 = fractional_part.parse().map_err(|_| OxizError::ParseError {
        position: 0,
        message: format!("invalid fractional part in decimal: {fractional_part}"),
    })?;

    // Convert to rational: integer_part + fractional_part / 10^fractional_digits
    let denominator = 10_i64
        .checked_pow(fractional_digits as u32)
        .ok_or_else(|| OxizError::ParseError {
            position: 0,
            message: format!("decimal has too many fractional digits: {s}"),
        })?;

    // Create rational: (integer_part * denominator + fractional_value) / denominator
    let numerator = integer_value
        .checked_mul(denominator)
        .and_then(|n| n.checked_add(fractional_value))
        .ok_or_else(|| OxizError::ParseError {
            position: 0,
            message: format!("decimal value overflow: {s}"),
        })?;

    Ok(Rational64::new(numerator, denominator))
}

/// Parse a term from a string
pub fn parse_term(input: &str, manager: &mut TermManager) -> Result<TermId> {
    let mut parser = Parser::new(input, manager);
    parser.parse_term()
}

/// Parse an SMT-LIB2 script
pub fn parse_script(input: &str, manager: &mut TermManager) -> Result<Vec<Command>> {
    let mut parser = Parser::new(input, manager);
    let mut commands = Vec::new();
    while let Some(cmd) = parser.parse_command()? {
        commands.push(cmd);
    }
    Ok(commands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constants() {
        let mut manager = TermManager::new();

        let t = parse_term("true", &mut manager).expect("should parse true");
        assert_eq!(t, manager.mk_true());

        let f = parse_term("false", &mut manager).expect("should parse false");
        assert_eq!(f, manager.mk_false());

        let n = parse_term("42", &mut manager).expect("should parse 42");
        let expected = manager.mk_int(42);
        assert_eq!(n, expected);
    }

    #[test]
    fn test_parse_boolean_ops() {
        let mut manager = TermManager::new();

        let not_true = parse_term("(not true)", &mut manager).expect("should parse (not true)");
        assert_eq!(not_true, manager.mk_false());

        let and_expr =
            parse_term("(and true false)", &mut manager).expect("should parse (and true false)");
        assert_eq!(and_expr, manager.mk_false());

        let or_expr =
            parse_term("(or true false)", &mut manager).expect("should parse (or true false)");
        assert_eq!(or_expr, manager.mk_true());
    }

    #[test]
    fn test_parse_arithmetic() {
        let mut manager = TermManager::new();

        let _add = parse_term("(+ 1 2 3)", &mut manager).expect("should parse (+ 1 2 3)");
        let _lt = parse_term("(< x y)", &mut manager).expect("should parse (< x y)");
    }

    #[test]
    fn test_parse_script() {
        let mut manager = TermManager::new();
        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Int)
            (assert (< x y))
            (check-sat)
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse script");
        assert_eq!(commands.len(), 5);
    }

    #[test]
    fn test_parse_define_sort() {
        let mut manager = TermManager::new();
        let script = r#"
            (define-sort MyInt () Int)
            (declare-const x MyInt)
            (check-sat)
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse define-sort script");
        assert_eq!(commands.len(), 3);

        // Check that define-sort command is correctly parsed
        match &commands[0] {
            Command::DefineSort(name, params, body) => {
                assert_eq!(name, "MyInt");
                assert!(params.is_empty());
                assert_eq!(body, "Int");
            }
            _ => panic!("Expected DefineSort command"),
        }
    }

    #[test]
    fn test_parse_define_fun() {
        let mut manager = TermManager::new();
        let script = r#"
            (define-fun double ((x Int)) Int (+ x x))
            (declare-const y Int)
            (assert (= y (double 5)))
            (check-sat)
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse define-fun script");
        assert_eq!(commands.len(), 4);

        // Check that define-fun command is correctly parsed
        match &commands[0] {
            Command::DefineFun(name, params, ret_sort, _body) => {
                assert_eq!(name, "double");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].0, "x");
                assert_eq!(params[0].1, "Int");
                assert_eq!(ret_sort, "Int");
            }
            _ => panic!("Expected DefineFun command"),
        }
    }

    #[test]
    fn test_parse_define_fun_nullary() {
        let mut manager = TermManager::new();
        let script = r#"
            (define-fun five () Int 5)
            (assert (= 5 (five)))
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse nullary define-fun script");
        assert_eq!(commands.len(), 3);

        match &commands[0] {
            Command::DefineFun(name, params, ret_sort, _body) => {
                assert_eq!(name, "five");
                assert!(params.is_empty());
                assert_eq!(ret_sort, "Int");
            }
            _ => panic!("Expected DefineFun command"),
        }
    }

    #[test]
    fn test_parse_new_commands() {
        let mut manager = TermManager::new();
        let script = r#"
            (set-logic QF_LIA)
            (get-assertions)
            (get-assignment)
            (get-proof)
            (get-option :produce-models)
            (reset-assertions)
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse new commands script");
        assert_eq!(commands.len(), 7);

        assert!(matches!(&commands[0], Command::SetLogic(_)));
        assert!(matches!(&commands[1], Command::GetAssertions));
        assert!(matches!(&commands[2], Command::GetAssignment));
        assert!(matches!(&commands[3], Command::GetProof));
        assert!(matches!(&commands[4], Command::GetOption(opt) if opt == "produce-models"));
        assert!(matches!(&commands[5], Command::ResetAssertions));
        assert!(matches!(&commands[6], Command::CheckSat));
    }

    #[test]
    fn test_parse_check_sat_assuming() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const p Bool)
            (declare-const q Bool)
            (check-sat-assuming (p q))
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse check-sat-assuming script");
        assert_eq!(commands.len(), 3);

        match &commands[2] {
            Command::CheckSatAssuming(assumptions) => {
                assert_eq!(assumptions.len(), 2);
            }
            _ => panic!("Expected CheckSatAssuming command"),
        }
    }

    #[test]
    fn test_parse_simplify() {
        let mut manager = TermManager::new();
        let script = r#"
            (simplify (+ 1 2))
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse simplify script");
        assert_eq!(commands.len(), 1);

        assert!(matches!(&commands[0], Command::Simplify(_)));
    }

    #[test]
    fn test_parse_annotations() {
        let mut manager = TermManager::new();

        // Test :named annotation
        let mut parser = Parser::new("(! (> x 0) :named myAssertion)", &mut manager);
        let term = parser.parse_term().expect("should parse annotated term");

        // Check that annotations were stored
        assert!(parser.annotations.contains_key(&term));
        let attrs = &parser.annotations[&term];
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].key, "named");
        assert!(matches!(
            attrs[0].value,
            Some(AttributeValue::Symbol(ref s)) if s == "myAssertion"
        ));
    }

    #[test]
    fn test_parse_pattern_annotation() {
        let mut manager = TermManager::new();

        // Test :pattern annotation with term list
        let mut parser = Parser::new(
            "(forall ((x Int)) (! (> x 0) :pattern ((f x))))",
            &mut manager,
        );
        let _term = parser
            .parse_term()
            .expect("should parse pattern annotation");

        // The annotation should be present on the body of the forall
        // We just verify that it parses without error for now
    }

    #[test]
    fn test_parse_multiple_annotations() {
        let mut manager = TermManager::new();

        // Test multiple annotations
        let mut parser = Parser::new("(! (> x 0) :named test :weight 10)", &mut manager);
        let term = parser
            .parse_term()
            .expect("should parse multiple annotations");

        // Check annotations
        assert!(parser.annotations.contains_key(&term));
        let attrs = &parser.annotations[&term];
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].key, "named");
        assert_eq!(attrs[1].key, "weight");
    }

    #[test]
    fn test_error_recovery() {
        let mut manager = TermManager::new();

        // Valid script to test error recovery infrastructure
        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (check-sat)
        "#;

        // Parse with recovery enabled
        let mut parser = Parser::with_recovery(script, &mut manager);

        // Parse commands
        let mut count = 0;
        while let Ok(Some(_)) = parser.parse_command() {
            count += 1;
        }

        // Should successfully parse all valid commands
        assert_eq!(count, 3);
        assert!(!parser.has_errors());
    }

    #[test]
    fn test_error_recovery_infrastructure() {
        let mut manager = TermManager::new();

        // Simple valid script
        let script = r#"
            (set-logic QF_LIA)
            (check-sat)
        "#;

        let mut parser = Parser::with_recovery(script, &mut manager);
        let mut commands = Vec::new();

        while let Ok(Some(cmd)) = parser.parse_command() {
            commands.push(cmd);
        }

        // Should successfully parse valid commands
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn test_parse_decimal_literals() {
        let mut manager = TermManager::new();

        // Test simple decimal
        let decimal = parse_term("5.5", &mut manager).expect("should parse 5.5");
        if let Some(term) = manager.get(decimal) {
            assert_eq!(term.sort, manager.sorts.real_sort);
            if let crate::ast::TermKind::RealConst(r) = &term.kind {
                assert_eq!(*r, Rational64::new(11, 2)); // 5.5 = 11/2
            } else {
                panic!("Expected RealConst");
            }
        } else {
            panic!("Term not found");
        }

        // Test decimal with many fractional digits
        let pi_approx = parse_term("3.14159", &mut manager).expect("should parse 3.14159");
        if let Some(term) = manager.get(pi_approx) {
            assert_eq!(term.sort, manager.sorts.real_sort);
            if let crate::ast::TermKind::RealConst(r) = &term.kind {
                assert_eq!(*r, Rational64::new(314159, 100000));
            } else {
                panic!("Expected RealConst");
            }
        } else {
            panic!("Term not found");
        }

        // Test zero decimal
        let zero = parse_term("0.0", &mut manager).expect("should parse 0.0");
        if let Some(term) = manager.get(zero) {
            assert_eq!(term.sort, manager.sorts.real_sort);
            if let crate::ast::TermKind::RealConst(r) = &term.kind {
                assert_eq!(*r, Rational64::new(0, 1));
            } else {
                panic!("Expected RealConst");
            }
        } else {
            panic!("Term not found");
        }
    }

    #[test]
    fn test_parse_real_arithmetic() {
        let mut manager = TermManager::new();
        let script = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (declare-const y Real)
            (assert (= (+ x y) 5.5))
            (assert (> x 0.0))
            (assert (< y 10.25))
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse real arithmetic script");
        assert_eq!(commands.len(), 7); // set-logic, 2 declares, 3 asserts, check-sat

        // Verify that Real constants are declared
        assert!(matches!(&commands[0], Command::SetLogic(_)));
    }

    #[test]
    fn test_parse_unary_minus_real() {
        let mut manager = TermManager::new();

        // Test unary minus with Real
        let neg_real = parse_term("(- 2.5)", &mut manager).expect("should parse (- 2.5)");
        if let Some(term) = manager.get(neg_real) {
            // Should be a subtraction: 0.0 - 2.5
            if let crate::ast::TermKind::Sub(lhs, rhs) = &term.kind {
                // Check lhs is Real zero
                if let Some(lhs_term) = manager.get(*lhs) {
                    assert_eq!(lhs_term.sort, manager.sorts.real_sort);
                    if let crate::ast::TermKind::RealConst(r) = &lhs_term.kind {
                        assert_eq!(*r, Rational64::new(0, 1));
                    } else {
                        panic!("Expected RealConst for zero");
                    }
                } else {
                    panic!("LHS term not found");
                }
                // Check rhs is 2.5
                if let Some(rhs_term) = manager.get(*rhs) {
                    assert_eq!(rhs_term.sort, manager.sorts.real_sort);
                    if let crate::ast::TermKind::RealConst(r) = &rhs_term.kind {
                        assert_eq!(*r, Rational64::new(5, 2)); // 2.5 = 5/2
                    } else {
                        panic!("Expected RealConst for 2.5");
                    }
                } else {
                    panic!("RHS term not found");
                }
            } else {
                panic!("Expected Sub for unary minus");
            }
        } else {
            panic!("Term not found");
        }
    }

    #[test]
    fn test_parse_array_sort() {
        let mut manager = TermManager::new();

        // Test simple array sort: (Array Int Int)
        let script = r#"
            (declare-const arr (Array Int Int))
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse simple array sort script");
        assert_eq!(commands.len(), 2);

        // Test nested array sort: (Array Int (Array Int Bool))
        let script = r#"
            (declare-const nested (Array Int (Array Int Bool)))
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse nested array sort script");
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn test_parse_string_literal() {
        let mut manager = TermManager::new();

        // Test simple string literal
        let term =
            parse_term("\"hello\"", &mut manager).expect("should parse string literal hello");
        let string_sort = manager.sorts.string_sort();
        if let Some(t) = manager.get(term) {
            assert_eq!(t.sort, string_sort);
            if let crate::ast::TermKind::StringLit(s) = &t.kind {
                assert_eq!(s, "hello");
            } else {
                panic!("Expected StringLit");
            }
        } else {
            panic!("Term not found");
        }

        // Test string literal in expression
        let script = r#"
            (declare-const s String)
            (assert (= s "world"))
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse string literal script");
        assert_eq!(commands.len(), 3);

        // Test empty string literal
        let term = parse_term("\"\"", &mut manager).expect("should parse empty string literal");
        if let Some(t) = manager.get(term) {
            if let crate::ast::TermKind::StringLit(s) = &t.kind {
                assert_eq!(s, "");
            } else {
                panic!("Expected StringLit");
            }
        } else {
            panic!("Term not found");
        }
    }

    #[test]
    fn test_parse_array_operations() {
        let mut manager = TermManager::new();

        // Test array select and store operations
        let script = r#"
            (declare-const arr (Array Int Int))
            (assert (= (select arr 0) 42))
            (assert (= (select (store arr 1 100) 1) 100))
            (check-sat)
        "#;

        let commands =
            parse_script(script, &mut manager).expect("should parse array operations script");
        assert_eq!(commands.len(), 4); // declare, 2 asserts, check-sat
    }
}
