//! SMT-LIB2 Parser

use super::lexer::{Lexer, TokenKind};
use crate::ast::{TermId, TermManager};
use crate::error::{OxizError, Result};
use crate::sort::SortId;
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
                    // Unary minus
                    let zero = self.manager.mk_int(0);
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
            "concat" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_concat(lhs, rhs)
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
            let sort_name = self.expect_symbol()?;
            let sort = self.parse_sort_name(&sort_name)?;
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
                let sort = self.expect_symbol()?;
                self.expect_rparen()?;
                let sort_id = self.parse_sort_name(&sort)?;
                self.constants.insert(name.clone(), sort_id);
                Command::DeclareConst(name, sort)
            }
            "declare-fun" => {
                let name = self.expect_symbol()?;
                self.expect_lparen()?;
                let mut arg_sorts = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    arg_sorts.push(self.expect_symbol()?);
                }
                let ret_sort = self.expect_symbol()?;
                self.expect_rparen()?;

                if arg_sorts.is_empty() {
                    let sort_id = self.parse_sort_name(&ret_sort)?;
                    self.constants.insert(name.clone(), sort_id);
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
                        if let TokenKind::Numeral(n) = self
                            .lexer
                            .next_token()
                            .expect("token exists after peek check")
                            .kind
                        {
                            n.parse().unwrap_or(1)
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
                        if let TokenKind::Numeral(n) = self
                            .lexer
                            .next_token()
                            .expect("token exists after peek check")
                            .kind
                        {
                            n.parse().unwrap_or(1)
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

        let t = parse_term("true", &mut manager).unwrap();
        assert_eq!(t, manager.mk_true());

        let f = parse_term("false", &mut manager).unwrap();
        assert_eq!(f, manager.mk_false());

        let n = parse_term("42", &mut manager).unwrap();
        let expected = manager.mk_int(42);
        assert_eq!(n, expected);
    }

    #[test]
    fn test_parse_boolean_ops() {
        let mut manager = TermManager::new();

        let not_true = parse_term("(not true)", &mut manager).unwrap();
        assert_eq!(not_true, manager.mk_false());

        let and_expr = parse_term("(and true false)", &mut manager).unwrap();
        assert_eq!(and_expr, manager.mk_false());

        let or_expr = parse_term("(or true false)", &mut manager).unwrap();
        assert_eq!(or_expr, manager.mk_true());
    }

    #[test]
    fn test_parse_arithmetic() {
        let mut manager = TermManager::new();

        let _add = parse_term("(+ 1 2 3)", &mut manager).unwrap();
        let _lt = parse_term("(< x y)", &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
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

        let commands = parse_script(script, &mut manager).unwrap();
        assert_eq!(commands.len(), 1);

        assert!(matches!(&commands[0], Command::Simplify(_)));
    }

    #[test]
    fn test_parse_annotations() {
        let mut manager = TermManager::new();

        // Test :named annotation
        let mut parser = Parser::new("(! (> x 0) :named myAssertion)", &mut manager);
        let term = parser.parse_term().unwrap();

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
        let _term = parser.parse_term().unwrap();

        // The annotation should be present on the body of the forall
        // We just verify that it parses without error for now
    }

    #[test]
    fn test_parse_multiple_annotations() {
        let mut manager = TermManager::new();

        // Test multiple annotations
        let mut parser = Parser::new("(! (> x 0) :named test :weight 10)", &mut manager);
        let term = parser.parse_term().unwrap();

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
}
