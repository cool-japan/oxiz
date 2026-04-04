//! SMT-LIB2 command parsing

use super::super::lexer::TokenKind;
use super::{Command, Parser};
use crate::ast::{RoundingMode, TermId};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;

impl<'a> Parser<'a> {
    /// Expect an opening parenthesis '('
    pub(super) fn expect_lparen(&mut self) -> Result<()> {
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

    /// Expect a symbol token and return its string value
    pub(super) fn expect_symbol(&mut self) -> Result<String> {
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

    /// Expect a keyword token (e.g., :named) and return its string value (without leading colon)
    pub(super) fn expect_keyword(&mut self) -> Result<String> {
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

    /// Expect a string literal token and return its content
    pub(super) fn expect_string(&mut self) -> Result<String> {
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

    /// Parse an IEEE 754 rounding mode symbol (RNE, RNA, RTP, RTN, RTZ or long forms)
    pub(super) fn parse_rounding_mode(&mut self) -> Result<RoundingMode> {
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

    /// Parse a single SMT-LIB2 top-level command.
    /// Returns `None` on EOF.
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
                // option value is optional / may be missing
                let val = self.expect_symbol().unwrap_or_default();
                self.expect_rparen()?;
                Command::SetOption(opt, val)
            }
            "declare-const" => {
                let name = self.expect_symbol()?;
                let sort_id = self.parse_sort()?;
                self.expect_rparen()?;
                self.constants.insert(name.clone(), sort_id);
                let sort_str = self.sort_id_to_string(sort_id);
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
                } else {
                    self.functions
                        .insert(name.clone(), (arg_sort_ids.clone(), ret_sort_id));
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
                let n = self.parse_optional_numeral(1)?;
                self.expect_rparen()?;
                Command::Push(n)
            }
            "pop" => {
                let n = self.parse_optional_numeral(1)?;
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
            "get-unsat-core" => {
                self.expect_rparen()?;
                Command::GetUnsatCore
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
                // Peek to decide whether the value is a string literal or a symbol
                // without consuming the token on a failed match.
                let value = if let Some(tok) = self.lexer.peek()
                    && matches!(tok.kind, TokenKind::StringLit(_))
                {
                    self.expect_string()?
                } else {
                    self.expect_symbol()?
                };
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

                self.sort_aliases
                    .insert(name.clone(), (params.clone(), sort_expr.clone()));

                Command::DefineSort(name, params, sort_expr)
            }
            "define-fun" => {
                // (define-fun name ((param sort) ...) ret-sort body)
                let name = self.expect_symbol()?;
                self.expect_lparen()?;

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
                    let param_sort_id = self.parse_sort()?;
                    let param_sort = self.sort_id_to_string(param_sort_id);
                    self.expect_rparen()?;
                    params.push((param_name, param_sort));
                }

                let ret_sort_id = self.parse_sort()?;
                let ret_sort = self.sort_id_to_string(ret_sort_id);

                // Save any shadowed bindings
                let old_bindings: Vec<(String, TermId)> = params
                    .iter()
                    .filter_map(|(pname, _)| self.bindings.get(pname).map(|&t| (pname.clone(), t)))
                    .collect();

                // Create placeholder vars for parameters
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

                // Register function definition
                self.function_defs
                    .insert(name.clone(), (params.clone(), body));

                // For nullary define-fun, inline it directly as a binding
                if params.is_empty() {
                    self.bindings.insert(name.clone(), body);
                }

                Command::DefineFun(name, params, ret_sort, body)
            }
            "declare-datatypes" => self.parse_declare_datatypes()?,
            "declare-datatype" => self.parse_declare_datatype()?,
            _ => {
                // Skip unknown command (balanced paren skipping)
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

    /// Parse an optional numeral from the token stream; return `default` if none present
    fn parse_optional_numeral(&mut self, default: u32) -> Result<u32> {
        if let Some(t) = self.lexer.peek()
            && matches!(t.kind, TokenKind::Numeral(_))
            && let Some(token) = self.lexer.next_token()
            && let TokenKind::Numeral(n) = token.kind
        {
            return n.parse::<u32>().map_err(|_| OxizError::ParseError {
                position: token.start,
                message: format!("invalid numeral: {n}"),
            });
        }
        Ok(default)
    }

    /// Parse `(declare-datatypes (...) (...))` — multi-datatype form
    fn parse_declare_datatypes(&mut self) -> Result<Command> {
        // (declare-datatypes ((name1 arity1) (name2 arity2) ...)
        //                    ((constructors1 ...) (constructors2 ...)))
        self.expect_lparen()?;

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
            // Skip the arity
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

        let mut constructors = Vec::new();

        // Opening paren for the first datatype's constructors
        self.expect_lparen()?;

        loop {
            if let Some(t) = self.lexer.peek()
                && matches!(t.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }

            self.expect_lparen()?;
            let ctor_name = self.expect_symbol()?;

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

        // Close outer constructor list and outer command list
        self.expect_rparen()?;
        self.expect_rparen()?;

        let name = datatype_names
            .first()
            .cloned()
            .unwrap_or_else(|| "UnknownDatatype".to_string());

        let dt_sort = self.manager.sorts.mk_datatype_sort(&name);
        for (ctor_name, _selectors) in &constructors {
            self.dt_constructors.insert(ctor_name.clone(), dt_sort);
        }

        Ok(Command::DeclareDatatype { name, constructors })
    }

    /// Parse `(declare-datatype name (...))` — single-datatype form
    fn parse_declare_datatype(&mut self) -> Result<Command> {
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

            self.expect_lparen()?;
            let ctor_name = self.expect_symbol()?;

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

        let dt_sort = self.manager.sorts.mk_datatype_sort(&name);
        for (ctor_name, _selectors) in &constructors {
            self.dt_constructors.insert(ctor_name.clone(), dt_sort);
        }

        Ok(Command::DeclareDatatype { name, constructors })
    }
}
