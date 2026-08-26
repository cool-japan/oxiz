//! SMT-LIB2 command parsing

use super::super::lexer::TokenKind;
use super::{AttributeValue, Command, Parser};
use crate::ast::{RoundingMode, TermId};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
#[cfg(feature = "profiling")]
use crate::profiling::{ProfilingCategory, ScopedTimer};
use crate::sort::DataTypeConstructor;

/// String-typed constructors for [`Command::DeclareDatatype`] paired with
/// the fully sort-resolved constructor definitions registered with the
/// sort manager, as produced by [`Parser::parse_datatype_constructor_group`].
type DatatypeConstructorGroup = (
    Vec<(String, Vec<(String, String)>)>,
    Vec<DataTypeConstructor>,
);

/// Outcome of reading one top-level `( ... )` form.
///
/// The `Skipped` case exists so [`Parser::parse_command`] can retry with a
/// loop; see its doc comment for why a tail call was not good enough.
enum CommandStep {
    /// End of input reached; there was no further form to read.
    Eof,
    /// A recognized command was parsed.
    Parsed(Command),
    /// An unrecognized command was balance-skipped and produced no command.
    Skipped,
}

/// The rounding-mode argument of an `fp.*` operator, as written.
///
/// SMT-LIB lets this position hold either a literal mode symbol or any
/// `RoundingMode`-sorted term. OxiZ supports the two spellings a real
/// benchmark uses — a literal, or a declared/bound rounding-mode symbol — and
/// the distinction survives to the builder because the two are compiled very
/// differently: a literal becomes the corresponding `TermKind::Fp*` node
/// directly, while a symbolic mode has to be expanded into a five-way `ite`
/// over the modes it could be.
#[derive(Debug, Clone, Copy)]
pub(super) enum RmOperand {
    /// A literal `RNE`/`roundTowardZero`/... symbol.
    Concrete(RoundingMode),
    /// A `RoundingMode`-sorted term (a declared constant or a bound
    /// variable), to be resolved by case split at build time.
    Symbolic(TermId),
}

/// Whether `symbol` is one of the ten literal rounding-mode spellings.
///
/// The single source of truth for that set: [`Parser::parse_rounding_mode`]
/// maps them to modes, [`Parser::parse_rounding_mode_operand`] uses this to
/// give a literal priority over any symbol lookup, and
/// `Parser::next_token_is_rounding_mode` uses it to tell the RM-first
/// `to_fp` form from the bit-pattern one.
pub(super) fn is_rounding_mode_literal(symbol: &str) -> bool {
    matches!(
        symbol,
        "RNE"
            | "roundNearestTiesToEven"
            | "RNA"
            | "roundNearestTiesToAway"
            | "RTP"
            | "roundTowardPositive"
            | "RTN"
            | "roundTowardNegative"
            | "RTZ"
            | "roundTowardZero"
    )
}

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

    /// Resolve a sort reference, preferring a fully-resolved `define-sort`
    /// alias registered in the sort manager (see the `"define-sort"` command
    /// handler below) over the parser's own textual, symbol-only alias table
    /// consulted deep inside [`Parser::parse_sort_name`].
    ///
    /// This lets `(define-sort IA () (Array Int Int))` followed by
    /// `(declare-fun f () IA)` resolve `IA` to a genuine `Array` sort
    /// instead of falling through to a fresh, semantically-unrelated
    /// `Uninterpreted` sort with no diagnostic. Nested references (e.g.
    /// `(Array IA Int)`) still go through `parse_sort`'s ordinary recursive
    /// descent and therefore don't see this table -- a narrower, documented
    /// gap that would require changes to `parse_sort_name` itself.
    pub(super) fn resolve_sort(&mut self) -> Result<crate::sort::SortId> {
        if let Some(token) = self.lexer.peek()
            && let TokenKind::Symbol(name) = &token.kind
            && let Some(sort_id) = self.manager.sorts.resolve_alias(name)
        {
            self.lexer.next_token();
            return Ok(sort_id);
        }
        self.parse_sort()
    }

    /// Spell out, for a diagnostic, the alias cycle a rejected `define-sort`
    /// would have closed.
    ///
    /// `body_symbol` is the raw source symbol of a bare-symbol body, before
    /// [`Parser::parse_sort`] flattened it through the existing alias table;
    /// following that table from it recovers the intermediate names, so
    /// `(define-sort A () B)` followed by `(define-sort B () A)` reports
    /// `B -> A -> B` rather than the flattened `B -> B`. A compound body has
    /// no such chain and is reported by its printed form, e.g.
    /// `A -> (Array A Int)`.
    fn sort_alias_cycle_path(
        &self,
        name: &str,
        body_symbol: Option<&str>,
        body_text: &str,
    ) -> String {
        let Some(body_symbol) = body_symbol else {
            return format!("{name} -> {body_text}");
        };
        let mut path = vec![name.to_string(), body_symbol.to_string()];
        let mut current = body_symbol.to_string();
        // The existing table is acyclic (this very check keeps it so), hence
        // this walk terminates; the bound only makes that independent of the
        // invariant holding.
        for _ in 0..self.sort_aliases.len() {
            if current == name {
                break;
            }
            match self.sort_aliases.get(&current) {
                Some((params, base)) if params.is_empty() => {
                    current = base.clone();
                    path.push(current.clone());
                }
                _ => break,
            }
        }
        path.join(" -> ")
    }

    /// Parse a single SMT-LIB attribute value (as used by `set-info` and
    /// similar commands), accepting any faithful token shape -- symbol,
    /// numeral, decimal, string, hex/binary literal, or a parenthesized
    /// s-expression -- and returning it as a verbatim string.
    ///
    /// Attribute values are solver metadata (e.g. `:smt-lib-version 2.6`,
    /// `:pattern (...)`) that never participate in solving semantics, so a
    /// textual round-trip is faithful. The s-expression case uses an
    /// explicit depth counter rather than recursion so a maliciously deep
    /// attribute value cannot exhaust the stack.
    pub(super) fn parse_info_attribute_value(&mut self) -> Result<String> {
        let first = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected attribute value, found end of input".to_string(),
            })?;

        if !matches!(first.kind, TokenKind::LParen) {
            let start = first.start;
            return Self::attribute_token_text(first.kind).ok_or_else(|| OxizError::ParseError {
                position: start,
                message: "unsupported attribute value token".to_string(),
            });
        }

        let mut parts = vec!["(".to_string()];
        let mut depth = 1usize;
        while depth > 0 {
            let token = self
                .lexer
                .next_token()
                .ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unterminated s-expression in attribute value".to_string(),
                })?;
            match token.kind {
                TokenKind::LParen => {
                    depth += 1;
                    parts.push("(".to_string());
                }
                TokenKind::RParen => {
                    depth -= 1;
                    parts.push(")".to_string());
                }
                TokenKind::Eof => {
                    return Err(OxizError::ParseError {
                        position: token.start,
                        message: "unterminated s-expression in attribute value".to_string(),
                    });
                }
                other => {
                    let start = token.start;
                    let text =
                        Self::attribute_token_text(other).ok_or_else(|| OxizError::ParseError {
                            position: start,
                            message: "unsupported attribute value token".to_string(),
                        })?;
                    parts.push(text);
                }
            }
        }

        let mut out = String::new();
        for p in &parts {
            if p == ")" {
                out.push(')');
            } else if out.is_empty() || out.ends_with('(') {
                out.push_str(p);
            } else {
                out.push(' ');
                out.push_str(p);
            }
        }
        Ok(out)
    }

    /// Render a single (non-paren) attribute-value token as verbatim text,
    /// or `None` if the token kind can never carry a scalar attribute value
    /// (`LParen`/`RParen`/`Eof`, all handled by the caller separately).
    fn attribute_token_text(kind: TokenKind) -> Option<String> {
        match kind {
            TokenKind::Symbol(s) => Some(s),
            TokenKind::Numeral(n) => Some(n),
            TokenKind::Decimal(d) => Some(d),
            TokenKind::StringLit(s) => Some(s),
            TokenKind::Hexadecimal(h) => Some(h),
            TokenKind::Binary(b) => Some(b),
            TokenKind::Keyword(k) => Some(format!(":{k}")),
            TokenKind::LParen | TokenKind::RParen | TokenKind::Eof => None,
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

    /// Parse the rounding-mode argument of an `fp.*` operator, accepting a
    /// symbolic mode as well as a literal one.
    ///
    /// A literal spelling always wins: the ten mode names are SMT-LIB-reserved
    /// so nothing may shadow them. Otherwise a symbol that resolves — through
    /// the `let`/quantifier bindings first, then the declared constants — to a
    /// `RoundingMode`-sorted term is taken symbolically. Anything else falls
    /// through to [`Parser::parse_rounding_mode`], which produces exactly the
    /// error it always did (so a typo'd mode name still reads as
    /// "unknown rounding mode", not as an unknown constant).
    pub(super) fn parse_rounding_mode_operand(&mut self) -> Result<RmOperand> {
        let symbol = match self.lexer.peek().map(|t| t.kind) {
            Some(TokenKind::Symbol(s)) => s,
            _ => return self.parse_rounding_mode().map(RmOperand::Concrete),
        };
        if is_rounding_mode_literal(&symbol) {
            return self.parse_rounding_mode().map(RmOperand::Concrete);
        }
        match self.rounding_mode_symbol_term(&symbol) {
            Some(term) => {
                self.lexer.next_token();
                Ok(RmOperand::Symbolic(term))
            }
            None => self.parse_rounding_mode().map(RmOperand::Concrete),
        }
    }

    /// The `RoundingMode`-sorted term a bare symbol denotes, if any.
    ///
    /// `None` means the symbol is not a rounding mode — either undeclared, or
    /// declared at some other sort — and the caller reports it as such.
    pub(super) fn rounding_mode_symbol_term(&mut self, symbol: &str) -> Option<TermId> {
        let rm_sort = self.manager.sorts.rounding_mode_sort;
        if let Some(&term) = self.bindings.get(symbol) {
            return self
                .manager
                .get(term)
                .is_some_and(|t| t.sort == rm_sort)
                .then_some(term);
        }
        if self.constants.get(symbol) == Some(&rm_sort) {
            return Some(self.manager.mk_var(symbol, rm_sort));
        }
        None
    }

    /// Return the `:named` annotation attached to `term`, if any.
    ///
    /// Looks the term up in `self.annotations` (populated by the `(! ... )`
    /// annotation form) and extracts the string value of a `:named` attribute.
    /// Used by the `assert` command to promote `(assert (! phi :named foo))`
    /// into a name-carrying [`Command::AssertNamed`].
    fn named_annotation(&self, term: TermId) -> Option<String> {
        self.annotations.get(&term).and_then(|attrs| {
            attrs.iter().find(|a| a.key == "named").and_then(|a| {
                match &a.value {
                    Some(AttributeValue::Symbol(s)) => Some(s.clone()),
                    // A `:named` with any other value shape (or none) is
                    // malformed; treat it as unnamed rather than guessing.
                    _ => None,
                }
            })
        })
    }

    /// Parse a single SMT-LIB2 top-level command.
    /// Returns `None` on EOF.
    ///
    /// Unrecognized commands are balance-skipped and the next form is read.
    /// That retry is a *loop*, not a tail call: skipping used to `return
    /// self.parse_command()`, and because a live `cmd_name: String` has to be
    /// dropped after the call returns, LLVM cannot turn it into a jump. A
    /// script of N unknown commands — a documented, tested, fully valid input
    /// ("lenient interoperability") — therefore consumed N native stack
    /// frames and aborted the process well before N reached a million.
    pub fn parse_command(&mut self) -> Result<Option<Command>> {
        loop {
            match self.parse_command_step()? {
                CommandStep::Eof => return Ok(None),
                CommandStep::Parsed(cmd) => return Ok(Some(cmd)),
                CommandStep::Skipped => {}
            }
        }
    }

    /// Read exactly one top-level `( ... )` form.
    ///
    /// Callers must go through [`Parser::parse_command`], which turns a
    /// [`CommandStep::Skipped`] outcome into another iteration.
    fn parse_command_step(&mut self) -> Result<CommandStep> {
        #[cfg(feature = "profiling")]
        let _timer = ScopedTimer::new(ProfilingCategory::Parser);
        let token = match self.lexer.next_token() {
            Some(t) if matches!(t.kind, TokenKind::Eof) => return Ok(CommandStep::Eof),
            Some(t) => t,
            None => return Ok(CommandStep::Eof),
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
                // Accept any faithful value token (symbol/bool, numeral,
                // decimal, string, hex or binary literal) instead of only
                // symbols; error out rather than silently dropping the
                // value as an empty string when it doesn't fit that shape.
                let value_token = self
                    .lexer
                    .next_token()
                    .ok_or_else(|| OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "set-option ':{opt}': expected a value, found end of input"
                        ),
                    })?;
                let val = match value_token.kind {
                    TokenKind::Symbol(s) => s,
                    TokenKind::Numeral(n) => n,
                    TokenKind::Decimal(d) => d,
                    TokenKind::StringLit(s) => s,
                    TokenKind::Hexadecimal(h) => h,
                    TokenKind::Binary(b) => b,
                    other => {
                        return Err(OxizError::ParseError {
                            position: value_token.start,
                            message: format!(
                                "set-option ':{opt}': unsupported value token {other:?}"
                            ),
                        });
                    }
                };
                self.expect_rparen()?;
                Command::SetOption(opt, val)
            }
            "declare-const" => {
                let name = self.expect_symbol()?;
                let sort_id = self.resolve_sort()?;
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
                    let sort_id = self.resolve_sort()?;
                    arg_sort_ids.push(sort_id);
                    arg_sorts.push(self.sort_id_to_string(sort_id));
                }
                let ret_sort_id = self.resolve_sort()?;
                let ret_sort = self.sort_id_to_string(ret_sort_id);
                self.expect_rparen()?;

                if arg_sorts.is_empty() {
                    self.constants.insert(name.clone(), ret_sort_id);
                } else {
                    // A genuine function: `RoundingMode` may appear neither in
                    // an argument nor in the result, because the closure axiom
                    // that gives the sort its five elements attaches only to a
                    // nullary declaration.  Checked here rather than in
                    // `parse_sort` so that the nullary spelling
                    // `(declare-fun m () RoundingMode)` still works.
                    for &arg_sort_id in &arg_sort_ids {
                        self.reject_unsupported_rounding_mode_position(
                            arg_sort_id,
                            "a function argument sort",
                        )?;
                    }
                    self.reject_unsupported_rounding_mode_position(
                        ret_sort_id,
                        "the result sort of a function with arguments",
                    )?;
                    self.functions
                        .insert(name.clone(), (arg_sort_ids.clone(), ret_sort_id));
                }
                Command::DeclareFun(name, arg_sorts, ret_sort)
            }
            "assert" => {
                let term = self.parse_term()?;
                self.expect_rparen()?;
                // Thread a top-level `:named` annotation into the command so
                // the solver can register the assertion by name (required for
                // `(get-unsat-core)`).  When the asserted expression is
                // `(! phi :named foo)`, `parse_term` returns `phi` and records
                // the attributes against it in `self.annotations`, so the
                // returned term id is exactly the key to look up.  A `:named`
                // buried on a *sub*-expression (e.g. `(=> (! a :named x) b)`)
                // is correctly ignored here: its annotation key is the inner
                // term, not the returned top-level term.
                match self.named_annotation(term) {
                    Some(name) => Command::AssertNamed(term, name),
                    None => Command::Assert(term),
                }
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
            "get-consequences" => {
                // (get-consequences (assumptions...) (variables...))
                // Two parenthesized term-lists, parsed exactly like the inner
                // list of `check-sat-assuming`.  Undeclared symbols are already
                // rejected by `parse_term` in script mode, so no extra symbol
                // validation is needed here.
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
                self.expect_lparen()?;
                let mut variables = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    variables.push(self.parse_term()?);
                }
                self.expect_rparen()?;
                Command::GetConsequences(assumptions, variables)
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
                // Accept any faithful attribute value shape (string, symbol,
                // numeral, decimal, hex/binary literal, or a parenthesized
                // s-expression) instead of only string/symbol. The standard
                // `(set-info :smt-lib-version 2.6)` header lexes its value as
                // a `Decimal` token, and rejecting it here used to abort
                // parsing of the *entire* script since `parse_script` parses
                // commands eagerly.
                let value = self.parse_info_attribute_value()?;
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

                if !params.is_empty() {
                    // Parametric aliases (e.g. `(define-sort Pair (X Y) ...)`)
                    // require substituting X/Y at each instantiation site.
                    // Neither this parser's flat name -> name alias table
                    // (`sort_aliases`, consulted by `parse_sort_name`) nor the
                    // sort manager's `define_alias` registry (one fixed
                    // `SortId` per name) can express that. Registering the
                    // definition anyway would let a later reference to the
                    // alias silently fall through to a fresh, unrelated
                    // `Uninterpreted` sort with no diagnostic -- exactly the
                    // miscompilation this rejects. Consume the body so the
                    // token stream stays in sync, then fail honestly instead
                    // of fabricating a wrong sort.
                    let position = self.lexer.position();
                    self.parse_sort()?;
                    self.expect_rparen()?;
                    return Err(OxizError::ParseError {
                        position,
                        message: format!(
                            "define-sort '{name}': parametric sort aliases ({} parameter(s)) are not supported",
                            params.len()
                        ),
                    });
                }

                // A bare-symbol body (e.g. `Int`, `MyOtherAlias`) round-trips
                // correctly through the parser's existing name -> name alias
                // table, so keep registering it there too: that lets nested
                // references (e.g. `(Array MyInt Int)`, resolved by
                // `parse_sort`'s ordinary recursive descent rather than
                // `resolve_sort`) still see it.
                let body_symbol = match self.lexer.peek().map(|t| t.kind) {
                    Some(TokenKind::Symbol(s)) => Some(s),
                    _ => None,
                };
                let is_bare_symbol = body_symbol.is_some();

                // Parse the body with the full sort grammar (not just a bare
                // symbol) so compound bodies like `(Array Int Int)` or
                // `(_ BitVec 32)` parse instead of erroring, and resolve to a
                // concrete `SortId` up front.
                let body_position = self.lexer.position();
                let sort_id = self.parse_sort()?;
                self.expect_rparen()?;
                let sort_expr = self.sort_id_to_string(sort_id);

                // Soundness: a sort abbreviation defined in terms of itself has
                // no fixed point, so there is nothing to register. The body
                // parses fine -- a not-yet-known symbol becomes a fresh
                // `Uninterpreted` sort carrying that very name -- and only a
                // *later* reference reveals the problem, by which point
                // `parse_sort_name` is chasing `A -> A` forever. Reject it
                // here, where the cycle is still nameable; z3 likewise rejects
                // a `define-sort` body that names an unknown sort at the
                // definition rather than at the use.
                //
                // Both `sort_aliases` and the sort manager's `define_alias`
                // registry are written below, so this check must precede both
                // for the two tables to stay consistent.
                if self.sort_mentions_name(sort_id, &name) {
                    return Err(OxizError::ParseError {
                        position: body_position,
                        message: format!(
                            "define-sort '{name}': cyclic sort abbreviation ({}); a sort \
                             abbreviation may not be defined in terms of itself, directly or \
                             through other abbreviations",
                            self.sort_alias_cycle_path(&name, body_symbol.as_deref(), &sort_expr)
                        ),
                    });
                }

                if is_bare_symbol {
                    self.sort_aliases
                        .insert(name.clone(), (params.clone(), sort_expr.clone()));
                }
                // Always register the fully-resolved sort in the sort
                // manager's general alias table so top-level references to
                // the alias (via `resolve_sort`, used by `declare-const`,
                // `declare-fun`, and `define-fun`) recover the exact sort
                // even for a compound body that the name -> name table
                // cannot express.
                self.manager.sorts.define_alias(&name, sort_id);

                Command::DefineSort(name, params, sort_expr)
            }
            "define-fun" => {
                // (define-fun name ((param sort) ...) ret-sort body)
                let name = self.expect_symbol()?;
                self.expect_lparen()?;

                let mut params: Vec<(String, String)> = Vec::new();
                // Parallel to `params`, keeps the already-resolved `SortId`
                // for each parameter so placeholder-var creation below can
                // reuse it directly instead of re-resolving the stringified
                // sort. Re-resolving `sort_id_to_string(param_sort_id)`
                // through `parse_sort_name` cannot round-trip a compound
                // sort like `(Array Int Int)` (that function only
                // understands flat sort *names*), which would silently give
                // the placeholder variable an unrelated `Uninterpreted`
                // sort and break sort-checking while parsing the body.
                let mut param_sort_ids: Vec<crate::sort::SortId> = Vec::new();
                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    self.expect_lparen()?;
                    let param_name = self.expect_symbol()?;
                    let param_sort_id = self.resolve_sort()?;
                    let param_sort = self.sort_id_to_string(param_sort_id);
                    self.expect_rparen()?;
                    params.push((param_name, param_sort));
                    param_sort_ids.push(param_sort_id);
                }

                let ret_sort_id = self.resolve_sort()?;
                let ret_sort = self.sort_id_to_string(ret_sort_id);

                // Save any shadowed bindings
                let old_bindings: Vec<(String, TermId)> = params
                    .iter()
                    .filter_map(|(pname, _)| self.bindings.get(pname).map(|&t| (pname.clone(), t)))
                    .collect();

                // Create placeholder vars for parameters, reusing the
                // already-resolved sort rather than re-parsing it from text.
                // The minted `TermId`s are kept in declaration order below —
                // expansion at the call site substitutes by walking this
                // exact list, never by recreating a `Var` from the bare name
                // (see `FunctionMacro::formal_vars`'s doc comment).
                let mut formal_vars: Vec<TermId> = Vec::with_capacity(params.len());
                for ((pname, _psort), &sort_id) in params.iter().zip(param_sort_ids.iter()) {
                    let param_term = self.manager.mk_var(pname, sort_id);
                    self.bindings.insert(pname.clone(), param_term);
                    formal_vars.push(param_term);
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
                self.function_defs.insert(
                    name.clone(),
                    super::FunctionMacro {
                        formal_vars,
                        formal_decls: params.clone(),
                        expansion: body,
                    },
                );

                // For nullary define-fun, inline it directly as a binding.
                // Because that inlining is a *substitution* at every later
                // occurrence, the depth of `body` is charged against the
                // parser's nesting budget here — a chain of such definitions
                // otherwise builds an arbitrarily deep term while every
                // individual command stays shallow. See
                // `Parser::charge_binding_depth`.
                if params.is_empty() {
                    self.charge_binding_depth(&name, body)?;
                    self.bindings.insert(name.clone(), body);
                }

                Command::DefineFun(name, params, ret_sort, body)
            }
            "declare-datatypes" => self.parse_declare_datatypes()?,
            "declare-datatype" => self.parse_declare_datatype()?,
            "declare-sort" => {
                // (declare-sort <symbol> <numeral>)
                let name = self.expect_symbol()?;
                let arity = self.parse_optional_numeral(0)?;
                self.expect_rparen()?;
                // Record the parametric-sort declaration (arity 0 is the
                // common "uninterpreted sort" case; arity > 0 is recorded
                // for well-formedness but application of such sorts beyond
                // `Array`/`BitVec`/`FloatingPoint` is not yet supported and
                // will surface its own honest parse error at use-site).
                self.manager
                    .sorts
                    .declare_parametric_sort(&name, arity as usize);
                Command::DeclareSort(name, arity)
            }
            // Recursive definitions used to be hard-rejected here, because the
            // alternative at the time — balance-skipping the command — left the
            // defined symbol as an unconstrained uninterpreted function and
            // silently changed the sat/unsat answer. They are now parsed for
            // real (see `parser::recfun`) and the definitional axiom is
            // discharged by the solver. `Command::DefineFunsRec`'s doc comment
            // carries the obligation the old rejection used to enforce: a
            // consumer that cannot solve recursive definitions must still fail
            // honestly rather than skip the command.
            "define-fun-rec" => Command::DefineFunsRec(vec![self.parse_define_fun_rec()?]),
            "define-funs-rec" => Command::DefineFunsRec(self.parse_define_funs_rec()?),
            "get-unsat-assumptions" => {
                // The assumptions passed to the most recent
                // `check-sat-assuming` are tracked by the solver context, which
                // reports an unsatisfiable subset after an `unsat` verdict.
                self.expect_rparen()?;
                Command::GetUnsatAssumptions
            }
            _ => {
                // Genuinely unrecognized (e.g. vendor/tooling-specific)
                // commands are skipped for lenient interoperability, same
                // as before. Commands with real solving-semantics impact
                // are special-cased above and rejected honestly instead.
                let mut depth = 1;
                while depth > 0 {
                    match self.lexer.next_token().map(|t| t.kind) {
                        Some(TokenKind::LParen) => depth += 1,
                        Some(TokenKind::RParen) => depth -= 1,
                        Some(TokenKind::Eof) | None => break,
                        _ => {}
                    }
                }
                return Ok(CommandStep::Skipped);
            }
        };

        Ok(CommandStep::Parsed(cmd))
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

        // Register every declared name's sort *before* any constructor group is
        // parsed, so that a field mentioning one of them — the recursive
        // `(tail Lst)`, or a mutually recursive `(left Tree)` — resolves to the
        // datatype's own sort instead of a fresh uninterpreted one.
        for dt_name in &datatype_names {
            let dt_sort = self.manager.sorts.mk_datatype_sort(dt_name);
            self.dt_sorts.insert(dt_name.clone(), dt_sort);
        }

        // Parse the constructor-group list: exactly one group per declared
        // datatype name, in the same order (this correctly handles both
        // multiple independent datatypes and mutually recursive ones,
        // since all `datatype_names` are known before any group's
        // selectors are parsed).
        self.expect_lparen()?;

        let mut constructors: Vec<(String, Vec<(String, String)>)> = Vec::new();

        for dt_name in &datatype_names {
            self.expect_lparen()?;

            let (dt_constructors, ctor_defs) = self.parse_datatype_constructor_group()?;

            let dt_sort = self.manager.sorts.mk_datatype_sort(dt_name);
            for (ctor_name, _selectors) in &dt_constructors {
                self.dt_constructors.insert(ctor_name.clone(), dt_sort);
            }
            self.manager.sorts.declare_datatype(dt_name, ctor_defs);

            constructors.extend(dt_constructors);
        }

        // Close the constructor-group list and the whole command.
        self.expect_rparen()?;
        self.expect_rparen()?;

        // `Command::DeclareDatatype` only carries a single (name,
        // constructors) pair. Join all declared names so a multi/mutual
        // datatype script's summary doesn't silently drop which datatypes
        // were declared; the authoritative per-datatype sort definitions
        // and constructor->sort bindings above are already correctly
        // registered on the parser/sort-manager state regardless of how
        // this summary command is shaped.
        let name = datatype_names.join(",");

        Ok(Command::DeclareDatatype { name, constructors })
    }

    /// Parse one `(ctor (selector sort) ...) ...` constructor group for a
    /// single datatype, i.e. the body between an already-consumed opening
    /// '(' and its matching closing ')'. Returns both the string-typed
    /// form (used by [`Command::DeclareDatatype`]) and the fully-typed
    /// [`crate::sort::DataTypeConstructor`] form (used to register the
    /// datatype's real definition, including selector sorts, with the
    /// sort manager).
    fn parse_datatype_constructor_group(&mut self) -> Result<DatatypeConstructorGroup> {
        let mut constructors = Vec::new();
        let mut ctor_defs = Vec::new();

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
            let mut selector_defs = Vec::new();
            loop {
                if let Some(t) = self.lexer.peek()
                    && matches!(t.kind, TokenKind::RParen)
                {
                    self.lexer.next_token();
                    break;
                }

                self.expect_lparen()?;
                let selector_name = self.expect_symbol()?;
                // Parse a full sort expression (not just a bare symbol) so
                // parametric selector sorts like `(Array Int Int)` or
                // `(_ BitVec 8)` parse correctly instead of erroring.
                let selector_sort_id = self.resolve_sort()?;
                // A datatype field is reached through a selector application,
                // which is not a nullary declaration, so no closure axiom can
                // pin its rounding mode to the five real ones.
                self.reject_unsupported_rounding_mode_position(
                    selector_sort_id,
                    "a datatype field sort",
                )?;
                let selector_sort = self.sort_id_to_string(selector_sort_id);
                self.expect_rparen()?;
                selector_defs.push((self.manager.intern_str(&selector_name), selector_sort_id));
                // Record the selector as a declared symbol so that a later
                // `(head l)` resolves to a `DtSelector` of the declared result
                // sort instead of an uninterpreted `Bool`-sorted application.
                self.dt_selectors
                    .insert(selector_name.clone(), selector_sort_id);
                selectors.push((selector_name, selector_sort));
            }

            ctor_defs.push(DataTypeConstructor {
                name: self.manager.intern_str(&ctor_name),
                selectors: selector_defs.into_iter().collect(),
            });
            constructors.push((ctor_name, selectors));
        }

        Ok((constructors, ctor_defs))
    }

    /// Parse `(declare-datatype name (...))` — single-datatype form
    fn parse_declare_datatype(&mut self) -> Result<Command> {
        let name = self.expect_symbol()?;
        self.expect_lparen()?;

        // Register the sort before the constructor group is parsed so a
        // recursive field `(tail Lst)` resolves to this datatype (see the
        // multi-datatype form for the same reasoning).
        let dt_sort = self.manager.sorts.mk_datatype_sort(&name);
        self.dt_sorts.insert(name.clone(), dt_sort);

        let (constructors, ctor_defs) = self.parse_datatype_constructor_group()?;

        self.expect_rparen()?;
        for (ctor_name, _selectors) in &constructors {
            self.dt_constructors.insert(ctor_name.clone(), dt_sort);
        }
        self.manager.sorts.declare_datatype(&name, ctor_defs);

        Ok(Command::DeclareDatatype { name, constructors })
    }
}
