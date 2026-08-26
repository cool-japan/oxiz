//! SMT-LIB2 Parser
#![allow(clippy::while_let_loop)] // Parser uses explicit loop control

use super::lexer::{Lexer, TokenKind};
use crate::ast::{TermId, TermManager};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use num_rational::Rational64;

mod build;
mod commands;
mod indexed;
mod recfun;
mod sorts;
mod terms;

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
    /// Assert a term carrying a `:named` annotation (the assertion name).
    ///
    /// Produced when the top-level asserted expression is `(! phi :named foo)`,
    /// so the solver can track the assertion by name for `(get-unsat-core)`.
    AssertNamed(TermId, String),
    /// Check sat
    CheckSat,
    /// Check sat with assumptions
    CheckSatAssuming(Vec<TermId>),
    /// Get consequences: `(get-consequences (assumptions...) (variables...))`.
    ///
    /// Returns the literals over `variables` that are entailed by the current
    /// assertions together with `assumptions`.
    GetConsequences(Vec<TermId>, Vec<TermId>),
    /// Get model
    GetModel,
    /// Get value
    GetValue(Vec<TermId>),
    /// Get unsat core
    GetUnsatCore,
    /// Get unsat assumptions (the failed subset of the assumptions passed to
    /// the most recent `check-sat-assuming` that returned `unsat`).
    GetUnsatAssumptions,
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
    /// One `(define-fun-rec ..)` (a single-element vector) or one
    /// `(define-funs-rec ..)` (one element per function in the mutually
    /// recursive group).
    ///
    /// # Consumers must never skip this command
    ///
    /// A recursive definition is the *only* thing that constrains the symbol it
    /// defines: the parser deliberately registers the name as an ordinary
    /// declared function so that call sites build a real `Apply` node, and it
    /// never expands the body at those call sites (expansion would not
    /// terminate).  A consumer whose `match` lets this command fall into a
    /// catch-all therefore keeps every `(f x)` application but drops every
    /// constraint on `f`, i.e. it silently solves a *strictly weaker* problem —
    /// turning `unsat` into `sat` with a meaningless model.  That is exactly the
    /// failure the previous hard parse rejection existed to prevent.
    ///
    /// Any consumer that cannot solve recursive definitions **must** answer with
    /// an explicit error (or an honest `unknown`), never by ignoring the
    /// command.
    DefineFunsRec(Vec<RecFunDecl>),
}

/// One function of a `define-fun-rec` / `define-funs-rec` group.
///
/// Unlike a `FunctionMacro` (this parser's internal representation for
/// non-recursive `define-fun`), this is *not* expanded at call sites — see
/// the parser's internal `rec_function_defs` table. It is handed to the solver, which discharges
/// it by fuel-bounded unfolding of the definitional equation
/// `forall x. f(x) = body`.
#[derive(Debug, Clone)]
pub struct RecFunDecl {
    /// The defined function's name.
    pub name: String,
    /// `(name, sort-string)` for each formal parameter, in declaration order.
    pub params: Vec<(String, String)>,
    /// Each formal parameter's `Var` term, in declaration order, exactly as
    /// bound while `body` was parsed.
    ///
    /// Never re-derive these from `params`: [`TermManager::mk_var`]
    /// hash-conses on the `(name, sort)` pair, so reconstructing a `Var` from
    /// the bare name with a guessed sort mints a *different* term that the body
    /// never mentions — every substitution against it would silently no-op and
    /// leave the parameter free in the unfolded body. This is the same hazard
    /// documented at length on the parser's internal `FunctionMacro::formal_vars`.
    pub formal_vars: Vec<TermId>,
    /// The declared return sort, as a sort string (resolved by the consumer the
    /// same way `declare-fun`'s return sort is).
    pub ret_sort: String,
    /// The body, possibly mentioning `formal_vars` and applications of any
    /// function in the same `define-funs-rec` group (including itself).
    pub body: TermId,
}

/// A recorded `define-fun` macro, ready for call-site expansion.
///
/// See the doc comment on [`Parser::function_defs`] for why `formal_vars`
/// must hold the parameters' actual interned `TermId`s rather than anything
/// that has to be re-derived (name, declared sort text, …) at the call site.
#[derive(Debug, Clone)]
pub(super) struct FunctionMacro {
    /// Each formal parameter's `Var` term, in declaration order, exactly as
    /// bound while `expansion` was parsed.
    pub(super) formal_vars: Vec<TermId>,
    /// `(name, sort-string)` pairs, kept for `Command::DefineFun` /
    /// introspection use — not consulted by expansion itself.
    #[allow(dead_code)]
    pub(super) formal_decls: Vec<(String, String)>,
    /// The function body, possibly mentioning `formal_vars`.
    pub(super) expansion: TermId,
}

/// Parser state
pub struct Parser<'a> {
    pub(super) lexer: Lexer<'a>,
    pub(super) manager: &'a mut TermManager,
    /// Variable bindings (for let expressions)
    pub(super) bindings: FxHashMap<String, TermId>,
    /// Declared constants
    pub(super) constants: FxHashMap<String, SortId>,
    /// Declared functions
    #[allow(dead_code)]
    pub(super) functions: FxHashMap<String, (Vec<SortId>, SortId)>,
    /// Sort aliases from define-sort
    pub(super) sort_aliases: FxHashMap<String, (Vec<String>, String)>,
    /// Function definitions from define-fun.
    ///
    /// Call-site expansion (see [`Parser::expand_defined_fun`] in `terms.rs`)
    /// substitutes each formal parameter for its actual argument by walking
    /// [`FunctionMacro::formal_vars`], the *exact* `TermId`s minted while the
    /// body was parsed. Re-deriving a parameter's `Var` term from its bare
    /// name at expansion time is unsafe: [`TermManager::mk_var`] hash-conses
    /// on the `(name, sort)` pair, so recovering the wrong sort for the name
    /// (a same-named global constant of a different sort, or no sort
    /// information at all) mints a *different* term that the body never
    /// mentions — the substitution then silently no-ops and the parameter
    /// stays free in the expanded body.
    pub(super) function_defs: FxHashMap<String, FunctionMacro>,
    /// Recursive function definitions from `define-fun-rec` /
    /// `define-funs-rec`, keyed by name.
    ///
    /// Deliberately *not* a second `function_defs`: a recursive body is never
    /// expanded at a call site, because expansion would not terminate (`f`'s
    /// body mentions `f`). The name is instead registered in
    /// [`Parser::functions`] (arity > 0) or [`Parser::constants`] (arity 0)
    /// *before* the body is parsed, so both the body's self-references and every
    /// later call site build an ordinary `Apply` / `Var` node with the declared
    /// sort. The bodies collected here travel out in
    /// [`Command::DefineFunsRec`], and the solver discharges them by
    /// fuel-bounded unfolding.
    ///
    /// The map itself is what makes a redefinition detectable and what lets a
    /// later group refer to an earlier one; it is never consulted by term
    /// construction.
    pub(super) rec_function_defs: FxHashMap<String, RecFunDecl>,
    /// Term annotations (term -> attributes)
    pub(super) annotations: FxHashMap<TermId, Vec<Attribute>>,
    /// Error recovery mode enabled
    #[allow(dead_code)]
    pub(super) recovery_mode: bool,
    /// Collected errors during parsing
    #[allow(dead_code)]
    pub(super) errors: Vec<OxizError>,
    /// Datatype constructor names -> (datatype_sort, arity/selector_info)
    /// For nullary constructors (enums), the Vec is empty
    pub(super) dt_constructors: FxHashMap<String, SortId>,
    /// Datatype selector (accessor) names -> the selector's result sort.
    ///
    /// Populated by `declare-datatype(s)`. Without it a selector application
    /// like `(head l)` has no declaration to resolve against, so it used to
    /// degrade to a `Bool`-sorted uninterpreted apply (losing its real `Int`
    /// sort) and would now be rejected outright by the strict undeclared-symbol
    /// rule.
    pub(super) dt_selectors: FxHashMap<String, SortId>,
    /// Datatype *sort* names -> the `SortKind::Datatype` sort they denote.
    ///
    /// Registered the moment a `declare-datatype(s)` names a datatype, i.e.
    /// *before* its constructor group is parsed, so that a recursive field such
    /// as `(tail Lst)` resolves to the datatype under construction rather than
    /// to a fresh uninterpreted sort. See `Parser::parse_sort_name`.
    pub(super) dt_sorts: FxHashMap<String, SortId>,
    /// Whether we are parsing a full SMT-LIB *script* (`parse_script`) rather
    /// than an ad-hoc bare term (`parse_term`).
    ///
    /// In script mode every symbol must be declared before use, so an unknown
    /// symbol is a hard error instead of being silently minted as a fresh
    /// Bool variable. The bare-term convenience path stays lenient so that
    /// free variables can still be constructed without a declaration prologue.
    /// This replaces the earlier "any declaration table is non-empty" heuristic,
    /// which wrongly stayed lenient for a script whose symbols were all
    /// undeclared.
    pub(super) script_mode: bool,
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
            rec_function_defs: FxHashMap::default(),
            annotations: FxHashMap::default(),
            recovery_mode: false,
            errors: Vec::new(),
            dt_constructors: FxHashMap::default(),
            dt_selectors: FxHashMap::default(),
            dt_sorts: FxHashMap::default(),
            script_mode: false,
        }
    }

    /// Create a new parser whose environment is pre-seeded with declarations
    /// supplied by an embedding context.
    ///
    /// The plain [`Parser::new`] path always starts with an empty
    /// constants/functions map, which means a symbol declared *outside* the
    /// current text fragment (for example a constant registered by the
    /// `oxiz-wasm` JS API or an `oxiz-solver` `Context`) is rejected in script
    /// mode or mis-sorted as a fresh `Bool` variable in bare-term mode. This
    /// constructor lets an embedder register those declarations up front so the
    /// symbols resolve with their true sorts.
    ///
    /// Because the seeded declarations describe a real environment, strict
    /// undeclared-symbol resolution is enabled: a *seeded* symbol resolves to a
    /// variable of its declared sort, while a genuinely-unknown symbol still
    /// produces an honest parse error instead of a silently mis-sorted term.
    /// Seed further symbols incrementally with [`Parser::seed_declaration`] /
    /// [`Parser::seed_function`].
    ///
    /// Note: for an *external* embedder to reach this API, the `Parser` type
    /// must be re-exported from `smtlib/mod.rs` (which currently re-exports only
    /// `parse_term` / `parse_script` / `Command`); that re-export lives outside
    /// this file. Until then the seeding API is reachable in-crate (e.g. by an
    /// `oxiz-solver` `Context` that lives in this crate) and via tests.
    #[allow(dead_code)]
    pub fn with_context<I>(input: &'a str, manager: &'a mut TermManager, declarations: I) -> Self
    where
        I: IntoIterator<Item = (String, SortId)>,
    {
        let mut parser = Self::new(input, manager);
        for (name, sort) in declarations {
            parser.constants.insert(name, sort);
        }
        // A seeded context describes real declarations, so unknown symbols must
        // be rejected rather than silently minted as fresh Bool variables.
        parser.script_mode = true;
        parser
    }

    /// Pre-register a declared constant (or nullary function) symbol so that it
    /// resolves to a variable of `sort` during parsing.
    ///
    /// This is the builder-style counterpart to [`Parser::with_context`]: an
    /// embedder can chain `seed_declaration` calls on a parser built with
    /// [`Parser::new`] or [`Parser::with_context`] before invoking
    /// [`Parser::parse_term`] / [`Parser::parse_script`]. Seeded symbols resolve
    /// with their true sort; unseeded symbols keep whatever undeclared-symbol
    /// behavior the parser's mode dictates.
    #[allow(dead_code)]
    pub fn seed_declaration(&mut self, name: impl Into<String>, sort: SortId) -> &mut Self {
        self.constants.insert(name.into(), sort);
        self
    }

    /// Pre-register a declared function symbol with its parameter sorts and
    /// return sort, so that applications like `(f x)` are built with the correct
    /// result sort instead of defaulting to `Bool`.
    #[allow(dead_code)]
    pub fn seed_function(
        &mut self,
        name: impl Into<String>,
        params: Vec<SortId>,
        ret: SortId,
    ) -> &mut Self {
        self.functions.insert(name.into(), (params, ret));
        self
    }

    /// Enable or disable strict undeclared-symbol resolution.
    ///
    /// When strict (`true`), an unknown symbol is a hard parse error — matching
    /// full-script semantics where every symbol must be declared. When lenient
    /// (`false`), an unknown symbol in a bare term is minted as a fresh
    /// `Bool`-sorted variable. Embedders parsing an isolated fragment against a
    /// seeded context typically want strict resolution.
    #[allow(dead_code)]
    pub fn set_strict_symbols(&mut self, strict: bool) -> &mut Self {
        self.script_mode = strict;
        self
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
            rec_function_defs: FxHashMap::default(),
            annotations: FxHashMap::default(),
            recovery_mode: true,
            errors: Vec::new(),
            dt_constructors: FxHashMap::default(),
            dt_selectors: FxHashMap::default(),
            dt_sorts: FxHashMap::default(),
            script_mode: false,
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
}

/// Parse a decimal string to a Rational64
/// Handles decimal literals like "5.5", "3.14159", "0.0", etc.
pub(super) fn parse_decimal_to_rational(s: &str) -> Result<Rational64> {
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
    // A full script must reference only declared symbols; enable strict
    // undeclared-symbol rejection regardless of how many declarations appear.
    parser.script_mode = true;
    let mut commands = Vec::new();
    while let Some(cmd) = parser.parse_command()? {
        commands.push(cmd);
    }
    // Surface any lexical errors accumulated while tokenizing. The lexer keeps
    // producing a best-effort token stream after a malformed token (so the
    // command loop above can still make progress), recording each problem in
    // `errors()` instead of aborting. A genuine SMT-LIB script must be
    // lexically well-formed, so once the whole input has been consumed we
    // reject it if any lexical error was seen rather than silently solving a
    // corrupted problem (leftover of todo-1174).
    if let Some(err) = parser.lexer.errors().first() {
        return Err(OxizError::ParseError {
            position: err.pos,
            message: format!("lexical error: {}", err.message),
        });
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
            _ => panic!("expected DefineSort command"),
        }
    }

    #[test]
    fn test_parse_define_fun() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const x Int)
            (declare-const y Int)
            (define-fun myFunc ((a Int) (b Int)) Bool (< a b))
            (assert (myFunc x y))
            (check-sat)
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse define-fun script");
        assert_eq!(commands.len(), 5);

        match &commands[2] {
            Command::DefineFun(name, params, ret_sort, _body) => {
                assert_eq!(name, "myFunc");
                assert_eq!(params.len(), 2);
                assert_eq!(ret_sort, "Bool");
            }
            _ => panic!("expected DefineFun command"),
        }
    }

    #[test]
    fn test_parse_define_fun_nullary() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const a Int)
            (declare-const b Int)
            (define-fun arr () (Array Int Int) ((as const (Array Int Int)) 0))
            (assert (= (select arr 3) 0))
            (check-sat)
        "#;

        let _commands =
            parse_script(script, &mut manager).expect("should parse define-fun nullary script");
    }

    #[test]
    fn test_parse_new_commands() {
        let mut manager = TermManager::new();
        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (> x 0))
            (check-sat)
            (get-unsat-core)
            (get-assertions)
            (get-assignment)
            (get-proof)
            (reset-assertions)
            (echo "hello")
            (set-info :author "test")
            (get-info :version)
            (exit)
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse new commands");
        assert_eq!(commands.len(), 13);
    }

    #[test]
    fn test_parse_check_sat_assuming() {
        let mut manager = TermManager::new();
        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Int)
            (assert (> x 0))
            (check-sat-assuming (true false))
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse check-sat-assuming");
        assert_eq!(commands.len(), 5);

        match &commands[4] {
            Command::CheckSatAssuming(assumptions) => {
                assert_eq!(assumptions.len(), 2);
            }
            _ => panic!("expected CheckSatAssuming command"),
        }
    }

    #[test]
    fn test_parse_simplify() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const x Int)
            (simplify (+ x 0))
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse simplify");
        assert_eq!(commands.len(), 2);

        match &commands[1] {
            Command::Simplify(_term) => {}
            _ => panic!("expected Simplify command"),
        }
    }

    #[test]
    fn test_parse_annotations() {
        let mut manager = TermManager::new();
        let expr =
            parse_term("(! true :named foo)", &mut manager).expect("should parse annotated term");
        assert_eq!(expr, manager.mk_true());
    }

    #[test]
    fn test_parse_pattern_annotation() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const f Int)
            (assert (forall ((x Int)) (! (> x 0) :pattern (x))))
        "#;

        let _commands =
            parse_script(script, &mut manager).expect("should parse pattern annotation");
    }

    #[test]
    fn test_parse_multiple_annotations() {
        let mut manager = TermManager::new();
        let expr = parse_term("(! true :named foo :weight 3)", &mut manager).expect("should parse");
        assert_eq!(expr, manager.mk_true());
    }

    #[test]
    fn test_error_recovery() {
        let mut manager = TermManager::new();
        let result = parse_term("(+ x", &mut manager);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_recovery_infrastructure() {
        let mut manager = TermManager::new();
        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (> x 0))
            (unknown-command arg1 arg2)
            (check-sat)
        "#;

        // Unknown commands should be silently skipped
        let commands =
            parse_script(script, &mut manager).expect("should skip unknown commands and continue");
        // set-logic, declare-const, assert, check-sat (unknown-command is skipped)
        assert_eq!(commands.len(), 4);
    }

    #[test]
    fn test_parse_decimal_literals() {
        let mut manager = TermManager::new();

        let zero_point_five = parse_term("0.5", &mut manager).expect("should parse 0.5");
        let expected_half = manager.mk_real(num_rational::Rational64::new(1, 2));
        assert_eq!(zero_point_five, expected_half);

        let five_point_five = parse_term("5.5", &mut manager).expect("should parse 5.5");
        let expected_5_5 = manager.mk_real(num_rational::Rational64::new(11, 2));
        assert_eq!(five_point_five, expected_5_5);

        let three_point_14 = parse_term("3.14", &mut manager).expect("should parse 3.14");
        let expected_314 = manager.mk_real(num_rational::Rational64::new(314, 100));
        assert_eq!(three_point_14, expected_314);

        let zero = parse_term("0.0", &mut manager).expect("should parse 0.0");
        let expected_zero = manager.mk_real(num_rational::Rational64::new(0, 1));
        assert_eq!(zero, expected_zero);
    }

    #[test]
    fn test_parse_real_arithmetic() {
        let mut manager = TermManager::new();

        let add = parse_term("(+ 1.5 2.5)", &mut manager).expect("should parse real addition");
        let _one_half = manager.mk_real(num_rational::Rational64::new(3, 2));
        let _five_half = manager.mk_real(num_rational::Rational64::new(5, 2));

        // Verify it's an addition node
        let term = manager.get(add).expect("term should exist");
        match &term.kind {
            crate::ast::TermKind::Add(_) => {}
            _ => panic!("expected Add term, got {:?}", term.kind),
        }
    }

    #[test]
    fn test_parse_unary_minus_real() {
        let mut manager = TermManager::new();

        // Test (- 3.5) - should parse as negation of 3.5
        let neg_real = parse_term("(- 3.5)", &mut manager).expect("should parse (- 3.5)");
        let term = manager.get(neg_real).expect("term should exist");
        match &term.kind {
            crate::ast::TermKind::Neg(_) => {}
            crate::ast::TermKind::RealConst(r) => {
                // Might be constant-folded
                assert!(
                    *r < num_rational::Rational64::new(0, 1),
                    "should be negative"
                );
            }
            _ => panic!("expected Neg or RealConst term, got {:?}", term.kind),
        }

        // Test (- 0.0) - should parse as negation of zero
        let neg_zero = parse_term("(- 0.0)", &mut manager).expect("should parse (- 0.0)");
        let term2 = manager.get(neg_zero).expect("term should exist");
        match &term2.kind {
            crate::ast::TermKind::Neg(_) => {}
            crate::ast::TermKind::RealConst(_) => {}
            _ => panic!("expected Neg or RealConst term, got {:?}", term2.kind),
        }

        // Test (- 1.5 0.5) - should parse as subtraction
        let sub_real = parse_term("(- 1.5 0.5)", &mut manager).expect("should parse (- 1.5 0.5)");
        let term3 = manager.get(sub_real).expect("term should exist");
        match &term3.kind {
            crate::ast::TermKind::Sub(_, _) => {}
            crate::ast::TermKind::RealConst(r) => {
                assert_eq!(*r, num_rational::Rational64::new(1, 1));
            }
            _ => panic!("expected Sub or RealConst term, got {:?}", term3.kind),
        }
    }

    #[test]
    fn test_parse_array_sort() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (assert (= (select a i) 0))
            (check-sat)
        "#;

        let commands = parse_script(script, &mut manager).expect("should parse array sort");
        assert_eq!(commands.len(), 4);

        match &commands[0] {
            Command::DeclareConst(name, sort) => {
                assert_eq!(name, "a");
                assert!(
                    sort.contains("Array"),
                    "sort should mention Array: {}",
                    sort
                );
            }
            _ => panic!("expected DeclareConst"),
        }
    }

    #[test]
    fn test_parse_string_literal() {
        let mut manager = TermManager::new();

        // Parse a simple string literal
        let s = parse_term(r#""hello""#, &mut manager).expect("should parse string literal");
        let term = manager.get(s).expect("term should exist");
        match &term.kind {
            crate::ast::TermKind::StringLit(val) => {
                assert_eq!(val, "hello");
            }
            _ => panic!("expected StringLit term, got {:?}", term.kind),
        }

        // Parse a string concatenation
        let concat = parse_term(r#"(str.++ "hello" " world")"#, &mut manager)
            .expect("should parse string concatenation");
        let concat_term = manager.get(concat).expect("term should exist");
        match &concat_term.kind {
            crate::ast::TermKind::StrConcat(_, _) => {}
            _ => panic!("expected StrConcat term, got {:?}", concat_term.kind),
        }

        // Parse string contains
        let contains = parse_term(r#"(str.contains "hello world" "world")"#, &mut manager)
            .expect("should parse string contains");
        let contains_term = manager.get(contains).expect("term should exist");
        match &contains_term.kind {
            crate::ast::TermKind::StrContains(_, _) => {}
            _ => panic!("expected StrContains term, got {:?}", contains_term.kind),
        }
    }

    #[test]
    fn test_parse_nary_core_operators() {
        // Regression: chainable / n-ary core operators `=`, `<`, `<=`, `>`,
        // `>=`, `=>`, `xor`, `-` were previously rejected with more than two
        // operands. They must now parse per the SMT-LIB grammar.
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const a Int)
            (declare-const b Int)
            (declare-const c Int)
            (assert (= a b c))
            (assert (< a b c))
            (assert (> a b c))
            (assert (<= a b c))
            (assert (>= a b c))
            (assert (=> (> a 0) (> b 0) (> c 0)))
            (assert (xor (> a 0) (> b 0) (> c 0)))
            (assert (= (- a b c) 0))
        "#;
        let commands =
            parse_script(script, &mut manager).expect("should parse n-ary core operators");
        // 3 declares + 8 asserts.
        assert_eq!(commands.len(), 11);
    }

    #[test]
    fn test_nary_eq_expands_to_conjunction() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let t = {
            let mut parser = Parser::with_context(
                "(= a b c)",
                &mut manager,
                [
                    ("a".to_string(), int_sort),
                    ("b".to_string(), int_sort),
                    ("c".to_string(), int_sort),
                ],
            );
            parser.parse_term().expect("should parse (= a b c)")
        };
        match &manager.get(t).expect("term should exist").kind {
            crate::ast::TermKind::And(atoms) => {
                assert_eq!(atoms.len(), 2, "(= a b c) => (and (= a b) (= b c))");
            }
            other => panic!("expected And of two Eq atoms, got {other:?}"),
        }
    }

    #[test]
    fn test_binary_eq_keeps_eq_kind() {
        // The binary case must be unchanged: a lone `(= a b)` stays an `Eq`
        // node, not a one-element conjunction.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let t = {
            let mut parser = Parser::with_context(
                "(= a b)",
                &mut manager,
                [("a".to_string(), int_sort), ("b".to_string(), int_sort)],
            );
            parser.parse_term().expect("should parse (= a b)")
        };
        match &manager.get(t).expect("term should exist").kind {
            crate::ast::TermKind::Eq(_, _) => {}
            other => panic!("expected Eq term, got {other:?}"),
        }
    }

    #[test]
    fn test_nary_minus_left_associative() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let t = {
            let mut parser = Parser::with_context(
                "(- a b c)",
                &mut manager,
                [
                    ("a".to_string(), int_sort),
                    ("b".to_string(), int_sort),
                    ("c".to_string(), int_sort),
                ],
            );
            parser.parse_term().expect("should parse (- a b c)")
        };
        // (- a b c) = (- (- a b) c): outer node is a Sub whose lhs is a Sub.
        match &manager.get(t).expect("term should exist").kind {
            crate::ast::TermKind::Sub(lhs, _) => {
                match &manager.get(*lhs).expect("lhs should exist").kind {
                    crate::ast::TermKind::Sub(_, _) => {}
                    other => panic!("expected left-nested Sub, got {other:?}"),
                }
            }
            other => panic!("expected Sub term, got {other:?}"),
        }
    }

    #[test]
    fn test_seed_declaration_resolves_sort() {
        // A seeded constant must resolve to a variable of its declared sort,
        // even though the constant was not declared in the parsed text.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let t = {
            let mut parser = Parser::new("x", &mut manager);
            parser.seed_declaration("x", int_sort);
            parser.parse_term().expect("seeded x should resolve")
        };
        let sort = manager.get(t).expect("term should exist").sort;
        assert_eq!(sort, int_sort, "seeded symbol keeps its true sort");
    }

    #[test]
    fn test_with_context_seeds_and_stays_strict() {
        // `with_context` seeds declarations and enables strict resolution:
        // seeded symbols resolve, genuinely-unknown symbols still error.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Seeded symbol resolves.
        {
            let mut parser = Parser::with_context("y", &mut manager, [("y".to_string(), int_sort)]);
            let t = parser.parse_term().expect("seeded y should resolve");
            let sort = parser.manager.get(t).expect("term should exist").sort;
            assert_eq!(sort, int_sort);
        }

        // Unseeded symbol errors in the strict seeded context.
        {
            let empty: Vec<(String, SortId)> = Vec::new();
            let mut parser = Parser::with_context("z", &mut manager, empty);
            let result = parser.parse_term();
            assert!(
                result.is_err(),
                "unseeded symbol must error in a strict seeded context"
            );
        }
    }

    #[test]
    fn test_parse_array_operations() {
        let mut manager = TermManager::new();
        let script = r#"
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const v Int)
            (assert (= (select (store a i v) i) v))
            (check-sat)
        "#;

        let _commands = parse_script(script, &mut manager).expect("should parse array operations");
    }
}
