//! SMT-LIB2 Parser
#![allow(clippy::while_let_loop)] // Parser uses explicit loop control

use super::lexer::{Lexer, TokenKind};
use crate::ast::{TermId, TermManager};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use num_rational::Rational64;

mod commands;
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
    /// Function definitions from define-fun
    pub(super) function_defs: FxHashMap<String, (Vec<(String, String)>, TermId)>,
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
