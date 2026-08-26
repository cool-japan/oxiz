//! Regression tests for audited defects in the SMT-LIB2 command parser
//! (see `oxiz-core/src/smtlib/parser/commands.rs`).
//!
//! Findings fixed:
//!   1. Unknown/unsupported-but-recognized commands (`define-fun-rec`,
//!      `define-funs-rec`) used to be silently balance-skipped and parsing
//!      continued as if nothing happened. For `define-fun-rec` in particular,
//!      a later application of the dropped function would fall through to an
//!      unconstrained, Bool-sorted uninterpreted-function term (see
//!      `terms.rs`), silently producing a wrong sat/unsat answer with no
//!      diagnostic. Both commands are now *implemented* (see
//!      `parser::recfun`): they parse to a `DefineFunsRec` command carrying
//!      each function's signature and body, and the solver discharges the
//!      definitional axiom by fuel-bounded unfolding. The tests below pin the
//!      shape of that command, since it is what stops the definition from
//!      being dropped. (`get-unsat-assumptions`, formerly in this group, is
//!      likewise now fully implemented.)
//!   2. `set-option` discarded numeral/string/decimal option values,
//!      always storing `""` because it only accepted `Symbol` tokens.
//!   3. `declare-datatypes` (the multi/mutual-datatype form) only ever
//!      parsed the *first* constructor group and then errored trying to
//!      match a second group's opening paren against the closing paren it
//!      expected, so any script declaring more than one datatype failed to
//!      parse at all. Selector sorts were also restricted to bare symbols,
//!      rejecting parametric sorts like `(Array Int Int)`.
//!
//! `declare-sort` is also covered here since it was previously silently
//! skipped along with genuinely-unrecognized commands.

use oxiz_core::ast::{TermKind, TermManager};
use oxiz_core::smtlib::{Command, parse_script};

// ---------------------------------------------------------------------
// Finding: define-fun-rec / define-funs-rec silently skipped
// ---------------------------------------------------------------------

#[test]
fn define_fun_rec_parses_to_a_definition_not_a_dropped_command() {
    let mut manager = TermManager::new();
    // Historically this script parsed as 4 commands with the definition
    // *dropped*, so `f` was an unconstrained UF at its use site. It must now
    // yield a real `DefineFunsRec` carrying `f`'s body.
    let script = r#"
        (declare-const x Int)
        (define-fun-rec f ((n Int)) Int (ite (= n 0) 0 (+ n (f (- n 1)))))
        (assert (= x (f 3)))
        (check-sat)
    "#;
    let commands = parse_script(script, &mut manager).expect("define-fun-rec must parse");
    assert_eq!(commands.len(), 4);
    match &commands[1] {
        Command::DefineFunsRec(defs) => {
            assert_eq!(defs.len(), 1, "define-fun-rec defines exactly one function");
            let def = &defs[0];
            assert_eq!(def.name, "f");
            assert_eq!(def.params, vec![("n".to_string(), "Int".to_string())]);
            assert_eq!(def.ret_sort, "Int");
            assert_eq!(
                def.formal_vars.len(),
                1,
                "one interned formal var per parameter"
            );
            // The formal var must be the *Int*-sorted `n` the body mentions,
            // not a re-derived Bool placeholder (see `RecFunDecl::formal_vars`).
            let formal = manager.get(def.formal_vars[0]).expect("formal var exists");
            assert_eq!(formal.sort, manager.sorts.int_sort);
            assert!(matches!(formal.kind, TermKind::Var(_)));
        }
        other => panic!("expected DefineFunsRec, got {other:?}"),
    }
}

#[test]
fn define_fun_rec_body_calls_itself_through_an_apply_node() {
    // The registration-before-body ordering is what makes the self-reference
    // resolve at all; if it regressed, the body parse would fail outright with
    // "unknown constant or symbol: f".
    let mut manager = TermManager::new();
    let script = "(define-fun-rec f ((n Int)) Int (ite (= n 0) 0 (f (- n 1))))";
    let commands = parse_script(script, &mut manager).expect("recursive body must parse");
    let Command::DefineFunsRec(defs) = &commands[0] else {
        panic!("expected DefineFunsRec");
    };
    assert!(
        mentions_apply_of(&manager, defs[0].body, "f"),
        "the body must contain a real `(f ..)` application node"
    );
}

#[test]
fn define_funs_rec_parses_mutual_recursion() {
    let mut manager = TermManager::new();
    let script = r#"
        (define-funs-rec
          ((is-even ((n Int)) Bool) (is-odd ((n Int)) Bool))
          ((ite (= n 0) true (is-odd (- n 1)))
           (ite (= n 0) false (is-even (- n 1)))))
        (check-sat)
    "#;
    let commands = parse_script(script, &mut manager)
        .expect("define-funs-rec must parse, including the forward reference to is-odd");
    assert_eq!(commands.len(), 2);
    let Command::DefineFunsRec(defs) = &commands[0] else {
        panic!("expected DefineFunsRec, got {:?}", commands[0]);
    };
    assert_eq!(defs.len(), 2, "the whole mutually recursive group is kept");
    assert_eq!(defs[0].name, "is-even");
    assert_eq!(defs[1].name, "is-odd");
    assert_eq!(defs[0].ret_sort, "Bool");
    assert_eq!(defs[1].ret_sort, "Bool");
    // Parsing *all* signatures before *any* body is exactly what lets each
    // body call its sibling.
    assert!(
        mentions_apply_of(&manager, defs[0].body, "is-odd"),
        "is-even's body must call is-odd"
    );
    assert!(
        mentions_apply_of(&manager, defs[1].body, "is-even"),
        "is-odd's body must call is-even"
    );
}

#[test]
fn define_fun_rec_nullary_is_accepted() {
    // An arity-0 recursive definition is legal SMT-LIB. It registers as a
    // constant, so the self-reference resolves as a `Var`.
    let mut manager = TermManager::new();
    let script = "(define-fun-rec c () Int (+ c 1))";
    let commands = parse_script(script, &mut manager).expect("nullary define-fun-rec must parse");
    let Command::DefineFunsRec(defs) = &commands[0] else {
        panic!("expected DefineFunsRec");
    };
    assert_eq!(defs[0].name, "c");
    assert!(defs[0].params.is_empty());
    assert!(defs[0].formal_vars.is_empty());
    assert_eq!(defs[0].ret_sort, "Int");
}

#[test]
fn define_funs_rec_body_count_mismatch_is_an_error() {
    // Dropping or duplicating a body would silently mis-pair definitions.
    let mut manager = TermManager::new();
    let script = r#"
        (define-funs-rec
          ((f ((n Int)) Int) (g ((n Int)) Int))
          ((ite (= n 0) 0 (g (- n 1)))))
    "#;
    let result = parse_script(script, &mut manager);
    assert!(
        result.is_err(),
        "a body list shorter than the declaration list must be an explicit error"
    );
}

/// Whether `term` mentions an `Apply` node whose function symbol is `name`.
fn mentions_apply_of(manager: &TermManager, term: oxiz_core::ast::TermId, name: &str) -> bool {
    oxiz_core::ast::traversal::collect_subterms(term, manager)
        .into_iter()
        .any(|t| match manager.get(t).map(|t| &t.kind) {
            Some(TermKind::Apply { func, .. }) => manager.resolve_str(*func) == name,
            _ => false,
        })
}

#[test]
fn get_unsat_assumptions_parses_to_command() {
    // `get-unsat-assumptions` is now implemented (it used to be rejected as
    // unsupported): the parser must produce a dedicated `GetUnsatAssumptions`
    // command so the solver context can report the failed assumptions from the
    // most recent `check-sat-assuming`.
    let mut manager = TermManager::new();
    let script = "(check-sat) (get-unsat-assumptions)";
    let commands = parse_script(script, &mut manager)
        .expect("get-unsat-assumptions must parse successfully now that it is implemented");
    assert!(
        commands
            .iter()
            .any(|c| matches!(c, Command::GetUnsatAssumptions)),
        "parser must emit a GetUnsatAssumptions command"
    );
}

#[test]
fn genuinely_unknown_command_is_still_leniently_skipped() {
    // Commands with no defined SMT-LIB semantics we understand (vendor /
    // tooling extensions) are still balance-skipped for interoperability;
    // only recognized-but-unsupported commands with real solving-semantics
    // impact are hard-rejected.
    let mut manager = TermManager::new();
    let script = "(set-logic QF_LIA) (some-vendor-extension foo bar) (check-sat)";
    let commands = parse_script(script, &mut manager)
        .expect("unrecognized vendor/tooling commands should still be leniently skipped");
    assert_eq!(commands.len(), 2);
}

// ---------------------------------------------------------------------
// Finding: set-option silently drops numeral/string values
// ---------------------------------------------------------------------

#[test]
fn set_option_preserves_numeral_value() {
    let mut manager = TermManager::new();
    let commands = parse_script("(set-option :timeout 5000)", &mut manager)
        .expect("set-option with a numeral value should parse");
    assert_eq!(commands.len(), 1);
    match &commands[0] {
        Command::SetOption(key, val) => {
            assert_eq!(key, "timeout");
            assert_eq!(
                val, "5000",
                "numeral value must be preserved, not dropped to \"\""
            );
        }
        other => panic!("expected SetOption, got {other:?}"),
    }
}

#[test]
fn set_option_preserves_string_value() {
    let mut manager = TermManager::new();
    let commands = parse_script(
        r#"(set-option :some.string.opt "hello world")"#,
        &mut manager,
    )
    .expect("set-option with a string value should parse");
    match &commands[0] {
        Command::SetOption(key, val) => {
            assert_eq!(key, "some.string.opt");
            assert_eq!(val, "hello world");
        }
        other => panic!("expected SetOption, got {other:?}"),
    }
}

#[test]
fn set_option_preserves_symbol_value() {
    let mut manager = TermManager::new();
    let commands = parse_script("(set-option :produce-models true)", &mut manager)
        .expect("set-option with a symbol value should parse");
    match &commands[0] {
        Command::SetOption(key, val) => {
            assert_eq!(key, "produce-models");
            assert_eq!(val, "true");
        }
        other => panic!("expected SetOption, got {other:?}"),
    }
}

#[test]
fn set_option_missing_value_is_an_error() {
    let mut manager = TermManager::new();
    let result = parse_script("(set-option :timeout)", &mut manager);
    assert!(
        result.is_err(),
        "a missing set-option value must be an explicit error, not silently \"\""
    );
}

// ---------------------------------------------------------------------
// Finding: declare-datatypes only parses the first constructor group
// ---------------------------------------------------------------------

#[test]
fn declare_datatypes_multi_datatype_parses_all_groups() {
    let mut manager = TermManager::new();
    let script = r#"
        (declare-datatypes ((Color 0) (Shape 0))
          (((red) (green) (blue))
           ((circle) (square) (triangle))))
    "#;
    let commands = parse_script(script, &mut manager)
        .expect("multi-datatype declare-datatypes should parse every constructor group");
    assert_eq!(commands.len(), 1);
    match &commands[0] {
        Command::DeclareDatatype { name, constructors } => {
            assert!(name.contains("Color"), "name should mention Color: {name}");
            assert!(name.contains("Shape"), "name should mention Shape: {name}");
            let ctor_names: Vec<&str> = constructors.iter().map(|(n, _)| n.as_str()).collect();
            for expected in ["red", "green", "blue", "circle", "square", "triangle"] {
                assert!(
                    ctor_names.contains(&expected),
                    "constructor '{expected}' missing from parsed result: {ctor_names:?}"
                );
            }
        }
        other => panic!("expected DeclareDatatype, got {other:?}"),
    }
}

#[test]
fn declare_datatypes_second_group_constructors_are_usable_afterward() {
    // This is the exact failure mode from the audit: previously
    // `expect_rparen` errored trying to consume the second constructor
    // group's opening paren, so the whole script failed to parse.
    let mut manager = TermManager::new();
    let script = r#"
        (declare-datatypes ((Color 0) (Shape 0))
          (((red) (green))
           ((circle) (square))))
        (declare-const c Color)
        (declare-const s Shape)
        (assert (= c red))
        (assert (= s circle))
        (check-sat)
    "#;
    let commands = parse_script(script, &mut manager)
        .expect("a script using constructors from both datatypes should parse successfully");
    // declare-datatypes, 2x declare-const, 2x assert, check-sat
    assert_eq!(commands.len(), 6);
}

#[test]
fn declare_datatypes_mutually_recursive_selectors_parse() {
    let mut manager = TermManager::new();
    let script = r#"
        (declare-datatypes ((Tree 0) (TreeList 0))
          (((leaf) (node (value Int) (children TreeList)))
           ((tnil) (tcons (thead Tree) (ttail TreeList)))))
    "#;
    let commands = parse_script(script, &mut manager)
        .expect("mutually recursive datatype selectors should parse");
    assert_eq!(commands.len(), 1);
}

#[test]
fn declare_datatype_selector_with_parametric_sort_parses() {
    // Previously selector sorts only accepted bare symbols (`expect_symbol`),
    // so a parametric sort like `(Array Int Int)` failed to parse at all.
    let mut manager = TermManager::new();
    let script = "(declare-datatype Box ((mk-box (contents (Array Int Int)))))";
    let commands = parse_script(script, &mut manager)
        .expect("a parametric selector sort should parse, not error");
    match &commands[0] {
        Command::DeclareDatatype { constructors, .. } => {
            let (ctor_name, selectors) = &constructors[0];
            assert_eq!(ctor_name, "mk-box");
            assert_eq!(selectors[0].0, "contents");
            assert_eq!(selectors[0].1, "(Array Int Int)");
        }
        other => panic!("expected DeclareDatatype, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Finding: declare-sort silently skipped
// ---------------------------------------------------------------------

#[test]
fn declare_sort_is_implemented_not_silently_skipped() {
    let mut manager = TermManager::new();
    let script = r#"
        (declare-sort MySort 0)
        (declare-const x MySort)
        (assert (= x x))
        (check-sat)
    "#;
    let commands = parse_script(script, &mut manager)
        .expect("declare-sort should parse and not be silently skipped");
    assert_eq!(commands.len(), 4);
    match &commands[0] {
        Command::DeclareSort(name, arity) => {
            assert_eq!(name, "MySort");
            assert_eq!(*arity, 0);
        }
        other => panic!("expected DeclareSort, got {other:?}"),
    }
}
