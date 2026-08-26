//! Regression tests for two parser defects found by the 0.3.1 soundness
//! sweep:
//!
//!   1. `smtlib/parser/sorts.rs`: `parse_sort_name` resolved a 0-arity
//!      `define-sort` alias by *re-entering itself* with the alias body. The
//!      `define-sort` handler stored a body naming an unknown symbol verbatim
//!      (an unknown symbol parses as `SortKind::Uninterpreted`, which
//!      `sort_id_to_string` renders back as its own bare name), so
//!      `(define-sort A () A)` stored `A -> A` and any later *nested*
//!      reference such as `(Array A Int)` recursed forever and aborted the
//!      process with a stack overflow. `parse_sort`'s depth cap is not
//!      re-entered on that path, and would be the wrong shape of fix anyway:
//!      the recursion is infinite, not merely deep. Mutually-referential
//!      definitions (`A -> B`, `B -> A`) behaved the same way.
//!
//!   2. `smtlib/parser/build.rs`: `build_variadic` folds the n-ary operators
//!      with no n-ary term representation -- `=>`, `xor`, `-`, `div`, `/`,
//!      `str.++` -- into binary chains. Only *syntactic* nesting was charged
//!      against `MAX_PARSE_DEPTH`, so a flat `(str.++ x1 ... x100000)` of
//!      parenthesis depth 2 sailed through the limit and produced a
//!      100 000-deep term, voiding every downstream mitigation that assumed
//!      "the parser bounds term depth at 1024" -- from a single-line input.
//!
//! Every deep structure below is built with an **iterative** loop, and every
//! parse runs on a thread with an explicitly small (1 MiB) stack: a stack
//! overflow aborts the whole process, so "the call returned at all" is itself
//! part of the assertion.

use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::parse_script;

/// Run `body` on a 1 MiB stack and return its value.
///
/// Frame sizes differ between the debug and release profiles, so the stack is
/// pinned rather than inherited; the pre-fix behaviour was an abort, which no
/// assertion can catch, so the harness exists to make the *return* meaningful.
fn on_small_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(body)
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not overflow its stack")
}

/// Parse `script` on a small stack, returning the error text on failure.
fn parse_on_small_stack(script: String) -> Result<(), String> {
    on_small_stack(move || {
        let mut manager = TermManager::new();
        parse_script(&script, &mut manager)
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
}

// ============ Finding 1: self- and mutually-referential sort aliases ========

#[test]
fn self_referential_define_sort_is_rejected_at_the_definition() {
    let err = parse_on_small_stack("(define-sort A () A)\n(check-sat)\n".to_string())
        .expect_err("a sort abbreviation defined as itself has no fixed point");
    assert!(
        err.contains("cyclic sort abbreviation") && err.contains("A -> A"),
        "the diagnostic must name the cycle, got: {err}"
    );
}

#[test]
fn self_referential_alias_used_nested_does_not_recurse_forever() {
    // The pre-fix crash needed a *nested* reference: `(declare-const x A)`
    // resolves through `Parser::resolve_sort`, which short-circuits on the
    // sort manager's alias table and never reaches `parse_sort_name`. Only
    // `parse_sort`'s ordinary recursive descent, as used inside
    // `(Array A Int)`, took the looping path.
    let err = parse_on_small_stack(
        "(define-sort A () A)\n(declare-const x (Array A Int))\n(check-sat)\n".to_string(),
    )
    .expect_err("the cyclic definition must be rejected before it can be referenced");
    assert!(
        err.contains("cyclic sort abbreviation"),
        "unexpected error: {err}"
    );
}

#[test]
fn mutually_referential_define_sorts_are_rejected() {
    let err = parse_on_small_stack(
        "(define-sort A () B)\n(define-sort B () A)\n(declare-const x (Array B Int))\n".to_string(),
    )
    .expect_err("`A -> B` plus `B -> A` closes a cycle");
    // The second definition is the one that closes the cycle, and the
    // diagnostic recovers the intermediate name from the raw source symbol
    // rather than reporting the flattened `B -> B`.
    assert!(
        err.contains("define-sort 'B'") && err.contains("B -> A -> B"),
        "the diagnostic must name the whole cycle, got: {err}"
    );
}

#[test]
fn three_way_alias_cycle_is_rejected() {
    let err = parse_on_small_stack(
        "(define-sort A () B)\n(define-sort B () C)\n(define-sort C () A)\n".to_string(),
    )
    .expect_err("`A -> B -> C -> A` is a cycle");
    assert!(err.contains("C -> A -> B -> C"), "unexpected error: {err}");
}

#[test]
fn compound_body_naming_the_alias_itself_is_rejected() {
    // `(define-sort A () (Array A Int))` does not loop -- the inner `A`
    // becomes a *fresh* free sort unrelated to the abbreviation -- but that
    // silent divergence between the two `A`s is the same defect one step
    // removed, so it is rejected with the same diagnostic.
    let err = parse_on_small_stack("(define-sort A () (Array A Int))\n".to_string())
        .expect_err("a body naming its own abbreviation is not well founded");
    assert!(
        err.contains("A -> (Array A Int)"),
        "unexpected error: {err}"
    );
}

#[test]
fn legitimate_sort_abbreviations_still_resolve() {
    // A chain of abbreviations, an abbreviation of a `declare-sort`ed sort,
    // and a compound body must all keep working: the cycle check must not be
    // a blanket rejection of forward references.
    for script in [
        "(define-sort A () Int)\n(define-sort B () A)\n(declare-const x (Array B B))\n",
        "(declare-sort U 0)\n(define-sort A () U)\n(declare-const x (Array A Int))\n",
        "(define-sort IA () (Array Int Int))\n(declare-fun f () IA)\n",
        "(define-sort W () (_ BitVec 32))\n(declare-const w W)\n",
    ] {
        parse_on_small_stack(script.to_string())
            .unwrap_or_else(|e| panic!("`{script}` must parse, got: {e}"));
    }
}

// ================= Finding 2: folded n-ary chains vs MAX_PARSE_DEPTH =======

/// `(op x0 x1 ... x{n-1})` over constants of `sort`, optionally wrapped so the
/// assertion is Bool-sorted.
fn flat_application(op: &str, n: usize, sort: &str, wrap_open: &str) -> String {
    let mut script = String::new();
    for i in 0..n {
        script.push_str(&format!("(declare-const x{i} {sort})\n"));
    }
    script.push_str("(assert ");
    script.push_str(wrap_open);
    script.push('(');
    script.push_str(op);
    for i in 0..n {
        script.push_str(&format!(" x{i}"));
    }
    script.push(')');
    if !wrap_open.is_empty() {
        script.push(')');
    }
    script.push_str(")\n");
    script
}

#[test]
fn flat_folded_operators_with_100k_operands_are_rejected_not_crashed() {
    // Each of these has *syntactic* nesting depth 2 but folds into a chain of
    // ~100 000 binary nodes. Before the fix `str.++` aborted the process
    // outright; the others built the deep term and only failed later. Now the
    // depth charged for the fold makes the parser's own bound true, so each is
    // rejected up front with the documented error.
    for (op, sort, wrap) in [
        ("str.++", "String", "(= \"\" "),
        ("=>", "Bool", ""),
        ("xor", "Bool", ""),
        ("div", "Int", "(= 0 "),
        ("/", "Real", "(= 0.0 "),
        ("-", "Int", "(= 0 "),
    ] {
        let script = flat_application(op, 12_500, sort, wrap);
        let Err(err) = parse_on_small_stack(script) else {
            panic!("`{op}` with 12500 operands must be rejected, not accepted");
        };
        assert!(
            err.contains("term nesting too deep"),
            "`{op}` must fail with the documented depth error, got: {err}"
        );
    }
}

#[test]
fn genuinely_nary_operators_are_unaffected_by_the_fold_charge() {
    // `and`, `or`, `distinct`, `+`, `*`, `re.++`, `re.union` and `re.inter`
    // build n-ary `TermKind`s of depth 1, so no chain is folded and no depth
    // is charged: 100 000 operands must still parse.
    for (op, sort, wrap) in [
        ("and", "Bool", ""),
        ("or", "Bool", ""),
        ("+", "Int", "(= 0 "),
        ("*", "Int", "(= 0 "),
        ("distinct", "Int", ""),
    ] {
        let script = flat_application(op, 12_500, sort, wrap);
        parse_on_small_stack(script)
            .unwrap_or_else(|e| panic!("n-ary `{op}` with 12500 operands must parse, got: {e}"));
    }
}

#[test]
fn ordinary_folded_applications_still_parse() {
    // The charge must not disturb the arities real scripts use.
    for script in [
        "(declare-const a String)\n(assert (= \"abcd\" (str.++ a \"b\" \"c\" \"d\")))\n",
        "(assert (= 1 (- 6 2 3)))\n",
        "(assert (xor true false false))\n",
        "(assert (=> true true true))\n",
        "(assert (= 2 (div 12 3 2)))\n",
        "(assert (= 1.0 (/ 4.0 2.0 2.0)))\n",
    ] {
        parse_on_small_stack(script.to_string())
            .unwrap_or_else(|e| panic!("`{script}` must parse, got: {e}"));
    }
}

#[test]
fn a_fold_just_inside_the_budget_is_accepted_and_just_outside_is_rejected() {
    // `MAX_PARSE_DEPTH` is 1024 and a `str.++` fold costs one level per extra
    // operand, so the boundary is observable: this pins the bound as a real
    // contract rather than an accident of the chosen constant.
    let accepted = flat_application("str.++", 1000, "String", "(= \"\" ");
    parse_on_small_stack(accepted).expect("a 1000-operand str.++ stays inside the budget");

    let rejected = flat_application("str.++", 1100, "String", "(= \"\" ");
    let err = parse_on_small_stack(rejected).expect_err("a 1100-operand str.++ exceeds it");
    assert!(
        err.contains("term nesting too deep"),
        "unexpected error: {err}"
    );
}
