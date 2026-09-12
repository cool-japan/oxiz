//! Regression tests for the `solver-cmd-wiring` audit wave.
//!
//! `Context::execute_script` used to silently swallow
//! `declare-sort`/`define-sort`/`define-fun`/`declare-datatype(s)`
//! (`// Ignore these commands for now`). The concrete, observable defect
//! this caused: `Context::parse_sort_name` (used to resolve the sort
//! *strings* carried by `Command::DeclareConst`/`Command::DeclareFun`)
//! only recognized `Bool`/`Int`/`Real`/legacy-`BitVecN` and silently
//! defaulted every other sort name -- including any user-declared
//! uninterpreted sort, datatype, or compound `(Array ..)`/`(_ BitVec
//! ..)` sort -- to `Bool`. That desynced the constant registered for
//! `get-model`/`get-value` output from the (correctly sorted) term the
//! parser actually built for use in assertions, so `get-model` reported
//! a `Bool`-sorted phantom variable with a meaningless value instead of
//! the constant that was actually solved for.
//!
//! Each test below pins down one restored behavior.

use oxiz_solver::Context;

/// `declare-sort` must give the constant its real (uninterpreted) sort,
/// not silently downgrade it to `Bool` -- otherwise `get-model` reports
/// a value for an unrelated phantom `Bool` variable while the real,
/// asserted-over constant of sort `S` never appears at all.
#[test]
fn declare_sort_constant_appears_in_model_with_its_own_sort() {
    let mut ctx = Context::new();
    let script = r#"
        (declare-sort S 0)
        (declare-const x S)
        (declare-const y S)
        (assert (= x y))
        (check-sat)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output, vec!["sat"]);

    let model = ctx.get_model().expect("sat result must produce a model");
    assert_eq!(model.len(), 2, "both x and y must be registered constants");
    for (name, sort, _value) in &model {
        assert_eq!(
            sort, "S",
            "constant {name} must keep its declared sort S, not Bool"
        );
    }

    // `x` and `y` were asserted equal: their reported values must agree
    // (this would fail if they had silently become two independent,
    // unconstrained Bool phantoms instead of the real constrained term).
    let x_value = &model
        .iter()
        .find(|(n, _, _)| n == "x")
        .expect("x in model")
        .2;
    let y_value = &model
        .iter()
        .find(|(n, _, _)| n == "y")
        .expect("y in model")
        .2;
    assert_eq!(
        x_value, y_value,
        "x = y was asserted, so their model values must match"
    );

    assert!(
        ctx.declared_sort_names()
            .any(|(n, arity)| n == "S" && arity == 0),
        "declare-sort must be tracked for introspection"
    );
}

/// A `(declare-const arr (Array Int Int))` must resolve to a genuine
/// Array sort end-to-end (not `Bool`), so an `Array`-typed constraint
/// over it is actually solved as such.
#[test]
fn declare_const_array_sort_resolves_correctly() {
    let mut ctx = Context::new();
    let script = r#"
        (declare-const arr (Array Int Int))
        (assert (= (select arr 0) 5))
        (check-sat)
        (get-model)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output[0], "sat");

    let model = ctx.get_model().expect("sat result must produce a model");
    let (_, sort, _) = model
        .iter()
        .find(|(n, _, _)| n == "arr")
        .expect("arr must be a registered constant");
    assert_eq!(sort, "(Array Int Int)", "arr must keep its full Array sort");
}

/// A `(_ BitVec n)` sort string (the form `oxiz_core`'s printer emits)
/// must resolve to a real bit-vector sort of the right width, not
/// `Bool`.
#[test]
fn declare_const_bitvec_sort_resolves_correctly() {
    let mut ctx = Context::new();
    let script = r#"
        (declare-const bv (_ BitVec 8))
        (assert (= bv (_ bv200 8)))
        (check-sat)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output, vec!["sat"]);

    let model = ctx.get_model().expect("sat result must produce a model");
    let (_, sort, value) = model
        .iter()
        .find(|(n, _, _)| n == "bv")
        .expect("bv must be a registered constant");
    assert_eq!(sort, "(_ BitVec 8)");
    // 200 = 0b1100_1000 = 0xc8. U-Z13: a width that is a multiple of four
    // prints `#x`, the same way `(get-value)` has always printed it; before
    // that fix `(get-model)` answered `#b11001000` here while `(get-value)`
    // answered `#xc8` for the same constant.
    assert_eq!(value, "#xc8", "unexpected bv value: {value}");
}

/// A nullary `define-fun` must be observable in `get-model`/`get-value`
/// with its actual (defined) value, not silently dropped.
#[test]
fn nullary_define_fun_is_visible_in_model() {
    let mut ctx = Context::new();
    let script = r#"
        (define-fun answer () Int 42)
        (check-sat)
        (get-value (answer))
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output[0], "sat");
    assert!(
        output[1].contains("42"),
        "define-fun'd constant must evaluate to its defined body: {}",
        output[1]
    );
}

/// A parameterized `define-fun` must be registered for signature
/// introspection (`get_fun_signature`/`declared_function_names`), like
/// `declare-fun` is.
///
/// Note: this only checks *registration*. Whether in-script call sites
/// of a parameterized `define-fun` are actually substituted soundly is
/// an `oxiz-core` parser concern (`Parser::parse_apply`'s defined-function
/// substitution, `oxiz-core/src/smtlib/parser/terms.rs`), outside this
/// package's owned files -- and, discovered while writing this test, is
/// itself currently broken there: the substitution looks up each
/// parameter's sort via `self.constants.get(param_name)` (the
/// *declared-constant* table) instead of the parameter-sort table
/// actually populated for `define-fun` params, silently falls back to
/// `Bool` when the name isn't a declared constant, and therefore builds
/// a substitution key that never matches the real (correctly-sorted)
/// parameter variable inside `body` -- so calls are left unsubstituted
/// and the parameter stays a free, unconstrained variable. Left for a
/// separate `oxiz-core` fix; flagged here rather than silently worked
/// around.
#[test]
fn parameterized_define_fun_registers_signature() {
    let mut ctx = Context::new();
    let script = r#"
        (define-fun double ((x Int)) Int (* 2 x))
        (check-sat)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output, vec!["sat"]);

    let sig = ctx
        .get_fun_signature("double")
        .expect("double must be registered");
    assert_eq!(sig.0.len(), 1, "double must be registered with 1 argument");
    assert_eq!(
        sig.0[0], ctx.terms.sorts.int_sort,
        "double's argument sort must be Int"
    );
    assert_eq!(
        sig.1, ctx.terms.sorts.int_sort,
        "double's return sort must be Int"
    );
    assert!(ctx.declared_function_names().any(|n| n == "double"));
}

/// `declare-datatype` must expose its constructors/selectors as
/// declared functions (mirroring `declare-fun`, as Z3 does implicitly)
/// and in-script constructor application must resolve to the real
/// datatype sort.
#[test]
fn declare_datatype_registers_constructors_and_selectors() {
    let mut ctx = Context::new();
    let script = r#"
        (declare-datatype Pair ((mk-pair (first Int) (second Int))))
        (declare-const p Pair)
        (assert (= p (mk-pair 1 2)))
        (assert (= (first p) 1))
        (check-sat)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output, vec!["sat"]);

    let sig = ctx
        .get_fun_signature("mk-pair")
        .expect("constructor mk-pair must be registered");
    assert_eq!(sig.0.len(), 2, "mk-pair must take 2 arguments");

    let sel_sig = ctx
        .get_fun_signature("first")
        .expect("selector first must be registered");
    assert_eq!(sel_sig.0.len(), 1, "selector first must take 1 argument");
    assert_eq!(sel_sig.1, ctx.terms.sorts.int_sort, "first must return Int");
}

/// A constant declared with a datatype sort must be reported in
/// `get-model` with its real datatype name, resolved through
/// `SortManager`'s own interner (`datatype_name`) -- not through
/// `TermManager`'s *separate* interner, which holds unrelated spurs.
///
/// Regression: `format_sort_name`'s `Datatype` arm previously called
/// `self.terms.resolve_str(spur)`, resolving the datatype-name spur (an
/// index into `SortManager`'s interner) against `TermManager`'s interner
/// -- yielding a plausible-but-wrong string (e.g. `p` reported with sort
/// `"first"` instead of `"Pair"`).
#[test]
fn declare_datatype_constant_reports_its_datatype_sort_in_model() {
    let mut ctx = Context::new();
    let script = r#"
        (declare-datatype Pair ((mk-pair (first Int) (second Int))))
        (declare-const p Pair)
        (assert (= p (mk-pair 1 2)))
        (check-sat)
        (get-model)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output[0], "sat");

    let model = ctx.get_model().expect("sat result must produce a model");
    let (_, sort, _) = model
        .iter()
        .find(|(n, _, _)| n == "p")
        .expect("p must be a registered constant");
    assert_eq!(
        sort, "Pair",
        "datatype-sorted constant p must report its real sort Pair, \
         not a spur resolved through the wrong interner"
    );
}

/// A 0-arity `define-sort` alias must be usable when solving; it must
/// not desync the constant's registered sort from the one used in
/// assertions.
#[test]
fn define_sort_alias_resolves_to_its_target() {
    let mut ctx = Context::new();
    let script = r#"
        (define-sort MyInt () Int)
        (declare-const x MyInt)
        (assert (= x 7))
        (check-sat)
    "#;
    let output = ctx.execute_script(script).expect("script should parse");
    assert_eq!(output, vec!["sat"]);

    let model = ctx.get_model().expect("sat result must produce a model");
    let (_, sort, value) = model
        .iter()
        .find(|(n, _, _)| n == "x")
        .expect("x must be a registered constant");
    assert_eq!(
        sort, "Int",
        "MyInt alias must resolve to its target sort Int"
    );
    assert_eq!(value, "7");
}
