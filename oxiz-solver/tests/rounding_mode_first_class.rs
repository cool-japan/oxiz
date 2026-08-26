//! End-to-end tests for `RoundingMode` as a first-class sort.
//!
//! `(declare-const m RoundingMode)` used to be rejected at parse time: rounding
//! modes existed only as literal `RNE`/`RTZ`/... symbols baked into `fp.*`
//! operators, so a mode-sorted symbol had no representation. It is now
//! `SortKind::RoundingMode`, whose five inhabitants are nullary `Var` terms —
//! which means EUF decides equalities between them for free, and the *only*
//! thing the solver has to supply is the sort's cardinality.
//!
//! That cardinality is the subject of most of what follows. It takes two
//! axioms (`oxiz_solver::context::rounding_mode`), and each guards a different
//! failure:
//!
//! - **closure**, per declared constant, stops a sixth element from existing;
//! - **distinctness** of the five modes stops them collapsing into fewer.
//!
//! `distinct_five_modes_is_sat` and `distinct_six_modes_is_unsat` are the pair
//! that pins both at once: drop distinctness and the first flips to `unsat`,
//! drop closure and the second flips to `sat`.

use oxiz_solver::Context;

/// Run a script, returning its outputs. A parse/execution error is returned as
/// a single `ERROR: ..` line so a test can assert on it.
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(outputs) => outputs,
        Err(err) => vec![format!("ERROR: {err}")],
    }
}

/// Run a script and return its single `check-sat` verdict.
fn verdict(script: &str) -> String {
    let outputs = run(script);
    let [verdict] = outputs.as_slice() else {
        panic!("expected exactly one output line, got {outputs:?}");
    };
    verdict.clone()
}

/// Declare `count` rounding-mode constants named `m0..m{count-1}`.
fn declare_modes(count: usize) -> String {
    (0..count)
        .map(|i| format!("(declare-const m{i} RoundingMode)"))
        .collect()
}

// ---------------------------------------------------------------------------
// Declaration and model reporting.
// ---------------------------------------------------------------------------

/// The end-to-end round trip a first-class sort needs: declared, constrained,
/// solved, and reported back.
///
/// The reported value is the canonical *long* spelling, matching Z3 — and, more
/// to the point, a spelling that is valid SMT-LIB input, unlike the `?`
/// placeholder or an `@uc_RoundingMode_0` abstract witness.
#[test]
fn a_declared_rounding_mode_constant_solves_and_appears_in_the_model() {
    let outputs = run("(declare-const m RoundingMode)(assert (= m RNE))(check-sat)(get-model)");
    let [verdict, model] = outputs.as_slice() else {
        panic!("expected a verdict and a model, got {outputs:?}");
    };
    assert_eq!(verdict, "sat");
    assert!(
        model.contains("(define-fun m () RoundingMode roundNearestTiesToEven)"),
        "model must report the mode by its canonical long name: {model}"
    );
    assert!(
        !model.contains('?') && !model.contains("@uc_"),
        "a rounding mode is never a placeholder or an uninterpreted witness: {model}"
    );
}

/// The model must report the mode the assertions actually force, not the sort's
/// default.
///
/// This is the shape that made the naive implementation report a *wrong*
/// model: with no model entry for `m`, `get-model` fell through to the sort
/// default `roundNearestTiesToEven` and printed it even for
/// `(assert (= m RTZ))`. The value is now read out of the SAT assignment of
/// the closure axiom's own equality atoms, so it is the solve's own choice.
#[test]
fn the_model_reports_the_mode_the_assertions_force() {
    for (spelling, expected) in [
        ("RNA", "roundNearestTiesToAway"),
        ("RTP", "roundTowardPositive"),
        ("RTN", "roundTowardNegative"),
        ("RTZ", "roundTowardZero"),
        ("roundTowardZero", "roundTowardZero"),
    ] {
        let outputs = run(&format!(
            "(declare-const m RoundingMode)(assert (= m {spelling}))(check-sat)(get-model)"
        ));
        let [verdict, model] = outputs.as_slice() else {
            panic!("{spelling}: expected a verdict and a model, got {outputs:?}");
        };
        assert_eq!(verdict, "sat", "{spelling}");
        assert!(
            model.contains(&format!("(define-fun m () RoundingMode {expected})")),
            "{spelling}: model must report {expected}: {model}"
        );
    }
}

/// The same, forced by *exclusion* rather than by a direct equality: only one
/// mode is left, and the model must find it.
#[test]
fn the_model_reports_the_only_mode_left_after_excluding_four() {
    let outputs = run("(declare-const m RoundingMode)
         (assert (not (= m RNE)))
         (assert (not (= m RNA)))
         (assert (not (= m RTP)))
         (assert (not (= m RTN)))
         (check-sat)
         (get-model)");
    let [verdict, model] = outputs.as_slice() else {
        panic!("expected a verdict and a model, got {outputs:?}");
    };
    assert_eq!(verdict, "sat");
    assert!(
        model.contains("(define-fun m () RoundingMode roundTowardZero)"),
        "the one remaining mode must be reported: {model}"
    );
}

/// `(get-value ..)` and `(get-model)` must agree about the same constant.
#[test]
fn get_value_reports_the_same_mode_as_get_model() {
    let outputs = run(
        "(declare-const m RoundingMode)(assert (= m RTP))(check-sat)(get-value (m))(get-model)",
    );
    let [verdict, value, model] = outputs.as_slice() else {
        panic!("expected three outputs, got {outputs:?}");
    };
    assert_eq!(verdict, "sat");
    assert_eq!(value, "((m roundTowardPositive))");
    assert!(model.contains("roundTowardPositive"), "{model}");
}

// ---------------------------------------------------------------------------
// Cardinality: exactly five.
// ---------------------------------------------------------------------------

/// Five pairwise-distinct rounding modes fit: the sort has exactly five
/// elements, so this is tight but satisfiable.
///
/// Fails if the distinctness axiom is missing — without it the five mode
/// constants may be collapsed into fewer classes and there is no room for five
/// distinct values.
#[test]
fn distinct_five_modes_is_sat() {
    let script = format!(
        "{}(assert (distinct m0 m1 m2 m3 m4))(check-sat)",
        declare_modes(5)
    );
    assert_eq!(verdict(&script), "sat");
}

/// Six pairwise-distinct rounding modes do not fit. **The** cardinality test.
///
/// Fails if the closure axiom is missing: a `RoundingMode` constant would then
/// be free to take a sixth value that is none of the five modes, and the
/// pigeonhole argument this rests on would not apply.
#[test]
fn distinct_six_modes_is_unsat() {
    let script = format!(
        "{}(assert (distinct m0 m1 m2 m3 m4 m5))(check-sat)",
        declare_modes(6)
    );
    assert_eq!(verdict(&script), "unsat");
}

/// The five-distinct model must actually name five *different* modes — an
/// assignment that repeats one would satisfy the printed model's syntax while
/// falsifying the `distinct` it came from.
#[test]
fn the_five_distinct_modes_get_five_different_values() {
    let script = format!(
        "{}(assert (distinct m0 m1 m2 m3 m4))(check-sat)(get-model)",
        declare_modes(5)
    );
    let outputs = run(&script);
    let [verdict, model] = outputs.as_slice() else {
        panic!("expected a verdict and a model, got {outputs:?}");
    };
    assert_eq!(verdict, "sat");
    for mode in [
        "roundNearestTiesToEven",
        "roundNearestTiesToAway",
        "roundTowardPositive",
        "roundTowardNegative",
        "roundTowardZero",
    ] {
        assert_eq!(
            model.matches(mode).count(),
            1,
            "each mode must be used exactly once: {model}"
        );
    }
}

// ---------------------------------------------------------------------------
// Symbolic rounding modes in `fp.*` operators.
// ---------------------------------------------------------------------------

/// A *symbolic* mode inside `fp.add` solves, in the same fragment the
/// equivalent literal mode solves in.
///
/// The parser compiles `(fp.add m x y)` into a five-way `ite` over
/// `(= m RNE)` … whose leaves are ordinary concrete `fp.add` nodes; the
/// concrete FP model finder picks a mode, follows the selected branch, and
/// verifies the result. `concrete_rounding_mode_path_is_unchanged` is the
/// control: same formula, literal mode.
#[test]
fn a_symbolic_rounding_mode_inside_fp_add_is_sat() {
    assert_eq!(
        verdict(
            "(set-logic QF_FP)
             (declare-const m RoundingMode)
             (declare-const x (_ FloatingPoint 8 24))
             (declare-const y (_ FloatingPoint 8 24))
             (declare-const z (_ FloatingPoint 8 24))
             (assert (= x ((_ to_fp 8 24) RNE 1.5)))
             (assert (= y ((_ to_fp 8 24) RNE 2.5)))
             (assert (= z (fp.add m x y)))
             (check-sat)"
        ),
        "sat"
    );
}

/// The control for the test above: replacing the symbolic mode with a literal
/// one must not change the verdict. If this ever fails, the symbolic result
/// above proves nothing.
#[test]
fn concrete_rounding_mode_path_is_unchanged() {
    assert_eq!(
        verdict(
            "(set-logic QF_FP)
             (declare-const x (_ FloatingPoint 8 24))
             (declare-const y (_ FloatingPoint 8 24))
             (declare-const z (_ FloatingPoint 8 24))
             (assert (= x ((_ to_fp 8 24) RNE 1.5)))
             (assert (= y ((_ to_fp 8 24) RNE 2.5)))
             (assert (= z (fp.add RNE x y)))
             (check-sat)"
        ),
        "sat"
    );
}

/// The indexed conversions take a symbolic mode through the same case split.
#[test]
fn a_symbolic_rounding_mode_inside_an_indexed_conversion_is_sat() {
    assert_eq!(
        verdict(
            "(set-logic QF_FP)
             (declare-const m RoundingMode)
             (declare-const x (_ FloatingPoint 8 24))
             (assert (= x ((_ to_fp 8 24) m 1.5)))
             (check-sat)"
        ),
        "sat"
    );
}

/// A rounding mode pinned by an equality must reach the floating-point
/// evaluation, not just the EUF core: `RTZ` truncates where `RNE` rounds, so
/// the two disagree on this value and only one of the two verdicts can be
/// `sat`.
#[test]
fn a_pinned_symbolic_mode_selects_that_modes_arithmetic() {
    // 1/3 is not representable in float32. Rounding toward zero lands strictly
    // below rounding to nearest, so asserting `<` between the two results is
    // satisfiable exactly when each `to_fp` used its own mode.
    let script = "(set-logic QF_FP)
         (declare-const m RoundingMode)
         (declare-const lo (_ FloatingPoint 8 24))
         (declare-const hi (_ FloatingPoint 8 24))
         (assert (= m RTZ))
         (assert (= lo ((_ to_fp 8 24) m (/ 1.0 3.0))))
         (assert (= hi ((_ to_fp 8 24) RTP (/ 1.0 3.0))))
         (assert (fp.lt lo hi))
         (check-sat)";
    assert_eq!(verdict(script), "sat");
}

// ---------------------------------------------------------------------------
// The reserved-name half: `RegLan` stays un-declarable, and the regular
// expression sublanguage keeps working.
// ---------------------------------------------------------------------------

/// The companion to the parser-level `RegLan` tests: reserving the name leaves
/// the regular-language operators not merely parseable but *solvable*.
#[test]
fn reserving_reglan_does_not_disable_regex_solving() {
    assert_eq!(
        verdict(
            r#"(set-logic QF_S)
               (declare-const s String)
               (assert (str.in_re s (re.++ (str.to_re "a") re.allchar)))
               (check-sat)"#
        ),
        "sat"
    );
    // And the membership really is decided, not vacuously accepted: a string
    // that cannot match the language is refuted.
    assert_eq!(
        verdict(
            r#"(set-logic QF_S)
               (assert (str.in_re "b" (re.++ (str.to_re "a") re.allchar)))
               (check-sat)"#
        ),
        "unsat"
    );
}

/// `(declare-const r RegLan)` stays rejected, with a message that says the name
/// is reserved rather than claiming the theory is unimplemented.
#[test]
fn reglan_stays_reserved_with_an_honest_message() {
    let outputs = run("(declare-const r RegLan)");
    let [message] = outputs.as_slice() else {
        panic!("expected a single error line, got {outputs:?}");
    };
    let message = message.to_lowercase();
    assert!(message.starts_with("error:"), "{message}");
    assert!(
        message.contains("reglan") && message.contains("reserved"),
        "{message}"
    );
    assert!(
        !message.contains("not implemented") && !message.contains("not yet implemented"),
        "the re.* sublanguage is implemented; the message must not say otherwise: {message}"
    );
}

// ---------------------------------------------------------------------------
// Internal axioms stay internal.
// ---------------------------------------------------------------------------

/// The closure and distinctness axioms are OxiZ's encoding of a built-in sort,
/// not something the user wrote, so `(get-assertions)` must not report them.
#[test]
fn cardinality_axioms_do_not_leak_into_get_assertions() {
    let outputs = run("(set-option :produce-assertions true)
         (declare-const m RoundingMode)
         (assert (= m RTZ))
         (check-sat)
         (get-assertions)");
    let assertions = outputs
        .last()
        .unwrap_or_else(|| panic!("expected output, got {outputs:?}"));
    assert!(
        !assertions.contains("distinct"),
        "the distinctness axiom must stay internal: {assertions}"
    );
    assert!(
        !assertions.contains("or "),
        "the closure axiom must stay internal: {assertions}"
    );
}
