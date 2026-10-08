//! U-Z13 — one radix rule for every printer of a bit-vector value.
//!
//! SMT-LIB gives a bit-vector value two spellings: `#b` with exactly `width`
//! binary digits, which is legal at **every** width, and `#x` with exactly
//! `width / 4` hexadecimal digits, which exists only when the width is a
//! multiple of four (a hex digit *is* four bits, so `#x` at any other width
//! would silently widen the value).
//!
//! U-Z13 is therefore an **inconsistency** bug, not an illegality one: nothing
//! the pre-fix code printed was invalid SMT-LIB, but the *same constant* came
//! back spelled two or three different ways depending on which command asked
//! and on whether the model had pinned it. The fix picks one spelling per
//! width — `#x` where it exists, `#b` otherwise, which is what the shared
//! printer and Z3 already did — and applies it in every printer. One rule, one
//! answer per width, independent of how the value came to be known. Legitimate
//! `#b` output elsewhere (the `(fp #b.. #b.. #b..)` bit-triples, for instance)
//! is a different thing and is untouched.
//!
//! # What OxiZ 0.3.3 / 0.3.4 answered before this fix
//!
//! Three printers disagreed. Measured on the 0.3.4 tree at commit `6bdf958`
//! with the same scripts [`assigned_script`] / [`free_script`] build below
//! (constant `x` of width `w`, pinned to `5` or left unconstrained):
//!
//! | width | `(get-value)` assigned | `(get-value)` unconstrained | `(get-model)` |
//! |---|---|---|---|
//! | 8  | `#x05`                 | `#b00000000`                 | `#b00000101` |
//! | 12 | `#x005`                | `#b000000000000`             | `#b000000000101` |
//! | 13 | `#b0000000000101`      | `#b0000000000000`            | `#b0000000000101` |
//! | 64 | `#x0000000000000005`   | `#b0…0` (64 digits)          | `#b0…0101` (64 digits) |
//! | 65 | `#b0…0101` (65 digits) | `#b0…0` (65 digits)          | `#b0…0101` (65 digits) |
//!
//! So at every width divisible by four the same constant was reported two or
//! three different ways: hex from `(get-value)` when the model had pinned it,
//! binary from `(get-value)` when it had not, and binary from `(get-model)`
//! always. A consumer comparing the two commands' answers for one constant
//! saw them disagree.
//!
//! The cause was two hand-rolled `format!("#b{:0>width$}", ..)` calls in
//! `oxiz-solver/src/context/model_fmt.rs` (`format_value`, and the
//! `SortKind::BitVec` arm of `default_value`) that never consulted the width,
//! while `(get-value)` on an *assigned* constant went through the shared
//! `oxiz_core::smtlib::Printer`, which had the rule right all along. Every
//! assertion below fails on 0.3.3/0.3.4 at widths 8, 12 and 64.

use oxiz_solver::Context;

/// The radix SMT-LIB prescribes for `width`, as a `#x` / `#b` prefix.
fn expected_prefix(width: u32) -> &'static str {
    if width.is_multiple_of(4) { "#x" } else { "#b" }
}

/// The number of digits SMT-LIB prescribes for `width` in that radix.
fn expected_digits(width: u32) -> usize {
    if width.is_multiple_of(4) {
        (width / 4) as usize
    } else {
        width as usize
    }
}

/// The literal for `value` at `width`, spelled the one legal way.
fn expected_literal(value: u32, width: u32) -> String {
    if width.is_multiple_of(4) {
        format!("#x{value:0>digits$x}", digits = expected_digits(width))
    } else {
        format!("#b{value:0>digits$b}", digits = expected_digits(width))
    }
}

/// A script whose single constant `x` is pinned to `5` by an equality, so the
/// model *assigns* it.
fn assigned_script(width: u32) -> String {
    format!(
        "(set-logic QF_BV)\n\
         (declare-const x (_ BitVec {width}))\n\
         (assert (= x (_ bv5 {width})))\n\
         (check-sat)\n\
         (get-value (x))\n\
         (get-model)\n"
    )
}

/// A script whose single constant `x` is not constrained at all, so the model
/// leaves it unassigned and both commands must fall back to the sort default.
fn free_script(width: u32) -> String {
    format!(
        "(set-logic QF_BV)\n\
         (declare-const x (_ BitVec {width}))\n\
         (check-sat)\n\
         (get-value (x))\n\
         (get-model)\n"
    )
}

/// Run `script` and return `(get-value line, get-model block)`.
fn get_value_and_model(script: &str) -> (String, String) {
    let mut ctx = Context::new();
    let output = match ctx.execute_script(script) {
        Ok(output) => output,
        Err(err) => panic!("execute_script failed: {err}"),
    };
    assert_eq!(
        output.first().map(String::as_str),
        Some("sat"),
        "expected the script to be sat, got {output:?}"
    );
    let get_value = output
        .get(1)
        .unwrap_or_else(|| panic!("no (get-value) response in {output:?}"))
        .clone();
    let get_model = output
        .get(2)
        .unwrap_or_else(|| panic!("no (get-model) response in {output:?}"))
        .clone();
    (get_value, get_model)
}

/// The bit-vector literal inside `text`, i.e. the one token starting `#x`/`#b`.
fn literal_in(text: &str) -> String {
    let start = text
        .find(['#'])
        .unwrap_or_else(|| panic!("no bit-vector literal in {text:?}"));
    let rest = &text[start..];
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '#'))
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

/// The widths that exercise both sides of the rule: `8`/`12`/`64` are
/// multiples of four (`#x`), `13`/`65` are not (`#b`), and `65` is also past
/// the 64-bit word boundary where the value no longer fits a `u64`.
const WIDTHS: [u32; 5] = [8, 12, 13, 64, 65];

/// An *assigned* constant prints with the width's radix, from both commands.
///
/// 0.3.3/0.3.4: `(get-value)` was already right; `(get-model)` said `#b` at
/// every width, so the two disagreed at 8, 12 and 64.
#[test]
fn assigned_constant_uses_the_width_radix_in_both_commands() {
    for width in WIDTHS {
        let (get_value, get_model) = get_value_and_model(&assigned_script(width));
        let expected = expected_literal(5, width);
        assert_eq!(
            literal_in(&get_value),
            expected,
            "(get-value) at width {width}: {get_value}"
        );
        assert_eq!(
            literal_in(&get_model),
            expected,
            "(get-model) at width {width}: {get_model}"
        );
    }
}

/// An *unconstrained* constant prints with the same rule — the radix is a
/// function of the width alone, never of whether the model pinned the value.
///
/// 0.3.3/0.3.4: `#b` from both commands at every width, so an 8-bit
/// unconstrained constant answered `#b00000000` where the same constant, once
/// pinned, answered `#x05`.
#[test]
fn unconstrained_constant_uses_the_width_radix_in_both_commands() {
    for width in WIDTHS {
        let (get_value, get_model) = get_value_and_model(&free_script(width));
        let expected = expected_literal(0, width);
        assert_eq!(
            literal_in(&get_value),
            expected,
            "(get-value) at width {width}: {get_value}"
        );
        assert_eq!(
            literal_in(&get_model),
            expected,
            "(get-model) at width {width}: {get_model}"
        );
    }
}

/// The four printers agree with each other, whatever the rule says.
///
/// This is the invariant a consumer actually depends on, checked without
/// restating the expected strings: for one width, the assigned and the
/// unconstrained answers must use the same prefix and the same digit count,
/// and `(get-value)` must agree with `(get-model)` in both cases.
///
/// 0.3.3/0.3.4: failed at widths 8, 12 and 64 (`#x05` vs `#b00000101`).
#[test]
fn every_printer_agrees_on_prefix_and_digit_count() {
    for width in WIDTHS {
        let (assigned_value, assigned_model) = get_value_and_model(&assigned_script(width));
        let (free_value, free_model) = get_value_and_model(&free_script(width));
        let literals = [
            ("(get-value) assigned", literal_in(&assigned_value)),
            ("(get-model) assigned", literal_in(&assigned_model)),
            ("(get-value) unconstrained", literal_in(&free_value)),
            ("(get-model) unconstrained", literal_in(&free_model)),
        ];
        for (label, literal) in &literals {
            assert!(
                literal.starts_with(expected_prefix(width)),
                "{label} at width {width}: {literal} does not start with {}",
                expected_prefix(width)
            );
            assert_eq!(
                literal.len() - 2,
                expected_digits(width),
                "{label} at width {width}: {literal} has the wrong digit count"
            );
        }
    }
}

/// Widths 8 and 64 spelled out, so a future change to the shared helper has to
/// face the literal strings and not only the derived rule.
///
/// 0.3.3/0.3.4: `(get-model)` answered `#b00000101` / 64 binary digits, and
/// the unconstrained `(get-value)` answered `#b00000000` / 64 binary zeros.
#[test]
fn width_eight_and_sixty_four_literals_are_pinned() {
    let (value, model) = get_value_and_model(&assigned_script(8));
    assert_eq!(literal_in(&value), "#x05");
    assert_eq!(literal_in(&model), "#x05");
    let (value, model) = get_value_and_model(&free_script(8));
    assert_eq!(literal_in(&value), "#x00");
    assert_eq!(literal_in(&model), "#x00");

    let (value, model) = get_value_and_model(&assigned_script(64));
    assert_eq!(literal_in(&value), "#x0000000000000005");
    assert_eq!(literal_in(&model), "#x0000000000000005");
    let (value, model) = get_value_and_model(&free_script(64));
    assert_eq!(literal_in(&value), "#x0000000000000000");
    assert_eq!(literal_in(&model), "#x0000000000000000");

    // Width 13 has no `#x` form at all: 13 is not a multiple of four.
    let (value, model) = get_value_and_model(&assigned_script(13));
    assert_eq!(literal_in(&value), "#b0000000000101");
    assert_eq!(literal_in(&model), "#b0000000000101");

    // Width 65 likewise, and past the 64-bit word boundary.
    let (value, model) = get_value_and_model(&free_script(65));
    assert_eq!(literal_in(&value), format!("#b{}", "0".repeat(65)));
    assert_eq!(literal_in(&model), format!("#b{}", "0".repeat(65)));
}
