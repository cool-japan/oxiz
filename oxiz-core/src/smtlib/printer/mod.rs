//! SMT-LIB2 Printer
//!
//! This module provides two printers:
//! - [`Printer`]: A basic printer that outputs terms on a single line
//! - [`PrettyPrinter`]: A configurable pretty printer with indentation support

#[allow(unused_imports)]
use crate::prelude::*;

mod basic;
mod config;
mod model;
mod pretty;
mod proof;

// Re-export public types
pub use basic::Printer;

/// Render a bit-vector constant as a well-formed SMT-LIB2 literal.
///
/// Two invariants are enforced here, because every printer used to get both
/// of them wrong:
///
/// * **Unsigned, width-bounded value.**  A bit-vector literal denotes an
///   unsigned number in `[0, 2^width)`.  Values reaching the printer can fall
///   outside that range (the BV theory's integer relaxation can hand back a
///   negative or oversized value), which previously printed as nonsense such
///   as `#x-1`.  The value is therefore reduced modulo `2^width` first, which
///   is exactly the two's-complement wrap SMT-LIB prescribes.  The wrap
///   itself lives in [`crate::ast::bv_wrap_unsigned`], shared with the term
///   builder and the solver's model builder so the three cannot drift apart.
/// * **Radix matching the width.**  The `#x` form denotes exactly
///   `4 * digits` bits, so it is only usable when `width` is a multiple of
///   four; any other width must use the `#b` form with exactly `width` binary
///   digits.  Emitting `#x` with `width.div_ceil(4)` digits (as the printers
///   previously did) silently widens e.g. a 5-bit value to 8 bits.
pub(crate) fn format_bitvec_literal(value: &num_bigint::BigInt, width: u32) -> String {
    // Width 0 is not a legal SMT-LIB bit-vector sort; there is no literal
    // syntax for it, so fall back to the shortest well-formed binary literal
    // rather than emit a zero-digit `#x`/`#b` token that no parser accepts.
    if width == 0 {
        return "#b0".to_string();
    }

    let normalized = crate::ast::bv_wrap_unsigned(value, width);

    if width.is_multiple_of(4) {
        format!("#x{:0>digits$x}", normalized, digits = (width / 4) as usize)
    } else {
        format!("#b{:0>digits$b}", normalized, digits = width as usize)
    }
}

/// Render a string value as a well-formed SMT-LIB2 string literal, including
/// the enclosing double quotes.
///
/// This is the inverse of the lexer's literal decoding
/// ([`crate::smtlib::Lexer`]), and the single place any string value is turned
/// back into source text — every printer routes through it so the three copies
/// that used to exist cannot drift apart again.
///
/// # The rules
///
/// SMT-LIB 2.6 is not C: **a backslash is not an escape introducer** and there
/// is no `\"` and no `\\`.  The only in-literal escape the core language
/// defines is the doubled quote, and the Unicode Strings theory adds exactly
/// two `\u` forms *on input*:
///
/// * `"` is written `""` — one quote character.  The old `\"` re-parsed as a
///   backslash followed by the end of the literal, silently truncating the
///   value and corrupting the rest of the output.
/// * `\` stands for itself and is emitted verbatim, **unless** the character
///   after it is `u`, in which case the reader would try to read a `\u`
///   escape; the backslash is then written `\u{5c}`.  This is the only
///   situation in which a backslash needs any treatment at all.  The old
///   `\\` re-parsed as *two* backslashes.
/// * Every code point outside printable ASCII (`0x20..=0x7E`) is written
///   `\u{...}` in lowercase hexadecimal, so the output is portable ASCII.
///   Emitting the raw UTF-8 bytes (as the printers used to) is read back by
///   Z3 as one character *per byte*, so `é` came back as two characters.
///
/// The escape forms recognised on input are `\ud₃d₂d₁d₀` (exactly four hex
/// digits) and `\u{d...}` (one to five hex digits); both denote a code point
/// in `0..=0x2FFFF`.  The `\u{...}` form emitted here is therefore always
/// read back as the single character it denotes, for every `char` up to the
/// lexer's `MAX_STRING_CODE_POINT`.
///
/// Rust's `char` excludes the UTF-16 surrogate range `0xD800..=0xDFFF` by
/// construction, so this function can never emit a surrogate escape — which
/// is exactly the range the lexer rejects as unrepresentable, so the two
/// layers agree on the alphabet.
///
/// Reference: Z3's `zstring.cpp` (`zstring::encode`) and `ast_smt2_pp.cpp`
/// (`smt2_pp_environment::pp_string_literal`), which apply these two steps in
/// the same order.
///
/// # Limitation
///
/// A `char` above `MAX_STRING_CODE_POINT` (`0x2FFFF`) is outside the alphabet
/// of the Unicode Strings theory and has *no* literal spelling: a `\u{...}`
/// escape denoting a larger value is not an escape sequence at all, so no
/// reader — Z3's or OxiZ's — would read it back as one character.  Such a
/// value can only arise from a raw supplementary-plane character in the
/// input; it is still written as `\u{...}` here, since there is no
/// alternative encoding and the escape at least names the code point.
pub fn format_string_literal(value: &str) -> String {
    use core::fmt::Write as _;

    // Most values are plain ASCII, so the output is usually the input plus
    // the two quotes.
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => out.push_str("\"\""),
            // A backslash only needs escaping when it would otherwise start a
            // `\u` sequence.  The character after it is emitted either raw (so
            // a literal `u` really would follow the backslash) or as a
            // `\u{...}`/`""` escape, both of which begin with a character
            // other than `u`; so peeking at the *value* is enough.
            '\\' if chars.peek() == Some(&'u') => out.push_str("\\u{5c}"),
            ' '..='~' => out.push(c),
            _ => {
                // Infallible: writing into a `String` never fails.
                let _ = write!(out, "\\u{{{:x}}}", c as u32);
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod string_literal_tests {
    use super::format_string_literal;
    use crate::smtlib::{Lexer, TokenKind};

    /// Largest code point in the alphabet of the SMT-LIB 2.6 Unicode Strings
    /// theory; kept in step with the lexer's constant of the same name.
    const MAX_CODE_POINT: u32 = 0x2_FFFF;

    /// Decode an SMT-LIB string literal with the real lexer, panicking on any
    /// lexical error — the printer's output must always be clean input.
    fn decode(literal: &str) -> String {
        let mut lexer = Lexer::new(literal);
        let token = lexer
            .next_token()
            .expect("a string literal must lex to a token");
        assert!(
            !lexer.has_errors(),
            "the printer emitted {literal:?}, which the lexer rejects: {:?}",
            lexer.errors()
        );
        match token.kind {
            TokenKind::StringLit(s) => s,
            other => panic!("{literal:?} lexed as {other:?}, not a string literal"),
        }
    }

    /// The fixed table of tricky values: `(value, expected literal)`.
    ///
    /// Every expected literal was cross-checked against z3 4.15 by feeding it
    /// back to z3 and comparing `str.len` and equality with the original.
    /// Deterministic by construction — no RNG, no wall-clock.
    const CASES: &[(&str, &str)] = &[
        // Empty and plain ASCII: no escaping whatsoever.
        ("", r#""""#),
        ("hello world", r#""hello world""#),
        // The doubled quote is the only escape the core language defines.
        ("\"", r#""""""#),
        ("a\"b", r#""a""b""#),
        ("\"\"", r#""""""""#),
        // A backslash stands for itself.
        ("\\", r#""\""#),
        ("a\\b", r#""a\b""#),
        ("back\\slash", r#""back\slash""#),
        // ... unless it would start a `\u` escape on the way back in.
        ("\\u0041", r#""\u{5c}u0041""#),
        ("\\u{41}", r#""\u{5c}u{41}""#),
        // Only the backslash immediately before the `u` needs escaping.
        ("\\\\u", r#""\\u{5c}u""#),
        // A backslash before a character that itself escapes stays verbatim,
        // because the escape it is followed by does not begin with `u`.
        ("\\\u{e9}", r#""\\u{e9}""#),
        // Outside printable ASCII: `\u{...}`, lowercase, unpadded.
        ("\u{e9}", r#""\u{e9}""#),
        ("\u{1f600}", r#""\u{1f600}""#),
        ("\n\t\r", r#""\u{a}\u{9}\u{d}""#),
        // Every boundary of the printable-ASCII window and of the alphabet.
        ("\u{0}", r#""\u{0}""#),
        ("\u{1f}", r#""\u{1f}""#),
        (" ", r#"" ""#),
        ("~", r#""~""#),
        ("\u{7f}", r#""\u{7f}""#),
        ("\u{80}", r#""\u{80}""#),
        ("\u{ffff}", r#""\u{ffff}""#),
        ("\u{2ffff}", r#""\u{2ffff}""#),
        // Everything at once.
        ("x\u{e9}\"y\\z", r#""x\u{e9}""y\z""#),
    ];

    /// Each tricky value encodes to exactly the literal z3 would print.
    #[test]
    fn test_string_literal_encoding_table() {
        for (value, expected) in CASES {
            assert_eq!(
                &format_string_literal(value),
                expected,
                "{value:?} encoded wrongly"
            );
        }
    }

    /// Property-style round trip over the same fixed table: whatever the
    /// printer writes, the lexer must read back as the identical value.
    #[test]
    fn test_string_literal_round_trips() {
        for (value, _) in CASES {
            let literal = format_string_literal(value);
            assert_eq!(
                &decode(&literal),
                value,
                "{value:?} printed as {literal} did not round-trip"
            );
            // The literal must also be portable ASCII, so a reader that works
            // in bytes (as z3 does) sees exactly these characters.
            assert!(
                literal.is_ascii(),
                "{value:?} printed as non-ASCII text {literal}"
            );
        }
    }

    /// Spec rule 1: `"` is written `""`, and that is the *only* in-literal
    /// escape the core language defines. It is never written `\"` — a
    /// backslash before a quote would end the literal, not escape it.
    #[test]
    fn test_string_literal_quote_is_doubled_never_backslashed() {
        assert_eq!(format_string_literal("\""), r#""""""#);
        assert_eq!(format_string_literal("a\"b"), r#""a""b""#);
        for (value, _) in CASES {
            let literal = format_string_literal(value);
            // Inside the enclosing quotes, every `"` must come in a pair.
            let body = &literal[1..literal.len() - 1];
            let mut rest = body;
            while let Some(at) = rest.find('"') {
                assert!(
                    rest[at + 1..].starts_with('"'),
                    "{value:?} printed a lone quote, which ends the literal: {literal}"
                );
                rest = &rest[at + 2..];
            }
            // ... and the number of quotes inside is exactly twice the number
            // of quote characters in the value.
            assert_eq!(
                body.matches('"').count(),
                2 * value.matches('"').count(),
                "{value:?} printed the wrong number of quotes: {literal}"
            );
        }
    }

    /// Spec rule 2: `\` is not an escape introducer, so it is emitted
    /// verbatim and never doubled.
    #[test]
    fn test_string_literal_backslash_is_verbatim() {
        assert_eq!(format_string_literal("\\"), r#""\""#);
        assert_eq!(format_string_literal("a\\b"), r#""a\b""#);
        // A single backslash must survive as a single backslash.
        assert_eq!(decode(&format_string_literal("\\")), "\\");
        // `\\` is not an escape either: two backslashes stay two.
        assert_eq!(format_string_literal("\\\\"), r#""\\""#);
        assert_eq!(decode(&format_string_literal("\\\\")), "\\\\");
    }

    /// Spec rule 3: the `\u` forms *are* recognised on input, so a backslash
    /// that would begin one is escaped as `\u{5c}` — and only then.
    #[test]
    fn test_string_literal_backslash_before_u_is_escaped() {
        // The four-digit and the braced form, as literal text.
        assert_eq!(format_string_literal("\\u0041"), r#""\u{5c}u0041""#);
        assert_eq!(format_string_literal("\\u{41}"), r#""\u{5c}u{41}""#);
        assert_eq!(decode(&format_string_literal("\\u0041")), "\\u0041");
        assert_eq!(decode(&format_string_literal("\\u{41}")), "\\u{41}");
        // Not even a *malformed* `\u` may go out raw: the reader's lookahead
        // does not know it is malformed until it has consumed the backslash.
        assert_eq!(format_string_literal("\\uZZZZ"), r#""\u{5c}uZZZZ""#);
        // A backslash followed by anything else needs no treatment.
        for follower in ['a', 'U', 'v', '{', '\\', '0'] {
            let value = format!("\\{follower}");
            assert_eq!(
                format_string_literal(&value),
                format!("\"\\{follower}\""),
                "\\{follower} must keep its backslash verbatim"
            );
        }
        // A trailing backslash has nothing after it, so it stays verbatim.
        assert_eq!(format_string_literal("ab\\"), r#""ab\""#);
    }

    /// Spec rule 4, at every boundary: printable ASCII is `0x20..=0x7E`;
    /// everything below or above it is escaped.
    #[test]
    fn test_string_literal_boundary_code_points() {
        let boundaries: &[(u32, &str)] = &[
            (0x00, r#""\u{0}""#),
            (0x1f, r#""\u{1f}""#),
            (0x20, r#"" ""#),
            (0x7e, r#""~""#),
            (0x7f, r#""\u{7f}""#),
            (0x80, r#""\u{80}""#),
            (0xff, r#""\u{ff}""#),
            (0xffff, r#""\u{ffff}""#),
            (0x1_0000, r#""\u{10000}""#),
            (MAX_CODE_POINT, r#""\u{2ffff}""#),
        ];
        for (code_point, expected) in boundaries {
            let c = char::from_u32(*code_point).expect("boundary is a valid char");
            let value = c.to_string();
            assert_eq!(
                &format_string_literal(&value),
                expected,
                "U+{code_point:04X} encoded wrongly"
            );
            assert_eq!(decode(expected), value, "U+{code_point:04X} did not decode");
        }
    }

    /// Exhaustive sweep of the whole theory alphabet: every code point the
    /// SMT-LIB Unicode Strings theory admits and a Rust `char` can hold must
    /// survive a print/parse round trip, on its own and after a backslash.
    #[test]
    fn test_string_literal_every_code_point_in_the_alphabet_round_trips() {
        for code_point in 0..=MAX_CODE_POINT {
            // `None` exactly on the surrogate range, which a Rust `char`
            // cannot hold and the lexer rejects; the two layers agree.
            let Some(c) = char::from_u32(code_point) else {
                assert!(
                    (0xd800..=0xdfff).contains(&code_point),
                    "U+{code_point:04X} is not a char for an unexpected reason"
                );
                continue;
            };
            let value = c.to_string();
            let literal = format_string_literal(&value);
            assert!(
                literal.is_ascii(),
                "U+{code_point:04X} printed non-ASCII text"
            );
            assert_eq!(
                decode(&literal),
                value,
                "U+{code_point:04X} printed as {literal} did not round-trip"
            );
        }
    }

    /// The lookahead that decides whether a backslash needs escaping must be
    /// right for *every* possible follower, not just `u`.
    #[test]
    fn test_string_literal_backslash_followed_by_any_ascii_round_trips() {
        for code_point in 0..0x80u32 {
            let c = char::from_u32(code_point).expect("ASCII is a valid char");
            let value = format!("\\{c}");
            let literal = format_string_literal(&value);
            assert_eq!(
                decode(&literal),
                value,
                "\\U+{code_point:04X} printed as {literal} did not round-trip"
            );
        }
    }

    /// The surrogate range is unrepresentable in a Rust `char`, so the
    /// printer can never emit an escape the lexer would reject.
    #[test]
    fn test_string_literal_never_emits_a_surrogate_escape() {
        for surrogate in 0xd800..=0xdfffu32 {
            assert!(
                char::from_u32(surrogate).is_none(),
                "U+{surrogate:04X} must not be constructible as a char"
            );
        }
        for (value, _) in CASES {
            let literal = format_string_literal(value);
            for surrogate in [0xd800u32, 0xdabc, 0xdfff] {
                let escape = format!("\\u{{{surrogate:x}}}");
                assert!(
                    !literal.contains(&escape),
                    "{value:?} printed a surrogate escape: {literal}"
                );
            }
        }
    }

    /// **Control**: ordinary text must print completely unchanged. Escaping
    /// a character that needs no escaping would be silently ugly in every
    /// model OxiZ prints, so pin it: every printable ASCII character except
    /// the quote goes out as itself.
    #[test]
    fn test_control_plain_ascii_prints_unchanged() {
        for text in [
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "0123456789",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "!#$%&'()*+,-./:;<=>?@[]^_`{|}~",
            "a b\tc", // only the tab is outside the printable window
        ] {
            let literal = format_string_literal(text);
            if text.contains('\t') {
                assert_eq!(literal, "\"a b\\u{9}c\"");
                continue;
            }
            assert_eq!(
                literal,
                format!("\"{text}\""),
                "{text:?} must print unchanged inside its quotes"
            );
            assert!(
                !literal[1..literal.len() - 1].contains('\\'),
                "{text:?} gained a gratuitous escape: {literal}"
            );
        }
        // Character by character: `"` doubles, everything else in
        // `0x20..=0x7E` is emitted as itself.
        for code_point in 0x20..=0x7eu32 {
            let c = char::from_u32(code_point).expect("printable ASCII is a char");
            let expected = if c == '"' {
                r#""""""#.to_string()
            } else {
                format!("\"{c}\"")
            };
            assert_eq!(
                format_string_literal(&c.to_string()),
                expected,
                "U+{code_point:04X} must not be escaped"
            );
        }
    }

    /// Documented limitation: a code point above the theory's alphabet has no
    /// literal spelling at all. `\u{...}` with six hex digits is *not* an
    /// escape sequence for any reader, so neither z3 nor OxiZ reads it back
    /// as one character. Pinned so the gap stays visible rather than
    /// pretending it round-trips.
    #[test]
    fn test_code_point_above_the_alphabet_has_no_literal_spelling() {
        let out_of_alphabet = char::from_u32(MAX_CODE_POINT + 1).expect("U+30000 is a char");
        let value = out_of_alphabet.to_string();
        let literal = format_string_literal(&value);
        assert_eq!(literal, r#""\u{30000}""#);
        // The escape names the code point, but a reader takes the backslash
        // literally, so this is the one case that does not round-trip.
        assert_ne!(decode(&literal), value);
        assert_eq!(decode(&literal), "\\u{30000}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{TermId, TermKind, TermManager, model::Model};
    use config::PrettyConfig;
    use pretty::PrettyPrinter;

    /// All three renderers of a string value — the single-line printer, the
    /// pretty printer, and `model::Value`'s `Display` — must produce the same
    /// literal, because all three route through `format_string_literal`.
    /// Three separate copies of the escape rules is exactly how this defect
    /// survived for so long.
    #[test]
    fn test_all_string_renderers_agree() {
        for value in [
            "",
            "hello",
            "a\"b",
            "a\\b",
            "\\u0041",
            "\u{e9}",
            "\u{0}",
            "\u{2ffff}",
        ] {
            let mut manager = TermManager::new();
            let term = manager.mk_string_lit(value);
            let basic = Printer::new(&manager).print_term(term);
            let pretty = PrettyPrinter::new(&manager).print_term(term);
            let display = crate::model::Value::String(value.to_string()).to_string();
            assert_eq!(basic, pretty, "printers disagree on {value:?}");
            assert_eq!(basic, display, "Value::Display disagrees on {value:?}");
            assert_eq!(basic, format_string_literal(value));
        }
    }

    /// A string literal nested inside a term still prints as a well-formed
    /// literal, and the surrounding term is unaffected.
    #[test]
    fn test_string_literal_inside_a_term() {
        let mut manager = TermManager::new();
        let lit = manager.mk_string_lit("a\"b\\u0041\u{e9}");
        let len = manager.mk_str_len(lit);
        let printed = Printer::new(&manager).print_term(len);
        assert_eq!(printed, r#"(str.len "a""b\u{5c}u0041\u{e9}")"#);
    }

    #[test]
    fn test_print_constants() {
        let manager = TermManager::new();
        let printer = Printer::new(&manager);

        assert_eq!(printer.print_term(manager.mk_true()), "true");
        assert_eq!(printer.print_term(manager.mk_false()), "false");
    }

    #[test]
    fn test_print_compound() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let printer = Printer::new(&manager);
        assert_eq!(printer.print_term(and), "(and x y)");
    }

    /// Issue #17: a bit-vector constant must print as a well-formed literal of
    /// its declared width.  The printers used to emit the value in *decimal*
    /// behind a `#x` prefix, so `255` came out as `#x255` and a negative value
    /// as `#x-1`, and they used `width.div_ceil(4)` hex digits, silently
    /// widening any width that is not a multiple of four.
    #[test]
    fn test_issue_17_bitvec_literal_printing() {
        let mut manager = TermManager::new();
        let printer_of = |manager: &TermManager, term| Printer::new(manager).print_term(term);

        // Hex form: exactly width/4 digits, hexadecimal, zero padded.
        let cases: [(i64, u32, &str); 6] = [
            (0, 8, "#x00"),
            (5, 8, "#x05"),
            (255, 8, "#xff"),
            (0xdead, 16, "#xdead"),
            (1, 32, "#x00000001"),
            (0xf, 4, "#xf"),
        ];
        for (value, width, expected) in cases {
            let term = manager.mk_bitvec(value, width);
            assert_eq!(printer_of(&manager, term), expected);
        }

        // Widths that are not a multiple of four have no `#x` form; they must
        // print in binary with exactly `width` digits.
        let five_bit = manager.mk_bitvec(5i64, 5);
        assert_eq!(printer_of(&manager, five_bit), "#b00101");

        // Out-of-range values wrap into `[0, 2^width)` instead of printing a
        // sign or overflowing the digit count.
        let negative = manager.mk_bitvec(-1i64, 8);
        assert_eq!(printer_of(&manager, negative), "#xff");
        let oversized = manager.mk_bitvec(256i64, 8);
        assert_eq!(printer_of(&manager, oversized), "#x00");
    }

    /// Issue #17: comparisons that hold (or fail) for every assignment are
    /// folded to a boolean constant at construction time.
    #[test]
    fn test_issue_17_bv_comparison_folding() {
        let mut manager = TermManager::new();
        let width = 8;
        let bv_sort = manager.sorts.bitvec(width);
        let x = manager.mk_var("x", bv_sort);
        let zero = manager.mk_bitvec(0i64, width);
        let max_unsigned = manager.mk_bitvec(255i64, width);
        let min_signed = manager.mk_bitvec(128i64, width);
        let max_signed = manager.mk_bitvec(127i64, width);
        let (t, f) = (manager.mk_true(), manager.mk_false());

        // Always false.
        assert_eq!(manager.mk_bv_ult(x, zero), f);
        assert_eq!(manager.mk_bv_ult(max_unsigned, x), f);
        assert_eq!(manager.mk_bv_slt(x, min_signed), f);
        assert_eq!(manager.mk_bv_slt(max_signed, x), f);
        assert_eq!(manager.mk_bv_ult(x, x), f);
        assert_eq!(manager.mk_bv_slt(x, x), f);

        // Always true.
        assert_eq!(manager.mk_bv_ule(zero, x), t);
        assert_eq!(manager.mk_bv_ule(x, max_unsigned), t);
        assert_eq!(manager.mk_bv_sle(min_signed, x), t);
        assert_eq!(manager.mk_bv_sle(x, max_signed), t);
        assert_eq!(manager.mk_bv_ule(x, x), t);
        assert_eq!(manager.mk_bv_sle(x, x), t);

        // Literal-vs-literal is evaluated directly, in the right order.
        assert_eq!(manager.mk_bv_ult(zero, max_unsigned), t);
        assert_eq!(manager.mk_bv_slt(min_signed, max_signed), t);
        assert_eq!(manager.mk_bv_slt(max_signed, min_signed), f);

        // Genuinely contingent atoms are left alone.
        let contingent = manager.mk_bv_ule(x, zero);
        assert_ne!(contingent, t);
        assert_ne!(contingent, f);
    }

    // ────────────────────────────────────────────────────────────────────
    // Structural bit-vector constant folding.
    //
    // The comparison folding above only fires once the bound has actually
    // become a literal, so `bvadd`/`bvand`/`bvshl`/... over literal operands
    // must evaluate at construction time too.  Every expected value in this
    // section was cross-checked against z3 4.15.
    // ────────────────────────────────────────────────────────────────────

    /// Build the `width`-bit literal `value` and return its printed form —
    /// the folding assertions below check the *literal* a fold produced, so
    /// they also pin the `#x` (width divisible by four) versus `#b` radix.
    fn literal_of(manager: &mut TermManager, value: u64, width: u32) -> String {
        let term = manager.mk_bitvec(value, width);
        Printer::new(manager).print_term(term)
    }

    /// Assert `folded` is exactly the `width`-bit literal `expected`, and
    /// that it prints as a well-formed literal of that width.
    fn assert_folds_to(manager: &mut TermManager, folded: TermId, expected: u64, width: u32) {
        let expected_term = manager.mk_bitvec(expected, width);
        assert_eq!(
            folded,
            expected_term,
            "expected the fold to yield {}, got {}",
            literal_of(manager, expected, width),
            Printer::new(manager).print_term(folded)
        );
        let printed = Printer::new(manager).print_term(folded);
        let expected_radix = if width.is_multiple_of(4) { "#x" } else { "#b" };
        assert!(
            printed.starts_with(expected_radix),
            "a {width}-bit literal must print with the {expected_radix} radix, got {printed}"
        );
    }

    /// One folded operation per row, at five widths — including width 5,
    /// which is not a multiple of four and therefore prints in binary.
    #[test]
    fn test_bv_structural_folding_per_operation() {
        // (width, lhs, rhs, in-range shift distance)
        let cases: [(u32, u64, u64, u64); 5] = [
            (4, 0xd, 0x5, 2),
            (5, 0b10110, 0b00101, 2),
            (8, 0xa7, 0x0d, 3),
            (16, 0xbeef, 0x0123, 4),
            (32, 0xdead_beef, 0x0000_f00d, 8),
        ];
        // Expected results, in the order the operations are applied below.
        let expected: [[u64; 16]; 5] = [
            // width 4: a = 0xd (-3 signed), b = 0x5, shift 2
            [
                0x2, 0x8, 0x1, 0x5, 0xd, 0x8, 0x2, 0x3, 0x0, 0xd, 0x2, 0x4, 0x3, 0xf, 0x2, 0x3,
            ],
            // width 5: a = 0b10110 (-10 signed), b = 0b00101, shift 2
            [
                0b11011, 0b10001, 0b01110, 0b00100, 0b10111, 0b10011, 0b00100, 0b00010, 0b11110,
                0b00000, 0b00000, 0b11000, 0b00101, 0b11101, 0b01001, 0b01010,
            ],
            // width 8: a = 0xa7 (-89 signed), b = 0x0d, shift 3
            [
                0xb4, 0x9a, 0x7b, 0x05, 0xaf, 0xaa, 0x0c, 0x0b, 0xfa, 0xf5, 0x02, 0x38, 0x14, 0xf4,
                0x58, 0x59,
            ],
            // width 16: a = 0xbeef (-16657 signed), b = 0x0123, shift 4
            [
                0xc012, 0xbdcc, 0x09ad, 0x0023, 0xbfef, 0xbfcc, 0x00a7, 0x011a, 0xffc7, 0xffba,
                0x00dd, 0xeef0, 0x0bee, 0xfbee, 0x4110, 0x4111,
            ],
            // width 32: a = 0xdeadbeef, b = 0x0000f00d, shift 8
            [
                0xdeae_aefc,
                0xdeac_cee2,
                0x31d2_c223,
                0x0000_b00d,
                0xdead_feef,
                0xdead_4ee2,
                0x0000_ed79,
                0x0000_3fca,
                0xffff_dc77,
                0xffff_fce4,
                0x0000_ecf1,
                0xadbe_ef00,
                0x00de_adbe,
                0xffde_adbe,
                0x2152_4110,
                0x2152_4111,
            ],
        ];

        for (case, want) in cases.iter().zip(expected.iter()) {
            let (width, lhs_value, rhs_value, shift_value) = *case;
            let mut manager = TermManager::new();
            let lhs = manager.mk_bitvec(lhs_value, width);
            let rhs = manager.mk_bitvec(rhs_value, width);
            let shift = manager.mk_bitvec(shift_value, width);

            let folded = [
                manager.mk_bv_add(lhs, rhs),
                manager.mk_bv_sub(lhs, rhs),
                manager.mk_bv_mul(lhs, rhs),
                manager.mk_bv_and(lhs, rhs),
                manager.mk_bv_or(lhs, rhs),
                manager.mk_bv_xor(lhs, rhs),
                manager.mk_bv_udiv(lhs, rhs),
                manager.mk_bv_urem(lhs, rhs),
                manager.mk_bv_sdiv(lhs, rhs),
                manager.mk_bv_srem(lhs, rhs),
                manager.mk_bv_smod(lhs, rhs),
                manager.mk_bv_shl(lhs, shift),
                manager.mk_bv_lshr(lhs, shift),
                manager.mk_bv_ashr(lhs, shift),
                manager.mk_bv_not(lhs),
                manager.mk_bv_neg(lhs),
            ];
            for (folded, expected) in folded.iter().zip(want.iter()) {
                assert_folds_to(&mut manager, *folded, *expected, width);
            }

            // concat doubles the width; extract takes the top half.
            let concat = manager
                .try_mk_bv_concat(lhs, rhs)
                .expect("two bit-vector literals concat cleanly");
            assert_folds_to(
                &mut manager,
                concat,
                (lhs_value << width) | rhs_value,
                width * 2,
            );
            let high_half = width - width / 2;
            let extract = manager.mk_bv_extract(width - 1, width / 2, lhs);
            assert_folds_to(&mut manager, extract, lhs_value >> (width / 2), high_half);
        }
    }

    /// Division and remainder are **total** in SMT-LIB: every one of the five
    /// operations has a specified value at a zero divisor.  Cross-checked
    /// against z3 4.15: `bvudiv` gives all ones, `bvurem`/`bvsrem`/`bvsmod`
    /// give the dividend, and `bvsdiv` gives `-1` for a non-negative dividend
    /// and `1` for a negative one.
    #[test]
    fn test_bv_division_by_zero_is_total() {
        for width in [4u32, 5, 8, 16, 32] {
            let mut manager = TermManager::new();
            let all_ones = (1u64 << width) - 1;
            let zero = manager.mk_bitvec(0u64, width);
            // A non-negative and a negative dividend of this width.
            let positive_value = 7u64 % (1u64 << (width - 1));
            let negative_value = all_ones - positive_value + 1; // -positive
            let positive = manager.mk_bitvec(positive_value, width);
            let negative = manager.mk_bitvec(negative_value, width);

            for (dividend, dividend_value) in
                [(positive, positive_value), (negative, negative_value)]
            {
                let udiv = manager.mk_bv_udiv(dividend, zero);
                assert_folds_to(&mut manager, udiv, all_ones, width);

                let urem = manager.mk_bv_urem(dividend, zero);
                assert_folds_to(&mut manager, urem, dividend_value, width);

                let srem = manager.mk_bv_srem(dividend, zero);
                assert_folds_to(&mut manager, srem, dividend_value, width);

                let smod = manager.mk_bv_smod(dividend, zero);
                assert_folds_to(&mut manager, smod, dividend_value, width);
            }

            // bvsdiv is the only one whose zero-divisor value depends on the
            // dividend's sign.
            let sdiv_positive = manager.mk_bv_sdiv(positive, zero);
            assert_folds_to(&mut manager, sdiv_positive, all_ones, width);
            let sdiv_zero = manager.mk_bv_sdiv(zero, zero);
            assert_folds_to(&mut manager, sdiv_zero, all_ones, width);
            let sdiv_negative = manager.mk_bv_sdiv(negative, zero);
            assert_folds_to(&mut manager, sdiv_negative, 1, width);
        }
    }

    /// Shift distances of `0`, `1`, `width - 1`, `width` and beyond.
    /// `bvshl`/`bvlshr` saturate to zero at or past the width; `bvashr`
    /// saturates to a copy of the sign bit.
    #[test]
    fn test_bv_shift_distances_at_and_beyond_the_width() {
        for width in [4u32, 5, 8, 16, 32] {
            let mut manager = TermManager::new();
            let all_ones = (1u64 << width) - 1;
            let sign_bit = 1u64 << (width - 1);
            let ones = manager.mk_bitvec(all_ones, width);
            let one = manager.mk_bitvec(1u64, width);
            let negative = manager.mk_bitvec(sign_bit, width);

            for distance_value in [0u64, 1, u64::from(width) - 1] {
                let distance = manager.mk_bitvec(distance_value, width);
                let shift = u32::try_from(distance_value).expect("distance fits");
                let shl = manager.mk_bv_shl(one, distance);
                assert_folds_to(&mut manager, shl, 1u64 << shift, width);
                let lshr = manager.mk_bv_lshr(ones, distance);
                assert_folds_to(&mut manager, lshr, all_ones >> shift, width);
                // All ones stays all ones under an arithmetic shift.
                let ashr = manager.mk_bv_ashr(ones, distance);
                assert_folds_to(&mut manager, ashr, all_ones, width);
            }

            // At and beyond the width.  `width + 3` and the all-ones distance
            // both exceed it; for width 4 the all-ones distance is 15.
            for distance_value in [u64::from(width), u64::from(width) + 3, all_ones] {
                if distance_value >= (1u64 << width) {
                    continue;
                }
                let distance = manager.mk_bitvec(distance_value, width);
                let shl = manager.mk_bv_shl(ones, distance);
                assert_folds_to(&mut manager, shl, 0, width);
                let lshr = manager.mk_bv_lshr(ones, distance);
                assert_folds_to(&mut manager, lshr, 0, width);
                // Arithmetic shift keeps the sign: all ones stays all ones,
                // a non-negative value collapses to zero.
                let ashr_negative = manager.mk_bv_ashr(negative, distance);
                assert_folds_to(&mut manager, ashr_negative, all_ones, width);
                let ashr_positive = manager.mk_bv_ashr(one, distance);
                assert_folds_to(&mut manager, ashr_positive, 0, width);
            }
        }
    }

    /// The algebraic identities, all of which hold for *symbolic* `t` and so
    /// must fire even when only one operand is a literal.
    #[test]
    fn test_bv_algebraic_identities() {
        for width in [4u32, 5, 8, 16, 32] {
            let mut manager = TermManager::new();
            let bv_sort = manager.sorts.bitvec(width);
            let t = manager.mk_var("t", bv_sort);
            let all_ones_value = (1u64 << width) - 1;
            let zero = manager.mk_bitvec(0u64, width);
            let one = manager.mk_bitvec(1u64, width);
            let all_ones = manager.mk_bitvec(all_ones_value, width);

            // t - t -> 0, t ^ t -> 0.
            assert_eq!(manager.mk_bv_sub(t, t), zero, "t - t must fold to 0");
            assert_eq!(manager.mk_bv_xor(t, t), zero, "t ^ t must fold to 0");
            // t & t -> t, t | t -> t.
            assert_eq!(manager.mk_bv_and(t, t), t, "t & t must fold to t");
            assert_eq!(manager.mk_bv_or(t, t), t, "t | t must fold to t");
            // Absorbing constants.
            assert_eq!(manager.mk_bv_and(t, zero), zero, "t & 0 must fold to 0");
            assert_eq!(manager.mk_bv_and(zero, t), zero, "0 & t must fold to 0");
            assert_eq!(
                manager.mk_bv_or(t, all_ones),
                all_ones,
                "t | all-ones must fold to all-ones"
            );
            assert_eq!(
                manager.mk_bv_or(all_ones, t),
                all_ones,
                "all-ones | t must fold to all-ones"
            );
            assert_eq!(manager.mk_bv_mul(t, zero), zero, "t * 0 must fold to 0");
            assert_eq!(manager.mk_bv_mul(zero, t), zero, "0 * t must fold to 0");
            // Neutral constants.
            assert_eq!(manager.mk_bv_add(t, zero), t, "t + 0 must fold to t");
            assert_eq!(manager.mk_bv_add(zero, t), t, "0 + t must fold to t");
            assert_eq!(manager.mk_bv_sub(t, zero), t, "t - 0 must fold to t");
            assert_eq!(manager.mk_bv_mul(t, one), t, "t * 1 must fold to t");
            assert_eq!(manager.mk_bv_mul(one, t), t, "1 * t must fold to t");
            assert_eq!(
                manager.mk_bv_and(t, all_ones),
                t,
                "t & all-ones must fold to t"
            );
            assert_eq!(manager.mk_bv_or(t, zero), t, "t | 0 must fold to t");
            assert_eq!(manager.mk_bv_xor(t, zero), t, "t ^ 0 must fold to t");
            // Shifting by zero is the identity for all three shifts.
            assert_eq!(manager.mk_bv_shl(t, zero), t, "t << 0 must fold to t");
            assert_eq!(manager.mk_bv_lshr(t, zero), t, "t >>u 0 must fold to t");
            assert_eq!(manager.mk_bv_ashr(t, zero), t, "t >>s 0 must fold to t");
            // Complement is an involution.
            let not_t = manager.mk_bv_not(t);
            assert_eq!(
                manager.mk_bv_not(not_t),
                t,
                "bvnot (bvnot t) must fold to t"
            );
            // A literal shift distance of at least the width empties the
            // vector regardless of the (symbolic) value.
            let over_wide = manager.mk_bitvec(u64::from(width), width);
            if u64::from(width) < (1u64 << width) {
                assert_eq!(
                    manager.mk_bv_shl(t, over_wide),
                    zero,
                    "t << width must fold to 0"
                );
                assert_eq!(
                    manager.mk_bv_lshr(t, over_wide),
                    zero,
                    "t >>u width must fold to 0"
                );
            }
        }
    }

    /// Control: folding must not fire on symbolic operands.  The comparator
    /// circuits and the bit-blaster are only exercised while these terms
    /// survive as real structural nodes, so each one must still be the
    /// operation that was requested.
    #[test]
    fn test_bv_structural_folding_leaves_symbolic_terms_alone() {
        let mut manager = TermManager::new();
        let width = 8;
        let bv_sort = manager.sorts.bitvec(width);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let three = manager.mk_bitvec(3u64, width);

        macro_rules! assert_kind {
            ($term:expr, $pattern:pat, $label:literal) => {{
                let term = $term;
                let kind = &manager.get(term).expect("term should exist").kind;
                assert!(
                    matches!(kind, $pattern),
                    "{} over symbolic operands must not fold, got {kind:?}",
                    $label
                );
            }};
        }

        // Two symbolic operands: nothing is decidable.
        assert_kind!(manager.mk_bv_add(x, y), TermKind::BvAdd(..), "bvadd");
        assert_kind!(manager.mk_bv_sub(x, y), TermKind::BvSub(..), "bvsub");
        assert_kind!(manager.mk_bv_mul(x, y), TermKind::BvMul(..), "bvmul");
        assert_kind!(manager.mk_bv_and(x, y), TermKind::BvAnd(..), "bvand");
        assert_kind!(manager.mk_bv_or(x, y), TermKind::BvOr(..), "bvor");
        assert_kind!(manager.mk_bv_xor(x, y), TermKind::BvXor(..), "bvxor");
        assert_kind!(manager.mk_bv_udiv(x, y), TermKind::BvUdiv(..), "bvudiv");
        assert_kind!(manager.mk_bv_urem(x, y), TermKind::BvUrem(..), "bvurem");
        assert_kind!(manager.mk_bv_sdiv(x, y), TermKind::BvSdiv(..), "bvsdiv");
        assert_kind!(manager.mk_bv_srem(x, y), TermKind::BvSrem(..), "bvsrem");
        assert_kind!(manager.mk_bv_shl(x, y), TermKind::BvShl(..), "bvshl");
        assert_kind!(manager.mk_bv_lshr(x, y), TermKind::BvLshr(..), "bvlshr");
        assert_kind!(manager.mk_bv_ashr(x, y), TermKind::BvAshr(..), "bvashr");
        assert_kind!(manager.mk_bv_not(x), TermKind::BvNot(..), "bvnot");
        assert_kind!(
            manager
                .try_mk_bv_concat(x, y)
                .expect("bit-vector operands concat cleanly"),
            TermKind::BvConcat(..),
            "concat"
        );
        assert_kind!(
            manager.mk_bv_extract(5, 2, x),
            TermKind::BvExtract { .. },
            "extract"
        );

        // One literal operand that is not an identity element: still no fold.
        assert_kind!(manager.mk_bv_add(x, three), TermKind::BvAdd(..), "bvadd 3");
        assert_kind!(manager.mk_bv_mul(x, three), TermKind::BvMul(..), "bvmul 3");
        assert_kind!(manager.mk_bv_and(x, three), TermKind::BvAnd(..), "bvand 3");
        assert_kind!(manager.mk_bv_or(x, three), TermKind::BvOr(..), "bvor 3");
        assert_kind!(manager.mk_bv_xor(x, three), TermKind::BvXor(..), "bvxor 3");
        assert_kind!(
            manager.mk_bv_udiv(x, three),
            TermKind::BvUdiv(..),
            "bvudiv 3"
        );
        assert_kind!(manager.mk_bv_shl(x, three), TermKind::BvShl(..), "bvshl 3");
        assert_kind!(
            manager.mk_bv_ashr(x, three),
            TermKind::BvAshr(..),
            "bvashr 3"
        );
        // `bvashr` by an over-wide distance depends on the sign bit, so it is
        // *not* foldable while the value is symbolic (unlike shl/lshr).
        let over_wide = manager.mk_bitvec(u64::from(width), width);
        assert_kind!(
            manager.mk_bv_ashr(x, over_wide),
            TermKind::BvAshr(..),
            "bvashr width"
        );

        // The comparison atoms stay contingent over symbolic operands, so the
        // comparator circuits keep their coverage.
        let (t, f) = (manager.mk_true(), manager.mk_false());
        for atom in [
            manager.mk_bv_ult(x, y),
            manager.mk_bv_ule(x, y),
            manager.mk_bv_slt(x, y),
            manager.mk_bv_sle(x, y),
        ] {
            assert_ne!(atom, t);
            assert_ne!(atom, f);
        }
    }

    #[test]
    fn test_roundtrip() {
        let mut manager = TermManager::new();
        let input = "(and (or x y) (not z))";
        let term =
            crate::smtlib::parse_term(input, &mut manager).expect("test operation should succeed");

        let printer = Printer::new(&manager);
        let output = printer.print_term(term);

        // Note: Output might differ slightly due to canonicalization
        assert!(output.contains("and"));
        assert!(output.contains("or"));
        assert!(output.contains("not"));
    }

    // ==================== PrettyPrinter Tests ====================

    #[test]
    fn test_pretty_config_default() {
        let config = PrettyConfig::default();
        assert_eq!(config.indent_width, 2);
        assert_eq!(config.max_width, 80);
        assert!(!config.use_tabs);
        assert!(!config.print_sorts);
        assert_eq!(config.break_depth, 2);
    }

    #[test]
    fn test_pretty_config_compact() {
        let config = PrettyConfig::compact();
        assert_eq!(config.indent_width, 0);
        assert_eq!(config.max_width, usize::MAX);
        assert_eq!(config.break_depth, usize::MAX);
    }

    #[test]
    fn test_pretty_config_expanded() {
        let config = PrettyConfig::expanded();
        assert_eq!(config.max_width, 40);
        assert_eq!(config.break_depth, 1);
    }

    #[test]
    fn test_pretty_config_builder() {
        let config = PrettyConfig::default()
            .with_indent_width(4)
            .with_max_width(100)
            .with_tabs(true)
            .with_print_sorts(true)
            .with_break_depth(3);

        assert_eq!(config.indent_width, 4);
        assert_eq!(config.max_width, 100);
        assert!(config.use_tabs);
        assert!(config.print_sorts);
        assert_eq!(config.break_depth, 3);
    }

    #[test]
    fn test_pretty_printer_simple_term() {
        let manager = TermManager::new();
        let pretty = PrettyPrinter::new(&manager);

        let output = pretty.print_term(manager.mk_true());
        assert_eq!(output, "true");
    }

    #[test]
    fn test_pretty_printer_compound_term() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let pretty = PrettyPrinter::new(&manager);
        let output = pretty.print_term(and);
        assert_eq!(output, "(and x y)");
    }

    #[test]
    fn test_pretty_printer_compact() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y, z]);
        let prod = manager.mk_mul([sum, x]);

        let config = PrettyConfig::compact();
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(prod);

        // Compact mode should not break lines
        assert!(!output.contains('\n'));
        assert!(output.contains("(* (+ x y z) x)"));
    }

    #[test]
    fn test_pretty_printer_expanded() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);
        let w = manager.mk_var("w", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y, z, w]);

        let config = PrettyConfig::expanded();
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(sum);

        // Expanded mode with many terms should break lines
        // The exact format depends on the width calculation
        assert!(output.contains("+"));
        assert!(output.contains("x"));
        assert!(output.contains("y"));
        assert!(output.contains("z"));
        assert!(output.contains("w"));
    }

    #[test]
    fn test_pretty_printer_nested_ite() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let a = manager.mk_int(1);
        let b = manager.mk_int(2);
        let ite = manager.mk_ite(x, a, b);

        let config = PrettyConfig::default()
            .with_max_width(10)
            .with_break_depth(0);
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(ite);

        // Should break due to small max_width
        assert!(output.contains("ite"));
        assert!(output.contains("x"));
    }

    // ==================== Model Printing Tests ====================

    #[test]
    fn test_print_empty_model() {
        let manager = TermManager::new();
        let model = Model::new();
        let printer = Printer::new(&manager);

        let output = printer.print_model(&model);
        assert!(output.contains("(model"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_model_with_bool_assignment() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut model = Model::new();
        model.assign_bool(x, true);

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("define-fun x () Bool true"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_model_with_int_assignment() {
        let mut manager = TermManager::new();
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let mut model = Model::new();
        model.assign_int(y, num_bigint::BigInt::from(42));

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("define-fun y () Int 42"));
    }

    #[test]
    fn test_print_model_with_multiple_assignments() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let mut model = Model::new();
        model.assign_bool(x, false);
        model.assign_int(y, num_bigint::BigInt::from(10));

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("x"));
        assert!(output.contains("Bool"));
        assert!(output.contains("false"));
        assert!(output.contains("y"));
        assert!(output.contains("Int"));
        assert!(output.contains("10"));
    }

    #[test]
    fn test_print_model_with_bitvec_assignment() {
        let mut manager = TermManager::new();
        let bv_sort = manager.sorts.bitvec(8);
        let z = manager.mk_var("z", bv_sort);

        let mut model = Model::new();
        model.assign_bitvec(z, 0xFF, 8);

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("z"));
        assert!(output.contains("#xff"));
    }

    // ==================== Proof Printing Tests ====================

    #[test]
    fn test_print_empty_proof() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let false_term = manager.mk_false();

        let root = ProofNode::new(ProofId(0), ProofRule::Contradiction, false_term);
        proof.add_node(root);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("contradiction"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_proof_with_assumption() {
        use crate::ast::proof::*;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut proof = Proof::new();
        let assume_node = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            x,
        );
        proof.add_node(assume_node);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("assume"));
        assert!(output.contains("H1"));
        assert!(output.contains("conclusion"));
    }

    #[test]
    fn test_print_proof_with_resolution() {
        use crate::ast::proof::*;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let p1 = manager.mk_var("p", manager.sorts.bool_sort);
        let p2 = manager.mk_var("q", manager.sorts.bool_sort);

        let mut proof = Proof::new();

        // Add premise nodes
        let node1 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("A1".to_string()),
            },
            p1,
        );
        let node2 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("A2".to_string()),
            },
            p2,
        );

        // Add resolution node
        let resolution_node = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: x },
            manager.mk_true(),
            vec![ProofId(0), ProofId(1)],
        );

        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(resolution_node);
        proof.set_root(ProofId(2));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("resolution"));
        assert!(output.contains("premises"));
        assert!(output.contains("@p0"));
        assert!(output.contains("@p1"));
    }

    #[test]
    fn test_print_proof_with_metadata() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();

        let mut node = ProofNode::new(
            ProofId(0),
            ProofRule::TheoryLemma {
                theory: "LIA".to_string(),
            },
            manager.mk_false(),
        );
        node.add_metadata("source".to_string(), "farkas".to_string());

        proof.add_node(node);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("theory-lemma"));
        assert!(output.contains("LIA"));
        assert!(output.contains("metadata"));
        assert!(output.contains("source"));
        assert!(output.contains("farkas"));
    }

    // ────────────────────────────────────────────────────────────────────
    // Proof text quoting: `write_proof_rule`/`write_proof_node` embed four
    // pieces of free-form text (metadata values, `Assume` names, `TheoryLemma`
    // theory names, `Custom` rule names) into `(step ... :key "...")`-shaped
    // SMT-LIB output. All four used to `write!(w, "...\"{}\"...", text)` the
    // value raw, so a `"` in the text ended the literal early and corrupted
    // the rest of the proof; they now all route through
    // `format_string_literal`, exactly like the term printers.
    // ────────────────────────────────────────────────────────────────────

    /// Metadata values containing a quote, a backslash, a `\u`-prefixed
    /// literal substring, a non-ASCII code point, and a control character
    /// must all come out exactly as `format_string_literal` would encode
    /// them, with no bare (unescaped) quote in the output.
    #[test]
    fn test_proof_metadata_value_special_chars_are_escaped() {
        use crate::ast::proof::*;

        for value in ["a\"b", "a\\b", "\\u0041", "caf\u{e9}", "line\u{0}break"] {
            let manager = TermManager::new();
            let mut proof = Proof::new();
            let mut node = ProofNode::new(
                ProofId(0),
                ProofRule::TheoryLemma {
                    theory: "LIA".to_string(),
                },
                manager.mk_false(),
            );
            node.add_metadata("source".to_string(), value.to_string());
            proof.add_node(node);
            proof.set_root(ProofId(0));

            let output = Printer::new(&manager).print_proof(&proof);
            let expected_literal = format_string_literal(value);
            assert!(
                output.contains(&format!(":source {expected_literal}")),
                "{value:?} did not print as the expected literal {expected_literal} in {output}"
            );
        }
    }

    /// Control: a metadata value with no special characters prints as a
    /// plain quoted literal, unchanged and with no gratuitous escaping.
    #[test]
    fn test_proof_metadata_value_plain_ascii_unchanged() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let mut node = ProofNode::new(
            ProofId(0),
            ProofRule::TheoryLemma {
                theory: "LIA".to_string(),
            },
            manager.mk_false(),
        );
        node.add_metadata("source".to_string(), "farkas".to_string());
        proof.add_node(node);
        proof.set_root(ProofId(0));

        let output = Printer::new(&manager).print_proof(&proof);
        assert!(output.contains(":source \"farkas\""));
    }

    /// `Assume { name }` must escape the same way.
    #[test]
    fn test_proof_assume_name_special_chars_are_escaped() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let node = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1\"\u{e9}".to_string()),
            },
            manager.mk_true(),
        );
        proof.add_node(node);
        proof.set_root(ProofId(0));

        let output = Printer::new(&manager).print_proof(&proof);
        let expected_literal = format_string_literal("H1\"\u{e9}");
        assert!(
            output.contains(&format!(":name {expected_literal}")),
            "expected {expected_literal} in {output}"
        );
    }

    /// `TheoryLemma { theory }` must escape the same way.
    #[test]
    fn test_proof_theory_lemma_name_special_chars_are_escaped() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let node = ProofNode::new(
            ProofId(0),
            ProofRule::TheoryLemma {
                theory: "BV\\theory".to_string(),
            },
            manager.mk_false(),
        );
        proof.add_node(node);
        proof.set_root(ProofId(0));

        let output = Printer::new(&manager).print_proof(&proof);
        let expected_literal = format_string_literal("BV\\theory");
        assert!(
            output.contains(&format!(":theory {expected_literal}")),
            "expected {expected_literal} in {output}"
        );
    }

    /// `Custom { name }` must escape the same way.
    #[test]
    fn test_proof_custom_rule_name_special_chars_are_escaped() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let node = ProofNode::new(
            ProofId(0),
            ProofRule::Custom {
                name: "weird\"rule\u{0}".to_string(),
            },
            manager.mk_true(),
        );
        proof.add_node(node);
        proof.set_root(ProofId(0));

        let output = Printer::new(&manager).print_proof(&proof);
        let expected_literal = format_string_literal("weird\"rule\u{0}");
        assert!(
            output.contains(&format!(":name {expected_literal}")),
            "expected {expected_literal} in {output}"
        );
    }
}
