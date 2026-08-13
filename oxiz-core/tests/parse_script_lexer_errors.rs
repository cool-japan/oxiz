//! Regression test for the todo-1174 leftover in `smtlib/parser/mod.rs`:
//! `parse_script` must surface accumulated *lexical* errors as a
//! `ParseError` after the command loop, rather than silently solving a
//! lexically-corrupted script.
//!
//! The lexer keeps producing a best-effort token stream after a malformed
//! token (so the command loop can still make progress) while recording each
//! problem in `errors()`. Before this fix, `parse_script` ignored that list,
//! so an SMT-LIB script containing e.g. a leading-zero numeral (`007`, which
//! SMT-LIB numerals forbid) would parse "successfully". Now the whole script
//! is rejected once any lexical error has been seen.

use oxiz_core::ast::TermManager;
use oxiz_core::error::OxizError;
use oxiz_core::smtlib::parse_script;

#[test]
fn parse_script_rejects_a_lexically_malformed_numeral() {
    // `007` is a leading-zero numeral: the lexer records a lexical error but
    // still yields a best-effort token, so command parsing itself succeeds.
    // `parse_script` must nonetheless reject the whole script.
    let src = "(set-logic QF_LIA)(declare-const x Int)(assert (= x 007))(check-sat)";
    let mut manager = TermManager::new();
    let result = parse_script(src, &mut manager);

    match result {
        Err(OxizError::ParseError { message, .. }) => {
            assert!(
                message.contains("lexical error"),
                "a leading-zero numeral must be surfaced as a lexical error, got: {message}"
            );
        }
        Err(other) => panic!("expected a ParseError surfacing the lexical error, got {other:?}"),
        Ok(_) => panic!("a lexically-malformed script must not parse successfully"),
    }
}

#[test]
fn parse_script_accepts_a_lexically_well_formed_script() {
    // Control: a clean script (including a bare `0` and a decimal, both of
    // which are lexically valid) must still parse without a spurious lexical
    // error. This guards against the surfacing check over-triggering.
    let src = "(set-logic QF_LIRA)\
               (declare-const x Int)\
               (declare-const y Real)\
               (assert (= x 0))\
               (assert (= y 0.001))\
               (check-sat)";
    let mut manager = TermManager::new();
    assert!(
        parse_script(src, &mut manager).is_ok(),
        "a lexically well-formed script must not be rejected as malformed"
    );
}

#[test]
fn parse_script_rejects_an_unlexable_character_instead_of_hanging() {
    // A character that can neither start nor continue a symbol (here a NUL, but
    // `,` `[` `\` `'` and U+007F behave identically) used to make the lexer
    // return a zero-width `Symbol("")` at an unchanged position for ever, and
    // the parser's balanced-paren skip then spun on it — a real infinite loop,
    // with no timeout on wasm32. It must now be a plain lexical error.
    for source in [
        "(\u{0}check-sat)",
        "(check-sat)(,foo)",
        "(check-sat)([foo)",
        "(check-sat)(\\foo)",
        "(check-sat)('foo)",
        "(check-sat)(\u{7f}foo)",
    ] {
        let mut manager = TermManager::new();
        match parse_script(source, &mut manager) {
            Err(OxizError::ParseError { message, .. }) => {
                assert!(
                    message.contains("lexical error"),
                    "expected a lexical error for {source:?}, got: {message}"
                );
            }
            Err(other) => panic!("expected a ParseError for {source:?}, got {other:?}"),
            Ok(_) => panic!("{source:?} must not parse successfully"),
        }
    }
}
