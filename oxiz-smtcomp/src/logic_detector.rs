//! Automatic SMT-LIB logic detection from benchmark source text.
//!
//! SMT-LIB benchmarks normally announce their theory with a `(set-logic ...)`
//! directive near the top of the file. Real-world corpora are not so tidy:
//! some inputs omit the directive, some annotate it only in a trailing
//! `(set-info ...)` comment, and some ship unadorned scripts. The
//! [`detect_logic`] function bridges that gap by scanning the raw SMT-LIB
//! source for theory keywords and deriving the most specific standard logic
//! name it can.
//!
//! # Why text-level detection?
//!
//! The [`crate::loader::Loader`] pipeline reads SMT-LIB files as raw
//! `String` content before handing them off to [`oxiz_core::smtlib::parse_script`].
//! Detecting the logic at the text layer keeps the analysis crate-local —
//! we do not have to build a parser dependency, keep a [`oxiz_core::ast::TermManager`]
//! alive, or re-implement the term AST walk. SMT-LIB's surface syntax uses
//! reserved theory keywords (`bvadd`, `select`, `fp.add`, `forall`, …) that
//! are stable enough across the current language revision for a lightweight
//! tokenizer to classify a benchmark correctly.
//!
//! # Accuracy
//!
//! The scanner is deliberately conservative: when in doubt it picks a more
//! general logic (e.g., falls back to `"ALL"`) rather than claim a feature
//! that is not actually used. The detector is therefore safe to use as a
//! default when no `(set-logic)` header is present; an explicit
//! `(set-logic X)` always takes precedence.
//!
//! # Examples
//!
//! ```
//! use oxiz_smtcomp::logic_detector::detect_logic;
//!
//! let src = "(declare-const x Int)(assert (>= x 0))(check-sat)";
//! assert_eq!(detect_logic(src), "QF_LIA");
//! ```

use std::collections::HashSet;

/// Theory feature flags detected in an SMT-LIB benchmark.
///
/// Each field records whether the associated theory appears anywhere in the
/// scanned source. `TheoryBits` is a flat, copyable descriptor so it can be
/// combined, compared, and mapped to logic names without additional
/// allocation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TheoryBits {
    /// User-declared uninterpreted functions with non-empty argument lists.
    pub has_uf: bool,
    /// Integer sort or any integer-flavoured operator.
    pub has_int: bool,
    /// Real sort or any real-flavoured operator.
    pub has_real: bool,
    /// `BitVec` sort or any `bv*` operator.
    pub has_bv: bool,
    /// Array sort or any `select`/`store` operator.
    pub has_array: bool,
    /// `String` sort or any `str.*` operator.
    pub has_string: bool,
    /// `FloatingPoint` sort or any `fp.*` operator.
    pub has_fp: bool,
    /// `declare-datatype(s)` declarations.
    pub has_dt: bool,
    /// Nonlinear arithmetic (variable-by-variable `*` / `/` / `mod` / `pow`).
    pub has_nonlinear: bool,
    /// `forall` / `exists` binders.
    pub has_quantifier: bool,
}

impl TheoryBits {
    /// Whether any theory bit is set.
    ///
    /// Used by the logic mapper to distinguish "no signal — probably a pure
    /// Boolean script" from "something interesting was detected".
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !(self.has_uf
            || self.has_int
            || self.has_real
            || self.has_bv
            || self.has_array
            || self.has_string
            || self.has_fp
            || self.has_dt
            || self.has_nonlinear
            || self.has_quantifier)
    }
}

/// Detect the minimal SMT-LIB logic that covers the given benchmark source.
///
/// The detector first honours an explicit `(set-logic X)` directive if one is
/// present: SMT-LIB's spec gives the user's declaration priority over any
/// inferred signature. When no directive is found the scanner walks the raw
/// token stream, collects a [`TheoryBits`] fingerprint, and maps it to the
/// most specific standard logic name it can. Combinations that do not match
/// any standard logic fall back to `"ALL"`.
#[must_use]
pub fn detect_logic(smtlib_source: &str) -> String {
    if let Some(explicit) = extract_set_logic(smtlib_source) {
        return explicit;
    }
    let bits = detect_theory_bits(smtlib_source);
    logic_from_bits(&bits)
}

/// Scan the SMT-LIB source for theory feature flags.
///
/// Returns a [`TheoryBits`] recording which theories the scanner found
/// evidence for. The scanner is a single-pass tokenizer over the already-
/// tokenized source (see [`tokenize`]); it does not attempt full parsing and
/// is intentionally tolerant of non-standard extensions.
#[must_use]
pub fn detect_theory_bits(smtlib_source: &str) -> TheoryBits {
    let tokens = tokenize(smtlib_source);
    let token_set: HashSet<&str> = tokens.iter().map(String::as_str).collect();

    let mut bits = TheoryBits::default();

    scan_sort_keywords(&token_set, &mut bits);
    scan_arith_keywords(&token_set, &mut bits);
    scan_bv_keywords(&token_set, &mut bits);
    scan_array_keywords(&token_set, &mut bits);
    scan_string_keywords(&token_set, &mut bits);
    scan_fp_keywords(&token_set, &mut bits);
    scan_quantifier_keywords(&token_set, &mut bits);
    scan_datatype_keywords(&token_set, &mut bits);

    scan_uf_declarations(&tokens, &mut bits);
    scan_nonlinear_multiplication(&tokens, &mut bits);

    bits
}

/// Extract the argument of a `(set-logic X)` directive when one is present.
///
/// The scanner examines the first 200 tokens of the source — large enough to
/// skip through leading `(set-info ...)` blocks on real-world benchmarks but
/// small enough to bail quickly on inputs without a logic header.
fn extract_set_logic(source: &str) -> Option<String> {
    let tokens = tokenize(source);
    let mut iter = tokens.iter().take(200).peekable();
    while let Some(tok) = iter.next() {
        if tok == "set-logic"
            && let Some(next) = iter.peek()
            && !next.is_empty()
        {
            let candidate = next.trim_end_matches(')');
            if !candidate.is_empty() && candidate != "(" && candidate != ")" {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// Tokenize SMT-LIB source for keyword scanning.
///
/// The tokenizer splits on whitespace and parentheses, drops empty pieces,
/// and strips `;`-to-end-of-line comments and quoted strings so keywords
/// appearing in comments or string literals do not confuse the scanner.
fn tokenize(source: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_line_comment = false;
    let mut in_string = false;
    let mut in_quoted_symbol = false;

    let flush = |current: &mut String, tokens: &mut Vec<String>| {
        if !current.is_empty() {
            tokens.push(std::mem::take(current));
        }
    };

    for ch in source.chars() {
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
            }
            continue;
        }
        if in_string {
            // SMT-LIB string literal — consume until matching `"`.
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if in_quoted_symbol {
            if ch == '|' {
                in_quoted_symbol = false;
            }
            continue;
        }

        match ch {
            ';' => {
                flush(&mut current, &mut tokens);
                in_line_comment = true;
            }
            '"' => {
                flush(&mut current, &mut tokens);
                in_string = true;
            }
            '|' => {
                flush(&mut current, &mut tokens);
                in_quoted_symbol = true;
            }
            '(' | ')' => {
                flush(&mut current, &mut tokens);
                tokens.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                flush(&mut current, &mut tokens);
            }
            c => current.push(c),
        }
    }
    flush(&mut current, &mut tokens);
    tokens
}

/// Detect sort keywords that unambiguously mark a theory.
fn scan_sort_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if tokens.contains("Int") {
        bits.has_int = true;
    }
    if tokens.contains("Real") {
        bits.has_real = true;
    }
    if tokens.contains("BitVec") {
        bits.has_bv = true;
    }
    if tokens.contains("Array") {
        bits.has_array = true;
    }
    if tokens.contains("String") {
        bits.has_string = true;
    }
    if tokens.contains("FloatingPoint")
        || tokens.contains("Float16")
        || tokens.contains("Float32")
        || tokens.contains("Float64")
        || tokens.contains("Float128")
        || tokens.contains("RoundingMode")
    {
        bits.has_fp = true;
    }
}

/// Detect arithmetic operators that force integer or real flavouring.
fn scan_arith_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    const INT_OPS: &[&str] = &["mod", "div", "abs", "to_int", "int.pow"];
    for op in INT_OPS {
        if tokens.contains(op) {
            bits.has_int = true;
        }
    }
    const REAL_OPS: &[&str] = &["to_real", "is_int"];
    for op in REAL_OPS {
        if tokens.contains(op) {
            bits.has_real = true;
        }
    }
    // `/` is the real-division operator when applied as a head symbol. We
    // cannot reliably distinguish "/ as division" from "/ appearing in a
    // longer identifier" at the token layer, so we rely on the sort
    // declarations above to seed the Real flag and only treat `/` as a hint.
    if tokens.contains("/") {
        // `/` is a weak signal — many Int-only benchmarks include `/` as part
        // of path-like identifiers. Only promote to Real if no Int sort is
        // already declared.
        if !bits.has_int {
            bits.has_real = true;
        }
    }
}

/// Detect bit-vector operators beyond the `BitVec` sort.
fn scan_bv_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if bits.has_bv {
        return;
    }
    for tok in tokens {
        // Fixed-width BV prefix covers bvadd, bvsub, bvmul, bvand, bvor,
        // bvxor, bvshl, bvshr, bvlshr, bvashr, bvult, bvule, bvslt, bvsle,
        // bvneg, bvnot, bvnand, bvnor, bvxnor, bvcomp, bvudiv, bvurem,
        // bvsdiv, bvsrem, bvsmod.
        if tok.starts_with("bv") && tok.len() > 2 && tok.chars().skip(2).all(|c| c.is_alphabetic())
        {
            bits.has_bv = true;
            return;
        }
    }
    // `concat` and `extract` are BV operators but also appear in String
    // theory. They only imply BV if no string signal is present; we
    // therefore defer the decision until after the string scan.
    if tokens.contains("concat") && !tokens.contains("str.++") {
        bits.has_bv = true;
    }
}

/// Detect array operators.
fn scan_array_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if tokens.contains("select") || tokens.contains("store") {
        bits.has_array = true;
    }
}

/// Detect string operators.
fn scan_string_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if bits.has_string {
        return;
    }
    for tok in tokens {
        if tok.starts_with("str.") || *tok == "str.++" {
            bits.has_string = true;
            return;
        }
    }
}

/// Detect floating-point operators.
fn scan_fp_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if bits.has_fp {
        return;
    }
    for tok in tokens {
        if tok.starts_with("fp.") {
            bits.has_fp = true;
            return;
        }
    }
}

/// Detect universal/existential quantifiers.
fn scan_quantifier_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if tokens.contains("forall") || tokens.contains("exists") {
        bits.has_quantifier = true;
    }
}

/// Detect datatype declarations.
fn scan_datatype_keywords(tokens: &HashSet<&str>, bits: &mut TheoryBits) {
    if tokens.contains("declare-datatype")
        || tokens.contains("declare-datatypes")
        || tokens.contains("declare-codatatype")
        || tokens.contains("declare-codatatypes")
    {
        bits.has_dt = true;
    }
}

/// Detect uninterpreted-function declarations.
///
/// In SMT-LIB `(declare-fun f (<arg-sorts>) <ret-sort>)` introduces an
/// uninterpreted function iff the argument sort list is non-empty. A nullary
/// `(declare-fun x () Int)` is only a constant and does not contribute UF.
fn scan_uf_declarations(tokens: &[String], bits: &mut TheoryBits) {
    if bits.has_uf {
        return;
    }
    let mut i = 0;
    while i + 4 < tokens.len() {
        if tokens[i] == "(" && tokens[i + 1] == "declare-fun" {
            // tokens[i+2] = function name, tokens[i+3] = "(" opening arg list
            if tokens[i + 3] == "(" {
                // If the next token is `)` the arg list is empty — constant.
                if tokens[i + 4] != ")" {
                    bits.has_uf = true;
                    return;
                }
            }
        }
        i += 1;
    }
}

/// Heuristic nonlinear detection: flag a `(* a b)` where neither `a` nor `b`
/// is an obvious numeric literal. Also flags `int.pow` / `^` power operators.
fn scan_nonlinear_multiplication(tokens: &[String], bits: &mut TheoryBits) {
    if bits.has_nonlinear {
        return;
    }
    // `^` power, if present, is nonlinear by definition.
    if tokens.iter().any(|t| t == "^" || t == "int.pow") {
        bits.has_nonlinear = true;
        return;
    }
    // Look for `( * X Y )` patterns where both X and Y are non-literals.
    let mut i = 0;
    while i + 4 < tokens.len() {
        if tokens[i] == "(" && tokens[i + 1] == "*" {
            let a = &tokens[i + 2];
            let b = &tokens[i + 3];
            if !is_numeric_literal(a) && !is_numeric_literal(b) && a != "(" && b != "(" {
                bits.has_nonlinear = true;
                return;
            }
        }
        i += 1;
    }
}

/// Whether a token is a numeric literal (integer or decimal).
fn is_numeric_literal(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    let mut chars = tok.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if first == '-' {
        // A bare "-" is the subtraction operator, not a literal.
        if tok.len() == 1 {
            return false;
        }
    } else if !first.is_ascii_digit() {
        return false;
    }
    tok.chars().skip(1).all(|c| c.is_ascii_digit() || c == '.')
}

/// Map a [`TheoryBits`] fingerprint to the most specific standard logic name.
///
/// The order of the checks matters: richer combinations must be tested
/// before their supersets so that, for example, `UF + Int + Array` maps to
/// `"AUFLIA"` rather than being swallowed by the simpler `"UFLIA"` arm. Any
/// combination that is not covered by a standard logic falls back to
/// `"ALL"`.
#[must_use]
pub fn logic_from_bits(bits: &TheoryBits) -> String {
    let quant = bits.has_quantifier;
    let uf = bits.has_uf;
    let int = bits.has_int;
    let real = bits.has_real;
    let bv = bits.has_bv;
    let arr = bits.has_array;
    let string = bits.has_string;
    let fp = bits.has_fp;
    let dt = bits.has_dt;
    let nl = bits.has_nonlinear;

    // Floating-point: no standard logic currently mixes FP with quantifiers
    // or with other theories outside BV. Keep FP checks early and fall back
    // to ALL for exotic blends.
    if fp && !quant && !string && !arr && !dt {
        if bv {
            return "QF_BVFP".to_string();
        }
        if !int && !real && !uf {
            return "QF_FP".to_string();
        }
        return "ALL".to_string();
    }

    // String logics.
    if string && !quant && !bv && !arr && !fp && !dt {
        if int {
            return "QF_SLIA".to_string();
        }
        return "QF_S".to_string();
    }

    // Datatypes.
    if dt && !quant && !string && !fp {
        if uf {
            return "QF_UFDT".to_string();
        }
        return "QF_DT".to_string();
    }
    if dt && quant && uf && !string && !fp {
        return "UFDT".to_string();
    }

    // Bit-vector family (ordered most-specific → least-specific).
    if bv && !string && !fp && !dt {
        if quant {
            if arr && uf {
                return "AUFBV".to_string();
            }
            if arr {
                return "ABV".to_string();
            }
            if uf {
                return "UFBV".to_string();
            }
            return "BV".to_string();
        }
        if arr && uf {
            return "QF_AUFBV".to_string();
        }
        if arr {
            return "QF_ABV".to_string();
        }
        if uf {
            return "QF_UFBV".to_string();
        }
        return "QF_BV".to_string();
    }

    // Array + integer / real (no BV).
    if arr && !bv && !string && !fp && !dt {
        if quant {
            if int && real && uf {
                return "AUFLIRA".to_string();
            }
            if int && uf {
                return "AUFLIA".to_string();
            }
            if int {
                return "ALIA".to_string();
            }
        } else {
            if uf && int {
                return "QF_AUFLIA".to_string();
            }
            if int {
                return "QF_ALIA".to_string();
            }
            if real || (!int && !uf) {
                return "QF_AX".to_string();
            }
        }
    }

    // Pure arithmetic family.
    if !bv && !arr && !string && !fp && !dt {
        if quant {
            // Quantified arithmetic.
            if int && real && uf && nl {
                return "AUFNIRA".to_string();
            }
            if int && real && uf {
                return "AUFLIRA".to_string();
            }
            if int && nl && uf {
                return "UFNIA".to_string();
            }
            if real && nl && uf {
                return "UFNRA".to_string();
            }
            if int && uf {
                return "UFLIA".to_string();
            }
            if real && uf {
                return "UFLRA".to_string();
            }
            if int && nl {
                return "NIA".to_string();
            }
            if real && nl {
                return "NRA".to_string();
            }
            if int {
                return "LIA".to_string();
            }
            if real {
                return "LRA".to_string();
            }
            if uf {
                return "UF".to_string();
            }
        } else {
            // Quantifier-free arithmetic.
            if int && real && nl {
                return "QF_NIRA".to_string();
            }
            if int && real && uf {
                return "QF_UFLIRA".to_string();
            }
            if int && real {
                return "QF_LIRA".to_string();
            }
            if int && nl && uf {
                return "QF_UFNIA".to_string();
            }
            if real && nl && uf {
                return "QF_UFNRA".to_string();
            }
            if int && nl {
                return "QF_NIA".to_string();
            }
            if real && nl {
                return "QF_NRA".to_string();
            }
            if int && uf {
                return "QF_UFLIA".to_string();
            }
            if real && uf {
                return "QF_UFLRA".to_string();
            }
            if int {
                return "QF_LIA".to_string();
            }
            if real {
                return "QF_LRA".to_string();
            }
            if uf {
                return "QF_UF".to_string();
            }
        }
    }

    // Nothing matched a standard logic — fall back to ALL.
    "ALL".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const QF_LIA_SRC: &str = "
(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 0))
(check-sat)
";

    const QF_LIA_NO_HEADER: &str = "
(declare-const x Int)
(declare-const y Int)
(assert (>= (+ x y) 0))
(check-sat)
";

    const UFLIA_SRC_NO_HEADER: &str = "
(declare-fun f (Int) Int)
(assert (forall ((x Int)) (= (f x) x)))
(check-sat)
";

    const QF_AUFBV_SRC: &str = "
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(declare-fun g ((_ BitVec 8)) (_ BitVec 8))
(assert (= (g (select a #x00)) #xff))
(check-sat)
";

    const QF_BV_SRC: &str = "
(declare-const x (_ BitVec 32))
(declare-const y (_ BitVec 32))
(assert (= (bvadd x y) #x00000000))
(check-sat)
";

    const QF_LRA_SRC: &str = "
(declare-const x Real)
(assert (>= x 0.0))
(check-sat)
";

    const QF_NIA_SRC: &str = "
(declare-const x Int)
(declare-const y Int)
(assert (= (* x y) 10))
(check-sat)
";

    #[test]
    fn test_detect_qf_lia() {
        // Explicit header takes precedence.
        assert_eq!(detect_logic(QF_LIA_SRC), "QF_LIA");
        // And the scanner arrives at the same answer without a header.
        assert_eq!(detect_logic(QF_LIA_NO_HEADER), "QF_LIA");
    }

    #[test]
    fn test_detect_uflia() {
        assert_eq!(detect_logic(UFLIA_SRC_NO_HEADER), "UFLIA");
    }

    #[test]
    fn test_detect_qf_aufbv() {
        assert_eq!(detect_logic(QF_AUFBV_SRC), "QF_AUFBV");
    }

    #[test]
    fn test_fallback_all() {
        // Mix string + FP + BV + quantifier — no single standard logic
        // covers all four, so the detector must fall back.
        let src = "
(declare-const s String)
(declare-const f (_ FloatingPoint 8 24))
(declare-const b (_ BitVec 8))
(assert (forall ((x Int)) (> x 0)))
(check-sat)
";
        assert_eq!(detect_logic(src), "ALL");
    }

    #[test]
    fn test_detect_qf_bv() {
        assert_eq!(detect_logic(QF_BV_SRC), "QF_BV");
    }

    #[test]
    fn test_detect_qf_lra() {
        assert_eq!(detect_logic(QF_LRA_SRC), "QF_LRA");
    }

    #[test]
    fn test_detect_qf_nia() {
        assert_eq!(detect_logic(QF_NIA_SRC), "QF_NIA");
    }

    #[test]
    fn test_detect_explicit_wins_over_inference() {
        // Source declares an Int sort so inference would pick QF_LIA, but
        // the explicit header asks for UFLIA.
        let src = "
(set-logic UFLIA)
(declare-const x Int)
(assert (>= x 0))
(check-sat)
";
        assert_eq!(detect_logic(src), "UFLIA");
    }

    #[test]
    fn test_detect_qf_slia() {
        let src = "
(declare-const s String)
(declare-const n Int)
(assert (= (str.len s) n))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_SLIA");
    }

    #[test]
    fn test_detect_qf_fp() {
        let src = "
(declare-const x (_ FloatingPoint 8 24))
(declare-const y (_ FloatingPoint 8 24))
(assert (fp.eq x y))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_FP");
    }

    #[test]
    fn test_detect_uf_only() {
        // Pure Boolean uninterpreted-function fragment.
        let src = "
(declare-fun p (Bool) Bool)
(declare-const x Bool)
(assert (p x))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_UF");
    }

    #[test]
    fn test_detect_dt_fragment() {
        let src = "
(declare-datatypes ((Color 0)) (((Red) (Green) (Blue))))
(declare-const c Color)
(assert (= c Red))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_DT");
    }

    #[test]
    fn test_theory_bits_empty() {
        let bits = TheoryBits::default();
        assert!(bits.is_empty());
    }

    #[test]
    fn test_theory_bits_nonempty() {
        let bits = TheoryBits {
            has_int: true,
            ..TheoryBits::default()
        };
        assert!(!bits.is_empty());
    }

    #[test]
    fn test_comments_do_not_confuse_scanner() {
        // Keywords inside a `;` comment or a `"..."` string must be ignored.
        let src = "
; this mentions Int and BitVec and forall but is a comment
(set-info :name \"contains Int Real BitVec forall\")
(declare-const b Bool)
(assert b)
(check-sat)
";
        // Only pure Boolean content — should fall back since no theory bit
        // fires and there are no arithmetic/UF markers.
        let logic = detect_logic(src);
        assert_eq!(logic, "ALL");
    }

    #[test]
    fn test_nonlinear_with_constant_is_linear() {
        // `(* 2 x)` is linear, so detector must not mark nonlinear.
        let src = "
(declare-const x Int)
(assert (= (* 2 x) 6))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_LIA");
    }

    #[test]
    fn test_uf_constant_only_is_not_uf() {
        // `(declare-fun x () Int)` is nullary — a constant, not a UF symbol.
        let src = "
(declare-fun x () Int)
(assert (>= x 0))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_LIA");
    }
}
