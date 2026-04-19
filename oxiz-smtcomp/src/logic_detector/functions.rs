//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashSet;

use super::types::{ConstructKind, StructuralFeatures, TheoryBits};

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
        if tok == "set-logic" && let Some(next) = iter.peek() && !next.is_empty() {
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
    if tokens.contains("FloatingPoint") || tokens.contains("Float16")
        || tokens.contains("Float32") || tokens.contains("Float64")
        || tokens.contains("Float128") || tokens.contains("RoundingMode")
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
    if tokens.contains("/") {
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
        if tok.starts_with("bv") && tok.len() > 2
            && tok.chars().skip(2).all(|c| c.is_alphabetic())
        {
            bits.has_bv = true;
            return;
        }
    }
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
    if tokens.contains("declare-datatype") || tokens.contains("declare-datatypes")
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
            if tokens[i + 3] == "(" {
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
    if tokens.iter().any(|t| t == "^" || t == "int.pow") {
        bits.has_nonlinear = true;
        return;
    }
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
        if tok.len() == 1 {
            return false;
        }
    } else if !first.is_ascii_digit() {
        return false;
    }
    tok.chars().skip(1).all(|c| c.is_ascii_digit() || c == '.')
}
/// Increment the count for `width` inside `histogram`, inserting a new entry
/// `(width, 1)` if no existing entry matches.
fn histogram_increment(histogram: &mut Vec<(u32, u32)>, key: u32) {
    for entry in histogram.iter_mut() {
        if entry.0 == key {
            entry.1 = entry.1.saturating_add(1);
            return;
        }
    }
    histogram.push((key, 1));
}
/// Try to parse an unsigned decimal integer from `tok`.
fn parse_u32(tok: &str) -> Option<u32> {
    tok.parse::<u32>().ok()
}
/// Extract structural features from raw SMT-LIB source text.
///
/// The extractor makes a single pass over the token stream produced by
/// [`tokenize`] and collects the metrics that make up [`StructuralFeatures`].
/// It does **not** perform full parsing and does not require an AST.
///
/// The scanner maintains a `depth_stack` of [`ConstructKind`] entries so that
/// closing parentheses can correctly unwind per-construct depth counters.
#[must_use]
pub fn extract_structural_features(source: &str) -> StructuralFeatures {
    let mut features = StructuralFeatures::default();
    let mut let_depth = 0_u32;
    let mut quantifier_depth = 0_u32;
    let mut ite_depth = 0_u32;
    let mut depth_stack: Vec<ConstructKind> = Vec::new();
    let tokens = tokenize(source);
    let mut i = 0;
    while i < tokens.len() {
        let tok = &tokens[i];
        match tok.as_str() {
            "(" => {
                let next = tokens.get(i + 1).map(String::as_str);
                let max_depth = u32::try_from(depth_stack.len().saturating_add(1))
                    .unwrap_or(u32::MAX);
                features.max_term_depth = features.max_term_depth.max(max_depth);
                let parent = depth_stack.last().copied();
                let kind = match next {
                    Some("assert") => {
                        features.clause_count = features.clause_count.saturating_add(1);
                        ConstructKind::Assert
                    }
                    Some("forall") | Some("exists") => {
                        quantifier_depth = quantifier_depth.saturating_add(1);
                        features.max_quantifier_nesting = features
                            .max_quantifier_nesting
                            .max(quantifier_depth);
                        ConstructKind::Quantifier
                    }
                    Some("ite") => {
                        ite_depth = ite_depth.saturating_add(1);
                        features.max_ite_depth = features.max_ite_depth.max(ite_depth);
                        ConstructKind::Ite
                    }
                    Some("let") => {
                        let_depth = let_depth.saturating_add(1);
                        features.max_let_depth = features.max_let_depth.max(let_depth);
                        ConstructKind::Let
                    }
                    Some("and") | Some("or") | Some("not") | Some("=>") | Some("xor") => {
                        ConstructKind::Bool
                    }
                    Some("=")
                    | Some("<")
                    | Some("<=")
                    | Some(">")
                    | Some(">=")
                    | Some("distinct") => {
                        features.atom_count = features.atom_count.saturating_add(1);
                        ConstructKind::Other
                    }
                    Some("Array") => {
                        let dims = depth_stack
                            .iter()
                            .filter(|&&kind| kind == ConstructKind::Array)
                            .count()
                            .saturating_add(1);
                        let dims = u32::try_from(dims).unwrap_or(u32::MAX);
                        histogram_increment(&mut features.array_dim_histogram, dims);
                        ConstructKind::Array
                    }
                    Some("_") => {
                        if tokens.get(i + 2).map(String::as_str) == Some("BitVec")
                            && let Some(width) = tokens
                                .get(i + 3)
                                .and_then(|tok| parse_u32(tok))
                        {
                            histogram_increment(&mut features.bv_width_histogram, width);
                        }
                        ConstructKind::Other
                    }
                    Some(head) => {
                        if should_count_as_predicate_app(head, parent) {
                            features.atom_count = features.atom_count.saturating_add(1);
                        }
                        ConstructKind::Other
                    }
                    None => ConstructKind::Other,
                };
                depth_stack.push(kind);
            }
            ")" => {
                if let Some(closed) = depth_stack.pop() {
                    match closed {
                        ConstructKind::Let => let_depth = let_depth.saturating_sub(1),
                        ConstructKind::Quantifier => {
                            quantifier_depth = quantifier_depth.saturating_sub(1);
                        }
                        ConstructKind::Ite => ite_depth = ite_depth.saturating_sub(1),
                        ConstructKind::Assert
                        | ConstructKind::Bool
                        | ConstructKind::Array
                        | ConstructKind::Other => {}
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    features.bv_width_histogram.sort_unstable_by_key(|&(width, _)| width);
    features.array_dim_histogram.sort_unstable_by_key(|&(dims, _)| dims);
    features
}
/// Whether `head` should count as a predicate application in the current context.
fn should_count_as_predicate_app(head: &str, parent: Option<ConstructKind>) -> bool {
    if !matches!(
        parent, Some(ConstructKind::Assert | ConstructKind::Bool |
        ConstructKind::Quantifier)
    ) {
        return false;
    }
    if head.is_empty()
        || matches!(head, "(" | ")" | "_" | "let" | "ite" | "forall" | "exists")
    {
        return false;
    }
    !matches!(
        head, "assert" | "check-sat" | "check-sat-assuming" | "declare-const" |
        "declare-fun" | "define-const" | "define-fun" | "set-logic" | "set-info" |
        "set-option" | "Array" | "BitVec" | "and" | "or" | "not" | "=>" | "xor" | "+" |
        "-" | "*" | "/" | "div" | "mod" | "abs" | "to_real" | "to_int" | "select" |
        "store" | "concat" | "extract"
    )
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
    if fp && !quant && !string && !arr && !dt {
        if bv {
            return "QF_BVFP".to_string();
        }
        if !int && !real && !uf {
            return "QF_FP".to_string();
        }
        return "ALL".to_string();
    }
    if string && !quant && !bv && !arr && !fp && !dt {
        if int {
            return "QF_SLIA".to_string();
        }
        return "QF_S".to_string();
    }
    if dt && !quant && !string && !fp {
        if uf {
            return "QF_UFDT".to_string();
        }
        return "QF_DT".to_string();
    }
    if dt && quant && uf && !string && !fp {
        return "UFDT".to_string();
    }
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
    if !bv && !arr && !string && !fp && !dt {
        if quant {
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
        assert_eq!(detect_logic(QF_LIA_SRC), "QF_LIA");
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
        assert!(! bits.is_empty());
    }
    #[test]
    fn test_comments_do_not_confuse_scanner() {
        let src = "
; this mentions Int and BitVec and forall but is a comment
(set-info :name \"contains Int Real BitVec forall\")
(declare-const b Bool)
(assert b)
(check-sat)
";
        let logic = detect_logic(src);
        assert_eq!(logic, "ALL");
    }
    #[test]
    fn test_nonlinear_with_constant_is_linear() {
        let src = "
(declare-const x Int)
(assert (= (* 2 x) 6))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_LIA");
    }
    #[test]
    fn test_uf_constant_only_is_not_uf() {
        let src = "
(declare-fun x () Int)
(assert (>= x 0))
(check-sat)
";
        assert_eq!(detect_logic(src), "QF_LIA");
    }
    #[test]
    fn test_max_term_depth() {
        let src = "
(declare-const x Int)
(declare-const y Int)
(declare-const a Int)
(declare-const b Int)
(assert (= (+ (+ x y) (+ a b)) 0))
(check-sat)
";
        let features = extract_structural_features(src);
        assert!(
            features.max_term_depth >= 2, "expected max_term_depth >= 2, got {}",
            features.max_term_depth
        );
    }
    #[test]
    fn test_clause_count() {
        let src = "
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(assert (>= x 0))
(assert (>= y 0))
(assert (>= z 0))
(check-sat)
";
        let features = extract_structural_features(src);
        assert_eq!(
            features.clause_count, 3, "expected clause_count == 3, got {}", features
            .clause_count
        );
    }
    #[test]
    fn test_quantifier_nesting() {
        let src = "
(declare-fun p (Int Int) Bool)
(assert (forall ((x Int)) (forall ((y Int)) (p x y))))
(check-sat)
";
        let features = extract_structural_features(src);
        assert_eq!(
            features.max_quantifier_nesting, 2,
            "expected max_quantifier_nesting == 2, got {}", features
            .max_quantifier_nesting
        );
    }
    #[test]
    fn test_bv_width_histogram() {
        let src = "
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 32))
(assert (= (bvadd a a) a))
(check-sat)
";
        let features = extract_structural_features(src);
        let widths: Vec<u32> = features
            .bv_width_histogram
            .iter()
            .map(|&(w, _)| w)
            .collect();
        assert!(
            widths.contains(& 8), "expected width 8 in histogram, got {:?}", features
            .bv_width_histogram
        );
        assert!(
            widths.contains(& 32), "expected width 32 in histogram, got {:?}", features
            .bv_width_histogram
        );
    }
    #[test]
    fn test_ite_depth() {
        let src = "
(declare-const c1 Bool)
(declare-const c2 Bool)
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)
(assert (= (ite c1 (ite c2 a b) c) 0))
(check-sat)
";
        let features = extract_structural_features(src);
        assert_eq!(
            features.max_ite_depth, 2, "expected max_ite_depth == 2, got {}", features
            .max_ite_depth
        );
    }
    #[test]
    fn test_let_depth() {
        let src = "
(declare-const x Int)
(assert
  (let ((a (+ x 1)))
    (let ((b (+ a 1)))
      (= b 3))))
(check-sat)
";
        let features = extract_structural_features(src);
        assert_eq!(
            features.max_let_depth, 2, "expected max_let_depth == 2, got {}", features
            .max_let_depth
        );
    }
    #[test]
    fn test_structural_features_default_zero() {
        let features = extract_structural_features("(check-sat)");
        assert_eq!(features.max_term_depth, 1);
        assert_eq!(features.atom_count, 0);
        assert_eq!(features.clause_count, 0);
        assert_eq!(features.max_quantifier_nesting, 0);
        assert!(features.bv_width_histogram.is_empty());
        assert!(features.array_dim_histogram.is_empty());
        assert_eq!(features.max_ite_depth, 0);
        assert_eq!(features.max_let_depth, 0);
    }
}
