//! Regression tests for the two bit-vector `ite` findings cargo-formal
//! reported against 0.3.3 on 2026-09-15 (`#P2b-24` and `#P2b-25` in
//! `TODO.md`), both found again by `bv_ite_selfcheck_fuzz.rs` on the 0.3.4
//! tree before the fixes.
//!
//! # `#P2b-24` — an `ite` selector outside the bit-blaster's fragment made the
//! whole term a *free* bit-vector
//!
//! `theory_bv_encode::bit_blast_cond_operands` accepted only `not`/`and`/`or`
//! over `=`/`bvult`/`bvule`/`bvslt`/`bvsle`.  cargo-formal's printer also
//! emits `distinct`, `xor` and `=>` inside selectors.  One such selector made
//! `encode_bv_term_recursive` fail for the *root* of the atom, and every
//! caller answered that failure with `BvSolver::new_bv(root, width)` — a
//! fresh, unconstrained bit-vector standing in for a `bvsub` or an `ite`
//! whose semantics the solver knows perfectly well.  A second atom sharing
//! the sub-term then bit-blasted it for real (or, the same way, left a
//! second free copy), the two disagreed on every model, and the debug-build
//! circuit self-check in `debug_verify_bv_circuits` panicked with "the model
//! says … but the operation evaluates to …".  A release build published the
//! model as `sat` until `Solver::model_refutes_assertions` refused it, so
//! the user-visible verdict was `unknown` on a formula the solver had
//! actually decided — or a wrong `sat` wherever that gate cannot evaluate.
//!
//! Fixed at the root: the selector fragment now covers `xor`, `=>`, the Bool
//! `ite`, `=`/`distinct` over Bool and `distinct` over BV; the encoder
//! abstracts only genuinely opaque leaves (an uninterpreted application) and
//! does so *at the leaf*; a remaining failure marks the atom unmodelled
//! (`unknown`, never `sat`) instead of handing it free bits; and
//! `BvSolver::bv_ite` reports its own failure instead of silently building
//! nothing.
//!
//! # `#P2b-25` — a theory conflict explanation omitted the pinned selectors
//!
//! `TheoryManager::on_assignment` mirrors every outer Boolean assignment into
//! the bit-blaster through `BvSolver::assert_bool_value`, which installs a
//! unit clause on the `ite` selector's boolean node.  The embedded SAT solver
//! resolves against that unit exactly like against an `assert_eq`, but
//! `collect_conflict_terms` reported only the recorded constraint terms, so
//! an `Unsat` that rested on a pinned selector came back as if it rested on
//! the constraints alone.  The CDCL(T) core learned a clause the theory never
//! derived and a satisfiable width-63 formula answered `unsat` — a **false
//! proof**, and one an LRAT check of the Boolean skeleton cannot see.  Fixed
//! by journalling every pinned atom (`BvSolver::pinned_terms`) with the
//! scope that pinned it and blaming it in every explanation.
//!
//! # `#P2b-27`, `#P2b-28`, `#P2b-29` — found by the close-out review of the two
//! fixes above
//!
//! * `#P2b-27`: a Boolean that occurs *only* as an `ite` selector has no outer
//!   clause, the SAT core never assigns it, `build_model` recorded nothing and
//!   `(get-value)` printed the sort default `false` — a published model that
//!   violates the assertion the circuit satisfied with `p = true`.  The same
//!   missing entry made every assertion above `p` `Undetermined` to the model
//!   gate, which is how the pre-fix free-bit-vector model of the `#P2b-24`
//!   campaign got past it.  Fixed by publishing `BvSolver::bool_value`, and
//!   by the gate refusing a `sat` whose assertion it cannot evaluate for want
//!   of a Boolean's entry.
//! * `#P2b-28`: `bvult`/`bvule` were also mirrored into the linear arithmetic
//!   solver as an `i64`-rational relaxation; `#x7fffffffffffffff` made
//!   `x < k ⇒ x ≤ k − 1` and `−rhs` overflow (debug: panic, release: `unknown`
//!   for a satisfiable one-liner).  Fixed by retiring the mirror.
//! * `#P2b-29`: congruence never reached an opaque leaf *under* a bit-vector
//!   operation — `(= a b) ∧ (distinct (bvadd (f a) #x01) (bvadd (f b) #x01))`
//!   answered `sat` — and a circuit-entailed equality between two application
//!   arguments never reached congruence — `(= (bvadd x #x01) (bvadd y #x01))
//!   ∧ (distinct (g x) (g y))` answered `sat`.  Fixed by a bidirectional
//!   exchange in `final_check` (`TheoryManager::combine_bv_with_euf`): shared
//!   leaf equalities under EUF's explanation one way, lemmas entailed by
//!   EUF's refusal of the model's argument partition the other.
//!
//! Every script here was run against the unfixed tree before being kept, and
//! what that tree does is stated on each constant: the un-minimised `#P2b-24`
//! script panics there (debug build) while its declared-constant form and the
//! `c13` fixture answer `unknown` (the model gate refusing the free-bit
//! model), the `#P2b-25` one answers `unsat`, and the three later findings
//! answered as described above on the tree carrying only `#P2b-24`/`#P2b-25`.

use oxiz_solver::{Context, SolverResult};
use std::panic::{AssertUnwindSafe, catch_unwind};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_output(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script).unwrap_or_default()
}

fn verdict_of(outputs: &[String]) -> SolverResult {
    for token in outputs.iter().rev() {
        match token.trim() {
            "sat" => return SolverResult::Sat,
            "unsat" => return SolverResult::Unsat,
            "unknown" => return SolverResult::Unknown,
            _ => {}
        }
    }
    SolverResult::Unknown
}

fn run(script: &str) -> SolverResult {
    verdict_of(&run_output(script))
}

/// The value `(get-value (<name>))` printed for `name`, radix-agnostic.
fn printed_bv(outputs: &[String], name: &str) -> Option<u128> {
    let joined = outputs.join("\n");
    let key = format!("({name} ");
    let after = joined.find(&key).map(|at| &joined[at + key.len()..])?;
    let rest = after.trim_start().strip_prefix('#')?;
    let mut chars = rest.chars();
    let radix = match chars.next()? {
        'x' => 16,
        'b' => 2,
        _ => return None,
    };
    let digits: String = chars.take_while(char::is_ascii_alphanumeric).collect();
    u128::from_str_radix(&digits, radix).ok()
}

fn printed_bool(outputs: &[String], name: &str) -> Option<bool> {
    let joined = outputs.join("\n");
    let key = format!("({name} ");
    let after = joined.find(&key).map(|at| &joined[at + key.len()..])?;
    let after = after.trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// cargo-formal's `explain --blame` rewriting: `:produce-unsat-cores` after
/// the `(set-logic …)` line, every `(assert X)` becomes `(assert (! X :named
/// aK))`, `(get-unsat-core)` appended.
fn named_form(script: &str) -> String {
    let mut out = String::new();
    let mut next = 0usize;
    for line in script.lines() {
        if let Some(body) = line
            .strip_prefix("(assert ")
            .and_then(|rest| rest.strip_suffix(')'))
        {
            out.push_str(&format!("(assert (! {body} :named a{next}))\n"));
            next += 1;
        } else {
            out.push_str(line);
            out.push('\n');
            if line.starts_with("(set-logic ") {
                out.push_str("(set-option :produce-unsat-cores true)\n");
            }
        }
    }
    out.push_str("(get-unsat-core)\n");
    out
}

/// The names in the `(get-unsat-core)` answer.
fn printed_core(outputs: &[String]) -> Option<Vec<String>> {
    let line = outputs.iter().rev().find(|l| {
        l.starts_with('(') && !l.starts_with("(error") && !l.starts_with("((") && {
            let inner = l.trim_start_matches('(').trim_end_matches(')');
            inner.is_empty() || inner.split_whitespace().all(|w| w.starts_with('a'))
        }
    })?;
    let inner = line.trim_start_matches('(').trim_end_matches(')');
    Some(inner.split_whitespace().map(str::to_string).collect())
}

const M64: u128 = u64::MAX as u128;
const M63: u128 = (1u128 << 63) - 1;

fn signed64(v: u128) -> i64 {
    v as u64 as i64
}

// ---------------------------------------------------------------------------
// #P2b-24 — the minimal self-check reproducer (delta-debugged from the fuzz)
// ---------------------------------------------------------------------------

/// The delta-minimised fuzz script (seed 0 trial 44 of the pre-fix campaign,
/// `define-fun` form, `:random-seed 42`, no assertion at all): the selector
/// `(and (distinct v1 (ite (bvult v0 v0) v0 v0)) (bvsle v1 v1))` is outside
/// the unfixed fragment, so the `bvsub` under `t12` became a free bit-vector
/// while `t19` shared the same `ite` with a real circuit.
///
/// **Unfixed tree, debug build: panics** in the self-check on `BvSub(v0,
/// ite)` — "the model says 0x0 but the operation evaluates to
/// 0x8000000000000001" (with the seed; "0xfffffffffffffffe … 0x2" without
/// it).  Its release build publishes `sat` with a model violating
/// `t12 = v0 − ite(…)`.  There is nothing to satisfy, so the only thing to
/// pin on the fixed tree is that the answer is `sat` and nothing panics.
const SELFCHECK_PANIC: &str = "\
(set-logic QF_BV)
(set-option :random-seed 42)
(set-option :max-conflicts 200000)
(set-option :produce-models true)
(declare-const v0 (_ BitVec 64))
(declare-const v1 (_ BitVec 64))
(define-fun t12 () (_ BitVec 64) (bvsub v0 (ite (and (distinct v1 (ite (bvult v0 v0) v0 v0)) (bvsle v1 v1)) #x977d2041c3f87852 v0)))
(define-fun t16 () Bool (bvslt t12 v1))
(define-fun t19 () Bool (bvsle (ite (and (distinct v1 (ite (bvult v0 v0) v0 v0)) (bvsle v1 v1)) #x977d2041c3f87852 v0) v1))
(check-sat)
(get-value (v0 v1))
";

#[test]
fn p2b24_selfcheck_panic_script_is_sat_without_panicking() {
    let outcome = catch_unwind(AssertUnwindSafe(|| run_output(SELFCHECK_PANIC)));
    let outputs = outcome.unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        panic!("the unfixed tree's self-check panic is back: {message}")
    });
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
}

/// The same shape with the selector simplified to `(distinct v1 v0)` and the
/// two definitions spelled as declared constants, so `(get-value …)` prints
/// values rather than definition bodies (a `define-fun` name prints its body,
/// `#P2b-26`).
///
/// **Unfixed tree: `unknown`**, plain and named, with and without
/// `:random-seed` — the model gate refusing the free-bit model; it does *not*
/// panic there, because the self-check only fires on the atom checked last
/// and this ordering never puts the free copy there.  crates.io 0.3.3
/// answers it `sat` with a model in which `t12 = 0` where `v0 − K` is
/// required.  The fixed tree answers `sat` with a valid model.
const PANIC_MIN_DECLARED: &str = "\
(set-logic QF_BV)
(declare-const v0 (_ BitVec 64))
(declare-const v1 (_ BitVec 64))
(declare-const t12 (_ BitVec 64))
(declare-const t19 Bool)
(assert (= t12 (bvsub v0 (ite (distinct v1 v0) #x977d2041c3f87852 v0))))
(assert (= t19 (bvsle (ite (distinct v1 v0) #x977d2041c3f87852 v0) v1)))
(check-sat)
(get-value (v0 v1 t12 t19))
";

#[test]
fn p2b24_distinct_selector_shared_by_bvsub_and_bvsle_is_sat_with_a_valid_model() {
    let outputs = run_output(PANIC_MIN_DECLARED);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    let v0 = printed_bv(&outputs, "v0").unwrap_or(u128::MAX);
    let v1 = printed_bv(&outputs, "v1").unwrap_or(u128::MAX);
    let t12 = printed_bv(&outputs, "t12").unwrap_or(u128::MAX);
    let t19 = printed_bool(&outputs, "t19");
    assert!(v0 <= M64 && v1 <= M64 && t12 <= M64, "{outputs:?}");
    let sel = if v1 != v0 {
        0x977d_2041_c3f8_7852u128
    } else {
        v0
    };
    assert_eq!(
        t12,
        v0.wrapping_sub(sel) & M64,
        "t12 = v0 - ite: {outputs:?}"
    );
    assert_eq!(
        t19,
        Some(signed64(sel) <= signed64(v1)),
        "t19 = (bvsle ite v1): {outputs:?}"
    );
}

/// The named/core form of the same script answers the same verdict.
#[test]
fn p2b24_distinct_selector_named_form_agrees() {
    assert_eq!(run(&named_form(PANIC_MIN_DECLARED)), SolverResult::Sat);
}

// ---------------------------------------------------------------------------
// #P2b-24 — the cargo-formal conformance fixture c13 (unsat, so the free
// bit-vector shows up as a wrong verdict and not only as a panic)
// ---------------------------------------------------------------------------

/// `out = (ite (bvult a b) (ite (distinct a b) a b) b)` with the selector
/// pinned false by `(bvuge a b)` forces `out = b`, so `(distinct out b)` is
/// refutable.  The inner `ite`'s `distinct` selector made the whole outer
/// `ite` a free bit-vector on the unfixed tree.
///
/// **Unfixed tree: `unknown`**, plain and named, in a debug build too — the
/// model gate refusing the free-bit model, not the self-check (which never
/// sees this atom last).  crates.io 0.3.3 answers `sat`, the wrong verdict
/// the fixture pins (`u11_ite_selector_free_bits` there, `c13` in this
/// tree's notes).  Standard answer, and this tree's: `unsat`.
const C13: &str = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 64))
(declare-const b (_ BitVec 64))
(declare-const out (_ BitVec 64))
(declare-const twin (_ BitVec 64))
(assert (= twin (ite (bvult a b) a b)))
(assert (= out (ite (bvult a b) (ite (distinct a b) a b) b)))
(assert (bvuge a b))
(assert (distinct out b))
(check-sat)
";

#[test]
fn c13_ite_width64_selfcheck_is_unsat() {
    assert_eq!(run(C13), SolverResult::Unsat);
}

#[test]
fn c13_named_form_is_unsat_with_a_core_drawn_from_the_names() {
    let outputs = run_output(&named_form(C13));
    assert_eq!(verdict_of(&outputs), SolverResult::Unsat, "{outputs:?}");
    let core = printed_core(&outputs).unwrap_or_default();
    assert!(!core.is_empty(), "no core printed: {outputs:?}");
    for name in &core {
        assert!(
            ["a0", "a1", "a2", "a3"].contains(&name.as_str()),
            "core names {core:?} are not all named assertions: {outputs:?}"
        );
    }
}

/// The satisfiable twin of c13: with the selector pinned *true* the outer
/// `ite` selects the inner one, `a ≠ b` there selects `a`, and `out = a`.
#[test]
fn c13_selector_true_is_sat_with_out_equal_to_a() {
    let script = "\
(set-logic QF_BV)
(declare-const a (_ BitVec 64))
(declare-const b (_ BitVec 64))
(declare-const out (_ BitVec 64))
(assert (= out (ite (bvult a b) (ite (distinct a b) a b) b)))
(assert (bvult a b))
(check-sat)
(get-value (a b out))
";
    let outputs = run_output(script);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    let a = printed_bv(&outputs, "a").unwrap_or(u128::MAX);
    let b = printed_bv(&outputs, "b").unwrap_or(u128::MAX);
    let out = printed_bv(&outputs, "out").unwrap_or(u128::MAX);
    assert!(a < b, "{outputs:?}");
    assert_eq!(out, a, "{outputs:?}");
}

// ---------------------------------------------------------------------------
// #P2b-24 — the whole selector fragment cargo-formal emits
// ---------------------------------------------------------------------------

/// `(x, expected)` rows: `x = (ite <selector> #x01 #x02)` under pins that
/// make the selector's truth value known, then an assertion agreeing or
/// disagreeing with the branch that truth selects.  Every `distinct`/`xor`
/// /`=>`/Bool-`ite`/Bool-`=` selector here made the atom's whole term a free
/// bit-vector on the unfixed tree.
const FRAGMENT_ROWS: &[(&str, &str, SolverResult)] = &[
    (
        "distinct over BV, true",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (distinct a b) #x01 #x02)))(assert (= x #x02))",
        SolverResult::Unsat,
    ),
    (
        "distinct over BV, false",
        "(assert (= a #x03))(assert (= b #x03))\
         (assert (= x (ite (distinct a b) #x01 #x02)))(assert (= x #x02))",
        SolverResult::Sat,
    ),
    (
        "xor of two comparisons",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (xor (bvult a b) (= a b)) #x01 #x02)))(assert (= x #x02))",
        SolverResult::Unsat,
    ),
    (
        "implication, antecedent false",
        "(assert (= a #x05))(assert (= b #x04))\
         (assert (= x (ite (=> (bvult a b) (= a #x00)) #x01 #x02)))(assert (= x #x02))",
        SolverResult::Unsat,
    ),
    (
        "implication, consequent false",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (=> (bvult a b) (= a #x00)) #x01 #x02)))(assert (= x #x01))",
        SolverResult::Unsat,
    ),
    (
        "Bool ite as selector",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (ite (bvult a b) (= a #x03) (= b #x00)) #x01 #x02)))(assert (= x #x02))",
        SolverResult::Unsat,
    ),
    (
        "Bool equality (iff) as selector",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (= (bvult a b) (bvule a b)) #x01 #x02)))(assert (= x #x02))",
        SolverResult::Unsat,
    ),
    (
        "distinct over Bool as selector",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (distinct (bvult a b) (bvule a b)) #x01 #x02)))(assert (= x #x01))",
        SolverResult::Unsat,
    ),
    (
        "three-way distinct over BV, one pair equal",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (distinct a b #x03) #x01 #x02)))(assert (= x #x01))",
        SolverResult::Unsat,
    ),
    (
        "nested: distinct selector under an and under a not",
        "(assert (= a #x03))(assert (= b #x04))\
         (assert (= x (ite (not (and (distinct a b) (bvult a b))) #x01 #x02)))(assert (= x #x01))",
        SolverResult::Unsat,
    ),
];

#[test]
fn p2b24_every_selector_shape_in_the_fragment_is_decided() {
    for (name, body, expected) in FRAGMENT_ROWS {
        let script = format!(
            "(set-logic QF_BV)(declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))\
             (declare-const x (_ BitVec 8)){body}(check-sat)"
        );
        assert_eq!(run(&script), *expected, "{name}: {script}");
    }
}

/// A Bool variable that occurs *only* inside a selector still gets the
/// value the model needs, in **both** polarities: `x = 2` forces
/// `p = false`, `x = 1` forces `p = true` — the polarity that does *not*
/// coincide with the sort default, which the first version of this test
/// never checked (`#P2b-27`) — and pinning `p` true under `x = 2` makes the
/// formula unsatisfiable.
#[test]
fn p2b24_bool_variable_only_inside_a_selector_is_decided() {
    for (x_value, p_value) in [(2u128, false), (1u128, true)] {
        let sat = format!(
            "(set-logic QF_BV)(declare-const p Bool)(declare-const x (_ BitVec 8))\
             (assert (= x (ite p #x01 #x02)))(assert (= x #x{x_value:02x}))\
             (check-sat)(get-value (p x))"
        );
        let outputs = run_output(&sat);
        assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
        assert_eq!(printed_bv(&outputs, "x"), Some(x_value), "{outputs:?}");
        assert_eq!(
            printed_bool(&outputs, "p"),
            Some(p_value),
            "x = {x_value} forces p = {p_value}: {outputs:?}"
        );
    }
    let unsat = "\
(set-logic QF_BV)
(declare-const p Bool)
(declare-const x (_ BitVec 8))
(assert (= x (ite p #x01 #x02)))
(assert (= x #x02))
(assert p)
(check-sat)
";
    assert_eq!(run(unsat), SolverResult::Unsat);
}

/// A selector the bit-blaster genuinely cannot model — an arithmetic
/// comparison over `Int` variables — makes the atom *unmodelled*: the answer
/// is `unknown`, never a `sat` over free bits.  (This is the
/// `bv_atom_unmodelled` guard firing on a real input, which 0.3.4's changelog
/// had recorded as untested.)
#[test]
fn p2b24_selector_outside_the_fragment_is_unknown_not_sat() {
    let script = "\
(set-logic ALL)
(declare-const i Int)
(declare-const j Int)
(declare-const x (_ BitVec 8))
(assert (= x (ite (< i j) #x01 #x02)))
(assert (< i j))
(assert (= x #x02))
(check-sat)
";
    assert_ne!(run(script), SolverResult::Sat);
}

/// An uninterpreted application under a bit-vector operation is abstracted
/// at the leaf, not at the root: the adder over `(f a)` is real, so
/// `(bvadd (f a) #x01) = (f a)` is refutable at width 8 (no 8-bit value is a
/// fixed point of `+1`).
#[test]
fn p2b24_opaque_leaf_is_abstracted_at_the_leaf() {
    let script = "\
(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(assert (= (bvadd (f a) #x01) (f a)))
(check-sat)
";
    assert_eq!(run(script), SolverResult::Unsat);
}

// ---------------------------------------------------------------------------
// #P2b-25 — the false proof (delta-debugged from the fuzz, width 63)
// ---------------------------------------------------------------------------

/// Satisfiable — `v0 = 4, v1 = 0x20cb882730550eec, v2 = 1, v3 = 7, p0 =
/// true` is a witness — yet the unfixed tree answers `unsat`.  The selector
/// `t26` is pinned into the circuit by the outer solver's assignment of its
/// Tseitin variable, the embedded solver resolves against that pin, and the
/// conflict explanation blamed only the comparison atoms.  Every line is
/// load-bearing: `t12`'s definition is never asserted directly but adds the
/// `(= t12 …)` atom whose processing order exposes the gap.
const FALSE_PROOF_MIN: &str = "\
(set-logic QF_BV)
(set-option :random-seed 7)
(set-option :produce-models true)
(declare-const v0 (_ BitVec 63))
(declare-const v1 (_ BitVec 63))
(declare-const v2 (_ BitVec 63))
(declare-const v3 (_ BitVec 63))
(declare-const p0 Bool)
(define-fun t5 () Bool (bvult v2 v0))
(define-fun t12 () Bool (= #b111111111111111111111111111111111111111111111111111111111111100 (ite t5 v3 #b000000000000000000000000000000000000000000000000000000000001111)))
(define-fun t26 () Bool (and p0 p0 (= #b000000000000000000000000000000000000000000000000000000000000110 (bvsub (ite t5 v3 #b000000000000000000000000000000000000000000000000000000000001111) v2))))
(assert (bvuge #b000000000000000000000000000000000000000000000000000000000000010 v2))
(assert (bvuge (bvsub v2 (ite t26 #b000000000000000000000000000000000000000000000000000000000001111 (bvnot v3))) #b011111111111111111111111111111111111111111111111111111111111111))
(assert (bvult #b101101000011000111111101111101110001011111101011011110000001101 (concat ((_ extract 62 32) (bvnot v3)) ((_ extract 31 0) #b110011011011000100000000000000100100010111100000110011011100101))))
(check-sat)
(get-value (v0 v1 v2 v3 p0))
";

/// Reference evaluation of [`FALSE_PROOF_MIN`]'s three assertions.
fn false_proof_min_holds(v0: u128, v2: u128, v3: u128, p0: bool) -> bool {
    let fifteen = 15u128;
    let t5 = v2 < v0;
    let t7 = if t5 { v3 } else { fifteen };
    let t8 = t7.wrapping_sub(v2) & M63;
    let t26 = p0 && t8 == 6;
    let not_v3 = !v3 & M63;
    let t27 = if t26 { fifteen } else { not_v3 };
    let t28 = v2.wrapping_sub(t27) & M63;
    let a1 = 2 >= v2;
    let a2 = t28 >= ((1u128 << 62) - 1);
    let k = u128::from_str_radix(
        "101101000011000111111101111101110001011111101011011110000001101",
        2,
    )
    .unwrap_or(0);
    let c = u128::from_str_radix(
        "110011011011000100000000000000100100010111100000110011011100101",
        2,
    )
    .unwrap_or(0);
    let t23 = ((not_v3 >> 32) << 32) | (c & 0xffff_ffff);
    let a3 = k < t23;
    a1 && a2 && a3
}

#[test]
fn p2b25_reference_evaluator_accepts_the_known_witness() {
    assert!(false_proof_min_holds(4, 1, 7, true));
}

#[test]
fn p2b25_pinned_selector_conflict_is_not_a_false_proof() {
    let outputs = run_output(FALSE_PROOF_MIN);
    assert_eq!(
        verdict_of(&outputs),
        SolverResult::Sat,
        "a witness exists (v0 = 4, v2 = 1, v3 = 7, p0 = true): {outputs:?}"
    );
    let v0 = printed_bv(&outputs, "v0").unwrap_or(u128::MAX);
    let v2 = printed_bv(&outputs, "v2").unwrap_or(u128::MAX);
    let v3 = printed_bv(&outputs, "v3").unwrap_or(u128::MAX);
    let p0 = printed_bool(&outputs, "p0").unwrap_or(false);
    assert!(v0 <= M63 && v2 <= M63 && v3 <= M63, "{outputs:?}");
    assert!(
        false_proof_min_holds(v0, v2, v3, p0),
        "the published model does not satisfy the assertions: {outputs:?}"
    );
}

#[test]
fn p2b25_named_form_agrees_and_prints_no_core() {
    let outputs = run_output(&named_form(FALSE_PROOF_MIN));
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
}

/// The smallest shape of the same gap: the selector `c` is an outer atom
/// pinned into the circuit; `x = 1` under `c` and `x = 2` are inconsistent
/// *only* while `c` holds, so the solver must retract `c` and answer `sat`
/// with `c = false`.  (A conflict clause omitting `c` would have made the
/// search learn `¬(x = 2)` outright.)
#[test]
fn p2b25_conflict_under_a_pinned_selector_is_retractable() {
    let script = "\
(set-logic QF_BV)
(declare-const c Bool)
(declare-const a (_ BitVec 8))
(declare-const x (_ BitVec 8))
(assert (= c (bvult a #x10)))
(assert (= x (ite (bvult a #x10) #x01 #x02)))
(assert (or c (= a #x20)))
(assert (= x #x02))
(check-sat)
(get-value (c a x))
";
    let outputs = run_output(script);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bool(&outputs, "c"), Some(false), "{outputs:?}");
    assert_eq!(printed_bv(&outputs, "a"), Some(0x20), "{outputs:?}");
    assert_eq!(printed_bv(&outputs, "x"), Some(0x02), "{outputs:?}");
}

// ---------------------------------------------------------------------------
// #P2b-27 — a Boolean that occurs only as a selector is published from the
// circuit, in every shape the close-out review listed
// ---------------------------------------------------------------------------

/// `(x, expected p)` rows over selectors that contain a Boolean the outer
/// search never assigns.  Each script is satisfiable in exactly one
/// polarity of that Boolean, and the published model must print it — the
/// tree before `#P2b-27` printed the sort default `false` for every one of
/// them, a model that violates the first assertion (`p = false` selects
/// `#x02`, and `x = #x01` was asserted).
type SelectorRow = (&'static str, &'static str, &'static [(&'static str, bool)]);

const SELECTOR_ONLY_ROWS: &[SelectorRow] = &[
    (
        "xor of two free Booleans",
        "(set-logic QF_BV)(declare-const p Bool)(declare-const q Bool)(declare-const x (_ BitVec 8))\
         (assert (= x (ite (xor p q) #x01 #x02)))(assert (= x #x01))(assert (not q))\
         (check-sat)(get-value (p q x))",
        &[("p", true), ("q", false)],
    ),
    (
        "nested ite, both selectors free",
        "(set-logic QF_BV)(declare-const p Bool)(declare-const q Bool)(declare-const x (_ BitVec 8))\
         (assert (= x (ite p (ite q #x01 #x02) #x03)))(assert (= x #x01))\
         (check-sat)(get-value (p q x))",
        &[("p", true), ("q", true)],
    ),
    (
        "define-fun chain, the selector under a Bool definition",
        "(set-logic QF_BV)(declare-const p Bool)(declare-const x (_ BitVec 8))\
         (define-fun t1 () (_ BitVec 8) (ite p #x10 #x20))(define-fun t2 () Bool (= t1 #x10))\
         (assert t2)(assert (= x t1))(check-sat)(get-value (p x))",
        &[("p", true)],
    ),
    (
        "width 64",
        "(set-logic QF_BV)(declare-const p Bool)(declare-const x (_ BitVec 64))\
         (assert (= x (ite p #x0000000000000001 #x0000000000000002)))\
         (assert (= x #x0000000000000001))(check-sat)(get-value (p x))",
        &[("p", true)],
    ),
    (
        "the polarity that coincides with the default",
        "(set-logic QF_BV)(declare-const p Bool)(declare-const x (_ BitVec 8))\
         (assert (= x (ite p #x01 #x02)))(assert (= x #x02))(check-sat)(get-value (p x))",
        &[("p", false)],
    ),
];

#[test]
fn p2b27_every_selector_only_boolean_is_published_from_the_circuit() {
    for (name, script, expected) in SELECTOR_ONLY_ROWS {
        let outputs = run_output(script);
        assert_eq!(
            verdict_of(&outputs),
            SolverResult::Sat,
            "{name}: {outputs:?}"
        );
        for (var, value) in *expected {
            assert_eq!(
                printed_bool(&outputs, var),
                Some(*value),
                "{name}: the published value of {var} must be the circuit's: {outputs:?}"
            );
        }
    }
}

/// `(get-model)` agrees with `(get-value)`: the selector's `define-fun`
/// carries the circuit's value, not the default.
#[test]
fn p2b27_get_model_prints_the_selector_the_circuit_chose() {
    let script = "\
(set-logic QF_BV)
(declare-const p Bool)
(declare-const x (_ BitVec 8))
(assert (= x (ite p #x01 #x02)))
(assert (= x #x01))
(check-sat)
(get-model)
";
    let outputs = run_output(script);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    let joined = outputs.join("\n");
    assert!(
        joined.contains("(define-fun p () Bool true)"),
        "the model must define p as true: {outputs:?}"
    );
}

/// The `#P2b-27` script itself, verbatim — the pre-fix campaign's seed 1
/// trial 82, where the model gate let a free-bit-vector model through as
/// `sat` under `v0 = 0, v1 = #xff, p0 = false`.  It is unsatisfiable:
/// `t6` is `#x20` or a sign-extension of two bits, never `#xc3`, so `t8` is
/// false, `t10` true, `t11 = t9 = v0`, and `t16 = (bvsgt v0 v0)` cannot
/// hold.  The free bits are gone since `#P2b-24`; what this pins is the
/// verdict, and — on the satisfiable twin that replaces `t16` by
/// `(= t6 #x00)` — that the selector `p0`, which occurs in no outer clause,
/// is published with the value the circuit chose (`true`: `p0 = false`
/// gives `t6 = #x20`).
const P2B27_CAMPAIGN_PREFIX: &str = "\
(set-logic QF_BV)
(set-option :max-conflicts 200000)
(set-option :produce-models true)
(declare-const v0 (_ BitVec 8))
(declare-const v1 (_ BitVec 8))
(declare-const p0 Bool)
(define-fun t3 () (_ BitVec 2) ((_ extract 3 2) v0))
(define-fun t4 () (_ BitVec 8) ((_ sign_extend 6) t3))
(define-fun t6 () (_ BitVec 8) (ite p0 t4 #x20))
(define-fun t8 () Bool (= t6 #xc3))
(define-fun t9 () (_ BitVec 8) (ite t8 #xc3 v0))
(define-fun t10 () Bool (distinct t6 #xc3))
(define-fun t11 () (_ BitVec 8) (ite t10 t9 v1))
(define-fun t12 () (_ BitVec 8) (bvnot t9))
(define-fun t13 () (_ BitVec 8) (bvnot v0))
(define-fun t15 () (_ BitVec 8) (bvsub t9 #x0d))
(define-fun t16 () Bool (bvsgt t11 v0))
(define-fun t18 () Bool (= #xff t9))
(define-fun t19 () Bool (not t18))
";

#[test]
fn p2b27_the_campaign_script_is_decided_and_its_twin_publishes_the_selector() {
    let unsat = format!("{P2B27_CAMPAIGN_PREFIX}(assert t16)\n(assert t19)\n(check-sat)\n");
    assert_eq!(run(&unsat), SolverResult::Unsat);
    let twin = format!(
        "{P2B27_CAMPAIGN_PREFIX}(assert (= t6 #x00))\n(assert t19)\n(check-sat)\n\
         (get-value (v0 v1 p0))\n"
    );
    let outputs = run_output(&twin);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bool(&outputs, "p0"), Some(true), "{outputs:?}");
    let v0 = printed_bv(&outputs, "v0").unwrap_or(u128::MAX);
    assert_eq!(
        (v0 >> 2) & 0b11,
        0,
        "t4 = 0 needs bits 3..2 of v0 clear: {outputs:?}"
    );
}

// ---------------------------------------------------------------------------
// #P2b-28 — comparisons against the i64 boundary at widths 63 and 64
// ---------------------------------------------------------------------------

/// Whether `(op K v)` (or `(op v K)` when `constant_first` is false) has a
/// witness at `width`: a comparison against one constant is monotone in
/// `v`, so the boundary values decide it.
fn comparison_has_witness(op: &str, constant: u128, width: u32, constant_first: bool) -> bool {
    let m = if width == 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    let sign = 1u128 << (width - 1);
    let signed = |v: u128| -> i128 {
        if v & sign != 0 {
            (v as i128) - (1i128 << width)
        } else {
            v as i128
        }
    };
    let holds = |a: u128, b: u128| -> bool {
        match op {
            "bvult" => a < b,
            "bvule" => a <= b,
            "bvugt" => a > b,
            "bvuge" => a >= b,
            "bvslt" => signed(a) < signed(b),
            "bvsle" => signed(a) <= signed(b),
            other => panic!("unknown comparison {other}"),
        }
    };
    let candidates = [
        0,
        1,
        constant.wrapping_sub(1) & m,
        constant,
        (constant + 1) & m,
        m,
        sign,
        sign - 1,
    ];
    candidates.iter().any(|&v| {
        if constant_first {
            holds(constant, v)
        } else {
            holds(v, constant)
        }
    })
}

/// Every unsigned and signed comparison, both operand orders, against
/// `#x7fffffffffffffff`, `#x8000000000000000` and `#xffffffffffffffff` at
/// width 64 and against the corresponding width-63 boundaries: each one is
/// *decided*, the verdict matches the reference, and a `sat` witness
/// satisfies the comparison.  On the tree before `#P2b-28` the width-64
/// `bvult`/`bvule` rows against `#x7fffffffffffffff` panicked in a debug
/// build ("attempt to negate with overflow" from the arithmetic mirror of
/// the comparison) and answered `unknown` in release; `bvugt`/`bvuge`
/// desugar to the same atoms with the operands swapped.
#[test]
fn p2b28_every_boundary_comparison_is_decided_with_a_valid_witness() {
    let ops = ["bvult", "bvule", "bvugt", "bvuge", "bvslt", "bvsle"];
    let constants_64: [u128; 3] = [
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        u64::MAX as u128,
    ];
    let constants_63: [u128; 3] = [(1u128 << 62) - 1, 1u128 << 62, (1u128 << 63) - 1];
    for (width, constants) in [(64u32, constants_64), (63u32, constants_63)] {
        for op in ops {
            for constant in constants {
                for constant_first in [true, false] {
                    let literal = if width % 4 == 0 {
                        format!("#x{constant:0width$x}", width = (width / 4) as usize)
                    } else {
                        format!("#b{constant:0width$b}", width = width as usize)
                    };
                    let atom = if constant_first {
                        format!("({op} {literal} v)")
                    } else {
                        format!("({op} v {literal})")
                    };
                    let script = format!(
                        "(set-logic QF_BV)(declare-const v (_ BitVec {width}))(assert {atom})\
                         (check-sat)(get-value (v))"
                    );
                    let outcome = catch_unwind(AssertUnwindSafe(|| run_output(&script)));
                    let outputs = outcome.unwrap_or_else(|payload| {
                        let message = payload
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                            .unwrap_or_default();
                        panic!("{atom} at width {width} panicked: {message}")
                    });
                    let expected = if comparison_has_witness(op, constant, width, constant_first) {
                        SolverResult::Sat
                    } else {
                        SolverResult::Unsat
                    };
                    assert_eq!(
                        verdict_of(&outputs),
                        expected,
                        "{atom} at width {width}: {outputs:?}"
                    );
                    if expected == SolverResult::Sat {
                        let v = printed_bv(&outputs, "v").unwrap_or(u128::MAX);
                        let holds = comparison_has_witness(op, constant, width, constant_first)
                            && {
                                let m = (1u128 << width) - 1;
                                let sign = 1u128 << (width - 1);
                                let signed = |x: u128| -> i128 {
                                    if x & sign != 0 {
                                        (x as i128) - (1i128 << width)
                                    } else {
                                        x as i128
                                    }
                                };
                                let (a, b) = if constant_first {
                                    (constant, v)
                                } else {
                                    (v, constant)
                                };
                                v <= m
                                    && match op {
                                        "bvult" => a < b,
                                        "bvule" => a <= b,
                                        "bvugt" => a > b,
                                        "bvuge" => a >= b,
                                        "bvslt" => signed(a) < signed(b),
                                        _ => signed(a) <= signed(b),
                                    }
                            };
                        assert!(
                            holds,
                            "{atom} at width {width}: witness {v:#x} is wrong: {outputs:?}"
                        );
                    }
                }
            }
        }
    }
}

/// The cargo-formal fixture `c16_i64_max_bound_wide_comparison` (written
/// beside `c13`/`c14` for cargo-formal to carry): the one-liner itself.
#[test]
fn p2b28_c16_i64_max_bound_wide_comparison_is_sat() {
    let script = "\
(set-logic QF_BV)
(declare-const v (_ BitVec 64))
(assert (bvult #x7fffffffffffffff v))
(check-sat)
(get-value (v))
";
    let outputs = run_output(script);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    let v = printed_bv(&outputs, "v").unwrap_or(0);
    assert!(v > 0x7fff_ffff_ffff_ffff && v <= M64, "{outputs:?}");
}

// ---------------------------------------------------------------------------
// #P2b-29 — equality sharing between congruence closure and the circuit,
// both directions
// ---------------------------------------------------------------------------

/// EUF → bit-vector: a congruence between two opaque leaves *under* a
/// bit-vector operation reaches the circuit.  Every row answered `sat` on
/// the tree before `#P2b-29` (and on 0.3.3).
const EUF_TO_BV_ROWS: &[(&str, &str)] = &[
    (
        "bvadd over f(a), f(b) — the cargo-formal fixture c15",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))(assert (= a b))\
         (assert (distinct (bvadd (f a) #x01) (bvadd (f b) #x01)))(check-sat)",
    ),
    (
        "bvnot over f(a), f(b)",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))(assert (= a b))\
         (assert (distinct (bvnot (f a)) (bvnot (f b))))(check-sat)",
    ),
    (
        "f(a), f(b) inside two ite selectors",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))\
         (declare-const x (_ BitVec 8))(declare-const y (_ BitVec 8))(assert (= a b))\
         (assert (= x (ite (bvult (f a) #x10) #x01 #x02)))\
         (assert (= y (ite (bvult (f b) #x10) #x01 #x02)))(assert (distinct x y))(check-sat)",
    ),
    (
        "array selects under bvadd",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))(declare-const j (_ BitVec 8))(assert (= i j))\
         (assert (distinct (bvadd (select arr i) #x01) (bvadd (select arr j) #x01)))(check-sat)",
    ),
    (
        "the equality reached through a disjunction",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))(declare-const c (_ BitVec 8))\
         (assert (or (= a b) (= a c)))(assert (= b c))\
         (assert (distinct (bvadd (f a) #x01) (bvadd (f b) #x01)))(check-sat)",
    ),
];

#[test]
fn p2b29_congruence_under_a_bv_operation_is_refuted() {
    for (name, script) in EUF_TO_BV_ROWS {
        assert_eq!(run(script), SolverResult::Unsat, "{name}");
    }
}

/// Bit-vector → EUF: an equality the circuit *entails* between two
/// application arguments reaches congruence closure.  Every row answered
/// `sat` on the tree before `#P2b-29` (and on 0.3.3): nothing told EUF that
/// `x + 1 = y + 1` forces `x = y`.
const BV_TO_EUF_ROWS: &[(&str, &str)] = &[
    (
        "bvadd on both variables",
        "(set-logic QF_UFBV)(declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const x (_ BitVec 8))(declare-const y (_ BitVec 8))\
         (assert (= (bvadd x #x01) (bvadd y #x01)))(assert (distinct (g x) (g y)))(check-sat)",
    ),
    (
        "an opaque leaf as the argument: g(f(a)) vs g(f(b))",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))\
         (assert (= (bvadd (f a) #x01) (bvadd (f b) #x01)))\
         (assert (distinct (g (f a)) (g (f b))))(check-sat)",
    ),
    (
        "a variable against a constant argument",
        "(set-logic QF_UFBV)(declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const x (_ BitVec 8))(assert (= (bvadd x #x01) #x02))\
         (assert (distinct (g x) (g #x01)))(check-sat)",
    ),
    (
        "array indices",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))(declare-const j (_ BitVec 8))\
         (assert (= (bvadd i #x01) (bvadd j #x01)))(assert (distinct (select arr i) (select arr j)))\
         (check-sat)",
    ),
    (
        "arguments that occur in no bit-vector atom of their own",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))\
         (assert (= (bvadd (f a) #x01) (bvadd (f b) #x01)))\
         (assert (distinct (g (bvadd (f a) #x00)) (g (bvadd (f b) #x00))))(check-sat)",
    ),
    (
        "both directions in one script",
        "(set-logic QF_UFBV)(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))\
         (assert (= (bvadd a #x01) (bvadd b #x01)))\
         (assert (distinct (bvadd (g (f a)) #x01) (bvadd (g (f b)) #x01)))(check-sat)",
    ),
    (
        "the entailment under a case split",
        "(set-logic QF_UFBV)(declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const x (_ BitVec 8))(declare-const y (_ BitVec 8))(declare-const c Bool)\
         (assert (or c (= (bvadd x #x01) (bvadd y #x01))))(assert (=> c (= (bvsub x y) #x00)))\
         (assert (distinct (g x) (g y)))(check-sat)",
    ),
];

#[test]
fn p2b29_a_circuit_entailed_argument_equality_reaches_congruence() {
    for (name, script) in BV_TO_EUF_ROWS {
        assert_eq!(run(script), SolverResult::Unsat, "{name}");
    }
}

/// The satisfiable controls of both directions: the exchange must not turn
/// a coincidence of the circuit's model into a fact.  `(distinct (g x) (g
/// y))` over two free variables is the sharpest one — the first model has
/// `x = y = 0` by the phase default, and the pass must separate them rather
/// than answer `unknown`, let alone `unsat`.
#[test]
fn p2b29_satisfiable_controls_stay_sat_with_valid_models() {
    let free_pair = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 8)) (_ BitVec 8))
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(assert (distinct (g x) (g y)))
(check-sat)
(get-value (x y))
";
    let outputs = run_output(free_pair);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_ne!(
        printed_bv(&outputs, "x"),
        printed_bv(&outputs, "y"),
        "g(x) ≠ g(y) needs x ≠ y: {outputs:?}"
    );

    let constant_argument = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 8)) (_ BitVec 8))
(declare-const x (_ BitVec 8))
(assert (= (bvadd x #x01) #x03))
(assert (distinct (g x) (g #x01)))
(check-sat)
(get-value (x))
";
    let outputs = run_output(constant_argument);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bv(&outputs, "x"), Some(2), "{outputs:?}");

    let leaves_apart = "\
(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (distinct a b))
(assert (distinct (bvadd (f a) #x01) (bvadd (f b) #x01)))
(check-sat)
(get-value (a b))
";
    let outputs = run_output(leaves_apart);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_ne!(
        printed_bv(&outputs, "a"),
        printed_bv(&outputs, "b"),
        "{outputs:?}"
    );

    let split = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 8)) (_ BitVec 8))
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(declare-const c Bool)
(assert (or c (= (bvadd x #x01) (bvadd y #x01))))
(assert (distinct (g x) (g y)))
(check-sat)
(get-value (c x y))
";
    let outputs = run_output(split);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bool(&outputs, "c"), Some(true), "{outputs:?}");
    assert_ne!(
        printed_bv(&outputs, "x"),
        printed_bv(&outputs, "y"),
        "{outputs:?}"
    );
}

/// The named-core form blames only the two atoms the refutation rests on
/// (`(= c #x07)` is inert), in both directions.
#[test]
fn p2b29_named_cores_name_only_the_atoms_the_exchange_used() {
    for script in [
        "(set-logic QF_UFBV)(set-option :produce-unsat-cores true)\
         (declare-fun f ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const a (_ BitVec 8))(declare-const b (_ BitVec 8))(declare-const c (_ BitVec 8))\
         (assert (! (= a b) :named a0))\
         (assert (! (distinct (bvadd (f a) #x01) (bvadd (f b) #x01)) :named a1))\
         (assert (! (= c #x07) :named a2))(check-sat)(get-unsat-core)",
        "(set-logic QF_UFBV)(set-option :produce-unsat-cores true)\
         (declare-fun g ((_ BitVec 8)) (_ BitVec 8))\
         (declare-const x (_ BitVec 8))(declare-const y (_ BitVec 8))(declare-const c (_ BitVec 8))\
         (assert (! (= (bvadd x #x01) (bvadd y #x01)) :named a0))\
         (assert (! (distinct (g x) (g y)) :named a1))\
         (assert (! (= c #x07) :named a2))(check-sat)(get-unsat-core)",
    ] {
        let outputs = run_output(script);
        assert_eq!(verdict_of(&outputs), SolverResult::Unsat, "{outputs:?}");
        let mut core = printed_core(&outputs).unwrap_or_default();
        core.sort();
        assert_eq!(
            core,
            vec!["a0".to_string(), "a1".to_string()],
            "{outputs:?}"
        );
    }
}

/// Shared equalities are retracted with their scope: the refutation appears
/// and disappears with the `push`/`pop` that carries the equality, in both
/// directions, and a later satisfiable scope prints a valid model.
#[test]
fn p2b29_shared_equalities_are_retracted_with_their_scope() {
    let leaves = "\
(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (distinct (bvadd (f a) #x01) (bvadd (f b) #x01)))
(check-sat)
(push 1)
(assert (= a b))
(check-sat)
(pop 1)
(check-sat)
(assert (= a b))
(check-sat)
";
    let verdicts: Vec<String> = run_output(leaves)
        .into_iter()
        .filter(|l| matches!(l.as_str(), "sat" | "unsat" | "unknown"))
        .collect();
    assert_eq!(verdicts, ["sat", "unsat", "sat", "unsat"]);

    let arguments = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 8)) (_ BitVec 8))
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(assert (distinct (g x) (g y)))
(check-sat)
(push 1)
(assert (= (bvadd x #x01) (bvadd y #x01)))
(check-sat)
(pop 1)
(check-sat)
(push 1)
(assert (bvult x #x02))
(assert (bvult y #x02))
(check-sat)
(get-value (x y))
(pop 1)
(check-sat)
";
    let outputs = run_output(arguments);
    let verdicts: Vec<&str> = outputs
        .iter()
        .map(String::as_str)
        .filter(|l| matches!(*l, "sat" | "unsat" | "unknown"))
        .collect();
    assert_eq!(
        verdicts,
        ["sat", "unsat", "sat", "sat", "sat"],
        "{outputs:?}"
    );
    let x = printed_bv(&outputs, "x").unwrap_or(u128::MAX);
    let y = printed_bv(&outputs, "y").unwrap_or(u128::MAX);
    assert!(x < 2 && y < 2 && x != y, "{outputs:?}");
}

/// The non-convex shape: the circuit entails a *disjunction* of argument
/// equalities (`x ∈ {0, 1}`, `y = 0`, `z = 1`, so `x = y ∨ x = z`) and
/// neither disjunct alone, so no single entailed equality can be
/// propagated.  The lemma loop decides it anyway — EUF refutes each model's
/// partition, the lemma `x ≠ y ∨ …` then `x ≠ z ∨ …` moves the circuit, and
/// the two lemmas together are refuted — where the tree before `#P2b-29`
/// answered `sat`.
#[test]
fn p2b29_an_entailed_disjunction_of_equalities_is_refuted() {
    let script = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 8)) (_ BitVec 8))
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(declare-const z (_ BitVec 8))
(assert (bvule x #x01))
(assert (= y #x00))
(assert (= z #x01))
(assert (distinct (g x) (g y)))
(assert (distinct (g x) (g z)))
(check-sat)
";
    assert_eq!(run(script), SolverResult::Unsat);
}

// ---------------------------------------------------------------------------
// `#P2b-32` — a `select` nested under a bit-vector or arithmetic operator was
// never collected by the array-axiom walk (`array_axioms::collect_array_structure`
// descended through a hand-written child list naming only the Boolean
// connectives, `ite` and `Apply`), so no read-over-write instance was ever
// asserted for it and the read stayed a free leaf: a free bit-vector in the
// circuit, a free column in the tableau.  Every unsatisfiable row below
// answered `sat` on 0.3.3, on HEAD and on the tree carrying
// `#P2b-24`–`#P2b-29`, while the same read as a *direct* atom operand
// (`(distinct (select (store arr i #x05) i) #x05)`) was always refuted.  The
// model gate could not refuse the published models either — it had no arm
// for `select` and evaluated every one of them `Undetermined`.  Fixed by an
// exhaustive walk and by a read-over-write arm in the gate.
// ---------------------------------------------------------------------------

/// The bit-vector rows (QF_ABV): the review's a3, a5, a6, c7, c8, e2, e3.
const P2B32_BV_ROWS: &[(&str, &str)] = &[
    (
        "a3: a read-over-write hit under bvadd — the cargo-formal fixture c17",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))\
         (assert (distinct (bvadd (select (store arr i #x05) i) #x01) #x06))(check-sat)",
    ),
    (
        "a5: a variable stored value, both sides under bvadd",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))(declare-const v (_ BitVec 8))\
         (assert (distinct (bvadd (select (store arr i v) i) #x01) (bvadd v #x01)))(check-sat)",
    ),
    (
        "a6: the hit through an asserted index equality",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))(declare-const j (_ BitVec 8))(declare-const v (_ BitVec 8))\
         (assert (= i j))\
         (assert (distinct (bvadd (select (store arr i v) j) #x01) (bvadd v #x01)))(check-sat)",
    ),
    (
        "c7: the read under bvult",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))\
         (assert (bvult #x06 (select (store arr i #x05) i)))(check-sat)",
    ),
    (
        "c8: a read-over-write miss under bvadd, the base read pinned",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))(declare-const j (_ BitVec 8))(declare-const v (_ BitVec 8))\
         (assert (distinct i j))(assert (= (bvadd (select (store arr i v) j) #x01) #x06))\
         (assert (= (select arr j) #x07))(check-sat)",
    ),
    (
        "e2: the read under concat",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))\
         (assert (distinct (concat #x00 (select (store arr i #x05) i)) #x0005))(check-sat)",
    ),
    (
        "e3: the read under bvnot",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))\
         (assert (distinct (bvnot (select (store arr i #x05) i)) #xfa))(check-sat)",
    ),
    (
        "b4: the read feeding a variable that is then constrained",
        "(set-logic QF_ABV)(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))\
         (declare-const i (_ BitVec 8))(declare-const s (_ BitVec 8))\
         (assert (= s (bvadd (select (store arr i #x05) i) #x01)))(assert (distinct s #x06))\
         (check-sat)",
    ),
];

/// The integer twins (QF_AUFLIA): the review's a12, c3, d4 — the same walk,
/// the tableau's free column instead of the circuit's free leaf.
const P2B32_INT_ROWS: &[(&str, &str)] = &[
    (
        "a12: a read-over-write hit under +",
        "(set-logic QF_AUFLIA)(declare-const arr (Array Int Int))\
         (declare-const i Int)(declare-const j Int)(assert (= i j))\
         (assert (distinct (+ (select (store arr i 5) j) 1) 6))(check-sat)",
    ),
    (
        "c3: a read-over-write miss under +, the base read pinned",
        "(set-logic QF_AUFLIA)(declare-const arr (Array Int Int))\
         (declare-const i Int)(declare-const j Int)(declare-const v Int)\
         (assert (not (= i j)))(assert (= (+ (select (store arr i v) j) 1) 6))\
         (assert (= (select arr j) 7))(check-sat)",
    ),
    (
        "d4: the miss reached through a defined variable",
        "(set-logic QF_AUFLIA)(declare-const arr (Array Int Int))\
         (declare-const i Int)(declare-const j Int)(declare-const v Int)(declare-const y Int)\
         (assert (not (= i j)))(assert (= y (+ (select (store arr i v) j) 1)))(assert (= y 6))\
         (assert (= (select arr j) 7))(check-sat)",
    ),
];

#[test]
fn p2b32_read_over_write_under_a_bv_operation_is_refuted() {
    for (name, script) in P2B32_BV_ROWS {
        assert_eq!(run(script), SolverResult::Unsat, "{name}");
    }
}

#[test]
fn p2b32_read_over_write_under_an_arithmetic_operator_is_refuted() {
    for (name, script) in P2B32_INT_ROWS {
        assert_eq!(run(script), SolverResult::Unsat, "{name}");
    }
}

/// The satisfiable controls: a miss needs `i ≠ j`, and the published model
/// must show it; a hit that is consistent stays `sat`, and `(get-value)`
/// prints the read's value rather than its body.
#[test]
fn p2b32_satisfiable_controls_stay_sat_with_valid_models() {
    let miss_direct = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(declare-const j (_ BitVec 8))
(assert (distinct i j))
(assert (distinct (select (store arr i #x05) j) #x05))
(check-sat)
(get-value (i j))
";
    let outputs = run_output(miss_direct);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_ne!(
        printed_bv(&outputs, "i"),
        printed_bv(&outputs, "j"),
        "{outputs:?}"
    );

    let miss_under_bvadd = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(declare-const j (_ BitVec 8))
(assert (distinct (bvadd (select (store arr i #x05) j) #x01) #x06))
(check-sat)
(get-value (i j))
";
    let outputs = run_output(miss_under_bvadd);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_ne!(
        printed_bv(&outputs, "i"),
        printed_bv(&outputs, "j"),
        "the sum can differ from 6 only on a miss, which needs i ≠ j: {outputs:?}"
    );

    let hit = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(declare-const s (_ BitVec 8))
(assert (= s (bvadd (select (store arr i #x05) i) #x01)))
(check-sat)
(get-value (s (select (store arr i #x05) i)))
";
    let outputs = run_output(hit);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bv(&outputs, "s"), Some(6), "{outputs:?}");
    assert_eq!(
        printed_bv(&outputs, "(select (store arr i #x05) i)"),
        Some(5),
        "the read prints its value, not its body: {outputs:?}"
    );
}

/// cargo-formal's `explain --blame` form agrees and blames only the one
/// assertion.
#[test]
fn p2b32_named_form_agrees_with_a_core_drawn_from_the_names() {
    let script = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(declare-const c (_ BitVec 8))
(assert (distinct (bvadd (select (store arr i #x05) i) #x01) #x06))
(assert (= c #x07))
(check-sat)
";
    let outputs = run_output(&named_form(script));
    assert_eq!(verdict_of(&outputs), SolverResult::Unsat, "{outputs:?}");
    assert_eq!(
        printed_core(&outputs),
        Some(vec!["a0".to_string()]),
        "{outputs:?}"
    );
}

/// The read-over-write lemma is a consequence of the assertions in scope:
/// the refutation appears and disappears with the `push`/`pop` that carries
/// the index equality, and a later satisfiable scope prints a valid model.
#[test]
fn p2b32_the_lemma_follows_its_scope() {
    let script = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(declare-const j (_ BitVec 8))
(assert (distinct (bvadd (select (store arr i #x05) j) #x01) #x06))
(check-sat)
(push 1)
(assert (= i j))
(check-sat)
(pop 1)
(check-sat)
(get-value (i j))
(assert (= i j))
(check-sat)
";
    let outputs = run_output(script);
    let verdicts: Vec<&str> = outputs
        .iter()
        .map(String::as_str)
        .filter(|l| matches!(*l, "sat" | "unsat" | "unknown"))
        .collect();
    assert_eq!(verdicts, ["sat", "unsat", "sat", "unsat"], "{outputs:?}");
    assert_ne!(
        printed_bv(&outputs, "i"),
        printed_bv(&outputs, "j"),
        "{outputs:?}"
    );
}

/// `#P2b-26`, the `(get-value)` half: a bit-vector operator, a comparison,
/// an application whose value lives on a congruent application and a
/// `define-fun` name all print their **value**.  Before this every one of
/// them printed its body — `((bvadd v #x01) (bvadd v #x01))` — on 0.3.3 and
/// on every tree before the fix, because `Model::eval` folds only the
/// Boolean connectives and integer arithmetic.
#[test]
fn p2b26_get_value_folds_bit_vector_terms_comparisons_and_congruent_applications() {
    let bit_vectors = "\
(set-logic QF_BV)
(declare-const v (_ BitVec 8))
(assert (bvult #x7f v))
(assert (bvult v #x82))
(check-sat)
(get-value (v (bvadd v #x01) (bvult v #x81) (= v #x80)))
";
    let outputs = run_output(bit_vectors);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    let v = printed_bv(&outputs, "v").unwrap_or(u128::MAX);
    assert!(v == 0x80 || v == 0x81, "{outputs:?}");
    assert_eq!(
        printed_bv(&outputs, "(bvadd v #x01)"),
        Some((v + 1) & 0xff),
        "{outputs:?}"
    );
    assert_eq!(
        printed_bool(&outputs, "(bvult v #x81)"),
        Some(v < 0x81),
        "{outputs:?}"
    );
    assert_eq!(
        printed_bool(&outputs, "(= v #x80)"),
        Some(v == 0x80),
        "{outputs:?}"
    );

    let congruent = "\
(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (= a b))
(assert (= (bvadd (f a) #x01) (bvadd (f b) #x01)))
(assert (= (f a) #x07))
(check-sat)
(get-value (a b (f a) (f b)))
";
    let outputs = run_output(congruent);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bv(&outputs, "(f a)"), Some(7), "{outputs:?}");
    assert_eq!(
        printed_bv(&outputs, "(f b)"),
        Some(7),
        "f(b) = f(a) by congruence: {outputs:?}"
    );

    let defined = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(define-fun t () (_ BitVec 8) (bvadd x #x01))
(assert (= x #x01))
(check-sat)
(get-value (t x))
";
    let outputs = run_output(defined);
    assert_eq!(verdict_of(&outputs), SolverResult::Sat, "{outputs:?}");
    assert_eq!(printed_bv(&outputs, "x"), Some(1), "{outputs:?}");
    // The key is the term *as queried* — `t`, not the `define-fun` body the
    // parser inlined (SMT-LIB 2.6 §4.1.1, `#P2b-35`).  The value is the body's
    // value either way; it is the key that used to name a term the script
    // never asked about.
    assert_eq!(
        printed_bv(&outputs, "t"),
        Some(2),
        "the define-fun name keys its own value: {outputs:?}"
    );
    assert!(
        !outputs.iter().any(|line| line.contains("(bvadd x #x01)")),
        "the inlined body must not appear as a response key: {outputs:?}"
    );
}
