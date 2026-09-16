//! Regression tests for `#P2b-37` (array extensionality) and for `#P2b-36`'s
//! replacement of the array-constant *shadow flag* by a reserved symbol.
//!
//! # The extensionality hole (`#P2b-37`)
//!
//! `collect_array_structure` recorded an array-sorted (dis)equality only from
//! `TermKind::Eq`.  `distinct` over array-sorted operands fell into the
//! generic arm and contributed nothing, so `(distinct arr brr)` never got an
//! extensionality witness: the two arrays stayed free leaves the SAT core
//! could satisfy by fiat, and
//!
//! ```text
//! (distinct arr brr)
//! (= (select arr #b0) (select brr #b0))
//! (= (select arr #b1) (select brr #b1))
//! ```
//!
//! — unsatisfiable over a two-element index sort, and needing neither an array
//! constant nor a `store` — answered `sat` on 0.3.3 and on every tree before
//! this one.  A store's *own-index* read was the second hole: nothing collects
//! `select(store(b,i,v), i)` unless the script spells it, so
//! `(= (store ((as const A) #b1) i #b0) ((as const A) #b1))` had no read
//! anywhere and no family could fire.  The third was the witness index itself
//! never reaching select congruence, the fourth was two array terms shared
//! with the uninterpreted fragment never being compared at all, and the fifth
//! was an equality whose discriminating index lies *outside* both store
//! chains.
//!
//! # The reserved symbol (`#P2b-36`)
//!
//! An array constant has no term kind of its own; the parser interns it as an
//! `Apply`, and the *name* of that application was `(as const)` — a string a
//! script could also declare, through the quoted symbol `|(as const)|`.  The
//! solver used to notice the collision and switch the array-constant axiom off
//! for the rest of the script.  It is now interned under a name containing a
//! backslash instead, which SMT-LIB 2.6 section 3.1 excludes from both symbol
//! forms, so a script can declare `|(as const)|` as an ordinary function
//! *and* have its real array constants decided in the same breath.

use oxiz_solver::{Context, SolverResult};

/// Run a script and return the verdict of its final `check-sat`.
fn verdict(script: &str) -> SolverResult {
    let mut ctx = Context::new();
    let outputs = ctx.execute_script(script).unwrap_or_default();
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

/// Every `check-sat` verdict of a script, in order.
fn verdicts(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .unwrap_or_default()
        .into_iter()
        .filter(|line| matches!(line.trim(), "sat" | "unsat" | "unknown"))
        .collect()
}

/// A script header declaring two arrays over `(_ BitVec width)` in both
/// components.
fn header(width: u32) -> String {
    format!(
        "(set-logic QF_ABV)\n\
         (declare-const arr (Array (_ BitVec {width}) (_ BitVec {width})))\n\
         (declare-const brr (Array (_ BitVec {width}) (_ BitVec {width})))\n"
    )
}

/// Every index literal of `(_ BitVec width)`, in `#b…` spelling.
fn indices(width: u32) -> Vec<String> {
    (0..(1u32 << width))
        .map(|value| format!("#b{value:0w$b}", w = width as usize))
        .collect()
}

// ---------------------------------------------------------------------------
// (1a) `distinct` over array-sorted operands
// ---------------------------------------------------------------------------

/// `r3/ext/e10` and its width-2 twin `r3/ext2/f5`: two arrays asserted
/// `distinct` while *every* index of the (finite) domain is pinned equal.
///
/// Extensionality is the only way to refute it, and the witness only exists
/// once `distinct` records the pair.
#[test]
fn distinct_arrays_with_every_index_pinned_equal_are_unsat() {
    for width in [1, 2] {
        let pins: String = indices(width)
            .iter()
            .map(|index| format!("(assert (= (select arr {index}) (select brr {index})))\n"))
            .collect();
        let script = format!(
            "{}(assert (distinct arr brr))\n{pins}(check-sat)\n",
            header(width)
        );
        assert_eq!(
            verdict(&script),
            SolverResult::Unsat,
            "width {width}: every index pinned equal contradicts `distinct`"
        );
    }
}

/// The control: leave one index free and the same shape is satisfiable, so the
/// test above is not simply refusing every `distinct` over arrays.
#[test]
fn distinct_arrays_with_one_index_free_are_sat() {
    for width in [1, 2] {
        let all = indices(width);
        let pins: String = all
            .iter()
            .skip(1)
            .map(|index| format!("(assert (= (select arr {index}) (select brr {index})))\n"))
            .collect();
        let script = format!(
            "{}(assert (distinct arr brr))\n{pins}(check-sat)\n",
            header(width)
        );
        assert_eq!(
            verdict(&script),
            SolverResult::Sat,
            "width {width}: one free index leaves room for a witness"
        );
    }
}

/// `(not (= a b))` always reached the `Eq` arm and always worked; pinning it
/// keeps the two spellings of an array disequality in step.
#[test]
fn the_not_eq_spelling_agrees_with_distinct() {
    for width in [1, 2] {
        let pins: String = indices(width)
            .iter()
            .map(|index| format!("(assert (= (select arr {index}) (select brr {index})))\n"))
            .collect();
        let script = format!(
            "{}(assert (not (= arr brr)))\n{pins}(check-sat)\n",
            header(width)
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

// ---------------------------------------------------------------------------
// (1a) n-ary `distinct` and the cardinality argument
// ---------------------------------------------------------------------------

/// `n`-ary `distinct` is pairwise, so five pairwise-distinct arrays over
/// `(_ BitVec 1) -> (_ BitVec 1)` — a sort with exactly four inhabitants — is
/// unsatisfiable, while four is satisfiable.
///
/// Deciding it needs a witness for every one of the ten pairs, which is what
/// "all pairs of an n-ary `distinct`" buys.
#[test]
fn five_pairwise_distinct_arrays_over_a_four_element_sort_are_unsat() {
    let declarations: String = (1..=5)
        .map(|k| format!("(declare-const a{k} (Array (_ BitVec 1) (_ BitVec 1)))\n"))
        .collect();
    let script = format!(
        "(set-logic QF_ABV)\n{declarations}(assert (distinct a1 a2 a3 a4 a5))\n(check-sat)\n"
    );
    assert_eq!(verdict(&script), SolverResult::Unsat);
}

/// The control: three of them fit comfortably, and cheaply — the family must
/// decide the cardinality, not refuse every `distinct` over arrays.
#[test]
fn three_pairwise_distinct_arrays_over_a_four_element_sort_are_sat() {
    let declarations: String = (1..=3)
        .map(|k| format!("(declare-const a{k} (Array (_ BitVec 1) (_ BitVec 1)))\n"))
        .collect();
    let script =
        format!("(set-logic QF_ABV)\n{declarations}(assert (distinct a1 a2 a3))\n(check-sat)\n");
    assert_eq!(verdict(&script), SolverResult::Sat);
}

/// The *tight* control: four pairwise-distinct arrays exactly exhaust the
/// four-element sort, so the search has to enumerate the last assignment
/// rather than stumble on one.  It costs ~0.3 s in release and over a minute
/// in the debug profile the test suite runs under, which is why it is
/// `#[ignore]`d rather than run on every `cargo test`; the cheap three-array
/// control above covers the same direction on the default path.
#[test]
#[ignore = "exhausts the four-element array sort; ~75 s in the debug profile"]
fn four_pairwise_distinct_arrays_over_a_four_element_sort_are_sat() {
    let declarations: String = (1..=4)
        .map(|k| format!("(declare-const a{k} (Array (_ BitVec 1) (_ BitVec 1)))\n"))
        .collect();
    let script =
        format!("(set-logic QF_ABV)\n{declarations}(assert (distinct a1 a2 a3 a4))\n(check-sat)\n");
    assert_eq!(verdict(&script), SolverResult::Sat);
}

// ---------------------------------------------------------------------------
// (1b)(1c) store own-index reads and the witness-index congruence
// ---------------------------------------------------------------------------

/// `r3/ext/e04`: a `store` over an array constant asserted equal to that same
/// constant, with no `select` anywhere in the script.
///
/// The store writes `#b0` where the constant reads `#b1`, so it is unsat; the
/// only read that can show it is the store's *own* index.
#[test]
fn a_store_over_a_constant_equal_to_that_constant_is_unsat() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let one = format!("#b{:0w$b}", 1, w = width as usize);
        let zero = format!("#b{:0w$b}", 0, w = width as usize);
        let script = format!(
            "(set-logic QF_ABV)\n\
             (declare-const i (_ BitVec {width}))\n\
             (assert (= (store ((as const {sort}) {one}) i {zero}) ((as const {sort}) {one})))\n\
             (check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

/// `r3/ext/e05`: a `store` over a *free* array equal to an array constant
/// whose default differs from the stored value.  The discriminating index is
/// the store's own.
#[test]
fn a_store_whose_value_contradicts_the_constant_it_equals_is_unsat() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let one = format!("#b{:0w$b}", 1, w = width as usize);
        let zero = format!("#b{:0w$b}", 0, w = width as usize);
        let script = format!(
            "(set-logic QF_ABV)\n\
             (declare-const arr {sort})\n\
             (declare-const i (_ BitVec {width}))\n\
             (assert (= (store arr i {zero}) ((as const {sort}) {one})))\n\
             (check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

/// `r3/ext/e06`, the `#P2b-36` remainder: two array constants with different
/// defaults, asserted **equal**.  The script mentions neither `select` nor
/// `store`, so even the guard on the refinement loop had to learn to fire on a
/// bare array-sorted term.
#[test]
fn two_constants_with_different_defaults_are_not_equal() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let one = format!("#b{:0w$b}", 1, w = width as usize);
        let zero = format!("#b{:0w$b}", 0, w = width as usize);
        let script = format!(
            "(set-logic QF_ABV)\n\
             (assert (= ((as const {sort}) {one}) ((as const {sort}) {zero})))\n\
             (check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

/// The other polarity: two array constants with the *same* default, asserted
/// `distinct`.  Here the extensionality lemma itself does the refuting.
#[test]
fn two_constants_with_the_same_default_are_not_distinct() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let one = format!("#b{:0w$b}", 1, w = width as usize);
        let script = format!(
            "(set-logic QF_ABV)\n\
             (assert (distinct ((as const {sort}) {one}) ((as const {sort}) {one})))\n\
             (check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

/// Two array constants with different defaults asserted `distinct` stay
/// satisfiable — the family must decide, not refuse.
#[test]
fn two_constants_with_different_defaults_are_distinct() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let one = format!("#b{:0w$b}", 1, w = width as usize);
        let zero = format!("#b{:0w$b}", 0, w = width as usize);
        let script = format!(
            "(set-logic QF_ABV)\n\
             (assert (distinct ((as const {sort}) {one}) ((as const {sort}) {zero})))\n\
             (check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Sat, "width {width}");
    }
}

/// A store chain whose writes cover the *whole* index domain, asserted equal
/// to an array constant it contradicts.
///
/// The off-chain Skolem index cannot help here — there is no index outside a
/// chain that writes every one of them, and the cardinality guard refuses to
/// mint one — so the refutation rests entirely on the store's own-index reads
/// and the congruence that carries them across the equality.
#[test]
fn a_chain_covering_the_domain_is_refuted_by_its_own_index_reads() {
    let script = "(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(assert (= (store (store arr #b0 #b0) #b1 #b0) ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))
(check-sat)
";
    assert_eq!(verdict(script), SolverResult::Unsat);
}

/// The control: the same chain writing the constant's own default is
/// satisfiable, so the test above is not refusing every covering chain.
#[test]
fn a_chain_covering_the_domain_with_matching_writes_is_sat() {
    let script = "(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(assert (= (store (store arr #b0 #b1) #b1 #b1) ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))
(check-sat)
";
    assert_eq!(verdict(script), SolverResult::Sat);
}

// ---------------------------------------------------------------------------
// (1d) foreign pairs
// ---------------------------------------------------------------------------

/// `(distinct (f arr) (f brr))` with every index read pinned equal: no array
/// (dis)equality atom occurs anywhere, so the extensionality family has
/// nothing to fire on and only the ext rule for *shared* array terms closes it.
#[test]
fn distinct_applications_of_extensionally_equal_arrays_are_unsat() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let pins: String = indices(width)
            .iter()
            .map(|index| format!("(assert (= (select arr {index}) (select brr {index})))\n"))
            .collect();
        let script = format!(
            "(set-logic QF_AUFBV)\n\
             (declare-fun f ({sort}) (_ BitVec {width}))\n\
             (declare-const arr {sort})\n\
             (declare-const brr {sort})\n\
             {pins}(assert (distinct (f arr) (f brr)))\n(check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

/// The control: leave one index free and the two arrays may differ, so the
/// applications may too.
#[test]
fn distinct_applications_of_possibly_different_arrays_are_sat() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let pins: String = indices(width)
            .iter()
            .skip(1)
            .map(|index| format!("(assert (= (select arr {index}) (select brr {index})))\n"))
            .collect();
        let script = format!(
            "(set-logic QF_AUFBV)\n\
             (declare-fun f ({sort}) (_ BitVec {width}))\n\
             (declare-const arr {sort})\n\
             (declare-const brr {sort})\n\
             {pins}(assert (distinct (f arr) (f brr)))\n(check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Sat, "width {width}");
    }
}

// ---------------------------------------------------------------------------
// (1e) arrays of arrays
// ---------------------------------------------------------------------------

/// A witness read of an array-sorted range is itself an array term, so the
/// fixpoint loop picks the new pair up on the next round.
#[test]
fn distinct_arrays_of_arrays_with_every_entry_pinned_are_unsat() {
    for width in [1, 2] {
        let inner = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let outer = format!("(Array (_ BitVec {width}) {inner})");
        let mut pins = String::new();
        for outer_index in indices(width) {
            for inner_index in indices(width) {
                pins.push_str(&format!(
                    "(assert (= (select (select aa {outer_index}) {inner_index}) \
                     (select (select bb {outer_index}) {inner_index})))\n"
                ));
            }
        }
        let script = format!(
            "(set-logic QF_ABV)\n\
             (declare-const aa {outer})\n\
             (declare-const bb {outer})\n\
             (assert (distinct aa bb))\n{pins}(check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

// ---------------------------------------------------------------------------
// The off-chain Skolem index, and its cardinality guard
// ---------------------------------------------------------------------------

/// `(= (store ((as const A) #b0) i #b1) ((as const A) #b1))` over a
/// two-element index sort: the two sides agree only at `i`, and an index other
/// than `i` exists, so it is unsat.  No index in the formula names that other
/// index; the pair's off-chain Skolem index does.
#[test]
fn a_store_over_a_constant_equal_to_another_constant_is_unsat() {
    for width in [1, 2] {
        let sort = format!("(Array (_ BitVec {width}) (_ BitVec {width}))");
        let one = format!("#b{:0w$b}", 1, w = width as usize);
        let zero = format!("#b{:0w$b}", 0, w = width as usize);
        let script = format!(
            "(set-logic QF_ABV)\n\
             (declare-const i (_ BitVec {width}))\n\
             (assert (= (store ((as const {sort}) {zero}) i {one}) ((as const {sort}) {one})))\n\
             (check-sat)\n"
        );
        assert_eq!(verdict(&script), SolverResult::Unsat, "width {width}");
    }
}

/// The cardinality guard's tightest case: a `Bool`-indexed array has exactly
/// two indices, so a chain of *two* writes can cover the whole domain and no
/// off-chain index need exist.  The shape below is satisfiable (write both
/// indices to the constant's own default) and must stay so — minting a Skolem
/// index constrained off both writes would make it `unsat`.
#[test]
fn a_bool_indexed_chain_covering_the_domain_stays_sat() {
    let script = "(set-logic ALL)
(declare-const i Bool)
(declare-const j Bool)
(assert (= (store (store ((as const (Array Bool Int)) 0) i 1) j 1)
           ((as const (Array Bool Int)) 1)))
(check-sat)
";
    assert_eq!(verdict(script), SolverResult::Sat);
}

/// One write over a `Bool`-indexed array leaves the other index uncovered, so
/// the same shape with a single `store` is unsat — the guard admits the rule
/// exactly when the chain is shorter than the domain.
#[test]
fn a_bool_indexed_single_write_is_refuted() {
    let script = "(set-logic ALL)
(declare-const i Bool)
(assert (= (store ((as const (Array Bool Int)) 0) i 1) ((as const (Array Bool Int)) 1)))
(check-sat)
";
    assert_eq!(verdict(script), SolverResult::Unsat);
}

/// An *uninterpreted* index sort has no known cardinality, so the rule must
/// not fire at all: the shape below is satisfiable (the sort may have exactly
/// one element) and answering `unsat` would be a wrong answer.
#[test]
fn an_uninterpreted_index_sort_is_never_assumed_large() {
    let script = "(set-logic ALL)
(declare-sort U 0)
(declare-const i U)
(assert (= (store ((as const (Array U Int)) 0) i 1) ((as const (Array U Int)) 1)))
(check-sat)
";
    assert_ne!(
        verdict(script),
        SolverResult::Unsat,
        "a one-element index sort satisfies this"
    );
}

/// Two writes over a two-element index sort *can* cover the whole domain, so
/// the off-chain index is only available when the two write indices coincide —
/// which congruence at the chain's own indices is what forces here.  The
/// lemma is therefore guarded by `(distinct i0 i1)`, and this pair is refuted
/// through that guard.
#[test]
fn a_two_write_chain_over_a_two_element_domain_is_refuted_through_the_guard() {
    let script = "(set-logic QF_ABV)
(declare-const i0 (_ BitVec 1))
(declare-const i1 (_ BitVec 1))
(assert (= (store (store ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1) i1 #b1) i0 #b0)
           ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))
(check-sat)
";
    assert_eq!(verdict(script), SolverResult::Unsat);
}

/// The control: the same two-write chain *can* equal a constant when its two
/// writes cover the domain with the constant's own default, so the guarded
/// rule must not fire unconditionally.
#[test]
fn a_two_write_chain_covering_the_domain_stays_sat() {
    let script = "(set-logic QF_ABV)
(declare-const i0 (_ BitVec 1))
(declare-const i1 (_ BitVec 1))
(assert (= (store (store ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1) i1 #b0) i0 #b0)
           ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))
(check-sat)
";
    assert_eq!(verdict(script), SolverResult::Sat);
}

// ---------------------------------------------------------------------------
// push / pop around a witness lemma
// ---------------------------------------------------------------------------

/// A witness lemma asserted inside a scope must not survive the `pop` as a
/// stale dedup entry: the same goal has to be decided again afterwards.
#[test]
fn a_witness_lemma_does_not_survive_a_pop() {
    let script = "(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))
(push 1)
(assert (distinct arr brr))
(assert (= (select arr #b0) (select brr #b0)))
(assert (= (select arr #b1) (select brr #b1)))
(check-sat)
(pop 1)
(check-sat)
(push 1)
(assert (distinct arr brr))
(assert (= (select arr #b0) (select brr #b0)))
(assert (= (select arr #b1) (select brr #b1)))
(check-sat)
(pop 1)
";
    assert_eq!(verdicts(script), vec!["unsat", "sat", "unsat"]);
}

// ---------------------------------------------------------------------------
// The reserved symbol (`#P2b-36`)
// ---------------------------------------------------------------------------

/// A user-declared `|(as const)|` is an ordinary uninterpreted function — its
/// applications carry no array semantics — while a *real* array constant in
/// the same script is still decided.
#[test]
fn a_user_declared_as_const_is_an_ordinary_function() {
    let ordinary = "(set-logic QF_AUFBV)
(declare-fun |(as const)| ((_ BitVec 8)) (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (select (|(as const)| #x03) i) #x03))
(check-sat)
";
    assert_eq!(
        verdict(ordinary),
        SolverResult::Sat,
        "the user's function says nothing about its reads"
    );

    let mixed = "(set-logic QF_AUFBV)
(declare-fun |(as const)| ((_ BitVec 8)) (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (select (|(as const)| #x03) i) #x03))
(assert (= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x05) i) #x06))
(check-sat)
";
    assert_eq!(
        verdict(mixed),
        SolverResult::Unsat,
        "the real array constant is still decided in the same script"
    );
}

/// A backslash inside a quoted symbol is a lexical error (SMT-LIB 2.6
/// section 3.1), which is what reserves the array constant's internal name.
#[test]
fn a_backslash_in_a_quoted_symbol_is_a_lexical_error() {
    let mut ctx = Context::new();
    let error = ctx
        .execute_script("(set-logic QF_UF)\n(declare-const |a\\b| Bool)\n(check-sat)\n")
        .err()
        .map(|err| err.to_string())
        .unwrap_or_default();
    assert!(
        error.contains("backslash"),
        "expected a lexical error naming the backslash, got: {error}"
    );
}

/// `(reset)` needs no special handling for the array constant any more: the
/// recognition no longer depends on any sticky per-solver flag, so a script
/// that declares `|(as const)|`, resets, and then uses a real array constant
/// is decided.
#[test]
fn reset_needs_no_array_constant_bookkeeping() {
    let script = "(set-logic QF_AUFBV)
(declare-fun |(as const)| ((_ BitVec 8)) (Array (_ BitVec 8) (_ BitVec 8)))
(check-sat)
(reset)
(set-logic QF_ABV)
(declare-const i (_ BitVec 8))
(assert (= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x05) i) #x06))
(check-sat)
";
    let all = verdicts(script);
    assert_eq!(
        all.last().map(String::as_str),
        Some("unsat"),
        "the array constant is decided after a reset: {all:?}"
    );
}
