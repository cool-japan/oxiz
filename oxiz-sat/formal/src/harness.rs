//! Harnesses over the **public** `oxiz-sat` literal API.
//!
//! Nothing in this file is a copy of `oxiz-sat`: every target is called
//! through the real crate's public interface (`oxiz_sat::{Lit, Var, LBool}`,
//! re-exported from the private module `oxiz-sat/src/literal.rs` at
//! `oxiz-sat/src/lib.rs:227`), and `cargo-formal` lowers the reachable bodies
//! of that crate into the verification conditions (19 dependency bodies
//! lowered, all 19 reachable, in the run recorded in `README.md`).
//!
//! `Lit` packs a variable index and a sign into one `u32` as
//! `index << 1 | sign`, and `to_dimacs`/`from_dimacs` carry that packing
//! across the DIMACS boundary. Both encodings are total only for
//! `index <= Var::MAX_INDEX` (`= (1 << 31) - 2`), which is simultaneously the
//! largest index the shift can hold and the largest index whose DIMACS form
//! `index + 1` has an `i32` negation. The pair
//! [`lit_pos_roundtrip_bounded_harness`] / [`lit_pos_roundtrip_unbounded_harness`]
//! states exactly that: the round trip under the bound, and the bound's
//! necessity without it.
//!
//! Every entry point states the property it raises and the **measured** L1
//! verdict for it; `EXPECTED.toml` at the package root mirrors the same table
//! machine-readably, and `README.md` names the run that produced it.
//!
//! | Specimen | What it exercises | Measured outcomes |
//! |---|---|---|
//! | `Lit::pos`/`neg`/`var`/`code`/`from_code`/`negate` | the `index << 1 \| sign` packing and the bound it needs | `proved`, `refuted` |
//! | `Lit::from_dimacs`/`to_dimacs`/`try_to_dimacs` | the DIMACS boundary, including the inputs that have no literal | `proved`, `refuted` |
//! | `LBool` | the three-valued assignment the CDCL loop reads | `unknown` (the solver, not the encoder -- see `README.md`) |
//!
//! Values are compared through `.code()`, `.var().0` and the `is_*`
//! predicates rather than with `==` on `Lit`/`Var`/`LBool`. That is
//! deliberate: `==` on those types goes through their *derived* `PartialEq`
//! impls, which only have MIR in the module set if dependency-body lowering
//! reaches derive-generated impls. Writing the properties over the `u32` and
//! `bool` observers makes every harness independent of that question, and
//! costs nothing -- the packing is the property.
//!
//! A bare `debug_assert!` lowers to `core::panicking::panic`, so the property
//! key for one is `panic`, not `assert`. Six of the measured rows below carry
//! that key -- five raised by `debug_assert!`s and one by `to_dimacs`'s
//! documented unconditional panic -- and they are the interesting ones: they
//! are the checks the upstream fix added, and both refutations are among
//! them.

use oxiformal::prelude::*;

// A `#[harness]` body exists only under `formal` or under
// `all(test, oxiformal_runtime_checks)` (see `oxiformal_macros::harness`). In
// the plain build these names have no harness using them, and the
// `plain_tests` module below is compiled only under `cargo test`.
#[cfg_attr(
    not(any(formal, all(test, oxiformal_runtime_checks))),
    allow(unused_imports)
)]
use oxiz_sat::{LBool, Lit, Var};

/// Properties: `assert` (5 sites), `panic` (1 site -- the `debug_assert!`s in
/// `Var::new` `../src/literal.rs:36` and `Lit::pos` `:61` share one
/// obligation, because both expand at the same `core` macro site) and
/// `shift-overflow` (3 sites). **Measured L1 verdicts: all proved** -- 9
/// obligations, nothing but `proved`.
///
/// Under the documented bound, `Lit::pos` is injective and `Lit::var` inverts
/// it: the literal keeps the index, reads back positive, and its raw code is
/// exactly `index << 1`. The last conjunct is what makes this injectivity and
/// not just "the accessor agrees with the constructor".
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn lit_pos_roundtrip_bounded_harness() {
    let index: u32 = any();
    assume(index <= Var::MAX_INDEX);
    let var = Var::new(index);
    let lit = Lit::pos(var);
    assert(lit.var().0 == index);
    assert(lit.is_pos());
    assert(!lit.is_neg());
    assert(lit.sign());
    assert(lit.code() == index << 1);
}

/// Properties: `assert` (5 sites), `panic` (1 site -- the `debug_assert!`s of
/// `Var::new`, `Lit::neg` `../src/literal.rs:73` and `Lit::pos` `:61`) and
/// `shift-overflow` (4 sites). **Measured L1 verdicts: all proved** -- 10
/// obligations, nothing but `proved`.
///
/// The negative constructor is the positive one with bit 0 set: it keeps the
/// index, reads back negative, and `negate` maps it onto `Lit::pos` of the
/// same variable.
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn lit_neg_roundtrip_bounded_harness() {
    let index: u32 = any();
    assume(index <= Var::MAX_INDEX);
    let var = Var::new(index);
    let lit = Lit::neg(var);
    assert(lit.var().0 == index);
    assert(lit.is_neg());
    assert(!lit.sign());
    assert(lit.negate().code() == Lit::pos(var).code());
    assert(lit.code() == ((index << 1) | 1));
}

/// Properties: `panic` (the `debug_assert!(var.0 <= Var::MAX_INDEX)` at
/// `../src/literal.rs:61`), `assert` and `shift-overflow` (2 sites).
/// **Measured L1 verdicts: `panic` refuted**, counterexample
/// `index = 4294967295` (`u32::MAX`); **`assert` proved**, and both
/// `shift-overflow` obligations proved.
///
/// This is [`lit_pos_roundtrip_bounded_harness`] with the precondition
/// removed, and it is the harness that says the bound is *necessary* rather
/// than decorative. It builds the variable with the public tuple field
/// (`Var(index)`, `../src/literal.rs:9`) instead of `Var::new`, precisely so
/// that the first obligation the encoder reaches is `Lit::pos`'s own check and
/// not the constructor's -- the packing is what is under test here.
///
/// The split verdict is the point. The encoder raises the `debug_assert!` as
/// its own obligation and then continues along the edge where it holds, so the
/// round trip *is* proved for every index the constructor will now accept, and
/// the refutation is exactly the statement "an index above the bound can reach
/// this constructor". Above the bound `var.0 << 1` drops the top bit (Rust's
/// `<<` checks the *shift amount*, never the value), so before the fix
/// recorded in this working tree `Lit::pos(Var(1 << 31))` denoted **variable
/// 0**, positively, and every clause built from it constrained the wrong
/// variable. The reported counterexample `u32::MAX` is one of `2^31 + 1`
/// witnesses; the smallest is `Var::MAX_INDEX + 1`, and `plain_tests` runs
/// both.
///
/// Runtime-checks build: slightly over half of all `u32` are witnesses, so the
/// first draw fails with overwhelming probability -- marked `#[should_panic]`,
/// and measured to panic.
#[harness]
#[should_panic]
fn lit_pos_roundtrip_unbounded_harness() {
    let index: u32 = any();
    let var = Var(index);
    let lit = Lit::pos(var);
    assert(lit.var().0 == index);
}

/// Properties: `assert` (4 sites) and `shift-overflow` (1 site). **Measured L1
/// verdicts: all proved** -- 5 obligations, nothing but `proved`.
///
/// `negate` is `code ^ 1`: applying it twice is the identity, it never changes
/// the variable, and it always flips the sign. Stated over a raw code rather
/// than over a constructed literal, so it covers all `2^32` representable
/// literals -- including the ones above [`Var::MAX_INDEX`] that no constructor
/// will hand out any more.
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn lit_negate_involution_harness() {
    let code: u32 = any();
    let lit = Lit::from_code(code);
    let flipped = lit.negate();
    assert(flipped.negate().code() == lit.code());
    assert(flipped.var().0 == lit.var().0);
    assert(flipped.is_pos() != lit.is_pos());
    assert(flipped.is_neg() == lit.is_pos());
}

/// Properties: `assert` (3 sites) and `shift-overflow` (2 sites). **Measured
/// L1 verdicts: all proved** -- 5 obligations, nothing but `proved`.
///
/// `from_code`/`code` are inverse, `index` is the code widened to `usize` (a
/// zero-extending cast on every supported target, so no information is lost),
/// and `var` is the code's high 31 bits. Together these pin the whole raw
/// representation, which is what the solver's watch lists and assignment
/// arrays are indexed by.
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn lit_code_roundtrip_harness() {
    let code: u32 = any();
    let lit = Lit::from_code(code);
    assert(lit.code() == code);
    assert(lit.index() == code as usize);
    assert(lit.var().0 == (code >> 1));
}

/// Properties: `assert` (3 sites), `panic` (2 sites: `from_dimacs`'s
/// `debug_assert!` at `../src/literal.rs:89` and `to_dimacs`'s unconditional
/// panic at `:113`), `neg-overflow` (the `-magnitude` of `try_to_dimacs`,
/// `:129`), `arith-overflow` (2 sites) and `shift-overflow` (3 sites).
/// **Measured L1 verdicts: all proved** -- 11 obligations, nothing but
/// `proved`.
///
/// Under the two documented preconditions -- DIMACS has no literal `0`, and
/// `i32::MIN` has no positive counterpart -- `to_dimacs` inverts `from_dimacs`
/// exactly, the sign survives, and the variable index is `|d| - 1`.
///
/// Both `assume`s together keep the index inside the bound, so this harness
/// was already total before the upstream fix -- the vendored pre-fix twin
/// (`examples/ecosystem/oxiz-sat-lit`) proves the same round trip. What is new
/// here is that the two `panic` sites the fix added are *also* proved: the
/// preconditions are now checked, and on this domain they hold. The harness
/// that shows the fix changing an answer is
/// [`dimacs_negation_harness`], which drops the second `assume`.
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn dimacs_roundtrip_harness() {
    let dimacs: i32 = any();
    assume(dimacs != 0);
    assume(dimacs != i32::MIN);
    let lit = Lit::from_dimacs(dimacs);
    assert(lit.to_dimacs() == dimacs);
    assert(lit.is_pos() == (dimacs > 0));
    assert(lit.var().0 == (dimacs.unsigned_abs() - 1));
}

/// Property: `panic`, 2 sites. **Measured L1 verdict: refuted**,
/// counterexample `dimacs = -2147483648` (`i32::MIN`) -- the
/// `debug_assert!(lit != 0 && lit != i32::MIN)` at `../src/literal.rs:89`.
/// The **other** `panic` site, `to_dimacs`'s own unconditional panic at
/// `:113`, is **proved**, and so are the `neg-overflow`, the `arith-overflow`
/// and the three `shift-overflow` obligations: 6 of this harness's 7
/// obligations are proved.
///
/// This is [`dimacs_roundtrip_harness`] with the `i32::MIN` precondition
/// removed, so that the one input DIMACS cannot represent is inside the
/// harness's domain. Two facts are measured here, and together they are the
/// upstream fix:
///
/// 1. `from_dimacs(i32::MIN)` is now **rejected**. It used to build
///    `Lit::neg(Var(0x7fff_ffff))` without complaint.
/// 2. `to_dimacs` is now **total on every literal this harness can produce**.
///    It used to compute `(0x7fff_ffff + 1) as i32 == i32::MIN` and negate it,
///    which overflowed -- the vendored pre-fix twin measures exactly that as
///    `neg-overflow = refuted`. Here `neg-overflow` and the unconditional
///    `panic` of `:113` are both proved, because the encoder continues along
///    the edge where `from_dimacs`'s assertion holds, and on that edge the
///    index is at most [`Var::MAX_INDEX`].
///
/// So `to_dimacs`'s unconditional panic is *not* what the refuted row reports,
/// and it is not reachable from any public constructor. `plain_tests` states
/// it concretely instead, through `Lit::from_code`, which is the only way to
/// reach it at all.
///
/// Runtime-checks build: one witness out of `2^32`, so 256 uniform draws
/// essentially never find it -- narrow witness, deliberately **not** marked
/// `#[should_panic]`. The generated test passes, which is not evidence of
/// correctness; that is what the L1 verdict is for. Measured: it passes.
#[harness]
fn dimacs_negation_harness() {
    let dimacs: i32 = any();
    assume(dimacs != 0);
    let lit = Lit::from_dimacs(dimacs);
    let _ = lit.to_dimacs();
}

/// Properties: `assert` (5 sites), `panic` (the `debug_assert!` of the
/// `from_dimacs` round trip), `neg-overflow`, `arith-overflow` and
/// `shift-overflow` (3 sites). **Measured L1 verdicts: all proved** -- 11
/// obligations, nothing but `proved`.
///
/// `try_to_dimacs` is the total counterpart of `to_dimacs`: it is defined on
/// every one of the `2^32` raw codes, it is `Some` exactly on the codes whose
/// variable index is within [`Var::MAX_INDEX`], and where it is `Some` the
/// DIMACS number it returns is non-zero and round-trips back to the same
/// literal. That is the whole specification, and it is what lets a caller
/// handle an out-of-range literal instead of trusting it.
///
/// Drawing a raw code rather than an index is what makes the `None` arm
/// reachable at all: `from_code` is the only public constructor that can
/// produce a literal above the bound, which is exactly why the non-panicking
/// accessor is worth having.
///
/// `assert(dimacs != i32::MIN)` is load-bearing for the *measurement*, not for
/// the property. Without it the `panic` obligation of the `from_dimacs` call
/// below came back `unknown` (`solver-model-rejected`, measured), because the
/// solver had to rediscover that bound from the `i32::try_from` inside
/// `try_to_dimacs`. Stating it as its own assertion adds one proved obligation
/// and turns the whole harness green. It is a strengthening: the harness now
/// claims strictly more than it did.
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn try_to_dimacs_is_total_harness() {
    let code: u32 = any();
    let lit = Lit::from_code(code);
    match lit.try_to_dimacs() {
        Some(dimacs) => {
            assert(lit.var().0 <= Var::MAX_INDEX);
            assert(dimacs != 0);
            assert(dimacs != i32::MIN);
            assert(Lit::from_dimacs(dimacs).code() == lit.code());
        }
        None => {
            assert(lit.var().0 > Var::MAX_INDEX);
        }
    }
}

/// Property: `assert`, 8 sites. **Measured L1 verdict: unknown** -- 5 of the 8
/// sites are proved and 3 are `solver-model-rejected`, so the row records
/// `unknown` under the worst-of-sites rule. The three that are not decided are
/// the negation identities (the last three `assert`s below); the statements
/// about `from_bool` and about `Undef` are proved.
///
/// `LBool` is the three-valued assignment the CDCL loop reads on every
/// propagation. Negation is an involution on all three values, `from_bool` is
/// always defined and agrees with `is_true`/`is_false`, and `Undef` is its own
/// negation and stays undefined. Exercises the encoder's
/// enum-with-discriminant layout rather than bit arithmetic.
///
/// The `unknown` is the solver, not this harness, and that was measured rather
/// than assumed. Two re-spellings were tried against the same driver:
/// replacing the three biconditionals with an `if raw { .. } else { .. }` case
/// split gives 9 obligations of which 4 are model-rejected (strictly worse),
/// and hoisting `value.negate()` into a local plus an explicit
/// `assert(value.is_true() != value.is_false())` gives 9 obligations with the
/// *same* 3 model-rejected. cargo-formal is pinned to OxiZ 0.3.3, whose
/// Boolean-structure-over-bit-vector queries can return a model that does not
/// satisfy the formula (upstream U-Z10); cargo-formal's mandatory model check
/// refuses such a model, so the answer is `unknown` and never a `refuted` with
/// an invented witness. `README.md` says what moves when the pin moves.
///
/// Written over `is_true`/`is_false`/`is_defined` rather than `==` on `LBool`,
/// for the reason given in the module docs.
///
/// Runtime-checks build: green, unmarked (measured).
#[harness]
fn lbool_negate_involution_harness() {
    let raw: bool = any();
    let value = LBool::from_bool(raw);
    assert(value.is_defined());
    assert(value.is_true() == raw);
    assert(value.is_false() != raw);
    assert(!LBool::Undef.is_defined());
    assert(!LBool::Undef.negate().is_defined());
    assert(value.negate().is_true() == value.is_false());
    assert(value.negate().is_false() == value.is_true());
    assert(value.negate().negate().is_true() == value.is_true());
}

#[cfg(all(test, not(oxiformal_runtime_checks), not(formal)))]
mod plain_tests {
    use super::*;

    /// The bound the bounded harnesses assume, exercised at both ends.
    #[test]
    fn the_packing_round_trips_at_both_ends_of_the_index_range() {
        for index in [0u32, 1, 12345, Var::MAX_INDEX] {
            let var = Var::new(index);
            let pos = Lit::pos(var);
            let neg = Lit::neg(var);
            assert_eq!(pos.var().0, index);
            assert_eq!(neg.var().0, index);
            assert_eq!(pos.code(), index << 1);
            assert_eq!(neg.code(), (index << 1) | 1);
            assert!(pos.is_pos() && !pos.is_neg() && pos.sign());
            assert!(neg.is_neg() && !neg.sign());
            assert_eq!(pos.negate().code(), neg.code());
        }
    }

    /// The counterexample the L1 run reported for
    /// `lit_pos_roundtrip_unbounded_harness`, run concretely: `u32::MAX`.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn lit_pos_rejects_the_reported_index_counterexample() {
        let _ = Lit::pos(Var(u32::MAX));
    }

    /// The same refutation at its *smallest* witness: one index past the bound
    /// is already enough.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn lit_pos_rejects_one_index_past_the_bound() {
        let _ = Lit::pos(Var(Var::MAX_INDEX + 1));
    }

    /// The same refutation at the index the pre-fix code silently turned into
    /// variable 0: `2^31 << 1` is `0` in a `u32`.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn lit_pos_rejects_two_to_the_thirty_first() {
        let _ = Lit::pos(Var(1 << 31));
    }

    /// The counterexample the L1 run reported for `dimacs_negation_harness`,
    /// run concretely: `i32::MIN` is the one DIMACS number with no literal.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn from_dimacs_rejects_the_reported_dimacs_counterexample() {
        let _ = Lit::from_dimacs(i32::MIN);
    }

    /// The fact no harness can reach, because no public *constructor* reaches
    /// it: `to_dimacs` panics -- in **every** build profile, not only in debug
    /// -- for a literal whose index has no DIMACS form. `Lit::from_code` is
    /// the one way to build such a literal, which is why
    /// `dimacs_negation_harness` measures this same site as `proved`.
    ///
    /// This is where the upstream fix is visible as a replacement rather than
    /// an addition: the pre-fix code answered `i32::MIN` here, for *both*
    /// polarities, so a positive literal printed as a negative one and two
    /// distinct literals shared one DIMACS number.
    #[test]
    #[should_panic(expected = "Lit::to_dimacs")]
    fn to_dimacs_panics_for_a_literal_above_the_bound() {
        let _ = Lit::from_code((Var::MAX_INDEX + 1) << 1).to_dimacs();
    }

    /// The `None` arm of `try_to_dimacs_is_total_harness`, and its `Some` arm
    /// at the boundary.
    #[test]
    fn try_to_dimacs_is_none_exactly_above_the_bound() {
        let last_good = Lit::pos(Var::new(Var::MAX_INDEX));
        assert_eq!(last_good.try_to_dimacs(), Some(i32::MAX));
        assert_eq!(
            Lit::neg(Var::new(Var::MAX_INDEX)).try_to_dimacs(),
            Some(-i32::MAX)
        );

        let above = (Var::MAX_INDEX + 1) << 1;
        assert_eq!(Lit::from_code(above).var().0, Var::MAX_INDEX + 1);
        assert_eq!(Lit::from_code(above).try_to_dimacs(), None);
        assert_eq!(Lit::from_code(above | 1).try_to_dimacs(), None);
    }

    /// The concrete content of `dimacs_roundtrip_harness` at the extremes the
    /// two `assume`s leave in.
    #[test]
    fn dimacs_round_trips_for_the_representable_extremes() {
        for dimacs in [1, -1, 2, -2, 12345, -12345, i32::MAX, -i32::MAX] {
            let lit = Lit::from_dimacs(dimacs);
            assert_eq!(lit.to_dimacs(), dimacs);
            assert_eq!(lit.try_to_dimacs(), Some(dimacs));
            assert_eq!(lit.is_pos(), dimacs > 0);
            assert_eq!(lit.var().0, dimacs.unsigned_abs() - 1);
        }
    }

    /// The concrete content of `lit_negate_involution_harness` and
    /// `lit_code_roundtrip_harness`, over the codes that bracket every
    /// interesting case.
    #[test]
    fn the_raw_code_representation_is_exact() {
        for code in [
            0u32,
            1,
            2,
            3,
            0x7fff_ffff,
            0x8000_0000,
            u32::MAX - 1,
            u32::MAX,
        ] {
            let lit = Lit::from_code(code);
            assert_eq!(lit.code(), code);
            assert_eq!(lit.index(), code as usize);
            assert_eq!(lit.var().0, code >> 1);
            let flipped = lit.negate();
            assert_eq!(flipped.negate().code(), code);
            assert_eq!(flipped.var().0, lit.var().0);
            assert_ne!(flipped.is_pos(), lit.is_pos());
        }
    }

    /// The property `lbool_negate_involution_harness` states but the solver
    /// could not decide (`unknown`, `solver-model-rejected`), exhausted over
    /// the whole domain -- `LBool` has exactly three values, so this is not a
    /// sample but a proof by cases.
    #[test]
    fn lbool_negation_is_an_involution_on_all_three_values() {
        for raw in [true, false] {
            let value = LBool::from_bool(raw);
            assert!(value.is_defined());
            assert_eq!(value.is_true(), raw);
            assert_ne!(value.is_false(), raw);
            assert_eq!(value.negate().is_true(), value.is_false());
            assert_eq!(value.negate().is_false(), value.is_true());
            assert_eq!(value.negate().negate().is_true(), value.is_true());
        }
        assert!(!LBool::Undef.is_defined());
        assert!(!LBool::Undef.negate().is_defined());
    }
}
