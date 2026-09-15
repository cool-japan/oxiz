//! Contracted wrappers over the **public** `oxiz-sat` literal API.
//!
//! Each `spec_*` function below restates one of the nine `#[harness]` entry
//! points of [`crate::harness`] as a `#[requires]`/`#[ensures]` contract, and
//! the `#[proof_for_contract]` harness beside it is what `cargo formal check`
//! verifies. The contract is on the **wrapper**, not on `oxiz-sat`: the
//! wrapper calls exactly the `oxiz_sat::{Lit, Var, LBool}` methods the
//! corresponding harness calls, and nothing in `oxiz-sat` is annotated or
//! changed. `README.md` ("In-source contract candidates") says why the
//! contracts are not on `oxiz-sat/src/literal.rs` itself yet.
//!
//! Two rules keep every wrapper measurable:
//!
//! * A `#[proof_for_contract]` body narrows its inputs by **total arithmetic**
//!   (`min`, an `if`), never by `assume(..)`. Under
//!   `--cfg oxiformal_runtime_checks` such a body runs once, outside
//!   `oxiformal::rt::run_harness`, so an `AssumeViolation` would fail the
//!   test rather than skip the draw; and under `cargo formal` the generated
//!   check sibling assumes the `requires` anyway, so the narrowing costs one
//!   `ite` and changes no verdict.
//! * No wrapper calls another wrapper. A contracted callee is *replaced* by
//!   its contract under the default `modular = true`, which records an
//!   undischarged assumption on every obligation after the call and refuses
//!   the `lrat` certificate and the external-replay evidence for it. Every
//!   wrapper and every `ensures` closure calls `oxiz-sat` bodies only.
//!
//! Values are compared through `.code()`, `.var().0` and the `is_*` observers
//! rather than with `==` on `Lit`/`Var`/`LBool`, for the reason given in
//! [`crate::harness`]'s module docs.
//!
//! # Why every `ensures` closure mentions a parameter
//!
//! Three clauses below are conjunctions that would read more naturally as two
//! clauses, and one clause states its identity against
//! `LBool::from_bool(raw)` rather than against the returned value twice. Both
//! spellings are forced by a measured limitation of the `cargo-formal` driver
//! this package is verified with (`formal-driver` 0.1.0, built from
//! cargo-formal `0621fc0`): an `#[ensures]` closure that captures **nothing**
//! is a zero-sized type, so its MIR operand is a *constant* of closure type,
//! and the driver's constant decoder has no arm for it
//! (`driver/formal-driver/src/lower/konst.rs:134-137`,
//! `unsupported-rvalue(constant of type ...)`). The whole generated
//! `__formal_check_*` body is then lost and the wrapper is reported as
//! `unsupported(no-body)` -- no verdict at all.
//!
//! That was measured, not guessed: the first `--evidence lrat` run of this
//! module (2026-09-15) reported exactly the three wrappers that had a
//! capture-less clause -- [`spec_lit_pos`], [`spec_lit_neg`],
//! [`spec_lbool_from_bool`] -- as `unsupported(no-body)`, and the six whose
//! every clause mentions a parameter encoded. Re-spelling the four
//! capture-less clauses so that each mentions its wrapper's parameter is what
//! made all nine measurable; it is a change of spelling, not of claim, and the
//! doc comment of each affected wrapper says what it cost. The limitation is
//! reported upstream; nothing about it is a property of `oxiz-sat`.
//!
//! # Property keys
//!
//! The `N` `#[ensures]` clauses of one wrapper all expand at the wrapper's own
//! `fn` span, so the encoder mints them as `ensures`, `ensures-2`, ... in
//! declaration order (the property-table collision rule). `EXPECTED.toml`
//! lists them under those keys.

use oxiformal::prelude::*;
use oxiz_sat::{LBool, Lit, Var};

// --- 1. lit_pos_roundtrip_bounded_harness ------------------------------------

/// Contract twin of `lit_pos_roundtrip_bounded_harness`: under the documented
/// bound, `Lit::pos` keeps the index, reads back positive, and packs exactly
/// `index << 1`.
///
/// The second clause is a conjunction because the polarity half of it
/// (`is_pos() && !is_neg() && sign()`) captures nothing on its own; see the
/// module docs. It is fused with the packing half, which implies it: a code of
/// `index << 1` has bit 0 clear.
///
/// **Measured L1 verdicts: `ensures` proved, `ensures-2` unknown**
/// (`solver-model-rejected`); `panic` proved (1 site) and `shift-overflow`
/// proved (3 sites). The undecided row is the fused clause, and it is the one
/// price the re-spelling charged: the same two facts are `proved` as separate
/// `assert`s in `lit_pos_roundtrip_bounded_harness`. z3 4.15.4 answers `unsat`
/// on the same script (`replay --all --solver z3`), so the clause is true and
/// OxiZ 0.3.3 is what could not show it.
#[requires(index <= Var::MAX_INDEX)]
#[ensures(|r| r.var().0 == index)]
#[ensures(|r| r.code() == index << 1 && r.is_pos() && !r.is_neg() && r.sign())]
pub fn spec_lit_pos(index: u32) -> Lit {
    Lit::pos(Var::new(index))
}

/// Discharges [`spec_lit_pos`]'s contract over its whole domain. The `min`
/// keeps the runtime-checks draw inside the `requires` (see the module docs);
/// `cargo formal` sees every index in `0..=Var::MAX_INDEX`.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lit_pos)]
fn spec_lit_pos_contract() {
    let index: u32 = any();
    let _ = spec_lit_pos(index.min(Var::MAX_INDEX));
}

// --- 2. lit_neg_roundtrip_bounded_harness ------------------------------------

/// Contract twin of `lit_neg_roundtrip_bounded_harness`: the negative
/// constructor is the positive one with bit 0 set, and `negate` maps it onto
/// `Lit::pos` of the same variable.
///
/// The last clause is a conjunction for the reason the module docs give; the
/// packing `(index << 1) | 1` implies the polarity half fused into it.
///
/// **Measured L1 verdicts: all proved** -- `ensures`, `ensures-2`,
/// `ensures-3`, `panic` (1 site) and `shift-overflow` (4 sites).
#[requires(index <= Var::MAX_INDEX)]
#[ensures(|r| r.var().0 == index)]
#[ensures(|r| r.negate().code() == Lit::pos(Var::new(index)).code())]
#[ensures(|r| r.code() == ((index << 1) | 1) && r.is_neg() && !r.sign())]
pub fn spec_lit_neg(index: u32) -> Lit {
    Lit::neg(Var::new(index))
}

/// Discharges [`spec_lit_neg`]'s contract.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lit_neg)]
fn spec_lit_neg_contract() {
    let index: u32 = any();
    let _ = spec_lit_neg(index.min(Var::MAX_INDEX));
}

// --- 3. lit_pos_roundtrip_unbounded_harness ----------------------------------

/// Contract twin of `lit_pos_roundtrip_unbounded_harness`: [`spec_lit_pos`]
/// with the `requires` **removed**. The round trip is still claimed for every
/// index the constructor accepts, and the `debug_assert!` at
/// `../src/literal.rs:61` that rejects the rest is reached with the full
/// `u32` domain. Built with the public tuple field (`Var(index)`) rather than
/// `Var::new`, so the first check the encoder reaches is `Lit::pos`'s own.
///
/// A contract cannot state "an index above the bound is rejected" as a
/// postcondition; what it can do is show that the bound is *necessary*: the
/// `requires` of [`spec_lit_pos`] is exactly what turns this wrapper's
/// refuted `panic` row into a proved one.
///
/// **Measured L1 verdicts: `panic` refuted**, counterexample
/// `index = 4294967295` (`u32::MAX`), the same witness the `#[harness]` twin
/// reports; **`ensures` proved** on the surviving edge, and both
/// `shift-overflow` obligations proved. That split is the whole point: with
/// the `requires` of [`spec_lit_pos`] the `panic` row is proved, without it it
/// is refuted, and the round trip holds either way.
#[ensures(|r| r.var().0 == index)]
pub fn spec_lit_pos_unbounded(index: u32) -> Lit {
    Lit::pos(Var(index))
}

/// Discharges [`spec_lit_pos_unbounded`]'s contract over all of `u32`.
///
/// Runtime-checks build: **ignored**. A `#[proof_for_contract]` body runs once,
/// and slightly over half of all `u32` trip the bound, so one draw is a coin
/// flip fixed by the seed rather than a sample; the dense runtime witness is
/// carried by `lit_pos_roundtrip_unbounded_harness` (256 draws,
/// `#[should_panic]`, measured to panic) and by `plain_tests`.
#[proof_for_contract(spec_lit_pos_unbounded)]
#[cfg_attr(
    oxiformal_runtime_checks,
    ignore = "one draw is not a sample; lit_pos_roundtrip_unbounded_harness carries the runtime witness"
)]
fn spec_lit_pos_unbounded_contract() {
    let index: u32 = any();
    let _ = spec_lit_pos_unbounded(index);
}

// --- 4. lit_negate_involution_harness ----------------------------------------

/// Contract twin of `lit_negate_involution_harness`: `negate` is `code ^ 1`,
/// an involution that never changes the variable and always flips the sign,
/// over all `2^32` raw codes.
///
/// **Measured L1 verdicts: all proved** -- `ensures` through `ensures-4`, and
/// the one incidental `shift-overflow`.
#[ensures(|r| r.negate().code() == lit.code())]
#[ensures(|r| r.var().0 == lit.var().0)]
#[ensures(|r| r.is_pos() != lit.is_pos())]
#[ensures(|r| r.is_neg() == lit.is_pos())]
pub fn spec_lit_negate(lit: Lit) -> Lit {
    lit.negate()
}

/// Discharges [`spec_lit_negate`]'s contract over every raw code.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lit_negate)]
fn spec_lit_negate_contract() {
    let code: u32 = any();
    let _ = spec_lit_negate(Lit::from_code(code));
}

// --- 5. lit_code_roundtrip_harness -------------------------------------------

/// Contract twin of `lit_code_roundtrip_harness`: `from_code`/`code` are
/// inverse, `index` is the zero-extended code, `var` is the high 31 bits.
///
/// **Measured L1 verdicts: all proved** -- one `ensures` row and two
/// `shift-overflow` rows. The three clauses produce a *single* obligation
/// here (one `vc/NNNN.smt2`, goal assertion 0) rather than three: for a wrapper
/// whose body is `Lit::from_code(code)` the raw round trip discharges inside
/// the encoder, so only one assertion survives to be asked about. The
/// measured row count, not the clause count, is what `EXPECTED.toml` lists.
#[ensures(|r| r.code() == code)]
#[ensures(|r| r.index() == code as usize)]
#[ensures(|r| r.var().0 == (code >> 1))]
pub fn spec_lit_from_code(code: u32) -> Lit {
    Lit::from_code(code)
}

/// Discharges [`spec_lit_from_code`]'s contract.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lit_from_code)]
fn spec_lit_from_code_contract() {
    let code: u32 = any();
    let _ = spec_lit_from_code(code);
}

// --- 6. dimacs_roundtrip_harness ---------------------------------------------

/// Contract twin of `dimacs_roundtrip_harness`: under the two documented
/// preconditions -- DIMACS has no literal `0`, and `i32::MIN` has no positive
/// counterpart -- `to_dimacs` inverts `from_dimacs`, the sign survives, and
/// the variable index is `|d| - 1`.
///
/// **Measured L1 verdicts: `ensures`, `ensures-2` and `ensures-3` all
/// proved**, `neg-overflow` (`../src/literal.rs:129`), `arith-overflow`
/// (2 sites) and `shift-overflow` (3 sites) proved, and **`panic` unknown**
/// over 2 sites (1 unknown, `solver-model-rejected`, at `:89`'s
/// `debug_assert!`; 1 proved at `to_dimacs`'s `:113`). The undecided site is
/// the same one the `#[harness]` twin `dimacs_negation_harness` reports as
/// undecided in this run; z3 4.15.4 answers `unsat` on this one.
#[requires(dimacs != 0 && dimacs != i32::MIN)]
#[ensures(|r| r.to_dimacs() == dimacs)]
#[ensures(|r| r.is_pos() == (dimacs > 0))]
#[ensures(|r| r.var().0 == dimacs.unsigned_abs() - 1)]
pub fn spec_lit_from_dimacs(dimacs: i32) -> Lit {
    Lit::from_dimacs(dimacs)
}

/// Discharges [`spec_lit_from_dimacs`]'s contract. The `if` maps the two
/// inputs the `requires` excludes onto `1`, which is inside the domain, so
/// the runtime draw never trips the precondition.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lit_from_dimacs)]
fn spec_lit_from_dimacs_contract() {
    let dimacs: i32 = any();
    let dimacs = if dimacs == 0 || dimacs == i32::MIN {
        1
    } else {
        dimacs
    };
    let _ = spec_lit_from_dimacs(dimacs);
}

// --- 7. dimacs_negation_harness ----------------------------------------------

/// Contract twin of `dimacs_negation_harness`: [`spec_lit_from_dimacs`] with
/// the `i32::MIN` precondition **removed**, so the one DIMACS number that has
/// no literal is inside the domain. Two facts are measured here, and together
/// they are the upstream fix: `from_dimacs(i32::MIN)` is rejected by the
/// `debug_assert!` at `../src/literal.rs:89`, and `to_dimacs` is total on
/// every literal the surviving edge can produce.
///
/// **Measured L1 verdicts: `panic` refuted**, counterexample
/// `dimacs = -2147483648` (`i32::MIN`), over 2 sites of which the other
/// (`to_dimacs`'s `:113`) is proved; **`ensures` proved**, and
/// `neg-overflow`, `arith-overflow` and the three `shift-overflow`
/// obligations proved. In this run the contract form is the only place the
/// `i32::MIN` refutation is still *recorded as a refutation*: the
/// `#[harness]` twin's identical obligation came back
/// `solver-model-rejected` (see `README.md`).
#[requires(dimacs != 0)]
#[ensures(|r| r.to_dimacs() == dimacs)]
pub fn spec_lit_from_dimacs_nonzero(dimacs: i32) -> Lit {
    Lit::from_dimacs(dimacs)
}

/// Discharges [`spec_lit_from_dimacs_nonzero`]'s contract.
///
/// Runtime-checks build: one witness (`i32::MIN`) out of `2^32` and one draw
/// -- narrow witness, deliberately **not** marked `#[should_panic]`; the test
/// passing is not evidence of correctness, which is what the L1 verdict is
/// for. Measured: it passes.
#[proof_for_contract(spec_lit_from_dimacs_nonzero)]
fn spec_lit_from_dimacs_nonzero_contract() {
    let dimacs: i32 = any();
    let dimacs = if dimacs == 0 { 1 } else { dimacs };
    let _ = spec_lit_from_dimacs_nonzero(dimacs);
}

// --- 8. try_to_dimacs_is_total_harness ---------------------------------------

/// Contract twin of `try_to_dimacs_is_total_harness`: `try_to_dimacs` is
/// defined on every raw code, is `Some` exactly when the variable index is
/// within [`Var::MAX_INDEX`], and where it is `Some` the number is non-zero,
/// is not `i32::MIN`, and round-trips through `from_dimacs` to the same
/// literal.
///
/// The `d != 0 && d != i32::MIN` conjuncts precede the `from_dimacs` call in
/// the second clause on purpose: `&&` is a path condition, so the
/// `debug_assert!` inside `from_dimacs` is reached only where they hold --
/// the same bound the harness had to state explicitly to stop the solver
/// rediscovering it (`harness.rs`, harness 8).
///
/// **Measured L1 verdicts: `ensures` and `ensures-2` both unknown**
/// (`solver-model-rejected`) -- an `Option<i32>` postcondition is
/// Boolean structure over an enum, the shape OxiZ 0.3.3 is least able to
/// decide. Both carry the note *the bit-level engine proved this
/// (`vc/NNNN.lrat`); the SMT engine did not*, and z3 4.15.4 answers `unsat`
/// on both. Everything incidental is proved: `panic`, `neg-overflow`,
/// `arith-overflow` and three `shift-overflow` obligations.
#[ensures(|r| r.is_some() == (lit.var().0 <= Var::MAX_INDEX))]
#[ensures(|r| match *r {
    Some(d) => d != 0 && d != i32::MIN && Lit::from_dimacs(d).code() == lit.code(),
    None => lit.var().0 > Var::MAX_INDEX,
})]
pub fn spec_lit_try_to_dimacs(lit: Lit) -> Option<i32> {
    lit.try_to_dimacs()
}

/// Discharges [`spec_lit_try_to_dimacs`]'s contract over every raw code,
/// which is what makes the `None` arm reachable.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lit_try_to_dimacs)]
fn spec_lit_try_to_dimacs_contract() {
    let code: u32 = any();
    let _ = spec_lit_try_to_dimacs(Lit::from_code(code));
}

// --- 9. lbool_negate_involution_harness --------------------------------------

/// Contract twin of `lbool_negate_involution_harness`: `from_bool` is always
/// defined and agrees with `is_true`/`is_false`, and negation is an involution
/// on the two defined values. The two `Undef` facts of the harness are
/// constants, not properties of `from_bool`, and stay as `assert`s in the
/// harness body below.
///
/// The three negation identities are stated against `LBool::from_bool(raw)`
/// rather than against the returned value a second time. That is the same
/// formula -- the wrapper *returns* `LBool::from_bool(raw)` -- spelled so the
/// closure captures `raw`; see the module docs for why it has to. The first
/// clause is a conjunction for the same reason.
///
/// **Measured L1 verdicts: `ensures-2` proved, `ensures`, `ensures-3`,
/// `ensures-4` and `ensures-5` unknown** (`solver-model-rejected`), and the
/// harness body's two `Undef` `assert`s proved. The three undecided identities
/// are the contract form of the three undecided `assert`s of
/// `lbool_negate_involution_harness`: the contract form does not move them.
/// The fourth undecided row (`ensures`) is the fused
/// `is_defined() && is_true() == raw` clause, whose two halves are `proved`
/// separately in the `#[harness]` twin -- the second price of the re-spelling.
/// z3 4.15.4 answers `unsat` on all four.
#[ensures(|r| r.is_defined() && r.is_true() == raw)]
#[ensures(|r| r.is_false() != raw)]
#[ensures(|r| r.negate().is_true() == LBool::from_bool(raw).is_false())]
#[ensures(|r| r.negate().is_false() == LBool::from_bool(raw).is_true())]
#[ensures(|r| r.negate().negate().is_true() == LBool::from_bool(raw).is_true())]
pub fn spec_lbool_from_bool(raw: bool) -> LBool {
    LBool::from_bool(raw)
}

/// Discharges [`spec_lbool_from_bool`]'s contract, and states the two `Undef`
/// fixpoint facts beside it.
///
/// Runtime-checks build: green, unmarked (measured).
#[proof_for_contract(spec_lbool_from_bool)]
fn spec_lbool_from_bool_contract() {
    let raw: bool = any();
    let _ = spec_lbool_from_bool(raw);
    assert(!LBool::Undef.is_defined());
    assert(!LBool::Undef.negate().is_defined());
}

#[cfg(all(test, not(oxiformal_runtime_checks), not(formal)))]
mod plain_tests {
    use super::*;

    /// The wrappers are the functions they wrap: nothing a contract attribute
    /// adds survives a plain build.
    #[test]
    fn wrappers_are_identities_over_the_public_api() {
        for index in [0u32, 1, 12345, Var::MAX_INDEX] {
            assert_eq!(spec_lit_pos(index).code(), Lit::pos(Var::new(index)).code());
            assert_eq!(spec_lit_neg(index).code(), Lit::neg(Var::new(index)).code());
            assert_eq!(
                spec_lit_pos_unbounded(index).code(),
                Lit::pos(Var(index)).code()
            );
        }
        for code in [0u32, 1, 2, 3, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
            assert_eq!(spec_lit_from_code(code).code(), code);
            assert_eq!(spec_lit_negate(Lit::from_code(code)).code(), code ^ 1);
            assert_eq!(
                spec_lit_try_to_dimacs(Lit::from_code(code)),
                Lit::from_code(code).try_to_dimacs()
            );
        }
        for dimacs in [1, -1, 2, -2, i32::MAX, -i32::MAX] {
            assert_eq!(
                spec_lit_from_dimacs(dimacs).code(),
                Lit::from_dimacs(dimacs).code()
            );
            assert_eq!(
                spec_lit_from_dimacs_nonzero(dimacs).code(),
                Lit::from_dimacs(dimacs).code()
            );
        }
        assert!(spec_lbool_from_bool(true).is_true());
        assert!(spec_lbool_from_bool(false).is_false());
    }

    /// The refutation the unbounded wrapper is expected to measure, run
    /// concretely at the smallest witness.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn spec_lit_pos_unbounded_rejects_one_index_past_the_bound() {
        let _ = spec_lit_pos_unbounded(Var::MAX_INDEX + 1);
    }

    /// The refutation the nonzero DIMACS wrapper is expected to measure.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn spec_lit_from_dimacs_nonzero_rejects_i32_min() {
        let _ = spec_lit_from_dimacs_nonzero(i32::MIN);
    }
}
