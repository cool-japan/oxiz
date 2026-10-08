//! Re-fix pass 19 (decision (92); `TODO.md` decision (85)'s net, `#P2b-92`) —
//! the one code change of the pass and the item cargo-formal filed, pinned.
//!
//! * §1, the honesty net over a table that does not read back (decision
//!   (92)(d); adversarial recheck 18's finding 4, a code read, turned into a
//!   measured escape by this pass): `Context::printed_datatype_model_refuted`
//!   returned "clean" as soon as ANY declared function's printed table failed
//!   to parse back, so one unrelated unreadable table switched off the reading
//!   of every assertion.  The vehicle is a pre-existing printer defect (every
//!   build, 0.3.3 included): a symbol that needs `|…|` quoting prints unquoted,
//!   so a constructor named `|(a|` prints as `(a` and its table does not
//!   parse.  On `gen_dt.py` seed 30093154's `d00236` with such a function added
//!   (`$R/fix19/atk/unread/d236_qa.smt2`) re-fix pass 18's tree printed `sat`
//!   and a model whose third assertion is false at both checks — as `c702310`
//!   does, where re-fix pass 17's trajectory printed a correct one.  Now a
//!   function whose table does not read back is OPEN: every other assertion is
//!   still read (so that model answers `unknown` with the net's reason), and an
//!   assertion that applies the function goes to a fresh solver with the
//!   function uninterpreted — the model is withheld unless the assertion's
//!   negation is refuted there, i.e. unless it holds under every
//!   interpretation of the function; the verdict stands.  `#P2b-88`'s root
//!   fix decides `d00236` (both checks `sat` with models z3 confirms; beside
//!   the table, `sat` with the model withheld for it), so a false candidate
//!   beside the table is now `#P2b-95`'s collision (`U1_COLLISION`).
//! * §2, `#P2b-92` (filed by cargo-formal wave 5.8): a bit-vector bound
//!   against a constant near 2^63 overflowed the solver's own arithmetic on
//!   0.3.3 (`(not (bvule n #x7fffffffffffffff))`: a debug build panics with
//!   "attempt to add with overflow" in `ArithSolver::assert_gt`, a release
//!   build answers `unknown`).  The bit-vector shape is fixed in 0.3.4 by
//!   `#P2b-28` (the unsigned relaxation retired; `c702310` already answers
//!   `sat`) and pinned here in this test binary's profile; the same root — a
//!   strict integer bound `k` turned into `k + 1` in `Rational64` — is still
//!   reachable from an `Int` bound at `i64::MAX`, pinned as the item's hole.
//!
//! No test installs a wall clock (decision (16)).

use oxiz_solver::Context;

/// The script's responses (an error folded into one line).
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// Every `sat` / `unsat` / `unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

/// Whether any response prints a model.
fn prints_a_model(lines: &[String]) -> bool {
    lines.iter().any(|line| line.contains("(define-fun"))
}

// ---------------------------------------------------------------------------
// §1. The net reads a table that does not read back as OPEN, never clean.
// ---------------------------------------------------------------------------

/// `gen_dt.py` seed 30093154 `d00236` with an unrelated function into a
/// datatype whose constructor `|(a|` prints unquoted (`$R/fix19/atk/unread/
/// d236_qa.smt2`, verbatim).  On re-fix pass 19's tree the candidate model
/// made assertion 3 false at both checks (`h 1 = (cons 0 nil)` against
/// `(tl (h 2)) = (cons 4 nil)`) and both answered `unknown`.  That candidate
/// rode `#P2b-88`'s trajectory, and `#P2b-88`'s root fix moved it: both checks
/// are `sat` now (z3: `sat`, `sat`; `d00236` itself prints two models z3
/// confirms), and the model stays withheld because assertion 1 applies `sx`,
/// whose table does not read back, and holds only under some readings of it.
/// The false-candidate half of the property moved to [`U1_COLLISION`].
const D236_QA: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n\
(declare-fun h (Int) L)\n\
(declare-fun g (Int) C)\n\
(declare-fun k (L) Int)\n\
(declare-fun q (C) Int)\n\
(declare-fun w (Int) U)\n\
(declare-fun v (L) L)\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const l3 L)\n\
(declare-const c1 C)\n\
(declare-const c2 C)\n\
(declare-const p P)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(declare-datatypes ((D 0)) (((|(a|) (b))))\n\
(declare-fun sx (Int) D)\n\
(assert (= (sx 0) |(a|))\n\
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
(assert (or (and ((_ is nil) (cons (q c2) (h 1))) (distinct l3 l3)) (and (= (h (ite ((_ is cons) (cons 1 l3)) (hd (cons 1 l3)) 0)) (ite ((_ is cons) (h 2)) (tl (h 2)) nil)) (= (h (px p)) (cons 2 l1)))))\n\
(assert (= (w (k (v l1))) u1))\n\
(assert (and (= p (mk (k nil) c2)) (= p (mk (q (pc p)) (pc p)))))\n\
(check-sat)\n\
(get-model)\n\
(check-sat)\n\
(get-model)\n";

/// A candidate the exact reader shows false beside a table that does not read
/// back: `#P2b-95`'s collision (a selector under an uninterpreted function —
/// no theory values `(px p)`, so the rebuilt `p` prints it as `0` and `h`
/// collides at `0`; z3: `sat`), after the same `sx` prelude as [`D236_QA`].
/// Assertion 1 applies the unreadable `sx`; assertion 3 reads false, so the
/// check answers `unknown` naming it — the net reads every other assertion.
/// When `#P2b-95` is closed this candidate holds and the check is `sat` with
/// the model withheld for `sx`: re-derive the vehicle on the next false
/// candidate then.
const U1_COLLISION: &str = "(set-logic ALL)\n\
(declare-datatypes ((D 0)) (((|(a|) (b))))\n\
(declare-fun sx (Int) D)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-fun h (Int) L)\n\
(declare-const l3 L)\n\
(declare-const p P)\n\
(assert (= (sx 0) |(a|))\n\
(assert (= (cons 6 l3) (h 0)))\n\
(assert (= (h (px p)) l3))\n\
(check-sat)\n\
(get-model)\n";

/// These pins ride a printer defect (every build): a symbol that needs `|…|`
/// quoting prints unquoted.  If the printer ever quotes it, the tables read
/// back and the pins below lose their vehicle — this guard says so instead of
/// letting them redden with a misleading message.
fn assert_the_vehicle_is_still_there() {
    let lines = run(
        "(set-logic ALL)\n(declare-datatypes ((D 0)) (((|(a|) (b))))\n\
                     (declare-const d D)\n(assert (= d |(a|))\n(check-sat)\n(get-model)\n",
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("(define-fun d () D (a)")),
        "the printer now quotes symbols: this pin's vehicle (a table that does not read back) is \
         gone — re-derive the pin on another unreadable table\n{}",
        lines.join("\n")
    );
}

/// The quantifier-free prefix of the shapes below: a list goal, so the net
/// runs, and a function into the badly printed datatype.
const UNREAD_PRELUDE: &str = "(set-logic ALL)\n\
(declare-datatypes ((D 0)) (((|(a|) (b))))\n\
(declare-fun sx (Int) D)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(assert ((_ is cons) l1))\n";

#[test]
fn an_unreadable_table_no_longer_switches_the_net_off() {
    assert_the_vehicle_is_still_there();
    // A candidate the reader shows false is caught beside the unreadable
    // table: the check answers `unknown`, naming the false assertion.
    let lines = run(U1_COLLISION);
    assert_eq!(
        verdicts(&lines),
        vec!["unknown"],
        "assertion 3 reads false under the candidate model, so the check answers unknown \
         (before re-fix pass 19 the unreadable `sx` made the net print it)\n{}",
        lines.join("\n")
    );
    assert!(
        !prints_a_model(&lines),
        "no model is printed\n{}",
        lines.join("\n")
    );
    assert!(
        lines.iter().any(|line| line
            .contains("model check failed: assertion 3 reads false under the candidate model")),
        "the (get-model) names the false assertion\n{}",
        lines.join("\n")
    );
    // `d00236` beside the same table: decided (`#P2b-88`), and the model is
    // withheld for the table assertion 1 applies.
    let lines = run(D236_QA);
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "sat"],
        "z3: sat at both checks\n{}",
        lines.join("\n")
    );
    assert!(
        !prints_a_model(&lines),
        "no model is printed\n{}",
        lines.join("\n")
    );
    let reason = "model not certified: the printed interpretation of sx does not read back \
                  (assertion 1)";
    assert_eq!(
        lines.iter().filter(|line| line.contains(reason)).count(),
        2,
        "each (get-model) names the table and the assertion\n{}",
        lines.join("\n")
    );
}

#[test]
fn an_assertion_over_an_unreadable_table_is_withheld_unless_it_holds_under_every_reading() {
    assert_the_vehicle_is_still_there();
    // `qa1`: true for the printed table, which cannot be read; true for SOME
    // interpretation of `sx`, not for every one -> withheld, verdict kept.
    let qa1 = format!("{UNREAD_PRELUDE}(assert (= (sx 0) |(a|))\n(check-sat)\n(get-model)\n");
    // `qa5`: the same with a disjunction that still depends on `sx`.
    let qa5 = format!(
        "{UNREAD_PRELUDE}(assert (or (= (sx 0) |(a|) (= (sx 1) b)))\n(check-sat)\n(get-model)\n"
    );
    for (name, script) in [("qa1", &qa1), ("qa5", &qa5)] {
        let lines = run(script);
        assert_eq!(
            verdicts(&lines),
            vec!["sat"],
            "`{name}`\n{}",
            lines.join("\n")
        );
        assert!(
            lines.iter().any(|line| line.contains(
                "model not certified: the printed interpretation of sx does not read back \
                 (assertion 2)"
            )),
            "`{name}`: the model is withheld, naming the table and the assertion\n{}",
            lines.join("\n")
        );
        assert!(!prints_a_model(&lines), "`{name}`\n{}", lines.join("\n"));
    }
    // `qa4`: true under EVERY interpretation of `sx` (`D` has two values), so
    // its negation is refuted and nothing shows the model false: printed.
    let qa4 = format!(
        "{UNREAD_PRELUDE}(assert (or (= (sx 0) |(a|) (= (sx 1) b) (not (= (sx 0) (sx 1)))))\n\
         (check-sat)\n(get-model)\n"
    );
    let lines = run(&qa4);
    assert_eq!(verdicts(&lines), vec!["sat"], "`qa4`\n{}", lines.join("\n"));
    assert!(
        prints_a_model(&lines),
        "`qa4`: an assertion valid in `sx` does not withhold the model\n{}",
        lines.join("\n")
    );
    // `s1`: a sort named `|U V|` prints its witnesses as `@uc_U V_n`, which
    // parses by truncation into a free symbol — not a value, so the table is
    // unreadable too.
    let s1 = "(set-logic ALL)\n\
              (declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
              (declare-sort |U V| 0)\n(declare-fun g (Int) |U V|)\n(declare-const l1 L)\n\
              (assert ((_ is cons) l1))\n(assert (distinct (g 1) (g 2)))\n(check-sat)\n(get-model)\n";
    let lines = run(s1);
    assert_eq!(verdicts(&lines), vec!["sat"], "`s1`\n{}", lines.join("\n"));
    assert!(
        lines.iter().any(|line| line.contains(
            "model not certified: the printed interpretation of g does not read back (assertion 2)"
        )),
        "`s1`: withheld\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §2. `#P2b-92` — a bound near 2^63.
// ---------------------------------------------------------------------------

/// A 64-bit `#x…` literal from a `(get-value)` line, or `None`.
fn hex_after(lines: &[String], name: &str) -> Option<u64> {
    let joined = lines.join(" ");
    let key = format!("({name} #x");
    let start = joined.find(&key)? + key.len();
    u64::from_str_radix(joined.get(start..start + 16)?, 16).ok()
}

#[test]
fn a_bit_vector_bound_near_two_to_the_sixty_third_is_decided_without_overflow() {
    // cargo-formal's shape (the capacity rule of `formal-vcgen`'s `seq.rs`): a
    // strict bound one past `i64::MAX`, beside a 64-bit multiplication.  0.3.3:
    // `unknown` in release, "attempt to add with overflow" in debug.
    let above = "(set-logic QF_BV)\n(declare-const n (_ BitVec 64))\n\
                 (assert (not (bvule n #x7fffffffffffffff)))\n(check-sat)\n(get-value (n))\n";
    let lines = run(above);
    assert_eq!(verdicts(&lines), vec!["sat"], "{}", lines.join("\n"));
    assert!(
        hex_after(&lines, "n").is_some_and(|n| n > 0x7fff_ffff_ffff_ffff),
        "n is above the bound\n{}",
        lines.join("\n")
    );
    let product = "(set-logic QF_BV)\n(declare-const a (_ BitVec 64))\n(declare-const b (_ BitVec 64))\n\
                   (assert (bvule a #xffffffffffffffff))\n\
                   (assert (not (bvule a #x7fffffffffffffff)))\n\
                   (assert (bvule (bvmul a b) #x7fffffffffffffff))\n(check-sat)\n(get-value (a b))\n";
    let lines = run(product);
    assert_eq!(verdicts(&lines), vec!["sat"], "{}", lines.join("\n"));
    let (Some(a), Some(b)) = (hex_after(&lines, "a"), hex_after(&lines, "b")) else {
        panic!("a and b are printed\n{}", lines.join("\n"));
    };
    assert!(
        a > 0x7fff_ffff_ffff_ffff && a.wrapping_mul(b) <= 0x7fff_ffff_ffff_ffff,
        "the model holds\n{}",
        lines.join("\n")
    );
    // The unsatisfiable twins: the bound and its negation, and the capacity
    // rule's `n <= isize::MAX / 8` against `8n > isize::MAX - 7`.
    for script in [
        "(set-logic QF_BV)\n(declare-const n (_ BitVec 64))\n\
         (assert (not (bvule n #x7fffffffffffffff)))\n(assert (bvule n #x7fffffffffffffff))\n\
         (check-sat)\n",
        "(set-logic QF_BV)\n(declare-const a (_ BitVec 64))\n\
         (assert (not (bvule (bvmul a #x0000000000000008) #x7ffffffffffffff8)))\n\
         (assert (bvule a #x0fffffffffffffff))\n(check-sat)\n",
    ] {
        let lines = run(script);
        assert_eq!(verdicts(&lines), vec!["unsat"], "{}", lines.join("\n"));
    }
}

#[test]
fn an_int_bound_at_i64_max_still_overflows_the_arithmetic_solver() {
    // `#P2b-92`'s open half: `x > 9223372036854775807` is asserted as
    // `x >= 9223372036854775807 + 1` in `Rational64`
    // (`oxiz-theories/src/arithmetic/solver.rs`, `ArithSolver::assert_gt`):
    // a debug build panics with "attempt to add with overflow", a release
    // build answers `unknown`.  z3: `sat` for both spellings.
    for script in [
        "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 9223372036854775807))\n(check-sat)\n",
        "(set-logic QF_LIA)\n(declare-const x Int)\n\
         (assert (not (<= x 9223372036854775807)))\n(check-sat)\n",
    ] {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(script)));
        let Ok(lines) = outcome else {
            // The debug build's overflow panic: no answer, never a wrong one.
            continue;
        };
        let got = verdicts(&lines);
        assert_ne!(
            got,
            vec!["unsat"],
            "a WRONG unsat (z3: sat)\n{}",
            lines.join("\n")
        );
        if got == vec!["sat"] {
            panic!(
                "THE HOLE IS CLOSED (`#P2b-92`): an Int bound at i64::MAX is decided — invert \
                 this pin and close the item\n{}",
                lines.join("\n")
            );
        }
    }
    // One below the limit never needed the overflowing `+ 1`: decided.
    let lines = run("(set-logic QF_LIA)\n(declare-const x Int)\n\
                     (assert (< x (- 9223372036854775807 1)))\n\
                     (assert (> x 9223372036854775806))\n(check-sat)\n");
    assert_eq!(verdicts(&lines), vec!["unsat"], "{}", lines.join("\n"));
}
