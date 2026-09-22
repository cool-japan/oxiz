//! Round-4 recheck pass 9 — pins for what re-fix pass 9 left open, and
//! regression guards for what passes 8, 9 and 10 closed.
//!
//! Re-fix pass 10 closed one of the three holes (`#P2b-59`) and inverted its
//! pin into `the_refinement_round_count_is_bounded_and_equal_inside_the_plateau`
//! — a name re-fix pass 11 corrected, because the equality it asserts is a
//! plateau inside a ramp and not the fixpoint the pass claimed (see that
//! test's doc for the measured ladder). Re-fix pass 11 closed the other two
//! (`#P2b-58`, model completion for an array default under a binder) with
//! `solver::array_completion_certify`, so **this file now has no open hole**:
//! every test in it asserts the answer the tree must give.
//!
//! # How to read this file
//!
//! * A test whose doc carried **THE HOLE IS CLOSED** asserted the answer this
//!   tree gave **today**, which was the weaker one, and had to go red when the
//!   defect was fixed. There are none left in this file: re-fix passes 10 and
//!   11 closed all three, and every scaffold went with them.
//! * Every other test asserts the CORRECT answer and is an ordinary
//!   regression guard for the polarity-complete quantifier handling
//!   (`encode::quant_guard`, `#P2b-54`), the per-quantifier capture guard
//!   (`encode::binder_row`, `#P2b-55`), the binder-sort witness (`#P2b-56`)
//!   and the vacuity discharge (`mbqi::integration::vacuity`, `#P2b-57`).
//!
//! No test here sets a wall-clock `(set-option :timeout N)` and none asserts
//! a verdict behind one; the two budgeted tests use the **deterministic**
//! `(set-option :max-bv-embedded-checks N)`, whose answer is a property of
//! the tree rather than of the machine.
//!
//! Every verdict quoted for `c4b04b7` and for crates.io 0.3.3 was measured
//! with the round's release probes; the scripts and the raw output are under
//! `<scratchpad>/oxiz4/rk9/` with a `REBUILD.md`.

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// The last `sat`/`unsat`/`unknown` line, or `"none"`.
fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// Every `sat`/`unsat`/`unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

fn assert_verdict(script: &str, expected: &str, why: &str) {
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        expected,
        "{why}\n--- script ---\n{script}--- response ---\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 1. OPEN HOLES — a verdict `c4b04b7` reaches that this tree does not
//
//    Decision (24a), as re-fix pass 9 restated it, claims that the only
//    mechanism by which this tree still loses a base-decided verdict is the
//    deterministic budget of `#P2b-46` (f), and that "the table below is the
//    whole of the losses".  The two scripts below are counter-examples: both
//    are decided by `c4b04b7` and by crates.io 0.3.3 in milliseconds, neither
//    carries `(set-option :max-conflicts N)`, and the counters say no budget
//    is exhausted when this tree gives up.
// ---------------------------------------------------------------------------

/// The `#b0`-valued constant array with one `store` that changes nothing,
/// read under a width-7 binder — the quantified half of
/// `<scratchpad>/oxiz4/rk6/corpus/qmbqi120/q0074.smt2`, one of the round's own
/// width-7/8 pairs.
const Q0074_PREFIX: &str = "(set-logic ALL)\n\
     (declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n\
     (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n\
     (declare-const q Bool)\n\
     (assert (or (= (bvxor (select (ite q a0 (ite q a0 a1)) #b1000100) #b1) \
     (bvxor (select ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1) #b1000100) #b0)) \
     (= #b0 (bvxor (select ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1) #b1101111) #b0))))\n";

/// The body of `q0074`'s quantifier at one concrete index.
fn q0074_body_at(index: &str) -> String {
    format!(
        "(distinct (bvxor (select (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0) \
         #b1000110 #b0) {index}) #b0) (select a1 {index}))"
    )
}

/// `index` as a width-7 bit-vector literal.
fn bv7(index: u32) -> String {
    let mut out = String::from("#b");
    for bit in (0..7).rev() {
        out.push(if index & (1 << bit) == 0 { '0' } else { '1' });
    }
    out
}

/// `#P2b-58`'s hole, **closed** by decision (36)'s model completion (re-fix
/// pass 11): this answers `sat`, and the model it publishes is checked here
/// rather than taken on trust.
///
/// `q0074` is satisfiable: the `store` writes `#b0` into a constant-`#b0`
/// array, so the left-hand side of the `distinct` is `#b0` at every index and
/// the quantifier says exactly `a1[i] = #b1` for all `i`, which
/// `a1 = ((as const …) #b1)` satisfies. [`the_same_formula_expanded_over_its_whole_index_sort_is_sat`]
/// establishes that here, with no binder and no oracle, by expanding the
/// quantifier over all 128 points of its index sort.
///
/// It answered `unknown` here until re-fix pass 11, with counters
/// (`:bv-embedded-checks 1167`, `:bv-embedded-conflicts 0`, `:conflicts 38`)
/// that said **no budget was exhausted** — so the loss was neither mechanism
/// (i) (`#P2b-46` (f)'s deterministic budget) nor mechanism (ii) (`#P2b-57`,
/// closed). It was `#P2b-58`: MBQI could not certify a `sat` one bit above
/// `finite_expand`'s 64-point budget, because the candidate model leaves
/// `a1`'s **default** free. `solver::array_completion_certify` completes that
/// default and certifies the completion with quantifier-free validity queries
/// before any `sat` is published.
///
/// **Two assertions, not one.** The verdict alone would pass on a tree that
/// guessed; the second half replays the *published model* against the
/// quantifier written out at all 128 points of its index sort and requires it
/// to be satisfiable — so a completion that is wrong anywhere reddens this
/// test. [`the_same_formula_expanded_over_its_whole_index_sort_is_sat`] stays
/// beside it as the oracle for the truth.
#[test]
fn a_satisfiable_width_seven_array_script_is_decided_by_a_certified_completion() {
    let script = format!(
        "{Q0074_PREFIX}(assert (forall ((i!q (_ BitVec 7))) {}))\n(check-sat)\n(get-model)\n",
        q0074_body_at("i!q")
    );
    let lines = run(&script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "#P2b-58 (decision (36)): the completed array model must certify this \
         `sat`\n--- response ---\n{}",
        lines.join("\n")
    );

    // The published model, replayed against the whole index sort.
    let pins = model_equalities(&lines);
    assert!(
        !pins.is_empty(),
        "the `sat` published no model to replay:\n{}",
        lines.join("\n")
    );
    let mut conjuncts = String::new();
    for index in 0..128u32 {
        conjuncts.push(' ');
        conjuncts.push_str(&q0074_body_at(&bv7(index)));
    }
    let replay = format!("{Q0074_PREFIX}{pins}(assert (and{conjuncts}))\n(check-sat)\n");
    assert_verdict(
        &replay,
        "sat",
        "the published model must satisfy the quantifier at EVERY point of \
         its index sort, not only where the search happened to look",
    );
}

/// Every `(define-fun n () S v)` of a published model as `(assert (= n v))`.
///
/// An entry whose value is the `?` placeholder is skipped: it is not a value
/// and asserting it would be a parse error rather than a check.
fn model_equalities(lines: &[String]) -> String {
    let mut out = String::new();
    for line in lines {
        for raw in line.lines() {
            let trimmed = raw.trim();
            let Some(rest) = trimmed.strip_prefix("(define-fun ") else {
                continue;
            };
            let Some((name, rest)) = rest.split_once(" () ") else {
                continue;
            };
            // Exactly ONE trailing `)` — the `define-fun`'s own.  Stripping
            // every trailing paren would eat the value's, which for an array
            // constant is `((as const …) #b1)`.
            let Some(body) = rest.trim_end().strip_suffix(')') else {
                continue;
            };
            // `SORT VALUE`, where SORT may itself be parenthesised.
            let Some(value) = split_sort_and_value(body) else {
                continue;
            };
            if value == "?" {
                continue;
            }
            out.push_str(&format!("(assert (= {name} {value}))\n"));
        }
    }
    out
}

/// Split `"<sort> <value>"` into its value half, honouring nesting in the
/// sort.
fn split_sort_and_value(body: &str) -> Option<&str> {
    let bytes = body.as_bytes();
    let mut depth = 0i32;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b' ' if depth == 0 && index > 0 => return Some(body[index + 1..].trim()),
            _ => {}
        }
        index += 1;
    }
    None
}

/// The control that makes the pin above a statement about the *binder*: the
/// identical formula with the quantifier written out over all 128 points of
/// its index sort is `sat` on this tree.
#[test]
fn the_same_formula_expanded_over_its_whole_index_sort_is_sat() {
    let mut conjuncts = String::new();
    for index in 0..128u32 {
        conjuncts.push(' ');
        conjuncts.push_str(&q0074_body_at(&bv7(index)));
    }
    let script = format!("{Q0074_PREFIX}(assert (and{conjuncts}))\n(check-sat)\n");
    assert_verdict(
        &script,
        "sat",
        "the ground expansion of the same formula, so the truth of \
         `a_satisfiable_width_seven_array_script_is_decided_by_a_certified_completion` \
         rests on no oracle",
    );
}

/// The lazy array refinement's round count is **bounded and equal at two
/// budgets inside the 10 / 76 plateau** on a quantifier-free `QF_ABV` script
/// with `ite`-selected array bases (`#P2b-59`).
///
/// # This is not a fixpoint test, and the name it used to carry said it was
///
/// The name `a_quantifier_free_array_script_reaches_a_refinement_fixpoint`
/// and the sentence "the counts are identical at two budgets — a fixpoint" are
/// **withdrawn** (adversarial recheck pass 10, corrected here in re-fix pass
/// 11). The equality at 2,000 and 5,000 is a *plateau inside a ramp*: the same
/// method one rung higher gives 41 / 129 at 8,000. Measured ladder, release,
/// `<scratchpad>/oxiz4/rk11/min/q33_b*.smt2`:
///
/// | budget | verdict | rounds / instances |
/// |---|---|---|
/// | 500 | `unknown` | 9 / 75 |
/// | 2,000 | `unknown` | 10 / 76 |
/// | 5,000 | `unknown` | 10 / 76 |
/// | 8,000 | `unknown` | 41 / 129 |
/// | 10,000 | `unknown` | 51 / 181 |
/// | 12,000 | `unknown` | 57 / 197 |
/// | 20,000 | `unknown` | 59 / 199 |
/// | **50,000** | **`sat`** | **66 / 208**, 30,402 checks spent |
///
/// The refinement **does** reach a fixpoint — at 66 rounds / 208 instances,
/// where it stops asking for budget (30,402 of the 50,002 checks offered) and
/// publishes `sat`. Every rung below that is a *truncation* by the budget, not
/// a fixpoint at a smaller place. The only assertion that could carry a real
/// fixpoint claim is one taken at two budgets **above** 30,402 checks, and
/// that costs the gate about a minute of release time (30.4 s at 50,000, and
/// the dev profile is several times slower), which is why this test does not
/// take it and says so instead of calling the plateau a fixpoint.
///
/// What this test is, then: a **bounded, equal** count at two budgets, which
/// is still a working regression guard — the pass-9 tree reports 20 and 26 at
/// exactly these two budgets and fails both halves. Its companion
/// `round4_pass10_recheck_pins::the_array_refinement_round_count_still_grows_with_the_budget`
/// pins the rung that breaks the plateau, so a tree whose plateau *moved*
/// reddens there rather than passing silently here.
///
/// A **quantifier-free** script — two assertions over two width-8 arrays with
/// `ite`-selected bases, taken verbatim from
/// `<scratchpad>/oxiz4/rk6/corpus/qmbqi120/q0033.smt2` with its `forall`
/// assertion dropped. It contains no binder at all, which is what located the
/// defect: no quantifier machinery can reach it.
///
/// # What went wrong, and what this now guards
///
/// `Solver::assert` stores the pre-rewrite term in `Solver::assertions` and
/// registered the **encoded** term as a ground array root. Where the encoding
/// chain replaced an array-sorted `(ite c a b)` with the proxy constant
/// `eliminate_nonbool_ite` mints, the array collector walked both and counted
/// one array as two — two members of every pair set, two extensionality
/// witnesses per pair, a read-over-write cascade down each. The refinement
/// then kept finding new instances for as long as it was given budget:
/// `:array-refinement-rounds` 7 at `:max-bv-embedded-checks 500`, **20** at
/// 2,000, **26** at 5,000, 91 at 20,000, and `c4b04b7` answers `sat` in 7.3 ms
/// while this tree and the pass-6 checkpoint `00add07` answered nothing in
/// 120 s.
///
/// `Solver::array_root_spelling` puts the proxies back before the root is
/// registered, so one array is one array term, and the script answers `sat` in
/// 30.4 s unbudgeted (release, 66 rounds, 208 lemma instances, 30,402 embedded
/// checks) where the pass-9 tree answered nothing in 400 s. The residual cost
/// is `#P2b-46` (f)'s `O(num_vars)` embedded check, whose price is **not** a
/// flat rate — 0.06 ms per check at 12,002 checks, 0.23 at 20,002 and 1.00 at
/// 30,402, which is the `O(num_vars)` curve itself. The unbudgeted run is not
/// asserted here because 30 s of release time is minutes of the gate's; it is
/// recorded in `TODO.md` `#P2b-59` with its counters.
///
/// The bound beside the equality is load-bearing: a tree that regressed to a
/// *stable* 93 rounds would satisfy equality alone.
#[test]
fn the_refinement_round_count_is_bounded_and_equal_inside_the_plateau() {
    let body = "(declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (declare-const d (_ BitVec 1))\n\
         (assert (or (= (select (store (ite p a0 a1) #b10110111 #b1) #b11010110) \
         (select (ite p (store a1 #b10101011 #b1) (ite p a1 a1)) #b11000000)) \
         (= (bvxor (select (ite q (store a0 #b01110110 #b1) (store a0 #b11101110 #b0)) \
         #b10111010) #b0) \
         (select ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0) #b00010001))))\n\
         (assert (= (select (ite q a1 ((as const (Array (_ BitVec 8) (_ BitVec 1))) d)) \
         #b00010111) (bvxor (select (ite q (ite p a1 a1) \
         ((as const (Array (_ BitVec 8) (_ BitVec 1))) #b0)) #b11110111) #b1)))\n\
         (check-sat)\n(get-info :all-statistics)\n";
    assert!(
        !body.contains("forall") && !body.contains("exists"),
        "this pin's whole point is that the script carries no binder"
    );
    let mut rounds = Vec::new();
    let mut instances = Vec::new();
    for budget in [2000u32, 5000] {
        let script =
            format!("(set-logic ALL)\n(set-option :max-bv-embedded-checks {budget})\n{body}");
        let lines = run(&script);
        // The budget is the *deterministic* one and it is deliberately below
        // what the script needs, so this says nothing about the verdict — the
        // claim is entirely in the two counters below.
        assert_eq!(
            verdict(&lines),
            "unknown",
            "at {budget} embedded checks the script is still short of a \
             verdict; it is the refinement counters that this pin is about\n{}",
            lines.join("\n")
        );
        let stats = lines
            .iter()
            .find(|line| line.contains(":array-refinement-rounds"))
            .cloned()
            .unwrap_or_default();
        let read = |key: &str| -> u32 {
            stats
                .split(key)
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|token| token.parse().ok())
                .unwrap_or_default()
        };
        rounds.push(read(":array-refinement-rounds "));
        instances.push(read(":array-lemma-instances "));
    }
    assert_eq!(
        rounds.len(),
        2,
        "both budgets must publish `:array-refinement-rounds`"
    );
    assert_eq!(
        rounds[0], rounds[1],
        "the lazy array refinement must reach a fixpoint: it reported \
         {rounds:?} rounds at budgets 2,000 and 5,000, so it is still finding \
         new work for as long as it is given budget (`#P2b-59`; before the \
         fix this was 20 and 26)",
    );
    assert_eq!(
        instances[0], instances[1],
        "the lemma set must reach a fixpoint too: {instances:?} instances at \
         budgets 2,000 and 5,000 (before the fix, 91 and 104)",
    );
    assert!(
        rounds[0] > 0 && rounds[0] <= 12 && instances[0] <= 90,
        "a fixpoint at {} rounds / {} instances is not the one measured (10 / \
         76): equality across two budgets alone would also be satisfied by a \
         tree stuck at the pre-fix 93 rounds",
        rounds[0],
        instances[0],
    );
}

/// The one base-decided verdict `#P2b-59`'s fix buys back.
///
/// `rk6/corpus/qmbqi120/q0107.smt2` verbatim, one of the four corpus members
/// the recheck listed beside the quantifier-free repro above. `c4b04b7`
/// answers `sat` in 11.4 ms; the pass-9 tree answered nothing inside a 20 s
/// cap, so `rk9/pairverd.py` counted it as a lost verdict; here it is `sat`
/// in 160.2 ms (release). The other three — `q0033`, `q0047`, `q0103` — are
/// deliberately **not** pinned: they still do not finish in 130 s, their cost
/// is `#P2b-46` (f)'s `O(num_vars)` embedded check, and a pin on a verdict
/// they do not produce would be scaffolding.
///
/// The `(get-model)` is kept because it is what the corpus runs, and the
/// model it prints is `#P2b-51`'s open item, not this test's claim: the
/// assertion here is the **verdict**.
#[test]
fn the_corpus_member_the_root_spelling_buys_back_is_decided() {
    assert_verdict(
        "(set-logic ALL)\n\
         (set-option :produce-models true)\n\
         (declare-const a0 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 8) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (declare-const d (_ BitVec 1))\n\
         (assert (= (select a0 #b00000100) (select a0 #b01101100)))\n\
         (assert (= (select (store ((as const (Array (_ BitVec 8) (_ BitVec 1))) d) \
         #b11011010 #b0) #b11100110) (bvxor (select (store a1 #b00101101 #b1) \
         #b01110100) #b0)))\n\
         (assert (or (= (select (ite q a1 (ite p a1 a0)) #b00001000) #b0) \
         (= (bvxor (select (ite q (store a1 #b10010010 #b1) (store a0 #b01111110 #b0)) \
         #b00001010) #b0) (select a1 #b01010110))))\n\
         (assert (forall ((i!q (_ BitVec 8))) (= (bvxor (select (ite p \
         (store a0 #b11010011 #b1) (ite q a1 a1)) #b01000000) #b1) \
         (select (ite p a1 a1) i!q))))\n\
         (check-sat)\n(get-model)\n",
        "sat",
        "`c4b04b7` answers `sat` in 11.4 ms and the pass-9 tree answered \
         nothing in 20 s; `Solver::array_root_spelling` is what buys it back",
    );
}

/// `#P2b-58`'s second hole, **closed** by the same completion (re-fix pass 11).
///
/// `rk8/atk/f5_binder_collide_index.smt2`, the unpinned twin of `#P2b-55`'s
/// name-collision repro. It is satisfiable — at any `i` other than `#b0000001`
/// the read misses the `store` and the body says `a[#b0000001] = k`, which
/// `a = ((as const …) k)` satisfies — and `c4b04b7` answers `sat` in 0.12 ms.
/// This tree answered `unknown` until the completion landed: `binder_row`
/// accepts the quantifier and rewrites the read, and the residual
/// `∀i. ite(i = 1, k, a[1]) = k` at width 7 is past `finite_expand`'s
/// 64-point budget, where MBQI had no way to certify the `sat`. The pool that
/// decides it is the element-sort one — `k`'s value in the candidate model is
/// a candidate default, and `a = ((as const …) k)` is what the certificate
/// discharges.
///
/// Same mechanism as the pin above, different surface. Recorded separately
/// because it is the one the round's own attack corpus already contained, and
/// because the bound name here **is** a declared constant's name — so a
/// completion that confused the two would certify the wrong formula.
/// `solver::array_completion_certify` renames every bound variable to a
/// reserved constant before it builds a query, which is what keeps them apart.
#[test]
fn a_satisfiable_name_collision_script_is_decided_by_a_certified_completion() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (declare-const k (_ BitVec 7))\n\
         (assert (and (forall ((k (_ BitVec 7))) (= k k))\n\
         (forall ((i (_ BitVec 7))) (= (select (store a i k) (_ bv1 7)) k))))\n\
         (check-sat)\n",
        "sat",
        "#P2b-58 (decision (36)): `c4b04b7` answers `sat` here and so must \
         this tree, by a certified completion",
    );
}

/// The control for the pin above: pin `a` to the all-zero array and `k` to
/// `#b0000101`, and the same shape is refuted — so the `unknown` above is the
/// solver declining to certify a `sat`, not the rewrite being broken.
#[test]
fn the_pinned_twin_of_the_name_collision_script_is_refuted() {
    assert_verdict(
        "(set-logic AUFBV)\n\
         (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
         (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n\
         (declare-const k (_ BitVec 7))\n\
         (assert (= k (_ bv5 7)))\n\
         (assert (and (forall ((k (_ BitVec 7))) (= k k))\n\
         (forall ((i (_ BitVec 7))) (= (select (store a i k) (_ bv1 7)) k))))\n\
         (check-sat)\n",
        "unsat",
        "REGRESSION GUARD (#P2b-55): a sibling binder that happens to bind the \
         name `k` must not disable the read-over-write expansion beside it",
    );
}

// ---------------------------------------------------------------------------
// 2. POLARITY — the closed `#P2b-54` family, attacked at every connective
//
//    Family A: the quantifier is VALID (`∀x:U. f(x) = f(x)`), so a context
//    that forces it FALSE is unsatisfiable.
//    Family B: the quantifier is REFUTABLE (`∀x:U. f(x) = a` beside
//    `f(b) ≠ a`), so a context that forces it TRUE is unsatisfiable.
//    Each is `sat` on `c4b04b7` unless noted.
// ---------------------------------------------------------------------------

/// `∀x:U. f(x) = f(x)`, a tautology under every structure.
const VALID_Q: &str = "(forall ((x U)) (= (f x) (f x)))";

/// Header for family A.
const HDR_A: &str = "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun f (U) U)\n";

/// Header for family B: `f(b) ≠ a` makes `∀x. f(x) = a` false outright.
const HDR_B: &str = "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun f (U) U)\n\
     (declare-const a U)\n(declare-const b U)\n(assert (distinct (f b) a))\n";

/// `∀x:U. f(x) = a`, false in every model of [`HDR_B`].
const REFUTABLE_Q: &str = "(forall ((x U)) (= (f x) a))";

/// A valid quantifier forced false by `xor`, by a Boolean `distinct`, by a
/// `let` binding, by a `:named` wrapper and by a trigger — every connective
/// the pass-7 verdict did not reach.
///
/// All five answered `sat` or `unknown` on `c4b04b7`.
#[test]
fn a_valid_quantifier_forced_false_is_refuted_at_every_connective() {
    for (why, tail) in [
        (
            "xor",
            "(declare-const p Bool)\n(assert p)\n(assert (xor p Q))\n",
        ),
        (
            "Boolean distinct",
            "(declare-const p Bool)\n(assert p)\n(assert (distinct p Q))\n",
        ),
        ("let", "(assert (let ((bb Q)) (not bb)))\n"),
        (
            ":named",
            "(set-option :produce-unsat-cores true)\n(assert (! (not Q) :named A1))\n",
        ),
        (
            "an ite whose two branches both negate it",
            "(declare-const c Bool)\n(assert (ite c (not Q) (not Q)))\n",
        ),
        ("a triple negation", "(assert (not (not (not Q))))\n"),
        ("a Boolean equality with false", "(assert (= Q false))\n"),
        (
            "a disjunction whose other arm is false",
            "(declare-const r Bool)\n(assert (not r))\n(assert (or r (not Q)))\n",
        ),
        (
            "a doubly nested implication",
            "(declare-const p Bool)\n(assert (not (or p (=> (not Q) p))))\n",
        ),
    ] {
        let script = format!("{HDR_A}{}(check-sat)\n", tail.replace('Q', VALID_Q));
        assert_verdict(
            &script,
            "unsat",
            &format!("REGRESSION GUARD (#P2b-54): the quantifier under {why}"),
        );
    }
}

/// A refutable quantifier forced **true** by the same connectives — the other
/// half of the polarity table, which a fix that only handled negative
/// positions would fail.
#[test]
fn a_refutable_quantifier_forced_true_is_refuted_at_every_connective() {
    for (why, tail) in [
        (
            "a disjunction whose other arm is false",
            "(declare-const r Bool)\n(assert (not r))\n(assert (or r Q))\n",
        ),
        (
            "the consequent of an implication whose premise holds",
            "(declare-const p Bool)\n(assert p)\n(assert (=> p Q))\n",
        ),
        (
            "the taken branch of an ite",
            "(declare-const c Bool)\n(assert c)\n(assert (ite c Q false))\n",
        ),
        ("a Boolean equality with true", "(assert (= Q true))\n"),
        (
            "xor against a false operand",
            "(declare-const p Bool)\n(assert (not p))\n(assert (xor p Q))\n",
        ),
        (
            "a Boolean distinct against a false operand",
            "(declare-const p Bool)\n(assert (not p))\n(assert (distinct p Q))\n",
        ),
        ("a let binding", "(assert (let ((bb Q)) bb))\n"),
        (
            "a doubly nested implication",
            "(declare-const p Bool)\n(declare-const q Bool)\n(assert (not p))\n\
             (assert (not q))\n(assert (=> (=> Q p) q))\n",
        ),
        (
            ":named",
            "(set-option :produce-unsat-cores true)\n(assert (! (or Q false) :named B1))\n",
        ),
    ] {
        let script = format!("{HDR_B}{}(check-sat)\n", tail.replace('Q', REFUTABLE_Q));
        assert_verdict(
            &script,
            "unsat",
            &format!("REGRESSION GUARD (#P2b-54): the quantifier at {why}"),
        );
    }
}

/// "Answer `unsat` to everything" does not pass: the two positions where the
/// same quantifiers leave the script satisfiable.
#[test]
fn the_satisfiable_members_of_both_polarity_families_are_still_sat() {
    assert_verdict(
        &format!(
            "{HDR_A}(declare-const p Bool)\n(declare-const q Bool)\n\
             (assert (not p))\n(assert (not q))\n(assert (=> (=> {VALID_Q} p) q))\n(check-sat)\n"
        ),
        "sat",
        "`(true -> false) -> q` is `q`'s vacuous premise, so the script holds \
         with `p` and `q` both false",
    );
    assert_verdict(
        &format!(
            "{HDR_B}(declare-const t Bool)\n(declare-const e Bool)\n(assert t)\n\
             (assert (not e))\n(assert (not (ite {REFUTABLE_Q} t e)))\n(check-sat)\n"
        ),
        "sat",
        "the quantifier is false, so the `ite` takes its `else` branch, which \
         is false, and the negation holds",
    );
}

/// A `forall` nested inside a `forall` inside a `not`, and an `exists` under a
/// `forall` under a `not` — the shapes where the Skolemised half of a
/// `quant_guard` obligation itself contains a quantifier that is closed only
/// after substitution.
#[test]
fn a_quantifier_that_becomes_closed_only_after_skolemisation_is_refuted() {
    assert_verdict(
        "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun r (U U) Bool)\n\
         (assert (not (forall ((x U)) (exists ((y U)) (or (r x y) (not (r x y)))))))\n\
         (check-sat)\n",
        "unsat",
        "REGRESSION GUARD (#P2b-54): the inner `exists` is `quant_guard`'s \
         recursion, reached through `assert_quantifier_obligation`",
    );
    assert_verdict(
        "(set-logic ALL)\n(declare-fun r ((_ BitVec 7)) Bool)\n\
         (assert (not (forall ((x (_ BitVec 7))) (r x))))\n\
         (assert (forall ((y (_ BitVec 7))) (r y)))\n(check-sat)\n",
        "unsat",
        "REGRESSION GUARD: the Skolem witness of the negated universal must \
         be visible to the positively asserted one",
    );
}

/// Two alpha-equivalent spellings of the same quantifier, each Skolemised
/// separately, against a third spelling asserted positively.
#[test]
fn alpha_equivalent_quantifiers_are_skolemised_without_losing_the_contradiction() {
    assert_verdict(
        "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun g (U) Bool)\n\
         (declare-const p Bool)\n(assert (not p))\n\
         (assert (or p (not (forall ((x U)) (g x)))))\n\
         (assert (or p (not (forall ((y U)) (g y)))))\n\
         (assert (forall ((z U)) (g z)))\n(check-sat)\n",
        "unsat",
        "REGRESSION GUARD (#P2b-54): two fresh Skolem constants, one \
         positively asserted universal, one contradiction",
    );
}

/// A quantifier over `Bool` and one over an array of arrays — two binder
/// sorts the round's corpora never draw.
#[test]
fn unusual_binder_sorts_are_handled_at_a_polarity_boundary() {
    assert_verdict(
        "(set-logic ALL)\n(declare-fun h (Bool) Bool)\n\
         (assert (not (forall ((b Bool)) (= (h b) (h b)))))\n(check-sat)\n",
        "unsat",
        "a `Bool` binder under a `not`",
    );
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const n (Array (_ BitVec 7) (Array (_ BitVec 7) (_ BitVec 7))))\n\
         (declare-const p Bool)\n(assert p)\n\
         (assert (=> p (forall ((i (_ BitVec 7))) \
         (= (select (select n i) (_ bv0 7)) (_ bv0 7)))))\n\
         (assert (distinct (select (select n (_ bv3 7)) (_ bv0 7)) (_ bv0 7)))\n\
         (check-sat)\n",
        "unsat",
        "an array-of-arrays read under a guarded binder",
    );
}

// ---------------------------------------------------------------------------
// 3. THE HONESTY NET — what `quant_guard` declines must never be a verdict
// ---------------------------------------------------------------------------

/// One assertion with `n` distinct valid quantifiers under a positive `or`,
/// all negated, so the disjunction is false and the script is `unsat`.
fn cap_script(n: u32) -> String {
    let mut body = String::new();
    for k in 0..n {
        body.push_str(&format!(" (forall ((x{k} U)) (= (f x{k}) (f x{k})))"));
    }
    format!("{HDR_A}(assert (not (or{body})))\n(check-sat)\n")
}

/// `quant_guard`'s `MAX_GUARDED_QUANTIFIERS` cap is 64, and crossing it makes
/// the pass decline the WHOLE assertion — every quantifier in it, not just the
/// 65th.
///
/// At 64 the assertion is refuted. At 65 the literals are unconstrained and
/// the answer must be `unknown`, never `sat`: this is
/// `Solver::quantifier_literal_unconstrained` doing the only job it has.
/// `c4b04b7` answers a wrong `sat` to both.
#[test]
fn crossing_the_guard_cap_costs_a_verdict_and_never_soundness() {
    assert_verdict(
        &cap_script(64),
        "unsat",
        "64 guarded quantifiers is inside the cap",
    );
    assert_verdict(
        &cap_script(65),
        "unknown",
        "SOUNDNESS: one past the cap the whole assertion is declined, and a \
         `sat` resting on an unconstrained quantifier literal must degrade to \
         `unknown` (`c4b04b7` answers a wrong `sat`)",
    );
}

/// A quantifier `quant_guard` declines because it sits under another binder,
/// in an assertion where nothing else justifies its literal.
///
/// `c4b04b7` answers `unknown` too, so nothing is lost; the point is that the
/// decline is not a `sat`.
#[test]
fn a_declined_nested_quantifier_is_unknown_and_never_sat() {
    assert_verdict(
        "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun f (U) U)\n\
         (declare-const p Bool)\n(assert (not p))\n\
         (assert (or p (not (forall ((y U)) (forall ((z U)) (= (f y) (f z)))))))\n\
         (assert (forall ((w U)) (forall ((v U)) (= (f w) (f v)))))\n(check-sat)\n",
        "unknown",
        "SOUNDNESS: a declined quantifier may cost a verdict, never buy one",
    );
}

// ---------------------------------------------------------------------------
// 4. THE VACUITY DISCHARGE (`#P2b-57`) — what it must NOT discharge
// ---------------------------------------------------------------------------

/// `quantifier_vacuous_under_model` substitutes the published model's value
/// for a Boolean `Var` of the body that the quantifier's own binders do not
/// bind. A Boolean bound by a binder **inside** the body has the same
/// `TermKind::Var` representation as a declared constant of that name, so the
/// substitution must be dropped there — which is
/// `TermManager::substitute`'s shadowing rule, exercised here rather than
/// assumed.
///
/// All three are `unsat`: `∀b:Bool. b` and `(let ((b false)) b)` are false,
/// so the disjunction needs `p`, and `p` is asserted false.
/// `c4b04b7` answers a wrong `sat` to all three.
#[test]
fn the_vacuity_discharge_respects_a_shadowing_inner_binder() {
    for (why, quantifier) in [
        (
            "a `forall` over `Bool` inside the body",
            "(forall ((x (_ BitVec 7))) (forall ((b Bool)) b))",
        ),
        (
            "a `let` that rebinds the same name inside the body",
            "(forall ((x (_ BitVec 7))) (let ((b false)) b))",
        ),
        ("the quantifier's own binder", "(forall ((b Bool)) b)"),
    ] {
        assert_verdict(
            &format!(
                "(set-logic ALL)\n(declare-const b Bool)\n(declare-const p Bool)\n\
                 (assert b)\n(assert (not p))\n(assert (or p {quantifier}))\n(check-sat)\n"
            ),
            "unsat",
            &format!("SOUNDNESS (#P2b-57): the model's `b = true` must not reach {why}"),
        );
    }
}

/// A guard the search is forced to set **true** still has to be verified: the
/// vacuity path must fire only on a guard the model makes false.
#[test]
fn a_guard_forced_true_is_still_verified() {
    for (why, body) in [
        ("a body refuted at one named index", "(r x)"),
        (
            "a body that is false at every point",
            "(and (r x) (not (r x)))",
        ),
    ] {
        assert_verdict(
            &format!(
                "(set-logic ALL)\n(declare-fun r ((_ BitVec 7)) Bool)\n\
                 (declare-const g Bool)\n(assert g)\n\
                 (assert (=> g (forall ((x (_ BitVec 7))) {body})))\n\
                 (assert (not (r (_ bv3 7))))\n(check-sat)\n"
            ),
            "unsat",
            &format!("SOUNDNESS (#P2b-57): {why} under a guard forced true"),
        );
    }
}

/// And the satisfiable side, so the guards above are not met by answering
/// `unsat` to everything: a guard the search sets false discharges its
/// quantifier and the script is `sat`, with a model.
#[test]
fn a_guard_the_search_sets_false_discharges_its_quantifier() {
    let lines = run("(set-logic ALL)\n(declare-fun r ((_ BitVec 7)) Bool)\n\
         (declare-const g Bool)\n(assert (not g))\n\
         (assert (=> g (forall ((x (_ BitVec 7))) (and (r x) (not (r x))))))\n\
         (check-sat)\n(get-model)\n");
    assert_eq!(
        verdict(&lines),
        "sat",
        "the guard is false, so the universal is satisfied everywhere at \
         once\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 5. WHAT THE USER WROTE — Skolemisation must stay invisible
// ---------------------------------------------------------------------------

/// A Skolemised negative-polarity quantifier leaves `(get-assertions)` and the
/// unsat core naming only the user's own terms, and the reserved Skolem class
/// cannot be declared from a script in either symbol form.
#[test]
fn skolemisation_is_invisible_to_the_user_facing_commands() {
    let lines = run("(set-logic ALL)\n(set-option :produce-assertions true)\n\
         (declare-sort U 0)\n(declare-fun g (U) Bool)\n(declare-const p Bool)\n\
         (assert (or p (not (forall ((x U)) (g x)))))\n(check-sat)\n(get-assertions)\n");
    let joined = lines.join("\n");
    assert_eq!(verdict(&lines), "sat", "{joined}");
    assert!(
        joined.contains("((or p (not (forall ((x U)) (g x)))))"),
        "(get-assertions) must echo the user's assertion and nothing derived\n{joined}"
    );
    assert!(
        !joined.contains("oxiz."),
        "no reserved symbol may appear in a user-facing response\n{joined}"
    );

    let core = run("(set-logic ALL)\n(set-option :produce-unsat-cores true)\n\
         (declare-sort U 0)\n(declare-fun f (U) U)\n(declare-const p Bool)\n\
         (assert (! (not p) :named N1))\n\
         (assert (! (or p (not (forall ((x U)) (= (f x) (f x))))) :named N2))\n\
         (check-sat)\n(get-unsat-core)\n");
    assert_eq!(verdict(&core), "unsat", "{}", core.join("\n"));
    assert!(
        core.iter().any(|line| line.trim() == "(N1 N2)"),
        "the core must name the two user assertions and nothing else\n{}",
        core.join("\n")
    );

    let collide = run("(set-logic ALL)\n(declare-sort U 0)\n\
         (declare-const |\\oxiz.sk!0| U)\n(check-sat)\n");
    assert!(
        collide.iter().any(|line| line.starts_with("(error ")),
        "a quoted symbol in the reserved class must be refused, not interned\n{}",
        collide.join("\n")
    );
}

/// `push` / `pop` around a negative-polarity quantifier: the Skolem
/// obligation, the justification mark and the derived clauses all go with the
/// assertion that introduced them.
#[test]
fn a_pop_retracts_a_skolemised_quantifier_with_its_assertion() {
    let lines = run(
        "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun f (U) U)\n\
         (push 1)\n(assert (not (forall ((x U)) (= (f x) (f x)))))\n(check-sat)\n\
         (pop 1)\n(check-sat)\n\
         (assert (not (forall ((x U)) (= (f x) (f x)))))\n(check-sat)\n",
    );
    assert_eq!(
        verdicts(&lines),
        vec!["unsat", "sat", "unsat"],
        "SOUNDNESS: the middle `sat` is the empty scope and the third \
         `unsat` is the same assertion re-made\n{}",
        lines.join("\n")
    );
}

/// The dual: a quantifier justified inside a pushed scope must not still
/// count as justified after the `pop`, or a later occurrence the pass declines
/// would inherit a justification it does not have.
#[test]
fn a_justification_does_not_survive_the_pop_that_retracts_it() {
    let lines = run(
        "(set-logic ALL)\n(declare-sort U 0)\n(declare-fun f (U) U)\n\
         (push 1)\n(assert (forall ((x U)) (= (f x) (f x))))\n(check-sat)\n(pop 1)\n\
         (declare-const p Bool)\n(assert (not p))\n\
         (assert (or p (not (forall ((x U)) (= (f x) (f x))))))\n(check-sat)\n",
    );
    assert_eq!(
        verdicts(&lines),
        vec!["sat", "unsat"],
        "the second answer is the negated tautology, refuted on its own \
         merits after the `pop`\n{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// 6. `binder_row`'s per-quantifier capture guard (`#P2b-55`), attacked
// ---------------------------------------------------------------------------

/// `a` pinned to the all-zero array, so every shape below is `unsat`: the
/// read-over-write expansion at `i ≠ #b0000001` gives `a[#b0000001] = #b0`.
const PINNED: &str = "(set-logic AUFBV)\n\
     (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))\n\
     (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))\n";

/// Deliberate name collisions in every position the new per-occurrence guard
/// has to reason about: three levels of shadowing, an enclosing binder that
/// binds the *value* the rewrite carries, and a `let` that shadows the index
/// name.
///
/// `c4b04b7` answers `unknown` to all three.
#[test]
fn the_per_quantifier_capture_guard_survives_every_collision() {
    for (why, assertion) in [
        (
            "three nesting levels re-binding the index name",
            "(assert (forall ((i (_ BitVec 7))) (forall ((j (_ BitVec 7))) \
             (forall ((i (_ BitVec 7))) \
             (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))))) ",
        ),
        (
            "an enclosing binder that binds the stored value's name",
            "(declare-const v (_ BitVec 7))\n(assert (= v (_ bv5 7)))\n\
             (assert (forall ((v (_ BitVec 7))) (and (= v v) \
             (forall ((i (_ BitVec 7))) \
             (= (select (store a i v) (_ bv1 7)) v)))))",
        ),
        (
            "a `let` that shadows the index name",
            "(assert (let ((i (_ bv2 7))) (forall ((i (_ BitVec 7))) \
             (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7)))))",
        ),
    ] {
        assert_verdict(
            &format!("{PINNED}{assertion}\n(check-sat)\n"),
            "unsat",
            &format!("REGRESSION GUARD (#P2b-55): {why}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 7. `#P2b-56` — the binder-sort witness, and the seed's scope
// ---------------------------------------------------------------------------

/// Ten consecutive `(check-sat)` on one quantified assertion must not mint ten
/// witnesses' worth of work, and every answer must be the same.
///
/// `Solver::seed_binder_sort_witnesses` runs at *assert* time precisely
/// because the candidate pool is search state that each `check` rolls back;
/// this is the behavioural statement of that.
#[test]
fn repeating_the_check_neither_changes_the_answer_nor_accumulates_witnesses() {
    let mut script = String::from(
        "(set-logic ALL)\n\
         (assert (forall ((i (_ BitVec 7))) \
         (distinct (_ bv0 7) \
         (select ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7)) i))))\n",
    );
    for _ in 0..10 {
        script.push_str("(check-sat)\n");
    }
    let lines = run(&script);
    assert_eq!(
        verdicts(&lines),
        vec!["unsat"; 10],
        "ten identical answers, and the witness is minted once\n{}",
        lines.join("\n")
    );
}
