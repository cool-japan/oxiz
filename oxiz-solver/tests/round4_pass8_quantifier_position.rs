//! Round 4, re-fix pass 8: a quantifier in a **random Boolean position**
//! agrees with its own ground expansion.
//!
//! # What this corpus is for
//!
//! `#P2b-54` was not one spelling: it was every Boolean position that is not a
//! conjunct.  A fixed battery of eight spellings closes the eight it names and
//! says nothing about the ninth, so the guard that keeps
//! [`encode::quant_guard`] honest has to *generate* the position.
//!
//! Each pair below is one formula written twice:
//!
//! * the **quantified** script, with `(forall ((i!q (_ BitVec w))) body)` or
//!   `(exists …)` dropped into a randomly chosen Boolean context — `=>` on
//!   either side, `or`, `ite`, a Boolean `=`, `xor`, `not`, and a nested
//!   `and`/`or`;
//! * the **ground** script, identical except that the quantifier is replaced
//!   *in that same position* by the conjunction (`forall`) or disjunction
//!   (`exists`) of its body over the whole index sort.
//!
//! Over a finite index sort those two are the same formula, so ANY
//! disagreement is a defect and both directions are asserted:
//!
//! * `wrong_sat` — a quantified `sat` against a ground `unsat` — is
//!   `#P2b-54` itself, and was 100 % of the negative-polarity draws before the
//!   guard existed.
//! * `wrong_unsat` — a quantified `unsat` against a ground `sat` — is what an
//!   over-eager guard breaks: Skolemising a *positive* universal, or asserting
//!   an instance without its guard literal, produces exactly this.  It has
//!   never been non-zero and must stay zero.
//!
//! # Why two width bands
//!
//! At widths 1–2 (2 and 4 index points) `encode::finite_expand` consumes the
//! quantifier before the encoder runs, so the band measures the *expansion*
//! path at every polarity — including the polarities the expansion has to be
//! an equivalence at.  Width 7 (128 points) is past the 64-point budget, so
//! the quantifier survives to `quant_guard` and MBQI and the band measures the
//! guarded-instance path.  The defect began exactly one bit above the budget,
//! so a corpus that stops at width 2 cannot see it.
//!
//! The ground twins at width 7 carry 128 conjuncts, so the bodies drawn there
//! are deliberately single atoms: the twin is the oracle, and an oracle that
//! answers `unknown` measures nothing.  `undecided` is reported, not asserted.
//!
//! Rebuild of the out-of-tree campaign this is a scale model of:
//! `<scratchpad>/oxiz4/rf9/REBUILD.md`.

use oxiz_solver::Context;

/// xorshift64, so the corpus is a *fixed* corpus: the seeds below name every
/// script this test will ever run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One generated pair: the quantified script and the same formula with the
/// quantifier expanded in place over the finite index sort.
struct Pair {
    quantified: String,
    ground: String,
    context: &'static str,
}

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

fn bv_literal(value: u64, width: u32) -> String {
    let mut out = String::from("#b");
    for bit in (0..width).rev() {
        out.push(if (value >> bit) & 1 == 1 { '1' } else { '0' });
    }
    out
}

/// The Boolean positions a quantifier can occupy, with `{Q}` marking it.
///
/// Every one of these but the two `and` spellings is a position
/// `term_walk::asserted_children` does **not** descend through, which is
/// exactly the set `register_asserted_quantifiers` used to register nothing
/// for.  The two conjunctive spellings are in the list on purpose: they are
/// the position that already worked, and a guard that broke them would show
/// up here rather than in a bug report.
const CONTEXTS: [&str; 10] = [
    "(=> p {Q})",
    "(=> {Q} p)",
    "(or p {Q})",
    "(or (not {Q}) p)",
    "(ite p {Q} q)",
    "(= p {Q})",
    "(xor p {Q})",
    "(not {Q})",
    "(and p (or q {Q}))",
    "(and p {Q})",
];

fn elem_term(rng: &mut Rng, index_width: u32, elem_width: u32, bound: &str) -> String {
    match rng.below(4) {
        0 => bv_literal(rng.below(1 << elem_width), elem_width),
        1 => format!(
            "(select (store a {bound} {}) {})",
            bv_literal(rng.below(1 << elem_width), elem_width),
            bv_literal(rng.below(1 << index_width), index_width)
        ),
        2 => format!("(select a {bound})"),
        _ => format!(
            "(select a {})",
            bv_literal(rng.below(1 << index_width), index_width)
        ),
    }
}

fn atom(rng: &mut Rng, index_width: u32, elem_width: u32, bound: &str) -> String {
    let left = elem_term(rng, index_width, elem_width, bound);
    let right = elem_term(rng, index_width, elem_width, bound);
    if rng.below(4) == 0 {
        format!("(distinct {left} {right})")
    } else {
        format!("(= {left} {right})")
    }
}

/// One pair at `index_width`.
fn generate_pair(rng: &mut Rng, index_width: u32) -> Pair {
    let elem_width = if index_width >= 7 {
        1
    } else if rng.below(3) == 0 {
        2
    } else {
        1
    };
    let sort = format!("(Array (_ BitVec {index_width}) (_ BitVec {elem_width}))");
    let mut header = format!(
        "(set-logic ALL)\n\
         (declare-const a {sort})\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n"
    );
    // Pin the array sometimes, so a good share of the corpus is decidable
    // rather than trivially satisfiable.
    if rng.below(2) == 0 {
        header.push_str(&format!(
            "(assert (= a ((as const {sort}) {})))\n",
            bv_literal(rng.below(1 << elem_width), elem_width)
        ));
    }
    for _ in 0..rng.below(3) {
        let index = bv_literal(rng.below(1 << index_width), index_width);
        let ground = atom(rng, index_width, elem_width, &index);
        header.push_str(&format!("(assert {ground})\n"));
    }
    // Pin the Boolean the context branches on, so the context actually forces
    // something rather than being satisfied by its other arm.
    match rng.below(3) {
        0 => header.push_str("(assert p)\n"),
        1 => header.push_str("(assert (not p))\n"),
        _ => {}
    }

    let universal = rng.below(10) < 6;
    let body = atom(rng, index_width, elem_width, "i!q");
    let quantifier = if universal { "forall" } else { "exists" };
    let context = CONTEXTS[rng.below(CONTEXTS.len() as u64) as usize];

    let quantified_term = format!("({quantifier} ((i!q (_ BitVec {index_width}))) {body})");
    let mut expansion = String::new();
    for value in 0..(1u64 << index_width) {
        expansion.push(' ');
        expansion.push_str(&body.replace("i!q", &bv_literal(value, index_width)));
    }
    let joiner = if universal { "and" } else { "or" };
    let ground_term = format!("({joiner}{expansion})");

    Pair {
        quantified: format!(
            "{header}(assert {})\n(check-sat)\n",
            context.replace("{Q}", &quantified_term)
        ),
        ground: format!(
            "{header}(assert {})\n(check-sat)\n",
            context.replace("{Q}", &ground_term)
        ),
        context,
    }
}

/// The tally of one band.
struct Tally {
    wrong_sat: u32,
    wrong_unsat: u32,
    agree: u32,
    undecided: u32,
    first: String,
}

fn score(seed: u64, pairs: usize, widths: &[u32]) -> Tally {
    let mut rng = Rng(seed);
    let mut tally = Tally {
        wrong_sat: 0,
        wrong_unsat: 0,
        agree: 0,
        undecided: 0,
        first: String::new(),
    };
    for index in 0..pairs {
        let width = widths[index % widths.len()];
        let pair = generate_pair(&mut rng, width);
        let quantified = verdict(&run(&pair.quantified));
        let ground = verdict(&run(&pair.ground));
        match (quantified.as_str(), ground.as_str()) {
            ("sat", "unsat") => {
                tally.wrong_sat += 1;
                if tally.first.is_empty() {
                    tally.first = format!(
                        "context {}\n--- quantified (sat) ---\n{}--- ground (unsat) ---\n{}",
                        pair.context, pair.quantified, pair.ground
                    );
                }
            }
            ("unsat", "sat") => {
                tally.wrong_unsat += 1;
                if tally.first.is_empty() {
                    tally.first = format!(
                        "context {}\n--- quantified (unsat) ---\n{}--- ground (sat) ---\n{}",
                        pair.context, pair.quantified, pair.ground
                    );
                }
            }
            (a, b) if a == b => tally.agree += 1,
            _ => tally.undecided += 1,
        }
    }
    tally
}

fn assert_agrees(tally: &Tally, band: &str) {
    eprintln!(
        "[{band}] wrong_sat {} wrong_unsat {} agree {} undecided {}",
        tally.wrong_sat, tally.wrong_unsat, tally.agree, tally.undecided
    );
    assert_eq!(
        tally.wrong_unsat, 0,
        "[{band}] a quantified `unsat` against a ground `sat` is a wrong \
         `unsat`: a positive universal was Skolemised, or an instance was \
         asserted without its guard literal.  First offender:\n{}",
        tally.first
    );
    assert_eq!(
        tally.wrong_sat, 0,
        "[{band}] a quantified `sat` against a ground `unsat` is `#P2b-54`: \
         the quantifier's literal is free in this Boolean position.  First \
         offender:\n{}",
        tally.first
    );
}

/// **GUARD.**  Inside the finite-expansion budget, every Boolean position
/// agrees with its own expansion.
///
/// This band never reaches `quant_guard` — `finite_expand` replaces the
/// quantifier first — so what it pins is that the expansion is an equivalence
/// at *every* polarity, which is the claim its doc comment makes.
#[test]
fn a_quantifier_in_any_boolean_position_agrees_with_its_expansion_inside_the_budget() {
    let tally = score(0x0521_2026_0921_0001, 240, &[1, 2]);
    assert!(
        tally.agree >= 235,
        "[widths 1-2] only {} of 240 pairs were decided on both sides, \
         against 235 measured on this tree; the corpus has stopped measuring \
         anything",
        tally.agree
    );
    assert_agrees(&tally, "widths 1-2");
}

/// **GUARD.**  One index bit above the budget — where `#P2b-54` began — every
/// Boolean position still agrees with its own expansion.
///
/// At width 7 the quantifier survives the expansion, so this is the band that
/// exercises `quant_guard`'s two obligations and the guarded MBQI instances.
/// Fewer pairs, because each ground twin is 128 conjuncts.
///
/// The floor is the *strength* half of the guard, and re-fix pass 9 raised it
/// from 20 to the 55 it measures: `#P2b-57`'s two fixes — the vacuity
/// discharge and the narrowed blind-lemma filter — are exactly what turned 35
/// of these into decided verdicts, so a floor of 20 would no longer notice
/// their removal.  Disabling either reddens this assertion.
#[test]
fn a_quantifier_in_any_boolean_position_agrees_with_its_expansion_above_the_budget() {
    let tally = score(0x0521_2026_0921_0007, 60, &[7]);
    assert!(
        tally.agree >= 55,
        "[width 7] only {} of 60 pairs were decided on both sides, against 55 \
         measured on this tree; the corpus has stopped measuring the seam it \
         was written for",
        tally.agree
    );
    assert_agrees(&tally, "width 7");
}
