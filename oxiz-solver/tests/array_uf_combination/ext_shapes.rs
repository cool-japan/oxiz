//! Differential campaign for array **extensionality** (`#P2b-37`), scored
//! against an exhaustive oracle, with every published model replayed through
//! the reference semantics in this file.
//!
//! The parent campaign generates reads and writes; the defects `#P2b-37`
//! closes live in the shapes it never produced:
//!
//! * an array *(dis)equality* atom, `n`-ary `distinct` over arrays included —
//!   `(distinct arr brr)` with every index pinned equal is unsat and needs a
//!   witness the `distinct` arm never recorded;
//! * `store = ((as const S) d)`, whose only usable read is the store's own
//!   index and whose discriminating index may lie off the chain entirely;
//! * an array in a *foreign* position, `(f arr)` with `f : Array -> BV`,
//!   where no array atom occurs at all and only the ext rule for shared array
//!   terms closes the goal;
//! * arrays of arrays at width 1, where the witness read of one pair is
//!   itself an array term that the next refinement round has to pick up.
//!
//! Index *and* element sort are one bit wide in the bounded run, which is
//! where cardinality bites: the four inhabitants of
//! `(Array (_ BitVec 1) (_ BitVec 1))` make five pairwise-distinct arrays
//! unsat, and a two-element index domain makes a one-write store chain
//! refutable against a constant.
//!
//! Scoring: the oracle enumerates every interpretation of every declared
//! symbol, so both verdicts are checked against the truth; and every `sat`
//! answer's `(get-model)` block is parsed back into an interpretation and the
//! assertions re-evaluated under it, so a model that falsifies its own script
//! is a failure even when the verdict is right.

use super::{Rng, Run, run_script};
use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Problem model
// ---------------------------------------------------------------------------

/// How many array-valued symbols a problem may declare.
const ARRAYS: usize = 3;
/// How many index-sorted symbols a problem may declare.
const INDICES: usize = 2;
/// How many element-sorted symbols a problem may declare.
const ELEMENTS: usize = 2;
/// How many arrays-of-arrays a problem may declare (nested mode only).
const NESTED: usize = 2;

/// An index-sorted term.
#[derive(Clone, Debug)]
enum Idx {
    Var(usize),
    Lit(u8),
}

/// An element-sorted term.
#[derive(Clone, Debug)]
enum Elem {
    Var(usize),
    Lit(u8),
    /// `(select A x)`.
    Read(Box<Arr>, Idx),
    /// `(f A)` — an array in a foreign position.
    Foreign(Box<Arr>),
    /// `(g e)`.
    Mapped(Box<Elem>),
    /// `(select (select N x) y)`.
    ReadNested(usize, Idx, Idx),
}

/// An array-sorted term.
#[derive(Clone, Debug)]
enum Arr {
    Var(usize),
    /// `((as const S) d)`.
    Const(u8),
    /// `(store A x e)`.
    Store(Box<Arr>, Idx, Box<Elem>),
    /// `(select N x)` — a row of an array of arrays.
    Row(usize, Idx),
    /// `(ite (= x y) A B)` — an *array-sorted* `ite` (`#P2b-41`).
    ///
    /// The shape this generator was missing, and its absence is why four
    /// passes of review never saw the hole it hides: a `select` through an
    /// array-sorted `ite` used to be a free bit-vector, because the encoder
    /// refused to name an `Array`-sorted `ite` with a fresh variable and the
    /// array theory only noted its branches as *foreign*.  Seven lines were a
    /// wrong `sat`.
    ///
    /// The condition is an index equality rather than a fresh Boolean: it
    /// keeps the oracle total (`eval_array` decides it from the same
    /// interpretation it already has) while still making the branch depend on
    /// the search rather than on a constant.
    Ite(Idx, Idx, Box<Arr>, Box<Arr>),
}

/// A Boolean combination of atoms.
#[derive(Clone, Debug)]
enum Formula {
    ElemEq(Elem, Elem),
    ElemNe(Elem, Elem),
    ArrEq(Arr, Arr),
    /// `(distinct A B …)`, two or more operands.
    ArrDistinct(Vec<Arr>),
    NestedDistinct(usize, usize),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
}

/// One generated problem.
struct Problem {
    index_width: u32,
    elem_width: u32,
    /// Whether the problem declares arrays of arrays.
    nested: bool,
    asserts: Vec<Formula>,
}

impl Problem {
    fn index_count(&self) -> usize {
        1usize << self.index_width
    }
    fn elem_count(&self) -> usize {
        1usize << self.elem_width
    }
    /// The number of distinct `(Array index elem)` values.
    fn array_count(&self) -> usize {
        self.elem_count().pow(self.index_count() as u32)
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

fn gen_index(rng: &mut Rng, problem: &Problem) -> Idx {
    if rng.chance(1, 2) {
        Idx::Var(rng.below(INDICES as u64) as usize)
    } else {
        Idx::Lit(rng.below(problem.index_count() as u64) as u8)
    }
}

fn gen_array(rng: &mut Rng, problem: &Problem, depth: u32) -> Arr {
    if depth == 0 || rng.chance(1, 2) {
        return match rng.below(3) {
            0 => Arr::Const(rng.below(problem.elem_count() as u64) as u8),
            1 if problem.nested => {
                Arr::Row(rng.below(NESTED as u64) as usize, gen_index(rng, problem))
            }
            _ => Arr::Var(rng.below(ARRAYS as u64) as usize),
        };
    }
    if rng.chance(1, 3) {
        // An array-sorted `ite`, drawn often enough that a few hundred scripts
        // contain a good number of them (`#P2b-41`).
        return Arr::Ite(
            gen_index(rng, problem),
            gen_index(rng, problem),
            Box::new(gen_array(rng, problem, depth - 1)),
            Box::new(gen_array(rng, problem, depth - 1)),
        );
    }
    Arr::Store(
        Box::new(gen_array(rng, problem, depth - 1)),
        gen_index(rng, problem),
        Box::new(gen_elem(rng, problem, depth - 1)),
    )
}

fn gen_elem(rng: &mut Rng, problem: &Problem, depth: u32) -> Elem {
    if depth == 0 || rng.chance(1, 3) {
        return match rng.below(2) {
            0 => Elem::Var(rng.below(ELEMENTS as u64) as usize),
            _ => Elem::Lit(rng.below(problem.elem_count() as u64) as u8),
        };
    }
    match rng.below(5) {
        0 => Elem::Read(
            Box::new(gen_array(rng, problem, 1)),
            gen_index(rng, problem),
        ),
        1 => Elem::Foreign(Box::new(gen_array(rng, problem, 1))),
        2 => Elem::Mapped(Box::new(gen_elem(rng, problem, depth - 1))),
        3 if problem.nested => Elem::ReadNested(
            rng.below(NESTED as u64) as usize,
            gen_index(rng, problem),
            gen_index(rng, problem),
        ),
        _ => Elem::Read(
            Box::new(gen_array(rng, problem, 2)),
            gen_index(rng, problem),
        ),
    }
}

/// The planted shapes: the four families `#P2b-37` closed, drawn far more
/// often than random noise would produce them.
fn gen_planted(rng: &mut Rng, problem: &Problem) -> Vec<Formula> {
    let index_count = problem.index_count() as u8;
    let elem_count = problem.elem_count() as u8;
    match rng.below(8) {
        // (1f) a read through an array-sorted `ite`, with both branches pinned
        // (`#P2b-41`).  The planted form of the seven-line wrong `sat`: the
        // read is one branch's read or the other's, and both are pinned, so
        // demanding a third value is unsatisfiable.
        6 => {
            let pinned = rng.below(u64::from(elem_count)) as u8;
            let demanded = (pinned + 1) % elem_count;
            vec![
                Formula::ElemEq(
                    Elem::Read(
                        Box::new(Arr::Ite(
                            gen_index(rng, problem),
                            gen_index(rng, problem),
                            Box::new(Arr::Var(0)),
                            Box::new(Arr::Var(1)),
                        )),
                        Idx::Lit(0),
                    ),
                    Elem::Lit(demanded),
                ),
                Formula::ElemEq(
                    Elem::Read(Box::new(Arr::Var(0)), Idx::Lit(0)),
                    Elem::Lit(pinned),
                ),
                Formula::ElemEq(
                    Elem::Read(Box::new(Arr::Var(1)), Idx::Lit(0)),
                    Elem::Lit(pinned),
                ),
            ]
        }
        // (1f) an array-sorted `ite` under `distinct`, the shape a random
        // corpus reaches: both operands can denote the same array.
        7 => {
            vec![
                Formula::ArrDistinct(vec![
                    Arr::Ite(
                        Idx::Lit(0),
                        Idx::Lit(0),
                        Box::new(Arr::Var(0)),
                        Box::new(Arr::Var(0)),
                    ),
                    Arr::Ite(
                        gen_index(rng, problem),
                        gen_index(rng, problem),
                        Box::new(Arr::Var(1)),
                        Box::new(Arr::Var(0)),
                    ),
                ]),
                Formula::Not(Box::new(Formula::ArrDistinct(vec![
                    Arr::Var(1),
                    Arr::Var(0),
                ]))),
            ]
        }
        // (1a) `distinct` over two arrays with every index pinned equal, or
        // one index left free.
        0 => {
            let pinned = if rng.chance(3, 4) {
                index_count
            } else {
                index_count - 1
            };
            let mut out = vec![Formula::ArrDistinct(vec![Arr::Var(0), Arr::Var(1)])];
            for index in 0..pinned {
                out.push(Formula::ElemEq(
                    Elem::Read(Box::new(Arr::Var(0)), Idx::Lit(index)),
                    Elem::Read(Box::new(Arr::Var(1)), Idx::Lit(index)),
                ));
            }
            out
        }
        // (1a) `n`-ary `distinct`, where the array sort's cardinality decides.
        1 => {
            let arity = 2 + rng.below(ARRAYS as u64 - 1) as usize;
            vec![Formula::ArrDistinct(
                (0..arity).map(Arr::Var).collect::<Vec<_>>(),
            )]
        }
        // (1b)/(1c) `store = ((as const S) d)`, both polarities of the default.
        2 => {
            let base = rng.below(u64::from(elem_count)) as u8;
            let stored = rng.below(u64::from(elem_count)) as u8;
            let target = rng.below(u64::from(elem_count)) as u8;
            vec![Formula::ArrEq(
                Arr::Store(
                    Box::new(Arr::Const(base)),
                    gen_index(rng, problem),
                    Box::new(Elem::Lit(stored)),
                ),
                Arr::Const(target),
            )]
        }
        // (1b) `store` over a declared array against a constant.
        3 => {
            let stored = rng.below(u64::from(elem_count)) as u8;
            let target = rng.below(u64::from(elem_count)) as u8;
            vec![Formula::ArrEq(
                Arr::Store(
                    Box::new(Arr::Var(0)),
                    gen_index(rng, problem),
                    Box::new(Elem::Lit(stored)),
                ),
                Arr::Const(target),
            )]
        }
        // (1d) foreign array arguments with every index pinned equal.
        4 => {
            let pinned = if rng.chance(3, 4) {
                index_count
            } else {
                index_count - 1
            };
            let mut out = vec![Formula::ElemNe(
                Elem::Foreign(Box::new(Arr::Var(0))),
                Elem::Foreign(Box::new(Arr::Var(1))),
            )];
            for index in 0..pinned {
                out.push(Formula::ElemEq(
                    Elem::Read(Box::new(Arr::Var(0)), Idx::Lit(index)),
                    Elem::Read(Box::new(Arr::Var(1)), Idx::Lit(index)),
                ));
            }
            out
        }
        // (1e) arrays of arrays: `distinct` with every entry pinned equal.
        _ => {
            if !problem.nested {
                return vec![Formula::ArrDistinct(vec![Arr::Var(0), Arr::Var(1)])];
            }
            let mut out = vec![Formula::NestedDistinct(0, 1)];
            let pinned = if rng.chance(3, 4) { index_count } else { 1 };
            for outer in 0..pinned {
                for inner in 0..index_count {
                    out.push(Formula::ElemEq(
                        Elem::ReadNested(0, Idx::Lit(outer), Idx::Lit(inner)),
                        Elem::ReadNested(1, Idx::Lit(outer), Idx::Lit(inner)),
                    ));
                }
            }
            out
        }
    }
}

fn gen_problem(rng: &mut Rng, index_width: u32, elem_width: u32, nested: bool) -> Problem {
    let mut problem = Problem {
        index_width,
        elem_width,
        nested,
        asserts: Vec::new(),
    };
    problem.asserts = gen_planted(rng, &problem);
    // A sprinkling of random structure on top, but never enough to hide the
    // planted shape: an extra atom is drawn only half the time.
    if rng.chance(1, 2) {
        let atom = match rng.below(3) {
            0 => Formula::ElemEq(gen_elem(rng, &problem, 2), gen_elem(rng, &problem, 2)),
            1 => Formula::ElemNe(gen_elem(rng, &problem, 2), gen_elem(rng, &problem, 2)),
            _ => Formula::ArrEq(gen_array(rng, &problem, 2), gen_array(rng, &problem, 2)),
        };
        let atom = if rng.chance(1, 4) {
            Formula::Or(
                Box::new(atom),
                Box::new(Formula::ElemEq(Elem::Var(0), Elem::Var(1))),
            )
        } else if rng.chance(1, 6) {
            Formula::And(
                Box::new(atom),
                Box::new(Formula::ElemEq(Elem::Var(0), Elem::Var(0))),
            )
        } else if rng.chance(1, 5) {
            Formula::Not(Box::new(atom))
        } else {
            atom
        };
        problem.asserts.push(atom);
    }
    problem
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

fn bits(value: u8, width: u32) -> String {
    format!("#b{value:0w$b}", w = width as usize)
}

fn index_sort(problem: &Problem) -> String {
    format!("(_ BitVec {})", problem.index_width)
}

fn elem_sort(problem: &Problem) -> String {
    format!("(_ BitVec {})", problem.elem_width)
}

fn array_sort(problem: &Problem) -> String {
    format!("(Array {} {})", index_sort(problem), elem_sort(problem))
}

fn nested_sort(problem: &Problem) -> String {
    format!("(Array {} {})", index_sort(problem), array_sort(problem))
}

fn print_index(index: &Idx, problem: &Problem, out: &mut String) {
    match index {
        Idx::Var(k) => {
            let _ = write!(out, "i{k}");
        }
        Idx::Lit(value) => out.push_str(&bits(*value, problem.index_width)),
    }
}

fn print_array(array: &Arr, problem: &Problem, out: &mut String) {
    match array {
        Arr::Var(k) => {
            let _ = write!(out, "a{k}");
        }
        Arr::Const(default) => {
            let _ = write!(
                out,
                "((as const {}) {})",
                array_sort(problem),
                bits(*default, problem.elem_width)
            );
        }
        Arr::Store(base, index, value) => {
            out.push_str("(store ");
            print_array(base, problem, out);
            out.push(' ');
            print_index(index, problem, out);
            out.push(' ');
            print_elem(value, problem, out);
            out.push(')');
        }
        Arr::Row(k, index) => {
            let _ = write!(out, "(select n{k} ");
            print_index(index, problem, out);
            out.push(')');
        }
        Arr::Ite(left, right, then_branch, else_branch) => {
            out.push_str("(ite (= ");
            print_index(left, problem, out);
            out.push(' ');
            print_index(right, problem, out);
            out.push_str(") ");
            print_array(then_branch, problem, out);
            out.push(' ');
            print_array(else_branch, problem, out);
            out.push(')');
        }
    }
}

fn print_elem(elem: &Elem, problem: &Problem, out: &mut String) {
    match elem {
        Elem::Var(k) => {
            let _ = write!(out, "e{k}");
        }
        Elem::Lit(value) => out.push_str(&bits(*value, problem.elem_width)),
        Elem::Read(array, index) => {
            out.push_str("(select ");
            print_array(array, problem, out);
            out.push(' ');
            print_index(index, problem, out);
            out.push(')');
        }
        Elem::Foreign(array) => {
            out.push_str("(f ");
            print_array(array, problem, out);
            out.push(')');
        }
        Elem::Mapped(inner) => {
            out.push_str("(g ");
            print_elem(inner, problem, out);
            out.push(')');
        }
        Elem::ReadNested(k, outer, inner) => {
            let _ = write!(out, "(select (select n{k} ");
            print_index(outer, problem, out);
            out.push_str(") ");
            print_index(inner, problem, out);
            out.push(')');
        }
    }
}

fn print_formula(formula: &Formula, problem: &Problem, out: &mut String) {
    match formula {
        Formula::ElemEq(lhs, rhs) | Formula::ElemNe(lhs, rhs) => {
            let head = if matches!(formula, Formula::ElemEq(..)) {
                "="
            } else {
                "distinct"
            };
            let _ = write!(out, "({head} ");
            print_elem(lhs, problem, out);
            out.push(' ');
            print_elem(rhs, problem, out);
            out.push(')');
        }
        Formula::ArrEq(lhs, rhs) => {
            out.push_str("(= ");
            print_array(lhs, problem, out);
            out.push(' ');
            print_array(rhs, problem, out);
            out.push(')');
        }
        Formula::ArrDistinct(operands) => {
            out.push_str("(distinct");
            for operand in operands {
                out.push(' ');
                print_array(operand, problem, out);
            }
            out.push(')');
        }
        Formula::NestedDistinct(lhs, rhs) => {
            let _ = write!(out, "(distinct n{lhs} n{rhs})");
        }
        Formula::Not(inner) => {
            out.push_str("(not ");
            print_formula(inner, problem, out);
            out.push(')');
        }
        Formula::And(lhs, rhs) | Formula::Or(lhs, rhs) => {
            let head = if matches!(formula, Formula::And(..)) {
                "and"
            } else {
                "or"
            };
            let _ = write!(out, "({head} ");
            print_formula(lhs, problem, out);
            out.push(' ');
            print_formula(rhs, problem, out);
            out.push(')');
        }
    }
}

/// The SMT-LIB script for `problem`, declaring only the symbols it mentions.
fn render(problem: &Problem, with_model: bool, harness_clock_ms: Option<u64>) -> String {
    let mut body = String::new();
    for assertion in &problem.asserts {
        body.push_str("(assert ");
        print_formula(assertion, problem, &mut body);
        body.push_str(")\n");
    }
    // A *deterministic* budget per check, as the parent campaign does: an
    // `unknown` from an exhausted budget is scored (`unknown_decided`), never a
    // failure, so one expensive circuit cannot stall the suite while every
    // decided verdict is still checked against the oracle.
    //
    // `harness_clock_ms` is a **harness** budget and is `None` for every gate
    // test (`#P2b-46`, decision (16)).  A `(set-option :timeout N)` here did
    // not only put a verdict behind the machine's speed — it put the *tally*
    // there, because `score` replays a published model on the `sat` branch
    // only, so a script that ran out of clock was never model-checked and the
    // gate's `bad_model` bound moved with how many scripts got past it: a gate
    // that can go red on a *faster* machine, which is why no pass ever saw it.
    // With `:max-conflicts` as the only budget, which scripts answer `sat` is a
    // property of the formula and not of the host.
    //
    // The `#[ignore]`d 6,000-script campaign passes `Some(1000)` all the same,
    // and that is sound rather than a loophole: every bound it asserts is
    // *zero* (`wrong_sat`, `wrong_unsat`, `errors`, `panics`, `bad_model`), and
    // a zero bound is monotone in coverage — a machine fast enough to score
    // more scripts can only find more defects, never manufacture one.  Without
    // it the campaign does not finish: since this round's array work a single
    // generated script can spend the whole `BV_EMBEDDED_CHECK_CEILING` of
    // 250,000 embedded checks, which is minutes, and the 6,000-script run
    // passed nextest's 900 s ceiling without completing.
    let mut out = String::from(
        "(set-logic QF_AUFBV)\n         (set-option :produce-models true)\n         (set-option :max-conflicts 20000)\n",
    );
    if let Some(ms) = harness_clock_ms {
        let _ = writeln!(out, "(set-option :timeout {ms})");
    }
    for k in 0..ARRAYS {
        if body.contains(&format!("a{k}")) {
            let _ = writeln!(out, "(declare-const a{k} {})", array_sort(problem));
        }
    }
    if problem.nested {
        for k in 0..NESTED {
            if body.contains(&format!("n{k}")) {
                let _ = writeln!(out, "(declare-const n{k} {})", nested_sort(problem));
            }
        }
    }
    for k in 0..INDICES {
        if body.contains(&format!("i{k}")) {
            let _ = writeln!(out, "(declare-const i{k} {})", index_sort(problem));
        }
    }
    for k in 0..ELEMENTS {
        if body.contains(&format!("e{k}")) {
            let _ = writeln!(out, "(declare-const e{k} {})", elem_sort(problem));
        }
    }
    if body.contains("(f ") {
        let _ = writeln!(
            out,
            "(declare-fun f ({}) {})",
            array_sort(problem),
            elem_sort(problem)
        );
    }
    if body.contains("(g ") {
        let _ = writeln!(
            out,
            "(declare-fun g ({}) {})",
            elem_sort(problem),
            elem_sort(problem)
        );
    }
    out.push_str(&body);
    out.push_str("(check-sat)\n");
    if with_model {
        out.push_str("(get-model)\n");
    }
    out
}

/// Which symbols a rendered script actually declares.
///
/// The *slot numbers* matter, not the counts: a script may declare `a1` and
/// not `a0`, and an oracle that enumerated the used symbols compactly would
/// feed `a1`'s value to `a0` and leave `a1` at its default — reporting `unsat`
/// for a satisfiable problem.
struct Used {
    arrays: Vec<usize>,
    nested: Vec<usize>,
    indices: Vec<usize>,
    elements: Vec<usize>,
    of_array: bool,
    of_elem: bool,
}

fn used_symbols(problem: &Problem) -> Used {
    let script = render(problem, false, None);
    Used {
        arrays: (0..ARRAYS)
            .filter(|k| script.contains(&format!("(declare-const a{k} ")))
            .collect(),
        nested: (0..NESTED)
            .filter(|k| script.contains(&format!("(declare-const n{k} ")))
            .collect(),
        indices: (0..INDICES)
            .filter(|k| script.contains(&format!("(declare-const i{k} ")))
            .collect(),
        elements: (0..ELEMENTS)
            .filter(|k| script.contains(&format!("(declare-const e{k} ")))
            .collect(),
        of_array: script.contains("(declare-fun f "),
        of_elem: script.contains("(declare-fun g "),
    }
}

// ---------------------------------------------------------------------------
// Reference semantics
// ---------------------------------------------------------------------------

/// One interpretation of a problem's declared symbols.
///
/// An array is a table of `2^index_width` elements; an array of arrays is a
/// table of array *codes* (the table read as a base-`2^elem_width` numeral),
/// which is also how `f`'s argument is indexed.
struct Interp {
    index_count: usize,
    elem_count: usize,
    arrays: Vec<Vec<u8>>,
    nested: Vec<Vec<Vec<u8>>>,
    indices: Vec<u8>,
    elements: Vec<u8>,
    /// `of_array[code]`: the value of `f` at the array with that code.
    of_array: Vec<u8>,
    /// `of_elem[value]`: the value of `g` at that element.
    of_elem: Vec<u8>,
}

impl Interp {
    /// The integer code of an array table, in base `elem_count`.
    fn code(&self, table: &[u8]) -> usize {
        let mut code = 0usize;
        for &entry in table.iter().rev() {
            code = code * self.elem_count + entry as usize;
        }
        code
    }

    fn table_of(&self, code: usize) -> Vec<u8> {
        let mut code = code;
        let mut table = Vec::with_capacity(self.index_count);
        for _ in 0..self.index_count {
            table.push((code % self.elem_count) as u8);
            code /= self.elem_count;
        }
        table
    }
}

fn eval_index(index: &Idx, interp: &Interp) -> u8 {
    match index {
        Idx::Var(k) => interp.indices.get(*k).copied().unwrap_or(0),
        Idx::Lit(value) => *value,
    }
}

fn eval_array(array: &Arr, interp: &Interp) -> Vec<u8> {
    match array {
        Arr::Var(k) => interp
            .arrays
            .get(*k)
            .cloned()
            .unwrap_or_else(|| vec![0; interp.index_count]),
        Arr::Const(default) => vec![*default; interp.index_count],
        Arr::Store(base, index, value) => {
            let mut table = eval_array(base, interp);
            let at = eval_index(index, interp) as usize;
            let element = eval_elem(value, interp);
            if let Some(slot) = table.get_mut(at) {
                *slot = element;
            }
            table
        }
        Arr::Row(k, index) => {
            let outer = eval_index(index, interp) as usize;
            interp
                .nested
                .get(*k)
                .and_then(|rows| rows.get(outer))
                .cloned()
                .unwrap_or_else(|| vec![0; interp.index_count])
        }
        Arr::Ite(left, right, then_branch, else_branch) => {
            if eval_index(left, interp) == eval_index(right, interp) {
                eval_array(then_branch, interp)
            } else {
                eval_array(else_branch, interp)
            }
        }
    }
}

fn eval_elem(elem: &Elem, interp: &Interp) -> u8 {
    match elem {
        Elem::Var(k) => interp.elements.get(*k).copied().unwrap_or(0),
        Elem::Lit(value) => *value,
        Elem::Read(array, index) => {
            let table = eval_array(array, interp);
            let at = eval_index(index, interp) as usize;
            table.get(at).copied().unwrap_or(0)
        }
        Elem::Foreign(array) => {
            let table = eval_array(array, interp);
            let code = interp.code(&table);
            interp.of_array.get(code).copied().unwrap_or(0)
        }
        Elem::Mapped(inner) => {
            let value = eval_elem(inner, interp) as usize;
            interp.of_elem.get(value).copied().unwrap_or(0)
        }
        Elem::ReadNested(k, outer, inner) => {
            let outer = eval_index(outer, interp) as usize;
            let inner = eval_index(inner, interp) as usize;
            interp
                .nested
                .get(*k)
                .and_then(|rows| rows.get(outer))
                .and_then(|row| row.get(inner))
                .copied()
                .unwrap_or(0)
        }
    }
}

fn eval_formula(formula: &Formula, interp: &Interp) -> bool {
    match formula {
        Formula::ElemEq(lhs, rhs) => eval_elem(lhs, interp) == eval_elem(rhs, interp),
        Formula::ElemNe(lhs, rhs) => eval_elem(lhs, interp) != eval_elem(rhs, interp),
        Formula::ArrEq(lhs, rhs) => eval_array(lhs, interp) == eval_array(rhs, interp),
        Formula::ArrDistinct(operands) => {
            let tables: Vec<Vec<u8>> = operands.iter().map(|a| eval_array(a, interp)).collect();
            (0..tables.len()).all(|first| {
                ((first + 1)..tables.len()).all(|second| tables[first] != tables[second])
            })
        }
        Formula::NestedDistinct(lhs, rhs) => interp.nested.get(*lhs) != interp.nested.get(*rhs),
        Formula::Not(inner) => !eval_formula(inner, interp),
        Formula::And(lhs, rhs) => eval_formula(lhs, interp) && eval_formula(rhs, interp),
        Formula::Or(lhs, rhs) => eval_formula(lhs, interp) || eval_formula(rhs, interp),
    }
}

fn all_hold(problem: &Problem, interp: &Interp) -> bool {
    problem.asserts.iter().all(|a| eval_formula(a, interp))
}

// ---------------------------------------------------------------------------
// Exhaustive oracle
// ---------------------------------------------------------------------------

/// The largest interpretation space the oracle enumerates.
const ORACLE_BUDGET: u64 = 1 << 18;

/// Exhaustive truth, or `None` when the space is too large.
fn exhaustive(problem: &Problem) -> Option<bool> {
    let used = used_symbols(problem);
    let index_count = problem.index_count();
    let elem_count = problem.elem_count();
    let array_count = problem.array_count() as u64;
    let nested_count = array_count.checked_pow(index_count as u32)?;

    // Mixed-radix digits: one per declared symbol.
    let mut radices: Vec<u64> = Vec::new();
    for _ in &used.arrays {
        radices.push(array_count);
    }
    for _ in &used.nested {
        radices.push(nested_count);
    }
    for _ in &used.indices {
        radices.push(index_count as u64);
    }
    for _ in &used.elements {
        radices.push(elem_count as u64);
    }
    if used.of_array {
        radices.push((elem_count as u64).checked_pow(u32::try_from(array_count).ok()?)?);
    }
    if used.of_elem {
        radices.push((elem_count as u64).checked_pow(elem_count as u32)?);
    }
    let mut space: u64 = 1;
    for &radix in &radices {
        space = space.checked_mul(radix)?;
        if space > ORACLE_BUDGET {
            return None;
        }
    }

    let mut digits = vec![0u64; radices.len()];
    loop {
        let mut cursor = 0usize;
        let mut interp = Interp {
            index_count,
            elem_count,
            arrays: vec![vec![0; index_count]; ARRAYS],
            nested: vec![vec![vec![0; index_count]; index_count]; NESTED],
            indices: vec![0; INDICES],
            elements: vec![0; ELEMENTS],
            of_array: Vec::new(),
            of_elem: Vec::new(),
        };
        for &slot in &used.arrays {
            let code = digits[cursor] as usize;
            cursor += 1;
            interp.arrays[slot] = interp.table_of(code);
        }
        for &slot in &used.nested {
            let mut code = digits[cursor] as usize;
            cursor += 1;
            let mut rows = Vec::with_capacity(index_count);
            for _ in 0..index_count {
                let row_code = code % problem.array_count();
                code /= problem.array_count();
                rows.push(interp.table_of(row_code));
            }
            interp.nested[slot] = rows;
        }
        for &slot in &used.indices {
            interp.indices[slot] = digits[cursor] as u8;
            cursor += 1;
        }
        for &slot in &used.elements {
            interp.elements[slot] = digits[cursor] as u8;
            cursor += 1;
        }
        if used.of_array {
            let mut code = digits[cursor];
            cursor += 1;
            for _ in 0..array_count {
                interp.of_array.push((code % elem_count as u64) as u8);
                code /= elem_count as u64;
            }
        }
        if used.of_elem {
            let mut code = digits[cursor];
            for _ in 0..elem_count {
                interp.of_elem.push((code % elem_count as u64) as u8);
                code /= elem_count as u64;
            }
        }
        if all_hold(problem, &interp) {
            return Some(true);
        }

        let mut carry = true;
        for (digit, &radix) in digits.iter_mut().zip(radices.iter()) {
            if !carry {
                break;
            }
            *digit += 1;
            if *digit == radix {
                *digit = 0;
            } else {
                carry = false;
            }
        }
        if carry {
            return Some(false);
        }
    }
}

// ---------------------------------------------------------------------------
// Model replay
// ---------------------------------------------------------------------------

/// A parsed S-expression from a `(get-model)` block.
#[derive(Debug, Clone)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

fn parse_sexps(text: &str) -> Vec<Sexp> {
    let mut stack: Vec<Vec<Sexp>> = vec![Vec::new()];
    let mut token = String::new();
    let flush = |token: &mut String, stack: &mut Vec<Vec<Sexp>>| {
        if !token.is_empty()
            && let Some(top) = stack.last_mut()
        {
            top.push(Sexp::Atom(std::mem::take(token)));
        }
    };
    for character in text.chars() {
        match character {
            '(' => {
                flush(&mut token, &mut stack);
                stack.push(Vec::new());
            }
            ')' => {
                flush(&mut token, &mut stack);
                if let Some(done) = stack.pop()
                    && let Some(top) = stack.last_mut()
                {
                    top.push(Sexp::List(done));
                }
            }
            c if c.is_whitespace() => flush(&mut token, &mut stack),
            c => token.push(c),
        }
    }
    flush(&mut token, &mut stack);
    stack.pop().unwrap_or_default()
}

fn bit_literal(text: &str) -> Option<u8> {
    let digits = text.strip_prefix("#b")?;
    u8::from_str_radix(digits, 2).ok()
}

/// Read an array value — `((as const S) d)` or a `store` chain over one — as a
/// table.
fn array_value(sexp: &Sexp, interp_index_count: usize) -> Option<Vec<u8>> {
    match sexp {
        Sexp::List(items) => match items.as_slice() {
            // `((as const (Array …)) #b0)`
            [Sexp::List(head), value] if matches!(head.first(), Some(Sexp::Atom(a)) if a == "as") =>
            {
                let Sexp::Atom(text) = value else { return None };
                Some(vec![bit_literal(text)?; interp_index_count])
            }
            [Sexp::Atom(head), base, index, value] if head == "store" => {
                let mut table = array_value(base, interp_index_count)?;
                let (Sexp::Atom(index), Sexp::Atom(value)) = (index, value) else {
                    return None;
                };
                let at = bit_literal(index)? as usize;
                let element = bit_literal(value)?;
                *table.get_mut(at)? = element;
                Some(table)
            }
            _ => None,
        },
        Sexp::Atom(_) => None,
    }
}

/// Read an array-of-arrays value as a table of tables.
fn nested_value(sexp: &Sexp, index_count: usize) -> Option<Vec<Vec<u8>>> {
    match sexp {
        Sexp::List(items) => match items.as_slice() {
            [Sexp::List(head), value] if matches!(head.first(), Some(Sexp::Atom(a)) if a == "as") =>
            {
                let row = array_value(value, index_count)?;
                Some(vec![row; index_count])
            }
            [Sexp::Atom(head), base, index, value] if head == "store" => {
                let mut rows = nested_value(base, index_count)?;
                let Sexp::Atom(index) = index else {
                    return None;
                };
                let at = bit_literal(index)? as usize;
                let row = array_value(value, index_count)?;
                *rows.get_mut(at)? = row;
                Some(rows)
            }
            _ => None,
        },
        Sexp::Atom(_) => None,
    }
}

/// Evaluate a printed `define-fun` body — a nested `ite` over `(= x!0 …)`
/// guards — at one argument.
fn apply_body(body: &Sexp, argument: &ArgValue, index_count: usize) -> Option<u8> {
    match body {
        Sexp::Atom(text) => bit_literal(text),
        Sexp::List(items) => match items.as_slice() {
            [Sexp::Atom(head), guard, hit, miss] if head == "ite" => {
                if guard_matches(guard, argument, index_count)? {
                    apply_body(hit, argument, index_count)
                } else {
                    apply_body(miss, argument, index_count)
                }
            }
            _ => None,
        },
    }
}

/// The argument a printed interpretation is applied to.
enum ArgValue {
    Bits(u8),
    Table(Vec<u8>),
}

fn guard_matches(guard: &Sexp, argument: &ArgValue, index_count: usize) -> Option<bool> {
    let Sexp::List(items) = guard else {
        return None;
    };
    let [Sexp::Atom(head), Sexp::Atom(_variable), expected] = items.as_slice() else {
        return None;
    };
    if head != "=" {
        return None;
    }
    match argument {
        ArgValue::Bits(value) => {
            let Sexp::Atom(text) = expected else {
                return None;
            };
            Some(bit_literal(text)? == *value)
        }
        ArgValue::Table(table) => Some(array_value(expected, index_count)?.as_slice() == table),
    }
}

/// Parse a `(get-model)` block into an interpretation and check the problem's
/// assertions under it.
///
/// `None` when the block mentions a shape this reader does not cover; the
/// caller scores that separately from a model that is genuinely wrong.
fn model_satisfies(problem: &Problem, block: &str) -> Option<bool> {
    let forms = parse_sexps(block);
    let Some(Sexp::List(top)) = forms.first() else {
        return None;
    };
    let index_count = problem.index_count();
    let mut interp = Interp {
        index_count,
        elem_count: problem.elem_count(),
        arrays: vec![vec![0; index_count]; ARRAYS],
        nested: vec![vec![vec![0; index_count]; index_count]; NESTED],
        indices: vec![0; INDICES],
        elements: vec![0; ELEMENTS],
        of_array: Vec::new(),
        of_elem: Vec::new(),
    };
    let mut of_array_body: Option<Sexp> = None;
    let mut of_elem_body: Option<Sexp> = None;

    for form in top.iter().skip(1) {
        let Sexp::List(items) = form else { continue };
        let [
            Sexp::Atom(head),
            Sexp::Atom(name),
            Sexp::List(params),
            _sort,
            body,
        ] = items.as_slice()
        else {
            continue;
        };
        if head != "define-fun" {
            continue;
        }
        if !params.is_empty() {
            match name.as_str() {
                "f" => of_array_body = Some(body.clone()),
                "g" => of_elem_body = Some(body.clone()),
                _ => {}
            }
            continue;
        }
        let slot = name
            .strip_prefix('a')
            .and_then(|k| k.parse::<usize>().ok())
            .filter(|_| name.starts_with('a'));
        if let Some(k) = slot
            && k < ARRAYS
        {
            interp.arrays[k] = array_value(body, index_count)?;
            continue;
        }
        if let Some(k) = name.strip_prefix('n').and_then(|k| k.parse::<usize>().ok())
            && k < NESTED
        {
            interp.nested[k] = nested_value(body, index_count)?;
            continue;
        }
        let Sexp::Atom(text) = body else {
            return None;
        };
        let value = bit_literal(text)?;
        if let Some(k) = name.strip_prefix('i').and_then(|k| k.parse::<usize>().ok())
            && k < INDICES
        {
            interp.indices[k] = value;
        } else if let Some(k) = name.strip_prefix('e').and_then(|k| k.parse::<usize>().ok())
            && k < ELEMENTS
        {
            interp.elements[k] = value;
        }
    }

    // Materialise the two function tables from their printed bodies.
    let array_count = problem.array_count();
    interp.of_array = match &of_array_body {
        Some(body) => {
            let mut table = Vec::with_capacity(array_count);
            for code in 0..array_count {
                let argument = ArgValue::Table(interp.table_of(code));
                table.push(apply_body(body, &argument, index_count)?);
            }
            table
        }
        None => vec![0; array_count],
    };
    interp.of_elem = match &of_elem_body {
        Some(body) => {
            let mut table = Vec::with_capacity(problem.elem_count());
            for value in 0..problem.elem_count() {
                table.push(apply_body(body, &ArgValue::Bits(value as u8), index_count)?);
            }
            table
        }
        None => vec![0; problem.elem_count()],
    };

    Some(all_hold(problem, &interp))
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Campaign counters.
#[derive(Default, Debug)]
pub(super) struct ExtTally {
    pub(super) scripts: usize,
    pub(super) sat: usize,
    pub(super) unsat: usize,
    pub(super) unknown: usize,
    /// `unknown` on a formula the oracle decided: precision lost, not a
    /// failure.
    pub(super) unknown_decided: usize,
    pub(super) wrong_sat: usize,
    pub(super) wrong_unsat: usize,
    pub(super) bad_model: usize,
    /// A model the reader could not parse back; scored separately so an
    /// unrecognised *shape* is never mistaken for a wrong model.
    pub(super) unread_model: usize,
    pub(super) errors: usize,
    pub(super) panics: usize,
}

impl ExtTally {
    /// Wrong verdicts and crashes — the failures that make the run fail.
    ///
    /// `bad_model` is scored separately and pinned by the caller: a published
    /// model that falsifies its own script is a real defect, but it is a
    /// *pre-existing* one this round narrowed rather than closed (the 0.3.4
    /// base falsifies far more of these), so the campaign pins the residue
    /// instead of demanding zero and hiding a regression behind a `0 == 0`.
    pub(super) fn failures(&self) -> usize {
        self.wrong_sat + self.wrong_unsat + self.errors + self.panics
    }
}

fn score(
    problem: &Problem,
    tally: &mut ExtTally,
    first_failure: &mut Vec<String>,
    harness_clock_ms: Option<u64>,
) {
    let truth = exhaustive(problem);
    let script = render(problem, true, harness_clock_ms);
    tally.scripts += 1;
    let lines = match run_script(&script) {
        Run::Panic(message) => {
            tally.panics += 1;
            first_failure.push(format!("panic: {message}\n{script}"));
            return;
        }
        Run::Error(message) => {
            tally.errors += 1;
            first_failure.push(format!("error: {message}\n{script}"));
            return;
        }
        Run::Lines(lines) => lines,
    };
    match lines.first().map(String::as_str).unwrap_or("none") {
        "sat" => {
            tally.sat += 1;
            if truth == Some(false) {
                tally.wrong_sat += 1;
                first_failure.push(format!("wrong sat (oracle: unsat)\n{script}"));
                return;
            }
            let block = lines.get(1).map(String::as_str).unwrap_or("");
            match model_satisfies(problem, block) {
                Some(true) => {}
                Some(false) => {
                    tally.bad_model += 1;
                    first_failure.push(format!(
                        "model falsifies its own assertions\n{script}\n{block}"
                    ));
                }
                None => tally.unread_model += 1,
            }
        }
        "unsat" => {
            tally.unsat += 1;
            if truth == Some(true) {
                tally.wrong_unsat += 1;
                first_failure.push(format!("wrong unsat (a witness exists)\n{script}"));
            }
        }
        "unknown" => {
            tally.unknown += 1;
            if truth.is_some() {
                tally.unknown_decided += 1;
            }
        }
        other => {
            tally.errors += 1;
            first_failure.push(format!("no verdict: {other}\n{script}"));
        }
    }
}

/// The first failure of each kind, so one noisy kind cannot hide another.
pub(super) fn summarise(failures: &[String]) -> String {
    let mut kinds: Vec<&str> = Vec::new();
    let mut out = String::new();
    for failure in failures {
        let kind = failure.split(['(', '\n']).next().unwrap_or("");
        if kinds.contains(&kind) {
            continue;
        }
        kinds.push(kind);
        out.push_str(failure);
        out.push('\n');
    }
    out
}

/// Run `trials` problems per seed over the given sort widths.
pub(super) fn campaign(
    seeds: std::ops::Range<u64>,
    trials: usize,
    widths: &[(u32, u32, bool)],
    tally: &mut ExtTally,
    first_failure: &mut Vec<String>,
    harness_clock_ms: Option<u64>,
) {
    for seed in seeds {
        let mut rng = Rng::new(seed ^ 0x00E4_7000_0000_0000);
        for _ in 0..trials {
            let (index_width, elem_width, nested) = widths[rng.below(widths.len() as u64) as usize];
            let problem = gen_problem(&mut rng, index_width, elem_width, nested);
            score(&problem, tally, first_failure, harness_clock_ms);
        }
    }
}
