//! Differential campaign for the array / uninterpreted-function combination
//! (`#P2b-33`), scored against an **exhaustive** oracle.
//!
//! `bv_euf_combination` does this for the bit-vector / EUF exchange and its
//! `array` mode puts a `select` where an application would be; what neither it
//! nor the `#P2b-32` walk covered is the shape that broke: a read **under an
//! uninterpreted application**, `(distinct (f (select (store arr i v) i)) (f v))`,
//! where the read-over-write equality exists only as a lazily instantiated
//! lemma and has to reach `f`'s arguments through congruence closure.  That
//! answered `sat` — in QF_AUF, QF_AUFBV and QF_AUFLIA alike — because the guard
//! on the whole refinement loop (`Solver::has_array_ops`) was raised by a walk
//! that does not descend into an application's arguments, so no lemma was ever
//! built.  See `array_axioms::Solver::mark_array_ops`.
//!
//! So the generator here always has both an array and at least one function,
//! puts reads under applications and applications under reads, and nests store
//! chains two deep with indices drawn from a small pool so hits and misses both
//! occur.  At widths 1 and 2 the oracle enumerates every variable assignment
//! together with every *array table* and every *function table* — at width 1 a
//! 2-element array over 1-bit values is one of 4 tables, at width 2 one of 256
//! — so every verdict, `sat` and `unsat` alike, is checked against the truth.
//! At widths 3 and 4 the oracle samples witnesses, which proves `sat` only: an
//! `unsat` against a witness is a failure, and every `sat` is re-checked by
//! pinning the printed variable values.
//!
//! The bounded tests run on every `cargo test`; `array_uf_campaign` is the long
//! form (`OXIZ_AUF_SEED_LO/HI`, `OXIZ_AUF_TRIALS`).

use oxiz_solver::Context;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

// ---------------------------------------------------------------------------
// PRNG (splitmix64) — the same generator `bv_euf_combination` uses, so a seed
// means the same thing in both campaigns.
// ---------------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng {
            state: z ^ (z >> 31),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }
}

// ---------------------------------------------------------------------------
// Terms.
// ---------------------------------------------------------------------------

/// A bit-vector term over `num_vars` variables, `num_funcs` unary functions
/// and one array.
#[derive(Clone, Debug)]
enum Term {
    Var(usize),
    Const(u128),
    /// `(f<k> t)`.
    App(usize, Box<Term>),
    /// `(select <array> t)`.
    Read(Box<ArrTerm>, Box<Term>),
    Add(Box<Term>, Box<Term>),
    Xor(Box<Term>, Box<Term>),
    Not(Box<Term>),
    Ite(Box<Atom>, Box<Term>, Box<Term>),
}

/// An array term: the declared array, the SMT-LIB array constant, or a write
/// over one of those.
///
/// `Const` carries a literal default rather than a symbolic one, so it costs
/// the exhaustive oracle no enumeration bits at all — the budget arithmetic
/// below is unchanged by its presence (`#P2b-36`).
#[derive(Clone, Debug)]
enum ArrTerm {
    Base,
    Const(u128),
    Store(Box<ArrTerm>, Box<Term>, Box<Term>),
}

#[derive(Clone, Debug)]
enum Atom {
    Eq(Term, Term),
    Distinct(Term, Term),
    Ult(Term, Term),
    Ule(Term, Term),
    Or(Box<Atom>, Box<Atom>),
    Not(Box<Atom>),
}

struct Problem {
    width: u32,
    num_vars: usize,
    num_funcs: usize,
    asserts: Vec<Atom>,
}

fn mask(width: u32) -> u128 {
    (1u128 << width) - 1
}

fn gen_array(rng: &mut Rng, p: &Problem, depth: u32) -> ArrTerm {
    if depth == 0 || rng.chance(1, 3) {
        if rng.chance(1, 3) {
            return ArrTerm::Const(u128::from(rng.next_u64()) & mask(p.width));
        }
        return ArrTerm::Base;
    }
    ArrTerm::Store(
        Box::new(gen_array(rng, p, depth - 1)),
        Box::new(gen_term(rng, p, 0)),
        Box::new(gen_term(rng, p, 0)),
    )
}

fn gen_term(rng: &mut Rng, p: &Problem, depth: u32) -> Term {
    let w = p.width;
    if depth == 0 || rng.chance(1, 3) {
        return match rng.below(4) {
            0 => Term::Var(rng.below(p.num_vars as u64) as usize),
            1 => Term::Const(u128::from(rng.next_u64()) & mask(w)),
            2 => Term::Read(
                Box::new(gen_array(rng, p, 2)),
                Box::new(Term::Var(rng.below(p.num_vars as u64) as usize)),
            ),
            _ => Term::App(
                rng.below(p.num_funcs as u64) as usize,
                Box::new(Term::Var(rng.below(p.num_vars as u64) as usize)),
            ),
        };
    }
    let sub = |rng: &mut Rng| Box::new(gen_term(rng, p, depth - 1));
    match rng.below(6) {
        // A function OVER a read and a read under a function are the two
        // shapes this campaign exists for, so both are drawn twice as often
        // as the plain operators.
        0 | 1 => Term::App(rng.below(p.num_funcs as u64) as usize, sub(rng)),
        2 => Term::Read(Box::new(gen_array(rng, p, 2)), sub(rng)),
        3 => Term::Add(sub(rng), sub(rng)),
        4 => match rng.below(2) {
            0 => Term::Xor(sub(rng), sub(rng)),
            _ => Term::Not(sub(rng)),
        },
        _ => Term::Ite(Box::new(gen_atom(rng, p, depth - 1)), sub(rng), sub(rng)),
    }
}

fn gen_atom(rng: &mut Rng, p: &Problem, depth: u32) -> Atom {
    let a = gen_term(rng, p, depth);
    let b = gen_term(rng, p, depth);
    let base = match rng.below(4) {
        0 => Atom::Eq(a, b),
        1 => Atom::Distinct(a, b),
        2 => Atom::Ult(a, b),
        _ => Atom::Ule(a, b),
    };
    if depth > 0 && rng.chance(1, 5) {
        let other = gen_atom(rng, p, depth - 1);
        return Atom::Or(Box::new(base), Box::new(other));
    }
    if rng.chance(1, 6) {
        return Atom::Not(Box::new(base));
    }
    base
}

/// A planted shape whose reads occur **only** as arguments of an application
/// — `plant_shape`'s first three cases, the ones that need the guard fix.
fn argument_position_shape(rng: &mut Rng, p: &Problem) -> Atom {
    loop {
        let shape = plant_shape(rng, p);
        if !matches!(shape, Atom::Eq(Term::Read(..), _)) {
            return shape;
        }
    }
}

/// The `#P2b-33` and `#P2b-36` shapes, planted in half the problems so the
/// campaign spends its budget where the defects lived rather than on random
/// noise.
///
/// Half the store chains are laid over an array *constant* instead of the
/// declared array, which is what makes a planted miss reduce all the way to a
/// read with a known value; case `4` plants the bare array-constant read, the
/// shape that needs the `#P2b-36` axiom and nothing else.
fn plant_shape(rng: &mut Rng, p: &Problem) -> Atom {
    let i = Term::Var(0);
    let j = Term::Var((1 % p.num_vars.max(1)).min(p.num_vars - 1));
    let v = Term::Const(u128::from(rng.next_u64()) & mask(p.width));
    let default = u128::from(rng.next_u64()) & mask(p.width);
    let base = if rng.chance(1, 2) {
        ArrTerm::Const(default)
    } else {
        ArrTerm::Base
    };
    let write = ArrTerm::Store(
        Box::new(base.clone()),
        Box::new(i.clone()),
        Box::new(v.clone()),
    );
    let hit = Term::Read(Box::new(write.clone()), Box::new(i.clone()));
    let miss = Term::Read(Box::new(write), Box::new(j.clone()));
    let base_read = Term::Read(Box::new(base.clone()), Box::new(j));
    match rng.below(5) {
        // RoW-1 under an application: the row that answered `sat`.
        0 => Atom::Distinct(
            Term::App(0, Box::new(hit)),
            Term::App(0, Box::new(v.clone())),
        ),
        // RoW-2 under an application, guarded by the index disequality.
        1 => Atom::Or(
            Box::new(Atom::Eq(i, Term::Var(p.num_vars - 1))),
            Box::new(Atom::Distinct(
                Term::App(0, Box::new(miss)),
                Term::App(0, Box::new(base_read)),
            )),
        ),
        // The same read under an operator *inside* the application.
        2 => Atom::Distinct(
            Term::App(0, Box::new(Term::Add(Box::new(hit), Box::new(v.clone())))),
            Term::App(
                0,
                Box::new(Term::Add(Box::new(v.clone()), Box::new(v.clone()))),
            ),
        ),
        // A read whose index is itself an application.
        3 => Atom::Eq(
            Term::Read(Box::new(base), Box::new(Term::App(0, Box::new(i)))),
            v,
        ),
        // A read of an array constant under an application (`#P2b-36`): no
        // `store` over it at all, so read-over-write decides nothing and only
        // the array-constant axiom refutes the disequality.
        _ => Atom::Distinct(
            Term::App(
                0,
                Box::new(Term::Read(Box::new(ArrTerm::Const(default)), Box::new(i))),
            ),
            Term::App(0, Box::new(Term::Const(default))),
        ),
    }
}

fn gen_problem(rng: &mut Rng, width: u32) -> Problem {
    let mut p = Problem {
        width,
        num_vars: 1 + rng.below(2) as usize,
        num_funcs: 1 + rng.below(2) as usize,
        asserts: Vec::new(),
    };
    // HALF THE PROBLEMS ARE PLANTED SHAPES ALONE, and that is the point.
    //
    // A problem that *also* contains a read in ordinary operand position is
    // decided correctly even by the broken tree: any such read raises
    // `has_array_ops` through `track_theory_vars`, the refinement loop runs,
    // and the walk then collects the argument-position read too.  The defect
    // needs a formula whose reads are *only* under applications — `plant_shape`
    // 0..=2 are exactly those — so mixing random atoms in would hide it.
    // Verified: reverting `Solver::mark_array_ops` makes this campaign report
    // `wrong_sat`, and it does so within the bounded seeds.
    if rng.chance(1, 2) {
        let count = 1 + rng.below(2) as usize;
        for _ in 0..count {
            let planted = argument_position_shape(rng, &p);
            p.asserts.push(planted);
        }
        return p;
    }
    let count = 1 + rng.below(2) as usize;
    for _ in 0..count {
        let atom = gen_atom(rng, &p, 2);
        p.asserts.push(atom);
    }
    if rng.chance(1, 2) {
        let planted = plant_shape(rng, &p);
        p.asserts.push(planted);
    }
    p
}

// ---------------------------------------------------------------------------
// Printing.
// ---------------------------------------------------------------------------

fn print_const(value: u128, width: u32, out: &mut String) {
    if width.is_multiple_of(4) {
        let _ = write!(out, "#x{value:0w$x}", w = (width / 4) as usize);
    } else {
        let _ = write!(out, "#b{value:0w$b}", w = width as usize);
    }
}

fn print_array(a: &ArrTerm, p: &Problem, out: &mut String) {
    match a {
        ArrTerm::Base => out.push_str("arr"),
        ArrTerm::Const(default) => {
            let w = p.width;
            let _ = write!(out, "((as const (Array (_ BitVec {w}) (_ BitVec {w}))) ");
            print_const(*default, w, out);
            out.push(')');
        }
        ArrTerm::Store(base, index, value) => {
            out.push_str("(store ");
            print_array(base, p, out);
            out.push(' ');
            print_term(index, p, out);
            out.push(' ');
            print_term(value, p, out);
            out.push(')');
        }
    }
}

fn print_term(t: &Term, p: &Problem, out: &mut String) {
    match t {
        Term::Var(i) => {
            let _ = write!(out, "v{i}");
        }
        Term::Const(c) => print_const(*c, p.width, out),
        Term::App(f, a) => {
            let _ = write!(out, "(f{f} ");
            print_term(a, p, out);
            out.push(')');
        }
        Term::Read(array, index) => {
            out.push_str("(select ");
            print_array(array, p, out);
            out.push(' ');
            print_term(index, p, out);
            out.push(')');
        }
        Term::Add(a, b) | Term::Xor(a, b) => {
            let name = if matches!(t, Term::Add(..)) {
                "bvadd"
            } else {
                "bvxor"
            };
            let _ = write!(out, "({name} ");
            print_term(a, p, out);
            out.push(' ');
            print_term(b, p, out);
            out.push(')');
        }
        Term::Not(a) => {
            out.push_str("(bvnot ");
            print_term(a, p, out);
            out.push(')');
        }
        Term::Ite(c, a, b) => {
            out.push_str("(ite ");
            print_atom(c, p, out);
            out.push(' ');
            print_term(a, p, out);
            out.push(' ');
            print_term(b, p, out);
            out.push(')');
        }
    }
}

fn print_atom(a: &Atom, p: &Problem, out: &mut String) {
    match a {
        Atom::Eq(x, y) | Atom::Distinct(x, y) | Atom::Ult(x, y) | Atom::Ule(x, y) => {
            let name = match a {
                Atom::Eq(..) => "=",
                Atom::Distinct(..) => "distinct",
                Atom::Ult(..) => "bvult",
                _ => "bvule",
            };
            let _ = write!(out, "({name} ");
            print_term(x, p, out);
            out.push(' ');
            print_term(y, p, out);
            out.push(')');
        }
        Atom::Or(x, y) => {
            out.push_str("(or ");
            print_atom(x, p, out);
            out.push(' ');
            print_atom(y, p, out);
            out.push(')');
        }
        Atom::Not(x) => {
            out.push_str("(not ");
            print_atom(x, p, out);
            out.push(')');
        }
    }
}

/// The harness's deterministic embedded-check budget.
///
/// 500 is more than twice the largest count any script in `bench/` needs (207,
/// `extended_theories/QF_ABV/02_bv_array_overwrite.smt2`), so an ordinary
/// generated problem is decided well inside it and only a runaway is cut off.
/// Measured on the widths-1-and-2 gate below, in the **test** profile these
/// tests actually run in (the bit-blaster's circuit self-check is gated on
/// `debug_assertions` and makes this shape about 100x slower than release):
/// 72 scripts in 5 s at 500 with 4 `unknown_decided`, 52 s at 1,000 with 2,
/// and no completion inside 400 s at 2,000 — the cost per budget unit grows
/// with the budget, because each embedded check is `O(num_vars)` over a
/// variable table that grows with the search (`TODO.md` `#P2b-46` (f)).
/// A cut-off script is scored `unknown_decided` and is never a failure, and
/// every *other* bound these tests assert is **zero** (`wrong_sat`,
/// `wrong_unsat`, `bad_model`, panics, errors), so shrinking the budget can
/// only check less, never manufacture a pass over a real defect.
///
/// "Check less" is itself bounded: [`report`] asserts a **floor** on
/// `unknown_decided` for each gate — 4 of 72 for the exhaustive one, 0 of 40
/// for the sampled one, 1 of 480 for `ext_shapes` — so lowering this constant
/// makes those gates RED rather than fast.  Without that floor, setting this
/// constant to `1` passed both gates in 0.652 s and 0.021 s.
///
/// It cannot raise the solver's calibrated ceiling — see
/// `theory_manager::bv_budget`.
const HARNESS_EMBEDDED_CHECK_BUDGET: u64 = 500;

/// Render `p` as a script.
///
/// **No wall clock anywhere** (decision (16), finding R5-5).  A script that
/// renders `(set-option :timeout N)` asserts its verdicts behind a wall clock:
/// on a slower machine fewer scripts are decided, so every bound the test
/// checks gets weaker, and the test's strength becomes a property of the host
/// rather than of the solver.
///
/// Two *deterministic* budgets stand in its place, and they bound the same
/// runaways identically on every machine — that is the whole point of decision
/// (9).  `(set-option :max-conflicts 20000)` bounds the Boolean search, and
/// `(set-option :max-bv-embedded-checks …)` bounds the third currency neither
/// of the other two can see: the bit-vector bridge runs one complete embedded
/// `BvSolver::check` per bit-vector atom propagation, and a generated script
/// that never conflicts can still spend the whole calibrated ceiling there.
/// Measured on this file's own worst generated script (a width-2 problem with
/// a deeply nested `store`/`select` spine, `rf7/slow59.smt2`): `unknown` in
/// 384 ms at 2,000 embedded checks, 58.2 s at 20,000, 362.9 s at 100,000, and
/// no answer in 900 s at the 250,000 default.  Before this budget existed the
/// two gates below exceeded nextest's ceiling without finishing.
fn render(p: &Problem, named: bool, pins: Option<&[u128]>) -> String {
    let mut out = String::new();
    out.push_str("(set-logic QF_AUFBV)\n");
    if named {
        out.push_str("(set-option :produce-unsat-cores true)\n");
    }
    out.push_str("(set-option :produce-models true)\n");
    // A budget per check, as the adversarial probe does: a `unknown` from an
    // exhausted budget is scored (`unknown_decided`), never a failure, and one
    // slow circuit cannot stall the suite.
    out.push_str("(set-option :max-conflicts 20000)\n");
    let _ = writeln!(
        out,
        "(set-option :max-bv-embedded-checks {HARNESS_EMBEDDED_CHECK_BUDGET})"
    );
    let w = p.width;
    for i in 0..p.num_vars {
        let _ = writeln!(out, "(declare-const v{i} (_ BitVec {w}))");
    }
    let _ = writeln!(
        out,
        "(declare-const arr (Array (_ BitVec {w}) (_ BitVec {w})))"
    );
    for f in 0..p.num_funcs {
        let _ = writeln!(out, "(declare-fun f{f} ((_ BitVec {w})) (_ BitVec {w}))");
    }
    for (k, a) in p.asserts.iter().enumerate() {
        if named {
            out.push_str("(assert (! ");
        } else {
            out.push_str("(assert ");
        }
        print_atom(a, p, &mut out);
        if named {
            let _ = writeln!(out, " :named a{k}))");
        } else {
            out.push_str(")\n");
        }
    }
    if let Some(values) = pins {
        for (i, v) in values.iter().enumerate() {
            let _ = write!(out, "(assert (= v{i} ");
            print_const(*v, w, &mut out);
            out.push_str("))\n");
        }
    }
    out.push_str("(check-sat)\n(get-value (");
    for i in 0..p.num_vars {
        if i > 0 {
            out.push(' ');
        }
        let _ = write!(out, "v{i}");
    }
    out.push_str("))\n");
    if named {
        out.push_str("(get-unsat-core)\n");
    }
    out
}

// ---------------------------------------------------------------------------
// Reference semantics and the oracle.
// ---------------------------------------------------------------------------

struct Interp<'a> {
    width: u32,
    vars: &'a [u128],
    /// `array[index]`: the declared array's contents.
    array: &'a [u128],
    /// `tables[f][arg]`: the value of function `f` at `arg`.
    tables: &'a [Vec<u128>],
}

/// The largest array domain the oracle handles: `2^4`, the widest campaign
/// width.  A fixed-capacity table is what keeps the exhaustive loop free of
/// allocation — at width 2 it runs 2^18 times per problem, and a `Vec` per
/// `store` level made that the slowest part of the campaign by two orders of
/// magnitude.
const MAX_DOMAIN: usize = 16;

/// The contents of an array term: the base table with every write applied in
/// order, innermost first — exactly SMT-LIB's `store` semantics.
fn eval_array(a: &ArrTerm, i: &Interp<'_>) -> [u128; MAX_DOMAIN] {
    match a {
        ArrTerm::Base => {
            let mut table = [0u128; MAX_DOMAIN];
            for (slot, value) in table.iter_mut().zip(i.array) {
                *slot = *value;
            }
            table
        }
        ArrTerm::Const(default) => [*default & mask(i.width); MAX_DOMAIN],
        ArrTerm::Store(base, index, value) => {
            let mut table = eval_array(base, i);
            let at = eval_term(index, i) as usize;
            let v = eval_term(value, i);
            if let Some(slot) = table.get_mut(at) {
                *slot = v;
            }
            table
        }
    }
}

fn eval_term(t: &Term, i: &Interp<'_>) -> u128 {
    let m = mask(i.width);
    match t {
        Term::Var(v) => i.vars[*v],
        Term::Const(c) => *c,
        Term::App(f, a) => {
            let arg = eval_term(a, i) as usize;
            i.tables[*f][arg]
        }
        Term::Read(array, index) => {
            let table = eval_array(array, i);
            let at = eval_term(index, i) as usize;
            table.get(at).copied().unwrap_or(0)
        }
        Term::Add(a, b) => eval_term(a, i).wrapping_add(eval_term(b, i)) & m,
        Term::Xor(a, b) => eval_term(a, i) ^ eval_term(b, i),
        Term::Not(a) => !eval_term(a, i) & m,
        Term::Ite(c, a, b) => {
            if eval_atom(c, i) {
                eval_term(a, i)
            } else {
                eval_term(b, i)
            }
        }
    }
}

fn eval_atom(a: &Atom, i: &Interp<'_>) -> bool {
    match a {
        Atom::Eq(x, y) => eval_term(x, i) == eval_term(y, i),
        Atom::Distinct(x, y) => eval_term(x, i) != eval_term(y, i),
        Atom::Ult(x, y) => eval_term(x, i) < eval_term(y, i),
        Atom::Ule(x, y) => eval_term(x, i) <= eval_term(y, i),
        Atom::Or(x, y) => eval_atom(x, i) || eval_atom(y, i),
        Atom::Not(x) => !eval_atom(x, i),
    }
}

fn all_hold(p: &Problem, i: &Interp<'_>) -> bool {
    p.asserts.iter().all(|a| eval_atom(a, i))
}

/// Exhaustive truth when the space of variable assignments, array contents and
/// function tables is at most 2^20, else `None`.
///
/// The array is one more table of `2^width` entries — the same shape as a
/// function's — so the budget is spent the same way and the width-2 problems
/// that fit are those with one variable and one function
/// (2·2 + 4·2 + 4·2 = 20 bits).
fn exhaustive(p: &Problem) -> Option<bool> {
    let w = p.width;
    let domain = 1u64 << w;
    let table_bits = u64::from(w) * domain;
    let total_bits = u64::from(w) * p.num_vars as u64 + table_bits * (p.num_funcs as u64 + 1);
    if total_bits > 20 {
        return None;
    }
    let var_space = 1u64 << (u64::from(w) * p.num_vars as u64);
    let table_space = 1u64 << table_bits;
    let mut tables: Vec<Vec<u128>> = vec![vec![0; domain as usize]; p.num_funcs];
    let mut array: Vec<u128> = vec![0; domain as usize];
    let mut vars = vec![0u128; p.num_vars];
    for vi in 0..var_space {
        for (k, v) in vars.iter_mut().enumerate() {
            *v = u128::from((vi >> (u64::from(w) * k as u64)) & (domain - 1));
        }
        // One mixed-radix counter over the array's table and each function's.
        let mut idx = vec![0u64; p.num_funcs + 1];
        loop {
            for (arg, slot) in array.iter_mut().enumerate() {
                *slot = u128::from((idx[0] >> (u64::from(w) * arg as u64)) & (domain - 1));
            }
            for (f, table) in tables.iter_mut().enumerate() {
                for (arg, slot) in table.iter_mut().enumerate() {
                    *slot = u128::from((idx[f + 1] >> (u64::from(w) * arg as u64)) & (domain - 1));
                }
            }
            let interp = Interp {
                width: w,
                vars: &vars,
                array: &array,
                tables: &tables,
            };
            if all_hold(p, &interp) {
                return Some(true);
            }
            let mut carry = true;
            for slot in idx.iter_mut() {
                if !carry {
                    break;
                }
                *slot += 1;
                if *slot == table_space {
                    *slot = 0;
                } else {
                    carry = false;
                }
            }
            if carry {
                break;
            }
        }
    }
    Some(false)
}

/// Random witness search: proves `sat` only.
fn sampled_witness(rng: &mut Rng, p: &Problem, samples: usize) -> bool {
    let w = p.width;
    let domain = 1usize << w;
    let m = mask(w);
    for _ in 0..samples {
        let vars: Vec<u128> = (0..p.num_vars)
            .map(|_| match rng.below(3) {
                0 => u128::from(rng.below(4)) & m,
                1 => m - (u128::from(rng.below(4)) & m),
                _ => u128::from(rng.next_u64()) & m,
            })
            .collect();
        let array: Vec<u128> = (0..domain)
            .map(|_| u128::from(rng.next_u64()) & m)
            .collect();
        let tables: Vec<Vec<u128>> = (0..p.num_funcs)
            .map(|_| {
                (0..domain)
                    .map(|_| u128::from(rng.next_u64()) & m)
                    .collect()
            })
            .collect();
        let interp = Interp {
            width: w,
            vars: &vars,
            array: &array,
            tables: &tables,
        };
        if all_hold(p, &interp) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Running and scoring.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Run {
    Lines(Vec<String>),
    Error(String),
    Panic(String),
}

fn run_script(script: &str) -> Run {
    match catch_unwind(AssertUnwindSafe(|| {
        let mut ctx = Context::new();
        ctx.execute_script(script)
    })) {
        Ok(Ok(lines)) => Run::Lines(lines),
        Ok(Err(e)) => Run::Error(e.to_string()),
        Err(payload) => Run::Panic(
            payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string()),
        ),
    }
}

fn printed_bv(block: &str, name: &str) -> Option<u128> {
    let key = format!("({name} ");
    let after = block.find(&key).map(|at| &block[at + key.len()..])?;
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

#[derive(Default, Debug)]
struct Tally {
    scripts: usize,
    sat: usize,
    unsat: usize,
    unknown: usize,
    /// `unknown` on a formula the exhaustive oracle decided: precision lost.
    unknown_decided: usize,
    errors: usize,
    panics: usize,
    wrong_sat: usize,
    wrong_unsat: usize,
    bad_core: usize,
    bad_model: usize,
}

impl Tally {
    fn has_failures(&self) -> bool {
        self.errors
            + self.panics
            + self.wrong_sat
            + self.wrong_unsat
            + self.bad_core
            + self.bad_model
            > 0
    }
}

struct Failure {
    kind: &'static str,
    detail: String,
    script: String,
}

/// Runs one problem plain and named, scoring both against `truth` (`Some` when
/// the oracle decided it, `None` when only a sampled witness search ran and
/// found nothing).
fn run_problem(rng: &mut Rng, p: &Problem, tally: &mut Tally, failures: &mut Vec<Failure>) {
    let decided_truth = exhaustive(p);
    let truth = match decided_truth {
        Some(t) => Some(t),
        None => sampled_witness(rng, p, 2000).then_some(true),
    };
    let decided = decided_truth.is_some();
    for named in [false, true] {
        let script = render(p, named, None);
        tally.scripts += 1;
        let lines = match run_script(&script) {
            Run::Panic(msg) => {
                tally.panics += 1;
                failures.push(Failure {
                    kind: "panic",
                    detail: msg,
                    script,
                });
                continue;
            }
            Run::Error(msg) => {
                tally.errors += 1;
                failures.push(Failure {
                    kind: "error",
                    detail: msg,
                    script,
                });
                continue;
            }
            Run::Lines(lines) => lines,
        };
        let verdict = lines.first().map(String::as_str).unwrap_or("none");
        match verdict {
            "sat" => {
                tally.sat += 1;
                if truth == Some(false) {
                    tally.wrong_sat += 1;
                    failures.push(Failure {
                        kind: "wrong-sat",
                        detail: "the exhaustive oracle says unsat".to_string(),
                        script,
                    });
                    continue;
                }
                let block = lines.get(1).map(String::as_str).unwrap_or("");
                let mut values = Vec::new();
                for i in 0..p.num_vars {
                    match printed_bv(block, &format!("v{i}")) {
                        Some(v) => values.push(v),
                        None => {
                            tally.bad_model += 1;
                            failures.push(Failure {
                                kind: "sat-without-model-value",
                                detail: lines.join(" | "),
                                script: script.clone(),
                            });
                            break;
                        }
                    }
                }
                if values.len() == p.num_vars {
                    let pinned = render(p, false, Some(&values));
                    let pinned_verdict = match run_script(&pinned) {
                        Run::Lines(l) => l.first().cloned().unwrap_or_default(),
                        Run::Error(e) => format!("error: {e}"),
                        Run::Panic(m) => format!("panic: {m}"),
                    };
                    if pinned_verdict == "unsat" {
                        tally.bad_model += 1;
                        failures.push(Failure {
                            kind: "model-does-not-extend",
                            detail: format!("pinned values {values:x?} answer unsat"),
                            script: script.clone(),
                        });
                    }
                }
            }
            "unsat" => {
                tally.unsat += 1;
                if truth == Some(true) {
                    tally.wrong_unsat += 1;
                    failures.push(Failure {
                        kind: "wrong-unsat",
                        detail: "a witness exists".to_string(),
                        script: script.clone(),
                    });
                    continue;
                }
                if named {
                    let names: Vec<String> =
                        (0..p.asserts.len()).map(|k| format!("a{k}")).collect();
                    let core_line = lines.get(2).map(String::as_str).unwrap_or("");
                    let inner = core_line
                        .trim()
                        .trim_start_matches('(')
                        .trim_end_matches(')');
                    let core: Vec<&str> = inner.split_whitespace().collect();
                    if core.is_empty() || !core.iter().all(|n| names.iter().any(|m| m == n)) {
                        tally.bad_core += 1;
                        failures.push(Failure {
                            kind: "bad-core",
                            detail: lines.join(" | "),
                            script: script.clone(),
                        });
                    }
                }
            }
            "unknown" => {
                // Not a failure — `unknown` is always a legal verdict — but
                // lost precision, so every one is written out for inspection.
                tally.unknown += 1;
                if decided {
                    tally.unknown_decided += 1;
                }
                failures.push(Failure {
                    kind: if decided {
                        "unknown-decided"
                    } else {
                        "unknown"
                    },
                    detail: format!(
                        "oracle: {}",
                        match truth {
                            Some(true) => "sat",
                            Some(false) => "unsat",
                            None => "undecided (no sampled witness)",
                        }
                    ),
                    script,
                });
            }
            other => {
                tally.errors += 1;
                failures.push(Failure {
                    kind: "no-verdict",
                    detail: other.to_string(),
                    script,
                });
            }
        }
    }
}

fn campaign(
    seed: u64,
    trials: usize,
    widths: &[u32],
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) {
    let mut rng = Rng::new(seed);
    for _ in 0..trials {
        let width = widths[rng.below(widths.len() as u64) as usize];
        let p = gen_problem(&mut rng, width);
        run_problem(&mut rng, &p, tally, failures);
    }
}

/// Print the tally, dump the failing scripts, and assert the gate's bounds.
///
/// `max_unknown_decided` is the **floor on the gate's strength**, and it is
/// what keeps [`HARNESS_EMBEDDED_CHECK_BUDGET`] honest.  Every other bound this
/// function checks is zero (`wrong_sat`, `wrong_unsat`, `bad_core`,
/// `bad_model`, errors, panics), so a budget small enough to cut every script
/// off would satisfy all of them vacuously: measured, with the budget set to
/// `1` both gates PASS, in 0.652 s and 0.021 s against 4.2 s and 17.2 s at the
/// live 500.  `unknown_decided` counts exactly those cut-offs — an `unknown` on
/// a formula the oracle DECIDED — so bounding it above turns "the gate checked
/// nothing" into a red gate.
///
/// `min_decided` is the same floor in the other currency, and the sampled gate
/// needs it: that gate's oracle *samples* witnesses, so a script it cannot
/// decide is scored plain `unknown` rather than `unknown_decided` and the
/// count above stays 0 however little the gate does.  Measured, with the
/// budget set to `1`: the exhaustive and `ext_shapes` gates go red on
/// `unknown_decided` and the sampled gate still PASSED, in 0.022 s with 0 of
/// 40 scripts decided.
///
/// **Lowering `HARNESS_EMBEDDED_CHECK_BUDGET` must move these numbers**, and a
/// number that has to be raised to make a gate pass is a defect report, not a
/// maintenance chore.  `None` is for the `#[ignore]`d campaign, whose script
/// count is set from the environment and whose counts therefore are not fixed
/// properties of the tree.
fn report(
    tag: &str,
    tally: &Tally,
    failures: &[Failure],
    max_unknown_decided: Option<usize>,
    min_decided: Option<usize>,
) {
    eprintln!("[{tag}] {tally:?}");
    let dir = std::env::temp_dir().join("oxiz-array-uf-combination");
    let _ = std::fs::create_dir_all(&dir);
    for (i, f) in failures.iter().take(50).enumerate() {
        eprintln!("[{tag}] {}: {}", f.kind, f.detail);
        let text = format!("; kind: {}\n; {}\n{}", f.kind, f.detail, f.script);
        let _ = std::fs::write(dir.join(format!("{tag}-{i:03}-{}.smt2", f.kind)), text);
    }
    let real: Vec<&Failure> = failures
        .iter()
        .filter(|f| !f.kind.starts_with("unknown"))
        .collect();
    assert!(
        !tally.has_failures(),
        "[{tag}] {} failures; first [{}] {}\n{}",
        real.len(),
        real.first().map(|f| f.kind).unwrap_or("-"),
        real.first().map(|f| f.detail.as_str()).unwrap_or(""),
        real.first().map(|f| f.script.as_str()).unwrap_or("")
    );
    if let Some(floor) = min_decided {
        let decided = tally.sat + tally.unsat;
        assert!(
            decided >= floor,
            "[{tag}] only {decided} of {} scripts got a verdict, against a floor \
             of {floor}.  `unknown_decided` is the wrong currency for a gate \
             whose oracle samples witnesses — it stays 0 however little the \
             gate decides — so the floor is on the verdicts themselves.  \
             {tally:?}",
            tally.scripts
        );
    }
    if let Some(ceiling) = max_unknown_decided {
        assert!(
            tally.unknown_decided <= ceiling,
            "[{tag}] {} scripts the oracle decided were cut off by \
             `HARNESS_EMBEDDED_CHECK_BUDGET` ({HARNESS_EMBEDDED_CHECK_BUDGET}), \
             against a floor of {ceiling}.  This gate's strength is the number \
             of scripts it actually decides; see `report`'s doc.  {tally:?}",
            tally.unknown_decided
        );
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The array-*extensionality* shapes (`#P2b-37`): array (dis)equality atoms,
/// `n`-ary `distinct` over arrays, `store = ((as const S) d)`, foreign array
/// arguments and arrays of arrays, each scored against an exhaustive oracle
/// and each `sat` model replayed through the reference semantics.
#[path = "array_uf_combination/ext_shapes.rs"]
mod ext_shapes;

/// A fixed script this gate decides, asserted by name.
///
/// `tally.sat + tally.unsat > 0` used to stand here (and in the sampled gate
/// below), and while `render` emitted `(set-option :timeout 5000)` that was a
/// verdict assertion behind a wall clock — decision (16)'s letter, broken: on
/// a machine slow enough for every generated script to time out, the bound
/// would have been the only thing standing between the gate and a vacuous
/// pass, and it is exactly the bound a slow machine removes.  A *named*
/// script with a known answer says the same thing (the harness really does run
/// scripts and really does decide them) and says it independently of how many
/// generated problems the host got through.
///
/// `(select (store arr v0 v1) v0) = v1` is read-over-write's own instance, so
/// `distinct` from `v1` is unsatisfiable at any width; the sat twin pins the
/// other direction so a solver that answered `unsat` to everything would fail
/// here too.
fn the_harness_decides_a_fixed_script() {
    let unsat = "(set-logic QF_AUFBV)\n\
         (declare-const v0 (_ BitVec 2))\n\
         (declare-const v1 (_ BitVec 2))\n\
         (declare-const arr (Array (_ BitVec 2) (_ BitVec 2)))\n\
         (assert (distinct (select (store arr v0 v1) v0) v1))\n\
         (check-sat)\n";
    let lines = match run_script(unsat) {
        Run::Lines(lines) => lines,
        other => panic!("the harness must run this script: {other:?}"),
    };
    assert_eq!(
        lines.first().map(String::as_str),
        Some("unsat"),
        "read-over-write's own instance must be refuted, with no clock in \
         sight: this is the gate's proof that it ran anything at all"
    );
    let sat = "(set-logic QF_AUFBV)\n\
         (declare-const v0 (_ BitVec 2))\n\
         (declare-const v1 (_ BitVec 2))\n\
         (declare-const arr (Array (_ BitVec 2) (_ BitVec 2)))\n\
         (assert (= (select (store arr v0 v1) v0) v1))\n\
         (check-sat)\n";
    let lines = match run_script(sat) {
        Run::Lines(lines) => lines,
        other => panic!("the harness must run this script: {other:?}"),
    };
    assert_eq!(
        lines.first().map(String::as_str),
        Some("sat"),
        "and the satisfiable twin, so \"answer `unsat` to everything\" does \
         not pass this gate"
    );
}

/// Widths 1 and 2, every verdict the oracle decides checked against it.
#[test]
fn array_uf_exhaustive_small_widths() {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in 0..3 {
        campaign(seed, 12, &[1, 2], &mut tally, &mut failures);
    }
    // Measured on this tree at the live budget: 72 scripts, 68 decided, 4 cut
    // off.
    report("exhaustive", &tally, &failures, Some(4), Some(68));
    the_harness_decides_a_fixed_script();
}

/// Widths 3 and 4, sampled witnesses (an `unsat` against a witness fails) and
/// every `sat` model checked by pinning.
#[test]
fn array_uf_sampled_wider_widths() {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in 10..12 {
        campaign(seed, 10, &[3, 4], &mut tally, &mut failures);
    }
    // Measured on this tree at the live budget: 40 scripts, 38 decided, 0 cut
    // off.
    report("sampled", &tally, &failures, Some(0), Some(38));
    the_harness_decides_a_fixed_script();
}

/// The bounded extensionality campaign: width-1 index **and** element sorts —
/// where the four inhabitants of `(Array (_ BitVec 1) (_ BitVec 1))` make the
/// cardinality argument bite — plus width-2 elements and arrays of arrays at
/// width 1.
///
/// Every verdict is checked against an exhaustive oracle over every
/// interpretation of every declared symbol, and every `sat` model is parsed
/// back and re-evaluated, so a model that falsifies its own script fails the
/// run even when the verdict is right.
///
/// Reverting any one of the four families `#P2b-37` added makes this run
/// report `wrong_sat`; the counts are in the `#P2b-37` entry of `TODO.md`.
#[test]
fn array_ext_shapes_bounded() {
    let mut tally = ext_shapes::ExtTally::default();
    let mut first_failure = Vec::new();
    ext_shapes::campaign(
        0..12,
        40,
        &[(1, 1, false), (1, 1, true), (1, 2, false), (2, 1, false)],
        &mut tally,
        &mut first_failure,
    );
    eprintln!("[ext-shapes] {tally:?}");
    assert_eq!(
        tally.failures(),
        0,
        "{} failures; first of each kind:\n{}",
        tally.failures(),
        ext_shapes::summarise(&first_failure)
    );
    // `tally.sat + tally.unsat > 0` stood here, and decision (27) named it.
    // It is gone rather than replaced: `tally.sat + tally.unsat >= 479` four
    // assertions below is the same claim with a real floor, and a weak
    // assertion beside a strong one about the same quantity is a place a later
    // pass can weaken the strong one and still see green.
    // The published-model residue is **zero**, and this is now a bound that
    // means something.  While the generator rendered every script with
    // `(set-option :timeout 1000)` it did not: `score` replays a published
    // model only on the `sat` branch, so a script that ran out of clock was
    // never model-checked, and the count therefore moved with how many
    // scripts got past the clock — on a *faster* machine more scripts reach
    // `sat`, more models are replayed, and the tally can rise above whatever
    // bound is written here.  That is the inverse of ordinary load flakiness,
    // which is why no pass ever saw it go red (`#P2b-46`).  The clock is gone
    // and `:max-conflicts` decides instead, so which scripts answer `sat` is
    // a property of the formula: 480 scripts, 163 `sat`, 316 `unsat`, 1
    // `unknown`, 0 falsifying models.
    assert_eq!(
        tally.bad_model, 0,
        "a published model falsifies its own script: {tally:?}"
    );
    // The same floor `report` asserts for the two `array_uf` gates, and for
    // the same reason: every other bound here is zero, so without it a budget
    // small enough to cut every script off would pass vacuously.  Measured on
    // this tree at the live budget: 480 scripts, 1 cut off.
    assert!(
        tally.sat + tally.unsat >= 479,
        "the gate's strength is the number of scripts it decides: {tally:?}"
    );
    assert!(
        tally.unknown_decided <= 1,
        "scripts the oracle decided were cut off by \
         `HARNESS_EMBEDDED_CHECK_BUDGET` ({HARNESS_EMBEDDED_CHECK_BUDGET}); \
         this gate's strength is the number of scripts it decides: {tally:?}"
    );
}

/// The long extensionality campaign: `OXIZ_EXT_SEED_LO/HI` (default 0..120),
/// `OXIZ_EXT_TRIALS` (default 50 per seed) — 6,000 scripts by default.
#[test]
#[ignore = "long-running differential campaign; run explicitly"]
fn array_ext_shapes_campaign() {
    let lo = env_u64("OXIZ_EXT_SEED_LO", 0);
    let hi = env_u64("OXIZ_EXT_SEED_HI", 120);
    let trials = env_u64("OXIZ_EXT_TRIALS", 50) as usize;
    let mut tally = ext_shapes::ExtTally::default();
    let mut first_failure = Vec::new();
    ext_shapes::campaign(
        lo..hi,
        trials,
        &[
            (1, 1, false),
            (1, 1, true),
            (1, 2, false),
            (2, 1, false),
            (2, 2, false),
        ],
        &mut tally,
        &mut first_failure,
    );
    eprintln!("[ext-campaign] {tally:?}");
    assert_eq!(
        tally.failures(),
        0,
        "{} failures; first of each kind:\n{}",
        tally.failures(),
        ext_shapes::summarise(&first_failure)
    );
    // The published-model residue, asserted here too and not only in the
    // bounded gate: `ExtTally::failures` does not count it, so until
    // `#P2b-46` a falsifying model in the long run was printed and passed.
    //
    // **THIS ASSERTION IS CURRENTLY RED, and deliberately so (`#P2b-49`).**
    // The 6,000-script default reports `bad_model: 1`. It read 0 only while
    // `ext_shapes::render` still emitted `(set-option :timeout 1000)`, because
    // `score` replays a published model on the `sat` branch alone and the one
    // offending script never got that far — which is exactly the defect
    // decision (16) describes, found by its own fix. The script is two nested
    // arrays asserted `distinct` that print as the *same* constant array;
    // `c4b04b7` prints byte-identical output, so the defect is pre-existing and
    // outside this round's findings. The bound stays at zero and is **not**
    // narrowed: see `TODO.md` `#P2b-49` for the repro and the analysis.
    assert_eq!(
        tally.bad_model,
        0,
        "a published model falsifies its own script: {tally:?}\n{}",
        ext_shapes::summarise(&first_failure)
    );
}

/// Long form: `OXIZ_AUF_SEED_LO/HI` (default 0..16), `OXIZ_AUF_TRIALS`
/// (default 100 per seed).
#[test]
#[ignore = "long-running differential campaign; run explicitly"]
fn array_uf_campaign() {
    let lo = env_u64("OXIZ_AUF_SEED_LO", 0);
    let hi = env_u64("OXIZ_AUF_SEED_HI", 16);
    let trials = env_u64("OXIZ_AUF_TRIALS", 100) as usize;
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in lo..hi {
        // No clock here either: the deterministic embedded-check budget
        // `render` emits is what keeps one pathological circuit from stalling
        // the campaign, and it does so identically on every machine
        // (decision (16)).
        campaign(seed, trials, &[1, 2, 3, 4], &mut tally, &mut failures);
        eprintln!("[campaign] seed {seed}: {tally:?}");
    }
    report(
        &format!("campaign-{lo}-{hi}"),
        &tally,
        &failures,
        None,
        None,
    );
}
