//! Differential campaign for the bit-vector / EUF equality exchange
//! (`TheoryManager::combine_bv_with_euf`, `#P2b-29`), scored against an
//! **exhaustive** oracle: random QF_UFBV / QF_ABV formulas over one or two
//! bit-vector variables, one or two uninterpreted functions (or an array),
//! bit-vector arithmetic under and over the applications, and comparisons
//! and (dis)equalities as atoms.  At widths 1 and 2 the oracle enumerates
//! every variable assignment together with every *function table* (a
//! function on a 2-bit domain has 2^8 tables), so every verdict — `sat`,
//! `unsat` and `unknown` alike — is checked against the truth.  At widths 3,
//! 4 and 8 the oracle samples witnesses, which proves `sat` only, so an
//! `unsat` against a sampled witness is a failure and a `sat` is checked by
//! re-running the script with the printed variable values pinned.
//!
//! Both directions of the exchange are exercised: applications whose
//! arguments the circuit forces equal (bit-vector → EUF), and applications
//! whose results feed bit-vector operations (EUF → bit-vector).  Every
//! script also runs in cargo-formal's named-core form.
//!
//! The bounded tests run on every `cargo test`; `bv_euf_campaign` is the
//! long form (`OXIZ_BVEUF_SEED_LO/HI`, `OXIZ_BVEUF_TRIALS`).

use oxiz_solver::Context;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

// ---------------------------------------------------------------------------
// PRNG (splitmix64).
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

/// A bit-vector term over `num_vars` variables and `num_funcs` unary
/// functions (`select` on one array is function 0 when `array` is set).
#[derive(Clone, Debug)]
enum Term {
    Var(usize),
    Const(u128),
    App(usize, Box<Term>),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    And(Box<Term>, Box<Term>),
    Xor(Box<Term>, Box<Term>),
    Not(Box<Term>),
    Ite(Box<Atom>, Box<Term>, Box<Term>),
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
    array: bool,
    asserts: Vec<Atom>,
}

fn mask(width: u32) -> u128 {
    (1u128 << width) - 1
}

fn gen_term(rng: &mut Rng, p: &Problem, depth: u32) -> Term {
    let w = p.width;
    if depth == 0 || rng.chance(1, 3) {
        return match rng.below(3) {
            0 => Term::Var(rng.below(p.num_vars as u64) as usize),
            1 => Term::Const(u128::from(rng.next_u64()) & mask(w)),
            _ => Term::App(
                rng.below(p.num_funcs as u64) as usize,
                Box::new(gen_term(rng, p, depth.saturating_sub(1))),
            ),
        };
    }
    let sub = |rng: &mut Rng| Box::new(gen_term(rng, p, depth - 1));
    match rng.below(7) {
        0 => Term::App(rng.below(p.num_funcs as u64) as usize, sub(rng)),
        1 => Term::Add(sub(rng), sub(rng)),
        2 => Term::Sub(sub(rng), sub(rng)),
        3 => Term::And(sub(rng), sub(rng)),
        4 => Term::Xor(sub(rng), sub(rng)),
        5 => Term::Not(sub(rng)),
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
    if depth > 0 && rng.chance(1, 4) {
        let other = gen_atom(rng, p, depth - 1);
        return Atom::Or(Box::new(base), Box::new(other));
    }
    if rng.chance(1, 5) {
        return Atom::Not(Box::new(base));
    }
    base
}

fn gen_problem(rng: &mut Rng, width: u32) -> Problem {
    let mut p = Problem {
        width,
        num_vars: 1 + rng.below(2) as usize,
        num_funcs: 1 + rng.below(2) as usize,
        array: rng.chance(1, 4),
        asserts: Vec::new(),
    };
    if p.array {
        p.num_funcs = 1;
    }
    let count = 1 + rng.below(3) as usize;
    for _ in 0..count {
        let atom = gen_atom(rng, &p, 2);
        p.asserts.push(atom);
    }
    // Make the exchange matter: half the problems get a direct argument
    // equality the circuit entails, or a congruence under an operation.
    if rng.chance(1, 2) {
        let x = Term::Var(0);
        let y = Term::Var(rng.below(p.num_vars as u64) as usize);
        let c = Term::Const(u128::from(rng.next_u64()) & mask(width));
        let shape = rng.below(3);
        let atom = match shape {
            0 => Atom::Eq(
                Term::Add(Box::new(x.clone()), Box::new(c.clone())),
                Term::Add(Box::new(y.clone()), Box::new(c)),
            ),
            1 => Atom::Distinct(
                Term::Add(Box::new(Term::App(0, Box::new(x))), Box::new(c.clone())),
                Term::Add(Box::new(Term::App(0, Box::new(y))), Box::new(c)),
            ),
            _ => Atom::Distinct(Term::App(0, Box::new(x)), Term::App(0, Box::new(y))),
        };
        p.asserts.push(atom);
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

fn print_term(t: &Term, p: &Problem, out: &mut String) {
    match t {
        Term::Var(i) => {
            let _ = write!(out, "v{i}");
        }
        Term::Const(c) => print_const(*c, p.width, out),
        Term::App(f, a) => {
            if p.array {
                out.push_str("(select arr ");
            } else {
                let _ = write!(out, "(f{f} ");
            }
            print_term(a, p, out);
            out.push(')');
        }
        Term::Add(a, b) | Term::Sub(a, b) | Term::And(a, b) | Term::Xor(a, b) => {
            let name = match t {
                Term::Add(..) => "bvadd",
                Term::Sub(..) => "bvsub",
                Term::And(..) => "bvand",
                _ => "bvxor",
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

fn render(p: &Problem, named: bool, pins: Option<&[u128]>) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "(set-logic {})",
        if p.array { "QF_ABV" } else { "QF_UFBV" }
    );
    if named {
        out.push_str("(set-option :produce-unsat-cores true)\n");
    }
    out.push_str("(set-option :produce-models true)\n");
    let w = p.width;
    for i in 0..p.num_vars {
        let _ = writeln!(out, "(declare-const v{i} (_ BitVec {w}))");
    }
    if p.array {
        let _ = writeln!(
            out,
            "(declare-const arr (Array (_ BitVec {w}) (_ BitVec {w})))"
        );
    } else {
        for f in 0..p.num_funcs {
            let _ = writeln!(out, "(declare-fun f{f} ((_ BitVec {w})) (_ BitVec {w}))");
        }
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
    /// `tables[f][arg]`: the value of function `f` at `arg`.
    tables: &'a [Vec<u128>],
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
        Term::Add(a, b) => eval_term(a, i).wrapping_add(eval_term(b, i)) & m,
        Term::Sub(a, b) => eval_term(a, i).wrapping_sub(eval_term(b, i)) & m,
        Term::And(a, b) => eval_term(a, i) & eval_term(b, i),
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

/// Exhaustive truth when the space of variable assignments and function
/// tables is at most 2^20, else `None`.
fn exhaustive(p: &Problem) -> Option<bool> {
    let w = p.width;
    let domain = 1u64 << w;
    let table_bits = u64::from(w) * domain;
    let total_bits = u64::from(w) * p.num_vars as u64 + table_bits * p.num_funcs as u64;
    if total_bits > 20 {
        return None;
    }
    let var_space = 1u64 << (u64::from(w) * p.num_vars as u64);
    let table_space = 1u64 << table_bits;
    let mut tables: Vec<Vec<u128>> = vec![vec![0; domain as usize]; p.num_funcs];
    let mut vars = vec![0u128; p.num_vars];
    for vi in 0..var_space {
        for (k, v) in vars.iter_mut().enumerate() {
            *v = u128::from((vi >> (u64::from(w) * k as u64)) & (domain - 1));
        }
        let mut idx = vec![0u64; p.num_funcs];
        loop {
            for (f, table) in tables.iter_mut().enumerate() {
                for (arg, slot) in table.iter_mut().enumerate() {
                    *slot = u128::from((idx[f] >> (u64::from(w) * arg as u64)) & (domain - 1));
                }
            }
            let interp = Interp {
                width: w,
                vars: &vars,
                tables: &tables,
            };
            if all_hold(p, &interp) {
                return Some(true);
            }
            // Advance the mixed-radix table index.
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

/// Runs one problem plain and named, scoring both against `truth`
/// (`Some` when the oracle decided it, `None` when only a sampled witness
/// search ran and found nothing).
fn run_problem(rng: &mut Rng, p: &Problem, tally: &mut Tally, failures: &mut Vec<Failure>) {
    let truth = match exhaustive(p) {
        Some(t) => Some(t),
        None => sampled_witness(rng, p, 2000).then_some(true),
    };
    let decided = exhaustive(p).is_some();
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
                // The printed variable values must extend to a model: pin
                // them and the script must stay `sat`.
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
                // lost precision, so every one is written out with the
                // failures for inspection; on a formula the exhaustive
                // oracle decided it is counted separately.
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

fn report(tag: &str, tally: &Tally, failures: &[Failure]) {
    eprintln!("[{tag}] {tally:?}");
    let dir = std::env::temp_dir().join("oxiz-bv-euf-combination");
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
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Widths 1 and 2, every verdict checked against the exhaustive oracle.
#[test]
fn bv_euf_exhaustive_small_widths() {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in 0..4 {
        campaign(seed, 40, &[1, 2], &mut tally, &mut failures);
    }
    report("exhaustive", &tally, &failures);
    assert!(
        tally.sat + tally.unsat > 0,
        "nothing was decided: {tally:?}"
    );
}

/// Widths 3, 4 and 8, sampled witnesses (an `unsat` against a witness
/// fails) and every `sat` model checked by pinning.
#[test]
fn bv_euf_sampled_wider_widths() {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in 10..13 {
        campaign(seed, 25, &[3, 4, 8], &mut tally, &mut failures);
    }
    report("sampled", &tally, &failures);
}

/// Long form: `OXIZ_BVEUF_SEED_LO/HI` (default 0..16), `OXIZ_BVEUF_TRIALS`
/// (default 100 per seed).
#[test]
#[ignore = "long-running differential campaign; run explicitly"]
fn bv_euf_campaign() {
    let lo = env_u64("OXIZ_BVEUF_SEED_LO", 0);
    let hi = env_u64("OXIZ_BVEUF_SEED_HI", 16);
    let trials = env_u64("OXIZ_BVEUF_TRIALS", 100) as usize;
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in lo..hi {
        campaign(seed, trials, &[1, 2, 3, 4, 8], &mut tally, &mut failures);
        eprintln!("[campaign] seed {seed}: {tally:?}");
    }
    report(&format!("campaign-{lo}-{hi}"), &tally, &failures);
}
