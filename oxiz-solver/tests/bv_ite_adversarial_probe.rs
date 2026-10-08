//! Adversarial verification of the `#P2b-24` / `#P2b-25` bit-blaster fixes:
//! `ite` selectors of every Boolean shape the fragment now claims, at widths
//! 1/7/8/31/32/33/63/64, nested three deep, feeding every bit-vector
//! operation (multiplication, both divisions and remainders, shifts by
//! amounts at or past the width, `concat`/`extract`/`zero_extend`/
//! `sign_extend`), opaque leaves (an uninterpreted application, an array
//! `select`) inside selectors and operands, and outer Booleans pinned into
//! live selectors before and after the check, under `push`/`pop`, with
//! `:produce-unsat-cores` and repeated `(check-sat)`.
//!
//! Every `sat` model is re-evaluated by an independent `u128` reference
//! evaluator; every `unsat` at width ≤ 8 is confirmed by exhaustive search
//! (opaque leaves are free variables of that search); a wider `unsat` is
//! contradicted whenever random sampling finds a witness.  Every script is run
//! plain and in cargo-formal's named-core form, and every panic is caught.
//!
//! The bounded tests run on every `cargo test`; `adversarial_campaign` is the
//! long form (`OXIZ_ADV_SEED_LO/HI`, `OXIZ_ADV_TRIALS`).  Nothing here is
//! release-specific: the same file is meant to be run under `--release` too,
//! where the debug self-check is compiled out and verdicts must not change.

use oxiz_solver::Context;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The term language, evaluator and generator (a submodule file, not a test
/// target of its own).
#[path = "bv_ite_adversarial_probe/lang.rs"]
mod lang;
use lang::*;

// ---------------------------------------------------------------------------
// Script rendering.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Plain,
    Named,
}

fn logic_of(dag: &Dag) -> &'static str {
    match (dag.has_uf, dag.has_sel) {
        (false, false) => "QF_BV",
        (true, false) => "QF_UFBV",
        (false, true) => "QF_ABV",
        (true, true) => "QF_AUFBV",
    }
}

fn render_prelude(dag: &Dag, variant: Variant, max_conflicts: u64, out: &mut String) {
    let _ = writeln!(out, "(set-logic {})", logic_of(dag));
    if variant == Variant::Named {
        out.push_str("(set-option :produce-unsat-cores true)\n");
    }
    let _ = writeln!(out, "(set-option :max-conflicts {max_conflicts})");
    // A wall-clock budget per check so one wide divider circuit in a debug
    // build cannot stall a whole run; a budget `unknown` is not a failure.
    let _ = writeln!(
        out,
        "(set-option :timeout {})",
        env_u64("OXIZ_ADV_TIMEOUT_MS", 10_000)
    );
    out.push_str("(set-option :produce-models true)\n");
    let w = dag.base_width;
    for i in 0..dag.num_vars {
        let _ = writeln!(out, "(declare-const v{i} (_ BitVec {w}))");
    }
    for i in 0..dag.num_bools {
        let _ = writeln!(out, "(declare-const p{i} Bool)");
    }
    if dag.has_uf {
        let _ = writeln!(out, "(declare-fun f ((_ BitVec {w})) (_ BitVec {w}))");
        let _ = writeln!(out, "(declare-const u0 (_ BitVec {w}))");
        out.push_str("(assert (= u0 (f v0)))\n");
    }
    if dag.has_sel {
        let _ = writeln!(
            out,
            "(declare-const arr (Array (_ BitVec {w}) (_ BitVec {w})))"
        );
        let _ = writeln!(out, "(declare-const w0 (_ BitVec {w}))");
        out.push_str("(assert (= w0 (select arr v0)))\n");
    }
    for id in 0..dag.nodes.len() {
        if dag.named[id] {
            let _ = write!(out, "(define-fun t{id} () {} ", sort_name(dag.width[id]));
            print_body(dag, id, out);
            out.push_str(")\n");
        }
    }
}

fn render_assert(dag: &Dag, a: NodeId, name: Option<&str>, out: &mut String) {
    match name {
        Some(n) => {
            out.push_str("(assert (! ");
            print_ref(dag, a, out);
            let _ = writeln!(out, " :named {n}))");
        }
        None => {
            out.push_str("(assert ");
            print_ref(dag, a, out);
            out.push_str(")\n");
        }
    }
}

fn render_get_value(dag: &Dag, out: &mut String) {
    out.push_str("(get-value (");
    for i in 0..dag.num_vars {
        if i > 0 {
            out.push(' ');
        }
        let _ = write!(out, "v{i}");
    }
    for i in 0..dag.num_bools {
        let _ = write!(out, " p{i}");
    }
    if dag.has_uf {
        out.push_str(" u0");
    }
    if dag.has_sel {
        out.push_str(" w0");
    }
    out.push_str("))\n");
}

fn render(problem: &Problem, variant: Variant, max_conflicts: u64) -> String {
    let dag = &problem.dag;
    let mut out = String::new();
    render_prelude(dag, variant, max_conflicts, &mut out);
    for (k, a) in problem.asserts.iter().enumerate() {
        let name = format!("a{k}");
        render_assert(
            dag,
            *a,
            (variant == Variant::Named).then_some(name.as_str()),
            &mut out,
        );
    }
    out.push_str("(check-sat)\n");
    render_get_value(dag, &mut out);
    if variant == Variant::Named {
        out.push_str("(get-unsat-core)\n");
    }
    out
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

/// Runs `script`, reporting it as slow when it takes ≥ 5 s.
fn run_script_timed(script: &str, tally: &mut Tally, failures: &mut Vec<Failure>) -> Run {
    let started = std::time::Instant::now();
    let run = run_script(script);
    let elapsed = started.elapsed();
    if elapsed.as_secs() >= 5 {
        tally.slow += 1;
        eprintln!("[slow {elapsed:?}] {:?}", run_verdict(&run));
        failures.push(Failure {
            kind: "slow",
            detail: format!("{elapsed:?}"),
            script: script.to_string(),
        });
    }
    run
}

/// The first verdict line of a run, for logging.
fn run_verdict(run: &Run) -> String {
    match run {
        Run::Lines(lines) => lines.first().cloned().unwrap_or_default(),
        Run::Error(e) => format!("error: {e}"),
        Run::Panic(p) => format!("panic: {p}"),
    }
}

fn run_script(script: &str) -> Run {
    match catch_unwind(AssertUnwindSafe(|| {
        let mut ctx = Context::new();
        ctx.execute_script(script)
    })) {
        Ok(Ok(lines)) => Run::Lines(lines),
        Ok(Err(e)) => Run::Error(e.to_string()),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Run::Panic(message)
        }
    }
}

fn verdict_of(line: &str) -> Option<&'static str> {
    match line.trim() {
        "sat" => Some("sat"),
        "unsat" => Some("unsat"),
        "unknown" => Some("unknown"),
        _ => None,
    }
}

/// The value printed for `name` in a `(get-value …)` answer.
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

fn printed_bool(block: &str, name: &str) -> Option<bool> {
    let key = format!("({name} ");
    let after = block.find(&key).map(|at| &block[at + key.len()..])?;
    let after = after.trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// The names in a `(get-unsat-core)` answer, or `None` for anything else.
fn printed_core(block: &str) -> Option<Vec<String>> {
    let line = block.trim();
    if !line.starts_with('(') || line.starts_with("(error") || line.starts_with("((") {
        return None;
    }
    let inner = line.trim_start_matches('(').trim_end_matches(')');
    if !inner.split_whitespace().all(|w| w.starts_with('a')) {
        return None;
    }
    Some(inner.split_whitespace().map(str::to_string).collect())
}

/// Reads the model printed after a `sat`: `None` when a value is missing.
fn read_model(dag: &Dag, block: &str) -> Option<(Vec<u128>, Vec<bool>, u128, u128)> {
    let mut vars = Vec::new();
    for i in 0..dag.num_vars {
        vars.push(printed_bv(block, &format!("v{i}"))?);
    }
    let mut bools = Vec::new();
    for i in 0..dag.num_bools {
        bools.push(printed_bool(block, &format!("p{i}"))?);
    }
    let uf = if dag.has_uf {
        printed_bv(block, "u0")?
    } else {
        0
    };
    let sel = if dag.has_sel {
        printed_bv(block, "w0")?
    } else {
        0
    };
    Some((vars, bools, uf, sel))
}

/// Exhaustive truth at width ≤ 8: at most two bit-vector variables plus two
/// Booleans, with the opaque leaves as extra free variables (allowed only
/// with a single bit-vector variable so the search stays ≤ 2^24).
fn exhaustive(dag: &Dag, asserts: &[NodeId]) -> Option<bool> {
    let w = dag.base_width;
    if w > 8 || dag.num_vars > 2 || dag.num_bools > 2 {
        return None;
    }
    // At most one opaque leaf, and then only one bit-vector variable, so the
    // search stays within 2^18 assignments.
    let opaque_count = usize::from(dag.has_uf) + usize::from(dag.has_sel);
    if opaque_count > 1 || (opaque_count == 1 && dag.num_vars > 1) {
        return None;
    }
    let var_space = 1u64 << (u64::from(w) * dag.num_vars as u64);
    let bool_space = 1u64 << dag.num_bools;
    let uf_space = if dag.has_uf { 1u64 << w } else { 1 };
    let sel_space = if dag.has_sel { 1u64 << w } else { 1 };
    for bits in 0..bool_space {
        let bools: Vec<bool> = (0..dag.num_bools).map(|i| (bits >> i) & 1 == 1).collect();
        for idx in 0..var_space {
            let vars: Vec<u128> = (0..dag.num_vars)
                .map(|i| u128::from((idx >> (u64::from(w) * i as u64)) & (mask(w) as u64)))
                .collect();
            for uf in 0..uf_space {
                for sel in 0..sel_space {
                    let asg = Assignment {
                        vars: &vars,
                        bools: &bools,
                        uf: u128::from(uf),
                        sel: u128::from(sel),
                    };
                    if all_hold(dag, asserts, &asg) {
                        return Some(true);
                    }
                }
            }
        }
    }
    Some(false)
}

/// Random witness search (any width); proves `sat` only.
fn sampled_witness(rng: &mut Rng, dag: &Dag, asserts: &[NodeId], samples: usize) -> bool {
    let m = mask(dag.base_width);
    for _ in 0..samples {
        let vars: Vec<u128> = (0..dag.num_vars)
            .map(|_| match rng.below(4) {
                0 => u128::from(rng.below(8)) & m,
                1 => m - (u128::from(rng.below(8)) & m),
                _ => u128::from(rng.next_u64()) & m,
            })
            .collect();
        let bools: Vec<bool> = (0..dag.num_bools).map(|_| rng.chance(1, 2)).collect();
        let asg = Assignment {
            vars: &vars,
            bools: &bools,
            uf: u128::from(rng.next_u64()) & m,
            sel: u128::from(rng.next_u64()) & m,
        };
        if all_hold(dag, asserts, &asg) {
            return true;
        }
    }
    false
}

#[derive(Default, Debug)]
struct Tally {
    scripts: usize,
    checks: usize,
    sat: usize,
    unsat: usize,
    unknown: usize,
    errors: usize,
    panics: usize,
    wrong_sat: usize,
    wrong_unsat: usize,
    bad_core: usize,
    missing_model: usize,
    /// Scripts that took ≥ 5 s: reported (and written out), not failures.
    slow: usize,
}

struct Failure {
    kind: &'static str,
    detail: String,
    script: String,
}

impl Tally {
    fn has_failures(&self) -> bool {
        self.errors
            + self.panics
            + self.wrong_sat
            + self.wrong_unsat
            + self.bad_core
            + self.missing_model
            > 0
    }
}

/// Scores one `check-sat` answer against the truth of `asserts`.
#[allow(clippy::too_many_arguments)] // one call site per driver; a struct would only rename the fields
fn score_check(
    dag: &Dag,
    asserts: &[NodeId],
    verdict: &str,
    value_block: Option<&str>,
    truth: Option<bool>,
    script: &str,
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) {
    tally.checks += 1;
    match verdict {
        "sat" => {
            tally.sat += 1;
            let Some(block) = value_block else {
                tally.missing_model += 1;
                failures.push(Failure {
                    kind: "sat-without-get-value",
                    detail: String::new(),
                    script: script.to_string(),
                });
                return;
            };
            let Some((vars, bools, uf, sel)) = read_model(dag, block) else {
                tally.missing_model += 1;
                failures.push(Failure {
                    kind: "sat-without-model-value",
                    detail: block.to_string(),
                    script: script.to_string(),
                });
                return;
            };
            let asg = Assignment {
                vars: &vars,
                bools: &bools,
                uf,
                sel,
            };
            if !all_hold(dag, asserts, &asg) {
                tally.wrong_sat += 1;
                failures.push(Failure {
                    kind: "wrong-sat",
                    detail: format!(
                        "model {vars:x?} {bools:?} u0={uf:x} w0={sel:x} violates an assertion"
                    ),
                    script: script.to_string(),
                });
            } else if truth == Some(false) {
                tally.wrong_sat += 1;
                failures.push(Failure {
                    kind: "wrong-sat-vs-oracle",
                    detail: "exhaustive search says unsat, yet the model evaluates true"
                        .to_string(),
                    script: script.to_string(),
                });
            }
        }
        "unsat" => {
            tally.unsat += 1;
            if truth == Some(true) {
                tally.wrong_unsat += 1;
                failures.push(Failure {
                    kind: "wrong-unsat",
                    detail: "a witness exists".to_string(),
                    script: script.to_string(),
                });
            }
        }
        _ => tally.unknown += 1,
    }
}

fn truth_of(rng: &mut Rng, dag: &Dag, asserts: &[NodeId]) -> Option<bool> {
    match exhaustive(dag, asserts) {
        Some(t) => Some(t),
        None => sampled_witness(rng, dag, asserts, 3000).then_some(true),
    }
}

/// Runs one single-check script and scores it.
fn run_single(
    rng: &mut Rng,
    problem: &Problem,
    variant: Variant,
    max_conflicts: u64,
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) {
    let dag = &problem.dag;
    let script = render(problem, variant, max_conflicts);
    let truth = truth_of(rng, dag, &problem.asserts);
    tally.scripts += 1;
    match run_script_timed(&script, tally, failures) {
        Run::Panic(msg) => {
            tally.panics += 1;
            failures.push(Failure {
                kind: "panic",
                detail: msg,
                script,
            });
        }
        Run::Error(msg) => {
            tally.errors += 1;
            failures.push(Failure {
                kind: "error",
                detail: msg,
                script,
            });
        }
        Run::Lines(lines) => {
            let Some(verdict) = lines.first().and_then(|l| verdict_of(l)) else {
                tally.errors += 1;
                failures.push(Failure {
                    kind: "no-verdict",
                    detail: lines.join(" | "),
                    script,
                });
                return;
            };
            score_check(
                dag,
                &problem.asserts,
                verdict,
                lines.get(1).map(String::as_str),
                truth,
                &script,
                tally,
                failures,
            );
            if variant == Variant::Named && verdict == "unsat" {
                let names: Vec<String> = (0..problem.asserts.len())
                    .map(|k| format!("a{k}"))
                    .collect();
                match lines.get(2).and_then(|b| printed_core(b)) {
                    Some(core) if !core.is_empty() && core.iter().all(|n| names.contains(n)) => {}
                    other => {
                        tally.bad_core += 1;
                        failures.push(Failure {
                            kind: "bad-core",
                            detail: format!("{other:?} (names {names:?}): {}", lines.join(" | ")),
                            script,
                        });
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-check scripts: pins under push/pop, repeated check-sat, named cores.
// ---------------------------------------------------------------------------

/// One step of an incremental script.
enum Step {
    Push,
    Pop,
    Assert(NodeId),
    Check,
}

/// Builds a problem whose Booleans are used as selectors and an incremental
/// command sequence that pins them before and after checks, inside and
/// outside scopes.
fn gen_incremental(rng: &mut Rng, width: u32) -> (Problem, Vec<Step>) {
    let num_bools = 1 + rng.below(2) as usize;
    let cfg = GenConfig {
        base_width: width,
        num_vars: 2,
        num_bools,
        growth: 2 + rng.below(4) as usize,
        asserts: 0,
        opaque: false,
    };
    let mut problem = gen_problem(rng, &cfg);
    let dag = &mut problem.dag;
    // Make sure every Boolean is a selector somewhere.
    for i in 0..num_bools {
        let p = dag.num_vars + i;
        let cond = match rng.below(4) {
            0 => p,
            1 => {
                let other = pick_bool(rng, dag);
                dag.push(Node::BXor(p, other), 0, true)
            }
            2 => {
                let other = pick_bool(rng, dag);
                dag.push(Node::BImplies(other, p), 0, true)
            }
            _ => {
                let other = pick_bool(rng, dag);
                dag.push(Node::BEq(p, other), 0, true)
            }
        };
        let then = pick_bv(rng, dag, width);
        let els = pick_bv(rng, dag, width);
        dag.push(Node::Ite { cond, then, els }, width, true);
    }
    let mut steps = Vec::new();
    // Base assertions: a comparison over the grown terms.
    let base = gen_assert(rng, dag);
    steps.push(Step::Assert(base));
    if rng.chance(1, 2) {
        let extra = gen_assert(rng, dag);
        steps.push(Step::Assert(extra));
    }
    let mut depth = 0u32;
    let n_steps = 4 + rng.below(6) as usize;
    for _ in 0..n_steps {
        match rng.below(6) {
            0 if depth < 3 => {
                steps.push(Step::Push);
                depth += 1;
            }
            1 if depth > 0 => {
                steps.push(Step::Pop);
                depth -= 1;
            }
            2 | 3 => {
                // Pin a Boolean, or a variable to a constant, or a comparison.
                let a = match rng.below(3) {
                    0 => {
                        let p = dag.num_vars + rng.below(num_bools as u64) as usize;
                        if rng.chance(1, 2) {
                            p
                        } else {
                            dag.push(Node::BNot(p), 0, true)
                        }
                    }
                    _ => gen_assert(rng, dag),
                };
                steps.push(Step::Assert(a));
            }
            _ => steps.push(Step::Check),
        }
    }
    steps.push(Step::Check);
    // Pin after the last check, then check again: the shape that went
    // unexamined before `bv_pin_pending`.
    let p = dag.num_vars + rng.below(num_bools as u64) as usize;
    let pin = if rng.chance(1, 2) {
        p
    } else {
        dag.push(Node::BNot(p), 0, true)
    };
    steps.push(Step::Assert(pin));
    steps.push(Step::Check);
    (problem, steps)
}

fn render_incremental(problem: &Problem, steps: &[Step], variant: Variant) -> String {
    let dag = &problem.dag;
    let mut out = String::new();
    render_prelude(dag, variant, 50_000, &mut out);
    let mut next_name = 0usize;
    for step in steps {
        match step {
            Step::Push => out.push_str("(push 1)\n"),
            Step::Pop => out.push_str("(pop 1)\n"),
            Step::Assert(a) => {
                let name = format!("a{next_name}");
                next_name += 1;
                render_assert(
                    dag,
                    *a,
                    (variant == Variant::Named).then_some(name.as_str()),
                    &mut out,
                );
            }
            Step::Check => {
                out.push_str("(check-sat)\n");
                render_get_value(dag, &mut out);
                if variant == Variant::Named {
                    out.push_str("(get-unsat-core)\n");
                }
            }
        }
    }
    out
}

/// Runs one incremental script: every `check-sat` is scored against the
/// assertions active at that point.
fn run_incremental(
    rng: &mut Rng,
    problem: &Problem,
    steps: &[Step],
    variant: Variant,
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) {
    let dag = &problem.dag;
    let script = render_incremental(problem, steps, variant);
    tally.scripts += 1;
    let lines = match run_script_timed(&script, tally, failures) {
        Run::Panic(msg) => {
            tally.panics += 1;
            failures.push(Failure {
                kind: "panic",
                detail: msg,
                script,
            });
            return;
        }
        Run::Error(msg) => {
            tally.errors += 1;
            failures.push(Failure {
                kind: "error",
                detail: msg,
                script,
            });
            return;
        }
        Run::Lines(lines) => lines,
    };
    // Active assertions as a scope stack, with their names.
    let mut scopes: Vec<Vec<(NodeId, String)>> = vec![Vec::new()];
    let mut next_name = 0usize;
    let mut cursor = 0usize;
    for step in steps {
        match step {
            Step::Push => scopes.push(Vec::new()),
            Step::Pop => {
                scopes.pop();
            }
            Step::Assert(a) => {
                let name = format!("a{next_name}");
                next_name += 1;
                if let Some(top) = scopes.last_mut() {
                    top.push((*a, name));
                }
            }
            Step::Check => {
                let active: Vec<NodeId> = scopes.iter().flatten().map(|(a, _)| *a).collect();
                let names: Vec<String> = scopes.iter().flatten().map(|(_, n)| n.clone()).collect();
                let Some(verdict) = lines.get(cursor).and_then(|l| verdict_of(l)) else {
                    tally.errors += 1;
                    failures.push(Failure {
                        kind: "no-verdict",
                        detail: format!("at response {cursor}: {}", lines.join(" | ")),
                        script,
                    });
                    return;
                };
                let value_block = lines.get(cursor + 1).map(String::as_str);
                let truth = truth_of(rng, dag, &active);
                score_check(
                    dag,
                    &active,
                    verdict,
                    value_block,
                    truth,
                    &script,
                    tally,
                    failures,
                );
                cursor += 2;
                if variant == Variant::Named {
                    if verdict == "unsat" {
                        match lines.get(cursor).and_then(|b| printed_core(b)) {
                            Some(core)
                                if !core.is_empty() && core.iter().all(|n| names.contains(n)) => {}
                            other => {
                                tally.bad_core += 1;
                                failures.push(Failure {
                                    kind: "bad-core",
                                    detail: format!(
                                        "{other:?} (active names {names:?}): {}",
                                        lines.join(" | ")
                                    ),
                                    script: script.clone(),
                                });
                            }
                        }
                    }
                    cursor += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The selector zoo: every Boolean shape at every width, both verdicts.
// ---------------------------------------------------------------------------

/// Builds selector `shape` over `a`, `b` (bit-vectors) and `p`, `q`
/// (Booleans) inside `dag`.
fn zoo_selector(dag: &mut Dag, shape: usize, a: NodeId, b: NodeId, p: NodeId, q: NodeId) -> NodeId {
    let cmp = |dag: &mut Dag, op: CmpOp| dag.push(Node::Cmp { op, lhs: a, rhs: b }, 0, true);
    match shape {
        0 => p,
        1 => cmp(dag, CmpOp::Eq),
        2 => dag.push(Node::Distinct(vec![a, b]), 0, true),
        3 => cmp(dag, CmpOp::Ult),
        4 => cmp(dag, CmpOp::Ule),
        5 => cmp(dag, CmpOp::Slt),
        6 => cmp(dag, CmpOp::Sle),
        7 => cmp(dag, CmpOp::Ugt),
        8 => cmp(dag, CmpOp::Uge),
        9 => cmp(dag, CmpOp::Sgt),
        10 => cmp(dag, CmpOp::Sge),
        11 => {
            let c = cmp(dag, CmpOp::Ult);
            dag.push(Node::BNot(c), 0, true)
        }
        12 => {
            let c = cmp(dag, CmpOp::Ule);
            let d = dag.push(Node::Distinct(vec![a, b]), 0, true);
            dag.push(Node::BAnd(vec![c, d, p]), 0, true)
        }
        13 => {
            let c = cmp(dag, CmpOp::Slt);
            dag.push(Node::BOr(vec![c, q]), 0, true)
        }
        14 => {
            let c = cmp(dag, CmpOp::Ult);
            dag.push(Node::BXor(c, p), 0, true)
        }
        15 => {
            let c = cmp(dag, CmpOp::Sle);
            dag.push(Node::BImplies(p, c), 0, true)
        }
        16 => {
            let c = cmp(dag, CmpOp::Eq);
            let d = cmp(dag, CmpOp::Ugt);
            dag.push(Node::BIte(p, c, d), 0, true)
        }
        17 => {
            let c = cmp(dag, CmpOp::Uge);
            dag.push(Node::BEq(c, q), 0, true)
        }
        18 => {
            let c = cmp(dag, CmpOp::Sge);
            let e = dag.push(Node::BEq(c, p), 0, true);
            dag.push(Node::BEq(e, q), 0, true)
        }
        19 => dag.push(Node::BDistinct(vec![p, q]), 0, true),
        20 => {
            let c = cmp(dag, CmpOp::Ult);
            let d = cmp(dag, CmpOp::Eq);
            dag.push(Node::BDistinct(vec![c, d, p]), 0, true)
        }
        21 => dag.push(Node::BXor(p, q), 0, true),
        22 => dag.push(Node::BImplies(p, q), 0, true),
        23 => {
            let x = dag.push(Node::BIte(p, q, p), 0, true);
            dag.push(Node::BEq(x, q), 0, true)
        }
        _ => {
            let c = dag.push(Node::Distinct(vec![a, b]), 0, true);
            let d = cmp(dag, CmpOp::Slt);
            let n = dag.push(Node::BNot(d), 0, true);
            let x = dag.push(Node::BXor(c, n), 0, true);
            dag.push(Node::BImplies(x, q), 0, true)
        }
    }
}

const ZOO_SHAPES: usize = 25;

/// `x = (ite sel K1 K2)` with `a`, `b`, `p`, `q` pinned so `sel` is decided;
/// `x = K_wrong` must be `unsat`, `x = K_right` `sat`.  Also a three-deep
/// nest of the same shape.  Returns the scripts with their expected verdicts.
fn zoo_scripts(rng: &mut Rng, width: u32, shape: usize) -> Vec<(String, bool, Problem)> {
    let mut out = Vec::new();
    for nested in [false, true] {
        for want_sat in [false, true] {
            let mut dag = Dag::new(width, 3, 2);
            let (a, b, x) = (0, 1, 2);
            let (p, q) = (3, 4);
            let av = gen_const(rng, width);
            let bv = gen_const(rng, width);
            let pv = rng.chance(1, 2);
            let qv = rng.chance(1, 2);
            let ka = dag.konst(av, width);
            let kb = dag.konst(bv, width);
            let k1 = dag.konst(1, width);
            let k2 = dag.konst(2 & mask(width), width);
            let sel = zoo_selector(&mut dag, shape, a, b, p, q);
            let body = if nested {
                let sel2 = zoo_selector(&mut dag, (shape + 7) % ZOO_SHAPES, b, a, q, p);
                let sel3 = zoo_selector(&mut dag, (shape + 13) % ZOO_SHAPES, a, b, p, q);
                let inner = dag.push(
                    Node::Ite {
                        cond: sel3,
                        then: k1,
                        els: k2,
                    },
                    width,
                    true,
                );
                let mid = dag.push(
                    Node::Ite {
                        cond: sel2,
                        then: inner,
                        els: k2,
                    },
                    width,
                    true,
                );
                dag.push(
                    Node::Ite {
                        cond: sel,
                        then: mid,
                        els: k1,
                    },
                    width,
                    true,
                )
            } else {
                dag.push(
                    Node::Ite {
                        cond: sel,
                        then: k1,
                        els: k2,
                    },
                    width,
                    true,
                )
            };
            let pin_a = dag.push(
                Node::Cmp {
                    op: CmpOp::Eq,
                    lhs: a,
                    rhs: ka,
                },
                0,
                true,
            );
            let pin_b = dag.push(
                Node::Cmp {
                    op: CmpOp::Eq,
                    lhs: b,
                    rhs: kb,
                },
                0,
                true,
            );
            let pin_p = if pv {
                p
            } else {
                dag.push(Node::BNot(p), 0, true)
            };
            let pin_q = if qv {
                q
            } else {
                dag.push(Node::BNot(q), 0, true)
            };
            let def_x = dag.push(
                Node::Cmp {
                    op: CmpOp::Eq,
                    lhs: x,
                    rhs: body,
                },
                0,
                true,
            );
            // Decide the body's value under the pins.
            let vars = [av, bv, 0];
            let bools = [pv, qv];
            let asg = Assignment {
                vars: &vars,
                bools: &bools,
                uf: 0,
                sel: 0,
            };
            let mut memo = vec![None; dag.nodes.len()];
            let Val::Bv(value) = eval(&dag, body, &asg, &mut memo) else {
                continue;
            };
            let target = if want_sat {
                value
            } else {
                (value ^ 1) & mask(width)
            };
            let kt = dag.konst(target, width);
            let x_is = dag.push(
                Node::Cmp {
                    op: CmpOp::Eq,
                    lhs: x,
                    rhs: kt,
                },
                0,
                true,
            );
            // Pins before and after the definition, so the selector's value
            // reaches the circuit both before the node exists and after.
            let asserts = vec![pin_a, def_x, pin_p, x_is, pin_b, pin_q];
            let problem = Problem { dag, asserts };
            let script = render(&problem, Variant::Plain, 50_000);
            out.push((script, want_sat, problem));
        }
    }
    out
}

fn run_zoo(rng: &mut Rng, widths: &[u32], tally: &mut Tally, failures: &mut Vec<Failure>) {
    for &width in widths {
        for shape in 0..ZOO_SHAPES {
            for (script, want_sat, problem) in zoo_scripts(rng, width, shape) {
                tally.scripts += 1;
                match run_script_timed(&script, tally, failures) {
                    Run::Panic(msg) => {
                        tally.panics += 1;
                        failures.push(Failure {
                            kind: "panic",
                            detail: msg,
                            script,
                        });
                    }
                    Run::Error(msg) => {
                        tally.errors += 1;
                        failures.push(Failure {
                            kind: "error",
                            detail: msg,
                            script,
                        });
                    }
                    Run::Lines(lines) => {
                        let verdict = lines.first().and_then(|l| verdict_of(l)).unwrap_or("none");
                        score_check(
                            &problem.dag,
                            &problem.asserts,
                            verdict,
                            lines.get(1).map(String::as_str),
                            Some(want_sat),
                            &script,
                            tally,
                            failures,
                        );
                        // The zoo is fully decided: an `unknown` is a lost
                        // verdict on a formula the fragment claims to model.
                        if verdict != if want_sat { "sat" } else { "unsat" } {
                            failures.push(Failure {
                                kind: "zoo-undecided",
                                detail: format!(
                                    "width {width} shape {shape}: expected {}, got {verdict}",
                                    if want_sat { "sat" } else { "unsat" }
                                ),
                                script,
                            });
                            tally.errors += 1;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Campaign driver.
// ---------------------------------------------------------------------------

fn write_failures(tag: &str, failures: &[Failure]) {
    let dir = std::env::temp_dir().join("oxiz-bv-ite-adversarial");
    let _ = std::fs::create_dir_all(&dir);
    for (i, f) in failures.iter().take(100).enumerate() {
        let mut text = String::new();
        let _ = writeln!(text, "; kind: {}", f.kind);
        for line in f.detail.lines() {
            let _ = writeln!(text, "; {line}");
        }
        text.push_str(&f.script);
        let _ = std::fs::write(dir.join(format!("{tag}-{i:03}-{}.smt2", f.kind)), text);
    }
}

fn report(tag: &str, tally: &Tally, failures: &[Failure]) {
    eprintln!("[{tag}] {tally:?}");
    for f in failures {
        eprintln!("[{tag}] {}: {}", f.kind, f.detail);
    }
    write_failures(tag, failures);
}

fn random_trials(
    seed: u64,
    trials: usize,
    widths: &[u32],
    opaque_share: u64,
    max_conflicts: u64,
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) {
    let mut rng = Rng::new(seed);
    for _ in 0..trials {
        let width = *rng.pick(widths);
        let opaque = rng.chance(opaque_share, 10);
        let cfg = GenConfig {
            base_width: width,
            num_vars: if opaque { 1 } else { 2 + rng.below(2) as usize },
            num_bools: rng.below(3) as usize,
            growth: 3 + rng.below(5) as usize,
            asserts: 1 + rng.below(3) as usize,
            opaque,
        };
        let problem = gen_problem(&mut rng, &cfg);
        run_single(
            &mut rng,
            &problem,
            Variant::Plain,
            max_conflicts,
            tally,
            failures,
        );
        run_single(
            &mut rng,
            &problem,
            Variant::Named,
            max_conflicts,
            tally,
            failures,
        );
    }
}

fn incremental_trials(
    seed: u64,
    trials: usize,
    widths: &[u32],
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) {
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A);
    for _ in 0..trials {
        let width = *rng.pick(widths);
        let (problem, steps) = gen_incremental(&mut rng, width);
        run_incremental(&mut rng, &problem, &steps, Variant::Plain, tally, failures);
        run_incremental(&mut rng, &problem, &steps, Variant::Named, tally, failures);
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn assert_clean(tag: &str, tally: &Tally, failures: &[Failure]) {
    report(tag, tally, failures);
    let real: Vec<&Failure> = failures.iter().filter(|f| f.kind != "slow").collect();
    assert!(
        !tally.has_failures(),
        "[{tag}] {} failures: first [{}] {}\n{}",
        real.len(),
        real.first().map(|f| f.kind).unwrap_or("-"),
        real.first().map(|f| f.detail.as_str()).unwrap_or(""),
        real.first().map(|f| f.script.as_str()).unwrap_or("")
    );
}

/// Every selector shape at every width, plain and three-deep, both verdicts:
/// every one must be *decided* (an `unknown` fails too).
#[test]
fn adversarial_selector_zoo() {
    let mut rng = Rng::new(20_260_915);
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    run_zoo(&mut rng, &WIDTHS, &mut tally, &mut failures);
    assert_clean("zoo", &tally, &failures);
}

/// Bounded random slice: every width, opaque leaves in one script of five,
/// plain and named forms.
#[test]
fn adversarial_bounded_random() {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in 100..102 {
        random_trials(seed, 6, &WIDTHS, 2, 10_000, &mut tally, &mut failures);
    }
    assert_clean("bounded-random", &tally, &failures);
}

/// Bounded incremental slice: pins before/after checks, under push/pop,
/// named cores, repeated `check-sat`.
#[test]
fn adversarial_bounded_incremental() {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in 200..202 {
        incremental_trials(seed, 4, &WIDTHS, &mut tally, &mut failures);
    }
    assert_clean("bounded-incremental", &tally, &failures);
}

/// Long campaign: `OXIZ_ADV_SEED_LO/HI` (default 0..8), `OXIZ_ADV_TRIALS`
/// (random trials per seed, default 100), `OXIZ_ADV_INC_TRIALS` (incremental
/// trials per seed, default 30), `OXIZ_ADV_MAX_CONFLICTS` (default 20000).
#[test]
#[ignore = "long-running adversarial campaign; run explicitly"]
fn adversarial_campaign() {
    let lo = env_u64("OXIZ_ADV_SEED_LO", 0);
    let hi = env_u64("OXIZ_ADV_SEED_HI", 8);
    let trials = env_u64("OXIZ_ADV_TRIALS", 100) as usize;
    let inc_trials = env_u64("OXIZ_ADV_INC_TRIALS", 30) as usize;
    let max_conflicts = env_u64("OXIZ_ADV_MAX_CONFLICTS", 20_000);
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for seed in lo..hi {
        random_trials(
            seed,
            trials,
            &WIDTHS,
            2,
            max_conflicts,
            &mut tally,
            &mut failures,
        );
        incremental_trials(seed, inc_trials, &WIDTHS, &mut tally, &mut failures);
        eprintln!("[campaign] seed {seed} done: {tally:?}");
    }
    assert_clean(&format!("campaign-{lo}-{hi}"), &tally, &failures);
}

// ---------------------------------------------------------------------------
// Three holes this probe found on the tree carrying only `#P2b-24`/`#P2b-25`,
// each pinned here first as "still open" and inverted the day it closed
// (`#P2b-27`, `#P2b-28`, `#P2b-29`), a fourth the review of those three found
// beside them (`#P2b-32`), and a fifth the review of *that* one found
// (`#P2b-33`).  They stay as regression guards.
// ---------------------------------------------------------------------------

/// Congruence reaches an opaque leaf *under* a bit-vector operation
/// (`#P2b-29`): `(= a b) ∧ (distinct (bvadd (f a) #x01) (bvadd (f b) #x01))`
/// is unsatisfiable.  Before the EUF → bit-vector equality sharing in
/// `TheoryManager::combine_bv_with_euf` the encoder abstracted `(f a)` and
/// `(f b)` into two unrelated free bit-vectors, the EUF layer never told the
/// circuit that `f(a) = f(b)`, and the model gate could not evaluate an
/// opaque leaf whose value the model does not print — this tree, 0.3.3 and
/// the tree before `#P2b-24` all answered `sat`.  (The same pair *without*
/// the `bvadd` was always refuted by EUF alone, which is still checked.)
#[test]
fn p2b29_congruence_under_bv_operation_is_refuted() {
    let script = "\
(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (= a b))
(assert (distinct (bvadd (f a) #x01) (bvadd (f b) #x01)))
(check-sat)
";
    let verdict = match run_script(script) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        Run::Error(_) => "error",
        Run::Panic(_) => "panic",
    };
    assert_eq!(
        verdict, "unsat",
        "the wrong `sat` of the pre-#P2b-29 tree is back (now {verdict})"
    );
    let leaf_only = "\
(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (= a b))
(assert (distinct (f a) (f b)))
(check-sat)
";
    let leaf_verdict = match run_script(leaf_only) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        _ => "error",
    };
    assert_eq!(leaf_verdict, "unsat", "EUF refutes the bare pair");
}

/// A Boolean that occurs *only* as an `ite` selector has no outer clause,
/// so the SAT core never assigns it; `build_model` used to record nothing
/// for it and `(get-value (p))` / `(get-model)` printed a **default**
/// `false` — even when the circuit had to choose `p = true`.  `x = (ite p 1
/// 2) ∧ x = 1` is satisfiable only with `p = true`, and this tree, 0.3.3 and
/// the tree before `#P2b-24` all answered `sat` with `p = false`: a
/// published model that violates the first assertion, which the model gate
/// could not notice because the same missing entry made every assertion
/// above `p` `Undetermined` (the root cause of `#P2b-27`).  `build_model`
/// now publishes the value the embedded circuit chose
/// (`BvSolver::bool_value`), and the gate refuses a `sat` whose assertions
/// it cannot evaluate for want of a Boolean's entry.
#[test]
fn p2b27_selector_only_boolean_is_published_from_the_circuit() {
    let script = "\
(set-logic QF_BV)
(declare-const p Bool)
(declare-const x (_ BitVec 8))
(assert (= x (ite p #x01 #x02)))
(assert (= x #x01))
(check-sat)
(get-value (p x))
";
    let lines = match run_script(script) {
        Run::Lines(lines) => lines,
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(lines.first().map(String::as_str), Some("sat"), "{lines:?}");
    let block = lines.get(1).map(String::as_str).unwrap_or("");
    assert_eq!(printed_bv(block, "x"), Some(1), "{lines:?}");
    assert_eq!(
        printed_bool(block, "p"),
        Some(true),
        "the published model must honour the circuit: {lines:?}"
    );
}

/// `(bvult #x7fffffffffffffff v)` is satisfiable and is now decided
/// (`#P2b-28`).  A 64-bit unsigned comparison used to be handed to the
/// arithmetic solver as well, as bounded-integer arithmetic over `i64`
/// rationals: there `assert_lt` rewrote `x < k` into `x <= k − 1`, the
/// constant `k` was `−i64::MAX`, `k − 1` was `i64::MIN`, and `assert_le`'s
/// `-rhs` overflowed — a debug build panicked ("attempt to negate with
/// overflow"), a release build wrapped and answered `unknown`.  Found by
/// the fresh-seed campaign of `bv_ite_selfcheck_fuzz` on the tree carrying
/// only `#P2b-24`/`#P2b-25`; 0.3.3 behaved the same.  The mirror is gone:
/// the bit-blasted circuit decides every comparison exactly.
#[test]
fn p2b28_i64_max_bound_in_a_wide_comparison_is_decided() {
    let script = "\
(set-logic QF_BV)
(declare-const v (_ BitVec 64))
(assert (bvult #x7fffffffffffffff v))
(check-sat)
";
    let verdict = match run_script(script) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        Run::Error(_) => "error",
        Run::Panic(msg) => panic!("the #P2b-28 panic is back: {msg}"),
    };
    assert_eq!(verdict, "sat", "a satisfiable one-liner lost its verdict");
}

/// Read-over-write reaches a `select` *under* a bit-vector operation
/// (`#P2b-32`): `(distinct (bvadd (select (store arr i #x05) i) #x01) #x06)`
/// is unsatisfiable.  The array-axiom instantiator's structural walk
/// descended through a hand-written child list naming only the Boolean
/// connectives, `ite` and `Apply`, so a read nested under `bvadd` was never
/// collected, no read-over-write instance was asserted for it, the read
/// stayed a free bit-vector in the circuit, and the model gate — with no arm
/// for `select` — evaluated the assertion `Undetermined` and vouched for the
/// free bits: this tree, 0.3.3 and the tree before `#P2b-24` all answered
/// `sat`.  The exchange of `#P2b-29` cannot see it, because the equality
/// comes from an array axiom, not from congruence.  (The same read as a
/// *direct* atom operand was always refuted, which is still checked.)
#[test]
fn p2b32_read_over_write_under_a_bv_operation_is_refuted() {
    let script = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (bvadd (select (store arr i #x05) i) #x01) #x06))
(check-sat)
";
    let verdict = match run_script(script) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        Run::Error(_) => "error",
        Run::Panic(_) => "panic",
    };
    assert_eq!(
        verdict, "unsat",
        "the wrong `sat` of the pre-#P2b-32 tree is back (now {verdict})"
    );
    let direct = "\
(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (select (store arr i #x05) i) #x05))
(check-sat)
";
    let direct_verdict = match run_script(direct) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        _ => "error",
    };
    assert_eq!(
        direct_verdict, "unsat",
        "the direct read was always refuted"
    );
}

/// Read-over-write reaches a `select` that occurs only as the ARGUMENT of an
/// uninterpreted application (`#P2b-33`):
/// `(distinct (f (select (store arr i #x05) i)) (f #x05))` is unsatisfiable.
/// The guard on the whole lazy array-refinement loop
/// (`Solver::has_array_ops`) was raised by `track_theory_vars`, which
/// deliberately does not descend into an application's arguments, so a formula
/// whose only read sits there never ran `instantiate_array_axioms` at all: no
/// read-over-write instance existed, the read stayed a free bit-vector, and the
/// model gate — which sees the same free leaf — vouched for it.  This tree,
/// 0.3.3 and every tree before `#P2b-32` all answered `sat`; `#P2b-32` itself
/// did not touch it, because the walk it fixed was never reached.  The guard is
/// now the instantiator's own walk (`Solver::mark_array_ops`).  (The same read
/// under a bit-vector operator is `#P2b-32`, checked above; as a *direct* atom
/// operand it was always refuted, which is still checked here.)
#[test]
fn p2b33_read_over_write_under_an_application_is_refuted() {
    let script = "\
(set-logic QF_AUFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (f (select (store arr i #x05) i)) (f #x05)))
(check-sat)
";
    let verdict = match run_script(script) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        Run::Error(_) => "error",
        Run::Panic(_) => "panic",
    };
    assert_eq!(
        verdict, "unsat",
        "the wrong `sat` of the pre-#P2b-33 tree is back (now {verdict})"
    );
    let direct = "\
(set-logic QF_AUFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const arr (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const i (_ BitVec 8))
(assert (distinct (select (store arr i #x05) i) #x05))
(check-sat)
";
    let direct_verdict = match run_script(direct) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        _ => "error",
    };
    assert_eq!(
        direct_verdict, "unsat",
        "the direct read was always refuted"
    );
}

/// A read of the SMT-LIB array constant is its default (`#P2b-36`):
/// `(= (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00) #x00) #x05)`
/// is unsatisfiable.  The array constant has no term kind of its own — the
/// parser turns the qualified identifier `(as const (Array D R))` into an
/// ordinary uninterpreted `Apply` — so every theory saw an opaque array, the
/// read was a free bit-vector, and the model gate saw the same free leaf.
/// 0.3.3 and every tree before this one answered `sat`, as they did for the
/// `Int` spelling, for the read under `bvadd`, and for the read wrapped in an
/// uninterpreted function.  `build_const_array_reads` supplies the axiom.
///
/// The second script is the control the axiom must not break: `|(as const)|`
/// is a legal quoted SMT-LIB symbol that interns to exactly the name the
/// parser gives an array constant, so a script declaring it gets an
/// uninterpreted function and a genuine `sat`.
#[test]
fn p2b36_a_read_of_an_array_constant_is_refuted() {
    let script = "\
(set-logic QF_ABV)
(declare-const i (_ BitVec 8))
(assert (= (bvadd (select ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x02) i) #x01) #x06))
(check-sat)
";
    let verdict = match run_script(script) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        Run::Error(_) => "error",
        Run::Panic(_) => "panic",
    };
    assert_eq!(
        verdict, "unsat",
        "the wrong `sat` of the pre-#P2b-36 tree is back (now {verdict})"
    );
    let shadowed = "\
(set-logic QF_AUFBV)
(declare-fun |(as const)| ((_ BitVec 8)) (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select (|(as const)| #x00) #x00) #x05))
(check-sat)
";
    let shadowed_verdict = match run_script(shadowed) {
        Run::Lines(lines) => lines.first().and_then(|l| verdict_of(l)).unwrap_or("none"),
        _ => "error",
    };
    assert_eq!(
        shadowed_verdict, "sat",
        "a declared `|(as const)|` must stay an uninterpreted function"
    );
}
