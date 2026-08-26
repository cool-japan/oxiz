//! Issue #38: is `SolverConfig::enable_lazy_hyper_binary` worth having on by
//! default?
//!
//! The report was a ~12x conflict blowup on a QF_UF quasigroup family. The
//! guards in `Solver::check_hyper_binary_resolution` were already correct by
//! the time this file was written; what was missing was *measurement*. This
//! file is that measurement, kept in-tree so the decision recorded at the
//! flag's declaration site (`solver/config.rs`) can be re-derived rather than
//! believed.
//!
//! Layout:
//!
//! * two cheap, always-run tests that pin the pass's observable behaviour —
//!   that it fires at all (this codebase has a history of flags that turned
//!   out to be inert), and that walking the binary implication graph does not
//!   enter it at all, which is where the bulk of its wasted scanning used to
//!   come from;
//! * three `#[ignore]`d sweeps that run generated families with the flag on
//!   and off and print a table — [`measure_lazy_hyper_binary`] (minutes; the
//!   families that fit in a coffee break),
//!   [`measure_lazy_hyper_binary_large`] (hours; the sizes at which a
//!   difference has room to compound) and
//!   [`measure_lazy_hyper_binary_quasigroup`] (the reported family in depth).
//!   Run one with
//!
//!   ```text
//!   cargo nextest run -p oxiz-sat --run-ignored all \
//!       -E 'test(=measure_lazy_hyper_binary)' --no-capture
//!   ```
//!
//!   Note the `=`: without it the filter is a substring match and picks up all
//!   three. Nextest's default profile also terminates a test that has been
//!   running for three minutes, so the long sweeps are better run straight
//!   from the test binary (`--ignored --nocapture --test-threads 1`) than
//!   through nextest.
//!
//! Every family is generated in-process from a fixed seed, so the numbers are
//! reproducible without a corpus checked into the repository. `propagations`
//! is the machine-independent cost metric; wall time is reported too but is
//! only meaningful on an otherwise idle machine.
//!
//! # What counts as evidence
//!
//! Fixed before the sweep was run, so that the reading of it could not be
//! fitted to whatever came back. The question is a default, so the bar is a
//! *median over a family*, not a worst case:
//!
//! * A family's verdict is the median of its **conflict** ratios (on / off)
//!   over the pairs both arms decided. A median at or above 2.0 is a blowup;
//!   at or below 0.5, a win.
//! * A pair where **both** arms exhausted the conflict budget has a conflict
//!   ratio of exactly 1.00 by construction and is not evidence about
//!   conflicts. It is rated on its propagation ratio instead, which answers a
//!   narrower question: did the two arms' searches *diverge* over the same
//!   budget? It is not a cost comparison — scanning is read-only and never
//!   propagates, so the pass's own overhead does not appear in this number at
//!   all, and a 1.00 means "same search", not "free". A family measured only
//!   this way therefore cannot flip the default in either direction.
//! * A pair where exactly one arm ran out of budget yields an inequality, not
//!   a ratio. Those are printed individually and never folded into a median.

use oxiz_sat::{Lit, Solver, SolverConfig, SolverResult};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*, same shape as the one in issue35's tests)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Never let the state be the xorshift fixed point 0.
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform-ish value in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

struct Instance {
    family: &'static str,
    name: String,
    vars: usize,
    cnf: Vec<Vec<i32>>,
}

/// (a) Uniform random 3-SAT at the phase-transition ratio 4.26.
fn random_3sat(vars: usize, seed: u64) -> Instance {
    let mut rng = Rng::new(seed ^ (vars as u64) << 32);
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let clause_count = (vars as f64 * 4.26).round() as usize;
    let mut cnf = Vec::with_capacity(clause_count);
    for _ in 0..clause_count {
        let mut clause = Vec::with_capacity(3);
        while clause.len() < 3 {
            let var = rng.below(vars) + 1;
            if clause
                .iter()
                .any(|l: &i32| l.unsigned_abs() as usize == var)
            {
                continue;
            }
            let sign = if rng.next_u64() & 1 == 0 { 1 } else { -1 };
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            clause.push(sign * var as i32);
        }
        cnf.push(clause);
    }
    // One family tag per size: a median that mixed n=150 with n=250 would
    // hide exactly the size-dependence the sweep is looking for.
    let family = match vars {
        150 => "random-3sat-150",
        200 => "random-3sat-200",
        250 => "random-3sat-250",
        _ => "random-3sat",
    };
    Instance {
        family,
        name: format!("rand3sat-n{vars}-s{seed}"),
        vars,
        cnf,
    }
}

/// (b) Pigeonhole PHP(pigeons, pigeons - 1): resolution-hard UNSAT.
fn php(pigeons: usize) -> Instance {
    let holes = pigeons - 1;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let var = |p: usize, h: usize| (p * holes + h + 1) as i32;
    let mut cnf = Vec::new();
    for p in 0..pigeons {
        cnf.push((0..holes).map(|h| var(p, h)).collect());
    }
    for h in 0..holes {
        for p1 in 0..pigeons {
            for p2 in (p1 + 1)..pigeons {
                cnf.push(vec![-var(p1, h), -var(p2, h)]);
            }
        }
    }
    Instance {
        family: "php",
        name: format!("php-{pigeons}-{holes}"),
        vars: pigeons * holes,
        cnf,
    }
}

/// (c) Quasigroup (latin-square) completion — the reported family's shape,
/// encoded directly as CNF: `v(r, c, k)` is "cell (r, c) holds value k".
///
/// The partial assignment is carved out of a genuine latin square, so an
/// unmutated instance is satisfiable by construction. `swap_prefill` exchanges
/// the values of two prefilled cells, which usually (not always) makes the
/// remaining completion impossible; the measured verdict is reported rather
/// than assumed.
fn quasigroup(order: usize, fill_percent: usize, seed: u64, swap_prefill: bool) -> Instance {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let var = |r: usize, c: usize, k: usize| (r * order * order + c * order + k + 1) as i32;
    let mut cnf: Vec<Vec<i32>> = Vec::new();

    // Each cell holds exactly one value.
    for r in 0..order {
        for c in 0..order {
            cnf.push((0..order).map(|k| var(r, c, k)).collect());
            for k1 in 0..order {
                for k2 in (k1 + 1)..order {
                    cnf.push(vec![-var(r, c, k1), -var(r, c, k2)]);
                }
            }
        }
    }
    // Each value occurs exactly once per row and once per column.
    for k in 0..order {
        for r in 0..order {
            cnf.push((0..order).map(|c| var(r, c, k)).collect());
            for c1 in 0..order {
                for c2 in (c1 + 1)..order {
                    cnf.push(vec![-var(r, c1, k), -var(r, c2, k)]);
                }
            }
        }
        for c in 0..order {
            cnf.push((0..order).map(|r| var(r, c, k)).collect());
            for r1 in 0..order {
                for r2 in (r1 + 1)..order {
                    cnf.push(vec![-var(r1, c, k), -var(r2, c, k)]);
                }
            }
        }
    }

    // A random latin square: cyclic base, then permute rows, columns, values.
    let mut rng = Rng::new(seed ^ (order as u64) << 40 ^ (fill_percent as u64) << 20);
    let mut row_perm: Vec<usize> = (0..order).collect();
    let mut col_perm: Vec<usize> = (0..order).collect();
    let mut val_perm: Vec<usize> = (0..order).collect();
    rng.shuffle(&mut row_perm);
    rng.shuffle(&mut col_perm);
    rng.shuffle(&mut val_perm);
    let square = |r: usize, c: usize| val_perm[(row_perm[r] + col_perm[c]) % order];

    let mut cells: Vec<(usize, usize)> = (0..order)
        .flat_map(|r| (0..order).map(move |c| (r, c)))
        .collect();
    rng.shuffle(&mut cells);
    let keep = cells.len() * fill_percent / 100;
    let mut prefill: Vec<(usize, usize, usize)> = cells[..keep]
        .iter()
        .map(|&(r, c)| (r, c, square(r, c)))
        .collect();

    if swap_prefill && prefill.len() >= 2 {
        // Exchange the values of two prefilled cells that share neither row
        // nor column, so neither unit clash is visible at level 0.
        let mut swapped = false;
        'outer: for i in 0..prefill.len() {
            for j in (i + 1)..prefill.len() {
                if prefill[i].0 != prefill[j].0
                    && prefill[i].1 != prefill[j].1
                    && prefill[i].2 != prefill[j].2
                {
                    let tmp = prefill[i].2;
                    prefill[i].2 = prefill[j].2;
                    prefill[j].2 = tmp;
                    swapped = true;
                    break 'outer;
                }
            }
        }
        assert!(swapped, "quasigroup prefill had no swappable pair");
    }

    for (r, c, k) in prefill {
        cnf.push(vec![var(r, c, k)]);
    }

    let tag = if swap_prefill { "mut" } else { "sat" };
    Instance {
        family: "quasigroup",
        name: format!("qg{order}-f{fill_percent}-s{seed}-{tag}"),
        vars: order * order * order,
        cnf,
    }
}

/// (c') Quasigroup *existence* (the QG3 axiom `(x*y)*(y*x) = x` over an
/// idempotent quasigroup), which — unlike completion — genuinely searches.
///
/// `v(x, y, z)` is "`x * y = z`". The latin-square core is the same as in
/// [`quasigroup`]; on top of it every `x, y, a, b` gets the ternary
/// `(!v(x,y,a) | !v(y,x,b) | v(a,b,x))`, and idempotence `x * x = x` is
/// asserted as units. The ternaries are exactly the reason-clause size the
/// pass under measurement can exploit, and the units give it level-0
/// literals to discharge.
fn quasigroup_qg3(order: usize) -> Instance {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let var = |x: usize, y: usize, z: usize| (x * order * order + y * order + z + 1) as i32;
    let mut cnf: Vec<Vec<i32>> = Vec::new();

    for x in 0..order {
        for y in 0..order {
            cnf.push((0..order).map(|z| var(x, y, z)).collect());
            for z1 in 0..order {
                for z2 in (z1 + 1)..order {
                    cnf.push(vec![-var(x, y, z1), -var(x, y, z2)]);
                }
            }
        }
    }
    for z in 0..order {
        for x in 0..order {
            cnf.push((0..order).map(|y| var(x, y, z)).collect());
            for y1 in 0..order {
                for y2 in (y1 + 1)..order {
                    cnf.push(vec![-var(x, y1, z), -var(x, y2, z)]);
                }
            }
        }
        for y in 0..order {
            cnf.push((0..order).map(|x| var(x, y, z)).collect());
            for x1 in 0..order {
                for x2 in (x1 + 1)..order {
                    cnf.push(vec![-var(x1, y, z), -var(x2, y, z)]);
                }
            }
        }
    }
    // QG3: (x*y) * (y*x) = x.
    for x in 0..order {
        for y in 0..order {
            for a in 0..order {
                for b in 0..order {
                    cnf.push(vec![-var(x, y, a), -var(y, x, b), var(a, b, x)]);
                }
            }
        }
    }
    // Idempotence, the standard side condition for the QG benchmarks.
    for x in 0..order {
        cnf.push(vec![var(x, x, x)]);
    }

    Instance {
        family: "quasigroup-qg3",
        name: format!("qg3-{order}"),
        vars: order * order * order,
        cnf,
    }
}

/// (d) The family where lazy hyper-binary resolution *should* pay: long
/// implication chains encoded as guarded ternaries.
///
/// Each chain step is `(!guard | !x_i | x_{i+1})` with `guard` asserted as a
/// unit. That is exactly the shape the pass looks for — one other literal
/// false at level 0, one at the current level — so propagating along a chain
/// lets it learn the shortcut binary `(!x_i | x_{i+1})`. A random 3-SAT core
/// over the chain endpoints keeps the instance from being solved by
/// propagation alone.
fn guarded_chains(chains: usize, length: usize, core_vars: usize, seed: u64) -> Instance {
    let mut rng = Rng::new(seed ^ 0x00c0_ffee_0000_0000);
    let mut cnf: Vec<Vec<i32>> = Vec::new();

    // Layout: 1..=core_vars core, then `chains` guards, then the chain bodies.
    let guard_base = core_vars;
    let chain_base = core_vars + chains;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let guard = |i: usize| (guard_base + i + 1) as i32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let chain = |i: usize, j: usize| (chain_base + i * length + j + 1) as i32;

    for i in 0..chains {
        cnf.push(vec![guard(i)]);
        for j in 0..(length - 1) {
            cnf.push(vec![-guard(i), -chain(i, j), chain(i, j + 1)]);
        }
    }

    // Tie chain heads and tails into the core: a chain head is implied by a
    // core literal, and a chain tail feeds back into the core.
    for i in 0..chains {
        let head_trigger = rng.below(core_vars) + 1;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        cnf.push(vec![-(head_trigger as i32), chain(i, 0)]);
        let tail_target = rng.below(core_vars) + 1;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        cnf.push(vec![-chain(i, length - 1), tail_target as i32]);
    }

    // Random 3-SAT core at the phase transition over the core variables.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let clause_count = (core_vars as f64 * 4.26).round() as usize;
    for _ in 0..clause_count {
        let mut clause: Vec<i32> = Vec::with_capacity(3);
        while clause.len() < 3 {
            let var = rng.below(core_vars) + 1;
            if clause.iter().any(|l| l.unsigned_abs() as usize == var) {
                continue;
            }
            let sign = if rng.next_u64() & 1 == 0 { 1 } else { -1 };
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            clause.push(sign * var as i32);
        }
        cnf.push(clause);
    }

    Instance {
        family: "guarded-chains",
        name: format!("chains-c{chains}-l{length}-n{core_vars}-s{seed}"),
        vars: chain_base + chains * length,
        cnf,
    }
}

// ---------------------------------------------------------------------------
// Running one arm
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Run {
    result: SolverResult,
    conflicts: u64,
    propagations: u64,
    decisions: u64,
    learned: u64,
    hb_attempts: u64,
    hb_learned: u64,
    elapsed: Duration,
}

/// The two arms differ in exactly one field. `enable_lucky_phase` is off in
/// both so the pre-search scans cannot decide an instance before the CDCL loop
/// (which is what the pass under measurement lives in), and the PRNG is seeded
/// identically so the only difference between the arms is the flag.
fn arm_config(hyper: bool) -> SolverConfig {
    SolverConfig {
        enable_lazy_hyper_binary: hyper,
        enable_lucky_phase: false,
        ..SolverConfig::default()
    }
}

fn run_arm(instance: &Instance, hyper: bool, max_conflicts: u64) -> Run {
    let mut solver = Solver::with_config(arm_config(hyper));
    solver.set_random_seed(0x5eed_1234);
    solver.ensure_vars(instance.vars);
    for clause in &instance.cnf {
        solver.add_clause_dimacs(clause);
    }
    solver.set_max_conflicts(Some(max_conflicts));
    let start = Instant::now();
    let result = solver.solve();
    let elapsed = start.elapsed();
    let stats = solver.stats();
    Run {
        result,
        conflicts: stats.conflicts,
        propagations: stats.propagations,
        decisions: stats.decisions,
        learned: stats.learned_clauses,
        hb_attempts: stats.hyper_binary_attempts,
        hb_learned: stats.hyper_binary_learned,
        elapsed,
    }
}

// ---------------------------------------------------------------------------
// Cheap always-run behavioural pins
// ---------------------------------------------------------------------------

/// The pass must actually fire. Assumptions give a deterministic way to reach
/// decision level >= 2 (each assumption is decided on its own level), so the
/// guarded ternary `(!g | !a | b)` propagates `b` with `!g` false at level 0
/// and `!a` false at the current level — precisely the learnable shape.
#[test]
fn hyper_binary_pass_fires_on_guarded_ternary() {
    let build = |hyper: bool| {
        let mut solver = Solver::with_config(arm_config(hyper));
        solver.ensure_vars(6);
        // 1 = guard (unit), 2/3 = padding assumptions, 4 = antecedent,
        // 5 = consequent, 6 unused.
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[-1, -4, 5]);
        solver.add_clause_dimacs(&[2, 3, 4, 5, 6]);
        let assumptions = [
            Lit::from_dimacs(2),
            Lit::from_dimacs(3),
            Lit::from_dimacs(4),
        ];
        let (result, _) = solver.solve_with_assumptions(&assumptions);
        assert_eq!(result, SolverResult::Sat, "instance is satisfiable");
        let stats = solver.stats();
        (stats.hyper_binary_attempts, stats.hyper_binary_learned)
    };

    let (on_attempts, on_learned) = build(true);
    assert!(
        on_attempts > 0,
        "the pass must run at all with the flag on (attempts = {on_attempts})"
    );
    assert!(
        on_learned > 0,
        "the guarded ternary is exactly the learnable shape, yet nothing was \
         learned (attempts = {on_attempts})"
    );

    let (off_attempts, off_learned) = build(false);
    assert_eq!(
        (off_attempts, off_learned),
        (0, 0),
        "the flag must switch the pass off completely"
    );
}

/// Propagation along the *binary implication graph* must not enter the pass
/// at all. The reason there is a binary clause `{!lit, implied}` whose edge
/// `lit -> implied` is what propagation just walked, so the only resolvent the
/// pass could form is that same clause and the graph lookup always rejects it:
/// pure overhead on every binary propagation at decision level >= 2. Those
/// visits were roughly a fifth of the pass's total work in the measurement
/// recorded at `SolverConfig::enable_lazy_hyper_binary`, and the call site
/// there no longer makes the call.
#[test]
fn hyper_binary_skips_binary_reasons_entirely() {
    let mut solver = Solver::with_config(arm_config(true));
    solver.ensure_vars(8);
    // Binary chain 4 -> 5 -> 6 -> 7, reachable only after two padding levels.
    solver.add_clause_dimacs(&[-4, 5]);
    solver.add_clause_dimacs(&[-5, 6]);
    solver.add_clause_dimacs(&[-6, 7]);
    solver.add_clause_dimacs(&[1, 2, 3, 8]);
    let assumptions = [
        Lit::from_dimacs(1),
        Lit::from_dimacs(2),
        Lit::from_dimacs(4),
    ];
    let (result, _) = solver.solve_with_assumptions(&assumptions);
    assert_eq!(result, SolverResult::Sat);
    let stats = solver.stats();
    assert_eq!(
        (stats.hyper_binary_attempts, stats.hyper_binary_learned),
        (0, 0),
        "walking a binary chain at level >= 2 must cost the pass nothing"
    );
}

// ---------------------------------------------------------------------------
// The measurement itself
// ---------------------------------------------------------------------------

/// Below this many conflicts in *both* arms an instance says nothing about the
/// pass: the ratio of two single-digit counts is noise, and a `0 / 0` or
/// `n / 0` pair would poison the family median outright.
const RATEABLE_CONFLICTS: u64 = 20;

fn ratio(on: u64, off: u64) -> f64 {
    if off == 0 {
        if on == 0 { 1.0 } else { f64::INFINITY }
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            on as f64 / off as f64
        }
    }
}

/// Per-family accumulator for the summary table.
struct FamilySummary {
    family: &'static str,
    /// Conflict ratios (on / off) of the pairs that are actually rateable.
    conflict_ratios: Vec<f64>,
    /// Propagation ratios of the same pairs.
    propagation_ratios: Vec<f64>,
    /// Propagation ratios of pairs where *both* arms exhausted the same
    /// conflict budget. Their conflict ratio is pinned at 1.00 by the cap and
    /// carries no information, but comparing the propagations each arm spent
    /// reaching that identical conflict count still detects whether the two
    /// searches diverged — so these pairs are not thrown away, they are rated
    /// on a narrower axis. It is *not* a measure of the pass's cost: its scans
    /// are read-only and propagate nothing, so they never reach this counter.
    equal_budget_propagation_ratios: Vec<f64>,
    /// One-sided caps: one arm finished and the other ran out of budget, so
    /// the true conflict ratio is only *bounded* by what was observed (at
    /// least this much worse for the capped arm). Recorded so a family whose
    /// worst pairs are all one-sided cannot look clean by dropping them.
    bounded_ratios: Vec<f64>,
    /// Hyper-binary work done by the on arm, summed over the whole family.
    attempts: u64,
    /// Hyper-binary clauses the on arm actually produced.
    learned: u64,
    /// Pairs where at least one arm hit the conflict cap.
    censored: usize,
    /// Pairs decided too quickly for a ratio to mean anything.
    trivial: usize,
    /// All pairs in the family.
    pairs: usize,
}

impl FamilySummary {
    fn new(family: &'static str) -> Self {
        Self {
            family,
            conflict_ratios: Vec::new(),
            propagation_ratios: Vec::new(),
            equal_budget_propagation_ratios: Vec::new(),
            bounded_ratios: Vec::new(),
            attempts: 0,
            learned: 0,
            censored: 0,
            trivial: 0,
            pairs: 0,
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Run every instance twice — flag on, flag off — and print the per-instance
/// table plus a per-family summary.
fn run_sweep(instances: &[Instance], max_conflicts: u64) {
    println!(
        "\n{:<26} {:>6} {:>9} {:>9} {:>6} {:>11} {:>11} {:>6} {:>9} {:>9} {:>7} {:>7}",
        "instance",
        "verd",
        "confl/on",
        "confl/off",
        "c-rat",
        "props/on",
        "props/off",
        "p-rat",
        "hb-att",
        "hb-lrn",
        "ms/on",
        "ms/off",
    );

    let mut by_family: Vec<FamilySummary> = Vec::new();
    let mut identical_pairs = 0usize;
    let mut total_pairs = 0usize;
    let mut determinism_checked = false;

    for instance in instances {
        let on = run_arm(instance, true, max_conflicts);
        let off = run_arm(instance, false, max_conflicts);
        // One repeat is enough to establish that both arms are deterministic;
        // repeating every instance would double the cost of the whole sweep
        // for no extra information. It is spent on the first *cheap* pair
        // rather than simply the first one, because a sweep may deliberately
        // lead with its most expensive instance (so that a truncated run still
        // has the decision-relevant answer) and quadrupling that one's cost to
        // re-establish a property every other pair also has would be a poor
        // trade.
        if !determinism_checked && on.elapsed + off.elapsed < Duration::from_secs(30) {
            determinism_checked = true;
            let on_again = run_arm(instance, true, max_conflicts);
            let off_again = run_arm(instance, false, max_conflicts);
            assert_eq!(
                (on.conflicts, on.propagations, on.decisions),
                (
                    on_again.conflicts,
                    on_again.propagations,
                    on_again.decisions
                ),
                "{}: the on arm must be deterministic across repeats",
                instance.name
            );
            assert_eq!(
                (off.conflicts, off.propagations, off.decisions),
                (
                    off_again.conflicts,
                    off_again.propagations,
                    off_again.decisions
                ),
                "{}: the off arm must be deterministic across repeats",
                instance.name
            );
        }

        let censored = on.result == SolverResult::Unknown || off.result == SolverResult::Unknown;
        if !censored {
            assert_eq!(
                on.result, off.result,
                "{}: the flag must not change the verdict",
                instance.name
            );
        }

        total_pairs += 1;
        if (on.conflicts, on.propagations, on.decisions)
            == (off.conflicts, off.propagations, off.decisions)
        {
            identical_pairs += 1;
        }

        let verdict = match (on.result, censored) {
            (_, true) => "CAP",
            (SolverResult::Sat, _) => "SAT",
            (SolverResult::Unsat, _) => "UNS",
            (SolverResult::Unknown, _) => "UNK",
        };
        let conflict_ratio = ratio(on.conflicts, off.conflicts);
        let propagation_ratio = ratio(on.propagations, off.propagations);
        println!(
            "{:<26} {:>6} {:>9} {:>9} {:>6.2} {:>11} {:>11} {:>6.2} {:>9} {:>9} {:>7} {:>7}",
            instance.name,
            verdict,
            on.conflicts,
            off.conflicts,
            conflict_ratio,
            on.propagations,
            off.propagations,
            propagation_ratio,
            on.hb_attempts,
            on.hb_learned,
            on.elapsed.as_millis(),
            off.elapsed.as_millis(),
        );

        let entry = match by_family.iter_mut().find(|e| e.family == instance.family) {
            Some(entry) => entry,
            None => {
                by_family.push(FamilySummary::new(instance.family));
                let last = by_family.len() - 1;
                &mut by_family[last]
            }
        };
        entry.attempts += on.hb_attempts;
        entry.learned += on.hb_learned;
        entry.pairs += 1;
        if censored {
            entry.censored += 1;
            if on.result == SolverResult::Unknown
                && off.result == SolverResult::Unknown
                && on.conflicts == off.conflicts
            {
                entry
                    .equal_budget_propagation_ratios
                    .push(propagation_ratio);
            } else {
                entry.bounded_ratios.push(conflict_ratio);
            }
        } else if on.conflicts.max(off.conflicts) < RATEABLE_CONFLICTS {
            entry.trivial += 1;
        } else {
            entry.conflict_ratios.push(conflict_ratio);
            entry.propagation_ratios.push(propagation_ratio);
        }
        // Learned-clause bookkeeping sanity: hyper-binaries are learned
        // clauses too, and the off arm must not produce any.
        assert!(on.learned >= on.hb_learned);
        assert_eq!(off.hb_attempts, 0);
        assert_eq!(off.hb_learned, 0);
    }

    println!(
        "\n{:<18} {:>5} {:>7} {:>5} {:>11} {:>11} {:>6} {:>11} {:>11} {:>11}",
        "family",
        "pairs",
        "trivial",
        "rated",
        "med c-ratio",
        "med p-ratio",
        "atcap",
        "med p@cap",
        "hb-attempts",
        "hb-learned",
    );
    for summary in &mut by_family {
        println!(
            "{:<18} {:>5} {:>7} {:>5} {:>11.3} {:>11.3} {:>6} {:>11.3} {:>11} {:>11}",
            summary.family,
            summary.pairs,
            summary.trivial,
            summary.conflict_ratios.len(),
            median(&mut summary.conflict_ratios),
            median(&mut summary.propagation_ratios),
            summary.equal_budget_propagation_ratios.len(),
            median(&mut summary.equal_budget_propagation_ratios),
            summary.attempts,
            summary.learned,
        );
    }
    for summary in &by_family {
        if !summary.bounded_ratios.is_empty() {
            // Printed rather than folded into a median: with only one arm
            // finished these are inequalities, not measurements, and a median
            // of inequalities would misrepresent them.
            println!(
                "{}: {} one-sided cap(s), observed conflict ratio bound(s): {:?}",
                summary.family,
                summary.bounded_ratios.len(),
                summary.bounded_ratios,
            );
        }
    }
    println!(
        "\nidentical (conflicts, propagations, decisions) pairs: {identical_pairs}/{total_pairs}"
    );
    // Reported rather than asserted: a sweep in which *every* pair was too
    // slow to repeat is still a valid measurement, but the reader should know
    // that its determinism went unchecked.
    println!("determinism re-checked on a repeated pair: {determinism_checked}");
}

#[test]
#[ignore = "measurement harness for issue #38; minutes of CPU, run explicitly"]
fn measure_lazy_hyper_binary() {
    // Conflict cap: bounds the sweep.
    let max_conflicts: u64 = 25_000;

    // Ordered so the family the pass is supposed to *help* comes first: a
    // sweep cut short still answers the question that matters most.
    //
    // This is the affordable half of the measurement (single-digit minutes).
    // The sizes that cost hours — random 3-SAT at 200 and 250 variables, and
    // PHP(9,8) — live in [`measure_lazy_hyper_binary_large`] so that this one
    // stays runnable on a whim; both halves are recorded at
    // `SolverConfig::enable_lazy_hyper_binary`.
    let mut instances: Vec<Instance> = Vec::new();
    for seed in 1u64..=5 {
        instances.push(guarded_chains(8, 14, 140, seed));
    }
    for seed in 1u64..=5 {
        instances.push(random_3sat(150, seed));
    }
    for pigeons in [6usize, 7, 8] {
        instances.push(php(pigeons));
    }
    // Quasigroup *completion* at any order this port can afford is solved by
    // propagation alone; the quasigroup family that actually searches is the
    // QG3 existence problem (see `measure_lazy_hyper_binary_quasigroup`).
    instances.push(quasigroup_qg3(7));

    run_sweep(&instances, max_conflicts);
}

/// The expensive half: the sizes at which a difference, if there is one, has
/// room to compound.
///
/// A pass that is neutral on small instances can still be ruinous on large
/// ones, so the decision cannot rest on the cheap sweep alone. The cap is
/// raised to 60 000 conflicts because at 25 000 several of these pairs end
/// one-sided (one arm finished, the other did not), which is the least
/// informative outcome of all. Where both arms still exhaust the budget, the
/// pair is rated on propagations spent per identical conflict budget instead
/// of being discarded — see [`FamilySummary`].
///
/// Hours of CPU. Run it with the same command as the cheap sweep, substituting
/// this test's name.
#[test]
#[ignore = "measurement harness for issue #38; hours of CPU, run explicitly"]
fn measure_lazy_hyper_binary_large() {
    let mut instances: Vec<Instance> = Vec::new();
    // QG3 first, and deliberately: issue #38's report was about a quasigroup
    // family, so of everything in this sweep, order 9 is the instance whose
    // answer would actually reopen the decision. A run that gets cut short
    // must not be one that lost exactly that.
    instances.push(quasigroup_qg3(9));
    for seed in 1u64..=5 {
        instances.push(random_3sat(200, seed));
    }
    for seed in 1u64..=5 {
        instances.push(random_3sat(250, seed));
    }
    // Last: order 10 is 1000 variables and 10 000 ternaries, and unlike the
    // random families its cost at this cap is not bounded by anything already
    // measured.
    instances.push(quasigroup_qg3(10));
    run_sweep(&instances, 60_000);

    // PHP(9,8) at the *cheap* sweep's cap rather than this one's. The pass is
    // structurally near-inert on pigeonhole: every "some hole holds this
    // pigeon" clause has width `holes`, which the `2 ..= 4` reason-width guard
    // rejects outright, so the only clauses that ever reach the pass are the
    // width-2 at-most-one constraints reaching it through the watch lists.
    // That shows up in the cheap sweep as 4, 25 and 905 attempts for PHP(6,5),
    // PHP(7,6) and PHP(8,7) against tens to hundreds of thousands of
    // propagations. Four more times the budget would buy a more precise
    // version of a 1.00 that is already explained.
    run_sweep(&[php(9)], 25_000);
}

/// The same large families as [`measure_lazy_hyper_binary_large`], but on a
/// deliberately small *budget* instead of a cap.
///
/// A cap is chosen high enough that instances finish, which is what makes the
/// large sweep cost hours. A budget is chosen low enough that they all stop in
/// the same place: every pair then ends with both arms at exactly the budget,
/// which forfeits the conflict ratio (pinned at 1.00) but still answers
/// "did the two searches diverge at this size?" on families that are otherwise
/// out of reach. Minutes rather than hours, so it is the one to reach for on a
/// machine that is already busy, or as a pre-flight check before committing to
/// the large sweep.
///
/// Per the evidence rule at the top of this file, a family measured only this
/// way cannot move the default on its own.
#[test]
#[ignore = "measurement harness for issue #38; minutes of CPU, run explicitly"]
fn measure_lazy_hyper_binary_budgeted() {
    // Low enough that every instance below is still running when it is
    // reached: the smallest conflict count any of these families finished in
    // during the cap-based sweeps was 7563 (random 3-SAT, n=200).
    let budget: u64 = 3_000;
    let mut instances: Vec<Instance> = Vec::new();
    instances.push(quasigroup_qg3(9));
    for seed in 1u64..=5 {
        instances.push(random_3sat(200, seed));
    }
    for seed in 1u64..=5 {
        instances.push(random_3sat(250, seed));
    }
    instances.push(php(9));
    instances.push(quasigroup_qg3(10));
    run_sweep(&instances, budget);
}

/// The reported family, at sizes that actually reach the CDCL loop.
///
/// Quasigroup *completion* only becomes hard around the phase transition in
/// the fraction of prefilled cells, and only for orders well above the 7/8
/// used as a smoke test above: at order 7-8 with 35% of the cells given, both
/// arms finish in single-digit conflicts and every ratio is a meaningless
/// `1.00`. These orders/fills are where the search actually happens.
#[test]
#[ignore = "measurement harness for issue #38; minutes of CPU, run explicitly"]
fn measure_lazy_hyper_binary_quasigroup() {
    let max_conflicts: u64 = 25_000;
    let mut instances: Vec<Instance> = Vec::new();
    // Orders 7 and 8 only: 9 and 10 are the expensive end and belong to
    // [`measure_lazy_hyper_binary_large`], which runs them once at a cap high
    // enough for them to finish. Running them here too would double hours of
    // CPU to re-derive the same rows at a lower cap.
    for order in [7usize, 8] {
        instances.push(quasigroup_qg3(order));
    }
    for (order, fill) in [(10usize, 45usize), (12, 45), (12, 55), (15, 55), (15, 65)] {
        for seed in 1u64..=3 {
            instances.push(quasigroup(order, fill, seed, false));
            instances.push(quasigroup(order, fill, seed, true));
        }
    }
    run_sweep(&instances, max_conflicts);
}
