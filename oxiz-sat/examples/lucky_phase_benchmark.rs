//! Before/after benchmark for the pre-search lucky phase (issue #35).
//!
//! Generates four satisfiable families that a lucky scan is expected to settle
//! without any search — three of them at ~1000 variables, plus one small
//! instance built to be *hard for the search* — writes each as a DIMACS CNF
//! file into the system temp directory, and runs both configurations over them
//! through [`BenchmarkHarness`]:
//!
//! * **before** — `enable_lucky_phase: false`, i.e. the CDCL loop alone;
//! * **after** — the default configuration.
//!
//! Both runs go through the same `DimacsParser` → `Solver::solve()` path, so
//! the wall-clock numbers are comparable; the only difference is the flag.
//!
//! The first three families are cheap for the CDCL loop too, so they show the
//! phase's *floor*: a small constant-factor win from skipping search entirely.
//! The fourth (`guarded pigeonhole`) is the case the phase exists for, where
//! the search's default phase guess walks it into an exponentially hard core.
//!
//! Run with `cargo run --release --example lucky_phase_benchmark`. The
//! "before" runs are given a conflict budget so an unlucky family cannot turn
//! the benchmark into an unbounded search; a run that exhausts it is reported
//! as `Unknown` with its budget noted rather than silently skewing the table.

use oxiz_sat::{BenchmarkHarness, DimacsParser, Solver, SolverConfig, SolverResult};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Conflict budget for the lucky-phase-off runs.
const BEFORE_CONFLICT_BUDGET: u64 = 2_000_000;

/// xorshift64*, so every family below is reproducible.
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        self.0 % bound
    }
}

/// A generated instance: a name, its variable count, and its clauses.
struct Family {
    name: &'static str,
    /// Which lucky scan is expected to settle it.
    expected_scan: &'static str,
    num_vars: usize,
    clauses: Vec<Vec<i32>>,
}

/// Random 3-SAT with every literal positive: the all-true scan's home turf,
/// and the shape the issue's `simon-*` instances reduce to.
fn all_positive_3sat(num_vars: usize, num_clauses: usize, seed: u64) -> Family {
    let mut rng = Rng(seed);
    let clauses = (0..num_clauses)
        .map(|_| {
            (0..3)
                .map(|_| 1 + rng.below(num_vars as u64) as i32)
                .collect()
        })
        .collect();
    Family {
        name: "all-positive random 3-SAT",
        expected_scan: "AllTrue",
        num_vars,
        clauses,
    }
}

/// Blocks of three variables whose clauses are ordered so that a single
/// forward greedy pass satisfies them, while neither constant assignment
/// does: `(a∨b)` rules out all-false and `(¬b∨¬c)` rules out all-true.
fn ordered_chain(num_blocks: usize) -> Family {
    let mut clauses = Vec::with_capacity(num_blocks * 4);
    for block in 0..num_blocks {
        let base = (block * 3) as i32;
        let (a, b, c) = (base + 1, base + 2, base + 3);
        clauses.push(vec![a, b]);
        clauses.push(vec![-b, -c]);
        clauses.push(vec![a, -b]);
        clauses.push(vec![-b, c, -a]);
    }
    Family {
        name: "ordered chain",
        expected_scan: "ForwardOrdered",
        num_vars: num_blocks * 3,
        clauses,
    }
}

/// Blocks that defeat *both* greedy directions — the second half of each
/// block is emitted in reverse order — leaving the Horn least-model closure
/// as the first scan that works.
fn horn_closure_family(num_blocks: usize) -> Family {
    let trap = |base: i32| -> Vec<Vec<i32>> {
        let (a, b, c) = (base + 1, base + 2, base + 3);
        vec![vec![-a, -b], vec![a, b], vec![a, c], vec![-c, -b]]
    };
    let mut clauses = Vec::with_capacity(num_blocks * 8);
    for block in 0..num_blocks {
        let base = (block * 6) as i32;
        clauses.extend(trap(base));
        let mut mirrored = trap(base + 3);
        mirrored.reverse();
        clauses.extend(mirrored);
    }
    Family {
        name: "Horn-closure blocks",
        expected_scan: "HornLeastModel",
        num_vars: num_blocks * 6,
        clauses,
    }
}

/// The shape this whole phase exists for: a satisfiable instance with a *hard
/// unsatisfiable core hidden behind one guard literal* that the default phase
/// heuristic guesses wrong.
///
/// Every clause of PHP(`pigeons`, `holes`) — the classic exponentially hard
/// pigeonhole principle — is extended with a fresh guard variable `g` (index
/// 1, so it is also the first variable the search decides). Setting `g` true
/// satisfies the entire formula at a stroke, which is exactly what the
/// all-true scan does. The search, whose saved phase starts every variable
/// false, instead sets `g` false and is then obliged to *refute* the
/// pigeonhole core before it can learn the unit `(g)` — the same "seconds
/// versus microseconds" gap issue #35 reports on the `simon-*` instances.
fn guarded_pigeonhole(pigeons: usize, holes: usize) -> Family {
    let guard = (pigeons * holes + 1) as i32;
    let cell = |pigeon: usize, hole: usize| -> i32 { 1 + (pigeon * holes + hole) as i32 };
    let mut clauses = Vec::new();
    for pigeon in 0..pigeons {
        let mut clause = vec![guard];
        clause.extend((0..holes).map(|hole| cell(pigeon, hole)));
        clauses.push(clause);
    }
    for hole in 0..holes {
        for first in 0..pigeons {
            for second in (first + 1)..pigeons {
                clauses.push(vec![guard, -cell(first, hole), -cell(second, hole)]);
            }
        }
    }
    Family {
        name: "guarded pigeonhole",
        expected_scan: "AllTrue",
        num_vars: 1 + pigeons * holes,
        clauses,
    }
}

fn write_dimacs(family: &Family) -> std::io::Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "oxiz_sat_lucky_bench_{}.cnf",
        family.name.replace([' ', '-'], "_")
    ));
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    writeln!(file, "c {} ({})", family.name, family.expected_scan)?;
    writeln!(file, "p cnf {} {}", family.num_vars, family.clauses.len())?;
    for clause in &family.clauses {
        for lit in clause {
            write!(file, "{lit} ")?;
        }
        writeln!(file, "0")?;
    }
    file.flush()?;
    Ok(path)
}

/// One measured run. `budget` caps the search for the lucky-phase-off case.
struct Measurement {
    result: SolverResult,
    millis: f64,
    conflicts: u64,
    decisions: u64,
    propagations: u64,
    lucky_attempts: u64,
    lucky_successes: u64,
}

fn measure(
    harness: &mut BenchmarkHarness,
    path: &Path,
    config: SolverConfig,
    budget: Option<u64>,
) -> Option<Measurement> {
    // `BenchmarkHarness::run_file` owns solver construction, so a conflict
    // budget cannot be passed through the config; when one is requested the
    // same parse-then-solve sequence is run directly instead, timed the same
    // way. Both paths use `DimacsParser` on the same file.
    if let Some(max_conflicts) = budget {
        let mut parser = DimacsParser::new();
        let mut solver = Solver::with_config(config);
        if parser.parse_file(path, &mut solver).is_err() {
            return None;
        }
        solver.set_max_conflicts(Some(max_conflicts));
        let start = std::time::Instant::now();
        let result = solver.solve();
        let elapsed = start.elapsed();
        let stats = solver.stats();
        return Some(Measurement {
            result,
            millis: elapsed.as_secs_f64() * 1000.0,
            conflicts: stats.conflicts,
            decisions: stats.decisions,
            propagations: stats.propagations,
            lucky_attempts: stats.lucky_attempts,
            lucky_successes: stats.lucky_successes,
        });
    }

    let run = harness.run_file(path, config, None).ok()?;
    let stats = run.stats.solver_stats();
    Some(Measurement {
        result: run.result,
        millis: run.wall_time.as_secs_f64() * 1000.0,
        conflicts: stats.conflicts,
        decisions: stats.decisions,
        propagations: stats.propagations,
        lucky_attempts: stats.lucky_attempts,
        lucky_successes: stats.lucky_successes,
    })
}

fn lucky_disabled() -> SolverConfig {
    SolverConfig {
        enable_lucky_phase: false,
        ..SolverConfig::default()
    }
}

fn main() {
    let families = [
        all_positive_3sat(1000, 4300, 0x5EED_2035),
        ordered_chain(333),
        horn_closure_family(167),
        guarded_pigeonhole(8, 7),
    ];

    println!("Lucky phase benchmark (issue #35)");
    println!("budget for the lucky-phase-off runs: {BEFORE_CONFLICT_BUDGET} conflicts\n");
    println!(
        "{:<26} {:>5} {:>6} {:>9} {:>10} {:>11} {:>10}",
        "instance", "vars", "clauses", "config", "verdict", "wall (ms)", "conflicts"
    );
    println!("{}", "-".repeat(84));

    for family in &families {
        let path = match write_dimacs(family) {
            Ok(path) => path,
            Err(err) => {
                println!("{}: could not write DIMACS: {err}", family.name);
                continue;
            }
        };
        let mut harness = BenchmarkHarness::new();

        let before = measure(
            &mut harness,
            &path,
            lucky_disabled(),
            Some(BEFORE_CONFLICT_BUDGET),
        );
        let after = measure(&mut harness, &path, SolverConfig::default(), None);

        for (label, measurement) in [("before", &before), ("after", &after)] {
            match measurement {
                Some(m) => println!(
                    "{:<26} {:>5} {:>6} {:>9} {:>10} {:>11.3} {:>10}",
                    family.name,
                    family.num_vars,
                    family.clauses.len(),
                    label,
                    format!("{:?}", m.result),
                    m.millis,
                    m.conflicts
                ),
                None => println!("{:<26} {label}: run failed", family.name),
            }
        }

        if let (Some(before), Some(after)) = (&before, &after) {
            let speedup = if after.millis > 0.0 {
                before.millis / after.millis
            } else {
                f64::INFINITY
            };
            println!(
                "  → expected scan {}: {} attempt(s), {} hit(s); \
                 decisions {} → {}, propagations {} → {}; speedup {speedup:.1}x",
                family.expected_scan,
                after.lucky_attempts,
                after.lucky_successes,
                before.decisions,
                after.decisions,
                before.propagations,
                after.propagations
            );
        }
        println!();
        let _ = std::fs::remove_file(&path);
    }
}
