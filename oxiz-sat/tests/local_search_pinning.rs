//! Pins WalkSAT and ProbSAT, step for step, on fixed formulas and seeds.
//!
//! Local search is fully determined by its seed, the formula and the order in
//! which the unsatisfied clauses are kept, so the result, the final assignment
//! and the three statistics below change if any of those change. The expected
//! fingerprints were taken from the implementation before the unsatisfied
//! clauses were kept together with their clause references; the two must agree
//! exactly.

use oxiz_sat::{
    Clause, ClauseDatabase, ClauseId, Lit, LocalSearch, LocalSearchConfig, LocalSearchResult, Var,
};

/// A deterministic random 3-CNF over `num_vars` variables, three distinct
/// variables per clause (an LCG of its own, so the formula never depends on
/// another crate's generator).
fn random_3cnf(num_vars: u32, num_clauses: usize, mut state: u64) -> ClauseDatabase {
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let mut db = ClauseDatabase::new();
    for _ in 0..num_clauses {
        let mut lits: Vec<Lit> = Vec::with_capacity(3);
        while lits.len() < 3 {
            let var = Var::new((next() % u64::from(num_vars)) as u32);
            if lits.iter().any(|l| l.var() == var) {
                continue;
            }
            let lit = if next() % 2 == 0 {
                Lit::pos(var)
            } else {
                Lit::neg(var)
            };
            lits.push(lit);
        }
        db.add(Clause::new(lits, false));
    }
    db
}

fn fingerprint(
    name: &str,
    result: LocalSearchResult,
    assignment: Option<Vec<bool>>,
    ls: &LocalSearch,
) -> String {
    let bits: String = assignment
        .map(|a| a.iter().map(|&b| if b { '1' } else { '0' }).collect())
        .unwrap_or_else(|| "-".to_string());
    let stats = ls.stats();
    format!(
        "{name} {result:?} flips={} min_unsat={} improvements={} assignment={bits}",
        stats.flips, stats.min_unsat, stats.improvements
    )
}

/// One case: variables, clauses, formula seed, search seed, algorithm, and
/// the IDs of clauses removed before the search.
type Case = (u32, usize, u64, u64, &'static str, &'static [u32]);

/// The cases. Each is a run the search finishes within a few flips: the
/// break-count bookkeeping of a longer run underflows `u64` (an
/// arithmetic-overflow panic in a build with overflow checks), so a longer
/// run's outcome depends on the build profile and could not be pinned once
/// for both.
const CASES: &[Case] = &[
    (4, 10, 1, 7, "walksat", &[]),
    (5, 6, 5, 1, "probsat", &[]),
    (5, 6, 5, 7, "walksat", &[]),
    (6, 10, 3, 1, "walksat", &[]),
    (6, 10, 4, 7, "walksat", &[]),
    (8, 6, 3, 99, "walksat", &[]),
    (10, 6, 4, 7, "probsat", &[]),
    (12, 6, 4, 1, "walksat", &[]),
    (6, 16, 2, 7, "walksat", &[3, 9]),
    (10, 10, 1, 7, "walksat", &[0, 4]),
];

fn run_all() -> Vec<String> {
    let mut out = Vec::new();
    for &(num_vars, num_clauses, formula_seed, seed, algo, removed) in CASES {
        let mut db = random_3cnf(num_vars, num_clauses, formula_seed);
        for &id in removed {
            db.remove(ClauseId::new(id));
        }
        let config = LocalSearchConfig {
            max_flips: 400,
            random_seed: seed,
            ..Default::default()
        };
        let n = num_vars as usize;
        let mut ls = LocalSearch::new(n, config);
        let (result, assignment) = if algo == "walksat" {
            ls.solve_walksat(&db, n)
        } else {
            ls.solve_probsat(&db, n)
        };
        let name = format!(
            "n{num_vars} m{num_clauses} s{formula_seed} seed{seed} {algo} removed{removed:?}"
        );
        out.push(fingerprint(&name, result, assignment, &ls));
    }
    out
}

#[test]
fn walksat_and_probsat_are_pinned() {
    let got = run_all();
    let want: &[&str] = &[
        "n4 m10 s1 seed7 walksat removed[] Sat flips=3 min_unsat=0 improvements=1 assignment=1110",
        "n5 m6 s5 seed1 probsat removed[] Sat flips=2 min_unsat=0 improvements=2 assignment=11001",
        "n5 m6 s5 seed7 walksat removed[] Sat flips=2 min_unsat=0 improvements=2 assignment=10011",
        "n6 m10 s3 seed1 walksat removed[] Sat flips=2 min_unsat=0 improvements=2 assignment=100011",
        "n6 m10 s4 seed7 walksat removed[] Sat flips=3 min_unsat=0 improvements=1 assignment=100010",
        "n8 m6 s3 seed99 walksat removed[] Sat flips=3 min_unsat=0 improvements=1 assignment=10101110",
        "n10 m6 s4 seed7 probsat removed[] Sat flips=2 min_unsat=0 improvements=2 assignment=0010100010",
        "n12 m6 s4 seed1 walksat removed[] Sat flips=2 min_unsat=0 improvements=2 assignment=101010111110",
        "n6 m16 s2 seed7 walksat removed[3, 9] Sat flips=2 min_unsat=0 improvements=2 assignment=001011",
        "n10 m10 s1 seed7 walksat removed[0, 4] Sat flips=2 min_unsat=0 improvements=1 assignment=1111101010",
    ];
    assert_eq!(got, want, "fingerprints:\n{}", got.join("\n"));
}
