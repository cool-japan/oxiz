//! Pins the seeded stochastic local search and the totalizer encoding.
//!
//! Every expected value below was produced by the implementation these tests
//! guard, so a refactor that changes the search (the order of its random
//! draws, a skipped cache update, a different best assignment) or the clauses
//! a cardinality bound emits shows up here as a changed fingerprint.

use oxiz_opt::sls::SlsResult;
use oxiz_opt::{SlsConfig, SlsSolver, SoftClause, SoftId, Totalizer, Weight};
use oxiz_sat::Lit;

/// A small deterministic generator for clause literals (no `rand` needed).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn literal(&mut self, num_vars: u64) -> i32 {
        let var = (self.next() % num_vars) as i32 + 1;
        if self.next().is_multiple_of(2) {
            var
        } else {
            -var
        }
    }
}

fn soft(id: u32, lits: &[i32], weight: u64) -> SoftClause {
    SoftClause::new(
        SoftId(id),
        lits.iter().map(|&l| Lit::from_dimacs(l)),
        Weight::from(weight),
    )
}

fn fingerprint(solver: &SlsSolver, result: &SlsResult) -> String {
    format!(
        "{:?} cost={:?} iterations={} flips={} stats_best={:?} best={:?}",
        result,
        solver.best_cost(),
        solver.stats().iterations,
        solver.stats().flips,
        solver.stats().best_cost,
        solver.best_assignment()
    )
}

fn run(config: SlsConfig, hard: &[Vec<i32>], softs: &[SoftClause]) -> String {
    let mut solver = SlsSolver::with_config(config);
    for clause in hard {
        solver.add_hard(clause.clone());
    }
    for clause in softs {
        solver.add_soft(clause.clone());
    }
    match solver.solve() {
        Ok(result) => fingerprint(&solver, &result),
        Err(e) => format!("error: {e}"),
    }
}

fn random_instance(
    seed: u64,
    num_vars: u64,
    hard: usize,
    softs: usize,
) -> (Vec<Vec<i32>>, Vec<SoftClause>) {
    let mut lcg = Lcg(seed);
    let hard_clauses = (0..hard)
        .map(|_| (0..3).map(|_| lcg.literal(num_vars)).collect())
        .collect();
    let soft_clauses = (0..softs)
        .map(|i| {
            let len = 1 + (lcg.next() % 2) as usize;
            let lits: Vec<i32> = (0..len).map(|_| lcg.literal(num_vars)).collect();
            soft(i as u32, &lits, 1 + lcg.next() % 5)
        })
        .collect();
    (hard_clauses, soft_clauses)
}

#[test]
fn seeded_sls_search_is_pinned() {
    let small_config = |seed: u64| SlsConfig {
        max_iterations: 400,
        max_flips: 300,
        random_seed: seed,
        ..SlsConfig::default()
    };

    let mut prints = Vec::new();

    // Contradictory units: exactly one of the two must be violated.
    let contradictory = [soft(0, &[1], 1), soft(1, &[-1], 1)];
    prints.push(run(SlsConfig::default(), &[], &contradictory));

    // Hard clauses that pin a unique model, plus soft preferences against it.
    let hard = vec![vec![1, 2], vec![-1, 3], vec![-2, -3], vec![2]];
    let prefs = [soft(0, &[-2], 3), soft(1, &[1], 2), soft(2, &[-3, 1], 1)];
    for seed in [1, 7, 42] {
        prints.push(run(small_config(seed), &hard, &prefs));
    }

    // Two random 3-SAT-shaped instances, satisfiable and over-constrained.
    for (seed, vars, hard_count, soft_count) in [(11, 8, 20, 10), (29, 6, 40, 12)] {
        let (hard, softs) = random_instance(seed, vars, hard_count, soft_count);
        for config_seed in [3, 1234] {
            prints.push(run(small_config(config_seed), &hard, &softs));
        }
    }

    let expected: &[&str] = &[
        "Satisfiable cost=Int(1) iterations=10000 flips=10000 stats_best=Some(Int(1)) best=Some(Assignment { values: [false, false] })",
        "Satisfiable cost=Int(5) iterations=400 flips=129 stats_best=Some(Int(5)) best=Some(Assignment { values: [false, false, true, false] })",
        "Satisfiable cost=Int(3) iterations=400 flips=177 stats_best=Some(Int(3)) best=Some(Assignment { values: [true, true, true, false] })",
        "Satisfiable cost=Int(3) iterations=400 flips=177 stats_best=Some(Int(3)) best=Some(Assignment { values: [false, false, false, true] })",
        "Satisfiable cost=Int(7) iterations=400 flips=161 stats_best=Some(Int(7)) best=Some(Assignment { values: [true, false, false, false, true, false, false, true, false] })",
        "Satisfiable cost=Int(7) iterations=400 flips=171 stats_best=Some(Int(7)) best=Some(Assignment { values: [true, true, true, false, true, true, false, false, false] })",
        "Satisfiable cost=Int(15) iterations=400 flips=0 stats_best=Some(Int(15)) best=Some(Assignment { values: [true, true, true, true, true, true, true] })",
        "Satisfiable cost=Int(6) iterations=400 flips=0 stats_best=Some(Int(6)) best=Some(Assignment { values: [true, true, false, true, false, true, true] })",
    ];
    assert_eq!(prints, expected, "SLS fingerprints changed");
}

fn totalizer_fingerprint(inputs: usize, bounds: &[usize]) -> String {
    let lits: Vec<Lit> = (1..=inputs as i32).map(Lit::from_dimacs).collect();
    let mut totalizer = Totalizer::new(&lits, inputs as u32 + 1);
    let mut out = format!("n={inputs} size={}", totalizer.size());
    for &k in bounds {
        let output = totalizer.ensure_bound(k).map(|l| l.to_dimacs());
        let clauses: Vec<Vec<i32>> = totalizer
            .take_clauses()
            .iter()
            .map(|c| c.lits.iter().map(|l| l.to_dimacs()).collect())
            .collect();
        out.push_str(&format!(
            " | k={k} out={output:?} at_least={:?} at_most={:?} next_var={} clauses={clauses:?}",
            totalizer.at_least(k).map(|l| l.to_dimacs()),
            totalizer.at_most(k).map(|l| l.to_dimacs()),
            totalizer.next_var()
        ));
    }
    out
}

#[test]
fn totalizer_clauses_are_pinned() {
    let prints: Vec<String> = [
        (0, vec![1]),
        (1, vec![1, 2]),
        (2, vec![1, 2]),
        (3, vec![1, 3, 2]),
        (5, vec![2, 1, 4, 5]),
        (8, vec![3, 8]),
    ]
    .into_iter()
    .map(|(n, bounds)| totalizer_fingerprint(n, &bounds))
    .collect();

    let expected: &[&str] = &[
        "n=0 size=0 | k=1 out=None at_least=None at_most=None next_var=1 clauses=[]",
        "n=1 size=1 | k=1 out=Some(1) at_least=Some(1) at_most=None next_var=2 clauses=[] | k=2 out=None at_least=None at_most=None next_var=2 clauses=[]",
        "n=2 size=2 | k=1 out=Some(4) at_least=Some(4) at_most=None next_var=4 clauses=[[-2, 4], [-1, 4]] | k=2 out=Some(5) at_least=Some(5) at_most=None next_var=5 clauses=[[-1, -2, 5]]",
        "n=3 size=3 | k=1 out=Some(6) at_least=Some(6) at_most=None next_var=6 clauses=[[-2, 5], [-1, 5], [-3, 6], [-5, 6]] | k=3 out=Some(9) at_least=Some(9) at_most=None next_var=9 clauses=[[-1, -2, 7], [-5, -3, 8], [-7, 8], [-7, -3, 9]] | k=2 out=Some(8) at_least=Some(8) at_most=Some(-9) next_var=9 clauses=[]",
        "n=5 size=5 | k=2 out=Some(14) at_least=Some(14) at_most=None next_var=14 clauses=[[-2, 7], [-1, 7], [-1, -2, 8], [-4, 9], [-3, 9], [-3, -4, 10], [-9, 11], [-7, 11], [-10, 12], [-7, -9, 12], [-8, 12], [-5, 13], [-11, 13], [-11, -5, 14], [-12, 14]] | k=1 out=Some(13) at_least=Some(13) at_most=Some(-14) next_var=14 clauses=[] | k=4 out=Some(18) at_least=Some(18) at_most=None next_var=18 clauses=[[-7, -10, 15], [-8, -9, 15], [-8, -10, 16], [-12, -5, 17], [-15, 17], [-15, -5, 18], [-16, 18]] | k=5 out=Some(19) at_least=Some(19) at_most=None next_var=19 clauses=[[-16, -5, 19]]",
        "n=8 size=8 | k=3 out=Some(26) at_least=Some(26) at_most=None next_var=26 clauses=[[-2, 10], [-1, 10], [-1, -2, 11], [-4, 12], [-3, 12], [-3, -4, 13], [-12, 14], [-10, 14], [-13, 15], [-10, -12, 15], [-11, 15], [-10, -13, 16], [-11, -12, 16], [-6, 17], [-5, 17], [-5, -6, 18], [-8, 19], [-7, 19], [-7, -8, 20], [-19, 21], [-17, 21], [-20, 22], [-17, -19, 22], [-18, 22], [-17, -20, 23], [-18, -19, 23], [-21, 24], [-14, 24], [-22, 25], [-14, -21, 25], [-15, 25], [-23, 26], [-14, -22, 26], [-15, -21, 26], [-16, 26]] | k=8 out=Some(33) at_least=Some(33) at_most=None next_var=33 clauses=[[-11, -13, 27], [-18, -20, 28], [-28, 29], [-14, -23, 29], [-15, -22, 29], [-16, -21, 29], [-27, 29], [-14, -28, 30], [-15, -23, 30], [-16, -22, 30], [-27, -21, 30], [-15, -28, 31], [-16, -23, 31], [-27, -22, 31], [-16, -28, 32], [-27, -23, 32], [-27, -28, 33]]",
    ];
    assert_eq!(prints, expected, "totalizer fingerprints changed");
}
