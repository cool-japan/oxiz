use bench_profile::{parser_script, run_script, sat_propagation_script, theory_check_script};
use criterion::{Criterion, criterion_group, criterion_main};
use num_rational::Rational64;
use oxiz_core::ast::{EGraph, ENode, ENodeKind, TermId, TermManager};
use oxiz_core::profiling::{ProfilingCategory, ProfilingStats};
use oxiz_core::rewrite::{CombinedRewriter, Rewriter};
use oxiz_core::RewriteContext;
use oxiz_proof::ProofRecorder;
use oxiz_sat::{Lit, Solver as SatSolver};
use oxiz_solver::combination::coordinator::{SatResult, TheoryCoordinator, TheoryId, TheorySolver};
use oxiz_theories::arithmetic::{LinExpr, Simplex};
use oxiz_theories::array::ArraySolver;
use oxiz_theories::bv::{Constraint as BvConstraint, Interval, WordLevelPropagator};
use oxiz_theories::string::{ConstraintAutomaton, Dfa};
use oxiz_theories::Theory;
use std::hint::black_box;

fn print_snapshot(category: ProfilingCategory) {
    let snapshot = ProfilingStats::snapshot();
    println!(
        "{} => count={} total_ns={}",
        category,
        snapshot.count(category),
        snapshot.total_ns(category)
    );
}

fn bench_sat_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::SatPropagation.as_str());
    group.bench_function("solve", |b| {
        b.iter(|| {
            let mut solver = SatSolver::new();
            let a = solver.new_var();
            let b_var = solver.new_var();
            let c_var = solver.new_var();
            solver.add_clause([Lit::pos(a), Lit::pos(b_var)]);
            solver.add_clause([Lit::neg(a), Lit::pos(c_var)]);
            solver.add_clause([Lit::neg(b_var), Lit::pos(c_var)]);
            solver.add_clause([Lit::neg(c_var)]);
            black_box(solver.solve())
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::SatPropagation);
}

struct MockTheory;

impl TheorySolver for MockTheory {
    fn theory_id(&self) -> TheoryId {
        TheoryId::Arithmetic
    }

    fn assert_formula(&mut self, _formula: usize) -> Result<(), String> {
        Ok(())
    }

    fn check_sat(&mut self) -> Result<SatResult, String> {
        Ok(SatResult::Sat)
    }

    fn get_model(&self) -> Option<rustc_hash::FxHashMap<usize, usize>> {
        Some(rustc_hash::FxHashMap::default())
    }

    fn get_conflict(&self) -> Option<Vec<usize>> {
        None
    }

    fn backtrack(&mut self, _level: usize) -> Result<(), String> {
        Ok(())
    }

    fn get_implied_equalities(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }

    fn notify_equality(&mut self, _lhs: usize, _rhs: usize) -> Result<(), String> {
        Ok(())
    }
}

fn bench_theory_check(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::TheoryCheck.as_str());
    group.bench_function("coordinator", |b| {
        b.iter(|| {
            let mut coordinator = TheoryCoordinator::new(Default::default());
            coordinator.register_theory(Box::new(MockTheory));
            black_box(coordinator.check_sat())
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::TheoryCheck);
}

fn bench_egraph_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::EGraphMerge.as_str());
    group.bench_function("merge", |b| {
        b.iter(|| {
            let mut egraph = EGraph::new();
            let lhs = egraph.add(ENode {
                kind: ENodeKind::Var("x".to_string()),
                children: Vec::new(),
            });
            let rhs = egraph.add(ENode {
                kind: ENodeKind::Var("y".to_string()),
                children: Vec::new(),
            });
            black_box(egraph.merge(lhs, rhs))
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::EGraphMerge);
}

fn build_simplex() -> Simplex {
    let mut simplex = Simplex::new();
    let x0 = simplex.new_var();
    let x1 = simplex.new_var();
    simplex.set_lower(x0, Rational64::new(0, 1), 0);
    simplex.set_upper(x0, Rational64::new(2, 1), 1);
    simplex.set_lower(x1, Rational64::new(0, 1), 2);
    simplex.set_upper(x1, Rational64::new(2, 1), 3);
    let mut expr = LinExpr::new();
    expr.add_term(x0, Rational64::new(1, 1));
    expr.add_term(x1, Rational64::new(1, 1));
    expr.add_constant(Rational64::new(-3, 1));
    simplex.add_ge(expr, 4);
    simplex
}

fn bench_simplex_pivot(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::SimplexPivot.as_str());
    group.bench_function("check", |b| {
        b.iter(|| {
            let mut simplex = build_simplex();
            black_box(simplex.check())
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::SimplexPivot);
}

fn bench_bv_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::BvPropagation.as_str());
    group.bench_function("propagate", |b| {
        b.iter(|| {
            let mut propagator = WordLevelPropagator::new();
            let a = TermId::new(1);
            let b_term = TermId::new(2);
            let c_term = TermId::new(3);
            propagator.set_interval(a, Interval::new(1, 3, 8));
            propagator.set_interval(b_term, Interval::new(2, 4, 8));
            propagator.add_constraint(BvConstraint::Add(c_term, a, b_term));
            black_box(propagator.propagate())
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::BvPropagation);
}

fn bench_string_automata(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::StringAutomata.as_str());
    group.bench_function("accepts", |b| {
        b.iter(|| {
            let mut dfa = Dfa::new();
            let accepting = dfa.add_state();
            dfa.add_transition(0, 'a', accepting);
            dfa.add_default_transition(accepting, accepting);
            dfa.accepting.insert(accepting);
            let automaton = ConstraintAutomaton::from_dfa(dfa).with_prefix("a".to_string());
            black_box(automaton.accepts("aaaa"))
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::StringAutomata);
}

fn bench_array_extensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::ArrayExtensionality.as_str());
    group.bench_function("check", |b| {
        b.iter(|| {
            let mut solver = ArraySolver::new();
            let array = solver.intern_array(TermId::new(1));
            let index = solver.intern(TermId::new(2));
            let value = solver.intern(TermId::new(3));
            let store = solver.intern_store(TermId::new(4), array, index, value);
            let _ = solver.intern_select(TermId::new(5), store, index);
            black_box(solver.check())
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::ArrayExtensionality);
}

fn bench_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::ProofGeneration.as_str());
    group.bench_function("record", |b| {
        b.iter(|| {
            let mut recorder = ProofRecorder::new();
            let premise = recorder.record_input("p");
            black_box(recorder.record_derived("unit-resolution", &[premise], "p"))
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::ProofGeneration);
}

fn bench_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::Parser.as_str());
    group.bench_function("parse_script", |b| {
        b.iter(|| black_box(run_script(parser_script())))
    });
    group.finish();
    print_snapshot(ProfilingCategory::Parser);
}

fn bench_cache_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group(ProfilingCategory::CacheMiss.as_str());
    group.bench_function("rewrite_unique_terms", |b| {
        b.iter(|| {
            let mut manager = TermManager::new();
            let mut ctx = RewriteContext::new();
            let mut rewriter = CombinedRewriter::new();
            let x = manager.mk_var("x", manager.sorts.int_sort);
            for offset in 0..8 {
                let cst = manager.mk_int(offset);
                let term = manager.mk_add(vec![x, cst]);
                let _ = rewriter.rewrite(term, &mut ctx, &mut manager);
            }
            black_box(rewriter.stats().cache_misses)
        });
    });
    group.finish();
    print_snapshot(ProfilingCategory::CacheMiss);
}

fn bench_context_scripts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ContextScripts");
    group.bench_function("sat_script", |b| {
        b.iter(|| black_box(run_script(sat_propagation_script())))
    });
    group.bench_function("theory_script", |b| {
        b.iter(|| black_box(run_script(theory_check_script())))
    });
    group.finish();
}

criterion_group!(
    profile_benches,
    bench_sat_propagation,
    bench_theory_check,
    bench_egraph_merge,
    bench_simplex_pivot,
    bench_bv_propagation,
    bench_string_automata,
    bench_array_extensionality,
    bench_proof_generation,
    bench_parser,
    bench_cache_miss,
    bench_context_scripts,
);
criterion_main!(profile_benches);
