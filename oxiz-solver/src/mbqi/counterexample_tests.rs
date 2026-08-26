//! Unit tests for [`super::CounterExampleGenerator`].
//!
//! Split out into its own file (rather than an inline `mod tests` at the
//! bottom of `counterexample.rs`) purely to keep `counterexample.rs` under
//! the workspace's 2000-line-per-file ceiling after the iterative-evaluator
//! conversion -- `#[path = "counterexample_tests.rs"] mod tests;` keeps this
//! at the same `super::counterexample` module position it always occupied,
//! so every private-item access below still resolves exactly as before.
use super::*;

// ===== Substitution regression tests =====
//
// `apply_substitution` used to be a local recursive walk whose `TermKind`
// whitelist ended in `_ => term`, so a bound variable under any unlisted
// kind survived into the supposedly ground instance. All four copies in
// this module now delegate to `crate::mbqi::macros::utils::substitute`.

/// A body whose only variable occurrence sits under a kind the old
/// whitelist missed must still be substituted.
#[test]
fn substitution_reaches_kinds_outside_the_old_whitelist() {
    let mut m = TermManager::new();
    let bool_sort = m.sorts.bool_sort;
    let int_sort = m.sorts.int_sort;
    let bv8 = m.sorts.bitvec(8);

    let p = m.mk_var("p", bool_sort);
    let i = m.mk_var("i", int_sort);
    let b = m.mk_var("b", bv8);

    let p_name = match m.get(p).map(|t| &t.kind) {
        Some(TermKind::Var(n)) => *n,
        _ => panic!("p is a variable"),
    };
    let i_name = match m.get(i).map(|t| &t.kind) {
        Some(TermKind::Var(n)) => *n,
        _ => panic!("i is a variable"),
    };
    let b_name = match m.get(b).map(|t| &t.kind) {
        Some(TermKind::Var(n)) => *n,
        _ => panic!("b is a variable"),
    };

    let truth = m.mk_true();
    let two = m.mk_int(2);
    let ones = m.mk_bitvec(1, 8);

    // Every one of these was returned unchanged by the old walk.
    let xor = m.mk_xor(p, truth);
    let distinct = m.mk_distinct([i, two]);
    let bv_lt = m.mk_bv_ult(b, ones);
    let implies = m.mk_implies(p, truth);
    let nested = m.mk_forall([("z", int_sort)], distinct);

    let mut subst: FxHashMap<Spur, TermId> = FxHashMap::default();
    subst.insert(p_name, truth);
    subst.insert(i_name, two);
    subst.insert(b_name, ones);

    let subject = CounterExampleGenerator::new();
    for (label, term) in [
        ("xor", xor),
        ("distinct", distinct),
        ("bvult", bv_lt),
        ("implies", implies),
        ("nested forall", nested),
    ] {
        let result = subject.apply_substitution(term, &subst, &mut m);
        let free = m.free_vars_including_patterns(result);
        assert!(
            free.is_empty(),
            "{label}: substitution left free variables {free:?} in the result"
        );
    }
}

#[test]
fn test_counterexample_creation() {
    let cex = CounterExample::new(TermId::new(1), FxHashMap::default(), vec![], 0);
    assert_eq!(cex.quantifier, TermId::new(1));
    assert_eq!(cex.quality, 1.0);
}

#[test]
fn test_cex_generator_creation() {
    let generator = CounterExampleGenerator::new();
    assert_eq!(generator.max_cex_per_quantifier, 5);
    assert_eq!(generator.max_candidates_per_var, 10);
}

#[test]
fn test_cex_generator_with_limits() {
    let generator = CounterExampleGenerator::with_limits(10, 20, Duration::from_secs(2));
    assert_eq!(generator.max_cex_per_quantifier, 10);
    assert_eq!(generator.max_candidates_per_var, 20);
    assert_eq!(generator.max_search_time, Duration::from_secs(2));
}

#[test]
fn test_enumerate_combinations_empty() {
    let generator = CounterExampleGenerator::new();
    let combos = generator.enumerate_combinations(&[], 10, 100);
    assert_eq!(combos.len(), 1);
    assert!(combos[0].is_empty());
}

#[test]
fn test_enumerate_combinations_single() {
    let generator = CounterExampleGenerator::new();
    let candidates = vec![vec![TermId::new(1), TermId::new(2)]];
    let combos = generator.enumerate_combinations(&candidates, 10, 100);
    assert_eq!(combos.len(), 2);
}

#[test]
fn test_enumerate_combinations_multiple() {
    let generator = CounterExampleGenerator::new();
    let candidates = vec![
        vec![TermId::new(1), TermId::new(2)],
        vec![TermId::new(3), TermId::new(4)],
    ];
    let combos = generator.enumerate_combinations(&candidates, 10, 100);
    assert_eq!(combos.len(), 4); // 2 * 2
}

#[test]
fn test_enumerate_combinations_limit() {
    let generator = CounterExampleGenerator::new();
    let candidates = vec![
        vec![TermId::new(1), TermId::new(2), TermId::new(3)],
        vec![TermId::new(4), TermId::new(5), TermId::new(6)],
    ];
    let combos = generator.enumerate_combinations(&candidates, 10, 5);
    assert!(combos.len() <= 5);
}

#[test]
fn test_cex_stats_display() {
    let stats = CexStats {
        num_searches: 10,
        num_counterexamples_found: 5,
        num_combinations_tried: 100,
        num_timeouts: 1,
        total_time: Duration::from_millis(500),
    };
    let display = format!("{}", stats);
    assert!(display.contains("Searches: 10"));
    assert!(display.contains("CEX found: 5"));
}

#[test]
fn test_refinement_strategy() {
    assert_ne!(
        RefinementStrategy::None,
        RefinementStrategy::BlockCounterexamples
    );
}

// ---------------------------------------------------------------------
// Audit regression: Euclidean div/mod (solver-p3b, finding #3)
// ---------------------------------------------------------------------

#[test]
fn test_audit_euclidean_div_rem_helper() {
    // SMT-LIB Euclidean semantics: 0 <= r < |b|.
    let cases = [
        (7i64, 2i64, 3i64, 1i64),
        (-7, 2, -4, 1),
        (7, -2, -3, 1),
        (-7, -2, 4, 1),
        (6, 3, 2, 0),
        (-6, 3, -2, 0),
        (0, 5, 0, 0),
    ];
    for (a, b, eq, er) in cases {
        let (q, r) = euclidean_div_rem(&BigInt::from(a), &BigInt::from(b));
        assert_eq!(q, BigInt::from(eq), "div({a},{b})");
        assert_eq!(r, BigInt::from(er), "mod({a},{b})");
        // Verify the defining identity and remainder range.
        assert_eq!(BigInt::from(b) * &q + &r, BigInt::from(a));
        assert!(r >= BigInt::from(0) && r < BigInt::from(b.abs()));
    }
}

#[test]
fn test_audit_eval_div_mod_negative_euclidean() {
    let generator = CounterExampleGenerator::new();
    let mut manager = TermManager::new();
    let neg7 = manager.mk_int(BigInt::from(-7));
    let two = manager.mk_int(BigInt::from(2));

    let d = generator.eval_div(neg7, two, &mut manager);
    let m = generator.eval_modulo(neg7, two, &mut manager);

    // (div -7 2) = -4 (Euclidean), NOT -3 (truncated).
    assert!(
        matches!(manager.get(d).map(|t| &t.kind),
            Some(TermKind::IntConst(v)) if *v == BigInt::from(-4)),
        "eval_div(-7,2) must be Euclidean -4, got {:?}",
        manager.get(d).map(|t| t.kind.clone())
    );
    // (mod -7 2) = 1 (non-negative), NOT -1.
    assert!(
        matches!(manager.get(m).map(|t| &t.kind),
            Some(TermKind::IntConst(v)) if *v == BigInt::from(1)),
        "eval_modulo(-7,2) must be Euclidean 1, got {:?}",
        manager.get(m).map(|t| t.kind.clone())
    );
}

// ===== Iterative-evaluator regression tests =====
//
// `evaluate_under_model_cached` and the inline `Exists` search used to be a
// pair of mutually recursive functions; they now run as one explicit-stack
// frame machine. These tests pin (a) exact evaluation results through every
// frame kind, proving the conversion behavior-preserving, (b) that deep
// inputs return at all on a deliberately small thread stack (a stack
// overflow would abort the whole process, so returning is the proof), and
// (c) that the memo cache still bounds work on shared DAGs.

#[test]
fn evaluate_semantic_pins_through_every_frame_kind() {
    let generator = CounterExampleGenerator::new();
    let mut m = TermManager::new();
    let mut model = CompletedModel::new();
    let int_sort = m.sorts.int_sort;
    let bool_sort = m.sorts.bool_sort;

    let one = m.mk_int(1);
    let two = m.mk_int(2);
    let three = m.mk_int(3);
    let four = m.mk_int(4);
    let six = m.mk_int(6);
    let neg3 = m.mk_int(-3);
    let neg4 = m.mk_int(-4);
    let neg7 = m.mk_int(-7);
    let tt = m.mk_true();
    let ff = m.mk_false();

    // Symbolic variables resolved through the model (nothing folds at
    // construction because the operands are variables).
    let p = m.mk_var("p", bool_sort);
    let q = m.mk_var("q", bool_sort);
    let x = m.mk_var("x", int_sort);
    let y = m.mk_var("y", int_sort);
    model.set(p, tt);
    model.set(q, ff);
    model.set(x, three);
    model.set(y, neg7);

    let not_q = m.mk_not(q);
    let and_pq = m.mk_and([p, q]);
    let or_qp = m.mk_or([q, p]);
    let imp_pq = m.mk_implies(p, q);
    let imp_qp = m.mk_implies(q, p);
    let ite_p = m.mk_ite(p, one, two);
    let ite_q = m.mk_ite(q, one, two);
    let eq_x3 = m.mk_eq(x, three);
    let lt_x2 = m.mk_lt(x, two);
    let le_x3 = m.mk_le(x, three);
    let gt_x2 = m.mk_gt(x, two);
    let ge_x4 = m.mk_ge(x, four);
    let add_x1 = m.mk_add([x, one]);
    let sub_x1 = m.mk_sub(x, one);
    let mul_x2 = m.mk_mul([x, two]);
    let neg_x = m.mk_neg(x);
    let div_y2 = m.mk_div(y, two);
    let mod_y2 = m.mk_mod(y, two);

    let expectations = [
        (not_q, tt),
        (and_pq, ff),
        (or_qp, tt),
        (imp_pq, ff),
        (imp_qp, tt),
        (ite_p, one),
        (ite_q, two),
        (eq_x3, tt),
        (lt_x2, ff),
        (le_x3, tt),
        (gt_x2, tt),
        (ge_x4, ff),
        (add_x1, four),
        (sub_x1, two),
        (mul_x2, six),
        (neg_x, neg3),
        // Euclidean division/modulo: (div -7 2) = -4, (mod -7 2) = 1.
        (div_y2, neg4),
        (mod_y2, one),
    ];
    for (term, expected) in expectations {
        let result = generator.evaluate_under_model(term, &model, &mut m);
        assert_eq!(
            result, expected,
            "term {term:?} must evaluate exactly as the recursive version did"
        );
    }

    // A variable without a model value stays symbolic; Forall stays symbolic.
    let free = m.mk_var("free_var", int_sort);
    assert_eq!(generator.evaluate_under_model(free, &model, &mut m), free);
    let f_free = m.mk_apply("f", [free], int_sort);
    let body = m.mk_eq(f_free, one);
    let fa = m.mk_forall([("z", int_sort)], body);
    assert_eq!(generator.evaluate_under_model(fa, &model, &mut m), fa);

    // Apply: arguments are evaluated first, then the rebuilt application is
    // probed in the model (f(x) with x = 3 resolves through f(3)).
    let f_of_3 = m.mk_apply("f", [three], int_sort);
    let forty_two = m.mk_int(42);
    model.set(f_of_3, forty_two);
    let f_of_x = m.mk_apply("f", [x], int_sort);
    assert_eq!(
        generator.evaluate_under_model(f_of_x, &model, &mut m),
        forty_two
    );

    // Select try-1: `select(original_array, evaluated_index)` is probed
    // before the array itself is evaluated.
    let arr_sort = m.sorts.array(int_sort, int_sort);
    let arr = m.mk_var("arr", arr_sort);
    let sel_a3 = m.mk_select(arr, three);
    let nine = m.mk_int(9);
    model.set(sel_a3, nine);
    let sel_ax = m.mk_select(arr, x);
    assert_eq!(generator.evaluate_under_model(sel_ax, &model, &mut m), nine);
}

#[test]
fn evaluate_exists_pins_witness_no_witness_and_symbolic() {
    let generator = CounterExampleGenerator::new();
    let mut m = TermManager::new();
    let model = CompletedModel::new();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let tt = m.mk_true();
    let ff = m.mk_false();

    // 3 is among the default Int candidates (-2..=5): witness found.
    let three = m.mk_int(3);
    let eq3 = m.mk_eq(x, three);
    let exists3 = m.mk_exists([("x", int_sort)], eq3);
    assert_eq!(generator.evaluate_under_model(exists3, &model, &mut m), tt);

    // 100 is not: every candidate evaluates the body to False, which the
    // (unchanged) verdict rule reports as a proven-False exists.
    let hundred = m.mk_int(100);
    let eq100 = m.mk_eq(x, hundred);
    let exists100 = m.mk_exists([("x", int_sort)], eq100);
    assert_eq!(
        generator.evaluate_under_model(exists100, &model, &mut m),
        ff
    );

    // A body that stays symbolic for some candidate keeps the exists
    // symbolic: the verdict is the body itself, exactly as before.
    let u = m.mk_var("u", int_sort); // no model value
    let eq_u = m.mk_eq(x, u);
    let exists_u = m.mk_exists([("x", int_sort)], eq_u);
    assert_eq!(
        generator.evaluate_under_model(exists_u, &model, &mut m),
        eq_u
    );
}

/// Deep-nesting regression: 12 500 implication levels on a 128 KiB stack.
/// The old evaluator recursed once per level and would overflow here; the
/// frame machine must return (returning at all is the proof — an overflow
/// aborts the process).
#[test]
fn evaluate_deep_implies_chain_returns_on_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const DEPTH: usize = 12_500;

    std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let generator = CounterExampleGenerator::new();
            let mut m = TermManager::new();
            let model = CompletedModel::new();
            let bool_sort = m.sorts.bool_sort;
            let p = m.mk_var("p", bool_sort);
            let q = m.mk_var("q", bool_sort);
            let mut chain = q;
            for _ in 0..DEPTH {
                chain = m.mk_implies(p, chain);
            }
            // Everything is symbolic under the empty model, so the machine
            // rebuilds the identical hash-consed chain.
            let result = generator.evaluate_under_model(chain, &model, &mut m);
            assert_eq!(result, chain);
        })
        .expect("spawn deep-evaluation thread")
        .join()
        .expect("deep implies chain must evaluate without overflowing");
}

/// The old mutual-recursion edge: a chain of nested existentials, each level
/// of which re-entered the evaluator through the exists search (two native
/// frames per level; at this depth that overflowed a 1 MiB stack).  The stack
/// stays at 1 MiB here, unlike the scaled-down deep tests around it, because
/// the depth is already far below the level where construction cost matters.
/// Depth is 2_000 rather than 50k+ because *auxiliary test construction* is the
/// bottleneck, not the machine: each level's candidate substitution walks
/// the remaining O(depth) subterm to prove the variable absent, making
/// setup quadratic in depth.  The machine itself holds one heap frame per
/// level and no native frames.
#[test]
fn evaluate_deeply_nested_exists_returns_on_small_stack() {
    // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — depth is
    // 2_000 (sub-10,000), well under the scaling threshold used elsewhere
    // in this crate. See TODO.md "v0.3.2 backlog".
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let generator = CounterExampleGenerator::new();
            let mut m = TermManager::new();
            let model = CompletedModel::new();
            let int_sort = m.sorts.int_sort;
            let x0 = m.mk_var("x0", int_sort);
            let zero = m.mk_int(0);
            let tt = m.mk_true();
            let mut term = m.mk_eq(x0, zero);
            for i in 0..2_000 {
                let name = format!("x{i}");
                term = m.mk_exists([(name.as_str(), int_sort)], term);
            }
            // The innermost witness x0 = 0 certifies every enclosing level.
            let result = generator.evaluate_under_model(term, &model, &mut m);
            assert_eq!(result, tt);
        })
        .expect("spawn nested-exists thread")
        .join()
        .expect("nested exists chain must evaluate without overflowing");
}

/// Shared-DAG regression: 60 doubling levels reference each subterm twice
/// (2^60 paths).  The memo cache must bound the machine to one visit per
/// distinct term; the exact folded value pins the arithmetic.
#[test]
fn evaluate_shared_dag_add_doubling_is_memoized_and_exact() {
    let generator = CounterExampleGenerator::new();
    let mut m = TermManager::new();
    let model = CompletedModel::new();
    let one = m.mk_int(1);
    let mut t = one;
    for _ in 0..60 {
        let prev = t;
        t = m.mk_add([prev, prev]);
        assert_ne!(t, prev, "doubling must build a fresh Add node");
    }
    let result = generator.evaluate_under_model(t, &model, &mut m);
    let expected = m.mk_int(BigInt::from(1u128 << 60));
    assert_eq!(result, expected);
}

// ===== `CounterExample::term_size` regression tests =====
//
// `term_size` used to recurse once per nesting level and return a plain
// `usize` (no error channel); it is now an explicit-stack DFS.

/// Semantic pin: distinct-subterm counting, including the shared-subterm
/// rule (a repeated subterm is counted once) and the non-descending kinds.
#[test]
fn term_size_pins_distinct_subterm_count() {
    let mut m = TermManager::new();
    let int_sort = m.sorts.int_sort;
    let x = m.mk_var("x", int_sort);
    let one = m.mk_int(1);

    let cex = CounterExample::new(TermId::new(1), FxHashMap::default(), vec![], 0);

    // A leaf counts as 1.
    assert_eq!(cex.term_size(x, &m), 1);

    // Eq(x, 1): the node plus its two distinct children.
    let eq = m.mk_eq(x, one);
    assert_eq!(cex.term_size(eq, &m), 3);

    // Not(Eq(x, 1)): one more node on top.
    let not_eq = m.mk_not(eq);
    assert_eq!(cex.term_size(not_eq, &m), 4);

    // And(Eq, Not(Eq)): the shared `Eq` subterm is counted exactly once,
    // so this is `And` + `Eq` + `x` + `1` + `Not` = 5.
    let conj = m.mk_and([eq, not_eq]);
    assert_eq!(cex.term_size(conj, &m), 5);

    // `Add` is not one of the descended kinds: only the node itself counts.
    let add = m.mk_add([x, one]);
    assert_eq!(cex.term_size(add, &m), 1);
}

/// Deep-nesting regression: 12 500 `Eq` levels on a 128 KiB stack.  The old
/// recursive helper would overflow here; returning at all is the proof
/// (an overflow aborts the process).
#[test]
fn term_size_deep_eq_chain_returns_on_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const DEPTH: usize = 12_500;

    std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut m = TermManager::new();
            let bool_sort = m.sorts.bool_sort;
            let p = m.mk_var("p", bool_sort);
            let q = m.mk_var("q", bool_sort);
            // `Eq` nesting (rather than `Not`, which folds double negation
            // away and so never gains depth) builds `DEPTH` real levels.
            let mut chain = q;
            for _ in 0..DEPTH {
                let next = m.mk_eq(chain, p);
                assert_ne!(next, chain, "Eq nesting must add a level");
                chain = next;
            }
            let cex = CounterExample::new(TermId::new(1), FxHashMap::default(), vec![], 0);
            // `DEPTH` `Eq` nodes plus the two distinct leaves.
            assert_eq!(cex.term_size(chain, &m), DEPTH + 2);
        })
        .expect("spawn deep term_size thread")
        .join()
        .expect("deep Eq chain must be sized without overflowing");
}

/// Shared-DAG regression: 60 doubling levels of `And` reference each
/// subterm twice (2^60 paths).  The visited set must bound the walk to one
/// visit per distinct term, and the count pins that exactly.
#[test]
fn term_size_shared_dag_doubling_is_linear() {
    let mut m = TermManager::new();
    let bool_sort = m.sorts.bool_sort;
    let p = m.mk_var("p", bool_sort);
    let q = m.mk_var("q", bool_sort);
    let mut t = m.mk_and([p, q]);
    let mut nodes = 3usize; // And + p + q
    for _ in 0..60 {
        let prev = t;
        let not_prev = m.mk_not(prev);
        t = m.mk_and([prev, not_prev]);
        assert_ne!(t, prev, "doubling must build a fresh And node");
        nodes += 2; // the new Not and the new And
    }
    let cex = CounterExample::new(TermId::new(1), FxHashMap::default(), vec![], 0);
    assert_eq!(cex.term_size(t, &m), nodes);
}
