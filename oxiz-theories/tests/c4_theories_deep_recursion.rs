//! Deep-nesting regression tests for the theory-layer walks that used to
//! recurse once per input nesting level.
//!
//! Every walk exercised here returns a value with **no error channel**
//! (`bool`, `usize`, `TermId`, `Ordering`, `Arc<Regex>`, `()`, or drop glue),
//! so it could never have been given an honest depth limit — the only correct
//! fix is an explicit heap stack. The assertion in each test is simply that
//! the call *returns*: a native-stack overflow aborts the whole process, so
//! reaching the assertion at all is the proof.
//!
//! Each body runs on a deliberately small worker stack — a scaled-down model of
//! what an embedder's thread typically gets — so a regression fails here long
//! before it would on the main thread. What the stack size and the nesting
//! depths pin *together* is a bytes-per-frame threshold, never either number on
//! its own; see [`WORKER_STACK`].

use oxiz_core::ast::TermId;
// Only the `nlsat` section below builds real terms; every other walk here
// constructs raw `TermId`s directly.
#[cfg(feature = "nlsat")]
use oxiz_core::ast::TermManager;
use oxiz_theories::combination::TheoryCombiner;
use oxiz_theories::euf::ProofStep;
use oxiz_theories::euf::{EMatchEngine, Pattern, QuantifiedFormula, Trigger, VarId};
#[cfg(feature = "nlsat")]
use oxiz_theories::nlsat::{TermPolyTranslator, term_is_nonlinear};
use oxiz_theories::set::{SetExpr, SetSolver, SetSort};
use oxiz_theories::string::sequence::{IntExpr, SeqExpr, SeqRewriter};
use oxiz_theories::string::{Regex, RegexOp};

/// A worker stack small enough that any surviving recursion overflows it.
///
/// This constant and every nesting depth below are scaled **together**, and
/// only their ratio — about 21 bytes of stack per nesting level — decides what
/// these tests detect. A recursive walk needs tens of bytes per frame at the
/// very least, so it still overflows; an iterative one uses O(1) native stack,
/// so it still returns. Halving the stack without halving the depths would
/// merely make the tests flaky, and raising a depth back to 100_000 without
/// also restoring a 1 MiB stack buys no extra detection power at 64× the cost
/// (several of the walks exercised here are quadratic in the depth, and at the
/// original numbers this file exhausted machine memory).
const WORKER_STACK: usize = 1 << 17;

/// Run `body` on a small dedicated stack and return its value. Returning at
/// all is the property under test.
fn on_worker_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(WORKER_STACK)
        .spawn(body)
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort")
}

// ─────────────────────────────────────────────────────────────────────────────
// string::regex — derivative, comparator, Eq/Hash, Drop
// ─────────────────────────────────────────────────────────────────────────────

/// A regex nested `levels` deep. `re.loop` is the one operator whose smart
/// constructor never flattens or collapses, so it builds genuine depth.
fn deep_regex(levels: usize) -> std::sync::Arc<Regex> {
    let mut r = Regex::char('a');
    for _ in 0..levels {
        r = Regex::loop_bounded(r, 1, Some(2));
    }
    r
}

#[test]
fn deep_regex_derivative_returns() {
    // Lower than the other regex tests: rebuilding a `Concat` at every level
    // is quadratic in the *original* algorithm too, so the depth here is
    // chosen for runtime, not because the walk needs it.
    const LEVELS: usize = 2_500;
    let nullable = on_worker_stack(|| {
        let r = deep_regex(LEVELS);
        let d = r.derivative('a');
        d.is_nullable()
    });
    // The exact answer matters less than returning, but pin it anyway:
    // D_a(a{1,2}{1,2}…) accepts the empty string at every level.
    assert!(nullable);
}

#[test]
fn deep_regex_equality_and_hash_return() {
    const LEVELS: usize = 12_500;
    let (equal, same_hash) = on_worker_stack(|| {
        use std::hash::{Hash, Hasher};
        let a = deep_regex(LEVELS);
        let b = deep_regex(LEVELS);
        let mut ha = std::collections::hash_map::DefaultHasher::new();
        let mut hb = std::collections::hash_map::DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        (a == b, ha.finish() == hb.finish())
    });
    assert!(equal, "structurally identical regexes must compare equal");
    assert!(same_hash, "equal regexes must hash equally");
}

#[test]
fn deep_regex_union_sort_comparator_returns() {
    const LEVELS: usize = 12_500;
    let is_union = on_worker_stack(|| {
        // `union` sorts its operands with the hand-written comparator, so this
        // drives the comparator over the full depth of both operands.
        let u = Regex::union(vec![deep_regex(LEVELS), Regex::char('b')]);
        matches!(u.op, RegexOp::Union(_))
    });
    assert!(is_union);
}

#[test]
fn deep_regex_drop_returns() {
    const LEVELS: usize = 12_500;
    let dropped = on_worker_stack(|| {
        let r = deep_regex(LEVELS);
        drop(r);
        true
    });
    assert!(dropped);
}

#[test]
fn shared_regex_dag_hashing_is_not_exponential() {
    // 60 doubling levels: 2^60 nodes if the DAG is walked as a tree, 60 if the
    // cached subtree hash is used. Completing at all is the assertion.
    const DOUBLINGS: usize = 60;
    let ok = on_worker_stack(|| {
        use std::hash::{Hash, Hasher};
        let mut r = Regex::char('a');
        for _ in 0..DOUBLINGS {
            r = Regex::concat(vec![Regex::loop_bounded(r.clone(), 1, Some(2)), r]);
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        r.hash(&mut h);
        let clone = r.clone();
        (h.finish() != 0 || h.finish() == 0) && *clone == *r
    });
    assert!(ok);
}

// ─────────────────────────────────────────────────────────────────────────────
// nlsat — term_is_nonlinear, TermPolyTranslator::translate
//
// This section alone is behind `#[cfg(feature = "nlsat")]`: `oxiz_theories::
// nlsat`, and the `oxiz-nlsat` crate its translator hands polynomials to, are
// compiled only under that feature (on by default). Every other walk in this
// file is unconditional, so a `--no-default-features --features std` build
// still gets the full depth guarantee for the theories it kept.
// ─────────────────────────────────────────────────────────────────────────────

/// A `((x + 1) + 1) + 1 …` chain nested `levels` deep.
#[cfg(feature = "nlsat")]
fn deep_arith_chain(manager: &mut TermManager, levels: usize) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let one = manager.mk_int(1);
    let mut term = x;
    for _ in 0..levels {
        term = manager.mk_add(vec![term, one]);
    }
    term
}

#[cfg(feature = "nlsat")]
#[test]
fn deep_arith_term_nonlinearity_check_returns() {
    const LEVELS: usize = 12_500;
    let nonlinear = on_worker_stack(|| {
        let mut manager = TermManager::new();
        let term = deep_arith_chain(&mut manager, LEVELS);
        term_is_nonlinear(term, &manager)
    });
    // A pure `+` chain is linear; the point is that the answer arrives.
    assert!(!nonlinear);
}

#[cfg(feature = "nlsat")]
#[test]
fn deep_arith_term_translation_returns() {
    const LEVELS: usize = 12_500;
    let translated = on_worker_stack(|| {
        use oxiz_nlsat::nia::NiaSolver;
        let mut manager = TermManager::new();
        let term = deep_arith_chain(&mut manager, LEVELS);
        let mut nia = NiaSolver::default();
        let mut translator = TermPolyTranslator::new(&manager, &mut nia, true);
        translator.translate(term).is_some()
    });
    assert!(translated, "a deep `+` chain is a polynomial");
}

// ─────────────────────────────────────────────────────────────────────────────
// combination — union-find `find` in extract_assignments
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn long_union_find_chain_returns() {
    // Descending-id equalities are exactly the order that builds a linear
    // parent chain, because `union` links by the smaller raw id and never by
    // rank.
    const LINKS: u32 = 25_000;
    let (len, all_same) = on_worker_stack(|| {
        let model: Vec<(TermId, TermId)> = (1..=LINKS)
            .rev()
            .map(|i| (TermId::new(i), TermId::new(i - 1)))
            .collect();
        let combiner = TheoryCombiner::new();
        let assignments = combiner.extract_assignments(&model);
        let root = TermId::new(0);
        let all_same = assignments.values().all(|&r| r == root);
        (assignments.len(), all_same)
    });
    assert_eq!(len as u32, LINKS + 1);
    assert!(
        all_same,
        "every term in one transitive chain must map to the same representative"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// set — SetSort::nesting_depth, SetExpr::get_vars / extract_var / Drop
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn deep_set_sort_nesting_depth_returns() {
    const LEVELS: usize = 25_000;
    let depth = on_worker_stack(|| {
        let mut sort = SetSort::IntSet;
        for _ in 0..LEVELS {
            sort = SetSort::SetSet(Box::new(sort));
        }
        let depth = sort.nesting_depth();
        drop(sort);
        depth
    });
    assert_eq!(depth, LEVELS, "nesting depth must be exact, not truncated");
}

#[test]
fn deep_set_expr_vars_and_drop_return() {
    const LEVELS: usize = 12_500;
    let vars = on_worker_stack(|| {
        let mut expr = SetExpr::Empty;
        for _ in 0..LEVELS {
            expr = SetExpr::complement(expr);
        }
        let vars = expr.get_vars().len();
        drop(expr);
        vars
    });
    assert_eq!(vars, 0);
}

#[test]
fn deep_set_expr_constraint_extraction_returns() {
    // `extract_var` mints one auxiliary variable per level, so keep the depth
    // modest: the recursion, not the variable count, is what is under test.
    const LEVELS: usize = 2_500;
    let ok = on_worker_stack(|| {
        let mut expr = SetExpr::Empty;
        for _ in 0..LEVELS {
            expr = SetExpr::complement(expr);
        }
        let mut solver = SetSolver::new();
        solver
            .add_constraint(oxiz_theories::set::SetConstraint::Member {
                element: 1,
                set: expr,
                sign: true,
            })
            .is_ok()
    });
    assert!(ok);
}

// ─────────────────────────────────────────────────────────────────────────────
// euf — ProofStep terms/size/reasons/Display/Drop, Trigger/EMatcher patterns
// ─────────────────────────────────────────────────────────────────────────────

/// A left-leaning `Trans` chain `levels` deep, wrapped in `Symm` at every
/// other level so the endpoint walk has to flip direction repeatedly.
fn deep_proof(levels: usize) -> ProofStep {
    let mut proof = ProofStep::Given {
        left: TermId::new(0),
        right: TermId::new(1),
        reason: 0,
    };
    for i in 0..levels {
        proof = ProofStep::Trans {
            left: Box::new(proof),
            right: Box::new(ProofStep::Refl {
                term: TermId::new(1),
            }),
        };
        if i % 2 == 0 {
            proof = ProofStep::Symm {
                proof: Box::new(proof),
            };
        }
    }
    proof
}

#[test]
fn deep_proof_queries_and_drop_return() {
    const LEVELS: usize = 12_500;
    let (size, reasons, rendered) = on_worker_stack(|| {
        let proof = deep_proof(LEVELS);
        let (_l, _r) = proof.terms();
        let size = proof.size();
        let reasons = proof.reasons().len();
        let rendered = proof.to_string().len();
        drop(proof);
        (size, reasons, rendered)
    });
    // One `Given`, plus one `Trans` and one `Refl` per level, plus one `Symm`
    // on every other level.
    assert_eq!(size, 1 + 2 * LEVELS + LEVELS.div_ceil(2));
    assert_eq!(reasons, 1, "only the single `Given` carries a reason");
    assert!(rendered > 0);
}

#[test]
fn deep_trigger_pattern_var_collection_returns() {
    const LEVELS: usize = 12_500;
    let vars = on_worker_stack(|| {
        let mut pattern = Pattern::Var(VarId::new(0));
        for _ in 0..LEVELS {
            pattern = Pattern::App {
                func: TermId::new(7),
                args: vec![pattern],
            };
        }
        Trigger::new(vec![pattern]).vars().len()
    });
    assert_eq!(vars, 1);
}

#[test]
fn deep_ematch_pattern_matching_returns() {
    const LEVELS: usize = 6_250;
    let matched = on_worker_stack(|| {
        // f_1(f_2(… f_L(x) …)) with a distinct symbol per level, so each level
        // has exactly one candidate application and the test measures depth
        // rather than the matcher's quadratic candidate scan.
        let mut engine = EMatchEngine::new();
        let mut ground = TermId::new(1);
        engine.add_ground_term(ground);
        for i in 0..LEVELS {
            let func = TermId::new(i as u32 + 2);
            let app = TermId::new(1_000_000 + i as u32);
            engine.add_app(func, app, smallvec::smallvec![ground]);
            ground = app;
        }
        let mut pattern = Pattern::Var(VarId::new(0));
        for i in 0..LEVELS {
            pattern = Pattern::App {
                func: TermId::new(i as u32 + 2),
                args: vec![pattern],
            };
        }
        engine.add_formula(QuantifiedFormula::new(
            vec![VarId::new(0)],
            TermId::new(0),
            vec![Trigger::new(vec![pattern])],
        ));
        engine.match_all();
        engine.instantiations().len()
    });
    assert!(matched > 0, "the deep pattern must match the deep term");
}

// ─────────────────────────────────────────────────────────────────────────────
// string::sequence — SeqRewriter::simplify and Drop
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn deep_sequence_simplify_and_drop_return() {
    const LEVELS: usize = 12_500;
    let simplified = on_worker_stack(|| {
        // A reversal tower `Reverse(Reverse(… Literal("ab") …))`.
        let tower = || {
            let mut expr = SeqExpr::Literal("ab".to_string());
            for _ in 0..LEVELS {
                expr = SeqExpr::Reverse(Box::new(expr));
            }
            expr
        };
        // Half one: the iterative `Drop`. `simplify` moves each operand out
        // behind a placeholder as it descends, so it only ever hands a
        // one-level shell to the drop glue -- an explicit `drop` is the only
        // way to drive `SeqExpr`'s dismantler to the full depth.
        drop(tower());
        // Half two: the iterative `simplify` over the same depth.
        let out = SeqRewriter::new().simplify(tower());
        // An even number of reversals of "ab" folds back to "ab".
        matches!(&out, SeqExpr::Literal(s) if s == "ab" || s == "ba")
    });
    assert!(simplified, "a reversal tower must fold to a literal");
}

#[test]
fn deep_int_expr_drop_returns() {
    const LEVELS: usize = 12_500;
    let dropped = on_worker_stack(|| {
        let mut expr = IntExpr::Literal(0);
        for _ in 0..LEVELS {
            expr = IntExpr::Sub(Box::new(expr), Box::new(IntExpr::Literal(1)));
        }
        drop(expr);
        true
    });
    assert!(dropped);
}
