//! Tests for `ast/normal_forms` and its submodules.
//!
//! Split out for the same reason as `ast/manager/query/tests.rs`: keeps the
//! implementation files themselves well under the workspace's line-count
//! ceiling.

use super::*;
use crate::ast::TermManager;

/// Run `f` to completion on a dedicated thread with a 128 KiB stack --
/// deliberately far smaller than the default (several-MiB) main-thread
/// stack -- and return whatever it returns. Mirrors
/// `ast/manager/query/tests.rs`'s `run_on_small_stack`: a stack overflow
/// aborts the whole process rather than failing a single test gracefully,
/// so for the deep-nesting tests below, the call *returning at all* is
/// itself part of what is being asserted.
fn run_on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(f)
        .expect("spawning the constrained-stack test thread should succeed")
        .join()
        .expect("the constrained-stack thread must not panic")
}

/// Build a chain of `depth` nested `Not`s around `leaf`, via `intern`
/// directly (bypassing `mk_not`'s double-negation simplification, which
/// would otherwise collapse the chain instead of building it), built
/// iteratively (never recursively, which would overflow before the
/// assertion under test even runs).
fn deep_not_chain(manager: &mut TermManager, leaf: TermId, depth: usize) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    let mut term = leaf;
    for _ in 0..depth {
        term = manager.intern(TermKind::Not(term), bool_sort);
    }
    term
}

/// Build a chain of `depth` nested `And`s around `leaf`: `And(x_0, And(x_1,
/// And(x_2, ... leaf)))`, each `x_i` a fresh, distinctly-named Boolean
/// variable, built iteratively.
fn deep_and_chain(manager: &mut TermManager, leaf: TermId, depth: usize) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    let mut term = leaf;
    for i in (0..depth).rev() {
        let x_i = manager.mk_var(&format!("and_chain_{i}"), bool_sort);
        // Built via `intern` directly rather than `mk_and`: `mk_and`
        // flattens a nested `And` argument into its own arg list
        // (`TermKind::And(inner) => flat_args.extend(inner...)`), which
        // would make each of these `depth` construction steps copy the
        // *entire* chain built so far -- quadratic in `depth` just to
        // build the fixture, before the function under test ever runs.
        // `intern` skips that flattening, keeping construction linear and
        // the chain genuinely (deeply) nested rather than a flat `And` of
        // `depth` arguments.
        term = manager.intern(TermKind::And([x_i, term].into_iter().collect()), bool_sort);
    }
    term
}

/// Build a chain of `depth` nested `Forall`s around `leaf`, all distinctly
/// named, built iteratively: `forall x_0. forall x_1. ... forall x_{n-1}.
/// leaf`.
fn deep_forall_chain(manager: &mut TermManager, leaf: TermId, depth: usize) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let mut term = leaf;
    for i in (0..depth).rev() {
        let name = format!("forall_chain_{i}");
        term = manager.mk_forall([(name.as_str(), int_sort)], term);
    }
    term
}

// ===========================================================================
// Shallow behaviour-preservation tests: pin the exact output of each public
// entry point on a small, hand-computable formula.
// ===========================================================================

mod shallow_pinned_outputs {
    use super::*;

    #[test]
    fn to_cnf_distributes_or_over_and() {
        // (a and b) or c  ==>  (a or c) and (b or c)
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let c = m.mk_var("c", bool_sort);
        let and_ab = m.mk_and([a, b]);
        let formula = m.mk_or([and_ab, c]);

        let result = to_cnf(formula, &mut m);

        let clause_ac = m.mk_or([a, c]);
        let clause_bc = m.mk_or([b, c]);
        let expected = m.mk_and([clause_ac, clause_bc]);
        assert_eq!(result, expected);
        assert!(
            is_cnf(result, &m),
            "to_cnf's own output must satisfy is_cnf"
        );
    }

    #[test]
    fn to_cnf_eliminates_implication_and_pushes_negation() {
        // not(a -> b)  ==>  a and not(b)
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let implies = m.mk_implies(a, b);
        let formula = m.mk_not(implies);

        let result = to_cnf(formula, &mut m);

        let not_b = m.mk_not(b);
        let expected = m.mk_and([a, not_b]);
        assert_eq!(result, expected);
    }

    #[test]
    fn to_dnf_distributes_and_over_or() {
        // (a or b) and c  ==>  (a and c) or (b and c)
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let c = m.mk_var("c", bool_sort);
        let or_ab = m.mk_or([a, b]);
        let formula = m.mk_and([or_ab, c]);

        let result = to_dnf(formula, &mut m);

        let term_ac = m.mk_and([a, c]);
        let term_bc = m.mk_and([b, c]);
        let expected = m.mk_or([term_ac, term_bc]);
        assert_eq!(result, expected);
        assert!(
            is_dnf(result, &m),
            "to_dnf's own output must satisfy is_dnf"
        );
    }

    #[test]
    fn to_nnf_pushes_negation_through_and_eliminates_implication() {
        // not((a and b) -> c)  ==>  not(not(a and b) or c) ==> (a and b) and not(c)
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let c = m.mk_var("c", bool_sort);
        let and_ab = m.mk_and([a, b]);
        let implies = m.mk_implies(and_ab, c);
        let formula = m.mk_not(implies);

        let result = to_nnf(formula, &mut m);

        let and_ab_again = m.mk_and([a, b]);
        let not_c = m.mk_not(c);
        let expected = m.mk_and([and_ab_again, not_c]);
        assert_eq!(result, expected);
        assert!(
            is_nnf(result, &m),
            "to_nnf's own output must satisfy is_nnf"
        );
    }

    #[test]
    fn to_nnf_expands_xor() {
        // a xor b  ==>  (a or b) and (not(a) or not(b))
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let formula = m.mk_xor(a, b);

        let result = to_nnf(formula, &mut m);

        let clause1 = m.mk_or([a, b]);
        let not_a = m.mk_not(a);
        let not_b = m.mk_not(b);
        let clause2 = m.mk_or([not_a, not_b]);
        let expected = m.mk_and([clause1, clause2]);
        assert_eq!(result, expected);
    }

    #[test]
    fn simplify_boolean_removes_duplicate_conjuncts() {
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let formula = m.intern(TermKind::And([a, a].into_iter().collect()), bool_sort);

        let result = simplify_boolean(formula, &mut m);

        assert_eq!(result, a, "And(a, a) must simplify down to just a");
    }

    #[test]
    fn skolemize_replaces_positive_exists_with_skolem_constant() {
        // exists x. P(x)  ==>  P(sk!0)
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let p_x = m.mk_apply("P", [x], bool_sort);
        let formula = m.mk_exists([("x", int_sort)], p_x);

        let result = skolemize(formula, &mut m);

        let sk0 = m.mk_var("sk!0", int_sort);
        let expected = m.mk_apply("P", [sk0], bool_sort);
        assert_eq!(result, expected);
    }

    #[test]
    fn skolemize_gives_skolem_function_the_governing_universal_as_argument() {
        // forall y. exists x. P(x, y)  ==>  forall y. P(sk!0(y), y)
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let p_x_y = m.mk_apply("P", [x, y], bool_sort);
        let inner = m.mk_exists([("x", int_sort)], p_x_y);
        let formula = m.mk_forall([("y", int_sort)], inner);

        let result = skolemize(formula, &mut m);

        let y2 = m.mk_var("y", int_sort);
        let sk0_y = m.mk_apply("sk!0", [y2], int_sort);
        let p_sk0y_y = m.mk_apply("P", [sk0_y, y2], bool_sort);
        let expected = m.mk_forall([("y", int_sort)], p_sk0y_y);
        assert_eq!(result, expected);
    }

    #[test]
    fn skolemize_keeps_negative_exists_as_forall_and_does_not_skolemize_it() {
        // not(exists x. P(x))  ==>  not(forall x. P(x))'s body still bound:
        // the Exists node's own polarity is negative, so it is kept as a
        // (still-an-Exists) binder wrapping a recursively-Skolemized body,
        // per `skolemize_polar`'s doc comment -- here the body has nothing
        // further to Skolemize, so it is untouched.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let p_x = m.mk_apply("P", [x], bool_sort);
        let exists = m.mk_exists([("x", int_sort)], p_x);
        let formula = m.mk_not(exists);

        let result = skolemize(formula, &mut m);

        let expected = m.mk_not(exists);
        assert_eq!(
            result, expected,
            "a negatively-polarized exists must be kept as a binder, not Skolemized"
        );
    }

    #[test]
    fn eliminate_universal_quantifiers_replaces_bound_var_with_fresh_constant() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(x, zero);
        let formula = m.mk_forall([("x", int_sort)], body);

        let result = eliminate_universal_quantifiers(formula, &mut m);

        let u0 = m.mk_var("u_0", int_sort);
        let expected = m.mk_gt(u0, zero);
        assert_eq!(result, expected);
    }

    #[test]
    fn is_cnf_and_is_dnf_agree_on_a_single_clause() {
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let not_b = m.mk_not(b);
        let clause = m.mk_or([a, not_b]);
        assert!(is_cnf(clause, &m));
        assert!(
            is_dnf(clause, &m),
            "a single clause is trivially also a single DNF term"
        );
    }

    #[test]
    fn is_nnf_rejects_double_negation_and_implication() {
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let not_a = m.mk_not(a);
        let double_neg = m.intern(TermKind::Not(not_a), bool_sort);
        assert!(!is_nnf(double_neg, &m), "not(not(a)) is not in NNF");

        let implies = m.mk_implies(a, b);
        assert!(!is_nnf(implies, &m), "implication is not in NNF");

        assert!(
            is_nnf(not_a, &m),
            "not(a) (negation of a literal) is in NNF"
        );
    }

    #[test]
    fn extract_cnf_clauses_splits_top_level_and() {
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let a = m.mk_var("a", bool_sort);
        let b = m.mk_var("b", bool_sort);
        let c = m.mk_var("c", bool_sort);
        let clause1 = m.mk_or([a, b]);
        let formula = m.mk_and([clause1, c]);

        let clauses = extract_cnf_clauses(formula, &m);

        assert_eq!(clauses, vec![vec![a, b], vec![c]]);
    }

    #[test]
    fn match_shadowing_free_vars_still_free_vars_pin_for_skolemize() {
        // Regression sanity check tying this module to the Item 1 fix:
        // Skolemizing `exists x. P(x)` twice with independently-reset
        // counters must produce the *same* sk!0 both times (no
        // cross-call state leaking via TermManager), confirming
        // `skolemize` (as opposed to `skolemize_with_counter`) really does
        // start a fresh counter every call.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let p_x = m.mk_apply("P", [x], bool_sort);
        let formula = m.mk_exists([("x", int_sort)], p_x);

        let first = skolemize(formula, &mut m);
        let second = skolemize(formula, &mut m);
        assert_eq!(first, second);
    }
}

// ===========================================================================
// Deep-structure regression tests: built iteratively, run on a 128 KiB stack.
// The call returning at all is part of the assertion; the result must also
// be exactly correct, not merely "didn't crash".
// ===========================================================================

mod deep_structures_on_tiny_stack {
    use super::*;

    const DEPTH: usize = 12_500;

    #[test]
    fn to_cnf_survives_deep_double_negation_chain() {
        // Not^DEPTH(x), DEPTH even, collapses all the way down to x itself
        // under repeated double-negation elimination -- exercising DEPTH
        // levels of the iterative walk without ever triggering CNF's
        // distribution (so this is purely a stack-depth check, not an
        // output-size one; see the module doc comment on why the two are
        // kept separate).
        assert_eq!(DEPTH % 2, 0, "test assumes an even depth");
        let result = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let x = m.mk_var("x", m.sorts.bool_sort);
            let chain = deep_not_chain(&mut m, x, DEPTH);
            let cnf = to_cnf(chain, &mut m);
            cnf == x
        });
        assert!(result, "Not^12500(x) must simplify to exactly x");
    }

    #[test]
    fn to_dnf_survives_deep_double_negation_chain() {
        assert_eq!(DEPTH % 2, 0, "test assumes an even depth");
        let result = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let x = m.mk_var("x", m.sorts.bool_sort);
            let chain = deep_not_chain(&mut m, x, DEPTH);
            let dnf = to_dnf(chain, &mut m);
            dnf == x
        });
        assert!(result, "Not^12500(x) must simplify to exactly x");
    }

    #[test]
    fn to_nnf_survives_deep_double_negation_chain() {
        assert_eq!(DEPTH % 2, 0, "test assumes an even depth");
        let result = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let x = m.mk_var("x", m.sorts.bool_sort);
            let chain = deep_not_chain(&mut m, x, DEPTH);
            let nnf = to_nnf(chain, &mut m);
            nnf == x
        });
        assert!(result, "Not^12500(x) must simplify to exactly x");
    }

    #[test]
    fn simplify_boolean_survives_deep_and_chain() {
        // And(x_0, And(x_1, ... And(x_{n-1}, x_0) ...)): the innermost
        // reuses x_0 (the outermost's own conjunct) so the dedup step has
        // something to actually remove at every nesting level, without
        // changing which *distinct* variables appear.
        //
        // Depth is deliberately smaller than the other tests' 100,000
        // here: `simplify_boolean`'s rebuild step calls `manager.mk_and`,
        // which *flattens* a nested `And` argument into its caller's own
        // arg list. On a chain this deeply (right-)nested, each level's
        // rebuild copies the *entire* already-flattened chain below it,
        // making the rebuild itself O(depth^2) -- a pre-existing property
        // of `simplify_boolean`/`mk_and` this session's stack-depth
        // conversion neither introduced nor is trying to fix (it is a
        // rebuild-cost/output-shape concern, not a native-recursion-depth
        // one; the walk itself, via `simplify_boolean_children`, is still
        // a plain O(depth) iterative stack walk regardless of depth).
        // 8,000 is still comfortably beyond any plausible native stack
        // limit (this is exactly the shape that used to overflow before
        // conversion) while keeping the O(depth^2) rebuild tractable.
        const AND_CHAIN_DEPTH: usize = 8_000;
        let result = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let bool_sort = m.sorts.bool_sort;
            let x0 = m.mk_var("and_chain_0", bool_sort);
            let chain = deep_and_chain(&mut m, x0, AND_CHAIN_DEPTH);
            let simplified = simplify_boolean(chain, &mut m);
            m.get(simplified).is_some()
        });
        assert!(result, "simplify_boolean must return a well-formed term");
    }

    #[test]
    fn is_nnf_survives_deep_and_chain_and_reports_true() {
        // Every level is `And(bare_var, rest)`, which is valid NNF at every
        // level, so the whole chain must be reported as NNF.
        let result = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let bool_sort = m.sorts.bool_sort;
            let leaf = m.mk_var("and_chain_leaf", bool_sort);
            let chain = deep_and_chain(&mut m, leaf, DEPTH);
            is_nnf(chain, &m)
        });
        assert!(
            result,
            "a chain of And(var, ...) is valid NNF at every level"
        );
    }

    #[test]
    fn skolemize_survives_deep_forall_chain_and_preserves_depth() {
        // All-Forall, all positive polarity: nothing is effectively
        // existential, so every quantifier is kept and the result's shape
        // (in particular, its term_depth) must match the input's exactly --
        // Skolemization must be a complete no-op here, not merely
        // "returned without crashing".
        let (input_depth, result_depth) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let zero = m.mk_int(0);
            let body_var = m.mk_var("body_var", m.sorts.int_sort);
            let leaf = m.mk_gt(body_var, zero);
            let chain = deep_forall_chain(&mut m, leaf, DEPTH);
            let input_depth = m.term_depth(chain);
            let result = skolemize(chain, &mut m);
            (input_depth, m.term_depth(result))
        });
        assert_eq!(
            input_depth, result_depth,
            "an all-positive-polarity Forall chain must be fully preserved by skolemize"
        );
    }

    #[test]
    fn eliminate_universal_quantifiers_survives_deep_forall_chain_and_strips_all_binders() {
        // Every Forall is eliminated unconditionally, so the result must
        // have strictly smaller term_depth than the input (every binder
        // layer removed, leaving just the innermost body with fresh
        // constants substituted in).
        //
        // Depth is smaller than the other tests' 12,500 here for the same
        // kind of reason as `simplify_boolean`'s test above, but via a
        // different mechanism: `eliminate_universal_impl` calls
        // `TermManager::substitute` once *per Forall layer eliminated*, and
        // each such call walks the *entire remaining* (still `depth -
        // level` layers deep) chain below it (`substitute` has no "this
        // subtree cannot possibly contain the substitution key" short
        // circuit). That makes the total substitute-walk work across all
        // `depth` layers O(depth^2), regardless of walking each individual
        // `substitute` call iteratively rather than recursively -- again a
        // pre-existing algorithmic property of this function (identical in
        // the retired recursive version, which made the exact same
        // per-layer `substitute` call), not a stack-depth concern. 3,000 is
        // still far beyond any plausible native stack limit while keeping
        // the O(depth^2) substitute cost tractable.
        const ELIMINATE_CHAIN_DEPTH: usize = 3_000;
        let (input_depth, result_depth, result_is_forall) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let zero = m.mk_int(0);
            let body_var = m.mk_var("body_var", m.sorts.int_sort);
            let leaf = m.mk_gt(body_var, zero);
            let chain = deep_forall_chain(&mut m, leaf, ELIMINATE_CHAIN_DEPTH);
            let input_depth = m.term_depth(chain);
            let result = eliminate_universal_quantifiers(chain, &mut m);
            let result_depth = m.term_depth(result);
            let result_is_forall = matches!(
                m.get(result).map(|t| &t.kind),
                Some(TermKind::Forall { .. })
            );
            (input_depth, result_depth, result_is_forall)
        });
        assert!(
            result_depth < input_depth,
            "every Forall layer must be stripped: result_depth {result_depth} should be far below input_depth {input_depth}"
        );
        assert!(
            !result_is_forall,
            "no Forall may remain at the top of the result"
        );
    }
}

/// Regression: `is_nnf`'s catch-all used to answer `true` for a formula
/// containing `Xor`. `to_nnf` expands `a xor b` into `(a and not b) or
/// (not a and b)`, so such a formula is *not* a fixed point of the
/// conversion, and a caller guarding `to_nnf` behind `if !is_nnf(t)` skipped
/// the conversion it needed.
#[test]
fn is_nnf_rejects_xor() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let a = manager.mk_var("a", bool_sort);
    let b = manager.mk_var("b", bool_sort);
    let x = manager.mk_xor(a, b);

    assert!(!is_nnf(x, &manager), "xor is not in negation normal form");

    // Nested inside a conjunction it must still be rejected.
    let c = manager.mk_var("c", bool_sort);
    let conj = manager.mk_and([c, x]);
    assert!(!is_nnf(conj, &manager));

    // And the converted form must be accepted, so the two agree.
    let converted = to_nnf(x, &mut manager);
    assert!(is_nnf(converted, &manager));
}

/// Implications are rejected too (pre-existing behaviour, pinned alongside
/// the `Xor` fix so the two connectives cannot drift apart).
#[test]
fn is_nnf_rejects_implies_and_accepts_atoms() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let int_sort = manager.sorts.int_sort;

    let a = manager.mk_var("a", bool_sort);
    let b = manager.mk_var("b", bool_sort);
    let imp = manager.mk_implies(a, b);
    assert!(!is_nnf(imp, &manager));

    // An arithmetic atom carries no Boolean structure and is always NNF.
    let x = manager.mk_var("x", int_sort);
    let one = manager.mk_int(1);
    let atom = manager.mk_le(x, one);
    let disj = manager.mk_or([a, atom]);
    assert!(is_nnf(disj, &manager));
}
