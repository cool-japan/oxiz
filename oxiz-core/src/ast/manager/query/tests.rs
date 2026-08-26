//! Tests for `ast/manager/query` and its submodules.
//!
//! Split out of `ast/manager/query.rs` so that no file in the module
//! approaches the workspace's 2000-line ceiling (mirrors the precedent set
//! by `oxiz-theories/src/euf/solver.rs` -> `solver/{congruence,explain,
//! tests}.rs`). Being a child module of `query`, it can still call the
//! `pub(super)` helpers (`prepare_binder_subst`, `find_var_sort`) that
//! `query::substitute` exposes specifically for this file's direct
//! white-box tests.
use super::*;

/// Run `f` to completion on a dedicated thread with a 128 KiB stack --
/// deliberately far smaller than the default (several-MiB) main-thread
/// stack -- and return whatever it returns.
///
/// A stack overflow aborts the whole process rather than failing a single
/// test gracefully, so for the deep-nesting tests below, the call
/// *returning at all* is itself part of what is being asserted: if any of
/// `term_size`/`term_depth`/`substitute`/`simplify` still recursed natively
/// once per level of term nesting, a 12,500-deep term would overflow this
/// stack and the test binary would abort instead of reporting a failure.
fn run_on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(f)
        .expect("spawning the constrained-stack test thread should succeed")
        .join()
        .expect("the constrained-stack thread must not panic")
}

#[cfg(test)]
mod lint_regression_tests {
    //! Regression tests for the clippy `collapsible_if` / `type_complexity`
    //! lint fixes to `find_var_sort` and `prepare_binder_subst`. These pin
    //! down that the mechanical rewrites (nested `if let` -> `if let ... &&
    //! ...`, and the `BinderSubstPrep` type alias) preserved behavior
    //! exactly.
    use super::*;

    #[test]
    fn find_var_sort_locates_variable_occurrence() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let term = m.mk_gt(x, zero); // x > 0

        let TermKind::Var(x_name) = m.get(x).expect("x term").kind else {
            panic!("expected Var");
        };

        // Present: must find the sort of the matching-named variable.
        assert_eq!(m.find_var_sort(term, x_name), Some(int_sort));

        // Absent: a name that never occurs in `term` must yield None, not a
        // stray match on an unrelated subterm.
        let y_name = m.intern_str("y");
        assert_eq!(m.find_var_sort(term, y_name), None);

        // Same name, different sort must not be confused with a differently
        // sorted occurrence located elsewhere in the walk.
        let y_bool = m.mk_var("y", bool_sort);
        let combined = m.mk_and([term, y_bool]);
        assert_eq!(m.find_var_sort(combined, y_name), Some(bool_sort));
    }

    #[test]
    fn prepare_binder_subst_none_when_substitution_is_empty_after_shadowing() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let forty_two = m.mk_int(42);
        let body = m.mk_gt(x, forty_two);

        let TermKind::Var(x_name) = m.get(x).expect("x term").kind else {
            panic!("expected Var");
        };

        let mut subst = FxHashMap::default();
        subst.insert(x, forty_two);

        // x is shadowed by the binder, so the effective substitution is
        // empty and prepare_binder_subst must report None.
        let bound = [(x_name, int_sort)];
        assert!(m.prepare_binder_subst(&bound, body, &[], &subst).is_none());
    }

    #[test]
    fn prepare_binder_subst_returns_effective_substitution() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let forty_two = m.mk_int(42);
        let body = m.mk_gt(x, y);

        let mut subst = FxHashMap::default();
        subst.insert(y, forty_two);

        // y is free (not bound), so it must survive into the effective
        // substitution returned via the BinderSubstPrep-typed Some(..).
        let bound: [(Spur, SortId); 0] = [];
        let (effective, new_bound) = m
            .prepare_binder_subst(&bound, body, &[], &subst)
            .expect("y is unshadowed, so a non-empty substitution is expected");
        assert_eq!(effective.get(&y), Some(&forty_two));
        assert!(new_bound.is_empty(), "no capture, so no fresh binders");
    }
}

#[cfg(test)]
mod free_vars_binder_tests {
    //! Regression tests for: "free_vars counts quantifier-bound variables
    //! as free" — `Forall`/`Exists`/`Let` binders must shadow their bound
    //! names from the free-variable result.
    use super::*;

    #[test]
    fn forall_bound_variable_is_not_free() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(x, zero); // x > 0
        let forall = m.mk_forall([("x", int_sort)], body);

        assert!(
            m.free_vars_including_patterns(forall).is_empty(),
            "the quantifier-bound x must not be reported as free"
        );
    }

    #[test]
    fn exists_bound_variable_is_not_free() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(x, zero); // x > 0
        let exists = m.mk_exists([("x", int_sort)], body);

        assert!(
            m.free_vars(exists).is_empty(),
            "the quantifier-bound x must not be reported as free"
        );
    }

    #[test]
    fn forall_leaves_a_genuinely_free_sibling_variable_free() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let body = m.mk_gt(x, y); // x > y
        let forall = m.mk_forall([("x", int_sort)], body); // forall x. x > y

        let free = m.free_vars_including_patterns(forall);
        assert_eq!(free, vec![y], "x is bound, but y must remain free");
    }

    #[test]
    fn let_bound_variable_is_not_free_in_body() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let five = m.mk_int(5);
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(x, zero); // x > 0
        let let_term = m.mk_let([("x", five)], body); // let ((x 5)) (x > 0)

        assert!(
            m.free_vars(let_term).is_empty(),
            "the let-bound x must not be reported as free"
        );
    }

    #[test]
    fn let_binding_value_is_evaluated_in_outer_scope() {
        // `let ((x y)) (x > 0)`: the bound *value* `y` is evaluated in the
        // outer scope (before `x` shadows anything), so it must still be
        // reported free even though the body's `x` is bound.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(x, zero); // x > 0
        let let_term = m.mk_let([("x", y)], body);

        assert_eq!(m.free_vars(let_term), vec![y]);
    }

    #[test]
    fn shared_subterm_free_outside_a_shadowing_binder_is_still_reported() {
        // Regression test for the `visited` memo: the *same* hash-consed
        // `x` term is referenced both (a) bound, inside a nested
        // `forall x. x > 0`, and (b) unbound, as a sibling conjunct. If
        // the TermId-keyed traversal memo were consulted while under the
        // binder, visiting `x > 0` there would mark the shared `x > 0`
        // subterm (and its `x` leaf) as already-visited, and the second,
        // truly-free occurrence would be wrongly skipped.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let x_gt_0 = m.mk_gt(x, zero); // x > 0, reused below
        let inner = m.mk_forall([("x", int_sort)], x_gt_0); // forall x. x > 0
        let combined = m.mk_and([inner, x_gt_0]); // (forall x. x>0) & (x>0)

        assert_eq!(
            m.free_vars(combined),
            vec![x],
            "the second, unbound occurrence of x must be reported free \
             despite the shared subterm also appearing bound inside the \
             forall"
        );
    }

    #[test]
    fn nested_quantifiers_with_distinct_names_collect_only_free_vars() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let z = m.mk_var("z", int_sort);

        let inner_body = m.mk_gt(x, y); // x > y
        let inner = m.mk_exists([("y", int_sort)], inner_body); // exists y. x > y
        let x_gt_z = m.mk_gt(x, z); // x > z
        let conjunction = m.mk_and([inner, x_gt_z]);
        let outer = m.mk_forall([("x", int_sort)], conjunction); // forall x. (exists y. x>y) & (x>z)

        // `x` is bound by the outer forall, `y` is bound by the inner
        // exists; only `z` is free.
        assert_eq!(m.free_vars(outer), vec![z]);
    }

    #[test]
    fn forall_let_and_match_with_shadowing_collect_only_genuinely_free_vars() {
        // `let ((w 5)) (forall ((x Int)) (match s { some(x) => P(x, w),
        //                                            none    => Q(x, z) }))`
        //
        // * `w` is let-bound and stays in scope throughout the forall's
        //   entire body (including both match cases), so it must never be
        //   reported free.
        // * `x` is forall-bound; the `some` case additionally re-binds `x`
        //   (shadowing the forall's own `x` for that case only) -- either
        //   way, every occurrence of `x` is bound, in both cases.
        // * `s` (the match scrutinee) and `z` (free in the `none` case)
        //   are the only genuinely free variables.
        use crate::ast::term::MatchCase;

        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;

        let w = m.mk_var("w", int_sort);
        let five = m.mk_int(5);
        let x = m.mk_var("x", int_sort);
        let z = m.mk_var("z", int_sort);
        let s = m.mk_var("s", int_sort);

        let TermKind::Var(x_name) = m.get(x).expect("x term").kind else {
            panic!("expected Var");
        };

        let p_x_w = m.mk_apply("P", [x, w], bool_sort); // some(x) => P(x, w)
        let q_x_z = m.mk_apply("Q", [x, z], bool_sort); // none => Q(x, z)
        let some_ctor = m.intern_str("some");
        let none_ctor = m.intern_str("none");
        let cases: SmallVec<[MatchCase; 4]> = [
            MatchCase {
                constructor: Some(some_ctor),
                bindings: [x_name].into_iter().collect(),
                body: p_x_w,
            },
            MatchCase {
                constructor: Some(none_ctor),
                bindings: SmallVec::new(),
                body: q_x_z,
            },
        ]
        .into_iter()
        .collect();
        let match_term = m.intern(
            TermKind::Match {
                scrutinee: s,
                cases,
            },
            bool_sort,
        );

        let forall = m.mk_forall([("x", int_sort)], match_term);
        let let_term = m.mk_let([("w", five)], forall);

        let mut free = m.free_vars(let_term);
        free.sort_by_key(|&id| id.0);
        let mut expected = vec![s, z];
        expected.sort_by_key(|&id| id.0);
        assert_eq!(
            free, expected,
            "only the match scrutinee s and the none-case's z are genuinely free"
        );
    }

    #[test]
    fn deep_not_chain_free_vars_reaches_the_leaf_on_tiny_stack() {
        // Regression: `collect_free_vars` used to recurse natively once per
        // level of term nesting with *no* depth guard at all -- worse than
        // `substitute`/`simplify` had before their own conversions. Built
        // iteratively (never recursively, which would overflow before the
        // assertion runs) and run inside a thread with a deliberately small
        // 128 KiB stack: the call returning at all is part of the assertion,
        // but the exact free-variable set must also be correct -- exactly
        // the one leaf variable, reached through DEPTH levels of `Not`
        // via the generic `get_children` fallback path (see
        // `run_free_var_step`'s final `Some(_)` arm).
        const DEPTH: usize = 12_500;

        let (free, x) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let bool_sort = m.sorts.bool_sort;
            let x = m.mk_var("x", bool_sort);
            let mut term = x;
            for _ in 0..DEPTH {
                term = m.intern(TermKind::Not(term), bool_sort);
            }
            (m.free_vars(term), x)
        });

        assert_eq!(free, vec![x], "the only free variable must be the leaf x");
    }
}

#[cfg(test)]
mod size_depth_tests {
    //! Regression tests for the iterative `term_size`/`term_depth`
    //! conversion (see `query::size_depth`).
    use super::*;

    #[test]
    fn mixed_term_size_and_depth_are_hand_computed_values() {
        // A mixed term exercising binders, `Ite`, n-ary `Add` and `Store`,
        // with sizes/depths computed by hand from the documented
        // recurrence (leaves = size 1 / depth 0; every other node = 1 +
        // sum/max over its `get_children`), independently of the
        // implementation under test:
        //
        //   x, y, zero, five                          size=1  depth=0  (each)
        //   gt_x_0   = (x > 0)                         size=3  depth=1
        //   neg_x    = (- x)                            size=2  depth=1
        //   abs_x    = (ite gt_x_0 x neg_x)             size=7  depth=2
        //   add3     = (+ x y five)                     size=4  depth=1
        //   store_t  = (store x y five)                 size=4  depth=1
        //   z_body   = (z > 0)                          size=3  depth=1
        //   forall_t = (forall ((z Int)) z_body)        size=4  depth=2
        //   mixed    = (and abs_x add3 store_t forall_t)
        //            size = 1 + 7 + 4 + 4 + 4 = 20
        //            depth = 1 + max(2, 1, 1, 2) = 3
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;

        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let zero = m.mk_int(0);
        let five = m.mk_int(5);

        let gt_x_0 = m.mk_gt(x, zero);
        let neg_x = m.mk_neg(x);
        let abs_x = m.mk_ite(gt_x_0, x, neg_x);

        let add3 = m.mk_add([x, y, five]);

        // `mk_store`'s result sort is simply inherited from its first
        // argument, so `x` (an `Int`) stands in for an array here: this
        // test is pinning the *structural* size/depth count, not building
        // a well-sorted formula.
        let store_t = m.mk_store(x, y, five);

        let z_body = {
            let z = m.mk_var("z", int_sort);
            m.mk_gt(z, zero)
        };
        let forall_t = m.mk_forall([("z", int_sort)], z_body);

        let mixed = m.mk_and([abs_x, add3, store_t, forall_t]);

        assert_eq!(m.term_size(mixed), 20);
        assert_eq!(m.term_depth(mixed), 3);
    }

    #[test]
    fn deep_add_chain_size_and_depth_on_tiny_stack() {
        const DEPTH: usize = 12_500;
        let (size, depth) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let zero = m.mk_int(0);
            // Iteratively (never recursively) build a chain of depth
            // `DEPTH`: t_0 is a leaf, t_k = (+ t_{k-1} 0). By the
            // recurrence: size(t_k) = size(t_{k-1}) + 2, depth(t_k) =
            // depth(t_{k-1}) + 1, so after DEPTH steps from a leaf (size 1,
            // depth 0): size = 1 + 2*DEPTH, depth = DEPTH.
            let mut t = m.mk_int(1);
            for _ in 0..DEPTH {
                t = m.mk_add([t, zero]);
            }
            (m.term_size(t), m.term_depth(t))
        });

        assert_eq!(size, 1 + 2 * DEPTH);
        assert_eq!(depth, DEPTH);
    }
}

#[cfg(test)]
mod substitute_tests {
    //! Regression tests for the iterative, capture-avoiding `substitute`
    //! conversion (see `query::substitute`).
    use super::*;
    use crate::ast::term::MatchCase;
    use crate::ast::traversal::contains_term;

    #[test]
    fn capture_avoidance_renames_shadowing_bound_variable() {
        // `(forall ((y Int)) (P x y))[x := y]` must alpha-rename the bound
        // `y` (it would otherwise capture the substituted, free `y`),
        // yielding `(forall ((y!0 Int)) (P y y!0))`.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let p_x_y = m.mk_apply("P", [x, y], bool_sort);
        let forall_y = m.mk_forall([("y", int_sort)], p_x_y);

        let mut subst = FxHashMap::default();
        subst.insert(x, y);
        let result = m.substitute(forall_y, &subst);

        let y_fresh = m.mk_var("y!0", int_sort);
        let p_y_yfresh = m.mk_apply("P", [y, y_fresh], bool_sort);
        let expected = m.mk_forall([("y!0", int_sort)], p_y_yfresh);
        assert_eq!(result, expected);

        // And the un-renamed form must *not* be what we got (that would be
        // the capture bug this pins down).
        let captured_bug = m.mk_forall([("y", int_sort)], p_x_y);
        assert_ne!(result, captured_bug);
    }

    #[test]
    fn capture_avoidance_renames_shadowing_let_binding() {
        // `(let ((z 5)) (x + z))[x := z]` must alpha-rename the let-bound
        // `z` (it would otherwise capture the substituted, free `z`),
        // yielding `(let ((z!0 5)) (z + z!0))`. There is no `mk_let`-level
        // smart simplification that could mask a broken rename here: the
        // binding value `5` is unrelated to the capture, so any wrong
        // (unrenamed) result would be a genuine capture bug, not new-vs-
        // reused-term noise from the builder.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let z = m.mk_var("z", int_sort);
        let five = m.mk_int(5);
        let body = m.mk_add([x, z]); // x + z
        let let_term = m.mk_let([("z", five)], body); // let ((z 5)) (x + z)

        let mut subst = FxHashMap::default();
        subst.insert(x, z);
        let result = m.substitute(let_term, &subst);

        let z_fresh = m.mk_var("z!0", int_sort);
        let expected_body = m.mk_add([z, z_fresh]); // z + z!0
        let expected = m.mk_let([("z!0", five)], expected_body);
        assert_eq!(result, expected);

        let captured_bug = m.mk_let([("z", five)], body);
        assert_ne!(result, captured_bug);
    }

    #[test]
    fn capture_avoidance_renames_shadowing_match_case_binding() {
        // `match s { some(y) => P(x, y), none => Q(x) }` under `x := y`
        // must alpha-rename the `some` case's bound `y` (it would
        // otherwise capture the substituted, free `y`) while leaving the
        // `none` case's substitution unaffected by that rename (it binds
        // nothing, so only `x -> y` applies there):
        // `match s { some(y!0) => P(y, y!0), none => Q(y) }`.
        //
        // There is no `mk_match` smart constructor in this crate, so the
        // `Match` term is built directly via `intern`, mirroring how
        // `TermManager::substitute` itself reconstructs one.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let TermKind::Var(y_name) = m.get(y).expect("y term").kind else {
            panic!("expected Var");
        };
        let scrutinee = m.mk_var("s", int_sort);
        let some_ctor = m.intern_str("some");
        let none_ctor = m.intern_str("none");

        let p_x_y = m.mk_apply("P", [x, y], bool_sort); // some's case body
        let q_x = m.mk_apply("Q", [x], bool_sort); // none's case body
        let cases: SmallVec<[MatchCase; 4]> = [
            MatchCase {
                constructor: Some(some_ctor),
                bindings: [y_name].into_iter().collect(),
                body: p_x_y,
            },
            MatchCase {
                constructor: Some(none_ctor),
                bindings: SmallVec::new(),
                body: q_x,
            },
        ]
        .into_iter()
        .collect();
        let match_term = m.intern(TermKind::Match { scrutinee, cases }, bool_sort);

        let mut subst = FxHashMap::default();
        subst.insert(x, y);
        let result = m.substitute(match_term, &subst);

        let y_fresh = m.mk_var("y!0", int_sort);
        let p_y_yfresh = m.mk_apply("P", [y, y_fresh], bool_sort);
        let q_y = m.mk_apply("Q", [y], bool_sort);
        let expected_cases: SmallVec<[MatchCase; 4]> = [
            MatchCase {
                constructor: Some(some_ctor),
                bindings: [m.intern_str("y!0")].into_iter().collect(),
                body: p_y_yfresh,
            },
            MatchCase {
                constructor: Some(none_ctor),
                bindings: SmallVec::new(),
                body: q_y,
            },
        ]
        .into_iter()
        .collect();
        let expected = m.intern(
            TermKind::Match {
                scrutinee,
                cases: expected_cases,
            },
            bool_sort,
        );
        assert_eq!(result, expected);
        let y_fresh_name = m.intern_str("y!0");
        assert_eq!(y_fresh, m.intern(TermKind::Var(y_fresh_name), int_sort));
    }

    #[test]
    fn capture_avoidance_survives_deep_unrelated_nesting_on_tiny_stack() {
        // The same capture scenario as above, but wrapped in `DEPTH`
        // additional quantifiers that bind names unrelated to `x`/`y` --
        // built iteratively, never recursively. None of those wrapper
        // binders shadow anything relevant to the `x -> y` substitution,
        // so each one still needs its own substitution context opened
        // (nothing shadows `x` away), stressing the context arena at real
        // depth; only the innermost `forall y` actually needs alpha-
        // renaming. `DEPTH` is comfortably beyond the 1000-level cap the
        // recursive implementation used to bail out at.
        const DEPTH: usize = 12_500;

        let matches = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let int_sort = m.sorts.int_sort;
            let bool_sort = m.sorts.bool_sort;
            let x = m.mk_var("x", int_sort);
            let y = m.mk_var("y", int_sort);
            let p_x_y = m.mk_apply("P", [x, y], bool_sort);
            let inner_forall = m.mk_forall([("y", int_sort)], p_x_y);

            let dummy_names: Vec<String> = (0..DEPTH).map(|i| format!("w{i}")).collect();

            let mut deep_input = inner_forall;
            for name in &dummy_names {
                deep_input = m.mk_forall([(name.as_str(), int_sort)], deep_input);
            }

            let mut subst = FxHashMap::default();
            subst.insert(x, y);
            let result = m.substitute(deep_input, &subst);

            let y_fresh = m.mk_var("y!0", int_sort);
            let p_y_yfresh = m.mk_apply("P", [y, y_fresh], bool_sort);
            let expected_inner = m.mk_forall([("y!0", int_sort)], p_y_yfresh);
            let mut expected = expected_inner;
            for name in &dummy_names {
                expected = m.mk_forall([(name.as_str(), int_sort)], expected);
            }

            result == expected
        });

        assert!(
            matches,
            "capture-avoidance must hold at the bottom of a deeply nested \
             chain of unrelated binders"
        );
    }

    #[test]
    fn deep_add_chain_substitute_reaches_innermost_leaf_on_tiny_stack() {
        const DEPTH: usize = 12_500;

        let (size, depth, still_has_old_leaf, now_has_new_leaf) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let zero = m.mk_int(0);
            let leaf = m.mk_int(1);
            let mut t = leaf;
            for _ in 0..DEPTH {
                t = m.mk_add([t, zero]);
            }

            let y = m.mk_var("y", m.sorts.int_sort);
            let mut subst = FxHashMap::default();
            subst.insert(leaf, y);
            let result = m.substitute(t, &subst);

            (
                m.term_size(result),
                m.term_depth(result),
                contains_term(result, leaf, &m),
                contains_term(result, y, &m),
            )
        });

        // Swapping one leaf for another leaf preserves the overall shape.
        assert_eq!(size, 1 + 2 * DEPTH);
        assert_eq!(depth, DEPTH);
        // The substitution must have actually reached the bottom of the
        // DEPTH-deep chain: the old leaf is gone and the new one is
        // present, not silently left unsubstituted past some residual cap.
        assert!(!still_has_old_leaf);
        assert!(now_has_new_leaf);
    }
}

#[cfg(test)]
mod simplify_tests {
    //! Regression tests for the iterative `simplify` conversion (see
    //! `query::simplify`).
    use super::*;

    #[test]
    fn shallow_constant_folding_and_identities() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);

        // x + 0 simplifies to x.
        let x_plus_0 = m.mk_add([x, zero]);
        assert_eq!(m.simplify(x_plus_0), x);

        // 2 + 3 simplifies to the constant 5.
        let two = m.mk_int(2);
        let three = m.mk_int(3);
        let sum = m.mk_add([two, three]);
        let five = m.mk_int(5);
        assert_eq!(m.simplify(sum), five);

        // x < x simplifies to False (reflexivity), regardless of nesting.
        let refl = m.mk_lt(x, x);
        assert_eq!(m.simplify(refl), m.false_id);
    }

    #[test]
    fn deep_add_of_zero_chain_folds_to_constant_on_tiny_stack() {
        const DEPTH: usize = 12_500;

        let (result, expected) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let zero = m.mk_int(0);
            // t_0 = 1, t_k = (+ t_{k-1} 0); every level's constant-folding
            // keeps the running value pinned at 1, so simplifying the
            // whole depth-DEPTH chain must fold it straight down to the
            // constant 1, regardless of nesting depth.
            let mut t = m.mk_int(1);
            for _ in 0..DEPTH {
                t = m.mk_add([t, zero]);
            }
            let simplified = m.simplify(t);
            let one = m.mk_int(1);
            (simplified, one)
        });

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod pattern_free_var_tests {
    //! Regression tests for: free-variable collection used to ignore the
    //! `patterns` (SMT-LIB `:pattern` / trigger) field of
    //! `Forall`/`Exists`, so a variable occurring *only* inside a trigger
    //! was invisible to the capture-avoidance name-clash detector in
    //! `prepare_binder_subst` (and to `oxiz-solver`'s MBQI grounding
    //! guard).
    use super::*;

    #[test]
    fn free_vars_including_patterns_reports_trigger_only_variable() {
        // (forall ((x Int)) (! true :pattern ((f x y))))
        //
        // `y` occurs nowhere but the trigger, yet it is a genuine free
        // occurrence of `y` in the term.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let trigger = m.mk_apply("f", [x, y], int_sort);
        let body = m.mk_true();
        let forall = m.mk_forall_with_patterns([("x", int_sort)], body, [vec![trigger]]);

        assert_eq!(
            m.free_vars_including_patterns(forall),
            vec![y],
            "a variable occurring only in a trigger is still a free occurrence"
        );
    }

    #[test]
    fn substitute_fresh_binder_name_avoids_trigger_only_variable() {
        // (forall ((x Int)) (! (> y 0) :pattern ((f x x!0))))[y := x]
        //
        // Substituting the free `y` with `x` forces the bound `x` to be
        // alpha-renamed. The generated fresh name must avoid `x!0`, which
        // occurs free *only in the trigger*: picking `x!0` would capture
        // it.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let x0 = m.mk_var("x!0", int_sort);
        let trigger = m.mk_apply("f", [x, x0], int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(y, zero);
        let forall = m.mk_forall_with_patterns([("x", int_sort)], body, [vec![trigger]]);

        let mut subst = FxHashMap::default();
        subst.insert(y, x);
        let result = m.substitute(forall, &subst);

        let TermKind::Forall {
            vars,
            patterns,
            body: new_body,
        } = m.get(result).expect("result must be a Forall").kind.clone()
        else {
            panic!("expected a Forall");
        };

        let x0_name = m.intern_str("x!0");
        assert_ne!(
            vars.first().map(|&(name, _)| name),
            Some(x0_name),
            "the fresh binder name must not be x!0, which occurs free in the trigger"
        );

        // The trigger's free `x!0` must survive as a free occurrence of the
        // *same* variable, i.e. it must not have been captured by the new
        // binder nor rewritten.
        let renamed = vars
            .first()
            .map(|&(name, _)| m.intern(TermKind::Var(name), int_sort))
            .expect("the rebuilt Forall must still bind one variable");
        let trigger_terms: Vec<TermId> = patterns.iter().flat_map(|p| p.to_vec()).collect();
        assert_eq!(trigger_terms.len(), 1, "the single trigger must survive");
        let TermKind::Apply { args, .. } =
            m.get(trigger_terms[0]).expect("trigger term").kind.clone()
        else {
            panic!("expected the trigger to still be an Apply");
        };
        assert_eq!(
            args.to_vec(),
            vec![renamed, x0],
            "the trigger's bound occurrence must be renamed and its free x!0 left alone"
        );
        assert_eq!(m.free_vars_including_patterns(new_body), vec![x]);
    }

    #[test]
    fn free_vars_deliberately_still_skips_trigger_only_variable() {
        // Pins the *other* half of the deliberate two-variant split: plain
        // `free_vars` must keep agreeing with `get_children` (a quantifier's
        // only child is its body), so widening it is a visible, intentional
        // change rather than something a future edit can do by accident.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let trigger = m.mk_apply("f", [x, y], int_sort);
        let body = m.mk_true();
        let forall = m.mk_forall_with_patterns([("x", int_sort)], body, [vec![trigger]]);

        assert!(
            m.free_vars(forall).is_empty(),
            "the non-pattern-aware query must not report the trigger-only y"
        );
        assert_eq!(m.free_vars_including_patterns(forall), vec![y]);
    }

    #[test]
    fn free_vars_including_patterns_treats_trigger_bound_variables_as_bound() {
        // A trigger normally mentions the bound variables; those occurrences
        // are bound, not free, so walking patterns must happen *inside* the
        // quantifier's scope.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let trigger = m.mk_apply("f", [x], int_sort);
        let body = m.mk_true();
        let forall = m.mk_forall_with_patterns([("x", int_sort)], body, [vec![trigger]]);

        assert!(
            m.free_vars_including_patterns(forall).is_empty(),
            "the trigger's reference to the bound x must be treated as bound"
        );
    }
}
