//! Macro Support for MBQI
//!
//! This module provides utility macros and helper functions for MBQI implementation.
//! It includes macros for common patterns, debugging, and code generation.

#[allow(unused_imports)]
use crate::prelude::*;

/// Macro for creating a quantified formula with default parameters
#[macro_export]
macro_rules! quantifier {
    ($term:expr, $vars:expr, $body:expr, universal) => {
        $crate::mbqi::QuantifiedFormula::new($term, $vars, $body, true)
    };
    ($term:expr, $vars:expr, $body:expr, existential) => {
        $crate::mbqi::QuantifiedFormula::new($term, $vars, $body, false)
    };
}

/// Macro for creating an instantiation
#[macro_export]
macro_rules! instantiation {
    ($quantifier:expr, $subst:expr, $result:expr, $gen:expr) => {
        $crate::mbqi::Instantiation::new($quantifier, $subst, $result, $gen)
    };
}

/// Macro for debugging MBQI state
#[macro_export]
macro_rules! mbqi_debug {
    ($($arg:tt)*) => {
        #[cfg(all(feature = "std", feature = "mbqi-debug"))]
        {
            eprintln!("[MBQI DEBUG] {}", format!($($arg)*));
        }
    };
}

/// Macro for MBQI tracing
#[macro_export]
macro_rules! mbqi_trace {
    ($($arg:tt)*) => {
        #[cfg(all(feature = "std", feature = "mbqi-trace"))]
        {
            eprintln!("[MBQI TRACE] {}", format!($($arg)*));
        }
    };
}

/// Macro for timing MBQI operations
#[macro_export]
macro_rules! mbqi_time {
    ($name:expr, $block:block) => {{
        #[cfg(feature = "std")]
        let start = oxiz_time::Instant::now();
        let result = $block;
        #[cfg(feature = "std")]
        {
            let elapsed = start.elapsed();
            mbqi_debug!("{} took {:?}", $name, elapsed);
        }
        result
    }};
}

/// Macro for creating a model completion error
#[macro_export]
macro_rules! completion_error {
    ($msg:expr) => {
        $crate::mbqi::model_completion::CompletionError::CompletionFailed($msg.to_string())
    };
}

/// Macro for creating a finder error
#[macro_export]
macro_rules! finder_error {
    (unsat) => {
        $crate::mbqi::finite_model::FinderError::UnsatAtBound
    };
    (max_bound) => {
        $crate::mbqi::finite_model::FinderError::ExceededMaxBound
    };
    (resource) => {
        $crate::mbqi::finite_model::FinderError::ResourceLimit
    };
}

/// Utility term queries for MBQI, all delegating to [`TermManager`].
///
/// [`TermManager`]: oxiz_core::ast::TermManager
///
/// # Why every function here is a one-line adapter
///
/// This module used to carry five independently written, natively
/// recursive term walks (`is_ground`, `free_vars`, `substitute`,
/// `term_depth`, `term_size`), each reimplementing something
/// [`TermManager`] already does. All five shared the same two defects:
///
/// * **A whitelist plus a silent catch-all.** Each matched only
///   `Var`/`And`/`Or`/`Not`/`Neg`/`Apply` and then fell through
///   `_ => <default>` for everything else -- so *every* arithmetic,
///   bit-vector, string, array, floating-point, `Ite`, `Eq`, `Distinct`
///   and datatype operator was treated as a childless leaf.
///   `is_ground((+ x 1))` answered `true`, `free_vars((bvadd x #x01))`
///   answered `{}`, `substitute` returned the term *unchanged*, and
///   `term_depth`/`term_size` answered `1`. None of those signalled
///   anything to the caller.
/// * **No binder awareness at all.** There was no `Forall`/`Exists`/
///   `Let`/`Match` arm anywhere, so a bound variable was reported free
///   (or, via the catch-all, a quantifier body was never entered at all),
///   and `substitute` was not capture-avoiding.
///
/// Additionally all five recursed natively with no guard whatsoever, so a
/// deeply nested (but perfectly valid) term aborted the process with a
/// stack overflow.
///
/// The silent under-substitution is the worst of these: the identical
/// "return the term unchanged when we give up" behaviour in
/// [`TermManager::substitute`]'s retired depth cap was confirmed to be a
/// genuine soundness exposure, not merely degraded output -- quantifier
/// elimination, Spacer/PDR inductive-invariant checking, BMC unrolling
/// and `oxiz-theories`' proof-instantiation checker all treat the
/// substituted term as authoritative.
///
/// Rather than patch five whitelists (and have them drift again the next
/// time a `TermKind` variant is added), every function below now
/// delegates to the corresponding [`TermManager`] query, each of which is
/// exhaustive over `TermKind` (no catch-all arm, so a new variant is a
/// compile error), binder-aware, and driven by an explicit heap stack
/// rather than native recursion. This mirrors what
/// [`crate::z3_compat::ext3`]'s `Z3Context::substitute` and
/// `oxiz_core::ast::traversal::collect_free_vars` did with their own
/// duplicate walks.
///
/// [`TermManager::substitute`]: oxiz_core::ast::TermManager::substitute
pub mod utils {
    use crate::prelude::*;
    use oxiz_core::ast::{TermId, TermKind, TermManager};
    use oxiz_core::interner::Spur;

    /// The interned name of `term` if it is a `Var`, else `None`.
    ///
    /// [`TermManager`]'s free-variable queries return the `TermId` of each
    /// free occurrence, whereas this module's public API is keyed by
    /// variable *name*; this is the bridge between the two.
    fn var_name(term: TermId, manager: &TermManager) -> Option<Spur> {
        match manager.get(term).map(|t| &t.kind) {
            Some(TermKind::Var(name)) => Some(*name),
            _ => None,
        }
    }

    /// Check whether a term is ground, i.e. has no free variables.
    ///
    /// "Ground" here means exactly "the free-variable set is empty" under
    /// standard first-order scoping: a `Var` occurrence is *not* counted
    /// when it lies inside a `Forall`/`Exists`/`Let` binding of the same
    /// name and sort, or inside a `Match` case that pattern-binds it. So
    /// `(forall ((x Int)) (> x 0))` is ground, while
    /// `(forall ((x Int)) (P x z))` is not (`z` is free) and `(+ x 1)` is
    /// not.
    ///
    /// Expressed directly as `free_vars(..).is_empty()` (via
    /// [`TermManager::free_vars_including_patterns`]) rather than as its
    /// own structural walk -- see this module's doc comment.
    ///
    /// [`TermManager::free_vars_including_patterns`]: oxiz_core::ast::TermManager::free_vars_including_patterns
    pub fn is_ground(term: TermId, manager: &TermManager) -> bool {
        manager.free_vars_including_patterns(term).is_empty()
    }

    /// Collect the names of all free variables in a term.
    ///
    /// Delegates to [`TermManager::free_vars_including_patterns`] -- the
    /// **pattern-aware** variant, which also walks `Forall`/`Exists`
    /// trigger patterns (inside the owning quantifier's own scope, so a
    /// trigger's references to the bound variables stay bound). Every
    /// consumer of this module reasons about groundedness and about
    /// variable *names*, never about `get_children`-shaped term structure,
    /// and for such a guard over-reporting a free variable is safe while
    /// under-reporting is not: a variable occurring only in a trigger is
    /// still a live occurrence. MBQI's own grounding guard
    /// (`mbqi::sat_certify`, `mbqi::integration`) uses the same variant.
    ///
    /// Note that the result is a set of *names*: two distinct free
    /// variables sharing a name at different sorts collapse into one
    /// entry, which is inherent to the `FxHashSet<Spur>` return type this
    /// function has always had.
    ///
    /// [`TermManager::free_vars_including_patterns`]: oxiz_core::ast::TermManager::free_vars_including_patterns
    pub fn free_vars(term: TermId, manager: &TermManager) -> FxHashSet<Spur> {
        manager
            .free_vars_including_patterns(term)
            .into_iter()
            .filter_map(|var| var_name(var, manager))
            .collect()
    }

    /// Substitute free variables in a term by name.
    ///
    /// Each free occurrence of a variable whose name is a key of `subst`
    /// is replaced by the mapped term. Delegates to
    /// [`TermManager::substitute`], which is keyed by `TermId`, so the
    /// name-keyed map is first resolved against the term's actual free
    /// variable occurrences (a name that only ever occurs *bound* is
    /// therefore correctly left alone, as is a name that does not occur at
    /// all).
    ///
    /// Inherited from the core routine, and absent from the local walk
    /// this replaced: it descends into `Forall`/`Exists`/`Let`/`Match`
    /// bodies, bindings, cases and trigger patterns; it is
    /// capture-avoiding (a bound variable whose name would capture a free
    /// variable of a replacement term is alpha-renamed first, which in
    /// OxiZ is what makes substitution under a binder correct at all,
    /// since bound occurrences are ordinary hash-consed
    /// `TermKind::Var(name)` terms); it handles every `TermKind` variant
    /// explicitly; and it uses an explicit heap stack.
    ///
    /// [`TermManager::substitute`]: oxiz_core::ast::TermManager::substitute
    pub fn substitute(
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        if subst.is_empty() {
            return term;
        }
        let by_id: FxHashMap<TermId, TermId> = manager
            .free_vars_including_patterns(term)
            .into_iter()
            .filter_map(|var| {
                let name = var_name(var, manager)?;
                subst.get(&name).map(|&replacement| (var, replacement))
            })
            .collect();
        if by_id.is_empty() {
            return term;
        }
        manager.substitute(term, &by_id)
    }

    /// Calculate the depth of a term.
    ///
    /// Delegates to [`TermManager::term_depth`]: leaves (`Var`, boolean
    /// and numeric literals, string literals, and a term absent from the
    /// manager) have depth `0`, and every other node is one more than the
    /// maximum depth of its children. Note this is one *less* than the
    /// local walk this replaced reported for the terms it handled at all,
    /// which counted nodes rather than nesting levels; the
    /// [`TermManager`] convention is the one every other depth consumer in
    /// the workspace uses.
    ///
    /// [`TermManager::term_depth`]: oxiz_core::ast::TermManager::term_depth
    pub fn term_depth(term: TermId, manager: &TermManager) -> usize {
        manager.term_depth(term)
    }

    /// Calculate the size of a term (number of nodes).
    ///
    /// Delegates to [`TermManager::term_size`], which counts the size of
    /// the term as a *tree*: a subterm reached through two different
    /// parents (structural sharing under hash-consing) is counted once per
    /// path. The local walk this replaced counted each distinct `TermId`
    /// once instead, so a term with shared subterms now reports a larger
    /// size. This is the convention every other size consumer in the
    /// workspace uses, and it is the one that matches "number of nodes" as
    /// a measure of formula size.
    ///
    /// [`TermManager::term_size`]: oxiz_core::ast::TermManager::term_size
    pub fn term_size(term: TermId, manager: &TermManager) -> usize {
        manager.term_size(term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::{TermKind, TermManager};

    #[test]
    fn test_is_ground_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        assert!(utils::is_ground(term, &manager));
    }

    /// A literal is a leaf, so its depth is `0`.
    ///
    /// This expectation used to be `1`: the retired local walk fell through
    /// its catch-all arm for `True` (as it did for every kind outside its
    /// five-arm whitelist) and returned `1`, i.e. it counted nodes rather
    /// than nesting levels -- and inconsistently at that, since its own
    /// `unwrap_or(0)` leaf convention for an empty n-ary operator implied
    /// `0`. `TermManager::term_depth`, which `utils::term_depth` now
    /// delegates to, is the convention used everywhere else in the
    /// workspace: leaves are depth `0`.
    #[test]
    fn test_term_depth_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        assert_eq!(utils::term_depth(term, &manager), 0);
    }

    #[test]
    fn test_term_size_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        assert_eq!(utils::term_size(term, &manager), 1);
    }

    #[test]
    fn test_free_vars_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        let vars = utils::free_vars(term, &manager);
        assert_eq!(vars.len(), 0);
    }

    // ===== Regression tests for the silent-fallthrough bugs =====

    #[test]
    fn is_ground_rejects_arithmetic_containing_a_variable() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let one = m.mk_int(1);
        let sum = m.mk_add([x, one]);
        assert!(!utils::is_ground(sum, &m), "(+ x 1) is not ground");
    }

    #[test]
    fn is_ground_rejects_bitvector_containing_a_variable() {
        let mut m = TermManager::new();
        let bv8 = m.sorts.bitvec(8);
        let x = m.mk_var("x", bv8);
        let one = m.mk_bitvec(1, 8);
        let sum = m.mk_bv_add(x, one);
        assert!(!utils::is_ground(sum, &m), "(bvadd x #x01) is not ground");
    }

    #[test]
    fn is_ground_accepts_a_closed_quantifier() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(x, zero);
        let forall = m.mk_forall([("x", int_sort)], body);
        assert!(
            utils::is_ground(forall, &m),
            "(forall ((x Int)) (> x 0)) has no free variables"
        );
    }

    #[test]
    fn is_ground_rejects_a_quantifier_with_a_free_variable() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let z = m.mk_var("z", int_sort);
        let body = m.mk_apply("P", [x, z], bool_sort);
        let forall = m.mk_forall([("x", int_sort)], body);
        assert!(
            !utils::is_ground(forall, &m),
            "(forall ((x Int)) (P x z)) has z free"
        );
    }

    #[test]
    fn free_vars_ignores_bound_variables_but_reports_free_ones() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let z = m.mk_var("z", int_sort);
        let body = m.mk_apply("P", [x, z], bool_sort);
        let forall = m.mk_forall([("x", int_sort)], body);

        let vars = utils::free_vars(forall, &m);
        let names: Vec<&str> = vars.iter().map(|&s| m.resolve_str(s)).collect();
        assert_eq!(names, vec!["z"], "only z is free; x is bound");
    }

    #[test]
    fn free_vars_finds_variables_under_a_bitvector_operator() {
        let mut m = TermManager::new();
        let bv8 = m.sorts.bitvec(8);
        let x = m.mk_var("x", bv8);
        let one = m.mk_bitvec(1, 8);
        let sum = m.mk_bv_add(x, one);
        let vars = utils::free_vars(sum, &m);
        let names: Vec<&str> = vars.iter().map(|&s| m.resolve_str(s)).collect();
        assert_eq!(names, vec!["x"]);
    }

    #[test]
    fn free_vars_finds_variables_under_ite_and_let() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        // A non-constant condition: `mk_ite` folds a literal condition away.
        let cond = m.mk_var("b", bool_sort);
        let ite = m.mk_ite(cond, x, y);
        let mut names: Vec<String> = utils::free_vars(ite, &m)
            .iter()
            .map(|&s| m.resolve_str(s).to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec!["b".to_string(), "x".to_string(), "y".to_string()]
        );

        // (let ((a x)) (> a 0)) -- `x` is free, `a` is bound.
        let a = m.mk_var("a", int_sort);
        let zero = m.mk_int(0);
        let body = m.mk_gt(a, zero);
        let let_term = m.mk_let([("a", x)], body);
        let names: Vec<String> = utils::free_vars(let_term, &m)
            .iter()
            .map(|&s| m.resolve_str(s).to_string())
            .collect();
        assert_eq!(names, vec!["x".to_string()]);
    }

    #[test]
    fn substitute_replaces_under_a_bitvector_operator() {
        let mut m = TermManager::new();
        let bv8 = m.sorts.bitvec(8);
        let x = m.mk_var("x", bv8);
        let y = m.mk_var("y", bv8);
        let one = m.mk_bitvec(1, 8);
        let sum = m.mk_bv_add(x, one);

        let x_name = m.intern_str("x");
        let mut subst = FxHashMap::default();
        subst.insert(x_name, y);
        let result = utils::substitute(sum, &subst, &mut m);

        let expected = m.mk_bv_add(y, one);
        assert_eq!(result, expected, "x must be replaced inside the bvadd");
    }

    #[test]
    fn substitute_under_a_binder_avoids_capture() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", int_sort);
        let z = m.mk_var("z", int_sort);
        let body = m.mk_apply("P", [x, z], bool_sort);
        let forall = m.mk_forall([("x", int_sort)], body);

        // (forall ((x Int)) (P x z))[z := x]: the bound x must be renamed.
        let z_name = m.intern_str("z");
        let mut subst = FxHashMap::default();
        subst.insert(z_name, x);
        let result = utils::substitute(forall, &subst, &mut m);

        let kind = m
            .get(result)
            .map(|t| t.kind.clone())
            .expect("result term must exist");
        let TermKind::Forall {
            vars,
            body: new_body,
            ..
        } = kind
        else {
            panic!("expected a Forall, got {kind:?}");
        };
        let (renamed, sort) = vars
            .first()
            .map(|&(name, sort)| (m.resolve_str(name).to_string(), sort))
            .expect("exactly one bound variable");
        assert_ne!(renamed, "x", "the bound x must be alpha-renamed");
        let fresh = m.mk_var(&renamed, sort);
        let expected_body = m.mk_apply("P", [fresh, x], bool_sort);
        assert_eq!(new_body, expected_body);
    }

    #[test]
    fn term_depth_sees_through_an_unlisted_operator() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let one = m.mk_int(1);
        let sum = m.mk_add([x, one]);
        let zero = m.mk_int(0);
        let cmp = m.mk_gt(sum, zero); // (> (+ x 1) 0)
        assert_eq!(
            utils::term_depth(cmp, &m),
            2,
            "Gt over an Add must be deeper than the Add alone"
        );
    }

    #[test]
    fn term_size_sees_through_an_unlisted_operator() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let one = m.mk_int(1);
        let sum = m.mk_add([x, one]);
        assert_eq!(utils::term_size(sum, &m), 3, "(+ x 1) has three nodes");
    }

    /// Run `f` to completion on a dedicated thread with a 128 KiB stack --
    /// deliberately far smaller than the default main-thread stack.
    ///
    /// A stack overflow aborts the whole process rather than failing one
    /// test gracefully, so for the deep-nesting test below the call
    /// *returning at all* is itself part of the assertion.
    ///
    /// This stack and the depth below were scaled down together by a factor
    /// of 8 (from 1 MiB / 100 000).  The pin is the ~10 bytes of stack per
    /// nesting level, not the absolute depth, and the smaller pair keeps the
    /// interned terms out of swap.  Never raise one without the other.
    fn run_on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(f)
            .expect("spawning the constrained-stack test thread should succeed")
            .join()
            .expect("the constrained-stack thread must not panic")
    }

    #[test]
    fn utils_survive_a_deeply_nested_term_on_a_tiny_stack() {
        const DEPTH: usize = 12_500;

        let (ground, depth, size, free, substituted) = run_on_small_stack(|| {
            let mut m = TermManager::new();
            let int_sort = m.sorts.int_sort;
            let x = m.mk_var("x", int_sort);
            let y = m.mk_var("y", int_sort);
            // f(f(f(...f(x)...))) -- uninterpreted application never folds.
            let mut chain = x;
            for _ in 0..DEPTH {
                chain = m.mk_apply("f", [chain], int_sort);
            }
            let x_name = m.intern_str("x");
            let mut subst = FxHashMap::default();
            subst.insert(x_name, y);
            (
                utils::is_ground(chain, &m),
                utils::term_depth(chain, &m),
                utils::term_size(chain, &m),
                utils::free_vars(chain, &m).len(),
                utils::substitute(chain, &subst, &mut m) != chain,
            )
        });

        assert!(!ground);
        assert_eq!(depth, DEPTH);
        assert_eq!(size, DEPTH + 1);
        assert_eq!(free, 1);
        assert!(substituted);
    }
}
