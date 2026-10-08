//! Unit tests for the lazy array-axiom refinement loop.
//!
//! White-box on purpose, and split out of `array_axioms.rs` so that file keeps
//! room under the workspace's 2,000-line ceiling for the rules themselves.
//! What they pin is solver *internals*: that an exhausted instance budget
//! degrades to `Unknown` rather than to `Sat`, that the structure walk
//! terminates on a deep or shared store chain, and that a read reached only
//! through a bit-vector or arithmetic operator is still collected.  The
//! end-to-end verdict behaviour lives in `oxiz-solver/tests/`.

mod budget_honesty_tests {
    use super::super::*;
    use crate::solver::types::SolverResult;
    use oxiz_core::ast::TermManager;

    /// An exhausted instance budget must never be reported as a model.
    ///
    /// The cap makes `instantiate_array_axioms` return `false`, which is the
    /// same value it returns for "this candidate satisfies every axiom" — the
    /// one answer that licenses `Sat`. Pre-loading the dedup set to the cap
    /// simulates a formula that used the whole budget, and the verdict on a
    /// formula that genuinely needs an array lemma must then be `Unknown`
    /// rather than the `sat` the unchecked `false` would have produced.
    #[test]
    fn an_exhausted_instance_budget_is_unknown_not_sat() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);

        // `(not (= (store (store a 1 x) 2 y) (store (store a 2 y) 1 x)))` —
        // unsat, and only extensionality can show it.
        let a = tm.mk_var("a", array_sort);
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let one = tm.mk_int(1);
        let two = tm.mk_int(2);
        let lhs = {
            let inner = tm.mk_store(a, one, x);
            tm.mk_store(inner, two, y)
        };
        let rhs = {
            let inner = tm.mk_store(a, two, y);
            tm.mk_store(inner, one, x)
        };
        let eq = tm.mk_eq(lhs, rhs);
        let goal = tm.mk_not(eq);
        solver.assert(goal, &mut tm);

        // Fill the dedup set to the cap with throwaway ids so the very first
        // instantiation call is refused by the budget check.
        for i in 0..MAX_ARRAY_AXIOM_INSTANCES {
            let filler = tm.mk_var(&format!("!filler!{i}"), int_sort);
            solver.array_axiom_instances.insert(filler);
        }

        let verdict = solver.check(&mut tm);
        assert_eq!(
            verdict,
            SolverResult::Unknown,
            "an exhausted array-axiom budget must be reported honestly, never as sat"
        );
        assert!(
            solver.array_axioms_incomplete,
            "the budget exhaustion must be recorded"
        );
    }

    /// The same formula with the budget available is decided, so the gate above
    /// is not simply suppressing every array answer.
    #[test]
    fn an_available_budget_still_decides_the_same_formula() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let one = tm.mk_int(1);
        let two = tm.mk_int(2);
        let lhs = {
            let inner = tm.mk_store(a, one, x);
            tm.mk_store(inner, two, y)
        };
        let rhs = {
            let inner = tm.mk_store(a, two, y);
            tm.mk_store(inner, one, x)
        };
        let eq = tm.mk_eq(lhs, rhs);
        let goal = tm.mk_not(eq);
        solver.assert(goal, &mut tm);

        assert_eq!(
            solver.check(&mut tm),
            SolverResult::Unsat,
            "store commutativity is refutable when the budget is available"
        );
        assert!(
            !solver.array_axioms_incomplete,
            "nothing near the cap was needed"
        );
    }
}

mod s8_iterative_tests {
    use super::super::*;
    use oxiz_core::ast::TermManager;

    /// Nesting depth that would overflow the native stack under the previous
    /// recursive walk; the assertion is simply that the call **returns**.
    ///
    /// This depth and [`SMALL_STACK`] were scaled down together by a factor
    /// of 8 (from 60 000 on 1 MiB).  What the test pins is the ~17 bytes of
    /// stack available per level — far under any native frame — not the
    /// absolute depth, and the smaller pair costs a fraction of the memory
    /// the interner has to keep live.  Never raise one without the other.
    const DEEP: usize = 7_500;

    /// Worker stack for the deep-nesting test; see [`DEEP`].
    const SMALL_STACK: usize = 1 << 17;

    /// Build `store(store(...store(a, i, v)..., i, v), i, v)`, `depth` levels.
    fn deep_store_chain(tm: &mut TermManager, depth: usize) -> (TermId, TermId) {
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let base = tm.mk_var("a", array_sort);
        let idx = tm.mk_int(num_bigint::BigInt::from(1));
        let val = tm.mk_int(num_bigint::BigInt::from(7));
        let mut current = base;
        for _ in 0..depth {
            current = tm.mk_store(current, idx, val);
        }
        (current, idx)
    }

    #[test]
    fn s8_collect_array_structure_deep_store_chain_returns() {
        // A 128 KiB stack: the recursive version could not survive `DEEP`
        // frames, so returning at all is the proof of the conversion.
        let handle = std::thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(|| {
                let mut tm = TermManager::new();
                let (deep, idx) = deep_store_chain(&mut tm, DEEP);
                let select = tm.mk_select(deep, idx);
                let mut visited = FxHashSet::default();
                let mut out = ArrayStructure::default();
                collect_array_structure(select, &tm, &mut visited, &mut out, true);
                out.selects.len()
            })
            .expect("spawn deep-nesting worker");
        assert_eq!(handle.join().ok(), Some(1));
    }

    /// A doubling DAG: without the `visited` set this would expand
    /// exponentially instead of completing immediately.
    #[test]
    fn s8_collect_array_structure_shared_dag_completes() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let mut current = tm.mk_var("x", int_sort);
        for _ in 0..55 {
            current = tm.mk_add(vec![current, current]);
        }
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(current, &tm, &mut visited, &mut out, true);
        assert!(out.selects.is_empty());
    }

    /// Semantic pin: the walk still records selects, read indices, array
    /// equalities and `var = store(..)` aliases, in the recursive order.
    #[test]
    fn s8_collect_array_structure_records_same_structure() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let b = tm.mk_var("b", array_sort);
        let i = tm.mk_int(num_bigint::BigInt::from(1));
        let j = tm.mk_int(num_bigint::BigInt::from(2));
        let v = tm.mk_int(num_bigint::BigInt::from(9));
        let store_a = tm.mk_store(a, i, v);
        let alias = tm.mk_eq(b, store_a);
        let sel_i = tm.mk_select(a, i);
        let sel_j = tm.mk_select(a, j);
        let sel_eq = tm.mk_eq(sel_i, sel_j);
        let both = tm.mk_and(vec![alias, sel_eq]);

        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(both, &tm, &mut visited, &mut out, true);

        // `b = store(a, i, v)` is recorded as an alias and as an array-sorted
        // equality pair; the two selects are recorded left to right.
        assert_eq!(out.aliases.get(&b), Some(&store_a));
        assert_eq!(out.eq_pairs, vec![(b, store_a)]);
        assert_eq!(
            out.selects,
            vec![(sel_i, a, i), (sel_j, a, j)],
            "select order must match the recursive pre-order"
        );
        assert_eq!(out.read_indices.get(&a), Some(&vec![i, j]));
    }
}

mod p2b32_walk_tests {
    use super::super::*;
    use crate::solver::types::SolverResult;
    use oxiz_core::ast::TermManager;

    /// A `select` nested under a bit-vector operator is collected (`#P2b-32`).
    /// The hand-written child list this walk used to have named only the
    /// Boolean connectives, `ite` and `Apply`, so the read under `bvadd`
    /// below was invisible and no read-over-write instance was ever built.
    #[test]
    fn a_select_under_a_bit_vector_operator_is_collected() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let array_sort = tm.sorts.array(bv8, bv8);
        let arr = tm.mk_var("arr", array_sort);
        let i = tm.mk_var("i", bv8);
        let five = tm.mk_bitvec(5, 8);
        let one = tm.mk_bitvec(1, 8);
        let six = tm.mk_bitvec(6, 8);
        let store = tm.mk_store(arr, i, five);
        let read = tm.mk_select(store, i);

        for (name, wrapped) in [
            ("bvadd", tm.mk_bv_add(read, one)),
            ("bvnot", tm.mk_bv_not(read)),
        ] {
            let goal = tm.mk_distinct([wrapped, six]);
            let mut visited = FxHashSet::default();
            let mut out = ArrayStructure::default();
            collect_array_structure(goal, &tm, &mut visited, &mut out, true);
            assert_eq!(out.selects, vec![(read, store, i)], "under {name}");
        }
        let comparison = tm.mk_bv_ult(six, read);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(comparison, &tm, &mut visited, &mut out, true);
        assert_eq!(out.selects, vec![(read, store, i)], "under bvult");
    }

    /// The integer twin: a read under `+`, and the `<` comparison over it.
    #[test]
    fn a_select_under_an_arithmetic_operator_is_collected() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let arr = tm.mk_var("arr", array_sort);
        let i = tm.mk_var("i", int_sort);
        let five = tm.mk_int(5);
        let one = tm.mk_int(1);
        let six = tm.mk_int(6);
        let store = tm.mk_store(arr, i, five);
        let read = tm.mk_select(store, i);
        let sum = tm.mk_add([read, one]);
        let goal = tm.mk_distinct([sum, six]);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(goal, &tm, &mut visited, &mut out, true);
        assert_eq!(out.selects, vec![(read, store, i)], "under +");

        let comparison = tm.mk_lt(six, read);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(comparison, &tm, &mut visited, &mut out, true);
        assert_eq!(out.selects, vec![(read, store, i)], "under <");
    }

    /// A read under a binder is deliberately *not* collected: a ground lemma
    /// over a bound variable is an instance of nothing, and the old list
    /// stopped at binders too.
    #[test]
    fn a_select_under_a_binder_is_not_collected() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let array_sort = tm.sorts.array(bv8, bv8);
        let arr = tm.mk_var("arr", array_sort);
        let k = tm.mk_var("k", bv8);
        let five = tm.mk_bitvec(5, 8);
        let store = tm.mk_store(arr, k, five);
        let read = tm.mk_select(store, k);
        let body = tm.mk_eq(read, five);
        let quantified = tm.mk_forall([("k", bv8)], body);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(quantified, &tm, &mut visited, &mut out, true);
        assert!(out.selects.is_empty(), "no ground instance under a binder");
    }

    /// End to end through the builder API: the a3 shape and its integer
    /// twin are refuted, and the miss with a free second index stays `sat`.
    #[test]
    fn nested_read_over_write_is_decided() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let array_sort = tm.sorts.array(bv8, bv8);
        let arr = tm.mk_var("arr", array_sort);
        let i = tm.mk_var("i", bv8);
        let j = tm.mk_var("j", bv8);
        let five = tm.mk_bitvec(5, 8);
        let one = tm.mk_bitvec(1, 8);
        let six = tm.mk_bitvec(6, 8);
        let store = tm.mk_store(arr, i, five);

        let hit = tm.mk_select(store, i);
        let hit_sum = tm.mk_bv_add(hit, one);
        let refuted = tm.mk_distinct([hit_sum, six]);
        let mut solver = Solver::new();
        solver.assert(refuted, &mut tm);
        assert_eq!(solver.check(&mut tm), SolverResult::Unsat, "a3");

        let miss = tm.mk_select(store, j);
        let miss_sum = tm.mk_bv_add(miss, one);
        let satisfiable = tm.mk_distinct([miss_sum, six]);
        let mut solver = Solver::new();
        solver.assert(satisfiable, &mut tm);
        assert_eq!(solver.check(&mut tm), SolverResult::Sat, "a9");

        let int_sort = tm.sorts.int_sort;
        let int_array = tm.sorts.array(int_sort, int_sort);
        let iarr = tm.mk_var("iarr", int_array);
        let n = tm.mk_var("n", int_sort);
        let ifive = tm.mk_int(5);
        let ione = tm.mk_int(1);
        let isix = tm.mk_int(6);
        let istore = tm.mk_store(iarr, n, ifive);
        let iread = tm.mk_select(istore, n);
        let isum = tm.mk_add([iread, ione]);
        let irefuted = tm.mk_distinct([isum, isix]);
        let mut solver = Solver::new();
        solver.assert(irefuted, &mut tm);
        assert_eq!(solver.check(&mut tm), SolverResult::Unsat, "a12");
    }
}

/// One array is one array term (`#P2b-59`).
///
/// `Solver::assert` stores the **pre**-rewrite term in `Solver::assertions`
/// and registers the **encoded** one as a ground array root.  Where the
/// encoding chain replaced an array-sorted `(ite c a b)` with the fresh proxy
/// `eliminate_nonbool_ite` mints, the same array became reachable under two
/// names, and the collector — which walks both root sets — counted it twice:
/// two members of every pair set, two extensionality witnesses per pair, and a
/// read-over-write cascade down each.
///
/// The test states the fix directly, on the shape that found it
/// (`rk9/min/q33_a01.smt2`): the collector's array-term set contains no proxy
/// at all, and its size is the number of syntactically distinct arrays rather
/// than that number plus one per `ite`.  A verdict test cannot say this — the
/// script's verdict is a property of the search that follows — and a timing
/// test must not.
mod one_spelling_per_array_tests {
    use super::super::*;
    use oxiz_core::ast::TermManager;

    /// Every array-sorted term the collector reaches from `solver`'s two root
    /// sets, in first-encounter order.
    fn collected_array_terms(solver: &Solver, manager: &TermManager) -> Vec<TermId> {
        let mut collected = ArrayStructure::default();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        for &root in &solver.assertions {
            collect_array_structure(root, manager, &mut visited, &mut collected, true);
        }
        for &root in &solver.ground_array_roots {
            collect_array_structure(root, manager, &mut visited, &mut collected, false);
        }
        let mut out: Vec<TermId> = Vec::new();
        let push = |t: TermId, out: &mut Vec<TermId>| {
            if is_array_sorted(t, manager) && !out.contains(&t) {
                out.push(t);
            }
        };
        for &(_, array, _) in &collected.selects {
            push(array, &mut out);
        }
        for &(store, _, value) in &collected.stores {
            push(store, &mut out);
            push(value, &mut out);
        }
        for &term in &collected.foreign {
            push(term, &mut out);
        }
        for &(ite, _, then_branch, else_branch) in &collected.array_ites {
            push(ite, &mut out);
            push(then_branch, &mut out);
            push(else_branch, &mut out);
        }
        for &term in &collected.const_arrays {
            push(term, &mut out);
        }
        out
    }

    #[test]
    fn an_ite_selected_array_base_is_one_array_and_not_two() {
        let mut tm = TermManager::new();
        let index = tm.sorts.bitvec(8);
        let element = tm.sorts.bitvec(1);
        let array_sort = tm.sorts.array(index, element);

        let a0 = tm.mk_var("a0", array_sort);
        let a1 = tm.mk_var("a1", array_sort);
        let p = tm.mk_var("p", tm.sorts.bool_sort);
        let one = tm.mk_bitvec(1u32, 1);
        let zero = tm.mk_bitvec(0u32, 1);
        let i0 = tm.mk_bitvec(0xb7u32, 8);
        let i1 = tm.mk_bitvec(0xd6u32, 8);

        // `(= (select (store (ite p a0 a1) #xb7 #b1) #xd6) #b0)` — the shape of
        // `rk9/min/q33_a01.smt2`'s first assertion, with an `ite`-selected
        // store base and no quantifier anywhere.
        let base = tm.mk_ite(p, a0, a1);
        let store = tm.mk_store(base, i0, one);
        let read = tm.mk_select(store, i1);
        let goal = tm.mk_eq(read, zero);

        let mut solver = Solver::new();
        solver.assert(goal, &mut tm);

        let arrays = collected_array_terms(&solver, &tm);
        let prefix = oxiz_core::smtlib::reserved_name("iteelim", "");
        let printer = oxiz_core::smtlib::Printer::new(&tm);
        let printed: Vec<String> = arrays.iter().map(|&t| printer.print_term(t)).collect();
        let proxies: Vec<&String> = printed.iter().filter(|s| s.contains(&prefix)).collect();
        assert!(
            proxies.is_empty(),
            "the array collector reached the encoder's `ite` proxies, so one array is two array \
             terms: {proxies:?}"
        );
        // `a0`, `a1`, `(ite p a0 a1)` and `(store (ite p a0 a1) #xb7 #b1)`.
        // The proxy would make it five, and every pair set built out of it
        // would grow with the square of that.
        assert_eq!(arrays.len(), 4, "collected array terms: {printed:?}");
    }

    /// The other half of the same rule: below the enumeration limit the proxy
    /// **stays**, and that is deliberate.
    ///
    /// `Solver::array_root_spelling` re-spells only the proxies of arrays
    /// whose index sort is too large for
    /// [`super::super::ARRAY_INDEX_ENUMERATION_LIMIT`]. An enumerated pair
    /// mints no Skolem witness, so the duplicate costs at most a constant
    /// factor of lemmas over the one shared index set — while putting the
    /// `ite` back there hands `build_array_ite_reads` a conditional read pair
    /// at every element of the domain for every read the enumerated family
    /// creates. Measured over the 300-pair width-1/2 corpus of
    /// `round4_pass5_recheck_pins`: one pair went from `sat` in 8.5 ms to
    /// `sat` in 2,702 ms with the verdict unchanged, and the gate's 180 s
    /// ceiling became a TIMEOUT; with the limit respected it is `sat` in
    /// 2.9 ms.
    ///
    /// So this test asserts the *presence* of the proxy, and it reddens if a
    /// later pass widens the rule to every index sort. It is not a claim that
    /// two spellings are good — it is the record of where the trade turns.
    #[test]
    fn below_the_enumeration_limit_the_proxy_deliberately_stays() {
        let mut tm = TermManager::new();
        let index = tm.sorts.bitvec(2);
        let element = tm.sorts.bitvec(1);
        let array_sort = tm.sorts.array(index, element);

        let a0 = tm.mk_var("a0", array_sort);
        let a1 = tm.mk_var("a1", array_sort);
        let p = tm.mk_var("p", tm.sorts.bool_sort);
        let one = tm.mk_bitvec(1u32, 1);
        let zero = tm.mk_bitvec(0u32, 1);
        let i0 = tm.mk_bitvec(2u32, 2);
        let i1 = tm.mk_bitvec(3u32, 2);

        let base = tm.mk_ite(p, a0, a1);
        let store = tm.mk_store(base, i0, one);
        let read = tm.mk_select(store, i1);
        let goal = tm.mk_eq(read, zero);

        let mut solver = Solver::new();
        solver.assert(goal, &mut tm);

        let arrays = collected_array_terms(&solver, &tm);
        let prefix = oxiz_core::smtlib::reserved_name("iteelim", "");
        let printer = oxiz_core::smtlib::Printer::new(&tm);
        let printed: Vec<String> = arrays.iter().map(|&t| printer.print_term(t)).collect();
        assert!(
            printed.iter().any(|s| s.contains(&prefix)),
            "at an enumerable index sort the encoder's `ite` proxy must still \
             reach the collector: {printed:?}"
        );
    }
}
