//! Characterisation and depth tests for [`Solver::track_theory_vars`].
//!
//! The values pinned below were captured from the **recursive** implementation
//! that preceded the explicit-stack conversion, via a temporary verbatim copy
//! of it run side by side with the new walk over the generated corpus built by
//! [`build_corpus`] (628 terms, compared both term-by-term on fresh solvers and
//! accumulated in order on one solver; zero divergences).  The copy is gone;
//! these pins are what remains of it, so a future change to the walk that
//! alters *which* terms get a theory variable, *which* trail entries are
//! journalled, or *in what order* fails here.
use super::*;

// ---------------------------------------------------------------------
// Corpus construction
// ---------------------------------------------------------------------

/// Deterministic 64-bit xorshift, written inline so the corpus needs no
/// dependency and reproduces bit-for-bit on every platform.
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % (n as u64)) as usize
    }
}

/// Pools of terms by sort family, so generated nodes are at least plausibly
/// well-sorted, plus `all` in creation order (the corpus proper).
struct Corpus {
    ints: Vec<TermId>,
    reals: Vec<TermId>,
    bools: Vec<TermId>,
    bvs: Vec<TermId>,
    arrays: Vec<TermId>,
    strings: Vec<TermId>,
    all: Vec<TermId>,
}

/// Build a varied corpus of terms with a deterministic PRNG.
///
/// Everything is built through [`TermManager::intern_term`] rather than the
/// `mk_*` builders, so constant folding, double-negation elimination and
/// commutative-operand canonicalization cannot pre-compute or reorder anything:
/// the generated shapes are exactly the shapes the walk sees.  Because operands
/// are drawn from the pool of already-built terms, the result is a genuine DAG
/// — many subterms are reachable by two or more paths, which is what exercises
/// the `tracked_compound_terms` memo.
///
/// Coverage is deliberate: Int/Real/Bool/BitVector/Array/String sorts, both
/// variables and constants, every arm of the walk that descends, the numeric
/// and non-numeric variants of `Apply`/`Select`/`DtSelector`, and a sample of
/// the kinds the walk deliberately does not descend into (`Store`, `Implies`,
/// `Xor`, `Distinct`, `Let`, `StrLen`).  `Store` is the one of those that is
/// still not descended into but *does* have an effect: it sets
/// `has_array_ops`, because a formula that only writes arrays still needs the
/// lazy array-axiom refinement loop (see `solver::array_axioms`).
#[allow(clippy::too_many_lines)]
fn build_corpus(manager: &mut TermManager, rounds: usize) -> Corpus {
    let int_sort = manager.sorts.int_sort;
    let real_sort = manager.sorts.real_sort;
    let bool_sort = manager.sorts.bool_sort;
    let bv8 = manager.sorts.bitvec(8);
    let bv4 = manager.sorts.bitvec(4);
    let str_sort = manager.sorts.string_sort();
    let arr_sort = manager.sorts.array(int_sort, int_sort);
    let arr_bool_sort = manager.sorts.array(int_sort, bool_sort);

    let mut c = Corpus {
        ints: Vec::new(),
        reals: Vec::new(),
        bools: Vec::new(),
        bvs: Vec::new(),
        arrays: Vec::new(),
        strings: Vec::new(),
        all: Vec::new(),
    };

    // Leaves: variables of every sort family, plus constants.
    for i in 0..3 {
        let v = manager.mk_var(&format!("i{i}"), int_sort);
        c.ints.push(v);
        c.all.push(v);
        let v = manager.mk_var(&format!("r{i}"), real_sort);
        c.reals.push(v);
        c.all.push(v);
        let v = manager.mk_var(&format!("b{i}"), bool_sort);
        c.bools.push(v);
        c.all.push(v);
        let v = manager.mk_var(&format!("v{i}"), bv8);
        c.bvs.push(v);
        c.all.push(v);
        let v = manager.mk_var(&format!("w{i}"), bv4);
        c.bvs.push(v);
        c.all.push(v);
        let v = manager.mk_var(&format!("a{i}"), arr_sort);
        c.arrays.push(v);
        c.all.push(v);
        let v = manager.mk_var(&format!("s{i}"), str_sort);
        c.strings.push(v);
        c.all.push(v);
    }
    for k in 0..3i64 {
        let t = manager.mk_int(k);
        c.ints.push(t);
        c.all.push(t);
    }
    let t = manager.mk_true();
    c.bools.push(t);
    c.all.push(t);
    let t = manager.mk_false();
    c.bools.push(t);
    c.all.push(t);
    let t = manager.mk_bitvec(7i64, 8);
    c.bvs.push(t);
    c.all.push(t);
    // A Bool-sorted array, whose `select` is non-numeric.
    let ab = manager.mk_var("ab", arr_bool_sort);
    c.arrays.push(ab);
    c.all.push(ab);

    let mut rng = Xorshift64(0x2545_F491_4F6C_DD1D);

    for round in 0..rounds {
        let pick = rng.below(40);
        let i0 = c.ints[rng.below(c.ints.len())];
        let i1 = c.ints[rng.below(c.ints.len())];
        let r0 = c.reals[rng.below(c.reals.len())];
        let b0 = c.bools[rng.below(c.bools.len())];
        let b1 = c.bools[rng.below(c.bools.len())];
        let v0 = c.bvs[rng.below(c.bvs.len())];
        let v1 = c.bvs[rng.below(c.bvs.len())];
        let a0 = c.arrays[rng.below(c.arrays.len())];
        let s0 = c.strings[rng.below(c.strings.len())];

        let (term, family) = match pick {
            // n-ary arithmetic / boolean (the `Add | Mul | And | Or` arm).
            0 => (
                manager.intern_term(TermKind::Add([i0, i1, i0].into_iter().collect()), int_sort),
                'i',
            ),
            1 => (
                manager.intern_term(TermKind::Mul([i0, i1].into_iter().collect()), int_sort),
                'i',
            ),
            2 => (
                manager.intern_term(TermKind::And([b0, b1].into_iter().collect()), bool_sort),
                'b',
            ),
            3 => (
                manager.intern_term(TermKind::Or([b0, b1, b0].into_iter().collect()), bool_sort),
                'b',
            ),
            // Real-sorted n-ary, to exercise the Real branch of the Var arm.
            4 => (
                manager.intern_term(TermKind::Add([r0, r0].into_iter().collect()), real_sort),
                'r',
            ),
            // Binary arithmetic / comparisons.
            5 => (manager.intern_term(TermKind::Sub(i0, i1), int_sort), 'i'),
            6 => (manager.intern_term(TermKind::Eq(i0, i1), bool_sort), 'b'),
            7 => (manager.intern_term(TermKind::Lt(i0, i1), bool_sort), 'b'),
            8 => (manager.intern_term(TermKind::Le(r0, r0), bool_sort), 'b'),
            9 => (manager.intern_term(TermKind::Gt(i0, i1), bool_sort), 'b'),
            10 => (manager.intern_term(TermKind::Ge(i0, i1), bool_sort), 'b'),
            // Bitvector binary family.
            11 => (manager.intern_term(TermKind::BvAdd(v0, v1), bv8), 'v'),
            12 => (manager.intern_term(TermKind::BvSub(v0, v1), bv8), 'v'),
            13 => (manager.intern_term(TermKind::BvMul(v0, v1), bv8), 'v'),
            14 => (manager.intern_term(TermKind::BvAnd(v0, v1), bv8), 'v'),
            15 => (manager.intern_term(TermKind::BvOr(v0, v1), bv8), 'v'),
            16 => (manager.intern_term(TermKind::BvXor(v0, v1), bv8), 'v'),
            17 => (manager.intern_term(TermKind::BvUlt(v0, v1), bool_sort), 'b'),
            18 => (manager.intern_term(TermKind::BvUle(v0, v1), bool_sort), 'b'),
            19 => (manager.intern_term(TermKind::BvSlt(v0, v1), bool_sort), 'b'),
            20 => (manager.intern_term(TermKind::BvSle(v0, v1), bool_sort), 'b'),
            21 => (manager.intern_term(TermKind::BvShl(v0, v1), bv8), 'v'),
            22 => (manager.intern_term(TermKind::BvLshr(v0, v1), bv8), 'v'),
            23 => (manager.intern_term(TermKind::BvAshr(v0, v1), bv8), 'v'),
            24 => (manager.intern_term(TermKind::BvConcat(v0, v1), bv8), 'v'),
            // Bit extraction.
            25 => (
                manager.intern_term(
                    TermKind::BvExtract {
                        high: 3,
                        low: 0,
                        arg: v0,
                    },
                    bv4,
                ),
                'v',
            ),
            // BV division / remainder (sets `has_bv_arith_ops`).
            26 => (manager.intern_term(TermKind::BvUdiv(v0, v1), bv8), 'v'),
            27 => (manager.intern_term(TermKind::BvSdiv(v0, v1), bv8), 'v'),
            28 => (manager.intern_term(TermKind::BvUrem(v0, v1), bv8), 'v'),
            29 => (manager.intern_term(TermKind::BvSrem(v0, v1), bv8), 'v'),
            // Unary.
            30 => (manager.intern_term(TermKind::Neg(i0), int_sort), 'i'),
            31 => (manager.intern_term(TermKind::Not(b0), bool_sort), 'b'),
            32 => (manager.intern_term(TermKind::BvNot(v0), bv8), 'v'),
            // Ite: numeric (an arithmetic atom in its own right) and Bool.
            33 => (
                manager.intern_term(TermKind::Ite(b0, i0, i1), int_sort),
                'i',
            ),
            34 => (
                manager.intern_term(TermKind::Ite(b0, b1, b0), bool_sort),
                'b',
            ),
            // Div / Mod: opaque arithmetic atoms.
            35 => (manager.intern_term(TermKind::Div(i0, i1), int_sort), 'i'),
            36 => (manager.intern_term(TermKind::Mod(i0, i1), int_sort), 'i'),
            // Apply of numeric / non-numeric sort.
            37 => {
                let f = manager.intern_str("f");
                let sort = if round % 3 == 0 {
                    int_sort
                } else if round % 3 == 1 {
                    real_sort
                } else {
                    bool_sort
                };
                let t = manager.intern_term(
                    TermKind::Apply {
                        func: f,
                        args: [i0, i1].into_iter().collect(),
                    },
                    sort,
                );
                (
                    t,
                    match sort {
                        s if s == int_sort => 'i',
                        s if s == real_sort => 'r',
                        _ => 'b',
                    },
                )
            }
            // Select (numeric and Bool-sorted).
            38 => {
                if round % 2 == 0 {
                    (manager.intern_term(TermKind::Select(a0, i0), int_sort), 'i')
                } else {
                    (
                        manager.intern_term(TermKind::Select(ab, i0), bool_sort),
                        'b',
                    )
                }
            }
            // Datatype selector, plus kinds that the walk deliberately does not
            // descend into (Store, Implies, Xor, Distinct, Let, StrLen).
            // `Store` still sets `has_array_ops` -- see `build_corpus`' doc.
            _ => {
                let sel = manager.intern_str("head");
                match round % 8 {
                    0 => (
                        manager.intern_term(
                            TermKind::DtSelector {
                                selector: sel,
                                arg: i0,
                            },
                            int_sort,
                        ),
                        'i',
                    ),
                    1 => (
                        manager.intern_term(
                            TermKind::DtSelector {
                                selector: sel,
                                arg: i0,
                            },
                            bool_sort,
                        ),
                        'b',
                    ),
                    2 => (
                        manager.intern_term(TermKind::Store(a0, i0, i1), arr_sort),
                        'a',
                    ),
                    3 => (
                        manager.intern_term(TermKind::Implies(b0, b1), bool_sort),
                        'b',
                    ),
                    4 => (manager.intern_term(TermKind::Xor(b0, b1), bool_sort), 'b'),
                    5 => (
                        manager.intern_term(
                            TermKind::Distinct([i0, i1].into_iter().collect()),
                            bool_sort,
                        ),
                        'b',
                    ),
                    6 => {
                        let n = manager.intern_str("bound");
                        (
                            manager.intern_term(
                                TermKind::Let {
                                    bindings: [(n, i0)].into_iter().collect(),
                                    body: b0,
                                },
                                bool_sort,
                            ),
                            'b',
                        )
                    }
                    _ => (manager.intern_term(TermKind::StrLen(s0), int_sort), 'i'),
                }
            }
        };

        c.all.push(term);
        match family {
            'i' => c.ints.push(term),
            'r' => c.reals.push(term),
            'b' => c.bools.push(term),
            'v' => c.bvs.push(term),
            'a' => c.arrays.push(term),
            _ => c.strings.push(term),
        }
    }

    c
}

// ---------------------------------------------------------------------
// Observable-effect capture
// ---------------------------------------------------------------------

/// Canonical rendering of every observable effect the walk can have: the three
/// term sets (sorted — `TermId`s are deterministic), the two sticky flags, the
/// trail **in order**, and a witness for each theory solver's own interning.
///
/// Takes `&mut Solver` because the intern-order witness has to call
/// [`oxiz_theories::arithmetic::ArithSolver::intern`] — the only available read
/// of that map.  `intern` is idempotent for an already-interned term and hands
/// out `VarId`s sequentially otherwise, so probing the whole corpus in a fixed
/// order yields a sequence that is identical across two runs exactly when the
/// walk's own `intern` calls agreed in both membership and order.  It is called
/// last, once everything else has been captured.
fn render_effects(solver: &mut Solver, probe_terms: &[TermId]) -> String {
    use core::fmt::Write as _;
    let mut out = String::new();

    let mut arith: Vec<TermId> = solver.arith_terms.iter().copied().collect();
    arith.sort_unstable();
    let mut bv: Vec<TermId> = solver.bv_terms.iter().copied().collect();
    bv.sort_unstable();
    let mut memo: Vec<TermId> = solver.tracked_compound_terms.iter().copied().collect();
    memo.sort_unstable();

    let _ = writeln!(out, "arith_terms={arith:?}");
    let _ = writeln!(out, "bv_terms={bv:?}");
    let _ = writeln!(out, "memo={memo:?}");
    let _ = writeln!(
        out,
        "flags: bv_arith={} array={}",
        solver.has_bv_arith_ops, solver.has_array_ops
    );
    let _ = writeln!(out, "trail_len={}", solver.trail.len());
    for op in &solver.trail {
        let _ = writeln!(out, "  {op:?}");
    }
    for &t in probe_terms {
        let has_bv = solver.bv.get_bv(t).is_some();
        let var = solver.arith.intern(t);
        let _ = writeln!(out, "  probe {t:?} bv={has_bv} arith_var={var:?}");
    }
    out
}

/// FNV-1a over a canonical rendering, so one `u64` pins the whole effect set.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------
// Pinned characterisation
// ---------------------------------------------------------------------

/// Replay the whole corpus in order on one solver — the way a real assertion
/// sequence hits the walk, with shared sub-DAGs and the memo interacting across
/// terms — and pin every observable effect.
///
/// The digest is over the full canonical rendering (sets, flags, the ordered
/// trail, and the per-term intern/BV witnesses), so it also pins the *order* of
/// the trail entries and of the `ArithSolver::intern` calls, not just their
/// multiset.  The individual counts are asserted alongside it purely so a
/// failure says *what* moved before the digest says *that* something moved.
#[test]
fn corpus_effects_match_the_recursive_implementation() {
    let mut manager = TermManager::new();
    let corpus = build_corpus(&mut manager, 600);
    let terms = corpus.all.clone();
    assert_eq!(
        terms.len(),
        628,
        "corpus construction must be deterministic"
    );

    let mut solver = Solver::new();
    for &t in &terms {
        solver.track_theory_vars(t, &manager);
    }

    assert_eq!(solver.arith_terms.len(), 69, "arith_terms");
    assert_eq!(solver.bv_terms.len(), 6, "bv_terms");
    assert_eq!(solver.tracked_compound_terms.len(), 528, "memo");
    assert_eq!(solver.trail.len(), 603, "trail length");
    assert!(solver.has_bv_arith_ops, "bvudiv/bvsdiv/... set this flag");
    assert!(solver.has_array_ops, "select sets this flag");

    let digest = fnv1a(&render_effects(&mut solver, &terms));
    assert_eq!(
        digest, 0x3d75_800b_80fe_510f,
        "observable effects of the theory-variable walk changed"
    );
}

/// Same corpus, but each term walked on its own fresh solver, and the
/// per-solver digests combined.  This pins the effect of each term *in
/// isolation* — the memo cannot mask a lost child here, because nothing was
/// claimed before the term was walked — so it catches a regression that the
/// accumulated run above could hide.
#[test]
fn per_term_effects_match_the_recursive_implementation() {
    let mut manager = TermManager::new();
    let corpus = build_corpus(&mut manager, 600);
    let terms = corpus.all.clone();

    let mut combined: u64 = 0;
    for &t in &terms {
        let mut solver = Solver::new();
        solver.track_theory_vars(t, &manager);
        let d = fnv1a(&render_effects(&mut solver, &terms));
        combined = combined.rotate_left(7) ^ d;
    }
    assert_eq!(
        combined, 0xd969_4bf7_3c09_27f5,
        "per-term observable effects of the theory-variable walk changed"
    );
}

// ---------------------------------------------------------------------
// Hand-checkable behaviour, in case the digests above ever need re-pinning
// ---------------------------------------------------------------------

/// The three registration kinds and the memo, on a formula small enough to
/// verify by eye: `(bvudiv v (bvadd v w))` compared against `(f i)` and
/// `(mod i 7)`.
#[test]
fn registers_arith_bv_and_opaque_atoms() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let bv8 = manager.sorts.bitvec(8);

    let i = manager.mk_var("i", int_sort);
    let seven = manager.mk_int(7);
    let modt = manager.intern_term(TermKind::Mod(i, seven), int_sort);
    let f = manager.intern_str("f");
    let fi = manager.intern_term(
        TermKind::Apply {
            func: f,
            args: [i].into_iter().collect(),
        },
        int_sort,
    );
    let v = manager.mk_var("v", bv8);
    let w = manager.mk_var("w", bv8);
    let add = manager.intern_term(TermKind::BvAdd(v, w), bv8);
    let udiv = manager.intern_term(TermKind::BvUdiv(v, add), bv8);

    solver.track_theory_vars(modt, &manager);
    solver.track_theory_vars(fi, &manager);
    solver.track_theory_vars(udiv, &manager);

    // `i` (a numeric variable), `mod` and `f(i)` (opaque numeric atoms).
    assert!(solver.arith_terms.contains(&i));
    assert!(solver.arith_terms.contains(&modt));
    assert!(solver.arith_terms.contains(&fi));
    // The integer literal is a value, not a variable.
    assert!(!solver.arith_terms.contains(&seven));
    // Bitvector leaves get a BV variable of the right width, and are also
    // interned arithmetically (BV comparisons run as bounded integers).
    assert!(solver.bv_terms.contains(&v));
    assert!(solver.bv_terms.contains(&w));
    assert!(solver.bv.get_bv(v).is_some());
    // Compound nodes are memoised, leaves are not.
    assert!(solver.tracked_compound_terms.contains(&modt));
    assert!(solver.tracked_compound_terms.contains(&add));
    assert!(solver.tracked_compound_terms.contains(&udiv));
    assert!(!solver.tracked_compound_terms.contains(&i));
    // `bvudiv` is what makes the BV conflict detector look for arithmetic ops.
    assert!(solver.has_bv_arith_ops);
}

/// A sub-expression reachable by two paths is claimed once, so its trail entry
/// appears once: the memo is a *deduplication* of work, and double-journalling
/// it would make one `pop` un-claim a term another scope still owns.
#[test]
fn shared_subterm_is_claimed_exactly_once() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let i0 = manager.mk_var("i0", int_sort);
    let i1 = manager.mk_var("i1", int_sort);
    let shared = manager.intern_term(TermKind::Sub(i0, i1), int_sort);
    // `shared` sits under both operands of the top-level node.
    let top = manager.intern_term(
        TermKind::Add([shared, shared].into_iter().collect()),
        int_sort,
    );

    solver.track_theory_vars(top, &manager);

    let claims = solver
        .trail
        .iter()
        .filter(|op| matches!(op, TrailOp::TrackedCompoundAdded { term } if *term == shared))
        .count();
    assert_eq!(claims, 1, "a shared subterm must be claimed exactly once");
    let arith_adds = solver
        .trail
        .iter()
        .filter(|op| matches!(op, TrailOp::ArithTermAdded { term } if *term == i0))
        .count();
    assert_eq!(arith_adds, 1, "a variable must be interned exactly once");
}

/// The walk visits operands left to right, which is the order the trail (and
/// hence `ArithSolver`'s `VarId` allocation) records them in.  Pinning it keeps
/// the explicit stack's reverse-push discipline honest: pushing children in
/// forward order would silently reverse every trail segment.
#[test]
fn children_are_visited_left_to_right() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let a = manager.mk_var("a", int_sort);
    let b = manager.mk_var("b", int_sort);
    let c = manager.mk_var("c", int_sort);
    let inner = manager.intern_term(TermKind::Sub(b, c), int_sort);
    let top = manager.intern_term(TermKind::Add([a, inner].into_iter().collect()), int_sort);

    solver.track_theory_vars(top, &manager);

    let order: Vec<TermId> = solver
        .trail
        .iter()
        .filter_map(|op| match op {
            TrailOp::ArithTermAdded { term } => Some(*term),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        vec![a, b, c],
        "operands must be visited left to right"
    );
}

// ---------------------------------------------------------------------
// Native-stack bound
// ---------------------------------------------------------------------

/// `bvnot(bvnot(...bvnot(x)...))`, `depth` levels over an 8-bit variable, built
/// with a plain loop — a recursive builder would overflow before the walk under
/// test even ran.
///
/// `intern_term` is used directly rather than `mk_bv_not` so that no folding
/// rule (double-negation elimination) can collapse the chain, and each level's
/// argument is the unique previous level's `TermId`, so hash-consing cannot
/// merge two levels either.
fn build_bvnot_chain(manager: &mut TermManager, depth: usize) -> TermId {
    let bv8 = manager.sorts.bitvec(8);
    let mut term = manager.mk_var("deep_bv", bv8);
    for _ in 0..depth {
        term = manager.intern_term(TermKind::BvNot(term), bv8);
    }
    term
}

/// The point of the explicit-stack conversion: an embedder calling OxiZ from a
/// worker thread with a conventional ~1 MiB stack must get a normal return, not
/// a process abort.  The pinned stack here is an eighth of that, paired with an
/// eighth of the depth, which pins the same bytes-per-frame ratio.
///
/// A Rust stack overflow is not a panic — it is a fatal runtime abort that
/// `catch_unwind` cannot intercept — so **the fact that this test returns at all
/// is itself the assertion**.  The recursive predecessor died at ~1556 levels on
/// 1 MiB at `opt-level = 0` and ~4370 at `opt-level = 1`; on the 128 KiB stack
/// used here that scales to ~195 and ~546 levels, and 25_000 is two orders of
/// magnitude past both.  Asserting the exact memo size on top of that rules out
/// a silent partial walk (the "unhandled input quietly dropped" failure mode a
/// bare depth cap would have produced instead).
#[test]
fn survives_a_deep_chain_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/200k -> 128 KiB/25k): the
    // ~5 B-per-frame threshold is the pin, so never raise one alone.
    const STACK_SIZE: usize = 1 << 17; // 128 KiB
    const DEPTH: usize = 25_000;

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut solver = Solver::new();
            let mut manager = TermManager::new();
            let deepest = build_bvnot_chain(&mut manager, DEPTH);

            solver.track_theory_vars(deepest, &manager);

            // Every `bvnot` level is a claimed compound; the one leaf variable
            // is a BV term instead.
            assert_eq!(
                solver.tracked_compound_terms.len(),
                DEPTH,
                "the walk must reach every level of a {DEPTH}-deep chain, not stop partway"
            );
            assert_eq!(solver.bv_terms.len(), 1, "the chain's single leaf variable");
        })
        .expect("spawning a 128 KiB-stack thread should succeed");

    handle
        .join()
        .expect("the theory-variable walk must return on 128 KiB instead of overflowing it");
}
