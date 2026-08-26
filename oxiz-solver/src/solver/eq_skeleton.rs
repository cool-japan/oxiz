//! A static-transitivity fast path for formulas that are pure Equality Logic:
//! Boolean combinations of `(= a b)` / `(not (= a b))` over uninterpreted-sort
//! constants, with no functions, arithmetic, bit-vectors, arrays, strings, or
//! quantifiers anywhere in the assertion set.
//!
//! # Why CDCL(T) needs help here
//!
//! A disjunctive chain of equalities (`(= a1 a2) \/ (= a2 a3)`, repeated over
//! many variables so many different "who equals whom" assignments are
//! individually consistent) is the textbook adversarial case for combining a
//! SAT core with an EUF theory: the SAT skeleton has no idea that equality is
//! transitive, so it can commit to an assignment of the equality atoms that
//! *looks* locally fine and only discovers the contradiction — or the lack of
//! one — after EUF has walked the union-find and reported back, once per
//! conflict. That round trip, repeated once per branch the chain admits, is
//! what makes these instances blow up exponentially in practice even though
//! the underlying decision problem is easy.
//!
//! # The fix: give transitivity to the SAT core directly
//!
//! Build the equality graph (vertices are the constants, edges are the `(=
//! a b)` atoms actually written in the formula), close it into a *chordal*
//! graph by a min-fill elimination ordering (adding one fresh Boolean
//! variable per fill-in "chord" edge), and add the three transitivity clauses
//! for every triangle the chordal graph contains:
//!
//! ```text
//! e_xy /\ e_yz => e_xz        e_xy /\ e_xz => e_yz        e_xz /\ e_yz => e_xy
//! ```
//!
//! A chordal graph's only chordless cycles are triangles, so this clause set
//! is exactly what is needed (no cycle of length > 3 can smuggle in an
//! inconsistent assignment) and nothing more (no clause is spent on a cycle
//! transitivity already rules out through a shorter one). This is the
//! "Sparse" construction from Bryant & Velev, *Boolean Satisfiability with
//! Transitivity Constraints* (ACM TOCL, 2002): plain SAT over the original
//! Boolean skeleton conjoined with these clauses decides the formula exactly,
//! because every assignment satisfying the triangle clauses over a chordal
//! completion is, by construction, consistent with *some* genuine equivalence
//! relation on the vertices — which is precisely what "transitive, symmetric,
//! reflexive equality" requires and a bare Boolean skeleton cannot enforce on
//! its own.
//!
//! # The pipeline, and where it declines
//!
//! Four stages run in order, and the first three touch nothing outside this
//! module:
//!
//! 1. [`Solver::collect_equality_skeleton`] walks the assertions and either
//!    returns the equality graph or declines.
//! 2. [`chordal_completion`] eliminates vertices, recording the fill edges it
//!    needs and the order it used.
//! 3. [`triangles_from_elimination_order`] reads the triangles straight off
//!    that order.
//! 4. Only then are SAT variables allocated for the fill edges and the clauses
//!    asserted.
//!
//! Splitting it this way is what lets the size guards be *clean*: a formula
//! whose graph is too large, or whose chordal completion would cost more
//! clauses than [`MAX_TRANSITIVITY_TRIANGLES`] allows, is abandoned before the
//! SAT core has been touched at all, and the caller falls through to the
//! ordinary search with nothing added and nothing to undo.
//!
//! # UNSAT is trusted; SAT is checked
//!
//! There is no model to hold a false `Unsat` against, so an `Unsat` from this
//! path is reported directly — its soundness rests on the chordal
//! construction above being correctly implemented, which is what this
//! module's tests exist to pin down, not something a runtime check could
//! independently confirm.
//!
//! A reported `Sat`, in contrast, *is* checked: after a satisfying
//! assignment, an independent union-find over every edge (real or
//! fill-in) the SAT core decided `true` re-derives the same equivalence
//! classes the clauses were supposed to enforce, and every *real* equality
//! atom's decided truth value is cross-checked against it — catching a bug in
//! this module's own graph construction (a missed triangle, a misrouted edge
//! variable, …) before it becomes a wrong verdict. `model_refutes_assertions`
//! (the solver's usual model-verification backstop) is deliberately not
//! reused for this: its evaluator falls back on the arithmetic/EUF theory
//! solvers for concrete values, neither of which this fast path ever runs, so
//! every uninterpreted-constant equality would evaluate `Undetermined` and
//! the gate would pass *anything* — the union-find re-derivation below is
//! what makes the check actually load-bearing.
//!
//! The classes that survive that check are handed to the model layer as
//! *class memberships*, not as values: an uninterpreted sort has no literals
//! to print, so `(get-model)` renders them through the same `@uc_S_n` abstract
//! witnesses it uses for any other unconstrained constant of that sort, with
//! constants in one class sharing a witness and constants in different classes
//! getting different ones.

use rustc_hash::FxHashSet;

use oxiz_core::sort::SortKind;

use super::*;

/// Largest equality graph this path will take on, in vertices.
///
/// OxiZ tuning decision. Chordalization is quadratic in the vertex count per
/// elimination step and the completion can approach a clique, so the work and
/// the clause count both grow steeply; past this size the exponential blow-up
/// this path exists to avoid is no longer the dominant cost.
const MAX_EQUALITY_VERTICES: usize = 512;

/// Largest equality graph this path will take on, in distinct edges.
///
/// OxiZ tuning decision. A graph can be small in vertices and still dense
/// enough to make its chordal completion expensive, so density is bounded
/// separately from size.
const MAX_EQUALITY_EDGES: usize = 8192;

/// How many triangles the completed graph may contain before the whole path
/// is abandoned.
///
/// OxiZ tuning decision. Each triangle costs three ternary clauses, so this
/// is really a clause budget: past it, handing the SAT core the eager
/// encoding is a worse trade than letting the ordinary CDCL(T) search do its
/// theory round trips. Checked before a single clause is asserted, so
/// exceeding it costs nothing but the counting.
const MAX_TRANSITIVITY_TRIANGLES: usize = 200_000;

/// A vertex-index union-find with path compression, scoped to one fast-path
/// attempt: built once from every edge the SAT core decided `true` (both the
/// atoms the input formula actually asserts and the fill-in chords this
/// module added), then used to re-derive what the chordal transitivity
/// clauses were supposed to guarantee.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// The uninterpreted-sort constant `t` maps to a distinct union-find class
/// per vertex; `edge` is that class's Boolean variable for the graph edge
/// `(x, y)`.
struct EqualityGraph {
    /// Every distinct constant that appears in some equality atom, in
    /// first-seen order — index into this is the union-find vertex id.
    vertices: Vec<TermId>,
    /// Undirected adjacency, kept for chordalization; grows as fill edges are
    /// added.
    ///
    /// Hash sets, in this codebase's usual style: nothing downstream depends
    /// on neighbour *order*, and the two places that would (choosing which
    /// vertex to eliminate, and laying out fill edges) sort explicitly rather
    /// than leaning on a container to do it for them.
    adjacency: Vec<FxHashSet<usize>>,
    /// Canonical (min-index, max-index) vertex pair -> the Boolean variable
    /// standing for that pair's equality, for every edge the graph currently
    /// has (both the formula's own atoms and any fill-in chords added since).
    edge_var: FxHashMap<(usize, usize), Var>,
    /// The subset of `edge_var`'s keys that came directly from the formula
    /// (as opposed to a fill-in chord) — what the post-solve check
    /// cross-verifies.
    direct_edges: Vec<(usize, usize)>,
}

fn canonical_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// A vertex's still-uneliminated neighbours, in ascending index order.
///
/// Sorting here is what makes every later decision reproducible: min-fill tie
/// breaks, the order fill edges are laid down in, and therefore the order
/// fresh SAT variables are allocated in, all follow from this sequence.
fn live_neighbours(
    adjacency: &[FxHashSet<usize>],
    vertex: usize,
    eliminated: &[bool],
) -> Vec<usize> {
    let mut neighbours: Vec<usize> = adjacency[vertex]
        .iter()
        .copied()
        .filter(|&other| !eliminated[other])
        .collect();
    neighbours.sort_unstable();
    neighbours
}

/// How many edges are missing from `neighbours` before it is a clique.
fn fill_cost(adjacency: &[FxHashSet<usize>], neighbours: &[usize]) -> usize {
    let mut missing = 0;
    for (offset, &left) in neighbours.iter().enumerate() {
        for &right in &neighbours[offset + 1..] {
            if !adjacency[left].contains(&right) {
                missing += 1;
            }
        }
    }
    missing
}

/// Make `adjacency` chordal by min-fill elimination, and report the order
/// used.
///
/// Repeatedly eliminates whichever remaining vertex needs the fewest new
/// edges to turn its neighbourhood into a clique, adding exactly those edges.
/// The returned sequence is a *perfect elimination ordering* of the resulting
/// graph — for each vertex, the neighbours that come after it in the sequence
/// form a clique — which is the property
/// [`triangles_from_elimination_order`] reads the triangles off.
///
/// Ties in fill cost break toward the lowest vertex index, so the same graph
/// always produces the same completion.
fn chordal_completion(adjacency: &mut [FxHashSet<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut eliminated = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    for _ in 0..n {
        let Some(next) = (0..n)
            .filter(|&vertex| !eliminated[vertex])
            .min_by_key(|&vertex| {
                let neighbours = live_neighbours(adjacency, vertex, &eliminated);
                (fill_cost(adjacency, &neighbours), vertex)
            })
        else {
            break;
        };

        let neighbours = live_neighbours(adjacency, next, &eliminated);
        for (offset, &left) in neighbours.iter().enumerate() {
            for &right in &neighbours[offset + 1..] {
                adjacency[left].insert(right);
                adjacency[right].insert(left);
            }
        }
        eliminated[next] = true;
        order.push(next);
    }
    order
}

/// Every triangle of a chordal graph, each reported once, as vertex indices.
///
/// A perfect elimination ordering hands these over without any search: when a
/// vertex is eliminated, its remaining neighbourhood was just made into a
/// clique, so every pair drawn from the neighbours that outlive it closes a
/// triangle with it. Counting each triangle at its earliest-eliminated vertex
/// is what stops it being reported three times.
fn triangles_from_elimination_order(
    adjacency: &[FxHashSet<usize>],
    order: &[usize],
) -> Vec<[usize; 3]> {
    let mut position = vec![usize::MAX; adjacency.len()];
    for (slot, &vertex) in order.iter().enumerate() {
        position[vertex] = slot;
    }

    let mut triangles = Vec::new();
    for &vertex in order {
        let mut later: Vec<usize> = adjacency[vertex]
            .iter()
            .copied()
            .filter(|&other| position[other] > position[vertex])
            .collect();
        later.sort_unstable();
        for (offset, &left) in later.iter().enumerate() {
            for &right in &later[offset + 1..] {
                debug_assert!(
                    adjacency[left].contains(&right),
                    "a perfect elimination ordering makes every later neighbourhood a clique"
                );
                triangles.push([vertex, left, right]);
            }
        }
    }
    triangles
}

impl Solver {
    /// Uninterpreted-sort constant terms only: `Bool`, `Int`, `Real`,
    /// `BitVec`, arrays, strings, floats and datatypes all have their own
    /// theory-level equality semantics beyond bare transitivity (or, for
    /// `Bool`, are really an iff), so mixing any of them in disqualifies the
    /// whole formula from this path.
    ///
    /// `RoundingMode` is deliberately excluded for the same reason, even
    /// though its terms are nullary `Var`s that look exactly like plain
    /// uninterpreted constants: the sort has a *fixed cardinality of five*,
    /// enforced by axioms the solver asserts separately
    /// (`context::rounding_mode`).  A pure-equality graph reasons only about
    /// the equalities written in the formula, so it would happily satisfy six
    /// pairwise-distinct rounding modes.  Matching on
    /// `SortKind::Uninterpreted` rather than "is a nullary `Var`" is what
    /// keeps it out — conservatively, at the level of a
    /// single atom's operand sorts rather than trying to enumerate every way
    /// a formula could smuggle theory content in elsewhere.
    fn is_plain_uninterpreted_constant(term: TermId, manager: &TermManager) -> Option<()> {
        let node = manager.get(term)?;
        if !matches!(node.kind, TermKind::Var(_)) {
            return None;
        }
        let sort = manager.sorts.get(node.sort)?;
        matches!(sort.kind, SortKind::Uninterpreted(_)).then_some(())
    }

    /// Walk every assertion with an explicit stack (never native recursion —
    /// an adversarially deep Boolean skeleton must not be able to exhaust the
    /// call stack just because this fast path looked at it first), building
    /// the equality graph's *direct* edges. `None` — with no partial result —
    /// the moment anything outside `True`/`False`/`Not`/`And`/`Or`/`Eq` over a
    /// same-sort pair of plain uninterpreted constants is found: one
    /// disqualifying construct anywhere in the assertion set takes the whole
    /// formula out of this path.
    ///
    /// A graph past [`MAX_EQUALITY_VERTICES`] or [`MAX_EQUALITY_EDGES`] is
    /// declined the same way, for cost rather than correctness reasons.
    fn collect_equality_skeleton(&self, manager: &TermManager) -> Option<EqualityGraph> {
        let mut vertices: Vec<TermId> = Vec::new();
        let mut vertex_of: FxHashMap<TermId, usize> = FxHashMap::default();
        let mut direct_edges: FxHashMap<(usize, usize), Var> = FxHashMap::default();

        let mut stack: Vec<TermId> = self.assertions.clone();
        while let Some(term) = stack.pop() {
            let node = manager.get(term)?;
            match &node.kind {
                TermKind::True | TermKind::False => {}
                TermKind::Not(inner) => stack.push(*inner),
                TermKind::And(args) | TermKind::Or(args) => stack.extend(args.iter().copied()),
                TermKind::Eq(a, b) => {
                    Self::is_plain_uninterpreted_constant(*a, manager)?;
                    Self::is_plain_uninterpreted_constant(*b, manager)?;
                    if manager.get(*a)?.sort != manager.get(*b)?.sort {
                        return None;
                    }
                    let &var = self.term_to_var.get(&term)?;
                    let ia = *vertex_of.entry(*a).or_insert_with(|| {
                        vertices.push(*a);
                        vertices.len() - 1
                    });
                    let ib = *vertex_of.entry(*b).or_insert_with(|| {
                        vertices.push(*b);
                        vertices.len() - 1
                    });
                    if ia != ib {
                        direct_edges.insert(canonical_pair(ia, ib), var);
                    }
                    if vertices.len() > MAX_EQUALITY_VERTICES
                        || direct_edges.len() > MAX_EQUALITY_EDGES
                    {
                        return None;
                    }
                }
                // `ite`, `xor`, `=>`, `distinct`, function application,
                // arithmetic, bit-vectors, arrays, strings, datatypes,
                // quantifiers: every one of these carries semantics this
                // module does not model, so any occurrence disqualifies the
                // whole formula.
                _ => return None,
            }
        }
        if direct_edges.is_empty() {
            return None;
        }

        let n = vertices.len();
        let mut adjacency: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
        for &(i, j) in direct_edges.keys() {
            adjacency[i].insert(j);
            adjacency[j].insert(i);
        }
        let edges_key_list: Vec<(usize, usize)> = direct_edges.keys().copied().collect();
        Some(EqualityGraph {
            vertices,
            adjacency,
            edge_var: direct_edges,
            direct_edges: edges_key_list,
        })
    }

    /// Give every fill-in edge of the completed graph a Boolean variable, and
    /// assert the transitivity clauses of every triangle not already covered.
    ///
    /// Variables come from [`Solver::get_or_create_var`] keyed on the real
    /// equality term for the pair, exactly like any atom the formula wrote
    /// itself — so a chord that the input *does* mention somewhere else, and a
    /// repeated `check-sat` on an unchanged goal, both resolve to the variable
    /// already in use rather than a fresh one.
    ///
    /// Returns the number of triangles whose clauses were newly asserted.
    fn install_transitivity(
        &mut self,
        graph: &mut EqualityGraph,
        triangles: &[[usize; 3]],
        manager: &mut TermManager,
    ) -> usize {
        for &triangle in triangles {
            for (offset, &left) in triangle.iter().enumerate() {
                for &right in &triangle[offset + 1..] {
                    let pair = canonical_pair(left, right);
                    if graph.edge_var.contains_key(&pair) {
                        continue;
                    }
                    let eq_term = manager.mk_eq(graph.vertices[left], graph.vertices[right]);
                    let var = self.get_or_create_var(eq_term);
                    graph.edge_var.insert(pair, var);
                }
            }
        }

        let mut asserted = 0;
        for &triangle in triangles {
            // Key the dedup on the constants, not on vertex indices: indices
            // are an artefact of the order this particular walk happened to
            // meet the terms in, and a repeated `check-sat` must recognise the
            // same triangle whatever order that was.
            let Some(edges) = self.triangle_edge_vars(graph, triangle) else {
                // Unreachable: every pair of this triangle was just given a
                // variable above. Declining rather than indexing blindly if
                // that invariant is ever broken — and declining *before* the
                // bookkeeping below, so a triangle that emits no clause is
                // never recorded as though it had.
                continue;
            };
            let mut key = triangle.map(|vertex| graph.vertices[vertex]);
            key.sort_unstable();
            if !self.eq_transitivity_triangles.insert(key) {
                continue;
            }
            self.trail
                .push(TrailOp::EqTransitivityTriangleAdded { triangle: key });

            // "Any two of the three edges being true forces the third" —
            // three clauses, one per choice of which edge is the conclusion.
            for conclusion in 0..edges.len() {
                self.sat
                    .add_clause(edges.iter().enumerate().map(|(slot, &edge)| {
                        if slot == conclusion {
                            Lit::pos(edge)
                        } else {
                            Lit::neg(edge)
                        }
                    }));
            }
            asserted += 1;
        }
        asserted
    }

    /// The Boolean variables of a triangle's three edges.
    fn triangle_edge_vars(&self, graph: &EqualityGraph, triangle: [usize; 3]) -> Option<[Var; 3]> {
        let [x, y, z] = triangle;
        Some([
            *graph.edge_var.get(&canonical_pair(x, y))?,
            *graph.edge_var.get(&canonical_pair(y, z))?,
            *graph.edge_var.get(&canonical_pair(x, z))?,
        ])
    }

    /// Re-derive the equivalence classes the transitivity clauses were
    /// supposed to enforce, independently of them: union every edge (direct
    /// or fill-in) the SAT core decided `true`, then check that every
    /// *direct* edge's decided truth agrees with whether its endpoints ended
    /// up in the same class. A mismatch can only mean this module's own
    /// graph construction missed something (a fill edge, a triangle, a
    /// misrouted variable) — the clauses added are exactly the ones a correct
    /// chordal completion calls for, so if they were all present and correct,
    /// agreement here is guaranteed by the Bryant–Velev argument, not merely
    /// likely.
    fn verify_and_build_classes(&self, graph: &EqualityGraph) -> Option<Vec<usize>> {
        let sat_model = self.sat.model();
        let is_true = |var: Var| sat_model.get(var.index()).is_some_and(|v| v.is_true());

        let n = graph.vertices.len();
        let mut uf = UnionFind::new(n);
        for (&(i, j), &var) in &graph.edge_var {
            if is_true(var) {
                uf.union(i, j);
            }
        }
        for &(i, j) in &graph.direct_edges {
            let &var = graph.edge_var.get(&(i, j))?;
            let decided_true = is_true(var);
            let same_class = uf.find(i) == uf.find(j);
            if decided_true != same_class {
                return None;
            }
        }
        Some((0..n).map(|v| uf.find(v)).collect())
    }

    /// Entry point: if the whole assertion set is pure Equality Logic, decide
    /// it with plain SAT over the skeleton plus static transitivity clauses
    /// and return the verdict; otherwise (impure input, an oversized graph,
    /// or — defensively — a `Sat` this module's own re-derivation could not
    /// confirm) return `None` and add nothing further, so the caller falls
    /// through to the ordinary CDCL(T) search. Any transitivity clauses
    /// already installed by that point are harmless there: each one is a
    /// sound consequence of equality's semantics regardless of which theory
    /// eventually confirms the same facts.
    pub(super) fn try_pure_equality_fast_path(
        &mut self,
        manager: &mut TermManager,
    ) -> Option<SolverResult> {
        // Cheap prefilter ahead of the full assertion walk: these are
        // already tracked incrementally as assertions come in (see
        // `track_theory_vars`/`register_asserted_quantifiers`/the array/BV
        // encoders), so a formula that is quantified, touches arrays or
        // bit-vector arithmetic, or has any `Int`/`Real` term at all is
        // known impure in O(1) without walking a single term. This only
        // ever *skips* the walk for input it would have rejected anyway — a
        // formula genuinely restricted to Boolean connectives over
        // uninterpreted-sort equality never sets any of these, so nothing
        // that belongs on this path is filtered out here.
        if self.has_quantifiers
            || self.has_array_ops
            || self.has_bv_arith_ops
            || !self.arith_terms.is_empty()
        {
            return None;
        }
        // A previous verdict's classes must not outlive it: this attempt
        // either replaces them or leaves the model layer with nothing from
        // this module at all.
        self.equality_skeleton_classes.clear();

        let mut graph = self.collect_equality_skeleton(manager)?;
        let order = chordal_completion(&mut graph.adjacency);
        let triangles = triangles_from_elimination_order(&graph.adjacency, &order);
        if triangles.len() > MAX_TRANSITIVITY_TRIANGLES {
            // Nothing has been asserted yet, so declining here leaves the
            // solver exactly as it was found.
            return None;
        }
        self.install_transitivity(&mut graph, &triangles, manager);

        match self.sat.solve() {
            SatResult::Unsat => {
                // Same downgrade as `check_core`'s `Unsat` arm: this path
                // solves the very same `self.sat`, so a model-blocking clause
                // left there by an earlier `check` restricts it too, and "no
                // model outside the excluded region" is not `unsat`. See
                // `crate::solver::model_blocking`.
                if self.blocking_clauses_present() {
                    self.model = None;
                    self.unsat_core = None;
                    return Some(SolverResult::Unknown);
                }
                self.build_unsat_core();
                Some(SolverResult::Unsat)
            }
            SatResult::Sat => {
                let classes = self.verify_and_build_classes(&graph);
                let Some(classes) = classes else {
                    self.sat.backtrack_to_root();
                    return None;
                };
                self.build_model(manager);
                // Class memberships, not values. An uninterpreted sort has no
                // literals, so handing the model a concrete term here (an
                // `Int` tag, say) would make `(get-model)` print a value of
                // the wrong sort. Recording the partition instead lets the
                // model layer's existing `@uc_S_n` witness machinery name the
                // classes: same class, same witness; different classes,
                // different witnesses.
                for (vertex, &class) in graph.vertices.iter().zip(classes.iter()) {
                    self.equality_skeleton_classes.insert(*vertex, class as u32);
                }
                self.unsat_core = None;
                Some(SolverResult::Sat)
            }
            SatResult::Unknown => {
                self.sat.backtrack_to_root();
                None
            }
        }
    }
}

#[cfg(test)]
mod tests;
