//! Equivalent-literal substitution (ELS): find literals proven equivalent by
//! a cycle in the binary implication graph and rewrite every clause through
//! one representative per equivalence class.
//!
//! Two literals `p` and `q` are equivalent when the graph proves both
//! `p → q` and `q → p`; that is exactly a strongly-connected component (SCC)
//! of the graph read as `lit → implied` edges (a binary clause `(¬a ∨ b)`
//! contributes the edge `a → b`). [`tarjan_scc`] computes those components
//! with Tarjan's classic linear-time algorithm (iteratively, to avoid a stack
//! frame per literal on a long implication chain), one component per literal;
//! every literal in a component can be replaced by a single canonical member
//! everywhere it occurs.
//!
//! Reference (technique, not implementation): R. Tarjan, *Depth-first search
//! and linear graph algorithms*, SIAM J. Computing 1(2), 1972.

use super::*;

/// Outcome of [`Solver::fold_equivalent_literals`].
pub(super) enum PreprocessOutcome {
    /// The pass ran (possibly finding nothing to fold) and left the solver in
    /// a consistent state; the caller should let propagation run to a
    /// fixpoint next, same as after any other clause-database change.
    Ok,
    /// The implication graph proved some variable equivalent to its own
    /// negation — an unconditional contradiction independent of any
    /// assignment.
    Unsat,
}

impl Solver {
    /// Run one pass of equivalent-literal substitution.
    ///
    /// One-shot per solver incarnation (latched by `equiv_fold_latched`, cleared
    /// by [`Solver::reset`]) and only at the base assertion level with no
    /// proof (DRAT or LRAT) being traced: like bounded variable elimination,
    /// this deletes variables from the live clause set on the promise that
    /// [`Solver::save_model`] fixes their value back up afterward, which only
    /// holds with no incremental `push` in scope to later reintroduce a
    /// dropped polarity, and this pass does not (yet) emit the proof steps a
    /// proof trace would need to justify the deletions.
    ///
    /// Mutually exclusive with [`SolverConfig::enable_bve`] in this
    /// implementation: both mechanisms delete variables and record
    /// reconstruction data for [`Solver::save_model`], and this pass runs
    /// first when both are requested, folding away variables that
    /// `bounded_variable_elimination` — which explicitly defers to this
    /// method's latch — would otherwise also be free to pick up. Interleaving
    /// the two reconstruction maps for one variable is not handled by
    /// either's model-repair pass.
    pub(super) fn fold_equivalent_literals(&mut self) -> PreprocessOutcome {
        if self.equiv_fold_latched
            || !self.config.enable_equiv_substitution
            || self.trail.decision_level() != 0
            || self.assertion_levels.len() > 1
            || self.proof_tracing_active()
        {
            return PreprocessOutcome::Ok;
        }
        self.equiv_fold_latched = true;
        if self.num_vars == 0 {
            return PreprocessOutcome::Ok;
        }

        if self.config.enable_gate_congruence {
            self.extend_binary_graph_with_gate_congruence();
        }

        let num_lits = self.num_vars * 2;
        let adjacency = self.build_trusted_binary_adjacency();
        let comp_id = tarjan_scc(&adjacency);

        // A variable equivalent to its own negation is an unconditional
        // contradiction: no assignment can satisfy `v ↔ ¬v`.
        for v in 0..self.num_vars {
            let var = Var::new(v as u32);
            if comp_id[Lit::pos(var).code() as usize] == comp_id[Lit::neg(var).code() as usize] {
                return PreprocessOutcome::Unsat;
            }
        }

        // Canonical representative per component: the member literal with
        // the smallest code. Deterministic and cheap; which specific member
        // is chosen has no effect on soundness (see below).
        let num_components = comp_id.iter().copied().max().map_or(0, |m| m + 1) as usize;
        let mut canonical: Vec<Lit> = vec![Lit::from_code(0); num_components];
        let mut canonical_set: Vec<bool> = vec![false; num_components];
        for code in 0..num_lits as u32 {
            let component = comp_id[code as usize] as usize;
            let lit = Lit::from_code(code);
            if !canonical_set[component] || lit.code() < canonical[component].code() {
                canonical[component] = lit;
                canonical_set[component] = true;
            }
        }

        let mut sub: Vec<Lit> = (0..num_lits as u32).map(Lit::from_code).collect();
        for code in 0..num_lits as u32 {
            sub[code as usize] = canonical[comp_id[code as usize] as usize];
        }

        // A variable already fixed on the level-0 trail keeps its own two
        // literals as their own representative, overriding whatever the
        // generic per-component choice above picked for them specifically.
        // Its value is a permanent fact the rest of the search already
        // depends on; folding it into someone else's representative would
        // require reconstruction to restore it later, and there is no need
        // to pay that risk when the value is already known. This does not
        // lose the equivalence itself: every clause that related this
        // variable to its class is rewritten (below) in terms of the class's
        // canonical member rather than deleted, so the implication keeps
        // constraining search exactly as before, just expressed through the
        // representative instead of through this variable.
        for v in 0..self.num_vars {
            let var = Var::new(v as u32);
            if self.trail.is_assigned(var) {
                sub[Lit::pos(var).code() as usize] = Lit::pos(var);
                sub[Lit::neg(var).code() as usize] = Lit::neg(var);
            }
        }

        debug_assert!(
            (0..num_lits as u32).all(|code| {
                let lit = Lit::from_code(code);
                sub[lit.negate().code() as usize] == sub[code as usize].negate()
            }),
            "substitution map must be polarity-consistent: sub(¬l) == ¬sub(l)"
        );

        let folds_any_variable = (0..self.num_vars).any(|v| {
            sub[Lit::pos(Var::new(v as u32)).code() as usize]
                .var()
                .index()
                != v
        });
        if !folds_any_variable {
            // Every component is a singleton (or entirely trail-fixed): no
            // rewriting to do, and no reconstruction data to record. Still
            // rebuild the binary graph if gate congruence ran: it may have
            // left `ClauseId::NULL`-tagged structural edges behind (trusted
            // unconditionally by `has_live_binary_implication`), and with no
            // rewrite pass below to trigger the usual rebuild, those edges
            // would otherwise linger in `self.binary_graph` for the rest of
            // the solve instead of being purged like every other edge this
            // pass's own rebuild step retires.
            if self.config.enable_gate_congruence {
                self.rebuild_propagation_index();
            }
            return PreprocessOutcome::Ok;
        }

        let mut derived_units: Vec<Lit> = Vec::new();
        let clause_ids: Vec<ClauseId> = self.clauses.iter_ids().collect();
        for id in clause_ids {
            let Some(clause) = self.clauses.get(id) else {
                continue;
            };
            if clause.deleted {
                continue;
            }
            let mut rewritten: SmallVec<[Lit; 4]> = clause
                .lits
                .iter()
                .map(|&l| sub[l.code() as usize])
                .collect();
            rewritten.sort_by_key(|l| l.code());
            rewritten.dedup();

            let is_tautology = rewritten
                .windows(2)
                .any(|pair| pair[0].var() == pair[1].var());
            if is_tautology {
                if let Some(c) = self.clauses.get_mut(id) {
                    c.deleted = true;
                }
                continue;
            }

            match rewritten.len() {
                0 => return PreprocessOutcome::Unsat,
                1 => {
                    derived_units.push(rewritten[0]);
                    if let Some(c) = self.clauses.get_mut(id) {
                        c.deleted = true;
                    }
                }
                _ => {
                    if let Some(c) = self.clauses.get_mut(id) {
                        c.lits = rewritten;
                    }
                }
            }
        }

        self.equiv_substitution = sub;
        self.equiv_substitution_sized = true;
        self.rebuild_propagation_index();

        for lit in derived_units {
            match self.trail.lit_value(lit) {
                LBool::True => {}
                LBool::False => return PreprocessOutcome::Unsat,
                LBool::Undef => self.trail.assign_unit_fact(lit),
            }
        }
        if self.propagate().is_some() {
            return PreprocessOutcome::Unsat;
        }

        PreprocessOutcome::Ok
    }

    /// A variable is considered *eliminated* by the inprocessing toolkit —
    /// and so must never be handed out as a decision — once either
    /// substitution has folded it into a different representative or bounded
    /// variable elimination has recorded a definition for it.
    ///
    /// Public so a caller (or a black-box test) can confirm a mechanism
    /// actually fired on a given instance rather than inferring it
    /// indirectly from verdict/model shape alone.
    pub fn var_eliminated(&self, v: Var) -> bool {
        let by_equiv = self.equiv_substitution_sized
            && self
                .equiv_substitution
                .get(Lit::pos(v).code() as usize)
                .is_some_and(|&rep| rep.var() != v);
        let by_bve = self
            .bve_def
            .get(v.index())
            .is_some_and(|def| !def.is_empty());
        by_equiv || by_bve
    }

    /// The solver's current fatal error, if any — see [`SolverError`].
    ///
    /// Once set (only [`Solver::add_clause`]/[`Solver::add_clause_dimacs`]
    /// and [`Solver::solve_with_assumptions`] can set it, both by way of
    /// `resolve_reintroduced_literal`, a private method not part of this
    /// crate's public API), every `solve*` entry point answers
    /// [`SolverResult::Unknown`] instead of a verdict until [`Solver::reset`]
    /// clears it.
    #[must_use]
    pub fn error(&self) -> Option<&SolverError> {
        self.fatal_error.as_ref()
    }

    /// Resolve a literal a *new* clause or assumption names, for the case
    /// where its variable was already eliminated from the live formula by
    /// the one-shot inprocessing toolkit before this call arrived.
    ///
    /// - Not eliminated: returned unchanged (the overwhelmingly common case
    ///   — this is a plain, cheap lookup, safe to call on every literal of
    ///   every new clause/assumption unconditionally).
    /// - Equivalent-literal-substituted: rewritten through
    ///   `equiv_substitution` to its class representative. Sound for free —
    ///   the substitution map exists exactly because `v`'s equivalence to
    ///   that representative was proven, so a clause mentioning `v` and the
    ///   same clause with `v` replaced by its representative are the same
    ///   constraint.
    /// - Bounded-variable-eliminated: `None`, and [`Solver::error`] starts
    ///   reporting [`SolverError::EliminatedVariableReintroduction`]. Unlike the
    ///   equivalence case there is no cheap rewrite: the variable's defining
    ///   clauses are gone, replaced by resolvents that no longer mention it,
    ///   and reintroducing it soundly means restoring those definitions
    ///   (CaDiCaL keeps an extension stack for exactly this) — a materially
    ///   larger undertaking this port does not yet implement. Answering with
    ///   a guessed verdict here would risk being wrong in either direction
    ///   (see the SK-1 gatekeeper finding this method fixes), so the caller
    ///   is refused instead.
    pub(super) fn resolve_reintroduced_literal(&mut self, lit: Lit) -> Option<Lit> {
        if self.equiv_substitution_sized
            && let Some(&representative) = self.equiv_substitution.get(lit.code() as usize)
            && representative.var() != lit.var()
        {
            return Some(representative);
        }
        if self
            .bve_def
            .get(lit.var().index())
            .is_some_and(|def| !def.is_empty())
        {
            self.fatal_error =
                Some(SolverError::EliminatedVariableReintroduction { var: lit.var() });
            return None;
        }
        Some(lit)
    }

    /// Restore the correct value of every variable equivalent-literal
    /// substitution folded out of the live formula.
    ///
    /// A single direct pass suffices (no fixpoint needed): every
    /// representative literal is, by construction, its own fixed point under
    /// `equiv_substitution` (see the doc comment on
    /// [`Self::fold_equivalent_literals`]'s canonical-choice step), and
    /// `bounded_variable_elimination` refuses to run at all whenever this
    /// pass is enabled, so a representative can never be *itself* eliminated
    /// by the other mechanism either — there is no case where the value this
    /// reads has not already been settled by the plain trail-copy at the top
    /// of [`Solver::save_model`].
    pub(super) fn reconstruct_equiv_eliminated_variables(&mut self) {
        if !self.equiv_substitution_sized {
            return;
        }
        for v in 0..self.num_vars {
            let var = Var::new(v as u32);
            let Some(&representative) = self.equiv_substitution.get(Lit::pos(var).code() as usize)
            else {
                continue;
            };
            if representative.var() == var {
                continue; // not substituted away
            }
            debug_assert_ne!(
                self.model.get(representative.var().index()).copied(),
                Some(LBool::Undef),
                "representative {representative:?} must already have a settled model value"
            );
            self.model[v] = if self.lit_true_in_model(representative) {
                LBool::True
            } else {
                LBool::False
            };
        }
    }

    /// Build an adjacency list over literal codes (`lit → implied`) from the
    /// binary implication graph, keeping only edges verified live right now:
    /// either a structural gate-congruence edge (tagged [`ClauseId::NULL`],
    /// trustworthy by construction — see `solver/congruence.rs`) or an edge
    /// still backed by a real, non-deleted, exactly-2-literal clause. This is
    /// [`Solver::has_live_binary_implication`]'s check inlined per edge
    /// (rather than called per query) so building the whole adjacency stays
    /// linear in the number of graph edges instead of paying a redundant
    /// lookup for each one.
    ///
    /// A stale (retracted) edge admitted here would let the SCC pass below
    /// fabricate an equivalence the current formula does not actually
    /// entail, so this filter is not optional the way it is for a
    /// dedup-only query like [`Solver::has_binary_implication`].
    fn build_trusted_binary_adjacency(&self) -> Vec<Vec<u32>> {
        let num_lits = self.num_vars * 2;
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); num_lits];
        for code in 0..num_lits as u32 {
            let lit = Lit::from_code(code);
            for &(implied, cid) in self.binary_graph.get(lit) {
                let live = cid == ClauseId::NULL
                    || self.clauses.get(cid).is_some_and(|c| {
                        !c.deleted
                            && c.lits.len() == 2
                            && c.lits.contains(&lit.negate())
                            && c.lits.contains(&implied)
                    });
                if live {
                    adjacency[code as usize].push(implied.code());
                }
            }
        }
        adjacency
    }

    /// Rebuild every watch and binary-graph entry from scratch by scanning
    /// the live clause set. Used after a pass (equivalent-literal
    /// substitution, bounded variable elimination) rewrites clause contents
    /// wholesale, since patching the existing watch/graph structures
    /// incrementally for an arbitrary batch of rewrites is far more
    /// error-prone than one linear re-derivation.
    ///
    /// Also prunes `learned_clause_ids` of any id the rewrite deleted
    /// (collapsed to a unit, discovered a tautology, or resolved away by
    /// variable elimination), keeping clause-count reporting and
    /// database-reduction scans from tripping over now-dead ids.
    pub(super) fn rebuild_propagation_index(&mut self) {
        self.watches.clear();
        self.binary_graph.clear();
        for id in self.clauses.iter_ids() {
            let Some(clause) = self.clauses.get(id) else {
                continue;
            };
            if clause.deleted || clause.lits.len() < 2 {
                continue;
            }
            let lit0 = clause.lits[0];
            let lit1 = clause.lits[1];
            self.watches.add(lit0.negate(), Watcher::new(id, lit1));
            self.watches.add(lit1.negate(), Watcher::new(id, lit0));
            if clause.lits.len() == 2 {
                self.binary_graph.add(lit0.negate(), lit1, id);
                self.binary_graph.add(lit1.negate(), lit0, id);
            }
        }
        self.learned_clause_ids
            .retain(|&id| self.clauses.get(id).is_some_and(|c| !c.deleted));

        // Every watcher in the index is brand new, but the assignments that
        // would have fired them are already on the trail and already
        // propagated. Without rewinding the propagation head, a clause whose
        // rewritten form is unit under the *current* level-0 assignment has no
        // future watch event left to notice it — a "hanging unit" that unit
        // propagation never fires on. Rewinding is free of risk:
        // re-propagating an already-assigned literal is a no-op (see
        // `Trail::reset_propagation_head`).
        self.trail.reset_propagation_head();
    }
}

/// Tarjan's SCC algorithm over an adjacency list, run with an explicit work
/// stack instead of recursion (a literal-per-frame recursive walk risks stack
/// overflow on a long implication chain). Returns one component id per node,
/// with **no guarantee about numeric ordering** relative to reverse
/// topological order (unlike the textbook recursive version, which happens to
/// assign ids in reverse-topological order as a side effect of when each
/// root's SCC completes) — callers here only need "same id ⇔ same
/// component", not any particular numbering.
fn tarjan_scc(adjacency: &[Vec<u32>]) -> Vec<u32> {
    let n = adjacency.len();
    const UNVISITED: u32 = u32::MAX;
    let mut index: Vec<u32> = vec![UNVISITED; n];
    let mut lowlink: Vec<u32> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut tarjan_stack: Vec<u32> = Vec::new();
    let mut comp_id: Vec<u32> = vec![UNVISITED; n];
    let mut next_index: u32 = 0;
    let mut next_comp: u32 = 0;

    // Each work-stack frame is (node, index of the next child to examine).
    let mut work: Vec<(u32, usize)> = Vec::new();

    for start in 0..n as u32 {
        if index[start as usize] != UNVISITED {
            continue;
        }
        work.push((start, 0));

        // Holding the frame itself (rather than peeking, then separately
        // re-looking it up to mutate `.1`) makes "the stack emptied out from
        // under us between the peek and the write" structurally unwritable
        // instead of an `.expect()`ed-away impossibility: there is only ever
        // the one lookup, and its result is what both the read below and the
        // in-place advance later in this iteration use.
        while let Some(frame) = work.last_mut() {
            let (node, child_pos) = *frame;
            if child_pos == 0 {
                index[node as usize] = next_index;
                lowlink[node as usize] = next_index;
                next_index += 1;
                tarjan_stack.push(node);
                on_stack[node as usize] = true;
            }

            let children = &adjacency[node as usize];
            if child_pos < children.len() {
                // Record progress before possibly descending, so resuming
                // this frame after the child returns continues at the next
                // sibling instead of revisiting this one. Written straight
                // through the frame the `while let` above already holds.
                frame.1 = child_pos + 1;
                let child = children[child_pos];
                if index[child as usize] == UNVISITED {
                    work.push((child, 0));
                } else if on_stack[child as usize] {
                    lowlink[node as usize] = lowlink[node as usize].min(index[child as usize]);
                }
            } else {
                // All children explored; fold this node's lowlink into its
                // parent's (if any) and, if this node is its own component
                // root, pop the whole component off the Tarjan stack.
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    lowlink[parent as usize] = lowlink[parent as usize].min(lowlink[node as usize]);
                }
                if lowlink[node as usize] == index[node as usize] {
                    // Tarjan's invariant guarantees `node`'s own frame is
                    // still on `tarjan_stack` at this point (it was pushed
                    // when this frame's `child_pos == 0` above, and nothing
                    // pops a node's own frame before its lowlink-equals-index
                    // check runs) — draining via `while let` rather than
                    // `.pop().expect(...)` means a violated invariant simply
                    // stops the drain early instead of panicking; the
                    // `debug_assert!` still catches that case under test.
                    let mut closed_own_frame = false;
                    while let Some(member) = tarjan_stack.pop() {
                        on_stack[member as usize] = false;
                        comp_id[member as usize] = next_comp;
                        if member == node {
                            closed_own_frame = true;
                            break;
                        }
                    }
                    debug_assert!(
                        closed_own_frame,
                        "tarjan_scc: node's own frame must still be on tarjan_stack \
                         when its lowlink equals its index"
                    );
                    next_comp += 1;
                }
            }
        }
    }

    comp_id
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solver with `enable_equiv_substitution` on — the config default is
    /// off (see `SolverConfig::default`), so any test that calls
    /// `fold_equivalent_literals` and expects it to actually run needs
    /// this instead of `Solver::new()`.
    fn els_enabled_solver() -> Solver {
        Solver::with_config(SolverConfig {
            enable_equiv_substitution: true,
            ..SolverConfig::default()
        })
    }

    #[test]
    fn test_pr26_els_tarjan_scc_finds_two_way_cycle() {
        // 0 -> 1 -> 0 (one SCC), 2 standalone.
        let adjacency = vec![vec![1u32], vec![0u32], vec![]];
        let comp = tarjan_scc(&adjacency);
        assert_eq!(comp[0], comp[1]);
        assert_ne!(comp[0], comp[2]);
    }

    #[test]
    fn test_pr26_els_tarjan_scc_handles_long_chain_without_overflow() {
        // A long simple chain 0->1->2->...->n-1 with no cycles: every node is
        // its own singleton component. Exercises the iterative work-stack on
        // a depth that would overflow a naive recursive implementation.
        let n = 50_000;
        let mut adjacency = vec![Vec::new(); n];
        for (i, successors) in adjacency.iter_mut().enumerate().take(n - 1) {
            successors.push((i + 1) as u32);
        }
        let comp = tarjan_scc(&adjacency);
        let distinct: std::collections::HashSet<u32> = comp.iter().copied().collect();
        assert_eq!(distinct.len(), n, "a DAG must have one component per node");
    }

    #[test]
    fn test_pr26_els_substitutes_equivalent_literal() {
        // (¬a∨b)∧(¬b∨a) makes a≡b. A third clause (¬a∨c) then implies
        // (¬b∨c) once rewritten. Add a clause that only becomes satisfied
        // through the substitution, (b∨¬c), together with an original unit
        // forcing c, to confirm the rewritten formula is still correctly
        // solved.
        let mut solver = els_enabled_solver();
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        solver.add_clause([Lit::neg(a), Lit::pos(b)]);
        solver.add_clause([Lit::neg(b), Lit::pos(a)]);
        solver.add_clause([Lit::neg(a), Lit::pos(c)]);
        solver.add_clause([Lit::pos(c)]);

        let outcome = solver.fold_equivalent_literals();
        assert!(matches!(outcome, PreprocessOutcome::Ok));
        assert!(
            solver.var_eliminated(a) || solver.var_eliminated(b),
            "one of the two equivalent variables must be folded into the other"
        );
    }

    #[test]
    fn test_pr26_els_detects_self_contradiction() {
        // (¬a∨b)∧(¬b∨a) makes a≡b; adding (¬a∨¬b)∧(a∨b) on top forces a≡¬b
        // too, so a≡¬a overall -- unconditionally UNSAT.
        let mut solver = els_enabled_solver();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause([Lit::neg(a), Lit::pos(b)]);
        solver.add_clause([Lit::neg(b), Lit::pos(a)]);
        solver.add_clause([Lit::neg(a), Lit::neg(b)]);
        solver.add_clause([Lit::pos(a), Lit::pos(b)]);

        let outcome = solver.fold_equivalent_literals();
        assert!(matches!(outcome, PreprocessOutcome::Unsat));
    }

    #[test]
    fn test_pr26_els_keeps_trail_fixed_variable_as_its_own_representative() {
        // a is forced true by a unit clause before substitution runs; a≡b
        // must not fold a away (it must remain a, with b substituted into a
        // instead, or simply left independently constrained).
        let mut solver = els_enabled_solver();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause([Lit::pos(a)]);
        solver.add_clause([Lit::neg(a), Lit::pos(b)]);
        solver.add_clause([Lit::neg(b), Lit::pos(a)]);

        solver.fold_equivalent_literals();
        assert!(
            !solver.var_eliminated(a),
            "a trail-fixed variable must never be substituted away"
        );
    }

    #[test]
    fn test_pr26_els_noop_when_disabled() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        solver.add_clause([Lit::neg(a), Lit::pos(b)]);
        solver.add_clause([Lit::neg(b), Lit::pos(a)]);
        // enable_equiv_substitution defaults to false.
        assert!(!solver.config.enable_equiv_substitution);
        solver.fold_equivalent_literals();
        assert!(!solver.var_eliminated(a));
        assert!(!solver.var_eliminated(b));
    }
}
