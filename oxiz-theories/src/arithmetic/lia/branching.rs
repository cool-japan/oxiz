//\! Branch-and-bound methods for LIA solver

use super::super::simplex::{LinExpr, VarId};
use super::types::{BranchNode, LiaSolver};
use crate::config::BranchingHeuristic;
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, Zero};
use oxiz_core::error::{OxizError, Result};

/// Maximum number of root-node cut rounds attempted by [`LiaSolver::check`].
///
/// Each round derives at most one tableau-based Gomory Mixed-Integer cut from
/// the current LP optimum and re-solves.  The limit keeps the root loop from
/// degenerating into an unbounded cut-generation spiral on instances where the
/// GMI closure converges slowly; branch-and-bound remains the complete
/// procedure behind it, so a truncated cut loop only forgoes strengthening.
const ROOT_CUT_ROUND_LIMIT: usize = 8;

/// Reason code attached to root-node cuts asserted into the simplex.
///
/// Root cuts are globally valid consequences of the asserted constraint set
/// rather than of any single user assertion, so they carry their own marker
/// instead of borrowing an input constraint's reason.
const ROOT_CUT_REASON: u32 = u32::MAX;

/// Result of the root-node cut loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootCutOutcome {
    /// The LP is still feasible after the cut rounds; continue to B&B.
    Continue,
    /// A valid cut made the LP infeasible.  Because every emitted cut is a
    /// valid inequality of the integer hull (see `cuts.rs`), this is a proof
    /// that no integer-feasible point exists.
    IntegerInfeasible,
}

impl LiaSolver {
    /// Root-node cutting-plane loop.
    ///
    /// Repeatedly derives a tableau-based Gomory Mixed-Integer cut from the
    /// current LP optimum (via [`LiaSolver::generate_conflict_driven_cut`]),
    /// asserts it, and re-solves.  Runs only when
    /// [`LiaConfig::enable_gomory_cuts`](crate::config::LiaConfig::enable_gomory_cuts)
    /// is set, and is bounded by `min(ROOT_CUT_ROUND_LIMIT, config.max_cuts)`
    /// rounds.
    ///
    /// The loop stops early when no cut can be derived (the LP point is
    /// integral, or no sound cut exists for the selected variable) and when the
    /// simplex exhausts its pivot budget — in the latter case the cuts already
    /// asserted stay, which is sound because each is a valid inequality.
    ///
    /// Callers must invoke this only at the root, before any branch-and-bound
    /// `push`, so the slack rows the cuts introduce are scoped by the caller's
    /// own scope rather than by a branch that will be popped.
    fn root_cut_loop(&mut self) -> RootCutOutcome {
        if !self.config.enable_gomory_cuts {
            return RootCutOutcome::Continue;
        }

        let budget = ROOT_CUT_ROUND_LIMIT.min(self.config.max_cuts);
        for _ in 0..budget {
            let Some(cut) = self.generate_conflict_driven_cut() else {
                break; // nothing fractional left, or no sound cut derivable
            };

            // A GMI cut derived from the row of a fractional basic variable
            // evaluates to its positive right-hand side at the current LP point
            // (every non-basic slack `y_j` is zero there), so it must separate.
            // A non-separating "cut" would mean the derivation was corrupted.
            debug_assert!(
                {
                    let mut at_current = cut.constant;
                    for (v, c) in &cut.terms {
                        at_current += *c * self.simplex.value(*v);
                    }
                    at_current > Rational64::zero()
                },
                "root cut does not separate the current LP point"
            );

            self.simplex.add_le(cut, ROOT_CUT_REASON);
            self.cuts_generated += 1;

            match self.simplex.check() {
                // The cut removes no integer-feasible point, so an LP that is
                // infeasible under it has no integer solution either.
                Err(_) => return RootCutOutcome::IntegerInfeasible,
                Ok(()) if self.simplex.resource_limit_reached() => break,
                Ok(()) => {}
            }
        }

        RootCutOutcome::Continue
    }

    /// Check satisfiability with branch-and-bound
    pub fn check(&mut self) -> Result<bool> {
        // First check if the LP relaxation is feasible
        match self.simplex.check() {
            Ok(()) => {
                // LP is feasible: run root-node preprocessing before B&B.

                // Step 0: Root cutting planes.  Tableau-derived GMI cuts are
                // asserted here, before any branch-and-bound `push`, so their
                // slack rows are scoped by the caller's scope and a cut can
                // never be attributed to (and popped with) a branch.
                if self.root_cut_loop() == RootCutOutcome::IntegerInfeasible {
                    return Ok(false);
                }

                // Step 1: Probe variables — root bound tightening.
                // Tentatively fixes each integer variable to its bounds and
                // propagates implied tightenings. Failure is non-fatal; we
                // proceed even if probing returns an error.
                let _ = self.probe_variables(20);

                // Step 2: Feasibility pump — opportunistic incumbent search.
                // Try to find an integer-feasible solution cheaply before
                // spending time in branch-and-bound. If the pump succeeds we
                // pass the incumbent implicitly: branch_and_bound will verify
                // integrality again and can prune accordingly.
                if let Ok(Some(_int_sol)) = self.feasibility_pump(10) {
                    // Pump found an integer-feasible solution.  We don't
                    // short-circuit B&B here because we still need to check
                    // that the full constraint set (including any constraints
                    // added after the LP solve) is satisfied and to populate
                    // the simplex model with the verified incumbent.  B&B will
                    // detect the integer assignment quickly on its first pass.
                }

                // Step 3: Branch-and-bound — full integer feasibility check.
                self.branch_and_bound(0)
            }
            Err(_reasons) => {
                // LP is infeasible, so LIA is also infeasible
                Ok(false)
            }
        }
    }

    /// Branch-and-bound algorithm.
    ///
    /// Each branch is explored under a matched `simplex.push()` / `simplex.pop()`
    /// so that trying `x >= ceil(v)` never destroys the constraints needed for the
    /// sibling branch `x <= floor(v)`.  (The previous implementation called
    /// `simplex.reset()` between the two branches, which erased *every* constraint
    /// and made the down-branch trivially satisfiable — a soundness bug that made
    /// e.g. `2x = 1` over the integers report SAT.)
    ///
    /// On a satisfying leaf the winning branch's constraints are intentionally
    /// left in place (its `push` is not popped) so that the integral model stays
    /// queryable via `value()`; failed branches are always fully popped.
    ///
    /// Cutting planes are generated at the *root* only, by
    /// [`LiaSolver::check`] before the first branch is pushed: the generators in
    /// `cuts.rs` derive tableau-based Gomory Mixed-Integer / Chvátal-Gomory
    /// inequalities and carry a validity contract (documented there) that a cut
    /// never removes an integer-feasible point, returning [`None`] rather than
    /// fabricating an inequality when no sound derivation exists.
    ///
    /// Branch-and-bound itself remains cut-free by design.  A branch-local cut
    /// is derived from a tableau that already includes the branch bound, so it
    /// is valid only inside that branch and must be retracted with it; wiring
    /// that scoping correctly is a recorded follow-up, not a gap in soundness —
    /// branch-and-bound alone is a sound and (for bounded problems) complete
    /// integrality procedure.
    fn branch_and_bound(&mut self, depth: usize) -> Result<bool> {
        if depth > self.max_depth {
            return Err(OxizError::Internal(
                "branch-and-bound depth limit exceeded".to_string(),
            ));
        }

        // Check if the current solution is integer.
        let (var, value) = match self.find_fractional_var() {
            Some(vv) => vv,
            None => return Ok(true), // all variables integer-valued ⇒ SAT
        };

        let ceil_value = value.ceil().to_integer();
        let floor_value = value.floor().to_integer();

        self.branch_stack.push(BranchNode {
            var,
            branch_up: true,
            fractional_value: value,
        });

        // Branch up: x >= ceil(value).
        self.simplex.push();
        let mut up_expr = LinExpr::new();
        up_expr.add_term(var, Rational64::one());
        up_expr.add_constant(-Rational64::from_integer(ceil_value));
        self.simplex.add_ge(up_expr, 0);
        match self.simplex.check() {
            Ok(()) if !self.simplex.resource_limit_reached() => {
                if self.branch_and_bound(depth + 1)? {
                    self.update_pseudo_cost(var, true, (depth + 1) as f64);
                    self.branch_stack.pop();
                    return Ok(true); // keep constraints so the model persists
                }
            }
            Ok(()) => {
                // Simplex hit its pivot budget — undecidable, report honestly.
                self.simplex.pop();
                self.branch_stack.pop();
                return Err(OxizError::Internal(
                    "branch-and-bound: simplex pivot limit reached".to_string(),
                ));
            }
            Err(_) => {} // up-branch infeasible
        }
        self.simplex.pop();
        self.update_pseudo_cost(var, true, (depth + 1) as f64);

        // Branch down: x <= floor(value).
        self.simplex.push();
        let mut down_expr = LinExpr::new();
        down_expr.add_term(var, Rational64::one());
        down_expr.add_constant(-Rational64::from_integer(floor_value));
        self.simplex.add_le(down_expr, 0);
        match self.simplex.check() {
            Ok(()) if !self.simplex.resource_limit_reached() => {
                if self.branch_and_bound(depth + 1)? {
                    self.update_pseudo_cost(var, false, (depth + 1) as f64);
                    self.branch_stack.pop();
                    return Ok(true); // keep constraints so the model persists
                }
            }
            Ok(()) => {
                self.simplex.pop();
                self.branch_stack.pop();
                return Err(OxizError::Internal(
                    "branch-and-bound: simplex pivot limit reached".to_string(),
                ));
            }
            Err(_) => {} // down-branch infeasible
        }
        self.simplex.pop();
        self.update_pseudo_cost(var, false, (depth + 1) as f64);

        // Both branches are proven infeasible ⇒ integer-infeasible.
        self.branch_stack.pop();
        Ok(false)
    }

    /// Scope-balanced integer feasibility check.
    ///
    /// Runs the same pipeline as [`LiaSolver::check`] — root cuts, probing,
    /// feasibility pump, branch-and-bound — but leaves the simplex at exactly
    /// the scope depth it was entered at.  [`LiaSolver::check`] deliberately
    /// does *not*: it retains the winning branch's `push` so that the integral
    /// assignment stays queryable through [`LiaSolver::value`].  That is
    /// convenient for a one-shot caller and poisonous for one that keeps
    /// asserting lemmas afterwards, because every later assertion would land
    /// inside a branch scope and be silently retracted by the next `pop`.
    ///
    /// Instead of leaving the branch in place, this method **snapshots** the
    /// integral assignment at the satisfying leaf and pops every branch it
    /// pushed on the way out.
    ///
    /// Returns:
    /// * `Ok(Some(model))` — integer-feasible; `model` maps every integer
    ///   variable to its value at the satisfying leaf.
    /// * `Ok(None)` — proven integer-infeasible (LP infeasible, refuted by a
    ///   valid root cut, or every branch exhausted).
    /// * `Err(_)` — a resource limit (simplex pivot budget or branch depth)
    ///   stopped the search before a verdict was reached.  A resource limit is
    ///   never reported as a verdict.
    ///
    /// Root cuts asserted by this method are *not* retracted: they are valid
    /// inequalities of the constraint set present at the entry scope, so they
    /// belong to that scope and are undone by the caller's own `pop`.
    pub fn check_balanced(&mut self) -> Result<Option<FxHashMap<VarId, Rational64>>> {
        let entry_depth = self.simplex.scope_depth();
        let outcome = self.check_balanced_inner();
        debug_assert_eq!(
            self.simplex.scope_depth(),
            entry_depth,
            "check_balanced must leave the simplex at its entry scope depth"
        );
        outcome
    }

    /// Body of [`LiaSolver::check_balanced`]; see there for the contract.
    fn check_balanced_inner(&mut self) -> Result<Option<FxHashMap<VarId, Rational64>>> {
        match self.simplex.check() {
            Ok(()) => {
                if self.root_cut_loop() == RootCutOutcome::IntegerInfeasible {
                    return Ok(None);
                }

                // Root bound tightening; failure is non-fatal, and probing is
                // itself scope balanced (matched push/pop per probe).
                let _ = self.probe_variables(20);

                // Opportunistic incumbent search.  As in `check`, the pump's
                // answer is not trusted as a verdict: branch-and-bound verifies
                // integrality against the full constraint set.
                let _ = self.feasibility_pump(10);

                self.branch_and_bound_balanced(0)
            }
            // LP infeasible ⇒ integer-infeasible.
            Err(_reasons) => Ok(None),
        }
    }

    /// Snapshot the current assignment of every integer variable.
    fn integral_model_snapshot(&self) -> FxHashMap<VarId, Rational64> {
        self.int_vars
            .keys()
            .map(|&var| (var, self.simplex.value(var)))
            .collect()
    }

    /// Scope-balanced branch-and-bound.
    ///
    /// Mirrors [`LiaSolver::branch_and_bound`], with two differences: the
    /// satisfying leaf's assignment is captured into a model before unwinding,
    /// and **every** `push` is matched by a `pop` on every exit path — the
    /// winning branch, the infeasible branch, and both resource-limit paths.
    fn branch_and_bound_balanced(
        &mut self,
        depth: usize,
    ) -> Result<Option<FxHashMap<VarId, Rational64>>> {
        if depth > self.max_depth {
            return Err(OxizError::Internal(
                "branch-and-bound depth limit exceeded".to_string(),
            ));
        }

        // Check if the current solution is integer.
        let (var, value) = match self.find_fractional_var() {
            Some(vv) => vv,
            None => return Ok(Some(self.integral_model_snapshot())),
        };

        let ceil_value = value.ceil().to_integer();
        let floor_value = value.floor().to_integer();

        self.branch_stack.push(BranchNode {
            var,
            branch_up: true,
            fractional_value: value,
        });

        // Explore `x >= ceil(value)` then `x <= floor(value)`.
        for branch_up in [true, false] {
            self.simplex.push();
            let mut expr = LinExpr::new();
            expr.add_term(var, Rational64::one());
            if branch_up {
                expr.add_constant(-Rational64::from_integer(ceil_value));
                self.simplex.add_ge(expr, 0);
            } else {
                expr.add_constant(-Rational64::from_integer(floor_value));
                self.simplex.add_le(expr, 0);
            }

            let outcome = match self.simplex.check() {
                Ok(()) if !self.simplex.resource_limit_reached() => {
                    self.branch_and_bound_balanced(depth + 1)
                }
                Ok(()) => Err(OxizError::Internal(
                    "branch-and-bound: simplex pivot limit reached".to_string(),
                )),
                Err(_) => Ok(None), // this branch is infeasible
            };

            // Unconditional pop: the model was already captured at the leaf, so
            // nothing downstream depends on the branch scope staying alive.
            self.simplex.pop();
            self.update_pseudo_cost(var, branch_up, (depth + 1) as f64);

            match outcome {
                Ok(Some(model)) => {
                    self.branch_stack.pop();
                    return Ok(Some(model));
                }
                Ok(None) => {}
                Err(err) => {
                    self.branch_stack.pop();
                    return Err(err);
                }
            }
        }

        // Both branches are proven infeasible ⇒ integer-infeasible.
        self.branch_stack.pop();
        Ok(None)
    }

    /// Find a variable with fractional value using the configured branching heuristic
    pub(super) fn find_fractional_var(&self) -> Option<(VarId, Rational64)> {
        match self.config.branching_heuristic {
            BranchingHeuristic::FirstFractional => self.find_first_fractional(),
            BranchingHeuristic::MostFractional => self.find_most_fractional(),
            BranchingHeuristic::PseudoCost => self.find_pseudo_cost_var(),
            BranchingHeuristic::StrongBranching => self.find_strong_branching_var(),
        }
    }

    /// Find the first fractional variable (fastest, but may not be optimal)
    pub fn find_first_fractional(&self) -> Option<(VarId, Rational64)> {
        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                return Some((var, value));
            }
        }
        None
    }

    /// Find the most fractional variable (closest to 0.5)
    /// This heuristic prefers variables that are "most uncertain"
    pub fn find_most_fractional(&self) -> Option<(VarId, Rational64)> {
        let mut best_var = None;
        let mut best_fractionality = 0.0;

        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                let frac = value - value.floor();
                let frac_f64 = (*frac.numer() as f64) / (*frac.denom() as f64);
                // Fractionality is how close to 0.5 the fractional part is
                let fractionality = 0.5 - (frac_f64 - 0.5).abs();

                if fractionality > best_fractionality {
                    best_fractionality = fractionality;
                    best_var = Some((var, value));
                }
            }
        }
        best_var
    }

    /// Find variable using pseudo-cost heuristic
    /// Pseudo-cost estimates the expected objective change when branching on a variable
    pub fn find_pseudo_cost_var(&self) -> Option<(VarId, Rational64)> {
        let mut best_var = None;
        let mut best_score = -1.0;

        for &var in self.int_vars.keys() {
            let value = self.simplex.value(var);
            if !value.is_integer() {
                let frac = value - value.floor();
                let frac_f64 = (*frac.numer() as f64) / (*frac.denom() as f64);

                // Get pseudo-costs (default to 1.0 if no history)
                let (down_cost, up_cost, _count) = self
                    .pseudo_costs
                    .get(&var)
                    .copied()
                    .unwrap_or((1.0, 1.0, 0));

                // Score is the product of estimated down and up costs
                // This balances between variables that are expensive to branch on
                let down_score = down_cost * frac_f64;
                let up_score = up_cost * (1.0 - frac_f64);
                let score = down_score * up_score;

                if score > best_score {
                    best_score = score;
                    best_var = Some((var, value));
                }
            }
        }
        best_var
    }

    /// Update pseudo-cost estimates after branching
    ///
    /// Pseudo-costs estimate how much "work" is required when branching on a variable.
    /// We track the number of nodes explored in each branch direction.
    fn update_pseudo_cost(&mut self, var: VarId, branch_up: bool, cost: f64) {
        let entry = self.pseudo_costs.entry(var).or_insert((1.0, 1.0, 0));

        if branch_up {
            // Update up-branch cost using exponential moving average
            let alpha = 0.1; // Learning rate
            entry.1 = (1.0 - alpha) * entry.1 + alpha * cost;
        } else {
            // Update down-branch cost
            let alpha = 0.1;
            entry.0 = (1.0 - alpha) * entry.0 + alpha * cost;
        }

        entry.2 += 1; // Increment observation count
    }

    /// Strong branching: evaluate both branch directions before selecting
    ///
    /// For each fractional variable candidate:
    /// 1. Tentatively add x <= floor(value) and solve for a few iterations
    /// 2. Tentatively add x >= ceil(value) and solve for a few iterations
    /// 3. Score based on the dual bound improvement in both directions
    /// 4. Select the variable with best score
    ///
    /// This is more expensive than other heuristics but often reduces the
    /// branch-and-bound tree size by 20-50%, leading to faster overall solving.
    ///
    /// Reference: Achterberg (2007) "Constraint Integer Programming"
    pub fn find_strong_branching_var(&self) -> Option<(VarId, Rational64)> {
        // Collect all fractional variables with their values
        let mut candidates: Vec<(VarId, Rational64)> = self
            .int_vars
            .keys()
            .filter_map(|&var| {
                let value = self.simplex.value(var);
                if !value.is_integer() {
                    Some((var, value))
                } else {
                    None
                }
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Limit number of candidates if configured
        let max_candidates = self.config.strong_branching_candidates;
        if max_candidates > 0 && candidates.len() > max_candidates {
            // Use most fractional as tiebreaker for limiting candidates
            candidates.sort_by(|(_, val_a), (_, val_b)| {
                let frac_a = val_a - val_a.floor();
                let frac_b = val_b - val_b.floor();
                let frac_a_f64 = (*frac_a.numer() as f64) / (*frac_a.denom() as f64);
                let frac_b_f64 = (*frac_b.numer() as f64) / (*frac_b.denom() as f64);
                let dist_a = (frac_a_f64 - 0.5).abs();
                let dist_b = (frac_b_f64 - 0.5).abs();
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            candidates.truncate(max_candidates);
        }

        // Evaluate each candidate
        let mut best_var = None;
        let mut best_score = -1.0;

        for &(var, value) in &candidates {
            // Note: We can't actually modify self.simplex here since this is &self
            // In a full implementation, we would:
            // 1. Clone the simplex state (or use push/pop)
            // 2. Add constraint x <= floor(value), solve limited iterations
            // 3. Record bound improvement (down_gain)
            // 4. Restore state
            // 5. Add constraint x >= ceil(value), solve limited iterations
            // 6. Record bound improvement (up_gain)
            // 7. Score = min(down_gain, up_gain) * max(down_gain, up_gain)
            //
            // For now, we fall back to a simplified heuristic that combines
            // pseudo-costs with fractionality (hybrid approach)

            let frac = value - value.floor();
            let frac_f64 = (*frac.numer() as f64) / (*frac.denom() as f64);

            // Get pseudo-costs (default to 1.0 if no history)
            let (down_cost, up_cost, count) = self
                .pseudo_costs
                .get(&var)
                .copied()
                .unwrap_or((1.0, 1.0, 0));

            // If we have pseudo-cost history, use it; otherwise use fractionality
            let score = if count > 0 {
                // Weighted combination of pseudo-cost and fractionality
                let pc_score = down_cost * frac_f64 * up_cost * (1.0 - frac_f64);
                let frac_score = 0.5 - (frac_f64 - 0.5).abs();
                0.8 * pc_score + 0.2 * frac_score
            } else {
                // No history: use fractionality
                0.5 - (frac_f64 - 0.5).abs()
            };

            if score > best_score {
                best_score = score;
                best_var = Some((var, value));
            }
        }

        best_var
    }
}
