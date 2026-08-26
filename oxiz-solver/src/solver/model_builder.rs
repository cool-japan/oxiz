//! Model and unsat core building

#[allow(unused_imports)]
use crate::prelude::*;
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::{SortId, SortKind};
use smallvec::SmallVec;

use super::Solver;
use super::dt_axioms::{DeclInfo, resolve_decl, scan_datatype_terms};
use super::types::Constraint;
use super::types::{Model, UnsatCore};

/// The datatype equalities the search decided: the pairs it made *false*
/// (ordered by term id), and the adjacency of the ones it made *true*.
type DecidedEqualities = (Vec<(TermId, TermId)>, FxHashMap<TermId, Vec<TermId>>);

impl Solver {
    pub(super) fn build_model(&mut self, manager: &mut TermManager) {
        let mut model = Model::new();
        let sat_model = self.sat.model();

        // Get boolean values from SAT model
        for (&term, &var) in &self.term_to_var {
            let val = sat_model.get(var.index()).copied();
            if let Some(v) = val {
                let bool_val = if v.is_true() {
                    manager.mk_true()
                } else if v.is_false() {
                    manager.mk_false()
                } else {
                    continue;
                };
                model.set(term, bool_val);
            }
        }

        // Extract values from equality constraints (e.g., x = 5)
        // This handles cases where a variable is equated to a constant
        for (&var, constraint) in &self.var_to_constraint {
            // Check if the equality is assigned true in the SAT model
            let is_true = sat_model
                .get(var.index())
                .copied()
                .is_some_and(|v| v.is_true());

            if !is_true {
                continue;
            }

            if let Constraint::Eq(lhs, rhs) = constraint {
                // Check if one side is a tracked variable and the other is a constant.
                // Also handle Apply terms (uninterpreted function applications) that are
                // not in arith_terms due to the restriction on Apply terms with arith args.
                let lhs_is_apply = manager
                    .get(*lhs)
                    .is_some_and(|t| matches!(t.kind, TermKind::Apply { .. }));
                let rhs_is_apply = manager
                    .get(*rhs)
                    .is_some_and(|t| matches!(t.kind, TermKind::Apply { .. }));
                let (var_term, const_term) = if self.arith_terms.contains(lhs)
                    || self.bv_terms.contains(lhs)
                    || lhs_is_apply
                {
                    (*lhs, *rhs)
                } else if self.arith_terms.contains(rhs)
                    || self.bv_terms.contains(rhs)
                    || rhs_is_apply
                {
                    (*rhs, *lhs)
                } else {
                    continue;
                };

                // Check if const_term is actually a constant
                let Some(const_term_data) = manager.get(const_term) else {
                    continue;
                };

                match &const_term_data.kind {
                    TermKind::IntConst(n) => {
                        if let Some(val) = n.to_i64() {
                            let value_term = manager.mk_int(val);
                            model.set(var_term, value_term);
                        }
                    }
                    TermKind::RealConst(r) => {
                        let value_term = manager.mk_real(*r);
                        model.set(var_term, value_term);
                    }
                    TermKind::BitVecConst { value, width } => {
                        // The whole literal, not just the part that fits a
                        // `u64`: a 128-bit constant used to be dropped here and
                        // the variable then took a default of `0`.
                        let (value, width) = (value.clone(), *width);
                        let value_term = manager.mk_bitvec(value, width);
                        model.set(var_term, value_term);
                    }
                    _ => {}
                }
            }
        }

        // Get arithmetic values from theory solver
        // Iterate over tracked arithmetic terms
        for &term in &self.arith_terms {
            // Don't overwrite if already set (e.g., from equality extraction above)
            if model.get(term).is_some() {
                continue;
            }

            if let Some(value) = self.arith.value(term) {
                // Determine whether the term has Int or Real sort, and create the
                // matching constant kind.  Using the term sort (rather than the
                // denominator of the rational value) is essential: a Real-sorted
                // term whose arith model value happens to be an integer ratio (e.g.
                // 2/1) must be represented as RealConst(2), not IntConst(2).  If
                // stored as IntConst, mixed-type comparisons like (f(c) <= 1.0)
                // become symbolic because eval_le requires both sides to be the
                // same constant kind, preventing counterexample detection.
                let is_int_sort = manager
                    .get(term)
                    .map(|t| t.sort == manager.sorts.int_sort)
                    .unwrap_or(true);
                let value_term = if is_int_sort {
                    // Integer-sorted term: convert to BigInt
                    manager.mk_int(*value.numer())
                } else {
                    // Real-sorted term: always use RealConst regardless of denominator
                    manager.mk_real(value)
                };
                model.set(term, value_term);
            } else {
                // If no value from ArithSolver (e.g., unconstrained variable), use default
                // Get the sort to determine if it's Int or Real
                let is_int = manager
                    .get(term)
                    .map(|t| t.sort == manager.sorts.int_sort)
                    .unwrap_or(true);

                let value_term = if is_int {
                    manager.mk_int(0i64)
                } else {
                    manager.mk_real(num_rational::Rational64::from_integer(0))
                };
                model.set(term, value_term);
            }
        }

        // Get bitvector values.  Which theory owns a BV variable's value depends
        // on how it was actually constrained (see `bv_solver_is_authoritative`):
        //
        //   * BV structure (arithmetic, bitwise, shifts, concat/extract), BV
        //     (dis)equalities and BV *comparisons* are all genuinely bit-blasted
        //     — with constant operands pinned to their concrete bits — so
        //     `BvSolver::get_value` is a real witness.
        //   * Unsigned BV comparisons are *additionally* relaxed into the linear
        //     `ArithSolver` as unbounded integers.  That relaxation carries no
        //     `0 <= x < 2^width` domain bound, so its value can fall outside the
        //     bit-vector's range; it is only consulted when the BV solver has
        //     nothing, and is wrapped into range below either way.
        //
        // Establishing a single owning theory per problem keeps the extracted
        // model self-consistent instead of reading a stale value from the wrong
        // solver.
        let bv_authoritative = self.bv_solver_is_authoritative(manager);
        for &term in &self.bv_terms {
            // Don't overwrite if already set (shouldn't happen, but be safe)
            if model.get(term).is_some() {
                continue;
            }

            // Get the bitvector width from the term's sort
            let width = manager
                .get(term)
                .and_then(|t| manager.sorts.get(t.sort))
                .and_then(|s| s.bitvec_width())
                .unwrap_or(64);

            // `get_value_big` rather than `get_value`: the latter answers `None`
            // for every bit-vector wider than 64 bits, so a 96-bit witness fell
            // through to the arithmetic relaxation and finally to `0` — the
            // model printed `#x000…0` for a variable the search had pinned to a
            // huge constant.
            let bv_value = self.bv.get_value_big(term);
            let arith_value = self.arith.value(term);

            let raw_value = if bv_authoritative {
                // BV theory owns the model: prefer its bit-blasted witness, then
                // fall back to any bounded-integer value, then a default of 0.
                if let Some(bv_value) = bv_value {
                    num_bigint::BigInt::from(bv_value)
                } else if let Some(arith_value) = arith_value {
                    arith_value.to_integer().into()
                } else {
                    num_bigint::BigInt::ZERO
                }
            } else {
                // No BV atom was bit-blasted (e.g. a BV variable that only ever
                // appears under an uninterpreted function): the ArithSolver's
                // relaxation is all we have.
                if let Some(arith_value) = arith_value {
                    arith_value.to_integer().into()
                } else if let Some(bv_value) = bv_value {
                    num_bigint::BigInt::from(bv_value)
                } else {
                    num_bigint::BigInt::ZERO
                }
            };
            // A bit-vector model value must be a well-formed literal of the
            // declared width.  The arithmetic relaxation has no domain bound, so
            // it can hand back a negative or oversized integer (`(bvult x #x00)`
            // used to yield `x = -1`, printed as the malformed `#x-1`).  Wrap
            // into `[0, 2^width)` — the two's-complement reading SMT-LIB
            // prescribes — before interning the constant.
            let value_term =
                manager.mk_bitvec(oxiz_core::ast::bv_wrap_unsigned(&raw_value, width), width);
            model.set(term, value_term);
        }

        // String values.  No string theory participates in the CDCL(T) loop, so
        // `String`-sorted variables are still unassigned at this point; recover
        // them from the ground string decision procedure, which hands back only
        // assignments it has verified against every assertion.
        self.extract_string_model(&mut model, manager);

        // Datatype values.  Must run last: the reconstruction reads the tester
        // literals and the field values the passes above have already recorded.
        self.extract_datatype_model(&mut model, manager);

        // Numeric-UF-argument purification aliases.  Must run after every
        // pass above: `purify_numeric_uf_args` rewrote e.g. `f(3)` to
        // `f(v)` (plus `v = 3`) before encoding, so only `f(v)`'s equivalence
        // class ever received a value from the loops above -- `f(3)` itself
        // (still what `self.assertions` and any `get-value` query name) has
        // none. Propagate the purified twin's value back to the original
        // term wherever one was found, so a genuinely satisfiable model
        // resolves the term the user actually wrote.
        for (&original, &purified) in &self.numeric_purify_aliases {
            if model.get(original).is_some() {
                continue;
            }
            if let Some(value) = model.get(purified) {
                model.set(original, value);
            }
        }

        // Rounding-mode values.  Must run after the Boolean pass above, which
        // is what put the mode-equality atoms' truth values into scope.
        self.extract_rounding_mode_model(&mut model, manager);

        self.model = Some(model);
    }

    /// Read the value of every free `RoundingMode` constant out of the SAT
    /// assignment.
    ///
    /// This needs no search and no heuristic, because the solver has already
    /// decided it. `Context::declare_const` asserts a closure axiom
    /// `(or (= m RNE) (= m RNA) (= m RTP) (= m RTN) (= m RTZ))` for every
    /// declared rounding-mode constant, so *any* satisfying assignment sets at
    /// least one of those five equality atoms true — and the atom that is true
    /// names the mode the solve chose.  Reading it back is therefore exact and
    /// jointly consistent across several rounding-mode constants for free:
    /// the assignment came from one model, not from five independent guesses.
    ///
    /// Getting this right matters concretely. Without it a constant the model
    /// pinned would fall through to `Context::default_value`, which reports the
    /// sort's default `roundNearestTiesToEven` — so
    /// `(assert (not (= m RNE)))` would answer `sat` and then print a "model"
    /// that falsifies its own assertion.
    fn extract_rounding_mode_model(&self, model: &mut Model, manager: &mut TermManager) {
        // Nothing to extract, and — the reason this guard is first rather than
        // a mere optimization — nothing to *build*: `mk_rounding_mode` below
        // would otherwise set the manager's `rounding_mode_used` flag on every
        // solve, and `Context::check_sat` reads that flag to decide whether the
        // five-modes distinctness axiom is needed. A script with no rounding
        // mode in it would start asserting one extra axiom per `check-sat`.
        if !manager.rounding_mode_used() {
            return;
        }
        let rm_sort = manager.sorts.rounding_mode_sort;
        let mode_terms: Vec<(oxiz_core::ast::RoundingMode, TermId)> =
            oxiz_core::ast::RoundingMode::ALL
                .into_iter()
                .map(|rm| (rm, manager.mk_rounding_mode(rm)))
                .collect();
        let mode_ids: FxHashSet<TermId> = mode_terms.iter().map(|&(_, t)| t).collect();

        // Every free rounding-mode variable reachable from the assertions.
        let mut targets: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = self.assertions.clone();
        while let Some(term) = stack.pop() {
            if !visited.insert(term) {
                continue;
            }
            let Some(td) = manager.get(term) else {
                continue;
            };
            if td.sort == rm_sort
                && matches!(td.kind, TermKind::Var(_))
                && !mode_ids.contains(&term)
                && seen.insert(term)
            {
                targets.push(term);
            }
            let kind = td.kind.clone();
            super::term_walk::collect_structural_children(&kind, &mut stack);
        }
        if targets.is_empty() {
            return;
        }

        let sat_model = self.sat.model();
        for target in targets {
            if model.get(target).is_some() {
                continue;
            }
            for &(_, mode_term) in &mode_terms {
                // Built exactly as the closure axiom built it, so hash-consing
                // hands back that axiom's own atom rather than a fresh twin.
                let atom = manager.mk_eq(target, mode_term);
                let is_true = self
                    .term_to_var
                    .get(&atom)
                    .and_then(|var| sat_model.get(var.index()).copied())
                    .is_some_and(|v| v.is_true());
                if is_true {
                    model.set(target, mode_term);
                    break;
                }
            }
        }
    }

    /// Reconstruct a concrete constructor value for every datatype-sorted term
    /// the assertions mention.
    ///
    /// Without this the model simply has no entry for a datatype constant and
    /// `(get-model)` completes it from the sort default — the datatype's first
    /// nullary constructor — so `((_ is cons) l) ∧ (= (head l) 7)` was answered
    /// `sat` (correctly) with the witness `l = nil`, which satisfies neither
    /// conjunct.  The verdict was sound; the witness was not.
    ///
    /// The value is read out of the assignment the search actually committed:
    /// the constructor from whichever tester literal the SAT core set true (the
    /// exhaustiveness / exclusivity axioms guarantee exactly one), and each
    /// field from the accessor term's own model value.
    ///
    /// # Soundness with respect to the model-verification gate
    ///
    /// [`Solver::model_refutes_assertions`] runs *after* `build_model` and
    /// downgrades a `Sat` whose model provably falsifies an assertion.  Every
    /// entry added here is datatype-sorted with a [`TermKind::DtConstructor`]
    /// value, and `Solver::parse_value_term` — the only way a model entry
    /// reaches that gate — recognises exactly `True`/`False`/`IntConst`/
    /// `RealConst`.  A datatype entry therefore always evaluates to `None`
    /// ("inconclusive"), never to `Some(Bool(false))`, so no reconstruction —
    /// complete, partial, or absent — can turn a correct `sat` into `unknown`.
    fn extract_datatype_model(&self, model: &mut Model, manager: &mut TermManager) {
        // O(1) guard: datatype lemmas exist iff the assertion set mentions a
        // datatype term, so a datatype-free problem never pays for the walk.
        if self.dt_axiom_instances.is_empty() {
            return;
        }
        let Some(scan) = scan_datatype_terms(&self.assertions, manager) else {
            return;
        };

        // Resolve the declaration of every datatype sort reachable through the
        // scanned sorts' *fields* as well, so the recursion below never needs
        // the manager immutably while it is interning value terms.
        let mut decls = scan.decls;
        let mut pending: Vec<SortId> = decls
            .values()
            .flat_map(|decl| decl.constructors.iter())
            .flat_map(|constructor| constructor.fields.iter())
            .filter(|field| field.is_datatype)
            .map(|field| field.sort)
            .collect();
        while let Some(sort) = pending.pop() {
            if decls.contains_key(&sort) {
                continue;
            }
            let Some(decl) = resolve_decl(sort, manager) else {
                continue;
            };
            pending.extend(
                decl.constructors
                    .iter()
                    .flat_map(|constructor| constructor.fields.iter())
                    .filter(|field| field.is_datatype)
                    .map(|field| field.sort),
            );
            decls.insert(sort, decl);
        }

        // `scan.members` is already ordered by sort id and then term id, so the
        // reconstruction is deterministic.  Constructor applications written out
        // in the formula go first: such a term *is* its own value, so it is the
        // most concrete witness in its equality class, and propagating it is
        // what makes `(assert (= p (mk-pair 1 2)))` report `p = (mk-pair 1 2)`
        // instead of rebuilding `p` from accessors whose numbers the arithmetic
        // model never had to separate.
        let (_, equal_adjacency) = self.decided_dt_equalities(model, manager);
        for literal_pass in [true, false] {
            for (sort, terms) in &scan.members {
                for &term in terms {
                    if model.get(term).is_some() {
                        continue;
                    }
                    let is_literal = manager
                        .get(term)
                        .is_some_and(|node| matches!(node.kind, TermKind::DtConstructor { .. }));
                    if is_literal != literal_pass {
                        continue;
                    }
                    let Some(value) =
                        self.reconstruct_dt_value(term, *sort, &decls, model, manager)
                    else {
                        continue;
                    };
                    model.set(term, value);
                    // Everything the search proved equal to `term` denotes the
                    // same value, so it reports the same witness.
                    for member in equality_class(term, &equal_adjacency) {
                        Self::record_dt_value(member, value, model);
                    }
                }
            }
        }

        self.separate_disequal_dt_values(&decls, model, manager);
        debug_verify_dt_model(&self.assertions, model, manager);
    }

    /// Pull apart two datatype values that the search proved *distinct* but the
    /// reconstruction rendered identically.
    ///
    /// Distinctness of two applications of the *same* constructor lives entirely
    /// in their fields, and a field the theory never pinned has no witness of its
    /// own: `(assert (not (= p q)))` over `(mk-pair (fst Int) (snd Int))` leaves
    /// the linear solver free to report the same value for `(fst p)` and
    /// `(fst q)` (it discharges disequalities by case split, not by separating
    /// witnesses — the same effect `Solver::eval_in_model` documents for
    /// `distinct`), so both sides reconstructed to `(mk-pair 0 0)`.
    ///
    /// The repair only ever re-values a field whose accessor occurs in *no*
    /// assertion — one nothing in the formula constrains, where every value of
    /// the sort is equally legitimate — so it can only turn a wrong witness into
    /// a right one, never the reverse.  See [`Solver::separate_dt_value`] for
    /// why that, and not the arithmetic solver's `value()`, is the pin test.
    fn separate_disequal_dt_values(
        &self,
        decls: &FxHashMap<SortId, DeclInfo>,
        model: &mut Model,
        manager: &mut TermManager,
    ) {
        let (disequal, equal_adjacency) = self.decided_dt_equalities(model, manager);
        if disequal.is_empty() {
            return;
        }
        let asserted = assertion_subterms(&self.assertions, manager);
        for (left, right) in disequal {
            let (Some(left_value), Some(right_value)) = (model.get(left), model.get(right)) else {
                continue;
            };
            if left_value != right_value {
                continue;
            }
            // The whole class the search proved equal to `right` has to move
            // together, or the repair would break one of those equalities while
            // fixing the disequality.  A class containing *both* sides is a
            // contradictory assignment the repair must not paper over.
            let class = equality_class(right, &equal_adjacency);
            if class.contains(&left) {
                continue;
            }
            let Some(sort) = manager.get(right).map(|node| node.sort) else {
                continue;
            };
            self.separate_dt_value(right, sort, &class, &asserted, decls, model, manager);
        }
    }

    /// The datatype-sorted term pairs whose equality atom the search decided
    /// *false*, plus the adjacency of the ones it decided *true*.
    ///
    /// Read from the encoded `Eq` atoms rather than from `var_to_constraint`, so
    /// it sees every equality the Tseitin encoder internalised regardless of
    /// which theory ended up owning it — including the reconstruction axiom's
    /// own `t = Ci(sel(t)…)`, which relates every opaque datatype term to a
    /// constructor application.  Ordered by term id for determinism.
    fn decided_dt_equalities(&self, model: &Model, manager: &TermManager) -> DecidedEqualities {
        let mut disequal: Vec<(TermId, TermId)> = Vec::new();
        let mut adjacency: FxHashMap<TermId, Vec<TermId>> = FxHashMap::default();
        for &atom in self.term_to_var.keys() {
            let Some(TermKind::Eq(left, right)) = manager.get(atom).map(|node| &node.kind) else {
                continue;
            };
            let (left, right) = (*left, *right);
            if !manager
                .get(left)
                .is_some_and(|node| manager.sorts.is_datatype(node.sort))
            {
                continue;
            }
            match model.get(atom).and_then(|value| bool_value(value, manager)) {
                Some(false) => disequal.push((left.min(right), left.max(right))),
                Some(true) => {
                    adjacency.entry(left).or_default().push(right);
                    adjacency.entry(right).or_default().push(left);
                }
                None => {}
            }
        }
        disequal.sort_unstable();
        disequal.dedup();
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
        (disequal, adjacency)
    }

    /// Re-value one numeric field of `term`'s reconstructed value so that the
    /// value changes.
    ///
    /// Only fields of the outermost constructor are considered, and only
    /// `Int`/`Real` ones whose accessor occurs in *no* assertion — for every
    /// member of the equality class, so a field pinned through a term the search
    /// proved equal to `term` is left alone too.  That is the criterion that
    /// makes the repair safe: a term the assertions never mention has no
    /// user constraint on it at all (the only lemmas that speak about it are
    /// datatype axioms, which the repair moves *towards* satisfying), and it is
    /// also invisible to [`Solver::model_refutes_assertions`], which evaluates
    /// nothing but the assertions.  Note that the arithmetic solver's own
    /// `value()` is *not* a usable pin test: it reports a value for every
    /// tableau variable, constrained or not — reporting `0` for both `(fst p)`
    /// and `(fst q)` is exactly how the collision arises.
    ///
    /// A datatype all of whose scalar fields are pinned keeps its colliding
    /// value rather than acquiring a fabricated one.
    ///
    /// `class` is every term the search proved equal to `term`; all of them
    /// receive the new value so the repair cannot break an equality.
    #[allow(clippy::too_many_arguments)]
    fn separate_dt_value(
        &self,
        term: TermId,
        sort: SortId,
        class: &[TermId],
        asserted: &FxHashSet<TermId>,
        decls: &FxHashMap<SortId, DeclInfo>,
        model: &mut Model,
        manager: &mut TermManager,
    ) {
        let Some(decl) = decls.get(&sort) else {
            return;
        };
        let Some(TermKind::DtConstructor { constructor, args }) = model
            .get(term)
            .and_then(|value| manager.get(value))
            .cloned()
            .map(|node| node.kind)
        else {
            return;
        };
        let name = manager.resolve_str(constructor).to_string();
        let Some(index) = decl.constructors.iter().position(|c| c.name == name) else {
            return;
        };
        let fields: Vec<(String, SortId)> = decl.constructors[index]
            .fields
            .iter()
            .map(|field| (field.selector.clone(), field.sort))
            .collect();
        if fields.len() != args.len() {
            return;
        }
        let int_sort = manager.sorts.int_sort;
        let real_sort = manager.sorts.real_sort;
        for (position, (selector, field_sort)) in fields.into_iter().enumerate() {
            if field_sort != int_sort && field_sort != real_sort {
                continue;
            }
            let accessor = manager.mk_dt_selector(&selector, term, field_sort);
            let pinned = class.iter().any(|&member| {
                let member_accessor = manager.mk_dt_selector(&selector, member, field_sort);
                asserted.contains(&member_accessor)
            });
            if pinned || asserted.contains(&accessor) {
                continue;
            }
            let current = args[position];
            let bumped = match manager.get(current).map(|node| node.kind.clone()) {
                Some(TermKind::IntConst(value)) => manager.mk_int(value + 1),
                Some(TermKind::RealConst(value)) => {
                    manager.mk_real(value + num_rational::Rational64::from_integer(1))
                }
                _ => continue,
            };
            let mut new_args = args.to_vec();
            new_args[position] = bumped;
            let value = manager.mk_dt_constructor(&name, new_args, sort);
            // Publish the field too, so `(get-value ((fst q)))` reports the very
            // number `(get-model)` printed inside `q`.
            model.set(accessor, bumped);
            model.set(term, value);
            for &member in class {
                model.set(member, value);
            }
            return;
        }
    }

    /// Record `value` as the model entry for the datatype term `term`, unless it
    /// already has one.
    ///
    /// Restricted to *datatype-sorted* terms on purpose.  Recording a scalar
    /// field value here instead would put a number in front of
    /// [`Solver::model_refutes_assertions`] that no theory had committed to, and
    /// an unconstrained accessor such as `(head nil)` — which SMT-LIB leaves
    /// free — could then "falsify" `(= (head nil) 42)` and downgrade a correct
    /// `sat` to `unknown`.  A datatype value is inert for that gate (see
    /// [`Solver::extract_datatype_model`]), so memoising one is always safe.
    fn record_dt_value(term: TermId, value: TermId, model: &mut Model) {
        if model.get(term).is_none() {
            model.set(term, value);
        }
    }

    /// Build the constructor value of the datatype term `term`, or `None` when
    /// it cannot be determined (unknown sort, contradictory testers, or an
    /// ill-founded blind regress — see below).
    ///
    /// `None` is always safe: the term simply keeps no model entry and
    /// `(get-model)` falls back to the sort default, exactly as before this pass
    /// existed.
    ///
    /// # Iterative driver and termination
    ///
    /// The walk runs on an explicit heap stack of [`DtBuildFrame`]s, so a value
    /// of any depth the search actually committed to is reconstructed *in
    /// full*.  (A previous version cut the walk at a fixed depth and silently
    /// substituted sort defaults past the bound — a printed model that need not
    /// satisfy the constraints.)  Termination no longer relies on a depth cap;
    /// it is structural:
    ///
    /// * *Literal* steps descend into the arguments of an existing constructor
    ///   application — strictly smaller terms.
    /// * *Informed* steps (at least one tester literal of the term is decided,
    ///   or the term already has a memoised value) each consume a distinct
    ///   entry of the finite model, and the accessor terms along one path are
    ///   pairwise distinct, so a path holds finitely many of them.
    /// * *Blind* steps (no tester decided; the choice is the pure sort-default
    ///   fallback) are deterministic per sort.  Along a run of consecutive
    ///   blind steps that has left every *informative spine* (see
    ///   [`Solver::informative_spines`]), a repeated sort proves the
    ///   deterministic expansion would repeat forever — an ill-founded regress
    ///   with no ground value, reported honestly as `None`, never as a
    ///   fabricated value.  Off-spine blind runs are therefore bounded by the
    ///   number of distinct datatype sorts, and on-spine blind steps by the
    ///   total spine length.
    fn reconstruct_dt_value(
        &self,
        term: TermId,
        sort: SortId,
        decls: &FxHashMap<SortId, DeclInfo>,
        model: &mut Model,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let spines = self.informative_spines(model, manager);
        let mut frames: Vec<DtBuildFrame> = Vec::new();
        let mut step = DtStep::Enter {
            term,
            sort,
            blind: SmallVec::new(),
        };
        loop {
            step = match step {
                DtStep::Enter { term, sort, blind } => {
                    match self.open_dt_value(term, sort, blind, decls, &spines, model, manager) {
                        DtOpened::Done(value) => DtStep::Deliver(value),
                        DtOpened::Build(frame) => {
                            self.resume_build(frame, None, &mut frames, decls, model, manager)
                        }
                    }
                }
                DtStep::Deliver(value) => match frames.pop() {
                    None => return value,
                    Some(frame) => {
                        self.resume_build(frame, Some(value), &mut frames, decls, model, manager)
                    }
                },
            };
        }
    }

    /// Every term from which a chain of selector applications can still reach
    /// *informative* model data: a decided tester literal, or a datatype term
    /// with a memoised value.
    ///
    /// For each such target the whole selector spine (the target, its selector
    /// argument, that argument's argument, …) is included: a term is on a
    /// spine exactly when some selector chain from it reaches a target, and a
    /// selector child of an off-spine term is provably off-spine too.  The
    /// blind-regress detection in [`Solver::reconstruct_dt_value`] may
    /// therefore only fire off-spine, where no decided fact can ever be
    /// reached and the default expansion really is deterministic.
    fn informative_spines(&self, model: &Model, manager: &TermManager) -> FxHashSet<TermId> {
        let mut spines: FxHashSet<TermId> = FxHashSet::default();
        for (&key, &value) in model.assignments() {
            let Some(node) = manager.get(key) else {
                continue;
            };
            let root = match &node.kind {
                // A decided tester literal: informative about its argument.
                TermKind::DtTester { arg, .. } => *arg,
                // A datatype term with a memoised concrete value.
                _ if manager.sorts.is_datatype(node.sort) && is_value_term(value, manager) => key,
                _ => continue,
            };
            let mut cursor = root;
            loop {
                if !spines.insert(cursor) {
                    break;
                }
                match manager.get(cursor).map(|node| &node.kind) {
                    Some(TermKind::DtSelector { arg, .. }) => cursor = *arg,
                    _ => break,
                }
            }
        }
        spines
    }

    /// Open one datatype term of the reconstruction: either its value is
    /// already decided on the spot, or a constructor application must be
    /// assembled from its fields.
    #[allow(clippy::too_many_arguments)]
    fn open_dt_value(
        &self,
        term: TermId,
        sort: SortId,
        blind: SmallVec<[SortId; 4]>,
        decls: &FxHashMap<SortId, DeclInfo>,
        spines: &FxHashSet<TermId>,
        model: &mut Model,
        manager: &mut TermManager,
    ) -> DtOpened {
        // Memoised from an earlier reconstruction that walked through this term
        // as a field; also what makes `(get-value ((tail l)))` answer the same
        // sub-value `(get-model)` printed inside `l`.
        if let Some(value) = model.get(term)
            && is_value_term(value, manager)
        {
            return DtOpened::Done(Some(value));
        }
        let Some(decl) = decls.get(&sort) else {
            return DtOpened::Done(None);
        };

        // A literal constructor application is its own value — but its
        // arguments need not be (`(cons (head l) nil)`), so rebuild them.
        // Arguments are strictly smaller existing terms, so the blind-run
        // tracking restarts below them.
        if let Some(TermKind::DtConstructor { constructor, args }) =
            manager.get(term).map(|node| node.kind.clone())
        {
            let name = manager.resolve_str(constructor).to_string();
            let Some(index) = decl.constructors.iter().position(|c| c.name == name) else {
                return DtOpened::Done(None);
            };
            if decl.constructors[index].fields.len() != args.len() {
                // Built against a different declaration than the one in scope.
                return DtOpened::Done(None);
            }
            return DtOpened::Build(DtBuildFrame {
                sort,
                index,
                sources: args.to_vec(),
                position: 0,
                args: Vec::with_capacity(args.len()),
                child_blind: SmallVec::new(),
            });
        }

        let Some((index, informed)) = self.decided_constructor(term, decl, &*model, manager) else {
            return DtOpened::Done(None);
        };
        let child_blind = if informed || spines.contains(&term) {
            // A decided tester consumes a model entry, and a spine term makes
            // progress towards one; either way the run of pure-default steps
            // is broken, so the tracking restarts.
            SmallVec::new()
        } else {
            // Pure sort-default fallback with no reachable fact below: the
            // expansion of `sort` is deterministic, so a repeat proves an
            // infinite regress.  `None` is the honest answer — inventing a
            // value here is exactly what this reconstruction exists to avoid.
            if blind.contains(&sort) {
                return DtOpened::Done(None);
            }
            let mut extended = blind;
            extended.push(sort);
            extended
        };
        // The accessor terms are what the reconstruction axiom
        // `is_Ci(t) ⇒ t = Ci(sel_i1(t), …)` relates to `term`, so reading their
        // values reads the very fields the search committed to.
        let accessors: Vec<TermId> = decl.constructors[index]
            .fields
            .iter()
            .map(|field| (field.selector.clone(), field.sort))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(selector, field_sort)| manager.mk_dt_selector(&selector, term, field_sort))
            .collect();
        DtOpened::Build(DtBuildFrame {
            sort,
            index,
            sources: accessors,
            position: 0,
            args: Vec::new(),
            child_blind,
        })
    }

    /// Advance one pending constructor application: fold an incoming datatype
    /// field value (when resuming), fill scalar fields inline, and either
    /// request the next datatype field or deliver the finished value.
    ///
    /// Field semantics are identical to the recursive version: a field whose
    /// value cannot be determined falls back to [`ground_default_term`] — a
    /// legitimate witness for an unconstrained field — and only when that
    /// fails too does the whole application fail.
    fn resume_build(
        &self,
        mut frame: DtBuildFrame,
        incoming: Option<Option<TermId>>,
        frames: &mut Vec<DtBuildFrame>,
        decls: &FxHashMap<SortId, DeclInfo>,
        model: &mut Model,
        manager: &mut TermManager,
    ) -> DtStep {
        let Some(decl) = decls.get(&frame.sort) else {
            // The declaration was present when the frame was opened; treat a
            // vanished one as an honest failure rather than inventing a value.
            return DtStep::Deliver(None);
        };
        let fields: Vec<(SortId, bool)> = decl.constructors[frame.index]
            .fields
            .iter()
            .map(|field| (field.sort, field.is_datatype))
            .collect();

        if let Some(field_value) = incoming {
            // The datatype field at `frame.position` finished.
            let Some((field_sort, _)) = fields.get(frame.position).copied() else {
                return DtStep::Deliver(None);
            };
            let Some(&source) = frame.sources.get(frame.position) else {
                return DtStep::Deliver(None);
            };
            let value = match field_value {
                Some(value) => value,
                // A field the formula never constrains may take *any* value of
                // its sort: the sort default is a legitimate witness there,
                // not a guess.
                None => match ground_default_term(manager, field_sort) {
                    Some(value) => value,
                    None => return DtStep::Deliver(None),
                },
            };
            // The accessor now has a witness of its own; publish it so a
            // `(get-value ((tail l)))` sees the same sub-value that
            // `(get-model)` printed inside `l`.
            Self::record_dt_value(source, value, model);
            frame.args.push(value);
            frame.position += 1;
        }

        while let Some(&(field_sort, is_datatype)) = fields.get(frame.position) {
            let Some(&source) = frame.sources.get(frame.position) else {
                return DtStep::Deliver(None);
            };
            if is_datatype {
                let blind = frame.child_blind.clone();
                frames.push(frame);
                return DtStep::Enter {
                    term: source,
                    sort: field_sort,
                    blind,
                };
            }
            let value = match self.scalar_value(source, field_sort, &*model, manager) {
                Some(value) => value,
                None => match ground_default_term(manager, field_sort) {
                    Some(value) => value,
                    None => return DtStep::Deliver(None),
                },
            };
            frame.args.push(value);
            frame.position += 1;
        }

        let name = decl.constructors[frame.index].name.clone();
        let value = manager.mk_dt_constructor(&name, frame.args, frame.sort);
        DtStep::Deliver(Some(value))
    }

    /// The constructor index the search decided for the opaque datatype term
    /// `term`, or a deterministic choice when it decided none, plus whether the
    /// decision was *informed* — whether any tester literal of `term` (true
    /// *or* false) was decided at all, as opposed to the pure sort-default
    /// fallback.
    ///
    /// Returns `None` only when every constructor is ruled out, which the
    /// exhaustiveness axiom makes impossible for an axiomatised term; refusing
    /// to invent a value there keeps a genuinely broken assignment out of the
    /// printed model.
    fn decided_constructor(
        &self,
        term: TermId,
        decl: &DeclInfo,
        model: &Model,
        manager: &mut TermManager,
    ) -> Option<(usize, bool)> {
        let names: Vec<String> = decl
            .constructors
            .iter()
            .map(|constructor| constructor.name.clone())
            .collect();
        let mut ruled_out = vec![false; names.len()];
        let mut informed = false;
        for (index, name) in names.iter().enumerate() {
            let tester = manager.mk_dt_tester(name, term);
            match model
                .get(tester)
                .and_then(|value| bool_value(value, manager))
            {
                // Mutual exclusivity makes at most one tester true.
                Some(true) => return Some((index, true)),
                Some(false) => {
                    ruled_out[index] = true;
                    informed = true;
                }
                None => {}
            }
        }
        // No tester decided true: any constructor the search did not rule out
        // is a legitimate value.  Prefer one with no datatype-sorted field so
        // the reconstruction bottoms out immediately (`nil` over `cons`); this
        // is the same choice `ground_default_term` makes, so an unconstrained
        // datatype term reads the same whether it was axiomatised or not.
        base_constructor(decl, |index| !ruled_out[index])
            .or_else(|| (0..names.len()).find(|&index| !ruled_out[index]))
            .map(|index| (index, informed))
    }

    /// The model value of a non-datatype field, taken from the same theory that
    /// owns it in [`Solver::build_model`], or `None` when nothing constrains it.
    fn scalar_value(
        &self,
        term: TermId,
        sort: SortId,
        model: &Model,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        // A literal argument of a constructor application written out in the
        // formula (`(cons 1 nil)`) is already its own value; nothing else may
        // override it.
        if is_value_term(term, manager) {
            return Some(term);
        }
        // A direct entry already went through the theory-ownership rules above
        // (Boolean atoms from the SAT assignment, arithmetic from the tableau,
        // bit-vectors from the bit-blaster, strings from the ground solver).
        if let Some(value) = model.get(term)
            && is_value_term(value, manager)
        {
            return Some(value);
        }
        // An accessor introduced by the reconstruction axiom may carry a theory
        // value without ever having been registered for model extraction.
        if sort == manager.sorts.int_sort {
            let value = self.arith.value(term)?;
            return Some(manager.mk_int(*value.numer()));
        }
        if sort == manager.sorts.real_sort {
            let value = self.arith.value(term)?;
            return Some(manager.mk_real(value));
        }
        let width = manager
            .sorts
            .get(sort)
            .and_then(oxiz_core::sort::Sort::bitvec_width);
        if let Some(width) = width {
            // Full-width witness: `get_value` is `None` above 64 bits, which
            // would report the field as unconstrained and let the datatype
            // reconstruction fill it with a sort default instead.
            let value = self.bv.get_value_big(term)?;
            let wrapped = oxiz_core::ast::bv_wrap_unsigned(&num_bigint::BigInt::from(value), width);
            return Some(manager.mk_bitvec(wrapped, width));
        }
        None
    }

    /// Decide whether the `BvSolver`'s bit-blasted model is authoritative for
    /// BV terms in the current problem.
    ///
    /// It is authoritative when the problem contains genuine BV *structure* —
    /// any BV arithmetic/bitwise/shift/concat/extract operation — or any BV
    /// (dis)equality or comparison constraint.  All of those paths bit-blast
    /// their operands with constant bits pinned to concrete values, so
    /// `BvSolver::get_value` is a faithful witness.
    ///
    /// Comparisons count because `TheoryManager::process_constraint` now
    /// bit-blasts their operands through `encode_bv_term_recursive` (pinning
    /// `BitVecConst` bits) instead of allocating free bits via `new_bv`.  While
    /// they were unpinned the BV model was arbitrary for a comparison-only
    /// problem and the linear relaxation had to be trusted instead — but that
    /// relaxation carries no `0 <= x < 2^width` bound and, for signed
    /// comparisons, is deliberately never populated at all, so it produced
    /// values that violate the very atom that made the query SAT (`x = 0` for
    /// `(bvsle x #b10000000)`).
    fn bv_solver_is_authoritative(&self, manager: &TermManager) -> bool {
        // Any structural BV operation implies real bit-blasting.
        for &term in &self.bv_terms {
            if let Some(t) = manager.get(term)
                && Self::is_structural_bv_op(&t.kind)
            {
                return true;
            }
        }

        // Any BV (dis)equality or comparison also bit-blasts both operands with
        // pinned constants.  A disequality `a != b` is stored as an `Eq` atom
        // whose SAT variable is assigned false, so both cases surface as `Eq`;
        // `bvult`/`bvule`/`bvslt`/`bvsle` surface as `Lt`/`Le`.
        let is_bv = |tid: TermId| -> bool {
            manager
                .get(tid)
                .and_then(|t| manager.sorts.get(t.sort))
                .is_some_and(|s| s.is_bitvec())
        };
        for constraint in self.var_to_constraint.values() {
            let operands = match constraint {
                Constraint::Eq(lhs, rhs)
                | Constraint::Lt(lhs, rhs)
                | Constraint::Le(lhs, rhs)
                | Constraint::Gt(lhs, rhs)
                | Constraint::Ge(lhs, rhs) => (*lhs, *rhs),
                _ => continue,
            };
            if is_bv(operands.0) || is_bv(operands.1) {
                return true;
            }
        }

        false
    }

    /// Whether a `TermKind` is a structural BV operation (arithmetic, bitwise,
    /// shift, concat, or extract) — as opposed to a comparison, constant, or
    /// variable.  Structural ops are the ones the BV solver genuinely
    /// bit-blasts, making its model authoritative.
    fn is_structural_bv_op(kind: &TermKind) -> bool {
        matches!(
            kind,
            TermKind::BvNot(_)
                | TermKind::BvAnd(_, _)
                | TermKind::BvOr(_, _)
                | TermKind::BvXor(_, _)
                | TermKind::BvAdd(_, _)
                | TermKind::BvSub(_, _)
                | TermKind::BvMul(_, _)
                | TermKind::BvUdiv(_, _)
                | TermKind::BvSdiv(_, _)
                | TermKind::BvUrem(_, _)
                | TermKind::BvSrem(_, _)
                | TermKind::BvShl(_, _)
                | TermKind::BvLshr(_, _)
                | TermKind::BvAshr(_, _)
                | TermKind::BvConcat(_, _)
                | TermKind::BvExtract { .. }
        )
    }

    /// Canonical EUF congruence-class representative node for `term`.
    ///
    /// Returns `None` when the term was never interned into the congruence
    /// closure (it took part in no (dis)equality), so distinct such terms are
    /// treated as distinct.  Model output uses this to give uninterpreted-sort
    /// constants proven equal a *shared* abstract witness while keeping
    /// distinct constants distinct.
    pub(crate) fn euf_class_representative(&self, term: TermId) -> Option<u32> {
        if let Some(node) = self.euf.term_to_node(term) {
            return Some(self.euf.find_immutable(node));
        }
        // The pure-equality fast path decides its formulas without ever
        // running EUF, so its partition is the only record of which
        // uninterpreted constants it proved equal (see `eq_skeleton`). The two
        // sources cannot both be populated for one verdict, so there is no
        // ambiguity about which class id a caller is looking at.
        self.equality_skeleton_classes.get(&term).copied()
    }

    /// Build unsat core for trivial conflicts (assertion of false)
    pub(super) fn build_unsat_core_trivial_false(&mut self) {
        if !self.produce_unsat_cores {
            self.unsat_core = None;
            return;
        }

        // Find all assertions that are trivially false
        let mut core = UnsatCore::new();

        for (i, &term) in self.assertions.iter().enumerate() {
            if term == TermId::new(1) {
                // This is a false assertion
                core.indices.push(i as u32);

                // Find the name if there is one
                if let Some(named) = self.named_assertions.iter().find(|na| na.index == i as u32)
                    && let Some(ref name) = named.name
                {
                    core.names.push(name.clone());
                }
            }
        }

        self.unsat_core = Some(core);
    }

    /// Build the initial (conservative) unsat core after `check()` returned
    /// `Unsat`.
    ///
    /// This records every tracked assertion — a *valid* unsatisfiable set (a
    /// superset of any minimal core is still unsatisfiable), but not minimal on
    /// its own.  Minimization is deliberately left to query time: the SMT-LIB
    /// `(get-unsat-core)` path drives greedy deletion-based minimization via
    /// [`Solver::minimize_unsat_core`], which needs the `TermManager` to
    /// re-solve subsets (unavailable here).  Doing it eagerly for every `Unsat`
    /// solve — including the many that never issue `(get-unsat-core)` — would
    /// pay the re-solve cost unconditionally, so the split is intentional.
    ///
    /// True assumption-literal-based extraction (one selector per assertion,
    /// reading the SAT layer's failed-assumption set) would make this minimal
    /// without re-solving, but requires the encoder to gate each assertion
    /// behind a fresh selector variable — a larger change than this method.
    pub(super) fn build_unsat_core(&mut self) {
        if !self.produce_unsat_cores {
            self.unsat_core = None;
            return;
        }

        let mut core = UnsatCore::new();
        for na in &self.named_assertions {
            core.indices.push(na.index);
            if let Some(ref name) = na.name {
                core.names.push(name.clone());
            }
        }

        self.unsat_core = Some(core);
    }
}

/// One step of the iterative datatype reconstruction driver.
enum DtStep {
    /// Produce the value of the datatype term `term` of sort `sort`.  `blind`
    /// is the set of sorts expanded by consecutive off-spine pure-default
    /// steps on the current path (see [`Solver::reconstruct_dt_value`]).
    Enter {
        term: TermId,
        sort: SortId,
        blind: SmallVec<[SortId; 4]>,
    },
    /// A term finished with this value (`None` = honestly undeterminable);
    /// hand it to the innermost pending constructor application.
    Deliver(Option<TermId>),
}

/// What opening one datatype term produced: a decided value, or a constructor
/// application whose fields must be filled in first.
enum DtOpened {
    Done(Option<TermId>),
    Build(DtBuildFrame),
}

/// A pending constructor application of the iterative datatype
/// reconstruction: constructor `index` of the declaration of `sort`, applied
/// to the values of `sources` (the constructor's own arguments, or the
/// accessor terms standing for them).
struct DtBuildFrame {
    /// Sort of the value being built (also the key of its declaration).
    sort: SortId,
    /// Constructor index within that declaration.
    index: usize,
    /// Source terms the field values are read from.
    sources: Vec<TermId>,
    /// The field currently being produced (fields before it are in `args`).
    position: usize,
    /// Finished field values, in field order.
    args: Vec<TermId>,
    /// Blind-run sorts to hand to each datatype field.
    child_blind: SmallVec<[SortId; 4]>,
}

/// Whether `term` is a ground *value* term — something that can stand as a
/// model assignment on its own rather than an expression still to be evaluated.
fn is_value_term(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|node| {
        matches!(
            node.kind,
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_)
                | TermKind::FpLit { .. }
                | TermKind::FpPlusInfinity { .. }
                | TermKind::FpMinusInfinity { .. }
                | TermKind::FpPlusZero { .. }
                | TermKind::FpMinusZero { .. }
                | TermKind::FpNaN { .. }
                | TermKind::DtConstructor { .. }
        )
    })
}

/// Every sub-term occurring in `assertions`, quantifier bodies included.
///
/// Used as the "is this term pinned by the formula?" test: a term that occurs
/// nowhere in the assertions carries no user constraint, so a model completion
/// is free to give it any value of its sort.
fn assertion_subterms(assertions: &[TermId], manager: &TermManager) -> FxHashSet<TermId> {
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = assertions.to_vec();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(term) = stack.pop() {
        if !seen.insert(term) {
            continue;
        }
        let Some(node) = manager.get(term) else {
            continue;
        };
        children.clear();
        super::term_walk::collect_structural_children(&node.kind, &mut children);
        stack.extend(children.iter().copied());
    }
    seen
}

/// Every term reachable from `term` through decided-true equalities, `term`
/// included, in ascending term-id order.
fn equality_class(term: TermId, adjacency: &FxHashMap<TermId, Vec<TermId>>) -> Vec<TermId> {
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    let mut stack = vec![term];
    while let Some(next) = stack.pop() {
        if !seen.insert(next) {
            continue;
        }
        if let Some(neighbours) = adjacency.get(&next) {
            stack.extend(neighbours.iter().copied());
        }
    }
    let mut class: Vec<TermId> = seen.into_iter().collect();
    class.sort_unstable();
    class
}

/// The Boolean a model entry stands for, or `None` when the entry is not a
/// Boolean literal.
fn bool_value(term: TermId, manager: &TermManager) -> Option<bool> {
    match manager.get(term)?.kind {
        TermKind::True => Some(true),
        TermKind::False => Some(false),
        _ => None,
    }
}

/// Release build: the datatype model-validity net compiles away entirely.
#[cfg(not(debug_assertions))]
#[inline]
fn debug_verify_dt_model(_assertions: &[TermId], _model: &Model, _manager: &TermManager) {}

/// Debug-only model-validity net for the datatype reconstruction.
///
/// Substitutes the reconstructed values back into the original assertions and
/// evaluates their *datatype fragment* — testers and datatype equalities, under
/// the Boolean structure that connects them.  An assertion that comes out
/// definitively `false` means the printed model does not satisfy the formula,
/// which is precisely the defect this pass exists to remove: before it,
/// `((_ is cons) l) ∧ (= (head l) 7)` was answered `sat` with the witness
/// `l = nil`, and the tester evaluates to `false` under exactly that witness.
///
/// Deliberately three-valued and tolerant.  Anything outside the datatype
/// fragment — arithmetic, bit-vectors, uninterpreted applications, a term with
/// no reconstructed value — is *inconclusive*, and inconclusive poisons every
/// enclosing connective, so only a violation the reconstruction is genuinely
/// responsible for can fire.  A datatype equality follows the same asymmetry
/// [`Solver::eval_in_model`] uses for numbers: two *different* values falsify
/// it, but two identical values are not evidence that it holds, because the
/// theories below do not always separate the witnesses of terms they merely
/// proved distinct.  Compiles to nothing in release builds.
#[cfg(debug_assertions)]
fn debug_verify_dt_model(assertions: &[TermId], model: &Model, manager: &TermManager) {
    for &assertion in assertions {
        debug_assert!(
            dt_fragment_value(assertion, model, manager) != Some(false),
            "reconstructed datatype model falsifies an assertion: {}",
            oxiz_core::smtlib::Printer::new(manager).print_term(assertion)
        );
    }
}

/// Three-valued evaluation of `term`'s datatype fragment under `model`;
/// `None` means inconclusive.  See [`debug_verify_dt_model`].
///
/// Iterative (explicit frame stack), so the net keeps working — instead of
/// overflowing the stack or silently going inconclusive past a depth cap — on
/// arbitrarily deep Boolean structure.  Short-circuiting matches the
/// recursive original: `and` stops at the first definite `false`, `or` at the
/// first definite `true`, while an inconclusive operand only taints the
/// result; `=>` evaluates both sides.
#[cfg(debug_assertions)]
fn dt_fragment_value(term: TermId, model: &Model, manager: &TermManager) -> Option<bool> {
    /// One pending connective of the three-valued walk.
    enum FragFrame {
        Not,
        And {
            args: SmallVec<[TermId; 4]>,
            next: usize,
            all_true: bool,
        },
        Or {
            args: SmallVec<[TermId; 4]>,
            next: usize,
            all_false: bool,
        },
        ImpliesLhs {
            rhs: TermId,
        },
        ImpliesRhs {
            lhs: Option<bool>,
        },
    }

    let mut frames: Vec<FragFrame> = Vec::new();
    let mut current = term;
    'open: loop {
        // Evaluate leaves; descend through connectives.
        let mut value: Option<bool> = loop {
            let Some(node) = manager.get(current) else {
                break None;
            };
            match &node.kind {
                TermKind::True => break Some(true),
                TermKind::False => break Some(false),
                TermKind::Not(arg) => {
                    frames.push(FragFrame::Not);
                    current = *arg;
                }
                TermKind::And(args) => match args.first() {
                    Some(&first) => {
                        frames.push(FragFrame::And {
                            args: args.clone(),
                            next: 1,
                            all_true: true,
                        });
                        current = first;
                    }
                    None => break Some(true),
                },
                TermKind::Or(args) => match args.first() {
                    Some(&first) => {
                        frames.push(FragFrame::Or {
                            args: args.clone(),
                            next: 1,
                            all_false: true,
                        });
                        current = first;
                    }
                    None => break Some(false),
                },
                TermKind::Implies(a, b) => {
                    frames.push(FragFrame::ImpliesLhs { rhs: *b });
                    current = *a;
                }
                // `is_C(t)` is decided by the constructor of `t`'s
                // reconstructed value.
                TermKind::DtTester { constructor, arg } => {
                    let expected = *constructor;
                    let arg = *arg;
                    break model
                        .get(arg)
                        .and_then(|value| match &manager.get(value)?.kind {
                            TermKind::DtConstructor {
                                constructor: actual,
                                ..
                            } => Some(*actual == expected),
                            _ => None,
                        });
                }
                // Datatype values are hash-consed ground trees, so structural
                // equality is term-id equality.  Only the *negative* direction
                // is evidence.
                TermKind::Eq(left, right)
                    if manager
                        .get(*left)
                        .is_some_and(|node| manager.sorts.is_datatype(node.sort)) =>
                {
                    let (left, right) = (*left, *right);
                    break match (model.get(left), model.get(right)) {
                        (Some(left_value), Some(right_value)) => {
                            (left_value != right_value).then_some(false)
                        }
                        _ => None,
                    };
                }
                _ => break None,
            }
        };

        // Fold the finished operand into the pending connectives.
        loop {
            let Some(frame) = frames.pop() else {
                return value;
            };
            match frame {
                FragFrame::Not => value = value.map(|v| !v),
                FragFrame::And {
                    args,
                    next,
                    mut all_true,
                } => {
                    match value {
                        // Definite `false` decides the conjunction; the
                        // remaining operands are not evaluated.
                        Some(false) => {
                            value = Some(false);
                            continue;
                        }
                        Some(true) => {}
                        None => all_true = false,
                    }
                    if let Some(&child) = args.get(next) {
                        frames.push(FragFrame::And {
                            args,
                            next: next + 1,
                            all_true,
                        });
                        current = child;
                        continue 'open;
                    }
                    value = all_true.then_some(true);
                }
                FragFrame::Or {
                    args,
                    next,
                    mut all_false,
                } => {
                    match value {
                        Some(true) => {
                            value = Some(true);
                            continue;
                        }
                        Some(false) => {}
                        None => all_false = false,
                    }
                    if let Some(&child) = args.get(next) {
                        frames.push(FragFrame::Or {
                            args,
                            next: next + 1,
                            all_false,
                        });
                        current = child;
                        continue 'open;
                    }
                    value = all_false.then_some(false);
                }
                FragFrame::ImpliesLhs { rhs } => {
                    frames.push(FragFrame::ImpliesRhs { lhs: value });
                    current = rhs;
                    continue 'open;
                }
                FragFrame::ImpliesRhs { lhs } => {
                    value = match (lhs, value) {
                        (Some(false), _) | (_, Some(true)) => Some(true),
                        (Some(true), Some(false)) => Some(false),
                        _ => None,
                    };
                }
            }
        }
    }
}

/// The index of the constructor a *ground default* value of a datatype uses:
/// the first one, among those `allowed`, with no datatype-sorted field.
///
/// Preferring a field-free ("base") constructor is what makes a default
/// construction bottom out immediately — `nil` rather than `cons`, matching
/// what Z3 prints for an unconstrained list.  Shared between the solver's
/// reconstruction and `Context`'s sort defaults so both agree on which
/// constructor an underdetermined datatype value uses.
pub(super) fn base_constructor(
    decl: &super::dt_axioms::DeclInfo,
    allowed: impl Fn(usize) -> bool,
) -> Option<usize> {
    decl.constructors
        .iter()
        .enumerate()
        .position(|(index, c)| allowed(index) && !c.fields.iter().any(|field| field.is_datatype))
}

/// The index of the constructor a ground default value of the datatype `def`
/// uses, expressed directly over a [`oxiz_core::sort::DataTypeDef`].
///
/// The same policy as [`base_constructor`] — first constructor with no
/// datatype-sorted field, else the first constructor — for callers that hold a
/// raw declaration rather than the solver's resolved [`DeclInfo`].  Keeping the
/// choice in one place is what lets `(get-model)` and `(get-value ..)` report
/// the same value for a datatype constant nothing constrains.
pub(crate) fn default_constructor_index(
    def: &oxiz_core::sort::DataTypeDef,
    sorts: &oxiz_core::sort::SortManager,
) -> Option<usize> {
    if def.constructors.is_empty() {
        return None;
    }
    def.constructors
        .iter()
        .position(|constructor| {
            !constructor
                .selectors
                .iter()
                .any(|&(_, field_sort)| sorts.is_datatype(field_sort))
        })
        .or(Some(0))
}

/// A ground value term of `sort`, or `None` for a sort with no constructible
/// ground witness (arrays, uninterpreted sorts, sort parameters, and datatypes
/// whose declaration is missing or not well-founded).
///
/// This is the single source of sort defaults for model completion: the
/// solver's datatype reconstruction uses it for an unconstrained field, and
/// [`crate::Context`]'s `default_value_term` delegates to it so a `(get-value
/// ..)` completion and a reconstructed field never disagree about what "the
/// default of this sort" is.
///
/// Runs on an explicit heap stack, so a deeply nested (but well-founded)
/// datatype family is built in full instead of being cut at an arbitrary
/// depth.  Termination is structural: the default expansion of a sort is a
/// pure function of the sort, so a sort recurring on the current expansion
/// path proves an ill-founded declaration (`(declare-datatype T ((c (f
/// T))))`) whose regress would never bottom out — reported honestly as
/// `None`.  A per-call memo keyed on the sort both bounds the work on
/// DAG-shaped declarations and reuses the (deterministic) result.
pub(crate) fn ground_default_term(manager: &mut TermManager, sort: SortId) -> Option<TermId> {
    /// A pending datatype default: constructor `index`'s remaining field
    /// sorts, with finished field values in `args`.
    struct DefaultFrame {
        sort: SortId,
        ctor_name: String,
        field_sorts: Vec<SortId>,
        position: usize,
        args: Vec<TermId>,
    }

    let mut memo: FxHashMap<SortId, Option<TermId>> = FxHashMap::default();
    // Sorts of the datatype defaults currently being assembled (the frames'
    // sorts, innermost last) — the cycle-detection path.
    let mut path: Vec<SortId> = Vec::new();
    let mut frames: Vec<DefaultFrame> = Vec::new();
    let mut current = sort;

    'open: loop {
        // Produce the default of `current`, or descend into a datatype.
        let mut value: Option<TermId> = loop {
            if let Some(&hit) = memo.get(&current) {
                break hit;
            }
            if path.contains(&current) {
                // The deterministic expansion of this sort re-expands itself:
                // an ill-founded declaration with no ground value at all.
                break None;
            }
            if current == manager.sorts.bool_sort {
                break Some(manager.mk_false());
            }
            if current == manager.sorts.int_sort {
                break Some(manager.mk_int(0));
            }
            if current == manager.sorts.real_sort {
                break Some(manager.mk_real(num_rational::Rational64::new(0, 1)));
            }
            let Some(kind) = manager.sorts.get(current).map(|s| s.kind.clone()) else {
                break None;
            };
            match kind {
                SortKind::String => break Some(manager.mk_string_lit("")),
                SortKind::BitVec(width) => {
                    break Some(manager.mk_bitvec(num_bigint::BigInt::from(0), width));
                }
                SortKind::FloatingPoint { eb, sb } => {
                    break Some(manager.mk_fp_plus_zero(eb, sb));
                }
                // The reserved five-element `RoundingMode` sort: its default
                // is IEEE 754's own default mode.  Unlike an uninterpreted
                // sort (which falls through to `None` below because no ground
                // witness can be minted) this one has five ground terms, so
                // `(get-value (m))` over an unconstrained rounding mode
                // completes to a real mode instead of echoing the constant.
                SortKind::RoundingMode => {
                    break Some(manager.mk_rounding_mode(oxiz_core::ast::RoundingMode::RNE));
                }
                SortKind::Datatype(_) => {
                    let opened = (|| {
                        let name = manager.sorts.datatype_name(current)?.to_string();
                        let def = manager.sorts.get_datatype(&name)?;
                        let index = default_constructor_index(def, &manager.sorts)?;
                        let constructor = def.constructors.get(index)?;
                        let ctor_name = manager.resolve_str(constructor.name).to_string();
                        let field_sorts: Vec<SortId> = constructor
                            .selectors
                            .iter()
                            .map(|&(_, field_sort)| field_sort)
                            .collect();
                        Some((ctor_name, field_sorts))
                    })();
                    let Some((ctor_name, field_sorts)) = opened else {
                        break None;
                    };
                    match field_sorts.first().copied() {
                        Some(first) => {
                            path.push(current);
                            frames.push(DefaultFrame {
                                sort: current,
                                ctor_name,
                                field_sorts,
                                position: 1,
                                args: Vec::new(),
                            });
                            current = first;
                        }
                        None => {
                            // Field-free constructor: the default is immediate.
                            break Some(manager.mk_dt_constructor(&ctor_name, Vec::new(), current));
                        }
                    }
                }
                _ => break None,
            }
        };

        // Fold the finished (or failed) field into the pending frames.
        loop {
            let Some(mut frame) = frames.pop() else {
                memo.insert(sort, value);
                return value;
            };
            let Some(field_value) = value else {
                // A field with no ground default: the whole constructor —
                // and with it this sort — has none either.
                path.pop();
                memo.insert(frame.sort, None);
                value = None;
                continue;
            };
            frame.args.push(field_value);
            if let Some(&next) = frame.field_sorts.get(frame.position) {
                frame.position += 1;
                frames.push(frame);
                current = next;
                continue 'open;
            }
            path.pop();
            let built = manager.mk_dt_constructor(&frame.ctor_name, frame.args, frame.sort);
            memo.insert(frame.sort, Some(built));
            value = Some(built);
        }
    }
}
