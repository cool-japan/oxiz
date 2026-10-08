//! What a `sat` publishes, settled once per check: the model every renderer
//! reads, with fresh values for the classes no theory valued (`#P2b-71`,
//! `context::model_fmt::mint`), and — for every `sat` whose asserted goal
//! holds a quantifier — the verdict of the published-model certificate
//! (`#P2b-51`, decisions (54), (67), (68)); for a quantifier-free `sat` over
//! a datatype, an enumeration, an uninterpreted sort or an array over one,
//! the honesty net (decision (85), re-fix pass 18): a candidate model the
//! exact value reader shows false makes the check `unknown` — the verdict is
//! only as good as the model behind it.
//!
//! # Fresh values (`#P2b-71`)
//!
//! Every congruence class no theory valued — a constant, an application, a
//! read — gets a value of its own, keyed by the point it reads, so distinct
//! classes print distinct values and one point prints one value (`mint`).
//! The minted values live in a copy of the solver's model that `(get-model)`,
//! `(get-value)` and the function interpretations all read.  At a
//! quantifier-free `sat` the printed model is evaluated exactly once they
//! changed it, and where they make an assertion false the solver's own model
//! is printed instead (`printed_check`, decision (64)).
//!
//! # The published-model certificate (decisions (54)(i), (67), (68))
//!
//! At a `sat` whose ASSERTED goal holds a quantifier — any theory, including
//! a quantifier the search removed by Skolemisation, destructive equality
//! resolution or vacuous-binder elimination — and whose model no certificate
//! of the check already covered, every declared constant is rendered exactly
//! as `(get-model)` prints it and every uninterpreted function's printed
//! table is read back; the goal, with every application replaced by its
//! table, is certified by the completion's certificate
//! (`solver::array_completion_certify::printed`) or, where its queries cannot
//! decide, by exact evaluation (`printed_check::certified_by_evaluation`).
//! A model that fails with the fresh values and holds without them is printed
//! without them; one that fails either way is completed around what it got
//! right where its arrays allow, and otherwise not published:
//! `(get-model)` answers `(error "model not certified: …")`, and
//! `(get-value)` answers the same for a term that reads an uncertified symbol
//! — an array of an array goal (a term reading none still answers), any
//! declared constant or function of another goal.  The verdict is left as it
//! is (decision (40)).

use super::*;
use crate::prelude::{FxHashMap, FxHashSet};

/// What the last `sat` publishes; reset by every check and by every command
/// that invalidates the last result.
#[derive(Default, Debug)]
pub(in crate::context) struct PublishedModel {
    /// The solver's model with the fresh values of `#P2b-71` added, or `None`
    /// when nothing was minted (the solver's model is then read as is).
    model: Option<crate::solver::Model>,
    /// The error `(get-model)` answers when the model failed its certificate.
    uncertified: Option<String>,
    /// The arrays of a model that failed its certificate (every declared
    /// constant, for a goal with no array).
    uncertified_arrays: FxHashSet<TermId>,
    /// The functions of a model with no array that failed its certificate.
    uncertified_functions: FxHashSet<String>,
    /// Else values of printed function tables the certificate chose over
    /// the most common entry value (`Context::repair_else_values`).
    else_overrides: FxHashMap<String, String>,
    /// Why the last check answers `unknown` where the search answered `sat`:
    /// the honesty net read an assertion false under the candidate model
    /// (decision (85)); `(get-info :reason-unknown)` reports it.
    unknown_reason: Option<String>,
}

/// What the quantifier-free net reads of a candidate model
/// ([`Context::printed_datatype_model_refuted`]).
enum NetReading {
    /// Every assertion holds or stays open.
    Clean,
    /// The exact value reader reads this assertion (1-based) false.
    False(usize),
    /// The structural evaluator reads this assertion (1-based) false where
    /// the value reader is open, and a fresh solver refutes it closed.
    Refuted(usize),
    /// This assertion (1-based) applies a function whose printed table does
    /// not read back (named), and a fresh solver does not show it true under
    /// every interpretation of that function (decision (92)(d)).
    Unread(usize, String),
}

/// Else values the repair may certify.
const MAX_ELSE_REPAIRS: usize = 12;

/// Conflicts the fresh solver that confirms a datatype model's refutation may
/// spend (`Context::printed_datatype_model_refuted`); a closed assertion has
/// no free symbol, so a refutation is a propagation, and the budget only
/// bounds a pathological one.
const DATATYPE_NET_CONFLICTS: u64 = 10_000;

/// How an `@uc_` witness prefix is spelled for the term parser, which rejects
/// `@` symbols; the symbols it yields are replaced by sorted witness constants
/// straight after parsing, so the spelling never reaches a query.
pub(super) const WITNESS_PLACEHOLDER: &str = "oxiz__uc_witness__";

impl Context {
    /// Settle what the last check publishes (called wherever a check result
    /// is recorded): keep a correct candidate (decision (48)), mint fresh
    /// values (`#P2b-71`), then
    ///
    /// * at a `sat` whose **asserted** goal holds a quantifier — any theory,
    ///   including one the search removed by Skolemisation, destructive
    ///   equality resolution or vacuous-binder elimination (decision (67)) —
    ///   certify the model as printed, functions included (decision (68)),
    ///   unless a certificate already covered exactly this model;
    /// * at a quantifier-free `sat` the fresh values changed, check the
    ///   printed model exactly and print the solver's own model instead when
    ///   the fresh values made an assertion false (decision (64)).
    ///
    /// Returns the verdict the check answers: `result`, except a
    /// quantifier-free `sat` whose candidate model the honesty net reads
    /// false (decision (85)), which answers `unknown` — and then
    /// `last_result` is set to `unknown` here, since the net renders the
    /// model through a `sat` `last_result`.
    pub(in crate::context) fn settle_published_model(
        &mut self,
        result: SolverResult,
    ) -> SolverResult {
        self.published = PublishedModel::default();
        if result != SolverResult::Sat {
            return result;
        }
        self.keep_a_correct_candidate_model();
        self.refresh_published_model();
        let minted = self.published.model.is_some();
        if self.goal_is_quantified() {
            let covered = self.solver.model_already_certified()
                && !minted
                && !self
                    .declared_funs
                    .iter()
                    .any(|d| !d.interpreted && !d.arg_sorts.is_empty());
            if !covered {
                self.certify_published_model();
            }
            return result;
        }
        if minted && self.printed_model_falsifies().is_some() {
            let fresh = self.published.model.take();
            // Neither reading holds: the fresh values stay out, so the model
            // printed is the solver's own, as on every earlier build.
            if self.printed_model_falsifies().is_none() {
                return result;
            }
            drop(fresh);
        }
        self.withhold_a_falsifying_datatype_model(result)
    }

    /// `(get-info :reason-unknown)`: the honesty net's reason where it took a
    /// `sat` back (decision (85)), `incomplete` for any other `unknown`.
    pub(in crate::context) fn reason_unknown_info(&self) -> String {
        match (self.last_result, self.published.unknown_reason.as_ref()) {
            (Some(SolverResult::Unknown), Some(reason)) => format!(
                "(:reason-unknown {})",
                oxiz_core::smtlib::format_string_literal(reason)
            ),
            (Some(SolverResult::Unknown), None) => "(:reason-unknown incomplete)".to_string(),
            _ => "(:reason-unknown \"not applicable\")".to_string(),
        }
    }

    /// The honesty net (decision (85), re-fix pass 18) over a
    /// quantifier-free `sat` whose problem mentions a value no theory prints
    /// exactly — a declared symbol whose sort CONTAINS a datatype, an
    /// enumeration or an uninterpreted sort (a constant of one, an array whose
    /// index or element sort is one at any depth, a function into or over
    /// one), or an assertion with a term of such a sort (decisions (72)(a),
    /// (79)(a), (85)).  The candidate model, exactly as `(get-model)` would
    /// print it, is read against every assertion in scope:
    ///
    /// * an assertion the exact value reader (`printed_eval::ground_values`)
    ///   reads FALSE makes the verdict `unknown` ("model check failed:
    ///   assertion N reads false under the candidate model") and
    ///   `(get-model)` answers the certified-or-absent error: a `sat` is never
    ///   answered over a model the solver itself shows false, and no fresh
    ///   solver is asked to overrule that reading — recheck 17 measured one
    ///   (`gen_dt.py` seed 30100192 `d00318`, its second check) answering
    ///   `sat` on the closed, false assertion (`#P2b-90`), so the model was
    ///   printed;
    /// * an assertion the value reader leaves OPEN and the structural
    ///   evaluator (`array_completion_certify::evaluate_closed`) reads false
    ///   goes to a fresh solver as before: its `unsat` withholds the model
    ///   (the verdict stands), its `sat` or `unknown` prints it, since nothing
    ///   then shows the model false (the structural evaluator alone once read
    ///   a correct model false, `gen_dt.py` seed 30093154 `d00074`);
    /// * an assertion that applies a function whose printed table does not
    ///   read back is OPEN, never clean (decision (92)(d), re-fix pass 19):
    ///   it goes to a fresh solver with that function uninterpreted, and the
    ///   model is withheld unless the assertion's negation is refuted there —
    ///   the assertion true under every interpretation of the function —
    ///   while every other assertion is still read as above, so one
    ///   unreadable table no longer switches the net off (a constructor
    ///   named `|(a|` printed unquoted as `(a` let a falsifying model through
    ///   on every build since the net, `round4_pass19_fix_pins`).
    ///
    /// The datatype values are rebuilt after the search from the terms of
    /// the assertions as written, while the search decided the assertions as
    /// encoded (`#P2b-88`), so a candidate can print one value for two terms
    /// the search kept apart; this is where that is caught.  Returns the
    /// verdict the check answers.
    fn withhold_a_falsifying_datatype_model(&mut self, result: SolverResult) -> SolverResult {
        if !self.goal_has_value_sort_symbols() && !self.goal_mentions_value_sort() {
            return result;
        }
        match self.printed_datatype_model_refuted() {
            NetReading::Clean => result,
            NetReading::False(number) => {
                let reason = format!(
                    "model check failed: assertion {number} reads false under the candidate model"
                );
                self.withhold_every_symbol(&reason);
                self.published.unknown_reason = Some(reason);
                self.last_result = Some(SolverResult::Unknown);
                SolverResult::Unknown
            }
            NetReading::Refuted(number) => {
                self.withhold_every_symbol(&format!("assertion {number} is false"));
                result
            }
            NetReading::Unread(number, name) => {
                self.withhold_every_symbol(&format!(
                    "the printed interpretation of {name} does not read back (assertion {number})"
                ));
                result
            }
        }
    }

    /// Withhold every declared constant and function of the model, with
    /// `reason` as the error `(get-model)` answers.
    fn withhold_every_symbol(&mut self, reason: &str) {
        self.published.uncertified = Some(format!("model not certified: {reason}"));
        self.published.uncertified_arrays =
            self.declared_consts.iter().map(|decl| decl.term).collect();
        self.published.uncertified_functions = self
            .declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| d.name.clone())
            .collect();
    }

    /// The first assertion the printed model makes false, read as
    /// [`Self::withhold_a_falsifying_datatype_model`] documents: by the exact
    /// value reader (`printed_eval::ground_values`, interning nothing) —
    /// before decision (79)(a) the net only saw what the term builder folded
    /// while the model was substituted in, and `gen_dt.py` seed 30093154
    /// `d00239` published its first, falsifying model while its second and
    /// third were withheld — and, where that reader is open, by the
    /// structural evaluator confirmed by a fresh solver under the hypothesis
    /// that distinct `@uc_` witnesses are distinct elements (what the printed
    /// model says).  An assertion whose closing leaves a symbol other than a
    /// witness reads open.  The structural evaluator and the confirming
    /// solver run exactly where they ran before decision (85): both intern
    /// terms, and a later check's search reads the term arena.
    fn printed_datatype_model_refuted(&mut self) -> NetReading {
        let Some((printed, mut parse)) = self.printed_constants() else {
            return NetReading::Clean;
        };
        // A table that does not read back leaves its function uninterpreted
        // in the assertions; every other table is read (decision (92)(d)).
        let (funcs, unreadable) = self.printed_functions_readable(&mut parse);
        let witnesses: FxHashSet<TermId> = parse.witnesses.values().copied().collect();
        // The first assertion over an unreadable table that is not true under
        // every interpretation of it: the model is withheld for it only after
        // every assertion was read, so a later one the value reader reads
        // false still makes the check `unknown`.
        let mut unread: Option<NetReading> = None;
        for (number, assertion) in self.certified_goal().into_iter().enumerate() {
            let expanded = self.expand_printed_functions(assertion, &funcs);
            let closed = self.terms.substitute(expanded, &printed);
            let closed = self.terms.fold_constructor_accessors(closed);
            let free = self.terms.free_vars_including_patterns(closed);
            if !unreadable.is_empty() && self.applies_one_of(closed, &unreadable) {
                if unread.is_none() && !self.holds_under_every_reading(closed, &free, &parse) {
                    let name = self.first_applied_of(closed, &unreadable);
                    unread = Some(NetReading::Unread(number + 1, name));
                }
                continue;
            }
            if !free.iter().all(|var| witnesses.contains(var)) {
                continue;
            }
            let structural = crate::solver::array_completion_certify::evaluate_closed(
                &self.solver,
                closed,
                &mut self.terms,
            );
            let exact = self.ground_truth(closed, &witnesses);
            if exact == Some(false) {
                return NetReading::False(number + 1);
            }
            if structural.or(exact) != Some(false) {
                continue;
            }
            let mut confirm = crate::solver::Solver::new();
            confirm.set_conflict_limit(DATATYPE_NET_CONFLICTS);
            confirm.set_logic("ALL");
            // The witness hypotheses only where the assertion reads a
            // witness: building them interns a term, and a later check's
            // search reads the term arena (`printed_eval::ground_values`).
            if !free.is_empty() {
                for hypothesis in self.witness_hypotheses(&parse.witnesses) {
                    confirm.assert(hypothesis, &mut self.terms);
                }
            }
            confirm.assert(closed, &mut self.terms);
            if confirm.check(&mut self.terms) == SolverResult::Unsat {
                return NetReading::Refuted(number + 1);
            }
        }
        unread.unwrap_or(NetReading::Clean)
    }

    /// Whether `closed` — an assertion with the printed model substituted in
    /// and some function left uninterpreted because its printed table does
    /// not read back — holds under EVERY interpretation of what it leaves
    /// open: a fresh solver refutes its negation (decision (92)(d)).  Any
    /// other answer, `unknown` included, is `false`.
    fn holds_under_every_reading(
        &mut self,
        closed: TermId,
        free: &[TermId],
        parse: &super::printed_check::ParseBack,
    ) -> bool {
        let mut confirm = crate::solver::Solver::new();
        confirm.set_conflict_limit(DATATYPE_NET_CONFLICTS);
        confirm.set_logic("ALL");
        if free
            .iter()
            .any(|var| parse.witnesses.values().any(|w| w == var))
        {
            for hypothesis in self.witness_hypotheses(&parse.witnesses) {
                confirm.assert(hypothesis, &mut self.terms);
            }
        }
        let negated = self.terms.mk_not(closed);
        confirm.assert(negated, &mut self.terms);
        confirm.check(&mut self.terms) == SolverResult::Unsat
    }

    /// The first function named in `names` that `term` applies, by name
    /// order (for the error `(get-model)` answers).
    fn first_applied_of(&self, term: TermId, names: &FxHashSet<String>) -> String {
        let mut sorted: Vec<&String> = names.iter().collect();
        sorted.sort();
        for name in sorted {
            let single: FxHashSet<String> = std::iter::once(name.clone()).collect();
            if self.applies_one_of(term, &single) {
                return name.clone();
            }
        }
        String::new()
    }

    /// Whether an assertion in scope has a sub-term whose sort contains a
    /// datatype, an enumeration or an uninterpreted sort — the net's trigger
    /// for a goal that declares no symbol of one (a ground formula over
    /// constructor literals, recheck 17's `c318_b`).
    fn goal_mentions_value_sort(&self) -> bool {
        let mut stack: Vec<TermId> = self.certified_goal();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut sorts: FxHashMap<SortId, bool> = FxHashMap::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            let valued = *sorts
                .entry(data.sort)
                .or_insert_with(|| self.sort_contains_value_sort(data.sort));
            if valued {
                return true;
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        false
    }

    /// Whether a declared constant's sort, or a declared function's argument
    /// or range sort, CONTAINS a datatype (an enumeration is one) or an
    /// uninterpreted sort: is one, or is an array whose index or element sort
    /// contains one (decision (79)(a)).
    fn goal_has_value_sort_symbols(&self) -> bool {
        self.declared_consts
            .iter()
            .any(|decl| self.sort_contains_value_sort(decl.sort))
            || self.declared_funs.iter().any(|d| {
                !d.interpreted
                    && (self.sort_contains_value_sort(d.ret_sort)
                        || d.arg_sorts
                            .iter()
                            .any(|&s| self.sort_contains_value_sort(s)))
            })
    }

    /// Whether `sort` is, or is an array (at any depth) over, a datatype or
    /// an uninterpreted sort.
    fn sort_contains_value_sort(&self, sort: SortId) -> bool {
        let mut pending: Vec<SortId> = vec![sort];
        let mut seen: FxHashSet<SortId> = FxHashSet::default();
        while let Some(current) = pending.pop() {
            if !seen.insert(current) {
                continue;
            }
            match self.terms.sorts.get(current).map(|s| &s.kind) {
                Some(SortKind::Datatype(_) | SortKind::Uninterpreted(_)) => return true,
                Some(SortKind::Array { domain, range }) => {
                    pending.push(*domain);
                    pending.push(*range);
                }
                _ => {}
            }
        }
        false
    }

    /// Recompute the published model from the solver's: its model with the
    /// fresh values of `#P2b-71` added.  `true` when a declared constant the
    /// solver's model gave a value is printed with a different one.
    fn refresh_published_model(&mut self) {
        self.published.model = None;
        let Some(model) = self.solver.model().cloned() else {
            return;
        };
        let minted = self.mint_fresh_values(&model);
        if !minted.is_empty() {
            let mut published = model;
            for (term, value) in minted {
                published.set(term, value);
            }
            self.published.model = Some(published);
        }
    }

    /// Whether `term` applies one of the functions named in `names`.
    fn applies_one_of(&self, term: TermId, names: &FxHashSet<String>) -> bool {
        let mut stack: Vec<TermId> = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if let TermKind::Apply { func, .. } = &data.kind
                && names.contains(self.terms.resolve_str(*func))
            {
                return true;
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        false
    }

    /// The model every renderer reads: the solver's, with the fresh values of
    /// `#P2b-71` added when any were minted.
    pub(in crate::context) fn published_model(&self) -> Option<&crate::solver::Model> {
        self.published
            .model
            .as_ref()
            .or_else(|| self.solver.model())
    }

    /// `model` with the fresh values of `#P2b-71` added (for rendering a
    /// model other than the published one, as decision (48) renders the
    /// candidate a completion replaced).
    pub(in crate::context) fn with_fresh_values(
        &mut self,
        model: &crate::solver::Model,
    ) -> crate::solver::Model {
        let minted = self.mint_fresh_values(model);
        let mut out = model.clone();
        for (term, value) in minted {
            out.set(term, value);
        }
        out
    }

    /// The error `(get-model)` answers instead of a model that failed its
    /// certificate.
    pub(in crate::context) fn uncertified_model_error(&self) -> Option<String> {
        self.published
            .uncertified
            .as_ref()
            .map(|reason| format!("(error \"{reason}\")"))
    }

    /// The same error for a `(get-value)` whose terms read an array of a
    /// model that failed its certificate; `None` when none does.
    pub(in crate::context) fn uncertified_read_error(&self, terms: &[TermId]) -> Option<String> {
        let reason = self.published.uncertified.as_ref()?;
        let reads = terms.iter().any(|&term| {
            self.terms
                .free_vars_including_patterns(term)
                .into_iter()
                .any(|var| self.published.uncertified_arrays.contains(&var))
                || (!self.published.uncertified_functions.is_empty()
                    && self.applies_one_of(term, &self.published.uncertified_functions))
        });
        reads.then(|| format!("(error \"{reason}\")"))
    }

    /// Render every declared constant and function as `(get-model)` prints
    /// it, parse the values and tables back, and certify them against the
    /// asserted goal; record the error when they fail.
    ///
    /// An uninterpreted-sort witness `@uc_S_n` is no SMT-LIB literal: it is
    /// read back as a constant of `S` of its own, and the certificate is taken
    /// under the hypothesis that distinct witnesses of one sort are distinct
    /// elements — which is what the printed model says.  Where the printed
    /// model fails with the fresh values of `#P2b-71` and holds without them,
    /// the solver's own model is printed (decision (64)); where it fails
    /// either way, it is completed around what it got right when its arrays
    /// allow, and withheld otherwise.
    fn certify_published_model(&mut self) {
        let Err(reason) = self.certify_printed_goal() else {
            return;
        };
        if let Some(fresh) = self.published.model.take() {
            if self.certify_printed_goal().is_ok() {
                return;
            }
            self.published.model = Some(fresh);
        }
        // A table whose else value breaks a universal the search satisfied
        // through another interpretation: another else value, certified.
        if self.repair_else_values() {
            return;
        }
        // Complete it around what it got right, where the groups allow
        // (`array_completion_certify::printed`); the installed model is
        // certified and printed as installed.
        if let Some((printed, mut parse)) = self.printed_constants()
            && let Ok(funcs) = self.printed_functions(&mut parse)
        {
            let goal = self.expanded_goal(&funcs);
            let hypotheses = self.witness_hypotheses(&parse.witnesses);
            if self
                .solver
                .complete_around_printed(&goal, &printed, &hypotheses, &mut self.terms)
            {
                self.refresh_published_model();
                if self.certify_printed_goal().is_ok() {
                    return;
                }
            }
        }
        self.withhold_published_model(reason);
    }

    /// The else value a repair chose for `name`'s printed table, if any.
    pub(in crate::context) fn else_override(&self, name: &str) -> Option<String> {
        self.published.else_overrides.get(name).cloned()
    }

    /// Where the printed model fails only because a function table's else
    /// value — the most common entry value — breaks a universal the search
    /// satisfied through another interpretation (the projection of the
    /// relevant set, `mbqi::model_certify`'s searched default), try each
    /// table's other entry values as its else value, one function at a time,
    /// and keep the first the certificate passes.  `true` when one was kept (it is what `(get-model)` prints).
    /// Sound for the reason every certified model is: the certificate
    /// checks exactly the tables that are printed.
    fn repair_else_values(&mut self) -> bool {
        let Some(model) = self.published_model().cloned() else {
            return false;
        };
        let class_values = self.build_class_values(&model);
        let names: Vec<String> = self
            .declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| d.name.clone())
            .collect();
        let mut alternatives: Vec<(String, Vec<String>)> = Vec::new();
        for name in names {
            let Some(((entries, else_value, _), _)) =
                self.func_interp_reading(&name, &model, &class_values, false)
            else {
                continue;
            };
            // The table's own entry values first, then the literals the goal
            // spells of the function's range sort (as `mbqi::model_certify`
            // searches its defaults).
            let mut values: Vec<String> = Vec::new();
            for (_, value) in entries {
                if value != else_value && !values.contains(&value) {
                    values.push(value);
                }
            }
            for literal in self.goal_literals_of_range(&name) {
                if literal != else_value && !values.contains(&literal) {
                    values.push(literal);
                }
            }
            if !values.is_empty() {
                alternatives.push((name, values));
            }
        }
        let mut attempts = 0usize;
        for (name, values) in &alternatives {
            for value in values {
                if attempts >= MAX_ELSE_REPAIRS {
                    break;
                }
                attempts += 1;
                self.published.else_overrides = FxHashMap::default();
                self.published
                    .else_overrides
                    .insert(name.clone(), value.clone());
                if self.certify_printed_goal().is_ok() {
                    return true;
                }
            }
        }
        self.published.else_overrides = FxHashMap::default();
        false
    }

    /// Every literal of the asserted goal whose sort is `name`'s range sort,
    /// as `(get-model)` would print it, in term-id order.
    fn goal_literals_of_range(&self, name: &str) -> Vec<String> {
        let Some(range) = self
            .declared_funs
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.ret_sort)
        else {
            return Vec::new();
        };
        let mut found: Vec<TermId> = Vec::new();
        let mut stack: Vec<TermId> = self.certified_goal();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if data.sort == range
                && matches!(
                    data.kind,
                    TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. }
                        | TermKind::True
                        | TermKind::False
                )
            {
                found.push(current);
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        found.sort_unstable_by_key(|t| t.raw());
        found.dedup();
        found.into_iter().map(|t| self.format_value(t)).collect()
    }

    /// The printed model — constants and function tables — certified against
    /// the asserted goal.
    fn certify_printed_goal(&mut self) -> Result<(), String> {
        let Some((printed, mut parse)) = self.printed_constants() else {
            return Ok(());
        };
        let funcs = self.printed_functions(&mut parse)?;
        let goal = self.expanded_goal(&funcs);
        let originals = self.certified_goal();
        let hypotheses = self.witness_hypotheses(&parse.witnesses);
        let verdict = self.solver.certify_printed_model(
            &goal,
            &originals,
            &printed,
            &hypotheses,
            &mut self.terms,
        );
        match verdict {
            Err(reason) if !reason.contains(" is false at ") => {
                // The certificate's queries could not decide the goal: a
                // second, query-free certificate evaluates the printed model
                // exactly (`printed_eval`), for the whole goal and then
                // assertion by assertion — each one shown true by evaluation
                // or by the queries on its own.  The model is fixed, so the
                // assertions are independent statements about it.
                if self.certified_by_evaluation(&goal, &printed) {
                    return Ok(());
                }
                for (&assertion, &original) in goal.iter().zip(&originals) {
                    if self.assertion_certified_by_evaluation(assertion, &printed)
                        || self
                            .solver
                            .certify_printed_model(
                                &[assertion],
                                &[original],
                                &printed,
                                &hypotheses,
                                &mut self.terms,
                            )
                            .is_ok()
                    {
                        continue;
                    }
                    return Err(reason);
                }
                Ok(())
            }
            other => other,
        }
    }

    /// The asserted goal with every printed function replaced by its table.
    fn expanded_goal(&mut self, funcs: &super::printed_check::PrintedFunctions) -> Vec<TermId> {
        self.certified_goal()
            .into_iter()
            .map(|assertion| self.expand_printed_functions(assertion, funcs))
            .collect()
    }

    /// Withhold the model: `(get-model)` answers the error, and so does a
    /// `(get-value)` that reads an uncertified symbol — every array of an
    /// array goal (a term reading none still answers), every declared
    /// constant and function of any other goal.
    fn withhold_published_model(&mut self, reason: String) {
        self.published.uncertified = Some(format!("model not certified: {reason}"));
        let arrays: FxHashSet<TermId> = self
            .declared_consts
            .iter()
            .filter(|decl| {
                matches!(
                    self.terms.sorts.get(decl.sort).map(|s| &s.kind),
                    Some(SortKind::Array { .. })
                )
            })
            .map(|decl| decl.term)
            .collect();
        if arrays.is_empty() {
            self.published.uncertified_arrays =
                self.declared_consts.iter().map(|decl| decl.term).collect();
            self.published.uncertified_functions = self
                .declared_funs
                .iter()
                .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
                .map(|d| d.name.clone())
                .collect();
        } else {
            self.published.uncertified_arrays = arrays;
        }
    }

    /// The uninterpreted sorts a witness of the printed model can belong to,
    /// by the name `@uc_<name>_<n>` spells: each declared constant's sort and,
    /// for an array, its index and element sorts.
    pub(super) fn witness_sorts(&self) -> FxHashMap<String, SortId> {
        let mut out: FxHashMap<String, SortId> = FxHashMap::default();
        for root in self.declared_value_roots() {
            // Through array index and element sorts at any depth (an array
            // of arrays into `U`, recheck 17's `b05`).
            let mut pending: Vec<SortId> = vec![root];
            let mut seen: FxHashSet<SortId> = FxHashSet::default();
            while let Some(sort) = pending.pop() {
                if !seen.insert(sort) {
                    continue;
                }
                if let Some(SortKind::Array { domain, range }) =
                    self.terms.sorts.get(sort).map(|s| &s.kind)
                {
                    pending.push(*domain);
                    pending.push(*range);
                }
                if self.is_uninterpreted_sort(sort) {
                    out.insert(self.format_sort_name(sort), sort);
                }
            }
        }
        out
    }

    /// `value` with every witness symbol (`@uc_S_n`, spelled with
    /// [`WITNESS_PLACEHOLDER`]) the lenient term parser read as a fresh
    /// Boolean replaced by one constant of sort `S` per witness.
    pub(super) fn resolve_witnesses(
        &mut self,
        value: TermId,
        sorts: &FxHashMap<String, SortId>,
        witnesses: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
        for var in self.terms.free_vars_including_patterns(value) {
            if let Some(&witness) = witnesses.get(&var) {
                map.insert(var, witness);
                continue;
            }
            let Some(TermKind::Var(name)) = self.terms.get(var).map(|t| t.kind.clone()) else {
                continue;
            };
            let text = self.terms.resolve_str(name).to_string();
            let Some(rest) = text.strip_prefix(WITNESS_PLACEHOLDER) else {
                continue;
            };
            let Some((sort_name, index)) = rest.rsplit_once('_') else {
                continue;
            };
            if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let Some(&sort) = sorts.get(sort_name) else {
                continue;
            };
            let witness = self
                .terms
                .mk_var(&oxiz_core::smtlib::reserved_name("ucw", rest), sort);
            witnesses.insert(var, witness);
            map.insert(var, witness);
        }
        if map.is_empty() {
            value
        } else {
            self.terms.substitute(value, &map)
        }
    }

    /// Every datatype constructor a printed value can spell, by name, with
    /// its datatype's sort: the datatypes reachable from each declared
    /// constant's sort through array index / element sorts and constructor
    /// fields.  The term parser knows no constructor outside the script it
    /// parsed, so a printed `red` or `(cons 3 nil)` comes back as a fresh
    /// symbol / an uninterpreted application and is rebuilt here.
    pub(super) fn value_constructors(&self) -> FxHashMap<String, SortId> {
        let mut out: FxHashMap<String, SortId> = FxHashMap::default();
        let mut pending: Vec<SortId> = self.declared_value_roots();
        let mut seen: FxHashSet<SortId> = FxHashSet::default();
        while let Some(sort) = pending.pop() {
            if !seen.insert(sort) || seen.len() > 256 {
                continue;
            }
            match self.terms.sorts.get(sort).map(|s| s.kind.clone()) {
                Some(SortKind::Array { domain, range }) => {
                    pending.push(domain);
                    pending.push(range);
                }
                Some(SortKind::Datatype(_)) => {
                    let Some(name) = self.terms.sorts.datatype_name(sort) else {
                        continue;
                    };
                    let Some(def) = self.terms.sorts.get_datatype(name) else {
                        continue;
                    };
                    for constructor in &def.constructors {
                        out.insert(self.terms.resolve_str(constructor.name).to_string(), sort);
                        pending.extend(constructor.selectors.iter().map(|&(_, field)| field));
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// `value` with every constructor the parser did not know rebuilt as the
    /// constructor term (see [`Context::value_constructors`]).
    pub(super) fn resolve_constructors(
        &mut self,
        value: TermId,
        constructors: &FxHashMap<String, SortId>,
    ) -> TermId {
        if constructors.is_empty() {
            return value;
        }
        let mut done: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut stack: Vec<(TermId, bool)> = vec![(value, false)];
        while let Some((term, expanded)) = stack.pop() {
            if done.contains_key(&term) {
                continue;
            }
            let Some(kind) = self.terms.get(term).map(|t| t.kind.clone()) else {
                done.insert(term, term);
                continue;
            };
            if !expanded {
                stack.push((term, true));
                let children: Vec<TermId> = oxiz_core::ast::traversal::get_children(&kind)
                    .into_iter()
                    .collect();
                stack.extend(children.into_iter().map(|child| (child, false)));
                continue;
            }
            let get = |child: TermId, done: &FxHashMap<TermId, TermId>| {
                done.get(&child).copied().unwrap_or(child)
            };
            let rebuilt = match kind {
                TermKind::Var(name) => {
                    let name = self.terms.resolve_str(name).to_string();
                    match constructors.get(&name) {
                        Some(&sort) => self.terms.mk_dt_constructor(&name, [], sort),
                        None => term,
                    }
                }
                TermKind::Apply { func, args } => {
                    let name = self.terms.resolve_str(func).to_string();
                    let new_args: Vec<TermId> = args.iter().map(|&a| get(a, &done)).collect();
                    if let Some(&sort) = constructors.get(&name) {
                        self.terms.mk_dt_constructor(&name, new_args, sort)
                    } else if new_args.iter().zip(args.iter()).any(|(n, o)| n != o) {
                        let sort = self.terms.get(term).map(|t| t.sort);
                        match sort {
                            Some(sort) => self.terms.mk_apply(&name, new_args, sort),
                            None => term,
                        }
                    } else {
                        term
                    }
                }
                TermKind::DtConstructor { constructor, args } => {
                    let name = self.terms.resolve_str(constructor).to_string();
                    let new_args: Vec<TermId> = args.iter().map(|&a| get(a, &done)).collect();
                    let sort = self.terms.get(term).map(|t| t.sort);
                    match sort {
                        Some(sort) if new_args.iter().zip(args.iter()).any(|(n, o)| n != o) => {
                            self.terms.mk_dt_constructor(&name, new_args, sort)
                        }
                        _ => term,
                    }
                }
                TermKind::Store(array, index, stored) => {
                    let (array, index, stored) =
                        (get(array, &done), get(index, &done), get(stored, &done));
                    self.terms.mk_store(array, index, stored)
                }
                _ => term,
            };
            done.insert(term, rebuilt);
        }
        done.get(&value).copied().unwrap_or(value)
    }

    /// The sorts a printed value can be read at: every declared constant's,
    /// and every declared function's range and argument sorts (a table of a
    /// function into `(Array Int C)` spells `red`, a witness of a function
    /// over `U` spells `@uc_U_n`; before re-fix pass 18 neither read back, so
    /// the table was open to the net, recheck-18 battery `n16`).
    fn declared_value_roots(&self) -> Vec<SortId> {
        let mut roots: Vec<SortId> = self.declared_consts.iter().map(|d| d.sort).collect();
        for decl in self.declared_funs.iter().filter(|d| !d.interpreted) {
            roots.push(decl.ret_sort);
            roots.extend(decl.arg_sorts.iter().copied());
        }
        roots
    }

    /// `(distinct w₀ w₁ …)` over the witnesses of each sort.
    fn witness_hypotheses(&mut self, witnesses: &FxHashMap<TermId, TermId>) -> Vec<TermId> {
        let mut by_sort: FxHashMap<SortId, Vec<TermId>> = FxHashMap::default();
        for &witness in witnesses.values() {
            if let Some(sort) = self.terms.get(witness).map(|t| t.sort) {
                by_sort.entry(sort).or_default().push(witness);
            }
        }
        let mut groups: Vec<Vec<TermId>> = by_sort.into_values().collect();
        for group in &mut groups {
            group.sort_unstable_by_key(|term| term.raw());
            group.dedup();
        }
        groups.sort_unstable_by_key(|group| group.first().map(|term| term.raw()));
        groups
            .into_iter()
            .filter(|group| group.len() > 1)
            .map(|group| self.terms.mk_distinct(group))
            .collect()
    }
}
