//! Answering `(get-value (t1 … tn))` — the second of the two commands that
//! describe the model, and the one that has to agree with `(get-model)`
//! term for term (`#P2b-35`).
//!
//! Split out of `model_fmt` when that file reached its size limit; the code is
//! unchanged by the move.  What lives here is everything the query path needs
//! and the printing path does not: the completion substitution (unconstrained
//! constants, datatype constructor values, folded selectors), the readings
//! that answer a term the structural evaluator declines to fold (an array read
//! resolved through the class rendering, an application resolved through the
//! printed interpretation), and the response formatting itself.
//!
//! The invariant the whole module exists for: every answer here is the answer
//! `(get-model)` would give for the same term in the same model.  Where a
//! reading is duplicated rather than shared — `Context::array_class_read` is
//! the read-side twin of the array renderer — the two carry cross-references
//! and `tests/model_one_reading.rs` pins their agreement.

use super::*;

/// How deep an array-sorted `ite` chain [`Context::resolve_array_branch`] will
/// follow before giving up.  Terms are a DAG, not a tree, so this is a
/// belt-and-braces bound rather than a real limit: no well-formed script
/// nests array `ite`s anywhere near this deep.
const MAX_ITE_BRANCH_DEPTH: usize = 256;

impl Context {
    /// A ground *term* carrying the same default value that
    /// [`Context::default_value`] renders as a string, or `None` for sorts with
    /// no constructible ground witness (uninterpreted and array sorts, sort
    /// parameters, and ill-founded datatypes).
    ///
    /// Used to complete the model before a `(get-value ...)` evaluation, so a
    /// query over an unconstrained constant — including inside a compound term
    /// such as `(+ x 1)` — reduces to a real value instead of echoing itself.
    ///
    /// Delegates to the solver's
    /// [`ground_default_term`](crate::solver::model_builder::ground_default_term),
    /// which is also what fills in an unconstrained *field* of a reconstructed
    /// datatype value — one definition of "the default of this sort" rather
    /// than one per caller.
    fn default_value_term(&mut self, sort: SortId) -> Option<TermId> {
        crate::solver::model_builder::ground_default_term(&mut self.terms, sort)
    }

    /// The value string [`Context::get_model`] reports for `term`, when `term`
    /// is a declared constant that the model left unassigned.
    ///
    /// `(get-value ...)` and `(get-model)` must never disagree about the same
    /// constant, and `get_model`'s uninterpreted-sort witnesses (`@uc_S_n`) are
    /// numbered across the whole declaration list — so the answer is read back
    /// out of `get_model` itself rather than recomputed.
    fn unassigned_const_value(&self, term: TermId, model: &crate::solver::Model) -> Option<String> {
        let index = self.declared_consts.iter().position(|d| d.term == term)?;
        // A constant the model *did* assign keeps the ordinary evaluation path.
        if model.get(term).is_some() {
            return None;
        }
        let (_, _, value) = self.get_model()?.into_iter().nth(index)?;
        Some(value)
    }

    /// Answer a `(get-value (t1 .. tn))` request.
    ///
    /// `keys` carries the source spelling of each term, which is what the
    /// response pairs with the value; see the key selection below.
    ///
    /// SMT-LIB 2.6 §4.1.1: the command is available only in `sat` mode, so a
    /// missing/superseded check result is reported as an error rather than
    /// answered from stale state.  Each term is evaluated in the current model,
    /// which is first *completed* with the sort defaults `get_model` reports for
    /// unconstrained declared constants — otherwise `Model::eval` returns an
    /// unassigned constant unchanged and `(get-value (x))` answered `((x x))`,
    /// echoing the term instead of producing a value.
    ///
    /// A compound term is folded by the solver's structural evaluator first
    /// (`Solver::model_value_in`, `#P2b-26`): `Model::eval` knows the Boolean
    /// connectives and integer arithmetic only, so `(bvadd v #x01)`,
    /// `(bvult v #x81)`, `(select (store arr i #x05) i)`, a `define-fun`
    /// name standing for any of them — the parser inlines the body — and an
    /// application `(f b)` whose value lives on a congruent `(f a)` all
    /// echoed their body back.  The substitution path below is kept for what
    /// the evaluator declines to fold (a term over a defaulted constant, a
    /// strict comparison at its boundary), where echoing is the honest
    /// non-answer.
    pub(in crate::context) fn format_get_value(
        &mut self,
        terms: &[TermId],
        keys: &[String],
    ) -> String {
        const NO_MODEL: &str = "(error \"No model available\")";
        if self.last_result != Some(SolverResult::Sat) {
            return NO_MODEL.to_string();
        }
        // Owned so the evaluation below can borrow `self.terms` mutably; see
        // `get_model` for why an empty assertion stack — or a populated
        // algebraic side-channel — yields an empty model rather than an error.
        let model = match self.solver.model() {
            Some(model) => model.clone(),
            None if self.assertions.is_empty() || !self.solver.nl_algebraic_values().is_empty() => {
                crate::solver::Model::new()
            }
            None => return NO_MODEL.to_string(),
        };

        // Completion substitution: every declared constant with no model entry
        // maps to its sort default.
        //
        // A constant the algebraic side-channel *does* pin is excluded. Its
        // sort default is `0.0`, and substituting that would answer a compound
        // query like `(get-value ((* x x)))` with `0.0` for a goal whose
        // witness is `√2` — a fabricated value, and one contradicting the `2.0`
        // that the very same model implies. Left out of the map the term
        // survives evaluation unreduced and echoes back, which is the same
        // honest non-answer this path already gives for anything else it
        // cannot fold. (A bare `(get-value (x))` never reaches the completion
        // at all: `unassigned_const_value` answers it from `get_model` below,
        // which is where the `root-obj` rendering lives.)
        let unassigned: Vec<(TermId, SortId)> = self
            .declared_consts
            .iter()
            .filter(|d| model.get(d.term).is_none())
            .filter(|d| self.solver.nl_algebraic_value(d.term).is_none())
            .map(|d| (d.term, d.sort))
            .collect();
        let mut completion: crate::prelude::FxHashMap<TermId, TermId> =
            crate::prelude::FxHashMap::default();
        for (term, sort) in unassigned {
            if let Some(value) = self.default_value_term(sort) {
                completion.insert(term, value);
            }
        }
        // A datatype constant the model *did* assign goes into the
        // substitution too (`#P2b-35`).  `Model::eval` stops at a selector —
        // it has no arm for one — so `(get-value ((snd q)))` echoed `(snd q)`
        // beside a `(get-model)` that printed `q = (mk #b00 #b01)`, two
        // commands describing two different models.  With the constructor
        // value substituted, the datatype rewriter folds the selector to the
        // field the model chose.
        let datatype_values: Vec<(TermId, TermId)> = self
            .declared_consts
            .iter()
            .filter_map(|decl| Some((decl.term, model.get(decl.term)?)))
            .filter(|&(_, value)| {
                matches!(
                    self.terms.get(value).map(|t| &t.kind),
                    Some(TermKind::DtConstructor { .. })
                )
            })
            .collect();
        for (term, value) in datatype_values {
            completion.insert(term, value);
        }
        // …and the selector applications over them are folded here rather
        // than left to the rewriter, which has no datatype registry on this
        // path: `(snd (mk #b00 #b01))` is still a term, and a term is an echo.
        self.fold_dt_selectors(terms, &model, &mut completion);
        // The structural evaluator reads the SAME completed model (`#P2b-35`).
        // Completing only the substitution path left the two readings
        // disagreeing about an unconstrained constant: `(get-value (w))`
        // answered `#x00` from the completion while `(get-value ((bvult w v)))`
        // could not fold at all, because the evaluator saw `w` as a leaf with
        // no entry, and printed `(bvult #x00 v)` — a term, not a value.
        let mut completed_model = model.clone();
        for (&term, &value) in &completion {
            completed_model.set(term, value);
        }

        // The same canonical class → value map `(get-model)` prints from, so
        // the interpretation arm below answers out of the model this query is
        // describing rather than out of a second reading of it.
        let class_values = self.build_class_values(&completed_model);

        let mut values = Vec::with_capacity(terms.len());
        for &term in terms {
            let value_str = if let Some(value) = self.unassigned_const_value(term, &model) {
                // A bare unconstrained constant: report exactly what
                // `(get-model)` reports for it, witnesses included.
                value
            } else if let Some(value) =
                self.array_query_value(term, &completed_model, &class_values)
            {
                // An array-sorted query term (`#P2b-39`).  Before this arm the
                // structural reading below folded `(ite p a b)` to the *term*
                // `a` and printed it, and an array-sorted datatype selector
                // `(arr b)` echoed itself — decision (4) says an array value is
                // a store chain or an `(as const …)` value, never a term, and
                // an echo beside a `(get-model)` that prints the chain is two
                // commands describing two different models.  Placed *above* the
                // structural reading because that reading answers an array
                // `ite` with a term and would shadow this one.
                value
            } else if let Some(value) =
                self.solver
                    .model_value_in(term, &completed_model, &mut self.terms)
            {
                // The structural reading: bit-vector operators, comparisons,
                // read-over-write and congruent applications fold here.
                oxiz_core::smtlib::Printer::new(&self.terms).print_term(value)
            } else if let Some(value) = self.array_read_value(term, &completed_model, &class_values)
            {
                // A read of an array the model describes only through its
                // class (`#P2b-35`): no store to reduce, no published value of
                // its own, so the structural reading above cannot fold it and
                // it echoed — beside a `(get-model)` that prints the very
                // entry being asked for.  Answered from the same class
                // rendering that model prints.
                value
            } else if let Some(value) =
                self.applied_interp_value(term, &completed_model, &class_values)
            {
                // The interpretation `(get-model)` prints for this function
                // (`#P2b-35`).  An application the structural reading cannot
                // fold — no model entry, no congruent application with one —
                // used to echo itself, so `(get-value ((f a)))` answered
                // `(f a)` while `(get-model)` printed `f` as a *total*
                // function with an else-value: two commands describing two
                // different models.  The else-value is the one `(get-model)`
                // committed to, so it is the answer here too.
                value
            } else {
                let completed = if completion.is_empty() {
                    term
                } else {
                    self.terms.substitute(term, &completion)
                };
                // `Model::eval` substitutes and folds the Boolean structure but
                // leaves arithmetic/bit-vector applications of the substituted
                // constants unreduced (`(+ x 1)` → `(+ 0 1)`), so run the
                // rewriter over the result to reach an actual value.
                let value = model.eval(completed, &mut self.terms);
                let value = self.terms.simplify(value);
                oxiz_core::smtlib::Printer::new(&self.terms).print_term(value)
            };
            // The key is the term *as queried* (SMT-LIB 2.6 §4.1.1).  Printing
            // the parsed term instead answered `(get-value ((dbl a)))` with
            // the key `(bvadd a a)`, the inlined `define-fun` body: a reader
            // matching the response against its own query found nothing it
            // asked for (`#P2b-35`).  The source spelling is only missing for
            // a caller that built the query without a parser, and then the
            // parsed term is the best key there is.
            let term_str = match keys.get(values.len()) {
                Some(key) if !key.is_empty() => key.clone(),
                _ => oxiz_core::smtlib::Printer::new(&self.terms).print_term(term),
            };
            values.push(format!("({} {})", term_str, value_str));
        }
        format!("({})", values.join("\n "))
    }

    /// The value the printed model gives the array read `term`, when `term` is
    /// a `select` the structural reading could not fold.
    ///
    /// The index is evaluated first — a compound index term names the position
    /// its *value* names, not the one its leading variable does — and the read
    /// itself is answered by [`Context::array_class_read`], the read-side twin
    /// of the renderer `(get-model)` prints from.
    fn array_read_value(
        &mut self,
        term: TermId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
    ) -> Option<String> {
        let TermKind::Select(array, index) = self.terms.get(term)?.kind else {
            return None;
        };
        let sort = self.terms.get(array)?.sort;
        let Some(SortKind::Array { range, .. }) = self.terms.sorts.get(sort).map(|s| &s.kind)
        else {
            return None;
        };
        let range = *range;
        let index_value = match model.get(index) {
            Some(value) => value,
            None => self.solver.model_value_in(index, model, &mut self.terms)?,
        };
        // Resolve an `ite` array operand through the model's value for its
        // condition before the class is looked up, the way `publish_index_leaves`
        // folds a compound *index*.  Without it the whole `(ite p a b)` term is
        // handed to `array_class_read`, which has no class for a term the
        // congruence closure never interned.
        let array = self.resolve_array_branch(array, model)?;
        let mut visiting: Vec<TermId> = Vec::new();
        // No class for this array term is *not* "the array is the sort
        // default": it is "this reading cannot answer", and the caller then
        // falls back to the substitution path, whose echo a consumer can
        // detect.  Answering `default_value(range)` instead turned a
        // detectable non-answer into a wrong one — `(get-value ((select (ite p
        // a b) #b0)))` answered `#b0`, the *else* branch's value, beside a
        // `(get-model)` in the same run that printed `p = true` and
        // `(select a #b0) = #b1`.  A `(get-value)` answer that contradicts the
        // same run's `(get-model)` is the one failure mode a consumer's model
        // check cannot catch, because it trusts the value.
        //
        // The *background* of the chain is still an answer, though, and it is
        // the one `(get-model)` prints: a read at an index no `store` of the
        // published chain covers takes the chain's `(as const …)` value, not
        // the sort's.  Only when neither reading exists does the query decline.
        if let Some(value) = self.array_class_read(
            array,
            index_value,
            sort,
            model,
            class_values,
            0,
            &mut visiting,
        ) {
            return Some(value);
        }
        let mut visiting: Vec<TermId> = Vec::new();
        if let Some(value) = self.array_class_background(
            array,
            index_value,
            sort,
            model,
            class_values,
            0,
            &mut visiting,
        ) {
            return Some(value);
        }
        // Last, the renderer's *own* last resort, and only where the renderer
        // reaches it: `array_class_parts_inner` prints the sort default as the
        // base of a chain whose class pins some entries but names no
        // background, so at an index that chain does not cover the printed
        // array really is that default.  The gate — "does the renderer
        // describe this array at all" — is the whole fix.  Ungated, "no class
        // for this term" was answered with the sort default too, and
        // `(get-value ((select (ite p a b) #b0)))` answered `#b0`, the *else*
        // branch's value, beside a `(get-model)` in the same run printing
        // `p = true` and `(select a #b0) = #b1`.
        self.array_model_value(array, sort, model, class_values)
            .map(|_| self.default_value(range))
    }

    /// The store chain `(get-model)` would print for an array-sorted query
    /// term, or `None` when the term is not array-sorted or the model does not
    /// describe it.
    ///
    /// Three shapes reach here that `(get-model)` never has to render, because
    /// it only ever prints *declared constants*: an `ite` between two arrays,
    /// an array-sorted datatype field, and a `store` expression.  All three are
    /// resolved to the array term the model means — the `ite` through its
    /// condition's value, the selector through the constructor value
    /// [`Context::fold_dt_selectors`] put in the completed model — and then
    /// handed to the same [`Context::array_model_value`] renderer
    /// `(get-model)` uses, so the two commands cannot describe the same array
    /// differently.
    fn array_query_value(
        &mut self,
        term: TermId,
        model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
    ) -> Option<String> {
        let sort = self.terms.get(term)?.sort;
        if !matches!(
            self.terms.sorts.get(sort).map(|s| &s.kind),
            Some(SortKind::Array { .. })
        ) {
            return None;
        }
        // A folded datatype selector is recorded in the completed model as the
        // field term; anything else stands for itself.
        let resolved = model.get(term).unwrap_or(term);
        let resolved = self.resolve_array_branch(resolved, model)?;
        self.array_model_value(resolved, sort, model, class_values)
    }

    /// Follow an array-sorted `ite` down to the branch the model selects.
    ///
    /// `(ite p a b)` is not an array term the congruence closure interned, so
    /// no class rendering exists for it; the array the model *means* is `a` or
    /// `b`, and which one is decided by the model's value for `p`.  Nested
    /// `ite`s resolve by iterating, bounded by the term count so a malformed
    /// cycle cannot spin.  `None` when the condition has no definite value in
    /// the model, which is the honest non-answer.
    fn resolve_array_branch(
        &mut self,
        array: TermId,
        model: &crate::solver::Model,
    ) -> Option<TermId> {
        let mut current = array;
        for _ in 0..MAX_ITE_BRANCH_DEPTH {
            let TermKind::Ite(cond, then_branch, else_branch) = self.terms.get(current)?.kind
            else {
                return Some(current);
            };
            let value = match model.get(cond) {
                Some(value) => value,
                None => self.solver.model_value_in(cond, model, &mut self.terms)?,
            };
            current = match self.terms.get(value)?.kind {
                TermKind::True => then_branch,
                TermKind::False => else_branch,
                _ => return None,
            };
        }
        None
    }

    /// Map every datatype *selector* application inside `terms` whose argument
    /// the model resolves to a constructor value onto the field that
    /// constructor holds, adding the mapping to `completion` (`#P2b-35`).
    ///
    /// `(get-model)` prints `q = (mk #b00 #b01)`; `(get-value ((snd q)))`
    /// answered `(snd q)` — an echo, and the two commands then describe two
    /// different models.  `Model::eval` has no selector arm, and the core
    /// rewriter's datatype rule needs a registry this path does not build, so
    /// the fold is done here, against the same declaration table
    /// `Context::default_value` reads.
    ///
    /// Nested selectors (`(fst (snd r))`) resolve by iterating: each pass can
    /// only turn an unresolved selector into a value, so `selectors.len()`
    /// passes suffice and the loop cannot spin.
    fn fold_dt_selectors(
        &self,
        terms: &[TermId],
        model: &crate::solver::Model,
        completion: &mut crate::prelude::FxHashMap<TermId, TermId>,
    ) {
        let mut selectors: Vec<(TermId, oxiz_core::interner::Spur, TermId)> = Vec::new();
        let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();
        let mut stack: Vec<TermId> = terms.to_vec();
        let mut children: Vec<TermId> = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if let TermKind::DtSelector { selector, arg } = data.kind {
                selectors.push((current, selector, arg));
            }
            children.clear();
            crate::solver::array_axioms::ground_children(&data.kind, &mut children);
            stack.extend(children.iter().copied());
        }
        for _ in 0..selectors.len() {
            let mut progress = false;
            for &(term, selector, arg) in &selectors {
                if completion.contains_key(&term) {
                    continue;
                }
                let Some(value) = completion.get(&arg).copied().or_else(|| model.get(arg)) else {
                    continue;
                };
                let Some(field) = self.dt_constructor_field(value, selector) else {
                    continue;
                };
                completion.insert(term, field);
                progress = true;
            }
            if !progress {
                break;
            }
        }
    }

    /// The field `selector` names inside the constructor value `value`, or
    /// `None` when `value` is not a constructor application of a datatype
    /// whose constructor has that selector.
    fn dt_constructor_field(
        &self,
        value: TermId,
        selector: oxiz_core::interner::Spur,
    ) -> Option<TermId> {
        let data = self.terms.get(value)?;
        let TermKind::DtConstructor {
            constructor, args, ..
        } = &data.kind
        else {
            return None;
        };
        let name = self.terms.sorts.datatype_name(data.sort)?.to_string();
        let def = self.terms.sorts.get_datatype(&name)?;
        let ctor = def
            .constructors
            .iter()
            .find(|candidate| candidate.name == *constructor)?;
        let index = ctor
            .selectors
            .iter()
            .position(|&(name, _)| name == selector)?;
        args.get(index).copied()
    }

    /// The value the printed interpretation of a declared uninterpreted
    /// function gives `term`, when `term` is an application of one.
    ///
    /// The argument tuple is *evaluated* first, exactly as
    /// [`Context::func_interp_from`] evaluates the tuples it prints, and an
    /// argument tuple matching no entry answers the interpretation's
    /// else-value — the branch the printed `define-fun`'s innermost `ite`
    /// takes.  `None` when `term` is not such an application, when the
    /// function has no interpretation, or when an argument has no value in
    /// this model: an answer would then be a guess, and echoing the term is
    /// the honest non-answer.
    fn applied_interp_value(
        &self,
        term: TermId,
        model: &crate::solver::Model,
        class_values: &class_values::ClassValues,
    ) -> Option<String> {
        let TermKind::Apply { func, args } = &self.terms.get(term)?.kind else {
            return None;
        };
        let name = self.terms.resolve_str(*func).to_string();
        let declared = self
            .declared_funs
            .iter()
            .find(|d| d.name == name && !d.interpreted && !d.arg_sorts.is_empty())?;
        if declared.arg_sorts.len() != args.len() {
            return None;
        }
        let arg_strs: Option<Vec<String>> = args
            .iter()
            .map(|&arg| self.class_value_string(&[arg], model, class_values))
            .collect();
        let arg_strs = arg_strs?;
        let (entries, else_value, _) = self.func_interp_from(&name, model, class_values)?;
        Some(
            entries
                .into_iter()
                .find(|(entry_args, _)| *entry_args == arg_strs)
                .map_or(else_value, |(_, value)| value),
        )
    }
}
