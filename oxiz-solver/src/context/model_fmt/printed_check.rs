//! The printed model read back as an interpretation — constants **and
//! uninterpreted functions** — and checked against what was asserted
//! (re-fix pass 15: decisions (64), (67), (68)).
//!
//! # Why the functions are read back too (decision (68))
//!
//! The published-model certificate of decision (54)(i) interpreted an
//! uninterpreted function only at the ground applications the assertions
//! spell, and only where their arguments were literals the candidate pinned.
//! An application nested in another (`(f (f 7))`) or read at a bound variable
//! had no reading, so the certificate declined and a model that was correct
//! — `f` constantly `7` — was withheld (recheck 14 measured 95 of 132
//! withheld models correct).  The printed `define-fun` of a function *is* its
//! interpretation: the first entry whose arguments match, else the else
//! value.  Every application is replaced by that table, bottom-up and under
//! binders, so what the certificate checks is exactly the printed model.
//!
//! # The scope of the net (decision (67))
//!
//! The net used to run only where the solver's `has_array_ops` flag was set,
//! and it read the solver's assertions — after Skolemisation, destructive
//! equality resolution and vacuous-binder elimination had removed a
//! quantifier.  A scalar or uninterpreted goal (`q01`, `u04`), and an array
//! goal whose only read sits under a binder that never produced an instance
//! (`n02`), published a model falsifying its own universal.  The goal is now
//! whatever the script asserted: any quantifier in an asserted term puts the
//! `sat` under the certificate, whatever the theory.
//!
//! # The quantifier-free check (decisions (64), (68))
//!
//! A quantifier-free `sat` whose printed model the fresh values of `#P2b-71`
//! changed is evaluated once, exactly, under the printed interpretation: if
//! it makes an assertion false, the fresh values are dropped and the solver's
//! own model is printed (recheck 14's `r03`: a fresh value re-keyed a table
//! and lost an entry — c702310 printed a correct model there).

use super::*;
use crate::prelude::{FxHashMap, FxHashSet};

/// One printed function table: its entries in the order the printed `ite`
/// chain tests them, and the else value.
pub(in crate::context) struct PrintedFunction {
    entries: Vec<(Vec<TermId>, TermId)>,
    else_value: TermId,
}

impl PrintedFunction {
    /// The table applied to `args`: the value of the first entry whose
    /// points equal the arguments, else the else value.
    fn apply(&self, args: &[TermId], terms: &mut oxiz_core::ast::TermManager) -> TermId {
        let mut body = self.else_value;
        for (points, value) in self.entries.iter().rev() {
            if points.len() != args.len() {
                continue;
            }
            let conjuncts: Vec<TermId> = points
                .iter()
                .zip(args)
                .map(
                    |(&point, &arg)| match numeric_literal_equality(arg, point, terms) {
                        // `(f 1)` over a `Real` domain meets the printed point
                        // `1.0`: equal as numbers, whatever the literal's kind.
                        Some(equal) => terms.mk_bool(equal),
                        None => terms.mk_eq(arg, point),
                    },
                )
                .collect();
            let guard = terms.mk_and(conjuncts);
            body = terms.mk_ite(guard, *value, body);
        }
        body
    }
}

/// Whether two numeric literals denote the same number (an `Int` and a
/// `Real` literal compared by value); `None` unless both are numeric
/// literals.
fn numeric_literal_equality(
    a: TermId,
    b: TermId,
    terms: &oxiz_core::ast::TermManager,
) -> Option<bool> {
    let value = |term: TermId| -> Option<num_rational::BigRational> {
        match &terms.get(term)?.kind {
            TermKind::IntConst(n) => Some(num_rational::BigRational::from_integer(n.clone())),
            TermKind::RealConst(r) => Some(num_rational::BigRational::new(
                BigInt::from(*r.numer()),
                BigInt::from(*r.denom()),
            )),
            _ => None,
        }
    };
    Some(value(a)? == value(b)?)
}

/// Every printed function table, by the function's name.
pub(in crate::context) type PrintedFunctions = FxHashMap<String, PrintedFunction>;

/// One declared function's printed table, read back
/// (`Context::read_printed_table`).
enum TableReading {
    /// The function prints no table.
    Absent,
    /// The table, parsed back.
    Read(PrintedFunction),
    /// A printed value of the table does not parse back.
    Unreadable,
}

/// The pieces the parse-back of printed values needs, gathered once.
pub(in crate::context) struct ParseBack {
    /// Uninterpreted sorts a witness `@uc_S_n` can belong to, by name.
    pub(in crate::context) sorts: FxHashMap<String, SortId>,
    /// Datatype constructors a printed value can spell, by name.
    pub(in crate::context) constructors: FxHashMap<String, SortId>,
    /// Witness placeholder symbol → the witness constant it stands for.
    pub(in crate::context) witnesses: FxHashMap<TermId, TermId>,
}

impl Context {
    /// Whether some asserted term — as the script wrote it, before any
    /// quantifier was Skolemised, resolved or dropped — contains a
    /// quantifier (decision (67)).  Assumptions of a `check-sat-assuming`
    /// count too.
    pub(in crate::context) fn goal_is_quantified(&self) -> bool {
        let mut stack: Vec<TermId> = self.certified_goal();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if matches!(data.kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
                return true;
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        false
    }

    /// What a published model must satisfy: every assertion in scope and
    /// every assumption of the last check.
    pub(in crate::context) fn certified_goal(&self) -> Vec<TermId> {
        let mut goal = self.assertions.clone();
        goal.extend(self.last_assumptions.iter().copied());
        goal
    }

    /// Parse one printed value back into a term: witnesses become sorted
    /// constants, constructors constructor terms; `None` when the text does
    /// not parse.
    pub(in crate::context) fn parse_printed_value(
        &mut self,
        text: &str,
        parse: &mut ParseBack,
    ) -> Option<TermId> {
        // The parser rightly refuses an `@` symbol (SMT-LIB reserves it), so
        // each witness is spelled with a placeholder first.
        let spelled = text.replace("@uc_", super::published::WITNESS_PLACEHOLDER);
        let value = oxiz_core::smtlib::parse_term(&spelled, &mut self.terms).ok()?;
        let value = self.resolve_witnesses(value, &parse.sorts, &mut parse.witnesses);
        let value = self.resolve_constructors(value, &parse.constructors);
        Some(self.fold_numeral_value(value))
    }

    /// A printed numeric value as the literal it denotes: the printer spells a
    /// non-integral `Real` as `(/ 37 10)` and a negative one as `(/ -21 10)`
    /// or `(- 2.0)`, which parse back as a division / negation of literals —
    /// terms the exact evaluator reads as `Int` division or leaves open.
    /// Anything else is returned unchanged.
    fn fold_numeral_value(&mut self, value: TermId) -> TermId {
        let rational = |term: TermId, terms: &oxiz_core::ast::TermManager| match terms
            .get(term)
            .map(|t| &t.kind)
        {
            Some(TermKind::IntConst(n)) => i64::try_from(n.clone())
                .ok()
                .map(num_rational::Rational64::from_integer),
            Some(TermKind::RealConst(r)) => Some(*r),
            _ => None,
        };
        match self.terms.get(value).map(|t| t.kind.clone()) {
            Some(TermKind::Neg(inner)) => {
                let inner = self.fold_numeral_value(inner);
                match self.terms.get(inner).map(|t| t.kind.clone()) {
                    Some(TermKind::IntConst(n)) => self.terms.mk_int(-n),
                    Some(TermKind::RealConst(r)) => self.terms.mk_real(-r),
                    _ => value,
                }
            }
            Some(TermKind::Div(numerator, denominator)) => {
                let numerator = self.fold_numeral_value(numerator);
                let denominator = self.fold_numeral_value(denominator);
                match (
                    rational(numerator, &self.terms),
                    rational(denominator, &self.terms),
                ) {
                    (Some(n), Some(d)) if d != num_rational::Rational64::from_integer(0) => {
                        self.terms.mk_real(n / d)
                    }
                    _ => value,
                }
            }
            _ => value,
        }
    }

    /// Every declared uninterpreted function's printed table, parsed back.
    /// `Err` names the first function whose table does not parse back into
    /// values; the tables after it are not read (so nothing past it is
    /// interned).
    pub(in crate::context) fn printed_functions(
        &mut self,
        parse: &mut ParseBack,
    ) -> Result<PrintedFunctions, String> {
        let mut out: PrintedFunctions = FxHashMap::default();
        let Some(solver_model) = self.published_model().cloned() else {
            return Ok(out);
        };
        let class_values = self.build_class_values(&solver_model);
        for name in self.uninterpreted_function_names() {
            match self.read_printed_table(&name, &solver_model, &class_values, parse) {
                TableReading::Absent => {}
                TableReading::Read(table) => {
                    out.insert(name, table);
                }
                TableReading::Unreadable => {
                    return Err(format!(
                        "the printed interpretation of {name} does not read back"
                    ));
                }
            }
        }
        Ok(out)
    }

    /// Every declared uninterpreted function's printed table that parses
    /// back into values closed up to witnesses, and the names of those that
    /// do not — the honesty net's reader (decision (92)(d), re-fix pass 19):
    /// one unreadable table must not switch off the reading of every
    /// assertion, so every other table is still read.  A table counts as
    /// unreadable when a printed value does not parse (a constructor named
    /// `|(a|` prints unquoted as `(a`) or parses into something that is not a
    /// value (a sort named `|W X|` prints its witnesses as `@uc_W X_n`, which
    /// parses by truncation into a free symbol).
    pub(in crate::context) fn printed_functions_readable(
        &mut self,
        parse: &mut ParseBack,
    ) -> (PrintedFunctions, FxHashSet<String>) {
        let mut out: PrintedFunctions = FxHashMap::default();
        let mut unreadable: FxHashSet<String> = FxHashSet::default();
        let Some(solver_model) = self.published_model().cloned() else {
            return (out, unreadable);
        };
        let class_values = self.build_class_values(&solver_model);
        for name in self.uninterpreted_function_names() {
            match self.read_printed_table(&name, &solver_model, &class_values, parse) {
                TableReading::Absent => {}
                TableReading::Read(table) if self.table_is_closed(&table, parse) => {
                    out.insert(name, table);
                }
                TableReading::Read(_) | TableReading::Unreadable => {
                    unreadable.insert(name);
                }
            }
        }
        (out, unreadable)
    }

    /// The names of the declared uninterpreted functions with arguments, in
    /// declaration order.
    fn uninterpreted_function_names(&self) -> Vec<String> {
        self.declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| d.name.clone())
            .collect()
    }

    /// One function's printed table parsed back: its else value first, then
    /// its entries in order, stopping at the first value that does not parse.
    fn read_printed_table(
        &mut self,
        name: &str,
        solver_model: &crate::solver::Model,
        class_values: &super::class_values::ClassValues,
        parse: &mut ParseBack,
    ) -> TableReading {
        // Two entries at one argument tuple with different values (the
        // congruence closure kept two applications apart that the
        // arithmetic valued alike, `#P2b-74`) are read exactly as
        // `(get-model)` prints them — the first entry at a tuple wins —
        // and the certificate judges that function.
        let Some(((entries, else_text, arity), _)) =
            self.func_interp_reading(name, solver_model, class_values, false)
        else {
            return TableReading::Absent;
        };
        let Some(else_value) = self.parse_printed_value(&else_text, parse) else {
            return TableReading::Unreadable;
        };
        let mut table: Vec<(Vec<TermId>, TermId)> = Vec::with_capacity(entries.len());
        for (args, value) in entries {
            if args.len() != arity {
                continue;
            }
            let mut points: Vec<TermId> = Vec::with_capacity(args.len());
            for arg in &args {
                let Some(point) = self.parse_printed_value(arg, parse) else {
                    return TableReading::Unreadable;
                };
                points.push(point);
            }
            let Some(value) = self.parse_printed_value(&value, parse) else {
                return TableReading::Unreadable;
            };
            table.push((points, value));
        }
        TableReading::Read(PrintedFunction {
            entries: table,
            else_value,
        })
    }

    /// Whether every point and value of a parsed table is closed up to the
    /// witnesses the parse resolved.
    fn table_is_closed(&self, table: &PrintedFunction, parse: &ParseBack) -> bool {
        let closed = |term: TermId| {
            self.terms
                .free_vars_including_patterns(term)
                .into_iter()
                .all(|var| parse.witnesses.values().any(|&witness| witness == var))
        };
        closed(table.else_value)
            && table
                .entries
                .iter()
                .all(|(points, value)| closed(*value) && points.iter().all(|&point| closed(point)))
    }

    /// `term` with every application of a printed function replaced by its
    /// table, bottom-up; under a binder the bound variables stand as fresh
    /// constants while the body is rewritten, so no binder is renamed and no
    /// application is missed.
    pub(in crate::context) fn expand_printed_functions(
        &mut self,
        term: TermId,
        funcs: &PrintedFunctions,
    ) -> TermId {
        let mut binders: u32 = 0;
        self.expand_printed_with(term, funcs, &mut binders)
    }

    /// [`Self::expand_printed_functions`], numbering the fresh constants that
    /// stand for bound variables with `binders` (unique within one rewrite).
    fn expand_printed_with(
        &mut self,
        term: TermId,
        funcs: &PrintedFunctions,
        binders: &mut u32,
    ) -> TermId {
        if funcs.is_empty() {
            return term;
        }
        // The maximal quantifiers first, each rewritten on its own.
        let mut quantifiers: Vec<TermId> = Vec::new();
        let mut applications: Vec<TermId> = Vec::new();
        self.collect_rewrite_sites(term, funcs, &mut quantifiers, &mut applications);
        let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
        for quantifier in quantifiers {
            let rewritten = self.expand_under_binder(quantifier, funcs, binders);
            if rewritten != quantifier {
                map.insert(quantifier, rewritten);
            }
        }
        // Applications outside every binder, innermost first: each one's
        // arguments are rewritten before its table is applied to them.
        for application in applications {
            let Some(TermKind::Apply { func, args }) =
                self.terms.get(application).map(|t| t.kind.clone())
            else {
                continue;
            };
            let name = self.terms.resolve_str(func).to_string();
            let Some(table) = funcs.get(&name) else {
                continue;
            };
            let new_args: Vec<TermId> = args
                .iter()
                .map(|&arg| self.terms.substitute(arg, &map))
                .collect();
            let replacement = table.apply(&new_args, &mut self.terms);
            map.insert(application, replacement);
        }
        if map.is_empty() {
            term
        } else {
            self.terms.substitute(term, &map)
        }
    }

    /// The maximal quantifier nodes of `term` and, outside them, the
    /// applications of printed functions in post-order (arguments first).
    fn collect_rewrite_sites(
        &self,
        term: TermId,
        funcs: &PrintedFunctions,
        quantifiers: &mut Vec<TermId>,
        applications: &mut Vec<TermId>,
    ) {
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<(TermId, bool)> = vec![(term, false)];
        while let Some((current, expanded)) = stack.pop() {
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if expanded {
                if let TermKind::Apply { func, .. } = &data.kind
                    && funcs.contains_key(self.terms.resolve_str(*func))
                {
                    applications.push(current);
                }
                continue;
            }
            if !visited.insert(current) {
                continue;
            }
            if matches!(data.kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
                quantifiers.push(current);
                continue;
            }
            stack.push((current, true));
            for child in oxiz_core::ast::traversal::get_children(&data.kind) {
                stack.push((child, false));
            }
        }
    }

    /// One quantifier with its body rewritten by
    /// [`Self::expand_printed_functions`], the bound variables standing as
    /// fresh constants meanwhile.
    fn expand_under_binder(
        &mut self,
        quantifier: TermId,
        funcs: &PrintedFunctions,
        binders: &mut u32,
    ) -> TermId {
        let Some(kind) = self.terms.get(quantifier).map(|t| t.kind.clone()) else {
            return quantifier;
        };
        let (vars, body, universal) = match kind {
            TermKind::Forall { vars, body, .. } => (vars, body, true),
            TermKind::Exists { vars, body, .. } => (vars, body, false),
            _ => return quantifier,
        };
        let mut to_fresh: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut back: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut names: Vec<(String, SortId)> = Vec::with_capacity(vars.len());
        for &(name, sort) in &vars {
            let text = self.terms.resolve_str(name).to_string();
            let bound = self.terms.mk_var(&text, sort);
            *binders = binders.wrapping_add(1);
            let fresh = self.terms.mk_var(
                &oxiz_core::smtlib::reserved_name("pbv", &binders.to_string()),
                sort,
            );
            to_fresh.insert(bound, fresh);
            back.insert(fresh, bound);
            names.push((text, sort));
        }
        let opened = self.terms.substitute(body, &to_fresh);
        let rewritten = self.expand_printed_with(opened, funcs, binders);
        if rewritten == opened {
            return quantifier;
        }
        let closed = self.terms.substitute(rewritten, &back);
        let binders: Vec<(&str, SortId)> = names.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        if universal {
            self.terms.mk_forall(binders, closed)
        } else {
            self.terms.mk_exists(binders, closed)
        }
    }

    /// The printed model's reading of every declared constant, parsed back:
    /// `(printed, parse state)`.  A constant whose printed value does not
    /// read back into a term closed up to witnesses is left out.
    pub(in crate::context) fn printed_constants(
        &mut self,
    ) -> Option<(FxHashMap<TermId, TermId>, ParseBack)> {
        let rows = self.model_rows()?;
        let decls: Vec<TermId> = self.declared_consts.iter().map(|decl| decl.term).collect();
        let mut parse = ParseBack {
            sorts: self.witness_sorts(),
            constructors: self.value_constructors(),
            witnesses: FxHashMap::default(),
        };
        let mut printed: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (term, (_, _, text)) in decls.into_iter().zip(rows) {
            let Some(value) = self.parse_printed_value(&text, &mut parse) else {
                continue;
            };
            let closed = self
                .terms
                .free_vars_including_patterns(value)
                .into_iter()
                .all(|var| parse.witnesses.values().any(|&witness| witness == var));
            if closed {
                printed.insert(term, value);
            }
        }
        Some((printed, parse))
    }

    /// The quantifier-free check: the first assertion the printed model —
    /// constants and function tables as `(get-model)` prints them — makes
    /// **definitely** false, quoted; `None` when every assertion is true or
    /// cannot be evaluated exactly.
    pub(in crate::context) fn printed_model_falsifies(&mut self) -> Option<String> {
        let (printed, mut parse) = self.printed_constants()?;
        let funcs = match self.printed_functions(&mut parse) {
            Ok(funcs) => funcs,
            Err(reason) => return Some(reason),
        };
        let witnesses: FxHashSet<TermId> = parse.witnesses.values().copied().collect();
        for (number, assertion) in self.certified_goal().into_iter().enumerate() {
            let expanded = self.expand_printed_functions(assertion, &funcs);
            let closed = self.terms.substitute(expanded, &printed);
            let closed = self.terms.fold_constructor_accessors(closed);
            // Where the exact evaluator is open, comparisons, `ite`s and
            // reads over datatype and uninterpreted-sort values are read by
            // value, interning nothing (decision (79)(a),
            // `printed_eval::ground_values`).
            let holds = crate::solver::array_completion_certify::evaluate_closed(
                &self.solver,
                closed,
                &mut self.terms,
            )
            .or_else(|| self.ground_truth(closed, &witnesses));
            if holds == Some(false) {
                return Some(format!("assertion {} is false", number + 1));
            }
        }
        None
    }

    /// The value `(get-model)`'s printed interpretation gives `term`, as
    /// `(get-value)` prints it (decision (69)(9), (10)): the term with every
    /// printed function applied by its table and every declared constant
    /// replaced by its printed value, folded exactly.  So a compound over an
    /// application (`(+ (f j) 3)`), a comparison of two applications and a
    /// read at an applied index fold to a literal, a constructor is its own
    /// value, and a `Real` keeps its decimal point — the same model the
    /// printed one is, term for term.  `None` for an array- or
    /// uninterpreted-sorted term, or one the evaluator cannot fold (the
    /// ordinary readings then answer).
    pub(in crate::context) fn printed_reading_value(
        &mut self,
        term: TermId,
        reading: &PrintedReading,
    ) -> Option<String> {
        let sort = self.terms.get(term)?.sort;
        let kind = self.terms.sorts.get(sort).map(|s| s.kind.clone())?;
        if matches!(kind, SortKind::Array { .. } | SortKind::Uninterpreted(_)) {
            return None;
        }
        let expanded = self.expand_printed_functions(term, &reading.functions);
        let closed = self.terms.substitute(expanded, &reading.constants);
        // `(fst (mk j red))` with `j` printed is the field, not an echo
        // (recheck 15's minor 11).
        let closed = self.terms.fold_constructor_accessors(closed);
        if !self.terms.free_vars_including_patterns(closed).is_empty() {
            return None;
        }
        // A read of a datatype-indexed array is decided by value over the
        // printed store chain (`printed_eval::datatype_reads`).
        let closed = self.fold_datatype_reads(closed);
        if matches!(kind, SortKind::Datatype(_)) {
            return super::array_model::is_constructor_value(closed, &self.terms)
                .then(|| oxiz_core::smtlib::Printer::new(&self.terms).print_term(closed));
        }
        let value = match crate::solver::array_completion_certify::evaluate_closed_value(
            &self.solver,
            closed,
            &mut self.terms,
        ) {
            Some(value) => value,
            None => {
                // A selector over a constructor value (`(fst (mk 0 red))`) is
                // the rewriter's to fold, not the evaluator's.
                let simplified = self.terms.simplify(closed);
                crate::solver::array_completion_certify::evaluate_closed_value(
                    &self.solver,
                    simplified,
                    &mut self.terms,
                )?
            }
        };
        super::render_eval_value(&value, term, &self.terms)
    }

    /// The printed interpretation `(get-value)` reads, parsed back once per
    /// query; `None` when there is no printed model to read.
    pub(in crate::context) fn printed_reading(&mut self) -> Option<PrintedReading> {
        // An algebraic witness prints as a `root-obj` or a rounded rational
        // no literal reads back exactly; its readings stay the side
        // channel's own.
        if !self.solver.nl_algebraic_values().is_empty() {
            return None;
        }
        let (constants, mut parse) = self.printed_constants()?;
        let functions = self.printed_functions(&mut parse).ok()?;
        Some(PrintedReading {
            constants,
            functions,
        })
    }
}

/// `(get-model)`'s printed interpretation, parsed back: what
/// [`Context::printed_reading_value`] folds a `(get-value)` term under.
pub(in crate::context) struct PrintedReading {
    constants: FxHashMap<TermId, TermId>,
    functions: PrintedFunctions,
}
