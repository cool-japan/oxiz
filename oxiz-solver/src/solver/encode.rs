//! Term encoding (Tseitin transformation) for the SMT solver

#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, ToPrimitive, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_sat::{Lit, Var};
use smallvec::SmallVec;

use super::Solver;
use super::trail::TrailOp;
use super::types::{
    ArithConstraintType, Constraint, NamedAssertion, ParsedArithConstraint, Polarity, UnsatCore,
};

impl Solver {
    pub(super) fn get_or_create_var(&mut self, term: TermId) -> Var {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }

        let var = self.sat.new_var();
        self.term_to_var.insert(term, var);
        self.trail.push(TrailOp::VarCreated { var, term });

        while self.var_to_term.len() <= var.index() {
            self.var_to_term.push(TermId::new(0));
        }
        self.var_to_term[var.index()] = term;
        var
    }

    /// Track theory variables in a term for model extraction.
    /// Recursively scans a term to find Int/Real/BV variables and registers them.
    ///
    /// Compound terms that have already been fully traversed are recorded in
    /// `tracked_compound_terms` to avoid redundant O(depth) re-walks when the
    /// same sub-expression appears in multiple parent constraints.
    pub(super) fn track_theory_vars(&mut self, term_id: TermId, manager: &TermManager) {
        let Some(term) = manager.get(term_id) else {
            return;
        };

        match &term.kind {
            TermKind::Var(_) => {
                // Found a variable - check its sort and track appropriately
                let is_int = term.sort == manager.sorts.int_sort;
                let is_real = term.sort == manager.sorts.real_sort;

                if is_int || is_real {
                    if !self.arith_terms.contains(&term_id) {
                        self.arith_terms.insert(term_id);
                        self.trail.push(TrailOp::ArithTermAdded { term: term_id });
                        self.arith.intern(term_id);
                    }
                } else if let Some(sort) = manager.sorts.get(term.sort)
                    && sort.is_bitvec()
                    && !self.bv_terms.contains(&term_id)
                {
                    self.bv_terms.insert(term_id);
                    self.trail.push(TrailOp::BvTermAdded { term: term_id });
                    if let Some(width) = sort.bitvec_width() {
                        self.bv.new_bv(term_id, width);
                    }
                    // Also intern in ArithSolver for BV comparison constraints
                    // (BV comparisons are handled as bounded integer arithmetic)
                    self.arith.intern(term_id);
                }
            }
            // Recursively scan compound terms.
            // Guard: if this compound node was already fully traversed, skip it.
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                if self.tracked_compound_terms.contains(&term_id) {
                    return;
                }
                self.tracked_compound_terms.insert(term_id);
                // Collect args to avoid re-borrowing `self` during iteration
                let args_vec: SmallVec<[TermId; 8]> = args.iter().copied().collect();
                for arg in args_vec {
                    self.track_theory_vars(arg, manager);
                }
            }
            TermKind::Sub(lhs, rhs)
            | TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::BvAdd(lhs, rhs)
            | TermKind::BvSub(lhs, rhs)
            | TermKind::BvMul(lhs, rhs)
            | TermKind::BvAnd(lhs, rhs)
            | TermKind::BvOr(lhs, rhs)
            | TermKind::BvXor(lhs, rhs)
            | TermKind::BvUlt(lhs, rhs)
            | TermKind::BvUle(lhs, rhs)
            | TermKind::BvSlt(lhs, rhs)
            | TermKind::BvSle(lhs, rhs) => {
                if self.tracked_compound_terms.contains(&term_id) {
                    return;
                }
                self.tracked_compound_terms.insert(term_id);
                let (l, r) = (*lhs, *rhs);
                self.track_theory_vars(l, manager);
                self.track_theory_vars(r, manager);
            }
            // BV arithmetic operations (division/remainder)
            // These need the has_bv_arith_ops flag for conflict detection
            TermKind::BvUdiv(lhs, rhs)
            | TermKind::BvSdiv(lhs, rhs)
            | TermKind::BvUrem(lhs, rhs)
            | TermKind::BvSrem(lhs, rhs) => {
                if self.tracked_compound_terms.contains(&term_id) {
                    return;
                }
                self.tracked_compound_terms.insert(term_id);
                self.has_bv_arith_ops = true;
                let (l, r) = (*lhs, *rhs);
                self.track_theory_vars(l, manager);
                self.track_theory_vars(r, manager);
            }
            TermKind::Neg(arg) | TermKind::Not(arg) | TermKind::BvNot(arg) => {
                if self.tracked_compound_terms.contains(&term_id) {
                    return;
                }
                self.tracked_compound_terms.insert(term_id);
                let a = *arg;
                self.track_theory_vars(a, manager);
            }
            TermKind::Ite(cond, then_br, else_br) => {
                if self.tracked_compound_terms.contains(&term_id) {
                    return;
                }
                self.tracked_compound_terms.insert(term_id);
                let (c, t, e) = (*cond, *then_br, *else_br);
                self.track_theory_vars(c, manager);
                self.track_theory_vars(t, manager);
                self.track_theory_vars(e, manager);
            }
            // Uninterpreted function application: if the sort is numeric (Int or
            // Real), treat the whole application as an opaque arithmetic variable.
            // This supports the UFLIA / UFLRA combination: `f(k)` appearing in
            // `(> (f k) 10)` must be tracked so that its model value is extracted
            // and available to the MBQI counterexample generator.
            //
            // We do NOT recurse into the arguments here -- argument terms are
            // arithmetic values passed to an opaque symbol, not arithmetic
            // variables in their own right within this constraint.  (They will be
            // tracked separately when they appear in other constraints.)
            //
            // RESTRICTION: skip Apply terms that have an argument which is itself
            // an Apply term that is already in `arith_terms` (i.e. already has a
            // numeric model value in the arithmetic solver).  When `f(g(a))` is
            // added to arith AND `g(a)` also has an arith model value `v`, the
            // arith solver treats `f(g(a))` as independent from `f(v)`, leading
            // to theory combination conflicts with EUF (which knows via congruence
            // that `f(g(a)) = f(v)`).
            //
            // In contrast, terms like `f(sk(x))` where `sk(x)` is a fresh Skolem
            // constant (NOT in arith_terms) are safe to add because there are no
            // contradictory EUF congruence facts to violate.
            TermKind::Apply { args, .. } => {
                let is_int = term.sort == manager.sorts.int_sort;
                let is_real = term.sort == manager.sorts.real_sort;
                if is_int || is_real {
                    // Check: is any argument a *non-Skolem* Apply term that is
                    // already in arith?  When f(g(a)) is added to arith AND g(a)
                    // has an arith model value v, EUF applies congruence to derive
                    // f(g(a)) = f(v), conflicting with arith's independent
                    // assignment to f(g(a)).  Skolem-generated Apply terms (whose
                    // function names start with "sk!") are fresh constants created
                    // during quantifier Skolemization; EUF has no equality facts
                    // about them so no congruence conflict can arise.
                    let has_conflicting_apply_arg = args.iter().any(|&arg| {
                        manager.get(arg).is_some_and(|a| {
                            if let TermKind::Apply {
                                func,
                                args: inner_args,
                                ..
                            } = &a.kind
                            {
                                if inner_args.is_empty() {
                                    return false;
                                }
                                let fname = manager.resolve_str(*func);
                                let is_skolem = fname.starts_with("sk!");
                                !is_skolem && self.arith_terms.contains(&arg)
                            } else {
                                false
                            }
                        })
                    });
                    if !has_conflicting_apply_arg && !self.arith_terms.contains(&term_id) {
                        self.arith_terms.insert(term_id);
                        self.trail.push(TrailOp::ArithTermAdded { term: term_id });
                        self.arith.intern(term_id);
                    }
                }
            }

            // Array select with numeric sort: `(select a i) : Int/Real` is an
            // opaque arithmetic variable -- the array theory handles equality
            // propagation for equal indices, while arithmetic sees the result as
            // an unconstrained integer/real.  We register it here so that
            // constraints like `(> (select a 0) 7)` are tracked by the arithmetic
            // solver and model values are extracted correctly.
            TermKind::Select(_, _) => {
                let is_int = term.sort == manager.sorts.int_sort;
                let is_real = term.sort == manager.sorts.real_sort;
                if (is_int || is_real) && !self.arith_terms.contains(&term_id) {
                    self.arith_terms.insert(term_id);
                    self.trail.push(TrailOp::ArithTermAdded { term: term_id });
                    self.arith.intern(term_id);
                }
            }

            // Constants and other leaf terms - nothing to track
            _ => {}
        }
    }

    /// Parse an arithmetic comparison and extract linear expression.
    /// Returns: (terms with coefficients, constant, constraint_type).
    ///
    /// Results are cached by `reason` (the comparison term id).
    /// `ParsedArithConstraint` is purely structural — it depends only on the
    /// term graph — so the cache is safe to retain across CDCL backtracks.
    pub(super) fn parse_arith_comparison(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        constraint_type: ArithConstraintType,
        reason: TermId,
        manager: &TermManager,
    ) -> Option<ParsedArithConstraint> {
        // Fast path: return cached result if available.
        if let Some(cached) = self.arith_parse_cache.get(&reason) {
            return cached.clone();
        }

        let mut terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
        let mut constant = Rational64::zero();

        // Parse LHS (add positive coefficients)
        let lhs_ok =
            self.extract_linear_terms(lhs, Rational64::one(), &mut terms, &mut constant, manager);
        if lhs_ok.is_none() {
            self.arith_parse_cache.insert(reason, None);
            return None;
        }

        // Parse RHS (subtract, so coefficients are negated)
        // For lhs OP rhs, we want lhs - rhs OP 0
        let rhs_ok =
            self.extract_linear_terms(rhs, -Rational64::one(), &mut terms, &mut constant, manager);
        if rhs_ok.is_none() {
            self.arith_parse_cache.insert(reason, None);
            return None;
        }

        // Combine like terms
        let mut combined: FxHashMap<TermId, Rational64> = FxHashMap::default();
        for (term, coef) in terms {
            *combined.entry(term).or_insert(Rational64::zero()) += coef;
        }

        // Remove zero coefficients
        let final_terms: SmallVec<[(TermId, Rational64); 4]> =
            combined.into_iter().filter(|(_, c)| !c.is_zero()).collect();

        let result = ParsedArithConstraint {
            terms: final_terms,
            constant: -constant, // Move constant to RHS
            constraint_type,
            reason_term: reason,
        };

        self.arith_parse_cache.insert(reason, Some(result.clone()));
        Some(result)
    }

    /// Extract linear terms recursively from an arithmetic expression
    /// Returns None if the term is not linear
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn extract_linear_terms(
        &self,
        term_id: TermId,
        scale: Rational64,
        terms: &mut SmallVec<[(TermId, Rational64); 4]>,
        constant: &mut Rational64,
        manager: &TermManager,
    ) -> Option<()> {
        let term = manager.get(term_id)?;

        match &term.kind {
            // Integer constant
            TermKind::IntConst(n) => {
                if let Some(val) = n.to_i64() {
                    *constant += scale * Rational64::from_integer(val);
                    Some(())
                } else {
                    // BigInt too large, skip for now
                    None
                }
            }

            // Rational constant
            TermKind::RealConst(r) => {
                *constant += scale * *r;
                Some(())
            }

            // Bitvector constant - treat as integer
            TermKind::BitVecConst { value, .. } => {
                if let Some(val) = value.to_i64() {
                    *constant += scale * Rational64::from_integer(val);
                    Some(())
                } else {
                    // BigInt too large, skip for now
                    None
                }
            }

            // Variable (or bitvector variable - treat as integer variable)
            TermKind::Var(_) => {
                terms.push((term_id, scale));
                Some(())
            }

            // Uninterpreted function application whose sort is numeric -- treat
            // as an opaque arithmetic variable.  This is the UFLIA / UFLRA case:
            // e.g. `f(k)` in `(> (f k) 10)` where `f : Int -> Int`.  By
            // representing `f(k)` as an arithmetic variable we ensure that
            //   (a) the arithmetic solver tracks it and assigns it a model value,
            //   (b) the constraint `f(k) > 10` is handled consistently with any
            //       later instantiation that produces `f(k) <= 10`.
            //
            // Restriction: only treat flat Apply terms (all args atomic) as
            // arithmetic variables.  Nested applications like `f(f(k))` are
            // handled by the EUF solver; including them in arith would require
            // full Nelson-Oppen equality propagation to avoid spurious UNSAT.
            TermKind::Apply { args, .. } => {
                let sort = term.sort;
                let is_numeric = sort == manager.sorts.int_sort || sort == manager.sorts.real_sort;
                if is_numeric {
                    // Skip if any argument is a *non-Skolem* Apply term that is
                    // already in arith_terms.  This mirrors the restriction in
                    // `track_theory_vars` and avoids EUF/arith congruence conflicts.
                    // Skolem Apply terms (prefix "sk!") are safe because EUF has no
                    // equality facts about fresh Skolem symbols.
                    let has_conflicting_apply_arg = args.iter().any(|&arg| {
                        manager.get(arg).is_some_and(|a| {
                            if let TermKind::Apply {
                                func,
                                args: inner_args,
                                ..
                            } = &a.kind
                            {
                                if inner_args.is_empty() {
                                    return false;
                                }
                                let fname = manager.resolve_str(*func);
                                let is_skolem = fname.starts_with("sk!");
                                !is_skolem && self.arith_terms.contains(&arg)
                            } else {
                                false
                            }
                        })
                    });
                    if !has_conflicting_apply_arg {
                        terms.push((term_id, scale));
                        Some(())
                    } else {
                        // Non-Skolem nested Apply in arith -- cannot safely represent.
                        None
                    }
                } else {
                    // Non-numeric Apply (e.g. uninterpreted predicate) -- not linear.
                    None
                }
            }

            // Array select with numeric sort: treat `(select a i) : Int/Real` as
            // an opaque arithmetic atom with the given scale coefficient.  This
            // allows expressions such as `(+ (select a 0) (select a 1))` to be
            // parsed as linear arithmetic sums.
            TermKind::Select(_, _) => {
                let sort = term.sort;
                let is_numeric = sort == manager.sorts.int_sort || sort == manager.sorts.real_sort;
                if is_numeric {
                    terms.push((term_id, scale));
                    Some(())
                } else {
                    // Select of non-numeric sort (e.g. Bool array) -- not linear.
                    None
                }
            }

            // Addition
            TermKind::Add(args) => {
                for &arg in args {
                    self.extract_linear_terms(arg, scale, terms, constant, manager)?;
                }
                Some(())
            }

            // Subtraction
            TermKind::Sub(lhs, rhs) => {
                self.extract_linear_terms(*lhs, scale, terms, constant, manager)?;
                self.extract_linear_terms(*rhs, -scale, terms, constant, manager)?;
                Some(())
            }

            // Negation
            TermKind::Neg(arg) => self.extract_linear_terms(*arg, -scale, terms, constant, manager),

            // Multiplication of linear terms.  A product is linear iff at most one
            // factor is non-constant.  Each factor may itself be a nested
            // expression that reduces to a pure constant (e.g. `(- 3.0)`,
            // `(+ 1 2)`) or to a single scaled variable (e.g. `(- x)`).
            TermKind::Mul(args) => {
                let mut const_product = Rational64::one();
                // The single non-constant factor, if any, represented as a sum of
                // (variable, coefficient) pairs.  The factor must be linear-as-a-whole
                // (exactly one variable term, no additive constant) for the product
                // to remain linear.
                let mut var_factor: Option<(TermId, Rational64)> = None;

                for &arg in args {
                    let mut sub_terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
                    let mut sub_constant = Rational64::zero();
                    self.extract_linear_terms(
                        arg,
                        Rational64::one(),
                        &mut sub_terms,
                        &mut sub_constant,
                        manager,
                    )?;

                    if sub_terms.is_empty() {
                        // Pure constant factor — absorb into product.
                        const_product *= sub_constant;
                    } else if sub_terms.len() == 1 && sub_constant.is_zero() {
                        // Exactly one scaled variable with no additive constant,
                        // e.g. `x`, `(- x)`, `(* 2 x)`.  Record as the variable
                        // factor; if we already have one, the product is nonlinear.
                        if var_factor.is_some() {
                            return None;
                        }
                        var_factor = Some(sub_terms[0]);
                    } else {
                        // Either multi-variable (e.g. `(+ x y)`), or a linear
                        // expression with a constant offset (e.g. `(+ 1 x)`).
                        // Multiplying such a factor by another variable yields a
                        // nonlinear product.
                        return None;
                    }
                }

                let new_scale = scale * const_product;
                match var_factor {
                    Some((v, coef)) => {
                        terms.push((v, new_scale * coef));
                        Some(())
                    }
                    None => {
                        *constant += new_scale;
                        Some(())
                    }
                }
            }

            // Not linear
            _ => None,
        }
    }

    /// Assert a term
    pub fn assert(&mut self, term: TermId, manager: &mut TermManager) {
        let index = self.assertions.len();
        self.assertions.push(term);
        self.trail.push(TrailOp::AssertionAdded { index });
        self.invalidate_fp_cache();

        // Check if this is a boolean constant first
        if let Some(t) = manager.get(term) {
            match t.kind {
                TermKind::False => {
                    // Mark that we have a false assertion
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: None,
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                TermKind::True => {
                    // True is always satisfied, no need to encode
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: None,
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                _ => {}
            }
        }

        // Apply simplification if enabled
        let term_to_encode = if self.config.simplify {
            self.simplifier.simplify(term, manager)
        } else {
            term
        };

        // Check again if simplification produced a constant
        if let Some(t) = manager.get(term_to_encode) {
            match t.kind {
                TermKind::False => {
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    return;
                }
                TermKind::True => {
                    // Simplified to true, no need to encode
                    return;
                }
                _ => {}
            }
        }

        // Check for datatype constructor mutual exclusivity
        // If we see (= var Constructor), track it and check for conflicts
        if let Some(t) = manager.get(term_to_encode).cloned() {
            if let TermKind::Eq(lhs, rhs) = &t.kind {
                if let Some((var_term, constructor)) =
                    self.extract_dt_var_constructor(*lhs, *rhs, manager)
                {
                    if let Some(&existing_con) = self.dt_var_constructors.get(&var_term) {
                        if existing_con != constructor {
                            // Variable constrained to two different constructors - UNSAT
                            if !self.has_false_assertion {
                                self.has_false_assertion = true;
                                self.trail.push(TrailOp::FalseAssertionSet);
                            }
                            return;
                        }
                    } else {
                        self.dt_var_constructors.insert(var_term, constructor);
                    }
                }
            }
        }

        // Collect polarity information if polarity-aware encoding is enabled
        if self.polarity_aware {
            self.collect_polarities(term_to_encode, Polarity::Positive, manager);
        }

        // Encode the assertion immediately
        let lit = self.encode(term_to_encode, manager);
        self.sat.add_clause([lit]);

        // For Not(Eq(a,b)) assertions on arithmetic terms, eagerly add the
        // arithmetic disequality split (a<b OR a>b) so that ArithSolver assigns
        // distinct values from the very first SAT solve iteration.  Without this,
        // the ArithSolver may not enforce disequalities correctly.
        self.add_arith_diseq_split(term_to_encode, manager);

        if self.produce_unsat_cores {
            let na_index = self.named_assertions.len();
            self.named_assertions.push(NamedAssertion {
                term,
                name: None,
                index: index as u32,
            });
            self.trail
                .push(TrailOp::NamedAssertionAdded { index: na_index });
        }
    }

    /// Assert a named term (for unsat core tracking)
    pub fn assert_named(&mut self, term: TermId, name: &str, manager: &mut TermManager) {
        let index = self.assertions.len();
        self.assertions.push(term);
        self.trail.push(TrailOp::AssertionAdded { index });
        self.invalidate_fp_cache();

        // Check if this is a boolean constant first
        if let Some(t) = manager.get(term) {
            match t.kind {
                TermKind::False => {
                    // Mark that we have a false assertion
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: Some(name.to_string()),
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                TermKind::True => {
                    // True is always satisfied, no need to encode
                    if self.produce_unsat_cores {
                        let na_index = self.named_assertions.len();
                        self.named_assertions.push(NamedAssertion {
                            term,
                            name: Some(name.to_string()),
                            index: index as u32,
                        });
                        self.trail
                            .push(TrailOp::NamedAssertionAdded { index: na_index });
                    }
                    return;
                }
                _ => {}
            }
        }

        // Collect polarity information if polarity-aware encoding is enabled
        if self.polarity_aware {
            self.collect_polarities(term, Polarity::Positive, manager);
        }

        // Encode the assertion immediately
        let lit = self.encode(term, manager);
        self.sat.add_clause([lit]);

        // Eagerly add arith diseq split for Not(Eq(a,b)) assertions
        self.add_arith_diseq_split(term, manager);

        if self.produce_unsat_cores {
            let na_index = self.named_assertions.len();
            self.named_assertions.push(NamedAssertion {
                term,
                name: Some(name.to_string()),
                index: index as u32,
            });
            self.trail
                .push(TrailOp::NamedAssertionAdded { index: na_index });
        }
    }

    /// Get the unsat core (after check() returned Unsat)
    #[must_use]
    pub fn get_unsat_core(&self) -> Option<&UnsatCore> {
        self.unsat_core.as_ref()
    }

    /// Encode a term into SAT clauses using Tseitin transformation
    pub(super) fn encode(&mut self, term: TermId, manager: &mut TermManager) -> Lit {
        // Clone the term data to avoid borrowing issues
        let Some(t) = manager.get(term).cloned() else {
            let var = self.get_or_create_var(term);
            return Lit::pos(var);
        };

        match &t.kind {
            TermKind::True => {
                let var = self.get_or_create_var(manager.mk_true());
                self.sat.add_clause([Lit::pos(var)]);
                Lit::pos(var)
            }
            TermKind::False => {
                let var = self.get_or_create_var(manager.mk_false());
                self.sat.add_clause([Lit::neg(var)]);
                Lit::neg(var)
            }
            TermKind::Var(_) => {
                let var = self.get_or_create_var(term);
                // Track theory terms for model extraction
                let is_int = t.sort == manager.sorts.int_sort;
                let is_real = t.sort == manager.sorts.real_sort;

                if is_int || is_real {
                    // Track arithmetic terms
                    if !self.arith_terms.contains(&term) {
                        self.arith_terms.insert(term);
                        self.trail.push(TrailOp::ArithTermAdded { term });
                        // Register with arithmetic solver
                        self.arith.intern(term);
                    }
                } else if let Some(sort) = manager.sorts.get(t.sort)
                    && sort.is_bitvec()
                    && !self.bv_terms.contains(&term)
                {
                    self.bv_terms.insert(term);
                    self.trail.push(TrailOp::BvTermAdded { term });
                    // Register with BV solver if not already registered
                    if let Some(width) = sort.bitvec_width() {
                        self.bv.new_bv(term, width);
                    }
                }
                Lit::pos(var)
            }
            TermKind::Not(arg) => {
                let arg_lit = self.encode(*arg, manager);
                arg_lit.negate()
            }
            TermKind::And(args) => {
                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut arg_lits: Vec<Lit> = Vec::new();
                for &arg in args {
                    arg_lits.push(self.encode(arg, manager));
                }

                // Get polarity for optimization
                let polarity = if self.polarity_aware {
                    self.polarities
                        .get(&term)
                        .copied()
                        .unwrap_or(Polarity::Both)
                } else {
                    Polarity::Both
                };

                // result => all args (needed when result is positive)
                // ~result or arg1, ~result or arg2, ...
                if polarity != Polarity::Negative {
                    for &arg in &arg_lits {
                        self.sat.add_clause([result.negate(), arg]);
                    }
                }

                // all args => result (needed when result is negative)
                // ~arg1 or ~arg2 or ... or result
                if polarity != Polarity::Positive {
                    let mut clause: Vec<Lit> = arg_lits.iter().map(|l| l.negate()).collect();
                    clause.push(result);
                    self.sat.add_clause(clause);
                }

                result
            }
            TermKind::Or(args) => {
                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut arg_lits: Vec<Lit> = Vec::new();
                for &arg in args {
                    arg_lits.push(self.encode(arg, manager));
                }

                // Get polarity for optimization
                let polarity = if self.polarity_aware {
                    self.polarities
                        .get(&term)
                        .copied()
                        .unwrap_or(Polarity::Both)
                } else {
                    Polarity::Both
                };

                // result => some arg (needed when result is positive)
                // ~result or arg1 or arg2 or ...
                if polarity != Polarity::Negative {
                    let mut clause: Vec<Lit> = vec![result.negate()];
                    clause.extend(arg_lits.iter().copied());
                    self.sat.add_clause(clause);
                }

                // some arg => result (needed when result is negative)
                // ~arg1 or result, ~arg2 or result, ...
                if polarity != Polarity::Positive {
                    for &arg in &arg_lits {
                        self.sat.add_clause([arg.negate(), result]);
                    }
                }

                result
            }
            TermKind::Xor(lhs, rhs) => {
                let lhs_lit = self.encode(*lhs, manager);
                let rhs_lit = self.encode(*rhs, manager);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (lhs xor rhs)
                // result <=> (lhs and ~rhs) or (~lhs and rhs)

                // result => (lhs or rhs)
                self.sat.add_clause([result.negate(), lhs_lit, rhs_lit]);
                // result => (~lhs or ~rhs)
                self.sat
                    .add_clause([result.negate(), lhs_lit.negate(), rhs_lit.negate()]);

                // (lhs and ~rhs) => result
                self.sat.add_clause([lhs_lit.negate(), rhs_lit, result]);
                // (~lhs and rhs) => result
                self.sat.add_clause([lhs_lit, rhs_lit.negate(), result]);

                result
            }
            TermKind::Implies(lhs, rhs) => {
                let lhs_lit = self.encode(*lhs, manager);
                let rhs_lit = self.encode(*rhs, manager);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (~lhs or rhs)
                // result => ~lhs or rhs
                self.sat
                    .add_clause([result.negate(), lhs_lit.negate(), rhs_lit]);

                // (~lhs or rhs) => result
                // lhs or result, ~rhs or result
                self.sat.add_clause([lhs_lit, result]);
                self.sat.add_clause([rhs_lit.negate(), result]);

                result
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let cond_lit = self.encode(*cond, manager);
                let then_lit = self.encode(*then_br, manager);
                let else_lit = self.encode(*else_br, manager);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (cond ? then : else)
                // cond and result => then
                self.sat
                    .add_clause([cond_lit.negate(), result.negate(), then_lit]);
                // cond and then => result
                self.sat
                    .add_clause([cond_lit.negate(), then_lit.negate(), result]);

                // ~cond and result => else
                self.sat.add_clause([cond_lit, result.negate(), else_lit]);
                // ~cond and else => result
                self.sat.add_clause([cond_lit, else_lit.negate(), result]);

                result
            }
            TermKind::Eq(lhs, rhs) => {
                // Check if this is a boolean equality or theory equality
                let lhs_term = manager.get(*lhs);
                let is_bool_eq = lhs_term.is_some_and(|t| t.sort == manager.sorts.bool_sort);

                if is_bool_eq {
                    // Boolean equality: encode as iff
                    let lhs_lit = self.encode(*lhs, manager);
                    let rhs_lit = self.encode(*rhs, manager);

                    let result_var = self.get_or_create_var(term);
                    let result = Lit::pos(result_var);

                    // result <=> (lhs <=> rhs)
                    // result => (lhs => rhs) and (rhs => lhs)
                    self.sat
                        .add_clause([result.negate(), lhs_lit.negate(), rhs_lit]);
                    self.sat
                        .add_clause([result.negate(), rhs_lit.negate(), lhs_lit]);

                    // (lhs <=> rhs) => result
                    self.sat.add_clause([lhs_lit, rhs_lit, result]);
                    self.sat
                        .add_clause([lhs_lit.negate(), rhs_lit.negate(), result]);

                    result
                } else {
                    // Theory equality: create a fresh boolean variable
                    // Store the constraint for theory propagation
                    let var = self.get_or_create_var(term);
                    self.var_to_constraint
                        .insert(var, Constraint::Eq(*lhs, *rhs));
                    self.trail.push(TrailOp::ConstraintAdded { var });

                    // Track theory variables for model extraction
                    self.track_theory_vars(*lhs, manager);
                    self.track_theory_vars(*rhs, manager);

                    // Pre-parse arithmetic equality for ArithSolver
                    // Only for Int/Real sorts, not BitVec
                    let is_arith = lhs_term.is_some_and(|t| {
                        t.sort == manager.sorts.int_sort || t.sort == manager.sorts.real_sort
                    });
                    if is_arith {
                        // We use Le type as placeholder since equality will be asserted
                        // as both Le and Ge
                        if let Some(parsed) = self.parse_arith_comparison(
                            *lhs,
                            *rhs,
                            ArithConstraintType::Le,
                            term,
                            manager,
                        ) {
                            self.var_to_parsed_arith.insert(var, parsed);
                        }
                    }

                    Lit::pos(var)
                }
            }
            TermKind::Distinct(args) => {
                // Encode distinct as pairwise disequalities
                // distinct(a,b,c) <=> (a!=b) and (a!=c) and (b!=c)
                if args.len() <= 1 {
                    // trivially true
                    let var = self.get_or_create_var(manager.mk_true());
                    return Lit::pos(var);
                }

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut diseq_lits = Vec::new();
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        let eq = manager.mk_eq(args[i], args[j]);
                        let eq_lit = self.encode(eq, manager);
                        diseq_lits.push(eq_lit.negate());
                    }
                }

                // result => all disequalities
                for &diseq in &diseq_lits {
                    self.sat.add_clause([result.negate(), diseq]);
                }

                // all disequalities => result
                let mut clause: Vec<Lit> = diseq_lits.iter().map(|l| l.negate()).collect();
                clause.push(result);
                self.sat.add_clause(clause);

                result
            }
            TermKind::Let { bindings, body } => {
                // For encoding, we can substitute the bindings into the body
                // This is a simplification - a more sophisticated approach would
                // memoize the bindings
                let substituted = *body;
                for (name, value) in bindings.iter().rev() {
                    // In a full implementation, we'd perform proper substitution
                    // For now, just encode the body directly
                    let _ = (name, value);
                }
                self.encode(substituted, manager)
            }
            // Theory atoms (arithmetic, bitvec, arrays, UF)
            // These get fresh boolean variables - the theory solver handles the semantics
            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                // Constants are theory terms, not boolean formulas
                // Should not appear at top level in boolean context
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Neg(_)
            | TermKind::Add(_)
            | TermKind::Sub(_, _)
            | TermKind::Mul(_)
            | TermKind::Div(_, _)
            | TermKind::Mod(_, _) => {
                // Arithmetic terms - should not appear at boolean top level
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Lt(lhs, rhs) => {
                // Arithmetic predicate: lhs < rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Lt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Le(lhs, rhs) => {
                // Arithmetic predicate: lhs <= rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Le(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Gt(lhs, rhs) => {
                // Arithmetic predicate: lhs > rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Gt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Gt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Ge(lhs, rhs) => {
                // Arithmetic predicate: lhs >= rhs
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Ge(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Ge, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvConcat(_, _)
            | TermKind::BvExtract { .. }
            | TermKind::BvNot(_)
            | TermKind::BvAnd(_, _)
            | TermKind::BvOr(_, _)
            | TermKind::BvXor(_, _)
            | TermKind::BvAdd(_, _)
            | TermKind::BvSub(_, _)
            | TermKind::BvMul(_, _)
            | TermKind::BvShl(_, _)
            | TermKind::BvLshr(_, _)
            | TermKind::BvAshr(_, _) => {
                // Bitvector terms - should not appear at boolean top level
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::BvUdiv(_, _)
            | TermKind::BvSdiv(_, _)
            | TermKind::BvUrem(_, _)
            | TermKind::BvSrem(_, _) => {
                // Bitvector arithmetic terms (division/remainder)
                // Mark that we have arithmetic BV ops for conflict checking
                self.has_bv_arith_ops = true;
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::BvUlt(lhs, rhs) => {
                // Bitvector unsigned less-than: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Lt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                // Parse as arithmetic constraint (bitvector as bounded integer)
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvUle(lhs, rhs) => {
                // Bitvector unsigned less-than-or-equal: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Le(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvSlt(lhs, rhs) => {
                // Bitvector signed less-than: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Lt(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvSle(lhs, rhs) => {
                // Bitvector signed less-than-or-equal: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.var_to_constraint
                    .insert(var, Constraint::Le(*lhs, *rhs));
                self.trail.push(TrailOp::ConstraintAdded { var });
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Select(_, _) | TermKind::Store(_, _, _) => {
                // Array operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Apply { .. } => {
                // Uninterpreted function application - theory term
                let var = self.get_or_create_var(term);
                // Register Bool-valued function applications as theory
                // constraints so that EUF congruence closure can detect
                // conflicts when the SAT solver assigns opposite polarities
                // to congruent applications (e.g., t(m)=true, t(co)=false,
                // but m=co implies t(m)=t(co)).
                if t.sort == manager.sorts.bool_sort {
                    self.var_to_constraint
                        .insert(var, Constraint::BoolApp(term));
                    self.trail.push(TrailOp::ConstraintAdded { var });
                }
                Lit::pos(var)
            }
            TermKind::Forall {
                patterns,
                body,
                vars,
            } => {
                // Universal quantifiers: register with MBQI
                self.has_quantifiers = true;

                // Check if body is Exists — if so, Skolemize the nested existential.
                // This handles the Forall-Exists pattern: ∀x. ∃y. φ(x,y) → ∀x. φ(x, f(x))
                let body_id = *body;
                let vars_clone = vars.clone();
                let patterns_clone = patterns.clone();
                let body_is_exists = manager
                    .get(body_id)
                    .map(|t| matches!(t.kind, TermKind::Exists { .. }))
                    .unwrap_or(false);

                if body_is_exists {
                    // Skolemize: ∀x. ∃y. φ(x,y) → ∀x. φ(x, sk(x))
                    // This eliminates the nested existential so MBQI can handle
                    // the resulting universal quantifier directly.
                    #[cfg(feature = "std")]
                    {
                        let mut sk_ctx = crate::skolemization::SkolemizationContext::new();
                        if let Ok(skolemized) = sk_ctx.skolemize(manager, term) {
                            // Register the Skolemized version with MBQI
                            self.mbqi.add_quantifier(skolemized, manager);
                            // Register with E-matching engine
                            let _ = self.ematch_engine.register_quantifier(skolemized, manager);

                            // Also collect Skolem function application terms from the
                            // Skolemized body as MBQI candidates.  These terms (e.g.
                            // sk(x)) must appear in the candidate pool so that other
                            // universal quantifiers can be instantiated with them.
                            self.collect_skolem_candidates(skolemized, manager);
                        } else {
                            // Skolemization failed — fall back to original
                            self.mbqi.add_quantifier(term, manager);
                            let _ = self.ematch_engine.register_quantifier(term, manager);
                        }
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        self.mbqi.add_quantifier(term, manager);
                        let _ = self.ematch_engine.register_quantifier(term, manager);
                    }
                } else {
                    self.mbqi.add_quantifier(term, manager);
                    // Register with E-matching engine for trigger-based instantiation
                    let _ = self.ematch_engine.register_quantifier(term, manager);
                }

                // Collect ground terms from patterns as candidates
                for pattern in &patterns_clone {
                    for &trigger in pattern {
                        self.mbqi.collect_ground_terms(trigger, manager);
                    }
                }
                // Create a boolean variable for the quantifier
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Exists { patterns, .. } => {
                // Existential quantifiers: register with MBQI for tracking
                self.has_quantifiers = true;
                self.mbqi.add_quantifier(term, manager);
                // Collect ground terms from patterns
                for pattern in patterns {
                    for &trigger in pattern {
                        self.mbqi.collect_ground_terms(trigger, manager);
                    }
                }
                // Create a boolean variable for the quantifier
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // String operations - theory terms and predicates
            TermKind::StringLit(_)
            | TermKind::StrConcat(_, _)
            | TermKind::StrLen(_)
            | TermKind::StrSubstr(_, _, _)
            | TermKind::StrAt(_, _)
            | TermKind::StrReplace(_, _, _)
            | TermKind::StrReplaceAll(_, _, _)
            | TermKind::StrToInt(_)
            | TermKind::IntToStr(_)
            | TermKind::StrInRe(_, _) => {
                // String terms - theory solver handles these
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::StrContains(_, _)
            | TermKind::StrPrefixOf(_, _)
            | TermKind::StrSuffixOf(_, _)
            | TermKind::StrIndexOf(_, _, _) => {
                // String predicates - theory atoms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point constants and special values
            TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. } => {
                // FP constants - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point operations
            TermKind::FpAbs(_)
            | TermKind::FpNeg(_)
            | TermKind::FpSqrt(_, _)
            | TermKind::FpRoundToIntegral(_, _)
            | TermKind::FpAdd(_, _, _)
            | TermKind::FpSub(_, _, _)
            | TermKind::FpMul(_, _, _)
            | TermKind::FpDiv(_, _, _)
            | TermKind::FpRem(_, _)
            | TermKind::FpMin(_, _)
            | TermKind::FpMax(_, _)
            | TermKind::FpFma(_, _, _, _) => {
                // FP operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point predicates
            TermKind::FpLeq(_, _)
            | TermKind::FpLt(_, _)
            | TermKind::FpGeq(_, _)
            | TermKind::FpGt(_, _)
            | TermKind::FpEq(_, _)
            | TermKind::FpIsNormal(_)
            | TermKind::FpIsSubnormal(_)
            | TermKind::FpIsZero(_)
            | TermKind::FpIsInfinite(_)
            | TermKind::FpIsNaN(_)
            | TermKind::FpIsNegative(_)
            | TermKind::FpIsPositive(_) => {
                // FP predicates - theory atoms that return bool
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point conversions
            TermKind::FpToFp { .. }
            | TermKind::FpToSBV { .. }
            | TermKind::FpToUBV { .. }
            | TermKind::FpToReal(_)
            | TermKind::RealToFp { .. }
            | TermKind::SBVToFp { .. }
            | TermKind::UBVToFp { .. } => {
                // FP conversions - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Datatype operations
            TermKind::DtConstructor { .. }
            | TermKind::DtTester { .. }
            | TermKind::DtSelector { .. } => {
                // Datatype operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Match expressions on datatypes
            TermKind::Match { .. } => {
                // Match expressions - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
        }
    }

    /// Walk a (possibly Skolemized) quantifier term and collect Apply terms
    /// whose function name starts with "sk" as MBQI instantiation candidates.
    ///
    /// These Skolem function applications (e.g. `sk!0(x)`) must be in the
    /// candidate pool so that MBQI can instantiate other universal quantifiers
    /// with Skolem terms, enabling cross-quantifier contradictions.
    fn collect_skolem_candidates(&mut self, term: TermId, manager: &TermManager) {
        let mut visited = FxHashSet::default();
        self.collect_skolem_candidates_rec(term, manager, &mut visited);
    }

    fn collect_skolem_candidates_rec(
        &mut self,
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) {
        if !visited.insert(term) {
            return;
        }
        let Some(t) = manager.get(term) else {
            return;
        };
        match &t.kind {
            TermKind::Apply { func, args } => {
                let fname = manager.resolve_str(*func);
                if fname.starts_with("sk") || fname.starts_with("skf") {
                    // Register the whole application as a candidate
                    self.mbqi.add_candidate(term, t.sort);
                }
                for &arg in args.iter() {
                    self.collect_skolem_candidates_rec(arg, manager, visited);
                }
            }
            TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                self.collect_skolem_candidates_rec(*body, manager, visited);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &a in args {
                    self.collect_skolem_candidates_rec(a, manager, visited);
                }
            }
            TermKind::Not(a) | TermKind::Neg(a) => {
                self.collect_skolem_candidates_rec(*a, manager, visited);
            }
            TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b) => {
                self.collect_skolem_candidates_rec(*a, manager, visited);
                self.collect_skolem_candidates_rec(*b, manager, visited);
            }
            TermKind::Add(args) | TermKind::Mul(args) => {
                for &a in args.iter() {
                    self.collect_skolem_candidates_rec(a, manager, visited);
                }
            }
            TermKind::Ite(c, t_br, e) => {
                self.collect_skolem_candidates_rec(*c, manager, visited);
                self.collect_skolem_candidates_rec(*t_br, manager, visited);
                self.collect_skolem_candidates_rec(*e, manager, visited);
            }
            TermKind::Select(a, i) => {
                self.collect_skolem_candidates_rec(*a, manager, visited);
                self.collect_skolem_candidates_rec(*i, manager, visited);
            }
            TermKind::Store(a, i, v) => {
                self.collect_skolem_candidates_rec(*a, manager, visited);
                self.collect_skolem_candidates_rec(*i, manager, visited);
                self.collect_skolem_candidates_rec(*v, manager, visited);
            }
            _ => {}
        }
    }

    /// Scan all Constraint::Eq entries in var_to_constraint that are currently
    /// assigned False by the SAT model and add arithmetic splits `(lhs < rhs)
    /// OR (lhs > rhs)` for each.  This ensures ArithSolver knows about
    /// disequalities that arise from SAT-level implication propagation (e.g.
    /// from MBQI-generated instantiations like `(=> (= f(a) f(b)) (= a b))`).
    pub(super) fn add_arith_diseq_splits_for_sat_model(&mut self, manager: &mut TermManager) {
        use super::types::Constraint;
        use oxiz_sat::LBool;

        let pairs: Vec<(TermId, TermId)> = self
            .var_to_constraint
            .iter()
            .filter_map(|(&var, constraint)| {
                if let Constraint::Eq(lhs, rhs) = constraint {
                    // Only Int or Real sorts
                    let lhs_is_numeric = manager.get(*lhs).is_some_and(|lt| {
                        lt.sort == manager.sorts.int_sort || lt.sort == manager.sorts.real_sort
                    });
                    if lhs_is_numeric && self.sat.model_value(var) == LBool::False {
                        Some((*lhs, *rhs))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        for (lhs, rhs) in pairs {
            let lt_term = manager.mk_lt(lhs, rhs);
            let gt_term = manager.mk_gt(lhs, rhs);
            // Only add if the clause isn't already a tautology or unit-forced
            let lt_lit = self.encode(lt_term, manager);
            let gt_lit = self.encode(gt_term, manager);
            self.sat.add_clause([lt_lit, gt_lit]);
        }
    }

    /// When a MBQI instantiation result is `(not (= a b))` where a and b have
    /// Int sort, add the arithmetic split `(a < b) OR (a > b)` as a SAT clause.
    /// This ensures the ArithSolver knows about the disequality and doesn't
    /// assign both a and b to equal values.
    pub(super) fn add_arith_diseq_split(&mut self, term: TermId, manager: &mut TermManager) {
        let mut visited = FxHashSet::default();
        self.add_arith_diseq_split_recursive(term, manager, &mut visited);
    }

    /// Add trichotomy clauses `Eq(a,b) OR Lt(a,b) OR Gt(a,b)` for every
    /// arithmetic `Eq(a,b)` sub-term in the given MBQI instantiation result.
    ///
    /// This ensures that when the SAT solver assigns an arithmetic Eq to false
    /// (disequality), the ArithSolver learns a strict ordering constraint
    /// (Lt or Gt) and doesn't assign equal values.
    ///
    /// Only called for MBQI instantiation results, not for all assertions,
    /// to avoid blowing up the clause database on non-quantified problems.
    pub(super) fn add_arith_eq_trichotomy(&mut self, term: TermId, manager: &mut TermManager) {
        let mut visited = FxHashSet::default();
        self.add_arith_eq_trichotomy_recursive(term, manager, &mut visited);
    }

    fn add_arith_eq_trichotomy_recursive(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
        visited: &mut FxHashSet<TermId>,
    ) {
        if !visited.insert(term) {
            return;
        }

        let Some(t) = manager.get(term).cloned() else {
            return;
        };

        match &t.kind {
            TermKind::Eq(lhs, rhs) => {
                let lhs_is_numeric = manager.get(*lhs).is_some_and(|lt| {
                    lt.sort == manager.sorts.int_sort || lt.sort == manager.sorts.real_sort
                });
                // Only add trichotomy when at least one side is an
                // uninterpreted function application (Apply). This is the
                // pattern that appears in injectivity / congruence axioms
                // where f(a)=f(b) needs to be split into f(a)<f(b) or
                // f(a)>f(b) when the equality is false.
                // Avoid Select terms -- the array theory handles those.
                let lhs_is_apply = manager
                    .get(*lhs)
                    .is_some_and(|lt| matches!(lt.kind, TermKind::Apply { .. }));
                let rhs_is_apply = manager
                    .get(*rhs)
                    .is_some_and(|rt| matches!(rt.kind, TermKind::Apply { .. }));
                if lhs_is_numeric && (lhs_is_apply || rhs_is_apply) {
                    let (l, r) = (*lhs, *rhs);
                    // Add trichotomy: Eq(a,b) OR Lt(a,b) OR Gt(a,b)
                    let eq_var = self.get_or_create_var(term);
                    let eq_lit = Lit::pos(eq_var);
                    let lt_term = manager.mk_lt(l, r);
                    let gt_term = manager.mk_gt(l, r);
                    let lt_lit = self.encode(lt_term, manager);
                    let gt_lit = self.encode(gt_term, manager);
                    self.sat.add_clause([eq_lit, lt_lit, gt_lit]);
                }
            }
            TermKind::Not(arg) => {
                self.add_arith_eq_trichotomy_recursive(*arg, manager, visited);
            }
            TermKind::And(args) => {
                let args_clone: Vec<TermId> = args.iter().copied().collect();
                for arg in args_clone {
                    self.add_arith_eq_trichotomy_recursive(arg, manager, visited);
                }
            }
            TermKind::Or(args) => {
                let args_clone: Vec<TermId> = args.iter().copied().collect();
                for arg in args_clone {
                    self.add_arith_eq_trichotomy_recursive(arg, manager, visited);
                }
            }
            TermKind::Implies(lhs, rhs) => {
                let (l, r) = (*lhs, *rhs);
                self.add_arith_eq_trichotomy_recursive(l, manager, visited);
                self.add_arith_eq_trichotomy_recursive(r, manager, visited);
            }
            TermKind::Ite(_, then_br, else_br) => {
                let (t, e) = (*then_br, *else_br);
                self.add_arith_eq_trichotomy_recursive(t, manager, visited);
                self.add_arith_eq_trichotomy_recursive(e, manager, visited);
            }
            _ => {}
        }
    }

    /// Recursively walk a term to find all `Not(Eq(a, b))` sub-terms with
    /// arithmetic sorts and add the split `(a < b) OR (a > b)` for each.
    ///
    /// This handles MBQI instantiation results that are implications like
    /// `(=> guard (not (= a b)))` where the disequality is nested inside
    /// the formula rather than at the top level.
    fn add_arith_diseq_split_recursive(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
        visited: &mut FxHashSet<TermId>,
    ) {
        if !visited.insert(term) {
            return;
        }

        let Some(t) = manager.get(term).cloned() else {
            return;
        };

        match &t.kind {
            TermKind::Not(inner) => {
                let inner_id = *inner;
                if let Some(inner_t) = manager.get(inner_id).cloned() {
                    if let TermKind::Eq(lhs, rhs) = &inner_t.kind {
                        let lhs_is_numeric = manager.get(*lhs).is_some_and(|lt| {
                            lt.sort == manager.sorts.int_sort || lt.sort == manager.sorts.real_sort
                        });
                        if lhs_is_numeric {
                            let (l, r) = (*lhs, *rhs);
                            // Build Lt(lhs, rhs) and Gt(lhs, rhs)
                            let lt_term = manager.mk_lt(l, r);
                            let gt_term = manager.mk_gt(l, r);

                            // Encode both and add the disjunction
                            let lt_lit = self.encode(lt_term, manager);
                            let gt_lit = self.encode(gt_term, manager);
                            self.sat.add_clause([lt_lit, gt_lit]);
                        }
                    }
                }
                // Also recurse into the inner term
                self.add_arith_diseq_split_recursive(inner_id, manager, visited);
            }
            TermKind::And(args) => {
                let args_clone: Vec<TermId> = args.iter().copied().collect();
                for arg in args_clone {
                    self.add_arith_diseq_split_recursive(arg, manager, visited);
                }
            }
            TermKind::Or(args) => {
                let args_clone: Vec<TermId> = args.iter().copied().collect();
                for arg in args_clone {
                    self.add_arith_diseq_split_recursive(arg, manager, visited);
                }
            }
            TermKind::Implies(_, rhs) => {
                // Recurse into the consequent -- that's where the disequality
                // typically lives in quantifier instantiation lemmas
                let rhs_id = *rhs;
                self.add_arith_diseq_split_recursive(rhs_id, manager, visited);
            }
            TermKind::Ite(_, then_br, else_br) => {
                let (t, e) = (*then_br, *else_br);
                self.add_arith_diseq_split_recursive(t, manager, visited);
                self.add_arith_diseq_split_recursive(e, manager, visited);
            }
            _ => {}
        }
    }
}
