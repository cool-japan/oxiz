//! Arithmetic-atom parsing: turning an Int/Real comparison term into the
//! linear form the theory solver consumes.
//!
//! Relocated out of [`super`] verbatim when that file reached the workspace's
//! 2000-line ceiling. This is one self-contained concern — recognising which
//! Int/Real terms need a theory variable of their own, and flattening a
//! comparison's two sides into `sum(coefficient * term) <=> constant` — with a
//! single entry point used by the encoder (`parse_arith_comparison`) and no
//! coupling to the Tseitin transformation itself.

#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, ToPrimitive, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use smallvec::SmallVec;

use super::super::Solver;
use super::super::trail::TrailOp;
use super::super::types::{ArithConstraintType, ParsedArithConstraint};

impl Solver {
    /// Register a compound Int/Real-sorted term as an opaque arithmetic atom.
    ///
    /// Used for `div`, `mod` and conditional values: the linear solver cannot
    /// express them as a combination of their operands, so each gets its own
    /// theory variable (and hence a model value), and its semantics arrive
    /// separately as the ground axioms asserted by
    /// [`Solver::instantiate_arith_axioms`].  Non-numeric terms are ignored.
    pub(super) fn register_arith_atom(
        &mut self,
        term_id: TermId,
        sort: SortId,
        manager: &TermManager,
    ) {
        if sort != manager.sorts.int_sort && sort != manager.sorts.real_sort {
            return;
        }
        if self.arith_terms.contains(&term_id) {
            return;
        }
        self.arith_terms.insert(term_id);
        self.trail.push(TrailOp::ArithTermAdded { term: term_id });
        self.arith.intern(term_id);
        if sort == manager.sorts.int_sort {
            self.arith.mark_int_term(term_id);
        }
    }

    /// Parse an arithmetic comparison and extract linear expression.
    /// Returns: (terms with coefficients, constant, constraint_type).
    ///
    /// Results are cached by `reason` (the comparison term id).
    /// `ParsedArithConstraint` is purely structural — it depends only on the
    /// term graph — so the cache is safe to retain across CDCL backtracks.
    pub(in crate::solver) fn parse_arith_comparison(
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

    /// Extract linear terms from an arithmetic expression.
    /// Returns None if the term is not linear.
    ///
    /// # Explicit stack, not native recursion
    ///
    /// The walk uses an explicit work-stack.  The recursive version's frames
    /// stacked *on top of* [`Solver::encode_depth`]'s at the leaf (the encoder
    /// calls [`Solver::parse_arith_comparison`] from a comparison arm), so the
    /// true worst-case native stack was `encode_depth × cap + this walk × its
    /// own depth` — the encoder's depth cap never bounded it.  Worse, the
    /// encoder does not descend into arithmetic operands at all: a single
    /// shallow atom `(< deep-arith-chain 0)` reached this walk with the whole
    /// chain, and `parse_arith_comparison` is also called from theory paths
    /// (e.g. `TermKind::Eq`) on MBQI instantiation results that never pass the
    /// assert-time depth gate.  A stack overflow here is a fatal process abort
    /// no `Result` can report, and a depth cap would be dishonest: `None`
    /// means "not linear", so a capped deep-but-linear chain would silently
    /// drop the atom's theory meaning and gate the whole problem to `Unknown`.
    ///
    /// Only the `Mul` arm carries resume state: a product is linear iff at
    /// most one factor is non-constant, so each factor is evaluated into a
    /// *fresh* accumulation context and classified when it completes.  The
    /// suspended parent context travels inside the `Mul` frame itself, which
    /// makes the "stack empty at finalize" case unrepresentable — no `pop().
    /// expect(..)` is needed anywhere.
    ///
    /// On failure the caller's buffers are untouched (the recursive version
    /// left partial writes behind); the only caller,
    /// [`Solver::parse_arith_comparison`], discards the buffers on `None`, so
    /// this is unobservable.
    pub(in crate::solver) fn extract_linear_terms(
        &self,
        term_id: TermId,
        scale: Rational64,
        terms: &mut SmallVec<[(TermId, Rational64); 4]>,
        constant: &mut Rational64,
        manager: &TermManager,
    ) -> Option<()> {
        /// One linear-accumulation context: the `(term, coefficient)` pairs
        /// and folded constant of the sub-expression currently being walked.
        struct Level {
            terms: SmallVec<[(TermId, Rational64); 4]>,
            constant: Rational64,
        }
        impl Level {
            fn new() -> Self {
                Level {
                    terms: SmallVec::new(),
                    constant: Rational64::zero(),
                }
            }
        }
        /// Resume state for one `Mul` node.  A product is linear iff at most
        /// one factor is non-constant; each factor is evaluated into a fresh
        /// [`Level`] and classified here when it completes.  The factor must
        /// be linear-as-a-whole (exactly one variable term, no additive
        /// constant) for the product to remain linear.
        struct MulFrame {
            args: SmallVec<[TermId; 4]>,
            /// Index of the next factor to evaluate; factors `..next-1` have
            /// been classified already, factor `next-1` (if `next > 0`) is the
            /// one whose result is sitting in the current level.
            next: usize,
            const_product: Rational64,
            /// The single non-constant factor seen so far, e.g. `x`, `(- x)`,
            /// `(* 2 x)`.  A second one makes the product nonlinear.
            var_factor: Option<(TermId, Rational64)>,
            /// The scale the whole product contributes at.
            scale: Rational64,
            /// The suspended accumulation context of the `Mul`'s parent,
            /// restored when the product finalizes.
            parent: Level,
        }
        enum Work {
            /// Fold `term` into the current level at the given scale.
            Visit(TermId, Rational64),
            /// Classify the factor that just finished (when `next > 0`) and
            /// either evaluate the next factor or finalize the product.
            Mul(MulFrame),
        }

        let mut cur = Level::new();
        let mut work: Vec<Work> = vec![Work::Visit(term_id, scale)];

        while let Some(item) = work.pop() {
            match item {
                Work::Visit(id, sc) => {
                    let term = manager.get(id)?;
                    match &term.kind {
                        // Integer constant
                        TermKind::IntConst(n) => {
                            // BigInt too large for i64 -> not linear (honest
                            // reject; the atom stays gated).
                            let val = n.to_i64()?;
                            cur.constant += sc * Rational64::from_integer(val);
                        }

                        // Rational constant
                        TermKind::RealConst(r) => {
                            cur.constant += sc * *r;
                        }

                        // Bitvector constant - treat as integer
                        TermKind::BitVecConst { value, .. } => {
                            let val = value.to_i64()?;
                            cur.constant += sc * Rational64::from_integer(val);
                        }

                        // Variable (or bitvector variable - treat as integer variable)
                        TermKind::Var(_) => {
                            cur.terms.push((id, sc));
                        }

                        // Uninterpreted function application whose sort is numeric -- treat
                        // as an opaque arithmetic variable.  This is the UFLIA / UFLRA case:
                        // e.g. `f(k)` in `(> (f k) 10)` where `f : Int -> Int`.  By
                        // representing `f(k)` as an arithmetic variable we ensure that
                        //   (a) the arithmetic solver tracks it and assigns it a model value,
                        //   (b) the constraint `f(k) > 10` is handled consistently with any
                        //       later instantiation that produces `f(k) <= 10`.
                        //
                        // Nested applications (`f(f(k))`) are opaque arithmetic variables
                        // exactly like flat ones.  Excluding them — the mirror of the old
                        // restriction in `track_theory_vars` — did not make the solver
                        // conservative, it made it *wrong*: failing the linear parse leaves
                        // the whole atom without a theory meaning, so it survives as a free
                        // boolean and the solver reports `sat` for formulas it never
                        // satisfied.  The Nelson-Oppen equality propagation the exclusion
                        // was waiting for is now in place and explained
                        // (`TheoryManager::assert_explained_equality`).
                        TermKind::Apply { .. } => {
                            let sort = term.sort;
                            let is_numeric =
                                sort == manager.sorts.int_sort || sort == manager.sorts.real_sort;
                            if !is_numeric {
                                // Non-numeric Apply (e.g. uninterpreted predicate) -- not linear.
                                return None;
                            }
                            cur.terms.push((id, sc));
                        }

                        // Array select with numeric sort: treat `(select a i) : Int/Real` as
                        // an opaque arithmetic atom with the given scale coefficient.  This
                        // allows expressions such as `(+ (select a 0) (select a 1))` to be
                        // parsed as linear arithmetic sums.
                        TermKind::Select(_, _) => {
                            let sort = term.sort;
                            let is_numeric =
                                sort == manager.sorts.int_sort || sort == manager.sorts.real_sort;
                            if !is_numeric {
                                // Select of non-numeric sort (e.g. Bool array) -- not linear.
                                return None;
                            }
                            cur.terms.push((id, sc));
                        }

                        // Datatype accessor with numeric sort: `(head l) : Int` is an opaque
                        // arithmetic atom, exactly like `(select a i)`.  Without this the
                        // linear parse of `(= (head l) 10)` failed, no constraint reached the
                        // tableau, and `(= (head l) 10) ∧ (= (head l) 11)` was answered `sat`
                        // — the accessor is one ground term and cannot hold two values.
                        // `dt_axioms` supplies the rest of the accessor's meaning; here it
                        // only has to be *a* variable so that two occurrences agree.
                        TermKind::DtSelector { .. } => {
                            let sort = term.sort;
                            let is_numeric =
                                sort == manager.sorts.int_sort || sort == manager.sorts.real_sort;
                            if !is_numeric {
                                // Accessor of a non-numeric field -- not a linear atom.
                                return None;
                            }
                            cur.terms.push((id, sc));
                        }

                        // Addition: fold every operand at the same scale.
                        // Children are pushed in reverse so they pop (and hence
                        // append to `cur.terms`) left-to-right, exactly like the
                        // recursive descent did.
                        TermKind::Add(args) => {
                            for &arg in args.iter().rev() {
                                work.push(Work::Visit(arg, sc));
                            }
                        }

                        // Subtraction
                        TermKind::Sub(lhs, rhs) => {
                            work.push(Work::Visit(*rhs, -sc));
                            work.push(Work::Visit(*lhs, sc));
                        }

                        // Negation
                        TermKind::Neg(arg) => {
                            work.push(Work::Visit(*arg, -sc));
                        }

                        // Multiplication of linear terms.  Suspend the current
                        // context inside the frame; each factor is evaluated
                        // into a fresh one (matching the recursive version's
                        // per-factor `sub_terms`/`sub_constant` buffers).
                        TermKind::Mul(args) => {
                            work.push(Work::Mul(MulFrame {
                                args: args.iter().copied().collect(),
                                next: 0,
                                const_product: Rational64::one(),
                                var_factor: None,
                                scale: sc,
                                parent: core::mem::replace(&mut cur, Level::new()),
                            }));
                        }

                        // Integer `div`/`mod` and Int/Real-sorted `ite`: opaque arithmetic
                        // atoms.  Their meaning is not expressible as a linear combination
                        // of their operands, so the linear solver gets a variable and the
                        // *definition* arrives separately as ground axioms — see
                        // [`Solver::instantiate_arith_axioms`].  Until those axioms are
                        // asserted the term stays in nobody's theory, which is exactly what
                        // the honesty gate in `encode_guards` watches for.
                        //
                        // Real-sorted `Div` is deliberately excluded: `(/ x y)` is exact
                        // rational division, whose defining identity `x = y * (x / y)` is
                        // nonlinear, so it keeps failing the parse and stays gated.
                        TermKind::Mod(_, _) if term.sort == manager.sorts.int_sort => {
                            cur.terms.push((id, sc));
                        }
                        TermKind::Div(_, _) if term.sort == manager.sorts.int_sort => {
                            cur.terms.push((id, sc));
                        }
                        TermKind::Ite(_, _, _)
                            if term.sort == manager.sorts.int_sort
                                || term.sort == manager.sorts.real_sort =>
                        {
                            cur.terms.push((id, sc));
                        }

                        // Not linear.  The catch-all is the honest reject
                        // channel here — "shape the linear solver cannot
                        // represent" — so a future `TermKind` variant fails
                        // the parse (and the atom stays gated) rather than
                        // being mis-folded.
                        _ => return None,
                    }
                }
                Work::Mul(mut frame) => {
                    if frame.next > 0 {
                        // Classify the factor whose evaluation just completed
                        // into the current (per-factor) level.
                        if cur.terms.is_empty() {
                            // Pure constant factor — absorb into product.
                            frame.const_product *= cur.constant;
                        } else if cur.terms.len() == 1 && cur.constant.is_zero() {
                            // Exactly one scaled variable with no additive constant,
                            // e.g. `x`, `(- x)`, `(* 2 x)`.  Record as the variable
                            // factor; if we already have one, the product is nonlinear.
                            if frame.var_factor.is_some() {
                                return None;
                            }
                            frame.var_factor = Some(cur.terms[0]);
                        } else {
                            // Either multi-variable (e.g. `(+ x y)`), or a linear
                            // expression with a constant offset (e.g. `(+ 1 x)`).
                            // Multiplying such a factor by another variable yields a
                            // nonlinear product.
                            return None;
                        }
                    }
                    if frame.next < frame.args.len() {
                        let arg = frame.args[frame.next];
                        frame.next += 1;
                        cur = Level::new();
                        work.push(Work::Mul(frame));
                        work.push(Work::Visit(arg, Rational64::one()));
                    } else {
                        // All factors classified: restore the parent context
                        // and contribute the product to it.
                        let new_scale = frame.scale * frame.const_product;
                        cur = frame.parent;
                        match frame.var_factor {
                            Some((v, coef)) => cur.terms.push((v, new_scale * coef)),
                            None => cur.constant += new_scale,
                        }
                    }
                }
            }
        }

        // The work stack is empty, so every `Mul` frame has finalized and
        // `cur` is the root-level context again.  Only now touch the caller's
        // buffers, preserving the recursive version's append order.
        for pair in cur.terms {
            terms.push(pair);
        }
        *constant += cur.constant;
        Some(())
    }
}
