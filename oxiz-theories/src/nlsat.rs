//! NLSAT Theory Wrapper
//!
//! This module wraps the NLSAT solver (from oxiz-nlsat) to provide Theory trait
//! implementation for nonlinear arithmetic (QF_NIA and QF_NRA).
//!
//! ## Architecture
//!
//! - `NlsatTheory`: Main wrapper implementing `Theory` trait
//! - Handles both Real (QF_NRA) and Integer (QF_NIA) nonlinear arithmetic
//! - Delegates to `NlsatSolver` (real) or `NiaSolver` (integer)
//! - `TermPolyTranslator`: Converts `TermId` AST nodes to `Polynomial` representations
//! - `dispatch_nia_constraints`: Runs `NiaSolver` over a set of NIA assertions
//! - `dispatch_nra_constraints`: Runs `NlsatSolver` over a set of NRA assertions
//!
//! ## Reference
//!
//! - Z3's NLSAT integration in nlsat/nlsat_explain.cpp
//! - NLSAT solver: oxiz-nlsat::solver::NlsatSolver
//! - Integer solver: oxiz-nlsat::nia::NiaSolver

use crate::nl_eval::Interpretation;
use crate::nl_witness::{AlgebraicValue, NlWitnessValue};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::theory::{Theory, TheoryId, TheoryResult};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::Result;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::cad::CadPoint;
use oxiz_nlsat::nia::{NiaConfig, NiaSolver, VarType};
use oxiz_nlsat::solver::{NlsatSolver, SolverResult};
use oxiz_nlsat::types::AtomKind;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public result type for dispatch functions
// ─────────────────────────────────────────────────────────────────────────────

/// The definitive result from a nonlinear dispatch call.
///
/// `Unknown` is not included: `dispatch_*` functions return `None` to signal
/// "fall through to CDCL(T)" instead of wrapping Unknown.
///
/// A `Sat` carries the witness that justifies it. Callers that must answer
/// `(get-model)` / `(get-value ...)` need concrete values, and the only values
/// that can be reported without risking a wrong answer are the ones the
/// procedure actually found — reconstructing them afterwards from a verdict
/// alone would be guessing. The witness may be *partial* (a decision procedure
/// that decided the problem without pinning every leaf leaves the rest open),
/// so a caller installing it must verify before publishing it; see
/// [`crate::nl_eval::holds_under`].
///
/// # The two witness channels are exclusive, never blended
///
/// `witness` speaks [`Interpretation`]'s value language, which is rational.
/// A real cell decomposition can produce a value no rational equals (`√2` for
/// `x² = 2`), and `algebraic` is the channel for exactly that case — see
/// [`crate::nl_witness`].
///
/// At most one of the two is ever populated for a given `Sat`:
///
/// * every variable rational → `witness` holds them, `algebraic` is empty
///   (byte-for-byte the behaviour that predates the algebraic channel);
/// * some variable irrational → `algebraic` holds **every** variable,
///   rationals included, and `witness` is left *empty*.
///
/// The second half of that split is what keeps the caller safe. An
/// `algebraic` map covering only the irrational variables would leave the
/// rest to be filled in from sort defaults by whatever reports the model,
/// producing an assignment that satisfies nothing; and a `witness` populated
/// alongside it would be partial in a way `holds_under` reads as *false*,
/// tripping `oxiz-solver`'s debug assertion on a perfectly correct `sat`.
/// Populating exactly one, completely, avoids both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlDispatchResult {
    /// The constraint set is satisfiable, witnessed by these values.
    Sat {
        /// The rational witness. Empty when `algebraic` is populated.
        witness: Box<Interpretation>,
        /// Exact values for *every* variable of the problem, populated only
        /// when at least one of them is irrational. Empty otherwise.
        algebraic: FxHashMap<TermId, NlWitnessValue>,
    },
    /// The constraint set is unsatisfiable.
    Unsat,
}

impl NlDispatchResult {
    /// A `Sat` whose witness is `interp`, with no algebraic values.
    #[must_use]
    pub fn sat(interp: Interpretation) -> Self {
        Self::Sat {
            witness: Box::new(interp),
            algebraic: FxHashMap::default(),
        }
    }

    /// A `Sat` witnessed entirely by exact values, at least one of which is
    /// irrational. The rational channel is left empty; see the type's docs
    /// for why the two are never populated together.
    #[must_use]
    pub fn sat_algebraic(algebraic: FxHashMap<TermId, NlWitnessValue>) -> Self {
        Self::Sat {
            witness: Box::new(Interpretation::empty()),
            algebraic,
        }
    }

    /// A `Sat` carrying no values at all — for a procedure that established
    /// satisfiability without producing an assignment.
    #[must_use]
    pub fn sat_unwitnessed() -> Self {
        Self::sat(Interpretation::empty())
    }

    /// The rational witness, or `None` for `Unsat`.
    #[must_use]
    pub fn witness(&self) -> Option<&Interpretation> {
        match self {
            Self::Sat { witness, .. } => Some(witness),
            Self::Unsat => None,
        }
    }

    /// The exact-value witness, empty unless this `Sat` needed the algebraic
    /// channel. `None` for `Unsat`.
    #[must_use]
    pub fn algebraic_witness(&self) -> Option<&FxHashMap<TermId, NlWitnessValue>> {
        match self {
            Self::Sat { algebraic, .. } => Some(algebraic),
            Self::Unsat => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Term→Polynomial translator
// ─────────────────────────────────────────────────────────────────────────────

/// Translates `TermId` AST nodes to `Polynomial` values for use with
/// the NLSAT / NIA solver.
///
/// Maintains a cache of `TermId → polynomial variable index` so that each
/// unique variable term receives a stable index.
///
/// ## Integer `div`/`mod`
///
/// SMT-LIB's `Ints` theory defines `div`/`mod` *Euclidean-style*: for any
/// integers `m` and nonzero `n`, `(div m n)` and `(mod m n)` are the unique
/// `q`, `r` satisfying `m = n·q + r` with `0 ≤ r < |n|` — the remainder is
/// never negative, regardless of either operand's sign. (`(div 7 (- 2))` is
/// `-3` and `(mod 7 (- 2))` is `1`: `7 = (-2)·(-3) + 1`. `(div (- 7) 2)` is
/// `-4` and `(mod (- 7) 2)` is `1`: `-7 = 2·(-4) + 1`.) `div`/`mod` by `0`
/// are left uninterpreted by the standard — any value is admissible, so this
/// translator does not attempt to give them a polynomial meaning at all.
///
/// A polynomial has no division operator, so a `div`/`mod` occurrence with a
/// resolvable nonzero constant divisor is instead given a meaning by
/// introducing fresh quotient/remainder variables `q`, `r` and asserting the
/// Euclidean identity above as ordinary polynomial side constraints —
/// `Self::ensure_divmod_witness` does the encoding, `Self::divmod_leaf`
/// wires it into `translate_poly`. This mirrors the *ground-lemma* Euclidean
/// encoding `oxiz-solver`'s `arith_axioms` module asserts for the linear
/// (simplex) arithmetic path — same convention, same divisor-constant
/// folding (see `resolve_int_divisor`), just phrased as CAD-visible
/// polynomial atoms instead of SAT-core clauses, for the nonlinear problems
/// that never reach the simplex path at all.  A symbolic divisor still has no
/// polynomial encoding (the identity `m = n·q + r` would itself be
/// nonlinear in `n` and `q`) and is left untranslated, same as before.
pub struct TermPolyTranslator<'a> {
    manager: &'a TermManager,
    nlsat: &'a mut NiaSolver,
    var_cache: HashMap<TermId, u32>,
    integer_mode: bool,
    /// `(dividend, divisor) → (quotient_var, remainder_var)`, so repeated
    /// occurrences of the same `div`/`mod` term share one pair of witnesses
    /// (and one copy of the identity) instead of re-deriving it.
    divmod_witnesses: HashMap<(TermId, TermId), (u32, u32)>,
    /// Euclidean identity and range atoms collected by
    /// [`Self::ensure_divmod_witness`], to be folded into the dispatcher's
    /// atom set once translation of the whole problem is done.
    divmod_side_constraints: Vec<PolyAtom>,
}

impl<'a> TermPolyTranslator<'a> {
    /// Create a new translator.
    pub fn new(manager: &'a TermManager, nlsat: &'a mut NiaSolver, integer_mode: bool) -> Self {
        Self {
            manager,
            nlsat,
            var_cache: HashMap::new(),
            integer_mode,
            divmod_witnesses: HashMap::new(),
            divmod_side_constraints: Vec::new(),
        }
    }

    /// Translate a term into a `Polynomial`.
    ///
    /// Returns `None` for sub-expressions that cannot be expressed as a
    /// polynomial (e.g. a symbolic-divisor `div`/`mod`, an uninterpreted
    /// function application).
    pub fn translate(&mut self, term_id: TermId) -> Option<Polynomial> {
        let manager = self.manager;
        translate_poly(manager, self, term_id)
    }

    /// Euclidean `div`/`mod` side constraints collected so far (see the
    /// struct-level doc comment). The caller folds these into the atom set
    /// once every assertion has been translated.
    fn take_divmod_side_constraints(&mut self) -> Vec<PolyAtom> {
        std::mem::take(&mut self.divmod_side_constraints)
    }

    /// Give `(div dividend divisor)` / `(mod dividend divisor)` a polynomial
    /// meaning, when the node is `Int`-sorted and `divisor` resolves to a
    /// nonzero integer constant.
    ///
    /// Returns the `(quotient_var, remainder_var)` pair. `None` means the
    /// occurrence could not be given a polynomial meaning at all — the node
    /// is not `Int`-sorted (SMT-LIB's `/` is exact rational division and
    /// shares this same `TermKind::Div` node when the dividend is `Real`;
    /// only the `Ints` theory's `div`/`mod` are Euclidean, and only when the
    /// *node itself* is `Int`-sorted, mirroring the `is_int` guard
    /// `oxiz-solver`'s ground-lemma encoder checks for the same reason), a
    /// symbolic divisor (the defining identity would itself be nonlinear in
    /// the divisor), a divisor of `0` (uninterpreted per SMT-LIB — this
    /// translator has no polynomial form of the congruence-only fact the
    /// ground-lemma encoder uses instead), or `i64::MIN` (no representable
    /// `|n|`) — the caller (`Self::divmod_leaf`) propagates that `None` the
    /// same way any other untranslatable sub-term does, so the containing
    /// atom is dropped and extraction is marked incomplete rather than
    /// asserting a partial or wrong identity.
    fn ensure_divmod_witness(&mut self, dividend: TermId, divisor: TermId) -> Option<(u32, u32)> {
        if let Some(&witnesses) = self.divmod_witnesses.get(&(dividend, divisor)) {
            return Some(witnesses);
        }

        // The Euclidean identity only holds meaning for the `Ints` theory's
        // `div`/`mod`; `TermKind::Div` also stands for exact rational `/`
        // when the node is `Real`-sorted (both keywords build the same node
        // — see `oxiz-core`'s parser). Substituting an integer quotient/
        // remainder pair for real division would silently change the
        // node's meaning (and, since `q`/`r` are `Integer`-typed, would
        // wrongly force the dividend to an integer value) rather than
        // merely fail to translate it, so this must be checked before
        // anything else.
        let dividend_is_int = self
            .manager
            .get(dividend)
            .is_some_and(|t| t.sort == self.manager.sorts.int_sort);
        if !dividend_is_int {
            return None;
        }

        // Resolve and validate the divisor *before* emitting anything, so a
        // symbolic or zero divisor never leaves a partial identity behind.
        let n = resolve_int_divisor(divisor, self.manager)?;
        if n == 0 {
            return None;
        }
        let abs_n_minus_one = n.checked_abs().and_then(|a| a.checked_sub(1))?;

        let dividend_poly = self.translate(dividend)?;

        let q = self.nlsat.nlsat_mut().new_arith_var();
        let r = self.nlsat.nlsat_mut().new_arith_var();
        self.nlsat.set_var_type(q, VarType::Integer);
        self.nlsat.set_var_type(r, VarType::Integer);
        let q_poly = Polynomial::from_var(q);
        let r_poly = Polynomial::from_var(r);
        let n_poly = Polynomial::constant(BigRational::from_integer(BigInt::from(n)));

        // m = n*q + r
        let reconstructed = Polynomial::add(&Polynomial::mul(&n_poly, &q_poly), &r_poly);
        self.divmod_side_constraints.push(PolyAtom {
            poly: Polynomial::sub(&dividend_poly, &reconstructed),
            kind: AtomKind::Eq,
            positive: true,
            synthetic: true,
        });
        // 0 <= r, phrased as NOT(r < 0)
        self.divmod_side_constraints.push(PolyAtom {
            poly: r_poly.clone(),
            kind: AtomKind::Lt,
            positive: false,
            synthetic: true,
        });
        // r <= |n|-1, phrased as NOT(r - (|n|-1) > 0)
        let upper = Polynomial::constant(BigRational::from_integer(BigInt::from(abs_n_minus_one)));
        self.divmod_side_constraints.push(PolyAtom {
            poly: Polynomial::sub(&r_poly, &upper),
            kind: AtomKind::Gt,
            positive: false,
            synthetic: true,
        });

        self.divmod_witnesses.insert((dividend, divisor), (q, r));
        Some((q, r))
    }

    fn get_or_create_var(&mut self, term_id: TermId) -> u32 {
        if let Some(&v) = self.var_cache.get(&term_id) {
            return v;
        }
        let v = self.nlsat.nlsat_mut().new_arith_var();
        // Assign integrality from the variable's *actual* sort, not the global
        // `integer_mode` flag. In mixed QF_NIRA problems only genuinely
        // Int-sorted variables may carry the integrality constraint; Real
        // variables must stay real (the NiaSolver default), otherwise a
        // satisfiable non-integral real assignment is spuriously rejected and
        // the solver reports a false UNSAT.
        // Reference: Z3's mixed Int/Real handling in nlsat/nlsat_solver.cpp.
        let is_int_var = self
            .manager
            .get(term_id)
            .map(|t| t.sort == self.manager.sorts.int_sort)
            .unwrap_or(self.integer_mode);
        if is_int_var {
            self.nlsat.set_var_type(v, VarType::Integer);
        }
        self.var_cache.insert(term_id, v);
        v
    }

    /// Return the variable mapping (for model extraction).
    pub fn var_cache(&self) -> &HashMap<TermId, u32> {
        &self.var_cache
    }
}

impl PolyVarSource for TermPolyTranslator<'_> {
    fn var_for(&mut self, term_id: TermId) -> u32 {
        self.get_or_create_var(term_id)
    }

    fn divmod_leaf(&mut self, lhs: TermId, rhs: TermId, is_div: bool) -> Option<Polynomial> {
        let (q, r) = self.ensure_divmod_witness(lhs, rhs)?;
        Some(Polynomial::from_var(if is_div { q } else { r }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared iterative term→polynomial translation
// ─────────────────────────────────────────────────────────────────────────────

/// The one thing the two translators do differently: mint (or look up) the
/// polynomial variable index for a term, and (only `TermPolyTranslator`)
/// give a `div`/`mod` node a polynomial meaning.
trait PolyVarSource {
    /// The polynomial variable index standing for `term_id`.
    fn var_for(&mut self, term_id: TermId) -> u32;

    /// Give `(div lhs rhs)` (`is_div`) or `(mod lhs rhs)` a polynomial
    /// meaning. The default `None` matches `RealPolyTranslator`
    /// unconditionally (`QF_NRA` has no `Ints`-theory `div`/`mod` at all).
    /// `TermPolyTranslator::divmod_leaf` additionally returns `None` at
    /// runtime, per occurrence, for a `Real`-sorted `Div` node — see
    /// [`TermPolyTranslator::ensure_divmod_witness`]'s doc comment for why
    /// exact rational `/` (which shares this same `TermKind::Div` node with
    /// `Ints`' Euclidean `div`) cannot be given this encoding.
    fn divmod_leaf(&mut self, _lhs: TermId, _rhs: TermId, _is_div: bool) -> Option<Polynomial> {
        None
    }
}

/// How a node's polynomial is assembled from its operands'.
#[derive(Debug, Clone, Copy)]
enum PolyCombine {
    /// Unary `-`, over the sum of the (single) operand.
    Neg,
    /// n-ary `+`, folded left to right from zero.
    Add,
    /// n-ary `*`, folded left to right from one.
    Mul,
    /// Binary `-`: the first operand minus the rest.
    Sub,
}

/// One pending arithmetic node of the iterative translation.
struct PolyFrame {
    /// How to combine the operands.
    combine: PolyCombine,
    /// Operand terms still to translate, reversed so `pop` yields them in the
    /// same left-to-right order the recursive version used (which matters:
    /// `var_for` mints variable indices as a side effect).
    pending: Vec<TermId>,
    /// Operands translated so far, in operand order.
    done: Vec<Polynomial>,
}

impl PolyFrame {
    /// Fold this node's operands into its polynomial.
    fn finish(self) -> Polynomial {
        match self.combine {
            PolyCombine::Neg => Polynomial::neg(&fold_add(self.done)),
            PolyCombine::Add => fold_add(self.done),
            PolyCombine::Mul => {
                let mut acc = Polynomial::one();
                for p in &self.done {
                    acc = Polynomial::mul(&acc, p);
                }
                acc
            }
            PolyCombine::Sub => {
                let mut operands = self.done.into_iter();
                let mut acc = operands.next().unwrap_or_else(Polynomial::zero);
                for p in operands {
                    acc = Polynomial::sub(&acc, &p);
                }
                acc
            }
        }
    }
}

/// Sum a list of polynomials left to right, starting from zero — which is also
/// exactly what a one-element list needs, so unary nodes can reuse it instead
/// of asserting their operand is present.
fn fold_add(parts: Vec<Polynomial>) -> Polynomial {
    let mut acc = Polynomial::zero();
    for p in &parts {
        acc = Polynomial::add(&acc, p);
    }
    acc
}

/// What translating one term needs: a polynomial already, or its operands.
enum PolyOpened {
    /// A constant or a variable.
    Leaf(Polynomial),
    /// An arithmetic operator whose operands must be translated first.
    Frame(PolyFrame),
}

/// Classify one term for [`translate_poly`]. `None` means "not expressible as
/// a polynomial", exactly as the recursive version's `_ => None` did.
fn open_poly<S: PolyVarSource>(
    manager: &TermManager,
    src: &mut S,
    term_id: TermId,
) -> Option<PolyOpened> {
    // The operand list is copied out before `src` is touched, so the borrow of
    // `manager` never overlaps the `&mut` borrow `var_for` needs.
    enum Shape {
        Const(Polynomial),
        Var,
        Op(PolyCombine, Vec<TermId>),
        /// `(div lhs rhs)` (`is_div`) or `(mod lhs rhs)`: not an ordinary
        /// polynomial operator, so it bypasses `PolyFrame`/`PolyCombine`
        /// entirely and asks the source to supply a leaf directly (see
        /// [`PolyVarSource::divmod_leaf`]).
        DivMod {
            lhs: TermId,
            rhs: TermId,
            is_div: bool,
        },
    }
    let shape = {
        let term = manager.get(term_id)?;
        match &term.kind {
            TermKind::IntConst(n) => {
                Shape::Const(Polynomial::constant(BigRational::from_integer(n.clone())))
            }
            TermKind::RealConst(r) => Shape::Const(Polynomial::constant(BigRational::new(
                BigInt::from(r.numer().to_i64().unwrap_or(0)),
                BigInt::from(r.denom().to_i64().unwrap_or(1)),
            ))),
            TermKind::Var(_) => Shape::Var,
            TermKind::Neg(inner) => Shape::Op(PolyCombine::Neg, vec![*inner]),
            TermKind::Add(args) => {
                Shape::Op(PolyCombine::Add, args.iter().rev().copied().collect())
            }
            TermKind::Sub(lhs, rhs) => Shape::Op(PolyCombine::Sub, vec![*rhs, *lhs]),
            TermKind::Mul(args) => {
                Shape::Op(PolyCombine::Mul, args.iter().rev().copied().collect())
            }
            TermKind::Div(lhs, rhs) => Shape::DivMod {
                lhs: *lhs,
                rhs: *rhs,
                is_div: true,
            },
            TermKind::Mod(lhs, rhs) => Shape::DivMod {
                lhs: *lhs,
                rhs: *rhs,
                is_div: false,
            },
            _ => return None,
        }
    };
    Some(match shape {
        Shape::Const(p) => PolyOpened::Leaf(p),
        Shape::Var => PolyOpened::Leaf(Polynomial::from_var(src.var_for(term_id))),
        Shape::Op(combine, pending) => PolyOpened::Frame(PolyFrame {
            combine,
            pending,
            done: Vec::new(),
        }),
        Shape::DivMod { lhs, rhs, is_div } => PolyOpened::Leaf(src.divmod_leaf(lhs, rhs, is_div)?),
    })
}

/// Translate an arithmetic term into a polynomial with an explicit stack.
///
/// The recursive version descended once per nesting level of an entirely
/// input-controlled term (and cloned the whole `TermKind`, `Vec<TermId>` and
/// `BigInt` included, into every frame). Shared subterms are translated once
/// and reused, so a `let`-shared doubling term cannot re-expand exponentially.
fn translate_poly<S: PolyVarSource>(
    manager: &TermManager,
    src: &mut S,
    root: TermId,
) -> Option<Polynomial> {
    let mut memo: HashMap<TermId, Polynomial> = HashMap::new();
    let mut frames: Vec<PolyFrame> = match open_poly(manager, src, root)? {
        PolyOpened::Leaf(p) => return Some(p),
        PolyOpened::Frame(f) => vec![f],
    };
    // A finished operand polynomial travelling back to the frame that asked
    // for it, paired with the term it came from so it can be memoised.
    let mut carry: Option<Polynomial> = None;
    // The term each frame is translating, parallel to `frames`.
    let mut frame_terms: Vec<TermId> = vec![root];

    while !frames.is_empty() {
        let next = match frames.last_mut() {
            Some(top) => {
                if let Some(p) = carry.take() {
                    top.done.push(p);
                }
                top.pending.pop()
            }
            // Unreachable: the loop condition just checked non-emptiness.
            None => break,
        };
        match next {
            Some(child) => {
                if let Some(hit) = memo.get(&child) {
                    carry = Some(hit.clone());
                    continue;
                }
                match open_poly(manager, src, child)? {
                    PolyOpened::Leaf(p) => {
                        memo.insert(child, p.clone());
                        carry = Some(p);
                    }
                    PolyOpened::Frame(f) => {
                        frames.push(f);
                        frame_terms.push(child);
                    }
                }
            }
            None => match (frames.pop(), frame_terms.pop()) {
                (Some(frame), Some(term)) => {
                    let built = frame.finish();
                    memo.insert(term, built.clone());
                    carry = Some(built);
                }
                // Unreachable: the two stacks are pushed and popped together.
                _ => break,
            },
        }
    }

    carry
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: nonlinearity detection
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if `term_id` (recursively) contains a `Mul` node where at
/// least two non-constant operands are multiplied together.
pub fn term_is_nonlinear(term_id: TermId, manager: &TermManager) -> bool {
    // Explicit stack plus a visited set. This is the first thing the NIA/NRA
    // dispatcher does for every assertion, so its depth is the assertion's
    // nesting depth (input-controlled) and its breadth is the assertion DAG
    // (which a `let`-shared term makes exponential to re-walk). It returns
    // `bool`, so a depth cap could only answer "linear" for a nonlinear
    // problem and hand the whole logic to the wrong solver.
    let mut stack: Vec<TermId> = vec![term_id];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = manager.get(current) else {
            continue;
        };
        match &term.kind {
            TermKind::Mul(args) => {
                let non_const_count = args.iter().filter(|&&a| !is_const_term(a, manager)).count();
                if non_const_count >= 2 {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            TermKind::Add(args) | TermKind::And(args) => stack.extend(args.iter().copied()),
            TermKind::Sub(lhs, rhs)
            | TermKind::Eq(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            TermKind::Neg(inner) => stack.push(*inner),
            _ => {}
        }
    }
    false
}

fn is_const_term(term_id: TermId, manager: &TermManager) -> bool {
    manager
        .get(term_id)
        .map(|t| matches!(&t.kind, TermKind::IntConst(_) | TermKind::RealConst(_)))
        .unwrap_or(false)
}

/// Maximum nesting explored while folding a `div`/`mod` divisor expression
/// down to a constant. A divisor is a tiny expression in practice; this only
/// bounds an adversarially deep one.
const MAX_DIVISOR_FOLD_DEPTH: u32 = 32;

/// The `i64` value of `term` if it is a *constant* integer expression, or
/// `None` if it is not constant, does not fit `i64`, or overflows while
/// folding.
///
/// Folds exactly the shapes `oxiz-solver`'s `arith_axioms::int_constant`
/// does (`IntConst`, `Neg`, `Sub`, `Add`, `Mul`, all `i64`-checked) so this
/// translator's notion of "the divisor is the constant `n`" can never
/// disagree with the ground-lemma encoder's for the same term: a divisor
/// either resolves to the identical value on both paths or is left symbolic
/// on both. `oxiz-theories` sits below `oxiz-solver` in the dependency
/// graph, so the two copies cannot share code; this one is instead pinned by
/// matching tests on both paths (the folded-divisor case in
/// `oxiz-theories/tests/pr27_nia_divmod.rs` and the equivalent one in
/// `oxiz-solver/tests/pr27_divmod_semantics.rs`) that each exercise the
/// identical `(- (* 2 3) 1)`-style expression -- the fold shapes, depth cap
/// (`MAX_DIVISOR_FOLD_DEPTH`), and `i64`-checked arithmetic are deliberately
/// kept identical between the two copies by inspection, not by a proof.
fn resolve_int_divisor(term: TermId, manager: &TermManager) -> Option<i64> {
    resolve_int_divisor_at(term, manager, 0)
}

fn resolve_int_divisor_at(term: TermId, manager: &TermManager, depth: u32) -> Option<i64> {
    if depth >= MAX_DIVISOR_FOLD_DEPTH {
        return None;
    }
    match &manager.get(term)?.kind {
        TermKind::IntConst(n) => n.to_i64(),
        TermKind::Neg(a) => resolve_int_divisor_at(*a, manager, depth + 1)?.checked_neg(),
        TermKind::Sub(a, b) => {
            let lhs = resolve_int_divisor_at(*a, manager, depth + 1)?;
            let rhs = resolve_int_divisor_at(*b, manager, depth + 1)?;
            lhs.checked_sub(rhs)
        }
        TermKind::Add(args) => {
            let mut sum: i64 = 0;
            for &a in args {
                sum = sum.checked_add(resolve_int_divisor_at(a, manager, depth + 1)?)?;
            }
            Some(sum)
        }
        TermKind::Mul(args) => {
            let mut product: i64 = 1;
            for &a in args {
                product = product.checked_mul(resolve_int_divisor_at(a, manager, depth + 1)?)?;
            }
            Some(product)
        }
        _ => None,
    }
}

/// Whether the term mentions an operator the polynomial translation cannot
/// express. Same shape (and same reasons for being iterative + memoised) as
/// [`term_is_nonlinear`]: `bool` return, `check_sat` path, input-controlled
/// depth and sharing.
fn contains_non_polynomial_ops(term_id: TermId, manager: &TermManager) -> bool {
    let mut stack: Vec<TermId> = vec![term_id];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = manager.get(current) else {
            continue;
        };
        match &term.kind {
            // An `Int`-sorted node with a resolvable nonzero constant
            // divisor is handled by `TermPolyTranslator::
            // ensure_divmod_witness`'s Euclidean encoding below — not
            // "unsupported" any more, though a symbolic or zero divisor
            // still is, and so is a `Real`-sorted `Div` node (exact rational
            // `/` shares this same `TermKind` but has no Euclidean meaning;
            // see `ensure_divmod_witness`'s doc comment). Either way, when
            // it *is* handled, both operands are still walked so a
            // *genuinely* unsupported sub-term (nested inside the dividend,
            // say) is still found.
            TermKind::Div(lhs, rhs) | TermKind::Mod(lhs, rhs) => {
                let is_int_node = term.sort == manager.sorts.int_sort;
                if !is_int_node || resolve_int_divisor(*rhs, manager).is_none_or(|n| n == 0) {
                    return true;
                }
                stack.push(*lhs);
                stack.push(*rhs);
            }
            TermKind::Apply { .. }
            | TermKind::Forall { .. }
            | TermKind::Exists { .. }
            | TermKind::Let { .. }
            | TermKind::Match { .. } => return true,
            TermKind::Neg(inner) | TermKind::Not(inner) => stack.push(*inner),
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Distinct(args) => stack.extend(args.iter().copied()),
            TermKind::Ite(cond, then_term, else_term) => {
                stack.push(*cond);
                stack.push(*then_term);
                stack.push(*else_term);
            }
            TermKind::Xor(lhs, rhs) | TermKind::Implies(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            TermKind::Sub(lhs, rhs)
            | TermKind::Eq(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            _ => {}
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial atom (internal representation)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PolyAtom {
    poly: Polynomial,
    kind: AtomKind,
    /// `true` → atom appears positively; `false` → negated literal.
    positive: bool,
    /// `true` for a side-constraint manufactured by this module itself (the
    /// Euclidean `div`/`mod` witness identities — see
    /// [`TermPolyTranslator::ensure_divmod_witness`]), as opposed to an atom
    /// translated from the input problem. See its use in
    /// [`dispatch_nia_constraints`]'s `unsat_is_trustworthy` computation for
    /// why the distinction matters.
    synthetic: bool,
}

impl PolyAtom {
    /// An atom translated directly from an input assertion.
    fn from_problem(poly: Polynomial, kind: AtomKind, positive: bool) -> Self {
        Self {
            poly,
            kind,
            positive,
            synthetic: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Assertion-level translation (integer mode)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract polynomial atoms from a top-level assertion.
///
/// `incomplete` is set to `true` whenever some part of the assertion could
/// **not** be captured as a pure conjunction of polynomial atoms — an
/// unrecognized top-level connective (`Or`/`Not`/`Distinct`/`Ite`/…) or a
/// comparison whose operand does not translate to a polynomial (e.g. it
/// contains `Div`/`Mod`/an uninterpreted apply). The dispatcher must treat a
/// `Sat` verdict as untrustworthy once `incomplete` is set, because the solver
/// then only sees a strictly weaker (relaxed) subproblem. Reference: Z3's
/// nlsat/nlsat_solver.cpp only trusts a model for the full atom set.
fn extract_poly_atoms(
    term_id: TermId,
    manager: &TermManager,
    translator: &mut TermPolyTranslator<'_>,
    out: &mut Vec<PolyAtom>,
    incomplete: &mut bool,
) {
    // Iterative conjunction descent: an assertion is an implicit conjunction
    // and `(and A (and B …))` nests as deep as the input makes it. Conjuncts
    // are pushed in reverse so they pop left to right, the order the recursive
    // descent used (and the order atoms land in `out`).
    let mut worklist = vec![term_id];
    while let Some(current) = worklist.pop() {
        let Some(term) = manager.get(current) else {
            *incomplete = true;
            continue;
        };
        let kind = term.kind.clone();
        match &kind {
            TermKind::Eq(lhs, rhs) => {
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&lp, &rp),
                        AtomKind::Eq,
                        true,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Lt(lhs, rhs) => {
                // lhs < rhs → rhs - lhs > 0
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&rp, &lp),
                        AtomKind::Gt,
                        true,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Le(lhs, rhs) => {
                // lhs <= rhs → rhs - lhs >= 0 → NOT(rhs - lhs < 0)
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&rp, &lp),
                        AtomKind::Lt,
                        false,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Gt(lhs, rhs) => {
                // lhs > rhs → lhs - rhs > 0
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&lp, &rp),
                        AtomKind::Gt,
                        true,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Ge(lhs, rhs) => {
                // lhs >= rhs → NOT(lhs - rhs < 0)
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&lp, &rp),
                        AtomKind::Lt,
                        false,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::And(args) => worklist.extend(args.iter().rev().copied()),
            _ => {
                // Any other top-level shape (Or/Not/Distinct/Ite/…) belongs to the
                // boolean abstraction layer, not to this pure-conjunction fast
                // path. Dropping it silently would let the solver "prove" Sat on a
                // relaxed problem, so flag the extraction as incomplete instead.
                *incomplete = true;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NIA dispatch: public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Decide nonlinear integer arithmetic assertions.
///
/// Returns:
/// - `Some(NlDispatchResult::Unsat)` if the system is provably UNSAT,
/// - `Some(NlDispatchResult::Sat { .. })` with a witness if a model was found,
/// - `None` when neither could be established (the caller answers `unknown`).
///
/// This is the cell-decomposition core, and it is the only nonlinear
/// procedure in this crate that can prove *unsatisfiability* — so a caller
/// composing it with the search-based procedures (`nl_repair_search`,
/// `nl_ground_reduce`, which answer `Sat` or nothing) must run this one first
/// and treat its verdict as final. `oxiz-solver`'s `check_nlsat` is that
/// caller; keeping the composition there rather than here is what lets the
/// searches be budget-gated by solver configuration without this function's
/// contract depending on it.
///
/// Both linear and nonlinear assertions are passed so the solver has full context.
pub fn dispatch_nia_constraints(
    assertions: &[TermId],
    manager: &TermManager,
    integer_mode: bool,
) -> Option<NlDispatchResult> {
    let has_nl = assertions.iter().any(|&a| term_is_nonlinear(a, manager));
    if !has_nl {
        return None;
    }
    let has_unsupported_ops = assertions
        .iter()
        .any(|&a| contains_non_polynomial_ops(a, manager));

    let config = NiaConfig {
        enable_cutting_planes: true,
        ..NiaConfig::default()
    };
    let mut nia = NiaSolver::with_config(config);
    let mut translator = TermPolyTranslator::new(manager, &mut nia, integer_mode);

    let mut poly_atoms: Vec<PolyAtom> = Vec::new();
    let mut incomplete = false;
    for &assertion in assertions {
        extract_poly_atoms(
            assertion,
            manager,
            &mut translator,
            &mut poly_atoms,
            &mut incomplete,
        );
    }
    // Fold in the Euclidean `div`/`mod` witness identities accumulated while
    // translating the assertions above (see `TermPolyTranslator`'s
    // struct-level doc comment). These are theorems of the theory — for any
    // dividend and nonzero constant divisor an integer quotient/remainder
    // pair satisfying the identity always exists — so asserting them changes
    // nothing about whether the *problem* atoms are satisfiable; they exist
    // only to give the `div`/`mod` leaves translated above a meaning CAD can
    // reason about.
    poly_atoms.extend(translator.take_divmod_side_constraints());

    if poly_atoms.is_empty() {
        return None;
    }

    // An atom counts toward the univariate requirement below unless it is
    // one of the synthetic witness identities just folded in: those are
    // always safe to trust regardless of how many variables they mention,
    // because they assert nothing beyond "some q, r exist with this
    // relationship to variables already present elsewhere" — they cannot by
    // themselves be the reason a genuinely satisfiable problem is declared
    // `Unsat`. A non-synthetic (i.e. translated straight from the input)
    // atom mentioning more than one variable still has to clear the
    // univariate bar, unchanged from before.
    let unsat_is_trustworthy = !has_unsupported_ops
        && poly_atoms
            .iter()
            .all(|atom| atom.synthetic || atom.poly.is_univariate());
    // A `Sat` verdict is only sound when the solver saw the *entire* assertion
    // set as a conjunction of translatable atoms. If any top-level term was
    // dropped (a disjunction, an untranslatable operand, …) the solver worked
    // on a strictly weaker problem, so its model may violate the dropped
    // constraint — fall through to CDCL(T) instead of trusting Sat.
    let sat_is_trustworthy = !incomplete;

    for atom in &poly_atoms {
        let atom_id = translator
            .nlsat
            .nlsat_mut()
            .new_ineq_atom(atom.poly.clone(), atom.kind);
        let lit = translator
            .nlsat
            .nlsat()
            .atom_literal(atom_id, atom.positive);
        translator.nlsat.nlsat_mut().add_clause(vec![lit]);
    }

    match translator.nlsat.solve() {
        SolverResult::Sat if sat_is_trustworthy => {
            // Belt-and-suspenders: re-check the returned model against every
            // atom this dispatch itself asserted (problem atoms and
            // synthetic `div`/`mod` witnesses alike) before trusting `Sat`.
            // `sat_is_trustworthy` already established that `poly_atoms` is
            // the *entire* problem, so a model that satisfies all of them is
            // known-good independent of trusting the CAD search that found
            // it — cheap (linear in the atom count) and catches any future
            // soundness gap in the underlying solver before it reaches the
            // caller as a wrong answer, not just the specific one this
            // change fixed.
            if model_satisfies_atoms(&translator, &poly_atoms) {
                Some(NlDispatchResult::sat(witness_from_translator(&translator)))
            } else {
                None
            }
        }
        SolverResult::Unsat if unsat_is_trustworthy => Some(NlDispatchResult::Unsat),
        SolverResult::Sat | SolverResult::Unsat | SolverResult::Unknown => None,
    }
}

/// Recover a term-level witness from the integer solver's own model.
///
/// The translator holds the only map from problem terms to the polynomial
/// variable indices the solver reasons about, so this is where a solver-side
/// assignment becomes something a caller can print or check. Variables the
/// model left unassigned are simply absent — a partial witness, which the
/// caller must verify before publishing (see [`NlDispatchResult`]).
fn witness_from_translator(translator: &TermPolyTranslator<'_>) -> Interpretation {
    let mut interp = Interpretation::empty();
    let Some(model) = translator.nlsat.nlsat().get_model() else {
        return interp;
    };
    for (&term, &poly_var) in translator.var_cache() {
        if let Some(value) = model.arith_value(poly_var) {
            interp.pin_num(term, value.clone());
        }
    }
    interp
}

/// The real-arithmetic analogue of [`witness_from_translator`].
///
/// **All-or-nothing**, unlike its integer sibling. The NLSAT solver may now
/// witness a real variable with an exact algebraic number (`x = √2` for
/// `x² = 2`), which `Model::arith_value` reports as absent because no rational
/// stands for it. Pinning the *other* variables and leaving that one out would
/// produce a witness that is partial in a way the caller cannot detect and
/// that `crate::nl_eval::holds_under` would evaluate to `false` — turning a
/// correct `sat` into a re-check failure, and tripping the debug assertion in
/// `oxiz-solver`'s `check_nlsat` that exists to catch a dispatcher claiming
/// `Sat` on a witness that does not hold.
///
/// The contract is therefore: a witness this solver cannot represent simply
/// leaves the model unset. The verdict is unaffected — it is the dispatcher's,
/// decided by its own trust conditions — and an empty interpretation is
/// exactly what a caller needs to see to know it must not publish values.
///
/// The algebraic value itself is *not* lost: it is reported through the
/// separate exact-value channel that [`algebraic_witness_from_real_translator`]
/// builds, which is what `(get-model)` renders as `root-obj`. The two channels
/// stay disjoint exactly as described on [`NlDispatchResult`].
fn witness_from_real_translator(translator: &RealPolyTranslator<'_>) -> Interpretation {
    let mut interp = Interpretation::empty();
    let Some(model) = translator.nlsat.get_model() else {
        return interp;
    };
    for (&term, &poly_var) in &translator.var_cache {
        let Some(value) = model.arith_value(poly_var) else {
            return Interpretation::empty();
        };
        interp.pin_num(term, value.clone());
    }
    interp
}

/// The exact-value witness, for the real models [`witness_from_real_translator`]
/// has to decline.
///
/// Returns an **empty** map — meaning "this channel does not apply, use the
/// rational one" — in all of these cases:
///
/// * there is no model at all;
/// * every variable's value is rational (the rational channel covers it, and
///   populating both would blend two witness sources for one model);
/// * any variable is missing from the model, or carries an algebraic point
///   whose defining polynomial this code cannot put into `root-obj` normal
///   form (not univariate, or degree 0).
///
/// The last case is the load-bearing one: the map is **all-or-nothing**. A map
/// covering some variables would be completed with sort defaults by whatever
/// renders the model, publishing an assignment that satisfies nothing. Refusing
/// wholesale restores the modelless-but-correct `sat` this function replaced,
/// which is always a safe answer.
fn algebraic_witness_from_real_translator(
    translator: &RealPolyTranslator<'_>,
) -> FxHashMap<TermId, NlWitnessValue> {
    let empty = FxHashMap::default;
    let Some(model) = translator.nlsat.get_model() else {
        return empty();
    };
    let mut values: FxHashMap<TermId, NlWitnessValue> = FxHashMap::default();
    let mut saw_algebraic = false;
    for (&term, &poly_var) in &translator.var_cache {
        let Some(point) = model.arith_point(poly_var) else {
            return empty();
        };
        let value = match point {
            CadPoint::Rational(value) => NlWitnessValue::Rational(value.clone()),
            CadPoint::Algebraic {
                lo,
                hi,
                poly,
                index,
            } => {
                let Some(coefficients) = root_obj_coefficients(poly) else {
                    return empty();
                };
                saw_algebraic = true;
                NlWitnessValue::Algebraic(AlgebraicValue {
                    coefficients,
                    root_index: *index,
                    lower: lo.clone(),
                    upper: hi.clone(),
                })
            }
        };
        values.insert(term, value);
    }
    if saw_algebraic { values } else { empty() }
}

/// Put a `CadPoint::Algebraic` defining polynomial into `root-obj` normal
/// form: integer coefficients indexed by degree, primitive, positive leading
/// coefficient.
///
/// `None` when the polynomial is not a univariate polynomial of degree ≥ 1,
/// which is the only shape `root-obj` can spell. (An algebraic point's
/// polynomial is a *univariate specialisation* produced by the CAD lifting
/// step, so this should always succeed; refusing rather than asserting keeps
/// a future change to that invariant a lost model instead of a wrong one.)
///
/// # Why normalising after the fact is sound
///
/// `root_index` was determined against the polynomial as isolated, and the
/// three operations here — clearing denominators, dividing out the integer
/// content, negating for a positive leading coefficient — are multiplication
/// by a nonzero rational. That changes no root and reorders none, so the
/// index still selects the same number. See [`AlgebraicValue`].
fn root_obj_coefficients(poly: &Polynomial) -> Option<Vec<BigInt>> {
    let vars = poly.vars();
    let &[var] = vars.as_slice() else {
        return None;
    };
    let degree = poly.degree(var) as usize;
    if degree == 0 {
        return None;
    }

    // Rational coefficients, indexed by degree.
    let mut rational = vec![BigRational::zero(); degree + 1];
    for term in poly.terms() {
        let power = match term.monomial.vars() {
            [] => 0usize,
            [vp] if vp.var == var => vp.power as usize,
            // A monomial mentioning another variable contradicts `vars()`
            // above; treat it as unrenderable rather than dropping it.
            _ => return None,
        };
        let slot = rational.get_mut(power)?;
        *slot = &*slot + &term.coeff;
    }

    // Clear denominators: multiply through by the LCM of them.
    let mut denominator_lcm = BigInt::from(1);
    for coefficient in &rational {
        denominator_lcm = denominator_lcm.lcm(coefficient.denom());
    }
    let mut integral: Vec<BigInt> = rational
        .iter()
        .map(|coefficient| coefficient.numer() * (&denominator_lcm / coefficient.denom()))
        .collect();

    // Divide out the content.
    let mut content = BigInt::from(0);
    for coefficient in &integral {
        content = content.gcd(coefficient);
    }
    if content.is_zero() {
        // Every coefficient zero: the zero polynomial has no isolated root.
        return None;
    }
    for coefficient in &mut integral {
        *coefficient /= &content;
    }

    // Force a positive leading coefficient.
    let leading = integral.last()?;
    if leading.is_zero() {
        return None;
    }
    if leading < &BigInt::from(0) {
        for coefficient in &mut integral {
            *coefficient = -&*coefficient;
        }
    }
    Some(integral)
}

/// Re-evaluate every atom `dispatch_nia_constraints` asserted against the
/// solver's own returned model. `false` on a missing model, a variable the
/// model left unassigned, or an atom the model does not actually satisfy —
/// any of which means the `Sat` this model came with must not be trusted.
fn model_satisfies_atoms(translator: &TermPolyTranslator<'_>, poly_atoms: &[PolyAtom]) -> bool {
    let Some(model) = translator.nlsat.nlsat().get_model() else {
        return false;
    };
    let assignment: FxHashMap<u32, BigRational> = model
        .arith_values
        .iter()
        .map(|(&v, r)| (v, r.clone()))
        .collect();
    for atom in poly_atoms {
        let Some(value) = atom.poly.try_eval(&assignment) else {
            return false;
        };
        let sign_holds = match atom.kind {
            AtomKind::Eq => value.is_zero(),
            AtomKind::Lt => value < BigRational::zero(),
            AtomKind::Gt => value > BigRational::zero(),
            // `extract_poly_atoms`/`extract_real_poly_atoms` and the
            // Euclidean encoding above only ever build `Eq`/`Lt`/`Gt` atoms;
            // any other kind reaching here is unexpected, so refuse to trust
            // it rather than guessing at its semantics.
            _ => return false,
        };
        if sign_holds != atom.positive {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// NRA dispatch (real arithmetic)
// ─────────────────────────────────────────────────────────────────────────────

struct RealPolyTranslator<'a> {
    manager: &'a TermManager,
    nlsat: &'a mut NlsatSolver,
    var_cache: HashMap<TermId, u32>,
}

impl<'a> RealPolyTranslator<'a> {
    fn new(manager: &'a TermManager, nlsat: &'a mut NlsatSolver) -> Self {
        Self {
            manager,
            nlsat,
            var_cache: HashMap::new(),
        }
    }

    fn translate(&mut self, term_id: TermId) -> Option<Polynomial> {
        let manager = self.manager;
        translate_poly(manager, self, term_id)
    }

    fn get_or_create_var(&mut self, term_id: TermId) -> u32 {
        if let Some(&v) = self.var_cache.get(&term_id) {
            return v;
        }
        let v = self.nlsat.new_arith_var();
        self.var_cache.insert(term_id, v);
        v
    }
}

impl PolyVarSource for RealPolyTranslator<'_> {
    fn var_for(&mut self, term_id: TermId) -> u32 {
        self.get_or_create_var(term_id)
    }
}

/// Real-arithmetic analogue of [`extract_poly_atoms`]. See its documentation
/// for the meaning of `incomplete`.
fn extract_real_poly_atoms(
    term_id: TermId,
    manager: &TermManager,
    translator: &mut RealPolyTranslator<'_>,
    out: &mut Vec<PolyAtom>,
    incomplete: &mut bool,
) {
    // Iterative conjunction descent: an assertion is an implicit conjunction
    // and `(and A (and B …))` nests as deep as the input makes it. Conjuncts
    // are pushed in reverse so they pop left to right, the order the recursive
    // descent used (and the order atoms land in `out`).
    let mut worklist = vec![term_id];
    while let Some(current) = worklist.pop() {
        let Some(term) = manager.get(current) else {
            *incomplete = true;
            continue;
        };
        let kind = term.kind.clone();
        match &kind {
            TermKind::Eq(lhs, rhs) => {
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&lp, &rp),
                        AtomKind::Eq,
                        true,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Lt(lhs, rhs) => {
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&rp, &lp),
                        AtomKind::Gt,
                        true,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Le(lhs, rhs) => {
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&rp, &lp),
                        AtomKind::Lt,
                        false,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Gt(lhs, rhs) => {
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&lp, &rp),
                        AtomKind::Gt,
                        true,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::Ge(lhs, rhs) => {
                if let (Some(lp), Some(rp)) =
                    (translator.translate(*lhs), translator.translate(*rhs))
                {
                    out.push(PolyAtom::from_problem(
                        Polynomial::sub(&lp, &rp),
                        AtomKind::Lt,
                        false,
                    ));
                } else {
                    *incomplete = true;
                }
            }
            TermKind::And(args) => worklist.extend(args.iter().rev().copied()),
            _ => {
                *incomplete = true;
            }
        }
    }
}

/// Dispatch nonlinear real arithmetic assertions to `NlsatSolver`.
pub fn dispatch_nra_constraints(
    assertions: &[TermId],
    manager: &TermManager,
) -> Option<NlDispatchResult> {
    let has_nl = assertions.iter().any(|&a| term_is_nonlinear(a, manager));
    if !has_nl {
        return None;
    }

    let mut nlsat = NlsatSolver::new();
    let mut translator = RealPolyTranslator::new(manager, &mut nlsat);

    let mut poly_atoms: Vec<PolyAtom> = Vec::new();
    let mut incomplete = false;
    for &assertion in assertions {
        extract_real_poly_atoms(
            assertion,
            manager,
            &mut translator,
            &mut poly_atoms,
            &mut incomplete,
        );
    }

    if poly_atoms.is_empty() {
        return None;
    }

    let unsat_is_trustworthy = poly_atoms.iter().all(|atom| atom.kind != AtomKind::Eq);
    // See `dispatch_nia_constraints`: trusting Sat under a dropped (relaxed)
    // constraint is unsound, so only accept Sat when extraction was complete.
    let sat_is_trustworthy = !incomplete;

    for atom in &poly_atoms {
        let atom_id = translator.nlsat.new_ineq_atom(atom.poly.clone(), atom.kind);
        let lit = translator.nlsat.atom_literal(atom_id, atom.positive);
        translator.nlsat.add_clause(vec![lit]);
    }

    match translator.nlsat.solve() {
        SolverResult::Sat if sat_is_trustworthy => {
            // Exactly one of the two witness channels, never both — see
            // `NlDispatchResult`. The algebraic one applies only when some
            // variable's value is irrational, and then it carries the whole
            // model; otherwise the rational one does, unchanged.
            let algebraic = algebraic_witness_from_real_translator(&translator);
            Some(if algebraic.is_empty() {
                NlDispatchResult::sat(witness_from_real_translator(&translator))
            } else {
                NlDispatchResult::sat_algebraic(algebraic)
            })
        }
        SolverResult::Unsat if unsat_is_trustworthy => Some(NlDispatchResult::Unsat),
        SolverResult::Sat | SolverResult::Unsat | SolverResult::Unknown => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NlsatTheory – Theory trait wrapper
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct NlsatContextState {
    level: usize,
}

enum NlsatSolverWrapper {
    Real(NlsatSolver),
    Integer(NiaSolver),
}

impl core::fmt::Debug for NlsatSolverWrapper {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Real(_) => write!(f, "NlsatSolverWrapper::Real(..)"),
            Self::Integer(_) => write!(f, "NlsatSolverWrapper::Integer(..)"),
        }
    }
}

impl NlsatSolverWrapper {
    fn new(integer: bool) -> Self {
        if integer {
            Self::Integer(NiaSolver::new())
        } else {
            Self::Real(NlsatSolver::new())
        }
    }

    fn solve(&mut self) -> SolverResult {
        match self {
            Self::Real(s) => s.solve(),
            Self::Integer(s) => s.solve(),
        }
    }
}

/// NLSAT Theory Solver for nonlinear arithmetic.
///
/// Supports both real (QF_NRA) and integer (QF_NIA) nonlinear arithmetic.
/// Full constraint translation happens in `dispatch_nia_constraints` /
/// `dispatch_nra_constraints`; this wrapper integrates with the `Theory` trait.
#[derive(Debug)]
pub struct NlsatTheory {
    solver: NlsatSolverWrapper,
    context_stack: Vec<NlsatContextState>,
    is_integer: bool,
    last_result: Option<SolverResult>,
    asserted_terms: Vec<TermId>,
}

impl NlsatTheory {
    /// Create a new NLSAT theory solver.
    ///
    /// * `integer` – true for QF_NIA, false for QF_NRA.
    pub fn new(integer: bool) -> Self {
        Self {
            solver: NlsatSolverWrapper::new(integer),
            context_stack: Vec::new(),
            is_integer: integer,
            last_result: None,
            asserted_terms: Vec::new(),
        }
    }
}

impl Theory for NlsatTheory {
    fn id(&self) -> TheoryId {
        if self.is_integer {
            TheoryId::NIA
        } else {
            TheoryId::NRA
        }
    }

    fn name(&self) -> &str {
        if self.is_integer { "NIA" } else { "NRA" }
    }

    fn can_handle(&self, _term: TermId) -> bool {
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        self.asserted_terms.push(term);
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        self.asserted_terms.push(term);
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        let result = self.solver.solve();
        self.last_result = Some(result);
        match result {
            SolverResult::Sat => Ok(TheoryResult::Sat),
            SolverResult::Unsat => {
                let conflict = self.asserted_terms.clone();
                Ok(TheoryResult::Unsat(conflict))
            }
            SolverResult::Unknown => Ok(TheoryResult::Unknown),
        }
    }

    fn push(&mut self) {
        self.context_stack.push(NlsatContextState {
            level: self.asserted_terms.len(),
        });
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            self.asserted_terms.truncate(state.level);
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.is_integer);
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        Vec::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::TermManager;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    // ── `root-obj` polynomial normal form ─────────────────────────────────────

    /// Build a univariate polynomial in variable 0 from `numerator/denominator`
    /// pairs indexed by degree.
    fn poly(coefficients: &[(i64, i64)]) -> Polynomial {
        let rationals: Vec<BigRational> = coefficients
            .iter()
            .map(|&(n, d)| BigRational::new(n.into(), d.into()))
            .collect();
        Polynomial::univariate(0, &rationals)
    }

    fn ints(values: &[i64]) -> Vec<BigInt> {
        values.iter().copied().map(BigInt::from).collect()
    }

    /// The plain case: `x² − 2` is already in normal form.
    #[test]
    fn root_obj_coefficients_keeps_a_primitive_polynomial() {
        assert_eq!(
            root_obj_coefficients(&poly(&[(-2, 1), (0, 1), (1, 1)])),
            Some(ints(&[-2, 0, 1]))
        );
    }

    /// Denominators are cleared: `x² − 1/2` becomes `2x² − 1`, which is what
    /// `z3 4.15.4` prints for `x² = 1/2` — `(root-obj (+ (* 2 (^ x 2)) (- 1)) 1)`.
    #[test]
    fn root_obj_coefficients_clears_denominators() {
        assert_eq!(
            root_obj_coefficients(&poly(&[(-1, 2), (0, 1), (1, 1)])),
            Some(ints(&[-1, 0, 2]))
        );
    }

    /// The integer content is divided out: `2x² − 4` becomes `x² − 2`, which
    /// is what z3 prints for `2x² = 4`.
    #[test]
    fn root_obj_coefficients_divides_out_the_content() {
        assert_eq!(
            root_obj_coefficients(&poly(&[(-4, 1), (0, 1), (2, 1)])),
            Some(ints(&[-2, 0, 1]))
        );
    }

    /// A negative leading coefficient is negated away: `−x² + 2` becomes
    /// `x² − 2`, which is what z3 prints for `2 − x² = 0`.
    #[test]
    fn root_obj_coefficients_forces_a_positive_leading_coefficient() {
        assert_eq!(
            root_obj_coefficients(&poly(&[(2, 1), (0, 1), (-1, 1)])),
            Some(ints(&[-2, 0, 1]))
        );
    }

    /// All three normalisations at once: `−(3/2)x² + 3` → `x² − 2`.
    #[test]
    fn root_obj_coefficients_composes_all_three_normalisations() {
        assert_eq!(
            root_obj_coefficients(&poly(&[(3, 1), (0, 1), (-3, 2)])),
            Some(ints(&[-2, 0, 1]))
        );
    }

    /// Interior zero coefficients are preserved by *degree index*, not
    /// squeezed out — the renderer relies on the index being the exponent.
    #[test]
    fn root_obj_coefficients_are_indexed_by_degree() {
        // x⁵ − 3x − 1
        assert_eq!(
            root_obj_coefficients(&poly(&[(-1, 1), (-3, 1), (0, 1), (0, 1), (0, 1), (1, 1)])),
            Some(ints(&[-1, -3, 0, 0, 0, 1]))
        );
    }

    /// A polynomial `root-obj` cannot spell is refused rather than mangled:
    /// the caller then reports no model at all, which is always safe.
    #[test]
    fn root_obj_coefficients_refuses_unrenderable_polynomials() {
        // Constant: no roots to index.
        assert_eq!(root_obj_coefficients(&poly(&[(7, 1)])), None);
        // Zero polynomial: every point is a root, none is isolated.
        assert_eq!(root_obj_coefficients(&Polynomial::zero()), None);
        // Multivariate: `x·y − 1` has no univariate `root-obj` form.
        use oxiz_math::polynomial::{Monomial, MonomialOrder, Term};
        let xy = Polynomial::from_terms(
            vec![
                Term::new(rat(1), Monomial::from_powers([(0, 1), (1, 1)])),
                Term::new(rat(-1), Monomial::unit()),
            ],
            MonomialOrder::default(),
        );
        assert_eq!(root_obj_coefficients(&xy), None);
    }

    // ── Theory trait tests ────────────────────────────────────────────────────

    #[test]
    fn test_nlsat_theory_new() {
        let theory_nia = NlsatTheory::new(true);
        assert_eq!(theory_nia.id(), TheoryId::NIA);
        assert_eq!(theory_nia.name(), "NIA");
        assert!(theory_nia.is_integer);

        let theory_nra = NlsatTheory::new(false);
        assert_eq!(theory_nra.id(), TheoryId::NRA);
        assert_eq!(theory_nra.name(), "NRA");
        assert!(!theory_nra.is_integer);
    }

    #[test]
    fn test_nlsat_theory_push_pop() {
        let mut theory = NlsatTheory::new(false);
        assert_eq!(theory.context_stack.len(), 0);
        theory.push();
        assert_eq!(theory.context_stack.len(), 1);
        theory.push();
        assert_eq!(theory.context_stack.len(), 2);
        theory.pop();
        assert_eq!(theory.context_stack.len(), 1);
        theory.pop();
        assert_eq!(theory.context_stack.len(), 0);
    }

    #[test]
    fn test_nlsat_theory_reset() {
        let mut theory = NlsatTheory::new(false);
        let term = TermId::new(1);
        let _ = theory.assert_true(term);
        assert!(!theory.asserted_terms.is_empty());
        theory.reset();
        assert!(theory.asserted_terms.is_empty());
        assert!(theory.context_stack.is_empty());
    }

    #[test]
    fn test_nlsat_theory_can_handle() {
        let theory = NlsatTheory::new(false);
        assert!(theory.can_handle(TermId::new(1)));
    }

    #[test]
    fn test_nlsat_theory_check_placeholder() {
        let mut theory = NlsatTheory::new(false);
        let result = theory.check().expect("check should succeed");
        assert!(matches!(result, TheoryResult::Sat));
    }

    // ── Translator unit tests ──────────────────────────────────────────────────

    #[test]
    fn test_translator_constant() {
        let mut manager = TermManager::new();
        let five = manager.mk_int(5);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(five).expect("constant should translate");
        assert!(poly.is_constant());
        assert_eq!(poly.constant_value(), rat(5));
    }

    #[test]
    fn test_translator_variable() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(x).expect("variable should translate");
        assert!(poly.is_linear());
        assert_eq!(poly.num_terms(), 1);
    }

    #[test]
    fn test_translator_add() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add(vec![x, y]);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(sum).expect("add should translate");
        assert_eq!(poly.num_terms(), 2);
    }

    #[test]
    fn test_translator_mul_vars() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let product = manager.mk_mul(vec![x, y]);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(product).expect("mul should translate");
        // x * y is a single monomial of degree 2
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(poly.total_degree(), 2);
    }

    #[test]
    fn test_translator_square() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let square = manager.mk_mul(vec![x, x]);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(square).expect("x*x should translate");
        // x^2 — single term, degree 2
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(poly.total_degree(), 2);
    }

    #[test]
    fn test_translator_neg() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let neg_x = manager.mk_neg(x);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(neg_x).expect("neg should translate");
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(poly.leading_coeff(), rat(-1));
    }

    #[test]
    fn test_translator_sub() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let two = manager.mk_int(2);
        let x_minus_2 = manager.mk_sub(x, two);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t.translate(x_minus_2).expect("sub should translate");
        // x - 2 → two terms: x and -2
        assert_eq!(poly.num_terms(), 2);
    }

    #[test]
    fn test_translator_triple_product() {
        // (* x y z) — degree-3 monomial
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);
        let triple = manager.mk_mul(vec![x, y, z]);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t
            .translate(triple)
            .expect("triple product should translate");
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(poly.total_degree(), 3);
    }

    #[test]
    fn test_translator_factored_product() {
        // (* (+ x 1) (- y 2)) → xy - 2x + y - 2
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let xp1 = manager.mk_add(vec![x, one]);
        let ym2 = manager.mk_sub(y, two);
        let product = manager.mk_mul(vec![xp1, ym2]);
        let mut nia = NiaSolver::new();
        let mut t = TermPolyTranslator::new(&manager, &mut nia, true);
        let poly = t
            .translate(product)
            .expect("factored product should translate");
        // (x+1)(y-2) = xy - 2x + y - 2  → 4 terms
        assert_eq!(poly.num_terms(), 4);
        assert_eq!(poly.total_degree(), 2);
    }

    // ── term_is_nonlinear tests ────────────────────────────────────────────────

    #[test]
    fn test_term_is_nonlinear_square() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let square = manager.mk_mul(vec![x, x]);
        assert!(term_is_nonlinear(square, &manager));
    }

    #[test]
    fn test_term_is_nonlinear_product_xy() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let xy = manager.mk_mul(vec![x, y]);
        assert!(term_is_nonlinear(xy, &manager));
    }

    #[test]
    fn test_term_is_nonlinear_linear_is_false() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let three = manager.mk_int(3);
        let three_x = manager.mk_mul(vec![three, x]);
        assert!(!term_is_nonlinear(three_x, &manager));
    }

    #[test]
    fn test_term_is_nonlinear_constant() {
        let mut manager = TermManager::new();
        let c = manager.mk_int(42);
        assert!(!term_is_nonlinear(c, &manager));
    }

    // ── dispatch integration tests ─────────────────────────────────────────────

    #[test]
    fn test_dispatch_nia_x_squared_eq_4_sat() {
        // x * x = 4 → SAT (x=2 or x=-2)
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let square = manager.mk_mul(vec![x, x]);
        let four = manager.mk_int(4);
        let eq = manager.mk_eq(square, four);
        let result = dispatch_nia_constraints(&[eq], &manager, true);
        // SAT or Unknown (unknown means solver fell through)
        assert!(
            matches!(result, Some(NlDispatchResult::Sat { .. }) | None),
            "x*x=4 should be SAT or unknown, got {:?}",
            result
        );
    }

    #[test]
    fn test_dispatch_nia_x_squared_neg_unsat() {
        // x * x = -1 → UNSAT (no integer square is negative)
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let square = manager.mk_mul(vec![x, x]);
        let neg_one = manager.mk_int(-1);
        let eq = manager.mk_eq(square, neg_one);
        let result = dispatch_nia_constraints(&[eq], &manager, true);
        assert!(
            matches!(result, Some(NlDispatchResult::Unsat) | None),
            "x*x=-1 should be UNSAT or unknown, got {:?}",
            result
        );
    }

    #[test]
    fn test_dispatch_nra_x_squared_neg_unsat() {
        // x * x < 0 → UNSAT (no real square is negative)
        let mut manager = TermManager::new();
        let real_sort = manager.sorts.real_sort;
        let x = manager.mk_var("x", real_sort);
        let square = manager.mk_mul(vec![x, x]);
        let zero = manager.mk_int(0);
        let lt = manager.mk_lt(square, zero);
        let result = dispatch_nra_constraints(&[lt], &manager);
        assert!(
            matches!(result, Some(NlDispatchResult::Unsat) | None),
            "x*x<0 should be UNSAT or unknown, got {:?}",
            result
        );
    }
}
