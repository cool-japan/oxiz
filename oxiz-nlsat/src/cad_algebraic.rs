//! Exact real-algebraic sample points for the standalone full-CAD path.
//!
//! The base decomposition and lifting in [`crate::cad`] partition the real line
//! at the roots of the projection polynomials. A rational midpoint of an
//! isolating interval is **not** the root it brackets, so collapsing an
//! irrational root to its interval midpoint makes the zero-dimensional (root)
//! cells lose sign-invariance: a polynomial that vanishes exactly at the true
//! root is generally non-zero at the midpoint. This module fixes that by
//! representing each root exactly — as a rational when the root is rational, and
//! as a [`CadPoint::Algebraic`] carrying its defining polynomial and isolating
//! interval otherwise — and by evaluating polynomial signs *at the root* via
//! interval refinement rather than by substituting a rational approximation.
//!
//! Only the open one-dimensional cells strictly *between* consecutive roots may
//! be sampled at an arbitrary rational; those samples are chosen from strictly
//! inside the gap between the neighbouring roots' isolating intervals, so they
//! are provably in the correct cell.
//!
//! Residual (honestly unsolved here): lifting through *several* variables still
//! substitutes a rational approximation of a lower-level algebraic sample point
//! into the next-level polynomials (see `CadPoint::approximate` used by
//! `lift_single_cell`). Doing that exactly requires full multivariate
//! real-algebraic-number arithmetic; the base-level decomposition produced here
//! is exact, and higher-level lifting remains an approximation.
//!
//! The same missing capability bounds the NLSAT search's use of these points:
//! `crate::solver::witness_algebraic` may hold at most **one** algebraic value
//! at a time, because a polynomial over two algebraic points is exactly the
//! multivariate arithmetic that does not exist here. See that module for how
//! the restriction is enforced and why exceeding it would be a wrong `sat`.
//!
//! Reference: Z3's `nlsat`/`algebraic_numbers` sign-at-root evaluation.

use crate::cad::{CadPoint, SturmSequence};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::gcd::PolynomialGcd;
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;

/// Safety cap on interval bisections; well-separated distinct real roots reach
/// disjoint isolating intervals far sooner than this.
const MAX_REFINE_STEPS: usize = 4096;

/// Above this magnitude the rational-root-theorem divisor search is skipped
/// (the root is then kept as an exact algebraic number rather than spending
/// unbounded time enumerating divisors of a huge integer).
const MAX_RATIONAL_ROOT_MAGNITUDE: i64 = 1_000_000;

/// A single real root of a projection polynomial, retained as an exact
/// algebraic object (defining polynomial + isolating interval) rather than a
/// rational approximation.
#[derive(Debug, Clone)]
pub(crate) struct RootSample {
    /// Lower bound of the isolating interval (equals `hi` for a rational root).
    pub(crate) lo: BigRational,
    /// Upper bound of the isolating interval.
    pub(crate) hi: BigRational,
    /// A polynomial that vanishes at this root (the univariate specialization it
    /// was isolated from).
    pub(crate) poly: Polynomial,
    /// 1-based index of this root among `poly`'s real roots (left to right).
    pub(crate) index: u32,
    /// The variable the root is a value of.
    pub(crate) var: Var,
}

impl RootSample {
    /// Whether the root is exactly the rational `lo == hi`.
    fn is_rational(&self) -> bool {
        self.lo == self.hi
    }

    /// Convert to a [`CadPoint`]: a rational point for a rational root, an
    /// exact algebraic point (defining polynomial + isolating interval) for an
    /// irrational one.
    pub(crate) fn to_point(&self) -> CadPoint {
        if self.is_rational() {
            CadPoint::rational(self.lo.clone())
        } else {
            CadPoint::algebraic(
                self.lo.clone(),
                self.hi.clone(),
                self.poly.clone(),
                self.index,
            )
        }
    }

    /// Halve the isolating interval, keeping the half that still contains the
    /// root (determined by counting `poly`'s roots in the lower half). For an
    /// irrational root the midpoint is never the root, so this is unambiguous.
    fn refine(&mut self) {
        if self.is_rational() {
            return;
        }
        let mid = (&self.lo + &self.hi) / BigRational::from_integer(2.into());
        let sturm = SturmSequence::new(&self.poly, self.var);
        if sturm.count_roots_in(&self.lo, &mid) >= 1 {
            self.hi = mid;
        } else {
            self.lo = mid;
        }
    }

    /// Whether the isolating intervals of `self` and `other` fail to be
    /// **strictly** separated — sharing an endpoint counts.
    ///
    /// Touching must count. Two brackets meeting at `x` (say `[-3, 0]` around
    /// `−√2` and the degenerate `[0, 0]` of the rational root `0`) make
    /// [`open_sample`] return `(0 + 0)/2 = 0`, which is the *root*, not a point
    /// of the open cell between them — so the cell `(−√2, 0)` would get no
    /// representative at all. A caller enumerating one point per cell (see
    /// `crate::solver::witness_algebraic`) would then miss every solution
    /// living there and wrongly conclude that none exists. Distinct real roots
    /// have a positive separation, so demanding strict disjointness always
    /// terminates.
    fn overlaps(&self, other: &RootSample) -> bool {
        self.hi >= other.lo && other.hi >= self.lo
    }
}

/// Decide whether two roots are the same real number.
///
/// Two isolated roots are equal iff their common divisor `gcd(p, q)` has a root
/// inside the overlap of their isolating intervals: such a common root lies in
/// each interval and each interval isolates a single root, so it must be both.
fn roots_equal(a: &RootSample, b: &RootSample) -> bool {
    // Fast path: two rational roots.
    if a.is_rational() && b.is_rational() {
        return a.lo == b.lo;
    }
    let lo = if a.lo >= b.lo {
        a.lo.clone()
    } else {
        b.lo.clone()
    };
    let hi = if a.hi <= b.hi {
        a.hi.clone()
    } else {
        b.hi.clone()
    };
    if lo > hi {
        return false;
    }
    let mut engine = PolynomialGcd::new();
    let g = engine.gcd(&a.poly, &b.poly);
    if g.is_zero() || g.is_constant() || g.degree(a.var) == 0 {
        return false;
    }
    let sturm = SturmSequence::new(&g, a.var);
    sturm.count_roots_in(&lo, &hi) >= 1
}

/// Isolate every real root of `polys` (as a value of `var`) into a sorted,
/// deduplicated list of exact [`RootSample`]s with pairwise-disjoint isolating
/// intervals.
pub(crate) fn isolate_root_samples(polys: &[Polynomial], var: Var) -> Vec<RootSample> {
    let mut samples: Vec<RootSample> = Vec::new();
    for poly in polys {
        if poly.degree(var) == 0 {
            continue;
        }
        let sturm = SturmSequence::new(poly, var);
        for (idx, (lo, hi)) in sturm.isolate_roots().into_iter().enumerate() {
            // Collapse an exactly-rational root to a degenerate interval so it
            // is represented (and, crucially, substituted during lifting)
            // exactly rather than as a rational midpoint approximation.
            let (lo, hi) = match try_exact_rational_root(poly, var, &lo, &hi) {
                Some(r) => (r.clone(), r),
                None => (lo, hi),
            };
            samples.push(RootSample {
                lo,
                hi,
                poly: poly.clone(),
                index: (idx + 1) as u32,
                var,
            });
        }
    }
    normalize_roots(samples)
}

/// Deduplicate equal roots, refine intervals to be pairwise **strictly**
/// disjoint, and sort ascending. After this the `i`-th and `(i+1)`-th roots
/// satisfy `roots[i].hi < roots[i+1].lo`, so any rational in the open interval
/// `(hi, lo)` — the midpoint [`open_sample`] picks, in particular — sits
/// strictly between the two real roots. Merely *touching* brackets would break
/// that: see [`RootSample::overlaps`].
fn normalize_roots(samples: Vec<RootSample>) -> Vec<RootSample> {
    // Deduplicate equal roots.
    let mut unique: Vec<RootSample> = Vec::new();
    'outer: for s in samples {
        for u in &unique {
            if roots_equal(u, &s) {
                continue 'outer;
            }
        }
        unique.push(s);
    }

    // Refine until all isolating intervals are pairwise disjoint. The remaining
    // roots are distinct reals, so a positive minimum separation exists and the
    // loop terminates well before the safety cap.
    let mut steps = 0;
    loop {
        let mut refined = false;
        for i in 0..unique.len() {
            for j in (i + 1)..unique.len() {
                if unique[i].overlaps(&unique[j]) {
                    unique[i].refine();
                    unique[j].refine();
                    refined = true;
                }
            }
        }
        steps += 1;
        if !refined || steps > MAX_REFINE_STEPS {
            break;
        }
    }

    unique.sort_by(|a, b| a.lo.cmp(&b.lo));
    unique
}

/// Sign (`-1`, `0`, `1`) of a univariate polynomial `poly` (in `var`) at the
/// rational `r`.
pub(crate) fn sign_at_rational(poly: &Polynomial, var: Var, r: &BigRational) -> i8 {
    let mut map = FxHashMap::default();
    map.insert(var, r.clone());
    match poly.try_eval(&map) {
        Some(v) => rational_sign(&v),
        None => 0,
    }
}

/// Sign (`-1`, `0`, `1`) of a univariate polynomial `q` (in `var`) at the
/// algebraic number `alpha`, the root of `defining` isolated by `[lo, hi]`.
///
/// Sound: `q(alpha) = 0` is detected exactly via `gcd(defining, q)` having a
/// root in `[lo, hi]`; otherwise the interval is refined until `q` has no root
/// inside it, at which point `q`'s sign is constant on the interval and is read
/// off at the midpoint.
///
/// On refinement-budget exhaustion this returns a *midpoint approximation* of
/// the sign, which is a guess, not a proof. Callers whose decision must be
/// exact (accepting a satisfying witness, refuting a candidate point) must use
/// [`try_sign_at_algebraic`] instead and treat `None` as "cannot decide".
pub(crate) fn sign_at_algebraic(
    q: &Polynomial,
    var: Var,
    defining: &Polynomial,
    lo: &BigRational,
    hi: &BigRational,
) -> i8 {
    sign_at_algebraic_inner(q, var, defining, lo, hi).0
}

/// [`sign_at_algebraic`] without the approximate fallback: `None` means the
/// refinement budget ran out before the sign could be *proved*, so nothing at
/// all is known about it.
pub(crate) fn try_sign_at_algebraic(
    q: &Polynomial,
    var: Var,
    defining: &Polynomial,
    lo: &BigRational,
    hi: &BigRational,
) -> Option<i8> {
    let (sign, exhausted) = sign_at_algebraic_inner(q, var, defining, lo, hi);
    (!exhausted).then_some(sign)
}

/// Shared implementation: `(sign, exhausted)`. When `exhausted` is `true` the
/// sign is the midpoint approximation of the last refined interval and carries
/// no proof.
fn sign_at_algebraic_inner(
    q: &Polynomial,
    var: Var,
    defining: &Polynomial,
    lo: &BigRational,
    hi: &BigRational,
) -> (i8, bool) {
    if q.is_zero() {
        return (0, false);
    }
    if q.degree(var) == 0 {
        // Constant in `var`: sign is the constant's sign.
        return (sign_at_rational(q, var, lo), false);
    }

    // Exact test for q(alpha) == 0.
    let mut engine = PolynomialGcd::new();
    let g = engine.gcd(defining, q);
    if !g.is_zero() && !g.is_constant() && g.degree(var) > 0 {
        let sturm = SturmSequence::new(&g, var);
        if sturm.count_roots_in(lo, hi) >= 1 {
            return (0, false);
        }
    }

    // q(alpha) != 0: refine [a, b] (isolating alpha for `defining`) until `q`
    // has no root inside, then q is sign-invariant there.
    let mut a = lo.clone();
    let mut b = hi.clone();
    let def_sturm = SturmSequence::new(defining, var);
    for _ in 0..MAX_REFINE_STEPS {
        let q_sturm = SturmSequence::new(q, var);
        if q_sturm.count_roots_in(&a, &b) == 0 {
            let mid = (&a + &b) / BigRational::from_integer(2.into());
            return (sign_at_rational(q, var, &mid), false);
        }
        let mid = (&a + &b) / BigRational::from_integer(2.into());
        if def_sturm.count_roots_in(&a, &mid) >= 1 {
            b = mid;
        } else {
            a = mid;
        }
    }

    // Unreachable for well-formed input; fall back to the (approximate) midpoint
    // sign rather than fabricating a zero — flagged as unproven.
    let mid = (&a + &b) / BigRational::from_integer(2.into());
    (sign_at_rational(q, var, &mid), true)
}

/// Try to recover an *exact* rational root of `poly` (in `var`) inside the
/// isolating interval `(lo, hi]` via the rational root theorem. Returns `None`
/// when the root is irrational or the coefficients are too large to enumerate
/// divisors economically (in which case the caller keeps the exact algebraic
/// representation).
///
/// # The interval is half-open, and that matters
///
/// Sturm isolation counts roots in `(a, b]`, so a rational sitting exactly at
/// `lo` is *not* the root this bracket isolates — it is the neighbouring
/// bracket's. Accepting it collapses this bracket onto the wrong number and
/// silently loses the root that actually lives here: for `x³ − 2x`, `−√2`'s
/// bracket can be `(-3, 0]`, and admitting the rational root `0` at its lower
/// endpoint replaced `−√2` with a duplicate of `0`, which deduplication then
/// discarded — leaving the polynomial with two of its three real roots.
/// Anything enumerating cells from these samples (`crate::cad`'s lifting,
/// `crate::solver::witness_algebraic`'s candidate walk) would then never look
/// at `−√2` or the cells beside it.
fn try_exact_rational_root(
    poly: &Polynomial,
    var: Var,
    lo: &BigRational,
    hi: &BigRational,
) -> Option<BigRational> {
    let coeffs = univariate_rational_coeffs(poly, var)?;
    if coeffs.len() < 2 {
        return None;
    }
    // Clear denominators to obtain integer coefficients a_0 .. a_n.
    let mut denom_lcm = BigInt::one();
    for c in &coeffs {
        denom_lcm = lcm(&denom_lcm, c.denom());
    }
    let int_coeffs: Vec<BigInt> = coeffs
        .iter()
        .map(|c| c.numer() * (&denom_lcm / c.denom()))
        .collect();

    let a0 = &int_coeffs[0];
    let an = int_coeffs.last()?;
    if an.is_zero() {
        return None;
    }

    // x = 0 is a root iff the constant term is zero.
    if a0.is_zero() {
        let zero = BigRational::zero();
        if &zero > lo && &zero <= hi {
            return Some(zero);
        }
    }

    let cap = BigInt::from(MAX_RATIONAL_ROOT_MAGNITUDE);
    if a0.abs() > cap || an.abs() > cap {
        return None; // too large: keep the algebraic representation
    }

    let p_divs = divisors(&a0.abs());
    let q_divs = divisors(&an.abs());
    for p in &p_divs {
        for q in &q_divs {
            for sign in [BigInt::one(), -BigInt::one()] {
                let cand = BigRational::new(&sign * p, q.clone());
                if &cand <= lo || &cand > hi {
                    continue; // outside the half-open bracket `(lo, hi]`
                }
                if sign_at_rational(poly, var, &cand) == 0 {
                    return Some(cand);
                }
            }
        }
    }
    None
}

/// Univariate rational coefficients of `poly` in `var` (index = power), or
/// `None` if `poly` is not univariate in `var`.
fn univariate_rational_coeffs(poly: &Polynomial, var: Var) -> Option<Vec<BigRational>> {
    let degree = poly.degree(var) as usize;
    let mut coeffs = vec![BigRational::zero(); degree + 1];
    for term in poly.terms() {
        let mut power = 0usize;
        for vp in term.monomial.vars() {
            if vp.var == var {
                power = vp.power as usize;
            } else {
                return None; // another variable present: not univariate
            }
        }
        if power >= coeffs.len() {
            return None;
        }
        coeffs[power] += term.coeff.clone();
    }
    Some(coeffs)
}

/// Positive divisors of a non-negative integer (bounded by construction — the
/// caller caps the magnitude via [`MAX_RATIONAL_ROOT_MAGNITUDE`]).
fn divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![BigInt::one()];
    }
    let mut out = Vec::new();
    let mut i = BigInt::one();
    while &i * &i <= *n {
        if (n % &i).is_zero() {
            out.push(i.clone());
            let other = n / &i;
            if other != i {
                out.push(other);
            }
        }
        i += BigInt::one();
    }
    out
}

/// Least common multiple of two integers (both treated as non-negative).
fn lcm(a: &BigInt, b: &BigInt) -> BigInt {
    if a.is_zero() || b.is_zero() {
        return BigInt::one();
    }
    let g = gcd(a.abs(), b.abs());
    (a * b).abs() / g
}

/// Euclidean GCD of two non-negative integers.
fn gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let t = &a % &b;
        a = b;
        b = t;
    }
    a.abs()
}

/// Sign of `x` as `-1`, `0`, or `1`.
fn rational_sign(x: &BigRational) -> i8 {
    if x.is_zero() {
        0
    } else if x > &BigRational::zero() {
        1
    } else {
        -1
    }
}

impl CadPoint {
    /// Exact sign (`-1`, `0`, `1`) of the univariate polynomial `poly` (in
    /// `var`) evaluated at this point. Rational points substitute directly;
    /// algebraic points use `sign_at_algebraic` so a polynomial vanishing at
    /// the true root reports sign `0` and never a spurious non-zero from a
    /// rational approximation.
    pub fn sign_of(&self, poly: &Polynomial, var: Var) -> i8 {
        match self {
            CadPoint::Rational(r) => sign_at_rational(poly, var, r),
            CadPoint::Algebraic {
                lo,
                hi,
                poly: defining,
                ..
            } => sign_at_algebraic(poly, var, defining, lo, hi),
        }
    }

    /// [`Self::sign_of`] with no approximate fallback: `None` means the sign
    /// could not be *proved* within the refinement budget.
    ///
    /// Any decision that a wrong answer would ride on — accepting a point as a
    /// satisfying witness, refuting it as part of an emptiness proof, deciding
    /// a theory atom's truth value — must go through this and treat `None` as
    /// "unknown", never as a sign. `sign_of`'s budget-exhaustion fallback is a
    /// midpoint guess and would turn a fabricated sign into a fabricated
    /// verdict.
    pub fn try_sign_of(&self, poly: &Polynomial, var: Var) -> Option<i8> {
        match self {
            CadPoint::Rational(r) => Some(sign_at_rational(poly, var, r)),
            CadPoint::Algebraic {
                lo,
                hi,
                poly: defining,
                ..
            } => try_sign_at_algebraic(poly, var, defining, lo, hi),
        }
    }
}

/// A rational strictly between two consecutive (strictly disjoint) roots, or
/// strictly outside the outermost root when a neighbour is absent. With
/// strictly disjoint intervals `left.hi < right.lo` (guaranteed by
/// [`normalize_roots`]), the midpoint of `[left.hi, right.lo]` is `> left` and
/// `< right`.
pub(crate) fn open_sample(left: Option<&RootSample>, right: Option<&RootSample>) -> BigRational {
    match (left, right) {
        (None, None) => BigRational::zero(),
        (None, Some(r)) => &r.lo - BigRational::one(),
        (Some(l), None) => &l.hi + BigRational::one(),
        (Some(l), Some(r)) => (&l.hi + &r.lo) / BigRational::from_integer(2.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(num_bigint::BigInt::from(n))
    }

    /// x^2 - 2: roots ±√2 are irrational, so they must be represented as exact
    /// algebraic points, and a polynomial vanishing there must report sign 0.
    #[test]
    fn irrational_root_is_algebraic_and_sign_at_root_is_zero() {
        // x^2 - 2
        let p = Polynomial::univariate(0, &[rat(-2), rat(0), rat(1)]);
        let roots = isolate_root_samples(std::slice::from_ref(&p), 0);
        assert_eq!(roots.len(), 2, "x^2 - 2 has two real roots");
        for r in &roots {
            assert!(!r.is_rational(), "±√2 are irrational");
            // The defining polynomial vanishes at its own root.
            assert_eq!(
                r.to_point().sign_of(&p, 0),
                0,
                "x^2 - 2 must be exactly zero at ±√2 (not a spurious midpoint sign)"
            );
        }
    }

    /// The midpoint of the isolating interval of √2 is NOT a root: substituting
    /// it gives a non-zero sign, which is exactly the unsoundness this fixes.
    #[test]
    fn midpoint_substitution_would_be_nonzero() {
        let p = Polynomial::univariate(0, &[rat(-2), rat(0), rat(1)]); // x^2 - 2
        let roots = isolate_root_samples(std::slice::from_ref(&p), 0);
        let positive_root = roots
            .iter()
            .find(|r| r.to_point().approximate() > rat(0))
            .expect("√2 present");
        let midpoint = (&positive_root.lo + &positive_root.hi) / rat(2);
        // The midpoint is a rational approximation, not the root.
        assert_ne!(
            sign_at_rational(&p, 0, &midpoint),
            0,
            "the rational midpoint of √2's interval is not a root of x^2 - 2"
        );
    }

    /// A different polynomial's sign at √2: q(x) = x - 1 is positive at √2 ≈ 1.41.
    #[test]
    fn sign_of_other_polynomial_at_algebraic_root() {
        let p = Polynomial::univariate(0, &[rat(-2), rat(0), rat(1)]); // x^2 - 2
        let q = Polynomial::univariate(0, &[rat(-1), rat(1)]); // x - 1
        let roots = isolate_root_samples(&[p], 0);
        let positive_root = roots
            .iter()
            .find(|r| r.to_point().approximate() > rat(0))
            .expect("√2 present");
        assert_eq!(
            positive_root.to_point().sign_of(&q, 0),
            1,
            "x - 1 is positive at √2"
        );

        let q2 = Polynomial::univariate(0, &[rat(-2), rat(1)]); // x - 2
        assert_eq!(
            positive_root.to_point().sign_of(&q2, 0),
            -1,
            "x - 2 is negative at √2 (< 2)"
        );
    }

    /// Rational roots (x^2 - 4 → ±2) stay rational and open samples fall
    /// strictly between the roots.
    #[test]
    fn rational_roots_and_open_samples() {
        let p = Polynomial::univariate(0, &[rat(-4), rat(0), rat(1)]); // x^2 - 4
        let roots = isolate_root_samples(&[p], 0);
        assert_eq!(roots.len(), 2);
        assert!(roots.iter().all(|r| r.is_rational()));
        assert_eq!(roots[0].lo, rat(-2));
        assert_eq!(roots[1].lo, rat(2));

        let between = open_sample(Some(&roots[0]), Some(&roots[1]));
        assert!(between > rat(-2) && between < rat(2));
        let before = open_sample(None, Some(&roots[0]));
        assert!(before < rat(-2));
        let after = open_sample(Some(&roots[1]), None);
        assert!(after > rat(2));
    }

    /// Regression: an irrational root's wide bracket can end exactly where a
    /// rational root sits (`−√2` isolated by `[-3, 0]`, next to the root `0`).
    /// If merely-touching brackets were accepted as "disjoint", the open cell
    /// between them would be sampled *at* the shared endpoint — i.e. at the
    /// root — leaving that cell with no representative at all. A caller
    /// enumerating one point per cell would then miss every solution in
    /// `(−√2, 0)` and wrongly report the constraint set empty.
    #[test]
    fn consecutive_brackets_are_strictly_separated_and_open_samples_land_inside() {
        // x * (x^2 - 2): roots -√2, 0, √2.
        let p = Polynomial::univariate(0, &[rat(0), rat(-2), rat(0), rat(1)]);
        let roots = isolate_root_samples(std::slice::from_ref(&p), 0);
        assert_eq!(roots.len(), 3, "roots are -√2, 0, √2");

        for pair in roots.windows(2) {
            assert!(
                pair[0].hi < pair[1].lo,
                "brackets [{}, {}] and [{}, {}] must be strictly separated",
                pair[0].lo,
                pair[0].hi,
                pair[1].lo,
                pair[1].hi
            );
            let between = open_sample(Some(&pair[0]), Some(&pair[1]));
            assert!(
                between > pair[0].hi && between < pair[1].lo,
                "{between} must sit strictly inside the gap"
            );
            // The strongest statement: the open sample is not a root of `p`.
            assert_ne!(
                sign_at_rational(&p, 0, &between),
                0,
                "{between} is an open-cell sample, so p must not vanish there"
            );
        }
    }

    /// Roots shared by two different polynomials must be deduplicated.
    #[test]
    fn shared_roots_are_deduplicated() {
        // (x-1) and (x-1)(x+1): the common root x = 1 must appear once.
        let p1 = Polynomial::univariate(0, &[rat(-1), rat(1)]); // x - 1
        let p2 = Polynomial::univariate(0, &[rat(-1), rat(0), rat(1)]); // x^2 - 1
        let roots = isolate_root_samples(&[p1, p2], 0);
        // Distinct roots: {-1, 1}. x = 1 comes from both and must not double.
        assert_eq!(roots.len(), 2, "shared root x=1 deduplicated → {{-1, 1}}");
    }
}
