//! Cylindrical Algebraic Decomposition (CAD) for NLSAT.
//!
//! This module implements the CAD algorithm for solving non-linear real
//! arithmetic problems. CAD decomposes R^n into cells on which each
//! polynomial has constant sign.
//!
//! ## Key Components
//!
//! - **Projection**: Computes polynomials in n-1 variables from n-variable polynomials
//! - **Lifting**: Extends partial assignments to full assignments
//! - **Cell decomposition**: Partitions the real line into sign-invariant regions
//!
//! ## Reference
//!
//! - Z3's `nlsat/nlsat_solver.cpp`
//! - Collins' original CAD paper (1975)
//! - McCallum's improved projection (1988)

use crate::var_order::{OrderingStrategy, VariableOrdering};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::{NULL_VAR, Polynomial, Var};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// A cell in the CAD decomposition.
/// Represents a connected region of R^k where all constraints have constant sign.
#[derive(Debug, Clone, PartialEq)]
pub struct CadCell {
    /// The variable ordering (x_1, x_2, ..., x_k).
    pub var_order: Vec<Var>,
    /// Sample point for each variable.
    pub sample: Vec<CadPoint>,
    /// Signs of input polynomials at this cell.
    pub signs: Vec<i8>,
}

/// A point in a CAD cell (either a root or a sample between roots).
#[derive(Debug, Clone, PartialEq)]
pub enum CadPoint {
    /// A rational sample point.
    Rational(BigRational),
    /// An algebraic number represented by isolating interval and polynomial.
    Algebraic {
        /// Isolating interval lower bound.
        lo: BigRational,
        /// Isolating interval upper bound.
        hi: BigRational,
        /// Minimal polynomial (univariate).
        poly: Polynomial,
        /// Root index (1-based, counting from left).
        index: u32,
    },
}

impl CadPoint {
    /// Create a rational point.
    pub fn rational(r: BigRational) -> Self {
        Self::Rational(r)
    }

    /// Create an algebraic point.
    pub fn algebraic(lo: BigRational, hi: BigRational, poly: Polynomial, index: u32) -> Self {
        Self::Algebraic {
            lo,
            hi,
            poly,
            index,
        }
    }

    /// Get a rational approximation of the point.
    pub fn approximate(&self) -> BigRational {
        match self {
            CadPoint::Rational(r) => r.clone(),
            CadPoint::Algebraic { lo, hi, .. } => (lo + hi) / BigRational::from_integer(2.into()),
        }
    }

    /// Check if this point is exactly rational.
    pub fn is_rational(&self) -> bool {
        matches!(self, CadPoint::Rational(_))
    }

    /// Get as rational if it is one.
    pub fn as_rational(&self) -> Option<&BigRational> {
        match self {
            CadPoint::Rational(r) => Some(r),
            _ => None,
        }
    }
}

/// Projection set - polynomials resulting from projection.
#[derive(Debug, Clone, Default)]
pub struct ProjectionSet {
    /// Polynomials in the projection, keyed by maximum variable.
    polys: FxHashMap<Var, Vec<Polynomial>>,
}

impl ProjectionSet {
    /// Create an empty projection set.
    pub fn new() -> Self {
        Self {
            polys: FxHashMap::default(),
        }
    }

    /// Add a polynomial to the projection set.
    pub fn add(&mut self, poly: Polynomial) {
        if poly.is_zero() || poly.is_constant() {
            return; // Skip trivial polynomials
        }

        let max_var = poly.max_var();
        if max_var == NULL_VAR {
            return;
        }

        self.polys.entry(max_var).or_default().push(poly);
    }

    /// Get polynomials for a given maximum variable.
    pub fn get(&self, var: Var) -> &[Polynomial] {
        self.polys.get(&var).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all polynomials.
    pub fn all(&self) -> impl Iterator<Item = &Polynomial> {
        self.polys.values().flatten()
    }

    /// Get all variables.
    pub fn vars(&self) -> impl Iterator<Item = Var> + '_ {
        self.polys.keys().copied()
    }

    /// Number of polynomials.
    pub fn len(&self) -> usize {
        self.polys.values().map(|v| v.len()).sum()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }

    /// Merge another projection set into this one.
    pub fn merge(&mut self, other: &ProjectionSet) {
        for polys in other.polys.values() {
            for poly in polys {
                self.add(poly.clone());
            }
        }
    }

    /// Remove duplicate polynomials.
    pub fn deduplicate(&mut self) {
        for polys in self.polys.values_mut() {
            // Sort by leading coefficient and terms for deduplication
            polys.sort_by(|a, b| {
                let a_terms = a.num_terms();
                let b_terms = b.num_terms();
                a_terms.cmp(&b_terms).then_with(|| {
                    let a_deg = a.total_degree();
                    let b_deg = b.total_degree();
                    a_deg.cmp(&b_deg)
                })
            });
            polys.dedup_by(|a, b| {
                // Check if a and b are scalar multiples of each other
                if a.num_terms() != b.num_terms() {
                    return false;
                }
                if a.is_zero() && b.is_zero() {
                    return true;
                }
                if a.is_zero() || b.is_zero() {
                    return false;
                }
                // Compare after making both monic
                let a_monic = a.make_monic();
                let b_monic = b.make_monic();
                a_monic == b_monic
            });
        }
    }
}

/// CAD projection operator.
///
/// Computes the projection of a set of polynomials, eliminating the
/// highest variable and producing polynomials in fewer variables.
#[derive(Debug)]
pub struct CadProjection {
    /// Use McCallum's reduced projection when possible.
    use_mccallum: bool,
    /// Cache for computed resultants (polynomial pair hash -> resultant).
    /// Uses RwLock for thread-safe concurrent access.
    resultant_cache: std::sync::RwLock<FxHashMap<(u64, u64), Polynomial>>,
    /// Enable caching.
    enable_cache: bool,
    /// Enable parallel projection computation.
    parallel: bool,
}

impl Default for CadProjection {
    fn default() -> Self {
        Self::new()
    }
}

impl CadProjection {
    /// Create a new CAD projection operator.
    pub fn new() -> Self {
        Self {
            use_mccallum: true,
            resultant_cache: std::sync::RwLock::new(FxHashMap::default()),
            enable_cache: true,
            parallel: true, // Enable by default
        }
    }

    /// Create with Collins' original (more complete) projection.
    pub fn collins() -> Self {
        Self {
            use_mccallum: false,
            resultant_cache: std::sync::RwLock::new(FxHashMap::default()),
            enable_cache: true,
            parallel: true, // Enable by default
        }
    }

    /// Create with custom settings.
    pub fn with_config(use_mccallum: bool, parallel: bool) -> Self {
        Self {
            use_mccallum,
            resultant_cache: std::sync::RwLock::new(FxHashMap::default()),
            enable_cache: true,
            parallel,
        }
    }

    /// Create a projection operator with caching control.
    pub fn with_caching(use_mccallum: bool, enable_cache: bool) -> Self {
        Self {
            use_mccallum,
            resultant_cache: std::sync::RwLock::new(FxHashMap::default()),
            enable_cache,
            parallel: true, // Enable by default
        }
    }

    /// Compute a hash for a polynomial (simple hash based on degree and coefficients).
    fn poly_hash(&self, poly: &Polynomial) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        poly.degree(NULL_VAR).hash(&mut hasher);
        // Hash first few terms for efficiency
        for (i, term) in poly.terms().iter().take(5).enumerate() {
            i.hash(&mut hasher);
            term.coeff.numer().hash(&mut hasher);
            term.coeff.denom().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Compute the resultant of two polynomials with caching.
    fn compute_resultant_cached(&self, p1: &Polynomial, p2: &Polynomial, var: Var) -> Polynomial {
        if !self.enable_cache {
            // Caching disabled, compute directly
            return p1.resultant(p2, var);
        }

        // Compute hashes for cache lookup
        let h1 = self.poly_hash(p1);
        let h2 = self.poly_hash(p2);
        let cache_key = if h1 < h2 { (h1, h2) } else { (h2, h1) };

        // Check cache with read lock. Only a write guard poisons a `RwLock`,
        // and the only write is the single `insert` below: its value (a clone
        // of a resultant computed before the lock is taken) is built before
        // the map is touched, and its `(u64, u64)` key hashes and compares
        // without panicking, so a poisoned lock still guards a well-formed
        // map whose every entry is a whole value an insert stored, and
        // `into_inner` reads it as it is. (Whether a cached entry is the
        // resultant of *these* operands is a separate question of how
        // `cache_key` is formed; poisoning adds nothing to it.)
        {
            let cache = self
                .resultant_cache
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        // Not in cache, compute it
        let resultant = p1.resultant(p2, var);

        // Store in cache with write lock. As at the read above, this `insert`
        // is the only mutation and leaves the map either without the entry
        // or with the whole entry, so a poisoned lock guards a map whose
        // entries are each a whole stored value and `into_inner` is sound.
        {
            let mut cache = self
                .resultant_cache
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.insert(cache_key, resultant.clone());
        }

        resultant
    }

    /// Project polynomials, eliminating the given variable.
    ///
    /// Returns polynomials in variables < var that define the cylindrical
    /// structure needed for sign-invariant decomposition.
    pub fn project(&self, polys: &[Polynomial], var: Var) -> ProjectionSet {
        let mut result = ProjectionSet::new();

        // Filter polynomials that actually contain this variable
        let relevant: Vec<&Polynomial> = polys.iter().filter(|p| p.degree(var) > 0).collect();

        if relevant.is_empty() {
            // No polynomials contain this variable
            return result;
        }

        // 1. Add leading coefficients (coefficients of highest degree term)
        let leading_coeffs: Vec<Polynomial> = if self.parallel {
            relevant
                .par_iter()
                .map(|poly| poly.leading_coeff_wrt(var))
                .filter(|lc| !lc.is_zero() && !lc.is_constant())
                .collect()
        } else {
            relevant
                .iter()
                .map(|poly| poly.leading_coeff_wrt(var))
                .filter(|lc| !lc.is_zero() && !lc.is_constant())
                .collect()
        };
        for lc in leading_coeffs {
            result.add(lc);
        }

        // 2. Add discriminants (for each polynomial)
        let discriminants: Vec<Polynomial> = if self.parallel {
            relevant
                .par_iter()
                .map(|poly| poly.discriminant(var))
                .filter(|disc| !disc.is_zero() && !disc.is_constant())
                .collect()
        } else {
            relevant
                .iter()
                .map(|poly| poly.discriminant(var))
                .filter(|disc| !disc.is_zero() && !disc.is_constant())
                .collect()
        };
        for disc in discriminants {
            result.add(disc);
        }

        // 3. Add resultants (for pairs of polynomials)
        let resultants: Vec<Polynomial> = if self.parallel {
            // Create pairs of indices for parallel computation
            let pairs: Vec<(usize, usize)> = (0..relevant.len())
                .flat_map(|i| ((i + 1)..relevant.len()).map(move |j| (i, j)))
                .collect();

            pairs
                .par_iter()
                .filter_map(|&(i, j)| {
                    let res = self.compute_resultant_cached(relevant[i], relevant[j], var);
                    if !res.is_zero() && !res.is_constant() {
                        Some(res)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            let mut results = Vec::new();
            for i in 0..relevant.len() {
                for j in (i + 1)..relevant.len() {
                    let res = self.compute_resultant_cached(relevant[i], relevant[j], var);
                    if !res.is_zero() && !res.is_constant() {
                        results.push(res);
                    }
                }
            }
            results
        };
        for res in resultants {
            result.add(res);
        }

        // 4. For Collins' projection, also add derivatives
        if !self.use_mccallum {
            let derivative_resultants: Vec<Polynomial> = if self.parallel {
                relevant
                    .par_iter()
                    .filter_map(|poly| {
                        let deriv = poly.derivative(var);
                        if !deriv.is_zero() && deriv.degree(var) > 0 {
                            let res = self.compute_resultant_cached(poly, &deriv, var);
                            if !res.is_zero() && !res.is_constant() {
                                Some(res)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                let mut results = Vec::new();
                for poly in &relevant {
                    let deriv = poly.derivative(var);
                    if !deriv.is_zero() && deriv.degree(var) > 0 {
                        let res = self.compute_resultant_cached(poly, &deriv, var);
                        if !res.is_zero() && !res.is_constant() {
                            results.push(res);
                        }
                    }
                }
                results
            };
            for res in derivative_resultants {
                result.add(res);
            }
        }

        // 5. Recursively include lower-variable polynomials
        for poly in polys {
            if poly.degree(var) == 0 && poly.max_var() != NULL_VAR {
                result.add(poly.clone());
            }
        }

        result.deduplicate();
        result
    }

    /// Compute full projection sequence, eliminating variables from high to low.
    ///
    /// Given polynomials in variables x_1, ..., x_n, computes:
    /// - P_n = input polynomials
    /// - P_{n-1} = project(P_n, x_n)
    /// - P_{n-2} = project(P_{n-1}, x_{n-1})
    /// - ...
    /// - P_1 = project(P_2, x_2)
    pub fn full_projection(&self, polys: &[Polynomial], var_order: &[Var]) -> Vec<ProjectionSet> {
        if var_order.is_empty() || polys.is_empty() {
            return Vec::new();
        }

        let mut projections = Vec::with_capacity(var_order.len());
        let mut current: Vec<Polynomial> = polys.to_vec();

        // Project from highest variable down
        for &var in var_order.iter().rev() {
            let proj = self.project(&current, var);
            projections.push(proj.clone());

            // Prepare for next level
            current = proj.all().cloned().collect();
        }

        projections.reverse();
        projections
    }
}

/// Root isolation using Sturm sequences.
///
/// Counts and isolates real roots of univariate polynomials.
///
/// Safety-net recursion depth for [`SturmSequence::isolate_in_interval_bounded`].
/// See [`SturmSequence::isolate_roots`] for why this is a defensive bound
/// rather than a correctness-affecting one.
const MAX_ROOT_ISOLATION_DEPTH: u32 = 4096;

#[derive(Debug)]
pub struct SturmSequence {
    /// The Sturm sequence (p, p', p % p', ...).
    sequence: Vec<Polynomial>,
    /// The variable.
    var: Var,
}

/// Sign (`1`, `-1`, or `0`) of `poly`'s leading coefficient with respect to
/// `var`, treated as a free function so it can be used before a
/// [`SturmSequence`] exists (e.g. while still building `sequence`) as well
/// as from instance methods.
fn poly_lc_sign(poly: &Polynomial, var: Var) -> i8 {
    let lc_poly = poly.leading_coeff_wrt(var);

    // If the LC is a constant, get its sign
    let lc = if lc_poly.is_constant() {
        lc_poly.leading_coeff()
    } else {
        // For multivariate, assume we're dealing with univariate at this level
        poly.leading_coeff()
    };

    if lc.is_positive() {
        1
    } else if lc.is_negative() {
        -1
    } else {
        0
    }
}

impl SturmSequence {
    /// Compute the Sturm sequence for a polynomial.
    pub fn new(poly: &Polynomial, var: Var) -> Self {
        let mut sequence = Vec::new();

        if poly.is_zero() {
            return Self { sequence, var };
        }

        // P_0 = p
        let p0 = poly.primitive();
        sequence.push(p0.clone());

        // P_1 = p'
        let p1 = poly.derivative(var).primitive();
        if p1.is_zero() {
            // Polynomial has no variable or is constant
            return Self { sequence, var };
        }
        sequence.push(p1);

        // P_{i+1} = -rem(P_{i-1}, P_i)
        let max_iters = poly.degree(var) as usize + 10;
        for _ in 0..max_iters {
            let len = sequence.len();
            if len < 2 {
                break;
            }

            let a = &sequence[len - 2];
            let b = &sequence[len - 1];

            if b.is_zero() || b.degree(var) == 0 {
                break;
            }

            // `Polynomial::pseudo_remainder(divisor, var)` returns `r`
            // satisfying `lc(divisor)^d * a = q * divisor + r` for some
            // `d >= 0` that depends on internal reduction steps not
            // exposed to callers. Whenever that implicit scale factor
            // `lc(divisor)^d` is negative, `r` has the OPPOSITE sign of
            // the true (unscaled) polynomial remainder `rem(a, b)` that
            // Sturm's sign-change theorem requires at every point,
            // corrupting `count_roots`/`isolate_roots` for any polynomial
            // with a negative leading coefficient (audit finding: e.g.
            // `p = 4 - x^2` produces a chain with zero net sign changes
            // despite having two real roots).
            //
            // Since `d`'s parity isn't observable from outside
            // `pseudo_remainder`, sidestep it entirely: dividing by `b` or
            // by `-b` yields the same remainder (up to sign), so instead
            // divide by a version of `b` whose leading coefficient is
            // always positive. Then `lc(divisor)^d` is positive for every
            // `d`, so the returned remainder is guaranteed to be a
            // *positive* scalar multiple of the true `rem(a, b)`,
            // preserving its sign everywhere.
            let divisor = if poly_lc_sign(b, var) < 0 {
                b.neg()
            } else {
                b.clone()
            };

            let remainder = a.pseudo_remainder(&divisor, var);
            if remainder.is_zero() {
                break;
            }

            // Negate and make primitive (Sturm's recurrence).
            let neg_rem = remainder.neg().primitive();
            sequence.push(neg_rem);
        }

        Self { sequence, var }
    }

    /// Count sign changes at a point.
    fn sign_changes_at(&self, x: &BigRational) -> u32 {
        let mut changes = 0;
        let mut prev_sign: Option<i8> = None;

        for poly in &self.sequence {
            // Evaluate polynomial at x
            let mut eval_map = FxHashMap::default();
            eval_map.insert(self.var, x.clone());

            // For multivariate, we'd need full evaluation
            // For now, assume univariate
            let val = poly.eval(&eval_map);
            let sign = if val.is_zero() {
                continue; // Skip zeros
            } else if val.is_positive() {
                1
            } else {
                -1
            };

            if let Some(prev) = prev_sign
                && prev != sign
            {
                changes += 1;
            }
            prev_sign = Some(sign);
        }

        changes
    }

    /// Get the sign of leading coefficient with respect to the variable at infinity.
    /// For univariate polynomials, this is just the sign of the leading coeff.
    fn lc_sign(&self, poly: &Polynomial) -> i8 {
        poly_lc_sign(poly, self.var)
    }

    /// Count sign changes at negative infinity.
    fn sign_changes_at_neg_inf(&self) -> u32 {
        let mut changes = 0;
        let mut prev_sign: Option<i8> = None;

        for poly in &self.sequence {
            if poly.is_zero() {
                continue;
            }

            // At -∞, sign is determined by leading coefficient and degree parity
            let degree = poly.degree(self.var);
            let lc_sign = self.lc_sign(poly);
            if lc_sign == 0 {
                continue;
            }

            // At -∞: sign = lc_sign * (-1)^degree
            let sign = if degree % 2 == 0 { lc_sign } else { -lc_sign };

            if let Some(prev) = prev_sign
                && prev != sign
            {
                changes += 1;
            }
            prev_sign = Some(sign);
        }

        changes
    }

    /// Count sign changes at positive infinity.
    fn sign_changes_at_pos_inf(&self) -> u32 {
        let mut changes = 0;
        let mut prev_sign: Option<i8> = None;

        for poly in &self.sequence {
            if poly.is_zero() {
                continue;
            }

            // At +∞, sign is determined by leading coefficient
            let lc_sign = self.lc_sign(poly);
            if lc_sign == 0 {
                continue;
            }

            if let Some(prev) = prev_sign
                && prev != lc_sign
            {
                changes += 1;
            }
            prev_sign = Some(lc_sign);
        }

        changes
    }

    /// Count the number of distinct real roots.
    pub fn count_roots(&self) -> u32 {
        if self.sequence.is_empty() {
            return 0;
        }

        let v_neg_inf = self.sign_changes_at_neg_inf();
        let v_pos_inf = self.sign_changes_at_pos_inf();

        v_neg_inf.saturating_sub(v_pos_inf)
    }

    /// Count roots in interval (a, b].
    pub fn count_roots_in(&self, a: &BigRational, b: &BigRational) -> u32 {
        if self.sequence.is_empty() || a >= b {
            return 0;
        }

        let v_a = self.sign_changes_at(a);
        let v_b = self.sign_changes_at(b);

        v_a.saturating_sub(v_b)
    }

    /// Isolate all real roots into disjoint intervals.
    ///
    /// Returns a list of (lo, hi) intervals, each containing exactly one root.
    pub fn isolate_roots(&self) -> Vec<(BigRational, BigRational)> {
        let num_roots = self.count_roots();
        if num_roots == 0 {
            return Vec::new();
        }

        // Start with a large interval that contains all roots
        // Use Cauchy's bound: |root| <= 1 + max|a_i/a_n|
        let bound = self.root_bound();
        let neg_bound = -bound.clone();

        // Bisection to isolate roots. `BigRational` bisection is EXACT (no
        // floating-point precision loss: halving a rational just grows its
        // denominator), and Sturm's theorem guarantees `count_roots_in`
        // correctly reports the number of *distinct* real roots in any
        // sub-interval. Since distinct real roots always have a positive
        // (if possibly tiny) minimum pairwise separation, repeated
        // bisection is mathematically guaranteed to eventually isolate
        // each one into its own interval -- there is no principled width
        // threshold below which "close enough" roots may be merged (the
        // audit finding this fixes: the previous 1e-6 width cutoff would
        // silently merge roots closer together than that, breaking the
        // "each interval contains exactly one root" contract and
        // corrupting downstream root-index lookups).
        //
        // `MAX_ROOT_ISOLATION_DEPTH` is therefore a pure safety net against
        // runaway recursion (e.g. a latent bug elsewhere producing a
        // non-square-free/inconsistent sequence), not a correctness
        // parameter: at depth 2^-MAX_ROOT_ISOLATION_DEPTH the interval
        // width is astronomically smaller than any root separation
        // reachable by realistic rational-coefficient polynomials, so this
        // limit should never actually be hit for well-formed input.
        self.isolate_in_interval_bounded(neg_bound, bound, num_roots, MAX_ROOT_ISOLATION_DEPTH)
    }

    /// Compute a bound on the absolute value of all roots.
    fn root_bound(&self) -> BigRational {
        if self.sequence.is_empty() {
            return BigRational::one();
        }

        let poly = &self.sequence[0];
        if poly.is_zero() {
            return BigRational::one();
        }

        let lc = poly.leading_coeff();
        if lc.is_zero() {
            return BigRational::one();
        }

        // Cauchy's bound: |root| <= 1 + max(|a_{n-1}|, |a_{n-2}|, ..., |a_0|) / |a_n|
        // where a_n is the leading coefficient
        let lc_abs = lc.abs();
        let mut max_ratio = BigRational::zero();

        for term in poly.terms() {
            // Skip the leading term itself
            let term_degree = term.monomial.degree(self.var);
            if term_degree == poly.degree(self.var) {
                continue;
            }
            let ratio = term.coeff.abs() / &lc_abs;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }

        BigRational::one() + max_ratio
    }

    /// Isolate roots within an interval using exact-rational bisection.
    ///
    /// Bisects until every returned interval contains exactly one root
    /// (`expected_roots == 1`), with `depth` acting purely as a defensive
    /// budget ceiling (see [`Self::isolate_roots`]). There is deliberately
    /// NO width-based early-out: any such threshold would (unsoundly) merge
    /// together distinct roots closer together than it, breaking the "one
    /// root per interval" contract that callers rely on for root-index
    /// lookups.
    ///
    /// Implemented with an explicit heap work-stack rather than native
    /// recursion: each level owns two `BigRational` bounds plus bisection
    /// temporaries, so `MAX_ROOT_ISOLATION_DEPTH` levels of native frames
    /// would consume on the order of a mebibyte — enough to overflow a
    /// small worker-thread stack *before* the depth ceiling can fire. The
    /// heap stack makes the ceiling the only thing that can stop the walk.
    ///
    /// Emission order is identical to the former recursion (depth-first,
    /// left sub-interval before right, hence ascending) because the right
    /// half is pushed first and therefore popped last.
    fn isolate_in_interval_bounded(
        &self,
        lo: BigRational,
        hi: BigRational,
        expected_roots: u32,
        depth: u32,
    ) -> Vec<(BigRational, BigRational)> {
        let mut result = Vec::new();
        let mut work: Vec<(BigRational, BigRational, u32, u32)> =
            vec![(lo, hi, expected_roots, depth)];

        while let Some((lo, hi, expected_roots, depth)) = work.pop() {
            if expected_roots == 0 {
                continue;
            }

            if expected_roots == 1 {
                result.push((lo, hi));
                continue;
            }

            if depth == 0 {
                // Defensive-only fallback: with a correct Sturm chain and
                // exact rational bisection, `MAX_ROOT_ISOLATION_DEPTH`
                // should never actually be exhausted while
                // `expected_roots > 1` (see `isolate_roots`). Reaching this
                // branch indicates either a pathologically ill-conditioned
                // input or a bug upstream (e.g. an incorrect root count);
                // rather than panicking or fabricating a single interval
                // that falsely claims to isolate multiple roots, drop this
                // sub-range so callers observe a short result
                // (already-handled by downstream root-index lookups)
                // instead of a wrong one.
                continue;
            }

            // Bisect
            let mid = (&lo + &hi) / BigRational::from_integer(2.into());

            // Count roots in each half.
            // count_roots_in(a, b) gives roots in (a, b] per Sturm's theorem
            let roots_left = self.count_roots_in(&lo, &mid);

            // For the right half, we need roots in (mid, hi]
            let roots_right = self.count_roots_in(&mid, &hi);

            // Push right before left so that left is processed first,
            // preserving the ascending output order of the recursive form.
            if roots_right > 0 {
                work.push((mid.clone(), hi, roots_right, depth - 1));
            }

            if roots_left > 0 {
                work.push((lo, mid, roots_left, depth - 1));
            }
        }

        result
    }
}

/// Lifting phase - extend partial assignment to full assignment.
#[derive(Debug)]
pub struct CadLifter {
    /// Variable ordering.
    var_order: Vec<Var>,
    /// Projection polynomials at each level.
    projections: Vec<ProjectionSet>,
}

impl CadLifter {
    /// Create a new lifter from projection data.
    pub fn new(var_order: Vec<Var>, projections: Vec<ProjectionSet>) -> Self {
        Self {
            var_order,
            projections,
        }
    }

    /// Lift a partial assignment to the next variable.
    ///
    /// Given an assignment to variables x_1, ..., x_{k-1}, compute the
    /// cell decomposition for x_k.
    pub fn lift(&self, assignment: &FxHashMap<Var, BigRational>, var_idx: usize) -> Vec<CadPoint> {
        if var_idx >= self.var_order.len() {
            return Vec::new();
        }

        let var = self.var_order[var_idx];
        let polys = self
            .projections
            .get(var_idx)
            .map(|p| p.get(var))
            .unwrap_or(&[]);

        // Substitute the partial assignment into each polynomial
        let mut univariate_polys: Vec<Polynomial> = Vec::new();

        for poly in polys {
            let mut sub_poly = poly.clone();
            for (&v, val) in assignment {
                if v != var {
                    sub_poly = sub_poly.eval_at(v, val);
                }
            }
            if !sub_poly.is_zero() && sub_poly.degree(var) > 0 {
                univariate_polys.push(sub_poly);
            }
        }

        // Isolate the real roots exactly (rational vs. algebraic), sorted with
        // disjoint intervals.
        let roots = crate::cad_algebraic::isolate_root_samples(&univariate_polys, var);

        // Create cell points: open samples strictly between roots and the exact
        // root points themselves.
        let mut points = Vec::new();

        match roots.last() {
            None => {
                // No roots - just one cell, sample at 0
                points.push(CadPoint::rational(crate::cad_algebraic::open_sample(
                    None, None,
                )));
            }
            Some(last) => {
                // Before first root
                points.push(CadPoint::rational(crate::cad_algebraic::open_sample(
                    None,
                    Some(&roots[0]),
                )));

                for i in 0..roots.len() {
                    // The exact root itself
                    points.push(roots[i].to_point());

                    // Between this root and next
                    if i + 1 < roots.len() {
                        points.push(CadPoint::rational(crate::cad_algebraic::open_sample(
                            Some(&roots[i]),
                            Some(&roots[i + 1]),
                        )));
                    }
                }

                // After last root
                points.push(CadPoint::rational(crate::cad_algebraic::open_sample(
                    Some(last),
                    None,
                )));
            }
        }

        points
    }
}

/// Configuration for CAD computation.
#[derive(Debug, Clone)]
pub struct CadConfig {
    /// Maximum number of cells to enumerate.
    pub max_cells: usize,
    /// Use McCallum's reduced projection.
    pub use_mccallum: bool,
    /// Enable caching of projection results.
    pub cache_projections: bool,
    /// Sample point selection strategy.
    pub sample_strategy: SampleStrategy,
    /// Variable ordering strategy.
    pub ordering_strategy: Option<OrderingStrategy>,
    /// Enable parallel projection computation using Rayon.
    pub parallel_projection: bool,
    /// Enable parallel lifting with work-stealing using Rayon.
    pub parallel_lifting: bool,
}

impl Default for CadConfig {
    fn default() -> Self {
        Self {
            max_cells: 100_000,
            use_mccallum: true,
            cache_projections: true,
            sample_strategy: SampleStrategy::Rational,
            ordering_strategy: Some(OrderingStrategy::Brown),
            parallel_projection: true, // Enable by default for better performance
            parallel_lifting: true,    // Enable work-stealing by default for better performance
        }
    }
}

/// Strategy for selecting sample points in cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStrategy {
    /// Prefer rational sample points.
    Rational,
    /// Prefer integer sample points.
    Integer,
    /// Use cell midpoints.
    Midpoint,
    /// Use algebraic numbers when needed.
    Algebraic,
}

/// Full CAD decomposer - computes cylindrical algebraic decomposition.
///
/// This implements the complete CAD algorithm:
/// 1. Projection: Compute projection sets eliminating variables from high to low
/// 2. Base: Decompose R into cells based on roots of lowest-variable polynomials
/// 3. Lifting: Extend cells to higher dimensions by adding variables one at a time
pub struct CadDecomposer {
    /// Configuration.
    config: CadConfig,
    /// Projection operator.
    projection: CadProjection,
    /// Variable ordering (from lowest to highest).
    var_order: Vec<Var>,
    /// Projection sets at each level.
    projection_sets: Vec<ProjectionSet>,
    /// Current number of cells enumerated.
    num_cells: usize,
}

impl CadDecomposer {
    /// Create a new CAD decomposer.
    pub fn new(config: CadConfig, var_order: Vec<Var>) -> Self {
        let projection =
            CadProjection::with_config(config.use_mccallum, config.parallel_projection);

        Self {
            config,
            projection,
            var_order,
            projection_sets: Vec::new(),
            num_cells: 0,
        }
    }

    /// Create a CAD decomposer with automatic variable ordering.
    ///
    /// The variable ordering is computed from the polynomials using the
    /// strategy specified in the config (or Brown's heuristic by default).
    pub fn with_auto_ordering(config: CadConfig, polynomials: &[Polynomial]) -> Self {
        let strategy = config.ordering_strategy.unwrap_or(OrderingStrategy::Brown);
        let ordering_computer = VariableOrdering::new(strategy, polynomials.to_vec());
        let var_order = ordering_computer.compute();

        Self::new(config, var_order)
    }

    /// Compute the CAD for a set of polynomials.
    ///
    /// Returns all cells in the decomposition.
    pub fn decompose(&mut self, polys: &[Polynomial]) -> Result<Vec<CadCell>, CadError> {
        if self.var_order.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Projection phase
        self.projection_sets = self.projection.full_projection(polys, &self.var_order);

        // For univariate case, add original polynomials to projection_sets[0]
        // since projection may filter them out as constants
        if self.var_order.len() == 1 && !polys.is_empty() {
            let var = self.var_order[0];
            if self.projection_sets.is_empty() {
                self.projection_sets.push(ProjectionSet::new());
            }
            for poly in polys {
                if poly.degree(var) > 0 {
                    self.projection_sets[0].add(poly.clone());
                }
            }
        }

        // 2. Base phase: decompose R (lowest variable)
        let base_cells = self.decompose_base()?;

        // 3. Lifting phase: extend to higher dimensions
        let mut current_cells = base_cells;
        for level in 1..self.var_order.len() {
            current_cells = self.lift_cells(&current_cells, level)?;

            if self.num_cells >= self.config.max_cells {
                return Err(CadError::TooManyCells);
            }
        }

        Ok(current_cells)
    }

    /// Decompose the base level (univariate case).
    fn decompose_base(&mut self) -> Result<Vec<CadCell>, CadError> {
        if self.var_order.is_empty() {
            return Ok(Vec::new());
        }

        let var = self.var_order[0];
        let polys = if !self.projection_sets.is_empty() {
            self.projection_sets[0].get(var)
        } else {
            &[]
        };

        // Isolate the real roots as exact algebraic samples (rational roots stay
        // rational; irrational roots keep their defining polynomial + isolating
        // interval), sorted with pairwise-disjoint intervals.
        let roots = crate::cad_algebraic::isolate_root_samples(polys, var);

        // Create cells: open cells sampled strictly between roots, plus the
        // exact root points themselves.
        let mut cells = Vec::new();

        match roots.last() {
            None => {
                // No roots - single cell covering all of R
                let sample = self.select_sample_point(None, None, var);
                cells.push(CadCell {
                    var_order: vec![var],
                    sample: vec![CadPoint::rational(sample)],
                    signs: Vec::new(),
                });
                self.num_cells += 1;

                if self.num_cells >= self.config.max_cells {
                    return Err(CadError::TooManyCells);
                }
            }
            Some(last) => {
                // Cell before first root: sample below the first root's isolating
                // interval lower bound (hence below the true root).
                let sample = self.select_sample_point(None, Some(&roots[0].lo), var);
                cells.push(CadCell {
                    var_order: vec![var],
                    sample: vec![CadPoint::rational(sample)],
                    signs: Vec::new(),
                });
                self.num_cells += 1;

                if self.num_cells >= self.config.max_cells {
                    return Err(CadError::TooManyCells);
                }

                for i in 0..roots.len() {
                    // Cell at the (exact) root
                    cells.push(CadCell {
                        var_order: vec![var],
                        sample: vec![roots[i].to_point()],
                        signs: Vec::new(),
                    });
                    self.num_cells += 1;

                    if self.num_cells >= self.config.max_cells {
                        return Err(CadError::TooManyCells);
                    }

                    // Cell between this root and next: sampled strictly inside the
                    // gap between the two roots' isolating intervals.
                    if i + 1 < roots.len() {
                        let sample = self.select_sample_point(
                            Some(&roots[i].hi),
                            Some(&roots[i + 1].lo),
                            var,
                        );
                        cells.push(CadCell {
                            var_order: vec![var],
                            sample: vec![CadPoint::rational(sample)],
                            signs: Vec::new(),
                        });
                        self.num_cells += 1;

                        if self.num_cells >= self.config.max_cells {
                            return Err(CadError::TooManyCells);
                        }
                    }
                }

                // Cell after last root: sample above the last root's interval.
                let sample = self.select_sample_point(Some(&last.hi), None, var);
                cells.push(CadCell {
                    var_order: vec![var],
                    sample: vec![CadPoint::rational(sample)],
                    signs: Vec::new(),
                });
                self.num_cells += 1;

                if self.num_cells >= self.config.max_cells {
                    return Err(CadError::TooManyCells);
                }
            }
        }

        Ok(cells)
    }

    /// Helper function to lift a single cell (used by both sequential and parallel versions).
    fn lift_single_cell(&self, cell: &CadCell, var: Var, level: usize) -> Vec<CadCell> {
        let mut result = Vec::new();

        // Build partial assignment from the cell's sample
        let mut assignment = FxHashMap::default();
        for (i, point) in cell.sample.iter().enumerate() {
            let v = cell.var_order[i];
            assignment.insert(v, point.approximate());
        }

        // Get polynomials at this level
        let polys = if level < self.projection_sets.len() {
            self.projection_sets[level].get(var)
        } else {
            &[]
        };

        // Substitute to get univariate polynomials in var
        let mut univariate_polys = Vec::new();
        for poly in polys {
            let mut sub_poly = poly.clone();
            for (&v, val) in &assignment {
                if v != var {
                    sub_poly = sub_poly.eval_at(v, val);
                }
            }
            if !sub_poly.is_zero() && sub_poly.degree(var) > 0 {
                univariate_polys.push(sub_poly);
            }
        }

        // Isolate the real roots in this cell exactly (rational vs. algebraic),
        // sorted with disjoint intervals.
        let cell_roots = crate::cad_algebraic::isolate_root_samples(&univariate_polys, var);

        // Create lifted cells
        match cell_roots.last() {
            None => {
                // No roots - single cell
                let sample = self.select_sample_point(None, None, var);
                let mut new_sample = cell.sample.clone();
                new_sample.push(CadPoint::rational(sample));

                let mut new_var_order = cell.var_order.clone();
                new_var_order.push(var);

                result.push(CadCell {
                    var_order: new_var_order,
                    sample: new_sample,
                    signs: Vec::new(),
                });
            }
            Some(last) => {
                // Before first root
                let sample = self.select_sample_point(None, Some(&cell_roots[0].lo), var);
                let mut new_sample = cell.sample.clone();
                new_sample.push(CadPoint::rational(sample));
                let mut new_var_order = cell.var_order.clone();
                new_var_order.push(var);
                result.push(CadCell {
                    var_order: new_var_order.clone(),
                    sample: new_sample,
                    signs: Vec::new(),
                });

                for i in 0..cell_roots.len() {
                    // At the exact root
                    let mut new_sample = cell.sample.clone();
                    new_sample.push(cell_roots[i].to_point());
                    let mut new_var_order = cell.var_order.clone();
                    new_var_order.push(var);
                    result.push(CadCell {
                        var_order: new_var_order.clone(),
                        sample: new_sample,
                        signs: Vec::new(),
                    });

                    // Between roots: strictly inside the gap between the two roots'
                    // isolating intervals.
                    if i + 1 < cell_roots.len() {
                        let sample = self.select_sample_point(
                            Some(&cell_roots[i].hi),
                            Some(&cell_roots[i + 1].lo),
                            var,
                        );
                        let mut new_sample = cell.sample.clone();
                        new_sample.push(CadPoint::rational(sample));
                        let mut new_var_order = cell.var_order.clone();
                        new_var_order.push(var);
                        result.push(CadCell {
                            var_order: new_var_order.clone(),
                            sample: new_sample,
                            signs: Vec::new(),
                        });
                    }
                }

                // After last root
                let sample = self.select_sample_point(Some(&last.hi), None, var);
                let mut new_sample = cell.sample.clone();
                new_sample.push(CadPoint::rational(sample));
                let mut new_var_order = cell.var_order.clone();
                new_var_order.push(var);
                result.push(CadCell {
                    var_order: new_var_order,
                    sample: new_sample,
                    signs: Vec::new(),
                });
            }
        }

        result
    }

    /// Lift cells to the next variable level.
    ///
    /// Uses work-stealing parallelism when `parallel_lifting` is enabled in config.
    fn lift_cells(&mut self, cells: &[CadCell], level: usize) -> Result<Vec<CadCell>, CadError> {
        if level >= self.var_order.len() {
            return Ok(cells.to_vec());
        }

        let var = self.var_order[level];

        let lifted_cells = if self.config.parallel_lifting {
            // Parallel version using Rayon's work-stealing
            cells
                .par_iter()
                .flat_map(|cell| self.lift_single_cell(cell, var, level))
                .collect()
        } else {
            // Sequential version
            let mut result = Vec::new();
            for cell in cells {
                result.extend(self.lift_single_cell(cell, var, level));
            }
            result
        };

        // Update cell count and check limit
        let new_cell_count = lifted_cells.len();
        self.num_cells += new_cell_count;

        if self.num_cells >= self.config.max_cells {
            return Err(CadError::TooManyCells);
        }

        Ok(lifted_cells)
    }

    /// Select a sample point between two bounds (or unbounded).
    fn select_sample_point(
        &self,
        lower: Option<&BigRational>,
        upper: Option<&BigRational>,
        _var: Var,
    ) -> BigRational {
        match self.config.sample_strategy {
            SampleStrategy::Rational | SampleStrategy::Algebraic => {
                match (lower, upper) {
                    (None, None) => BigRational::zero(),
                    (None, Some(u)) => u - BigRational::one(),
                    (Some(l), None) => l + BigRational::one(),
                    (Some(l), Some(u)) => {
                        // Midpoint
                        (l + u) / BigRational::from_integer(2.into())
                    }
                }
            }
            SampleStrategy::Integer => {
                // Try to find an integer between bounds
                match (lower, upper) {
                    (None, None) => BigRational::zero(),
                    (None, Some(u)) => u.floor() - BigRational::one(),
                    (Some(l), None) => l.ceil() + BigRational::one(),
                    (Some(l), Some(u)) => {
                        let mid = (l + u) / BigRational::from_integer(2.into());
                        let candidate = mid.round();
                        if lower.is_none_or(|lo| &candidate > lo)
                            && upper.is_none_or(|hi| &candidate < hi)
                        {
                            candidate
                        } else {
                            mid
                        }
                    }
                }
            }
            SampleStrategy::Midpoint => match (lower, upper) {
                (None, None) => BigRational::zero(),
                (None, Some(u)) => u - BigRational::one(),
                (Some(l), None) => l + BigRational::one(),
                (Some(l), Some(u)) => (l + u) / BigRational::from_integer(2.into()),
            },
        }
    }

    /// Get statistics about the decomposition.
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// Get reference to the projection sets.
    pub fn projection_sets(&self) -> &[ProjectionSet] {
        &self.projection_sets
    }
}

/// Error type for CAD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CadError {
    /// Too many cells were generated.
    TooManyCells,
    /// Maximum projection depth exceeded.
    ProjectionDepthExceeded,
    /// Invalid variable ordering.
    InvalidVariableOrder,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_projection_set() {
        let mut proj = ProjectionSet::new();

        // x^2 - 1
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-1, &[])]);
        proj.add(p);

        assert_eq!(proj.len(), 1);
        assert!(!proj.is_empty());
    }

    #[test]
    fn test_cad_projection_basic() {
        let proj = CadProjection::new();

        // p = x^2 + y - 1 (var y = 1 is being eliminated)
        let p = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (1, &[(1, 1)]), // y
            (-1, &[]),      // -1
        ]);

        let result = proj.project(&[p], 1);

        // Should include leading coefficient (which is 1, so constant)
        // and discriminant
        // Projection may or may not produce results depending on polynomial structure
        let _ = result.len();
    }

    #[test]
    fn test_sturm_sequence_quadratic() {
        // x^2 - 4 has roots at x = -2 and x = 2
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-4, &[])]);
        let sturm = SturmSequence::new(&poly, 0);

        assert_eq!(sturm.count_roots(), 2);

        let intervals = sturm.isolate_roots();
        assert_eq!(intervals.len(), 2);
    }

    /// Regression test for the audit finding: `SturmSequence` built its
    /// chain from sign-unnormalized pseudo-remainders, so a NEGATIVE
    /// leading coefficient corrupted the sign-change count. `4 - x^2`
    /// (leading coefficient -1) has roots at x = -2 and x = 2, but the
    /// pre-fix chain `(-x^2+4, -x, +1)` gave zero net sign changes
    /// (`count_roots() == 0`) despite the two real roots.
    #[test]
    fn test_sturm_sequence_negative_leading_coefficient() {
        // 4 - x^2 = -x^2 + 4, roots at x = -2 and x = 2.
        let poly = Polynomial::from_coeffs_int(&[(-1, &[(0, 2)]), (4, &[])]);
        let sturm = SturmSequence::new(&poly, 0);

        assert_eq!(
            sturm.count_roots(),
            2,
            "4 - x^2 has two real roots (-2 and 2)"
        );

        let intervals = sturm.isolate_roots();
        assert_eq!(intervals.len(), 2);
    }

    /// Same shape but cubic, to check the fix isn't accidentally specific
    /// to a single degree/parity combination: `-(x^3 - x) = -x^3 + x` has
    /// roots at -1, 0, 1 (leading coefficient -1, odd degree).
    #[test]
    fn test_sturm_sequence_negative_leading_coefficient_cubic() {
        let poly = Polynomial::from_coeffs_int(&[(-1, &[(0, 3)]), (1, &[(0, 1)])]);
        let sturm = SturmSequence::new(&poly, 0);

        assert_eq!(sturm.count_roots(), 3, "-x^3 + x has three real roots");

        let intervals = sturm.isolate_roots();
        assert_eq!(intervals.len(), 3);
    }

    /// Regression test for the audit finding: `isolate_in_interval_bounded`
    /// used to silently return a single "isolating" interval once its
    /// width dropped below a hardcoded `1e-6` epsilon, even when it still
    /// contained multiple distinct roots. This constructs two roots
    /// (`x = 0` and `x = 1/2_000_000`) separated by `5e-7` -- closer
    /// together than that removed cutoff -- and checks they are still
    /// isolated into two disjoint intervals.
    /// Regression test for the audit finding that the bisection recursion
    /// could exhaust a small thread stack *before* `MAX_ROOT_ISOLATION_DEPTH`
    /// could fire: each level owned two `BigRational` bounds plus bisection
    /// temporaries, so 4096 levels was on the order of a mebibyte.
    ///
    /// Two roots separated by `2^-2000` force ~2000 bisection levels. Run on
    /// a 1 MiB stack, the assertion is that the call *returns* (a stack
    /// overflow aborts the process) with both roots still isolated.
    #[test]
    fn test_sturm_sequence_deep_bisection_on_small_stack() {
        let worker = std::thread::Builder::new().stack_size(1 << 20).spawn(|| {
            let x = Polynomial::from_var(0);
            let separation = BigRational::new(1.into(), num_bigint::BigInt::from(1u8) << 2000);
            let factor2 = Polynomial::sub(&x, &Polynomial::constant(separation));
            let poly = Polynomial::mul(&x, &factor2);

            let sturm = SturmSequence::new(&poly, 0);
            let intervals = sturm.isolate_roots();
            (sturm.count_roots(), intervals)
        });
        let (count, intervals) = match worker.map(std::thread::JoinHandle::join) {
            Ok(Ok(result)) => result,
            _ => panic!("deep-bisection worker thread did not complete"),
        };
        assert_eq!(count, 2);
        assert_eq!(intervals.len(), 2, "got {intervals:?}");
        // Emission order is ascending, as in the recursive form.
        assert!(intervals[0].1 <= intervals[1].0);
    }

    #[test]
    fn test_sturm_sequence_isolates_roots_closer_than_legacy_epsilon() {
        let x = Polynomial::from_var(0);
        let r2 = BigRational::new(1.into(), 2_000_000.into());
        let factor2 = Polynomial::sub(&x, &Polynomial::constant(r2));
        let poly = Polynomial::mul(&x, &factor2);

        let sturm = SturmSequence::new(&poly, 0);
        assert_eq!(
            sturm.count_roots(),
            2,
            "two distinct roots at 0 and 1/2_000_000"
        );

        let intervals = sturm.isolate_roots();
        assert_eq!(
            intervals.len(),
            2,
            "roots closer than the old 1e-6 cutoff must still be isolated separately, got {intervals:?}"
        );

        // Each interval must be genuinely non-degenerate (lo < hi) and the
        // two intervals must be disjoint (no overlap) -- i.e. actually
        // isolating, not just two copies of the same merged range.
        for (lo, hi) in &intervals {
            assert!(lo < hi, "interval must be non-degenerate: ({lo}, {hi})");
        }
        let (lo0, hi0) = &intervals[0];
        let (lo1, hi1) = &intervals[1];
        assert!(
            hi0 <= lo1 || hi1 <= lo0,
            "intervals must be disjoint: {intervals:?}"
        );
    }

    #[test]
    fn test_sturm_sequence_no_roots() {
        // x^2 + 1 has no real roots
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[])]);
        let sturm = SturmSequence::new(&poly, 0);

        assert_eq!(sturm.count_roots(), 0);
    }

    #[test]
    fn test_sturm_sequence_cubic() {
        // x^3 - x = x(x-1)(x+1) has roots at -1, 0, 1
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 3)]), (-1, &[(0, 1)])]);
        let sturm = SturmSequence::new(&poly, 0);

        assert_eq!(sturm.count_roots(), 3);

        let intervals = sturm.isolate_roots();
        assert_eq!(intervals.len(), 3);
    }

    #[test]
    fn test_cad_point() {
        let p = CadPoint::rational(rat(5));
        assert!(p.is_rational());
        assert_eq!(p.as_rational(), Some(&rat(5)));
        assert_eq!(p.approximate(), rat(5));

        let alg = CadPoint::algebraic(
            rat(1),
            rat(2),
            Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-2, &[])]),
            1,
        );
        assert!(!alg.is_rational());
        let approx = alg.approximate();
        assert!(approx > rat(1) && approx < rat(2));
    }

    #[test]
    fn test_cad_lifter() {
        // Test using univariate polynomial constructor
        let poly = Polynomial::univariate(0, &[rat(-4), rat(0), rat(1)]); // -4 + 0*x + 1*x^2
        assert_eq!(poly.degree(0), 2);

        let sturm = SturmSequence::new(&poly, 0);
        assert_eq!(sturm.count_roots(), 2);

        let intervals = sturm.isolate_roots();
        assert_eq!(intervals.len(), 2);

        // Verify the roots are approximately at -2 and 2
        let mut roots: Vec<BigRational> = intervals
            .iter()
            .map(|(lo, hi)| (lo + hi) / BigRational::from_integer(2.into()))
            .collect();
        roots.sort();

        // Check first root is near -2
        assert!(roots[0] < rat(-1));
        assert!(roots[0] > rat(-3));

        // Check second root is near 2
        assert!(roots[1] > rat(1));
        assert!(roots[1] < rat(3));
    }

    #[test]
    fn test_full_projection() {
        let proj = CadProjection::new();

        // Simple linear polynomial: y - 1 (just a hyperplane)
        let linear = Polynomial::from_coeffs_int(&[
            (1, &[(1, 1)]), // y
            (-1, &[]),      // -1
        ]);

        let var_order = vec![0, 1]; // x, then y
        let projections = proj.full_projection(&[linear], &var_order);

        // Should have projections for both levels
        assert_eq!(projections.len(), 2);
    }

    #[test]
    fn test_cad_decomposer_univariate() {
        // Test CAD decomposition for univariate polynomial
        // p(x) = x^2 - 4 has roots at -2 and 2
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-4, &[])]);

        let config = CadConfig::default();
        let mut decomposer = CadDecomposer::new(config, vec![0]);

        let cells = decomposer.decompose(&[poly]).expect("Decomposition failed");

        // Should have 5 cells: (-inf, -2), {-2}, (-2, 2), {2}, (2, inf)
        assert_eq!(cells.len(), 5);
        assert_eq!(decomposer.num_cells(), 5);
    }

    #[test]
    fn test_cad_decomposer_bivariate() {
        // Test CAD decomposition for bivariate polynomial
        // p(x, y) = x + y
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 1)])]);

        let config = CadConfig {
            max_cells: 1000,
            ..Default::default()
        };
        let mut decomposer = CadDecomposer::new(config, vec![0, 1]);

        let result = decomposer.decompose(&[poly]);
        assert!(result.is_ok());

        let cells = result.expect("test operation should succeed");
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_sample_strategies() {
        let config_rational = CadConfig {
            sample_strategy: SampleStrategy::Rational,
            ..Default::default()
        };
        let decomposer_rational = CadDecomposer::new(config_rational, vec![0]);

        let sample = decomposer_rational.select_sample_point(Some(&rat(1)), Some(&rat(3)), 0);
        assert_eq!(sample, rat(2)); // Midpoint

        let config_integer = CadConfig {
            sample_strategy: SampleStrategy::Integer,
            ..Default::default()
        };
        let decomposer_integer = CadDecomposer::new(config_integer, vec![0]);

        let sample = decomposer_integer.select_sample_point(Some(&rat(0)), Some(&rat(10)), 0);
        // Should be near the midpoint, possibly an integer
        assert!(sample > rat(0) && sample < rat(10));
    }

    #[test]
    fn test_cad_error_too_many_cells() {
        // Test that we get an error when too many cells are generated
        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (-4, &[])]);

        let config = CadConfig {
            max_cells: 2, // Very small limit
            ..Default::default()
        };
        let mut decomposer = CadDecomposer::new(config, vec![0]);

        let result = decomposer.decompose(&[poly]);
        assert_eq!(result, Err(CadError::TooManyCells));
    }

    #[test]
    fn test_cad_with_auto_ordering() {
        // Test CAD with automatic variable ordering
        // p1 = x^2 + y - 1, p2 = x + y^2
        let p1 = Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (1, &[(1, 1)]), // y
            (-1, &[]),      // -1
        ]);
        let p2 = Polynomial::from_coeffs_int(&[
            (1, &[(0, 1)]), // x
            (1, &[(1, 2)]), // y^2
        ]);

        let config = CadConfig::default();
        let mut decomposer = CadDecomposer::with_auto_ordering(config, &[p1, p2]);

        // Should have computed an ordering
        assert!(!decomposer.var_order.is_empty());
        assert_eq!(decomposer.var_order.len(), 2);

        // Should be able to decompose
        let result = decomposer.decompose(&[
            Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 1)]), (-1, &[])]),
            Polynomial::from_coeffs_int(&[(1, &[(0, 1)]), (1, &[(1, 2)])]),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cad_with_different_ordering_strategies() {
        use crate::var_order::OrderingStrategy;

        let poly = Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (1, &[(1, 1)]), (-1, &[])]);

        // Test with Brown's heuristic
        let config_brown = CadConfig {
            ordering_strategy: Some(OrderingStrategy::Brown),
            ..Default::default()
        };
        let decomposer_brown =
            CadDecomposer::with_auto_ordering(config_brown, std::slice::from_ref(&poly));
        assert_eq!(decomposer_brown.var_order.len(), 2);

        // Test with min degree
        let config_min_deg = CadConfig {
            ordering_strategy: Some(OrderingStrategy::MinDegree),
            ..Default::default()
        };
        let decomposer_min_deg = CadDecomposer::with_auto_ordering(config_min_deg, &[poly]);
        assert_eq!(decomposer_min_deg.var_order.len(), 2);
    }
}
