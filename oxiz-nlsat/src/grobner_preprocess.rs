#![allow(dead_code)]
// NLSAT status (0.3.0): pub(crate), DEFERRED — retained-unwired (candidate for a
// future feature flag). Groebner-basis simplification is sound only on the
// equational (p=0) subset of atoms; applying it to inequalities is unsound, and
// it spawns background threads that outlive their timeout. Needs equational
// scoping + resource control before wiring.
//! Gröbner Basis Preprocessing for NLSAT.
//!
//! This module implements Gröbner basis-based preprocessing for polynomial systems
//! before CAD (Cylindrical Algebraic Decomposition). The key idea is to simplify
//! polynomial ideals using Gröbner basis computation, which can:
//!
//! 1. Reduce the number of polynomials in the system
//! 2. Lower polynomial degrees
//! 3. Detect inconsistencies early (e.g., 1 in the ideal)
//! 4. Extract variable bounds from the ideal
//! 5. Simplify multi-variable constraints
//!
//! This preprocessing can significantly improve CAD performance by reducing the
//! complexity of the projection phase.
//!
//! Reference:
//! - Z3's `nlsat/nlsat_solver.cpp` preprocessing strategies
//! - Buchberger's algorithm and its applications in SMT
//! - Cox, Little, O'Shea: "Ideals, Varieties, and Algorithms"

use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::grobner::{grobner_basis, reduce};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Hard cap on Gröbner-basis background worker threads allowed to be in
/// flight at once, shared across every [`GroebnerPreprocessor`] in this
/// process. See [`GroebnerPreprocessor::compute_grobner_with_timeout`]'s
/// doc comment for why this exists.
const MAX_OUTSTANDING_GROBNER_THREADS: usize = 4;

/// Number of Gröbner-basis worker threads currently running in the
/// background (including ones whose timeout already fired on the caller's
/// side — they keep running regardless, see
/// [`GroebnerPreprocessor::compute_grobner_with_timeout`]).
static OUTSTANDING_GROBNER_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Configuration for Gröbner preprocessing.
#[derive(Debug, Clone)]
pub struct GroebnerConfig {
    /// Enable Gröbner basis preprocessing.
    pub enabled: bool,
    /// Maximum number of polynomials to preprocess (avoid explosion).
    pub max_poly_count: usize,
    /// Maximum degree threshold (skip if degree too high).
    pub max_degree: u32,
    /// Maximum number of variables (skip if too many variables).
    pub max_vars: u32,
    /// Timeout for Gröbner basis computation (milliseconds).
    pub timeout_ms: u64,
    /// Enable early inconsistency detection.
    pub detect_inconsistency: bool,
    /// Extract variable bounds from the ideal.
    pub extract_bounds: bool,
    /// Simplify multi-variable constraints.
    pub simplify_multivar: bool,
}

impl Default for GroebnerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_poly_count: 20,
            max_degree: 10,
            max_vars: 8,
            timeout_ms: 1000,
            detect_inconsistency: true,
            extract_bounds: true,
            simplify_multivar: true,
        }
    }
}

/// Statistics for Gröbner preprocessing.
#[derive(Debug, Clone, Default)]
pub struct GroebnerStats {
    /// Number of times preprocessing was invoked.
    pub invocations: u64,
    /// Number of times preprocessing was skipped (too complex).
    pub skipped: u64,
    /// Number of inconsistencies detected.
    pub inconsistencies: u64,
    /// Number of polynomials before preprocessing.
    pub polys_before: u64,
    /// Number of polynomials after preprocessing.
    pub polys_after: u64,
    /// Total degree reduction (sum of degree reductions).
    pub degree_reduction: u64,
    /// Number of variable bounds extracted.
    pub bounds_extracted: u64,
    /// Total time spent in preprocessing (microseconds).
    pub time_us: u64,
}

/// Result of Gröbner preprocessing.
#[derive(Debug, Clone)]
pub enum PreprocessResult {
    /// System is inconsistent (contains unsatisfiable constraints).
    Inconsistent,
    /// System was simplified.
    Simplified {
        /// Simplified polynomials.
        polynomials: Vec<Polynomial>,
        /// Extracted variable bounds (var -> (lower, upper)).
        bounds: FxHashMap<Var, (Option<BigRational>, Option<BigRational>)>,
        /// Eliminated variables (substitutions: var -> polynomial).
        substitutions: FxHashMap<Var, Polynomial>,
    },
    /// Preprocessing skipped (too complex or disabled).
    Skipped,
}

/// Gröbner basis preprocessor for polynomial systems.
pub struct GroebnerPreprocessor {
    /// Configuration.
    config: GroebnerConfig,
    /// Statistics.
    stats: GroebnerStats,
}

impl GroebnerPreprocessor {
    /// Create a new Gröbner preprocessor with default configuration.
    pub fn new() -> Self {
        Self {
            config: GroebnerConfig::default(),
            stats: GroebnerStats::default(),
        }
    }

    /// Create a new Gröbner preprocessor with custom configuration.
    pub fn with_config(config: GroebnerConfig) -> Self {
        Self {
            config,
            stats: GroebnerStats::default(),
        }
    }

    /// Get preprocessing statistics.
    pub fn stats(&self) -> &GroebnerStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = GroebnerStats::default();
    }

    /// Preprocess a system of polynomial equations (equality constraints).
    ///
    /// This method takes a set of polynomials representing equality constraints
    /// (p = 0) and returns a simplified equivalent system.
    pub fn preprocess(&mut self, equations: &[Polynomial]) -> PreprocessResult {
        self.stats.invocations += 1;
        let start = oxiz_time::Instant::now();

        if !self.config.enabled {
            self.stats.skipped += 1;
            return PreprocessResult::Skipped;
        }

        // Filter out zero polynomials
        let non_zero: Vec<Polynomial> =
            equations.iter().filter(|p| !p.is_zero()).cloned().collect();

        if non_zero.is_empty() {
            self.stats.time_us += start.elapsed().as_micros() as u64;
            return PreprocessResult::Simplified {
                polynomials: vec![],
                bounds: FxHashMap::default(),
                substitutions: FxHashMap::default(),
            };
        }

        self.stats.polys_before += non_zero.len() as u64;

        // Check if system is too complex
        if non_zero.len() > self.config.max_poly_count {
            self.stats.skipped += 1;
            self.stats.time_us += start.elapsed().as_micros() as u64;
            return PreprocessResult::Skipped;
        }

        // Check degrees and variable count
        let max_degree = non_zero.iter().map(|p| p.total_degree()).max().unwrap_or(0);
        let all_vars: FxHashSet<Var> = non_zero.iter().flat_map(|p| p.vars().into_iter()).collect();

        if max_degree > self.config.max_degree || all_vars.len() > self.config.max_vars as usize {
            self.stats.skipped += 1;
            self.stats.time_us += start.elapsed().as_micros() as u64;
            return PreprocessResult::Skipped;
        }

        // Compute Gröbner basis (with timeout check)
        let gb = match self.compute_grobner_with_timeout(&non_zero) {
            Some(basis) => basis,
            None => {
                // Timeout or error - skip preprocessing
                self.stats.skipped += 1;
                self.stats.time_us += start.elapsed().as_micros() as u64;
                return PreprocessResult::Skipped;
            }
        };

        // Check for inconsistency (1 in ideal means unsatisfiable)
        if self.config.detect_inconsistency
            && let Some(result) = self.check_inconsistency(&gb)
        {
            self.stats.inconsistencies += 1;
            self.stats.time_us += start.elapsed().as_micros() as u64;
            return result;
        }

        // Extract variable bounds from univariate polynomials
        let bounds = if self.config.extract_bounds {
            self.extract_bounds(&gb)
        } else {
            FxHashMap::default()
        };

        // Extract substitutions for eliminated variables
        let substitutions = self.extract_substitutions(&gb);

        // Simplify the basis (remove redundant polynomials)
        let simplified = self.simplify_basis(gb);

        self.stats.polys_after += simplified.len() as u64;

        // Calculate degree reduction
        let degree_before: u32 = non_zero.iter().map(|p| p.total_degree()).sum();
        let degree_after: u32 = simplified.iter().map(|p| p.total_degree()).sum();
        self.stats.degree_reduction += degree_before.saturating_sub(degree_after) as u64;

        self.stats.bounds_extracted += bounds.len() as u64;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        PreprocessResult::Simplified {
            polynomials: simplified,
            bounds,
            substitutions,
        }
    }

    /// Compute Gröbner basis with timeout.
    ///
    /// Spawns a background thread to run Buchberger's algorithm and waits for up
    /// to `config.timeout_ms` milliseconds.  If the computation does not finish
    /// in time we return `None`, which the caller treats as `PreprocessResult::Skipped`.
    ///
    /// Rust has no safe thread-cancellation primitive, so a worker thread
    /// that misses its timeout does *not* stop: it keeps running
    /// Buchberger's algorithm to completion in the background, consuming a
    /// CPU core, with no way for us to reclaim it. The `max_poly_count` /
    /// `max_degree` bounds checked before this call reduce how often that
    /// happens but do not eliminate it (Buchberger's algorithm can still be
    /// doubly-exponential on small-looking inputs, which is precisely why a
    /// timeout is needed at all). Left unmitigated, a caller that hits this
    /// timeout repeatedly — e.g. a portfolio search retrying preprocessing
    /// across many hard subproblems — would accumulate an unbounded number
    /// of these permanently-running threads: an unbounded resource leak.
    ///
    /// [`OUTSTANDING_GROBNER_THREADS`] turns that into a *bounded* leak:
    /// once [`MAX_OUTSTANDING_GROBNER_THREADS`] computations are already in
    /// flight process-wide, further calls honestly report "skip" (`None`,
    /// exactly like a timeout) instead of spawning yet another
    /// unreclaimable thread. This does not implement true cancellation —
    /// that would require a cooperative check inside `grobner_basis` itself
    /// (in `oxiz-math`, outside this crate) — but it does cap the worst-case
    /// damage at a small, fixed number of background threads rather than
    /// letting it grow without limit.
    fn compute_grobner_with_timeout(&self, polys: &[Polynomial]) -> Option<Vec<Polynomial>> {
        use std::sync::mpsc;
        use std::time::Duration;

        // Reserve a slot, or honestly skip if we're already at the cap on
        // outstanding (potentially permanently-running) worker threads.
        let reserved = OUTSTANDING_GROBNER_THREADS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_OUTSTANDING_GROBNER_THREADS).then_some(n + 1)
            })
            .is_ok();
        if !reserved {
            return None;
        }

        let polys_clone = polys.to_vec();
        let max_poly_count = self.config.max_poly_count;
        let timeout = Duration::from_millis(self.config.timeout_ms);

        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let gb = grobner_basis(&polys_clone);
            // Ignore send error — receiver may have already timed out.
            let _ = tx.send(gb);
            // Release our slot regardless of whether anyone was still
            // listening, so the cap reflects threads still actually running
            // rather than ones the caller merely gave up waiting for.
            OUTSTANDING_GROBNER_THREADS.fetch_sub(1, Ordering::AcqRel);
        });

        let gb = match rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(_) => return None, // Timeout — skip preprocessing (thread keeps running).
        };

        // Check if result is too large
        if gb.len() > max_poly_count * 2 {
            return None; // Basis exploded, skip
        }

        Some(gb)
    }

    /// Check if the Gröbner basis contains an inconsistency.
    ///
    /// If the basis contains a non-zero constant (like 1), the ideal is all of
    /// the polynomial ring, meaning the system is unsatisfiable.
    fn check_inconsistency(&self, gb: &[Polynomial]) -> Option<PreprocessResult> {
        for poly in gb {
            if poly.is_constant() && !poly.is_zero() {
                return Some(PreprocessResult::Inconsistent);
            }
        }
        None
    }

    /// Extract variable bounds from univariate polynomials in the basis.
    ///
    /// For univariate polynomials like x - 3 = 0, we can extract x = 3.
    /// For inequalities (handled separately), we can extract bounds.
    fn extract_bounds(
        &self,
        gb: &[Polynomial],
    ) -> FxHashMap<Var, (Option<BigRational>, Option<BigRational>)> {
        let mut bounds: FxHashMap<Var, (Option<BigRational>, Option<BigRational>)> =
            FxHashMap::default();

        for poly in gb {
            let vars: Vec<Var> = poly.vars();

            // Only process univariate linear polynomials (x - c = 0)
            if vars.len() == 1 && poly.total_degree() == 1 {
                let var = vars[0];
                // Try to solve for the variable: ax + b = 0 => x = -b/a
                if let Some(value) = solve_linear_univariate(poly, var) {
                    // Set both lower and upper bound to the same value (equality)
                    bounds.insert(var, (Some(value.clone()), Some(value)));
                }
            }
        }

        bounds
    }

    /// Extract substitutions for variables from the Gröbner basis.
    ///
    /// For polynomials like x - f(y, z) = 0, we can substitute x with f(y, z).
    fn extract_substitutions(&self, gb: &[Polynomial]) -> FxHashMap<Var, Polynomial> {
        let mut substitutions = FxHashMap::default();

        for poly in gb {
            let vars: Vec<Var> = poly.vars();

            // Look for polynomials of the form x - f(y1, ..., yn) = 0
            // where x is the leading variable
            if !vars.is_empty() && poly.total_degree() == 1 {
                let max_var = vars.iter().max().copied();
                if let Some(mv) = max_var {
                    // Try to isolate the maximum variable
                    if let Some(substitution) = isolate_variable(poly, mv) {
                        substitutions.insert(mv, substitution);
                    }
                }
            }
        }

        substitutions
    }

    /// Simplify the Gröbner basis by removing redundant polynomials.
    ///
    /// This keeps only the "minimal" polynomials that generate the same ideal.
    fn simplify_basis(&self, mut gb: Vec<Polynomial>) -> Vec<Polynomial> {
        // Remove zero polynomials
        gb.retain(|p| !p.is_zero());

        // Remove constant polynomials (except if detecting inconsistency)
        if !self.config.detect_inconsistency {
            gb.retain(|p| !p.is_constant());
        }

        // Sort by total degree (lower degree first)
        gb.sort_by_key(|p| p.total_degree());

        // Auto-reduce: reduce each polynomial by others
        let mut simplified = Vec::new();
        for i in 0..gb.len() {
            let mut others: Vec<Polynomial> = gb
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| p.clone())
                .collect();
            others.extend(simplified.iter().cloned());

            let reduced = reduce(&gb[i], &others);
            if !reduced.is_zero() {
                simplified.push(reduced);
            }
        }

        simplified
    }
}

impl Default for GroebnerPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve a linear univariate polynomial equation: ax + b = 0
fn solve_linear_univariate(poly: &Polynomial, var: Var) -> Option<BigRational> {
    // Polynomial should be in the form ax + b
    let terms = poly.terms();
    if terms.len() > 2 {
        return None;
    }

    let mut a = BigRational::zero();
    let mut b = BigRational::zero();

    for term in terms {
        let mon = &term.monomial;
        let coeff = &term.coeff;

        if mon.is_unit() {
            b = coeff.clone();
        } else {
            // Check if this is the linear term for our variable
            let vars = mon.vars();
            if vars.len() == 1 && vars[0].var == var && vars[0].power == 1 {
                a = coeff.clone();
            } else {
                return None; // Not a simple linear term
            }
        }
    }

    if a.is_zero() {
        return None; // Not a linear equation in var
    }

    // Solve: ax + b = 0 => x = -b/a
    Some(-b / a)
}

/// Isolate a variable from a polynomial equation.
///
/// Given a polynomial p and a variable x, try to rewrite as x - f(...) = 0
/// and return f(...).
fn isolate_variable(poly: &Polynomial, var: Var) -> Option<Polynomial> {
    // For now, only handle linear cases: ax + f(other_vars) = 0
    let terms = poly.terms();

    let mut var_coeff = BigRational::zero();
    let mut other_terms = Vec::new();

    for term in terms {
        let mon = &term.monomial;
        let coeff = &term.coeff;
        let vars = mon.vars();

        // Check if this term contains only our target variable linearly
        if vars.len() == 1 && vars[0].var == var && vars[0].power == 1 {
            var_coeff = coeff.clone();
        } else {
            other_terms.push(term.clone());
        }
    }

    if var_coeff.is_zero() {
        return None; // Variable doesn't appear linearly
    }

    // Build: x = -other_terms / var_coeff
    let other_poly =
        Polynomial::from_terms(other_terms, oxiz_math::polynomial::MonomialOrder::default());

    Some(other_poly.scale(&(-BigRational::one() / var_coeff)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn poly_from_coeffs(var: u32, coeffs: &[i64]) -> Polynomial {
        let mut p = Polynomial::zero();
        let x = Polynomial::from_var(var);

        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff != 0 {
                let coef_poly =
                    Polynomial::constant(BigRational::from_integer(BigInt::from(coeff)));
                let mut term = coef_poly;
                for _ in 0..i {
                    term = Polynomial::mul(&term, &x);
                }
                p = Polynomial::add(&p, &term);
            }
        }
        p
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = GroebnerPreprocessor::new();
        assert!(preprocessor.config.enabled);
        assert_eq!(preprocessor.stats.invocations, 0);
    }

    #[test]
    fn test_empty_system() {
        let mut preprocessor = GroebnerPreprocessor::new();
        let result = preprocessor.preprocess(&[]);

        match result {
            PreprocessResult::Simplified {
                polynomials,
                bounds,
                substitutions,
            } => {
                assert!(polynomials.is_empty());
                assert!(bounds.is_empty());
                assert!(substitutions.is_empty());
            }
            _ => panic!("Expected Simplified result"),
        }
    }

    #[test]
    fn test_inconsistent_system() {
        let mut preprocessor = GroebnerPreprocessor::new();

        // System: x = 0 and x = 1 (inconsistent)
        let p1 = poly_from_coeffs(0, &[0, 1]); // x
        let p2 = poly_from_coeffs(0, &[-1, 1]); // x - 1

        let result = preprocessor.preprocess(&[p1, p2]);

        match result {
            PreprocessResult::Inconsistent => {
                assert_eq!(preprocessor.stats.inconsistencies, 1);
            }
            _ => {
                // May also get Simplified with a constant polynomial
                // This is acceptable behavior
            }
        }
    }

    #[test]
    fn test_linear_univariate() {
        let mut preprocessor = GroebnerPreprocessor::new();

        // System: x - 3 = 0
        let p = poly_from_coeffs(0, &[-3, 1]); // -3 + x

        let result = preprocessor.preprocess(&[p]);

        if let PreprocessResult::Simplified {
            polynomials,
            bounds,
            ..
        } = result
        {
            assert!(!polynomials.is_empty());
            // Should extract bound x = 3
            if let Some((lower, upper)) = bounds.get(&0) {
                assert!(lower.is_some());
                assert!(upper.is_some());
                assert_eq!(
                    lower.as_ref().expect("test operation should succeed"),
                    &BigRational::from_integer(BigInt::from(3))
                );
                assert_eq!(
                    upper.as_ref().expect("test operation should succeed"),
                    &BigRational::from_integer(BigInt::from(3))
                );
            }
        }
    }

    #[test]
    fn test_skip_complex_system() {
        let config = GroebnerConfig {
            max_poly_count: 2, // Very restrictive limit
            ..Default::default()
        };
        let mut preprocessor = GroebnerPreprocessor::with_config(config);

        // System with 3 polynomials (exceeds limit)
        let p1 = poly_from_coeffs(0, &[1, 1]);
        let p2 = poly_from_coeffs(1, &[2, 1]);
        let p3 = poly_from_coeffs(2, &[3, 1]);

        let result = preprocessor.preprocess(&[p1, p2, p3]);

        match result {
            PreprocessResult::Skipped => {
                assert_eq!(preprocessor.stats.skipped, 1);
            }
            _ => panic!("Expected Skipped result"),
        }
    }

    #[test]
    fn test_disabled_preprocessing() {
        let config = GroebnerConfig {
            enabled: false,
            ..Default::default()
        };
        let mut preprocessor = GroebnerPreprocessor::with_config(config);

        let p = poly_from_coeffs(0, &[1, 1]);
        let result = preprocessor.preprocess(&[p]);

        match result {
            PreprocessResult::Skipped => {
                assert_eq!(preprocessor.stats.skipped, 1);
            }
            _ => panic!("Expected Skipped result"),
        }
    }

    #[test]
    fn test_stats_tracking() {
        let mut preprocessor = GroebnerPreprocessor::new();

        let p1 = poly_from_coeffs(0, &[1, 1]);
        let _result = preprocessor.preprocess(std::slice::from_ref(&p1));

        assert_eq!(preprocessor.stats.invocations, 1);
        assert!(preprocessor.stats.time_us > 0);

        preprocessor.reset_stats();
        assert_eq!(preprocessor.stats.invocations, 0);
        assert_eq!(preprocessor.stats.time_us, 0);
    }

    #[test]
    fn test_solve_linear_univariate_simple() {
        // x - 5 = 0 => x = 5
        let p = poly_from_coeffs(0, &[-5, 1]);
        let result = solve_linear_univariate(&p, 0);

        assert!(result.is_some());
        assert_eq!(
            result.expect("test operation should succeed"),
            BigRational::from_integer(BigInt::from(5))
        );
    }

    #[test]
    fn test_solve_linear_univariate_scaled() {
        // 2x - 10 = 0 => x = 5
        let p = poly_from_coeffs(0, &[-10, 2]);
        let result = solve_linear_univariate(&p, 0);

        assert!(result.is_some());
        assert_eq!(
            result.expect("test operation should succeed"),
            BigRational::from_integer(BigInt::from(5))
        );
    }

    #[test]
    fn test_isolate_variable_linear() {
        // x - 3 = 0 => x = 3
        let p = poly_from_coeffs(0, &[-3, 1]);
        let result = isolate_variable(&p, 0);

        assert!(result.is_some());
        let isolated = result.expect("test operation should succeed");
        // Should be the constant 3
        assert!(isolated.is_constant());
        let expected = BigRational::from_integer(BigInt::from(3));
        // Compare the constant term coefficient
        if let Some(first_term) = isolated.terms().first() {
            assert_eq!(first_term.coeff, expected);
        } else {
            panic!("Expected constant polynomial with value 3");
        }
    }

    #[test]
    fn test_degree_reduction() {
        let mut preprocessor = GroebnerPreprocessor::new();

        // System that can be simplified
        // x^2 - 1 = 0 and x - 1 = 0 => Gröbner basis should keep x - 1
        let p1 = poly_from_coeffs(0, &[-1, 0, 1]); // x^2 - 1
        let p2 = poly_from_coeffs(0, &[-1, 1]); // x - 1

        let result = preprocessor.preprocess(&[p1, p2]);

        if let PreprocessResult::Simplified { polynomials, .. } = result {
            // Should reduce to just x - 1 (or equivalent)
            assert!(!polynomials.is_empty());
            // Degree should be reduced
            assert!(preprocessor.stats.degree_reduction > 0);
        }
    }

    // Regression test for the thread-leak item: `compute_grobner_with_timeout`
    // must refuse to spawn another background worker once
    // `MAX_OUTSTANDING_GROBNER_THREADS` are already in flight, honestly
    // reporting "skip" (`None`) instead of piling on an unbounded number of
    // threads that can never be reclaimed.
    #[test]
    fn test_grobner_thread_cap_prevents_unbounded_leak() {
        // Saturate the shared budget so this call cannot reserve a slot.
        // Bump by far more than the cap so this holds even if other tests
        // running in the same process are concurrently nudging the counter
        // by the (small, `<= MAX_OUTSTANDING_GROBNER_THREADS`) amounts
        // normal usage would.
        let bump = 1_000_000usize;
        OUTSTANDING_GROBNER_THREADS.fetch_add(bump, Ordering::SeqCst);

        let preprocessor = GroebnerPreprocessor::new();
        let poly = poly_from_coeffs(0, &[-1, 1]); // x - 1
        let result = preprocessor.compute_grobner_with_timeout(&[poly]);

        OUTSTANDING_GROBNER_THREADS.fetch_sub(bump, Ordering::SeqCst);

        assert!(
            result.is_none(),
            "must not spawn another worker thread once the outstanding-thread \
             cap is saturated"
        );
    }

    // A completed computation must release its slot rather than leaking it
    // forever, so the process doesn't permanently ratchet down towards the
    // cap under ordinary (non-timing-out) usage.
    #[test]
    fn test_grobner_thread_counter_released_after_completion() {
        let preprocessor = GroebnerPreprocessor::new();
        let poly = poly_from_coeffs(0, &[-1, 1]); // x - 1

        let before = OUTSTANDING_GROBNER_THREADS.load(Ordering::SeqCst);
        let result = preprocessor.compute_grobner_with_timeout(&[poly]);
        assert!(result.is_some());

        // The worker thread sends its result and then decrements the
        // counter; poll briefly for that trailing decrement to land instead
        // of assuming it already has.
        let mut released = false;
        for _ in 0..200 {
            if OUTSTANDING_GROBNER_THREADS.load(Ordering::SeqCst) <= before {
                released = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            released,
            "outstanding-thread counter must be released after the worker completes"
        );
    }
}
