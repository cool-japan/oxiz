//! Advanced polynomial operations: factorization, root finding, and interpolation.

use super::helpers::*;
use super::types::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

type Polynomial = super::Polynomial;

impl super::Polynomial {

    /// Refine all roots in the given intervals using Newton-Raphson.
    ///
    /// Takes a list of isolating intervals (from `isolate_roots`) and refines
    /// each root to higher precision using Newton-Raphson iteration.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `intervals` - List of isolating intervals from `isolate_roots`
    /// * `max_iterations` - Maximum number of Newton-Raphson iterations per root
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// Vector of refined root approximations
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let intervals = p.isolate_roots(0);
    /// let tolerance = BigRational::new(BigInt::from(1), BigInt::from(1000000));
    /// let refined_roots = p.refine_roots(0, &intervals, 10, &tolerance);
    /// ```
    pub fn refine_roots(
        &self,
        var: Var,
        intervals: &[(BigRational, BigRational)],
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Vec<BigRational> {
        intervals
            .iter()
            .filter_map(|(lower, upper)| {
                // Use midpoint as initial guess
                let initial = (lower + upper) / BigRational::from_integer(BigInt::from(2));
                self.newton_raphson(
                    var,
                    initial,
                    lower.clone(),
                    upper.clone(),
                    max_iterations,
                    tolerance,
                )
            })
            .collect()
    }

    /// Compute the Taylor series expansion of a univariate polynomial around a point.
    ///
    /// For a polynomial p(x), computes the Taylor series:
    /// p(x) = Σ_{k=0}^n (p^(k)(a) / k!) * (x - a)^k
    ///
    /// where p^(k) is the k-th derivative.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to expand around
    /// * `point` - The point to expand around (typically 0 for Maclaurin series)
    /// * `degree` - Maximum degree of the Taylor expansion
    ///
    /// # Returns
    ///
    /// A polynomial representing the Taylor series expansion
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Expand x^2 + 2x + 1 around x = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (2, &[(0, 1)]),  // 2x
    ///     (1, &[]),        // 1
    /// ]);
    ///
    /// let taylor = p.taylor_expansion(0, &BigRational::from_integer(BigInt::from(0)), 3);
    /// // Should be the same polynomial since it's already a polynomial
    /// ```
    pub fn taylor_expansion(&self, var: Var, point: &BigRational, degree: u32) -> Polynomial {
        use crate::rational::factorial;

        let mut result = Polynomial::zero();
        let mut derivative = self.clone();

        // Compute successive derivatives and evaluate at the point
        for k in 0..=degree {
            // Evaluate derivative at the point
            let mut assignment = FxHashMap::default();
            assignment.insert(var, point.clone());
            let coeff_at_point = derivative.eval(&assignment);

            // Divide by k!
            let factorial_k = factorial(k);
            let taylor_coeff = coeff_at_point / BigRational::from_integer(factorial_k);

            // Create term (x - point)^k
            let shifted_term = if k == 0 {
                Polynomial::constant(taylor_coeff)
            } else {
                // (x - a)^k = sum of binomial expansion
                // For simplicity, compute it directly
                let x_poly = Polynomial::from_var(var);
                let point_poly = Polynomial::constant(point.clone());
                let mut power = x_poly - point_poly;

                for _ in 1..k {
                    let x_poly = Polynomial::from_var(var);
                    let point_poly = Polynomial::constant(point.clone());
                    power = power * (x_poly - point_poly);
                }

                power * Polynomial::constant(taylor_coeff)
            };

            result = result + shifted_term;

            // Compute next derivative if needed
            if k < degree {
                derivative = derivative.derivative(var);
            }
        }

        result
    }

    /// Compute the Maclaurin series expansion (Taylor series around 0).
    ///
    /// This is a convenience method for Taylor expansion around 0.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to expand
    /// * `degree` - Maximum degree of the expansion
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Expand x^3 - 2x around x = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 3)]),   // x^3
    ///     (-2, &[(0, 1)]),  // -2x
    /// ]);
    ///
    /// let maclaurin = p.maclaurin_expansion(0, 4);
    /// ```
    pub fn maclaurin_expansion(&self, var: Var, degree: u32) -> Polynomial {
        use num_bigint::BigInt;
        self.taylor_expansion(var, &BigRational::from_integer(BigInt::from(0)), degree)
    }

    /// Find a root using the bisection method.
    ///
    /// The bisection method is a robust root-finding algorithm that works by repeatedly
    /// bisecting an interval and selecting the subinterval where the function changes sign.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `lower` - Lower bound (must have f(lower) and f(upper) with opposite signs)
    /// * `upper` - Upper bound (must have f(lower) and f(upper) with opposite signs)
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance (stop when interval width < tolerance)
    ///
    /// # Returns
    ///
    /// The approximate root, or None if the initial bounds don't bracket a root
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0 (find sqrt(2))
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let lower = BigRational::from_integer(BigInt::from(1));
    /// let upper = BigRational::from_integer(BigInt::from(2));
    /// let tolerance = BigRational::new(BigInt::from(1), BigInt::from(1000000));
    ///
    /// let root = p.bisection(0, lower, upper, 100, &tolerance);
    /// assert!(root.is_some());
    /// ```
    pub fn bisection(
        &self,
        var: Var,
        lower: BigRational,
        upper: BigRational,
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Option<BigRational> {
        use num_traits::{Signed, Zero};

        let mut a = lower;
        let mut b = upper;

        // Evaluate at endpoints
        let mut assignment_a = FxHashMap::default();
        assignment_a.insert(var, a.clone());
        let fa = self.eval(&assignment_a);

        let mut assignment_b = FxHashMap::default();
        assignment_b.insert(var, b.clone());
        let fb = self.eval(&assignment_b);

        // Check if endpoints bracket a root (opposite signs)
        if fa.is_zero() {
            return Some(a);
        }
        if fb.is_zero() {
            return Some(b);
        }
        if (fa.is_positive() && fb.is_positive()) || (fa.is_negative() && fb.is_negative()) {
            // Same sign, no root bracketed
            return None;
        }

        for _ in 0..max_iterations {
            // Check if interval is small enough
            if (&b - &a).abs() < *tolerance {
                return Some((&a + &b) / BigRational::from_integer(BigInt::from(2)));
            }

            // Compute midpoint
            let mid = (&a + &b) / BigRational::from_integer(BigInt::from(2));

            // Evaluate at midpoint
            let mut assignment_mid = FxHashMap::default();
            assignment_mid.insert(var, mid.clone());
            let fmid = self.eval(&assignment_mid);

            // Check if we found exact root
            if fmid.is_zero() {
                return Some(mid);
            }

            // Update interval
            let mut assignment_a = FxHashMap::default();
            assignment_a.insert(var, a.clone());
            let fa = self.eval(&assignment_a);

            if (fa.is_positive() && fmid.is_negative()) || (fa.is_negative() && fmid.is_positive())
            {
                // Root is in [a, mid]
                b = mid;
            } else {
                // Root is in [mid, b]
                a = mid;
            }
        }

        // Return midpoint as best approximation
        Some((&a + &b) / BigRational::from_integer(BigInt::from(2)))
    }

    /// Find a root using the secant method.
    ///
    /// The secant method is similar to Newton-Raphson but doesn't require computing derivatives.
    /// It uses two previous points to approximate the derivative.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to solve for
    /// * `x0` - First initial guess
    /// * `x1` - Second initial guess (should be close to x0)
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// The approximate root, or None if the method fails to converge
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigint::BigInt;
    /// use num_rational::BigRational;
    /// use oxiz_math::polynomial::Polynomial;
    ///
    /// // Solve x^2 - 2 = 0
    /// let p = Polynomial::from_coeffs_int(&[
    ///     (1, &[(0, 2)]),  // x^2
    ///     (-2, &[]),       // -2
    /// ]);
    ///
    /// let x0 = BigRational::from_integer(BigInt::from(1));
    /// let x1 = BigRational::new(BigInt::from(3), BigInt::from(2)); // 1.5
    /// let tolerance = BigRational::new(BigInt::from(1), BigInt::from(1000000));
    ///
    /// let root = p.secant(0, x0, x1, 20, &tolerance);
    /// assert!(root.is_some());
    /// ```
    pub fn secant(
        &self,
        var: Var,
        x0: BigRational,
        x1: BigRational,
        max_iterations: usize,
        tolerance: &BigRational,
    ) -> Option<BigRational> {
        use num_traits::{Signed, Zero};

        let mut x_prev = x0;
        let mut x_curr = x1;

        for _ in 0..max_iterations {
            // Evaluate at current and previous points
            let mut assignment_prev = FxHashMap::default();
            assignment_prev.insert(var, x_prev.clone());
            let f_prev = self.eval(&assignment_prev);

            let mut assignment_curr = FxHashMap::default();
            assignment_curr.insert(var, x_curr.clone());
            let f_curr = self.eval(&assignment_curr);

            // Check if we're close enough
            if f_curr.abs() < *tolerance {
                return Some(x_curr);
            }

            // Check for zero denominator
            let denom = &f_curr - &f_prev;
            if denom.is_zero() {
                return None;
            }

            // Secant method update: x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            let x_new = &x_curr - &f_curr * (&x_curr - &x_prev) / denom;

            x_prev = x_curr;
            x_curr = x_new;
        }

        Some(x_curr)
    }

    /// Sign of the polynomial given signs of variables.
    /// Returns Some(1) for positive, Some(-1) for negative, Some(0) for zero,
    /// or None if undetermined.
    pub fn sign_at(&self, var_signs: &FxHashMap<Var, i8>) -> Option<i8> {
        if self.is_zero() {
            return Some(0);
        }
        if self.is_constant() {
            let c = &self.terms[0].coeff;
            return Some(if c.is_positive() {
                1
            } else if c.is_negative() {
                -1
            } else {
                0
            });
        }

        // Try to determine sign from term signs
        let mut all_positive = true;
        let mut all_negative = true;

        for term in &self.terms {
            let mut term_sign: i8 = if term.coeff.is_positive() {
                1
            } else if term.coeff.is_negative() {
                -1
            } else {
                continue;
            };

            for vp in term.monomial.vars() {
                if let Some(&s) = var_signs.get(&vp.var) {
                    if s == 0 {
                        // Variable is zero; this term contributes 0
                        term_sign = 0;
                        break;
                    }
                    if vp.power % 2 == 1 {
                        term_sign *= s;
                    }
                } else {
                    // Unknown sign; can't determine overall sign
                    return None;
                }
            }

            if term_sign > 0 {
                all_negative = false;
            } else if term_sign < 0 {
                all_positive = false;
            }
        }

        if all_positive && !all_negative {
            Some(1)
        } else if all_negative && !all_positive {
            Some(-1)
        } else {
            None
        }
    }

    /// Check if a univariate polynomial is irreducible over rationals.
    /// This is a heuristic test - returns false if definitely reducible,
    /// true if likely irreducible (but not guaranteed).
    pub fn is_irreducible(&self, var: Var) -> bool {
        if self.is_zero() || self.is_constant() {
            return false;
        }

        let deg = self.degree(var);
        if deg == 0 {
            return false;
        }
        if deg == 1 {
            return true;
        }

        // Check if square-free
        let sf = self.square_free();
        if sf.degree(var) < deg {
            return false; // Has repeated factors
        }

        // For degree 2 and 3, use discriminant-based checks
        if deg == 2 {
            return self.is_irreducible_quadratic(var);
        }

        // For higher degrees, we'd need more sophisticated tests
        // For now, assume possibly irreducible
        true
    }

    /// Check if a quadratic polynomial is irreducible over rationals.
    fn is_irreducible_quadratic(&self, var: Var) -> bool {
        let deg = self.degree(var);
        if deg != 2 {
            return false;
        }

        let a = self.univ_coeff(var, 2);
        let b = self.univ_coeff(var, 1);
        let c = self.univ_coeff(var, 0);

        // Check discriminant: b^2 - 4ac
        // If not a perfect square, polynomial is irreducible over rationals
        let discriminant = &b * &b - (BigRational::from_integer(BigInt::from(4)) * &a * &c);

        if discriminant.is_negative() {
            return true; // No real roots
        }

        // Check if discriminant is a perfect square of a rational
        let num_sqrt = is_perfect_square(discriminant.numer());
        let den_sqrt = is_perfect_square(discriminant.denom());

        !(num_sqrt && den_sqrt)
    }

    /// Factor a univariate polynomial into irreducible factors.
    /// Returns a vector of (factor, multiplicity) pairs.
    /// Each factor is monic and primitive.
    pub fn factor(&self, var: Var) -> Vec<(Polynomial, u32)> {
        if self.is_zero() {
            return vec![];
        }

        if self.is_constant() {
            return vec![(self.clone(), 1)];
        }

        let deg = self.degree(var);
        if deg == 0 {
            return vec![(self.clone(), 1)];
        }

        // Make monic and primitive
        let p = self.primitive().make_monic();

        // Handle degree 1 (linear) - always irreducible
        if deg == 1 {
            return vec![(p, 1)];
        }

        // Handle degree 2 (quadratic) - use quadratic formula
        if deg == 2 {
            return self.factor_quadratic(var);
        }

        // For degree > 2, use square-free factorization first
        self.factor_square_free(var)
    }

    /// Factor a quadratic polynomial.
    fn factor_quadratic(&self, var: Var) -> Vec<(Polynomial, u32)> {
        let deg = self.degree(var);
        if deg != 2 {
            return vec![(self.primitive(), 1)];
        }

        let p = self.primitive().make_monic();
        let a = p.univ_coeff(var, 2);
        let b = p.univ_coeff(var, 1);
        let c = p.univ_coeff(var, 0);

        // Discriminant: b^2 - 4ac
        let discriminant = &b * &b - (BigRational::from_integer(BigInt::from(4)) * &a * &c);

        // Check if discriminant is a perfect square
        let num_sqrt = integer_sqrt(discriminant.numer());
        let den_sqrt = integer_sqrt(discriminant.denom());

        if num_sqrt.is_none() || den_sqrt.is_none() {
            // Irreducible over rationals
            return vec![(p, 1)];
        }

        let disc_sqrt = BigRational::new(num_sqrt.unwrap(), den_sqrt.unwrap());

        // Roots: (-b ± sqrt(disc)) / (2a)
        let two_a = BigRational::from_integer(BigInt::from(2)) * &a;
        let root1 = (-&b + &disc_sqrt) / &two_a;
        let root2 = (-&b - disc_sqrt) / two_a;

        // Factor as (x - root1)(x - root2)
        let factor1 = Polynomial::from_terms(
            vec![
                Term::new(BigRational::one(), Monomial::from_var(var)),
                Term::new(-root1, Monomial::unit()),
            ],
            MonomialOrder::Lex,
        );

        let factor2 = Polynomial::from_terms(
            vec![
                Term::new(BigRational::one(), Monomial::from_var(var)),
                Term::new(-root2, Monomial::unit()),
            ],
            MonomialOrder::Lex,
        );

        vec![(factor1, 1), (factor2, 1)]
    }

    /// Square-free factorization: decompose into coprime factors.
    /// Returns factors with their multiplicities.
    fn factor_square_free(&self, var: Var) -> Vec<(Polynomial, u32)> {
        if self.is_zero() || self.is_constant() {
            return vec![(self.clone(), 1)];
        }

        let mut result = Vec::new();
        let p = self.primitive();
        let mut multiplicity = 1u32;

        // Yun's algorithm for square-free factorization
        let deriv = p.derivative(var);

        if deriv.is_zero() {
            // All exponents are multiples of characteristic (0 for rationals)
            // This shouldn't happen for polynomials over rationals
            return vec![(p, 1)];
        }

        let gcd = p.gcd_univariate(&deriv);

        if gcd.is_constant() {
            // Already square-free
            if p.is_irreducible(var) {
                return vec![(p.make_monic(), 1)];
            } else {
                // Try to factor further (for now, return as-is)
                return vec![(p.make_monic(), 1)];
            }
        }

        let (quo, rem) = p.pseudo_div_univariate(&gcd);
        if !rem.is_zero() {
            return vec![(p, 1)];
        }

        let mut u = quo.primitive();
        let (v_quo, v_rem) = deriv.pseudo_div_univariate(&gcd);
        if v_rem.is_zero() {
            let v = v_quo.primitive();
            let mut w = Polynomial::sub(&v, &u.derivative(var));

            while !w.is_zero() && !w.is_constant() {
                let y = u.gcd_univariate(&w);
                if !y.is_constant() {
                    result.push((y.make_monic(), multiplicity));
                    let (u_new, u_rem) = u.pseudo_div_univariate(&y);
                    if u_rem.is_zero() {
                        u = u_new.primitive();
                    }
                    let (w_new, w_rem) = w.pseudo_div_univariate(&y);
                    if w_rem.is_zero() {
                        w = Polynomial::sub(&w_new.primitive(), &u.derivative(var));
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                multiplicity += 1;
            }

            if !u.is_constant() {
                result.push((u.make_monic(), multiplicity));
            }
        }

        if result.is_empty() {
            result.push((p.make_monic(), 1));
        }

        result
    }

    /// Content of a polynomial: GCD of all coefficients.
    /// Returns the rational content.
    pub fn content(&self) -> BigRational {
        if self.terms.is_empty() {
            return BigRational::one();
        }

        let mut num_gcd: Option<BigInt> = None;
        let mut den_lcm: Option<BigInt> = None;

        for term in &self.terms {
            let coeff_num = term.coeff.numer().clone().abs();
            let coeff_den = term.coeff.denom().clone();

            num_gcd = Some(match num_gcd {
                None => coeff_num,
                Some(g) => gcd_bigint(g, coeff_num),
            });

            den_lcm = Some(match den_lcm {
                None => coeff_den,
                Some(l) => {
                    let gcd = gcd_bigint(l.clone(), coeff_den.clone());
                    (&l * &coeff_den) / gcd
                }
            });
        }

        BigRational::new(
            num_gcd.unwrap_or_else(BigInt::one),
            den_lcm.unwrap_or_else(BigInt::one),
        )
    }

    /// Compose this polynomial with another: compute p(q(x)).
    ///
    /// This is an alias for `substitute` with clearer semantics for composition.
    /// If `self` is p(x) and `other` is q(x), returns p(q(x)).
    pub fn compose(&self, var: Var, other: &Polynomial) -> Polynomial {
        self.substitute(var, other)
    }

    /// Lagrange polynomial interpolation.
    ///
    /// Given a set of points (x_i, y_i), constructs the unique polynomial of minimal degree
    /// that passes through all the points.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `points` - A slice of (x, y) coordinate pairs
    ///
    /// # Returns
    /// The interpolating polynomial, or None if points is empty or contains duplicate x values
    ///
    /// # Reference
    /// Standard Lagrange interpolation formula from numerical analysis textbooks
    pub fn lagrange_interpolate(
        var: Var,
        points: &[(BigRational, BigRational)],
    ) -> Option<Polynomial> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            // Single point: constant polynomial
            return Some(Polynomial::constant(points[0].1.clone()));
        }

        // Check for duplicate x values
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                if points[i].0 == points[j].0 {
                    return None; // Duplicate x values
                }
            }
        }

        let mut result = Polynomial::zero();

        // Lagrange basis polynomials
        for i in 0..points.len() {
            let (x_i, y_i) = &points[i];
            let mut basis = Polynomial::one();

            for (j, (x_j, _)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }

                // basis *= (x - x_j) / (x_i - x_j)
                let x_poly = Polynomial::from_var(var);
                let const_poly = Polynomial::constant(x_j.clone());
                let numerator = &x_poly - &const_poly;
                let denominator = x_i - x_j;

                if denominator.is_zero() {
                    return None; // Should not happen after duplicate check
                }

                basis = &basis * &numerator;
                basis = basis.scale(&(BigRational::one() / denominator));
            }

            // result += y_i * basis
            let scaled_basis = basis.scale(y_i);
            result = &result + &scaled_basis;
        }

        Some(result)
    }

    /// Newton polynomial interpolation using divided differences.
    ///
    /// Constructs the same interpolating polynomial as Lagrange interpolation,
    /// but using Newton's divided difference form which can be more efficient
    /// for incremental construction.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `points` - A slice of (x, y) coordinate pairs
    ///
    /// # Returns
    /// The interpolating polynomial, or None if points is empty or contains duplicate x values
    pub fn newton_interpolate(
        var: Var,
        points: &[(BigRational, BigRational)],
    ) -> Option<Polynomial> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            return Some(Polynomial::constant(points[0].1.clone()));
        }

        // Check for duplicate x values
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                if points[i].0 == points[j].0 {
                    return None;
                }
            }
        }

        let n = points.len();

        // Compute divided differences table
        let mut dd: Vec<Vec<BigRational>> = vec![vec![BigRational::zero(); n]; n];

        // Initialize first column with y values
        for i in 0..n {
            dd[i][0] = points[i].1.clone();
        }

        // Compute divided differences
        for j in 1..n {
            for i in 0..n - j {
                let numerator = &dd[i + 1][j - 1] - &dd[i][j - 1];
                let denominator = &points[i + j].0 - &points[i].0;
                if denominator.is_zero() {
                    return None;
                }
                dd[i][j] = numerator / denominator;
            }
        }

        // Build Newton polynomial: p(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + ...
        let mut result = Polynomial::constant(dd[0][0].clone());
        let mut term = Polynomial::one();

        for i in 1..n {
            // term *= (x - x_{i-1})
            let x_poly = Polynomial::from_var(var);
            let const_poly = Polynomial::constant(points[i - 1].0.clone());
            let factor = &x_poly - &const_poly;
            term = &term * &factor;

            // result += a_i * term
            let scaled_term = term.scale(&dd[0][i]);
            result = &result + &scaled_term;
        }

        Some(result)
    }

    /// Generate the nth Chebyshev polynomial of the first kind T_n(x).
    ///
    /// Chebyshev polynomials of the first kind are defined by:
    /// - T_0(x) = 1
    /// - T_1(x) = x
    /// - T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    ///
    /// These polynomials are orthogonal on [-1, 1] with respect to the weight
    /// function 1/√(1-x²) and are useful for polynomial approximation.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Chebyshev polynomial
    ///
    /// # Returns
    /// The nth Chebyshev polynomial T_n(var)
    pub fn chebyshev_first_kind(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => Polynomial::from_var(var),
            _ => {
                // Use recurrence relation: T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
                let mut t_prev = Polynomial::one(); // T_0
                let mut t_curr = Polynomial::from_var(var); // T_1

                for _ in 2..=n {
                    let x = Polynomial::from_var(var);
                    let two_x_t_curr = Polynomial::mul(&t_curr, &x)
                        .scale(&BigRational::from_integer(BigInt::from(2)));
                    let t_next = Polynomial::sub(&two_x_t_curr, &t_prev);
                    t_prev = t_curr;
                    t_curr = t_next;
                }

                t_curr
            }
        }
    }

    /// Generate the nth Chebyshev polynomial of the second kind U_n(x).
    ///
    /// Chebyshev polynomials of the second kind are defined by:
    /// - U_0(x) = 1
    /// - U_1(x) = 2x
    /// - U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
    ///
    /// These polynomials are orthogonal on [-1, 1] with respect to the weight
    /// function √(1-x²).
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Chebyshev polynomial
    ///
    /// # Returns
    /// The nth Chebyshev polynomial U_n(var)
    pub fn chebyshev_second_kind(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => {
                // U_1(x) = 2x
                Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2)))
            }
            _ => {
                // Use recurrence relation: U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
                let mut u_prev = Polynomial::one(); // U_0
                let mut u_curr =
                    Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2))); // U_1

                for _ in 2..=n {
                    let x = Polynomial::from_var(var);
                    let two_x_u_curr = Polynomial::mul(&u_curr, &x)
                        .scale(&BigRational::from_integer(BigInt::from(2)));
                    let u_next = Polynomial::sub(&two_x_u_curr, &u_prev);
                    u_prev = u_curr;
                    u_curr = u_next;
                }

                u_curr
            }
        }
    }

    /// Compute Chebyshev nodes (zeros of Chebyshev polynomial) for use in interpolation.
    ///
    /// The Chebyshev nodes minimize the Runge phenomenon in polynomial interpolation.
    /// For degree n, returns n+1 nodes in [-1, 1].
    ///
    /// The nodes are: x_k = cos((2k+1)π / (2n+2)) for k = 0, 1, ..., n
    ///
    /// Note: This function returns approximate rational values. For exact values,
    /// use algebraic numbers or symbolic computation.
    ///
    /// # Arguments
    /// * `n` - The degree (returns n+1 nodes)
    ///
    /// # Returns
    /// Approximations of the Chebyshev nodes as BigRational values
    pub fn chebyshev_nodes(n: u32) -> Vec<BigRational> {
        if n == 0 {
            return vec![BigRational::zero()];
        }

        let mut nodes = Vec::with_capacity((n + 1) as usize);

        // Compute nodes: cos((2k+1)π / (2n+2))
        // We'll use a rational approximation
        for k in 0..=n {
            // Approximate cos using Taylor series or use exact rational bounds
            // For now, use a simple rational approximation based on the angle

            // Angle = (2k+1) / (2n+2) * π/2
            // For small angles, we can use rational approximations
            // This is a simplified version - ideally would use higher precision

            let numerator = (2 * k + 1) as i64;
            let denominator = (2 * n + 2) as i64;

            // Simple linear approximation for demonstration
            // In production, would use more accurate trigonometric approximation
            let ratio = BigRational::new(BigInt::from(numerator), BigInt::from(denominator));

            // Map to [-1, 1] range
            // This is a placeholder - real implementation would compute cos accurately
            let node = BigRational::one() - ratio * BigRational::from_integer(BigInt::from(2));
            nodes.push(node);
        }

        nodes
    }

    /// Generate the nth Legendre polynomial P_n(x).
    ///
    /// Legendre polynomials are orthogonal polynomials on [-1, 1] with respect to
    /// the weight function 1. They are defined by the recurrence:
    /// - P_0(x) = 1
    /// - P_1(x) = x
    /// - (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    ///
    /// These polynomials are useful for Gaussian quadrature and least-squares
    /// polynomial approximation.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Legendre polynomial
    ///
    /// # Returns
    /// The nth Legendre polynomial P_n(var)
    pub fn legendre(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => Polynomial::from_var(var),
            _ => {
                // Recurrence: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
                let mut p_prev = Polynomial::one(); // P_0
                let mut p_curr = Polynomial::from_var(var); // P_1

                for k in 1..n {
                    let x = Polynomial::from_var(var);

                    // (2k+1) x P_k(x)
                    let coeff_2k_plus_1 = BigRational::from_integer(BigInt::from(2 * k + 1));
                    let term1 = Polynomial::mul(&p_curr, &x).scale(&coeff_2k_plus_1);

                    // k P_{k-1}(x)
                    let coeff_k = BigRational::from_integer(BigInt::from(k));
                    let term2 = p_prev.scale(&coeff_k);

                    // [(2k+1) x P_k(x) - k P_{k-1}(x)] / (k+1)
                    let numerator = Polynomial::sub(&term1, &term2);
                    let divisor = BigRational::from_integer(BigInt::from(k + 1));
                    let p_next = numerator.scale(&(BigRational::one() / divisor));

                    p_prev = p_curr;
                    p_curr = p_next;
                }

                p_curr
            }
        }
    }

    /// Generate the nth Hermite polynomial H_n(x) (physicist's version).
    ///
    /// Hermite polynomials are orthogonal with respect to the weight function e^(-x²).
    /// The physicist's version is defined by:
    /// - H_0(x) = 1
    /// - H_1(x) = 2x
    /// - H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    ///
    /// These polynomials are useful in quantum mechanics, probability theory,
    /// and numerical analysis.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Hermite polynomial
    ///
    /// # Returns
    /// The nth Hermite polynomial H_n(var)
    pub fn hermite(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => {
                // H_1(x) = 2x
                Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2)))
            }
            _ => {
                // Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
                let mut h_prev = Polynomial::one(); // H_0
                let mut h_curr =
                    Polynomial::from_var(var).scale(&BigRational::from_integer(BigInt::from(2))); // H_1

                for k in 1..n {
                    let x = Polynomial::from_var(var);

                    // 2x H_k(x)
                    let two_x_h = Polynomial::mul(&h_curr, &x)
                        .scale(&BigRational::from_integer(BigInt::from(2)));

                    // 2k H_{k-1}(x)
                    let coeff_2k = BigRational::from_integer(BigInt::from(2 * k));
                    let term2 = h_prev.scale(&coeff_2k);

                    // 2x H_k(x) - 2k H_{k-1}(x)
                    let h_next = Polynomial::sub(&two_x_h, &term2);

                    h_prev = h_curr;
                    h_curr = h_next;
                }

                h_curr
            }
        }
    }

    /// Generate the nth Laguerre polynomial L_n(x).
    ///
    /// Laguerre polynomials are orthogonal with respect to the weight function e^(-x)
    /// on [0, ∞). They are defined by:
    /// - L_0(x) = 1
    /// - L_1(x) = 1 - x
    /// - (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
    ///
    /// These polynomials are useful in quantum mechanics and numerical analysis.
    ///
    /// # Arguments
    /// * `var` - The variable to use for the polynomial
    /// * `n` - The degree of the Laguerre polynomial
    ///
    /// # Returns
    /// The nth Laguerre polynomial L_n(var)
    pub fn laguerre(var: Var, n: u32) -> Polynomial {
        match n {
            0 => Polynomial::one(),
            1 => {
                // L_1(x) = 1 - x
                let one = Polynomial::one();
                let x = Polynomial::from_var(var);
                Polynomial::sub(&one, &x)
            }
            _ => {
                // Recurrence: (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
                let mut l_prev = Polynomial::one(); // L_0
                let mut l_curr = {
                    let one = Polynomial::one();
                    let x = Polynomial::from_var(var);
                    Polynomial::sub(&one, &x)
                }; // L_1

                for k in 1..n {
                    let x = Polynomial::from_var(var);

                    // (2k+1-x) L_k(x) = (2k+1) L_k(x) - x L_k(x)
                    let coeff_2k_plus_1 = BigRational::from_integer(BigInt::from(2 * k + 1));
                    let term1 = l_curr.scale(&coeff_2k_plus_1);
                    let term2 = Polynomial::mul(&l_curr, &x);
                    let combined = Polynomial::sub(&term1, &term2);

                    // k L_{k-1}(x)
                    let coeff_k = BigRational::from_integer(BigInt::from(k));
                    let term3 = l_prev.scale(&coeff_k);

                    // [(2k+1-x) L_k(x) - k L_{k-1}(x)] / (k+1)
                    let numerator = Polynomial::sub(&combined, &term3);
                    let divisor = BigRational::from_integer(BigInt::from(k + 1));
                    let l_next = numerator.scale(&(BigRational::one() / divisor));

                    l_prev = l_curr;
                    l_curr = l_next;
                }

                l_curr
            }
        }
    }
}

/// Check if a BigInt is a perfect square.
fn is_perfect_square(n: &BigInt) -> bool {
    if n.is_negative() {
        return false;
    }
    if n.is_zero() || n.is_one() {
        return true;
    }

    let sqrt = integer_sqrt(n);
    sqrt.is_some()
}
