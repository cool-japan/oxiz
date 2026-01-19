# oxiz-math TODO

Last Updated: 2026-01-06

Reference: Z3's `math/` directory at `../z3/src/math/`

## Progress: ~99% Complete

## Dependencies
- **oxiz-core**: Term representation for polynomial constraints

## Provides (enables other crates)
- **oxiz-nlsat**: Polynomial arithmetic, Sturm sequences, root isolation
- **oxiz-theories**: Simplex algorithm for LRA/LIA
- **oxiz-opt**: LP solver foundation, interior point method

---

## High Priority

### Simplex Algorithm
- [x] Basic simplex tableau implementation
- [x] Bland's anti-cycling rule
- [x] Incremental pivot operations
- [x] Unboundedness detection
- [x] Infeasibility detection with explanation
- [x] Delta-rational support for strict inequalities
- [x] Integration with theory solver interface
  - [x] Model extraction
  - [x] Feasibility checking
  - [x] Theory propagation
  - [x] Constraint assertion
  - [x] Conflict tracking and unsat core

### Polynomial Arithmetic
- [x] Univariate polynomial representation
- [x] Multivariate polynomial representation
- [x] Polynomial addition, subtraction, multiplication
- [x] Polynomial division with remainder
- [x] GCD computation
- [x] Square-free decomposition
- [x] Sturm sequence computation
- [x] Real root isolation (interval bisection)
- [x] Factorization (for NLSAT)
  - [x] Irreducibility testing
  - [x] Quadratic factorization
  - [x] Square-free factorization (Yun's algorithm)
  - [x] Integer square root for perfect squares
  - [x] Polynomial content computation

### Interval Arithmetic
- [x] Closed interval representation
- [x] Open/half-open intervals
- [x] Interval arithmetic operations (+, -, *, /)
- [x] Interval union and intersection
- [x] Bound propagation interface
  - [x] Convex hull operation
  - [x] Tighten lower/upper bounds
  - [x] Constraint propagation (add, sub, mul, div)
  - [x] Subset and overlap checking
  - [x] Interval widening

## Medium Priority

### Gröbner Bases
- [x] Buchberger's algorithm
- [x] F4/F5 algorithms (faster alternatives)
- [x] Ideal membership testing
- [x] Integration with NRA solver
  - [x] Polynomial constraint representation (equality, inequality)
  - [x] NraSolver with SAT checking
  - [x] Model extraction from Gröbner basis
  - [x] Polynomial simplification using Gröbner basis
  - [x] Ideal membership checking (implies_zero)

### Real Closure
- [x] Algebraic number representation
- [x] Root isolation for polynomials
- [x] Comparison of algebraic numbers
- [x] Arithmetic on algebraic numbers

### Linear Programming
- [x] Dense matrix operations
- [x] Sparse matrix support
- [x] Dual simplex
- [x] Interior point method (basic implementation exists)

## Low Priority

### Decision Diagrams
- [x] BDD (Binary Decision Diagrams)
- [x] ZDD (Zero-suppressed BDDs)
- [x] ADD (Algebraic Decision Diagrams)

### Hilbert Basis
- [x] Hilbert basis computation
- [x] Integer cone operations

## Enhancements (2026)

### NRA Solver Improvements
- [x] Enhanced inequality handling
  - [x] Simplification of inequalities using Gröbner basis
  - [x] Trivial satisfiability/unsatisfiability detection
  - [x] Support for all relation types (=, !=, >, >=, <, <=)
  - [x] Constant polynomial evaluation

### Integration & Testing
- [x] Cross-module integration tests
  - [x] Gröbner basis with root isolation
  - [x] NRA solver with algebraic numbers
  - [x] Interval arithmetic with polynomials
  - [x] Delta rationals ordering
  - [x] Matrix operations
  - [x] Polynomial GCD with factorization
  - [x] Real closure root isolation

### Polynomial Algorithm Improvements (January 2026)
- [x] Descartes' rule of signs for root isolation optimization
  - [x] Sign variation counting for positive roots
  - [x] Sign variation counting for negative roots
  - [x] Integration with root isolation to skip empty intervals
- [x] Subresultant polynomial remainder sequence (PRS)
  - [x] Subresultant normalization to prevent coefficient explosion
  - [x] More efficient than naive pseudo-remainder sequence
  - [x] Useful for GCD computation and resultant calculation
- [x] Orthogonal polynomial families
  - [x] Chebyshev polynomials of the first kind (T_n)
  - [x] Chebyshev polynomials of the second kind (U_n)
  - [x] Chebyshev nodes generation for interpolation
  - [x] Legendre polynomials (P_n) for Gaussian quadrature
  - [x] Hermite polynomials (H_n, physicist's version)
  - [x] Laguerre polynomials (L_n)
- [x] Efficient polynomial evaluation
  - [x] Horner's method for univariate polynomials
  - [x] Reduced multiplication count from O(n²) to O(n)

## Recent Enhancements (January 2026)

### Rational Arithmetic Extensions
- [x] Extended Euclidean algorithm (Bézout coefficients)
  - [x] GCD computation with linear combination coefficients
  - [x] Useful for Diophantine equations and modular arithmetic
- [x] Continued fractions implementation
  - [x] Continued fraction expansion
  - [x] Convergent computation
  - [x] Best rational approximation within tolerance
  - [x] Integration with interval refinement

### Modular Arithmetic (January 4, 2026)
- [x] Modular exponentiation
  - [x] Binary exponentiation: (base^exp) mod m
  - [x] Efficient O(log exp) algorithm
- [x] Modular multiplicative inverse
  - [x] Using extended GCD
  - [x] Verification with Fermat's little theorem for primes
- [x] Chinese Remainder Theorem
  - [x] Solve systems of congruences: x ≡ a_i (mod m_i)
  - [x] Returns solution and combined modulus
- [x] Linear Diophantine equations
  - [x] Solve ax + by = c
  - [x] Uses extended GCD for solution finding

### Polynomial Root Bounds
- [x] Cauchy's root bound (existing, now with public API)
  - [x] Simple, conservative bound: 1 + max(|a_i|/|a_n|)
- [x] Fujiwara's root bound
  - [x] Tighter bound than Cauchy: 2 * max(|a_i/a_n|^(1/(n-i)))
  - [x] Nth root approximation using binary search
- [x] Lagrange's positive root bound
  - [x] Specialized bound for positive roots only
  - [x] Useful for root isolation optimization

### Number Theory Extensions (January 4, 2026)
- [x] Primality testing
  - [x] Miller-Rabin probabilistic primality test
  - [x] Configurable number of rounds for accuracy
- [x] Integer factorization
  - [x] Trial division with configurable limit
  - [x] Pollard's rho algorithm for larger factors
- [x] Quadratic residues
  - [x] Jacobi symbol computation
  - [x] Legendre symbol for primes
  - [x] Quadratic reciprocity implementation
- [x] Euler's totient function (φ)
  - [x] Count of integers coprime to n
  - [x] Uses prime factorization
- [x] Perfect power detection
  - [x] Detect if n = a^b for some a, b > 1
  - [x] Binary search for efficient computation
- [x] Square-free integer testing
  - [x] Check if divisible by any perfect square
  - [x] Uses trial division factorization

### Advanced Number Theory & Combinatorics (January 4, 2026)
- [x] Divisor functions
  - [x] Tau function τ(n) - count of divisors
  - [x] Sigma function σ(n) - sum of divisors
  - [x] Efficient computation using prime factorization
- [x] Möbius function μ(n)
  - [x] Returns 1, -1, or 0 based on prime factorization
  - [x] Used in number theoretic inversions
- [x] Carmichael's lambda function λ(n)
  - [x] Exponent of the multiplicative group (ℤ/nℤ)*
  - [x] Generalizes Euler's totient
  - [x] Essential for RSA cryptography
- [x] Binary GCD (Stein's algorithm)
  - [x] More efficient than Euclidean algorithm
  - [x] Uses bitwise operations instead of division
  - [x] Better performance for large integers
- [x] Tonelli-Shanks algorithm
  - [x] Modular square root computation
  - [x] Solves x² ≡ n (mod p) for prime p
  - [x] Handles special cases (p ≡ 3 mod 4)
- [x] Combinatorial functions
  - [x] Factorial n!
  - [x] Binomial coefficient C(n,k)
  - [x] Optimized computation with symmetry

### Performance Optimizations (January 5, 2026)
- [x] Karatsuba multiplication for univariate polynomials
  - [x] O(n^1.585) complexity instead of O(n^2) for naive multiplication
  - [x] Automatic threshold-based selection (16+ terms)
  - [x] Recursive divide-and-conquer algorithm
  - [x] Particularly efficient for large-degree polynomial operations

### January 6, 2026 Enhancements (Morning)
- [x] Rational Function Arithmetic
  - [x] Quotient of polynomials representation (p/q)
  - [x] Basic arithmetic operations (add, sub, mul, div, neg)
  - [x] Automatic reduction to lowest terms using GCD
  - [x] Derivative computation using quotient rule
  - [x] Evaluation at points with zero-denominator detection
- [x] Advanced Root Finding
  - [x] Newton-Raphson iteration for polynomial root refinement
  - [x] Bound checking to ensure convergence
  - [x] Batch root refinement from isolating intervals
  - [x] Configurable tolerance and iteration limits
- [x] Taylor Series Expansion
  - [x] Taylor series expansion around arbitrary points
  - [x] Maclaurin series (Taylor around 0) convenience method
  - [x] Automatic computation of successive derivatives
  - [x] Factorial normalization for Taylor coefficients

### January 6, 2026 Enhancements (Afternoon)
- [x] Additional Root Finding Methods
  - [x] Bisection method for robust polynomial root finding
  - [x] Always converges when bracketing a root
  - [x] Sign change detection for interval selection
  - [x] Secant method for derivative-free root finding
  - [x] Two-point approximation of derivative
  - [x] Faster convergence than bisection
- [x] Rational Function Advanced Operations
  - [x] Proper/improper rational function detection
  - [x] Polynomial long division for improper fractions
  - [x] Separation into polynomial + proper fraction form
  - [x] Partial fraction decomposition framework (placeholder for full implementation)
- [x] Polynomial Integration (Antiderivative)
  - [x] Indefinite integration with respect to a variable
  - [x] Proper handling of constant terms
  - [x] Multivariate polynomial integration support
  - [x] Integration-differentiation round-trip verification
  - [x] Complements existing derivative functionality
  - [x] Definite integration over intervals [a, b]
  - [x] Fundamental theorem of calculus: F(b) - F(a)
- [x] Calculus and Optimization Features
  - [x] Critical point detection (finding extrema)
  - [x] Solving f'(x) = 0 using root isolation
  - [x] Useful for optimization problems
- [x] Numerical Integration Methods
  - [x] Trapezoidal rule for numerical approximation
  - [x] Simpson's rule for higher accuracy
  - [x] Configurable number of subintervals
  - [x] Verification against symbolic integration

### January 7, 2026 Enhancements
- [x] Matrix decompositions for linear algebra
  - [x] QR decomposition using modified Gram-Schmidt process
  - [x] Rational arithmetic adaptation (unnormalized orthogonal vectors)
  - [x] Exact reconstruction guarantee: QR = A
  - [x] Cholesky decomposition for symmetric positive definite matrices
  - [x] Perfect rational square detection for decomposition
  - [x] Lower triangular factorization: A = L * L^T
  - [x] Comprehensive test coverage for both decompositions
  - [x] Edge case handling (non-symmetric, non-positive-definite, etc.)
- [x] Matrix inverse computation
  - [x] Using LU decomposition with partial pivoting
  - [x] Efficient column-by-column solving via forward/backward substitution
  - [x] Identity matrix verification helper
  - [x] Comprehensive testing (2x2, 3x3, identity, singular, non-square)
  - [x] Clippy-compliant implementation (no warnings)
- [x] Multivariable calculus for polynomials
  - [x] Gradient computation (∇f) - vector of first partial derivatives
  - [x] Hessian matrix (H) - matrix of second partial derivatives
  - [x] Jacobian matrix for vector-valued polynomial functions
  - [x] Useful for optimization, convexity analysis, and Taylor expansions
  - [x] Full test coverage including symmetry verification
  - [x] Doctests for all three functions

## Future Enhancements

- [x] OxiBLAS integration for large-scale LP
  - **Priority:** Medium - for optimization problems with 1000+ variables
  - Implemented in `src/blas.rs` with DGEMM, DGEMV, DDOT, DNRM2, DSCAL, DAXPY
  - Cache-blocked GEMM for performance on large matrices
  - Pure Rust implementation optimized for vectorization
- [x] Arbitrary precision floating-point (MPFR-like)
  - **Priority:** Low - for extreme precision requirements
  - Implemented in `src/mpfr.rs` with ArbitraryFloat type
  - Configurable precision and rounding modes
  - Supports add, sub, mul, div, sqrt, and comparison operations

---

## Completed

- [x] Crate structure setup
- [x] Rational utilities module (floor, ceil, round, gcd, lcm, pow, etc.)
- [x] Basic interval arithmetic (bounds, operations, sign detection, division)
- [x] Basic polynomial arithmetic (multivariate, operations, GCD, derivatives)
- [x] Simplex tableau implementation (pivot, bounds, feasibility checking)
- [x] Delta-rational numbers for handling strict inequalities
- [x] Real root isolation using Sturm sequences and interval bisection
- [x] Square-free polynomial decomposition
- [x] Gröbner bases (Buchberger, F4/F5) with NRA integration
- [x] Complete number theory extensions (Miller-Rabin, Pollard rho, Jacobi/Legendre)
- [x] Orthogonal polynomials (Chebyshev, Legendre, Hermite, Laguerre)
- [x] Modular arithmetic (CRT, modular inverse, exponentiation)
- [x] Rational function arithmetic (quotient of polynomials, polynomial division)
- [x] Newton-Raphson root refinement
- [x] Bisection method for root finding
- [x] Secant method for root finding
- [x] Taylor series expansion
