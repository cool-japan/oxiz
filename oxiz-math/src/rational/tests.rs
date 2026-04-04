use super::*;
use num_bigint::BigInt;
use num_rational::BigRational;

fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn int_rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

#[test]
fn test_floor() {
    assert_eq!(floor(&rat(7, 2)), BigInt::from(3));
    assert_eq!(floor(&rat(-7, 2)), BigInt::from(-4));
    assert_eq!(floor(&int_rat(5)), BigInt::from(5));
}

#[test]
fn test_ceil() {
    assert_eq!(ceil(&rat(7, 2)), BigInt::from(4));
    assert_eq!(ceil(&rat(-7, 2)), BigInt::from(-3));
    assert_eq!(ceil(&int_rat(5)), BigInt::from(5));
}

#[test]
fn test_round() {
    assert_eq!(round(&rat(7, 2)), BigInt::from(4)); // 3.5 -> 4 (round to even)
    assert_eq!(round(&rat(5, 2)), BigInt::from(2)); // 2.5 -> 2 (round to even)
    assert_eq!(round(&rat(9, 2)), BigInt::from(4)); // 4.5 -> 4 (round to even)
}

#[test]
fn test_frac() {
    assert_eq!(frac(&rat(7, 2)), rat(1, 2));
    assert_eq!(frac(&int_rat(5)), int_rat(0));
}

#[test]
fn test_gcd() {
    let a = rat(6, 1);
    let b = rat(9, 1);
    assert_eq!(gcd(&a, &b), int_rat(3));
}

#[test]
fn test_lcm() {
    let a = rat(6, 1);
    let b = rat(9, 1);
    assert_eq!(lcm(&a, &b), int_rat(18));
}

#[test]
fn test_pow() {
    let r = rat(2, 3);
    assert_eq!(pow_int(&r, 0), int_rat(1));
    assert_eq!(pow_int(&r, 1), rat(2, 3));
    assert_eq!(pow_int(&r, 2), rat(4, 9));
    assert_eq!(pow_int(&r, -1), rat(3, 2));
}

#[test]
fn test_sign() {
    assert_eq!(sign(&rat(5, 1)), 1);
    assert_eq!(sign(&rat(-5, 1)), -1);
    assert_eq!(sign(&rat(0, 1)), 0);
}

#[test]
fn test_clamp() {
    let val = rat(5, 1);
    let min_val = rat(0, 1);
    let max_val = rat(10, 1);
    assert_eq!(clamp(&val, &min_val, &max_val), rat(5, 1));
    assert_eq!(clamp(&rat(-5, 1), &min_val, &max_val), rat(0, 1));
    assert_eq!(clamp(&rat(15, 1), &min_val, &max_val), rat(10, 1));
}

#[test]
fn test_min_max() {
    let a = rat(3, 1);
    let b = rat(5, 1);
    assert_eq!(min(&a, &b), rat(3, 1));
    assert_eq!(max(&a, &b), rat(5, 1));
}

#[test]
fn test_mediant() {
    let a = rat(1, 2);
    let b = rat(2, 3);
    // Mediant: (1+2)/(2+3) = 3/5
    assert_eq!(mediant(&a, &b), rat(3, 5));
}

#[test]
fn test_approx_eq() {
    let a = rat(1, 3);
    let b = rat(333, 1000);
    let epsilon = rat(1, 100);
    assert!(approx_eq(&a, &b, &epsilon));
}

#[test]
fn test_gcd_extended() {
    let a = BigInt::from(240);
    let b = BigInt::from(46);
    let (gcd, x, y) = gcd_extended(a.clone(), b.clone());
    assert_eq!(gcd, BigInt::from(2));
    assert_eq!(&a * &x + &b * &y, gcd);

    // Test with coprime numbers
    let a = BigInt::from(17);
    let b = BigInt::from(13);
    let (gcd, x, y) = gcd_extended(a.clone(), b.clone());
    assert_eq!(gcd, BigInt::from(1));
    assert_eq!(&a * &x + &b * &y, gcd);
}

#[test]
fn test_continued_fraction() {
    // 22/7 = 3 + 1/7
    let r = rat(22, 7);
    let cf = continued_fraction(&r);
    assert_eq!(cf, vec![BigInt::from(3), BigInt::from(7)]);

    // 3/1 = 3
    let r = rat(3, 1);
    let cf = continued_fraction(&r);
    assert_eq!(cf, vec![BigInt::from(3)]);

    // 7/3 = 2 + 1/3
    let r = rat(7, 3);
    let cf = continued_fraction(&r);
    assert_eq!(cf, vec![BigInt::from(2), BigInt::from(3)]);
}

#[test]
fn test_from_continued_fraction() {
    let cf = vec![BigInt::from(3), BigInt::from(7)];
    let r = from_continued_fraction(&cf);
    assert_eq!(r, rat(22, 7));

    let cf = vec![BigInt::from(2), BigInt::from(3)];
    let r = from_continued_fraction(&cf);
    assert_eq!(r, rat(7, 3));
}

#[test]
fn test_continued_fraction_roundtrip() {
    let r = rat(355, 113); // Better pi approximation
    let cf = continued_fraction(&r);
    let reconstructed = from_continued_fraction(&cf);
    assert_eq!(r, reconstructed);
}

#[test]
fn test_convergents() {
    // For 22/7
    let cf = vec![BigInt::from(3), BigInt::from(7)];
    let convs = convergents(&cf);
    assert_eq!(convs.len(), 2);
    assert_eq!(convs[0], rat(3, 1));
    assert_eq!(convs[1], rat(22, 7));
}

#[test]
fn test_best_rational_approximation() {
    // Approximate pi approx 3.14159...
    let pi_approx = rat(31416, 10000);
    let epsilon = rat(1, 100);
    let approx = best_rational_approximation(&pi_approx, &epsilon);

    // Should find a simple approximation
    assert!((pi_approx - &approx).abs() <= epsilon);

    // Denominator should be reasonably small
    assert!(approx.denom() < &BigInt::from(1000));
}

#[test]
fn test_mod_pow() {
    // 2^10 mod 1000 = 1024 mod 1000 = 24
    let base = BigInt::from(2);
    let exp = BigInt::from(10);
    let m = BigInt::from(1000);
    assert_eq!(mod_pow(&base, &exp, &m), BigInt::from(24));

    // 3^5 mod 7 = 243 mod 7 = 5
    let base = BigInt::from(3);
    let exp = BigInt::from(5);
    let m = BigInt::from(7);
    assert_eq!(mod_pow(&base, &exp, &m), BigInt::from(5));

    // 0^n mod m = 0
    let base = BigInt::from(0);
    let exp = BigInt::from(100);
    let m = BigInt::from(7);
    assert_eq!(mod_pow(&base, &exp, &m), BigInt::from(0));

    // a^0 mod m = 1
    let base = BigInt::from(123);
    let exp = BigInt::from(0);
    let m = BigInt::from(7);
    assert_eq!(mod_pow(&base, &exp, &m), BigInt::from(1));
}

#[test]
fn test_mod_inverse() {
    // 3 * 5 equiv 1 (mod 7)
    let a = BigInt::from(3);
    let m = BigInt::from(7);
    let inv = mod_inverse(&a, &m).expect("test operation should succeed");
    assert_eq!((&a * &inv) % &m, BigInt::from(1));

    // 15 * 7 equiv 1 (mod 26)
    let a = BigInt::from(15);
    let m = BigInt::from(26);
    let inv = mod_inverse(&a, &m).expect("test operation should succeed");
    assert_eq!((&a * &inv) % &m, BigInt::from(1));

    // 2 has no inverse mod 4 (not coprime)
    let a = BigInt::from(2);
    let m = BigInt::from(4);
    assert!(mod_inverse(&a, &m).is_none());
}

#[test]
fn test_chinese_remainder() {
    // x equiv 2 (mod 3), x equiv 3 (mod 5), x equiv 2 (mod 7)
    // Solution: x = 23 (mod 105)
    let congruences = vec![
        (BigInt::from(2), BigInt::from(3)),
        (BigInt::from(3), BigInt::from(5)),
        (BigInt::from(2), BigInt::from(7)),
    ];
    let (x, m) = chinese_remainder(&congruences).expect("test operation should succeed");

    assert_eq!(m, BigInt::from(105)); // 3 * 5 * 7
    assert_eq!(&x % BigInt::from(3), BigInt::from(2));
    assert_eq!(&x % BigInt::from(5), BigInt::from(3));
    assert_eq!(&x % BigInt::from(7), BigInt::from(2));

    // Simple case: x equiv 1 (mod 2), x equiv 2 (mod 3)
    // Solution: x = 5 (mod 6)
    let congruences = vec![
        (BigInt::from(1), BigInt::from(2)),
        (BigInt::from(2), BigInt::from(3)),
    ];
    let (x, m) = chinese_remainder(&congruences).expect("test operation should succeed");

    assert_eq!(m, BigInt::from(6));
    assert_eq!(&x % BigInt::from(2), BigInt::from(1));
    assert_eq!(&x % BigInt::from(3), BigInt::from(2));
}

#[test]
fn test_solve_linear_diophantine() {
    // 3x + 5y = 1
    // One solution: x = 2, y = -1
    let a = BigInt::from(3);
    let b = BigInt::from(5);
    let c = BigInt::from(1);
    let (x, y) = solve_linear_diophantine(&a, &b, &c).expect("test operation should succeed");
    assert_eq!(&a * &x + &b * &y, c);

    // 6x + 9y = 3
    // Solution exists (gcd(6,9) = 3 divides 3)
    let a = BigInt::from(6);
    let b = BigInt::from(9);
    let c = BigInt::from(3);
    let (x, y) = solve_linear_diophantine(&a, &b, &c).expect("test operation should succeed");
    assert_eq!(&a * &x + &b * &y, c);

    // 2x + 4y = 3
    // No solution (gcd(2,4) = 2 does not divide 3)
    let a = BigInt::from(2);
    let b = BigInt::from(4);
    let c = BigInt::from(3);
    assert!(solve_linear_diophantine(&a, &b, &c).is_none());
}

#[test]
fn test_mod_inverse_fermat() {
    // For prime modulus, verify Fermat's little theorem: a^(p-1) equiv 1 (mod p)
    let p = BigInt::from(17); // prime
    let a = BigInt::from(5);

    let inv = mod_inverse(&a, &p).expect("test operation should succeed");
    assert_eq!((&a * &inv) % &p, BigInt::from(1));

    // Also verify using Fermat: a^(p-2) equiv a^(-1) (mod p)
    let fermat_inv = mod_pow(&a, &(&p - BigInt::from(2)), &p);
    assert_eq!(inv, fermat_inv);
}

#[test]
fn test_is_prime() {
    // Test small primes
    assert!(is_prime(&BigInt::from(2), 5));
    assert!(is_prime(&BigInt::from(3), 5));
    assert!(is_prime(&BigInt::from(5), 5));
    assert!(is_prime(&BigInt::from(7), 5));
    assert!(is_prime(&BigInt::from(11), 5));
    assert!(is_prime(&BigInt::from(17), 5));
    assert!(is_prime(&BigInt::from(97), 5));

    // Test composites
    assert!(!is_prime(&BigInt::from(1), 5));
    assert!(!is_prime(&BigInt::from(4), 5));
    assert!(!is_prime(&BigInt::from(6), 5));
    assert!(!is_prime(&BigInt::from(8), 5));
    assert!(!is_prime(&BigInt::from(9), 5));
    assert!(!is_prime(&BigInt::from(15), 5));
    assert!(!is_prime(&BigInt::from(100), 5));

    // Test larger prime
    assert!(is_prime(&BigInt::from(1009), 10));
}

#[test]
fn test_trial_division() {
    // 60 = 2^2 * 3 * 5
    let n = BigInt::from(60);
    let factors = trial_division(&n, 100);
    assert_eq!(factors.len(), 4);
    assert_eq!(factors[0], BigInt::from(2));
    assert_eq!(factors[1], BigInt::from(2));
    assert_eq!(factors[2], BigInt::from(3));
    assert_eq!(factors[3], BigInt::from(5));

    // 17 is prime
    let n = BigInt::from(17);
    let factors = trial_division(&n, 100);
    assert_eq!(factors.len(), 0); // No factors found (n itself is not added if n == original n)

    // 100 = 2^2 * 5^2
    let n = BigInt::from(100);
    let factors = trial_division(&n, 100);
    assert_eq!(factors.len(), 4);
}

#[test]
fn test_pollard_rho() {
    // 8051 = 83 * 97
    let n = BigInt::from(8051);
    if let Some(factor) = pollard_rho(&n) {
        assert!(n.clone() % &factor == BigInt::from(0));
        assert!(factor > BigInt::from(1) && factor < n);
    }

    // 15 = 3 * 5
    let n = BigInt::from(15);
    if let Some(factor) = pollard_rho(&n) {
        assert!(n.clone() % &factor == BigInt::from(0));
        assert!(factor > BigInt::from(1) && factor < n);
    }
}

#[test]
fn test_jacobi_symbol() {
    // (2/5) = -1
    assert_eq!(jacobi_symbol(&BigInt::from(2), &BigInt::from(5)), -1);
    // (3/5) = -1
    assert_eq!(jacobi_symbol(&BigInt::from(3), &BigInt::from(5)), -1);
    // (4/5) = 1
    assert_eq!(jacobi_symbol(&BigInt::from(4), &BigInt::from(5)), 1);
    // (1/5) = 1
    assert_eq!(jacobi_symbol(&BigInt::from(1), &BigInt::from(5)), 1);
    // (0/5) = 0
    assert_eq!(jacobi_symbol(&BigInt::from(0), &BigInt::from(5)), 0);

    // (2/15) = 1
    assert_eq!(jacobi_symbol(&BigInt::from(2), &BigInt::from(15)), 1);
}

#[test]
fn test_legendre_symbol() {
    // For p = 5: quadratic residues are {1, 4}
    assert_eq!(legendre_symbol(&BigInt::from(1), &BigInt::from(5)), 1);
    assert_eq!(legendre_symbol(&BigInt::from(4), &BigInt::from(5)), 1);
    assert_eq!(legendre_symbol(&BigInt::from(2), &BigInt::from(5)), -1);
    assert_eq!(legendre_symbol(&BigInt::from(3), &BigInt::from(5)), -1);

    // For p = 7: quadratic residues are {1, 2, 4}
    assert_eq!(legendre_symbol(&BigInt::from(1), &BigInt::from(7)), 1);
    assert_eq!(legendre_symbol(&BigInt::from(2), &BigInt::from(7)), 1);
    assert_eq!(legendre_symbol(&BigInt::from(4), &BigInt::from(7)), 1);
    assert_eq!(legendre_symbol(&BigInt::from(3), &BigInt::from(7)), -1);
}

#[test]
fn test_euler_totient() {
    assert_eq!(euler_totient(&BigInt::from(1)), BigInt::from(1));
    assert_eq!(euler_totient(&BigInt::from(2)), BigInt::from(1)); // phi(2) = 1
    assert_eq!(euler_totient(&BigInt::from(9)), BigInt::from(6)); // phi(9) = 6
    assert_eq!(euler_totient(&BigInt::from(10)), BigInt::from(4)); // phi(10) = 4
    assert_eq!(euler_totient(&BigInt::from(12)), BigInt::from(4)); // phi(12) = 4
    assert_eq!(euler_totient(&BigInt::from(15)), BigInt::from(8)); // phi(15) = 8
}

#[test]
fn test_is_perfect_power() {
    // 8 = 2^3
    assert_eq!(
        is_perfect_power(&BigInt::from(8)),
        Some((BigInt::from(2), 3))
    );
    // 27 = 3^3
    assert_eq!(
        is_perfect_power(&BigInt::from(27)),
        Some((BigInt::from(3), 3))
    );
    // 16 = 2^4 (or 4^2, but should find smallest exponent >= 2)
    let result = is_perfect_power(&BigInt::from(16));
    assert!(result.is_some());
    let (base, exp) = result.expect("test operation should succeed");
    assert_eq!(base.pow(exp), BigInt::from(16));

    // 10 is not a perfect power
    assert_eq!(is_perfect_power(&BigInt::from(10)), None);
    assert_eq!(is_perfect_power(&BigInt::from(15)), None);
}

#[test]
fn test_is_square_free() {
    // Square-free numbers
    assert!(is_square_free(&BigInt::from(1)));
    assert!(is_square_free(&BigInt::from(6))); // 6 = 2 * 3
    assert!(is_square_free(&BigInt::from(10))); // 10 = 2 * 5
    assert!(is_square_free(&BigInt::from(15))); // 15 = 3 * 5

    // Not square-free
    assert!(!is_square_free(&BigInt::from(4))); // 4 = 2^2
    assert!(!is_square_free(&BigInt::from(12))); // 12 = 4 * 3
    assert!(!is_square_free(&BigInt::from(18))); // 18 = 9 * 2
    assert!(!is_square_free(&BigInt::from(8))); // 8 = 2^3
}

#[test]
fn test_divisor_count() {
    assert_eq!(divisor_count(&BigInt::from(1)), BigInt::from(1));
    assert_eq!(divisor_count(&BigInt::from(12)), BigInt::from(6)); // 1, 2, 3, 4, 6, 12
    assert_eq!(divisor_count(&BigInt::from(28)), BigInt::from(6)); // 1, 2, 4, 7, 14, 28
    assert_eq!(divisor_count(&BigInt::from(6)), BigInt::from(4)); // 1, 2, 3, 6
}

#[test]
fn test_divisor_sum() {
    assert_eq!(divisor_sum(&BigInt::from(1)), BigInt::from(1));
    assert_eq!(divisor_sum(&BigInt::from(6)), BigInt::from(12)); // 1+2+3+6
    assert_eq!(divisor_sum(&BigInt::from(12)), BigInt::from(28)); // 1+2+3+4+6+12
    assert_eq!(divisor_sum(&BigInt::from(28)), BigInt::from(56)); // 1+2+4+7+14+28
}

#[test]
fn test_mobius() {
    assert_eq!(mobius(&BigInt::from(1)), 1);
    assert_eq!(mobius(&BigInt::from(2)), -1); // 2 is prime
    assert_eq!(mobius(&BigInt::from(6)), 1); // 6 = 2*3 (2 primes)
    assert_eq!(mobius(&BigInt::from(30)), -1); // 30 = 2*3*5 (3 primes)
    assert_eq!(mobius(&BigInt::from(12)), 0); // 12 = 2^2*3 (not square-free)
    assert_eq!(mobius(&BigInt::from(4)), 0); // 4 = 2^2 (not square-free)
}

#[test]
fn test_carmichael_lambda() {
    assert_eq!(carmichael_lambda(&BigInt::from(1)), BigInt::from(1));
    assert_eq!(carmichael_lambda(&BigInt::from(8)), BigInt::from(2)); // lambda(8) = 2
    assert_eq!(carmichael_lambda(&BigInt::from(15)), BigInt::from(4)); // lambda(15) = 4
    assert_eq!(carmichael_lambda(&BigInt::from(9)), BigInt::from(6)); // lambda(9) = phi(9) = 6
}

#[test]
fn test_gcd_binary() {
    assert_eq!(
        gcd_binary(BigInt::from(48), BigInt::from(18)),
        BigInt::from(6)
    );
    assert_eq!(
        gcd_binary(BigInt::from(100), BigInt::from(35)),
        BigInt::from(5)
    );
    assert_eq!(
        gcd_binary(BigInt::from(17), BigInt::from(19)),
        BigInt::from(1)
    );
    assert_eq!(
        gcd_binary(BigInt::from(0), BigInt::from(5)),
        BigInt::from(5)
    );
    assert_eq!(
        gcd_binary(BigInt::from(5), BigInt::from(0)),
        BigInt::from(5)
    );

    // Compare with standard GCD
    let a = BigInt::from(12345);
    let b = BigInt::from(67890);
    assert_eq!(gcd_binary(a.clone(), b.clone()), gcd_bigint(a, b));
}

#[test]
fn test_tonelli_shanks() {
    // 4 is a quadratic residue mod 7 (2^2 = 4)
    let result = tonelli_shanks(&BigInt::from(4), &BigInt::from(7));
    assert!(result.is_some());
    if let Some(x) = result {
        let p = BigInt::from(7);
        assert_eq!((&x * &x) % &p, BigInt::from(4) % &p);
    }

    // 2 is a quadratic residue mod 7 (3^2 = 9 = 2 mod 7)
    let result = tonelli_shanks(&BigInt::from(2), &BigInt::from(7));
    assert!(result.is_some());
    if let Some(x) = result {
        let p = BigInt::from(7);
        assert_eq!((&x * &x) % &p, BigInt::from(2));
    }

    // 3 is NOT a quadratic residue mod 7
    assert!(tonelli_shanks(&BigInt::from(3), &BigInt::from(7)).is_none());

    // Test with larger prime
    let result = tonelli_shanks(&BigInt::from(5), &BigInt::from(11));
    assert!(result.is_some());
    if let Some(x) = result {
        let p = BigInt::from(11);
        assert_eq!((&x * &x) % &p, BigInt::from(5));
    }
}

#[test]
fn test_factorial() {
    assert_eq!(factorial(0), BigInt::from(1));
    assert_eq!(factorial(1), BigInt::from(1));
    assert_eq!(factorial(5), BigInt::from(120));
    assert_eq!(factorial(10), BigInt::from(3628800));
    assert_eq!(factorial(3), BigInt::from(6));
}

#[test]
fn test_binomial() {
    assert_eq!(binomial(5, 2), BigInt::from(10));
    assert_eq!(binomial(10, 3), BigInt::from(120));
    assert_eq!(binomial(5, 0), BigInt::from(1));
    assert_eq!(binomial(5, 5), BigInt::from(1));
    assert_eq!(binomial(7, 3), BigInt::from(35));
    assert_eq!(binomial(10, 5), BigInt::from(252));

    // Edge cases
    assert_eq!(binomial(5, 6), BigInt::from(0)); // k > n
}
