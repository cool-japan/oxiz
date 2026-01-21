//! Helper functions for LIA solver

/// Compute GCD of two integers using Euclidean algorithm
pub(super) fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();

    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }

    a
}

/// Compute LCM of two integers
#[allow(dead_code)]
pub(super) fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a.abs() * b.abs()) / gcd(a, b)
}

/// Extended GCD algorithm: returns (gcd, x, y) such that ax + by = gcd
#[allow(dead_code)]
pub(super) fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a, 1, 0);
    }

    let (g, x1, y1) = extended_gcd(b, a % b);
    let x = y1;
    let y = x1 - (a / b) * y1;

    (g, x, y)
}
