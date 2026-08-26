// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super`], kept in a sibling file the way
//! `arithmetic/simplex` and `arithmetic/solver` do. Same module tree, so
//! `use super::*` still reaches the crate-private items under test.

use super::*;

/// Deterministic sampler: a fixed-increment LCG, seeded per test so a
/// failure is always reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn in_range(&mut self, lo: i64, hi: i64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let span = (hi - lo + 1) as u64;
        lo + ((self.0 >> 33) % span) as i64
    }
}

fn r(n: i64) -> Rational64 {
    Rational64::from_integer(n)
}

/// Evaluate `expr ⋈ 0` under an assignment.
fn holds(a: &LinAtom, assign: &[(VarId, Rational64)]) -> bool {
    let mut acc = a.expr.constant;
    for (v, c) in &a.expr.terms {
        let val = assign
            .iter()
            .find(|(w, _)| w == v)
            .map(|(_, x)| *x)
            .expect("every variable in the atom must be assigned");
        acc += *c * val;
    }
    let z = Rational64::zero();
    match a.kind {
        LinAtomKind::Le => acc <= z,
        LinAtomKind::Ge => acc >= z,
        LinAtomKind::Eq => acc == z,
        LinAtomKind::Lt => acc < z,
        LinAtomKind::Gt => acc > z,
    }
}

fn all_hold(l: &Lemma, assign: &[(VarId, Rational64)]) -> bool {
    l.atoms.iter().all(|a| holds(a, assign))
}

const V: VarId = 0;
const X: VarId = 1;
const Y: VarId = 2;

#[test]
fn sign_lemma_holds_on_every_sampled_point() {
    let mut rng = Rng::new(0x5eed_0001);
    for &(sx, sy) in &[
        (Sign::Pos, Sign::Pos),
        (Sign::Pos, Sign::Neg),
        (Sign::Neg, Sign::Pos),
        (Sign::Neg, Sign::Neg),
    ] {
        let l = sign(V, sx, sy, &[1, 2]).expect("representable");
        assert_eq!(l.scope, LemmaScope::BranchLocal);
        for _ in 0..200 {
            let mag_x = rng.in_range(1, 40);
            let mag_y = rng.in_range(1, 40);
            let x = if sx == Sign::Pos { mag_x } else { -mag_x };
            let y = if sy == Sign::Pos { mag_y } else { -mag_y };
            let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
            assert!(all_hold(&l, &assign), "sign lemma failed at {x} * {y}");
        }
    }
}

#[test]
fn zero_lemma_holds_when_a_factor_vanishes() {
    let mut rng = Rng::new(0x5eed_0002);
    let l = zero(V, &[3]).expect("representable");
    for _ in 0..200 {
        let y = rng.in_range(-50, 50);
        let x = 0i64;
        let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
        assert!(all_hold(&l, &assign));
    }
}

#[test]
fn neutral_lemma_holds_when_a_factor_is_one() {
    let mut rng = Rng::new(0x5eed_0003);
    let l = neutral(V, X, &[4]).expect("representable");
    for _ in 0..200 {
        let x = rng.in_range(-50, 50);
        let y = 1i64;
        let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
        assert!(all_hold(&l, &assign));
    }
}

#[test]
fn proportion_lemma_holds_in_all_four_quadrants() {
    let mut rng = Rng::new(0x5eed_0004);
    for &(sx, sy) in &[
        (Sign::Pos, Sign::Pos),
        (Sign::Pos, Sign::Neg),
        (Sign::Neg, Sign::Pos),
        (Sign::Neg, Sign::Neg),
    ] {
        let l = proportion(V, Y, sx, sy, &[5]).expect("representable");
        for _ in 0..200 {
            // |x| >= 1 is the premise, so sample magnitudes from 1 up.
            let mag_x = rng.in_range(1, 30);
            let x = if sx == Sign::Pos { mag_x } else { -mag_x };
            let mag_y = rng.in_range(1, 30);
            let y = if sy == Sign::Pos { mag_y } else { -mag_y };
            let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
            assert!(
                all_hold(&l, &assign),
                "proportion failed at x={x}, y={y} ({sx:?}, {sy:?})"
            );
        }
    }
}

#[test]
fn order_lemma_holds_under_its_premises() {
    // ac <= bc given a <= b and c > 0.
    const AC: VarId = 3;
    const BC: VarId = 4;
    let mut rng = Rng::new(0x5eed_0005);
    let l = order(AC, BC, &[6, 7]).expect("representable");
    for _ in 0..300 {
        let a = rng.in_range(-30, 30);
        let b = a + rng.in_range(0, 30);
        let c = rng.in_range(1, 30);
        let assign = [(AC, r(a * c)), (BC, r(b * c))];
        assert!(all_hold(&l, &assign), "order failed at a={a}, b={b}, c={c}");
    }
}

#[test]
fn monotonicity_lemma_holds_under_its_premises() {
    // ac <= bd given 0 <= a <= b and 0 <= c <= d.
    const AC: VarId = 3;
    const BD: VarId = 4;
    let mut rng = Rng::new(0x5eed_0006);
    let l = monotonicity(AC, BD, &[8]).expect("representable");
    for _ in 0..300 {
        let a = rng.in_range(0, 30);
        let b = a + rng.in_range(0, 30);
        let c = rng.in_range(0, 30);
        let d = c + rng.in_range(0, 30);
        let assign = [(AC, r(a * c)), (BD, r(b * d))];
        assert!(all_hold(&l, &assign));
    }
}

#[test]
fn tangent_lemma_holds_on_the_side_it_claims() {
    let mut rng = Rng::new(0x5eed_0007);
    for _ in 0..400 {
        let a = rng.in_range(-15, 15);
        let b = rng.in_range(-15, 15);
        let x = rng.in_range(-15, 15);
        let y = rng.in_range(-15, 15);
        let above = (x - a) * (y - b) >= 0;
        let l = tangent(V, X, Y, r(a), r(b), above, &[9]).expect("representable");
        let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
        assert!(
            all_hold(&l, &assign),
            "tangent at ({a},{b}) failed at ({x},{y}), above={above}"
        );
    }
}

#[test]
fn square_tangent_is_global_and_unconditional() {
    let mut rng = Rng::new(0x5eed_0008);
    for _ in 0..400 {
        let a = rng.in_range(-40, 40);
        let x = rng.in_range(-40, 40);
        let l = square_tangent(V, X, r(a)).expect("representable");
        assert_eq!(l.scope, LemmaScope::Global);
        assert!(l.premises.is_empty());
        let assign = [(V, r(x * x)), (X, r(x))];
        assert!(
            all_hold(&l, &assign),
            "x^2 tangent at a={a} failed at x={x}"
        );
    }
}

#[test]
fn mccormick_cuts_hold_everywhere_in_the_box() {
    let mut rng = Rng::new(0x5eed_0009);
    for _ in 0..200 {
        let xl = rng.in_range(-20, 20);
        let xu = xl + rng.in_range(0, 20);
        let yl = rng.in_range(-20, 20);
        let yu = yl + rng.in_range(0, 20);
        let bx = Box2 {
            xl: r(xl),
            xu: r(xu),
            yl: r(yl),
            yu: r(yu),
        };
        let l = mccormick(V, X, Y, &bx, &[10]).expect("representable");
        assert_eq!(l.atoms.len(), 4);
        for _ in 0..20 {
            let x = rng.in_range(xl, xu);
            let y = rng.in_range(yl, yu);
            let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
            assert!(
                all_hold(&l, &assign),
                "McCormick over [{xl},{xu}]x[{yl},{yu}] failed at ({x},{y})"
            );
        }
    }
}

#[test]
fn mccormick_cut_is_exactly_the_slack_product() {
    // Each cut's expression must be algebraically identical to the slack
    // product it was derived from, at every sampled point.
    let mut rng = Rng::new(0x5eed_000a);
    let (xl, xu, yl, yu) = (-3i64, 5i64, -2i64, 7i64);
    let bx = Box2 {
        xl: r(xl),
        xu: r(xu),
        yl: r(yl),
        yu: r(yu),
    };
    let l = mccormick(V, X, Y, &bx, &[]).expect("representable");
    let corners = [(xl, yl), (xu, yu), (xu, yl), (xl, yu)];
    for _ in 0..300 {
        let x = rng.in_range(-30, 30);
        let y = rng.in_range(-30, 30);
        let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
        for (i, at) in l.atoms.iter().enumerate() {
            let mut lhs = at.expr.constant;
            for (v, c) in &at.expr.terms {
                let val = assign
                    .iter()
                    .find(|(w, _)| w == v)
                    .map(|(_, t)| *t)
                    .expect("assigned");
                lhs += *c * val;
            }
            let (cx, cy) = corners[i];
            let slack = (x - cx) * (y - cy);
            assert_eq!(
                lhs,
                r(slack),
                "cut {i} must equal its slack product at ({x},{y})"
            );
        }
    }
}

#[test]
fn mccormick_is_tight_at_the_corners() {
    // At a corner of the box the bilinear surface and its envelope must
    // touch: two of the four cuts are satisfied with equality.
    let (xl, xu, yl, yu) = (-3i64, 5i64, -2i64, 7i64);
    let bx = Box2 {
        xl: r(xl),
        xu: r(xu),
        yl: r(yl),
        yu: r(yu),
    };
    let l = mccormick(V, X, Y, &bx, &[]).expect("representable");
    for (x, y) in [(xl, yl), (xl, yu), (xu, yl), (xu, yu)] {
        let assign = [(V, r(x * y)), (X, r(x)), (Y, r(y))];
        assert!(all_hold(&l, &assign));
        let tight = l
            .atoms
            .iter()
            .filter(|a| {
                let mut acc = a.expr.constant;
                for (v, c) in &a.expr.terms {
                    let val = assign
                        .iter()
                        .find(|(w, _)| w == v)
                        .map(|(_, t)| *t)
                        .expect("assigned");
                    acc += *c * val;
                }
                acc == Rational64::zero()
            })
            .count();
        assert!(tight >= 2, "expected two tight cuts at ({x},{y})");
    }
}

#[test]
fn repeated_variable_coefficients_are_merged() {
    // v = x*x: McCormick over a square box passes the same variable for
    // both factors, so the two coefficients must fold into one term.
    let bx = Box2 {
        xl: r(0),
        xu: r(4),
        yl: r(0),
        yu: r(4),
    };
    let l = mccormick(V, X, X, &bx, &[]).expect("representable");
    for at in &l.atoms {
        assert!(
            at.expr.terms.len() <= 2,
            "x and y coefficients must merge into a single term"
        );
    }
    let mut rng = Rng::new(0x5eed_000b);
    for _ in 0..200 {
        let x = rng.in_range(0, 4);
        let assign = [(V, r(x * x)), (X, r(x))];
        assert!(all_hold(&l, &assign));
    }
}

#[test]
fn unrepresentable_bounds_are_declined_not_wrapped() {
    let huge = Rational64::from_integer(i64::MAX / 2);
    assert!(
        mccormick(
            V,
            X,
            Y,
            &Box2 {
                xl: huge,
                xu: huge,
                yl: huge,
                yu: huge,
            },
            &[]
        )
        .is_none(),
        "xl*yl overflows and must be declined"
    );
    assert!(square_tangent(V, X, huge).is_none());
    assert!(tangent(V, X, Y, huge, huge, true, &[]).is_none());
}
