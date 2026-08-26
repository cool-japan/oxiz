//! Characterisation and equivalence tests for
//! [`Solver::evaluate_int_expr_with_array_axiom`].
//!
//! The evaluator decides `check_int_ordering_conflict`, which answers `Unsat`
//! outright when the two sides of an `<`/`<=`/`>`/`>=` assertion both fold and
//! the ordering does not hold.  A value that is off by one, or an operand that
//! is silently dropped from a sum, is therefore a spurious `unsat`.  Every arm
//! is pinned here by value, and the whole corpus is pinned by digest.

use super::Solver;
use crate::prelude::*;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use smallvec::{SmallVec, smallvec};

/// The array-variable alias map, as `collect_array_var_aliases` builds it.
type Aliases = FxHashMap<TermId, TermId>;

/// A scratch term manager plus the alias map the evaluator reads.
struct Env {
    manager: TermManager,
    aliases: Aliases,
    solver: Solver,
}

impl Env {
    fn new() -> Self {
        Self {
            manager: TermManager::new(),
            aliases: Aliases::default(),
            solver: Solver::new(),
        }
    }

    fn int(&mut self, value: i64) -> TermId {
        self.manager.mk_int(value)
    }

    fn big_int(&mut self, value: BigInt) -> TermId {
        self.manager.mk_int(value)
    }

    fn int_var(&mut self, name: &str) -> TermId {
        let sort = self.manager.sorts.int_sort;
        self.manager.mk_var(name, sort)
    }

    fn array_var(&mut self, name: &str) -> TermId {
        let int_sort = self.manager.sorts.int_sort;
        let sort = self.manager.sorts.array(int_sort, int_sort);
        self.manager.mk_var(name, sort)
    }

    /// Intern `kind` at integer sort *without* the builder's folding, so the
    /// evaluator is the only thing that computes a value.
    fn node(&mut self, kind: TermKind) -> TermId {
        let sort = self.manager.sorts.int_sort;
        self.manager.intern_term(kind, sort)
    }

    fn add(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();
        self.node(TermKind::Add(args))
    }

    fn mul(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();
        self.node(TermKind::Mul(args))
    }

    fn sub(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.node(TermKind::Sub(lhs, rhs))
    }

    /// A raw `select` node — `mk_select` would apply read-over-write itself.
    fn select(&mut self, array: TermId, index: TermId) -> TermId {
        self.node(TermKind::Select(array, index))
    }

    /// A raw `store` node at array sort.
    fn store(&mut self, array: TermId, index: TermId, value: TermId) -> TermId {
        let int_sort = self.manager.sorts.int_sort;
        let sort = self.manager.sorts.array(int_sort, int_sort);
        self.manager
            .intern_term(TermKind::Store(array, index, value), sort)
    }

    fn eval(&self, term: TermId) -> Option<BigInt> {
        self.solver
            .evaluate_int_expr_with_array_axiom(term, &self.aliases, &self.manager)
    }

    fn rendered(&self, term: TermId) -> String {
        match self.eval(term) {
            None => "-".to_string(),
            Some(value) => value.to_string(),
        }
    }
}

fn val(value: i64) -> Option<BigInt> {
    Some(BigInt::from(value))
}

// ── Leaves and the unhandled kinds ────────────────────────────────────────

/// An integer literal evaluates to itself, at arbitrary magnitude — the
/// evaluator is exact `BigInt` arithmetic, not `i64`.
#[test]
fn literals_evaluate_exactly() {
    let mut env = Env::new();
    let small = env.int(-7);
    assert_eq!(env.eval(small), val(-7));

    let huge = BigInt::from(1u8) << 200u32;
    let big = env.big_int(huge.clone());
    assert_eq!(env.eval(big), Some(huge));
}

/// Kinds with no arm are reported as *not evaluable*, never as a defaulted
/// zero: `check_int_ordering_conflict` skips the assertion entirely.
#[test]
fn unhandled_kinds_are_not_evaluable() {
    let mut env = Env::new();
    let a = env.int(6);
    let b = env.int(3);
    // `Le` is a Boolean relation, not an integer expression; it folds only in an
    // `ite` condition.  `Ite` with an *integer* first operand is not an `ite`
    // with a condition at all.
    let unhandled = [TermKind::Le(a, b), TermKind::Ite(a, a, b)];
    for kind in unhandled {
        let term = env.node(kind);
        assert_eq!(env.eval(term), None, "expected no value for {term:?}");
    }
    let variable = env.int_var("x");
    assert_eq!(env.eval(variable), None);
    let real = env.manager.mk_real(num_rational::Rational64::new(1, 2));
    assert_eq!(env.eval(real), None);
}

/// Every integer-expression kind this evaluator implements folds.
///
/// A regression guard for the four added after the iterative conversion: `Neg`,
/// `Div`, `Mod` and `Ite` each answered *not evaluable* before.
#[test]
fn every_supported_integer_kind_folds() {
    let mut env = Env::new();
    let a = env.int(6);
    let b = env.int(3);
    let yes = env.manager.mk_true();
    let kinds = [
        TermKind::Add(smallvec![a, b]),
        TermKind::Mul(smallvec![a, b]),
        TermKind::Sub(a, b),
        TermKind::Neg(a),
        TermKind::Div(a, b),
        TermKind::Mod(a, b),
        TermKind::Ite(yes, a, b),
    ];
    for kind in kinds {
        let term = env.node(kind);
        assert!(env.eval(term).is_some(), "expected a value for {term:?}");
    }
}

// ── Arithmetic ────────────────────────────────────────────────────────────

/// `+` sums every operand.  A dropped operand is the failure mode this pins.
#[test]
fn add_sums_every_operand() {
    let mut env = Env::new();
    let args: Vec<TermId> = (1..=5).map(|value| env.int(value)).collect();
    let sum = env.add(args);
    assert_eq!(env.eval(sum), val(15));
}

/// The identity of an empty `+` is zero and of an empty `*` is one, matching
/// the accumulator each arm starts from.
#[test]
fn empty_folds_are_the_identities() {
    let mut env = Env::new();
    let sum = env.add([]);
    let product = env.mul([]);
    assert_eq!(env.eval(sum), val(0));
    assert_eq!(env.eval(product), val(1));
}

/// A one-operand fold is that operand.
#[test]
fn single_operand_folds() {
    let mut env = Env::new();
    let seven = env.int(7);
    let sum = env.add([seven]);
    let product = env.mul([seven]);
    assert_eq!(env.eval(sum), val(7));
    assert_eq!(env.eval(product), val(7));
}

/// `*` multiplies every operand, exactly — no `i64` wrap-around.
#[test]
fn mul_is_exact() {
    let mut env = Env::new();
    let big = env.big_int(BigInt::from(1u8) << 70u32);
    let product = env.mul([big, big, big]);
    assert_eq!(env.eval(product), Some(BigInt::from(1u8) << 210u32));
}

/// `-` is not commutative and is unbounded below: results go negative and stay
/// exact.
#[test]
fn sub_is_ordered_and_signed() {
    let mut env = Env::new();
    let two = env.int(2);
    let five = env.int(5);
    let forward = env.sub(two, five);
    let backward = env.sub(five, two);
    assert_eq!(env.eval(forward), val(-3));
    assert_eq!(env.eval(backward), val(3));
}

/// Mixed nesting folds bottom-up: `(1 + 2) * (10 - 4) = 18`.
#[test]
fn nested_arithmetic_folds_bottom_up() {
    let mut env = Env::new();
    let one = env.int(1);
    let two = env.int(2);
    let ten = env.int(10);
    let four = env.int(4);
    let sum = env.add([one, two]);
    let diff = env.sub(ten, four);
    let product = env.mul([sum, diff]);
    assert_eq!(env.eval(product), val(18));
}

/// One unevaluable operand anywhere makes the whole term unevaluable — the sum
/// does not quietly proceed with the operands it did manage to fold.
#[test]
fn one_unevaluable_operand_defeats_the_whole_term() {
    let mut env = Env::new();
    let x = env.int_var("x");
    let one = env.int(1);
    let two = env.int(2);
    let sum = env.add([one, x, two]);
    assert_eq!(env.eval(sum), None);

    // Also in the trailing position, and under `-` on either side.
    let trailing = env.add([one, two, x]);
    assert_eq!(env.eval(trailing), None);
    let left = env.sub(x, one);
    let right = env.sub(one, x);
    assert_eq!(env.eval(left), None);
    assert_eq!(env.eval(right), None);
}

/// A shared subterm folds identically under every parent; the evaluator keeps
/// no memo table and that is not observable in the answer.
#[test]
fn shared_subterms_fold_identically() {
    let mut env = Env::new();
    let three = env.int(3);
    let four = env.int(4);
    let shared = env.add([three, four]);
    let doubled = env.add([shared, shared]);
    let squared = env.mul([shared, shared]);
    assert_eq!(env.eval(shared), val(7));
    assert_eq!(env.eval(doubled), val(14));
    assert_eq!(env.eval(squared), val(49));
}

// ── The read-over-write arm ───────────────────────────────────────────────

/// `select(store(a, i, v), i)` folds to `v`.
#[test]
fn read_over_write_at_the_same_index() {
    let mut env = Env::new();
    let array = env.array_var("a");
    let index = env.int(3);
    let stored = env.int(42);
    let store = env.store(array, index, stored);
    let select = env.select(store, index);
    assert_eq!(env.eval(select), val(42));
}

/// A read at a *different* index is not decided by the axiom, so the term does
/// not fold — the evaluator must not fall through to the stored value.
#[test]
fn read_at_a_different_index_does_not_fold() {
    let mut env = Env::new();
    let array = env.array_var("a");
    let index = env.int(3);
    let other = env.int(4);
    let stored = env.int(42);
    let store = env.store(array, index, stored);
    let select = env.select(store, other);
    assert_eq!(env.eval(select), None);
}

/// A read of a plain array variable does not fold.
#[test]
fn read_of_an_unaliased_variable_does_not_fold() {
    let mut env = Env::new();
    let array = env.array_var("a");
    let index = env.int(3);
    let select = env.select(array, index);
    assert_eq!(env.eval(select), None);
}

/// The axiom is applied to a fixpoint: a stored value that is itself a
/// read-over-write folds through, however many levels deep.
#[test]
fn read_over_write_folds_through_nested_stores() {
    let mut env = Env::new();
    let array = env.array_var("a");
    let index = env.int(3);
    let mut value = env.int(9);
    for _ in 0..4 {
        let store = env.store(array, index, value);
        value = env.select(store, index);
    }
    assert_eq!(env.eval(value), val(9));

    // And under arithmetic.
    let one = env.int(1);
    let sum = env.add([value, one]);
    assert_eq!(env.eval(sum), val(10));
}

/// The value the axiom lands on is dispatched normally, so a stored *sum*
/// folds too.
#[test]
fn the_stored_value_is_evaluated_in_turn() {
    let mut env = Env::new();
    let array = env.array_var("a");
    let index = env.int(3);
    let two = env.int(2);
    let five = env.int(5);
    let stored = env.add([two, five]);
    let store = env.store(array, index, stored);
    let select = env.select(store, index);
    assert_eq!(env.eval(select), val(7));
}

/// A read through an array variable aliased to a store folds via the alias
/// map: `B = store(A, i, 42)` makes `select(B, i)` fold to `42`.
#[test]
fn read_through_an_alias_folds() {
    let mut env = Env::new();
    let base = env.array_var("a");
    let aliased = env.array_var("b");
    let index = env.int(3);
    let stored = env.int(42);
    let store = env.store(base, index, stored);
    env.aliases.insert(aliased, store);
    let select = env.select(aliased, index);
    assert_eq!(env.eval(select), val(42));
}

/// With a non-empty alias map, a select the alias arm cannot resolve still
/// falls back to the plain axiom.
#[test]
fn a_non_empty_alias_map_still_falls_back() {
    let mut env = Env::new();
    let base = env.array_var("a");
    let other = env.array_var("b");
    let index = env.int(3);
    let stored = env.int(42);
    let store = env.store(base, index, stored);
    // `other` is aliased to something irrelevant to the select below.
    let elsewhere = env.int(1);
    let unrelated = env.store(base, elsewhere, stored);
    env.aliases.insert(other, unrelated);

    let direct = env.select(store, index);
    assert_eq!(env.eval(direct), val(42));
}

/// An alias whose index does not match the read is not applied.
#[test]
fn an_alias_at_a_different_index_does_not_fold() {
    let mut env = Env::new();
    let base = env.array_var("a");
    let aliased = env.array_var("b");
    let index = env.int(3);
    let other = env.int(4);
    let stored = env.int(42);
    let store = env.store(base, index, stored);
    env.aliases.insert(aliased, store);
    let select = env.select(aliased, other);
    assert_eq!(env.eval(select), None);
}

/// A chain of aliases, each storing a read of the previous one, folds all the
/// way down to the literal at the bottom.
fn alias_chain(env: &mut Env, length: usize) -> TermId {
    let base = env.array_var("a");
    let index = env.int(0);
    let bottom = env.int(11);
    let mut value = bottom;
    for level in 0..length {
        let aliased = env.array_var(&format!("b{level}"));
        let store = env.store(base, index, value);
        env.aliases.insert(aliased, store);
        value = env.select(aliased, index);
    }
    value
}

#[test]
fn an_alias_chain_folds_to_the_bottom() {
    let mut env = Env::new();
    let top = alias_chain(&mut env, 5);
    assert_eq!(env.eval(top), val(11));
}

// ── Negation, Euclidean division and `ite` ────────────────────────────────

/// Arithmetic negation.
#[test]
fn negation_folds() {
    let mut env = Env::new();
    let seven = env.int(7);
    let negated = env.node(TermKind::Neg(seven));
    assert_eq!(env.eval(negated), val(-7));

    let negative = env.int(-7);
    let double = env.node(TermKind::Neg(negative));
    assert_eq!(env.eval(double), val(7));
}

/// `div` is **Euclidean**, not truncating: the remainder is always
/// non-negative, so `(div -7 2)` is `-4` and not Rust's `-3`.
///
/// This is the convention `oxiz_core`'s `rewrite::arith` and model evaluator
/// already use (`i64::div_euclid`), and getting it wrong by one would refute
/// satisfiable formulas.
#[test]
fn div_is_euclidean() {
    let mut env = Env::new();
    let cases = [
        (7i64, 2i64, 3i64),
        (-7, 2, -4),
        (7, -2, -3),
        (-7, -2, 4),
        (6, 3, 2),
        (-6, 3, -2),
        (0, 5, 0),
    ];
    for (lhs, rhs, expected) in cases {
        let lhs_term = env.int(lhs);
        let rhs_term = env.int(rhs);
        let term = env.node(TermKind::Div(lhs_term, rhs_term));
        assert_eq!(env.eval(term), val(expected), "div {lhs} {rhs}");
    }
}

/// `mod` is the Euclidean remainder, so it is never negative: `(mod -7 2)` is
/// `1`, and the sign of the *divisor* does not change that.
#[test]
fn modulo_is_euclidean_and_never_negative() {
    let mut env = Env::new();
    let cases = [
        (7i64, 2i64, 1i64),
        (-7, 2, 1),
        (7, -2, 1),
        (-7, -2, 1),
        (6, 3, 0),
        (-6, 3, 0),
    ];
    for (lhs, rhs, expected) in cases {
        let lhs_term = env.int(lhs);
        let rhs_term = env.int(rhs);
        let term = env.node(TermKind::Mod(lhs_term, rhs_term));
        assert_eq!(env.eval(term), val(expected), "mod {lhs} {rhs}");
        // The identity `a = b*q + r` with `0 <= r < |b|` must hold.
        if let (Some(q), Some(r)) = (
            {
                let d = env.node(TermKind::Div(lhs_term, rhs_term));
                env.eval(d)
            },
            env.eval(term),
        ) {
            assert_eq!(BigInt::from(rhs) * q + &r, BigInt::from(lhs));
            assert!(r >= BigInt::ZERO && r < BigInt::from(rhs.abs()));
        }
    }
}

/// `div` and `mod` by zero are **uninterpreted** in SMT-LIB, not total, so they
/// must not fold at all.
///
/// This is the opposite of the bit-vector division family, where SMT-LIB does
/// specify a result at a zero divisor.  Folding to any particular value here
/// would claim a fact the theory does not state, and because the caller reports
/// `Unsat` on a failed ordering, that fabricated value could refute a
/// satisfiable formula.
#[test]
fn division_by_zero_does_not_fold() {
    let mut env = Env::new();
    let zero = env.int(0);
    for numerator in [7i64, 0, -7] {
        let lhs = env.int(numerator);
        let quotient = env.node(TermKind::Div(lhs, zero));
        let remainder = env.node(TermKind::Mod(lhs, zero));
        assert_eq!(env.eval(quotient), None, "div {numerator} 0");
        assert_eq!(env.eval(remainder), None, "mod {numerator} 0");
    }
}

/// A Real-sorted `div` is exact rational division, not Euclidean, so this
/// evaluator does not fold it even when both operands are integer literals.
#[test]
fn real_sorted_division_does_not_fold() {
    let mut env = Env::new();
    let seven = env.int(7);
    let two = env.int(2);
    let real_sort = env.manager.sorts.real_sort;
    let real_div = env
        .manager
        .intern_term(TermKind::Div(seven, two), real_sort);
    assert_eq!(env.eval(real_div), None);

    let real_mod = env
        .manager
        .intern_term(TermKind::Mod(seven, two), real_sort);
    assert_eq!(env.eval(real_mod), None);
}

/// `ite` folds only the **taken** branch: the untaken one here cannot be folded
/// at all, so visiting it would make the whole term unevaluable.
#[test]
fn ite_evaluates_only_the_taken_branch() {
    let mut env = Env::new();
    let taken = env.int(11);
    let unevaluable = env.int_var("never");
    let yes = env.manager.mk_true();
    let no = env.manager.mk_false();

    let then_taken = env.node(TermKind::Ite(yes, taken, unevaluable));
    assert_eq!(env.eval(then_taken), val(11));

    let else_taken = env.node(TermKind::Ite(no, unevaluable, taken));
    assert_eq!(env.eval(else_taken), val(11));
}

/// Each of the five integer relations decides a branch.
#[test]
fn every_integer_relation_decides_a_branch() {
    let mut env = Env::new();
    let three = env.int(3);
    let five = env.int(5);
    let then_value = env.int(1);
    let else_value = env.int(0);
    let cases = [
        (TermKind::Eq(three, five), 0i64),
        (TermKind::Lt(three, five), 1),
        (TermKind::Le(three, five), 1),
        (TermKind::Gt(three, five), 0),
        (TermKind::Ge(three, five), 0),
        (TermKind::Eq(three, three), 1),
        (TermKind::Le(three, three), 1),
        (TermKind::Lt(three, three), 0),
    ];
    for (condition, expected) in cases {
        let condition = env.node(condition);
        let term = env.node(TermKind::Ite(condition, then_value, else_value));
        assert_eq!(env.eval(term), val(expected), "condition {condition:?}");
    }
}

/// A condition can be built from `not`, `and` and `or`, and the connectives
/// short-circuit on the first decisive operand.
#[test]
fn connective_conditions_short_circuit() {
    let mut env = Env::new();
    let then_value = env.int(1);
    let else_value = env.int(0);
    let yes = env.manager.mk_true();
    let no = env.manager.mk_false();
    let bool_sort = env.manager.sorts.bool_sort;
    let undecidable = env.manager.mk_var("p", bool_sort);

    let negated = env.node(TermKind::Not(yes));
    let negated_ite = env.node(TermKind::Ite(negated, then_value, else_value));
    assert_eq!(env.eval(negated_ite), val(0));

    // `(and false p)` is false without ever opening `p`.
    let conjunction = env.node(TermKind::And(smallvec![no, undecidable]));
    let short_false = env.node(TermKind::Ite(conjunction, then_value, else_value));
    assert_eq!(env.eval(short_false), val(0));

    // `(or true p)` is true without ever opening `p`.
    let disjunction = env.node(TermKind::Or(smallvec![yes, undecidable]));
    let short_true = env.node(TermKind::Ite(disjunction, then_value, else_value));
    assert_eq!(env.eval(short_true), val(1));

    // Nothing to short-circuit on: the whole term does not fold.
    let blocked = env.node(TermKind::And(smallvec![undecidable, no]));
    let blocked_ite = env.node(TermKind::Ite(blocked, then_value, else_value));
    assert_eq!(env.eval(blocked_ite), None);

    // Empty connectives are their identities.
    let empty_and = env.node(TermKind::And(smallvec![]));
    let empty_and_ite = env.node(TermKind::Ite(empty_and, then_value, else_value));
    assert_eq!(env.eval(empty_and_ite), val(1));
    let empty_or = env.node(TermKind::Or(smallvec![]));
    let empty_or_ite = env.node(TermKind::Ite(empty_or, then_value, else_value));
    assert_eq!(env.eval(empty_or_ite), val(0));
}

/// An undecidable condition leaves the whole term unfolded.
#[test]
fn undecidable_ite_does_not_fold() {
    let mut env = Env::new();
    let then_value = env.int(1);
    let else_value = env.int(0);
    let bool_sort = env.manager.sorts.bool_sort;
    let undecidable = env.manager.mk_var("p", bool_sort);
    let term = env.node(TermKind::Ite(undecidable, then_value, else_value));
    assert_eq!(env.eval(term), None);

    // A relation over an unbound integer variable is equally undecidable.
    let x = env.int_var("x");
    let relation = env.node(TermKind::Lt(x, then_value));
    let relation_ite = env.node(TermKind::Ite(relation, then_value, else_value));
    assert_eq!(env.eval(relation_ite), None);
}

/// A condition can read through the array axiom, since its operands are opened
/// in number position like any other.
#[test]
fn ite_condition_can_read_through_the_array_axiom() {
    let mut env = Env::new();
    let array = env.array_var("a");
    let index = env.int(0);
    let stored = env.int(42);
    let store = env.store(array, index, stored);
    let read = env.select(store, index);
    let five = env.int(5);
    let then_value = env.int(1);
    let else_value = env.int(0);
    let condition = env.node(TermKind::Gt(read, five));
    let term = env.node(TermKind::Ite(condition, then_value, else_value));
    assert_eq!(env.eval(term), val(1));
}

/// The new kinds compose: `(mod (- (div 17 5)) 3)` is `(mod -3 3)` = `0`.
#[test]
fn new_kinds_compose() {
    let mut env = Env::new();
    let seventeen = env.int(17);
    let five = env.int(5);
    let three = env.int(3);
    let quotient = env.node(TermKind::Div(seventeen, five));
    let negated = env.node(TermKind::Neg(quotient));
    let remainder = env.node(TermKind::Mod(negated, three));
    assert_eq!(env.eval(quotient), val(3));
    assert_eq!(env.eval(negated), val(-3));
    assert_eq!(env.eval(remainder), val(0));
}

// ── Corpus equivalence ────────────────────────────────────────────────────

/// A deterministic 64-bit xorshift, so the corpus below is reproducible
/// without a random-number dependency.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// The largest expanded (tree, not DAG) node count a corpus term may have.
const CORPUS_TREE_CAP: u64 = 24;

/// One growing pool of terms, with each term's expanded tree size alongside.
struct Pool {
    terms: Vec<TermId>,
    sizes: Vec<u64>,
}

impl Pool {
    fn new() -> Self {
        Self {
            terms: Vec::new(),
            sizes: Vec::new(),
        }
    }

    fn push(&mut self, term: TermId, size: u64) {
        self.terms.push(term);
        self.sizes.push(size);
    }
}

/// Draw an `Add`, `Mul` or `Sub` node over operands taken from `pool`,
/// returning it with its expanded tree size.  `None` when the draw would
/// exceed the tree cap.
fn draw_operator(
    env: &mut Env,
    rng: &mut Xorshift,
    pool: &Pool,
    op_count: usize,
) -> Option<(TermId, u64)> {
    let arity = rng.below(4);
    let mut operands: SmallVec<[TermId; 4]> = SmallVec::new();
    let mut size = 1u64;
    for _ in 0..arity {
        let pick = rng.below(pool.terms.len());
        size += pool.sizes[pick];
        operands.push(pool.terms[pick]);
    }
    if size > CORPUS_TREE_CAP {
        return None;
    }
    // Draw two operands for a strictly binary shape, replacing the n-ary size.
    let binary = |rng: &mut Xorshift, size: &mut u64| {
        let left = rng.below(pool.terms.len());
        let right = rng.below(pool.terms.len());
        *size = pool.sizes[left] + pool.sizes[right] + 1;
        (pool.terms[left], pool.terms[right])
    };
    let kind = match rng.below(op_count) {
        0 => TermKind::Add(operands),
        1 => TermKind::Mul(operands),
        2 => {
            let (left, right) = binary(rng, &mut size);
            TermKind::Sub(left, right)
        }
        // Slots below here exist only in the extended corpus.
        3 => {
            let (left, right) = binary(rng, &mut size);
            TermKind::Div(left, right)
        }
        4 => {
            let (left, right) = binary(rng, &mut size);
            TermKind::Mod(left, right)
        }
        5 => {
            let arg = rng.below(pool.terms.len());
            size = pool.sizes[arg] + 1;
            TermKind::Neg(pool.terms[arg])
        }
        // An `ite` over a relation between the two operands, so the condition is
        // decidable exactly when both operands are.
        _ => {
            let (left, right) = binary(rng, &mut size);
            let condition = match rng.below(5) {
                0 => TermKind::Eq(left, right),
                1 => TermKind::Lt(left, right),
                2 => TermKind::Le(left, right),
                3 => TermKind::Gt(left, right),
                _ => TermKind::Ge(left, right),
            };
            let condition = env.node(condition);
            size += 1;
            TermKind::Ite(condition, left, right)
        }
    };
    if size > CORPUS_TREE_CAP {
        return None;
    }
    Some((env.node(kind), size))
}

/// The shapes the *legacy* int corpus draws from: the three the pre-extension
/// evaluator supported.  Frozen so that corpus stays byte-identical to the one
/// the pre-extension implementation was compared against.
const LEGACY_CORPUS_OPS: usize = 3;

/// Every shape [`draw_operator`] can draw, adding `div`, `mod`, `neg` and `ite`.
const EXTENDED_CORPUS_OPS: usize = 7;

/// Build a corpus of integer terms in two layers, the same way the bit-vector
/// corpus does.
///
/// The first layer draws only from leaves that fold — literals of both signs,
/// a read-over-write, and a read resolved through the alias map — so the sums,
/// products and differences are exercised on terms that actually produce a
/// value.  The second layer adds an unbound variable, a non-folding read and
/// two kinds with no arm, so the "not evaluable" propagation is exercised from
/// every operand position.
///
/// Terms are interned raw throughout, so the builder's folding never
/// pre-computes an answer the evaluator was supposed to produce.
fn build_corpus(clean_count: usize, mixed: usize, op_count: usize) -> (Env, Vec<TermId>) {
    let mut env = Env::new();
    let mut rng = Xorshift(0x2545_F491_4F6C_DD1D);
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    let mut corpus: Vec<TermId> = Vec::new();
    let mut pool = Pool::new();

    for value in [-7i64, -1, 0, 1, 2, 3, 11] {
        let term = env.int(value);
        pool.push(term, 1);
    }

    // A read that folds through the plain axiom, and one that folds only
    // through the alias map.
    let base = env.array_var("a");
    let index = env.int(0);
    let other_index = env.int(1);
    let stored = env.int(5);
    let store = env.store(base, index, stored);
    let folding = env.select(store, index);
    let aliased = env.array_var("b");
    env.aliases.insert(aliased, store);
    let through_alias = env.select(aliased, index);
    for term in [folding, through_alias] {
        pool.push(term, 2);
    }

    seen.extend(pool.terms.iter().copied());
    corpus.extend(pool.terms.iter().copied());

    let mut produced = 0usize;
    let mut attempts = 0usize;
    while produced < clean_count && attempts < clean_count * 64 {
        attempts += 1;
        let Some((term, size)) = draw_operator(&mut env, &mut rng, &pool, op_count) else {
            continue;
        };
        if !seen.insert(term) {
            // Hash consing gave back an existing node; keep drawing.
            continue;
        }
        pool.push(term, size);
        corpus.push(term);
        produced += 1;
    }
    assert_eq!(produced, clean_count, "clean pool");

    // Poison leaves: an unbound variable, a read the axiom does not decide,
    // and two kinds with no arm.
    let variable = env.int_var("x");
    let not_folding = env.select(store, other_index);
    let one = pool.terms[3];
    let divided = env.node(TermKind::Div(one, one));
    let negated = env.node(TermKind::Neg(one));
    for term in [variable, not_folding, divided, negated] {
        pool.push(term, 1);
        corpus.push(term);
    }

    produced = 0;
    attempts = 0;
    while produced < mixed && attempts < mixed * 64 {
        attempts += 1;
        let Some((term, size)) = draw_operator(&mut env, &mut rng, &pool, op_count) else {
            continue;
        };
        if !seen.insert(term) {
            continue;
        }
        pool.push(term, size);
        corpus.push(term);
        produced += 1;
    }
    assert_eq!(produced, mixed, "mixed pool");

    (env, corpus)
}

/// FNV-1a over the rendered evaluation of every corpus term, in corpus order.
fn corpus_digest(env: &Env, terms: &[TermId]) -> (u64, usize) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut evaluable = 0usize;
    for (index, &term) in terms.iter().enumerate() {
        let rendered = env.rendered(term);
        if rendered != "-" {
            evaluable += 1;
        }
        for byte in format!("{index}:{rendered};").bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    (hash, evaluable)
}

/// Corpus terms that fold: the recursive implementation managed 3003, and the
/// four listed on [`legacy_corpus_matches_the_pinned_digest`] were added when
/// `Neg`, `Div`, `Mod` and `Ite` gained arms.
const LEGACY_PINNED_EVALUABLE: usize = 3003 + 4;
/// FNV-1a digest over the legacy corpus.
const LEGACY_PINNED_DIGEST: u64 = 8_667_480_659_813_980_293;

/// The legacy corpus digest, pinned.
///
/// This corpus draws only from the three kinds the pre-extension evaluator
/// supported, and is generated draw for draw identically to the one the
/// **recursive** implementation was measured against — whose digest was
/// `7013243568558455799` over 3003 evaluable terms.  Every one of its 3013
/// answers was then compared term by term against a faithful restatement of the
/// pre-extension semantics:
///
/// * **3009 identical** — same value, or `None` in both.
/// * **4 answered where it did not**, all reached through the two poison leaves
///   the corpus plants: `(div 1 1) = 1` and `(- 1) = -1`, plus
///   `(- (- 1) 2) = -3` and `(+ x (div 1 1)) = -6`.
/// * **0 regressions** — no term it could answer is answered differently now.
#[test]
fn legacy_corpus_matches_the_pinned_digest() {
    let (env, terms) = build_corpus(2500, 500, LEGACY_CORPUS_OPS);
    let (digest, evaluable) = corpus_digest(&env, &terms);
    assert_eq!(terms.len(), 3013, "corpus size");
    assert_eq!(evaluable, LEGACY_PINNED_EVALUABLE, "evaluable corpus terms");
    assert_eq!(digest, LEGACY_PINNED_DIGEST, "corpus digest");
}

/// Corpus terms that fold in the extended corpus.
const EXTENDED_PINNED_EVALUABLE: usize = 966;
/// FNV-1a digest over the extended corpus.
const EXTENDED_PINNED_DIGEST: u64 = 16_094_252_581_770_523_361;

/// The extended corpus digest, pinned.
///
/// Same generator, drawing from `div`, `mod`, `neg` and `ite` over each of the
/// five integer relations as well.  Nothing pins these answers to a previous
/// implementation — there was none — so the value tests above are what establish
/// them and this digest is what keeps them from drifting.  Fewer terms fold than
/// in the legacy corpus (966 of 3013) precisely because `0` is one of the corpus
/// literals and `div`/`mod` by zero must *not* fold, so the refusal path is
/// exercised heavily; the Euclidean values at negative operands are pinned by
/// [`div_is_euclidean`] and [`modulo_is_euclidean_and_never_negative`].
#[test]
fn extended_corpus_matches_the_pinned_digest() {
    let (env, terms) = build_corpus(2500, 500, EXTENDED_CORPUS_OPS);
    let (digest, evaluable) = corpus_digest(&env, &terms);
    assert_eq!(terms.len(), 3013, "corpus size");
    assert_eq!(
        evaluable, EXTENDED_PINNED_EVALUABLE,
        "evaluable corpus terms"
    );
    assert_eq!(digest, EXTENDED_PINNED_DIGEST, "corpus digest");
}

// ── Native stack usage ────────────────────────────────────────────────────

/// A left-nested `-` chain of `depth` levels, and its expected value.
fn sub_chain(env: &mut Env, depth: usize) -> (TermId, BigInt) {
    let one = env.int(1);
    let mut term = env.int(0);
    for _ in 0..depth {
        term = env.sub(term, one);
    }
    (term, BigInt::from(-(depth as i64)))
}

/// The evaluator's native stack usage is constant in the term's depth.
///
/// The chain is evaluated on a **128 KiB** thread — one eighth of the 1 MiB
/// this test used to pin, paired with one eighth of the historical 200 000
/// levels so the ~5 bytes of stack per level it actually pins is unchanged at
/// a 64th of the construction cost.  The recursive implementation this
/// replaced aborted the *process* at roughly 540 levels on a stack this size
/// (4 300 on 1 MiB).  The assertion that matters is that the thread returned
/// at all — a stack overflow is not a catchable failure.
#[test]
fn deep_arithmetic_evaluates_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/200k -> 128 KiB/25k): the
    // ~5 B-per-frame threshold is the pin, so never raise one alone.
    const DEPTH: usize = 25_000;

    let observed = std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut env = Env::new();
            let (term, expected) = sub_chain(&mut env, DEPTH);
            (env.eval(term), expected)
        })
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort");

    assert_eq!(observed.0, Some(observed.1));
}

/// The same for a right-leaning chain, which keeps a finished left-hand value
/// in every frame, and for an n-ary fold nested through its first operand.
#[test]
fn deep_right_leaning_and_nary_terms_evaluate_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const DEPTH: usize = 12_500;

    let observed = std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut env = Env::new();
            let one = env.int(1);
            let mut right = env.int(0);
            for _ in 0..DEPTH {
                right = env.sub(one, right);
            }
            let mut nary = env.int(0);
            for _ in 0..DEPTH {
                nary = env.add([nary, one]);
            }
            (env.eval(right), env.eval(nary))
        })
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort");

    // `1 - (1 - (… - 0))` alternates between 0 and 1; an even chain is 0.
    assert_eq!(observed.0, val(0));
    assert_eq!(observed.1, val(DEPTH as i64));
}

/// The alias-resolution chain is not native recursion either.  Each level is a
/// *rewrite*, not a structural child, so the recursive implementation charged a
/// native frame per alias hop as well.
#[test]
fn a_deep_alias_chain_folds_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/50k -> 128 KiB/6.25k): the
    // ~21 B-per-frame threshold is the pin, so never raise one alone.
    const DEPTH: usize = 6_250;

    let observed = std::thread::Builder::new()
        .stack_size(1 << 17)
        .spawn(|| {
            let mut env = Env::new();
            let top = alias_chain(&mut env, DEPTH);
            env.eval(top)
        })
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort");

    assert_eq!(observed, val(11));
}

// ── The alias rewrite cycle ───────────────────────────────────────────────

/// An alias whose stored value is the very read being resolved makes the
/// read-over-write rewrite cycle: `B = store(A, i, select(B, i))` rewrites
/// `select(B, i)` to itself.
///
/// The recursive implementation followed that rewrite as a tail call and so
/// **aborted the process** with a stack overflow — on well-sorted SMT-LIB input
/// (`collect_array_var_aliases` records the alias for any `(= B (store …))`
/// with `B` a variable, and applies no acyclicity test).  The iterative
/// implementation detects the repeat and answers "not evaluable", which is the
/// same answer it gives for every other term it cannot fold, and which makes
/// `check_int_ordering_conflict` skip the assertion.
#[test]
fn a_self_referential_alias_is_not_evaluable() {
    let mut env = Env::new();
    let base = env.array_var("a");
    let aliased = env.array_var("b");
    let index = env.int(0);
    let read = env.select(aliased, index);
    let store = env.store(base, index, read);
    env.aliases.insert(aliased, store);

    assert_eq!(env.eval(read), None);
}

/// A two-step alias cycle: `B` stores a read of `C` and `C` stores a read of
/// `B`.
#[test]
fn a_two_step_alias_cycle_is_not_evaluable() {
    let mut env = Env::new();
    let base = env.array_var("a");
    let first = env.array_var("b");
    let second = env.array_var("c");
    let index = env.int(0);
    let read_first = env.select(first, index);
    let read_second = env.select(second, index);
    let store_first = env.store(base, index, read_second);
    let store_second = env.store(base, index, read_first);
    env.aliases.insert(first, store_first);
    env.aliases.insert(second, store_second);

    assert_eq!(env.eval(read_first), None);
    assert_eq!(env.eval(read_second), None);
}

/// The whole array check survives the cyclic alias end to end, through the
/// public assertion path rather than a hand-built alias map.
///
/// `(= b (store a 0 (select b 0)))` together with `(< (select b 0) 5)` is the
/// smallest input that reaches the cycle from `check_array_constraints`.
#[test]
fn a_cyclic_alias_assertion_does_not_abort_the_array_check() {
    // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — the alias
    // graph here is a small fixed shape (a single cycle), sub-10,000-depth
    // by a huge margin, so 1 MiB is intentionally generous. See TODO.md
    // "v0.3.2 backlog".
    let observed = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let mut env = Env::new();
            let base = env.array_var("a");
            let aliased = env.array_var("b");
            let index = env.int(0);
            let read = env.select(aliased, index);
            let store = env.store(base, index, read);
            let alias_eq = env.manager.mk_eq(aliased, store);
            let five = env.int(5);
            let ordering = env.manager.mk_lt(read, five);
            env.solver.assertions = vec![alias_eq, ordering];

            // The alias the collector derives is exactly the cyclic one.
            let aliases = env.solver.collect_array_var_aliases(&env.manager);
            let recorded = aliases.get(&aliased).copied();
            let conflict = env.solver.check_array_constraints(&env.manager);
            (recorded, store, conflict)
        })
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not abort");

    assert_eq!(observed.0, Some(observed.1), "cyclic alias is recorded");
    // Nothing is refuted: the read cannot be folded, so no ordering conflict.
    assert!(!observed.2);
}
