//! A closed quantifier-free assertion over datatype, enumeration and
//! uninterpreted-sort VALUES, decided by value — without interning a term
//! (re-fix pass 17, decision (79)(a)).
//!
//! # Why the quantifier-free net needs it
//!
//! The exact evaluator the printed-model checks run on
//! (`solver::array_completion_certify::evaluate_closed`, the model gate's
//! evaluator) has no value for a datatype- or uninterpreted-sorted term, so
//! an equality, a `distinct`, an `ite` or an array read of such a sort stays
//! open there.  The quantifier-free datatype net (decision (72)(a),
//! `Context::withhold_a_falsifying_datatype_model`) therefore only ever saw
//! the comparisons the term builder had already folded while the printed
//! model was substituted in — two literal constructor applications, which
//! `mk_eq` decides at construction (`#P2b-76`).  Whether a check's model was
//! caught depended on how its printed tables happened to read: `gen_dt.py`
//! seed 30093154 `d00239`'s first model prints `h` as an `ite` chain over its
//! argument, so `(= (cons (k (v nil)) l3) (h (+ 1 (- 1))))` closed to an
//! equality between an `ite` whose conditions compare `(+ 1 (- 1))` with
//! numerals and the list `(cons 5 (cons 2 nil))` — open, published, and
//! false; its second and third checks closed the same assertion to two
//! literal constructor applications, which folded to `false` and were
//! withheld.  A `distinct` of two constructor values was never decided at
//! all, and an array whose element sort is a datatype, an enumeration or an
//! uninterpreted sort (`#P2b-89`) was never read.
//!
//! # What is read
//!
//! A closed term is evaluated to a VALUE: a scalar (the model gate's own
//! exact evaluator reads every sub-term with no datatype- or
//! uninterpreted-sorted part, so every scalar operator means here what it
//! means there), a constructor applied to values, or an uninterpreted-sort
//! witness (`@uc_S_n`, read back as a constant of its own).  On top of the
//! scalars: a constructor application, a selector or tester of a constructor
//! value, the Boolean connectives (three-valued: an open operand decides
//! nothing it could change), an `ite` by its condition, a read of a printed
//! store chain by comparing its keys with the index, and `+`, `-`, `*`, the
//! order comparisons over numbers whose operands are values.  Two values
//! compare constructor by constructor and then field by field, numbers by
//! their exact value, witnesses by identity: two distinct witnesses are two
//! distinct elements, which is what the printed model says and the
//! hypothesis every certificate of it takes.  Anything else is open.  Sound
//! for the reason every evaluation of the printed model is: each step reads
//! the model as printed.
//!
//! # Why nothing is interned
//!
//! The reading runs at every quantifier-free `sat` over such a sort, and the
//! search of a LATER check is sensitive to what the term arena holds (the
//! note on `(get-value)`'s printed reading says the same): a first version
//! that rewrote the assertion through the term builder moved later checks'
//! trajectories — `gen_dt.py` seed 30093154 `d00261`'s third check printed
//! a model pass 16 printed and z3 confirms, and that version found another.
//! So the reading borrows the term manager shared and builds nothing.

use super::*;
use crate::prelude::{FxHashMap, FxHashSet};
use crate::solver::EvalVal;
use oxiz_core::interner::Spur;

/// Nesting depth of a closed term the reading follows (a deeper term is
/// open; the round's assertions are a few dozen levels deep).
const MAX_GROUND_DEPTH: u32 = 512;

/// Constructor fields one value comparison may visit.
const MAX_VALUE_PAIRS: usize = 4_096;

/// The value of a closed term.
#[derive(Clone, Debug, PartialEq)]
enum GroundValue {
    /// A Boolean, number or bit-vector.
    Scalar(EvalVal),
    /// A constructor applied to values.
    Constructor(Spur, Vec<GroundValue>),
    /// An uninterpreted-sort witness: an element of its own.
    Witness(TermId),
}

impl Context {
    /// The truth value of `term` — a closed Boolean term, with no free
    /// symbol but the uninterpreted-sort witnesses in `witnesses` — read by
    /// value (see the module docs); `None` where the reading is open.
    pub(in crate::context) fn ground_truth(
        &self,
        term: TermId,
        witnesses: &FxHashSet<TermId>,
    ) -> Option<bool> {
        let mut reader = GroundReader {
            ctx: self,
            witnesses,
            memo: FxHashMap::default(),
            valued_sorts: FxHashMap::default(),
        };
        match reader.value(term, 0)? {
            GroundValue::Scalar(EvalVal::Bool(holds)) => Some(holds),
            _ => None,
        }
    }
}

/// One reading of closed terms: memoised per term, so a shared sub-term is
/// read once.
struct GroundReader<'a> {
    ctx: &'a Context,
    witnesses: &'a FxHashSet<TermId>,
    memo: FxHashMap<TermId, Option<GroundValue>>,
    /// Whether a term has a datatype- or uninterpreted-sorted part (or a
    /// witness), by term.
    valued_sorts: FxHashMap<TermId, bool>,
}

impl GroundReader<'_> {
    fn value(&mut self, term: TermId, depth: u32) -> Option<GroundValue> {
        if depth > MAX_GROUND_DEPTH {
            return None;
        }
        if let Some(known) = self.memo.get(&term) {
            return known.clone();
        }
        let value = self.compute(term, depth);
        self.memo.insert(term, value.clone());
        value
    }

    fn compute(&mut self, term: TermId, depth: u32) -> Option<GroundValue> {
        if self.witnesses.contains(&term) {
            return Some(GroundValue::Witness(term));
        }
        let data = self.ctx.terms.get(term)?;
        let sort = data.sort;
        let kind = data.kind.clone();
        // A scalar sub-term with no datatype or uninterpreted part is the
        // model gate's to read: every operator means what it means there.
        if self.is_scalar_sort(sort)
            && !self.has_valued_sort(term, depth)
            && let Some(value) = crate::solver::array_completion_certify::evaluate_closed_value_pure(
                &self.ctx.solver,
                term,
                &self.ctx.terms,
            )
        {
            return Some(GroundValue::Scalar(value));
        }
        let next = depth + 1;
        match kind {
            TermKind::True => Some(boolean(true)),
            TermKind::False => Some(boolean(false)),
            TermKind::DtConstructor { constructor, args } => {
                let mut fields = Vec::with_capacity(args.len());
                for arg in args {
                    fields.push(self.value(arg, next)?);
                }
                Some(GroundValue::Constructor(constructor, fields))
            }
            TermKind::DtSelector { selector, arg } => {
                let GroundValue::Constructor(built, fields) = self.value(arg, next)? else {
                    return None;
                };
                let arg_sort = self.ctx.terms.get(arg)?.sort;
                let position = self.field_position(arg_sort, built, selector)?;
                fields.get(position).cloned()
            }
            TermKind::DtTester { constructor, arg } => {
                let GroundValue::Constructor(built, _) = self.value(arg, next)? else {
                    return None;
                };
                let arg_sort = self.ctx.terms.get(arg)?.sort;
                self.has_constructor(arg_sort, constructor)
                    .then(|| boolean(built == constructor))
            }
            TermKind::Not(inner) => Some(boolean(!self.truth(inner, next)?)),
            TermKind::And(args) => {
                let mut open = false;
                for arg in args {
                    match self.truth(arg, next) {
                        Some(false) => return Some(boolean(false)),
                        Some(true) => {}
                        None => open = true,
                    }
                }
                (!open).then(|| boolean(true))
            }
            TermKind::Or(args) => {
                let mut open = false;
                for arg in args {
                    match self.truth(arg, next) {
                        Some(true) => return Some(boolean(true)),
                        Some(false) => {}
                        None => open = true,
                    }
                }
                (!open).then(|| boolean(false))
            }
            TermKind::Implies(premise, conclusion) => {
                match (self.truth(premise, next), self.truth(conclusion, next)) {
                    (Some(false), _) | (_, Some(true)) => Some(boolean(true)),
                    (Some(true), Some(false)) => Some(boolean(false)),
                    _ => None,
                }
            }
            TermKind::Xor(left, right) => {
                let (a, b) = (self.truth(left, next)?, self.truth(right, next)?);
                Some(boolean(a != b))
            }
            TermKind::Ite(condition, then_branch, else_branch) => {
                match self.truth(condition, next) {
                    Some(true) => self.value(then_branch, next),
                    Some(false) => self.value(else_branch, next),
                    None => {
                        let a = self.value(then_branch, next)?;
                        let b = self.value(else_branch, next)?;
                        (self.equal(&a, &b) == Some(true)).then_some(a)
                    }
                }
            }
            TermKind::Eq(left, right) => {
                let a = self.value(left, next)?;
                let b = self.value(right, next)?;
                Some(boolean(self.equal(&a, &b)?))
            }
            TermKind::Distinct(args) => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.value(arg, next)?);
                }
                for (position, a) in values.iter().enumerate() {
                    for b in values.iter().skip(position + 1) {
                        if self.equal(a, b)? {
                            return Some(boolean(false));
                        }
                    }
                }
                Some(boolean(true))
            }
            TermKind::Select(array, index) => {
                let index = self.value(index, next)?;
                self.read(array, &index, next)
            }
            TermKind::Neg(inner) => {
                let number = self.number(inner, next)?;
                let numer = number.numer().checked_neg()?;
                Some(number_value(num_rational::Rational64::new(
                    numer,
                    *number.denom(),
                )))
            }
            TermKind::Add(args) => {
                let mut sum = num_rational::Rational64::from_integer(0);
                for arg in args {
                    let term = self.number(arg, next)?;
                    sum = num_traits::CheckedAdd::checked_add(&sum, &term)?;
                }
                Some(number_value(sum))
            }
            TermKind::Mul(args) => {
                let mut product = num_rational::Rational64::from_integer(1);
                for arg in args {
                    let factor = self.number(arg, next)?;
                    product = num_traits::CheckedMul::checked_mul(&product, &factor)?;
                }
                Some(number_value(product))
            }
            TermKind::Sub(left, right) => {
                let (a, b) = (self.number(left, next)?, self.number(right, next)?);
                Some(number_value(num_traits::CheckedSub::checked_sub(&a, &b)?))
            }
            TermKind::Lt(left, right) => self.compare(left, right, next, |a, b| a < b),
            TermKind::Le(left, right) => self.compare(left, right, next, |a, b| a <= b),
            TermKind::Gt(left, right) => self.compare(left, right, next, |a, b| a > b),
            TermKind::Ge(left, right) => self.compare(left, right, next, |a, b| a >= b),
            _ => None,
        }
    }

    /// The Boolean value of `term`, where it is read.
    fn truth(&mut self, term: TermId, depth: u32) -> Option<bool> {
        match self.value(term, depth)? {
            GroundValue::Scalar(EvalVal::Bool(holds)) => Some(holds),
            _ => None,
        }
    }

    /// The number `term` reads, where it is read.
    fn number(&mut self, term: TermId, depth: u32) -> Option<num_rational::Rational64> {
        match self.value(term, depth)? {
            GroundValue::Scalar(EvalVal::Num(number)) => Some(number),
            _ => None,
        }
    }

    /// An order comparison of two numbers.
    fn compare(
        &mut self,
        left: TermId,
        right: TermId,
        depth: u32,
        holds: impl Fn(&num_rational::Rational64, &num_rational::Rational64) -> bool,
    ) -> Option<GroundValue> {
        let (a, b) = (self.number(left, depth)?, self.number(right, depth)?);
        Some(boolean(holds(&a, &b)))
    }

    /// The value `(select array index)` reads where `array` is a printed
    /// store chain (or an `ite` over such chains): the entry whose key equals
    /// `index`, passing every key that differs, then the constant array's
    /// default.
    fn read(&mut self, array: TermId, index: &GroundValue, depth: u32) -> Option<GroundValue> {
        let mut level = array;
        let mut steps = depth;
        loop {
            steps += 1;
            if steps > MAX_GROUND_DEPTH {
                return None;
            }
            match self.ctx.terms.get(level)?.kind.clone() {
                TermKind::Store(inner, key, value) => {
                    let key = self.value(key, steps)?;
                    if self.equal(&key, index)? {
                        return self.value(value, steps);
                    }
                    level = inner;
                }
                TermKind::Ite(condition, then_branch, else_branch) => {
                    level = if self.truth(condition, steps)? {
                        then_branch
                    } else {
                        else_branch
                    };
                }
                TermKind::Apply { func, args }
                    if args.len() == 1
                        && self.ctx.terms.resolve_str(func)
                            == oxiz_core::smtlib::CONST_ARRAY_FUNC =>
                {
                    return self.value(*args.first()?, steps);
                }
                _ => return None,
            }
        }
    }

    /// Whether two values are one: `Some(true)` / `Some(false)` where they
    /// decide it, `None` otherwise (see the module docs).
    fn equal(&self, left: &GroundValue, right: &GroundValue) -> Option<bool> {
        let mut pairs: Vec<(&GroundValue, &GroundValue)> = vec![(left, right)];
        let mut visited = 0usize;
        while let Some((a, b)) = pairs.pop() {
            visited += 1;
            if visited > MAX_VALUE_PAIRS {
                return None;
            }
            match (a, b) {
                (GroundValue::Scalar(x), GroundValue::Scalar(y)) => {
                    if x != y {
                        return Some(false);
                    }
                }
                (GroundValue::Witness(x), GroundValue::Witness(y)) => {
                    if x != y {
                        return Some(false);
                    }
                }
                (
                    GroundValue::Constructor(name_a, fields_a),
                    GroundValue::Constructor(name_b, fields_b),
                ) => {
                    if name_a != name_b {
                        return Some(false);
                    }
                    if fields_a.len() != fields_b.len() {
                        return None;
                    }
                    pairs.extend(fields_a.iter().zip(fields_b.iter()));
                }
                _ => return None,
            }
        }
        Some(true)
    }

    /// The position of `selector` among `constructor`'s fields in the
    /// datatype `sort`; `None` when `constructor` has no such field (the
    /// selector is then unspecified there, and the reading is open).
    fn field_position(&self, sort: SortId, constructor: Spur, selector: Spur) -> Option<usize> {
        let sorts = &self.ctx.terms.sorts;
        let def = sorts.get_datatype(sorts.datatype_name(sort)?)?;
        let declared = def.constructors.iter().find(|c| c.name == constructor)?;
        declared
            .selectors
            .iter()
            .position(|&(field, _)| field == selector)
    }

    /// Whether `constructor` is a constructor of the datatype `sort`.
    fn has_constructor(&self, sort: SortId, constructor: Spur) -> bool {
        let sorts = &self.ctx.terms.sorts;
        sorts
            .datatype_name(sort)
            .and_then(|name| sorts.get_datatype(name))
            .is_some_and(|def| def.constructors.iter().any(|c| c.name == constructor))
    }

    fn is_scalar_sort(&self, sort: SortId) -> bool {
        matches!(
            self.ctx.terms.sorts.get(sort).map(|s| &s.kind),
            Some(SortKind::Bool | SortKind::Int | SortKind::Real | SortKind::BitVec(_))
        )
    }

    /// Whether `term` has a datatype- or uninterpreted-sorted sub-term (or a
    /// witness) — memoised, depth-bounded (a term past the bound counts as
    /// having one, so it is read structurally rather than handed whole to
    /// the scalar evaluator).
    fn has_valued_sort(&mut self, term: TermId, depth: u32) -> bool {
        if let Some(&known) = self.valued_sorts.get(&term) {
            return known;
        }
        if depth > MAX_GROUND_DEPTH {
            return true;
        }
        let answer = self.witnesses.contains(&term)
            || match self.ctx.terms.get(term) {
                None => true,
                Some(data) => {
                    let valued = matches!(
                        self.ctx.terms.sorts.get(data.sort).map(|s| &s.kind),
                        Some(SortKind::Datatype(_) | SortKind::Uninterpreted(_))
                    );
                    let children: Vec<TermId> = oxiz_core::ast::traversal::get_children(&data.kind)
                        .into_iter()
                        .collect();
                    valued
                        || children
                            .into_iter()
                            .any(|child| self.has_valued_sort(child, depth + 1))
                }
            };
        self.valued_sorts.insert(term, answer);
        answer
    }
}

fn boolean(holds: bool) -> GroundValue {
    GroundValue::Scalar(EvalVal::Bool(holds))
}

fn number_value(number: num_rational::Rational64) -> GroundValue {
    GroundValue::Scalar(EvalVal::Num(number))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context with the list datatype `L`, an uninterpreted sort `U`, and
    /// one constant of each declared: `(context, L, U)`.
    fn context() -> (Context, SortId, SortId) {
        let mut ctx = Context::new();
        let script = "(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
                      (declare-sort U 0)\n(declare-const l1 L)\n(declare-const u1 U)\n";
        assert!(ctx.execute_script(script).is_ok());
        let sort_of = |ctx: &Context, name: &str| {
            ctx.declared_consts
                .iter()
                .find(|decl| decl.name == name)
                .map(|decl| decl.sort)
        };
        let (Some(list), Some(uninterpreted)) = (sort_of(&ctx, "l1"), sort_of(&ctx, "u1")) else {
            panic!("the declarations are in scope");
        };
        (ctx, list, uninterpreted)
    }

    /// `[n₀, n₁, …]` as a constructor term.
    fn list(ctx: &mut Context, sort: SortId, items: &[i64]) -> TermId {
        let mut out = ctx.terms.mk_dt_constructor("nil", [], sort);
        for &item in items.iter().rev() {
            let head = ctx.terms.mk_int(item);
            out = ctx.terms.mk_dt_constructor("cons", [head, out], sort);
        }
        out
    }

    /// `d00239`'s first check: an equality between an `ite` over
    /// `(+ 1 (- 1))` and a list, and a `distinct` of lists — both open to the
    /// exact evaluator, both read by value, and nothing interned for it.
    #[test]
    fn an_ite_over_a_closed_condition_and_a_distinct_of_lists_are_read() {
        let (mut ctx, sort, _) = context();
        let none = FxHashSet::default();
        let one = ctx.terms.mk_int(1);
        let minus_one = ctx.terms.mk_neg(one);
        let zero_sum = ctx.terms.mk_add([one, minus_one]);
        let zero = ctx.terms.mk_int(0);
        let condition = ctx.terms.mk_eq(zero_sum, zero);
        let then_list = list(&mut ctx, sort, &[2, 2]);
        let else_list = list(&mut ctx, sort, &[6]);
        let chosen = ctx.terms.mk_ite(condition, then_list, else_list);
        let other = list(&mut ctx, sort, &[5, 2]);
        let equality = ctx.terms.mk_eq(chosen, other);
        let same = list(&mut ctx, sort, &[2, 2]);
        let holds = ctx.terms.mk_eq(chosen, same);
        let a = list(&mut ctx, sort, &[3]);
        let b = list(&mut ctx, sort, &[0, 2]);
        let apart = ctx.terms.mk_distinct([a, b]);
        let c = list(&mut ctx, sort, &[3]);
        let repeated = ctx.terms.mk_distinct([a, b, c]);
        let interned = ctx.terms.len();
        assert_eq!(ctx.ground_truth(equality, &none), Some(false));
        assert_eq!(ctx.ground_truth(holds, &none), Some(true));
        assert_eq!(ctx.ground_truth(apart, &none), Some(true));
        assert_eq!(ctx.ground_truth(repeated, &none), Some(false));
        assert_eq!(ctx.terms.len(), interned, "the reading interns no term");
    }

    /// A read of a printed store chain into a datatype: by value, at a key
    /// it names and at one it does not (the default); a selector of the
    /// value read.
    #[test]
    fn a_read_of_a_list_valued_store_chain_is_decided() {
        let (mut ctx, sort, _) = context();
        let none = FxHashSet::default();
        let int = ctx.terms.sorts.int_sort;
        let array_sort = ctx.terms.sorts.array(int, sort);
        let nil = list(&mut ctx, sort, &[]);
        let base = ctx
            .terms
            .mk_apply(oxiz_core::smtlib::CONST_ARRAY_FUNC, [nil], array_sort);
        let zero = ctx.terms.mk_int(0);
        let one_list = list(&mut ctx, sort, &[1]);
        let chain = ctx.terms.mk_store(base, zero, one_list);
        let read_zero = ctx.terms.mk_select(chain, zero);
        let two_list = list(&mut ctx, sort, &[2]);
        let wrong = ctx.terms.mk_eq(read_zero, two_list);
        assert_eq!(ctx.ground_truth(wrong, &none), Some(false), "a[0] = [1]");
        let three = ctx.terms.mk_int(3);
        let read_three = ctx.terms.mk_select(chain, three);
        let default = ctx.terms.mk_eq(read_three, nil);
        assert_eq!(ctx.ground_truth(default, &none), Some(true), "a[3] = nil");
        let shorter = list(&mut ctx, sort, &[]);
        let longer = list(&mut ctx, sort, &[1, 1]);
        let apart = ctx.terms.mk_distinct([read_zero, shorter, longer]);
        assert_eq!(
            ctx.ground_truth(apart, &none),
            Some(true),
            "[1], [], [1, 1]"
        );
    }

    /// Witnesses compare by identity (two witnesses are two elements); a
    /// declared constant is no value, and its comparison stays open.
    #[test]
    fn witnesses_compare_by_identity_and_a_symbol_stays_open() {
        let (mut ctx, sort, uninterpreted) = context();
        let w0 = ctx.terms.mk_var("w0", uninterpreted);
        let w1 = ctx.terms.mk_var("w1", uninterpreted);
        let witnesses: FxHashSet<TermId> = [w0, w1].into_iter().collect();
        let apart = ctx.terms.mk_eq(w0, w1);
        assert_eq!(ctx.ground_truth(apart, &witnesses), Some(false));
        let symbol = ctx.terms.mk_var("l1", sort);
        let one_list = list(&mut ctx, sort, &[1]);
        let open = ctx.terms.mk_eq(symbol, one_list);
        assert_eq!(ctx.ground_truth(open, &witnesses), None);
        let u1 = ctx.terms.mk_var("u1", uninterpreted);
        let open = ctx.terms.mk_eq(w0, u1);
        assert_eq!(ctx.ground_truth(open, &witnesses), None);
    }
}
