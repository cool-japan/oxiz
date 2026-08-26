//! Iterative, memoizing evaluation of terms against a stored model snapshot.
//!
//! Split out of `context.rs` to keep that file under the 2000-line ceiling;
//! this is pure code motion plus the recursion-to-heap-stack conversion.

use super::ModelValue;
use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;

/// Which of the two model evaluators a pending sub-term must be run through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EvalKind {
    /// Boolean interpretation (`model_eval_bool`).
    Bool,
    /// Numeric interpretation (`model_eval_num`).
    Num,
}

/// One entry on the explicit model-evaluation work-stack.
enum EvalTask {
    /// Resolve `term` directly if it is a model entry / leaf, otherwise
    /// schedule its operands followed by the matching [`EvalTask::Combine`].
    Expand(EvalKind, TermId),
    /// Every operand of `term` is memoized; fold them into `term`'s value.
    Combine(EvalKind, TermId),
}

/// Iterative, memoizing evaluator for soft-constraint / objective terms
/// against a stored model snapshot.
///
/// The model maps *leaf* terms (variables / atoms the solver assigned) to
/// concrete [`ModelValue`]s. A soft constraint term, however, may be an
/// arbitrary boolean expression such as `(not p)`, `(and p q)`, or an
/// arithmetic atom like `(<= x 3)` — none of which appear directly as a
/// model key, so the term structure has to be walked, looking leaves up in
/// the model.
///
/// That walk used to be a pair of mutually recursive functions. Both defects
/// that entails are fixed here:
///
/// * **Stack depth** — soft-constraint terms come straight from user
///   `.smt2` input, so nesting depth is attacker-controlled and the native
///   stack could be blown. The walk now runs on an explicit heap stack.
///   A depth cap was not an option: both entry points return `Option`,
///   where `None` already means "undeterminable", so a cap would be
///   indistinguishable from a genuine unknown and would silently inflate
///   the reported cost.
/// * **Shared sub-terms** — the term DAG is hash-consed, so the recursive
///   form re-expanded every shared node, taking exponential time on a DAG
///   with only linearly many nodes. Both interpretations are memoized on
///   `TermId`, making the walk linear in DAG size. Memoization is sound
///   because evaluation is pure: it reads the (fixed) model and the
///   (fixed) term structure, and there are no binders to make a value
///   context-dependent.
///
/// Operands are expanded eagerly rather than short-circuited (the recursive
/// form stopped early on, say, a `false` conjunct). That is observationally
/// equivalent — evaluation has no side effects — and the memo keeps the
/// extra work bounded by DAG size.
struct ModelEvaluation<'a> {
    model: &'a FxHashMap<TermId, ModelValue>,
    tm: &'a TermManager,
    bool_memo: FxHashMap<TermId, Option<bool>>,
    num_memo: FxHashMap<TermId, Option<BigRational>>,
}

impl<'a> ModelEvaluation<'a> {
    fn new(model: &'a FxHashMap<TermId, ModelValue>, tm: &'a TermManager) -> Self {
        Self {
            model,
            tm,
            bool_memo: FxHashMap::default(),
            num_memo: FxHashMap::default(),
        }
    }

    /// Memoized boolean value of an already-evaluated sub-term.
    ///
    /// A missing entry is treated as "undeterminable", the same conservative
    /// answer a genuine failure produces. The driver always schedules every
    /// operand before the `Combine` that reads it, so a miss cannot occur.
    fn cached_bool(&self, term: TermId) -> Option<bool> {
        self.bool_memo.get(&term).copied().flatten()
    }

    /// Memoized numeric value of an already-evaluated sub-term (see
    /// [`Self::cached_bool`] for the miss policy).
    fn cached_num(&self, term: TermId) -> Option<BigRational> {
        self.num_memo.get(&term).cloned().flatten()
    }

    /// Drive the walk until `term` has a memo entry for `kind`.
    fn run(&mut self, kind: EvalKind, term: TermId) {
        // Copy the shared references out of `self` so the borrow checker
        // lets the loop mutate the memo maps while a `&Term` is live.
        let tm = self.tm;
        let model = self.model;
        let mut stack = vec![EvalTask::Expand(kind, term)];

        while let Some(task) = stack.pop() {
            match task {
                EvalTask::Expand(EvalKind::Bool, t) => {
                    if self.bool_memo.contains_key(&t) {
                        continue;
                    }
                    // A direct model entry always wins (leaf assignment).
                    if let Some(ModelValue::Bool(b)) = model.get(&t) {
                        self.bool_memo.insert(t, Some(*b));
                        continue;
                    }
                    let Some(node) = tm.get(t) else {
                        self.bool_memo.insert(t, None);
                        continue;
                    };
                    let mut operands: Vec<EvalTask> = Vec::new();
                    match &node.kind {
                        TermKind::True | TermKind::False => {}
                        TermKind::Not(a) => operands.push(EvalTask::Expand(EvalKind::Bool, *a)),
                        TermKind::And(args) | TermKind::Or(args) => operands
                            .extend(args.iter().map(|&a| EvalTask::Expand(EvalKind::Bool, a))),
                        TermKind::Xor(a, b) | TermKind::Implies(a, b) => {
                            operands.push(EvalTask::Expand(EvalKind::Bool, *a));
                            operands.push(EvalTask::Expand(EvalKind::Bool, *b));
                        }
                        TermKind::Ite(c, then_t, else_t) => {
                            operands.push(EvalTask::Expand(EvalKind::Bool, *c));
                            operands.push(EvalTask::Expand(EvalKind::Bool, *then_t));
                            operands.push(EvalTask::Expand(EvalKind::Bool, *else_t));
                        }
                        TermKind::Eq(a, b) => {
                            // `Eq` tries boolean equality first, then numeric,
                            // so both interpretations of both sides are needed.
                            operands.push(EvalTask::Expand(EvalKind::Bool, *a));
                            operands.push(EvalTask::Expand(EvalKind::Bool, *b));
                            operands.push(EvalTask::Expand(EvalKind::Num, *a));
                            operands.push(EvalTask::Expand(EvalKind::Num, *b));
                        }
                        TermKind::Distinct(args) => operands
                            .extend(args.iter().map(|&a| EvalTask::Expand(EvalKind::Num, a))),
                        TermKind::Lt(a, b)
                        | TermKind::Le(a, b)
                        | TermKind::Gt(a, b)
                        | TermKind::Ge(a, b) => {
                            operands.push(EvalTask::Expand(EvalKind::Num, *a));
                            operands.push(EvalTask::Expand(EvalKind::Num, *b));
                        }
                        // Any other kind is not modelled by this lightweight
                        // evaluator; report "undeterminable" rather than
                        // fabricating a value.
                        _ => {
                            self.bool_memo.insert(t, None);
                            continue;
                        }
                    }
                    stack.push(EvalTask::Combine(EvalKind::Bool, t));
                    stack.extend(operands);
                }
                EvalTask::Expand(EvalKind::Num, t) => {
                    if self.num_memo.contains_key(&t) {
                        continue;
                    }
                    if let Some(mv) = model.get(&t) {
                        let value = match mv {
                            ModelValue::Int(n) => Some(BigRational::from(n.clone())),
                            ModelValue::Rational(r) => Some(r.clone()),
                            ModelValue::BitVec(_, n) => Some(BigRational::from(n.clone())),
                            ModelValue::Bool(_) => None,
                        };
                        self.num_memo.insert(t, value);
                        continue;
                    }
                    let Some(node) = tm.get(t) else {
                        self.num_memo.insert(t, None);
                        continue;
                    };
                    let mut operands: Vec<EvalTask> = Vec::new();
                    match &node.kind {
                        TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. } => {}
                        TermKind::Neg(a) => operands.push(EvalTask::Expand(EvalKind::Num, *a)),
                        TermKind::Add(args) | TermKind::Mul(args) => operands
                            .extend(args.iter().map(|&a| EvalTask::Expand(EvalKind::Num, a))),
                        TermKind::Sub(a, b) => {
                            operands.push(EvalTask::Expand(EvalKind::Num, *a));
                            operands.push(EvalTask::Expand(EvalKind::Num, *b));
                        }
                        // Operations this evaluator does not model exactly
                        // (e.g. integer division / modulo): stay conservative.
                        _ => {
                            self.num_memo.insert(t, None);
                            continue;
                        }
                    }
                    stack.push(EvalTask::Combine(EvalKind::Num, t));
                    stack.extend(operands);
                }
                EvalTask::Combine(EvalKind::Bool, t) => {
                    let value = self.combine_bool(t);
                    self.bool_memo.insert(t, value);
                }
                EvalTask::Combine(EvalKind::Num, t) => {
                    let value = self.combine_num(t);
                    self.num_memo.insert(t, value);
                }
            }
        }
    }

    /// Fold the memoized operand values of a boolean node into its value.
    fn combine_bool(&self, term: TermId) -> Option<bool> {
        let t = self.tm.get(term)?;
        match &t.kind {
            TermKind::True => Some(true),
            TermKind::False => Some(false),
            TermKind::Not(a) => self.cached_bool(*a).map(|b| !b),
            TermKind::And(args) => {
                let mut all_true = true;
                for &a in args {
                    match self.cached_bool(a) {
                        Some(true) => {}
                        Some(false) => return Some(false),
                        None => all_true = false,
                    }
                }
                if all_true { Some(true) } else { None }
            }
            TermKind::Or(args) => {
                let mut any_unknown = false;
                for &a in args {
                    match self.cached_bool(a) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => any_unknown = true,
                    }
                }
                if any_unknown { None } else { Some(false) }
            }
            TermKind::Xor(a, b) => Some(self.cached_bool(*a)? ^ self.cached_bool(*b)?),
            TermKind::Implies(a, b) => match self.cached_bool(*a) {
                Some(false) => Some(true),
                Some(true) => self.cached_bool(*b),
                None => match self.cached_bool(*b) {
                    Some(true) => Some(true),
                    _ => None,
                },
            },
            TermKind::Ite(c, then_t, else_t) => match self.cached_bool(*c)? {
                true => self.cached_bool(*then_t),
                false => self.cached_bool(*else_t),
            },
            TermKind::Eq(a, b) => {
                // Try boolean equality first, then numeric equality.
                if let (Some(ba), Some(bb)) = (self.cached_bool(*a), self.cached_bool(*b)) {
                    return Some(ba == bb);
                }
                let na = self.cached_num(*a)?;
                let nb = self.cached_num(*b)?;
                Some(na == nb)
            }
            TermKind::Distinct(args) => {
                let mut vals = Vec::with_capacity(args.len());
                for &a in args {
                    vals.push(self.cached_num(a)?);
                }
                for i in 0..vals.len() {
                    for j in (i + 1)..vals.len() {
                        if vals[i] == vals[j] {
                            return Some(false);
                        }
                    }
                }
                Some(true)
            }
            TermKind::Lt(a, b) => Some(self.cached_num(*a)? < self.cached_num(*b)?),
            TermKind::Le(a, b) => Some(self.cached_num(*a)? <= self.cached_num(*b)?),
            TermKind::Gt(a, b) => Some(self.cached_num(*a)? > self.cached_num(*b)?),
            TermKind::Ge(a, b) => Some(self.cached_num(*a)? >= self.cached_num(*b)?),
            _ => None,
        }
    }

    /// Fold the memoized operand values of a numeric node into its value.
    fn combine_num(&self, term: TermId) -> Option<BigRational> {
        let t = self.tm.get(term)?;
        match &t.kind {
            TermKind::IntConst(n) => Some(BigRational::from(n.clone())),
            TermKind::RealConst(r) => Some(BigRational::new(
                BigInt::from(*r.numer()),
                BigInt::from(*r.denom()),
            )),
            TermKind::BitVecConst { value, .. } => Some(BigRational::from(value.clone())),
            TermKind::Neg(a) => Some(-self.cached_num(*a)?),
            TermKind::Add(args) => {
                let mut acc = BigRational::from(BigInt::from(0));
                for &a in args {
                    acc += self.cached_num(a)?;
                }
                Some(acc)
            }
            TermKind::Sub(a, b) => Some(self.cached_num(*a)? - self.cached_num(*b)?),
            TermKind::Mul(args) => {
                let mut acc = BigRational::from(BigInt::from(1));
                for &a in args {
                    acc *= self.cached_num(a)?;
                }
                Some(acc)
            }
            _ => None,
        }
    }
}

/// Evaluate a boolean-valued term against a stored model.
///
/// See [`ModelEvaluation`] for why this is an iterative memoized walk rather
/// than plain recursion.
///
/// Returns `None` when the value cannot be determined from the model.
pub(super) fn model_eval_bool(
    term: TermId,
    model: &FxHashMap<TermId, ModelValue>,
    tm: &TermManager,
) -> Option<bool> {
    let mut eval = ModelEvaluation::new(model, tm);
    eval.run(EvalKind::Bool, term);
    eval.cached_bool(term)
}

/// Evaluate a numeric (integer/real) term against a stored model.
///
/// Returns `None` when the term uses an operation this lightweight evaluator
/// does not model exactly (e.g. integer division / modulo), so callers stay
/// conservative rather than fabricating a value.
#[cfg(test)]
fn model_eval_num(
    term: TermId,
    model: &FxHashMap<TermId, ModelValue>,
    tm: &TermManager,
) -> Option<BigRational> {
    let mut eval = ModelEvaluation::new(model, tm);
    eval.run(EvalKind::Num, term);
    eval.cached_num(term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    /// A soft-constraint term nested 12_500 levels deep, evaluated on a
    /// 128 KiB worker stack. The assertion is simply that the call *returns*:
    /// a stack overflow aborts the whole process, so returning at all is
    /// the proof that the walk no longer uses the native stack.
    ///
    /// `Add` nodes are used for the nesting because `mk_and` flattens
    /// nested conjunctions (so it cannot build a deep boolean spine); a
    /// single relational atom on top routes the walk through the boolean
    /// evaluator into the deep numeric spine.
    #[test]
    fn deep_nesting_returns_instead_of_overflowing() {
        let worker = std::thread::Builder::new().stack_size(1 << 17).spawn(|| {
            let mut tm = TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let one = tm.mk_int(1);
            let mut term = x;
            for _ in 0..12_500 {
                term = tm.mk_add(vec![term, one]);
            }
            let huge = tm.mk_int(BigInt::from(1_000_000_000));
            let atom = tm.mk_le(term, huge);

            let mut model = FxHashMap::default();
            model.insert(x, ModelValue::Int(BigInt::from(0)));
            model_eval_bool(atom, &model, &tm)
        });
        let value = match worker.map(std::thread::JoinHandle::join) {
            Ok(Ok(value)) => value,
            _ => panic!("deep-nesting worker thread did not complete"),
        };
        // 0 + 12_500 ones = 12_500 <= 1_000_000_000.
        assert_eq!(value, Some(true));
    }

    /// A doubling DAG `t_k = (+ t_{k-1} t_{k-1})`: 60 levels, 61 distinct
    /// hash-consed nodes, but 2^60 tree-unfoldings. Without the memo this
    /// never completes; with it the walk is linear in node count.
    #[test]
    fn shared_dag_is_memoized() {
        let mut tm = TermManager::new();
        let x = tm.mk_var("x", tm.sorts.int_sort);
        let mut term = x;
        for _ in 0..60 {
            term = tm.mk_add(vec![term, term]);
        }
        let mut model = FxHashMap::default();
        model.insert(x, ModelValue::Int(BigInt::from(1)));
        let expected = BigRational::from(BigInt::one() << 60u32);
        assert_eq!(model_eval_num(term, &model, &tm), Some(expected));
    }

    /// Semantic pins: the combination rules must survive the conversion
    /// unchanged, including the partial-information cases where `None`
    /// means "undeterminable" rather than "false".
    #[test]
    fn combination_semantics_preserved() {
        let mut tm = TermManager::new();
        let bool_sort = tm.sorts.bool_sort;
        let p = tm.mk_var("p", bool_sort);
        let q = tm.mk_var("q", bool_sort);
        let mut model = FxHashMap::default();
        model.insert(p, ModelValue::Bool(false));
        // `q` is deliberately left unassigned.

        // (and p q) is false because p is false, even though q is unknown.
        let and_pq = tm.mk_and(vec![p, q]);
        assert_eq!(model_eval_bool(and_pq, &model, &tm), Some(false));

        // (or p q) is undeterminable: p is false and q is unknown.
        let or_pq = tm.mk_or(vec![p, q]);
        assert_eq!(model_eval_bool(or_pq, &model, &tm), None);

        // (=> p q) is true because the antecedent is false.
        let imp = tm.mk_implies(p, q);
        assert_eq!(model_eval_bool(imp, &model, &tm), Some(true));

        // (not p) is true.
        let not_p = tm.mk_not(p);
        assert_eq!(model_eval_bool(not_p, &model, &tm), Some(true));

        // Arithmetic atoms over an assigned integer leaf.
        let x = tm.mk_var("x", tm.sorts.int_sort);
        model.insert(x, ModelValue::Int(BigInt::from(3)));
        let three = tm.mk_int(3);
        let le = tm.mk_le(x, three);
        assert_eq!(model_eval_bool(le, &model, &tm), Some(true));
        let lt = tm.mk_lt(x, three);
        assert_eq!(model_eval_bool(lt, &model, &tm), Some(false));

        // An unmodelled operation (integer division) stays undeterminable
        // rather than fabricating a value.
        let div = tm.mk_div(x, three);
        assert_eq!(model_eval_num(div, &model, &tm), None);
    }
}
