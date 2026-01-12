//! Model Evaluator
//!
//! Evaluates terms under a given model assignment.

use super::{Model, Value};
use crate::ast::{TermId, TermKind, TermManager};
use num_rational::Rational64;
use std::collections::HashMap;

/// Result of evaluation
#[derive(Debug, Clone)]
pub enum EvalResult {
    /// Successful evaluation
    Ok(Value),
    /// Term has no value in model
    Undefined(TermId),
    /// Evaluation error
    Error(String),
}

impl EvalResult {
    /// Check if evaluation succeeded
    pub fn is_ok(&self) -> bool {
        matches!(self, EvalResult::Ok(_))
    }

    /// Get value if successful
    pub fn value(&self) -> Option<&Value> {
        match self {
            EvalResult::Ok(v) => Some(v),
            _ => None,
        }
    }

    /// Unwrap value or panic
    pub fn unwrap(self) -> Value {
        match self {
            EvalResult::Ok(v) => v,
            EvalResult::Undefined(t) => panic!("Term {:?} is undefined", t),
            EvalResult::Error(e) => panic!("Evaluation error: {}", e),
        }
    }
}

/// Cache for evaluated terms
#[derive(Debug, Default)]
pub struct EvalCache {
    cache: HashMap<TermId, Value>,
}

impl EvalCache {
    /// Create a new cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Get cached value
    pub fn get(&self, term: TermId) -> Option<&Value> {
        self.cache.get(&term)
    }

    /// Insert value into cache
    pub fn insert(&mut self, term: TermId, value: Value) {
        self.cache.insert(term, value);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Model evaluator with caching
#[derive(Debug)]
pub struct ModelEvaluator<'a> {
    model: &'a Model,
    cache: EvalCache,
    use_cache: bool,
}

impl<'a> ModelEvaluator<'a> {
    /// Create a new evaluator
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            cache: EvalCache::new(),
            use_cache: true,
        }
    }

    /// Create evaluator without caching
    pub fn without_cache(model: &'a Model) -> Self {
        Self {
            model,
            cache: EvalCache::new(),
            use_cache: false,
        }
    }

    /// Evaluate a term
    pub fn eval(&mut self, term: TermId, manager: &TermManager) -> EvalResult {
        // Check cache first
        if self.use_cache
            && let Some(v) = self.cache.get(term)
        {
            return EvalResult::Ok(v.clone());
        }

        // Check model assignment
        if let Some(v) = self.model.get(term) {
            if self.use_cache {
                self.cache.insert(term, v.clone());
            }
            return EvalResult::Ok(v.clone());
        }

        // Evaluate based on term structure
        let result = self.eval_term(term, manager);

        // Cache result
        if self.use_cache
            && let EvalResult::Ok(ref v) = result
        {
            self.cache.insert(term, v.clone());
        }

        result
    }

    /// Internal evaluation
    fn eval_term(&mut self, term: TermId, manager: &TermManager) -> EvalResult {
        let t = match manager.get(term) {
            Some(t) => t,
            None => return EvalResult::Error(format!("Unknown term: {:?}", term)),
        };

        match &t.kind {
            // Constants
            TermKind::True => EvalResult::Ok(Value::Bool(true)),
            TermKind::False => EvalResult::Ok(Value::Bool(false)),
            TermKind::IntConst(n) => {
                // Convert BigInt to i64 if possible
                let val: i64 = n.try_into().unwrap_or(0);
                EvalResult::Ok(Value::Int(val))
            }
            TermKind::RealConst(r) => EvalResult::Ok(Value::Rational(*r)),
            TermKind::BitVecConst { value, width } => {
                let val: u64 = value.try_into().unwrap_or(0);
                EvalResult::Ok(Value::BitVec(*width, val))
            }

            // Variables - look up in model
            TermKind::Var(_) => match self.model.get(term) {
                Some(v) => EvalResult::Ok(v.clone()),
                None => EvalResult::Undefined(term),
            },

            // Boolean operations
            TermKind::Not(inner) => match self.eval(*inner, manager) {
                EvalResult::Ok(Value::Bool(b)) => EvalResult::Ok(Value::Bool(!b)),
                EvalResult::Ok(_) => EvalResult::Error("Not: expected bool".to_string()),
                e => e,
            },
            TermKind::And(args) => self.eval_and(args.as_slice(), manager),
            TermKind::Or(args) => self.eval_or(args.as_slice(), manager),
            TermKind::Xor(a, b) => match (self.eval(*a, manager), self.eval(*b, manager)) {
                (EvalResult::Ok(Value::Bool(x)), EvalResult::Ok(Value::Bool(y))) => {
                    EvalResult::Ok(Value::Bool(x ^ y))
                }
                (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                    EvalResult::Error("Xor: expected bools".to_string())
                }
                (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
                (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
            },
            TermKind::Implies(a, b) => match (self.eval(*a, manager), self.eval(*b, manager)) {
                (EvalResult::Ok(Value::Bool(x)), EvalResult::Ok(Value::Bool(y))) => {
                    EvalResult::Ok(Value::Bool(!x || y))
                }
                (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                    EvalResult::Error("Implies: expected bools".to_string())
                }
                (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
                (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
            },
            TermKind::Ite(cond, then_branch, else_branch) => match self.eval(*cond, manager) {
                EvalResult::Ok(Value::Bool(true)) => self.eval(*then_branch, manager),
                EvalResult::Ok(Value::Bool(false)) => self.eval(*else_branch, manager),
                EvalResult::Ok(_) => EvalResult::Error("Ite: condition must be bool".to_string()),
                e => e,
            },

            // Equality
            TermKind::Eq(a, b) => match (self.eval(*a, manager), self.eval(*b, manager)) {
                (EvalResult::Ok(v1), EvalResult::Ok(v2)) => EvalResult::Ok(Value::Bool(v1 == v2)),
                (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
                (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
            },
            TermKind::Distinct(args) => self.eval_distinct(args.as_slice(), manager),

            // Arithmetic
            TermKind::Add(args) => self.eval_add(args.as_slice(), manager),
            TermKind::Sub(a, b) => self.eval_sub(*a, *b, manager),
            TermKind::Mul(args) => self.eval_mul(args.as_slice(), manager),
            TermKind::Div(a, b) => self.eval_div(*a, *b, manager),
            TermKind::Neg(a) => self.eval_neg(*a, manager),
            TermKind::Lt(a, b) => self.eval_lt(*a, *b, manager),
            TermKind::Le(a, b) => self.eval_le(*a, *b, manager),
            TermKind::Gt(a, b) => self.eval_lt(*b, *a, manager),
            TermKind::Ge(a, b) => self.eval_le(*b, *a, manager),

            // Bitvector operations
            TermKind::BvNot(a) => self.eval_bvnot(*a, manager),
            TermKind::BvAnd(a, b) => self.eval_bvand(*a, *b, manager),
            TermKind::BvOr(a, b) => self.eval_bvor(*a, *b, manager),
            TermKind::BvXor(a, b) => self.eval_bvxor(*a, *b, manager),
            TermKind::BvAdd(a, b) => self.eval_bvadd(*a, *b, manager),
            TermKind::BvSub(a, b) => self.eval_bvsub(*a, *b, manager),
            TermKind::BvMul(a, b) => self.eval_bvmul(*a, *b, manager),

            // Unhandled - return undefined for now
            _ => EvalResult::Undefined(term),
        }
    }

    fn eval_and(&mut self, args: &[TermId], manager: &TermManager) -> EvalResult {
        for arg in args {
            match self.eval(*arg, manager) {
                EvalResult::Ok(Value::Bool(false)) => return EvalResult::Ok(Value::Bool(false)),
                EvalResult::Ok(Value::Bool(true)) => continue,
                EvalResult::Ok(_) => return EvalResult::Error("And: expected bool".to_string()),
                e => return e,
            }
        }
        EvalResult::Ok(Value::Bool(true))
    }

    fn eval_or(&mut self, args: &[TermId], manager: &TermManager) -> EvalResult {
        for arg in args {
            match self.eval(*arg, manager) {
                EvalResult::Ok(Value::Bool(true)) => return EvalResult::Ok(Value::Bool(true)),
                EvalResult::Ok(Value::Bool(false)) => continue,
                EvalResult::Ok(_) => return EvalResult::Error("Or: expected bool".to_string()),
                e => return e,
            }
        }
        EvalResult::Ok(Value::Bool(false))
    }

    fn eval_distinct(&mut self, args: &[TermId], manager: &TermManager) -> EvalResult {
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            match self.eval(*arg, manager) {
                EvalResult::Ok(v) => values.push(v),
                e => return e,
            }
        }

        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                if values[i] == values[j] {
                    return EvalResult::Ok(Value::Bool(false));
                }
            }
        }
        EvalResult::Ok(Value::Bool(true))
    }

    fn eval_add(&mut self, args: &[TermId], manager: &TermManager) -> EvalResult {
        let mut sum = Rational64::from_integer(0);
        for arg in args {
            match self.eval(*arg, manager) {
                EvalResult::Ok(Value::Int(n)) => sum += Rational64::from_integer(n),
                EvalResult::Ok(Value::Rational(r)) => sum += r,
                EvalResult::Ok(_) => return EvalResult::Error("Add: expected number".to_string()),
                e => return e,
            }
        }
        if *sum.denom() == 1 {
            EvalResult::Ok(Value::Int(*sum.numer()))
        } else {
            EvalResult::Ok(Value::Rational(sum))
        }
    }

    fn eval_sub(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::Int(x)), EvalResult::Ok(Value::Int(y))) => {
                EvalResult::Ok(Value::Int(x - y))
            }
            (EvalResult::Ok(v1), EvalResult::Ok(v2)) => {
                match (v1.as_rational(), v2.as_rational()) {
                    (Some(r1), Some(r2)) => {
                        let r = r1 - r2;
                        if *r.denom() == 1 {
                            EvalResult::Ok(Value::Int(*r.numer()))
                        } else {
                            EvalResult::Ok(Value::Rational(r))
                        }
                    }
                    _ => EvalResult::Error("Sub: expected numbers".to_string()),
                }
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_mul(&mut self, args: &[TermId], manager: &TermManager) -> EvalResult {
        let mut product = Rational64::from_integer(1);
        for arg in args {
            match self.eval(*arg, manager) {
                EvalResult::Ok(Value::Int(n)) => product *= Rational64::from_integer(n),
                EvalResult::Ok(Value::Rational(r)) => product *= r,
                EvalResult::Ok(_) => return EvalResult::Error("Mul: expected number".to_string()),
                e => return e,
            }
        }
        if *product.denom() == 1 {
            EvalResult::Ok(Value::Int(*product.numer()))
        } else {
            EvalResult::Ok(Value::Rational(product))
        }
    }

    fn eval_div(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(v1), EvalResult::Ok(v2)) => {
                match (v1.as_rational(), v2.as_rational()) {
                    (Some(r1), Some(r2)) => {
                        if r2 == Rational64::from_integer(0) {
                            EvalResult::Error("Division by zero".to_string())
                        } else {
                            let r = r1 / r2;
                            if *r.denom() == 1 {
                                EvalResult::Ok(Value::Int(*r.numer()))
                            } else {
                                EvalResult::Ok(Value::Rational(r))
                            }
                        }
                    }
                    _ => EvalResult::Error("Div: expected numbers".to_string()),
                }
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_neg(&mut self, a: TermId, manager: &TermManager) -> EvalResult {
        match self.eval(a, manager) {
            EvalResult::Ok(Value::Int(n)) => EvalResult::Ok(Value::Int(-n)),
            EvalResult::Ok(Value::Rational(r)) => EvalResult::Ok(Value::Rational(-r)),
            EvalResult::Ok(_) => EvalResult::Error("Neg: expected number".to_string()),
            e => e,
        }
    }

    fn eval_lt(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(v1), EvalResult::Ok(v2)) => {
                match (v1.as_rational(), v2.as_rational()) {
                    (Some(r1), Some(r2)) => EvalResult::Ok(Value::Bool(r1 < r2)),
                    _ => EvalResult::Error("Lt: expected numbers".to_string()),
                }
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_le(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(v1), EvalResult::Ok(v2)) => {
                match (v1.as_rational(), v2.as_rational()) {
                    (Some(r1), Some(r2)) => EvalResult::Ok(Value::Bool(r1 <= r2)),
                    _ => EvalResult::Error("Le: expected numbers".to_string()),
                }
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_bvnot(&mut self, a: TermId, manager: &TermManager) -> EvalResult {
        match self.eval(a, manager) {
            EvalResult::Ok(Value::BitVec(w, v)) => {
                let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
                EvalResult::Ok(Value::BitVec(w, !v & mask))
            }
            EvalResult::Ok(_) => EvalResult::Error("BvNot: expected bitvector".to_string()),
            e => e,
        }
    }

    fn eval_bvand(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::BitVec(w1, v1)), EvalResult::Ok(Value::BitVec(w2, v2))) => {
                if w1 != w2 {
                    EvalResult::Error("BvAnd: width mismatch".to_string())
                } else {
                    EvalResult::Ok(Value::BitVec(w1, v1 & v2))
                }
            }
            (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                EvalResult::Error("BvAnd: expected bitvectors".to_string())
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_bvor(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::BitVec(w1, v1)), EvalResult::Ok(Value::BitVec(w2, v2))) => {
                if w1 != w2 {
                    EvalResult::Error("BvOr: width mismatch".to_string())
                } else {
                    EvalResult::Ok(Value::BitVec(w1, v1 | v2))
                }
            }
            (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                EvalResult::Error("BvOr: expected bitvectors".to_string())
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_bvxor(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::BitVec(w1, v1)), EvalResult::Ok(Value::BitVec(w2, v2))) => {
                if w1 != w2 {
                    EvalResult::Error("BvXor: width mismatch".to_string())
                } else {
                    EvalResult::Ok(Value::BitVec(w1, v1 ^ v2))
                }
            }
            (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                EvalResult::Error("BvXor: expected bitvectors".to_string())
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_bvadd(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::BitVec(w1, v1)), EvalResult::Ok(Value::BitVec(w2, v2))) => {
                if w1 != w2 {
                    EvalResult::Error("BvAdd: width mismatch".to_string())
                } else {
                    let mask = if w1 >= 64 { u64::MAX } else { (1u64 << w1) - 1 };
                    EvalResult::Ok(Value::BitVec(w1, v1.wrapping_add(v2) & mask))
                }
            }
            (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                EvalResult::Error("BvAdd: expected bitvectors".to_string())
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_bvsub(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::BitVec(w1, v1)), EvalResult::Ok(Value::BitVec(w2, v2))) => {
                if w1 != w2 {
                    EvalResult::Error("BvSub: width mismatch".to_string())
                } else {
                    let mask = if w1 >= 64 { u64::MAX } else { (1u64 << w1) - 1 };
                    EvalResult::Ok(Value::BitVec(w1, v1.wrapping_sub(v2) & mask))
                }
            }
            (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                EvalResult::Error("BvSub: expected bitvectors".to_string())
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    fn eval_bvmul(&mut self, a: TermId, b: TermId, manager: &TermManager) -> EvalResult {
        match (self.eval(a, manager), self.eval(b, manager)) {
            (EvalResult::Ok(Value::BitVec(w1, v1)), EvalResult::Ok(Value::BitVec(w2, v2))) => {
                if w1 != w2 {
                    EvalResult::Error("BvMul: width mismatch".to_string())
                } else {
                    let mask = if w1 >= 64 { u64::MAX } else { (1u64 << w1) - 1 };
                    EvalResult::Ok(Value::BitVec(w1, v1.wrapping_mul(v2) & mask))
                }
            }
            (EvalResult::Ok(_), EvalResult::Ok(_)) => {
                EvalResult::Error("BvMul: expected bitvectors".to_string())
            }
            (e @ EvalResult::Undefined(_), _) | (_, e @ EvalResult::Undefined(_)) => e,
            (e @ EvalResult::Error(_), _) | (_, e @ EvalResult::Error(_)) => e,
        }
    }

    /// Clear the evaluation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_cache() {
        let mut cache = EvalCache::new();
        let t1 = TermId::from(1u32);

        assert!(cache.is_empty());

        cache.insert(t1, Value::Bool(true));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(t1), Some(&Value::Bool(true)));

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_eval_result() {
        let ok = EvalResult::Ok(Value::Int(42));
        assert!(ok.is_ok());
        assert_eq!(ok.value(), Some(&Value::Int(42)));

        let undef = EvalResult::Undefined(TermId::from(1u32));
        assert!(!undef.is_ok());
        assert_eq!(undef.value(), None);
    }
}
