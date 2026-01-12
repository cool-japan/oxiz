//! Model validation utilities
//!
//! This module provides functionality to validate that a model satisfies
//! a set of assertions, which is crucial for correctness checking.

use crate::ast::{Model, ModelValue, TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

/// Evaluate a term under a given model
///
/// Returns `None` if the term cannot be fully evaluated (e.g., uninterpreted function
/// without interpretation, or variable without assignment).
pub fn eval_term(term_id: TermId, manager: &TermManager, model: &Model) -> Option<ModelValue> {
    let term = manager.get(term_id)?;

    match &term.kind {
        // Constants
        TermKind::True => Some(ModelValue::Bool(true)),
        TermKind::False => Some(ModelValue::Bool(false)),
        TermKind::IntConst(n) => Some(ModelValue::Int(n.clone())),
        TermKind::RealConst(r) => {
            // Convert Rational<i64> to BigRational
            let numer = BigInt::from(*r.numer());
            let denom = BigInt::from(*r.denom());
            Some(ModelValue::Real(BigRational::new(numer, denom)))
        }
        TermKind::BitVecConst { value, width } => {
            // Convert BigInt to u64 for model value
            let val_u64 = value.iter_u64_digits().next().unwrap_or(0);
            Some(ModelValue::BitVec {
                value: val_u64,
                width: *width,
            })
        }

        // Variables - look up in model
        TermKind::Var(_) => model.get_assignment(term_id).cloned(),

        // Boolean operations
        TermKind::Not(arg) => {
            let val = eval_term(*arg, manager, model)?;
            match val {
                ModelValue::Bool(b) => Some(ModelValue::Bool(!b)),
                _ => None,
            }
        }

        TermKind::And(args) => {
            for &arg in args {
                let val = eval_term(arg, manager, model)?;
                match val {
                    ModelValue::Bool(false) => return Some(ModelValue::Bool(false)),
                    ModelValue::Bool(true) => continue,
                    _ => return None,
                }
            }
            Some(ModelValue::Bool(true))
        }

        TermKind::Or(args) => {
            for &arg in args {
                let val = eval_term(arg, manager, model)?;
                match val {
                    ModelValue::Bool(true) => return Some(ModelValue::Bool(true)),
                    ModelValue::Bool(false) => continue,
                    _ => return None,
                }
            }
            Some(ModelValue::Bool(false))
        }

        TermKind::Implies(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            match (lhs_val, rhs_val) {
                (ModelValue::Bool(a), ModelValue::Bool(b)) => Some(ModelValue::Bool(!a || b)),
                _ => None,
            }
        }

        TermKind::Xor(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            match (lhs_val, rhs_val) {
                (ModelValue::Bool(a), ModelValue::Bool(b)) => Some(ModelValue::Bool(a != b)),
                _ => None,
            }
        }

        // Comparisons
        TermKind::Eq(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            Some(ModelValue::Bool(lhs_val == rhs_val))
        }

        TermKind::Lt(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            Some(ModelValue::Bool(compare_lt(&lhs_val, &rhs_val)?))
        }

        TermKind::Le(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            Some(ModelValue::Bool(compare_le(&lhs_val, &rhs_val)?))
        }

        TermKind::Gt(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            Some(ModelValue::Bool(compare_lt(&rhs_val, &lhs_val)?))
        }

        TermKind::Ge(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            Some(ModelValue::Bool(compare_le(&rhs_val, &lhs_val)?))
        }

        // Arithmetic
        TermKind::Add(args) => {
            if args.is_empty() {
                return Some(ModelValue::Int(BigInt::zero()));
            }
            let first = eval_term(args[0], manager, model)?;
            let mut result = first;
            for &arg in &args[1..] {
                let val = eval_term(arg, manager, model)?;
                result = add_values(&result, &val)?;
            }
            Some(result)
        }

        TermKind::Mul(args) => {
            if args.is_empty() {
                return Some(ModelValue::Int(BigInt::one()));
            }
            let first = eval_term(args[0], manager, model)?;
            let mut result = first;
            for &arg in &args[1..] {
                let val = eval_term(arg, manager, model)?;
                result = mul_values(&result, &val)?;
            }
            Some(result)
        }

        TermKind::Sub(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            sub_values(&lhs_val, &rhs_val)
        }

        TermKind::Neg(arg) => {
            let val = eval_term(*arg, manager, model)?;
            neg_value(&val)
        }

        TermKind::Div(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            div_values(&lhs_val, &rhs_val)
        }

        TermKind::Mod(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            mod_values(&lhs_val, &rhs_val)
        }

        // ITE
        TermKind::Ite(cond, then_branch, else_branch) => {
            let cond_val = eval_term(*cond, manager, model)?;
            match cond_val {
                ModelValue::Bool(true) => eval_term(*then_branch, manager, model),
                ModelValue::Bool(false) => eval_term(*else_branch, manager, model),
                _ => None,
            }
        }

        // Bit-vector operations (simplified - full implementation would be more complex)
        TermKind::BvNot(arg) => {
            let val = eval_term(*arg, manager, model)?;
            match val {
                ModelValue::BitVec { value, width } => {
                    let mask = (1u64 << width) - 1;
                    Some(ModelValue::BitVec {
                        value: (!value) & mask,
                        width,
                    })
                }
                _ => None,
            }
        }

        TermKind::BvAnd(lhs, rhs) => {
            let lhs_val = eval_term(*lhs, manager, model)?;
            let rhs_val = eval_term(*rhs, manager, model)?;
            match (lhs_val, rhs_val) {
                (
                    ModelValue::BitVec {
                        value: v1,
                        width: w1,
                    },
                    ModelValue::BitVec {
                        value: v2,
                        width: w2,
                    },
                ) if w1 == w2 => Some(ModelValue::BitVec {
                    value: v1 & v2,
                    width: w1,
                }),
                _ => None,
            }
        }

        // For other operations, we can't evaluate without more information
        _ => None,
    }
}

/// Validate that a model satisfies an assertion
pub fn validate_assertion(assertion: TermId, manager: &TermManager, model: &Model) -> Result<bool> {
    match eval_term(assertion, manager, model) {
        Some(ModelValue::Bool(b)) => Ok(b),
        Some(_) => Err(OxizError::Internal(
            "Assertion did not evaluate to a boolean value".to_string(),
        )),
        None => Err(OxizError::Internal(
            "Could not fully evaluate assertion under model".to_string(),
        )),
    }
}

/// Validate that a model satisfies all assertions
pub fn validate_model(assertions: &[TermId], manager: &TermManager, model: &Model) -> Result<bool> {
    for &assertion in assertions {
        if !validate_assertion(assertion, manager, model)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Cached term evaluator for improved performance
///
/// This evaluator maintains a cache of already-evaluated terms to avoid
/// redundant computation when the same subterms appear multiple times.
pub struct CachedEvaluator<'a> {
    manager: &'a TermManager,
    model: &'a Model,
    cache: FxHashMap<TermId, Option<ModelValue>>,
}

impl<'a> CachedEvaluator<'a> {
    /// Create a new cached evaluator
    #[must_use]
    pub fn new(manager: &'a TermManager, model: &'a Model) -> Self {
        Self {
            manager,
            model,
            cache: FxHashMap::default(),
        }
    }

    /// Evaluate a term using the cache
    pub fn eval(&mut self, term_id: TermId) -> Option<ModelValue> {
        // Check cache first
        if let Some(cached) = self.cache.get(&term_id) {
            return cached.clone();
        }

        // Evaluate and cache the result
        let result = eval_term_internal(term_id, self.manager, self.model, &mut self.cache);
        self.cache.insert(term_id, result.clone());
        result
    }

    /// Validate an assertion using the cached evaluator
    pub fn validate_assertion(&mut self, assertion: TermId) -> Result<bool> {
        match self.eval(assertion) {
            Some(ModelValue::Bool(b)) => Ok(b),
            Some(_) => Err(OxizError::Internal(
                "Assertion did not evaluate to a boolean value".to_string(),
            )),
            None => Err(OxizError::Internal(
                "Could not fully evaluate assertion under model".to_string(),
            )),
        }
    }

    /// Validate multiple assertions using the cached evaluator
    pub fn validate_assertions(&mut self, assertions: &[TermId]) -> Result<bool> {
        for &assertion in assertions {
            if !self.validate_assertion(assertion)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Get the number of cached evaluations
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the evaluation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Internal evaluation function that uses the cache
fn eval_term_internal(
    term_id: TermId,
    manager: &TermManager,
    model: &Model,
    cache: &mut FxHashMap<TermId, Option<ModelValue>>,
) -> Option<ModelValue> {
    // Check cache first
    if let Some(cached) = cache.get(&term_id) {
        return cached.clone();
    }

    let term = manager.get(term_id)?;

    let result = match &term.kind {
        // Constants
        TermKind::True => Some(ModelValue::Bool(true)),
        TermKind::False => Some(ModelValue::Bool(false)),
        TermKind::IntConst(n) => Some(ModelValue::Int(n.clone())),
        TermKind::RealConst(r) => {
            let numer = BigInt::from(*r.numer());
            let denom = BigInt::from(*r.denom());
            Some(ModelValue::Real(BigRational::new(numer, denom)))
        }
        TermKind::BitVecConst { value, width } => {
            let val_u64 = value.iter_u64_digits().next().unwrap_or(0);
            Some(ModelValue::BitVec {
                value: val_u64,
                width: *width,
            })
        }

        // Variables
        TermKind::Var(_) => model.get_assignment(term_id).cloned(),

        // Boolean operations
        TermKind::Not(arg) => {
            let val = eval_term_internal(*arg, manager, model, cache)?;
            match val {
                ModelValue::Bool(b) => Some(ModelValue::Bool(!b)),
                _ => None,
            }
        }

        TermKind::And(args) => {
            for &arg in args {
                let val = eval_term_internal(arg, manager, model, cache)?;
                match val {
                    ModelValue::Bool(false) => return Some(ModelValue::Bool(false)),
                    ModelValue::Bool(true) => continue,
                    _ => return None,
                }
            }
            Some(ModelValue::Bool(true))
        }

        TermKind::Or(args) => {
            for &arg in args {
                let val = eval_term_internal(arg, manager, model, cache)?;
                match val {
                    ModelValue::Bool(true) => return Some(ModelValue::Bool(true)),
                    ModelValue::Bool(false) => continue,
                    _ => return None,
                }
            }
            Some(ModelValue::Bool(false))
        }

        TermKind::Implies(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            match (lhs_val, rhs_val) {
                (ModelValue::Bool(a), ModelValue::Bool(b)) => Some(ModelValue::Bool(!a || b)),
                _ => None,
            }
        }

        TermKind::Xor(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            match (lhs_val, rhs_val) {
                (ModelValue::Bool(a), ModelValue::Bool(b)) => Some(ModelValue::Bool(a != b)),
                _ => None,
            }
        }

        // Comparisons
        TermKind::Eq(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            Some(ModelValue::Bool(lhs_val == rhs_val))
        }

        TermKind::Lt(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            Some(ModelValue::Bool(compare_lt(&lhs_val, &rhs_val)?))
        }

        TermKind::Le(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            Some(ModelValue::Bool(compare_le(&lhs_val, &rhs_val)?))
        }

        TermKind::Gt(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            Some(ModelValue::Bool(compare_lt(&rhs_val, &lhs_val)?))
        }

        TermKind::Ge(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            Some(ModelValue::Bool(compare_le(&rhs_val, &lhs_val)?))
        }

        // Arithmetic
        TermKind::Add(args) => {
            if args.is_empty() {
                return Some(ModelValue::Int(BigInt::zero()));
            }
            let first = eval_term_internal(args[0], manager, model, cache)?;
            let mut result = first;
            for &arg in &args[1..] {
                let val = eval_term_internal(arg, manager, model, cache)?;
                result = add_values(&result, &val)?;
            }
            Some(result)
        }

        TermKind::Mul(args) => {
            if args.is_empty() {
                return Some(ModelValue::Int(BigInt::one()));
            }
            let first = eval_term_internal(args[0], manager, model, cache)?;
            let mut result = first;
            for &arg in &args[1..] {
                let val = eval_term_internal(arg, manager, model, cache)?;
                result = mul_values(&result, &val)?;
            }
            Some(result)
        }

        TermKind::Sub(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            sub_values(&lhs_val, &rhs_val)
        }

        TermKind::Neg(arg) => {
            let val = eval_term_internal(*arg, manager, model, cache)?;
            neg_value(&val)
        }

        TermKind::Div(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            div_values(&lhs_val, &rhs_val)
        }

        TermKind::Mod(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            mod_values(&lhs_val, &rhs_val)
        }

        // ITE
        TermKind::Ite(cond, then_branch, else_branch) => {
            let cond_val = eval_term_internal(*cond, manager, model, cache)?;
            match cond_val {
                ModelValue::Bool(true) => eval_term_internal(*then_branch, manager, model, cache),
                ModelValue::Bool(false) => eval_term_internal(*else_branch, manager, model, cache),
                _ => None,
            }
        }

        // Bit-vector operations
        TermKind::BvNot(arg) => {
            let val = eval_term_internal(*arg, manager, model, cache)?;
            match val {
                ModelValue::BitVec { value, width } => {
                    let mask = (1u64 << width) - 1;
                    Some(ModelValue::BitVec {
                        value: (!value) & mask,
                        width,
                    })
                }
                _ => None,
            }
        }

        TermKind::BvAnd(lhs, rhs) => {
            let lhs_val = eval_term_internal(*lhs, manager, model, cache)?;
            let rhs_val = eval_term_internal(*rhs, manager, model, cache)?;
            match (lhs_val, rhs_val) {
                (
                    ModelValue::BitVec {
                        value: v1,
                        width: w1,
                    },
                    ModelValue::BitVec {
                        value: v2,
                        width: w2,
                    },
                ) if w1 == w2 => Some(ModelValue::BitVec {
                    value: v1 & v2,
                    width: w1,
                }),
                _ => None,
            }
        }

        _ => None,
    };

    cache.insert(term_id, result.clone());
    result
}

// Helper functions for arithmetic operations

fn compare_lt(lhs: &ModelValue, rhs: &ModelValue) -> Option<bool> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) => Some(a < b),
        (ModelValue::Real(a), ModelValue::Real(b)) => Some(a < b),
        _ => None,
    }
}

fn compare_le(lhs: &ModelValue, rhs: &ModelValue) -> Option<bool> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) => Some(a <= b),
        (ModelValue::Real(a), ModelValue::Real(b)) => Some(a <= b),
        _ => None,
    }
}

fn add_values(lhs: &ModelValue, rhs: &ModelValue) -> Option<ModelValue> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) => Some(ModelValue::Int(a + b)),
        (ModelValue::Real(a), ModelValue::Real(b)) => Some(ModelValue::Real(a + b)),
        _ => None,
    }
}

fn mul_values(lhs: &ModelValue, rhs: &ModelValue) -> Option<ModelValue> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) => Some(ModelValue::Int(a * b)),
        (ModelValue::Real(a), ModelValue::Real(b)) => Some(ModelValue::Real(a * b)),
        _ => None,
    }
}

fn sub_values(lhs: &ModelValue, rhs: &ModelValue) -> Option<ModelValue> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) => Some(ModelValue::Int(a - b)),
        (ModelValue::Real(a), ModelValue::Real(b)) => Some(ModelValue::Real(a - b)),
        _ => None,
    }
}

fn neg_value(val: &ModelValue) -> Option<ModelValue> {
    match val {
        ModelValue::Int(n) => Some(ModelValue::Int(-n)),
        ModelValue::Real(r) => Some(ModelValue::Real(-r)),
        _ => None,
    }
}

fn div_values(lhs: &ModelValue, rhs: &ModelValue) -> Option<ModelValue> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) if !b.is_zero() => Some(ModelValue::Int(a / b)),
        (ModelValue::Real(a), ModelValue::Real(b)) if !b.is_zero() => Some(ModelValue::Real(a / b)),
        _ => None,
    }
}

fn mod_values(lhs: &ModelValue, rhs: &ModelValue) -> Option<ModelValue> {
    match (lhs, rhs) {
        (ModelValue::Int(a), ModelValue::Int(b)) if !b.is_zero() => Some(ModelValue::Int(a % b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_eval_constants() {
        let mut manager = TermManager::new();
        let model = Model::new();

        let true_term = manager.mk_true();
        assert_eq!(
            eval_term(true_term, &manager, &model),
            Some(ModelValue::Bool(true))
        );

        let false_term = manager.mk_false();
        assert_eq!(
            eval_term(false_term, &manager, &model),
            Some(ModelValue::Bool(false))
        );

        let int_term = manager.mk_int(42);
        assert_eq!(
            eval_term(int_term, &manager, &model),
            Some(ModelValue::Int(BigInt::from(42)))
        );
    }

    #[test]
    fn test_eval_variable() {
        let mut manager = TermManager::new();
        let mut model = Model::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        model.assign_int(x, BigInt::from(10));

        assert_eq!(
            eval_term(x, &manager, &model),
            Some(ModelValue::Int(BigInt::from(10)))
        );
    }

    #[test]
    fn test_eval_arithmetic() {
        let mut manager = TermManager::new();
        let model = Model::new();

        // 2 + 3 = 5
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let sum = manager.mk_add(vec![two, three]);

        assert_eq!(
            eval_term(sum, &manager, &model),
            Some(ModelValue::Int(BigInt::from(5)))
        );

        // 2 * 3 = 6
        let prod = manager.mk_mul(vec![two, three]);
        assert_eq!(
            eval_term(prod, &manager, &model),
            Some(ModelValue::Int(BigInt::from(6)))
        );
    }

    #[test]
    fn test_eval_comparison() {
        let mut manager = TermManager::new();
        let model = Model::new();

        let two = manager.mk_int(2);
        let three = manager.mk_int(3);

        // 2 < 3 = true
        let lt = manager.mk_lt(two, three);
        assert_eq!(
            eval_term(lt, &manager, &model),
            Some(ModelValue::Bool(true))
        );

        // 2 > 3 = false
        let gt = manager.mk_gt(two, three);
        assert_eq!(
            eval_term(gt, &manager, &model),
            Some(ModelValue::Bool(false))
        );
    }

    #[test]
    fn test_validate_assertion_simple() {
        let manager = TermManager::new();
        let model = Model::new();

        // true
        let assertion = manager.mk_true();
        assert!(validate_assertion(assertion, &manager, &model).unwrap());

        // false
        let assertion = manager.mk_false();
        assert!(!validate_assertion(assertion, &manager, &model).unwrap());
    }

    #[test]
    fn test_validate_assertion_with_variable() {
        let mut manager = TermManager::new();
        let mut model = Model::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);

        // x = 5
        let eq = manager.mk_eq(x, five);

        // Model: x = 5
        model.assign_int(x, BigInt::from(5));
        assert!(validate_assertion(eq, &manager, &model).unwrap());

        // Model: x = 10
        model.assign_int(x, BigInt::from(10));
        assert!(!validate_assertion(eq, &manager, &model).unwrap());
    }

    #[test]
    fn test_validate_model_multiple_assertions() {
        let mut manager = TermManager::new();
        let mut model = Model::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // x > 0
        let zero = manager.mk_int(0);
        let assertion1 = manager.mk_gt(x, zero);

        // y < 10
        let ten = manager.mk_int(10);
        let assertion2 = manager.mk_lt(y, ten);

        // x + y = 15
        let sum = manager.mk_add(vec![x, y]);
        let fifteen = manager.mk_int(15);
        let assertion3 = manager.mk_eq(sum, fifteen);

        let assertions = vec![assertion1, assertion2, assertion3];

        // Model: x = 5, y = 10 (doesn't satisfy y < 10)
        model.assign_int(x, BigInt::from(5));
        model.assign_int(y, BigInt::from(10));
        assert!(!validate_model(&assertions, &manager, &model).unwrap());

        // Model: x = 7, y = 8 (satisfies all)
        model.assign_int(x, BigInt::from(7));
        model.assign_int(y, BigInt::from(8));
        assert!(validate_model(&assertions, &manager, &model).unwrap());
    }

    #[test]
    fn test_eval_ite() {
        let mut manager = TermManager::new();
        let model = Model::new();

        let cond = manager.mk_true();
        let then_val = manager.mk_int(1);
        let else_val = manager.mk_int(2);

        let ite = manager.mk_ite(cond, then_val, else_val);

        assert_eq!(
            eval_term(ite, &manager, &model),
            Some(ModelValue::Int(BigInt::from(1)))
        );
    }

    #[test]
    fn test_cached_evaluator_basic() {
        let manager = TermManager::new();
        let model = Model::new();

        let mut evaluator = CachedEvaluator::new(&manager, &model);

        let true_term = manager.mk_true();
        assert_eq!(evaluator.eval(true_term), Some(ModelValue::Bool(true)));

        // Evaluating again should use the cache
        assert_eq!(evaluator.eval(true_term), Some(ModelValue::Bool(true)));
        assert_eq!(evaluator.cache_size(), 1);
    }

    #[test]
    fn test_cached_evaluator_shared_subterms() {
        let mut manager = TermManager::new();
        let mut model = Model::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        model.assign_int(x, BigInt::from(5));

        let two = manager.mk_int(2);
        let x_plus_2 = manager.mk_add(vec![x, two]);

        // Build two terms that share the x+2 subterm
        // (x+2) * 3 and (x+2) + 4
        let three = manager.mk_int(3);
        let four = manager.mk_int(4);
        let term1 = manager.mk_mul(vec![x_plus_2, three]);
        let term2 = manager.mk_add(vec![x_plus_2, four]);

        let mut evaluator = CachedEvaluator::new(&manager, &model);

        // Evaluate first term - should cache x+2
        let result1 = evaluator.eval(term1);
        assert_eq!(result1, Some(ModelValue::Int(BigInt::from(21)))); // (5+2)*3 = 21

        // Evaluate second term - should reuse cached x+2
        let result2 = evaluator.eval(term2);
        assert_eq!(result2, Some(ModelValue::Int(BigInt::from(11)))); // (5+2)+4 = 11

        // Cache should contain entries for x, two, x+2, three, term1, four, term2
        assert!(evaluator.cache_size() >= 5);
    }

    #[test]
    fn test_cached_evaluator_validate_assertions() {
        let mut manager = TermManager::new();
        let mut model = Model::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        model.assign_int(x, BigInt::from(5));
        model.assign_int(y, BigInt::from(10));

        // Create assertions
        let zero = manager.mk_int(0);
        let fifteen = manager.mk_int(15);

        let assertion1 = manager.mk_gt(x, zero); // x > 0
        let assertion2 = manager.mk_gt(y, x); // y > x
        let sum = manager.mk_add(vec![x, y]);
        let assertion3 = manager.mk_eq(sum, fifteen); // x + y = 15

        let assertions = vec![assertion1, assertion2, assertion3];

        let mut evaluator = CachedEvaluator::new(&manager, &model);

        // All assertions should be satisfied
        assert!(evaluator.validate_assertions(&assertions).unwrap());
    }

    #[test]
    fn test_cached_evaluator_clear_cache() {
        let manager = TermManager::new();
        let model = Model::new();

        let mut evaluator = CachedEvaluator::new(&manager, &model);

        let true_term = manager.mk_true();
        evaluator.eval(true_term);
        assert_eq!(evaluator.cache_size(), 1);

        evaluator.clear_cache();
        assert_eq!(evaluator.cache_size(), 0);
    }

    #[test]
    fn test_cached_evaluator_complex_formula() {
        let mut manager = TermManager::new();
        let mut model = Model::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        model.assign_int(x, BigInt::from(3));

        // Build formula: (x * x) + (x * x) = 18
        let x_squared = manager.mk_mul(vec![x, x]);
        let sum = manager.mk_add(vec![x_squared, x_squared]);
        let eighteen = manager.mk_int(18);
        let formula = manager.mk_eq(sum, eighteen);

        let mut evaluator = CachedEvaluator::new(&manager, &model);

        // Should evaluate to true: (3*3) + (3*3) = 9 + 9 = 18
        assert!(evaluator.validate_assertion(formula).unwrap());

        // x_squared should only be evaluated once and cached
        assert!(evaluator.cache_size() >= 3); // At least x, x_squared, and sum
    }
}
