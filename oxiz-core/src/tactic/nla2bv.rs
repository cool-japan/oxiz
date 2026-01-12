//! Nonlinear Arithmetic to Bit-Vector tactic (NLA2BV)
//!
//! This tactic converts nonlinear integer arithmetic formulas to equivalent
//! bit-vector formulas when the variables are bounded. This enables the use
//! of efficient bit-blasting SAT solvers for NLA problems.
//!
//! # Algorithm
//!
//! 1. Extract bounds for all integer variables from the formula
//! 2. Compute the required bit-width for each variable
//! 3. Replace integer variables with bit-vectors
//! 4. Convert arithmetic operations to bit-vector operations
//! 5. Handle overflow by using extended precision
//!
//! # Supported Operations
//!
//! - Integer constants
//! - Addition, subtraction, negation
//! - Multiplication (the "nonlinear" part)
//! - Division, modulo (unsigned)
//! - Comparisons (=, <, <=, >, >=)
//!
//! # Reference
//!
//! Based on Z3's `nla2bv_tactic` in `src/tactic/arith/nla2bv_tactic.cpp`

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use crate::sort::SortId;
use crate::tactic::{Goal, Tactic, TacticResult};
use lasso::Spur;
use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;

/// Comparison operator for bounds extraction
#[derive(Debug, Clone, Copy)]
enum CompOp {
    Le,
    Lt,
    Ge,
    Gt,
}

/// Data extracted from a term for bounds processing (to avoid borrow issues)
enum BoundsExtractionData {
    Comparison {
        op: CompOp,
        lhs: TermId,
        rhs: TermId,
    },
    Args(Vec<TermId>),
    Negation(TermId),
}

/// Data extracted from arithmetic term for conversion
enum ArithConvData {
    IntConst(BigInt),
    Var { name: Spur, var_width: Option<u32> },
    Neg(TermId),
    Add(Vec<TermId>),
    Sub(TermId, TermId),
    Mul(Vec<TermId>),
    Div(TermId, TermId),
    Mod(TermId, TermId),
    Ite(TermId, TermId, TermId),
    Passthrough,
}

/// Data extracted from term for boolean/formula conversion
enum TermConvData {
    Passthrough,
    And(Vec<TermId>),
    Or(Vec<TermId>),
    Not(TermId),
    Implies(TermId, TermId),
    Ite(TermId, TermId, TermId),
    Eq(TermId, TermId),
    Lt(TermId, TermId),
    Le(TermId, TermId),
    Gt(TermId, TermId),
    Ge(TermId, TermId),
    Forall {
        vars: Vec<(Spur, SortId)>,
        body: TermId,
        patterns: Vec<Vec<TermId>>,
    },
    Exists {
        vars: Vec<(Spur, SortId)>,
        body: TermId,
        patterns: Vec<Vec<TermId>>,
    },
    TryArith,
}

/// Configuration for NLA to BV conversion
#[derive(Debug, Clone)]
pub struct Nla2BvConfig {
    /// Default bit width when bounds are not available
    pub default_bit_width: u32,
    /// Maximum bit width allowed (to prevent blowup)
    pub max_bit_width: u32,
    /// Whether to use signed bitvectors (vs unsigned)
    pub use_signed: bool,
    /// Whether to handle division by zero specially
    pub handle_div_zero: bool,
}

impl Default for Nla2BvConfig {
    fn default() -> Self {
        Self {
            default_bit_width: 32,
            max_bit_width: 64,
            use_signed: true,
            handle_div_zero: true,
        }
    }
}

/// Variable bound information
#[derive(Debug, Clone)]
struct VarBounds {
    /// Lower bound (inclusive)
    lower: Option<BigInt>,
    /// Upper bound (inclusive)
    upper: Option<BigInt>,
}

impl VarBounds {
    fn new() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    /// Update with a lower bound constraint
    fn update_lower(&mut self, val: BigInt) {
        match &self.lower {
            Some(current) if current >= &val => {}
            _ => self.lower = Some(val),
        }
    }

    /// Update with an upper bound constraint
    fn update_upper(&mut self, val: BigInt) {
        match &self.upper {
            Some(current) if current <= &val => {}
            _ => self.upper = Some(val),
        }
    }

    /// Check if bounds are complete (both present)
    fn is_bounded(&self) -> bool {
        self.lower.is_some() && self.upper.is_some()
    }

    /// Get the range size if bounded
    fn range_size(&self) -> Option<BigInt> {
        match (&self.lower, &self.upper) {
            (Some(l), Some(u)) => Some(u - l + 1),
            _ => None,
        }
    }

    /// Compute required unsigned bit width for the range
    fn required_bit_width(&self) -> Option<u32> {
        let range = self.range_size()?;
        if range <= BigInt::from(0) {
            return Some(1);
        }
        // bits = ceil(log2(range))
        let bits = range.bits() as u32;
        Some(bits.max(1))
    }
}

/// NLA to BV conversion tactic
#[derive(Debug)]
pub struct Nla2BvTactic<'a> {
    manager: &'a mut TermManager,
    config: Nla2BvConfig,
    /// Extracted bounds for variables
    var_bounds: FxHashMap<Spur, VarBounds>,
    /// Mapping from integer variables to bitvector variables
    var_to_bv: FxHashMap<TermId, TermId>,
    /// Bit width for each variable
    var_widths: FxHashMap<Spur, u32>,
}

impl<'a> Nla2BvTactic<'a> {
    /// Create a new NLA to BV tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self::with_config(manager, Nla2BvConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(manager: &'a mut TermManager, config: Nla2BvConfig) -> Self {
        Self {
            manager,
            config,
            var_bounds: FxHashMap::default(),
            var_to_bv: FxHashMap::default(),
            var_widths: FxHashMap::default(),
        }
    }

    /// Apply the tactic
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Phase 1: Extract bounds from constraints
        self.extract_bounds(&goal.assertions);

        // Phase 2: Check if all variables are bounded
        if !self.check_all_bounded(&goal.assertions) {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 3: Compute bit widths
        self.compute_bit_widths();

        // Phase 4: Convert constraints
        let mut new_assertions = Vec::new();
        for &assertion in &goal.assertions {
            match self.convert_term(assertion) {
                Some(converted) => new_assertions.push(converted),
                None => return Ok(TacticResult::NotApplicable),
            }
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Extract variable bounds from constraints
    fn extract_bounds(&mut self, assertions: &[TermId]) {
        for &assertion in assertions {
            self.extract_bounds_from_term(assertion);
        }
    }

    /// Extract bounds from a single term
    fn extract_bounds_from_term(&mut self, term: TermId) {
        // Extract all needed info first to avoid borrow issues
        let Some(t) = self.manager.get(term) else {
            return;
        };

        // Clone data we need for recursive calls
        let kind_data: BoundsExtractionData = match &t.kind {
            TermKind::Le(lhs, rhs) => BoundsExtractionData::Comparison {
                op: CompOp::Le,
                lhs: *lhs,
                rhs: *rhs,
            },
            TermKind::Lt(lhs, rhs) => BoundsExtractionData::Comparison {
                op: CompOp::Lt,
                lhs: *lhs,
                rhs: *rhs,
            },
            TermKind::Ge(lhs, rhs) => BoundsExtractionData::Comparison {
                op: CompOp::Ge,
                lhs: *lhs,
                rhs: *rhs,
            },
            TermKind::Gt(lhs, rhs) => BoundsExtractionData::Comparison {
                op: CompOp::Gt,
                lhs: *lhs,
                rhs: *rhs,
            },
            TermKind::And(args) => BoundsExtractionData::Args(args.to_vec()),
            TermKind::Or(args) => BoundsExtractionData::Args(args.to_vec()),
            TermKind::Not(inner) => BoundsExtractionData::Negation(*inner),
            _ => return,
        };

        // Now process without holding borrow
        match kind_data {
            BoundsExtractionData::Comparison { op, lhs, rhs } => {
                self.process_comparison(op, lhs, rhs);
            }
            BoundsExtractionData::Args(args) => {
                for arg in args {
                    self.extract_bounds_from_term(arg);
                }
            }
            BoundsExtractionData::Negation(inner) => {
                self.extract_negated_bounds(inner);
            }
        }
    }

    /// Process a comparison for bounds extraction
    fn process_comparison(&mut self, op: CompOp, lhs: TermId, rhs: TermId) {
        match op {
            CompOp::Le => {
                // x <= c => x.upper = c
                if let Some(var) = self.get_var_name(lhs)
                    && let Some(c) = self.get_int_const(rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_upper(c);
                }
                // c <= x => x.lower = c
                if let Some(var) = self.get_var_name(rhs)
                    && let Some(c) = self.get_int_const(lhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_lower(c);
                }
            }
            CompOp::Lt => {
                // x < c => x.upper = c - 1
                if let Some(var) = self.get_var_name(lhs)
                    && let Some(c) = self.get_int_const(rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_upper(c - 1);
                }
                // c < x => x.lower = c + 1
                if let Some(var) = self.get_var_name(rhs)
                    && let Some(c) = self.get_int_const(lhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_lower(c + 1);
                }
            }
            CompOp::Ge => {
                // x >= c => x.lower = c
                if let Some(var) = self.get_var_name(lhs)
                    && let Some(c) = self.get_int_const(rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_lower(c);
                }
                // c >= x => x.upper = c
                if let Some(var) = self.get_var_name(rhs)
                    && let Some(c) = self.get_int_const(lhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_upper(c);
                }
            }
            CompOp::Gt => {
                // x > c => x.lower = c + 1
                if let Some(var) = self.get_var_name(lhs)
                    && let Some(c) = self.get_int_const(rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_lower(c + 1);
                }
                // c > x => x.upper = c - 1
                if let Some(var) = self.get_var_name(rhs)
                    && let Some(c) = self.get_int_const(lhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_upper(c - 1);
                }
            }
        }
    }

    /// Extract bounds from negated comparisons
    fn extract_negated_bounds(&mut self, term: TermId) {
        let Some(t) = self.manager.get(term) else {
            return;
        };

        match &t.kind {
            // !(x <= c) => x > c => x >= c + 1
            TermKind::Le(lhs, rhs) => {
                if let Some(var) = self.get_var_name(*lhs)
                    && let Some(c) = self.get_int_const(*rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_lower(c + 1);
                }
            }
            // !(x < c) => x >= c
            TermKind::Lt(lhs, rhs) => {
                if let Some(var) = self.get_var_name(*lhs)
                    && let Some(c) = self.get_int_const(*rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_lower(c.clone());
                }
            }
            // !(x >= c) => x < c => x <= c - 1
            TermKind::Ge(lhs, rhs) => {
                if let Some(var) = self.get_var_name(*lhs)
                    && let Some(c) = self.get_int_const(*rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_upper(c - 1);
                }
            }
            // !(x > c) => x <= c
            TermKind::Gt(lhs, rhs) => {
                if let Some(var) = self.get_var_name(*lhs)
                    && let Some(c) = self.get_int_const(*rhs)
                {
                    self.var_bounds
                        .entry(var)
                        .or_insert_with(VarBounds::new)
                        .update_upper(c.clone());
                }
            }
            _ => {}
        }
    }

    /// Get variable name if term is a variable
    fn get_var_name(&self, term: TermId) -> Option<Spur> {
        let t = self.manager.get(term)?;
        match &t.kind {
            TermKind::Var(name) => Some(*name),
            _ => None,
        }
    }

    /// Get integer constant value
    fn get_int_const(&self, term: TermId) -> Option<BigInt> {
        let t = self.manager.get(term)?;
        match &t.kind {
            TermKind::IntConst(val) => Some(val.clone()),
            _ => None,
        }
    }

    /// Check if all integer variables in the formula are bounded
    fn check_all_bounded(&self, assertions: &[TermId]) -> bool {
        let mut all_vars = Vec::new();
        for &assertion in assertions {
            self.collect_int_vars(assertion, &mut all_vars);
        }

        for var in all_vars {
            if let Some(bounds) = self.var_bounds.get(&var) {
                if !bounds.is_bounded() {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    /// Collect all integer variables from a term
    fn collect_int_vars(&self, term: TermId, vars: &mut Vec<Spur>) {
        let Some(t) = self.manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Var(name) => {
                // Check if it's an integer variable (by context - appears in arithmetic)
                if !vars.contains(name) {
                    vars.push(*name);
                }
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args.iter() {
                    self.collect_int_vars(arg, vars);
                }
            }
            TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Eq(a, b) => {
                self.collect_int_vars(*a, vars);
                self.collect_int_vars(*b, vars);
            }
            TermKind::Neg(a) | TermKind::Not(a) => {
                self.collect_int_vars(*a, vars);
            }
            TermKind::Ite(c, t, e) => {
                self.collect_int_vars(*c, vars);
                self.collect_int_vars(*t, vars);
                self.collect_int_vars(*e, vars);
            }
            _ => {}
        }
    }

    /// Compute bit widths for all variables
    fn compute_bit_widths(&mut self) {
        for (var, bounds) in &self.var_bounds {
            let width = bounds
                .required_bit_width()
                .unwrap_or(self.config.default_bit_width)
                .min(self.config.max_bit_width);

            // For signed arithmetic, add one bit for sign
            let final_width = if self.config.use_signed {
                width + 1
            } else {
                width
            };

            self.var_widths.insert(*var, final_width);
        }
    }

    /// Convert an integer term to bitvector
    fn convert_term(&mut self, term: TermId) -> Option<TermId> {
        let t = self.manager.get(term)?;

        // Extract data to avoid holding borrow during recursive calls
        let conv_data: TermConvData = match &t.kind {
            TermKind::True | TermKind::False => TermConvData::Passthrough,
            TermKind::And(args) => TermConvData::And(args.to_vec()),
            TermKind::Or(args) => TermConvData::Or(args.to_vec()),
            TermKind::Not(inner) => TermConvData::Not(*inner),
            TermKind::Implies(a, b) => TermConvData::Implies(*a, *b),
            TermKind::Ite(c, t, e) => TermConvData::Ite(*c, *t, *e),
            TermKind::Eq(a, b) => TermConvData::Eq(*a, *b),
            TermKind::Lt(a, b) => TermConvData::Lt(*a, *b),
            TermKind::Le(a, b) => TermConvData::Le(*a, *b),
            TermKind::Gt(a, b) => TermConvData::Gt(*a, *b),
            TermKind::Ge(a, b) => TermConvData::Ge(*a, *b),
            TermKind::BvUlt(_, _)
            | TermKind::BvUle(_, _)
            | TermKind::BvSlt(_, _)
            | TermKind::BvSle(_, _) => TermConvData::Passthrough,
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => TermConvData::Forall {
                vars: vars.to_vec(),
                body: *body,
                patterns: patterns.iter().map(|p| p.to_vec()).collect(),
            },
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => TermConvData::Exists {
                vars: vars.to_vec(),
                body: *body,
                patterns: patterns.iter().map(|p| p.to_vec()).collect(),
            },
            _ => TermConvData::TryArith,
        };

        // Now process without holding borrow
        match conv_data {
            TermConvData::Passthrough => Some(term),

            TermConvData::And(args) => {
                let converted: Option<Vec<_>> =
                    args.into_iter().map(|a| self.convert_term(a)).collect();
                let converted = converted?;
                Some(self.manager.mk_and(converted))
            }

            TermConvData::Or(args) => {
                let converted: Option<Vec<_>> =
                    args.into_iter().map(|a| self.convert_term(a)).collect();
                let converted = converted?;
                Some(self.manager.mk_or(converted))
            }

            TermConvData::Not(inner) => {
                let converted = self.convert_term(inner)?;
                Some(self.manager.mk_not(converted))
            }

            TermConvData::Implies(a, b) => {
                let a_conv = self.convert_term(a)?;
                let b_conv = self.convert_term(b)?;
                Some(self.manager.mk_implies(a_conv, b_conv))
            }

            TermConvData::Ite(c, t, e) => {
                let c_conv = self.convert_term(c)?;
                let t_conv = self.convert_arith_to_bv(t)?;
                let e_conv = self.convert_arith_to_bv(e)?;
                Some(self.manager.mk_ite(c_conv, t_conv, e_conv))
            }

            TermConvData::Eq(a, b) => {
                // Try arithmetic first, fall back to boolean
                if let (Some(a_bv), Some(b_bv)) =
                    (self.convert_arith_to_bv(a), self.convert_arith_to_bv(b))
                {
                    Some(self.manager.mk_eq(a_bv, b_bv))
                } else {
                    // Boolean equality
                    let a_conv = self.convert_term(a)?;
                    let b_conv = self.convert_term(b)?;
                    Some(self.manager.mk_eq(a_conv, b_conv))
                }
            }

            TermConvData::Lt(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                if self.config.use_signed {
                    Some(self.manager.mk_bv_slt(a_bv, b_bv))
                } else {
                    Some(self.manager.mk_bv_ult(a_bv, b_bv))
                }
            }

            TermConvData::Le(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                if self.config.use_signed {
                    Some(self.manager.mk_bv_sle(a_bv, b_bv))
                } else {
                    Some(self.manager.mk_bv_ule(a_bv, b_bv))
                }
            }

            TermConvData::Gt(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                // a > b  <==>  b < a
                if self.config.use_signed {
                    Some(self.manager.mk_bv_slt(b_bv, a_bv))
                } else {
                    Some(self.manager.mk_bv_ult(b_bv, a_bv))
                }
            }

            TermConvData::Ge(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                // a >= b  <==>  b <= a
                if self.config.use_signed {
                    Some(self.manager.mk_bv_sle(b_bv, a_bv))
                } else {
                    Some(self.manager.mk_bv_ule(b_bv, a_bv))
                }
            }

            TermConvData::Forall {
                vars,
                body,
                patterns,
            } => {
                let body_conv = self.convert_term(body)?;
                // Convert Spur names to owned strings first
                let var_names: Vec<_> = vars
                    .iter()
                    .map(|(name, sort)| (self.manager.resolve_str(*name).to_string(), *sort))
                    .collect();
                // Now create references for the API
                let var_strs: Vec<_> = var_names
                    .iter()
                    .map(|(name, sort)| (name.as_str(), *sort))
                    .collect();
                Some(
                    self.manager
                        .mk_forall_with_patterns(var_strs, body_conv, patterns),
                )
            }

            TermConvData::Exists {
                vars,
                body,
                patterns,
            } => {
                let body_conv = self.convert_term(body)?;
                // Convert Spur names to owned strings first
                let var_names: Vec<_> = vars
                    .iter()
                    .map(|(name, sort)| (self.manager.resolve_str(*name).to_string(), *sort))
                    .collect();
                // Now create references for the API
                let var_strs: Vec<_> = var_names
                    .iter()
                    .map(|(name, sort)| (name.as_str(), *sort))
                    .collect();
                Some(
                    self.manager
                        .mk_exists_with_patterns(var_strs, body_conv, patterns),
                )
            }

            TermConvData::TryArith => {
                // Try to convert as arithmetic, return original if not applicable
                self.convert_arith_to_bv(term).or(Some(term))
            }
        }
    }

    /// Convert an arithmetic expression to bitvector
    fn convert_arith_to_bv(&mut self, term: TermId) -> Option<TermId> {
        // Check cache first
        if let Some(&cached) = self.var_to_bv.get(&term) {
            return Some(cached);
        }

        let t = self.manager.get(term)?;
        let width = self.get_expression_width(term);

        // Extract data to avoid holding borrow during recursive calls
        let conv_data: ArithConvData = match &t.kind {
            TermKind::IntConst(val) => ArithConvData::IntConst(val.clone()),
            TermKind::Var(name) => ArithConvData::Var {
                name: *name,
                var_width: self.var_widths.get(name).copied(),
            },
            TermKind::Neg(inner) => ArithConvData::Neg(*inner),
            TermKind::Add(args) => ArithConvData::Add(args.to_vec()),
            TermKind::Sub(a, b) => ArithConvData::Sub(*a, *b),
            TermKind::Mul(args) => ArithConvData::Mul(args.to_vec()),
            TermKind::Div(a, b) => ArithConvData::Div(*a, *b),
            TermKind::Mod(a, b) => ArithConvData::Mod(*a, *b),
            TermKind::Ite(c, t, e) => ArithConvData::Ite(*c, *t, *e),
            TermKind::BitVecConst { .. }
            | TermKind::BvAdd(_, _)
            | TermKind::BvSub(_, _)
            | TermKind::BvMul(_, _)
            | TermKind::BvUdiv(_, _)
            | TermKind::BvSdiv(_, _)
            | TermKind::BvUrem(_, _)
            | TermKind::BvSrem(_, _) => ArithConvData::Passthrough,
            _ => return None,
        };

        // Now process without holding borrow
        let result = match conv_data {
            ArithConvData::IntConst(val) => {
                let bv_val = if self.config.use_signed && val.is_negative() {
                    let modulus = BigInt::from(1) << width;
                    (&val % &modulus + &modulus) % &modulus
                } else {
                    val
                };
                Some(self.manager.mk_bitvec(bv_val, width))
            }

            ArithConvData::Var { name, var_width } => {
                let w = var_width.unwrap_or(width);
                let name_str = self.manager.resolve_str(name).to_string();
                let bv_name = format!("{}_bv", name_str);
                let bv_sort = self.manager.sorts.bitvec(w);
                let bv_var = self.manager.mk_var(&bv_name, bv_sort);
                Some(bv_var)
            }

            ArithConvData::Neg(inner) => {
                let inner_bv = self.convert_arith_to_bv(inner)?;
                Some(self.manager.mk_bv_neg(inner_bv))
            }

            ArithConvData::Add(args) => {
                if args.is_empty() {
                    return Some(self.manager.mk_bitvec(BigInt::zero(), width));
                }
                let mut result = self.convert_arith_to_bv(args[0])?;
                for arg in args.into_iter().skip(1) {
                    let arg_bv = self.convert_arith_to_bv(arg)?;
                    result = self.manager.mk_bv_add(result, arg_bv);
                }
                Some(result)
            }

            ArithConvData::Sub(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                Some(self.manager.mk_bv_sub(a_bv, b_bv))
            }

            ArithConvData::Mul(args) => {
                if args.is_empty() {
                    return Some(self.manager.mk_bitvec(BigInt::from(1), width));
                }
                let mut result = self.convert_arith_to_bv(args[0])?;
                for arg in args.into_iter().skip(1) {
                    let arg_bv = self.convert_arith_to_bv(arg)?;
                    result = self.manager.mk_bv_mul(result, arg_bv);
                }
                Some(result)
            }

            ArithConvData::Div(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                if self.config.use_signed {
                    Some(self.manager.mk_bv_sdiv(a_bv, b_bv))
                } else {
                    Some(self.manager.mk_bv_udiv(a_bv, b_bv))
                }
            }

            ArithConvData::Mod(a, b) => {
                let a_bv = self.convert_arith_to_bv(a)?;
                let b_bv = self.convert_arith_to_bv(b)?;
                if self.config.use_signed {
                    Some(self.manager.mk_bv_srem(a_bv, b_bv))
                } else {
                    Some(self.manager.mk_bv_urem(a_bv, b_bv))
                }
            }

            ArithConvData::Ite(c, t, e) => {
                let c_conv = self.convert_term(c)?;
                let t_bv = self.convert_arith_to_bv(t)?;
                let e_bv = self.convert_arith_to_bv(e)?;
                Some(self.manager.mk_ite(c_conv, t_bv, e_bv))
            }

            ArithConvData::Passthrough => Some(term),
        };

        // Cache the result
        if let Some(res) = result {
            self.var_to_bv.insert(term, res);
            return Some(res);
        }

        result
    }

    /// Get the bit width for an expression
    fn get_expression_width(&self, term: TermId) -> u32 {
        let Some(t) = self.manager.get(term) else {
            return self.config.default_bit_width;
        };

        match &t.kind {
            TermKind::Var(name) => self
                .var_widths
                .get(name)
                .copied()
                .unwrap_or(self.config.default_bit_width),

            TermKind::IntConst(val) => {
                // Compute required bits for constant
                let bits = if val.is_negative() {
                    val.abs().bits() as u32 + 1
                } else {
                    val.bits() as u32
                };
                bits.max(1).min(self.config.max_bit_width)
            }

            TermKind::Add(args) | TermKind::Mul(args) => {
                // For add/mul, we may need extra bits to prevent overflow
                let max_width = args
                    .iter()
                    .map(|&a| self.get_expression_width(a))
                    .max()
                    .unwrap_or(self.config.default_bit_width);

                // Add log2(n) bits for addition of n terms
                let extra_bits = (args.len() as f64).log2().ceil() as u32;
                (max_width + extra_bits).min(self.config.max_bit_width)
            }

            TermKind::Sub(a, b) => {
                let a_w = self.get_expression_width(*a);
                let b_w = self.get_expression_width(*b);
                a_w.max(b_w)
            }

            _ => self.config.default_bit_width,
        }
    }
}

/// Stateless wrapper for tactic trait
#[derive(Debug)]
pub struct StatelessNla2BvTactic {
    #[allow(dead_code)]
    config: Nla2BvConfig,
}

impl StatelessNla2BvTactic {
    /// Create a new stateless wrapper
    pub fn new() -> Self {
        Self {
            config: Nla2BvConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: Nla2BvConfig) -> Self {
        Self { config }
    }
}

impl Default for StatelessNla2BvTactic {
    fn default() -> Self {
        Self::new()
    }
}

impl Tactic for StatelessNla2BvTactic {
    fn name(&self) -> &str {
        "nla2bv"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Need a TermManager - for now return NotApplicable
        // In practice, the solver would provide the manager
        Ok(TacticResult::NotApplicable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_nla2bv_config_default() {
        let config = Nla2BvConfig::default();
        assert_eq!(config.default_bit_width, 32);
        assert_eq!(config.max_bit_width, 64);
        assert!(config.use_signed);
        assert!(config.handle_div_zero);
    }

    #[test]
    fn test_var_bounds_basic() {
        let mut bounds = VarBounds::new();
        assert!(!bounds.is_bounded());

        bounds.update_lower(BigInt::from(0));
        assert!(!bounds.is_bounded());

        bounds.update_upper(BigInt::from(100));
        assert!(bounds.is_bounded());

        assert_eq!(bounds.range_size(), Some(BigInt::from(101)));
        assert_eq!(bounds.required_bit_width(), Some(7)); // ceil(log2(101)) = 7
    }

    #[test]
    fn test_var_bounds_negative() {
        let mut bounds = VarBounds::new();
        bounds.update_lower(BigInt::from(-10));
        bounds.update_upper(BigInt::from(10));

        assert!(bounds.is_bounded());
        assert_eq!(bounds.range_size(), Some(BigInt::from(21)));
    }

    #[test]
    fn test_nla2bv_extract_bounds() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // x <= 100
        let x = manager.mk_var("x", int_sort);
        let x_name = manager.intern_str("x"); // Get the name now before creating tactic
        let c100 = manager.mk_int(BigInt::from(100));
        let le = manager.mk_le(x, c100);

        // x >= 0
        let c0 = manager.mk_int(BigInt::from(0));
        let ge = manager.mk_ge(x, c0);

        let goal = Goal::new(vec![le, ge]);
        let mut tactic = Nla2BvTactic::new(&mut manager);
        tactic.extract_bounds(&goal.assertions);

        let bounds = tactic.var_bounds.get(&x_name);
        assert!(bounds.is_some());
        let bounds = bounds.expect("bounds should exist");
        assert!(bounds.is_bounded());
    }

    #[test]
    fn test_nla2bv_tactic_not_applicable() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // x * y = z without bounds - should be not applicable
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);
        let mul = manager.mk_mul([x, y]);
        let eq = manager.mk_eq(mul, z);

        let goal = Goal::new(vec![eq]);
        let mut tactic = Nla2BvTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal);

        assert!(matches!(result, Ok(TacticResult::NotApplicable)));
    }

    #[test]
    fn test_nla2bv_with_bounds() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // 0 <= x <= 10
        let x = manager.mk_var("x", int_sort);
        let c0 = manager.mk_int(BigInt::from(0));
        let c10 = manager.mk_int(BigInt::from(10));
        let ge = manager.mk_ge(x, c0);
        let le = manager.mk_le(x, c10);

        // x * x = y with 0 <= y <= 100
        let y = manager.mk_var("y", int_sort);
        let mul = manager.mk_mul([x, x]);
        let eq = manager.mk_eq(mul, y);
        let c100 = manager.mk_int(BigInt::from(100));
        let y_ge = manager.mk_ge(y, c0);
        let y_le = manager.mk_le(y, c100);

        let goal = Goal::new(vec![ge, le, eq, y_ge, y_le]);
        let mut tactic = Nla2BvTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal);

        // Should produce subgoals (converted to BV)
        assert!(matches!(result, Ok(TacticResult::SubGoals(_))));
    }

    #[test]
    fn test_stateless_wrapper() {
        let tactic = StatelessNla2BvTactic::new();
        assert_eq!(tactic.name(), "nla2bv");

        let goal = Goal::empty();
        let result = tactic.apply(&goal);
        assert!(matches!(result, Ok(TacticResult::NotApplicable)));
    }
}
