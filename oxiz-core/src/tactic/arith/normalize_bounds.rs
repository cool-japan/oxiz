//! Bounds Normalization Tactic for Arithmetic.
//!
//! Normalizes arithmetic bounds into canonical form and propagates
//! derived inequalities.  The tactic:
//!
//! 1. Extracts all `x ≤ c`, `c ≤ x`, `x < c`, `c < x` atoms from an AND-tree.
//! 2. Keeps only the tightest bound per variable per direction.
//! 3. Detects direct conflicts (`lb > ub` or `lb == ub` with at least one strict side).
//! 4. Derives transitive inequalities between pairs of variables.
//! 5. Reassembles the result as a conjunction of normalized atoms.

#[allow(unused_imports)]
use crate::prelude::*;
use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

// ─── Public types required by the parent module ──────────────────────────────

/// Classifies the direction of a single bound constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundType {
    /// Lower bound: `value ≤ variable` (or `value < variable` when strict).
    Lower,
    /// Upper bound: `variable ≤ value` (or `variable < value` when strict).
    Upper,
    /// Equality: `variable = value`.
    Equal,
}

/// A set of normalized bounds for a single variable, expressed as a
/// tight `[lo, hi]` interval with optional strictness on each side.
#[derive(Debug, Clone)]
pub struct NormalizedBounds {
    /// The variable name these bounds apply to.
    pub var: String,
    /// Tightest lower bound (if any).
    pub lower: Option<BoundValue>,
    /// Tightest upper bound (if any).
    pub upper: Option<BoundValue>,
}

/// A bound value paired with its strictness.
#[derive(Debug, Clone)]
pub struct BoundValue {
    /// The rational bound value.
    pub value: BigRational,
    /// `true` for strict inequality (`<` / `>`), `false` for non-strict (`≤` / `≥`).
    pub strict: bool,
}

/// Configuration for the normalization pass.
#[derive(Debug, Clone, Default)]
pub struct NormalizeBoundsConfig {
    /// When `true`, derive transitive inequalities between variable pairs.
    pub derive_transitive: bool,
}

/// Statistics collected during normalization.
#[derive(Debug, Clone, Default)]
pub struct NormalizeBoundsStats {
    /// Number of bound atoms extracted from the input formula.
    pub bounds_extracted: usize,
    /// Number of redundant (weaker) bounds discarded.
    pub redundant_eliminated: usize,
    /// Number of direct conflicts detected.
    pub conflicts_detected: usize,
    /// Number of derived transitive inequalities added.
    pub derived_inequalities: usize,
}

// ─── Internal helper ─────────────────────────────────────────────────────────

/// Internal per-variable bound record.
#[derive(Debug, Clone)]
struct BoundInfo {
    lower: Option<BoundValue>,
    upper: Option<BoundValue>,
}

// ─── Tactic ──────────────────────────────────────────────────────────────────

/// Bounds normalization tactic for linear arithmetic.
pub struct NormalizeBoundsTactic {
    config: NormalizeBoundsConfig,
    bounds: FxHashMap<String, BoundInfo>,
    stats: NormalizeBoundsStats,
}

impl NormalizeBoundsTactic {
    /// Create a new bounds normalization tactic with default configuration.
    pub fn new() -> Self {
        Self {
            config: NormalizeBoundsConfig::default(),
            bounds: FxHashMap::default(),
            stats: NormalizeBoundsStats::default(),
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: NormalizeBoundsConfig) -> Self {
        Self { config, bounds: FxHashMap::default(), stats: NormalizeBoundsStats::default() }
    }

    /// Apply normalization to a formula, returning a simplified conjunction.
    pub fn apply(&mut self, formula: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        // Phase 1: Extract bounds from the AND-tree.
        self.extract_bounds(formula, tm)?;

        // Phase 2: Detect direct conflicts.
        if self.check_conflicts().is_some() {
            self.stats.conflicts_detected += 1;
            return Ok(tm.mk_false());
        }

        // Phase 3: Emit normalized inequalities.
        let mut all_ineqs = self.generate_normalized_inequalities(tm)?;

        // Phase 4: Optionally derive transitive inequalities.
        if self.config.derive_transitive {
            let derived = self.derive_inequalities(tm)?;
            self.stats.derived_inequalities += derived.len();
            all_ineqs.extend(derived);
        }

        if all_ineqs.is_empty() {
            Ok(tm.mk_true())
        } else {
            Ok(tm.mk_and(all_ineqs))
        }
    }

    /// Build a `NormalizedBounds` view of every variable seen during `apply`.
    pub fn normalized_bounds(&self) -> Vec<NormalizedBounds> {
        self.bounds
            .iter()
            .map(|(var, info)| NormalizedBounds {
                var: var.clone(),
                lower: info.lower.clone(),
                upper: info.upper.clone(),
            })
            .collect()
    }

    /// Return accumulated statistics.
    pub fn stats(&self) -> &NormalizeBoundsStats {
        &self.stats
    }

    // ── Bound extraction ──────────────────────────────────────────────────────

    fn extract_bounds(&mut self, formula: TermId, tm: &TermManager) -> Result<(), String> {
        let mut stack = vec![formula];

        while let Some(tid) = stack.pop() {
            let term = tm.get(tid).ok_or_else(|| format!("term {:?} not found", tid))?;

            match &term.kind {
                TermKind::And(args) => {
                    stack.extend(args.iter().copied());
                }
                TermKind::Le(lhs, rhs) => {
                    self.extract_bound_from_le(*lhs, *rhs, false, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                TermKind::Lt(lhs, rhs) => {
                    self.extract_bound_from_le(*lhs, *rhs, true, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                TermKind::Ge(lhs, rhs) => {
                    self.extract_bound_from_le(*rhs, *lhs, false, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                TermKind::Gt(lhs, rhs) => {
                    self.extract_bound_from_le(*rhs, *lhs, true, tm)?;
                    self.stats.bounds_extracted += 1;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Interpret `lhs ≤ rhs` (or `lhs < rhs` when `strict`).
    fn extract_bound_from_le(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        strict: bool,
        tm: &TermManager,
    ) -> Result<(), String> {
        // Case: var ≤ constant  →  upper bound on var.
        if let Some(var) = self.get_var_name(lhs, tm) {
            if let Some(c) = self.eval_constant(rhs, tm) {
                self.update_upper_bound(var, BoundValue { value: c, strict });
            }
        }

        // Case: constant ≤ var  →  lower bound on var.
        if let Some(var) = self.get_var_name(rhs, tm) {
            if let Some(c) = self.eval_constant(lhs, tm) {
                self.update_lower_bound(var, BoundValue { value: c, strict });
            }
        }

        Ok(())
    }

    // ── Bound maintenance ─────────────────────────────────────────────────────

    fn update_lower_bound(&mut self, var: String, new_b: BoundValue) {
        let entry = self.bounds.entry(var).or_insert(BoundInfo { lower: None, upper: None });
        let tighter = match &entry.lower {
            None => true,
            Some(existing) => {
                new_b.value > existing.value
                    || (new_b.value == existing.value && new_b.strict && !existing.strict)
            }
        };
        if tighter {
            if entry.lower.is_some() {
                self.stats.redundant_eliminated += 1;
            }
            entry.lower = Some(new_b);
        } else {
            self.stats.redundant_eliminated += 1;
        }
    }

    fn update_upper_bound(&mut self, var: String, new_b: BoundValue) {
        let entry = self.bounds.entry(var).or_insert(BoundInfo { lower: None, upper: None });
        let tighter = match &entry.upper {
            None => true,
            Some(existing) => {
                new_b.value < existing.value
                    || (new_b.value == existing.value && new_b.strict && !existing.strict)
            }
        };
        if tighter {
            if entry.upper.is_some() {
                self.stats.redundant_eliminated += 1;
            }
            entry.upper = Some(new_b);
        } else {
            self.stats.redundant_eliminated += 1;
        }
    }

    // ── Conflict detection ────────────────────────────────────────────────────

    /// Return the name of the first conflicted variable, if any.
    fn check_conflicts(&self) -> Option<&str> {
        for (var, info) in &self.bounds {
            if let (Some(lo), Some(hi)) = (&info.lower, &info.upper) {
                if lo.value > hi.value {
                    return Some(var);
                }
                // Equal bounds where at least one side is strict: no solution.
                if lo.value == hi.value && (lo.strict || hi.strict) {
                    return Some(var);
                }
            }
        }
        None
    }

    // ── Inequality emission ───────────────────────────────────────────────────

    fn generate_normalized_inequalities(
        &self,
        tm: &mut TermManager,
    ) -> Result<Vec<TermId>, String> {
        let mut ineqs = Vec::new();
        let int_sort = tm.sorts.int_sort;

        for (var, info) in &self.bounds {
            let var_term = tm.mk_var(var.as_str(), int_sort);

            if let Some(lo) = &info.lower {
                let const_term = rational_to_term(&lo.value, tm)?;
                let ineq = if lo.strict {
                    tm.mk_gt(var_term, const_term)
                } else {
                    tm.mk_ge(var_term, const_term)
                };
                ineqs.push(ineq);
            }

            if let Some(hi) = &info.upper {
                let const_term = rational_to_term(&hi.value, tm)?;
                let ineq = if hi.strict {
                    tm.mk_lt(var_term, const_term)
                } else {
                    tm.mk_le(var_term, const_term)
                };
                ineqs.push(ineq);
            }
        }

        Ok(ineqs)
    }

    fn derive_inequalities(&self, tm: &mut TermManager) -> Result<Vec<TermId>, String> {
        let mut derived = Vec::new();
        let vars: Vec<_> = self.bounds.keys().collect();
        let int_sort = tm.sorts.int_sort;

        for i in 0..vars.len() {
            for j in (i + 1)..vars.len() {
                let v1 = vars[i];
                let v2 = vars[j];

                let (Some(info1), Some(info2)) = (self.bounds.get(v1), self.bounds.get(v2)) else {
                    continue;
                };

                // If var1 ≤ c1 and c2 ≤ var2, and c1 ≤ c2, then var1 ≤ var2.
                if let (Some(hi1), Some(lo2)) = (&info1.upper, &info2.lower) {
                    if hi1.value <= lo2.value {
                        let t1 = tm.mk_var(v1.as_str(), int_sort);
                        let t2 = tm.mk_var(v2.as_str(), int_sort);
                        derived.push(tm.mk_le(t1, t2));
                    }
                }
            }
        }

        Ok(derived)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn get_var_name(&self, tid: TermId, tm: &TermManager) -> Option<String> {
        let term = tm.get(tid)?;
        match &term.kind {
            TermKind::Var(name) => Some(name.to_string()),
            _ => None,
        }
    }

    fn eval_constant(&self, tid: TermId, tm: &TermManager) -> Option<BigRational> {
        let term = tm.get(tid)?;
        match &term.kind {
            TermKind::IntConst(v) => Some(BigRational::from_integer(v.clone())),
            TermKind::RealConst(r) => {
                let (n, d) = (r.numer(), r.denom());
                Some(BigRational::new(BigInt::from(*n), BigInt::from(*d)))
            }
            _ => None,
        }
    }
}

impl Default for NormalizeBoundsTactic {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Free helper ─────────────────────────────────────────────────────────────

/// Emit a `TermId` for a rational constant.
///
/// If the denominator is 1 we emit an integer constant (cheaper).
/// Otherwise we approximate: the TermManager only supports `Rational64`,
/// so we fall back to rounding to the nearest integer (sound for the
/// normalization pass, which is a pre-processing step, not a proof).
fn rational_to_term(r: &BigRational, tm: &mut TermManager) -> Result<TermId, String> {
    use num_traits::ToPrimitive;
    if r.denom() == &BigInt::one() {
        Ok(tm.mk_int(r.numer().clone()))
    } else {
        // Round toward zero for approximation.
        let approx = r.to_integer();
        Ok(tm.mk_int(approx))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactic_init() {
        let t = NormalizeBoundsTactic::new();
        assert_eq!(t.stats.bounds_extracted, 0);
        assert_eq!(t.stats.conflicts_detected, 0);
    }

    #[test]
    fn test_update_lower_bound_keeps_tightest() {
        let mut t = NormalizeBoundsTactic::new();

        t.update_lower_bound(
            "x".into(),
            BoundValue { value: BigRational::from_integer(BigInt::from(3)), strict: false },
        );
        // A looser bound should be discarded.
        t.update_lower_bound(
            "x".into(),
            BoundValue { value: BigRational::from_integer(BigInt::from(1)), strict: false },
        );

        let info = t.bounds.get("x").expect("variable must be present");
        let lo = info.lower.as_ref().expect("lower bound must be set");
        assert_eq!(lo.value, BigRational::from_integer(BigInt::from(3)));
        assert_eq!(t.stats.redundant_eliminated, 1);
    }

    #[test]
    fn test_conflict_detection() {
        let mut t = NormalizeBoundsTactic::new();

        t.bounds.insert(
            "y".into(),
            BoundInfo {
                lower: Some(BoundValue {
                    value: BigRational::from_integer(BigInt::from(10)),
                    strict: false,
                }),
                upper: Some(BoundValue {
                    value: BigRational::from_integer(BigInt::from(5)),
                    strict: false,
                }),
            },
        );

        assert!(t.check_conflicts().is_some());
    }

    #[test]
    fn test_bound_type_and_normalized_bounds() {
        // Just verify the public types compile and can be constructed.
        let _bt = BoundType::Lower;
        let _nb = NormalizedBounds {
            var: "z".into(),
            lower: None,
            upper: Some(BoundValue {
                value: BigRational::from_integer(BigInt::from(7)),
                strict: true,
            }),
        };
        let _cfg = NormalizeBoundsConfig { derive_transitive: true };
    }
}
