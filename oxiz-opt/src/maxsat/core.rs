//! Core data structures for MaxSAT solving.
//!
//! This module defines the Weight type, SoftId, SoftClause, and Core structures
//! used by the MaxSAT solver algorithms.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use oxiz_sat::{Lit, Solver as SatSolver, SolverResult};
use smallvec::SmallVec;

/// Weight of a soft clause (can be integer, rational, or infinite).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Weight {
    /// Integer weight
    Int(BigInt),
    /// Rational weight
    Rational(BigRational),
    /// Infinite weight (effectively hard)
    Infinite,
}

impl Weight {
    /// Create a unit weight (1)
    pub fn one() -> Self {
        Weight::Int(BigInt::one())
    }

    /// Create a zero weight
    pub fn zero() -> Self {
        Weight::Int(BigInt::zero())
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        match self {
            Weight::Int(n) => n.is_zero(),
            Weight::Rational(r) => r.is_zero(),
            Weight::Infinite => false,
        }
    }

    /// Check if this is infinite
    pub fn is_infinite(&self) -> bool {
        matches!(self, Weight::Infinite)
    }

    /// Add two weights
    pub fn add(&self, other: &Weight) -> Weight {
        match (self, other) {
            (Weight::Infinite, _) | (_, Weight::Infinite) => Weight::Infinite,
            (Weight::Int(a), Weight::Int(b)) => Weight::Int(a + b),
            (Weight::Rational(a), Weight::Rational(b)) => Weight::Rational(a + b),
            (Weight::Int(a), Weight::Rational(b)) | (Weight::Rational(b), Weight::Int(a)) => {
                let a_rat = BigRational::from(a.clone());
                Weight::Rational(a_rat + b)
            }
        }
    }

    /// Subtract two weights (saturating at zero)
    pub fn sub(&self, other: &Weight) -> Weight {
        match (self, other) {
            (Weight::Infinite, Weight::Infinite) => Weight::zero(),
            (Weight::Infinite, _) => Weight::Infinite,
            (_, Weight::Infinite) => Weight::zero(),
            (Weight::Int(a), Weight::Int(b)) => {
                if a >= b {
                    Weight::Int(a - b)
                } else {
                    Weight::zero()
                }
            }
            (Weight::Rational(a), Weight::Rational(b)) => {
                if a >= b {
                    Weight::Rational(a - b)
                } else {
                    Weight::zero()
                }
            }
            (Weight::Int(a), Weight::Rational(b)) => {
                let a_rat = BigRational::from(a.clone());
                if a_rat >= *b {
                    Weight::Rational(a_rat - b)
                } else {
                    Weight::zero()
                }
            }
            (Weight::Rational(a), Weight::Int(b)) => {
                let b_rat = BigRational::from(b.clone());
                if *a >= b_rat {
                    Weight::Rational(a - b_rat)
                } else {
                    Weight::zero()
                }
            }
        }
    }

    /// Get the minimum of two weights (non-consuming version)
    pub fn min_weight(&self, other: &Weight) -> Weight {
        if self <= other {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Get the maximum of two weights (non-consuming version)
    pub fn max_weight(&self, other: &Weight) -> Weight {
        if self >= other {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Multiply weight by a scalar
    pub fn mul_scalar(&self, scalar: i64) -> Weight {
        if scalar == 0 {
            return Weight::zero();
        }
        if scalar < 0 {
            // Multiplying by negative doesn't make sense for weights
            return Weight::zero();
        }

        match self {
            Weight::Infinite => Weight::Infinite,
            Weight::Int(n) => Weight::Int(n * BigInt::from(scalar)),
            Weight::Rational(r) => Weight::Rational(r * BigInt::from(scalar)),
        }
    }

    /// Divide weight by a scalar (returns None if scalar is 0)
    pub fn div_scalar(&self, scalar: i64) -> Option<Weight> {
        if scalar == 0 {
            return None;
        }
        if scalar < 0 {
            return None;
        }

        match self {
            Weight::Infinite => Some(Weight::Infinite),
            Weight::Int(n) => {
                // Convert to rational for division
                let result = BigRational::from(n.clone()) / BigInt::from(scalar);
                Some(Weight::Rational(result))
            }
            Weight::Rational(r) => Some(Weight::Rational(r / BigInt::from(scalar))),
        }
    }

    /// Check if this weight is one
    pub fn is_one(&self) -> bool {
        match self {
            Weight::Int(n) => n.is_one(),
            Weight::Rational(r) => r.is_one(),
            Weight::Infinite => false,
        }
    }

    /// Convert to integer if possible
    pub fn to_int(&self) -> Option<BigInt> {
        match self {
            Weight::Int(n) => Some(n.clone()),
            Weight::Rational(r) if r.is_integer() => Some(r.numer().clone()),
            _ => None,
        }
    }

    /// Convert to rational
    pub fn to_rational(&self) -> Option<BigRational> {
        match self {
            Weight::Int(n) => Some(BigRational::from(n.clone())),
            Weight::Rational(r) => Some(r.clone()),
            Weight::Infinite => None,
        }
    }

    /// Try to convert to i64 if the value fits
    pub fn to_i64(&self) -> Option<i64> {
        self.to_int()?.to_i64()
    }

    /// Create an infinite weight
    pub fn infinite() -> Self {
        Weight::Infinite
    }

    /// Get the absolute value (weights are always non-negative)
    pub fn abs(&self) -> Weight {
        self.clone()
    }
}

impl Default for Weight {
    fn default() -> Self {
        Weight::one()
    }
}

impl From<i64> for Weight {
    fn from(n: i64) -> Self {
        Weight::Int(BigInt::from(n))
    }
}

impl From<i32> for Weight {
    fn from(n: i32) -> Self {
        Weight::Int(BigInt::from(n))
    }
}

impl From<u64> for Weight {
    fn from(n: u64) -> Self {
        Weight::Int(BigInt::from(n))
    }
}

impl From<u32> for Weight {
    fn from(n: u32) -> Self {
        Weight::Int(BigInt::from(n))
    }
}

impl From<usize> for Weight {
    fn from(n: usize) -> Self {
        Weight::Int(BigInt::from(n))
    }
}

impl From<BigInt> for Weight {
    fn from(n: BigInt) -> Self {
        Weight::Int(n)
    }
}

impl From<BigRational> for Weight {
    fn from(r: BigRational) -> Self {
        Weight::Rational(r)
    }
}

impl From<(i64, i64)> for Weight {
    /// Create a rational weight from a (numerator, denominator) tuple.
    ///
    /// # Example
    /// ```
    /// use oxiz_opt::Weight;
    /// let w: Weight = (3, 2).into(); // Creates 3/2
    /// ```
    fn from((num, denom): (i64, i64)) -> Self {
        Weight::Rational(BigRational::new(BigInt::from(num), BigInt::from(denom)))
    }
}

impl std::ops::Add for Weight {
    type Output = Weight;

    fn add(self, other: Weight) -> Weight {
        Weight::add(&self, &other)
    }
}

impl std::ops::Add for &Weight {
    type Output = Weight;

    fn add(self, other: &Weight) -> Weight {
        Weight::add(self, other)
    }
}

impl std::ops::Sub for Weight {
    type Output = Weight;

    fn sub(self, other: Weight) -> Weight {
        Weight::sub(&self, &other)
    }
}

impl std::ops::Sub for &Weight {
    type Output = Weight;

    fn sub(self, other: &Weight) -> Weight {
        Weight::sub(self, other)
    }
}

impl std::ops::AddAssign for Weight {
    fn add_assign(&mut self, other: Weight) {
        *self = Weight::add(self, &other);
    }
}

impl std::ops::SubAssign for Weight {
    fn sub_assign(&mut self, other: Weight) {
        *self = Weight::sub(self, &other);
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Weight::Int(n) => write!(f, "{}", n),
            Weight::Rational(r) => write!(f, "{}", r),
            Weight::Infinite => write!(f, "∞"),
        }
    }
}

/// Unique identifier for a soft constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoftId(pub u32);

impl SoftId {
    /// Create a new soft ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Convert to usize for indexing
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for SoftId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for SoftId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<SoftId> for u32 {
    fn from(id: SoftId) -> Self {
        id.0
    }
}

impl From<SoftId> for usize {
    fn from(id: SoftId) -> Self {
        id.0 as usize
    }
}

/// A soft constraint with weight
#[derive(Debug, Clone)]
pub struct SoftClause {
    /// Unique identifier
    pub id: SoftId,
    /// The clause literals
    pub lits: SmallVec<[Lit; 4]>,
    /// Weight of this soft constraint
    pub weight: Weight,
    /// Relaxation variable (if added)
    pub relax_var: Option<Lit>,
    /// Current assignment value
    value: oxiz_sat::LBool,
}

impl SoftClause {
    /// Create a new soft clause
    pub fn new(id: SoftId, lits: impl IntoIterator<Item = Lit>, weight: Weight) -> Self {
        Self {
            id,
            lits: lits.into_iter().collect(),
            weight,
            relax_var: None,
            value: oxiz_sat::LBool::Undef,
        }
    }

    /// Create a unit soft clause
    pub fn unit(id: SoftId, lit: Lit, weight: Weight) -> Self {
        Self::new(id, [lit], weight)
    }

    /// Check if this soft clause is satisfied
    pub fn is_satisfied(&self) -> bool {
        self.value == oxiz_sat::LBool::True
    }

    /// Set the value from a model
    pub fn set_value(&mut self, satisfied: bool) {
        self.value = if satisfied {
            oxiz_sat::LBool::True
        } else {
            oxiz_sat::LBool::False
        };
    }
}

/// Core found during MaxSAT solving
#[derive(Debug, Clone)]
pub struct Core {
    /// Soft clause IDs in this core
    pub soft_ids: SmallVec<[SoftId; 8]>,
    /// Minimum weight of soft clauses in this core
    pub min_weight: Weight,
}

impl Core {
    /// Create a new core
    pub fn new(soft_ids: impl IntoIterator<Item = SoftId>, min_weight: Weight) -> Self {
        Self {
            soft_ids: soft_ids.into_iter().collect(),
            min_weight,
        }
    }

    /// Get the size of this core
    pub fn size(&self) -> usize {
        self.soft_ids.len()
    }

    /// Minimize this core by removing unnecessary soft clauses
    ///
    /// Uses a deletion-based approach: try removing each soft clause
    /// from the core and check if the remaining clauses are still unsatisfiable.
    /// If they are, the clause was unnecessary and can be removed.
    ///
    /// Reference: "On Minimal Correction Subsets" (Liffiton & Sakallah, 2008)
    pub fn minimize(
        &mut self,
        _solver: &mut SatSolver,
        soft_clauses: &[SoftClause],
        hard_clauses: &[SmallVec<[Lit; 4]>],
    ) -> usize {
        if self.soft_ids.len() <= 1 {
            return 0; // Can't minimize a unit or empty core
        }

        let original_size = self.soft_ids.len();
        let mut minimized_ids: SmallVec<[SoftId; 8]> = SmallVec::new();

        // Try removing each soft clause from the core
        for (idx, &soft_id) in self.soft_ids.iter().enumerate() {
            // Build a temporary core without this clause
            let mut test_core: SmallVec<[SoftId; 8]> = SmallVec::new();
            test_core.extend(minimized_ids.iter().copied());
            test_core.extend(self.soft_ids.iter().skip(idx + 1).copied());

            // Check if the core without this clause is still unsatisfiable
            if test_core.is_empty() {
                // We need at least this clause
                minimized_ids.push(soft_id);
                continue;
            }

            // Create a temporary solver with hard clauses and test core
            let mut temp_solver = SatSolver::new();

            // Add all hard clauses
            for hard_clause in hard_clauses {
                temp_solver.add_clause(hard_clause.iter().copied());
            }

            // Add soft clauses from test core
            for &test_id in &test_core {
                if let Some(soft_clause) = soft_clauses.iter().find(|c| c.id == test_id) {
                    temp_solver.add_clause(soft_clause.lits.iter().copied());
                }
            }

            // Check satisfiability
            match temp_solver.solve() {
                SolverResult::Sat => {
                    // Core without this clause is satisfiable, so we need this clause
                    minimized_ids.push(soft_id);
                }
                SolverResult::Unsat => {
                    // Core is still unsatisfiable without this clause, so remove it
                    // (don't add to minimized_ids)
                }
                SolverResult::Unknown => {
                    // Conservative: keep the clause if we can't determine
                    minimized_ids.push(soft_id);
                }
            }
        }

        let removed = original_size - minimized_ids.len();
        self.soft_ids = minimized_ids;
        removed
    }

    /// Strengthen core assumptions by finding a smaller set of assumptions
    /// that still produce an unsatisfiable core.
    ///
    /// This is similar to minimization but works on the assumption level.
    /// Uses binary search to find a minimal set of assumptions.
    ///
    /// Reference: "Improving MCS Enumeration" (Marques-Silva et al., 2013)
    #[allow(dead_code)]
    pub fn strengthen_assumptions(
        assumptions: &[Lit],
        solver: &mut SatSolver,
    ) -> SmallVec<[Lit; 8]> {
        if assumptions.len() <= 1 {
            return assumptions.iter().copied().collect();
        }

        let mut strengthened: SmallVec<[Lit; 8]> = SmallVec::new();
        let mut remaining: Vec<Lit> = assumptions.to_vec();

        // Greedy approach: try removing each assumption
        while !remaining.is_empty() {
            let mut found_removable = false;

            for i in 0..remaining.len() {
                // Test without this assumption
                let mut test_assumptions: SmallVec<[Lit; 8]> = SmallVec::new();
                test_assumptions.extend(strengthened.iter().copied());
                test_assumptions.extend(
                    remaining
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &lit)| if idx != i { Some(lit) } else { None }),
                );

                if test_assumptions.is_empty() {
                    // Can't remove the last one
                    strengthened.push(remaining[i]);
                    remaining.remove(i);
                    found_removable = true;
                    break;
                }

                // Check if still unsatisfiable without this assumption
                let (result, _) = solver.solve_with_assumptions(&test_assumptions);
                match result {
                    SolverResult::Unsat => {
                        // Still unsat, so this assumption is not needed
                        remaining.remove(i);
                        found_removable = true;
                        break;
                    }
                    _ => {
                        // Sat or unknown, assumption is needed
                    }
                }
            }

            if !found_removable {
                // All remaining assumptions are necessary
                strengthened.extend(remaining.drain(..));
                break;
            }
        }

        strengthened
    }
}
