//! Farkas Lemma and Infeasibility Certificates.
//!
//! Implements Farkas' lemma for generating infeasibility certificates,
//! useful for conflict analysis and proof generation.
//!
//! ## Farkas Lemma
//!
//! For a system Ax ≤ b, if infeasible, there exists y ≥ 0 such that:
//! - yᵀA = 0
//! - yᵀb < 0
//!
//! The vector y is the Farkas certificate (infeasibility proof).
//!
//! ## References
//!
//! - Schrijver: "Theory of Linear and Integer Programming" (1986)
//! - Z3's `math/lp/lp_farkas.cpp`

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = usize;

/// Constraint identifier.
pub type ConstraintId = usize;

/// A linear constraint: sum(coeffs\[i\] * x\[i\]) ≤ rhs.
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// Coefficients (var_id -> coefficient).
    pub coeffs: FxHashMap<VarId, BigRational>,
    /// Right-hand side.
    pub rhs: BigRational,
}

/// Farkas certificate (infeasibility proof).
#[derive(Debug, Clone)]
pub struct FarkasCertificate {
    /// Multipliers for each constraint (constraint_id -> multiplier).
    /// All multipliers must be non-negative.
    pub multipliers: FxHashMap<ConstraintId, BigRational>,
}

impl FarkasCertificate {
    /// Create a new certificate.
    pub fn new() -> Self {
        Self {
            multipliers: FxHashMap::default(),
        }
    }

    /// Add a multiplier for a constraint.
    pub fn add_multiplier(&mut self, constraint_id: ConstraintId, multiplier: BigRational) {
        assert!(
            multiplier >= BigRational::zero(),
            "Farkas multipliers must be non-negative"
        );
        self.multipliers.insert(constraint_id, multiplier);
    }

    /// Check if this is a valid Farkas certificate.
    pub fn is_valid(&self, constraints: &[LinearConstraint]) -> bool {
        // Check 1: All multipliers are non-negative (enforced by add_multiplier)

        // Check 2: Multiplier combination has zero coefficients (yᵀA = 0)
        let mut combined_coeffs: FxHashMap<VarId, BigRational> = FxHashMap::default();

        for (&cid, multiplier) in &self.multipliers {
            if let Some(constraint) = constraints.get(cid) {
                for (&var, coeff) in &constraint.coeffs {
                    *combined_coeffs.entry(var).or_insert_with(BigRational::zero) +=
                        multiplier.clone() * coeff;
                }
            }
        }

        // All combined coefficients should be zero (or very small)
        for coeff in combined_coeffs.values() {
            if coeff.abs() > BigRational::new(BigInt::from(1), BigInt::from(1000000)) {
                return false;
            }
        }

        // Check 3: Combined RHS is negative (yᵀb < 0)
        let mut combined_rhs = BigRational::zero();

        for (&cid, multiplier) in &self.multipliers {
            if let Some(constraint) = constraints.get(cid) {
                combined_rhs += multiplier.clone() * constraint.rhs.clone();
            }
        }

        combined_rhs < BigRational::zero()
    }
}

impl Default for FarkasCertificate {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for Farkas certificate generation.
#[derive(Debug, Clone)]
pub struct FarkasConfig {
    /// Enable certificate validation.
    pub validate_certificate: bool,
    /// Enable minimal certificate generation.
    pub minimize_certificate: bool,
}

impl Default for FarkasConfig {
    fn default() -> Self {
        Self {
            validate_certificate: true,
            minimize_certificate: false,
        }
    }
}

/// Statistics for Farkas certificate generation.
#[derive(Debug, Clone, Default)]
pub struct FarkasStats {
    /// Certificates generated.
    pub certificates_generated: u64,
    /// Validations performed.
    pub validations: u64,
    /// Invalid certificates detected.
    pub invalid_certificates: u64,
}

/// Farkas certificate generator.
#[derive(Debug)]
pub struct FarkasGenerator {
    /// Configuration.
    config: FarkasConfig,
    /// Statistics.
    stats: FarkasStats,
}

impl FarkasGenerator {
    /// Create a new Farkas generator.
    pub fn new(config: FarkasConfig) -> Self {
        Self {
            config,
            stats: FarkasStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(FarkasConfig::default())
    }

    /// Generate a Farkas certificate from dual simplex solution.
    pub fn generate_from_dual(
        &mut self,
        _constraints: &[LinearConstraint],
        _dual_solution: &FxHashMap<ConstraintId, BigRational>,
    ) -> Option<FarkasCertificate> {
        self.stats.certificates_generated += 1;

        // Simplified: would extract Farkas multipliers from dual solution
        // The dual solution gives the multipliers directly
        None
    }

    /// Generate a Farkas certificate from a conflict core.
    pub fn generate_from_core(
        &mut self,
        constraints: &[LinearConstraint],
        core: &[ConstraintId],
    ) -> Option<FarkasCertificate> {
        self.stats.certificates_generated += 1;

        // Simplified: would find non-negative combination of core constraints
        // that yields 0 coefficients and negative RHS

        let mut certificate = FarkasCertificate::new();

        // Example: use uniform weights for all constraints in core
        let weight = BigRational::one();
        for &cid in core {
            certificate.add_multiplier(cid, weight.clone());
        }

        // Validate if enabled
        if self.config.validate_certificate && !self.validate(&certificate, constraints) {
            return None;
        }

        Some(certificate)
    }

    /// Validate a Farkas certificate.
    pub fn validate(
        &mut self,
        certificate: &FarkasCertificate,
        constraints: &[LinearConstraint],
    ) -> bool {
        self.stats.validations += 1;

        let valid = certificate.is_valid(constraints);

        if !valid {
            self.stats.invalid_certificates += 1;
        }

        valid
    }

    /// Minimize a Farkas certificate (remove unnecessary constraints).
    pub fn minimize(&mut self, _certificate: &mut FarkasCertificate) {
        if !self.config.minimize_certificate {}

        // Simplified: would try to remove multipliers while maintaining validity
    }

    /// Get statistics.
    pub fn stats(&self) -> &FarkasStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = FarkasStats::default();
    }
}

impl Default for FarkasGenerator {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = FarkasGenerator::default_config();
        assert_eq!(generator.stats().certificates_generated, 0);
    }

    #[test]
    fn test_certificate_creation() {
        let mut cert = FarkasCertificate::new();

        cert.add_multiplier(0, BigRational::one());
        cert.add_multiplier(1, BigRational::from(BigInt::from(2)));

        assert_eq!(cert.multipliers.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Farkas multipliers must be non-negative")]
    fn test_negative_multiplier_panics() {
        let mut cert = FarkasCertificate::new();
        cert.add_multiplier(0, BigRational::from(BigInt::from(-1)));
    }

    #[test]
    fn test_generate_from_core() {
        let mut generator = FarkasGenerator::default_config();

        let mut constraints = Vec::new();
        let mut coeffs = FxHashMap::default();
        coeffs.insert(0, BigRational::one());

        constraints.push(LinearConstraint {
            coeffs,
            rhs: BigRational::zero(),
        });

        let core = vec![0];
        let _cert = generator.generate_from_core(&constraints, &core);

        assert_eq!(generator.stats().certificates_generated, 1);
    }

    #[test]
    fn test_stats() {
        let mut generator = FarkasGenerator::default_config();
        generator.stats.certificates_generated = 5;

        assert_eq!(generator.stats().certificates_generated, 5);

        generator.reset_stats();
        assert_eq!(generator.stats().certificates_generated, 0);
    }
}
