//! Difficulty classification for SMT benchmarks.
//!
//! Defines [`DifficultyClass`], a five-bucket ordinal scale that maps
//! expected solver runtime to a human-readable difficulty label.

/// Five-level difficulty scale for SMT benchmarks.
///
/// Ordered from easiest to hardest so `Trivial < Easy < Medium < Hard < VeryHard`.
/// This ordering is used when sorting benchmarks for LPT scheduling.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum DifficultyClass {
    /// Expected runtime below 0.1 seconds.
    Trivial,
    /// Expected runtime in the 0.1–1 s range.
    Easy,
    /// Expected runtime in the 1–10 s range.
    Medium,
    /// Expected runtime in the 10–60 s range.
    Hard,
    /// Expected runtime at or above 60 seconds.
    VeryHard,
}

impl DifficultyClass {
    /// Map a runtime in seconds to the corresponding difficulty bucket.
    ///
    /// The bucket boundaries are: 0.1 s, 1 s, 10 s, 60 s.
    #[must_use]
    pub fn from_runtime_seconds(s: f64) -> Self {
        if s < 0.1 {
            Self::Trivial
        } else if s < 1.0 {
            Self::Easy
        } else if s < 10.0 {
            Self::Medium
        } else if s < 60.0 {
            Self::Hard
        } else {
            Self::VeryHard
        }
    }

    /// Human-readable label suitable for display and JSON keys.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Trivial => "trivial",
            Self::Easy => "easy",
            Self::Medium => "medium",
            Self::Hard => "hard",
            Self::VeryHard => "very_hard",
        }
    }

    /// Exclusive upper bound in seconds for this class, or `None` for `VeryHard`.
    #[must_use]
    pub fn upper_bound_seconds(&self) -> Option<f64> {
        match self {
            Self::Trivial => Some(0.1),
            Self::Easy => Some(1.0),
            Self::Medium => Some(10.0),
            Self::Hard => Some(60.0),
            Self::VeryHard => None,
        }
    }
}

impl std::fmt::Display for DifficultyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_boundaries() {
        assert_eq!(
            DifficultyClass::from_runtime_seconds(0.0),
            DifficultyClass::Trivial
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(0.05),
            DifficultyClass::Trivial
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(0.1),
            DifficultyClass::Easy
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(0.5),
            DifficultyClass::Easy
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(1.0),
            DifficultyClass::Medium
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(5.0),
            DifficultyClass::Medium
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(10.0),
            DifficultyClass::Hard
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(30.0),
            DifficultyClass::Hard
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(60.0),
            DifficultyClass::VeryHard
        );
        assert_eq!(
            DifficultyClass::from_runtime_seconds(120.0),
            DifficultyClass::VeryHard
        );
    }

    #[test]
    fn test_ordering() {
        assert!(DifficultyClass::Trivial < DifficultyClass::Easy);
        assert!(DifficultyClass::Easy < DifficultyClass::Medium);
        assert!(DifficultyClass::Medium < DifficultyClass::Hard);
        assert!(DifficultyClass::Hard < DifficultyClass::VeryHard);
    }

    #[test]
    fn test_labels() {
        assert_eq!(DifficultyClass::Trivial.label(), "trivial");
        assert_eq!(DifficultyClass::VeryHard.label(), "very_hard");
    }

    #[test]
    fn test_upper_bound() {
        assert_eq!(DifficultyClass::Trivial.upper_bound_seconds(), Some(0.1));
        assert_eq!(DifficultyClass::VeryHard.upper_bound_seconds(), None);
    }
}
