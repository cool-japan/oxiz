use crate::SolverResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchStatus {
    Correct, // Both solvers agree
    Wrong,   // Different results (SAT vs UNSAT)
    Timeout, // One or both timed out
    Error,   // Parse/execution error in one or both
}

pub fn compare_results(oxiz: &SolverResult, z3: &SolverResult) -> MatchStatus {
    match (oxiz, z3) {
        // Both agree on SAT
        (SolverResult::Sat, SolverResult::Sat) => MatchStatus::Correct,

        // Both agree on UNSAT
        (SolverResult::Unsat, SolverResult::Unsat) => MatchStatus::Correct,

        // Both agree on UNKNOWN
        (SolverResult::Unknown, SolverResult::Unknown) => MatchStatus::Correct,

        // One returned UNKNOWN, the other a definite answer
        // This is acceptable - UNKNOWN is a valid response
        (SolverResult::Unknown, _) | (_, SolverResult::Unknown) => MatchStatus::Correct,

        // Timeout cases
        (SolverResult::Timeout, _) | (_, SolverResult::Timeout) => MatchStatus::Timeout,

        // Error cases
        (SolverResult::Error(_), _) | (_, SolverResult::Error(_)) => MatchStatus::Error,

        // Disagreement on SAT/UNSAT - this is a real problem!
        (SolverResult::Sat, SolverResult::Unsat) | (SolverResult::Unsat, SolverResult::Sat) => {
            MatchStatus::Wrong
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_both_sat() {
        let status = compare_results(&SolverResult::Sat, &SolverResult::Sat);
        assert_eq!(status, MatchStatus::Correct);
    }

    #[test]
    fn test_both_unsat() {
        let status = compare_results(&SolverResult::Unsat, &SolverResult::Unsat);
        assert_eq!(status, MatchStatus::Correct);
    }

    #[test]
    fn test_both_unknown() {
        let status = compare_results(&SolverResult::Unknown, &SolverResult::Unknown);
        assert_eq!(status, MatchStatus::Correct);
    }

    #[test]
    fn test_disagreement() {
        let status = compare_results(&SolverResult::Sat, &SolverResult::Unsat);
        assert_eq!(status, MatchStatus::Wrong);
    }

    #[test]
    fn test_unknown_vs_sat() {
        let status = compare_results(&SolverResult::Unknown, &SolverResult::Sat);
        assert_eq!(status, MatchStatus::Correct);
    }

    #[test]
    fn test_timeout() {
        let status = compare_results(&SolverResult::Timeout, &SolverResult::Sat);
        assert_eq!(status, MatchStatus::Timeout);
    }

    #[test]
    fn test_error() {
        let status = compare_results(&SolverResult::Error("test".to_string()), &SolverResult::Sat);
        assert_eq!(status, MatchStatus::Error);
    }
}
