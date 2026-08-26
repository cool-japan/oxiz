//! Binary proof-log serialisation for [`Context::check_sat`].
//!
//! Split into a child module so the (already large) `context` module stays
//! under the 2000-line policy limit; being a child of `context`, it retains
//! full access to `Context`'s private fields.

use crate::solver::SolverResult;
use std::path::Path;

use super::Context;

impl Context {
    /// Serialise a proof log entry for the given result.
    ///
    /// For `Unsat`, resolution proof steps are emitted when available;
    /// for `Sat` and `Unknown`, a single axiom node is written so the log is
    /// never empty and can be cleanly replayed.
    pub(super) fn write_proof_log(
        &self,
        path: &Path,
        result: SolverResult,
    ) -> std::result::Result<(), oxiz_proof::logging::LoggingError> {
        use oxiz_proof::logging::ProofLogger;
        use oxiz_proof::proof::{ProofNodeId, ProofStep};
        use smallvec::SmallVec;

        let mut logger = ProofLogger::create(path)?;

        match result {
            SolverResult::Unsat => {
                if let Some(proof) = self.solver.get_proof() {
                    let mut counter: u32 = 0;
                    for step in proof.steps() {
                        let entry = match step {
                            crate::solver::ProofStep::Input { index, .. } => ProofStep::Axiom {
                                conclusion: format!("input-clause-{}", index),
                            },
                            crate::solver::ProofStep::Resolution {
                                index,
                                left,
                                right,
                                pivot,
                                ..
                            } => {
                                let mut premises: SmallVec<[ProofNodeId; 4]> = SmallVec::new();
                                premises.push(ProofNodeId(*left));
                                premises.push(ProofNodeId(*right));
                                let mut args: SmallVec<[String; 2]> = SmallVec::new();
                                args.push(format!("{:?}", pivot));
                                ProofStep::Inference {
                                    rule: "resolution".to_string(),
                                    premises,
                                    conclusion: format!("resolution-{}", index),
                                    args,
                                }
                            }
                            crate::solver::ProofStep::TheoryLemma { index, theory, .. } => {
                                ProofStep::Axiom {
                                    conclusion: format!("theory-lemma-{}-{}", theory, index),
                                }
                            }
                        };
                        logger.log_step(ProofNodeId(counter), &entry)?;
                        counter += 1;
                    }
                    if counter == 0 {
                        // Proof object present but empty — emit minimal witness.
                        logger.log_step(
                            ProofNodeId(0),
                            &ProofStep::Axiom {
                                conclusion: "unsat".to_string(),
                            },
                        )?;
                    }
                } else {
                    logger.log_step(
                        ProofNodeId(0),
                        &ProofStep::Axiom {
                            conclusion: "unsat".to_string(),
                        },
                    )?;
                }
            }
            SolverResult::Sat => {
                logger.log_step(
                    ProofNodeId(0),
                    &ProofStep::Axiom {
                        conclusion: "sat".to_string(),
                    },
                )?;
            }
            SolverResult::Unknown => {
                logger.log_step(
                    ProofNodeId(0),
                    &ProofStep::Axiom {
                        conclusion: "unknown".to_string(),
                    },
                )?;
            }
        }

        logger.flush()?;
        logger.close()
    }
}
