//! Proof log replay and offline verification.
//!
//! `ProofReplayer` reads a binary proof log produced by `ProofLogger` and
//! re-executes every step, verifying that:
//!
//! 1. All premise references point to already-seen nodes.
//! 2. The binary format is well-formed.
//! 3. The proof is structurally consistent (no forward references, valid UTF-8
//!    strings, etc.).
//!
//! Semantic soundness checking (i.e. confirming that a rule application is
//! logically valid) is delegated to `crate::checker::ProofChecker` and is
//! performed on the reconstructed in-memory `Proof` object.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use thiserror::Error;

use crate::proof::{Proof, ProofNodeId, ProofStep};

// Re-use the same constants as logging.rs without a public dependency:
const MAGIC: &[u8; 8] = b"OXIZPROF";
const FORMAT_VERSION: u32 = 1;
const KIND_AXIOM: u8 = 0;
const KIND_INFERENCE: u8 = 1;

/// Errors that can occur while replaying a proof log.
#[derive(Error, Debug)]
pub enum ProofError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// File did not begin with the expected magic bytes.
    #[error("invalid file magic; expected OXIZPROF, got {found:?}")]
    InvalidMagic {
        /// Bytes that were actually read.
        found: [u8; 8],
    },

    /// The log file was written by an incompatible format version.
    #[error(
        "unsupported format version {version}; this reader supports version {}",
        FORMAT_VERSION
    )]
    UnsupportedVersion {
        /// Version found in the file.
        version: u32,
    },

    /// An unknown record kind byte was encountered.
    #[error("unknown record kind {0:#x}")]
    UnknownKind(u8),

    /// A premise ID referenced a node that has not been seen yet.
    #[error(
        "forward reference: node {referencing} references premise {missing} which has not been defined"
    )]
    ForwardReference {
        /// Node that contains the invalid reference.
        referencing: u32,
        /// Node ID that has not been defined yet.
        missing: u32,
    },

    /// A byte slice could not be decoded as UTF-8.
    #[error("invalid UTF-8 in proof step: {0}")]
    InvalidUtf8(String),

    /// The replay produced zero steps (empty or truncated file).
    #[error("proof log is empty or truncated after the header")]
    EmptyLog,

    /// A length-prefixed field had an unreasonably large value (guards OOM).
    #[error("field length {0} exceeds the safety limit")]
    FieldTooLarge(u32),
}

/// Result type for replay operations.
pub type ReplayResult<T> = Result<T, ProofError>;

/// Maximum allowed byte length for a single string field (64 MiB).
const MAX_FIELD_BYTES: u32 = 64 * 1024 * 1024;

/// Maximum number of premises per inference step (guards against OOM loops).
const MAX_PREMISES: u32 = 1_000_000;

/// Maximum number of arguments per inference step.
const MAX_ARGS: u32 = 1_000;

/// The result of a proof replay/verification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// The proof log was replayed successfully and all structural checks passed.
    Valid,
    /// A structural or semantic error was found.
    Invalid(String),
    /// The proof was structurally sound but did not contain a final conclusion;
    /// semantic verification could not be completed.
    Incomplete,
}

impl VerificationResult {
    /// Return `true` if the result is `Valid`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Return the reason string if the result is `Invalid`.
    #[must_use]
    pub fn invalid_reason(&self) -> Option<&str> {
        match self {
            Self::Invalid(reason) => Some(reason.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Valid => write!(f, "valid"),
            Self::Invalid(reason) => write!(f, "invalid: {}", reason),
            Self::Incomplete => write!(f, "incomplete"),
        }
    }
}

/// Statistics collected during a replay run.
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    /// Total number of records processed.
    pub total_steps: u64,
    /// Number of `Axiom` records.
    pub axiom_count: u64,
    /// Number of `Inference` records.
    pub inference_count: u64,
    /// Total number of premise references across all inference steps.
    pub total_premises: u64,
}

/// Reads and verifies a proof log produced by `ProofLogger`.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use oxiz_proof::replay::{ProofReplayer, VerificationResult};
///
/// let result = ProofReplayer::replay_from_file(Path::new("/tmp/proof.oxizlog"))
///     .expect("replay failed");
///
/// match result {
///     VerificationResult::Valid => println!("Proof is valid"),
///     VerificationResult::Invalid(reason) => eprintln!("Invalid: {}", reason),
///     VerificationResult::Incomplete => println!("Proof is incomplete"),
/// }
/// ```
pub struct ProofReplayer {
    /// The reconstructed proof DAG.
    proof: Proof,
    /// Mapping from log node IDs to in-memory `ProofNodeId`.
    id_map: HashMap<u32, ProofNodeId>,
    /// Statistics from the last replay.
    stats: ReplayStats,
}

impl ProofReplayer {
    /// Create a new, empty replayer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: Proof::new(),
            id_map: HashMap::new(),
            stats: ReplayStats::default(),
        }
    }

    /// Read, replay, and structurally verify a proof log file.
    ///
    /// This is the primary entry point.  It opens `path`, reads the header,
    /// then processes every record in sequence, rebuilding the proof DAG and
    /// checking that all references are backward-pointing.
    ///
    /// Returns `Ok(VerificationResult)` even when the proof is logically
    /// incomplete or invalid — the caller can inspect the variant to determine
    /// the outcome.  Only hard I/O or format errors are returned as `Err`.
    pub fn replay_from_file(path: &Path) -> ReplayResult<VerificationResult> {
        let mut replayer = Self::new();
        replayer.replay(path)
    }

    /// Replay a proof log and return the verification result.
    ///
    /// Mutates `self`, accumulating the proof and stats.
    pub fn replay(&mut self, path: &Path) -> ReplayResult<VerificationResult> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and validate header
        read_header(&mut reader)?;

        // Read records until EOF
        let mut steps_read = 0u64;
        loop {
            match read_record(&mut reader) {
                Ok(Some(record)) => {
                    self.incorporate_record(record)?;
                    steps_read += 1;
                }
                Ok(None) => break, // clean EOF
                Err(ProofError::Io(ref e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if steps_read == 0 {
                        return Err(ProofError::EmptyLog);
                    }
                    // Truncated file after at least one valid record.
                    return Ok(VerificationResult::Incomplete);
                }
                Err(e) => return Err(e),
            }
        }

        if steps_read == 0 {
            return Err(ProofError::EmptyLog);
        }

        // Run structural validation on the reconstructed proof.
        Ok(self.assess())
    }

    /// Consume `self` and return the reconstructed `Proof`.
    #[must_use]
    pub fn into_proof(self) -> Proof {
        self.proof
    }

    /// Return a reference to the reconstructed `Proof`.
    #[must_use]
    pub fn proof(&self) -> &Proof {
        &self.proof
    }

    /// Return replay statistics from the most recent `replay()` call.
    #[must_use]
    pub fn stats(&self) -> &ReplayStats {
        &self.stats
    }

    // ────────────────────────────── internals ────────────────────────────────

    /// Add a decoded log record to the in-memory proof DAG.
    fn incorporate_record(&mut self, record: LogRecord) -> ReplayResult<()> {
        let log_id = record.log_id;

        let mem_id = match record.step {
            ProofStep::Axiom { ref conclusion } => {
                self.stats.axiom_count += 1;
                self.proof.add_axiom(conclusion.clone())
            }
            ProofStep::Inference {
                ref rule,
                ref premises,
                ref conclusion,
                ref args,
            } => {
                self.stats.inference_count += 1;
                self.stats.total_premises += premises.len() as u64;

                // Translate log-level premise IDs → in-memory IDs.
                // premises are stored as ProofNodeId, whose inner u32 is the log ID.
                let mut mem_premises = Vec::with_capacity(premises.len());
                for p_node_id in premises.iter() {
                    let p_log_id = p_node_id.0;
                    let mem_premise = self.id_map.get(&p_log_id).copied().ok_or(
                        ProofError::ForwardReference {
                            referencing: log_id,
                            missing: p_log_id,
                        },
                    )?;
                    mem_premises.push(mem_premise);
                }

                self.proof.add_inference_with_args(
                    rule.clone(),
                    mem_premises,
                    args.iter().cloned().collect(),
                    conclusion.clone(),
                )
            }
        };

        self.stats.total_steps += 1;
        self.id_map.insert(log_id, mem_id);
        Ok(())
    }

    /// Assess the overall validity of the reconstructed proof.
    fn assess(&self) -> VerificationResult {
        let stats = self.proof.stats();
        if stats.total_nodes == 0 {
            return VerificationResult::Incomplete;
        }
        // A proof with at least one node and no forward-reference errors is valid.
        VerificationResult::Valid
    }
}

impl Default for ProofReplayer {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────── binary decoding ────────────────────────────────

/// An in-memory representation of one decoded log record.
struct LogRecord {
    /// The node ID recorded in the log.
    log_id: u32,
    /// The proof step.
    step: ProofStep,
}

/// Read and validate the 32-byte file header.
fn read_header<R: Read>(r: &mut R) -> ReplayResult<()> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(ProofError::InvalidMagic { found: magic });
    }

    let version = read_u32(r)?;
    if version != FORMAT_VERSION {
        return Err(ProofError::UnsupportedVersion { version });
    }

    // flags (reserved, skip 4 bytes)
    let _flags = read_u32(r)?;

    // 16 reserved bytes
    let mut reserved = [0u8; 16];
    r.read_exact(&mut reserved)?;

    Ok(())
}

/// Read one record from the log, returning `None` on clean EOF.
fn read_record<R: Read>(r: &mut R) -> ReplayResult<Option<LogRecord>> {
    // Attempt to read the kind byte; a clean EOF here is expected.
    let mut kind_buf = [0u8; 1];
    match r.read(&mut kind_buf) {
        Ok(0) => return Ok(None),
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(ProofError::Io(e)),
    }
    let kind = kind_buf[0];

    let log_id = read_u32(r)?;
    let conclusion = read_string(r)?;

    let step = match kind {
        KIND_AXIOM => ProofStep::Axiom { conclusion },
        KIND_INFERENCE => {
            let rule = read_string(r)?;

            let premise_count = read_u32(r)?;
            if premise_count > MAX_PREMISES {
                return Err(ProofError::FieldTooLarge(premise_count));
            }
            let mut premises = smallvec::SmallVec::<[ProofNodeId; 4]>::new();
            for _ in 0..premise_count {
                premises.push(ProofNodeId(read_u32(r)?));
            }

            let arg_count = read_u32(r)?;
            if arg_count > MAX_ARGS {
                return Err(ProofError::FieldTooLarge(arg_count));
            }
            let mut args = smallvec::SmallVec::<[String; 2]>::new();
            for _ in 0..arg_count {
                args.push(read_string(r)?);
            }

            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            }
        }
        other => return Err(ProofError::UnknownKind(other)),
    };

    Ok(Some(LogRecord { log_id, step }))
}

/// Read a 4-byte little-endian u32.
fn read_u32<R: Read>(r: &mut R) -> ReplayResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a length-prefixed UTF-8 string.
fn read_string<R: Read>(r: &mut R) -> ReplayResult<String> {
    let len = read_u32(r)?;
    if len > MAX_FIELD_BYTES {
        return Err(ProofError::FieldTooLarge(len));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| ProofError::InvalidUtf8(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::ProofLogger;
    use crate::proof::{ProofNodeId, ProofStep};
    use smallvec::SmallVec;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("oxiz_replay_{}.oxizlog", name))
    }

    #[test]
    fn test_roundtrip_axiom() {
        let path = temp_path("roundtrip_axiom");
        let _ = std::fs::remove_file(&path);

        {
            let mut logger = ProofLogger::create(&path).expect("create");
            let step = ProofStep::Axiom {
                conclusion: "(= x 0)".to_string(),
            };
            logger.log_step(ProofNodeId(0), &step).expect("log");
            logger.close().expect("close");
        }

        let result = ProofReplayer::replay_from_file(&path).expect("replay");
        assert_eq!(result, VerificationResult::Valid);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_roundtrip_inference_chain() {
        let path = temp_path("roundtrip_chain");
        let _ = std::fs::remove_file(&path);

        {
            let mut logger = ProofLogger::create(&path).expect("create");

            // Step 0: axiom  p
            logger
                .log_step(
                    ProofNodeId(0),
                    &ProofStep::Axiom {
                        conclusion: "p".to_string(),
                    },
                )
                .expect("log");

            // Step 1: axiom  p → q
            logger
                .log_step(
                    ProofNodeId(1),
                    &ProofStep::Axiom {
                        conclusion: "(=> p q)".to_string(),
                    },
                )
                .expect("log");

            // Step 2: modus ponens  p, p→q  ⊢  q
            logger
                .log_step(
                    ProofNodeId(2),
                    &ProofStep::Inference {
                        rule: "mp".to_string(),
                        premises: SmallVec::from_vec(vec![ProofNodeId(0), ProofNodeId(1)]),
                        conclusion: "q".to_string(),
                        args: SmallVec::new(),
                    },
                )
                .expect("log");

            logger.close().expect("close");
        }

        let mut replayer = ProofReplayer::new();
        let result = replayer.replay(&path).expect("replay");
        assert_eq!(result, VerificationResult::Valid);
        assert_eq!(replayer.stats().total_steps, 3);
        assert_eq!(replayer.stats().axiom_count, 2);
        assert_eq!(replayer.stats().inference_count, 1);
        assert_eq!(replayer.stats().total_premises, 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_empty_log_is_error() {
        let path = temp_path("empty_log");
        let _ = std::fs::remove_file(&path);

        // Write only the header, no records.
        {
            let mut logger = ProofLogger::create(&path).expect("create");
            logger.close().expect("close");
        }

        let result = ProofReplayer::replay_from_file(&path);
        assert!(matches!(result, Err(ProofError::EmptyLog)));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bad_magic() {
        let path = temp_path("bad_magic");
        let _ = std::fs::remove_file(&path);

        std::fs::write(
            &path,
            b"BADMAGIC\x01\x00\x00\x00\x00\x00\x00\x00\
                                 \x00\x00\x00\x00\x00\x00\x00\x00\
                                 \x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .expect("write");

        let err = ProofReplayer::replay_from_file(&path).expect_err("should fail");
        assert!(matches!(err, ProofError::InvalidMagic { .. }));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_verification_result_display() {
        assert_eq!(VerificationResult::Valid.to_string(), "valid");
        assert_eq!(
            VerificationResult::Invalid("bad step".to_string()).to_string(),
            "invalid: bad step"
        );
        assert_eq!(VerificationResult::Incomplete.to_string(), "incomplete");
    }
}
