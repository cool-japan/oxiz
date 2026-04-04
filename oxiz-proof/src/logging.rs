//! Proof step logging to persistent binary log files.
//!
//! `ProofLogger` streams proof steps to disk in a compact binary format during
//! solving.  The resulting log file can later be read by `ProofReplayer` in
//! `crate::replay` for offline verification.
//!
//! # Binary Format
//!
//! Each log file begins with a fixed-size header, followed by zero or more
//! *records*.  Every record encodes one `ProofStep`:
//!
//! ```text
//! ┌──────────────────── FILE HEADER (32 bytes) ─────────────────────┐
//! │ magic: [u8; 8]  = b"OXIZPROF"                                   │
//! │ version: u32    = FORMAT_VERSION                                 │
//! │ flags: u32      = 0 (reserved)                                   │
//! │ reserved: [u8; 16]                                               │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌──────────────── RECORD (variable length) ───────────────────────┐
//! │ kind: u8        = 0 (Axiom) | 1 (Inference)                     │
//! │ node_id: u32    (little-endian)                                  │
//! │ conclusion_len: u32                                              │
//! │ conclusion: [u8; conclusion_len]  (UTF-8)                        │
//! │  ── if kind == Inference ──────────────────────────────────────  │
//! │ rule_len: u32                                                    │
//! │ rule: [u8; rule_len]             (UTF-8)                         │
//! │ premise_count: u32                                               │
//! │ premises: [u32; premise_count]   (node IDs, little-endian)      │
//! │ arg_count: u32                                                   │
//! │ for each arg:                                                    │
//! │   arg_len: u32 + arg: [u8; arg_len]                              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::proof::{ProofNodeId, ProofStep};

/// Magic bytes at the start of every OxiZ proof log.
const MAGIC: &[u8; 8] = b"OXIZPROF";

/// Current binary format version.
const FORMAT_VERSION: u32 = 1;

/// Record kind byte for `ProofStep::Axiom`.
const KIND_AXIOM: u8 = 0;

/// Record kind byte for `ProofStep::Inference`.
const KIND_INFERENCE: u8 = 1;

/// Errors produced by `ProofLogger`.
#[derive(Error, Debug)]
pub enum LoggingError {
    /// Underlying I/O failure.
    #[error("I/O error while logging proof: {0}")]
    Io(#[from] io::Error),

    /// Attempted to log a step after the logger has been closed.
    #[error("attempted to write to a closed ProofLogger")]
    AlreadyClosed,
}

/// Result type for logging operations.
pub type LoggingResult<T> = Result<T, LoggingError>;

/// Streams proof steps to a binary log file during solving.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use oxiz_proof::logging::ProofLogger;
/// use oxiz_proof::proof::{ProofStep, ProofNodeId};
///
/// let mut logger = ProofLogger::create(Path::new("/tmp/proof.oxizlog"))
///     .expect("failed to create log");
///
/// logger.log_step(ProofNodeId(0), &ProofStep::Axiom {
///     conclusion: "true".to_string(),
/// }).expect("failed to log step");
///
/// logger.close().expect("failed to close log");
/// ```
pub struct ProofLogger {
    /// Buffered writer to the log file.
    writer: Option<BufWriter<File>>,
    /// Path to the log file (kept for diagnostics).
    path: PathBuf,
    /// Number of steps written so far.
    step_count: u64,
}

impl ProofLogger {
    /// Create a new proof log at `path`, overwriting any existing file.
    ///
    /// Writes the file header immediately.
    pub fn create(path: &Path) -> LoggingResult<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer)?;
        writer.flush()?;

        Ok(Self {
            writer: Some(writer),
            path: path.to_path_buf(),
            step_count: 0,
        })
    }

    /// Open an existing proof log for appending.
    ///
    /// Does **not** write a new header; the file must already have a valid header
    /// written by a previous `ProofLogger::create` call.
    pub fn append(path: &Path) -> LoggingResult<Self> {
        let file = OpenOptions::new().append(true).open(path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer: Some(writer),
            path: path.to_path_buf(),
            step_count: 0,
        })
    }

    /// Append a single `ProofStep` to the log.
    ///
    /// `node_id` is the `ProofNodeId` assigned to this step in the live proof
    /// DAG; it is recorded so that cross-references in `Inference` premises can
    /// be resolved during replay.
    ///
    /// # Errors
    ///
    /// Returns `LoggingError::AlreadyClosed` if called after `close()`.
    pub fn log_step(&mut self, node_id: ProofNodeId, step: &ProofStep) -> LoggingResult<()> {
        let writer = self.writer.as_mut().ok_or(LoggingError::AlreadyClosed)?;

        write_step(writer, node_id, step)?;
        self.step_count += 1;
        Ok(())
    }

    /// Flush the internal write buffer to the OS.
    ///
    /// Under normal operation you do not need to call this explicitly;
    /// `close()` flushes before closing.  Call `flush()` for crash-safety
    /// checkpointing during long-running solves.
    pub fn flush(&mut self) -> LoggingResult<()> {
        if let Some(ref mut w) = self.writer {
            w.flush()?;
        }
        Ok(())
    }

    /// Flush and close the log file.
    ///
    /// After this call, further `log_step` calls will return
    /// `LoggingError::AlreadyClosed`.
    pub fn close(&mut self) -> LoggingResult<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        Ok(())
    }

    /// Return the number of proof steps written since this logger was created.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Return the path of the log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return `true` if the logger has been closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.writer.is_none()
    }
}

impl Drop for ProofLogger {
    fn drop(&mut self) {
        // Best-effort flush on drop; ignore errors since we can't propagate.
        if let Some(ref mut w) = self.writer {
            let _ = w.flush();
        }
    }
}

// ──────────────────────────────── helpers ────────────────────────────────────

/// Write the 32-byte file header.
fn write_header<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    // flags (reserved)
    w.write_all(&0u32.to_le_bytes())?;
    // 16 reserved bytes
    w.write_all(&[0u8; 16])?;
    Ok(())
}

/// Encode and write one proof step record.
fn write_step<W: Write>(w: &mut W, node_id: ProofNodeId, step: &ProofStep) -> io::Result<()> {
    match step {
        ProofStep::Axiom { conclusion } => {
            w.write_all(&[KIND_AXIOM])?;
            w.write_all(&node_id.0.to_le_bytes())?;
            write_bytes(w, conclusion.as_bytes())?;
        }
        ProofStep::Inference {
            rule,
            premises,
            conclusion,
            args,
        } => {
            w.write_all(&[KIND_INFERENCE])?;
            w.write_all(&node_id.0.to_le_bytes())?;
            write_bytes(w, conclusion.as_bytes())?;
            write_bytes(w, rule.as_bytes())?;

            // premises
            let premise_count = premises.len() as u32;
            w.write_all(&premise_count.to_le_bytes())?;
            for p in premises.iter() {
                w.write_all(&p.0.to_le_bytes())?;
            }

            // args
            let arg_count = args.len() as u32;
            w.write_all(&arg_count.to_le_bytes())?;
            for arg in args.iter() {
                write_bytes(w, arg.as_bytes())?;
            }
        }
    }
    Ok(())
}

/// Write a length-prefixed byte slice (u32 LE length + data).
fn write_bytes<W: Write>(w: &mut W, data: &[u8]) -> io::Result<()> {
    let len = data.len() as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::{ProofNodeId, ProofStep};
    use smallvec::SmallVec;

    fn temp_log_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oxiz_proof_log_{}.oxizlog", name))
    }

    #[test]
    fn test_create_and_close() {
        let path = temp_log_path("create_close");
        let _ = std::fs::remove_file(&path);

        let mut logger = ProofLogger::create(&path).expect("create failed");
        assert!(!logger.is_closed());
        logger.close().expect("close failed");
        assert!(logger.is_closed());
        assert!(path.exists());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_log_axiom() {
        let path = temp_log_path("log_axiom");
        let _ = std::fs::remove_file(&path);

        let mut logger = ProofLogger::create(&path).expect("create failed");
        let step = ProofStep::Axiom {
            conclusion: "(assert (= x 0))".to_string(),
        };
        logger.log_step(ProofNodeId(0), &step).expect("log failed");
        logger.close().expect("close failed");

        assert!(path.metadata().expect("stat failed").len() > 32);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_log_inference() {
        let path = temp_log_path("log_inference");
        let _ = std::fs::remove_file(&path);

        let mut logger = ProofLogger::create(&path).expect("create failed");

        let axiom = ProofStep::Axiom {
            conclusion: "p".to_string(),
        };
        logger.log_step(ProofNodeId(0), &axiom).expect("log failed");

        let inference = ProofStep::Inference {
            rule: "mp".to_string(),
            premises: SmallVec::from_vec(vec![ProofNodeId(0)]),
            conclusion: "q".to_string(),
            args: SmallVec::new(),
        };
        logger
            .log_step(ProofNodeId(1), &inference)
            .expect("log failed");

        assert_eq!(logger.step_count(), 2);
        logger.close().expect("close failed");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_after_close_fails() {
        let path = temp_log_path("after_close");
        let _ = std::fs::remove_file(&path);

        let mut logger = ProofLogger::create(&path).expect("create failed");
        logger.close().expect("close failed");

        let step = ProofStep::Axiom {
            conclusion: "p".to_string(),
        };
        let result = logger.log_step(ProofNodeId(0), &step);
        assert!(matches!(result, Err(LoggingError::AlreadyClosed)));

        let _ = std::fs::remove_file(&path);
    }
}
