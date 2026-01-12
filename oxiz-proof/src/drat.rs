//! DRAT proof format for SAT proofs.
//!
//! DRAT (Deletion Resolution Asymmetric Tautology) is the standard format
//! for SAT solver proofs, checkable by tools like `drat-trim`.
//!
//! ## Format
//!
//! DRAT proofs consist of a sequence of lines, each representing either:
//! - A clause addition (RAT clause)
//! - A clause deletion (marked with 'd')
//!
//! The proof demonstrates that the original formula is unsatisfiable by
//! deriving the empty clause.

use std::io::{self, Write};

/// A literal in DRAT format (signed integer, 0 is terminator)
pub type Lit = i32;

/// A clause in DRAT format
pub type Clause = Vec<Lit>;

/// A DRAT proof step
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DratStep {
    /// Add a clause (learned or derived)
    Add(Clause),
    /// Delete a clause
    Delete(Clause),
}

/// DRAT proof writer
///
/// Records proof steps and can output in text or binary format.
#[derive(Debug, Default)]
pub struct DratProof {
    /// Proof steps
    steps: Vec<DratStep>,
    /// Whether to use binary format
    binary: bool,
}

impl DratProof {
    /// Create a new DRAT proof writer (text format)
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            binary: false,
        }
    }

    /// Create a new DRAT proof writer (binary format)
    #[must_use]
    pub fn binary() -> Self {
        Self {
            steps: Vec::new(),
            binary: true,
        }
    }

    /// Add a clause to the proof
    pub fn add_clause(&mut self, clause: impl Into<Clause>) {
        self.steps.push(DratStep::Add(clause.into()));
    }

    /// Delete a clause from the proof
    pub fn delete_clause(&mut self, clause: impl Into<Clause>) {
        self.steps.push(DratStep::Delete(clause.into()));
    }

    /// Get the number of proof steps
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the proof steps
    #[must_use]
    pub fn steps(&self) -> &[DratStep] {
        &self.steps
    }

    /// Clear all proof steps
    pub fn clear(&mut self) {
        self.steps.clear();
    }

    /// Write the proof in text format
    pub fn write_text<W: Write>(&self, mut writer: W) -> io::Result<()> {
        for step in &self.steps {
            match step {
                DratStep::Add(clause) => {
                    for &lit in clause {
                        write!(writer, "{} ", lit)?;
                    }
                    writeln!(writer, "0")?;
                }
                DratStep::Delete(clause) => {
                    write!(writer, "d ")?;
                    for &lit in clause {
                        write!(writer, "{} ", lit)?;
                    }
                    writeln!(writer, "0")?;
                }
            }
        }
        Ok(())
    }

    /// Write the proof in binary format
    ///
    /// Binary DRAT format uses:
    /// - 'a' (0x61) prefix for additions
    /// - 'd' (0x64) prefix for deletions
    /// - Variable-length encoding for literals (similar to LEB128)
    pub fn write_binary<W: Write>(&self, mut writer: W) -> io::Result<()> {
        for step in &self.steps {
            match step {
                DratStep::Add(clause) => {
                    writer.write_all(b"a")?;
                    for &lit in clause {
                        self.write_lit_binary(&mut writer, lit)?;
                    }
                    self.write_lit_binary(&mut writer, 0)?;
                }
                DratStep::Delete(clause) => {
                    writer.write_all(b"d")?;
                    for &lit in clause {
                        self.write_lit_binary(&mut writer, lit)?;
                    }
                    self.write_lit_binary(&mut writer, 0)?;
                }
            }
        }
        Ok(())
    }

    /// Write a literal in binary format (variable-length encoding)
    fn write_lit_binary<W: Write>(&self, writer: &mut W, lit: Lit) -> io::Result<()> {
        // Convert signed literal to unsigned
        // lit > 0: 2*lit
        // lit < 0: 2*(-lit) + 1
        // lit = 0: 0 (terminator)
        let value = if lit == 0 {
            0u64
        } else if lit > 0 {
            (lit as u64) << 1
        } else {
            (((-lit) as u64) << 1) | 1
        };

        // Variable-length encoding
        let mut val = value;
        loop {
            let byte = (val & 0x7f) as u8;
            val >>= 7;
            if val == 0 {
                writer.write_all(&[byte])?;
                break;
            } else {
                writer.write_all(&[byte | 0x80])?;
            }
        }
        Ok(())
    }

    /// Write the proof in the configured format
    pub fn write<W: Write>(&self, writer: W) -> io::Result<()> {
        if self.binary {
            self.write_binary(writer)
        } else {
            self.write_text(writer)
        }
    }

    /// Convert to string (text format)
    #[must_use]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut buf = Vec::new();
        self.write_text(&mut buf)
            .expect("writing to Vec should not fail");
        String::from_utf8(buf).expect("DRAT output is ASCII")
    }
}

/// Trait for SAT solvers that can produce DRAT proofs
pub trait DratProofProducer {
    /// Enable DRAT proof production
    fn enable_proof(&mut self);

    /// Disable DRAT proof production
    fn disable_proof(&mut self);

    /// Get the DRAT proof (if available)
    fn get_proof(&self) -> Option<&DratProof>;

    /// Take the DRAT proof, leaving None
    fn take_proof(&mut self) -> Option<DratProof>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drat_proof_add() {
        let mut proof = DratProof::new();
        proof.add_clause(vec![1, 2, -3]);
        proof.add_clause(vec![4]);
        proof.add_clause(vec![]); // Empty clause (contradiction)

        let output = proof.to_string();
        assert!(output.contains("1 2 -3 0"));
        assert!(output.contains("4 0"));
        assert!(output.contains("0")); // Empty clause
    }

    #[test]
    fn test_drat_proof_delete() {
        let mut proof = DratProof::new();
        proof.add_clause(vec![1, 2]);
        proof.delete_clause(vec![1, 2]);

        let output = proof.to_string();
        assert!(output.contains("1 2 0"));
        assert!(output.contains("d 1 2 0"));
    }

    #[test]
    fn test_drat_proof_binary() {
        let mut proof = DratProof::binary();
        proof.add_clause(vec![1, -2]);
        proof.delete_clause(vec![3]);

        let mut buf = Vec::new();
        proof.write(&mut buf).unwrap();

        // Check binary format starts with 'a' and 'd'
        assert_eq!(buf[0], b'a');
        // Find the 'd' for deletion
        assert!(buf.contains(&b'd'));
    }

    #[test]
    fn test_drat_proof_clear() {
        let mut proof = DratProof::new();
        proof.add_clause(vec![1, 2]);
        assert!(!proof.is_empty());

        proof.clear();
        assert!(proof.is_empty());
    }

    #[test]
    fn test_drat_lit_encoding() {
        let proof = DratProof::new();
        let mut buf = Vec::new();

        // Test encoding of small positive literal
        proof.write_lit_binary(&mut buf, 1).unwrap();
        assert_eq!(buf, vec![2]); // 1 << 1 = 2

        buf.clear();
        proof.write_lit_binary(&mut buf, -1).unwrap();
        assert_eq!(buf, vec![3]); // (1 << 1) | 1 = 3

        buf.clear();
        proof.write_lit_binary(&mut buf, 0).unwrap();
        assert_eq!(buf, vec![0]); // Terminator

        buf.clear();
        proof.write_lit_binary(&mut buf, 64).unwrap();
        // 64 << 1 = 128, needs 2 bytes: 0x80 | 0, 0x01
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn test_drat_proof_steps() {
        let mut proof = DratProof::new();
        proof.add_clause(vec![1, 2]);
        proof.delete_clause(vec![1]);
        proof.add_clause(vec![-3]);

        assert_eq!(proof.len(), 3);
        assert_eq!(proof.steps()[0], DratStep::Add(vec![1, 2]));
        assert_eq!(proof.steps()[1], DratStep::Delete(vec![1]));
        assert_eq!(proof.steps()[2], DratStep::Add(vec![-3]));
    }
}
