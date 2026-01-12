//! DRAT and LRAT proof generation for SAT solving
//!
//! DRAT (Deletion Resolution Asymmetric Tautology) is a proof format
//! that allows verification of UNSAT results from SAT solvers.
//!
//! LRAT (Labelled Resolution Asymmetric Tautology) is an extension of DRAT
//! that includes clause IDs and resolution hints for more efficient verification.

use crate::literal::Lit;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// DRAT proof logger
#[derive(Debug)]
pub struct DratProof {
    /// Writer for the proof file
    writer: Option<BufWriter<File>>,
    /// Whether proof logging is enabled
    enabled: bool,
}

impl DratProof {
    /// Create a new DRAT proof logger (disabled)
    pub fn new() -> Self {
        Self {
            writer: None,
            enabled: false,
        }
    }

    /// Enable proof logging to a file
    pub fn enable(&mut self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = File::create(path)?;
        self.writer = Some(BufWriter::new(file));
        self.enabled = true;
        Ok(())
    }

    /// Disable proof logging
    pub fn disable(&mut self) {
        self.enabled = false;
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.flush();
        }
    }

    /// Check if proof logging is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log clause addition
    pub fn add_clause(&mut self, lits: &[Lit]) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Some(writer) = &mut self.writer {
            // Write literals in DIMACS format
            for &lit in lits {
                write!(writer, "{} ", lit.to_dimacs())?;
            }
            writeln!(writer, "0")?;
        }

        Ok(())
    }

    /// Log clause deletion
    pub fn delete_clause(&mut self, lits: &[Lit]) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Some(writer) = &mut self.writer {
            // Deletion is marked with 'd' prefix
            write!(writer, "d ")?;
            for &lit in lits {
                write!(writer, "{} ", lit.to_dimacs())?;
            }
            writeln!(writer, "0")?;
        }

        Ok(())
    }

    /// Flush the proof to disk
    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }
}

impl Default for DratProof {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DratProof {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// LRAT proof logger
///
/// LRAT extends DRAT with clause IDs and resolution hints for more efficient verification
#[derive(Debug)]
pub struct LratProof {
    /// Writer for the proof file
    writer: Option<BufWriter<File>>,
    /// Whether proof logging is enabled
    enabled: bool,
    /// Next clause ID to assign
    next_id: u64,
}

impl LratProof {
    /// Create a new LRAT proof logger (disabled)
    pub fn new() -> Self {
        Self {
            writer: None,
            enabled: false,
            next_id: 1,
        }
    }

    /// Enable proof logging to a file
    pub fn enable(&mut self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = File::create(path)?;
        self.writer = Some(BufWriter::new(file));
        self.enabled = true;
        Ok(())
    }

    /// Disable proof logging
    pub fn disable(&mut self) {
        self.enabled = false;
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.flush();
        }
    }

    /// Check if proof logging is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log clause addition with hints
    ///
    /// # Arguments
    /// * `lits` - The literals in the clause
    /// * `hints` - Clause IDs used in the derivation (for RAT checking)
    ///
    /// # Returns
    /// The ID assigned to this clause
    pub fn add_clause(&mut self, lits: &[Lit], hints: &[u64]) -> std::io::Result<u64> {
        if !self.enabled {
            let id = self.next_id;
            self.next_id += 1;
            return Ok(id);
        }

        let clause_id = self.next_id;
        self.next_id += 1;

        if let Some(writer) = &mut self.writer {
            // LRAT format: <id> <lits> 0 [<hints>] 0
            write!(writer, "{} ", clause_id)?;

            // Write literals
            for &lit in lits {
                write!(writer, "{} ", lit.to_dimacs())?;
            }
            write!(writer, "0")?;

            // Write hints if provided
            if !hints.is_empty() {
                write!(writer, " ")?;
                for &hint in hints {
                    write!(writer, "{} ", hint)?;
                }
                write!(writer, "0")?;
            }

            writeln!(writer)?;
        }

        Ok(clause_id)
    }

    /// Log original clause (from input formula)
    ///
    /// Original clauses are added with their sequential IDs
    pub fn add_original_clause(&mut self, lits: &[Lit]) -> std::io::Result<u64> {
        self.add_clause(lits, &[])
    }

    /// Log clause deletion
    ///
    /// # Arguments
    /// * `clause_id` - The ID of the clause to delete
    pub fn delete_clause(&mut self, clause_id: u64) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if let Some(writer) = &mut self.writer {
            // LRAT deletion format: <id> d <clause_id> 0
            writeln!(writer, "{} d {} 0", self.next_id, clause_id)?;
            self.next_id += 1;
        }

        Ok(())
    }

    /// Log the empty clause (proof of UNSAT)
    pub fn add_empty_clause(&mut self, hints: &[u64]) -> std::io::Result<u64> {
        self.add_clause(&[], hints)
    }

    /// Flush the proof to disk
    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }
        Ok(())
    }

    /// Get the next clause ID that will be assigned
    pub fn next_id(&self) -> u64 {
        self.next_id
    }
}

impl Default for LratProof {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LratProof {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Proof trimmer for removing unnecessary clauses
///
/// This analyzes a proof and removes clauses that are not needed for the final derivation
#[derive(Debug)]
pub struct ProofTrimmer {
    /// Clauses that are actually used in the proof
    used_clauses: std::collections::HashSet<u64>,
    /// The final clause ID (usually the empty clause)
    #[allow(dead_code)]
    final_clause_id: u64,
}

impl ProofTrimmer {
    /// Create a new proof trimmer
    pub fn new(final_clause_id: u64) -> Self {
        let mut used = std::collections::HashSet::new();
        used.insert(final_clause_id);

        Self {
            used_clauses: used,
            final_clause_id,
        }
    }

    /// Mark a clause and its dependencies as used
    pub fn mark_used(&mut self, clause_id: u64, dependencies: &[u64]) {
        if self.used_clauses.insert(clause_id) {
            // Also mark dependencies as used
            for &dep_id in dependencies {
                self.used_clauses.insert(dep_id);
            }
        }
    }

    /// Check if a clause is used in the trimmed proof
    pub fn is_used(&self, clause_id: u64) -> bool {
        self.used_clauses.contains(&clause_id)
    }

    /// Trim a proof by removing unused clauses
    ///
    /// This reads a proof file and writes a trimmed version
    pub fn trim_proof(
        &self,
        input_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> std::io::Result<usize> {
        use std::io::{BufRead, BufReader};

        let input = File::open(input_path)?;
        let reader = BufReader::new(input);

        let output = File::create(output_path)?;
        let mut writer = BufWriter::new(output);

        let mut trimmed_count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // Parse the clause ID from the line
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // First part should be the clause ID
            if let Ok(clause_id) = parts[0].parse::<u64>() {
                if self.is_used(clause_id) {
                    // Keep this clause
                    writeln!(writer, "{}", line)?;
                } else {
                    // Trim this clause
                    trimmed_count += 1;
                }
            } else {
                // Not a valid clause line, keep it (might be a comment)
                writeln!(writer, "{}", line)?;
            }
        }

        writer.flush()?;

        Ok(trimmed_count)
    }

    /// Get the number of used clauses
    pub fn num_used_clauses(&self) -> usize {
        self.used_clauses.len()
    }
}

impl Default for ProofTrimmer {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;
    use std::fs;
    use std::io::Read;

    #[test]
    fn test_drat_proof() {
        let path = "/tmp/test_drat.proof";
        let mut proof = DratProof::new();

        assert!(!proof.is_enabled());

        proof.enable(path).unwrap();
        assert!(proof.is_enabled());

        let v0 = Var(0);
        let v1 = Var(1);

        // Add a clause: x0 ∨ x1
        proof.add_clause(&[Lit::pos(v0), Lit::pos(v1)]).unwrap();

        // Add a clause: ~x0
        proof.add_clause(&[Lit::neg(v0)]).unwrap();

        // Delete a clause
        proof.delete_clause(&[Lit::pos(v0), Lit::pos(v1)]).unwrap();

        proof.flush().unwrap();
        proof.disable();

        // Read the proof file and verify
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert!(contents.contains("1 2 0"));
        assert!(contents.contains("-1 0"));
        assert!(contents.contains("d 1 2 0"));

        // Clean up
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_disabled_proof() {
        let mut proof = DratProof::new();

        // Should not error even though not enabled
        let v0 = Var(0);
        proof.add_clause(&[Lit::pos(v0)]).unwrap();
        proof.delete_clause(&[Lit::pos(v0)]).unwrap();
    }

    #[test]
    fn test_lrat_proof() {
        let path = "/tmp/test_lrat.proof";
        let mut proof = LratProof::new();

        assert!(!proof.is_enabled());

        proof.enable(path).unwrap();
        assert!(proof.is_enabled());

        let v0 = Var(0);
        let v1 = Var(1);

        // Add an original clause: x0 ∨ x1
        let id1 = proof
            .add_original_clause(&[Lit::pos(v0), Lit::pos(v1)])
            .unwrap();
        assert_eq!(id1, 1);

        // Add a clause with hints: ~x0 (derived from clause 1)
        let id2 = proof.add_clause(&[Lit::neg(v0)], &[1]).unwrap();
        assert_eq!(id2, 2);

        // Delete clause 1
        proof.delete_clause(id1).unwrap();

        proof.flush().unwrap();
        proof.disable();

        // Read the proof file and verify
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert!(contents.contains("1 1 2 0"));
        assert!(contents.contains("2 -1 0 1 0"));
        assert!(contents.contains("d 1 0"));

        // Clean up
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_lrat_empty_clause() {
        let path = "/tmp/test_lrat_empty.proof";
        let mut proof = LratProof::new();

        proof.enable(path).unwrap();

        // Add empty clause (UNSAT proof) with hints
        let id = proof.add_empty_clause(&[1, 2, 3]).unwrap();
        assert_eq!(id, 1);

        proof.flush().unwrap();
        proof.disable();

        // Read the proof file
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        assert!(contents.contains("1 0 1 2 3 0"));

        // Clean up
        fs::remove_file(path).unwrap();
    }
}
