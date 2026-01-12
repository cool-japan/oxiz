//! Checkpointing module for long-running solver tasks
//!
//! This module provides functionality to save and restore solver state,
//! enabling interruption and resumption of long-running computations.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique checkpoint ID
    pub id: String,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Problem being solved (SMT-LIB2 script)
    pub problem: String,
    /// Logic used
    pub logic: Option<String>,
    /// Solver state (serialized)
    pub solver_state: SolverState,
    /// Progress information
    pub progress: ProgressInfo,
    /// Configuration options
    pub options: Vec<(String, String)>,
}

/// Solver state that can be checkpointed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverState {
    /// Learned clauses
    pub learned_clauses: Vec<Vec<i32>>,
    /// Variable assignments
    pub assignments: Vec<(usize, bool)>,
    /// Decision level
    pub decision_level: usize,
    /// Number of conflicts so far
    pub conflicts: usize,
    /// Number of decisions so far
    pub decisions: usize,
    /// Number of propagations so far
    pub propagations: usize,
    /// VSIDS scores (variable activity)
    pub vsids_scores: Vec<(usize, f64)>,
    /// Restart count
    pub restarts: usize,
}

/// Progress tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// Time elapsed in milliseconds
    pub elapsed_ms: u128,
    /// Estimated completion percentage (0-100)
    pub completion_pct: f64,
    /// Current phase (e.g., "preprocessing", "solving", "model-extraction")
    pub phase: String,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Checkpoint {
    /// Create a new checkpoint
    #[allow(dead_code)]
    pub fn new(
        problem: String,
        logic: Option<String>,
        solver_state: SolverState,
        progress: ProgressInfo,
        options: Vec<(String, String)>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let id = format!("checkpoint_{}", timestamp);

        Self {
            id,
            timestamp,
            problem,
            logic,
            solver_state,
            progress,
            options,
        }
    }

    /// Save checkpoint to file
    pub fn save(&self, checkpoint_dir: &Path) -> Result<PathBuf, String> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(checkpoint_dir)
            .map_err(|e| format!("Failed to create checkpoint directory: {}", e))?;

        let checkpoint_file = checkpoint_dir.join(format!("{}.json", self.id));

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize checkpoint: {}", e))?;

        fs::write(&checkpoint_file, json)
            .map_err(|e| format!("Failed to write checkpoint: {}", e))?;

        Ok(checkpoint_file)
    }

    /// Load checkpoint from file
    pub fn load(checkpoint_file: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(checkpoint_file)
            .map_err(|e| format!("Failed to read checkpoint: {}", e))?;

        let checkpoint: Self = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize checkpoint: {}", e))?;

        Ok(checkpoint)
    }

    /// List all checkpoints in a directory
    pub fn list_checkpoints(checkpoint_dir: &Path) -> Result<Vec<PathBuf>, String> {
        if !checkpoint_dir.exists() {
            return Ok(Vec::new());
        }

        let entries = fs::read_dir(checkpoint_dir)
            .map_err(|e| format!("Failed to read checkpoint directory: {}", e))?;

        let mut checkpoints = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                checkpoints.push(path);
            }
        }

        // Sort by modification time (newest first)
        checkpoints.sort_by(|a, b| {
            let a_time = fs::metadata(a).and_then(|m| m.modified()).ok();
            let b_time = fs::metadata(b).and_then(|m| m.modified()).ok();
            b_time.cmp(&a_time)
        });

        Ok(checkpoints)
    }

    /// Delete a checkpoint file
    #[allow(dead_code)]
    pub fn delete(&self, checkpoint_dir: &Path) -> Result<(), String> {
        let checkpoint_file = checkpoint_dir.join(format!("{}.json", self.id));
        fs::remove_file(&checkpoint_file).map_err(|e| format!("Failed to delete checkpoint: {}", e))
    }

    /// Get the most recent checkpoint for a problem
    #[allow(dead_code)]
    pub fn get_latest(checkpoint_dir: &Path, problem_hash: &str) -> Result<Option<Self>, String> {
        let checkpoints = Self::list_checkpoints(checkpoint_dir)?;

        for checkpoint_file in checkpoints {
            if let Ok(checkpoint) = Self::load(&checkpoint_file) {
                // Simple hash check - in production, use a proper hash
                if checkpoint.id.contains(problem_hash) {
                    return Ok(Some(checkpoint));
                }
            }
        }

        Ok(None)
    }
}

impl SolverState {
    /// Create a new empty solver state
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            learned_clauses: Vec::new(),
            assignments: Vec::new(),
            decision_level: 0,
            conflicts: 0,
            decisions: 0,
            propagations: 0,
            vsids_scores: Vec::new(),
            restarts: 0,
        }
    }
}

impl Default for SolverState {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressInfo {
    /// Create a new progress info
    #[allow(dead_code)]
    pub fn new(elapsed_ms: u128, completion_pct: f64, phase: String) -> Self {
        Self {
            elapsed_ms,
            completion_pct,
            phase,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata
    #[allow(dead_code)]
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Checkpoint manager for handling periodic checkpointing
#[allow(dead_code)]
pub struct CheckpointManager {
    /// Directory where checkpoints are saved
    checkpoint_dir: PathBuf,
    /// Checkpoint interval in seconds
    interval_secs: u64,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Last checkpoint time
    last_checkpoint: Option<std::time::Instant>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    #[allow(dead_code)]
    pub fn new(checkpoint_dir: PathBuf, interval_secs: u64, max_checkpoints: usize) -> Self {
        Self {
            checkpoint_dir,
            interval_secs,
            max_checkpoints,
            last_checkpoint: None,
        }
    }

    /// Check if it's time to create a new checkpoint
    #[allow(dead_code)]
    pub fn should_checkpoint(&self) -> bool {
        if let Some(last) = self.last_checkpoint {
            last.elapsed().as_secs() >= self.interval_secs
        } else {
            true // First checkpoint
        }
    }

    /// Save a checkpoint and update internal state
    #[allow(dead_code)]
    pub fn save_checkpoint(&mut self, checkpoint: Checkpoint) -> Result<PathBuf, String> {
        let path = checkpoint.save(&self.checkpoint_dir)?;
        self.last_checkpoint = Some(std::time::Instant::now());

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(path)
    }

    /// Remove old checkpoints keeping only the most recent max_checkpoints
    #[allow(dead_code)]
    fn cleanup_old_checkpoints(&self) -> Result<(), String> {
        let mut checkpoints = Checkpoint::list_checkpoints(&self.checkpoint_dir)?;

        if checkpoints.len() > self.max_checkpoints {
            // Keep only the most recent max_checkpoints
            for checkpoint_file in checkpoints.drain(self.max_checkpoints..) {
                let _ = fs::remove_file(checkpoint_file);
            }
        }

        Ok(())
    }

    /// Load the most recent checkpoint
    #[allow(dead_code)]
    pub fn load_latest(&self) -> Result<Option<Checkpoint>, String> {
        let checkpoints = Checkpoint::list_checkpoints(&self.checkpoint_dir)?;

        if let Some(checkpoint_file) = checkpoints.first() {
            Ok(Some(Checkpoint::load(checkpoint_file)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new(
            "(assert true)".to_string(),
            Some("QF_LIA".to_string()),
            SolverState::new(),
            ProgressInfo::new(1000, 50.0, "solving".to_string()),
            vec![("timeout".to_string(), "30".to_string())],
        );

        assert!(!checkpoint.id.is_empty());
        assert_eq!(checkpoint.problem, "(assert true)");
        assert_eq!(checkpoint.logic, Some("QF_LIA".to_string()));
    }

    #[test]
    fn test_checkpoint_save_load() {
        use std::env;
        let temp_dir = env::temp_dir().join(format!("oxiz_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let checkpoint = Checkpoint::new(
            "(assert (= x 1))".to_string(),
            Some("QF_LIA".to_string()),
            SolverState::new(),
            ProgressInfo::new(2000, 75.0, "solving".to_string()),
            vec![],
        );

        let path = checkpoint.save(&temp_dir).unwrap();
        assert!(path.exists());

        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.problem, checkpoint.problem);
        assert_eq!(loaded.logic, checkpoint.logic);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_checkpoint_manager() {
        use std::env;
        let temp_dir = env::temp_dir().join(format!("oxiz_test_mgr_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut manager = CheckpointManager::new(temp_dir.clone(), 10, 3);

        assert!(manager.should_checkpoint());

        let checkpoint1 = Checkpoint::new(
            "(assert true)".to_string(),
            None,
            SolverState::new(),
            ProgressInfo::new(1000, 33.0, "solving".to_string()),
            vec![],
        );

        manager.save_checkpoint(checkpoint1).unwrap();

        // Immediately shouldn't checkpoint again
        assert!(!manager.should_checkpoint());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[ignore] // Ignored because it requires time-based delays
    fn test_checkpoint_cleanup() {
        use std::env;
        let temp_dir = env::temp_dir().join(format!("oxiz_test_cleanup_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let mut manager = CheckpointManager::new(temp_dir.clone(), 0, 2);

        // Create 3 checkpoints
        for i in 0..3 {
            // Sleep to ensure different timestamps
            if i > 0 {
                std::thread::sleep(std::time::Duration::from_millis(1100));
            }
            let checkpoint = Checkpoint::new(
                format!("(assert (= x {}))", i),
                None,
                SolverState::new(),
                ProgressInfo::new(1000 * i as u128, 33.0, "solving".to_string()),
                vec![],
            );
            manager.save_checkpoint(checkpoint).unwrap();
        }

        // Should have only 2 checkpoints (cleanup happens after each save)
        let checkpoints = Checkpoint::list_checkpoints(&temp_dir).unwrap();
        assert_eq!(checkpoints.len(), 2);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
