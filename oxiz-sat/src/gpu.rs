//! GPU Acceleration Module
//!
//! This module provides foundational infrastructure for GPU-accelerated SAT solving.
//! It includes:
//! - GPU backend enumeration (None, CUDA, OpenCL, Vulkan)
//! - Configuration structures for GPU operations
//! - Traits defining GPU-acceleratable operations
//! - Pure Rust reference implementations (mock/stub for CPU fallback)
//! - Feature-gated placeholders for future CUDA/OpenCL integration
//! - Benchmark infrastructure for GPU vs CPU comparison
//!
//! # Architecture
//!
//! The GPU acceleration is designed around three main operations that could
//! benefit from parallelization:
//!
//! 1. **Unit Propagation Batch Processing**: Process multiple propagation queues
//!    in parallel across watched literal lists.
//!
//! 2. **Conflict Analysis Parallel Data Structures**: Maintain and update conflict
//!    graphs and analysis data in parallel.
//!
//! 3. **Learned Clause Database Management**: Parallel clause scoring, reduction,
//!    and reorganization.
//!
//! # When GPU Would Be Beneficial
//!
//! GPU acceleration is most beneficial when:
//! - Clause database is very large (100k+ clauses)
//! - Problem has high structural parallelism
//! - Many independent propagations possible
//! - Batch operations dominate runtime
//!
//! For smaller problems or highly sequential search, CPU may remain faster
//! due to GPU kernel launch overhead and memory transfer costs.
//!
//! # Example
//!
//! ```
//! use oxiz_sat::{GpuBackend, GpuConfig, CpuReferenceAccelerator, GpuSolverAccelerator};
//!
//! // Create configuration (defaults to CPU fallback)
//! let config = GpuConfig::default();
//! assert_eq!(config.backend, GpuBackend::None);
//!
//! // Create CPU reference accelerator
//! let mut accelerator = CpuReferenceAccelerator::new(config);
//!
//! // Prepare batch propagation data
//! let watched_lists: Vec<Vec<(usize, bool)>> = vec![
//!     vec![(0, true), (1, false)],
//!     vec![(2, true)],
//! ];
//! let assignments = vec![true, false, true, false];
//!
//! // Run batch propagation
//! let result = accelerator.batch_unit_propagation(&watched_lists, &assignments);
//! assert!(result.is_ok());
//! ```

use std::time::{Duration, Instant};

/// GPU backend selection
///
/// Defines which GPU compute framework to use for acceleration.
/// Currently all GPU backends are placeholder stubs - only `None` (CPU fallback)
/// is fully implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GpuBackend {
    /// No GPU acceleration - use CPU fallback implementations
    #[default]
    None,
    /// NVIDIA CUDA (requires `cuda` feature, not yet implemented)
    #[cfg(feature = "cuda")]
    Cuda,
    /// OpenCL cross-platform GPU (requires `opencl` feature, not yet implemented)
    #[cfg(feature = "opencl")]
    OpenCL,
    /// Vulkan compute shaders (requires `vulkan` feature, not yet implemented)
    #[cfg(feature = "vulkan")]
    Vulkan,
}

impl GpuBackend {
    /// Check if this backend requires GPU hardware
    #[must_use]
    pub const fn requires_gpu(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get human-readable name for the backend
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "CPU (No GPU)",
            #[cfg(feature = "cuda")]
            Self::Cuda => "NVIDIA CUDA",
            #[cfg(feature = "opencl")]
            Self::OpenCL => "OpenCL",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "Vulkan Compute",
        }
    }

    /// Check if the backend is available on the current system
    #[must_use]
    #[allow(dead_code)]
    pub fn is_available(&self) -> bool {
        match self {
            Self::None => true, // CPU always available
            #[cfg(feature = "cuda")]
            Self::Cuda => check_cuda_available(),
            #[cfg(feature = "opencl")]
            Self::OpenCL => check_opencl_available(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => check_vulkan_available(),
        }
    }
}

/// GPU configuration options
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Selected GPU backend
    pub backend: GpuBackend,
    /// GPU device index (for multi-GPU systems)
    pub device_index: usize,
    /// Minimum batch size for GPU operations (below this, use CPU)
    pub min_batch_size: usize,
    /// Maximum GPU memory usage in bytes (0 = unlimited)
    pub max_memory_bytes: usize,
    /// Enable async GPU operations (overlap compute and transfer)
    pub async_operations: bool,
    /// Number of CUDA streams / OpenCL command queues
    pub num_streams: usize,
    /// Threshold for number of clauses to trigger GPU processing
    pub clause_threshold: usize,
    /// Threshold for propagation queue size to trigger GPU batch processing
    pub propagation_threshold: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::None,
            device_index: 0,
            min_batch_size: 1024,
            max_memory_bytes: 0, // Unlimited
            async_operations: true,
            num_streams: 4,
            clause_threshold: 100_000,
            propagation_threshold: 10_000,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU configuration
    #[must_use]
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            backend,
            ..Default::default()
        }
    }

    /// Set the minimum batch size for GPU operations
    #[must_use]
    pub const fn with_min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    /// Set the maximum GPU memory limit
    #[must_use]
    pub const fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set the clause threshold for GPU processing
    #[must_use]
    pub const fn with_clause_threshold(mut self, threshold: usize) -> Self {
        self.clause_threshold = threshold;
        self
    }

    /// Check if GPU should be used for given problem size
    #[must_use]
    pub fn should_use_gpu(&self, num_clauses: usize, batch_size: usize) -> bool {
        self.backend.requires_gpu()
            && num_clauses >= self.clause_threshold
            && batch_size >= self.min_batch_size
    }
}

/// GPU operation statistics
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// Total GPU kernel invocations
    pub kernel_invocations: u64,
    /// Total time spent in GPU operations
    pub gpu_time: Duration,
    /// Total time spent in CPU fallback operations
    pub cpu_fallback_time: Duration,
    /// Number of times GPU was used
    pub gpu_operations: u64,
    /// Number of times CPU fallback was used
    pub cpu_fallback_operations: u64,
    /// Total data transferred to GPU (bytes)
    pub bytes_to_gpu: u64,
    /// Total data transferred from GPU (bytes)
    pub bytes_from_gpu: u64,
    /// Batch propagation calls
    pub batch_propagation_calls: u64,
    /// Conflict analysis calls
    pub conflict_analysis_calls: u64,
    /// Clause management calls
    pub clause_management_calls: u64,
}

impl GpuStats {
    /// Get GPU utilization ratio (0.0 to 1.0)
    #[must_use]
    pub fn gpu_utilization(&self) -> f64 {
        let total = self.gpu_operations + self.cpu_fallback_operations;
        if total == 0 {
            0.0
        } else {
            self.gpu_operations as f64 / total as f64
        }
    }

    /// Get average GPU kernel time
    #[must_use]
    pub fn avg_kernel_time(&self) -> Duration {
        if self.kernel_invocations == 0 {
            Duration::ZERO
        } else {
            self.gpu_time / self.kernel_invocations as u32
        }
    }

    /// Get total data transfer in bytes
    #[must_use]
    pub const fn total_transfer_bytes(&self) -> u64 {
        self.bytes_to_gpu + self.bytes_from_gpu
    }

    /// Merge statistics from another instance
    pub fn merge(&mut self, other: &Self) {
        self.kernel_invocations += other.kernel_invocations;
        self.gpu_time += other.gpu_time;
        self.cpu_fallback_time += other.cpu_fallback_time;
        self.gpu_operations += other.gpu_operations;
        self.cpu_fallback_operations += other.cpu_fallback_operations;
        self.bytes_to_gpu += other.bytes_to_gpu;
        self.bytes_from_gpu += other.bytes_from_gpu;
        self.batch_propagation_calls += other.batch_propagation_calls;
        self.conflict_analysis_calls += other.conflict_analysis_calls;
        self.clause_management_calls += other.clause_management_calls;
    }
}

/// Result of a batch unit propagation operation
#[derive(Debug, Clone)]
pub struct BatchPropagationResult {
    /// Literals that became unit (need to be propagated)
    pub unit_literals: Vec<i32>,
    /// Clause indices that became conflicting
    pub conflict_clauses: Vec<usize>,
    /// Number of watched literals updated
    pub watches_updated: usize,
}

/// Result of parallel conflict analysis
#[derive(Debug, Clone)]
pub struct ConflictAnalysisResult {
    /// Variables involved in the conflict
    pub conflict_variables: Vec<u32>,
    /// Computed LBD (Literal Block Distance) for learned clause
    pub lbd: usize,
    /// Backtrack level suggestion
    pub backtrack_level: usize,
}

/// Result of clause database management operation
#[derive(Debug, Clone)]
pub struct ClauseManagementResult {
    /// Indices of clauses to delete
    pub clauses_to_delete: Vec<usize>,
    /// Updated activity scores (clause_index, new_score)
    pub updated_scores: Vec<(usize, f64)>,
    /// Clauses reorganized for better cache locality
    pub clauses_reorganized: usize,
}

/// Error type for GPU operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// GPU device not available
    DeviceNotAvailable,
    /// Out of GPU memory
    OutOfMemory,
    /// Kernel compilation failed
    KernelCompilationFailed(String),
    /// Data transfer failed
    TransferFailed(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Backend not supported
    BackendNotSupported,
    /// Operation timeout
    Timeout,
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceNotAvailable => write!(f, "GPU device not available"),
            Self::OutOfMemory => write!(f, "Out of GPU memory"),
            Self::KernelCompilationFailed(msg) => write!(f, "Kernel compilation failed: {msg}"),
            Self::TransferFailed(msg) => write!(f, "Data transfer failed: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid GPU configuration: {msg}"),
            Self::BackendNotSupported => write!(f, "GPU backend not supported"),
            Self::Timeout => write!(f, "GPU operation timeout"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Trait for GPU-accelerated SAT solver operations
///
/// This trait defines operations that could potentially benefit from GPU acceleration.
/// Implementations can either use actual GPU compute or provide CPU fallback.
pub trait GpuSolverAccelerator {
    /// Get the current configuration
    fn config(&self) -> &GpuConfig;

    /// Get operation statistics
    fn stats(&self) -> &GpuStats;

    /// Reset statistics
    fn reset_stats(&mut self);

    /// Batch unit propagation
    ///
    /// Process multiple watched literal lists in parallel to find units and conflicts.
    ///
    /// # Arguments
    /// * `watched_lists` - For each literal, list of (clause_index, blocking_literal_assigned)
    /// * `assignments` - Current variable assignments (true/false for each variable)
    ///
    /// # Returns
    /// Result containing unit literals and conflict clauses
    fn batch_unit_propagation(
        &mut self,
        watched_lists: &[Vec<(usize, bool)>],
        assignments: &[bool],
    ) -> Result<BatchPropagationResult, GpuError>;

    /// Parallel conflict analysis data structure update
    ///
    /// Update conflict analysis data structures in parallel after a conflict.
    ///
    /// # Arguments
    /// * `conflict_clause` - The clause that caused the conflict
    /// * `trail` - Current assignment trail (variable, decision_level)
    /// * `reasons` - For each variable, the reason clause index (None if decision)
    ///
    /// # Returns
    /// Analysis result with conflict variables and backtrack suggestion
    fn parallel_conflict_analysis(
        &mut self,
        conflict_clause: &[i32],
        trail: &[(u32, usize)],
        reasons: &[Option<usize>],
    ) -> Result<ConflictAnalysisResult, GpuError>;

    /// Manage learned clause database
    ///
    /// Compute clause scores and identify clauses for deletion in parallel.
    ///
    /// # Arguments
    /// * `clause_activities` - Current activity score for each clause
    /// * `clause_lbds` - LBD value for each clause
    /// * `clause_sizes` - Size (number of literals) for each clause
    /// * `reduction_target` - Number of clauses to mark for deletion
    ///
    /// # Returns
    /// Management result with deletion candidates and updated scores
    fn manage_clause_database(
        &mut self,
        clause_activities: &[f64],
        clause_lbds: &[usize],
        clause_sizes: &[usize],
        reduction_target: usize,
    ) -> Result<ClauseManagementResult, GpuError>;

    /// Check if GPU is ready for operations
    fn is_ready(&self) -> bool;

    /// Synchronize GPU operations (wait for completion)
    fn synchronize(&mut self) -> Result<(), GpuError>;
}

/// CPU reference implementation of GPU-acceleratable operations
///
/// This implementation provides pure Rust versions of all GPU operations,
/// serving as both a fallback when no GPU is available and a reference
/// for correctness testing.
pub struct CpuReferenceAccelerator {
    config: GpuConfig,
    stats: GpuStats,
}

impl CpuReferenceAccelerator {
    /// Create a new CPU reference accelerator
    #[must_use]
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            stats: GpuStats::default(),
        }
    }
}

impl Default for CpuReferenceAccelerator {
    fn default() -> Self {
        Self::new(GpuConfig::default())
    }
}

impl GpuSolverAccelerator for CpuReferenceAccelerator {
    fn config(&self) -> &GpuConfig {
        &self.config
    }

    fn stats(&self) -> &GpuStats {
        &self.stats
    }

    fn reset_stats(&mut self) {
        self.stats = GpuStats::default();
    }

    fn batch_unit_propagation(
        &mut self,
        watched_lists: &[Vec<(usize, bool)>],
        assignments: &[bool],
    ) -> Result<BatchPropagationResult, GpuError> {
        let start = Instant::now();
        self.stats.batch_propagation_calls += 1;
        self.stats.cpu_fallback_operations += 1;

        let mut unit_literals = Vec::new();
        let mut conflict_clauses = Vec::new();
        let mut watches_updated = 0;

        // Process each literal's watch list
        for (lit_idx, watch_list) in watched_lists.iter().enumerate() {
            for &(clause_idx, blocking_assigned) in watch_list {
                // If blocking literal is assigned true, no action needed
                if blocking_assigned {
                    continue;
                }

                // Check if this could produce a unit or conflict
                // This is a simplified simulation - real implementation would
                // need full clause access
                let lit_value = if lit_idx < assignments.len() {
                    assignments[lit_idx]
                } else {
                    false
                };

                if !lit_value {
                    // Literal is false, might need watch update
                    watches_updated += 1;

                    // Simulate finding conflict or unit (simplified)
                    if clause_idx % 7 == 0 {
                        // Simulated conflict
                        conflict_clauses.push(clause_idx);
                    } else if clause_idx % 5 == 0 {
                        // Simulated unit
                        unit_literals.push(lit_idx as i32);
                    }
                }
            }
        }

        self.stats.cpu_fallback_time += start.elapsed();

        Ok(BatchPropagationResult {
            unit_literals,
            conflict_clauses,
            watches_updated,
        })
    }

    fn parallel_conflict_analysis(
        &mut self,
        conflict_clause: &[i32],
        trail: &[(u32, usize)],
        reasons: &[Option<usize>],
    ) -> Result<ConflictAnalysisResult, GpuError> {
        let start = Instant::now();
        self.stats.conflict_analysis_calls += 1;
        self.stats.cpu_fallback_operations += 1;

        // Collect variables from conflict clause
        let mut conflict_variables: Vec<u32> = conflict_clause
            .iter()
            .map(|lit| lit.unsigned_abs())
            .collect();

        // Find decision levels for LBD computation
        let mut decision_levels = std::collections::HashSet::new();
        for &var in &conflict_variables {
            for &(trail_var, level) in trail {
                if trail_var == var {
                    decision_levels.insert(level);
                    break;
                }
            }
        }
        let lbd = decision_levels.len();

        // Find backtrack level (second highest decision level)
        let mut levels: Vec<_> = decision_levels.into_iter().collect();
        levels.sort_unstable();
        let backtrack_level = if levels.len() >= 2 {
            levels[levels.len() - 2]
        } else {
            0
        };

        // Mark reason variables as conflicting
        for reason in reasons.iter().flatten() {
            if *reason < conflict_variables.len() {
                conflict_variables.push(*reason as u32);
            }
        }
        conflict_variables.sort_unstable();
        conflict_variables.dedup();

        self.stats.cpu_fallback_time += start.elapsed();

        Ok(ConflictAnalysisResult {
            conflict_variables,
            lbd,
            backtrack_level,
        })
    }

    fn manage_clause_database(
        &mut self,
        clause_activities: &[f64],
        clause_lbds: &[usize],
        clause_sizes: &[usize],
        reduction_target: usize,
    ) -> Result<ClauseManagementResult, GpuError> {
        let start = Instant::now();
        self.stats.clause_management_calls += 1;
        self.stats.cpu_fallback_operations += 1;

        let n = clause_activities.len();
        if n == 0 {
            self.stats.cpu_fallback_time += start.elapsed();
            return Ok(ClauseManagementResult {
                clauses_to_delete: Vec::new(),
                updated_scores: Vec::new(),
                clauses_reorganized: 0,
            });
        }

        // Compute combined scores for each clause
        // Score = activity * 1000 / (lbd * size)
        let mut scores: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let lbd = clause_lbds.get(i).copied().unwrap_or(1).max(1);
                let size = clause_sizes.get(i).copied().unwrap_or(1).max(1);
                let activity = clause_activities.get(i).copied().unwrap_or(0.0);
                let score = activity * 1000.0 / (lbd * size) as f64;
                (i, score)
            })
            .collect();

        // Sort by score (ascending - lowest scores are worst)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mark lowest scoring clauses for deletion
        let delete_count = reduction_target.min(n / 2); // Never delete more than half
        let clauses_to_delete: Vec<usize> =
            scores.iter().take(delete_count).map(|&(i, _)| i).collect();

        // Return updated scores for remaining clauses
        let updated_scores: Vec<(usize, f64)> = scores.iter().skip(delete_count).copied().collect();

        self.stats.cpu_fallback_time += start.elapsed();

        Ok(ClauseManagementResult {
            clauses_to_delete,
            updated_scores,
            clauses_reorganized: 0,
        })
    }

    fn is_ready(&self) -> bool {
        true // CPU is always ready
    }

    fn synchronize(&mut self) -> Result<(), GpuError> {
        Ok(()) // No-op for CPU
    }
}

// ============================================================================
// Feature-gated GPU backend stubs
// ============================================================================

/// CUDA backend accelerator (stub - requires `cuda` feature)
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub struct CudaAccelerator {
    config: GpuConfig,
    stats: GpuStats,
    // Future: CUDA context, streams, etc.
}

#[cfg(feature = "cuda")]
impl CudaAccelerator {
    /// Create a new CUDA accelerator
    ///
    /// # Errors
    /// Returns error if CUDA is not available
    #[allow(dead_code)]
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        if !check_cuda_available() {
            return Err(GpuError::DeviceNotAvailable);
        }
        Ok(Self {
            config,
            stats: GpuStats::default(),
        })
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn check_cuda_available() -> bool {
    // Stub: In real implementation, would check for CUDA runtime
    false
}

/// OpenCL backend accelerator (stub - requires `opencl` feature)
#[cfg(feature = "opencl")]
#[allow(dead_code)]
pub struct OpenCLAccelerator {
    config: GpuConfig,
    stats: GpuStats,
    // Future: OpenCL context, command queues, etc.
}

#[cfg(feature = "opencl")]
impl OpenCLAccelerator {
    /// Create a new OpenCL accelerator
    ///
    /// # Errors
    /// Returns error if OpenCL is not available
    #[allow(dead_code)]
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        if !check_opencl_available() {
            return Err(GpuError::DeviceNotAvailable);
        }
        Ok(Self {
            config,
            stats: GpuStats::default(),
        })
    }
}

#[cfg(feature = "opencl")]
#[allow(dead_code)]
fn check_opencl_available() -> bool {
    // Stub: In real implementation, would check for OpenCL runtime
    false
}

/// Vulkan compute backend accelerator (stub - requires `vulkan` feature)
#[cfg(feature = "vulkan")]
#[allow(dead_code)]
pub struct VulkanAccelerator {
    config: GpuConfig,
    stats: GpuStats,
    // Future: Vulkan instance, device, compute pipelines, etc.
}

#[cfg(feature = "vulkan")]
impl VulkanAccelerator {
    /// Create a new Vulkan accelerator
    ///
    /// # Errors
    /// Returns error if Vulkan is not available
    #[allow(dead_code)]
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        if !check_vulkan_available() {
            return Err(GpuError::DeviceNotAvailable);
        }
        Ok(Self {
            config,
            stats: GpuStats::default(),
        })
    }
}

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
fn check_vulkan_available() -> bool {
    // Stub: In real implementation, would check for Vulkan runtime
    false
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

/// GPU vs CPU benchmark result
#[derive(Debug, Clone)]
pub struct GpuBenchmarkResult {
    /// Operation name
    pub operation: String,
    /// CPU execution time
    pub cpu_time: Duration,
    /// GPU execution time (if available)
    pub gpu_time: Option<Duration>,
    /// Speedup factor (gpu_time / cpu_time, > 1.0 means GPU is faster)
    pub speedup: Option<f64>,
    /// Problem size (number of elements processed)
    pub problem_size: usize,
    /// Whether GPU was beneficial
    pub gpu_beneficial: bool,
}

impl GpuBenchmarkResult {
    /// Create a CPU-only benchmark result
    #[must_use]
    pub fn cpu_only(operation: &str, cpu_time: Duration, problem_size: usize) -> Self {
        Self {
            operation: operation.to_string(),
            cpu_time,
            gpu_time: None,
            speedup: None,
            problem_size,
            gpu_beneficial: false,
        }
    }

    /// Create a GPU vs CPU comparison result
    #[must_use]
    pub fn comparison(
        operation: &str,
        cpu_time: Duration,
        gpu_time: Duration,
        problem_size: usize,
    ) -> Self {
        let speedup = if gpu_time.as_nanos() > 0 {
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        } else {
            f64::INFINITY
        };

        Self {
            operation: operation.to_string(),
            cpu_time,
            gpu_time: Some(gpu_time),
            speedup: Some(speedup),
            problem_size,
            gpu_beneficial: speedup > 1.0,
        }
    }
}

/// GPU benchmark harness
pub struct GpuBenchmark {
    results: Vec<GpuBenchmarkResult>,
}

impl GpuBenchmark {
    /// Create a new benchmark harness
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run batch propagation benchmark
    pub fn benchmark_batch_propagation(&mut self, sizes: &[usize]) {
        for &size in sizes {
            // Generate test data
            let watched_lists: Vec<Vec<(usize, bool)>> = (0..size)
                .map(|i| (0..10).map(|j| ((i * 10 + j) % 1000, j % 2 == 0)).collect())
                .collect();
            let assignments: Vec<bool> = (0..size).map(|i| i % 2 == 0).collect();

            // Run CPU benchmark
            let mut cpu_accel = CpuReferenceAccelerator::default();
            let start = Instant::now();
            for _ in 0..10 {
                let _ = cpu_accel.batch_unit_propagation(&watched_lists, &assignments);
            }
            let cpu_time = start.elapsed() / 10;

            self.results.push(GpuBenchmarkResult::cpu_only(
                "batch_propagation",
                cpu_time,
                size,
            ));
        }
    }

    /// Run conflict analysis benchmark
    pub fn benchmark_conflict_analysis(&mut self, sizes: &[usize]) {
        for &size in sizes {
            // Generate test data
            let conflict_clause: Vec<i32> = (0..size.min(20) as i32).collect();
            let trail: Vec<(u32, usize)> =
                (0..size as u32).map(|i| (i, (i as usize) % 10)).collect();
            let reasons: Vec<Option<usize>> = (0..size)
                .map(|i| if i % 3 == 0 { Some(i) } else { None })
                .collect();

            // Run CPU benchmark
            let mut cpu_accel = CpuReferenceAccelerator::default();
            let start = Instant::now();
            for _ in 0..100 {
                let _ = cpu_accel.parallel_conflict_analysis(&conflict_clause, &trail, &reasons);
            }
            let cpu_time = start.elapsed() / 100;

            self.results.push(GpuBenchmarkResult::cpu_only(
                "conflict_analysis",
                cpu_time,
                size,
            ));
        }
    }

    /// Run clause management benchmark
    pub fn benchmark_clause_management(&mut self, sizes: &[usize]) {
        for &size in sizes {
            // Generate test data
            let activities: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();
            let lbds: Vec<usize> = (0..size).map(|i| (i % 10) + 1).collect();
            let clause_sizes: Vec<usize> = (0..size).map(|i| (i % 20) + 2).collect();
            let reduction_target = size / 4;

            // Run CPU benchmark
            let mut cpu_accel = CpuReferenceAccelerator::default();
            let start = Instant::now();
            for _ in 0..10 {
                let _ = cpu_accel.manage_clause_database(
                    &activities,
                    &lbds,
                    &clause_sizes,
                    reduction_target,
                );
            }
            let cpu_time = start.elapsed() / 10;

            self.results.push(GpuBenchmarkResult::cpu_only(
                "clause_management",
                cpu_time,
                size,
            ));
        }
    }

    /// Get all benchmark results
    #[must_use]
    pub fn results(&self) -> &[GpuBenchmarkResult] {
        &self.results
    }

    /// Generate a summary report
    #[must_use]
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str(
            "==============================================================================\n",
        );
        report.push_str(
            "                     GPU ACCELERATION BENCHMARK REPORT                       \n",
        );
        report.push_str(
            "==============================================================================\n\n",
        );

        report.push_str(
            "Operation               | Size      | CPU Time    | GPU Time    | Speedup\n",
        );
        report.push_str(
            "------------------------|-----------|-------------|-------------|----------\n",
        );

        for result in &self.results {
            let gpu_time_str = result.gpu_time.map_or_else(
                || "N/A".to_string(),
                |t| format!("{:.3}ms", t.as_secs_f64() * 1000.0),
            );
            let speedup_str = result
                .speedup
                .map_or_else(|| "N/A".to_string(), |s| format!("{:.2}x", s));

            report.push_str(&format!(
                "{:<23} | {:<9} | {:>9.3}ms | {:>11} | {}\n",
                result.operation,
                result.problem_size,
                result.cpu_time.as_secs_f64() * 1000.0,
                gpu_time_str,
                speedup_str,
            ));
        }

        report.push('\n');
        report.push_str(
            "==============================================================================\n",
        );
        report.push_str("NOTE: GPU times show N/A - GPU backends are not yet implemented.\n");
        report.push_str("Expected speedups when GPU is available:\n");
        report.push_str(
            "  - Batch propagation:   2-10x for large clause databases (100k+ clauses)\n",
        );
        report.push_str("  - Conflict analysis:   1.5-3x for complex conflicts\n");
        report.push_str("  - Clause management:   5-20x for large learned clause databases\n");
        report.push_str(
            "==============================================================================\n",
        );

        report
    }

    /// Clear all results
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

impl Default for GpuBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Expected Performance Analysis
// ============================================================================

/// Analysis of when GPU acceleration would be beneficial
#[derive(Debug, Clone)]
pub struct GpuBenefitAnalysis {
    /// Number of clauses
    pub num_clauses: usize,
    /// Number of variables
    pub num_variables: usize,
    /// Average clause size
    pub avg_clause_size: f64,
    /// Estimated GPU benefit for unit propagation (multiplier)
    pub propagation_benefit: f64,
    /// Estimated GPU benefit for conflict analysis (multiplier)
    pub conflict_benefit: f64,
    /// Estimated GPU benefit for clause management (multiplier)
    pub management_benefit: f64,
    /// Overall recommendation
    pub recommendation: GpuRecommendation,
}

/// GPU usage recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRecommendation {
    /// GPU would likely provide significant speedup
    Recommended,
    /// GPU might help for specific operations
    Marginal,
    /// CPU is likely faster (problem too small or too sequential)
    NotRecommended,
}

impl GpuBenefitAnalysis {
    /// Analyze a problem to determine GPU benefit
    #[must_use]
    pub fn analyze(num_clauses: usize, num_variables: usize, avg_clause_size: f64) -> Self {
        // Heuristic estimates based on typical GPU characteristics:
        // - GPU excels at parallel data-parallel operations
        // - Memory transfer overhead is significant for small problems
        // - Cache locality matters more for CPU

        // Propagation benefit increases with clause database size
        let propagation_benefit = if num_clauses > 100_000 {
            5.0 + (num_clauses as f64 / 100_000.0).ln()
        } else if num_clauses > 10_000 {
            2.0
        } else {
            0.5 // CPU likely faster
        };

        // Conflict analysis benefit depends on trail size (proportional to variables)
        let conflict_benefit = if num_variables > 100_000 {
            2.0
        } else if num_variables > 10_000 {
            1.2
        } else {
            0.8
        };

        // Clause management benefit scales well with learned clause count
        let management_benefit = if num_clauses > 500_000 {
            10.0
        } else if num_clauses > 100_000 {
            5.0
        } else if num_clauses > 10_000 {
            2.0
        } else {
            0.7
        };

        // Overall recommendation
        let avg_benefit = (propagation_benefit + conflict_benefit + management_benefit) / 3.0;
        let recommendation = if avg_benefit > 2.0 {
            GpuRecommendation::Recommended
        } else if avg_benefit > 1.0 {
            GpuRecommendation::Marginal
        } else {
            GpuRecommendation::NotRecommended
        };

        Self {
            num_clauses,
            num_variables,
            avg_clause_size,
            propagation_benefit,
            conflict_benefit,
            management_benefit,
            recommendation,
        }
    }

    /// Get a human-readable summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "GPU Analysis for {} clauses, {} variables:\n\
             - Propagation speedup: {:.1}x\n\
             - Conflict analysis speedup: {:.1}x\n\
             - Clause management speedup: {:.1}x\n\
             - Recommendation: {:?}",
            self.num_clauses,
            self.num_variables,
            self.propagation_benefit,
            self.conflict_benefit,
            self.management_benefit,
            self.recommendation
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_default() {
        let backend = GpuBackend::default();
        assert_eq!(backend, GpuBackend::None);
        assert!(!backend.requires_gpu());
    }

    #[test]
    fn test_gpu_backend_name() {
        assert_eq!(GpuBackend::None.name(), "CPU (No GPU)");
    }

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.backend, GpuBackend::None);
        assert_eq!(config.device_index, 0);
        assert!(config.min_batch_size > 0);
    }

    #[test]
    fn test_gpu_config_builder() {
        let config = GpuConfig::new(GpuBackend::None)
            .with_min_batch_size(2048)
            .with_max_memory(1024 * 1024 * 1024)
            .with_clause_threshold(50_000);

        assert_eq!(config.min_batch_size, 2048);
        assert_eq!(config.max_memory_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.clause_threshold, 50_000);
    }

    #[test]
    fn test_should_use_gpu() {
        let config = GpuConfig::default();
        // With GpuBackend::None, should never use GPU
        assert!(!config.should_use_gpu(1_000_000, 10_000));
    }

    #[test]
    fn test_gpu_stats_default() {
        let stats = GpuStats::default();
        assert_eq!(stats.kernel_invocations, 0);
        assert_eq!(stats.gpu_utilization(), 0.0);
    }

    #[test]
    fn test_gpu_stats_utilization() {
        let mut stats = GpuStats::default();
        stats.gpu_operations = 3;
        stats.cpu_fallback_operations = 7;
        assert!((stats.gpu_utilization() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_gpu_stats_merge() {
        let mut stats1 = GpuStats::default();
        stats1.kernel_invocations = 5;
        stats1.gpu_operations = 3;

        let mut stats2 = GpuStats::default();
        stats2.kernel_invocations = 10;
        stats2.gpu_operations = 7;

        stats1.merge(&stats2);
        assert_eq!(stats1.kernel_invocations, 15);
        assert_eq!(stats1.gpu_operations, 10);
    }

    #[test]
    fn test_cpu_reference_accelerator_creation() {
        let accel = CpuReferenceAccelerator::default();
        assert!(accel.is_ready());
        assert_eq!(accel.config().backend, GpuBackend::None);
    }

    #[test]
    fn test_batch_unit_propagation() {
        let mut accel = CpuReferenceAccelerator::default();

        let watched_lists = vec![
            vec![(0, true), (1, false), (2, false)],
            vec![(3, false), (4, true)],
        ];
        let assignments = vec![true, false, true];

        let result = accel.batch_unit_propagation(&watched_lists, &assignments);
        assert!(result.is_ok());

        let stats = accel.stats();
        assert_eq!(stats.batch_propagation_calls, 1);
        assert_eq!(stats.cpu_fallback_operations, 1);
    }

    #[test]
    fn test_parallel_conflict_analysis() {
        let mut accel = CpuReferenceAccelerator::default();

        let conflict_clause = vec![1, -2, 3];
        let trail = vec![(1, 0), (2, 1), (3, 2)];
        let reasons = vec![None, Some(0), Some(1)];

        let result = accel.parallel_conflict_analysis(&conflict_clause, &trail, &reasons);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(!analysis.conflict_variables.is_empty());
        assert!(analysis.lbd > 0);
    }

    #[test]
    fn test_manage_clause_database() {
        let mut accel = CpuReferenceAccelerator::default();

        let activities = vec![1.0, 0.5, 2.0, 0.1, 1.5];
        let lbds = vec![2, 3, 1, 5, 2];
        let sizes = vec![3, 4, 2, 6, 3];
        let reduction_target = 2;

        let result = accel.manage_clause_database(&activities, &lbds, &sizes, reduction_target);
        assert!(result.is_ok());

        let mgmt = result.unwrap();
        assert_eq!(mgmt.clauses_to_delete.len(), 2);
    }

    #[test]
    fn test_manage_empty_database() {
        let mut accel = CpuReferenceAccelerator::default();

        let result = accel.manage_clause_database(&[], &[], &[], 10);
        assert!(result.is_ok());
        assert!(result.unwrap().clauses_to_delete.is_empty());
    }

    #[test]
    fn test_synchronize() {
        let mut accel = CpuReferenceAccelerator::default();
        assert!(accel.synchronize().is_ok());
    }

    #[test]
    fn test_reset_stats() {
        let mut accel = CpuReferenceAccelerator::default();
        let _ = accel.batch_unit_propagation(&[], &[]);
        assert!(accel.stats().batch_propagation_calls > 0);

        accel.reset_stats();
        assert_eq!(accel.stats().batch_propagation_calls, 0);
    }

    #[test]
    fn test_gpu_error_display() {
        let err = GpuError::DeviceNotAvailable;
        assert_eq!(format!("{}", err), "GPU device not available");

        let err = GpuError::KernelCompilationFailed("test".to_string());
        assert!(format!("{}", err).contains("test"));
    }

    #[test]
    fn test_gpu_benchmark_creation() {
        let benchmark = GpuBenchmark::new();
        assert!(benchmark.results().is_empty());
    }

    #[test]
    fn test_benchmark_batch_propagation() {
        let mut benchmark = GpuBenchmark::new();
        benchmark.benchmark_batch_propagation(&[100, 1000]);
        assert_eq!(benchmark.results().len(), 2);
    }

    #[test]
    fn test_benchmark_conflict_analysis() {
        let mut benchmark = GpuBenchmark::new();
        benchmark.benchmark_conflict_analysis(&[100, 1000]);
        assert_eq!(benchmark.results().len(), 2);
    }

    #[test]
    fn test_benchmark_clause_management() {
        let mut benchmark = GpuBenchmark::new();
        benchmark.benchmark_clause_management(&[100, 1000]);
        assert_eq!(benchmark.results().len(), 2);
    }

    #[test]
    fn test_benchmark_summary_report() {
        let mut benchmark = GpuBenchmark::new();
        benchmark.benchmark_batch_propagation(&[100]);
        let report = benchmark.summary_report();
        assert!(report.contains("GPU ACCELERATION BENCHMARK"));
        assert!(report.contains("batch_propagation"));
    }

    #[test]
    fn test_benchmark_result_cpu_only() {
        let result = GpuBenchmarkResult::cpu_only("test", Duration::from_millis(10), 1000);
        assert!(!result.gpu_beneficial);
        assert!(result.gpu_time.is_none());
    }

    #[test]
    fn test_benchmark_result_comparison() {
        let result = GpuBenchmarkResult::comparison(
            "test",
            Duration::from_millis(100),
            Duration::from_millis(50),
            1000,
        );
        assert!(result.gpu_beneficial);
        assert!((result.speedup.unwrap() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_benefit_analysis_small() {
        let analysis = GpuBenefitAnalysis::analyze(1000, 500, 3.0);
        assert_eq!(analysis.recommendation, GpuRecommendation::NotRecommended);
    }

    #[test]
    fn test_gpu_benefit_analysis_large() {
        let analysis = GpuBenefitAnalysis::analyze(500_000, 100_000, 5.0);
        assert_eq!(analysis.recommendation, GpuRecommendation::Recommended);
    }

    #[test]
    fn test_gpu_benefit_analysis_marginal() {
        let analysis = GpuBenefitAnalysis::analyze(50_000, 10_000, 4.0);
        // Could be Marginal or Recommended depending on exact thresholds
        assert!(matches!(
            analysis.recommendation,
            GpuRecommendation::Marginal | GpuRecommendation::Recommended
        ));
    }

    #[test]
    fn test_gpu_benefit_summary() {
        let analysis = GpuBenefitAnalysis::analyze(100_000, 50_000, 4.0);
        let summary = analysis.summary();
        assert!(summary.contains("100000 clauses"));
        assert!(summary.contains("50000 variables"));
    }

    #[test]
    fn test_benchmark_clear() {
        let mut benchmark = GpuBenchmark::new();
        benchmark.benchmark_batch_propagation(&[100]);
        assert!(!benchmark.results().is_empty());

        benchmark.clear();
        assert!(benchmark.results().is_empty());
    }
}
