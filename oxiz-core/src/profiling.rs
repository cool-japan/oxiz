//! Shared profiling utilities for cross-crate hot-path instrumentation.

use portable_atomic::AtomicU64;
use portable_atomic::Ordering;
use std::fmt;
use std::sync::OnceLock;
use std::time::Instant;

const CATEGORY_COUNT: usize = 10;

/// Named profiling categories for hot-path measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum ProfilingCategory {
    /// SAT clause propagation.
    SatPropagation = 0,
    /// Theory checking dispatch and execution.
    TheoryCheck = 1,
    /// E-graph merge operations.
    EGraphMerge = 2,
    /// Simplex pivot operations.
    SimplexPivot = 3,
    /// Bit-vector propagation.
    BvPropagation = 4,
    /// String automata checks.
    StringAutomata = 5,
    /// Array extensionality checks.
    ArrayExtensionality = 6,
    /// Proof generation and recording.
    ProofGeneration = 7,
    /// SMT-LIB parser entry points.
    Parser = 8,
    /// Cache miss handling.
    CacheMiss = 9,
}

impl ProfilingCategory {
    /// Return every profiling category in a stable order.
    #[must_use]
    pub const fn all() -> &'static [Self; CATEGORY_COUNT] {
        &[
            Self::SatPropagation,
            Self::TheoryCheck,
            Self::EGraphMerge,
            Self::SimplexPivot,
            Self::BvPropagation,
            Self::StringAutomata,
            Self::ArrayExtensionality,
            Self::ProofGeneration,
            Self::Parser,
            Self::CacheMiss,
        ]
    }

    /// Return the stable display name for this category.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SatPropagation => "SatPropagation",
            Self::TheoryCheck => "TheoryCheck",
            Self::EGraphMerge => "EGraphMerge",
            Self::SimplexPivot => "SimplexPivot",
            Self::BvPropagation => "BvPropagation",
            Self::StringAutomata => "StringAutomata",
            Self::ArrayExtensionality => "ArrayExtensionality",
            Self::ProofGeneration => "ProofGeneration",
            Self::Parser => "Parser",
            Self::CacheMiss => "CacheMiss",
        }
    }

    #[must_use]
    const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for ProfilingCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn build_atomic_array() -> Box<[AtomicU64]> {
    (0..CATEGORY_COUNT)
        .map(|_| AtomicU64::new(0))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn count_counters() -> &'static [AtomicU64] {
    static COUNTS: OnceLock<Box<[AtomicU64]>> = OnceLock::new();
    COUNTS.get_or_init(build_atomic_array)
}

fn total_ns_counters() -> &'static [AtomicU64] {
    static TOTALS: OnceLock<Box<[AtomicU64]>> = OnceLock::new();
    TOTALS.get_or_init(build_atomic_array)
}

/// RAII timer that records aggregate category timings on drop.
#[derive(Debug)]
pub struct ScopedTimer {
    category: ProfilingCategory,
    start: Instant,
}

impl ScopedTimer {
    /// Start timing a profiling category.
    #[must_use]
    pub fn new(category: ProfilingCategory) -> Self {
        count_counters()[category.index()].fetch_add(1, Ordering::Relaxed);
        Self {
            category,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed_ns = self.start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        total_ns_counters()[self.category.index()].fetch_add(elapsed_ns, Ordering::Relaxed);
    }
}

/// Snapshot entry for one profiling category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfilingCategorySnapshot {
    /// Category label.
    pub category: ProfilingCategory,
    /// Number of timed samples.
    pub count: u64,
    /// Total time across all samples in nanoseconds.
    pub total_ns: u64,
}

/// Immutable view of current profiling counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilingSnapshot {
    entries: Vec<ProfilingCategorySnapshot>,
}

impl ProfilingSnapshot {
    /// Return the sampled count for one category.
    #[must_use]
    pub fn count(&self, category: ProfilingCategory) -> u64 {
        self.entry(category).count
    }

    /// Return the aggregated nanoseconds for one category.
    #[must_use]
    pub fn total_ns(&self, category: ProfilingCategory) -> u64 {
        self.entry(category).total_ns
    }

    /// Iterate over all category snapshots.
    pub fn iter(&self) -> impl Iterator<Item = &ProfilingCategorySnapshot> {
        self.entries.iter()
    }

    /// Serialize the snapshot into a compact JSON object.
    #[must_use]
    pub fn to_json(&self) -> String {
        let body = self
            .entries
            .iter()
            .map(|entry| {
                format!(
                    "\"{}\":{{\"count\":{},\"total_ns\":{}}}",
                    entry.category.as_str(),
                    entry.count,
                    entry.total_ns
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{body}}}")
    }

    fn entry(&self, category: ProfilingCategory) -> &ProfilingCategorySnapshot {
        &self.entries[category.index()]
    }
}

/// Aggregated profiling statistics.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProfilingStats;

impl ProfilingStats {
    /// Read the current profiling counters into an immutable snapshot.
    #[must_use]
    pub fn snapshot() -> ProfilingSnapshot {
        let counts = count_counters();
        let totals = total_ns_counters();
        let entries = ProfilingCategory::all()
            .iter()
            .copied()
            .map(|category| ProfilingCategorySnapshot {
                category,
                count: counts[category.index()].load(Ordering::Relaxed),
                total_ns: totals[category.index()].load(Ordering::Relaxed),
            })
            .collect();
        ProfilingSnapshot { entries }
    }
}
