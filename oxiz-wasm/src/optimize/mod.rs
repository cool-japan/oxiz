//! WASM Size Optimization Tools
//!
//! This module provides tools and utilities for optimizing WebAssembly bundle size.
//! The goal is to achieve sub-2MB compressed bundle size while maintaining full functionality.
//!
//! # Optimization Strategies
//!
//! 1. **Dead Code Elimination**: Remove unused functions and data
//! 2. **Symbol Stripping**: Remove debug symbols and unnecessary metadata
//! 3. **LTO Configuration**: Link-time optimization settings
//! 4. **Compression**: Brotli/gzip compression helpers
//!
//! # Target Metrics
//!
//! - Uncompressed: < 6MB
//! - Gzip: < 2.5MB
//! - Brotli: < 2MB
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::optimize::{OptimizationConfig, optimize_wasm};
//!
//! let config = OptimizationConfig::aggressive();
//! let optimized = optimize_wasm(&wasm_bytes, &config)?;
//! ```

pub mod compression;
pub mod dead_code_elim;
pub mod lto_config;
pub mod symbol_stripping;

pub use compression::*;
pub use dead_code_elim::*;
pub use lto_config::*;
pub use symbol_stripping::*;

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization (fastest build)
    None,
    /// Basic optimization (balanced)
    Basic,
    /// Aggressive optimization (smallest size)
    Aggressive,
    /// Maximum optimization (slowest build, smallest size)
    Maximum,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Optimization level
    pub level: OptLevel,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Strip debug symbols
    pub strip_symbols: bool,
    /// Enable LTO
    pub lto: bool,
    /// Enable compression
    pub compress: bool,
    /// Compression format
    pub compression_format: CompressionFormat,
    /// Custom opt-level override
    pub custom_opt_level: Option<String>,
}

impl OptimizationConfig {
    /// Create a new optimization configuration
    pub fn new(level: OptLevel) -> Self {
        let (dead_code, strip, lto, compress) = match level {
            OptLevel::None => (false, false, false, false),
            OptLevel::Basic => (true, false, false, false),
            OptLevel::Aggressive => (true, true, true, true),
            OptLevel::Maximum => (true, true, true, true),
        };

        Self {
            level,
            dead_code_elimination: dead_code,
            strip_symbols: strip,
            lto,
            compress,
            compression_format: CompressionFormat::Brotli,
            custom_opt_level: None,
        }
    }

    /// Create configuration for development (fast builds)
    pub fn development() -> Self {
        Self::new(OptLevel::None)
    }

    /// Create configuration for production (balanced)
    pub fn production() -> Self {
        Self::new(OptLevel::Basic)
    }

    /// Create configuration for aggressive size optimization
    pub fn aggressive() -> Self {
        Self::new(OptLevel::Aggressive)
    }

    /// Create configuration for maximum size optimization
    pub fn maximum() -> Self {
        let mut config = Self::new(OptLevel::Maximum);
        config.custom_opt_level = Some("z".to_string());
        config
    }

    /// Get Cargo opt-level string
    pub fn cargo_opt_level(&self) -> &str {
        if let Some(ref custom) = self.custom_opt_level {
            custom
        } else {
            match self.level {
                OptLevel::None => "0",
                OptLevel::Basic => "2",
                OptLevel::Aggressive => "s",
                OptLevel::Maximum => "z",
            }
        }
    }

    /// Get estimated size reduction percentage
    pub fn estimated_reduction(&self) -> f64 {
        match self.level {
            OptLevel::None => 0.0,
            OptLevel::Basic => 25.0,
            OptLevel::Aggressive => 50.0,
            OptLevel::Maximum => 70.0,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::production()
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Optimized size in bytes
    pub optimized_size: usize,
    /// Compressed size in bytes (if compression enabled)
    pub compressed_size: Option<usize>,
    /// Time taken for optimization in milliseconds
    pub optimization_time_ms: f64,
    /// Functions eliminated
    pub functions_eliminated: usize,
    /// Symbols stripped
    pub symbols_stripped: usize,
}

impl OptimizationStats {
    /// Calculate size reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        ((self.original_size - self.optimized_size) as f64 / self.original_size as f64) * 100.0
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> Option<f64> {
        self.compressed_size.map(|compressed| {
            if compressed == 0 {
                0.0
            } else {
                self.optimized_size as f64 / compressed as f64
            }
        })
    }

    /// Get human-readable size string
    pub fn format_size(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;

        if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_config_levels() {
        let dev = OptimizationConfig::development();
        assert_eq!(dev.level, OptLevel::None);
        assert!(!dev.dead_code_elimination);

        let prod = OptimizationConfig::production();
        assert_eq!(prod.level, OptLevel::Basic);

        let aggressive = OptimizationConfig::aggressive();
        assert_eq!(aggressive.level, OptLevel::Aggressive);
        assert!(aggressive.dead_code_elimination);
        assert!(aggressive.strip_symbols);
    }

    #[test]
    fn test_cargo_opt_level() {
        assert_eq!(OptimizationConfig::development().cargo_opt_level(), "0");
        assert_eq!(OptimizationConfig::production().cargo_opt_level(), "2");
        assert_eq!(OptimizationConfig::aggressive().cargo_opt_level(), "s");
        assert_eq!(OptimizationConfig::maximum().cargo_opt_level(), "z");
    }

    #[test]
    fn test_optimization_stats() {
        let stats = OptimizationStats {
            original_size: 10_000_000,
            optimized_size: 5_000_000,
            compressed_size: Some(2_000_000),
            optimization_time_ms: 1500.0,
            functions_eliminated: 100,
            symbols_stripped: 500,
        };

        assert_eq!(stats.reduction_percent(), 50.0);
        assert_eq!(stats.compression_ratio(), Some(2.5));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(OptimizationStats::format_size(500), "500 B");
        assert_eq!(OptimizationStats::format_size(2048), "2.00 KB");
        assert_eq!(OptimizationStats::format_size(2_097_152), "2.00 MB");
    }
}
