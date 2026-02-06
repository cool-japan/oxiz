//! Link-Time Optimization Configuration
//!
//! This module provides configuration and utilities for Link-Time Optimization (LTO)
//! in WebAssembly builds. LTO performs whole-program optimization across compilation
//! units, enabling better inlining, dead code elimination, and code generation.
//!
//! # LTO Modes
//!
//! - **No LTO**: Fast compilation, larger binaries
//! - **Thin LTO**: Balanced compilation time and optimization
//! - **Fat LTO**: Maximum optimization, slower compilation
//!
//! # Size Impact
//!
//! - Thin LTO: ~10-20% size reduction
//! - Fat LTO: ~20-30% size reduction
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::optimize::lto_config::{LtoConfig, LtoMode};
//!
//! let config = LtoConfig::new()
//!     .with_mode(LtoMode::Fat)
//!     .with_codegen_units(1)
//!     .with_opt_level("z");
//!
//! println!("Cargo flags: {}", config.to_cargo_flags());
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;

/// LTO mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtoMode {
    /// No LTO (fastest compilation)
    Off,
    /// Thin LTO (balanced)
    Thin,
    /// Fat LTO (maximum optimization)
    Fat,
}

impl LtoMode {
    /// Get the mode name for Cargo.toml
    pub fn cargo_name(&self) -> &'static str {
        match self {
            LtoMode::Off => "false",
            LtoMode::Thin => "thin",
            LtoMode::Fat => "fat",
        }
    }

    /// Get estimated size reduction percentage
    pub fn size_reduction(&self) -> f64 {
        match self {
            LtoMode::Off => 0.0,
            LtoMode::Thin => 15.0,
            LtoMode::Fat => 25.0,
        }
    }

    /// Get estimated compilation time multiplier
    pub fn compile_time_multiplier(&self) -> f64 {
        match self {
            LtoMode::Off => 1.0,
            LtoMode::Thin => 1.5,
            LtoMode::Fat => 3.0,
        }
    }
}

/// Optimization level for code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Default optimization
    O2,
    /// Aggressive optimization
    O3,
    /// Optimize for size
    Os,
    /// Optimize aggressively for size
    Oz,
}

impl OptLevel {
    /// Get the level name for Cargo.toml
    pub fn cargo_name(&self) -> &'static str {
        match self {
            OptLevel::O0 => "0",
            OptLevel::O1 => "1",
            OptLevel::O2 => "2",
            OptLevel::O3 => "3",
            OptLevel::Os => "s",
            OptLevel::Oz => "z",
        }
    }

    /// Get numeric value (0-3, Os=2, Oz=2)
    pub fn numeric_value(&self) -> u8 {
        match self {
            OptLevel::O0 => 0,
            OptLevel::O1 => 1,
            OptLevel::O2 => 2,
            OptLevel::O3 => 3,
            OptLevel::Os => 2,
            OptLevel::Oz => 2,
        }
    }

    /// Check if this is a size optimization level
    pub fn is_size_optimized(&self) -> bool {
        matches!(self, OptLevel::Os | OptLevel::Oz)
    }
}

/// Panic strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanicStrategy {
    /// Unwind panic (larger, allows unwinding)
    Unwind,
    /// Abort panic (smaller, immediate abort)
    Abort,
}

impl PanicStrategy {
    /// Get the strategy name for Cargo.toml
    pub fn cargo_name(&self) -> &'static str {
        match self {
            PanicStrategy::Unwind => "unwind",
            PanicStrategy::Abort => "abort",
        }
    }

    /// Get estimated size impact in bytes
    pub fn size_impact(&self) -> i32 {
        match self {
            PanicStrategy::Unwind => 50_000, // ~50KB overhead
            PanicStrategy::Abort => 0,
        }
    }
}

/// LTO configuration
#[derive(Debug, Clone)]
pub struct LtoConfig {
    /// LTO mode
    pub lto_mode: LtoMode,
    /// Optimization level
    pub opt_level: OptLevel,
    /// Number of codegen units (1 = maximum optimization)
    pub codegen_units: usize,
    /// Panic strategy
    pub panic_strategy: PanicStrategy,
    /// Strip symbols
    pub strip: bool,
    /// Embed bitcode
    pub embed_bitcode: bool,
    /// Additional RUSTFLAGS
    pub rustflags: Vec<String>,
    /// Profile name (release, wasm-release, etc.)
    pub profile: String,
}

impl LtoConfig {
    /// Create a new LTO configuration with defaults
    pub fn new() -> Self {
        Self {
            lto_mode: LtoMode::Off,
            opt_level: OptLevel::O2,
            codegen_units: 16,
            panic_strategy: PanicStrategy::Unwind,
            strip: false,
            embed_bitcode: false,
            rustflags: Vec::new(),
            profile: "release".to_string(),
        }
    }

    /// Create configuration for development
    pub fn development() -> Self {
        Self::new()
    }

    /// Create configuration for production
    pub fn production() -> Self {
        Self {
            lto_mode: LtoMode::Thin,
            opt_level: OptLevel::O2,
            codegen_units: 1,
            panic_strategy: PanicStrategy::Abort,
            strip: true,
            embed_bitcode: false,
            rustflags: Vec::new(),
            profile: "release".to_string(),
        }
    }

    /// Create configuration for maximum size optimization
    pub fn size_optimized() -> Self {
        Self {
            lto_mode: LtoMode::Fat,
            opt_level: OptLevel::Oz,
            codegen_units: 1,
            panic_strategy: PanicStrategy::Abort,
            strip: true,
            embed_bitcode: false,
            rustflags: vec!["-C".to_string(), "link-arg=-zstack-size=0".to_string()],
            profile: "wasm-release".to_string(),
        }
    }

    /// Set LTO mode
    pub fn with_mode(mut self, mode: LtoMode) -> Self {
        self.lto_mode = mode;
        self
    }

    /// Set optimization level
    pub fn with_opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set codegen units
    pub fn with_codegen_units(mut self, units: usize) -> Self {
        self.codegen_units = units;
        self
    }

    /// Set panic strategy
    pub fn with_panic_strategy(mut self, strategy: PanicStrategy) -> Self {
        self.panic_strategy = strategy;
        self
    }

    /// Enable stripping
    pub fn with_strip(mut self, strip: bool) -> Self {
        self.strip = strip;
        self
    }

    /// Add a RUSTFLAG
    pub fn add_rustflag(mut self, flag: impl Into<String>) -> Self {
        self.rustflags.push(flag.into());
        self
    }

    /// Set profile name
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = profile.into();
        self
    }

    /// Generate Cargo.toml profile section
    pub fn to_cargo_toml(&self) -> String {
        let mut toml = format!("[profile.{}]\n", self.profile);
        toml.push_str(&format!("lto = {}\n", self.lto_mode.cargo_name()));
        toml.push_str(&format!(
            "opt-level = \"{}\"\n",
            self.opt_level.cargo_name()
        ));
        toml.push_str(&format!("codegen-units = {}\n", self.codegen_units));
        toml.push_str(&format!(
            "panic = \"{}\"\n",
            self.panic_strategy.cargo_name()
        ));
        toml.push_str(&format!("strip = {}\n", self.strip));

        if self.embed_bitcode {
            toml.push_str("embed-bitcode = true\n");
        }

        toml
    }

    /// Generate RUSTFLAGS environment variable value
    pub fn to_rustflags(&self) -> String {
        self.rustflags.join(" ")
    }

    /// Generate Cargo build command arguments
    pub fn to_cargo_args(&self) -> Vec<String> {
        let mut args = vec!["build".to_string(), "--release".to_string()];

        if !self.profile.is_empty() && self.profile != "release" {
            args.push("--profile".to_string());
            args.push(self.profile.clone());
        }

        args
    }

    /// Get environment variables for build
    pub fn to_env_vars(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        if !self.rustflags.is_empty() {
            env.insert("RUSTFLAGS".to_string(), self.to_rustflags());
        }

        env
    }

    /// Estimate total size reduction percentage
    pub fn estimated_reduction(&self) -> f64 {
        let mut reduction = 0.0;

        // LTO contribution
        reduction += self.lto_mode.size_reduction();

        // Opt level contribution (if size-optimized)
        if self.opt_level.is_size_optimized() {
            reduction += 10.0;
        }

        // Codegen units (1 unit = more optimization)
        if self.codegen_units == 1 {
            reduction += 5.0;
        }

        // Panic strategy
        if self.panic_strategy == PanicStrategy::Abort {
            reduction += 3.0;
        }

        // Strip
        if self.strip {
            reduction += 7.0;
        }

        reduction.min(80.0) // Cap at 80%
    }

    /// Estimate compilation time multiplier
    pub fn estimated_compile_time(&self) -> f64 {
        let mut multiplier = 1.0;

        // LTO impact
        multiplier *= self.lto_mode.compile_time_multiplier();

        // Codegen units impact (fewer units = slower)
        multiplier *= 16.0 / self.codegen_units as f64;

        // Opt level impact
        match self.opt_level {
            OptLevel::O0 => multiplier *= 1.0,
            OptLevel::O1 => multiplier *= 1.2,
            OptLevel::O2 => multiplier *= 1.5,
            OptLevel::O3 => multiplier *= 2.0,
            OptLevel::Os | OptLevel::Oz => multiplier *= 1.8,
        }

        multiplier
    }
}

impl Default for LtoConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// LTO statistics
#[derive(Debug, Clone)]
pub struct LtoStats {
    /// Build time in seconds
    pub build_time_seconds: f64,
    /// Original size in bytes (without LTO)
    pub original_size: usize,
    /// Optimized size in bytes (with LTO)
    pub optimized_size: usize,
    /// Functions inlined
    pub functions_inlined: usize,
    /// Dead code eliminated (bytes)
    pub dead_code_bytes: usize,
}

impl LtoStats {
    /// Get size reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        ((self.original_size - self.optimized_size) as f64 / self.original_size as f64) * 100.0
    }

    /// Get bytes saved
    pub fn bytes_saved(&self) -> usize {
        self.original_size.saturating_sub(self.optimized_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lto_modes() {
        assert_eq!(LtoMode::Off.cargo_name(), "false");
        assert_eq!(LtoMode::Thin.cargo_name(), "thin");
        assert_eq!(LtoMode::Fat.cargo_name(), "fat");

        assert_eq!(LtoMode::Fat.size_reduction(), 25.0);
        assert_eq!(LtoMode::Fat.compile_time_multiplier(), 3.0);
    }

    #[test]
    fn test_opt_levels() {
        assert_eq!(OptLevel::O0.cargo_name(), "0");
        assert_eq!(OptLevel::Oz.cargo_name(), "z");

        assert!(OptLevel::Os.is_size_optimized());
        assert!(!OptLevel::O3.is_size_optimized());
    }

    #[test]
    fn test_panic_strategies() {
        assert_eq!(PanicStrategy::Unwind.cargo_name(), "unwind");
        assert_eq!(PanicStrategy::Abort.cargo_name(), "abort");

        assert!(PanicStrategy::Unwind.size_impact() > 0);
        assert_eq!(PanicStrategy::Abort.size_impact(), 0);
    }

    #[test]
    fn test_lto_config_presets() {
        let dev = LtoConfig::development();
        assert_eq!(dev.lto_mode, LtoMode::Off);
        assert_eq!(dev.codegen_units, 16);

        let prod = LtoConfig::production();
        assert_eq!(prod.lto_mode, LtoMode::Thin);
        assert_eq!(prod.codegen_units, 1);
        assert!(prod.strip);

        let size = LtoConfig::size_optimized();
        assert_eq!(size.lto_mode, LtoMode::Fat);
        assert_eq!(size.opt_level, OptLevel::Oz);
    }

    #[test]
    fn test_cargo_toml_generation() {
        let config = LtoConfig::production();
        let toml = config.to_cargo_toml();

        assert!(toml.contains("[profile.release]"));
        assert!(toml.contains("lto = thin"));
        assert!(toml.contains("codegen-units = 1"));
        assert!(toml.contains("panic = \"abort\""));
    }

    #[test]
    fn test_rustflags_generation() {
        let config = LtoConfig::new()
            .add_rustflag("-C")
            .add_rustflag("target-cpu=native");

        let flags = config.to_rustflags();
        assert!(flags.contains("-C target-cpu=native"));
    }

    #[test]
    fn test_estimated_reduction() {
        let minimal = LtoConfig::development();
        assert_eq!(minimal.estimated_reduction(), 0.0);

        let maximal = LtoConfig::size_optimized();
        assert!(maximal.estimated_reduction() > 40.0);
    }

    #[test]
    fn test_compile_time_estimation() {
        let dev = LtoConfig::development();
        let prod = LtoConfig::production();

        assert!(prod.estimated_compile_time() > dev.estimated_compile_time());
    }

    #[test]
    fn test_lto_stats() {
        let stats = LtoStats {
            build_time_seconds: 120.0,
            original_size: 10_000_000,
            optimized_size: 7_000_000,
            functions_inlined: 500,
            dead_code_bytes: 1_000_000,
        };

        assert_eq!(stats.reduction_percent(), 30.0);
        assert_eq!(stats.bytes_saved(), 3_000_000);
    }
}
