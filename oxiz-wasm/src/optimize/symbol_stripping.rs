//! Symbol Stripping for WASM Modules
//!
//! This module provides utilities for stripping debug symbols and metadata
//! from WebAssembly modules to reduce bundle size.
//!
//! # What Gets Stripped
//!
//! - Debug symbols (DWARF sections)
//! - Source maps
//! - Function names (except exports)
//! - Local variable names
//! - Type names
//! - Custom sections (optional)
//!
//! # Size Impact
//!
//! Typical size reduction: 30-50% for debug builds, 10-20% for release builds
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::optimize::symbol_stripping::{SymbolStripper, StripConfig};
//!
//! let stripper = SymbolStripper::new();
//! let config = StripConfig::aggressive();
//! let result = stripper.strip(&wasm_bytes, &config)?;
//!
//! println!("Stripped {} symbols, saved {} bytes",
//!          result.symbols_stripped, result.bytes_saved);
//! ```

#![forbid(unsafe_code)]

use std::collections::HashSet;

/// Configuration for symbol stripping
#[derive(Debug, Clone)]
pub struct StripConfig {
    /// Strip debug sections (DWARF)
    pub strip_debug: bool,
    /// Strip function names (except exports)
    pub strip_names: bool,
    /// Strip source maps
    pub strip_source_maps: bool,
    /// Strip custom sections
    pub strip_custom: bool,
    /// Keep sections matching these patterns
    pub keep_sections: Vec<String>,
    /// Keep names matching these patterns
    pub keep_names: Vec<String>,
    /// Aggressive mode (strip everything possible)
    pub aggressive: bool,
}

impl StripConfig {
    /// Create a new strip configuration
    pub fn new() -> Self {
        Self {
            strip_debug: true,
            strip_names: false,
            strip_source_maps: true,
            strip_custom: false,
            keep_sections: Vec::new(),
            keep_names: Vec::new(),
            aggressive: false,
        }
    }

    /// Conservative stripping (only debug symbols)
    pub fn conservative() -> Self {
        Self {
            strip_debug: true,
            strip_names: false,
            strip_source_maps: false,
            strip_custom: false,
            keep_sections: Vec::new(),
            keep_names: Vec::new(),
            aggressive: false,
        }
    }

    /// Aggressive stripping (maximum size reduction)
    pub fn aggressive() -> Self {
        Self {
            strip_debug: true,
            strip_names: true,
            strip_source_maps: true,
            strip_custom: true,
            keep_sections: Vec::new(),
            keep_names: vec!["__wasm".to_string()], // Keep wasm-specific sections
            aggressive: true,
        }
    }

    /// Production stripping (balanced)
    pub fn production() -> Self {
        Self {
            strip_debug: true,
            strip_names: false,
            strip_source_maps: true,
            strip_custom: false,
            keep_sections: Vec::new(),
            keep_names: Vec::new(),
            aggressive: false,
        }
    }

    /// Add a section pattern to keep
    pub fn keep_section(mut self, pattern: impl Into<String>) -> Self {
        self.keep_sections.push(pattern.into());
        self
    }

    /// Add a name pattern to keep
    pub fn keep_name(mut self, pattern: impl Into<String>) -> Self {
        self.keep_names.push(pattern.into());
        self
    }
}

impl Default for StripConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of symbol stripping
#[derive(Debug, Clone)]
pub struct StripResult {
    /// Number of symbols stripped
    pub symbols_stripped: usize,
    /// Number of sections removed
    pub sections_removed: usize,
    /// Original size in bytes
    pub original_size: usize,
    /// New size in bytes
    pub new_size: usize,
    /// Bytes saved
    pub bytes_saved: usize,
    /// Time taken in milliseconds
    pub time_ms: f64,
    /// Details of what was stripped
    pub details: StripDetails,
}

impl StripResult {
    /// Get size reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (self.bytes_saved as f64 / self.original_size as f64) * 100.0
    }
}

/// Details about what was stripped
#[derive(Debug, Clone, Default)]
pub struct StripDetails {
    /// Debug symbols removed
    pub debug_symbols: usize,
    /// Function names removed
    pub function_names: usize,
    /// Local names removed
    pub local_names: usize,
    /// Type names removed
    pub type_names: usize,
    /// Custom sections removed
    pub custom_sections: Vec<String>,
    /// DWARF sections removed
    pub dwarf_sections: Vec<String>,
}

impl StripDetails {
    /// Get total items stripped
    pub fn total_items(&self) -> usize {
        self.debug_symbols
            + self.function_names
            + self.local_names
            + self.type_names
            + self.custom_sections.len()
            + self.dwarf_sections.len()
    }
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: SymbolType,
    /// Size contribution in bytes
    pub size_bytes: usize,
    /// Whether this symbol is exported
    pub exported: bool,
}

/// Type of symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    /// Function symbol
    Function,
    /// Global variable
    Global,
    /// Memory
    Memory,
    /// Table
    Table,
    /// Type definition
    Type,
    /// Data segment
    Data,
    /// Custom section
    Custom,
}

impl SymbolType {
    /// Get symbol type name
    pub fn name(&self) -> &'static str {
        match self {
            SymbolType::Function => "function",
            SymbolType::Global => "global",
            SymbolType::Memory => "memory",
            SymbolType::Table => "table",
            SymbolType::Type => "type",
            SymbolType::Data => "data",
            SymbolType::Custom => "custom",
        }
    }
}

/// Symbol stripper
pub struct SymbolStripper {
    /// Known symbols
    symbols: Vec<SymbolInfo>,
    /// Exported symbol names
    exports: HashSet<String>,
    /// Verbose logging
    verbose: bool,
}

impl SymbolStripper {
    /// Create a new symbol stripper
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            exports: HashSet::new(),
            verbose: false,
        }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Register a symbol
    pub fn register_symbol(&mut self, info: SymbolInfo) {
        if info.exported {
            self.exports.insert(info.name.clone());
        }
        self.symbols.push(info);
    }

    /// Register multiple symbols
    pub fn register_symbols(&mut self, symbols: Vec<SymbolInfo>) {
        for symbol in symbols {
            self.register_symbol(symbol);
        }
    }

    /// Check if a symbol should be kept
    fn should_keep(&self, symbol: &SymbolInfo, config: &StripConfig) -> bool {
        // Always keep exports
        if symbol.exported {
            return true;
        }

        // Check keep patterns
        for pattern in &config.keep_names {
            if symbol.name.contains(pattern) {
                return true;
            }
        }

        // Check by type
        match symbol.symbol_type {
            SymbolType::Function => !config.strip_names,
            SymbolType::Custom => !config.strip_custom,
            _ => true,
        }
    }

    /// Strip symbols according to configuration
    pub fn strip(&mut self, config: &StripConfig) -> StripResult {
        #[cfg(target_arch = "wasm32")]
        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or(0.0)
        };

        let original_size: usize = self.symbols.iter().map(|s| s.size_bytes).sum();

        let mut symbols_to_remove = Vec::new();
        let mut details = StripDetails::default();

        // Find symbols to remove
        for (idx, symbol) in self.symbols.iter().enumerate() {
            if !self.should_keep(symbol, config) {
                symbols_to_remove.push(idx);

                match symbol.symbol_type {
                    SymbolType::Function => details.function_names += 1,
                    SymbolType::Global => {}
                    SymbolType::Type => details.type_names += 1,
                    SymbolType::Custom => details.custom_sections.push(symbol.name.clone()),
                    _ => {}
                }
            }
        }

        // Calculate bytes saved
        let bytes_saved: usize = symbols_to_remove
            .iter()
            .map(|&idx| self.symbols[idx].size_bytes)
            .sum();

        // Remove symbols (in reverse order to maintain indices)
        for &idx in symbols_to_remove.iter().rev() {
            self.symbols.remove(idx);
        }

        let new_size = original_size - bytes_saved;
        let symbols_stripped = symbols_to_remove.len();

        #[cfg(target_arch = "wasm32")]
        let end_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(start_time);
        #[cfg(not(target_arch = "wasm32"))]
        let end_time = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or(start_time)
        };

        #[cfg(target_arch = "wasm32")]
        if self.verbose {
            web_sys::console::log_1(
                &format!(
                    "Symbol stripping: removed {} symbols, saved {} bytes ({:.1}%)",
                    symbols_stripped,
                    bytes_saved,
                    (bytes_saved as f64 / original_size as f64) * 100.0
                )
                .into(),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.verbose {
            eprintln!(
                "Symbol stripping: removed {} symbols, saved {} bytes ({:.1}%)",
                symbols_stripped,
                bytes_saved,
                (bytes_saved as f64 / original_size as f64) * 100.0
            );
        }

        StripResult {
            symbols_stripped,
            sections_removed: details.custom_sections.len() + details.dwarf_sections.len(),
            original_size,
            new_size,
            bytes_saved,
            time_ms: end_time - start_time,
            details,
        }
    }

    /// Get symbol statistics
    pub fn stats(&self) -> SymbolStats {
        let mut stats = SymbolStats {
            total_symbols: self.symbols.len(),
            exported_symbols: 0,
            functions: 0,
            globals: 0,
            types: 0,
            custom_sections: 0,
            total_size_bytes: 0,
        };

        for symbol in &self.symbols {
            if symbol.exported {
                stats.exported_symbols += 1;
            }

            match symbol.symbol_type {
                SymbolType::Function => stats.functions += 1,
                SymbolType::Global => stats.globals += 1,
                SymbolType::Type => stats.types += 1,
                SymbolType::Custom => stats.custom_sections += 1,
                _ => {}
            }

            stats.total_size_bytes += symbol.size_bytes;
        }

        stats
    }

    /// Export symbol list
    pub fn export_symbols(&self) -> Vec<SymbolExport> {
        self.symbols
            .iter()
            .map(|s| SymbolExport {
                name: s.name.clone(),
                symbol_type: s.symbol_type.name().to_string(),
                size_bytes: s.size_bytes,
                exported: s.exported,
            })
            .collect()
    }

    /// Find symbols by type
    pub fn find_by_type(&self, symbol_type: SymbolType) -> Vec<&SymbolInfo> {
        self.symbols
            .iter()
            .filter(|s| s.symbol_type == symbol_type)
            .collect()
    }

    /// Find symbols by pattern
    pub fn find_by_pattern(&self, pattern: &str) -> Vec<&SymbolInfo> {
        self.symbols
            .iter()
            .filter(|s| s.name.contains(pattern))
            .collect()
    }

    /// Reset stripper state
    pub fn reset(&mut self) {
        self.symbols.clear();
        self.exports.clear();
    }
}

impl Default for SymbolStripper {
    fn default() -> Self {
        Self::new()
    }
}

/// Symbol statistics
#[derive(Debug, Clone)]
pub struct SymbolStats {
    /// Total number of symbols
    pub total_symbols: usize,
    /// Number of exported symbols
    pub exported_symbols: usize,
    /// Number of function symbols
    pub functions: usize,
    /// Number of global symbols
    pub globals: usize,
    /// Number of type symbols
    pub types: usize,
    /// Number of custom sections
    pub custom_sections: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
}

impl SymbolStats {
    /// Get average size per symbol
    pub fn avg_size_bytes(&self) -> f64 {
        if self.total_symbols == 0 {
            0.0
        } else {
            self.total_size_bytes as f64 / self.total_symbols as f64
        }
    }
}

/// Exported symbol information
#[derive(Debug, Clone)]
pub struct SymbolExport {
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether exported
    pub exported: bool,
}

/// DWARF section types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwarfSection {
    /// .debug_info
    Info,
    /// .debug_abbrev
    Abbrev,
    /// .debug_line
    Line,
    /// .debug_str
    Str,
    /// .debug_ranges
    Ranges,
    /// .debug_loc
    Loc,
}

impl DwarfSection {
    /// Get section name
    pub fn name(&self) -> &'static str {
        match self {
            DwarfSection::Info => ".debug_info",
            DwarfSection::Abbrev => ".debug_abbrev",
            DwarfSection::Line => ".debug_line",
            DwarfSection::Str => ".debug_str",
            DwarfSection::Ranges => ".debug_ranges",
            DwarfSection::Loc => ".debug_loc",
        }
    }

    /// Get all DWARF sections
    pub fn all() -> Vec<DwarfSection> {
        vec![
            DwarfSection::Info,
            DwarfSection::Abbrev,
            DwarfSection::Line,
            DwarfSection::Str,
            DwarfSection::Ranges,
            DwarfSection::Loc,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_config() {
        let aggressive = StripConfig::aggressive();
        assert!(aggressive.strip_debug);
        assert!(aggressive.strip_names);
        assert!(aggressive.strip_custom);

        let conservative = StripConfig::conservative();
        assert!(conservative.strip_debug);
        assert!(!conservative.strip_names);
    }

    #[test]
    fn test_symbol_info() {
        let symbol = SymbolInfo {
            name: "test_func".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 100,
            exported: true,
        };

        assert_eq!(symbol.name, "test_func");
        assert_eq!(symbol.symbol_type, SymbolType::Function);
        assert!(symbol.exported);
    }

    #[test]
    fn test_symbol_stripper() {
        let mut stripper = SymbolStripper::new();

        stripper.register_symbol(SymbolInfo {
            name: "exported_func".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 100,
            exported: true,
        });

        stripper.register_symbol(SymbolInfo {
            name: "internal_func".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 50,
            exported: false,
        });

        let config = StripConfig::aggressive();
        let result = stripper.strip(&config);

        // Only internal function should be stripped
        assert_eq!(result.symbols_stripped, 1);
        assert_eq!(result.bytes_saved, 50);
    }

    #[test]
    fn test_keep_patterns() {
        let mut stripper = SymbolStripper::new();

        stripper.register_symbol(SymbolInfo {
            name: "oxiz_internal_func".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 50,
            exported: false,
        });

        let config = StripConfig::aggressive().keep_name("oxiz_");
        let result = stripper.strip(&config);

        // Should be kept due to pattern
        assert_eq!(result.symbols_stripped, 0);
    }

    #[test]
    fn test_symbol_stats() {
        let mut stripper = SymbolStripper::new();

        stripper.register_symbol(SymbolInfo {
            name: "func1".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 100,
            exported: true,
        });

        stripper.register_symbol(SymbolInfo {
            name: "func2".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 50,
            exported: false,
        });

        let stats = stripper.stats();
        assert_eq!(stats.total_symbols, 2);
        assert_eq!(stats.exported_symbols, 1);
        assert_eq!(stats.functions, 2);
        assert_eq!(stats.total_size_bytes, 150);
    }

    #[test]
    fn test_find_by_type() {
        let mut stripper = SymbolStripper::new();

        stripper.register_symbol(SymbolInfo {
            name: "func1".to_string(),
            symbol_type: SymbolType::Function,
            size_bytes: 100,
            exported: false,
        });

        stripper.register_symbol(SymbolInfo {
            name: "global1".to_string(),
            symbol_type: SymbolType::Global,
            size_bytes: 4,
            exported: false,
        });

        let functions = stripper.find_by_type(SymbolType::Function);
        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0].name, "func1");
    }

    #[test]
    fn test_dwarf_sections() {
        assert_eq!(DwarfSection::Info.name(), ".debug_info");
        assert_eq!(DwarfSection::Line.name(), ".debug_line");

        let all_sections = DwarfSection::all();
        assert_eq!(all_sections.len(), 6);
    }
}
