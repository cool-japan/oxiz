//! Compression Utilities for WASM Modules
//!
//! This module provides compression utilities for WebAssembly modules,
//! supporting multiple compression formats optimized for web delivery.
//!
//! # Supported Formats
//!
//! - **Gzip**: Universal support, ~60-70% compression ratio
//! - **Brotli**: Better compression, ~70-75% compression ratio
//! - **Zstd**: Fast, ~65-70% compression ratio (optional)
//!
//! # Performance
//!
//! | Format | Compression Speed | Decompression Speed | Ratio |
//! |--------|------------------|---------------------|-------|
//! | Gzip   | Fast             | Fast                | 60-70%|
//! | Brotli | Slow             | Fast                | 70-75%|
//! | Zstd   | Very Fast        | Very Fast           | 65-70%|
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::optimize::compression::{Compressor, CompressionFormat, CompressionLevel};
//!
//! let compressor = Compressor::new();
//! let compressed = compressor.compress(
//!     &wasm_bytes,
//!     CompressionFormat::Brotli,
//!     CompressionLevel::Best
//! )?;
//!
//! println!("Compressed {} -> {} bytes ({:.1}% reduction)",
//!     wasm_bytes.len(), compressed.len(),
//!     (1.0 - compressed.len() as f64 / wasm_bytes.len() as f64) * 100.0
//! );
//! ```

#![forbid(unsafe_code)]

use std::io::{Read, Write};

/// Compression format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Brotli compression
    Brotli,
    /// Zstandard compression
    Zstd,
}

impl CompressionFormat {
    /// Get format name
    pub fn name(&self) -> &'static str {
        match self {
            CompressionFormat::None => "none",
            CompressionFormat::Gzip => "gzip",
            CompressionFormat::Brotli => "brotli",
            CompressionFormat::Zstd => "zstd",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionFormat::None => "",
            CompressionFormat::Gzip => ".gz",
            CompressionFormat::Brotli => ".br",
            CompressionFormat::Zstd => ".zst",
        }
    }

    /// Get typical compression ratio (0.0 = no compression, 1.0 = perfect compression)
    pub fn typical_ratio(&self) -> f64 {
        match self {
            CompressionFormat::None => 0.0,
            CompressionFormat::Gzip => 0.65,
            CompressionFormat::Brotli => 0.72,
            CompressionFormat::Zstd => 0.68,
        }
    }

    /// Get Content-Encoding header value
    pub fn content_encoding(&self) -> Option<&'static str> {
        match self {
            CompressionFormat::None => None,
            CompressionFormat::Gzip => Some("gzip"),
            CompressionFormat::Brotli => Some("br"),
            CompressionFormat::Zstd => Some("zstd"),
        }
    }
}

/// Compression level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Fastest compression (lowest ratio)
    Fastest,
    /// Fast compression
    Fast,
    /// Default compression (balanced)
    Default,
    /// Best compression (slowest)
    Best,
}

impl CompressionLevel {
    /// Get numeric level for gzip (1-9)
    pub fn gzip_level(&self) -> u32 {
        match self {
            CompressionLevel::Fastest => 1,
            CompressionLevel::Fast => 3,
            CompressionLevel::Default => 6,
            CompressionLevel::Best => 9,
        }
    }

    /// Get numeric level for brotli (0-11)
    pub fn brotli_level(&self) -> u32 {
        match self {
            CompressionLevel::Fastest => 1,
            CompressionLevel::Fast => 4,
            CompressionLevel::Default => 6,
            CompressionLevel::Best => 11,
        }
    }

    /// Get numeric level for zstd (1-22)
    pub fn zstd_level(&self) -> i32 {
        match self {
            CompressionLevel::Fastest => 1,
            CompressionLevel::Fast => 5,
            CompressionLevel::Default => 10,
            CompressionLevel::Best => 19,
        }
    }
}

/// Compression result
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (0.0-1.0)
    pub ratio: f64,
    /// Time taken in milliseconds
    pub time_ms: f64,
    /// Compression format used
    pub format: CompressionFormat,
    /// Compression level used
    pub level: CompressionLevel,
}

impl CompressionResult {
    /// Get space saved in bytes
    pub fn bytes_saved(&self) -> usize {
        self.original_size.saturating_sub(self.compressed_size)
    }

    /// Get reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (self.bytes_saved() as f64 / self.original_size as f64) * 100.0
    }

    /// Get compression speed in MB/s
    pub fn speed_mbps(&self) -> f64 {
        if self.time_ms == 0.0 {
            return 0.0;
        }
        (self.original_size as f64 / 1_000_000.0) / (self.time_ms / 1000.0)
    }
}

/// Compression error
#[derive(Debug, Clone)]
pub enum CompressionError {
    /// Invalid input
    InvalidInput(String),
    /// Compression failed
    CompressionFailed(String),
    /// Decompression failed
    DecompressionFailed(String),
    /// Unsupported format
    UnsupportedFormat(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CompressionError::CompressionFailed(msg) => write!(f, "Compression failed: {}", msg),
            CompressionError::DecompressionFailed(msg) => {
                write!(f, "Decompression failed: {}", msg)
            }
            CompressionError::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {}", fmt),
        }
    }
}

impl std::error::Error for CompressionError {}

/// Result type for compression operations
pub type CompressionOpResult<T> = Result<T, CompressionError>;

/// WASM module compressor
pub struct Compressor {
    /// Verbose logging
    verbose: bool,
}

impl Compressor {
    /// Create a new compressor
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Compress data
    pub fn compress(
        &self,
        data: &[u8],
        format: CompressionFormat,
        level: CompressionLevel,
    ) -> CompressionOpResult<Vec<u8>> {
        if data.is_empty() {
            return Err(CompressionError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

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

        let compressed = match format {
            CompressionFormat::None => data.to_vec(),
            CompressionFormat::Gzip => self.compress_gzip(data, level)?,
            CompressionFormat::Brotli => self.compress_brotli(data, level)?,
            CompressionFormat::Zstd => self.compress_zstd(data, level)?,
        };

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
            let ratio = compressed.len() as f64 / data.len() as f64;
            web_sys::console::log_1(
                &format!(
                    "Compressed {} bytes -> {} bytes ({:.1}% reduction) using {} in {:.2}ms",
                    data.len(),
                    compressed.len(),
                    (1.0 - ratio) * 100.0,
                    format.name(),
                    end_time - start_time
                )
                .into(),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.verbose {
            let ratio = compressed.len() as f64 / data.len() as f64;
            eprintln!(
                "Compressed {} bytes -> {} bytes ({:.1}% reduction) using {} in {:.2}ms",
                data.len(),
                compressed.len(),
                (1.0 - ratio) * 100.0,
                format.name(),
                end_time - start_time
            );
        }

        Ok(compressed)
    }

    /// Compress using gzip
    fn compress_gzip(&self, data: &[u8], level: CompressionLevel) -> CompressionOpResult<Vec<u8>> {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level.gzip_level()));
        encoder
            .write_all(data)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))
    }

    /// Compress using brotli
    fn compress_brotli(
        &self,
        data: &[u8],
        level: CompressionLevel,
    ) -> CompressionOpResult<Vec<u8>> {
        let mut output = Vec::new();
        let mut encoder =
            brotli::CompressorWriter::new(&mut output, 4096, level.brotli_level(), 22);
        encoder
            .write_all(data)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        encoder
            .flush()
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
        drop(encoder);
        Ok(output)
    }

    /// Compress using zstd
    fn compress_zstd(
        &self,
        _data: &[u8],
        _level: CompressionLevel,
    ) -> CompressionOpResult<Vec<u8>> {
        // Zstd support would require zstd crate
        Err(CompressionError::UnsupportedFormat(
            "zstd support not enabled".to_string(),
        ))
    }

    /// Decompress data
    pub fn decompress(
        &self,
        data: &[u8],
        format: CompressionFormat,
    ) -> CompressionOpResult<Vec<u8>> {
        if data.is_empty() {
            return Err(CompressionError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        match format {
            CompressionFormat::None => Ok(data.to_vec()),
            CompressionFormat::Gzip => self.decompress_gzip(data),
            CompressionFormat::Brotli => self.decompress_brotli(data),
            CompressionFormat::Zstd => self.decompress_zstd(data),
        }
    }

    /// Decompress gzip
    fn decompress_gzip(&self, data: &[u8]) -> CompressionOpResult<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut output = Vec::new();
        decoder
            .read_to_end(&mut output)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        Ok(output)
    }

    /// Decompress brotli
    fn decompress_brotli(&self, data: &[u8]) -> CompressionOpResult<Vec<u8>> {
        let mut output = Vec::new();
        let mut decoder = brotli::Decompressor::new(data, 4096);
        decoder
            .read_to_end(&mut output)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
        Ok(output)
    }

    /// Decompress zstd
    fn decompress_zstd(&self, _data: &[u8]) -> CompressionOpResult<Vec<u8>> {
        Err(CompressionError::UnsupportedFormat(
            "zstd support not enabled".to_string(),
        ))
    }

    /// Benchmark compression for all formats
    pub fn benchmark(&self, data: &[u8]) -> Vec<CompressionResult> {
        let formats = vec![CompressionFormat::Gzip, CompressionFormat::Brotli];

        let levels = vec![
            CompressionLevel::Fastest,
            CompressionLevel::Default,
            CompressionLevel::Best,
        ];

        let mut results = Vec::new();

        for format in &formats {
            for level in &levels {
                #[cfg(target_arch = "wasm32")]
                let start = web_sys::window()
                    .and_then(|w| w.performance())
                    .map(|p| p.now())
                    .unwrap_or(0.0);
                #[cfg(not(target_arch = "wasm32"))]
                let start = {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs_f64() * 1000.0)
                        .unwrap_or(0.0)
                };

                if let Ok(compressed) = self.compress(data, *format, *level) {
                    #[cfg(target_arch = "wasm32")]
                    let end = web_sys::window()
                        .and_then(|w| w.performance())
                        .map(|p| p.now())
                        .unwrap_or(start);
                    #[cfg(not(target_arch = "wasm32"))]
                    let end = {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs_f64() * 1000.0)
                            .unwrap_or(start)
                    };

                    results.push(CompressionResult {
                        original_size: data.len(),
                        compressed_size: compressed.len(),
                        ratio: compressed.len() as f64 / data.len() as f64,
                        time_ms: end - start,
                        format: *format,
                        level: *level,
                    });
                }
            }
        }

        results
    }

    /// Find best compression for given constraints
    pub fn find_best(
        &self,
        data: &[u8],
        max_time_ms: f64,
        target_ratio: f64,
    ) -> Option<(CompressionFormat, CompressionLevel)> {
        let results = self.benchmark(data);

        results
            .into_iter()
            .filter(|r| r.time_ms <= max_time_ms && r.ratio <= target_ratio)
            .min_by(|a, b| {
                a.compressed_size.cmp(&b.compressed_size).then_with(|| {
                    a.time_ms
                        .partial_cmp(&b.time_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|r| (r.format, r.level))
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression statistics aggregator
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total bytes compressed
    pub total_bytes_in: usize,
    /// Total bytes after compression
    pub total_bytes_out: usize,
    /// Number of compressions performed
    pub compressions: usize,
    /// Total time spent compressing
    pub total_time_ms: f64,
    /// Format usage counts
    pub format_usage: std::collections::HashMap<String, usize>,
}

impl CompressionStats {
    /// Create new compression stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a compression result
    pub fn add_result(&mut self, result: &CompressionResult) {
        self.total_bytes_in += result.original_size;
        self.total_bytes_out += result.compressed_size;
        self.compressions += 1;
        self.total_time_ms += result.time_ms;
        *self
            .format_usage
            .entry(result.format.name().to_string())
            .or_insert(0) += 1;
    }

    /// Get average compression ratio
    pub fn avg_ratio(&self) -> f64 {
        if self.total_bytes_in == 0 {
            0.0
        } else {
            self.total_bytes_out as f64 / self.total_bytes_in as f64
        }
    }

    /// Get average compression time
    pub fn avg_time_ms(&self) -> f64 {
        if self.compressions == 0 {
            0.0
        } else {
            self.total_time_ms / self.compressions as f64
        }
    }

    /// Get total bytes saved
    pub fn bytes_saved(&self) -> usize {
        self.total_bytes_in.saturating_sub(self.total_bytes_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_formats() {
        assert_eq!(CompressionFormat::Gzip.name(), "gzip");
        assert_eq!(CompressionFormat::Brotli.extension(), ".br");
        assert_eq!(CompressionFormat::Gzip.content_encoding(), Some("gzip"));
    }

    #[test]
    fn test_compression_levels() {
        assert_eq!(CompressionLevel::Fastest.gzip_level(), 1);
        assert_eq!(CompressionLevel::Best.gzip_level(), 9);
        assert_eq!(CompressionLevel::Best.brotli_level(), 11);
    }

    #[test]
    fn test_compressor_gzip() {
        let compressor = Compressor::new();
        let data = b"Hello, World! ".repeat(100);

        let compressed = compressor
            .compress(&data, CompressionFormat::Gzip, CompressionLevel::Default)
            .unwrap();

        assert!(compressed.len() < data.len());

        let decompressed = compressor
            .decompress(&compressed, CompressionFormat::Gzip)
            .unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_compressor_brotli() {
        let compressor = Compressor::new();
        let data = b"Test data for brotli compression. ".repeat(50);

        let compressed = compressor
            .compress(&data, CompressionFormat::Brotli, CompressionLevel::Default)
            .unwrap();

        assert!(compressed.len() < data.len());

        let decompressed = compressor
            .decompress(&compressed, CompressionFormat::Brotli)
            .unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_compression_result() {
        let result = CompressionResult {
            original_size: 1000,
            compressed_size: 600,
            ratio: 0.6,
            time_ms: 10.0,
            format: CompressionFormat::Gzip,
            level: CompressionLevel::Default,
        };

        assert_eq!(result.bytes_saved(), 400);
        assert_eq!(result.reduction_percent(), 40.0);
    }

    #[test]
    fn test_empty_data() {
        let compressor = Compressor::new();
        let result = compressor.compress(&[], CompressionFormat::Gzip, CompressionLevel::Default);

        assert!(result.is_err());
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();

        let result1 = CompressionResult {
            original_size: 1000,
            compressed_size: 600,
            ratio: 0.6,
            time_ms: 10.0,
            format: CompressionFormat::Gzip,
            level: CompressionLevel::Default,
        };

        let result2 = CompressionResult {
            original_size: 2000,
            compressed_size: 1200,
            ratio: 0.6,
            time_ms: 20.0,
            format: CompressionFormat::Brotli,
            level: CompressionLevel::Best,
        };

        stats.add_result(&result1);
        stats.add_result(&result2);

        assert_eq!(stats.compressions, 2);
        assert_eq!(stats.total_bytes_in, 3000);
        assert_eq!(stats.total_bytes_out, 1800);
        assert_eq!(stats.avg_ratio(), 0.6);
        assert_eq!(stats.avg_time_ms(), 15.0);
    }
}
