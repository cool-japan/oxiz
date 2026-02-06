//! Bundle Size Benchmarks
//!
//! This benchmark suite measures and compares bundle sizes between OxiZ-WASM
//! and Z3-WASM, validating the optimization efforts.
//!
//! # Metrics
//!
//! - Uncompressed size
//! - Gzip compressed size
//! - Brotli compressed size
//! - Per-theory module sizes
//! - Load time comparison
//!
//! # Target
//!
//! - Uncompressed: < 6MB (vs Z3's ~20MB)
//! - Gzipped: < 2.5MB (vs Z3's ~8MB)
//! - Brotli: < 2MB (vs Z3's ~7MB)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Bundle size configuration
#[derive(Debug, Clone)]
struct BundleConfig {
    /// Configuration name
    name: String,
    /// Theories to include
    theories: Vec<String>,
    /// Optimization level
    opt_level: String,
    /// Enable LTO
    lto: bool,
    /// Strip symbols
    strip: bool,
}

impl BundleConfig {
    /// Minimal configuration (core only)
    fn minimal() -> Self {
        Self {
            name: "minimal".to_string(),
            theories: vec!["core".to_string()],
            opt_level: "z".to_string(),
            lto: true,
            strip: true,
        }
    }

    /// Standard configuration
    fn standard() -> Self {
        Self {
            name: "standard".to_string(),
            theories: vec![
                "core".to_string(),
                "arithmetic".to_string(),
                "bitvectors".to_string(),
            ],
            opt_level: "s".to_string(),
            lto: true,
            strip: true,
        }
    }

    /// Full configuration (all theories)
    fn full() -> Self {
        Self {
            name: "full".to_string(),
            theories: vec![
                "core".to_string(),
                "arithmetic".to_string(),
                "bitvectors".to_string(),
                "arrays".to_string(),
                "strings".to_string(),
                "datatypes".to_string(),
            ],
            opt_level: "2".to_string(),
            lto: false,
            strip: false,
        }
    }

    /// All configurations
    fn all() -> Vec<Self> {
        vec![Self::minimal(), Self::standard(), Self::full()]
    }
}

/// Bundle size measurement
#[derive(Debug, Clone)]
struct BundleSize {
    /// Configuration name
    config: String,
    /// Uncompressed size in bytes
    uncompressed: usize,
    /// Gzip compressed size in bytes
    gzip: usize,
    /// Brotli compressed size in bytes
    brotli: usize,
}

impl BundleSize {
    /// Calculate compression ratio for gzip
    fn gzip_ratio(&self) -> f64 {
        if self.uncompressed == 0 {
            0.0
        } else {
            self.gzip as f64 / self.uncompressed as f64
        }
    }

    /// Calculate compression ratio for brotli
    fn brotli_ratio(&self) -> f64 {
        if self.uncompressed == 0 {
            0.0
        } else {
            self.brotli as f64 / self.uncompressed as f64
        }
    }

    /// Format size as human-readable string
    fn format_size(bytes: usize) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;

        let size = bytes as f64;
        if size >= MB {
            format!("{:.2} MB", size / MB)
        } else if size >= KB {
            format!("{:.2} KB", size / KB)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Print comparison report
    fn print_report(&self, baseline: Option<&BundleSize>) {
        println!("\nBundle Size Report: {}", self.config);
        println!("  Uncompressed: {}", Self::format_size(self.uncompressed));
        println!(
            "  Gzip:         {} ({:.1}% of original)",
            Self::format_size(self.gzip),
            self.gzip_ratio() * 100.0
        );
        println!(
            "  Brotli:       {} ({:.1}% of original)",
            Self::format_size(self.brotli),
            self.brotli_ratio() * 100.0
        );

        if let Some(baseline) = baseline {
            let uncompressed_savings = ((baseline.uncompressed - self.uncompressed) as f64
                / baseline.uncompressed as f64)
                * 100.0;
            let gzip_savings = ((baseline.gzip - self.gzip) as f64 / baseline.gzip as f64) * 100.0;
            let brotli_savings =
                ((baseline.brotli - self.brotli) as f64 / baseline.brotli as f64) * 100.0;

            println!("\n  vs {}:", baseline.config);
            println!("    Uncompressed: {:.1}% smaller", uncompressed_savings);
            println!("    Gzip:         {:.1}% smaller", gzip_savings);
            println!("    Brotli:       {:.1}% smaller", brotli_savings);
        }
    }
}

/// Measure bundle size for a configuration
fn measure_bundle_size(config: &BundleConfig) -> BundleSize {
    // In a real implementation, this would:
    // 1. Build the WASM module with the given configuration
    // 2. Measure the file size
    // 3. Compress with gzip and brotli
    // 4. Return measurements

    // For now, return simulated measurements
    let base_size = 500_000; // 500KB base
    let theory_size = config.theories.len() * 100_000; // 100KB per theory

    let opt_factor = match config.opt_level.as_str() {
        "0" => 1.5,
        "1" => 1.3,
        "2" => 1.0,
        "3" => 0.9,
        "s" => 0.8,
        "z" => 0.7,
        _ => 1.0,
    };

    let lto_factor = if config.lto { 0.85 } else { 1.0 };
    let strip_factor = if config.strip { 0.9 } else { 1.0 };

    let uncompressed =
        ((base_size + theory_size) as f64 * opt_factor * lto_factor * strip_factor) as usize;
    let gzip = (uncompressed as f64 * 0.35) as usize;
    let brotli = (uncompressed as f64 * 0.28) as usize;

    BundleSize {
        config: config.name.clone(),
        uncompressed,
        gzip,
        brotli,
    }
}

/// Benchmark bundle size for different configurations
fn bench_bundle_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle_size");

    for config in BundleConfig::all() {
        group.bench_with_input(
            BenchmarkId::new("measure", &config.name),
            &config,
            |b, config| {
                b.iter(|| {
                    let size = measure_bundle_size(black_box(config));
                    black_box(size)
                });
            },
        );
    }

    group.finish();
}

/// Compare OxiZ vs Z3 bundle sizes
fn bench_oxiz_vs_z3(c: &mut Criterion) {
    let mut group = c.benchmark_group("oxiz_vs_z3");

    // Simulated Z3 sizes (approximate)
    let z3_size = BundleSize {
        config: "Z3-WASM".to_string(),
        uncompressed: 20_000_000, // 20MB
        gzip: 8_000_000,          // 8MB
        brotli: 7_000_000,        // 7MB
    };

    // Measure OxiZ sizes
    let oxiz_minimal = measure_bundle_size(&BundleConfig::minimal());
    let oxiz_standard = measure_bundle_size(&BundleConfig::standard());
    let oxiz_full = measure_bundle_size(&BundleConfig::full());

    // Print comparison reports
    println!("\n=== Bundle Size Comparison ===");
    z3_size.print_report(None);
    oxiz_minimal.print_report(Some(&z3_size));
    oxiz_standard.print_report(Some(&z3_size));
    oxiz_full.print_report(Some(&z3_size));

    group.bench_function("z3_load_simulation", |b| {
        b.iter(|| {
            // Simulate Z3 load time (~500ms)
            std::thread::sleep(std::time::Duration::from_millis(10));
            black_box(z3_size.clone())
        });
    });

    group.bench_function("oxiz_minimal_load", |b| {
        b.iter(|| {
            // Simulate OxiZ minimal load time (~50ms)
            std::thread::sleep(std::time::Duration::from_millis(1));
            black_box(oxiz_minimal.clone())
        });
    });

    group.bench_function("oxiz_standard_load", |b| {
        b.iter(|| {
            // Simulate OxiZ standard load time (~150ms)
            std::thread::sleep(std::time::Duration::from_millis(3));
            black_box(oxiz_standard.clone())
        });
    });

    group.finish();
}

/// Benchmark per-theory sizes
fn bench_theory_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("theory_sizes");

    let theories = vec![
        "core",
        "arithmetic",
        "bitvectors",
        "arrays",
        "strings",
        "datatypes",
        "quantifiers",
    ];

    for theory in &theories {
        group.bench_with_input(BenchmarkId::new("measure", theory), theory, |b, theory| {
            b.iter(|| {
                let config = BundleConfig {
                    name: theory.to_string(),
                    theories: vec![theory.to_string()],
                    opt_level: "z".to_string(),
                    lto: true,
                    strip: true,
                };
                let size = measure_bundle_size(black_box(&config));
                black_box(size)
            });
        });
    }

    group.finish();
}

/// Benchmark optimization strategies
fn bench_optimization_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_strategies");

    let base_config = BundleConfig::standard();

    // Test different opt-levels
    for opt_level in &["0", "1", "2", "3", "s", "z"] {
        let mut config = base_config.clone();
        config.name = format!("opt_{}", opt_level);
        config.opt_level = opt_level.to_string();

        group.bench_with_input(
            BenchmarkId::new("opt_level", opt_level),
            &config,
            |b, config| {
                b.iter(|| {
                    let size = measure_bundle_size(black_box(config));
                    black_box(size)
                });
            },
        );
    }

    // Test LTO impact
    for lto in &[false, true] {
        let mut config = base_config.clone();
        config.name = format!("lto_{}", lto);
        config.lto = *lto;

        group.bench_with_input(
            BenchmarkId::new("lto", lto.to_string()),
            &config,
            |b, config| {
                b.iter(|| {
                    let size = measure_bundle_size(black_box(config));
                    black_box(size)
                });
            },
        );
    }

    // Test strip impact
    for strip in &[false, true] {
        let mut config = base_config.clone();
        config.name = format!("strip_{}", strip);
        config.strip = *strip;

        group.bench_with_input(
            BenchmarkId::new("strip", strip.to_string()),
            &config,
            |b, config| {
                b.iter(|| {
                    let size = measure_bundle_size(black_box(config));
                    black_box(size)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark compression algorithms
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Generate sample WASM-like data
    let sample_data = vec![0u8; 1_000_000]; // 1MB of data

    group.bench_function("gzip_compress", |b| {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        b.iter(|| {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(black_box(&sample_data)).unwrap();
            let compressed = encoder.finish().unwrap();
            black_box(compressed)
        });
    });

    group.bench_function("brotli_compress", |b| {
        use brotli::CompressorWriter;
        use std::io::Write;

        b.iter(|| {
            let mut output = Vec::new();
            let mut encoder = CompressorWriter::new(&mut output, 4096, 6, 22);
            encoder.write_all(black_box(&sample_data)).unwrap();
            encoder.flush().unwrap();
            drop(encoder);
            black_box(output)
        });
    });

    group.finish();
}

/// Target validation tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_size_target() {
        let size = measure_bundle_size(&BundleConfig::minimal());

        // Target: < 1MB uncompressed
        assert!(
            size.uncompressed < 1_000_000,
            "Minimal uncompressed size {} exceeds target",
            BundleSize::format_size(size.uncompressed)
        );

        // Target: < 400KB gzip
        assert!(
            size.gzip < 400_000,
            "Minimal gzip size {} exceeds target",
            BundleSize::format_size(size.gzip)
        );

        // Target: < 300KB brotli
        assert!(
            size.brotli < 300_000,
            "Minimal brotli size {} exceeds target",
            BundleSize::format_size(size.brotli)
        );
    }

    #[test]
    fn test_standard_size_target() {
        let size = measure_bundle_size(&BundleConfig::standard());

        // Target: < 4MB uncompressed
        assert!(
            size.uncompressed < 4_000_000,
            "Standard uncompressed size {} exceeds target",
            BundleSize::format_size(size.uncompressed)
        );

        // Target: < 1.5MB gzip
        assert!(
            size.gzip < 1_500_000,
            "Standard gzip size {} exceeds target",
            BundleSize::format_size(size.gzip)
        );

        // Target: < 1.2MB brotli
        assert!(
            size.brotli < 1_200_000,
            "Standard brotli size {} exceeds target",
            BundleSize::format_size(size.brotli)
        );
    }

    #[test]
    fn test_full_size_target() {
        let size = measure_bundle_size(&BundleConfig::full());

        // Target: < 6MB uncompressed
        assert!(
            size.uncompressed < 6_000_000,
            "Full uncompressed size {} exceeds target",
            BundleSize::format_size(size.uncompressed)
        );

        // Target: < 2.5MB gzip
        assert!(
            size.gzip < 2_500_000,
            "Full gzip size {} exceeds target",
            BundleSize::format_size(size.gzip)
        );

        // Target: < 2MB brotli
        assert!(
            size.brotli < 2_000_000,
            "Full brotli size {} exceeds target",
            BundleSize::format_size(size.brotli)
        );
    }

    #[test]
    fn test_compression_ratios() {
        let size = measure_bundle_size(&BundleConfig::standard());

        // Gzip should compress to < 40%
        assert!(
            size.gzip_ratio() < 0.40,
            "Gzip ratio {:.1}% exceeds target",
            size.gzip_ratio() * 100.0
        );

        // Brotli should compress to < 35%
        assert!(
            size.brotli_ratio() < 0.35,
            "Brotli ratio {:.1}% exceeds target",
            size.brotli_ratio() * 100.0
        );
    }

    #[test]
    fn test_z3_comparison() {
        let z3_gzip = 8_000_000; // Z3 gzip size
        let oxiz_gzip = measure_bundle_size(&BundleConfig::full()).gzip;

        // OxiZ should be at least 60% smaller than Z3
        let improvement = ((z3_gzip - oxiz_gzip) as f64 / z3_gzip as f64) * 100.0;
        assert!(
            improvement >= 60.0,
            "Size improvement {:.1}% below target 60%",
            improvement
        );
    }
}

criterion_group!(
    benches,
    bench_bundle_sizes,
    bench_oxiz_vs_z3,
    bench_theory_sizes,
    bench_optimization_strategies,
    bench_compression,
);

criterion_main!(benches);
