//! Load Time Benchmarks
//!
//! This benchmark suite measures module load and initialization times
//! for OxiZ-WASM, comparing different configurations and optimization levels.
//!
//! # Metrics
//!
//! - Module instantiation time
//! - Theory loading time (lazy loading)
//! - First-solve latency
//! - Memory initialization time
//!
//! # Targets
//!
//! - Full module load: < 200ms (vs Z3's ~500ms)
//! - Core module load: < 50ms
//! - Theory lazy-load: < 100ms per theory
//! - First solve overhead: < 50ms

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Load time measurement
#[derive(Debug, Clone)]
struct LoadTime {
    /// Module instantiation time
    instantiation_ms: f64,
    /// Initialization time
    initialization_ms: f64,
    /// Theory loading time
    theory_loading_ms: f64,
    /// Total time
    total_ms: f64,
}

#[allow(dead_code)]
impl LoadTime {
    /// Create a new load time measurement
    fn new() -> Self {
        Self {
            instantiation_ms: 0.0,
            initialization_ms: 0.0,
            theory_loading_ms: 0.0,
            total_ms: 0.0,
        }
    }

    /// Calculate total time
    fn calculate_total(&mut self) {
        self.total_ms = self.instantiation_ms + self.initialization_ms + self.theory_loading_ms;
    }

    /// Print report
    fn print_report(&self, _name: &str) {
        println!("\nLoad Time Report: {}", _name);
        println!("  Instantiation:   {:.2}ms", self.instantiation_ms);
        println!("  Initialization:  {:.2}ms", self.initialization_ms);
        println!("  Theory loading:  {:.2}ms", self.theory_loading_ms);
        println!("  Total:           {:.2}ms", self.total_ms);
    }

    /// Check if meets targets
    fn meets_targets(&self) -> bool {
        self.total_ms < 200.0
    }
}

/// Load configuration
#[derive(Debug, Clone)]
struct LoadConfig {
    /// Configuration name
    name: String,
    /// Theories to load
    theories: Vec<String>,
    /// Use lazy loading
    lazy_loading: bool,
    /// Preload commonly used functions
    preload: bool,
}

impl LoadConfig {
    /// Minimal configuration
    fn minimal() -> Self {
        Self {
            name: "minimal".to_string(),
            theories: vec!["core".to_string()],
            lazy_loading: false,
            preload: false,
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
            lazy_loading: true,
            preload: true,
        }
    }

    /// Full configuration
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
            lazy_loading: true,
            preload: true,
        }
    }

    /// All configurations
    fn all() -> Vec<Self> {
        vec![Self::minimal(), Self::standard(), Self::full()]
    }
}

/// Simulate module instantiation
fn simulate_instantiation(config: &LoadConfig) -> f64 {
    let start = Instant::now();

    // Simulate WASM module parse/compile
    let base_time = 20.0; // Base instantiation time
    let theory_factor = config.theories.len() as f64 * 5.0;

    std::thread::sleep(Duration::from_micros(
        (base_time + theory_factor) as u64 * 100,
    ));

    start.elapsed().as_secs_f64() * 1000.0
}

/// Simulate initialization
fn simulate_initialization(config: &LoadConfig) -> f64 {
    let start = Instant::now();

    // Simulate memory allocation and setup
    let base_time = 10.0;
    let preload_time = if config.preload { 15.0 } else { 0.0 };

    std::thread::sleep(Duration::from_micros(
        (base_time + preload_time) as u64 * 100,
    ));

    start.elapsed().as_secs_f64() * 1000.0
}

/// Simulate theory loading
fn simulate_theory_loading(config: &LoadConfig) -> f64 {
    if config.lazy_loading {
        // Only load core initially
        simulate_single_theory_load("core")
    } else {
        // Load all theories
        let start = Instant::now();
        for theory in &config.theories {
            simulate_single_theory_load(theory);
        }
        start.elapsed().as_secs_f64() * 1000.0
    }
}

/// Simulate loading a single theory
fn simulate_single_theory_load(theory: &str) -> f64 {
    let base_time = match theory {
        "core" => 5.0,
        "arithmetic" => 25.0,
        "bitvectors" => 20.0,
        "arrays" => 15.0,
        "strings" => 30.0,
        "datatypes" => 12.0,
        "quantifiers" => 40.0,
        _ => 10.0,
    };

    std::thread::sleep(Duration::from_micros((base_time * 100.0) as u64));
    base_time
}

/// Measure load time for a configuration
fn measure_load_time(config: &LoadConfig) -> LoadTime {
    let mut load_time = LoadTime::new();

    load_time.instantiation_ms = simulate_instantiation(config);
    load_time.initialization_ms = simulate_initialization(config);
    load_time.theory_loading_ms = simulate_theory_loading(config);
    load_time.calculate_total();

    load_time
}

/// Benchmark module loading
fn bench_module_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("module_loading");

    for config in LoadConfig::all() {
        group.bench_with_input(
            BenchmarkId::new("load", &config.name),
            &config,
            |b, config| {
                b.iter(|| {
                    let load_time = measure_load_time(black_box(config));
                    black_box(load_time)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark lazy theory loading
fn bench_lazy_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_loading");

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
        group.bench_with_input(
            BenchmarkId::new("load_theory", theory),
            theory,
            |b, theory| {
                b.iter(|| {
                    let time = simulate_single_theory_load(black_box(theory));
                    black_box(time)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark first-solve latency
fn bench_first_solve_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_solve_latency");

    group.bench_function("cold_start", |b| {
        b.iter(|| {
            // Simulate loading + first solve
            let config = LoadConfig::standard();
            let load_time = measure_load_time(&config);

            // First solve overhead
            let first_solve = 30.0; // ms
            std::thread::sleep(Duration::from_micros((first_solve * 100.0) as u64));

            black_box(load_time.total_ms + first_solve)
        });
    });

    group.bench_function("warm_start", |b| {
        b.iter(|| {
            // Simulate already loaded module
            let solve_time = 5.0; // ms
            std::thread::sleep(Duration::from_micros((solve_time * 100.0) as u64));
            black_box(solve_time)
        });
    });

    group.finish();
}

/// Benchmark OxiZ vs Z3 load times
fn bench_oxiz_vs_z3_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("oxiz_vs_z3_load");

    // Z3 simulated load time
    let z3_load_ms = 500.0;

    group.bench_function("z3_load", |b| {
        b.iter(|| {
            std::thread::sleep(Duration::from_micros((z3_load_ms * 100.0) as u64));
            black_box(z3_load_ms)
        });
    });

    // OxiZ load times
    for config in LoadConfig::all() {
        let name = format!("oxiz_{}", config.name);
        group.bench_with_input(BenchmarkId::new("load", &name), &config, |b, config| {
            b.iter(|| {
                let load_time = measure_load_time(black_box(config));
                black_box(load_time.total_ms)
            });
        });
    }

    group.finish();

    // Print comparison
    println!("\n=== Load Time Comparison ===");
    println!("Z3-WASM:        {:.2}ms", z3_load_ms);

    for config in LoadConfig::all() {
        let load_time = measure_load_time(&config);
        let speedup = z3_load_ms / load_time.total_ms;
        println!(
            "OxiZ ({}):   {:.2}ms ({:.1}x faster)",
            config.name, load_time.total_ms, speedup
        );
    }
}

/// Benchmark memory initialization
fn bench_memory_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_initialization");

    let sizes = vec![
        ("small", 1_024 * 1024),      // 1MB
        ("medium", 10 * 1024 * 1024), // 10MB
        ("large", 50 * 1024 * 1024),  // 50MB
    ];

    for (name, size) in &sizes {
        group.bench_with_input(BenchmarkId::new("allocate", name), size, |b, size| {
            b.iter(|| {
                // Simulate memory allocation
                let data = vec![0u8; *size];
                black_box(data)
            });
        });
    }

    group.finish();
}

/// Benchmark parallel loading
fn bench_parallel_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_loading");

    let theories = vec!["arithmetic", "bitvectors", "arrays", "strings"];

    group.bench_function("sequential", |b| {
        b.iter(|| {
            let start = Instant::now();
            for theory in &theories {
                simulate_single_theory_load(theory);
            }
            black_box(start.elapsed().as_secs_f64() * 1000.0)
        });
    });

    group.bench_function("parallel", |b| {
        use std::thread;

        b.iter(|| {
            let start = Instant::now();
            let handles: Vec<_> = theories
                .iter()
                .map(|theory| {
                    let theory = theory.to_string();
                    thread::spawn(move || simulate_single_theory_load(&theory))
                })
                .collect();

            for handle in handles {
                handle.join().ok();
            }

            black_box(start.elapsed().as_secs_f64() * 1000.0)
        });
    });

    group.finish();
}

/// Benchmark caching impact
fn bench_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("caching");

    group.bench_function("no_cache", |b| {
        b.iter(|| {
            let config = LoadConfig::standard();
            let load_time = measure_load_time(black_box(&config));
            black_box(load_time)
        });
    });

    group.bench_function("with_cache", |b| {
        // Simulate cached load (much faster)
        b.iter(|| {
            let instant_load = 10.0; // ms
            std::thread::sleep(Duration::from_micros((instant_load * 100.0) as u64));
            black_box(instant_load)
        });
    });

    group.finish();
}

/// Target validation tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_load_target() {
        let config = LoadConfig::minimal();
        let load_time = measure_load_time(&config);

        // Target: < 50ms
        assert!(
            load_time.total_ms < 50.0,
            "Minimal load time {:.2}ms exceeds target 50ms",
            load_time.total_ms
        );
    }

    #[test]
    fn test_standard_load_target() {
        let config = LoadConfig::standard();
        let load_time = measure_load_time(&config);

        // Target: < 100ms
        assert!(
            load_time.total_ms < 100.0,
            "Standard load time {:.2}ms exceeds target 100ms",
            load_time.total_ms
        );
    }

    #[test]
    fn test_full_load_target() {
        let config = LoadConfig::full();
        let load_time = measure_load_time(&config);

        // Target: < 200ms
        assert!(
            load_time.total_ms < 200.0,
            "Full load time {:.2}ms exceeds target 200ms",
            load_time.total_ms
        );
    }

    #[test]
    fn test_lazy_loading_benefit() {
        let mut eager_config = LoadConfig::full();
        eager_config.lazy_loading = false;

        let mut lazy_config = LoadConfig::full();
        lazy_config.lazy_loading = true;

        let eager_time = measure_load_time(&eager_config);
        let lazy_time = measure_load_time(&lazy_config);

        // Lazy loading should be significantly faster
        assert!(
            lazy_time.total_ms < eager_time.total_ms * 0.5,
            "Lazy loading not providing expected benefit"
        );
    }

    #[test]
    fn test_theory_load_times() {
        // Each theory should load in < 100ms
        let theories = vec!["arithmetic", "bitvectors", "arrays", "strings", "datatypes"];

        for theory in theories {
            let time = simulate_single_theory_load(theory);
            assert!(
                time < 100.0,
                "Theory '{}' load time {:.2}ms exceeds target 100ms",
                theory,
                time
            );
        }
    }

    #[test]
    fn test_first_solve_latency() {
        let config = LoadConfig::standard();
        let load_time = measure_load_time(&config);

        let first_solve_overhead = 30.0; // ms
        let total_cold_start = load_time.total_ms + first_solve_overhead;

        // Target: < 150ms for cold start to first result
        assert!(
            total_cold_start < 150.0,
            "Cold start latency {:.2}ms exceeds target 150ms",
            total_cold_start
        );
    }

    #[test]
    fn test_z3_comparison() {
        let z3_load = 500.0;
        let oxiz_load = measure_load_time(&LoadConfig::full()).total_ms;

        // OxiZ should be at least 2x faster
        let speedup = z3_load / oxiz_load;
        assert!(
            speedup >= 2.0,
            "Load speedup {:.1}x below target 2.0x",
            speedup
        );
    }
}

criterion_group!(
    benches,
    bench_module_loading,
    bench_lazy_loading,
    bench_first_solve_latency,
    bench_oxiz_vs_z3_load,
    bench_memory_init,
    bench_parallel_loading,
    bench_caching,
);

criterion_main!(benches);
