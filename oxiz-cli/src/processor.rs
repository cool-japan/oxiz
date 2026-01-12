//! File processing and execution logic for OxiZ CLI
//!
//! This module handles the core file processing functionality including:
//! - Sequential and parallel file processing
//! - File collection with glob pattern support
//! - Stdin input processing
//! - Watch mode for automatic reprocessing on file changes

use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::Instant;

use globset::{Glob, GlobSetBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use notify::{RecursiveMode, Watcher};
use rayon::prelude::*;
use sysinfo::System;
use walkdir::WalkDir;

use oxiz_solver::Context;

// Import from crate
use crate::{Args, InputFormat, Verbosity, apply_solver_options, execute_and_format};

// Import from crate modules
use crate::cache;
use crate::dimacs;

// Import from crate::format
use crate::format::{
    ProfilingData, SolverResult, SolverStats, eprintln_colored, export_statistics,
    format_dimacs_output, output_results, print_statistics, println_colored,
};

/// Process multiple files based on command-line arguments
pub(crate) fn run_files(ctx: &mut Context, args: &Args, verbosity: Verbosity) {
    let files = collect_files(&args.input, args.recursive);

    if files.is_empty() {
        eprintln_colored(args, "No files found matching the input patterns");
        std::process::exit(1);
    }

    if verbosity >= Verbosity::Normal && !args.smtcomp {
        println_colored(
            args,
            &format!("Processing {} file(s)...", files.len()),
            Some(owo_colors::AnsiColors::Cyan),
        );
    }

    // Initialize cache if enabled
    let mut result_cache = if args.cache {
        Some(cache::ResultCache::new(args.cache_dir.clone()))
    } else {
        None
    };

    // Initialize benchmark tracker if enabled
    let mut benchmark_tracker = args
        .benchmark_file
        .as_ref()
        .map(|path| cache::BenchmarkTracker::new(path.clone()));

    let start_time = Instant::now();
    let mut sys = if args.memory {
        let mut s = System::new_all();
        s.refresh_all();
        Some(s)
    } else {
        None
    };

    let results = if args.parallel && files.len() > 1 {
        process_files_parallel(&files, ctx, args, verbosity, &mut result_cache)
    } else {
        process_files_sequential(&files, ctx, args, verbosity, &mut result_cache)
    };

    let total_time = start_time.elapsed();

    // Calculate per-file statistics
    let mut times: Vec<u128> = results.iter().map(|r| r.time_ms).collect();
    times.sort_unstable();
    let min_time = times.first().copied().unwrap_or(0);
    let max_time = times.last().copied().unwrap_or(0);
    let avg_time = if !times.is_empty() {
        times.iter().sum::<u128>() / times.len() as u128
    } else {
        0
    };

    // Collect memory statistics
    let (memory_bytes, peak_memory) = if let Some(ref mut s) = sys {
        s.refresh_all();
        (s.used_memory(), s.used_memory())
    } else {
        (0, 0)
    };

    // Profiling data (if enabled)
    let profiling_data = if args.profile {
        let mut prof_data = vec![];

        prof_data.push(ProfilingData {
            operation: "Total Execution".to_string(),
            duration_us: total_time.as_micros(),
            memory_delta_bytes: memory_bytes as i64,
        });

        for (idx, result) in results.iter().enumerate() {
            prof_data.push(ProfilingData {
                operation: format!(
                    "File {}: {}",
                    idx + 1,
                    result.file.as_ref().unwrap_or(&"stdin".to_string())
                ),
                duration_us: result.time_ms * 1000,
                memory_delta_bytes: 0,
            });
        }

        Some(prof_data)
    } else {
        None
    };

    // Collect SAT solver statistics
    let sat_stats = ctx.stats();

    // Collect statistics
    let stats = SolverStats {
        execution_time_ms: total_time.as_millis(),
        files_processed: files.len(),
        memory_bytes,
        peak_memory_bytes: peak_memory,
        success_count: results.iter().filter(|r| r.error.is_none()).count(),
        error_count: results.iter().filter(|r| r.error.is_some()).count(),
        profiling_data,
        avg_time_per_file_ms: avg_time,
        min_time_ms: min_time,
        max_time_ms: max_time,
        decisions: sat_stats.decisions,
        propagations: sat_stats.propagations,
        conflicts: sat_stats.conflicts,
        restarts: sat_stats.restarts,
    };

    // Save benchmark data if requested
    if let Some(ref mut tracker) = benchmark_tracker {
        for result in results.iter() {
            if let Some(file_path) = result.file.as_ref() {
                let entry = cache::BenchmarkEntry {
                    problem: file_path.clone(),
                    result: if result.error.is_some() {
                        "error".to_string()
                    } else if result.result.contains("unsat") {
                        "unsat".to_string()
                    } else if result.result.contains("sat") {
                        "sat".to_string()
                    } else {
                        "unknown".to_string()
                    },
                    time_ms: result.time_ms,
                    memory_bytes,
                    decisions: stats.decisions,
                    conflicts: stats.conflicts,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    solver_version: env!("CARGO_PKG_VERSION").to_string(),
                };
                tracker.add_entry(entry);
            }
        }
        if let Err(e) = tracker.save()
            && verbosity >= Verbosity::Normal
        {
            eprintln_colored(args, &format!("Warning: Failed to save benchmarks: {}", e));
        }
    }

    // Output results
    output_results(&results, args, &stats);

    // Show statistics if requested
    if (args.stats || args.time || args.memory) && !args.smtcomp {
        print_statistics(&stats, args);
    }

    // Export statistics if requested
    if let Some(ref export_path) = args.export_stats {
        if let Err(e) = export_statistics(&stats, export_path, args) {
            if verbosity >= Verbosity::Normal {
                eprintln_colored(
                    args,
                    &format!("Warning: Failed to export statistics: {}", e),
                );
            }
        } else if verbosity >= Verbosity::Normal {
            println_colored(
                args,
                &format!("Statistics exported to {}", export_path.display()),
                Some(owo_colors::AnsiColors::Green),
            );
        }
    }
}

/// Collect files from input paths, supporting glob patterns and recursive traversal
fn collect_files(inputs: &[PathBuf], recursive: bool) -> Vec<PathBuf> {
    let mut files = Vec::new();

    for input in inputs {
        if input
            .to_str()
            .is_some_and(|s| s.contains('*') || s.contains('?'))
        {
            // Glob pattern
            if let Ok(glob) = Glob::new(input.to_str().unwrap_or("")) {
                let mut builder = GlobSetBuilder::new();
                builder.add(glob);
                if let Ok(set) = builder.build() {
                    let base_dir = input
                        .parent()
                        .and_then(|p| {
                            if p.as_os_str().is_empty() {
                                None
                            } else {
                                Some(p)
                            }
                        })
                        .unwrap_or_else(|| Path::new("."));

                    let walker = if recursive {
                        WalkDir::new(base_dir)
                    } else {
                        WalkDir::new(base_dir).max_depth(1)
                    };

                    for entry in walker.into_iter().filter_map(Result::ok) {
                        if entry.file_type().is_file() {
                            let path = entry.path();
                            if set.is_match(path) {
                                files.push(path.to_path_buf());
                            }
                        }
                    }
                }
            }
        } else if input.is_dir() && recursive {
            // Recursive directory processing
            for entry in WalkDir::new(input)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("smt2") {
                    files.push(path.to_path_buf());
                }
            }
        } else if input.is_file() {
            files.push(input.clone());
        }
    }

    files
}

/// Process files sequentially with optional progress bar
fn process_files_sequential(
    files: &[PathBuf],
    ctx: &mut Context,
    args: &Args,
    verbosity: Verbosity,
    cache: &mut Option<cache::ResultCache>,
) -> Vec<SolverResult> {
    // Show progress bar with ETA if requested
    let progress = if args.progress && verbosity >= Verbosity::Normal {
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} [ETA: {eta_precise}] {msg}",
                )
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    let mut results = Vec::new();
    let start_time = Instant::now();

    for (idx, file) in files.iter().enumerate() {
        if let Some(ref pb) = progress {
            let avg_time_per_file = if idx > 0 {
                start_time.elapsed().as_secs() / idx as u64
            } else {
                0
            };
            let msg = if avg_time_per_file > 0 {
                format!("{} (~{}s/file)", file.display(), avg_time_per_file)
            } else {
                format!("{}", file.display())
            };
            pb.set_message(msg);
        }

        let result = process_single_file(file, ctx, args, cache);
        results.push(result);

        if let Some(ref pb) = progress {
            pb.inc(1);
        }
    }

    if let Some(pb) = progress {
        let total_time = start_time.elapsed();
        pb.finish_with_message(format!(
            "Completed {} files in {:.2}s (avg: {:.2}s/file)",
            files.len(),
            total_time.as_secs_f64(),
            total_time.as_secs_f64() / files.len() as f64
        ));
    }

    results
}

/// Process files in parallel using rayon
fn process_files_parallel(
    files: &[PathBuf],
    _ctx: &Context,
    args: &Args,
    verbosity: Verbosity,
    _cache: &mut Option<cache::ResultCache>,
) -> Vec<SolverResult> {
    let progress = if args.progress && verbosity >= Verbosity::Normal {
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} [ETA: {eta_precise}] Parallel processing...")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    let results: Vec<_> = files
        .par_iter()
        .map(|file| {
            let mut ctx = Context::new();
            if let Some(ref logic) = args.logic {
                ctx.set_logic(logic);
            }

            // Apply resource limits and options
            apply_solver_options(&mut ctx, args);

            // Note: Cache not used in parallel mode to avoid synchronization overhead
            let result = process_single_file(file, &mut ctx, args, &mut None);

            if let Some(ref pb) = progress {
                pb.inc(1);
            }

            result
        })
        .collect();

    if let Some(pb) = progress {
        pb.finish_with_message("Done");
    }

    results
}

/// Process a single file and return the result
fn process_single_file(
    file: &Path,
    ctx: &mut Context,
    args: &Args,
    cache: &mut Option<cache::ResultCache>,
) -> SolverResult {
    let start = Instant::now();

    // Determine input format
    let use_dimacs = args.dimacs
        || args.input_format == Some(InputFormat::Dimacs)
        || file.extension().and_then(|s| s.to_str()) == Some("cnf");

    let use_qdimacs = args.input_format == Some(InputFormat::Qdimacs)
        || file.extension().and_then(|s| s.to_str()) == Some("qdimacs")
        || file.extension().and_then(|s| s.to_str()) == Some("qcnf");

    if use_qdimacs {
        // Parse QDIMACS format
        let file_handle = match fs::File::open(file) {
            Ok(f) => f,
            Err(e) => {
                return SolverResult {
                    file: Some(file.display().to_string()),
                    result: String::new(),
                    error: Some(format!("Failed to open file: {}", e)),
                    time_ms: start.elapsed().as_millis(),
                };
            }
        };

        let reader = BufReader::new(file_handle);
        let qcnf = match dimacs::QDimacsCnf::parse(reader) {
            Ok(c) => c,
            Err(e) => {
                return SolverResult {
                    file: Some(file.display().to_string()),
                    result: String::new(),
                    error: Some(format!("Failed to parse QDIMACS: {}", e)),
                    time_ms: start.elapsed().as_millis(),
                };
            }
        };

        // Convert to SMT-LIB2 and solve
        let script = qcnf.to_smtlib2();
        let result = execute_and_format(ctx, &script, args);

        let time_ms = start.elapsed().as_millis();

        SolverResult {
            file: Some(file.display().to_string()),
            result: result.clone(),
            error: if result.starts_with("(error") {
                Some(result)
            } else {
                None
            },
            time_ms,
        }
    } else if use_dimacs {
        // Parse DIMACS CNF format
        let file_handle = match fs::File::open(file) {
            Ok(f) => f,
            Err(e) => {
                return SolverResult {
                    file: Some(file.display().to_string()),
                    result: String::new(),
                    error: Some(format!("Failed to open file: {}", e)),
                    time_ms: start.elapsed().as_millis(),
                };
            }
        };

        let reader = BufReader::new(file_handle);
        let cnf = match dimacs::DimacsCnf::parse(reader) {
            Ok(c) => c,
            Err(e) => {
                return SolverResult {
                    file: Some(file.display().to_string()),
                    result: String::new(),
                    error: Some(format!("Failed to parse DIMACS: {}", e)),
                    time_ms: start.elapsed().as_millis(),
                };
            }
        };

        // Convert to SMT-LIB2 and solve
        let script = cnf.to_smtlib2();
        let result = execute_and_format(ctx, &script, args);

        // Format output based on args
        let formatted_result = if args.dimacs_output {
            format_dimacs_output(&result, cnf.num_vars)
        } else {
            result.clone()
        };

        let time_ms = start.elapsed().as_millis();

        SolverResult {
            file: Some(file.display().to_string()),
            result: formatted_result.clone(),
            error: if result.starts_with("(error") {
                Some(result)
            } else {
                None
            },
            time_ms,
        }
    } else {
        // Standard SMT-LIB2 format
        let script = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                return SolverResult {
                    file: Some(file.display().to_string()),
                    result: String::new(),
                    error: Some(format!("Failed to read file: {}", e)),
                    time_ms: start.elapsed().as_millis(),
                };
            }
        };

        // Check cache first
        if let Some(cache_ref) = cache
            && let Some(cached_entry) = cache_ref.get(&script)
        {
            return SolverResult {
                file: Some(file.display().to_string()),
                result: cached_entry.result.clone(),
                error: if cached_entry.result.starts_with("(error") {
                    Some(cached_entry.result)
                } else {
                    None
                },
                time_ms: cached_entry.time_ms,
            };
        }

        let result = execute_and_format(ctx, &script, args);
        let time_ms = start.elapsed().as_millis();

        // Store in cache if enabled
        if let Some(cache_ref) = cache {
            cache_ref.put(&script, &result, time_ms);
        }

        SolverResult {
            file: Some(file.display().to_string()),
            result: result.clone(),
            error: if result.starts_with("(error") {
                Some(result)
            } else {
                None
            },
            time_ms,
        }
    }
}

/// Process input from stdin
pub(crate) fn run_stdin(ctx: &mut Context, args: &Args, verbosity: Verbosity) {
    if verbosity >= Verbosity::Verbose {
        eprintln_colored(args, "Reading from stdin...");
    }

    let stdin = io::stdin();
    let mut script = String::new();

    for line in stdin.lock().lines() {
        match line {
            Ok(l) => {
                script.push_str(&l);
                script.push('\n');
            }
            Err(e) => {
                eprintln_colored(args, &format!("Error reading stdin: {}", e));
                std::process::exit(1);
            }
        }
    }

    let start = Instant::now();
    let result = execute_and_format(ctx, &script, args);
    let time_ms = start.elapsed().as_millis();

    let solver_result = SolverResult {
        file: None,
        result: result.clone(),
        error: if result.starts_with("(error") {
            Some(result)
        } else {
            None
        },
        time_ms,
    };

    // Collect SAT solver statistics
    let sat_stats = ctx.stats();

    let stats = SolverStats {
        execution_time_ms: time_ms,
        files_processed: 1,
        memory_bytes: 0,
        peak_memory_bytes: 0,
        success_count: if solver_result.error.is_none() { 1 } else { 0 },
        error_count: if solver_result.error.is_some() { 1 } else { 0 },
        profiling_data: if args.profile {
            Some(vec![ProfilingData {
                operation: "stdin".to_string(),
                duration_us: time_ms * 1000,
                memory_delta_bytes: 0,
            }])
        } else {
            None
        },
        avg_time_per_file_ms: time_ms,
        min_time_ms: time_ms,
        max_time_ms: time_ms,
        decisions: sat_stats.decisions,
        propagations: sat_stats.propagations,
        conflicts: sat_stats.conflicts,
        restarts: sat_stats.restarts,
    };

    output_results(&[solver_result], args, &stats);

    if args.time && !args.smtcomp {
        print_statistics(&stats, args);
    }
}

/// Watch files for changes and automatically reprocess
pub(crate) fn run_watch(ctx: &mut Context, args: &Args, verbosity: Verbosity) {
    let files = collect_files(&args.input, args.recursive);

    if files.is_empty() {
        eprintln_colored(args, "No files found to watch");
        std::process::exit(1);
    }

    if verbosity >= Verbosity::Normal {
        println_colored(
            args,
            &format!("Watching {} file(s) for changes...", files.len()),
            Some(owo_colors::AnsiColors::Cyan),
        );
    }

    let (tx, rx) = channel();

    let mut watcher = notify::recommended_watcher(move |res| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })
    .unwrap_or_else(|e| {
        eprintln_colored(args, &format!("Failed to create watcher: {}", e));
        std::process::exit(1);
    });

    for file in &files {
        if let Err(e) = watcher.watch(file, RecursiveMode::NonRecursive) {
            eprintln_colored(args, &format!("Failed to watch {}: {}", file.display(), e));
        }
    }

    // Initial run
    run_files(ctx, args, verbosity);

    loop {
        match rx.recv() {
            Ok(_event) => {
                if verbosity >= Verbosity::Normal {
                    println_colored(args, "\n--- File changed, rerunning ---", None);
                }
                run_files(ctx, args, verbosity);
            }
            Err(e) => {
                eprintln_colored(args, &format!("Watch error: {}", e));
                break;
            }
        }
    }
}
