//! WebSocket progress server for real-time benchmark monitoring.
//!
//! Build with `--features ws-progress` to enable this binary.
//!
//! # Usage
//!
//! ```text
//! ws_server --root /path/to/benchmarks --port 9090 [--threads N]
//! ```
//!
//! Connect any WebSocket client to `ws://127.0.0.1:<port>/progress` to receive
//! JSON-encoded [`oxiz_smtcomp::ParallelProgress`] events in real time.

#[cfg(not(feature = "ws-progress"))]
fn main() {
    eprintln!(
        "This binary requires the `ws-progress` feature.\n\
         Rebuild with: cargo build -p oxiz-smtcomp --features ws-progress --bin ws_server"
    );
    std::process::exit(1);
}

#[cfg(feature = "ws-progress")]
mod app {
    use oxiz_smtcomp::{
        Loader, LoaderConfig,
        parallel::{ParallelConfig, ParallelRunner},
        websocket::WsProgressServer,
    };
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::Duration;

    /// Parsed command-line arguments.
    struct Args {
        root: PathBuf,
        port: u16,
        threads: usize,
        timeout_secs: u64,
    }

    fn parse_args() -> Result<Args, String> {
        let raw: Vec<String> = std::env::args().collect();
        let mut root: Option<PathBuf> = None;
        let mut port: u16 = 9090;
        let mut threads: usize = 0;
        let mut timeout_secs: u64 = 60;
        let mut i = 1usize;

        while i < raw.len() {
            match raw[i].as_str() {
                "--root" => {
                    i += 1;
                    root = Some(PathBuf::from(
                        raw.get(i)
                            .ok_or_else(|| "--root requires a value".to_string())?,
                    ));
                }
                "--port" => {
                    i += 1;
                    port = raw
                        .get(i)
                        .ok_or_else(|| "--port requires a value".to_string())?
                        .parse::<u16>()
                        .map_err(|e| format!("invalid port: {e}"))?;
                }
                "--threads" => {
                    i += 1;
                    threads = raw
                        .get(i)
                        .ok_or_else(|| "--threads requires a value".to_string())?
                        .parse::<usize>()
                        .map_err(|e| format!("invalid threads: {e}"))?;
                }
                "--timeout" => {
                    i += 1;
                    timeout_secs = raw
                        .get(i)
                        .ok_or_else(|| "--timeout requires a value".to_string())?
                        .parse::<u64>()
                        .map_err(|e| format!("invalid timeout: {e}"))?;
                }
                "--help" | "-h" => {
                    println!(
                        "ws_server -- OxiZ WebSocket progress server\n\n\
                         USAGE:\n\
                         \tws_server --root <PATH> [--port <PORT>] [--threads <N>] [--timeout <SECS>]\n\n\
                         OPTIONS:\n\
                         \t--root     PATH  Benchmark root directory (required)\n\
                         \t--port     PORT  WebSocket port (default: 9090)\n\
                         \t--threads  N     Worker threads, 0 = all cores (default: 0)\n\
                         \t--timeout  SECS  Per-benchmark timeout in seconds (default: 60)"
                    );
                    std::process::exit(0);
                }
                unknown => {
                    return Err(format!("unknown argument: {unknown}"));
                }
            }
            i += 1;
        }

        let root = root.ok_or_else(|| "--root is required".to_string())?;
        Ok(Args {
            root,
            port,
            threads,
            timeout_secs,
        })
    }

    pub async fn run() {
        let args = match parse_args() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("error: {e}\nRun with --help for usage.");
                std::process::exit(1);
            }
        };

        // Discover benchmarks from root.
        let loader_config = LoaderConfig::new(&args.root);
        let loader = Loader::new(loader_config);
        let benchmarks = match loader.discover() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to discover benchmarks under {:?}: {e}", args.root);
                std::process::exit(1);
            }
        };
        if benchmarks.is_empty() {
            eprintln!(
                "No benchmarks found under {:?}. Check the --root path.",
                args.root
            );
            std::process::exit(1);
        }
        eprintln!("Discovered {} benchmarks.", benchmarks.len());

        // Start the WebSocket server.
        let addr: SocketAddr = SocketAddr::from(([127, 0, 0, 1], args.port));
        let ws_server = WsProgressServer::new(addr);
        let callback = ws_server.progress_callback();

        let _handle = ws_server.serve().await;
        eprintln!("WebSocket server started on ws://{addr}/progress");

        // Run the benchmarks in parallel, streaming progress via WebSocket.
        let config = ParallelConfig::new(Duration::from_secs(args.timeout_secs))
            .with_num_threads(args.threads);
        let runner = ParallelRunner::new(config);

        eprintln!("Running benchmarks...");
        let results = runner.run_from_meta_with_progress(&benchmarks, &loader, Some(callback));
        let solved = results
            .iter()
            .filter(|r| {
                matches!(
                    r.status,
                    oxiz_smtcomp::BenchmarkStatus::Sat | oxiz_smtcomp::BenchmarkStatus::Unsat
                )
            })
            .count();
        eprintln!("Done. {}/{} benchmarks solved.", solved, results.len());
    }
}

#[cfg(feature = "ws-progress")]
#[tokio::main]
async fn main() {
    app::run().await;
}
