//! Scratch probe (not part of the deliverable): runs the scripts named by
//! `OXIZ_PROBE_SCRIPT` (comma-separated paths) through `Context` and prints
//! every response line, catching panics.
use oxiz_solver::Context;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
#[ignore = "scratch probe"]
fn probe() {
    let paths = std::env::var("OXIZ_PROBE_SCRIPT").unwrap_or_default();
    for p in paths.split(',').filter(|p| !p.is_empty()) {
        let script = std::fs::read_to_string(p).unwrap_or_default();
        let started = std::time::Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut ctx = Context::new();
            ctx.execute_script(&script)
        }));
        let elapsed = started.elapsed();
        let name = std::path::Path::new(p)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        match r {
            Ok(Ok(lines)) => eprintln!("[{name}] ({elapsed:?}) {}", lines.join(" | ")),
            Ok(Err(e)) => eprintln!("[{name}] ({elapsed:?}) error: {e}"),
            Err(pl) => eprintln!(
                "[{name}] ({elapsed:?}) panic: {}",
                pl.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| pl.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default()
            ),
        }
    }
}
