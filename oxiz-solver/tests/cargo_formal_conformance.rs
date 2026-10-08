//! Runs cargo-formal's conformance fixtures (`crates/formal-conformance/
//! fixtures/*.smt2`, each paired with a `.expected` sidecar whose first line
//! is the standard-conformant verdict and whose optional second line
//! `upstream: <id>` marks a fixture the pinned crates.io OxiZ is known to get
//! wrong) against *this* tree.
//!
//! The fixtures live in the sibling repository, so this test is `#[ignore]`d
//! and reads their directory from `OXIZ_CONFORMANCE_FIXTURES`; without the
//! variable it is a no-op.  It reports one line per fixture and the agree
//! count, and fails only if an **untagged** fixture disagrees — a tagged one
//! that now agrees is progress to report upstream (the tag is stale), not a
//! failure here.
//!
//! The verdict reduction mirrors cargo-formal's `formal-conformance`: an
//! `(error …)` response line wins over any `check-sat` answer, a panic inside
//! `execute_script` is folded into `error`, otherwise the last
//! `sat`/`unsat`/`unknown` line is the verdict.

use oxiz_solver::Context;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

const KNOWN_VERDICTS: [&str; 4] = ["sat", "unsat", "unknown", "error"];

fn reduce_to_verdict(output: &[String]) -> Option<&'static str> {
    if output.iter().any(|line| line.starts_with("(error ")) {
        return Some("error");
    }
    output
        .iter()
        .rev()
        .find_map(|line| KNOWN_VERDICTS.iter().find(|&&v| v == line).copied())
}

fn run_script(script: &str) -> Vec<String> {
    match catch_unwind(AssertUnwindSafe(|| {
        let mut ctx = Context::new();
        ctx.execute_script(script)
    })) {
        Ok(Ok(lines)) => lines,
        Ok(Err(err)) => vec![format!("(error \"execute_script returned Err: {err}\")")],
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            vec![format!("(error \"oxiz panicked: {message}\")")]
        }
    }
}

struct Row {
    name: String,
    expected: String,
    actual: String,
    upstream: Option<String>,
}

fn run_fixture(smt2: &Path) -> Option<Row> {
    let name = smt2.file_stem()?.to_string_lossy().into_owned();
    let script = std::fs::read_to_string(smt2).ok()?;
    let expected_text = std::fs::read_to_string(smt2.with_extension("expected")).ok()?;
    let mut lines = expected_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty());
    let expected = lines.next()?.to_string();
    let upstream = lines
        .next()
        .and_then(|l| l.strip_prefix("upstream:"))
        .map(|rest| rest.trim().to_string());
    let output = run_script(&script);
    let actual = reduce_to_verdict(&output)
        .unwrap_or("no-verdict")
        .to_string();
    Some(Row {
        name,
        expected,
        actual,
        upstream,
    })
}

#[test]
#[ignore = "needs OXIZ_CONFORMANCE_FIXTURES pointing at cargo-formal's fixtures directory"]
fn cargo_formal_conformance_fixtures() {
    let Ok(dir) = std::env::var("OXIZ_CONFORMANCE_FIXTURES") else {
        eprintln!("OXIZ_CONFORMANCE_FIXTURES not set; nothing to run");
        return;
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        panic!("cannot list {dir}");
    };
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "smt2"))
        .collect();
    paths.sort();
    let rows: Vec<Row> = paths.iter().filter_map(|p| run_fixture(p)).collect();
    let mut agree = 0usize;
    let mut untagged_disagreements = Vec::new();
    for row in &rows {
        let agrees = row.expected == row.actual;
        if agrees {
            agree += 1;
        }
        let tag = row.upstream.as_deref().unwrap_or("-");
        eprintln!(
            "{:<40} expected {:<8} actual {:<8} {} (upstream: {tag})",
            row.name,
            row.expected,
            row.actual,
            if agrees { "agree" } else { "DISAGREE" }
        );
        if !agrees && row.upstream.is_none() {
            untagged_disagreements.push(row.name.clone());
        }
    }
    eprintln!("agree {agree}/{}", rows.len());
    assert!(
        untagged_disagreements.is_empty(),
        "untagged fixtures disagree: {untagged_disagreements:?}"
    );
}
