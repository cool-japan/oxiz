//! CLI orchestration for `--checkpoint` / `--resume` / `--resume-from`.
//!
//! These helpers sit on top of the [`crate::checkpoint`] module's
//! (de)serialization primitives and wire them into the solve pipeline: writing
//! a resumable record after a completed solve, and replaying one on resume.
//!
//! This is a *completed-problem* checkpoint, not a pause/resume of an
//! in-progress CDCL search — `oxiz-solver` exposes no hook to snapshot
//! mid-`check-sat` state, so the learned clauses / assignments are honestly
//! left empty (see [`crate::checkpoint::solver_state_from_counts`]). What it
//! genuinely provides is a durable record of a problem and its result that a
//! later `--resume` replays without re-solving.

use std::path::PathBuf;

use oxiz_solver::Context;

use crate::checkpoint;
use crate::{Args, Verbosity};

/// Resolve the checkpoint directory: explicit `--checkpoint-dir`, else the
/// platform config dir under `oxiz/checkpoints`. Never hardcodes an absolute
/// path — the default is derived from the OS config dir.
pub fn checkpoint_dir(args: &Args) -> Option<PathBuf> {
    if let Some(ref dir) = args.checkpoint_dir {
        return Some(dir.clone());
    }
    dirs::config_dir().map(|mut p| {
        p.push("oxiz");
        p.push("checkpoints");
        p
    })
}

/// Attempt to replay a previously-checkpointed result for `script`.
///
/// `--resume-from FILE` loads that specific checkpoint; `--resume` scans the
/// checkpoint directory for one whose problem matches `script` exactly. Returns
/// the recorded output lines when a matching, result-bearing checkpoint exists.
pub fn try_resume(script: &str, args: &Args) -> Option<Vec<String>> {
    if let Some(ref file) = args.resume_from {
        let checkpoint = checkpoint::Checkpoint::load(file).ok()?;
        if checkpoint.problem == script {
            return checkpoint.result_output();
        }
        return None;
    }
    let dir = checkpoint_dir(args)?;
    checkpoint::find_for_problem(&dir, script).and_then(|cp| cp.result_output())
}

/// Persist a completed-solve checkpoint (problem + config + real post-solve
/// counters + full output) so a later `--resume` can replay it.
///
/// Best-effort for the write itself — a write failure warns but never aborts
/// the solve. A system clock that reads a time before the Unix epoch leaves no
/// timestamp to name the checkpoint by; that stops the process with exit
/// status 1 and a message, as any other unusable environment does.
pub fn write(script: &str, args: &Args, ctx: &Context, output: &[String]) {
    let Some(dir) = checkpoint_dir(args) else {
        return;
    };

    let stats = ctx.stats();
    let state = checkpoint::solver_state_from_counts(
        stats.conflicts as usize,
        stats.decisions as usize,
        stats.propagations as usize,
        stats.restarts as usize,
    );

    let status = if output.iter().any(|l| l.trim() == "unsat") {
        "unsat"
    } else if output.iter().any(|l| l.trim() == "sat") {
        "sat"
    } else {
        "unknown"
    };

    let mut progress = checkpoint::ProgressInfo::new(0, 100.0, "completed".to_string());
    progress.metadata.insert(
        "note".to_string(),
        "completed-solve record; mid-search SAT internals are not captured".to_string(),
    );

    let mut cp = match checkpoint::Checkpoint::new(
        script.to_string(),
        args.logic.clone(),
        state,
        progress,
        Vec::new(),
    ) {
        Ok(cp) => cp,
        Err(e) => {
            eprintln!("error: cannot write checkpoint: {e}");
            std::process::exit(1);
        }
    };
    cp.set_result(status, output);

    match cp.save(&dir) {
        Ok(path) => {
            if args.verbosity >= Verbosity::Verbose {
                eprintln!("; checkpoint saved to {}", path.display());
            }
        }
        Err(e) => {
            if !args.quiet {
                eprintln!("warning: failed to write checkpoint: {e}");
            }
        }
    }
}
