//! ML-guided tactic selection wiring for `--ml-tactic-selection`.
//!
//! This is the CLI's integration point for the `oxiz-ml` crate: before a
//! solve it extracts formula features and asks the ML engine to recommend a
//! tactic, applies a conservative (correctness-preserving) solver option for
//! that tactic, and after the solve records the outcome so the model learns.
//! The learned model is persisted to (and reloaded from) a JSON file under the
//! user config directory, so learning accumulates across invocations.
//!
//! The flag is **off by default**; when off, none of this runs and the solve
//! is byte-for-byte unchanged.

use std::path::PathBuf;
use std::time::Duration;

use oxiz_ml::MlTacticEngine;
use oxiz_solver::Context;

use crate::Args;

/// Filename used to persist the learned tactic-selection model.
const MODEL_FILENAME: &str = "ml_tactic_model.json";

/// Environment variable that overrides the model path. It lets a user point
/// the persisted model at a location of their choosing, and lets the
/// integration tests, which run the binary as a child process, keep it out of
/// the real user config dir. The unit tests pass their path to
/// [`begin_with_model_path`] instead, so that they never change the process
/// environment, which tests running on other threads of the same process may
/// read.
const MODEL_PATH_ENV: &str = "OXIZ_ML_MODEL";

/// Resolve the on-disk model path (best-effort). Uses `OXIZ_ML_MODEL` when set,
/// otherwise the platform config dir under `oxiz/`. Returns `None` when neither
/// is available, in which case learning still works within a single run but is
/// not persisted. Never hardcodes an absolute path — it is derived from the
/// environment or the platform config dir.
fn model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(MODEL_PATH_ENV) {
        return Some(PathBuf::from(path));
    }
    dirs::config_dir().map(|mut p| {
        p.push("oxiz");
        p.push(MODEL_FILENAME);
        p
    })
}

/// An in-flight ML tactic-selection decision, created by [`begin`] and closed
/// out by [`MlSession::finish`] once the solve result is known.
pub struct MlSession {
    engine: MlTacticEngine,
    model_path: Option<PathBuf>,
    comment: String,
}

impl MlSession {
    /// The comment line describing the recommendation, to be surfaced in the
    /// solver output.
    pub fn comment(&self) -> &str {
        &self.comment
    }

    /// Record the solve outcome, retrain, and persist the updated model.
    ///
    /// `was_successful` should be `true` when the solve produced a definite
    /// answer (`sat`/`unsat`). Persistence failures are non-fatal (the flag is
    /// advisory) and simply skip saving.
    pub fn finish(mut self, was_successful: bool, elapsed: Duration) {
        self.engine
            .record_outcome(was_successful, elapsed.as_secs_f64(), 0);
        self.engine.retrain_now();

        if let Some(path) = self.model_path.as_ref()
            && let Ok(bytes) = self.engine.save_model()
        {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, bytes);
        }
    }
}

/// Map a recommended tactic to a conservative, correctness-preserving solver
/// option and apply it to the context.
///
/// Only options actually consumed by the solve loop (`simplify`,
/// `theory-mode`) are used, so the recommendation genuinely changes search
/// behaviour without ever changing the answer. The `portfolio` recommendation
/// is left advisory (it does not reroute control flow from here).
fn apply_tactic(ctx: &mut Context, tactic_name: &str) {
    match tactic_name {
        "simplify-preprocess" => ctx.set_option("simplify", "true"),
        "cdcl-core" => ctx.set_option("simplify", "false"),
        "eager-theory" => ctx.set_option("theory-mode", "eager"),
        "lazy-theory" => ctx.set_option("theory-mode", "lazy"),
        _ => {}
    }
}

/// Begin an ML tactic-selection session for `script`: extract features, get a
/// recommendation, apply the corresponding solver option, and return the
/// session (which carries a human-readable comment and will record feedback on
/// [`MlSession::finish`]).
pub fn begin(ctx: &mut Context, script: &str, _args: &Args) -> MlSession {
    begin_with_model_path(ctx, script, model_path())
}

/// [`begin`] with the model persisted at `path` (not persisted when `None`)
/// instead of the path [`model_path`] resolves.
fn begin_with_model_path(ctx: &mut Context, script: &str, path: Option<PathBuf>) -> MlSession {
    let mut engine = MlTacticEngine::new();
    if let Some(ref p) = path
        && let Ok(bytes) = std::fs::read(p)
    {
        // A corrupt/old model file must not break solving: ignore load errors
        // and continue with a fresh engine.
        let _ = engine.load_model(&bytes);
    }

    let recommendation = engine.recommend(script);
    apply_tactic(ctx, recommendation.tactic_name);

    let comment = format!(
        "; ml-tactic-selection: recommended '{}' (id {}, confidence {:.2}, est {:.2}s)",
        recommendation.tactic_name,
        recommendation.tactic_id,
        recommendation.confidence,
        recommendation.estimated_time
    );

    MlSession {
        engine,
        model_path: path,
        comment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_produces_recommendation_comment() {
        // Persist the model to a temp file so the test never touches the real
        // user config dir. The path is handed to the session directly rather
        // than through OXIZ_ML_MODEL: under `cargo test` the crate's unit
        // tests share one process, so changing its environment would race
        // with tests on other threads.
        let model = std::env::temp_dir().join(format!("oxiz_ml_test_{}.json", std::process::id()));

        let mut ctx = Context::new();
        let session = begin_with_model_path(
            &mut ctx,
            "(declare-const x Int)\n(assert (> x 0))\n(check-sat)\n",
            Some(model.clone()),
        );
        assert!(
            session
                .comment()
                .contains("ml-tactic-selection: recommended"),
            "got: {}",
            session.comment()
        );
        // Recording an outcome must not panic.
        session.finish(true, Duration::from_millis(1));

        let _ = std::fs::remove_file(&model);
    }
}
