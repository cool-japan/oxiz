//! REST API Server for OxiZ SMT Solver
//!
//! Provides HTTP REST API endpoints for the SMT solver:
//! - POST /solve - Submit SMT-LIB2 script and get result
//! - POST /check-sat - Quick check-sat endpoint
//! - GET /health - Health check endpoint
//! - GET /version - Get solver version
//! - POST /model - Get model after SAT result
//! - POST /optimize - Run optimization (MaxSMT)
//!
//! # Session isolation
//!
//! All state-bearing requests (`/solve`, `/check-sat`, `/model`,
//! `/optimize`) accept an optional `session_id` string. Each distinct
//! `session_id` gets its own private [`Context`] and last-result slot, so
//! concurrent clients using different session ids cannot observe or clobber
//! each other's solver state. Clients that omit `session_id` all share a
//! single `"default"` session -- this reproduces the older, unisolated
//! behavior for backward compatibility, so it still has the pre-existing
//! cross-client race if multiple unlabeled clients hit the server
//! concurrently. Clients that need isolation must supply a stable, unique
//! `session_id` themselves.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use oxiz_solver::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

fn default_session_id() -> String {
    "default".to_string()
}

/// Per-client solving state: an independent [`Context`] plus the last
/// solve's result, keyed by `session_id` in [`ServerState`].
struct Session {
    context: Mutex<Context>,
    last_result: Mutex<Option<LastResult>>,
}

impl Session {
    fn new() -> Self {
        Self {
            context: Mutex::new(Context::new()),
            last_result: Mutex::new(None),
        }
    }
}

/// Server state shared across all requests
pub struct ServerState {
    /// Sessions, keyed by client-supplied `session_id` (see module docs).
    sessions: Mutex<HashMap<String, Arc<Session>>>,
}

/// Look up (creating if necessary) the session for `session_id`.
async fn session_for(state: &ServerState, session_id: &str) -> Arc<Session> {
    let mut sessions = state.sessions.lock().await;
    Arc::clone(
        sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Session::new())),
    )
}

/// Stores the last solving result for model retrieval
struct LastResult {
    status: String,
}

/// Request body for /solve endpoint
#[derive(Debug, Deserialize)]
pub struct SolveRequest {
    /// SMT-LIB2 script to solve
    pub script: String,
    /// Optional logic to use (e.g., "QF_LIA", "QF_BV")
    #[serde(default)]
    pub logic: Option<String>,
    /// Optional timeout in milliseconds
    #[serde(default)]
    #[allow(dead_code)]
    pub timeout_ms: Option<u64>,
    /// See module docs on session isolation.
    #[serde(default = "default_session_id")]
    pub session_id: String,
}

/// Response body for /solve endpoint
#[derive(Debug, Serialize)]
pub struct SolveResponse {
    /// Result status: "sat", "unsat", "unknown", or "error"
    pub status: String,
    /// Time taken in milliseconds
    pub time_ms: u64,
    /// Model (if SAT and model is available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<std::collections::HashMap<String, String>>,
    /// Error message (if status is "error")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Full output from solver
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<String>>,
}

/// Request body for /check-sat endpoint
#[derive(Debug, Deserialize)]
pub struct CheckSatRequest {
    /// List of assertions in SMT-LIB2 format (used when `script` is not
    /// given). Each is wrapped as `(assert <assertion>)`.
    #[serde(default)]
    pub assertions: Vec<String>,
    /// Declarations needed by `assertions` (e.g. `"(declare-const x Int)"`),
    /// executed before them. Without these, any assertion mentioning a
    /// variable would previously fail to parse. Ignored when `script` is
    /// given.
    #[serde(default)]
    pub declarations: Vec<String>,
    /// A full SMT-LIB2 script to execute verbatim (declarations, asserts,
    /// and optionally its own `(check-sat)`). When given, `assertions` and
    /// `declarations` are ignored.
    #[serde(default)]
    pub script: Option<String>,
    /// Optional logic to use
    #[serde(default)]
    pub logic: Option<String>,
    /// See module docs on session isolation.
    #[serde(default = "default_session_id")]
    pub session_id: String,
}

/// Response body for /check-sat endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckSatResponse {
    /// Result: "sat", "unsat", "unknown", or "error"
    pub result: String,
    /// Time taken in milliseconds
    pub time_ms: u64,
    /// Error message when `result` is "error" -- a parse or execution
    /// failure is never masked as "unknown".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Request body for /model endpoint
#[derive(Debug, Deserialize)]
pub struct ModelRequest {
    /// Optional: re-solve with this script before getting model
    #[serde(default)]
    pub script: Option<String>,
    /// See module docs on session isolation.
    #[serde(default = "default_session_id")]
    pub session_id: String,
}

/// Response body for /model endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Whether model is available
    pub available: bool,
    /// The model as variable -> value mapping
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<std::collections::HashMap<String, String>>,
    /// Error message if model is not available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Request body for /optimize endpoint
#[derive(Debug, Deserialize)]
pub struct OptimizeRequest {
    /// SMT-LIB2 script with optimization objectives
    pub script: String,
    /// Optimization direction: "minimize" or "maximize"
    #[serde(default = "default_direction")]
    pub direction: String,
    /// Variable to optimize
    pub objective: String,
    /// Optional timeout in milliseconds
    #[serde(default)]
    #[allow(dead_code)]
    pub timeout_ms: Option<u64>,
    /// See module docs on session isolation.
    #[serde(default = "default_session_id")]
    pub session_id: String,
}

fn default_direction() -> String {
    "minimize".to_string()
}

/// Response body for /optimize endpoint
#[derive(Debug, Serialize)]
pub struct OptimizeResponse {
    /// Result status: "optimal", "sat", "unsat", "unknown", or "error"
    pub status: String,
    /// Optimal value found (if optimal)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimal_value: Option<String>,
    /// Time taken in milliseconds
    pub time_ms: u64,
    /// Model at optimal point
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<std::collections::HashMap<String, String>>,
    /// Error message (if status is "error")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response body for /health endpoint
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
}

/// Response body for /version endpoint
#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
}

/// Server startup time for health check
static START_TIME: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Create the REST API router
pub fn create_router() -> Router {
    // Initialize start time
    START_TIME.get_or_init(Instant::now);

    let state = Arc::new(ServerState {
        sessions: Mutex::new(HashMap::new()),
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/solve", post(handle_solve))
        .route("/check-sat", post(handle_check_sat))
        .route("/health", get(handle_health))
        .route("/version", get(handle_version))
        .route("/model", post(handle_model))
        .route("/optimize", post(handle_optimize))
        .layer(cors)
        .with_state(state)
}

/// Run the REST API server
pub async fn run_server(port: u16) -> anyhow::Result<()> {
    let router = create_router();

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;

    tracing::info!("OxiZ REST API server listening on http://0.0.0.0:{}", port);
    println!("OxiZ REST API server listening on http://0.0.0.0:{}", port);
    println!("Available endpoints:");
    println!("  POST /solve     - Submit SMT-LIB2 script and get result");
    println!("  POST /check-sat - Quick check-sat endpoint");
    println!("  GET  /health    - Health check endpoint");
    println!("  GET  /version   - Get solver version");
    println!("  POST /model     - Get model after SAT result");
    println!("  POST /optimize  - Run optimization (MaxSMT)");

    axum::serve(listener, router).await?;

    Ok(())
}

/// Handle POST /solve requests
async fn handle_solve(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<SolveRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    let session = session_for(&state, &request.session_id).await;
    let mut ctx = session.context.lock().await;

    // Reset context for fresh solve
    *ctx = Context::new();

    // Set logic if specified
    if let Some(ref logic) = request.logic {
        ctx.set_logic(logic);
    }

    // Execute the script
    let result = ctx.execute_script(&request.script);

    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            if let Some(err_line) = find_error_line(&output) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(SolveResponse {
                        status: "error".to_string(),
                        time_ms: elapsed,
                        model: None,
                        error: Some(err_line),
                        output: Some(output),
                    }),
                );
            }

            let status = determine_status(&output);
            let model = if status == "sat" {
                extract_model(&output)
            } else {
                None
            };

            // Store last result for model retrieval
            {
                let mut last = session.last_result.lock().await;
                *last = Some(LastResult {
                    status: status.clone(),
                });
            }

            (
                StatusCode::OK,
                Json(SolveResponse {
                    status,
                    time_ms: elapsed,
                    model,
                    error: None,
                    output: Some(output),
                }),
            )
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(SolveResponse {
                status: "error".to_string(),
                time_ms: elapsed,
                model: None,
                error: Some(e.to_string()),
                output: None,
            }),
        ),
    }
}

/// Build the SMT-LIB2 script for a `/check-sat` request: either the raw
/// `script` field verbatim (appending `(check-sat)` if the caller forgot
/// it), or `declarations` + `assertions` + `(check-sat)`.
fn build_check_sat_script(request: &CheckSatRequest) -> String {
    if let Some(ref script) = request.script {
        if script.contains("check-sat") {
            script.clone()
        } else {
            format!("{}\n(check-sat)\n", script)
        }
    } else {
        let mut script = String::new();
        for decl in &request.declarations {
            script.push_str(decl);
            script.push('\n');
        }
        for assertion in &request.assertions {
            script.push_str(&format!("(assert {})\n", assertion));
        }
        script.push_str("(check-sat)\n");
        script
    }
}

/// Handle POST /check-sat requests
async fn handle_check_sat(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<CheckSatRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    let session = session_for(&state, &request.session_id).await;
    let mut ctx = session.context.lock().await;

    // Reset context for fresh solve
    *ctx = Context::new();

    // Set logic if specified
    if let Some(ref logic) = request.logic {
        ctx.set_logic(logic);
    }

    let script = build_check_sat_script(&request);

    // Execute
    let result = ctx.execute_script(&script);

    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            if let Some(err_line) = find_error_line(&output) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(CheckSatResponse {
                        result: "error".to_string(),
                        time_ms: elapsed,
                        error: Some(err_line),
                    }),
                );
            }

            let result = determine_status(&output);
            (
                StatusCode::OK,
                Json(CheckSatResponse {
                    result,
                    time_ms: elapsed,
                    error: None,
                }),
            )
        }
        // A parse/execution failure is a real error, not "unknown" -- an
        // "unknown" result is reserved for a solver that genuinely could
        // not decide satisfiability.
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(CheckSatResponse {
                result: "error".to_string(),
                time_ms: elapsed,
                error: Some(e.to_string()),
            }),
        ),
    }
}

/// Handle GET /health requests
async fn handle_health() -> impl IntoResponse {
    let uptime = START_TIME.get().map(|t| t.elapsed().as_secs()).unwrap_or(0);

    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds: uptime,
    })
}

/// Handle GET /version requests
async fn handle_version() -> impl IntoResponse {
    Json(VersionResponse {
        name: "OxiZ SMT Solver".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        features: vec![
            "QF_LIA".to_string(),
            "QF_LRA".to_string(),
            "QF_BV".to_string(),
            "QF_AUFLIA".to_string(),
            "QF_UF".to_string(),
            "Optimization".to_string(),
            "Incremental".to_string(),
            "Proofs".to_string(),
        ],
    })
}

/// Handle POST /model requests
async fn handle_model(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<ModelRequest>,
) -> impl IntoResponse {
    let session = session_for(&state, &request.session_id).await;
    let mut ctx = session.context.lock().await;

    // If script is provided, solve it first
    if let Some(ref script) = request.script {
        // Reset context for fresh solve
        *ctx = Context::new();

        let result = ctx.execute_script(script);
        return match result {
            Ok(output) => {
                let status = determine_status(&output);
                if status == "sat" {
                    // Get model
                    let model_result = ctx.execute_script("(get-model)");
                    if let Ok(model_output) = model_result
                        && let Some(model) = extract_model(&model_output)
                    {
                        return Json(ModelResponse {
                            available: true,
                            model: Some(model),
                            error: None,
                        });
                    }
                }
                Json(ModelResponse {
                    available: false,
                    model: None,
                    error: Some(format!("Result is {}, no model available", status)),
                })
            }
            Err(e) => Json(ModelResponse {
                available: false,
                model: None,
                error: Some(format!("Failed to execute script: {}", e)),
            }),
        };
    }

    // Otherwise, use this session's last result -- never another client's,
    // since `session` is keyed by the caller's own `session_id`.
    let last = session.last_result.lock().await;
    if let Some(ref last_result) = *last {
        if last_result.status == "sat" {
            // Get model from this session's own context
            let model_result = ctx.execute_script("(get-model)");
            if let Ok(model_output) = model_result
                && let Some(model) = extract_model(&model_output)
            {
                return Json(ModelResponse {
                    available: true,
                    model: Some(model),
                    error: None,
                });
            }
        }
        return Json(ModelResponse {
            available: false,
            model: None,
            error: Some(format!(
                "Last result was {}, no model available",
                last_result.status
            )),
        });
    }

    Json(ModelResponse {
        available: false,
        model: None,
        error: Some("No previous solve result available".to_string()),
    })
}

/// Handle POST /optimize requests
async fn handle_optimize(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<OptimizeRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    let session = session_for(&state, &request.session_id).await;
    let mut ctx = session.context.lock().await;

    // Reset context for fresh solve
    *ctx = Context::new();

    // Enable optimization mode
    ctx.set_option("optimize", "true");

    // Build optimization command
    let direction_cmd = if request.direction == "maximize" {
        format!("(maximize {})", request.objective)
    } else {
        format!("(minimize {})", request.objective)
    };

    // Insert optimization command before check-sat
    let script = if request.script.contains("(check-sat)") {
        request
            .script
            .replace("(check-sat)", &format!("{}\n(check-sat)", direction_cmd))
    } else {
        format!("{}\n{}\n(check-sat)", request.script, direction_cmd)
    };

    // Execute
    let result = ctx.execute_script(&script);

    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            if let Some(err_line) = find_error_line(&output) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(OptimizeResponse {
                        status: "error".to_string(),
                        optimal_value: None,
                        time_ms: elapsed,
                        model: None,
                        error: Some(err_line),
                    }),
                );
            }

            let status = determine_optimization_status(&output);
            let (optimal_value, model) = if status == "optimal" || status == "sat" {
                // Try to extract optimal value and model
                let model = extract_model(&output);
                let optimal = model
                    .as_ref()
                    .and_then(|m| m.get(&request.objective).cloned());
                (optimal, model)
            } else {
                (None, None)
            };

            (
                StatusCode::OK,
                Json(OptimizeResponse {
                    status,
                    optimal_value,
                    time_ms: elapsed,
                    model,
                    error: None,
                }),
            )
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(OptimizeResponse {
                status: "error".to_string(),
                optimal_value: None,
                time_ms: elapsed,
                model: None,
                error: Some(e.to_string()),
            }),
        ),
    }
}

/// Find the first `(error ...)` line in solver output, if any. `execute_script`
/// generally returns `Err` for parse failures, but some command-level
/// failures are instead reported as an `(error ...)` string inside an
/// otherwise-`Ok` output; both must surface as a real error, never as a
/// silently-downgraded "unknown".
fn find_error_line(output: &[String]) -> Option<String> {
    output
        .iter()
        .find(|line| line.trim_start().starts_with("(error"))
        .cloned()
}

/// Determine the status from solver output
fn determine_status(output: &[String]) -> String {
    for line in output {
        let trimmed = line.trim().to_lowercase();
        if trimmed == "sat" || trimmed.starts_with("sat") {
            return "sat".to_string();
        }
        if trimmed == "unsat" || trimmed.starts_with("unsat") {
            return "unsat".to_string();
        }
        if trimmed == "unknown" || trimmed.starts_with("unknown") {
            return "unknown".to_string();
        }
    }
    "unknown".to_string()
}

/// Determine optimization status from solver output
fn determine_optimization_status(output: &[String]) -> String {
    for line in output {
        let trimmed = line.trim().to_lowercase();
        if trimmed.contains("optimal") {
            return "optimal".to_string();
        }
    }
    determine_status(output)
}

/// Extract model from solver output
///
/// `Context::format_model` (invoked by the `(get-model)` command) returns
/// its whole `(model ...)` block as a *single* string with embedded
/// newlines, not one output entry per line -- so each entry must itself be
/// split on `\n` before scanning for `define-fun` lines, or every
/// `(get-model)` response fails to yield any bindings.
fn extract_model(output: &[String]) -> Option<std::collections::HashMap<String, String>> {
    let mut model = std::collections::HashMap::new();

    for entry in output {
        for line in entry.lines() {
            // Look for define-fun lines like: (define-fun x () Int 42)
            if line.contains("define-fun")
                && let Some(parsed) = parse_define_fun(line)
            {
                model.insert(parsed.0, parsed.1);
            }
        }
    }

    if model.is_empty() { None } else { Some(model) }
}

/// Parse a define-fun line and extract variable name and value
///
/// Only the *constant* form `(define-fun name () Sort value)` yields a
/// binding.  Since `#P2b-34` a model also prints an interpretation for every
/// declared function, `(define-fun f ((x!0 S)) T (ite ...))`, and that line
/// names no variable: taking its last whitespace-separated token would report
/// the innermost `ite` branch as if it were `f`'s value.
fn parse_define_fun(line: &str) -> Option<(String, String)> {
    // Simple parsing for: (define-fun name () type value)
    let trimmed = line.trim();
    if !trimmed.starts_with("(define-fun") {
        return None;
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.get(2).is_some_and(|params| *params != "()") {
        return None;
    }
    if parts.len() >= 5 {
        let name = parts[1].to_string();
        // Value is the last part before the closing paren
        let value_part = parts.last()?;
        let value = value_part.trim_end_matches(')').to_string();
        return Some((name, value));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    #[test]
    fn test_determine_status() {
        assert_eq!(determine_status(&["sat".to_string()]), "sat");
        assert_eq!(determine_status(&["unsat".to_string()]), "unsat");
        assert_eq!(determine_status(&["unknown".to_string()]), "unknown");
        assert_eq!(determine_status(&["SAT".to_string()]), "sat");
    }

    #[test]
    fn test_parse_define_fun() {
        let result = parse_define_fun("(define-fun x () Int 42)");
        assert!(result.is_some());
        let (name, value) = result.expect("test operation should succeed");
        assert_eq!(name, "x");
        assert_eq!(value, "42");
    }

    #[test]
    fn test_parse_define_fun_skips_a_function_interpretation() {
        // `#P2b-34` made `(get-model)` print an interpretation for every
        // declared function.  Those lines carry a parameter list and a term
        // body, and name no variable -- the naive "last token" reading would
        // have bound `f` to the `ite`'s else-branch.
        let line = "  (define-fun f ((x!0 (_ BitVec 8))) (_ BitVec 8)                     (ite (= x!0 #x01) #x07 #x00))";
        assert!(parse_define_fun(line).is_none());
        // The constant form on the same model block is still read.
        let result = parse_define_fun("  (define-fun x () Int 42)");
        assert_eq!(
            result,
            Some(("x".to_string(), "42".to_string())),
            "a zero-arity define-fun is a variable binding"
        );
    }

    #[test]
    fn test_extract_model_from_multiline_model_block() {
        // `(get-model)` returns its whole block as ONE string with embedded
        // newlines, e.g. via `Context::format_model`. `extract_model` must
        // still find the bindings inside it.
        let output = vec!["(model\n\n  (define-fun p () Bool true)\n)".to_string()];
        let model = extract_model(&output).expect("should find a binding");
        assert_eq!(model.get("p"), Some(&"true".to_string()));
    }

    #[test]
    fn test_find_error_line() {
        assert_eq!(find_error_line(&["sat".to_string()]), None);
        assert_eq!(
            find_error_line(&["(error \"bad\")".to_string()]),
            Some("(error \"bad\")".to_string())
        );
    }

    async fn body_json<T: for<'de> Deserialize<'de>>(response: axum::response::Response) -> T {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("valid json")
    }

    #[tokio::test]
    async fn test_check_sat_with_declarations_does_not_error() {
        // Previously, assertions referencing any variable would fail to
        // parse because no declarations were ever sent, and that failure
        // was masked as "unknown" rather than surfaced as an error.
        let app = create_router();
        let body = serde_json::json!({
            "declarations": ["(declare-const x Int)"],
            "assertions": ["(> x 0)"]
        });
        let request = Request::builder()
            .method("POST")
            .uri("/check-sat")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
            .expect("build request");
        let response = app.oneshot(request).await.expect("request should succeed");
        assert_eq!(response.status(), StatusCode::OK);
        let parsed: CheckSatResponse = body_json(response).await;
        assert_eq!(parsed.result, "sat");
        assert!(parsed.error.is_none());
    }

    #[tokio::test]
    async fn test_check_sat_undeclared_variable_is_a_real_error_not_unknown() {
        let app = create_router();
        // No declarations at all: `x` is undeclared.
        let body = serde_json::json!({
            "assertions": ["(> x 0)"]
        });
        let request = Request::builder()
            .method("POST")
            .uri("/check-sat")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
            .expect("build request");
        let response = app.oneshot(request).await.expect("request should succeed");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let parsed: CheckSatResponse = body_json(response).await;
        assert_eq!(parsed.result, "error");
        assert!(parsed.error.is_some());
    }

    #[tokio::test]
    async fn test_model_is_isolated_per_session() {
        let app = create_router();

        // Client A solves a SAT problem under session "a".
        let solve_a = serde_json::json!({
            "script": "(declare-const p Bool)\n(assert p)\n(check-sat)\n",
            "session_id": "session-a"
        });
        let request = Request::builder()
            .method("POST")
            .uri("/solve")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&solve_a).expect("serialize")))
            .expect("build request");
        let response = app.clone().oneshot(request).await.expect("request ok");
        assert_eq!(response.status(), StatusCode::OK);

        // Client B, under a *different* session, has solved nothing yet.
        let model_b = serde_json::json!({ "session_id": "session-b" });
        let request = Request::builder()
            .method("POST")
            .uri("/model")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&model_b).expect("serialize")))
            .expect("build request");
        let response = app.clone().oneshot(request).await.expect("request ok");
        let parsed: ModelResponse = body_json(response).await;
        // Client B must NOT see client A's model.
        assert!(!parsed.available);

        // Client A can still retrieve its own model under its own session.
        let model_a = serde_json::json!({ "session_id": "session-a" });
        let request = Request::builder()
            .method("POST")
            .uri("/model")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&model_a).expect("serialize")))
            .expect("build request");
        let response = app.oneshot(request).await.expect("request ok");
        let parsed: ModelResponse = body_json(response).await;
        assert!(parsed.available);
    }
}
