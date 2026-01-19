//! REST API Server for OxiZ SMT Solver
//!
//! Provides HTTP REST API endpoints for the SMT solver:
//! - POST /solve - Submit SMT-LIB2 script and get result
//! - POST /check-sat - Quick check-sat endpoint
//! - GET /health - Health check endpoint
//! - GET /version - Get solver version
//! - POST /model - Get model after SAT result
//! - POST /optimize - Run optimization (MaxSMT)

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
};
use oxiz_solver::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

/// Server state shared across all requests
pub struct ServerState {
    /// Shared solver context for stateful operations
    context: Mutex<Context>,
    /// Last SAT result for model retrieval
    last_result: Mutex<Option<LastResult>>,
}

/// Stores the last solving result for model retrieval
#[allow(dead_code)]
struct LastResult {
    status: String,
    script: String,
}

/// Request body for /solve endpoint
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SolveRequest {
    /// SMT-LIB2 script to solve
    pub script: String,
    /// Optional logic to use (e.g., "QF_LIA", "QF_BV")
    #[serde(default)]
    pub logic: Option<String>,
    /// Optional timeout in milliseconds
    #[serde(default)]
    pub timeout_ms: Option<u64>,
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
    /// List of assertions in SMT-LIB2 format
    pub assertions: Vec<String>,
    /// Optional logic to use
    #[serde(default)]
    pub logic: Option<String>,
}

/// Response body for /check-sat endpoint
#[derive(Debug, Serialize)]
pub struct CheckSatResponse {
    /// Result: "sat", "unsat", or "unknown"
    pub result: String,
    /// Time taken in milliseconds
    pub time_ms: u64,
}

/// Request body for /model endpoint
#[derive(Debug, Deserialize)]
pub struct ModelRequest {
    /// Optional: re-solve with this script before getting model
    #[serde(default)]
    pub script: Option<String>,
}

/// Response body for /model endpoint
#[derive(Debug, Serialize)]
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
#[allow(dead_code)]
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
    pub timeout_ms: Option<u64>,
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
        context: Mutex::new(Context::new()),
        last_result: Mutex::new(None),
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

    let mut ctx = state.context.lock().await;

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
            let status = determine_status(&output);
            let model = if status == "sat" {
                extract_model(&output)
            } else {
                None
            };

            // Store last result for model retrieval
            {
                let mut last = state.last_result.lock().await;
                *last = Some(LastResult {
                    status: status.clone(),
                    script: request.script,
                });
            }

            Json(SolveResponse {
                status,
                time_ms: elapsed,
                model,
                error: None,
                output: Some(output),
            })
        }
        Err(e) => Json(SolveResponse {
            status: "error".to_string(),
            time_ms: elapsed,
            model: None,
            error: Some(e.to_string()),
            output: None,
        }),
    }
}

/// Handle POST /check-sat requests
async fn handle_check_sat(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<CheckSatRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    let mut ctx = state.context.lock().await;

    // Reset context for fresh solve
    *ctx = Context::new();

    // Set logic if specified
    if let Some(ref logic) = request.logic {
        ctx.set_logic(logic);
    }

    // Build script from assertions
    let mut script = String::new();
    for assertion in &request.assertions {
        script.push_str(&format!("(assert {})\n", assertion));
    }
    script.push_str("(check-sat)\n");

    // Execute
    let result = ctx.execute_script(&script);

    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            let result = determine_status(&output);
            Json(CheckSatResponse {
                result,
                time_ms: elapsed,
            })
        }
        Err(_) => Json(CheckSatResponse {
            result: "unknown".to_string(),
            time_ms: elapsed,
        }),
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
#[allow(clippy::collapsible_if)]
async fn handle_model(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<ModelRequest>,
) -> impl IntoResponse {
    let mut ctx = state.context.lock().await;

    // If script is provided, solve it first
    if let Some(ref script) = request.script {
        // Reset context for fresh solve
        *ctx = Context::new();

        let result = ctx.execute_script(script);
        if let Ok(output) = result {
            let status = determine_status(&output);
            if status == "sat" {
                // Get model
                let model_result = ctx.execute_script("(get-model)");
                if let Ok(model_output) = model_result {
                    if let Some(model) = extract_model(&model_output) {
                        return Json(ModelResponse {
                            available: true,
                            model: Some(model),
                            error: None,
                        });
                    }
                }
            }
            return Json(ModelResponse {
                available: false,
                model: None,
                error: Some(format!("Result is {}, no model available", status)),
            });
        } else {
            return Json(ModelResponse {
                available: false,
                model: None,
                error: Some("Failed to execute script".to_string()),
            });
        }
    }

    // Otherwise, check if we have a last result
    let last = state.last_result.lock().await;
    if let Some(ref last_result) = *last {
        if last_result.status == "sat" {
            // Get model from current context
            let model_result = ctx.execute_script("(get-model)");
            if let Ok(model_output) = model_result {
                if let Some(model) = extract_model(&model_output) {
                    return Json(ModelResponse {
                        available: true,
                        model: Some(model),
                        error: None,
                    });
                }
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

    let mut ctx = state.context.lock().await;

    // Reset context for fresh solve
    *ctx = Context::new();

    // Enable optimization mode
    ctx.set_option("optimize", "true");

    // Build optimization script
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

            Json(OptimizeResponse {
                status,
                optimal_value,
                time_ms: elapsed,
                model,
                error: None,
            })
        }
        Err(e) => Json(OptimizeResponse {
            status: "error".to_string(),
            optimal_value: None,
            time_ms: elapsed,
            model: None,
            error: Some(e.to_string()),
        }),
    }
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
#[allow(clippy::collapsible_if)]
fn extract_model(output: &[String]) -> Option<std::collections::HashMap<String, String>> {
    let mut model = std::collections::HashMap::new();

    for line in output {
        // Look for define-fun lines like: (define-fun x () Int 42)
        if line.contains("define-fun") {
            if let Some(parsed) = parse_define_fun(line) {
                model.insert(parsed.0, parsed.1);
            }
        }
    }

    if model.is_empty() { None } else { Some(model) }
}

/// Parse a define-fun line and extract variable name and value
fn parse_define_fun(line: &str) -> Option<(String, String)> {
    // Simple parsing for: (define-fun name () type value)
    let trimmed = line.trim();
    if !trimmed.starts_with("(define-fun") {
        return None;
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
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
        let (name, value) = result.unwrap();
        assert_eq!(name, "x");
        assert_eq!(value, "42");
    }
}
