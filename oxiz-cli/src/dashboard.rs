//! Web-based dashboard for monitoring solver progress
//!
//! This module provides a real-time web dashboard for monitoring SAT/SMT solver
//! statistics, including conflicts, decisions, propagations, memory usage, and more.

#![allow(dead_code)]

use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::{Html, IntoResponse},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use tokio::sync::broadcast;

/// Dashboard state shared across all connections
#[derive(Debug)]
pub struct DashboardState {
    /// Broadcast channel for sending updates to all WebSocket clients
    pub tx: broadcast::Sender<DashboardStats>,
    /// Current solver statistics
    pub stats: DashboardStats,
    /// Whether solving is currently paused
    pub paused: AtomicBool,
    /// Whether solving should be cancelled
    pub cancelled: AtomicBool,
    /// Current restart threshold
    pub restart_threshold: AtomicU64,
    /// Current phase
    pub phase: std::sync::Mutex<String>,
    /// Start time for elapsed calculation
    pub start_time: std::sync::Mutex<Option<std::time::Instant>>,
}

/// Solver statistics for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardStats {
    /// Number of conflicts
    pub conflicts: u64,
    /// Number of decisions
    pub decisions: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Number of restarts
    pub restarts: u64,
    /// Number of learned clauses
    pub learned_clauses: u64,
    /// Number of deleted clauses
    pub deleted_clauses: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Time elapsed in milliseconds
    pub time_elapsed_ms: u64,
    /// Current phase (preprocessing, solving, etc.)
    pub phase: String,
    /// Current decision level
    pub decision_level: u32,
    /// Average LBD of learned clauses
    pub avg_lbd: f64,
    /// Number of binary clauses
    pub binary_clauses: u64,
    /// Number of unit clauses
    pub unit_clauses: u64,
    /// Whether solving is paused
    pub paused: bool,
    /// Learned clause size distribution (buckets: 1-2, 3-5, 6-10, 11-20, 21+)
    pub clause_size_distribution: Vec<u64>,
    /// Variable activity levels (top 20 most active)
    pub top_variable_activities: Vec<f64>,
    /// Decision level history (last 100 values)
    pub decision_level_history: Vec<u32>,
}

impl DashboardState {
    /// Create a new dashboard state
    #[must_use]
    pub fn new() -> Arc<Self> {
        let (tx, _) = broadcast::channel(100);
        Arc::new(Self {
            tx,
            stats: DashboardStats::default(),
            paused: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            restart_threshold: AtomicU64::new(100),
            phase: std::sync::Mutex::new("idle".to_string()),
            start_time: std::sync::Mutex::new(None),
        })
    }

    /// Update statistics from the solver
    pub fn update_stats(&self, stats: DashboardStats) {
        // Ignore send errors (no receivers)
        let _ = self.tx.send(stats);
    }

    /// Check if solving is paused
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Check if solving is cancelled
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Set the current phase
    pub fn set_phase(&self, phase: &str) {
        if let Ok(mut p) = self.phase.lock() {
            *p = phase.to_string();
        }
    }

    /// Start timing
    pub fn start_timing(&self) {
        if let Ok(mut start) = self.start_time.lock() {
            *start = Some(std::time::Instant::now());
        }
    }

    /// Get elapsed time in milliseconds
    #[must_use]
    #[allow(clippy::collapsible_if)]
    pub fn elapsed_ms(&self) -> u64 {
        if let Ok(start) = self.start_time.lock() {
            if let Some(s) = *start {
                return s.elapsed().as_millis() as u64;
            }
        }
        0
    }
}

/// Command from the web interface to control the solver
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DashboardCommand {
    /// Pause solving
    Pause,
    /// Resume solving
    Resume,
    /// Cancel solving
    Cancel,
    /// Update a parameter
    SetParameter { name: String, value: String },
}

/// Start the dashboard HTTP server
pub async fn start_dashboard_server(
    state: Arc<DashboardState>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = Router::new()
        .route("/", get(serve_dashboard))
        .route("/ws", get(websocket_handler))
        .route("/api/stats", get(get_stats))
        .route("/api/pause", get(pause_solver))
        .route("/api/resume", get(resume_solver))
        .route("/api/cancel", get(cancel_solver))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    eprintln!("Dashboard server running at http://localhost:{}", port);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Serve the main dashboard HTML page
async fn serve_dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

/// Handle WebSocket connections
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

/// Handle an individual WebSocket connection
#[allow(clippy::collapsible_if)]
async fn handle_websocket(mut socket: WebSocket, state: Arc<DashboardState>) {
    let mut rx = state.tx.subscribe();

    // Send initial stats
    let initial_stats = DashboardStats {
        phase: state
            .phase
            .lock()
            .map_or_else(|_| "unknown".to_string(), |p| p.clone()),
        paused: state.is_paused(),
        time_elapsed_ms: state.elapsed_ms(),
        ..Default::default()
    };

    if let Ok(json) = serde_json::to_string(&initial_stats) {
        let _ = socket.send(Message::Text(json.into())).await;
    }

    loop {
        tokio::select! {
            // Handle incoming messages from client
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(cmd) = serde_json::from_str::<DashboardCommand>(&text) {
                            handle_command(&state, cmd);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
            // Forward stats updates to client
            stats = rx.recv() => {
                if let Ok(stats) = stats {
                    if let Ok(json) = serde_json::to_string(&stats) {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Handle a command from the web interface
#[allow(clippy::collapsible_if)]
fn handle_command(state: &DashboardState, cmd: DashboardCommand) {
    match cmd {
        DashboardCommand::Pause => {
            state.paused.store(true, Ordering::Relaxed);
        }
        DashboardCommand::Resume => {
            state.paused.store(false, Ordering::Relaxed);
        }
        DashboardCommand::Cancel => {
            state.cancelled.store(true, Ordering::Relaxed);
        }
        DashboardCommand::SetParameter { name, value } => {
            if name == "restart_threshold" {
                if let Ok(v) = value.parse::<u64>() {
                    state.restart_threshold.store(v, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Get current stats as JSON
async fn get_stats(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let stats = DashboardStats {
        phase: state
            .phase
            .lock()
            .map_or_else(|_| "unknown".to_string(), |p| p.clone()),
        paused: state.is_paused(),
        time_elapsed_ms: state.elapsed_ms(),
        ..Default::default()
    };

    axum::Json(stats)
}

/// Pause the solver
async fn pause_solver(State(state): State<Arc<DashboardState>>) -> &'static str {
    state.paused.store(true, Ordering::Relaxed);
    "paused"
}

/// Resume the solver
async fn resume_solver(State(state): State<Arc<DashboardState>>) -> &'static str {
    state.paused.store(false, Ordering::Relaxed);
    "resumed"
}

/// Cancel the solver
async fn cancel_solver(State(state): State<Arc<DashboardState>>) -> &'static str {
    state.cancelled.store(true, Ordering::Relaxed);
    "cancelled"
}

/// Embedded HTML/CSS/JS for the dashboard
const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OxiZ Solver Dashboard</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --accent-green: #4caf50;
            --accent-yellow: #ffc107;
            --accent-blue: #2196f3;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: var(--bg-secondary);
            border-radius: 10px;
        }

        .header h1 {
            font-size: 1.5em;
            color: var(--accent);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }

        .status-dot.paused {
            background: var(--accent-yellow);
            animation: none;
        }

        .status-dot.disconnected {
            background: var(--accent);
            animation: none;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            font-size: 1em;
            color: var(--text-secondary);
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--accent-blue);
        }

        .stat-label {
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        .progress-container {
            margin-top: 15px;
        }

        .progress-bar {
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent));
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .chart-container {
            height: 200px;
            position: relative;
            margin-top: 10px;
        }

        .chart-canvas {
            width: 100%;
            height: 100%;
        }

        .heatmap {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 2px;
            margin-top: 10px;
        }

        .heatmap-cell {
            aspect-ratio: 1;
            border-radius: 2px;
            background: var(--bg-secondary);
            transition: background 0.3s ease;
        }

        .control-panel {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-pause {
            background: var(--accent-yellow);
            color: #000;
        }

        .btn-resume {
            background: var(--accent-green);
            color: #fff;
        }

        .btn-cancel {
            background: var(--accent);
            color: #fff;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .param-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
        }

        .param-control label {
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .param-control input {
            background: var(--bg-secondary);
            border: 1px solid var(--bg-card);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 5px;
            width: 100px;
        }

        .distribution-bars {
            display: flex;
            align-items: flex-end;
            justify-content: space-around;
            height: 150px;
            margin-top: 10px;
            padding: 10px 0;
        }

        .dist-bar {
            width: 40px;
            background: linear-gradient(180deg, var(--accent-blue), var(--accent));
            border-radius: 4px 4px 0 0;
            transition: height 0.3s ease;
            position: relative;
        }

        .dist-bar::after {
            content: attr(data-label);
            position: absolute;
            bottom: -25px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.75em;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .phase-indicator {
            display: inline-block;
            padding: 5px 15px;
            background: var(--accent-blue);
            border-radius: 20px;
            font-size: 0.9em;
            margin-left: 10px;
        }

        .time-display {
            font-family: 'Courier New', monospace;
            font-size: 2em;
            color: var(--accent-green);
            text-align: center;
            margin: 10px 0;
        }

        .memory-bar {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 10px;
        }

        .memory-usage {
            flex: 1;
        }

        .memory-text {
            font-size: 1.2em;
            color: var(--accent-blue);
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>OxiZ Solver Dashboard</h1>
            <span class="phase-indicator" id="phase">Idle</span>
        </div>
        <div class="status-indicator">
            <span id="connection-status">Connected</span>
            <div class="status-dot" id="status-dot"></div>
        </div>
    </div>

    <div class="grid">
        <!-- Main Statistics Card -->
        <div class="card">
            <h2>Core Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="conflicts">0</div>
                    <div class="stat-label">Conflicts</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="decisions">0</div>
                    <div class="stat-label">Decisions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="propagations">0</div>
                    <div class="stat-label">Propagations</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="restarts">0</div>
                    <div class="stat-label">Restarts</div>
                </div>
            </div>
        </div>

        <!-- Clause Statistics -->
        <div class="card">
            <h2>Clause Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="learned-clauses">0</div>
                    <div class="stat-label">Learned Clauses</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="deleted-clauses">0</div>
                    <div class="stat-label">Deleted Clauses</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="binary-clauses">0</div>
                    <div class="stat-label">Binary Clauses</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="avg-lbd">0.00</div>
                    <div class="stat-label">Avg LBD</div>
                </div>
            </div>
        </div>

        <!-- Time and Memory -->
        <div class="card">
            <h2>Time & Memory</h2>
            <div class="time-display" id="time-elapsed">00:00:00</div>
            <div class="memory-bar">
                <div class="memory-usage">
                    <div class="memory-text" id="memory-usage">0 MB</div>
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" id="memory-progress" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="card">
            <h2>Control Panel</h2>
            <div class="control-panel">
                <button class="btn btn-pause" id="btn-pause" onclick="pauseSolver()">
                    <span>Pause</span>
                </button>
                <button class="btn btn-resume" id="btn-resume" onclick="resumeSolver()" disabled>
                    <span>Resume</span>
                </button>
                <button class="btn btn-cancel" id="btn-cancel" onclick="cancelSolver()">
                    <span>Cancel</span>
                </button>
            </div>
            <div class="param-control">
                <label for="restart-threshold">Restart Threshold:</label>
                <input type="number" id="restart-threshold" value="100" min="1" onchange="updateParameter('restart_threshold', this.value)">
            </div>
        </div>

        <!-- Decision Level Graph -->
        <div class="card">
            <h2>Decision Level History</h2>
            <div class="chart-container">
                <canvas id="decision-level-chart" class="chart-canvas"></canvas>
            </div>
        </div>

        <!-- Learned Clause Size Distribution -->
        <div class="card">
            <h2>Clause Size Distribution</h2>
            <div class="distribution-bars" id="clause-distribution">
                <div class="dist-bar" data-label="1-2" style="height: 10%"></div>
                <div class="dist-bar" data-label="3-5" style="height: 10%"></div>
                <div class="dist-bar" data-label="6-10" style="height: 10%"></div>
                <div class="dist-bar" data-label="11-20" style="height: 10%"></div>
                <div class="dist-bar" data-label="21+" style="height: 10%"></div>
            </div>
        </div>

        <!-- Variable Activity Heatmap -->
        <div class="card">
            <h2>Variable Activity (Top 20)</h2>
            <div class="heatmap" id="activity-heatmap">
                <!-- Will be populated by JS -->
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let decisionLevelHistory = [];
        const MAX_HISTORY_POINTS = 100;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('status-dot').classList.remove('disconnected');
            };

            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('status-dot').classList.add('disconnected');
                // Reconnect after 2 seconds
                setTimeout(connect, 2000);
            };

            ws.onerror = function(err) {
                console.error('WebSocket error:', err);
            };

            ws.onmessage = function(event) {
                const stats = JSON.parse(event.data);
                updateDashboard(stats);
            };
        }

        function updateDashboard(stats) {
            // Update core statistics
            document.getElementById('conflicts').textContent = formatNumber(stats.conflicts);
            document.getElementById('decisions').textContent = formatNumber(stats.decisions);
            document.getElementById('propagations').textContent = formatNumber(stats.propagations);
            document.getElementById('restarts').textContent = formatNumber(stats.restarts);

            // Update clause statistics
            document.getElementById('learned-clauses').textContent = formatNumber(stats.learned_clauses);
            document.getElementById('deleted-clauses').textContent = formatNumber(stats.deleted_clauses);
            document.getElementById('binary-clauses').textContent = formatNumber(stats.binary_clauses);
            document.getElementById('avg-lbd').textContent = stats.avg_lbd.toFixed(2);

            // Update time
            document.getElementById('time-elapsed').textContent = formatTime(stats.time_elapsed_ms);

            // Update memory
            const memoryMB = stats.memory_bytes / (1024 * 1024);
            document.getElementById('memory-usage').textContent = memoryMB.toFixed(1) + ' MB';
            // Assume 4GB max for progress bar
            const memoryPercent = Math.min((memoryMB / 4096) * 100, 100);
            document.getElementById('memory-progress').style.width = memoryPercent + '%';

            // Update phase
            document.getElementById('phase').textContent = stats.phase || 'Idle';

            // Update pause state
            if (stats.paused) {
                document.getElementById('status-dot').classList.add('paused');
                document.getElementById('btn-pause').disabled = true;
                document.getElementById('btn-resume').disabled = false;
            } else {
                document.getElementById('status-dot').classList.remove('paused');
                document.getElementById('btn-pause').disabled = false;
                document.getElementById('btn-resume').disabled = true;
            }

            // Update decision level history
            if (stats.decision_level !== undefined) {
                decisionLevelHistory.push(stats.decision_level);
                if (decisionLevelHistory.length > MAX_HISTORY_POINTS) {
                    decisionLevelHistory.shift();
                }
                drawDecisionLevelChart();
            }

            // Update clause size distribution
            if (stats.clause_size_distribution && stats.clause_size_distribution.length > 0) {
                updateClauseDistribution(stats.clause_size_distribution);
            }

            // Update activity heatmap
            if (stats.top_variable_activities && stats.top_variable_activities.length > 0) {
                updateActivityHeatmap(stats.top_variable_activities);
            }
        }

        function formatNumber(num) {
            if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        }

        function formatTime(ms) {
            const seconds = Math.floor(ms / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            return String(hours).padStart(2, '0') + ':' +
                   String(minutes % 60).padStart(2, '0') + ':' +
                   String(seconds % 60).padStart(2, '0');
        }

        function drawDecisionLevelChart() {
            const canvas = document.getElementById('decision-level-chart');
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();

            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const width = rect.width;
            const height = rect.height;
            const padding = 10;

            ctx.clearRect(0, 0, width, height);

            if (decisionLevelHistory.length < 2) return;

            const maxLevel = Math.max(...decisionLevelHistory, 1);
            const xStep = (width - 2 * padding) / (MAX_HISTORY_POINTS - 1);

            // Draw grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padding + (height - 2 * padding) * i / 4;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
            }

            // Draw line
            ctx.strokeStyle = '#2196f3';
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < decisionLevelHistory.length; i++) {
                const x = padding + i * xStep;
                const y = height - padding - (decisionLevelHistory[i] / maxLevel) * (height - 2 * padding);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            // Fill area under line
            ctx.fillStyle = 'rgba(33, 150, 243, 0.2)';
            ctx.lineTo(padding + (decisionLevelHistory.length - 1) * xStep, height - padding);
            ctx.lineTo(padding, height - padding);
            ctx.closePath();
            ctx.fill();
        }

        function updateClauseDistribution(distribution) {
            const bars = document.querySelectorAll('#clause-distribution .dist-bar');
            const maxVal = Math.max(...distribution, 1);

            bars.forEach((bar, i) => {
                if (i < distribution.length) {
                    const percent = (distribution[i] / maxVal) * 100;
                    bar.style.height = Math.max(percent, 5) + '%';
                }
            });
        }

        function updateActivityHeatmap(activities) {
            const container = document.getElementById('activity-heatmap');
            container.innerHTML = '';

            const maxActivity = Math.max(...activities, 0.001);

            for (let i = 0; i < 20; i++) {
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';

                if (i < activities.length) {
                    const intensity = activities[i] / maxActivity;
                    const r = Math.round(15 + intensity * 218);
                    const g = Math.round(49 + intensity * 120);
                    const b = Math.round(96 - intensity * 16);
                    cell.style.background = `rgb(${r}, ${g}, ${b})`;
                    cell.title = `Var ${i + 1}: ${activities[i].toFixed(4)}`;
                }

                container.appendChild(cell);
            }
        }

        function pauseSolver() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'Pause' }));
            }
        }

        function resumeSolver() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'Resume' }));
            }
        }

        function cancelSolver() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'Cancel' }));
            }
        }

        function updateParameter(name, value) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'SetParameter',
                    name: name,
                    value: value
                }));
            }
        }

        // Initialize
        connect();

        // Handle window resize
        window.addEventListener('resize', () => {
            drawDecisionLevelChart();
        });
    </script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_creation() {
        let state = DashboardState::new();
        assert!(!state.is_paused());
        assert!(!state.is_cancelled());
    }

    #[test]
    fn test_dashboard_pause_resume() {
        let state = DashboardState::new();

        state.paused.store(true, Ordering::Relaxed);
        assert!(state.is_paused());

        state.paused.store(false, Ordering::Relaxed);
        assert!(!state.is_paused());
    }

    #[test]
    fn test_dashboard_phase() {
        let state = DashboardState::new();
        state.set_phase("solving");

        let phase = state.phase.lock().unwrap();
        assert_eq!(*phase, "solving");
    }
}
