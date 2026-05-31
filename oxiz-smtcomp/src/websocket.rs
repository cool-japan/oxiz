//! WebSocket-based real-time progress monitoring for parallel benchmark runs.
//!
//! This module provides a lightweight WebSocket server that broadcasts
//! `ParallelProgress` events as JSON to all connected clients.
//!
//! # Feature Gate
//!
//! The entire module is gated behind the `ws-progress` Cargo feature.  A
//! default build of `oxiz-smtcomp` therefore never pulls in `tokio` or `axum`.
//!
//! # Usage
//!
//! ```no_run
//! # #[cfg(feature = "ws-progress")]
//! # {
//! use oxiz_smtcomp::websocket::WsProgressServer;
//! use oxiz_smtcomp::parallel::{ParallelConfig, ParallelRunner};
//! use std::net::SocketAddr;
//! use std::time::Duration;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let addr: SocketAddr = "127.0.0.1:9090".parse().unwrap();
//! let server = WsProgressServer::new(addr);
//! let callback = server.progress_callback();
//!
//! // Hand the callback to the runner.
//! let config = ParallelConfig::new(Duration::from_secs(60));
//! let runner = ParallelRunner::new(config);
//!
//! // Start the server and run benchmarks concurrently.
//! let _handle = server.serve().await;
//! // runner.run_all_with_progress(&benchmarks, Some(callback));
//! # });
//! # }
//! ```

#[cfg(feature = "ws-progress")]
mod inner {
    use crate::parallel::ParallelProgress;
    use axum::{
        Router,
        extract::{
            State,
            ws::{Message, WebSocket, WebSocketUpgrade},
        },
        response::IntoResponse,
        routing::get,
    };
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::sync::broadcast;

    /// A boxed progress callback compatible with [`crate::parallel::ParallelRunner::run_all_with_progress`].
    pub type ProgressCallback = Box<dyn Fn(ParallelProgress) + Send + Sync + 'static>;

    /// Shared server state threaded through Axum handlers.
    struct ServerState {
        sender: broadcast::Sender<String>,
    }

    /// A WebSocket server that broadcasts [`ParallelProgress`] events as JSON.
    ///
    /// Clients connect to `ws://<addr>/progress` and receive one JSON message
    /// per progress event.
    pub struct WsProgressServer {
        addr: SocketAddr,
        sender: broadcast::Sender<String>,
    }

    impl WsProgressServer {
        /// Create a new server bound to the given address.
        ///
        /// The underlying broadcast channel holds up to 256 messages.  Slow
        /// clients that fall more than 256 events behind will miss intermediate
        /// updates (events are not queued indefinitely).
        #[must_use]
        pub fn new(addr: SocketAddr) -> Self {
            let (sender, _) = broadcast::channel(256);
            Self { addr, sender }
        }

        /// Return a progress callback that serialises each [`ParallelProgress`]
        /// to JSON and broadcasts it to every connected WebSocket client.
        ///
        /// Pass the returned value to
        /// [`ParallelRunner::run_all_with_progress`](crate::parallel::ParallelRunner::run_all_with_progress).
        #[must_use]
        pub fn progress_callback(&self) -> ProgressCallback {
            let sender = self.sender.clone();
            Box::new(move |progress: ParallelProgress| {
                match serde_json::to_string(&progress) {
                    Ok(json) => {
                        // Ignore send errors — receivers may have all disconnected.
                        let _ = sender.send(json);
                    }
                    Err(e) => {
                        tracing::warn!("ws-progress: failed to serialise progress event: {e}");
                    }
                }
            })
        }

        /// Spawn the WebSocket server as a background [`tokio::task::JoinHandle`].
        ///
        /// The handle can be awaited to detect server termination, or simply
        /// dropped to let the task run until the process exits.
        pub async fn serve(&self) -> tokio::task::JoinHandle<()> {
            let state = Arc::new(ServerState {
                sender: self.sender.clone(),
            });
            let app = Router::new()
                .route("/progress", get(ws_handler))
                .with_state(state);
            let addr = self.addr;
            tokio::spawn(async move {
                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::error!("ws-progress: failed to bind {addr}: {e}");
                        return;
                    }
                };
                tracing::info!("ws-progress: listening on ws://{addr}/progress");
                if let Err(e) = axum::serve(listener, app).await {
                    tracing::error!("ws-progress: server error: {e}");
                }
            })
        }
    }

    /// Axum handler — upgrades an HTTP connection to WebSocket.
    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(state): State<Arc<ServerState>>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| handle_socket(socket, state))
    }

    /// Drive a single WebSocket connection, forwarding broadcast messages.
    async fn handle_socket(socket: WebSocket, state: Arc<ServerState>) {
        let rx = state.sender.subscribe();
        pump_socket(socket, rx).await;
    }

    /// Forward messages from the broadcast receiver to the WebSocket until the
    /// client disconnects or the channel is closed.
    async fn pump_socket(mut socket: WebSocket, mut rx: broadcast::Receiver<String>) {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    if socket.send(Message::Text(msg.into())).await.is_err() {
                        // Client disconnected.
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("ws-progress: client lagged, skipped {n} messages");
                }
            }
        }
    }
}

#[cfg(feature = "ws-progress")]
pub use inner::{ProgressCallback, WsProgressServer};
