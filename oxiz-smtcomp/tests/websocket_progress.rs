//! Integration tests for the WebSocket progress API.
//!
//! These tests are feature-gated; they only compile and run when
//! `--features ws-progress` is passed to Cargo.

#![cfg(feature = "ws-progress")]

use futures_util::StreamExt;
use oxiz_smtcomp::websocket::WsProgressServer;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::time::timeout;

/// Bind an OS-assigned port and return a listener + its address.
async fn bind_free_port() -> Option<(tokio::net::TcpListener, SocketAddr)> {
    let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
        Ok(listener) => listener,
        Err(err) if err.kind() == ErrorKind::PermissionDenied => return None,
        Err(err) => panic!("failed to bind free port: {err}"),
    };
    let addr = listener.local_addr().expect("failed to get local addr");
    Some((listener, addr))
}

#[tokio::test]
async fn test_receives_progress_events() {
    // 1. Pick an OS-assigned port so we don't collide with anything.
    let Some((probe, addr)) = bind_free_port().await else {
        return;
    };
    drop(probe); // release the port; WsProgressServer will re-bind it

    // 2. Start the WebSocket server.
    let server = WsProgressServer::new(addr);
    let callback = server.progress_callback();
    let _handle = server.serve().await;

    // Give the server a moment to start accepting connections.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 3. Connect a tokio-tungstenite client.
    let url = format!("ws://{addr}/progress");
    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("WebSocket client failed to connect");
    let (_, mut read) = ws_stream.split();

    // 4. Send a synthetic progress event via the callback.
    let progress = oxiz_smtcomp::ParallelProgress {
        total: 10,
        completed: 3,
        solved: 2,
        errors: 0,
        elapsed: Duration::from_millis(500),
    };
    callback(progress);

    // 5. Assert that at least one message is received within 5 seconds.
    let msg = timeout(Duration::from_secs(5), read.next())
        .await
        .expect("timed out waiting for WebSocket message")
        .expect("stream ended before message")
        .expect("WebSocket error");

    let text = msg.into_text().expect("message is not text");
    assert!(
        text.contains("completed"),
        "expected progress JSON to contain 'completed', got: {text}"
    );
    assert!(
        text.contains("total"),
        "expected progress JSON to contain 'total', got: {text}"
    );
}

#[tokio::test]
async fn test_multiple_clients_receive_events() {
    let Some((probe, addr)) = bind_free_port().await else {
        return;
    };
    drop(probe);

    let server = WsProgressServer::new(addr);
    let callback = server.progress_callback();
    let _handle = server.serve().await;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let url = format!("ws://{addr}/progress");

    // Connect two clients.
    let (ws1, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("client 1 failed");
    let (ws2, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("client 2 failed");
    let (_, mut read1) = ws1.split();
    let (_, mut read2) = ws2.split();

    let progress = oxiz_smtcomp::ParallelProgress {
        total: 5,
        completed: 1,
        solved: 1,
        errors: 0,
        elapsed: Duration::from_millis(100),
    };
    callback(progress);

    // Both clients should receive the message.
    let msg1 = timeout(Duration::from_secs(5), read1.next())
        .await
        .expect("client 1 timed out")
        .expect("stream ended")
        .expect("client 1 ws error");
    let msg2 = timeout(Duration::from_secs(5), read2.next())
        .await
        .expect("client 2 timed out")
        .expect("stream ended")
        .expect("client 2 ws error");

    let text1 = msg1.into_text().expect("not text");
    let text2 = msg2.into_text().expect("not text");
    assert_eq!(text1, text2, "both clients should receive the same JSON");
}

#[tokio::test]
async fn test_progress_json_fields() {
    let Some((probe, addr)) = bind_free_port().await else {
        return;
    };
    drop(probe);

    let server = WsProgressServer::new(addr);
    let callback = server.progress_callback();
    let _handle = server.serve().await;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let url = format!("ws://{addr}/progress");
    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("connect failed");
    let (_, mut read) = ws_stream.split();

    callback(oxiz_smtcomp::ParallelProgress {
        total: 100,
        completed: 42,
        solved: 30,
        errors: 5,
        elapsed: Duration::from_secs(7),
    });

    let msg = timeout(Duration::from_secs(5), read.next())
        .await
        .expect("timed out")
        .expect("stream ended")
        .expect("ws error");

    let text = msg.into_text().expect("not text");
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("failed to parse JSON from WebSocket");

    assert_eq!(value["total"], 100);
    assert_eq!(value["completed"], 42);
    assert_eq!(value["solved"], 30);
    assert_eq!(value["errors"], 5);
    // elapsed is serialised as { secs: N, nanos: N }
    assert!(
        value.get("elapsed").is_some(),
        "elapsed field must be present"
    );
}
