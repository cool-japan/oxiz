//! Tests for lock-free data structures

use super::*;
use std::sync::Arc;
use std::thread;

#[test]
fn test_lockfree_queue_basic() {
    let queue = LockFreeQueue::new();
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);

    queue.push(1);
    queue.push(2);
    queue.push(3);

    assert_eq!(queue.len(), 3);
    assert!(!queue.is_empty());

    assert_eq!(queue.pop(), Some(1));
    assert_eq!(queue.pop(), Some(2));
    assert_eq!(queue.pop(), Some(3));
    assert_eq!(queue.pop(), None);
    assert!(queue.is_empty());
}

#[test]
fn test_lockfree_queue_concurrent_push() {
    let queue = Arc::new(LockFreeQueue::new());
    let mut handles = vec![];

    for i in 0..4 {
        let q = Arc::clone(&queue);
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                q.push(i * 100 + j);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(queue.len(), 400);

    // Drain all items
    let mut count = 0;
    while queue.pop().is_some() {
        count += 1;
    }
    assert_eq!(count, 400);
}

#[test]
fn test_lockfree_queue_concurrent_push_pop() {
    let queue = Arc::new(LockFreeQueue::new());
    let mut handles = vec![];

    // Producers
    for i in 0..2 {
        let q = Arc::clone(&queue);
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                q.push(i * 100 + j);
            }
        }));
    }

    // Consumers
    let consumed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for _ in 0..2 {
        let q = Arc::clone(&queue);
        let c = Arc::clone(&consumed);
        handles.push(thread::spawn(move || {
            let mut local_count = 0;
            for _ in 0..200 {
                if q.pop().is_some() {
                    local_count += 1;
                }
                // Yield to allow other threads
                thread::yield_now();
            }
            c.fetch_add(local_count, std::sync::atomic::Ordering::Relaxed);
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Drain remaining items
    let mut remaining = 0;
    while queue.pop().is_some() {
        remaining += 1;
    }

    let total = consumed.load(std::sync::atomic::Ordering::Relaxed) + remaining;
    assert_eq!(total, 200);
}

#[test]
fn test_lockfree_queue_drop() {
    let queue = LockFreeQueue::new();
    queue.push(String::from("hello"));
    queue.push(String::from("world"));
    // Drop should clean up all nodes
    drop(queue);
}

#[test]
fn test_lockfree_queue_empty_pop() {
    let queue: LockFreeQueue<i32> = LockFreeQueue::new();
    assert_eq!(queue.pop(), None);
    assert_eq!(queue.pop(), None);
}

#[test]
fn test_clause_sharing_basic() {
    let sharing = LockFreeClauseSharing::new();

    // Export a small clause
    assert!(sharing.export_clause(vec![1, -2, 3], 2, 0));

    // Import it
    let clause = sharing.import_clause();
    assert!(clause.is_some());
    let clause = clause.expect("clause should be available");
    assert_eq!(clause.literals, vec![1, -2, 3]);
    assert_eq!(clause.lbd, 2);
    assert_eq!(clause.source_id, 0);
}

#[test]
fn test_clause_sharing_quality_filter() {
    let sharing = LockFreeClauseSharing::new();

    // Too long clause should be rejected
    let long_clause: Vec<i32> = (0..20).collect();
    assert!(!sharing.export_clause(long_clause, 1, 0));

    // High LBD clause should be rejected
    assert!(!sharing.export_clause(vec![1, 2], 10, 0));

    let stats = sharing.statistics();
    assert_eq!(stats.2, 2); // 2 rejected
}

#[test]
fn test_clause_sharing_batch_import() {
    let sharing = LockFreeClauseSharing::new();

    for i in 0..5 {
        sharing.export_clause(vec![i, i + 1], 1, 0);
    }

    let batch = sharing.import_batch(3);
    assert_eq!(batch.len(), 3);
    assert_eq!(sharing.pending_count(), 2);
}

#[test]
fn test_clause_sharing_concurrent() {
    let sharing = Arc::new(LockFreeClauseSharing::new());
    let mut handles = vec![];

    // Producers
    for id in 0..4u32 {
        let s = Arc::clone(&sharing);
        handles.push(thread::spawn(move || {
            for i in 0..10i32 {
                s.export_clause(vec![i, i + 1], 1, id);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    let stats = sharing.statistics();
    assert_eq!(stats.0, 40); // 40 exported

    // Import all
    let mut count = 0;
    while sharing.import_clause().is_some() {
        count += 1;
    }
    assert_eq!(count, 40);
}

#[test]
fn test_clause_sharing_custom_params() {
    let sharing = LockFreeClauseSharing::with_params(4, 2);

    // Clause len 5 > max 4 should be rejected
    assert!(!sharing.export_clause(vec![1, 2, 3, 4, 5], 1, 0));

    // Clause with LBD 3 > max 2 should be rejected
    assert!(!sharing.export_clause(vec![1, 2], 3, 0));

    // Valid clause
    assert!(sharing.export_clause(vec![1, 2, 3], 2, 0));
}
