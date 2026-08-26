//! Interning must retain each `TermKind` exactly once.
//!
//! `TermManager::intern` used to keep three copies of every kind it was
//! handed: one cloned into the `terms` vector, a second owned as the key of an
//! `FxHashMap<(TermKind, SortId), TermId>` intern cache, and -- under the
//! `arena` feature -- a third cloned into a bumpalo arena that nothing ever
//! read and that never freed. For a kind whose payload is large (a string
//! literal, a bignum constant) the payload dominates, so that layout retained
//! 2x the payload by default and 3x with `arena` on.
//!
//! The intern table now stores only a `TermId` and resolves collisions against
//! `terms[id]`, and the write-only arena copy is gone, so a kind is retained
//! exactly once in both configurations. This test pins that with a counting
//! global allocator.
//!
//! # Why this measures a slope
//!
//! The obvious test -- "interning N terms must cost about N x payload" -- does
//! not work at any modest payload size, because `Term` is 152 bytes wide (a
//! `TermKind` is 144: the enum is as large as its largest variant) and the
//! `terms` vector carries up to 2x that in growth slack. At a 100-byte payload
//! the *fixed* per-term overhead is already ~3.4x the payload, which swamps
//! the thing under test.
//!
//! So the measurement is taken twice, at two payload sizes, with everything
//! else held identical: same term count, same number of `Vec` growths, same
//! table capacity. Every fixed cost cancels in the difference, and the
//! marginal bytes retained per extra payload byte is then *exactly* the number
//! of copies kept -- 1.0 for the current layout, 2.0 for the old default one,
//! 3.0 for the old `--all-features` one. The counter records requested
//! (`Layout::size()`) bytes rather than malloc bucket sizes, and both `format!`
//! temporaries and `String::to_string`/`clone` allocate exactly, so the slope
//! is clean rather than merely close.
//!
//! # Why this lives in its own file
//!
//! A `#[global_allocator]` is process-wide, so it would also see allocations
//! made by any other test sharing the binary. This file therefore holds
//! exactly one `#[test]`, and the counter is additionally scoped to the thread
//! that armed it, so harness chatter on other threads cannot leak into the
//! measurement.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use std::alloc::System;

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

thread_local! {
    /// Net bytes handed out to *this* thread since the counter was armed.
    ///
    /// `const`-initialised so that touching it never itself allocates (a
    /// lazily initialised thread-local would re-enter the allocator from
    /// inside `alloc`), and holding a `Cell<isize>` so there is no TLS
    /// destructor to run -- and therefore no teardown window in which the
    /// value is gone but allocations are still arriving.
    static LIVE_BYTES: Cell<isize> = const { Cell::new(0) };

    /// Whether this thread is currently measuring.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Record a net change in live bytes, if the current thread is measuring.
fn record(delta: isize) {
    // `try_with` rather than `with`: during thread teardown a thread-local can
    // already be inaccessible, and a panic raised from inside the allocator
    // would abort the process.
    if ARMED.try_with(Cell::get).unwrap_or(false) {
        let _ = LIVE_BYTES.try_with(|live| live.set(live.get().saturating_add(delta)));
    }
}

/// A `System` passthrough that tracks net live bytes per measuring thread.
struct CountingAllocator;

// SAFETY: every method forwards to `System` unchanged, and only reads the
// returned pointer's nullness for bookkeeping. All the allocator-contract
// obligations are therefore discharged by `System` itself.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(layout.size() as isize);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        record(-(layout.size() as isize));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record(new_size as isize - layout.size() as isize);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run `f`, returning its value and the net live bytes it left behind.
fn measure<T>(f: impl FnOnce() -> T) -> (T, isize) {
    ARMED.with(|armed| armed.set(true));
    LIVE_BYTES.with(|live| live.set(0));
    let value = f();
    let bytes = LIVE_BYTES.with(Cell::get);
    ARMED.with(|armed| armed.set(false));
    (value, bytes)
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

/// Number of distinct string literals interned, per run.
const TERM_COUNT: usize = 10_000;

/// Payload length of the small run.
const SMALL_LEN: usize = 100;

/// Payload length of the large run.
///
/// The gap to `SMALL_LEN` is what the slope is measured over, so it is made
/// large enough that a stray kilobyte of allocator noise cannot shift the
/// result by anything near the 0.5-copy decision margin. Both runs still
/// perform the same *number* of allocations, so noise cancels rather than
/// accumulates.
const LARGE_LEN: usize = 1_100;

/// Extra payload bytes per term between the two runs.
const DELTA_LEN: usize = LARGE_LEN - SMALL_LEN;

/// Interning `TERM_COUNT` literals of `len` bytes, returning the net bytes the
/// interned terms retain.
///
/// The manager is constructed *outside* the measured region so that its fixed
/// start-up cost (sort table, string interner, the pre-interned `true` and
/// `false`) is charged to neither run.
fn retained_bytes(len: usize) -> usize {
    use oxiz_core::ast::TermManager;

    let mut manager = TermManager::new();
    let baseline_terms = manager.term_count();

    let (ids, bytes) = measure(|| {
        let mut ids = Vec::with_capacity(TERM_COUNT);
        for i in 0..TERM_COUNT {
            // Exactly `len` bytes, all distinct, and deliberately sharing
            // their first and last byte so that a hash reading only length and
            // endpoints would collide on every single one.
            let literal = format!("s{i:0width$}e", width = len - 2);
            assert_eq!(literal.len(), len);
            ids.push(manager.mk_string_lit(&literal));
        }
        ids
    });

    // Sanity: this is TERM_COUNT *distinct* terms, not one term interned
    // TERM_COUNT times -- otherwise the measurement would be of nothing.
    assert_eq!(ids.len(), TERM_COUNT);
    assert_eq!(
        manager.term_count() - baseline_terms,
        TERM_COUNT,
        "every literal must have allocated its own term"
    );
    let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
    assert_eq!(unique.len(), TERM_COUNT, "all ids must be distinct");

    // Every term must still be reachable at the end: dropping the manager or
    // the terms early would let the measurement pass for the wrong reason.
    assert!(manager.get(ids[0]).is_some());
    assert!(manager.get(ids[TERM_COUNT - 1]).is_some());

    usize::try_from(bytes).unwrap_or_else(|_| {
        panic!("interning freed more than it allocated ({bytes} net bytes), which cannot happen")
    })
}

/// Upper bound on retained copies of each interned `TermKind`.
///
/// Measured on this test (2026-08-25, aarch64-apple-darwin, `Term` = 152 B),
/// the "before" rows by re-adding the removed copies behind a temporary shim
/// and re-running this same test unchanged:
///
/// | layout                              | 100 B/term | 1100 B/term | slope |
/// |-------------------------------------|-----------:|------------:|------:|
/// | before, default (kind + map key)    |        709 |       2_709 |  2.00 |
/// | before, `--all-features` (+ arena)  |      1_018 |       4_018 |  3.00 |
/// | after, either configuration         |        345 |       1_345 |  1.00 |
///
/// The absolute columns are dominated by the fixed 152-byte `Term` plus `Vec`
/// growth slack (~257 B/term), which is why the bound is on the slope: it
/// isolates the payload, and the payload is what a redundant copy duplicates.
/// 1.5 sits halfway between the single-retention 1.0 this pins and the 2.0 the
/// cheapest regression (bringing back the owned map key) would produce, so
/// there is 50% margin on both sides.
const MAX_RETAINED_COPIES: f64 = 1.5;

#[test]
fn intern_retains_each_term_kind_exactly_once() {
    let small = retained_bytes(SMALL_LEN);
    let large = retained_bytes(LARGE_LEN);

    assert!(
        large > small,
        "the larger payload must retain more, got {large} <= {small}"
    );
    let copies = (large - small) as f64 / (TERM_COUNT * DELTA_LEN) as f64;

    // Reported unconditionally so the numbers are visible under `--nocapture`
    // when the bound ever needs re-deriving on another target.
    println!(
        "{TERM_COUNT} literals: {SMALL_LEN} B/term -> {small} B live \
         ({} B/term); {LARGE_LEN} B/term -> {large} B live ({} B/term); \
         slope = {copies:.2} retained copies per TermKind",
        small / TERM_COUNT,
        large / TERM_COUNT,
    );

    assert!(
        copies <= MAX_RETAINED_COPIES,
        "interning retains {copies:.2} copies of each TermKind \
         (limit {MAX_RETAINED_COPIES}): {DELTA_LEN} extra payload bytes per term \
         cost {} extra bytes per term. A redundant copy of the kind -- an owned \
         intern-table key, or the write-only arena clone -- has been reintroduced",
        (large - small) / TERM_COUNT,
    );
}
