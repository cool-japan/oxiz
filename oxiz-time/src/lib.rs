//! Wasm-safe drop-in replacements for the parts of `std::time` OxiZ uses.
//!
//! # Why this crate exists
//!
//! `std::time::Instant::now()` and `std::time::SystemTime::now()` **abort the
//! process on `wasm32-unknown-unknown`**. That target has no clock at all, so
//! the standard library's implementation is a hard `unreachable`:
//!
//! ```text
//! panicked at library/std/src/sys/time/unsupported.rs:13:9:
//! time not implemented on this platform
//! ```
//!
//! In a browser or in Node this surfaces as `RuntimeError: unreachable`, and
//! because the wasm release profile is `panic = "abort"` it is a trap, not an
//! unwind: the `Solver` / `Context` object that was mid-call is left borrowed
//! and every later call on it fails with *"recursive use of an object detected
//! which would lead to unsafe aliasing in rust"*. The session is unusable.
//!
//! OxiZ reads the clock on its **main `check-sat` path** (the search-start
//! stamp in `oxiz-solver`'s `check_core`, which gates non-convex LIA case-split
//! refinement, is taken on every check, not only when a timeout is set), so
//! before this crate existed *every* `check-sat` with at least one assertion
//! aborted the wasm instance. Routing every clock read through the types here
//! is what makes OxiZ usable from WebAssembly.
//!
//! # What it does
//!
//! * **On every target with a working clock** — which is every target except
//!   `wasm32-unknown-unknown` — [`Instant`], [`SystemTime`], [`SystemTimeError`]
//!   and [`UNIX_EPOCH`] are *re-exports of the `std::time` originals*. They are
//!   the same types, not wrappers: timing behaviour, precision and public API
//!   are bit-identical to using `std::time` directly. `wasm32-wasip1` and
//!   `wasm32-unknown-emscripten` do have clocks and keep the real ones.
//! * **On `wasm32-unknown-unknown`** (and in any `--no-default-features`, i.e.
//!   `no_std`, build) they are *frozen* stand-ins: `Instant::now()` always
//!   reads t = 0, `elapsed()` is always [`Duration::ZERO`], `SystemTime::now()`
//!   is always [`UNIX_EPOCH`]. Nothing traps and nothing allocates.
//!
//! # Consequences of the frozen clock — read this before relying on a timeout
//!
//! On `wasm32-unknown-unknown`:
//!
//! * **`(set-option :timeout N)` and `Solver::set_timeout_ms(N)` become
//!   no-ops.** A deadline is computed as `Instant::now() + N`, which is
//!   `0 + N > 0 = now()`, so it never compares as expired and the search runs
//!   to its natural end. A wasm caller that needs a wall-clock bound must
//!   impose it from the outside: use the conflict budget
//!   (`(set-option :max-conflicts N)` / `Solver::set_max_conflicts`) and/or
//!   terminate the Web Worker running the solve.
//! * Every duration a solver reports (statistics, per-theory timings, the
//!   `time_us` counters in conflict analysis) reads **0**. Measure around the
//!   wasm call from JavaScript instead.
//! * Time-based *heuristic* valves degrade to "always affordable" — which is
//!   exactly the branch OxiZ's own `#[cfg(not(feature = "std"))]` code already
//!   takes. No loop in the workspace uses a clock as its only exit; every one
//!   of them also has a structural bound (iteration cap, finite worklist), so
//!   freezing the clock cannot turn a terminating search into a hang.
//!
//! Frozen mode is detectable at compile time via [`IS_FROZEN`], which lets a
//! consumer statically assert that its *native* builds did not accidentally end
//! up on the stub clock (see the `const _: () = assert!(!oxiz_time::IS_FROZEN)`
//! guards in the other OxiZ crates).

#![no_std]

// The real clock: pulled in only on targets that have one, and only when the
// `std` feature is on. This is what makes `Instant` literally be
// `std::time::Instant` off-wasm.
#[cfg(all(
    feature = "std",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
extern crate std;

/// Re-export of [`core::time::Duration`], which is the exact same type as
/// `std::time::Duration`. Re-exported so a call site can write
/// `use oxiz_time::{Duration, Instant};` in place of
/// `use std::time::{Duration, Instant};`.
pub use core::time::Duration;

/// `true` when this build uses the frozen stub clock instead of a real one.
///
/// That happens on `wasm32-unknown-unknown` (no clock exists) and in `no_std`
/// builds (`--no-default-features`).
///
/// ```
/// if oxiz_time::IS_FROZEN {
///     // `:timeout` cannot fire in this build -- bound the search with
///     // `:max-conflicts`, or from outside the wasm module.
/// }
/// ```
///
/// Being a `const`, it also works in a compile-time assertion, which is how
/// the rest of the workspace proves its *native* builds did not accidentally
/// select the stub clock through a missing `oxiz-time/std` feature forward:
///
/// ```text
/// #[cfg(all(feature = "std", not(all(target_arch = "wasm32", target_os = "unknown"))))]
/// const _: () = assert!(!oxiz_time::IS_FROZEN, "oxiz-time/std must be forwarded");
/// ```
pub const IS_FROZEN: bool = cfg!(any(
    not(feature = "std"),
    all(target_arch = "wasm32", target_os = "unknown")
));

#[cfg(all(
    feature = "std",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
pub use std::time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH};

// Compile-time proof of the "native behaviour is bit-identical" claim: these
// functions only type-check if this crate's exported names *are* the `std`
// ones, so no re-export can silently drift into a wrapper type.
#[cfg(all(
    feature = "std",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
const _: () = {
    const fn _instant_is_std(x: std::time::Instant) -> Instant {
        x
    }
    const fn _system_time_is_std(x: std::time::SystemTime) -> SystemTime {
        x
    }
    assert!(!IS_FROZEN);
};

#[cfg(any(
    not(feature = "std"),
    all(target_arch = "wasm32", target_os = "unknown")
))]
pub use frozen::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH};

/// The frozen clock used on `wasm32-unknown-unknown` and in `no_std` builds.
///
/// Every type here mirrors the public API of its `std::time` counterpart for
/// the operations OxiZ performs, so the call sites are identical on both
/// branches. The only difference is that time never advances.
#[cfg(any(
    not(feature = "std"),
    all(target_arch = "wasm32", target_os = "unknown")
))]
mod frozen {
    use core::fmt;
    use core::ops::{Add, AddAssign, Sub, SubAssign};
    use core::time::Duration;

    /// A monotonic instant that never advances: `now()` always reads t = 0.
    ///
    /// Drop-in for `std::time::Instant`. Arithmetic saturates instead of
    /// panicking on overflow (`std` panics); since every value is t = 0 in
    /// practice this is unobservable, and saturating is the safer behaviour
    /// for a stub.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Instant(Duration);

    impl Instant {
        /// Always returns t = 0. Never traps.
        #[must_use]
        pub const fn now() -> Self {
            Self(Duration::ZERO)
        }

        /// Always [`Duration::ZERO`] — no time can pass on a frozen clock.
        #[must_use]
        pub const fn elapsed(&self) -> Duration {
            Duration::ZERO
        }

        /// Saturating, like [`Self::saturating_duration_since`]; the `std`
        /// version panics when `earlier` is later, which cannot happen here.
        #[must_use]
        pub const fn duration_since(&self, earlier: Self) -> Duration {
            self.0.saturating_sub(earlier.0)
        }

        #[must_use]
        pub const fn saturating_duration_since(&self, earlier: Self) -> Duration {
            self.0.saturating_sub(earlier.0)
        }

        #[must_use]
        pub const fn checked_duration_since(&self, earlier: Self) -> Option<Duration> {
            self.0.checked_sub(earlier.0)
        }

        /// Deadline construction. Returns `Some`, so a caller's
        /// `deadline = now().checked_add(budget)` is `Some(budget)` and the
        /// `now() >= deadline` test stays `false` for every non-zero budget —
        /// i.e. the timeout simply never fires (see the crate docs).
        #[must_use]
        pub const fn checked_add(&self, duration: Duration) -> Option<Self> {
            match self.0.checked_add(duration) {
                Some(d) => Some(Self(d)),
                None => None,
            }
        }

        #[must_use]
        pub const fn checked_sub(&self, duration: Duration) -> Option<Self> {
            match self.0.checked_sub(duration) {
                Some(d) => Some(Self(d)),
                None => None,
            }
        }
    }

    impl Add<Duration> for Instant {
        type Output = Instant;
        fn add(self, rhs: Duration) -> Instant {
            Self(self.0.saturating_add(rhs))
        }
    }

    impl AddAssign<Duration> for Instant {
        fn add_assign(&mut self, rhs: Duration) {
            self.0 = self.0.saturating_add(rhs);
        }
    }

    impl Sub<Duration> for Instant {
        type Output = Instant;
        fn sub(self, rhs: Duration) -> Instant {
            Self(self.0.saturating_sub(rhs))
        }
    }

    impl SubAssign<Duration> for Instant {
        fn sub_assign(&mut self, rhs: Duration) {
            self.0 = self.0.saturating_sub(rhs);
        }
    }

    impl Sub<Instant> for Instant {
        type Output = Duration;
        fn sub(self, rhs: Instant) -> Duration {
            self.0.saturating_sub(rhs.0)
        }
    }

    /// A wall-clock timestamp frozen at the Unix epoch.
    ///
    /// Drop-in for `std::time::SystemTime`. OxiZ only uses `SystemTime` to
    /// stamp unique temporary paths in test helpers; on a frozen clock those
    /// stamps are constant, so they would collide if such a helper were ever
    /// compiled and run for wasm (it never is — those helpers also need
    /// `std::fs` and `std::env::temp_dir`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SystemTime(Duration);

    /// The Unix epoch — and, on a frozen clock, also "now".
    pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::ZERO);

    impl SystemTime {
        /// The anchor of the frozen wall clock, identical to [`UNIX_EPOCH`].
        pub const UNIX_EPOCH: SystemTime = UNIX_EPOCH;

        /// Always [`UNIX_EPOCH`]. Never traps.
        #[must_use]
        pub const fn now() -> Self {
            UNIX_EPOCH
        }

        /// Always `Ok(Duration::ZERO)` for the frozen clock.
        ///
        /// # Errors
        ///
        /// Returns [`SystemTimeError`] when `earlier` is later than `self`,
        /// which a frozen clock cannot produce, but which is kept so the
        /// signature matches `std::time::SystemTime::duration_since`.
        pub const fn duration_since(&self, earlier: Self) -> Result<Duration, SystemTimeError> {
            match self.0.checked_sub(earlier.0) {
                Some(d) => Ok(d),
                None => Err(SystemTimeError(earlier.0.saturating_sub(self.0))),
            }
        }

        /// Always `Ok(Duration::ZERO)` for the frozen clock.
        ///
        /// # Errors
        ///
        /// Never, on a frozen clock; see [`Self::duration_since`].
        pub const fn elapsed(&self) -> Result<Duration, SystemTimeError> {
            Self::now().duration_since(*self)
        }

        #[must_use]
        pub const fn checked_add(&self, duration: Duration) -> Option<Self> {
            match self.0.checked_add(duration) {
                Some(d) => Some(Self(d)),
                None => None,
            }
        }

        #[must_use]
        pub const fn checked_sub(&self, duration: Duration) -> Option<Self> {
            match self.0.checked_sub(duration) {
                Some(d) => Some(Self(d)),
                None => None,
            }
        }
    }

    impl Add<Duration> for SystemTime {
        type Output = SystemTime;
        fn add(self, rhs: Duration) -> SystemTime {
            Self(self.0.saturating_add(rhs))
        }
    }

    impl Sub<Duration> for SystemTime {
        type Output = SystemTime;
        fn sub(self, rhs: Duration) -> SystemTime {
            Self(self.0.saturating_sub(rhs))
        }
    }

    /// Error returned by [`SystemTime::duration_since`] when the argument is
    /// later than the receiver. Mirrors `std::time::SystemTimeError`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SystemTimeError(Duration);

    impl SystemTimeError {
        /// How far the argument was ahead of the receiver.
        #[must_use]
        pub const fn duration(&self) -> Duration {
            self.0
        }
    }

    impl fmt::Display for SystemTimeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "second time provided was later than self")
        }
    }

    impl core::error::Error for SystemTimeError {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holds on both branches: a deadline built from `now()` is always in the
    /// future, so `now() >= deadline` is false right after construction. On
    /// the frozen clock it stays false forever — which is exactly why
    /// `:timeout` is documented as a no-op there.
    #[test]
    fn a_fresh_deadline_has_not_expired() {
        let start = Instant::now();
        let deadline = start
            .checked_add(Duration::from_secs(3600))
            .expect("an hour past now is representable");
        assert!(deadline > start);
        assert!(Instant::now() < deadline);
    }

    #[test]
    fn system_time_is_at_or_after_the_epoch() {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the wall clock is at or after the Unix epoch");
        // Frozen builds read exactly zero here; real ones read decades.
        assert_eq!(since_epoch == Duration::ZERO, IS_FROZEN);
    }

    /// The real-clock branch. Gated on exactly the cfg that selects it, so
    /// `--no-default-features` and wasm builds do not run these.
    #[cfg(all(
        feature = "std",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    ))]
    mod real_clock {
        use super::*;

        /// Requirement: off `wasm32-unknown-unknown`, OxiZ keeps the real
        /// clock. The workspace's test runs are native, so this is the branch
        /// under test; the frozen branch is covered by the wasm build gate.
        #[test]
        fn native_builds_use_the_real_clock() {
            const {
                assert!(
                    !IS_FROZEN,
                    "a native `oxiz-time` build must not select the frozen stub clock"
                )
            };
        }

        #[test]
        fn now_is_monotonic_and_elapsed_is_measurable() {
            let start = Instant::now();
            // Busy-wait rather than sleep: `Duration::ZERO` also satisfies a
            // `>=` test, so spin until the clock actually moves.
            let mut spins: u64 = 0;
            while start.elapsed() == Duration::ZERO {
                spins = spins.wrapping_add(1);
                assert!(spins < 1_000_000_000, "the clock never advanced");
            }
            let later = Instant::now();
            assert!(later > start);
            assert!(later.duration_since(start) > Duration::ZERO);
            assert!(later.checked_duration_since(start).is_some());
            assert!(start.checked_duration_since(later).is_none());
        }
    }

    /// The frozen branch. Not reachable from a native `cargo test`; kept so
    /// the semantics are pinned wherever it *is* compiled.
    #[cfg(any(
        not(feature = "std"),
        all(target_arch = "wasm32", target_os = "unknown")
    ))]
    mod frozen_clock {
        use super::*;

        #[test]
        fn the_clock_never_advances() {
            let start = Instant::now();
            assert!(IS_FROZEN);
            assert_eq!(start.elapsed(), Duration::ZERO);
            assert_eq!(Instant::now(), start);
            assert_eq!(Instant::now().duration_since(start), Duration::ZERO);
        }
    }
}
