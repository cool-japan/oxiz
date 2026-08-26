# oxiz-time

Wasm-safe drop-in replacements for the parts of `std::time` OxiZ uses.

## Why This Crate Exists

`std::time::Instant::now()` and `std::time::SystemTime::now()` **abort the
process on `wasm32-unknown-unknown`** — that target has no clock at all, so
the standard library's implementation is a hard `unreachable!()`. Because the
wasm release profile is `panic = "abort"`, this is a trap, not an unwind: the
`Solver` / `Context` object that was mid-call is left borrowed, and every
later call on it fails with *"recursive use of an object detected which would
lead to unsafe aliasing in rust"*.

OxiZ reads the clock on its main `check-sat` path, so before this crate
existed, every `(check-sat)` with at least one assertion permanently
poisoned the session on `wasm32-unknown-unknown`. Routing every clock read
through the types here is what makes OxiZ usable from WebAssembly.

The crate has no dependencies and is `no_std`; it sits below `oxiz-math` at
the bottom of the workspace dependency graph, so every other crate in the
workspace depends on it.

## What It Provides

| Type | On every target except `wasm32-unknown-unknown` | On `wasm32-unknown-unknown` (or `no_std`) |
|:-----|:--------------------------------------------------|:-------------------------------------------|
| `Instant` | re-export of `std::time::Instant` | frozen stub: `now()` always reads t = 0, `elapsed()` is always `Duration::ZERO` |
| `SystemTime`, `UNIX_EPOCH` | re-exports of `std::time::{SystemTime, UNIX_EPOCH}` | frozen stub: `now()` always returns `UNIX_EPOCH` |
| `SystemTimeError` | re-export of `std::time::SystemTimeError` | stub with the same `duration()` / `Display` / `Error` surface |

`wasm32-wasip1` and `wasm32-unknown-emscripten` do have working clocks and
keep the real types — the stub is scoped to `wasm32-unknown-unknown`
specifically. Off that one target, `Instant` and `SystemTime` are not
wrappers; they are the literal `std::time` types, so native timing behaviour
and precision are bit-identical to using `std::time` directly.

Also re-exported: `Duration` (`core::time::Duration`, identical on both
branches) and `IS_FROZEN: bool`, a `const` that is `true` exactly when the
stubs are active.

## Usage

```rust
use oxiz_time::{Duration, Instant};

let start = Instant::now();
// ... do work ...
if start.elapsed() > Duration::from_millis(500) {
    // budget exceeded — never observed on `wasm32-unknown-unknown`, where
    // the clock is frozen and `elapsed()` always reads `Duration::ZERO`
}

if oxiz_time::IS_FROZEN {
    // `:timeout` cannot fire in this build -- bound the search with
    // `:max-conflicts`, or from outside the wasm module.
}
```

## Feature Flags

- **`std`** (default): selects the real `std::time` re-exports on every
  target that has a working clock. Turning it off — or building for
  `wasm32-unknown-unknown`, which has no clock regardless of this flag —
  selects the frozen stubs instead.

## `wasm32-unknown-unknown` and Timeouts

On the frozen clock, nothing ever traps, but a wall-clock deadline can never
expire: `(set-option :timeout N)` / `Solver::set_timeout_ms(N)` becomes a
no-op there, and any duration OxiZ reports (statistics, per-theory timings)
reads `0`. Bound a wasm search a different way instead — `(set-option
:max-conflicts N)` / `Solver::set_max_conflicts`, or by terminating the Web
Worker running it — and measure wall time from JavaScript around the call if
you need it.

## License

Apache-2.0

---

Part of [OxiZ](https://github.com/cool-japan/oxiz), a Pure Rust SMT solver.
