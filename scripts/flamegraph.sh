#!/bin/bash
# Usage: ./scripts/flamegraph.sh [benchmark_file.smt2]
# Profiles oxiz-cli on the given benchmark and generates a flamegraph SVG
#
# =============================================================================
# TOP 5 HOT PATHS (identified via code analysis)
# =============================================================================
#
# 1. Simplex::pivot()  [oxiz-theories/src/arithmetic/simplex.rs ~L980]
#    The core tableau pivot operation. Called O(pivots) times per solve. Each
#    call iterates every other tableau row to substitute the pivot column, making
#    this O(rows * nonbasics_per_row) per pivot.  Dominant cost: FxHashMap
#    iteration + FastRational arithmetic (mul/add of SmallVec terms).
#    Optimization target: pivot_terms SmallVec<8> already avoids heap for
#    small problems; large problems still heap-allocate.  Consider dense matrix
#    representation for 50+ variable problems.
#
# 2. Simplex::find_pivot_col()  [oxiz-theories/src/arithmetic/simplex.rs ~L762]
#    Scans all non-basic terms in a basic variable's tableau row to select the
#    entering variable according to Bland/Dantzig/SteepestEdge/PartialPricing
#    rules.  For large rows the Dantzig scan (max |coef|) touches every term.
#    Optimization target: maintain a sorted heap of eligible non-basics with
#    cached improvement scores to reduce selection to O(log n).
#
# 3. Solver::check() main CDCL(T) loop  [oxiz-solver/src/solver/mod.rs ~L330]
#    The outer CDCL(T) loop calls sat.solve_with_theory() which drives SAT
#    + theory check/propagation in lock-step.  Each iteration can trigger
#    ArithSolver checks, EUF congruence closure, and BV bit-blasting.
#    Dominant cost: theory_manager dispatch + arith simplex re-check on
#    each new SAT partial assignment.  Optimization target: lazy theory
#    evaluation (only re-check changed literals) and incremental simplex
#    via basis caching (cached_assignments already present but limited use).
#
# 4. SatSolver::propagate()  [oxiz-sat/src/solver/propagate.rs ~L14]
#    Boolean Constraint Propagation (BCP) using two-watched literals.  Already
#    has SIMD-friendly blocker pre-filter (chunks of 8 with auto-vectorization).
#    Dominant cost: clause database random access (cache misses) when blocker
#    pre-filter does not fire.  Optimization target: clause inlining for binary
#    clauses (already done via binary_graph), tertiary-clause fast path, and
#    better watch list ordering (satisfied clauses last).
#
# 5. Simplex::update_assignment()  [oxiz-theories/src/arithmetic/simplex.rs ~L1062]
#    Recomputes every basic variable's value from its tableau row after each
#    pivot.  Called once per pivot; iterates all O(rows) entries of the tableau
#    plus all non-basic variables.  Optimization target: incremental update —
#    only recompute rows that contain the entering variable (already identified
#    during pivot); store a "dirty" flag per row and evaluate lazily.
#
# =============================================================================

set -e

BENCH=${1:-bench/z3_parity/benchmarks/qf_lia/lia_01_range.smt2}
OUTPUT=/tmp/oxiz_flamegraph.svg

# Ensure we're in the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Building oxiz-cli (release)..."
cargo build --release -p oxiz-cli 2>/dev/null

echo "Profiling: $BENCH"
echo "Output:    $OUTPUT"

if command -v cargo-flamegraph >/dev/null 2>&1 || cargo flamegraph --help >/dev/null 2>&1; then
    cargo flamegraph --bin oxiz-cli -o "$OUTPUT" -- "$BENCH" 2>/dev/null || {
        echo "cargo-flamegraph run failed; falling back to perf + inferno"
        if command -v perf >/dev/null 2>&1 && command -v inferno-collapse-perf >/dev/null 2>&1 && command -v inferno-flamegraph >/dev/null 2>&1; then
            perf record -g -- ./target/release/oxiz-cli "$BENCH"
            perf script | inferno-collapse-perf | inferno-flamegraph > "$OUTPUT"
        else
            echo "ERROR: neither cargo-flamegraph nor perf+inferno tools are available." >&2
            echo "Install with: cargo install flamegraph  (and ensure perf is available)" >&2
            exit 1
        fi
    }
else
    echo "cargo-flamegraph not installed; trying perf + inferno directly"
    if command -v perf >/dev/null 2>&1 && command -v inferno-collapse-perf >/dev/null 2>&1 && command -v inferno-flamegraph >/dev/null 2>&1; then
        perf record -g -- ./target/release/oxiz-cli "$BENCH"
        perf script | inferno-collapse-perf | inferno-flamegraph > "$OUTPUT"
    else
        echo "ERROR: cargo-flamegraph is not installed." >&2
        echo "Install with: cargo install flamegraph" >&2
        echo "On Linux you also need: sudo apt install linux-perf  (or equivalent)" >&2
        echo "On macOS use DTrace: cargo flamegraph uses dtrace automatically" >&2
        exit 1
    fi
fi

echo "Flamegraph written to $OUTPUT"
