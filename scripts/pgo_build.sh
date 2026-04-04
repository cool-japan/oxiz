#!/bin/bash
# Profile-Guided Optimization build script for OxiZ
#
# This script builds the oxiz CLI binary with PGO (Profile-Guided Optimization):
# 1. Builds an instrumented binary that collects runtime profile data
# 2. Runs representative SMT2 benchmarks to generate profile data
# 3. Merges the collected profiles
# 4. Rebuilds with PGO using the merged profile data
#
# Requirements:
#   - llvm-profdata (comes with LLVM toolchain / rustup component llvm-tools)
#   - Benchmark SMT2 files in bench/z3_parity/benchmarks/
#
# Usage: ./scripts/pgo_build.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PGO_DIR="$PROJ_DIR/target/pgo"

# Clean up previous PGO data
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR/profiles"

echo "=== Phase 1: Build with instrumentation ==="
RUSTFLAGS="-Cprofile-generate=$PGO_DIR/profiles" \
    cargo build --release --manifest-path "$PROJ_DIR/Cargo.toml" --bin oxiz

echo "=== Phase 2: Run training workload ==="
# Run representative benchmarks to generate profile data
BENCH_DIR="$PROJ_DIR/bench/z3_parity/benchmarks"
if [ -d "$BENCH_DIR" ]; then
    find "$BENCH_DIR" -name "*.smt2" -type f | while read -r smt2; do
        echo "  Running: $(basename "$smt2")"
        timeout 30 "$PROJ_DIR/target/release/oxiz" "$smt2" 2>/dev/null || true
    done
else
    echo "WARNING: No benchmark directory found at $BENCH_DIR"
    echo "  PGO optimization will be limited without training data."
fi

echo "=== Phase 3: Merge profiles ==="
# Find llvm-profdata (try rustup component first, then system)
PROFDATA=""
if command -v llvm-profdata &>/dev/null; then
    PROFDATA="llvm-profdata"
elif command -v rustup &>/dev/null; then
    # Try to find llvm-profdata via rustup
    TOOLCHAIN_DIR="$(rustup show home)/toolchains/$(rustup show active-toolchain | cut -d' ' -f1)"
    if [ -f "$TOOLCHAIN_DIR/lib/rustlib/$(rustc -vV | grep host | cut -d' ' -f2)/bin/llvm-profdata" ]; then
        PROFDATA="$TOOLCHAIN_DIR/lib/rustlib/$(rustc -vV | grep host | cut -d' ' -f2)/bin/llvm-profdata"
    fi
fi

if [ -z "$PROFDATA" ]; then
    echo "ERROR: llvm-profdata not found."
    echo "  Install it with: rustup component add llvm-tools"
    exit 1
fi

"$PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR/profiles/"

echo "=== Phase 4: Build with PGO ==="
RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" \
    cargo build --release --manifest-path "$PROJ_DIR/Cargo.toml" --bin oxiz

echo "=== PGO build complete ==="
echo "Binary: $PROJ_DIR/target/release/oxiz"
