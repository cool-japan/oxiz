#!/usr/bin/env bash
# Performance regression check for OxiZ
# Usage: ./scripts/perf_check.sh [--threshold N] [--update] [--json] [--github]
#
# Options:
#   -t, --threshold N    Regression threshold percentage (default: 5)
#   -u, --update         Update baseline with current results
#   --json               Output in JSON format
#   --github             Output with GitHub Actions annotations
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Building performance regression tool..."
cargo build --release --manifest-path bench/regression/Cargo.toml 2>&1

echo "Running performance regression check (threshold: 5%)..."
# Pass --threshold 5 as the default; caller args can override via "$@"
# If caller passes their own --threshold, it will take precedence because
# the argument parser reads the last value.
cargo run --release --manifest-path bench/regression/Cargo.toml -- --threshold 5 "$@"
