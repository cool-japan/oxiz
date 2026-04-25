#!/bin/bash
# Usage: ./scripts/flamegraph.sh [--category NAME]
# Profiles the bench-profile criterion harness and generates a flamegraph SVG.

set -euo pipefail

CATEGORY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)
            CATEGORY="${2:-}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

OUTPUT=/tmp/oxiz_profile_flamegraph.svg
COLLAPSED=/tmp/oxiz_profile_perf.folded

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FILTER_ARGS=()
if [[ -n "$CATEGORY" ]]; then
    FILTER_ARGS+=("$CATEGORY")
fi

echo "Building bench-profile (release)..."
cargo build --release -p bench-profile >/dev/null

echo "Profiling bench-profile"
if [[ -n "$CATEGORY" ]]; then
    echo "Category: $CATEGORY"
else
    echo "Category: all"
fi
echo "Output:   $OUTPUT"

if command -v cargo-flamegraph >/dev/null 2>&1 || cargo flamegraph --help >/dev/null 2>&1; then
    cargo flamegraph --bench profile_benchmarks -p bench-profile -o "$OUTPUT" -- "${FILTER_ARGS[@]}" >/dev/null 2>&1 || {
        echo "cargo-flamegraph failed; falling back to perf + inferno"
        if command -v perf >/dev/null 2>&1 && command -v inferno-collapse-perf >/dev/null 2>&1 && command -v inferno-flamegraph >/dev/null 2>&1; then
            perf record -g -- \
                cargo bench -p bench-profile --bench profile_benchmarks -- "${FILTER_ARGS[@]}" >/dev/null 2>&1
            perf script | inferno-collapse-perf > "$COLLAPSED"
            inferno-flamegraph < "$COLLAPSED" > "$OUTPUT"
        else
            echo "ERROR: neither cargo-flamegraph nor perf+inferno tools are available." >&2
            exit 1
        fi
    }
else
    if command -v perf >/dev/null 2>&1 && command -v inferno-collapse-perf >/dev/null 2>&1 && command -v inferno-flamegraph >/dev/null 2>&1; then
        perf record -g -- \
            cargo bench -p bench-profile --bench profile_benchmarks -- "${FILTER_ARGS[@]}" >/dev/null 2>&1
        perf script | inferno-collapse-perf > "$COLLAPSED"
        inferno-flamegraph < "$COLLAPSED" > "$OUTPUT"
    else
        echo "ERROR: cargo-flamegraph is not installed and perf+inferno is unavailable." >&2
        exit 1
    fi
fi

if [[ ! -f "$COLLAPSED" ]] && command -v perf >/dev/null 2>&1 && command -v inferno-collapse-perf >/dev/null 2>&1; then
    perf script | inferno-collapse-perf > "$COLLAPSED"
fi

if [[ -f "$COLLAPSED" ]]; then
    echo "Top 20 frames:"
    awk '{
        samples=$NF;
        $NF="";
        sub(/[ \t]+$/, "", $0);
        print samples "\t" $0;
    }' "$COLLAPSED" | sort -nr | head -20
else
    echo "Top 20 frames unavailable: collapsed perf data was not produced."
fi

echo "Flamegraph written to $OUTPUT"
