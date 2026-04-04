#!/bin/bash
# Compares OxiZ vs Z3 on z3_parity benchmarks
# Usage: ./scripts/perf_vs_z3.sh [benchmark_dir]
#   benchmark_dir defaults to bench/z3_parity/benchmarks
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BENCH_DIR="${1:-bench/z3_parity/benchmarks}"
RESULTS="/tmp/oxiz_vs_z3_$(date +%Y%m%d_%H%M%S).txt"
TIMEOUT_SEC=10

# Check if oxiz-cli binary exists; build if needed
OXIZ_BIN="$REPO_ROOT/target/release/oxiz-cli"
if [ ! -f "$OXIZ_BIN" ]; then
    echo "Building oxiz-cli (release)..."
    cargo build --release -p oxiz-cli 2>/dev/null
fi

# Check z3 availability
HAS_Z3=false
if command -v z3 >/dev/null 2>&1; then
    HAS_Z3=true
    Z3_VERSION=$(z3 --version 2>/dev/null | head -1 || echo "unknown")
fi

# Header
{
    echo "OxiZ vs Z3 Performance Comparison"
    echo "Date: $(date)"
    echo "OxiZ binary: $OXIZ_BIN"
    if $HAS_Z3; then
        echo "Z3: $Z3_VERSION"
    else
        echo "Z3: not installed — OxiZ-only timings"
    fi
    echo "Timeout: ${TIMEOUT_SEC}s per benchmark"
    echo "Benchmark dir: $BENCH_DIR"
    echo "================================="
    echo ""
} > "$RESULTS"

# Column header
if $HAS_Z3; then
    printf "%-60s %10s %10s %10s %10s %10s\n" \
        "Benchmark" "OxiZ_ms" "Z3_ms" "Ratio" "OxiZ_res" "Z3_res" >> "$RESULTS"
    printf "%-60s %10s %10s %10s %10s %10s\n" \
        "----------" "-------" "-----" "-----" "--------" "------" >> "$RESULTS"
else
    printf "%-60s %10s %10s\n" "Benchmark" "OxiZ_ms" "OxiZ_res" >> "$RESULTS"
    printf "%-60s %10s %10s\n" "----------" "-------" "--------" >> "$RESULTS"
fi

# Also write header to stdout
cat "$RESULTS"

# Counters
total=0
oxiz_faster=0
z3_faster=0
both_timeout=0

# Helper: run a command with timeout and return elapsed ms + result word
# Usage: timed_run <timeout_sec> <result_var_ms> <result_var_res> -- cmd args...
run_timed() {
    local timeout_s="$1"
    local out_ms_var="$2"
    local out_res_var="$3"
    shift 3
    # skip '--' separator if present
    if [ "$1" = "--" ]; then shift; fi

    local start_ns end_ns elapsed_ms exit_code raw_output result_word

    start_ns=$(date +%s%N 2>/dev/null || echo 0)
    raw_output=$(timeout "$timeout_s" "$@" 2>&1) || exit_code=$?
    end_ns=$(date +%s%N 2>/dev/null || echo 0)

    # Compute elapsed in ms (fallback to 0 if date +%N unavailable)
    if [ "$start_ns" -ne 0 ] && [ "$end_ns" -ne 0 ]; then
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    else
        elapsed_ms=0
    fi

    # Determine result from output
    if echo "$raw_output" | grep -qi "^unsat"; then
        result_word="unsat"
    elif echo "$raw_output" | grep -qi "^sat"; then
        result_word="sat"
    elif [ "${exit_code:-0}" -eq 124 ] || [ "${exit_code:-0}" -eq 137 ]; then
        result_word="timeout"
        elapsed_ms="${timeout_s}000"
    else
        result_word="unknown"
    fi

    printf -v "$out_ms_var" '%d' "$elapsed_ms"
    printf -v "$out_res_var" '%s' "$result_word"
}

# Collect all .smt2 files
mapfile -t SMT_FILES < <(find "$BENCH_DIR" -name "*.smt2" | sort)

if [ "${#SMT_FILES[@]}" -eq 0 ]; then
    echo "No .smt2 files found in $BENCH_DIR" | tee -a "$RESULTS"
    exit 1
fi

echo "Found ${#SMT_FILES[@]} benchmark files." | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

for smt_file in "${SMT_FILES[@]}"; do
    rel_path="${smt_file#$REPO_ROOT/}"
    # Truncate to 60 chars for display
    short_name="${rel_path: -60}"
    if [ "${#rel_path}" -gt 60 ]; then
        short_name="...${rel_path: -57}"
    fi

    oxiz_ms=0
    oxiz_res="skip"
    z3_ms=0
    z3_res="n/a"

    run_timed "$TIMEOUT_SEC" oxiz_ms oxiz_res -- "$OXIZ_BIN" "$smt_file"

    if $HAS_Z3; then
        run_timed "$TIMEOUT_SEC" z3_ms z3_res -- z3 "$smt_file"

        # Compute ratio (oxiz / z3), avoid division by zero
        if [ "$z3_ms" -gt 0 ]; then
            ratio=$(awk "BEGIN { printf \"%.2f\", $oxiz_ms / $z3_ms }")
        else
            ratio="N/A"
        fi

        line=$(printf "%-60s %10d %10d %10s %10s %10s\n" \
            "$short_name" "$oxiz_ms" "$z3_ms" "$ratio" "$oxiz_res" "$z3_res")

        # Update counters
        if [ "$oxiz_res" = "timeout" ] && [ "$z3_res" = "timeout" ]; then
            both_timeout=$(( both_timeout + 1 ))
        elif [ "$oxiz_res" != "timeout" ] && [ "$z3_res" != "timeout" ] && [ "$z3_ms" -gt 0 ]; then
            if [ "$oxiz_ms" -lt "$z3_ms" ]; then
                oxiz_faster=$(( oxiz_faster + 1 ))
            else
                z3_faster=$(( z3_faster + 1 ))
            fi
        fi
    else
        line=$(printf "%-60s %10d %10s\n" "$short_name" "$oxiz_ms" "$oxiz_res")
    fi

    echo "$line" | tee -a "$RESULTS"
    total=$(( total + 1 ))
done

# Summary footer
{
    echo ""
    echo "================================="
    echo "Summary: $total benchmarks"
    if $HAS_Z3; then
        echo "  OxiZ faster:    $oxiz_faster"
        echo "  Z3 faster:      $z3_faster"
        echo "  Both timed out: $both_timeout"
    fi
    echo "Full results: $RESULTS"
} | tee -a "$RESULTS"
