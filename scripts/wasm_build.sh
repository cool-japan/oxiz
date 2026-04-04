#!/bin/bash
# wasm_build.sh — Build OxiZ WASM and report bundle size.
#
# Usage:
#   ./scripts/wasm_build.sh              # minimal (size-optimized, default)
#   ./scripts/wasm_build.sh full         # all features
#   ./scripts/wasm_build.sh minimal      # size-optimized (explicit)
#
# Requirements:
#   - wasm-pack  (https://rustwasm.github.io/wasm-pack/)
#   - wasm-opt   (from binaryen, optional but recommended)
#
# Target: <2 MB uncompressed WASM binary with --features minimal

set -euo pipefail

PROFILE="${1:-minimal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WASM_CRATE="${REPO_ROOT}/oxiz-wasm"
PKG_DIR="${WASM_CRATE}/pkg"

echo "==> Building OxiZ WASM (profile: ${PROFILE})"
echo "    crate: ${WASM_CRATE}"

# Map profile name to Cargo features
case "${PROFILE}" in
  minimal)
    FEATURES="--no-default-features --features minimal"
    ;;
  full)
    FEATURES="--features full"
    ;;
  *)
    echo "ERROR: unknown profile '${PROFILE}'. Use 'minimal' or 'full'." >&2
    exit 1
    ;;
esac

# Build with wasm-pack
wasm-pack build "${WASM_CRATE}" --target web --release -- ${FEATURES}

echo ""
echo "==> Raw WASM bundle size:"
ls -lh "${PKG_DIR}"/*.wasm 2>/dev/null || echo "  (no .wasm files found in ${PKG_DIR})"

# Optional: run wasm-opt for additional size reduction (-Oz = optimize for size)
if command -v wasm-opt &>/dev/null; then
  echo ""
  echo "==> Running wasm-opt -Oz ..."
  for wasm_file in "${PKG_DIR}"/*.wasm; do
    opt_file="${wasm_file%.wasm}_opt.wasm"
    wasm-opt -Oz -o "${opt_file}" "${wasm_file}" 2>/dev/null || true
    if [[ -f "${opt_file}" ]]; then
      echo "    $(basename "${opt_file}"): $(du -h "${opt_file}" | cut -f1)"
    fi
  done
  echo ""
  echo "==> Final sizes (raw + opt):"
  ls -lh "${PKG_DIR}"/*.wasm 2>/dev/null
else
  echo ""
  echo "  (wasm-opt not found — install binaryen for additional ~10-15% size reduction)"
fi

echo ""
echo "==> Done."
