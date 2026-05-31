#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(cargo metadata --no-deps --format-version 1 | python3 -c 'import sys,json; d=json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"]=="oxiz-smtcomp"))')"
OUT_DIR="$(mktemp -d)"
ZIP_NAME="oxiz-smtcomp-2026-${VERSION}.zip"

echo "[package_smtcomp] Version: ${VERSION}"
echo "[package_smtcomp] Building release binary..."
cargo build --release -p oxiz-smtcomp --bin smtcomp2026

BINARY="$ROOT/target/release/smtcomp2026"
echo "[package_smtcomp] Binary: ${BINARY}"

echo "[package_smtcomp] Generating submission package..."
# Use a small Rust helper to call generate_submission_package
# We invoke the binary itself for validation, but packaging is done via shell
mkdir -p "${OUT_DIR}/bin"
cp "${BINARY}" "${OUT_DIR}/bin/smtcomp2026"

# Generate run scripts for each track
for TRACK in default incremental unsat_core model_validation proof_exhibition; do
    SCRIPT="${OUT_DIR}/bin/starexec_run_${TRACK}"
    if [ "${TRACK}" = "default" ]; then
        # Default / single-query script has no --track flag (backward compat)
        cat > "${SCRIPT}" <<EOF
#!/bin/bash
SCRIPT_DIR="\$(dirname "\$0")"
exec "\${SCRIPT_DIR}/smtcomp2026" --smtcomp "\$@"
EOF
    else
        cat > "${SCRIPT}" <<EOF
#!/bin/bash
SCRIPT_DIR="\$(dirname "\$0")"
exec "\${SCRIPT_DIR}/smtcomp2026" --smtcomp --track ${TRACK} "\$@"
EOF
    fi
    chmod +x "${SCRIPT}"
done

# Generate description and conf
cat > "${OUT_DIR}/description.txt" <<EOF
OxiZ ${VERSION} — Pure Rust SMT Solver
Next-generation SMT solving with 100% Z3 parity across 8 core logics.
https://github.com/cool-japan/oxiz
EOF

cat > "${OUT_DIR}/starexec_conf.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<solverConfig>
    <solver name="OxiZ" version="${VERSION}"/>
    <tracks>
        <track name="Single Query"/>
        <track name="Incremental"/>
        <track name="Unsat Core"/>
        <track name="Model Validation"/>
        <track name="Proof Exhibition"/>
    </tracks>
</solverConfig>
EOF

echo "[package_smtcomp] Creating zip: ${ZIP_NAME}"
(cd "${OUT_DIR}" && zip -r "${ROOT}/target/${ZIP_NAME}" .)

echo "[package_smtcomp] Done: target/${ZIP_NAME}"
rm -rf "${OUT_DIR}"
