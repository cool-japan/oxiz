#!/bin/bash
# Build Python bindings for OxiZ
# Usage: ./scripts/build_python.sh [--release|--debug|--develop]
#
# Requires: maturin (install with: pip install maturin)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OXIZ_PY_DIR="${SCRIPT_DIR}/../oxiz-py"

cd "$OXIZ_PY_DIR"

BUILD_MODE="--release"
DEVELOP_MODE=0

for arg in "$@"; do
    case "$arg" in
        --debug)
            BUILD_MODE=""
            ;;
        --release)
            BUILD_MODE="--release"
            ;;
        --develop)
            DEVELOP_MODE=1
            ;;
    esac
done

if [ "$DEVELOP_MODE" -eq 1 ]; then
    echo "Installing OxiZ Python bindings in development mode..."
    maturin develop $BUILD_MODE
    echo "Development install complete. 'import oxiz' should now work."
else
    echo "Building OxiZ Python bindings (wheel)..."
    maturin build $BUILD_MODE

    WHEEL_PATH=$(ls "${SCRIPT_DIR}/../target/wheels/oxiz-"*.whl 2>/dev/null | tail -1)
    if [ -n "$WHEEL_PATH" ]; then
        echo "Python wheel built: $WHEEL_PATH"
        echo "Install with: pip install '$WHEEL_PATH'"
        echo "Or run this script with --develop for an editable install."
    else
        echo "Wheel built (check target/wheels/ directory)"
    fi
fi
