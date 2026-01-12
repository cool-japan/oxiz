#!/usr/bin/env bash

# OxiZ WASM Build Script
# Provides various build configurations for different use cases

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    print_error "wasm-pack is not installed"
    echo "Install it with: cargo install wasm-pack"
    exit 1
fi

# Parse arguments
BUILD_TYPE="${1:-dev}"
TARGET="${2:-web}"

print_info "Building OxiZ WASM"
print_info "Build type: $BUILD_TYPE"
print_info "Target: $TARGET"

case "$BUILD_TYPE" in
    dev)
        print_info "Building development version (fast compile, larger size)..."
        wasm-pack build --target "$TARGET" --dev
        ;;

    release)
        print_info "Building release version (optimized)..."
        wasm-pack build --target "$TARGET" --release
        ;;

    profiling)
        print_info "Building profiling version (release with debug info)..."
        wasm-pack build --target "$TARGET" --profiling
        ;;

    optimized)
        print_info "Building highly optimized version (requires wasm-opt)..."

        # Check if wasm-opt is available
        if ! command -v wasm-opt &> /dev/null; then
            print_warn "wasm-opt not found, falling back to regular release build"
            print_warn "Install binaryen for maximum optimization:"
            print_warn "  macOS: brew install binaryen"
            print_warn "  Ubuntu: sudo apt-get install binaryen"
            wasm-pack build --target "$TARGET" --release
        else
            # Build release first
            wasm-pack build --target "$TARGET" --release

            print_info "Running wasm-opt for maximum size reduction..."

            # Backup original
            cp pkg/oxiz_wasm_bg.wasm pkg/oxiz_wasm_bg.wasm.backup

            # Optimize with wasm-opt
            wasm-opt -Oz --enable-bulk-memory \
                     pkg/oxiz_wasm_bg.wasm \
                     -o pkg/oxiz_wasm_bg.wasm

            # Show size comparison
            ORIG_SIZE=$(stat -f%z pkg/oxiz_wasm_bg.wasm.backup 2>/dev/null || stat -c%s pkg/oxiz_wasm_bg.wasm.backup 2>/dev/null)
            OPT_SIZE=$(stat -f%z pkg/oxiz_wasm_bg.wasm 2>/dev/null || stat -c%s pkg/oxiz_wasm_bg.wasm 2>/dev/null)
            REDUCTION=$(( (ORIG_SIZE - OPT_SIZE) * 100 / ORIG_SIZE ))

            print_info "Original size: $(numfmt --to=iec $ORIG_SIZE 2>/dev/null || echo "$ORIG_SIZE bytes")"
            print_info "Optimized size: $(numfmt --to=iec $OPT_SIZE 2>/dev/null || echo "$OPT_SIZE bytes")"
            print_info "Size reduction: ${REDUCTION}%"

            rm pkg/oxiz_wasm_bg.wasm.backup
        fi
        ;;

    all)
        print_info "Building all targets..."
        for target in web nodejs bundler; do
            print_info "Building for $target..."
            wasm-pack build --target "$target" --release --out-dir "pkg-$target"
        done
        print_info "All targets built successfully"
        ;;

    clean)
        print_info "Cleaning build artifacts..."
        rm -rf pkg pkg-* target
        print_info "Clean complete"
        exit 0
        ;;

    *)
        print_error "Unknown build type: $BUILD_TYPE"
        echo ""
        echo "Usage: ./build.sh [BUILD_TYPE] [TARGET]"
        echo ""
        echo "Build types:"
        echo "  dev         - Fast development build (default)"
        echo "  release     - Optimized release build"
        echo "  profiling   - Release with debug symbols"
        echo "  optimized   - Maximum optimization (requires wasm-opt)"
        echo "  all         - Build for all targets (web, nodejs, bundler)"
        echo "  clean       - Remove all build artifacts"
        echo ""
        echo "Targets:"
        echo "  web         - For direct browser usage (default)"
        echo "  nodejs      - For Node.js"
        echo "  bundler     - For webpack/rollup/etc"
        exit 1
        ;;
esac

# Copy TypeScript declarations to pkg
if [ -f "oxiz-wasm.d.ts" ] && [ -d "pkg" ]; then
    print_info "Copying TypeScript declarations..."
    cp oxiz-wasm.d.ts pkg/
fi

# Show final package size
if [ -d "pkg" ]; then
    WASM_SIZE=$(stat -f%z pkg/oxiz_wasm_bg.wasm 2>/dev/null || stat -c%s pkg/oxiz_wasm_bg.wasm 2>/dev/null)
    print_info "Final WASM size: $(numfmt --to=iec $WASM_SIZE 2>/dev/null || echo "$WASM_SIZE bytes")"

    # Estimate gzipped size
    if command -v gzip &> /dev/null; then
        GZIP_SIZE=$(gzip -c pkg/oxiz_wasm_bg.wasm | wc -c | tr -d ' ')
        print_info "Estimated gzipped size: $(numfmt --to=iec $GZIP_SIZE 2>/dev/null || echo "$GZIP_SIZE bytes")"
    fi
fi

print_info "Build complete!"
