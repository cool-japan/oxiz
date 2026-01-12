#!/usr/bin/env bash

# OxiZ WASM NPM Publishing Script
# Automates the publishing process with pre-publish checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -f "package.json" ]; then
    print_error "Must be run from oxiz-wasm directory"
    exit 1
fi

# Check if npm is logged in
print_step "Checking NPM authentication..."
if ! npm whoami &> /dev/null; then
    print_error "Not logged in to NPM. Run 'npm login' first."
    exit 1
fi

NPM_USER=$(npm whoami)
print_info "Logged in as: $NPM_USER"

# Parse version from Cargo.toml
CARGO_VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
print_info "Current version: $CARGO_VERSION"

# Check if this version is already published
if npm view oxiz-wasm@$CARGO_VERSION version &> /dev/null; then
    print_error "Version $CARGO_VERSION is already published!"
    print_warn "Update the version in Cargo.toml first"
    exit 1
fi

# Pre-publish checks
print_step "Running pre-publish checks..."

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    print_warn "There are uncommitted changes in the repository"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests
print_step "Running tests..."
cargo test

# Check for warnings
print_step "Checking for warnings..."
if cargo build --release 2>&1 | grep -i "warning:" > /dev/null; then
    print_warn "Build contains warnings. Review them carefully."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build for all targets
print_step "Building for all targets..."
./build.sh all

# Optimize the web build
print_step "Optimizing web build..."
if command -v wasm-opt &> /dev/null; then
    wasm-opt -Oz --enable-bulk-memory pkg-web/oxiz_wasm_bg.wasm -o pkg-web/oxiz_wasm_bg.wasm
    print_info "Web build optimized"
else
    print_warn "wasm-opt not found. Skipping optimization."
    print_warn "Install binaryen for smaller bundle size."
fi

# Show final sizes
print_step "Package sizes:"
for dir in pkg-web pkg-nodejs pkg-bundler; do
    if [ -d "$dir" ]; then
        WASM_SIZE=$(stat -f%z "$dir/oxiz_wasm_bg.wasm" 2>/dev/null || stat -c%s "$dir/oxiz_wasm_bg.wasm" 2>/dev/null)
        SIZE_MB=$(echo "scale=2; $WASM_SIZE / 1024 / 1024" | bc)
        echo "  $dir: ${SIZE_MB} MB"
    fi
done

# Dry run
print_step "Running npm publish dry-run..."
npm publish --dry-run

# Confirm publication
echo ""
print_warn "About to publish oxiz-wasm@$CARGO_VERSION to NPM"
read -p "Continue with publication? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Publication cancelled"
    exit 0
fi

# Publish to NPM
print_step "Publishing to NPM..."
npm publish

# Create git tag
print_step "Creating git tag..."
git tag -a "v$CARGO_VERSION" -m "Release version $CARGO_VERSION"

print_info "Successfully published oxiz-wasm@$CARGO_VERSION"
print_info "Don't forget to push the tag: git push origin v$CARGO_VERSION"

# Show CDN URLs
echo ""
print_info "Package will be available on CDNs within a few minutes:"
echo "  unpkg: https://unpkg.com/oxiz-wasm@$CARGO_VERSION/"
echo "  jsdelivr: https://cdn.jsdelivr.net/npm/oxiz-wasm@$CARGO_VERSION/"
