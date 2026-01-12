#!/usr/bin/env bash

# OxiZ WASM Version Bump Script
# Updates version in both Cargo.toml and package.json

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -f "package.json" ]; then
    print_error "Must be run from oxiz-wasm directory"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
print_info "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Determine bump type
BUMP_TYPE="${1:-patch}"

case "$BUMP_TYPE" in
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    minor)
        NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
        ;;
    patch)
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
    *)
        # Custom version provided
        if [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            NEW_VERSION="$1"
        else
            print_error "Invalid version format. Use: major, minor, patch, or X.Y.Z"
            echo "Usage: ./version-bump.sh [major|minor|patch|X.Y.Z]"
            exit 1
        fi
        ;;
esac

print_info "New version: $NEW_VERSION"

# Confirm
read -p "Bump version from $CURRENT_VERSION to $NEW_VERSION? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Version bump cancelled"
    exit 0
fi

# Update Cargo.toml
print_info "Updating Cargo.toml..."
sed -i.bak "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
rm Cargo.toml.bak

# Update package.json
print_info "Updating package.json..."
sed -i.bak "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$NEW_VERSION\"/" package.json
rm package.json.bak

# Update workspace Cargo.lock
print_info "Updating Cargo.lock..."
cargo update -p oxiz-wasm

# Create commit
print_info "Creating commit..."
git add Cargo.toml package.json ../Cargo.lock
git commit -m "Bump oxiz-wasm version to $NEW_VERSION"

print_info "Version bumped successfully!"
print_warn "Next steps:"
echo "  1. Review the changes: git show"
echo "  2. Run tests: cargo test"
echo "  3. Push changes: git push"
echo "  4. Publish: ./publish.sh"
