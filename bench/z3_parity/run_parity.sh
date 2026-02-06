#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OxiZ Z3 Parity Test Suite ===${NC}\n"

# Check if Z3 is installed
if ! command -v z3 &> /dev/null; then
    echo -e "${RED}ERROR: Z3 not found!${NC}"
    echo "Please install Z3:"
    echo "  macOS: brew install z3"
    echo "  Linux: sudo apt-get install z3"
    echo "  Or download from: https://github.com/Z3Prover/z3/releases"
    exit 1
fi

echo -e "${GREEN}✓ Z3 found:${NC} $(which z3)"
z3 --version

# Build OxiZ
echo -e "\n${YELLOW}Building OxiZ...${NC}"
cd ../..
cargo build --release --quiet

# Run parity tests
echo -e "\n${YELLOW}Running parity tests...${NC}"
cd bench/z3_parity
cargo run --release

# Check results
if [ -f "results.json" ]; then
    echo -e "\n${GREEN}✓ Results saved to results.json${NC}"
else
    echo -e "\n${RED}✗ No results file generated${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== Parity test complete ===${NC}"
