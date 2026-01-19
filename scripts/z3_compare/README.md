# Z3 Feature Comparison Scripts

This directory contains Python scripts for extracting Z3 parameters and tactics,
and comparing them against OxiZ's implemented features.

## Requirements

- Python 3.9+
- Z3 source code (available at https://github.com/Z3Prover/z3)

## Scripts Overview

### 1. extract_params.py

Extracts parameter definitions from Z3 source code.

**Features:**
- Parses `.pyg` parameter definition files
- Extracts from `params.register_*` calls
- Extracts from `params_ref.get_*` calls
- Handles parameters from smt/, sat/, opt/, nlsat/, tactic/ directories

**Usage:**
```bash
./extract_params.py --z3-src /path/to/z3/src --output z3_params.json
```

**Options:**
- `--z3-src`: Path to Z3 source directory (default: `../z3/src`)
- `--output, -o`: Output JSON file path (default: `z3_params.json`)
- `--verbose, -v`: Enable verbose output

### 2. extract_tactics.py

Extracts tactic registrations from Z3 source code.

**Features:**
- Parses `MK_TACTIC` and `MK_SIMPLE_TACTIC` macros
- Extracts from `add_tactic` and `register_tactic` calls
- Identifies tactic combinators (and-then, or-else, etc.)
- Maps tactics to OxiZ equivalents where available

**Usage:**
```bash
./extract_tactics.py --z3-src /path/to/z3/src --output z3_tactics.json
```

**Options:**
- `--z3-src`: Path to Z3 source directory (default: `../z3/src`)
- `--output, -o`: Output JSON file path (default: `z3_tactics.json`)
- `--verbose, -v`: Enable verbose output

### 3. compare_features.py

Compares extracted Z3 data against OxiZ implemented features.

**Features:**
- Loads extracted Z3 JSON files
- Compares against OxiZ feature inventory
- Calculates coverage percentages by category
- Identifies missing features
- Generates text and JSON reports

**Usage:**
```bash
# First, extract Z3 data
./extract_params.py --z3-src /path/to/z3/src
./extract_tactics.py --z3-src /path/to/z3/src

# Then run comparison
./compare_features.py --text
```

**Options:**
- `--params`: Path to Z3 params JSON file (default: `z3_params.json`)
- `--tactics`: Path to Z3 tactics JSON file (default: `z3_tactics.json`)
- `--output, -o`: Output JSON report path (default: `comparison_report.json`)
- `--text, -t`: Print text report to stdout
- `--no-json`: Do not write JSON output file

## Expected Z3 Source Location

The scripts expect Z3 source at `../z3/src` relative to the script directory,
which means:

```
/media/kitasan/Backup/
  z3/
    src/
      smt/
      sat/
      opt/
      nlsat/
      tactic/
      ...
  oxiz/
    scripts/
      z3_compare/
        extract_params.py
        extract_tactics.py
        compare_features.py
```

You can override this with the `--z3-src` option.

## Output Format

### z3_params.json

```json
{
  "metadata": {
    "z3_source_path": "/path/to/z3/src",
    "files_processed": 150,
    "total_parameters": 200,
    "extraction_errors": []
  },
  "parameters": [
    {
      "name": "restart.max",
      "param_type": "uint",
      "default_value": "UINT_MAX",
      "description": "Maximum number of restarts",
      "module": "sat",
      "category": "restart",
      "file_path": "sat/sat_params.pyg",
      "line_number": 42
    }
  ],
  "by_module": {
    "sat": ["restart.max", "random_seed", ...],
    "smt": [...]
  },
  "by_category": {
    "restart": [...],
    "randomization": [...]
  }
}
```

### z3_tactics.json

```json
{
  "metadata": {
    "z3_source_path": "/path/to/z3/src",
    "files_processed": 80,
    "total_tactics": 50,
    "total_combinators": 15,
    "extraction_errors": []
  },
  "tactics": [
    {
      "name": "sat",
      "description": "SAT solver",
      "category": "sat",
      "file_path": "tactic/sat_tactic.cpp",
      "line_number": 100,
      "oxiz_mapping": {
        "oxiz_module": "oxiz-sat",
        "oxiz_feature": "Solver"
      }
    }
  ],
  "combinators": [...],
  "by_category": {...},
  "oxiz_mappings": {...}
}
```

### comparison_report.json

```json
{
  "summary": {
    "overall_tactic_coverage_percent": 45.5,
    "overall_param_coverage_percent": 30.0,
    "total_z3_tactics": 65,
    "total_z3_params": 200,
    "total_mapped_tactics": 30,
    "total_mapped_params": 60
  },
  "tactic_coverage": [
    {
      "category": "sat",
      "z3_count": 10,
      "oxiz_count": 35,
      "mapped_count": 8,
      "coverage_percent": 80.0,
      "mapped_features": [...],
      "missing_features": [...]
    }
  ],
  "param_coverage": [...],
  "oxiz_features": {...},
  "oxiz_params": {...},
  "warnings": []
}
```

## Quick Start

```bash
cd /media/kitasan/Backup/oxiz/scripts/z3_compare

# Clone Z3 if you don't have it
git clone https://github.com/Z3Prover/z3.git ../../z3

# Make scripts executable
chmod +x *.py

# Extract Z3 features
./extract_params.py --z3-src ../../z3/src -v
./extract_tactics.py --z3-src ../../z3/src -v

# Generate comparison report
./compare_features.py --text

# View JSON output
cat comparison_report.json | python3 -m json.tool
```

## OxiZ Feature Categories

The comparison tracks these OxiZ feature areas:

- **sat**: SAT solver features (CDCL, preprocessing, proof generation)
- **smt**: SMT solver features (theories, combination, quantifiers)
- **nlsat**: Non-linear arithmetic (CAD, NLSAT algorithm)
- **optimization**: MaxSAT, MaxSMT, OMT, Pareto
- **preprocessing**: Simplification, normalization, rewriting
- **proof**: Proof formats and checking
- **chc**: Constrained Horn Clauses (Spacer/PDR)

## Updating OxiZ Feature Inventory

To update the OxiZ feature inventory in `compare_features.py`, edit the
`OXIZ_FEATURES` and `OXIZ_PARAM_CATEGORIES` dictionaries to reflect newly
implemented features.

## Troubleshooting

**No parameters/tactics extracted:**
- Verify Z3 source path is correct
- Check that the Z3 source contains .pyg files and source code
- Use `--verbose` flag for detailed output

**Low coverage percentages:**
- Some Z3 features may not have direct OxiZ equivalents
- Update `OXIZ_TACTIC_MAPPING` in extract_tactics.py for new mappings
- Update `OXIZ_FEATURES` in compare_features.py for new OxiZ features
