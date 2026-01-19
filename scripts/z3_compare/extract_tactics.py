#!/usr/bin/env python3
"""
Z3 Tactics Extraction Script

Parses Z3 source code to extract tactic registrations and their descriptions.
Maps tactics to OxiZ equivalents where available.

Output: JSON file with tactic names, descriptions, categories, and OxiZ mappings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Tactic:
    """Represents a Z3 tactic definition."""
    name: str
    description: str
    category: str
    file_path: str = ""
    line_number: int = 0
    dependencies: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    is_combinator: bool = False


@dataclass
class TacticCollection:
    """Collection of extracted tactics."""
    tactics: list[Tactic] = field(default_factory=list)
    combinators: list[Tactic] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)
    source_files_processed: int = 0
    z3_source_path: str = ""


# Mapping of Z3 tactics to OxiZ equivalents
OXIZ_TACTIC_MAPPING: dict[str, dict[str, str]] = {
    # SAT-related tactics
    "sat": {"oxiz_module": "oxiz-sat", "oxiz_feature": "Solver"},
    "sat-preprocess": {"oxiz_module": "oxiz-sat", "oxiz_feature": "Preprocessor"},
    "card2bv": {"oxiz_module": "oxiz-sat", "oxiz_feature": "CardinalityEncoder"},
    "bit-blast": {"oxiz_module": "oxiz-theories/bv", "oxiz_feature": "bit-blasting"},
    "pb2bv": {"oxiz_module": "oxiz-theories/pb", "oxiz_feature": "PbSolver"},
    "aig": {"oxiz_module": "oxiz-theories/bv", "oxiz_feature": "aig_builder"},

    # Arithmetic tactics
    "simplify": {"oxiz_module": "oxiz-core/rewrite", "oxiz_feature": "Rewriter"},
    "propagate-values": {"oxiz_module": "oxiz-core/rewrite", "oxiz_feature": "Rewriter"},
    "ctx-simplify": {"oxiz_module": "oxiz-core/rewrite", "oxiz_feature": "RewriteContext"},
    "normalize-bounds": {"oxiz_module": "oxiz-theories/arithmetic", "oxiz_feature": "normalize"},
    "lia2card": {"oxiz_module": "oxiz-opt", "oxiz_feature": "cardinality"},
    "diff-neq": {"oxiz_module": "oxiz-theories/diff_logic", "oxiz_feature": "DiffLogicSolver"},

    # SMT tactics
    "smt": {"oxiz_module": "oxiz-solver", "oxiz_feature": "Solver"},
    "qfnra-nlsat": {"oxiz_module": "oxiz-nlsat", "oxiz_feature": "NlsatSolver"},
    "nlsat": {"oxiz_module": "oxiz-nlsat", "oxiz_feature": "NlsatSolver"},
    "qfnia": {"oxiz_module": "oxiz-nlsat/nia", "oxiz_feature": "NiaSolver"},

    # Preprocessing tactics
    "tseitin-cnf": {"oxiz_module": "oxiz-core/ast", "oxiz_feature": "normal_forms::to_cnf"},
    "nnf": {"oxiz_module": "oxiz-core/ast", "oxiz_feature": "normal_forms::to_nnf"},
    "elim-and": {"oxiz_module": "oxiz-core/rewrite", "oxiz_feature": "BoolRewriter"},
    "elim-term-ite": {"oxiz_module": "oxiz-core/rewrite", "oxiz_feature": "ite_elimination"},
    "flatten-clauses": {"oxiz_module": "oxiz-core/tactic", "oxiz_feature": "FlattenTactic"},
    "distribute-forall": {"oxiz_module": "oxiz-theories/quantifier", "oxiz_feature": "QuantifierSolver"},

    # Optimization tactics
    "max-bv-sharing": {"oxiz_module": "oxiz-theories/bv", "oxiz_feature": "sharing"},
    "reduce-bv-size": {"oxiz_module": "oxiz-theories/bv", "oxiz_feature": "reduction"},

    # QE tactics
    "qe": {"oxiz_module": "oxiz-core/qe", "oxiz_feature": "QeLiteSolver"},
    "qe-lite": {"oxiz_module": "oxiz-core/qe", "oxiz_feature": "QeLiteSolver"},

    # Proof tactics
    "unit-subsume-simplify": {"oxiz_module": "oxiz-sat", "oxiz_feature": "SubsumptionChecker"},
    "purify-arith": {"oxiz_module": "oxiz-theories/combination", "oxiz_feature": "Purifier"},

    # Array tactics
    "solve-eqs": {"oxiz_module": "oxiz-theories/euf", "oxiz_feature": "EufSolver"},
    "ackermannize_bv": {"oxiz_module": "oxiz-theories/bv", "oxiz_feature": "ackermannization"},

    # Special tactics
    "skip": {"oxiz_module": "oxiz-core/tactic", "oxiz_feature": "SkipTactic"},
    "fail": {"oxiz_module": "oxiz-core/tactic", "oxiz_feature": "FailTactic"},
    "fail-if-undecided": {"oxiz_module": "oxiz-core/tactic", "oxiz_feature": "tactic::Goal"},

    # Model tactics
    "model_validate": {"oxiz_module": "oxiz-core/model", "oxiz_feature": "ModelEvaluator"},

    # Spacer/CHC tactics
    "horn": {"oxiz_module": "oxiz-spacer", "oxiz_feature": "SpacerSolver"},
    "pdr": {"oxiz_module": "oxiz-spacer", "oxiz_feature": "SpacerSolver"},
}


class Z3TacticExtractor:
    """Extracts tactics from Z3 source code."""

    # Regex patterns for tactic definitions
    PATTERNS: dict[str, re.Pattern[str]] = {
        # MK_TACTIC macro
        "mk_tactic": re.compile(
            r'MK_TACTIC\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
            re.MULTILINE
        ),
        # add_tactic style
        "add_tactic": re.compile(
            r'add_tactic\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"',
            re.MULTILINE | re.DOTALL
        ),
        # register_tactic style
        "register_tactic": re.compile(
            r'register_tactic\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"',
            re.MULTILINE | re.DOTALL
        ),
        # TACTIC_DESCRIPTION in tactic_cmds.cpp
        "tactic_desc": re.compile(
            r'{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*}',
            re.MULTILINE
        ),
        # tactic_factory style
        "tactic_factory": re.compile(
            r'class\s+(\w+)_tactic\s*:\s*public\s+tactic',
            re.MULTILINE
        ),
        # MK_SIMPLE_TACTIC
        "mk_simple_tactic": re.compile(
            r'MK_SIMPLE_TACTIC\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"',
            re.MULTILINE
        ),
    }

    # Tactic combinators (meta-tactics)
    COMBINATOR_NAMES: set[str] = {
        "and-then", "or-else", "par-or", "par-then", "try-for",
        "repeat", "repeat-until", "skip", "fail", "using-params",
        "with", "echo", "if", "when", "cond", "!",
    }

    TARGET_DIRS: list[str] = [
        "tactic",
        "sat/tactic",
        "smt/tactic",
        "nlsat/tactic",
        "qe/tactic",
        "ast/rewriter",
        "opt",
        "muz",
    ]

    def __init__(self, z3_source_path: str) -> None:
        self.z3_source_path = Path(z3_source_path)
        self.collection = TacticCollection(z3_source_path=z3_source_path)
        self.seen_tactics: set[str] = set()

    def extract_all(self) -> TacticCollection:
        """Extract tactics from all relevant Z3 directories."""
        if not self.z3_source_path.exists():
            self.collection.extraction_errors.append(
                f"Z3 source path does not exist: {self.z3_source_path}"
            )
            return self.collection

        # Process target directories
        for target_dir in self.TARGET_DIRS:
            dir_path = self.z3_source_path / target_dir
            if dir_path.exists():
                self._process_directory(dir_path)

        # Also search for tactic_cmds.cpp and similar files
        for pattern in ["**/tactic*.cpp", "**/tactic*.h", "**/*_tactic.cpp"]:
            for file_path in self.z3_source_path.glob(pattern):
                if file_path.is_file():
                    self._process_source_file(file_path)

        # Add known built-in combinators
        self._add_builtin_combinators()

        return self.collection

    def _process_directory(self, dir_path: Path) -> None:
        """Process all source files in a directory."""
        extensions = [".cpp", ".h", ".hpp"]
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                self._process_source_file(file_path)

    def _process_source_file(self, file_path: Path) -> None:
        """Process a single source file for tactic definitions."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            self.collection.source_files_processed += 1

            # Extract using different patterns
            self._extract_mk_tactic(content, file_path)
            self._extract_add_tactic(content, file_path)
            self._extract_register_tactic(content, file_path)
            self._extract_tactic_descriptions(content, file_path)

        except OSError as e:
            self.collection.extraction_errors.append(f"Error reading {file_path}: {e}")

    def _extract_mk_tactic(self, content: str, file_path: Path) -> None:
        """Extract tactics from MK_TACTIC macros."""
        for pattern_name in ["mk_tactic", "mk_simple_tactic"]:
            pattern = self.PATTERNS[pattern_name]
            for match in pattern.finditer(content):
                name = match.group(1)
                description = match.group(2)
                self._add_tactic(Tactic(
                    name=name,
                    description=description,
                    category=self._categorize_tactic(name),
                    file_path=str(file_path.relative_to(self.z3_source_path)),
                    line_number=content[:match.start()].count('\n') + 1,
                    is_combinator=name in self.COMBINATOR_NAMES
                ))

    def _extract_add_tactic(self, content: str, file_path: Path) -> None:
        """Extract tactics from add_tactic calls."""
        pattern = self.PATTERNS["add_tactic"]
        for match in pattern.finditer(content):
            name = match.group(1)
            description = match.group(2)
            self._add_tactic(Tactic(
                name=name,
                description=description,
                category=self._categorize_tactic(name),
                file_path=str(file_path.relative_to(self.z3_source_path)),
                line_number=content[:match.start()].count('\n') + 1,
                is_combinator=name in self.COMBINATOR_NAMES
            ))

    def _extract_register_tactic(self, content: str, file_path: Path) -> None:
        """Extract tactics from register_tactic calls."""
        pattern = self.PATTERNS["register_tactic"]
        for match in pattern.finditer(content):
            name = match.group(1)
            description = match.group(2)
            self._add_tactic(Tactic(
                name=name,
                description=description,
                category=self._categorize_tactic(name),
                file_path=str(file_path.relative_to(self.z3_source_path)),
                line_number=content[:match.start()].count('\n') + 1,
                is_combinator=name in self.COMBINATOR_NAMES
            ))

    def _extract_tactic_descriptions(self, content: str, file_path: Path) -> None:
        """Extract tactics from static description arrays."""
        # Look for arrays of tactic descriptions
        desc_pattern = re.compile(
            r'static\s+(?:const\s+)?(?:\w+\s+)*tactic_descr(?:iption)?s?\s*\[\]\s*=\s*\{([^}]+)\}',
            re.MULTILINE | re.DOTALL
        )
        for array_match in desc_pattern.finditer(content):
            array_content = array_match.group(1)
            for entry_match in self.PATTERNS["tactic_desc"].finditer(array_content):
                name = entry_match.group(1)
                description = entry_match.group(2)
                self._add_tactic(Tactic(
                    name=name,
                    description=description,
                    category=self._categorize_tactic(name),
                    file_path=str(file_path.relative_to(self.z3_source_path)),
                    line_number=content[:array_match.start()].count('\n') + 1,
                    is_combinator=name in self.COMBINATOR_NAMES
                ))

    def _add_builtin_combinators(self) -> None:
        """Add built-in tactic combinators."""
        combinators = [
            ("and-then", "Apply tactics sequentially"),
            ("or-else", "Apply first tactic, if fails apply second"),
            ("par-or", "Apply tactics in parallel, return first success"),
            ("par-then", "Apply first tactic, then apply second to all subgoals in parallel"),
            ("try-for", "Apply tactic with timeout"),
            ("repeat", "Repeat tactic until no progress"),
            ("repeat-until", "Repeat until condition met"),
            ("skip", "Do nothing (identity tactic)"),
            ("fail", "Always fail"),
            ("using-params", "Apply tactic with parameters"),
            ("with", "Apply tactic with parameters"),
            ("echo", "Print message"),
            ("if", "Conditional tactic"),
            ("when", "Apply tactic if probe succeeds"),
            ("cond", "Conditional tactic with probe"),
        ]
        for name, desc in combinators:
            if name not in self.seen_tactics:
                tactic = Tactic(
                    name=name,
                    description=desc,
                    category="combinator",
                    is_combinator=True
                )
                self.seen_tactics.add(name)
                self.collection.combinators.append(tactic)

    def _add_tactic(self, tactic: Tactic) -> None:
        """Add a tactic to the collection, avoiding duplicates."""
        if tactic.name not in self.seen_tactics:
            self.seen_tactics.add(tactic.name)
            if tactic.is_combinator:
                self.collection.combinators.append(tactic)
            else:
                self.collection.tactics.append(tactic)

    def _categorize_tactic(self, name: str) -> str:
        """Categorize a tactic based on its name."""
        name_lower = name.lower()

        if name in self.COMBINATOR_NAMES:
            return "combinator"
        if any(k in name_lower for k in ["sat", "cdcl"]):
            return "sat"
        if any(k in name_lower for k in ["smt"]):
            return "smt"
        if any(k in name_lower for k in ["nlsat", "nra", "nia"]):
            return "nonlinear"
        if any(k in name_lower for k in ["bv", "bit"]):
            return "bitvector"
        if any(k in name_lower for k in ["arith", "lia", "lra", "diff"]):
            return "arithmetic"
        if any(k in name_lower for k in ["qe", "quantif", "forall", "exists"]):
            return "quantifier"
        if any(k in name_lower for k in ["simplif", "rewrite", "propagat", "elim"]):
            return "simplification"
        if any(k in name_lower for k in ["cnf", "nnf", "tseitin"]):
            return "normalization"
        if any(k in name_lower for k in ["array"]):
            return "array"
        if any(k in name_lower for k in ["horn", "pdr", "spacer"]):
            return "chc"
        if any(k in name_lower for k in ["pb", "card"]):
            return "pseudo_boolean"
        return "general"


def get_oxiz_mapping(tactic_name: str) -> Optional[dict[str, str]]:
    """Get OxiZ mapping for a Z3 tactic."""
    return OXIZ_TACTIC_MAPPING.get(tactic_name)


def write_json_output(collection: TacticCollection, output_path: str) -> None:
    """Write the extracted tactics to a JSON file."""
    output_data = {
        "metadata": {
            "z3_source_path": collection.z3_source_path,
            "files_processed": collection.source_files_processed,
            "total_tactics": len(collection.tactics),
            "total_combinators": len(collection.combinators),
            "extraction_errors": collection.extraction_errors,
        },
        "tactics": [],
        "combinators": [asdict(t) for t in collection.combinators],
        "by_category": {},
        "oxiz_mappings": {},
    }

    # Add tactics with OxiZ mappings
    for tactic in collection.tactics:
        tactic_dict = asdict(tactic)
        mapping = get_oxiz_mapping(tactic.name)
        if mapping:
            tactic_dict["oxiz_mapping"] = mapping
        output_data["tactics"].append(tactic_dict)

    # Group by category
    all_tactics = collection.tactics + collection.combinators
    for tactic in all_tactics:
        if tactic.category not in output_data["by_category"]:
            output_data["by_category"][tactic.category] = []
        output_data["by_category"][tactic.category].append(tactic.name)

    # Build OxiZ mappings summary
    for name, mapping in OXIZ_TACTIC_MAPPING.items():
        output_data["oxiz_mappings"][name] = mapping

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Z3 tactics from source code"
    )
    parser.add_argument(
        "--z3-src",
        default="../z3/src",
        help="Path to Z3 source directory (default: ../z3/src)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="z3_tactics.json",
        help="Output JSON file path (default: z3_tactics.json)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Resolve path relative to script location
    script_dir = Path(__file__).parent
    z3_src = Path(args.z3_src)
    if not z3_src.is_absolute():
        z3_src = (script_dir / z3_src).resolve()

    if args.verbose:
        print(f"Z3 source path: {z3_src}")

    extractor = Z3TacticExtractor(str(z3_src))
    collection = extractor.extract_all()

    if args.verbose:
        print(f"Files processed: {collection.source_files_processed}")
        print(f"Tactics extracted: {len(collection.tactics)}")
        print(f"Combinators extracted: {len(collection.combinators)}")
        if collection.extraction_errors:
            print(f"Errors: {len(collection.extraction_errors)}")
            for err in collection.extraction_errors[:5]:
                print(f"  - {err}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    write_json_output(collection, str(output_path))

    if args.verbose:
        print(f"Output written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
