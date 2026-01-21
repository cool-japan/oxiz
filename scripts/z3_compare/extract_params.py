#!/usr/bin/env python3
"""
Z3 Parameter Extraction Script

Parses Z3 source code to extract parameter definitions from various modules
including smt/, sat/, opt/, nlsat/, and tactic/ directories.

Output: JSON file with parameter names, types, default values, and descriptions.
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
class Parameter:
    """Represents a Z3 parameter definition."""
    name: str
    param_type: str
    default_value: str
    description: str
    module: str
    category: str = ""
    file_path: str = ""
    line_number: int = 0


@dataclass
class ParameterCollection:
    """Collection of extracted parameters."""
    parameters: list[Parameter] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)
    source_files_processed: int = 0
    z3_source_path: str = ""


class Z3ParameterExtractor:
    """Extracts parameters from Z3 source code."""

    # Regex patterns for different parameter definition styles
    PATTERNS: dict[str, re.Pattern[str]] = {
        # DEF_PARAMS style: (name, type, default, description)
        "def_params": re.compile(
            r'DEF_PARAMS\s*\(\s*"([^"]+)"\s*,\s*(\w+)\s*,\s*([^,]+)\s*,\s*"([^"]+)"\s*\)',
            re.MULTILINE
        ),
        # params.register_* style
        "register_bool": re.compile(
            r'(?:params|p)\.register_bool_param\s*\(\s*"([^"]+)"\s*(?:,\s*([^,)]+))?\s*(?:,\s*"([^"]+)")?\s*\)',
            re.MULTILINE
        ),
        "register_uint": re.compile(
            r'(?:params|p)\.register_unsigned_param\s*\(\s*"([^"]+)"\s*(?:,\s*([^,)]+))?\s*(?:,\s*"([^"]+)")?\s*\)',
            re.MULTILINE
        ),
        "register_double": re.compile(
            r'(?:params|p)\.register_double_param\s*\(\s*"([^"]+)"\s*(?:,\s*([^,)]+))?\s*(?:,\s*"([^"]+)")?\s*\)',
            re.MULTILINE
        ),
        "register_string": re.compile(
            r'(?:params|p)\.register_string_param\s*\(\s*"([^"]+)"\s*(?:,\s*"([^"]+)")?\s*(?:,\s*"([^"]+)")?\s*\)',
            re.MULTILINE
        ),
        "register_symbol": re.compile(
            r'(?:params|p)\.register_sym_param\s*\(\s*"([^"]+)"\s*(?:,\s*([^,)]+))?\s*(?:,\s*"([^"]+)")?\s*\)',
            re.MULTILINE
        ),
        # params_ref.get_* style (indicates parameter usage)
        "get_bool": re.compile(
            r'(?:params_ref|p|m_params)\.get_bool\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
            re.MULTILINE
        ),
        "get_uint": re.compile(
            r'(?:params_ref|p|m_params)\.get_uint\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
            re.MULTILINE
        ),
        "get_double": re.compile(
            r'(?:params_ref|p|m_params)\.get_double\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
            re.MULTILINE
        ),
        "get_str": re.compile(
            r'(?:params_ref|p|m_params)\.get_str\s*\(\s*"([^"]+)"\s*,\s*"?([^")]+)"?\s*\)',
            re.MULTILINE
        ),
        # pyg file style (parameter definition files)
        "pyg_def": re.compile(
            r'^\s*\(\s*"([^"]+)"\s*,\s*(\w+)\s*,\s*([^,]+)\s*,\s*"([^"]+)"\s*\)',
            re.MULTILINE
        ),
        # REGISTER_MODULE_PARAMS style
        "module_params": re.compile(
            r'REGISTER_MODULE_PARAMS\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
            re.MULTILINE
        ),
    }

    TARGET_DIRS: list[str] = [
        "smt",
        "sat",
        "opt",
        "nlsat",
        "tactic",
        "params",
        "ast",
        "solver",
        "model",
        "muz",
        "qe",
    ]

    def __init__(self, z3_source_path: str) -> None:
        self.z3_source_path = Path(z3_source_path)
        self.collection = ParameterCollection(z3_source_path=z3_source_path)
        self.seen_params: set[str] = set()

    def extract_all(self) -> ParameterCollection:
        """Extract parameters from all relevant Z3 directories."""
        if not self.z3_source_path.exists():
            self.collection.extraction_errors.append(
                f"Z3 source path does not exist: {self.z3_source_path}"
            )
            return self.collection

        # Process .pyg parameter definition files first
        self._process_pyg_files()

        # Process source directories
        for target_dir in self.TARGET_DIRS:
            dir_path = self.z3_source_path / target_dir
            if dir_path.exists():
                self._process_directory(dir_path, target_dir)

        return self.collection

    def _process_pyg_files(self) -> None:
        """Process .pyg parameter definition files."""
        for pyg_file in self.z3_source_path.rglob("*.pyg"):
            self._process_pyg_file(pyg_file)

    def _process_pyg_file(self, file_path: Path) -> None:
        """Process a single .pyg file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            self.collection.source_files_processed += 1

            # Extract module name from file path
            module = file_path.stem.replace("_params", "")

            # Look for def_module pattern
            module_match = re.search(r'def_module_params\s*\(\s*"([^"]+)"', content)
            if module_match:
                module = module_match.group(1)

            # Find parameter definitions
            for match in re.finditer(
                r'^\s*\(\s*[\'"]([^\'"]+)[\'"]\s*,\s*(\w+)\s*,\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)',
                content,
                re.MULTILINE
            ):
                name = match.group(1)
                param_type = self._normalize_type(match.group(2))
                default_value = match.group(3).strip()
                description = match.group(4)

                self._add_parameter(Parameter(
                    name=name,
                    param_type=param_type,
                    default_value=default_value,
                    description=description,
                    module=module,
                    category=self._categorize_param(name, module),
                    file_path=str(file_path.relative_to(self.z3_source_path)),
                    line_number=content[:match.start()].count('\n') + 1
                ))

        except OSError as e:
            self.collection.extraction_errors.append(f"Error reading {file_path}: {e}")

    def _process_directory(self, dir_path: Path, module: str) -> None:
        """Process all source files in a directory."""
        extensions = [".cpp", ".h", ".hpp"]
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                self._process_source_file(file_path, module)

    def _process_source_file(self, file_path: Path, module: str) -> None:
        """Process a single source file for parameter definitions."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            self.collection.source_files_processed += 1

            # Try different extraction patterns
            self._extract_register_params(content, file_path, module)
            self._extract_get_params(content, file_path, module)
            self._extract_def_params(content, file_path, module)

        except OSError as e:
            self.collection.extraction_errors.append(f"Error reading {file_path}: {e}")

    def _extract_register_params(
        self, content: str, file_path: Path, module: str
    ) -> None:
        """Extract parameters from register_*_param calls."""
        type_patterns = [
            ("register_bool", "bool"),
            ("register_uint", "uint"),
            ("register_double", "double"),
            ("register_string", "string"),
            ("register_symbol", "symbol"),
        ]

        for pattern_name, param_type in type_patterns:
            pattern = self.PATTERNS[pattern_name]
            for match in pattern.finditer(content):
                name = match.group(1)
                default = match.group(2) if len(match.groups()) > 1 else ""
                desc = match.group(3) if len(match.groups()) > 2 else ""

                self._add_parameter(Parameter(
                    name=name,
                    param_type=param_type,
                    default_value=default.strip() if default else "",
                    description=desc.strip() if desc else "",
                    module=module,
                    category=self._categorize_param(name, module),
                    file_path=str(file_path.relative_to(self.z3_source_path)),
                    line_number=content[:match.start()].count('\n') + 1
                ))

    def _extract_get_params(
        self, content: str, file_path: Path, module: str
    ) -> None:
        """Extract parameters from get_* calls (provides default values)."""
        type_patterns = [
            ("get_bool", "bool"),
            ("get_uint", "uint"),
            ("get_double", "double"),
            ("get_str", "string"),
        ]

        for pattern_name, param_type in type_patterns:
            pattern = self.PATTERNS[pattern_name]
            for match in pattern.finditer(content):
                name = match.group(1)
                default = match.group(2) if len(match.groups()) > 1 else ""

                # Only add if we haven't seen this param before
                # get_* calls are secondary source of param info
                param_key = f"{module}:{name}"
                if param_key not in self.seen_params:
                    self._add_parameter(Parameter(
                        name=name,
                        param_type=param_type,
                        default_value=default.strip() if default else "",
                        description="",  # get_* calls don't have descriptions
                        module=module,
                        category=self._categorize_param(name, module),
                        file_path=str(file_path.relative_to(self.z3_source_path)),
                        line_number=content[:match.start()].count('\n') + 1
                    ))

    def _extract_def_params(
        self, content: str, file_path: Path, module: str
    ) -> None:
        """Extract parameters from DEF_PARAMS macros."""
        pattern = self.PATTERNS["def_params"]
        for match in pattern.finditer(content):
            name = match.group(1)
            param_type = self._normalize_type(match.group(2))
            default = match.group(3)
            desc = match.group(4)

            self._add_parameter(Parameter(
                name=name,
                param_type=param_type,
                default_value=default.strip(),
                description=desc,
                module=module,
                category=self._categorize_param(name, module),
                file_path=str(file_path.relative_to(self.z3_source_path)),
                line_number=content[:match.start()].count('\n') + 1
            ))

    def _add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the collection, avoiding duplicates."""
        key = f"{param.module}:{param.name}"
        if key not in self.seen_params:
            self.seen_params.add(key)
            self.collection.parameters.append(param)

    def _normalize_type(self, type_str: str) -> str:
        """Normalize parameter type names."""
        type_map = {
            "BOOL": "bool",
            "UINT": "uint",
            "DOUBLE": "double",
            "STRING": "string",
            "SYMBOL": "symbol",
            "bool_param": "bool",
            "uint_param": "uint",
            "double_param": "double",
            "string_param": "string",
            "symbol_param": "symbol",
        }
        return type_map.get(type_str.upper(), type_str.lower())

    def _categorize_param(self, name: str, module: str) -> str:
        """Categorize a parameter based on its name and module."""
        name_lower = name.lower()
        module_lower = module.lower()

        if any(k in name_lower for k in ["restart", "luby"]):
            return "restart"
        if any(k in name_lower for k in ["random", "seed"]):
            return "randomization"
        if any(k in name_lower for k in ["decay", "activity"]):
            return "activity"
        if any(k in name_lower for k in ["clause", "learn"]):
            return "clause_management"
        if any(k in name_lower for k in ["simplif", "preprocess", "elim"]):
            return "preprocessing"
        if any(k in name_lower for k in ["proof", "core"]):
            return "proof"
        if any(k in name_lower for k in ["timeout", "limit", "resource"]):
            return "resource_limits"
        if any(k in name_lower for k in ["verbose", "trace", "debug"]):
            return "diagnostics"
        if module_lower in ["sat", "smt", "nlsat", "opt"]:
            return module_lower
        return "general"


def write_json_output(collection: ParameterCollection, output_path: str) -> None:
    """Write the extracted parameters to a JSON file."""
    output_data = {
        "metadata": {
            "z3_source_path": collection.z3_source_path,
            "files_processed": collection.source_files_processed,
            "total_parameters": len(collection.parameters),
            "extraction_errors": collection.extraction_errors,
        },
        "parameters": [asdict(p) for p in collection.parameters],
        "by_module": {},
        "by_category": {},
    }

    # Group by module
    for param in collection.parameters:
        if param.module not in output_data["by_module"]:
            output_data["by_module"][param.module] = []
        output_data["by_module"][param.module].append(param.name)

    # Group by category
    for param in collection.parameters:
        if param.category not in output_data["by_category"]:
            output_data["by_category"][param.category] = []
        output_data["by_category"][param.category].append(param.name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Z3 parameters from source code"
    )
    parser.add_argument(
        "--z3-src",
        default="../z3/src",
        help="Path to Z3 source directory (default: ../z3/src)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="z3_params.json",
        help="Output JSON file path (default: z3_params.json)"
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

    extractor = Z3ParameterExtractor(str(z3_src))
    collection = extractor.extract_all()

    if args.verbose:
        print(f"Files processed: {collection.source_files_processed}")
        print(f"Parameters extracted: {len(collection.parameters)}")
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

    if not collection.parameters:
        print("Warning: No parameters extracted. Check Z3 source path.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
