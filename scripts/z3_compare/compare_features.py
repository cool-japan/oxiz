#!/usr/bin/env python3
"""
Z3 vs OxiZ Feature Comparison Script

Loads extracted Z3 data and compares against OxiZ implemented features.
Generates a comprehensive report showing coverage percentage and missing features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class FeatureCoverage:
    """Represents coverage of a feature category."""
    category: str
    z3_count: int
    oxiz_count: int
    mapped_count: int
    coverage_percent: float
    z3_features: list[str] = field(default_factory=list)
    oxiz_features: list[str] = field(default_factory=list)
    mapped_features: list[dict[str, str]] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Complete comparison report."""
    overall_tactic_coverage: float = 0.0
    overall_param_coverage: float = 0.0
    tactic_categories: list[FeatureCoverage] = field(default_factory=list)
    param_categories: list[FeatureCoverage] = field(default_factory=list)
    total_z3_tactics: int = 0
    total_z3_params: int = 0
    total_mapped_tactics: int = 0
    total_mapped_params: int = 0
    warnings: list[str] = field(default_factory=list)


# OxiZ implemented features (extracted from source analysis)
OXIZ_FEATURES: dict[str, list[str]] = {
    "sat": [
        "CDCL solver",
        "two-watched literals",
        "VSIDS branching",
        "LRB branching",
        "CHB branching",
        "VMTF branching",
        "first-UIP learning",
        "recursive minimization",
        "BCE preprocessing",
        "BVE preprocessing",
        "subsumption elimination",
        "incremental solving",
        "DRAT proof generation",
        "local search integration",
        "parallel portfolio",
        "AllSAT enumeration",
        "phase saving",
        "LBD clause management",
        "restarts (Luby, geometric)",
        "clause database reduction",
        "vivification",
        "distillation",
        "hyper-binary resolution",
        "gate detection",
        "community structure",
        "symmetry breaking",
        "cube-and-conquer",
        "XOR constraint handling",
        "cardinality encoding",
        "trail saving",
        "stabilization",
        "target phase",
        "extended resolution",
        "lookahead",
        "backbone detection",
        "ML branching",
    ],
    "smt": [
        "CDCL(T) framework",
        "Nelson-Oppen combination",
        "EUF theory (E-graphs)",
        "LRA theory (Simplex)",
        "LIA theory",
        "BV theory (bit-blasting)",
        "Array theory",
        "FP theory (IEEE 754)",
        "Datatype theory",
        "String theory",
        "Difference logic",
        "UTVPI",
        "Pseudo-Boolean",
        "Special relations",
        "User propagators",
        "Quantifier handling",
        "Recursive functions",
        "Model generation",
        "Unsat core extraction",
        "Proof production",
        "Interpolation",
    ],
    "nlsat": [
        "NLSAT algorithm",
        "CAD (Cylindrical Algebraic Decomposition)",
        "polynomial constraints",
        "variable ordering strategies",
        "interval arithmetic",
        "conflict explanation",
        "incremental CAD",
        "NIA (integer arithmetic)",
        "bound propagation",
        "cutting planes",
        "discriminant analysis",
        "monotonicity analysis",
        "root isolation",
        "chronological backtracking",
    ],
    "optimization": [
        "MaxSAT (weighted/unweighted)",
        "MaxSMT",
        "Fu-Malik algorithm",
        "OLL algorithm",
        "MSU3 algorithm",
        "WMax algorithm",
        "RC2 algorithm",
        "MaxHS algorithm",
        "IHS algorithm",
        "SortMax algorithm",
        "PMRES algorithm",
        "OMT (Optimization Modulo Theories)",
        "Pareto optimization",
        "LNS (Large Neighborhood Search)",
        "Hybrid SLS",
        "Portfolio solving",
        "Totalizer encoding",
        "Cardinality networks",
    ],
    "preprocessing": [
        "CNF conversion (Tseitin)",
        "NNF conversion",
        "simplification",
        "constant propagation",
        "unit propagation",
        "pure literal elimination",
        "subsumption",
        "self-subsumption",
        "variable elimination",
        "blocked clause elimination",
        "equivalent literal substitution",
        "ITE elimination",
        "AND/OR flattening",
    ],
    "proof": [
        "DRAT proof format",
        "LRAT proof format",
        "Alethe proof format",
        "LFSC proof format",
        "Carcara proof format",
        "proof checking",
        "proof trimming",
        "resolution proofs",
        "theory lemma proofs",
    ],
    "chc": [
        "Spacer algorithm",
        "PDR/IC3",
        "Horn clause solving",
        "parallel solving",
        "proof generation",
    ],
}

# Parameter categories in OxiZ
OXIZ_PARAM_CATEGORIES: dict[str, list[str]] = {
    "general": [
        "verbosity",
        "random_seed",
        "produce_proofs",
        "produce_models",
        "produce_unsat_cores",
        "incremental",
    ],
    "sat": [
        "restart_base",
        "restart_factor",
        "clause_decay",
        "var_decay",
        "restart_first",
        "clause_deletion",
        "max_learnt_size",
        "clause_deletion_fraction",
        "phase_saving",
        "use_vsids",
    ],
    "preprocessing": [
        "simplify.enable",
        "simplify.max_iterations",
        "simplify.subsumption",
        "simplify.variable_elimination",
        "simplify.blocked_clause_elimination",
        "simplify.equiv_literals",
    ],
    "resource_limits": [
        "time_limit",
        "decision_limit",
        "conflict_limit",
        "memory_limit",
    ],
}


class FeatureComparator:
    """Compares Z3 and OxiZ features."""

    def __init__(
        self,
        z3_params_path: Optional[str] = None,
        z3_tactics_path: Optional[str] = None,
    ) -> None:
        self.z3_params: dict[str, Any] = {}
        self.z3_tactics: dict[str, Any] = {}
        self.report = ComparisonReport()

        if z3_params_path and Path(z3_params_path).exists():
            with open(z3_params_path, encoding="utf-8") as f:
                self.z3_params = json.load(f)

        if z3_tactics_path and Path(z3_tactics_path).exists():
            with open(z3_tactics_path, encoding="utf-8") as f:
                self.z3_tactics = json.load(f)

    def compare_all(self) -> ComparisonReport:
        """Run complete comparison."""
        self._compare_tactics()
        self._compare_params()
        self._calculate_overall_coverage()
        return self.report

    def _compare_tactics(self) -> None:
        """Compare Z3 tactics against OxiZ features."""
        if not self.z3_tactics:
            self.report.warnings.append("No Z3 tactics data available")
            return

        z3_by_category = self.z3_tactics.get("by_category", {})
        oxiz_mappings = self.z3_tactics.get("oxiz_mappings", {})

        self.report.total_z3_tactics = self.z3_tactics.get("metadata", {}).get(
            "total_tactics", 0
        ) + self.z3_tactics.get("metadata", {}).get("total_combinators", 0)

        # Map Z3 categories to OxiZ feature categories
        category_mapping = {
            "sat": "sat",
            "smt": "smt",
            "nonlinear": "nlsat",
            "bitvector": "smt",
            "arithmetic": "smt",
            "quantifier": "smt",
            "simplification": "preprocessing",
            "normalization": "preprocessing",
            "array": "smt",
            "chc": "chc",
            "pseudo_boolean": "optimization",
            "combinator": "preprocessing",
            "general": "preprocessing",
        }

        for z3_cat, tactics in z3_by_category.items():
            oxiz_cat = category_mapping.get(z3_cat, "smt")
            oxiz_features = OXIZ_FEATURES.get(oxiz_cat, [])

            # Count mapped tactics
            mapped = [t for t in tactics if t in oxiz_mappings]
            missing = [t for t in tactics if t not in oxiz_mappings]

            self.report.total_mapped_tactics += len(mapped)

            coverage = FeatureCoverage(
                category=z3_cat,
                z3_count=len(tactics),
                oxiz_count=len(oxiz_features),
                mapped_count=len(mapped),
                coverage_percent=(len(mapped) / len(tactics) * 100) if tactics else 0,
                z3_features=tactics,
                oxiz_features=oxiz_features,
                mapped_features=[
                    {"z3": t, "oxiz": str(oxiz_mappings.get(t, {}))} for t in mapped
                ],
                missing_features=missing,
            )
            self.report.tactic_categories.append(coverage)

    def _compare_params(self) -> None:
        """Compare Z3 parameters against OxiZ parameters."""
        if not self.z3_params:
            self.report.warnings.append("No Z3 params data available")
            return

        z3_by_category = self.z3_params.get("by_category", {})
        self.report.total_z3_params = self.z3_params.get("metadata", {}).get(
            "total_parameters", 0
        )

        # Map Z3 param categories to OxiZ
        for z3_cat, params in z3_by_category.items():
            oxiz_cat = z3_cat if z3_cat in OXIZ_PARAM_CATEGORIES else "general"
            oxiz_params = OXIZ_PARAM_CATEGORIES.get(oxiz_cat, [])

            # Simple mapping based on name similarity
            mapped = []
            missing = []
            for p in params:
                p_lower = p.lower().replace("-", "_").replace(".", "_")
                matched = False
                for op in oxiz_params:
                    op_lower = op.lower().replace(".", "_")
                    if p_lower == op_lower or p_lower in op_lower or op_lower in p_lower:
                        mapped.append({"z3": p, "oxiz": op})
                        matched = True
                        break
                if not matched:
                    missing.append(p)

            self.report.total_mapped_params += len(mapped)

            coverage = FeatureCoverage(
                category=z3_cat,
                z3_count=len(params),
                oxiz_count=len(oxiz_params),
                mapped_count=len(mapped),
                coverage_percent=(len(mapped) / len(params) * 100) if params else 0,
                z3_features=params,
                oxiz_features=oxiz_params,
                mapped_features=mapped,
                missing_features=missing,
            )
            self.report.param_categories.append(coverage)

    def _calculate_overall_coverage(self) -> None:
        """Calculate overall coverage percentages."""
        if self.report.total_z3_tactics > 0:
            self.report.overall_tactic_coverage = (
                self.report.total_mapped_tactics / self.report.total_z3_tactics * 100
            )
        if self.report.total_z3_params > 0:
            self.report.overall_param_coverage = (
                self.report.total_mapped_params / self.report.total_z3_params * 100
            )


def generate_text_report(report: ComparisonReport) -> str:
    """Generate a human-readable text report."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("Z3 vs OxiZ Feature Comparison Report")
    lines.append("=" * 80)
    lines.append("")

    # Overall summary
    lines.append("OVERALL COVERAGE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Tactic Coverage: {report.overall_tactic_coverage:.1f}%")
    lines.append(f"  - Z3 Tactics: {report.total_z3_tactics}")
    lines.append(f"  - Mapped to OxiZ: {report.total_mapped_tactics}")
    lines.append("")
    lines.append(f"Parameter Coverage: {report.overall_param_coverage:.1f}%")
    lines.append(f"  - Z3 Parameters: {report.total_z3_params}")
    lines.append(f"  - Mapped to OxiZ: {report.total_mapped_params}")
    lines.append("")

    # Tactic coverage by category
    lines.append("TACTIC COVERAGE BY CATEGORY")
    lines.append("-" * 40)
    for cat in sorted(report.tactic_categories, key=lambda x: -x.coverage_percent):
        lines.append(
            f"  {cat.category:20s}: {cat.coverage_percent:5.1f}% "
            f"({cat.mapped_count}/{cat.z3_count} tactics)"
        )
    lines.append("")

    # Parameter coverage by category
    lines.append("PARAMETER COVERAGE BY CATEGORY")
    lines.append("-" * 40)
    for cat in sorted(report.param_categories, key=lambda x: -x.coverage_percent):
        lines.append(
            f"  {cat.category:20s}: {cat.coverage_percent:5.1f}% "
            f"({cat.mapped_count}/{cat.z3_count} params)"
        )
    lines.append("")

    # OxiZ implemented features summary
    lines.append("OXIZ IMPLEMENTED FEATURES")
    lines.append("-" * 40)
    for category, features in OXIZ_FEATURES.items():
        lines.append(f"  {category}: {len(features)} features")
    lines.append("")

    # Missing high-priority tactics
    lines.append("HIGH-PRIORITY MISSING TACTICS")
    lines.append("-" * 40)
    priority_missing: list[str] = []
    for cat in report.tactic_categories:
        if cat.category in ["sat", "smt", "arithmetic", "bitvector"]:
            priority_missing.extend(cat.missing_features[:5])
    for tactic in priority_missing[:15]:
        lines.append(f"  - {tactic}")
    lines.append("")

    # Warnings
    if report.warnings:
        lines.append("WARNINGS")
        lines.append("-" * 40)
        for warning in report.warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def generate_json_report(report: ComparisonReport) -> dict[str, Any]:
    """Generate JSON report data."""
    return {
        "summary": {
            "overall_tactic_coverage_percent": round(report.overall_tactic_coverage, 2),
            "overall_param_coverage_percent": round(report.overall_param_coverage, 2),
            "total_z3_tactics": report.total_z3_tactics,
            "total_z3_params": report.total_z3_params,
            "total_mapped_tactics": report.total_mapped_tactics,
            "total_mapped_params": report.total_mapped_params,
        },
        "tactic_coverage": [
            {
                "category": c.category,
                "z3_count": c.z3_count,
                "oxiz_count": c.oxiz_count,
                "mapped_count": c.mapped_count,
                "coverage_percent": round(c.coverage_percent, 2),
                "mapped_features": c.mapped_features,
                "missing_features": c.missing_features,
            }
            for c in report.tactic_categories
        ],
        "param_coverage": [
            {
                "category": c.category,
                "z3_count": c.z3_count,
                "oxiz_count": c.oxiz_count,
                "mapped_count": c.mapped_count,
                "coverage_percent": round(c.coverage_percent, 2),
                "mapped_features": c.mapped_features,
                "missing_features": c.missing_features,
            }
            for c in report.param_categories
        ],
        "oxiz_features": OXIZ_FEATURES,
        "oxiz_params": OXIZ_PARAM_CATEGORIES,
        "warnings": report.warnings,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Z3 features against OxiZ implementation"
    )
    parser.add_argument(
        "--params",
        default="z3_params.json",
        help="Path to Z3 params JSON file (default: z3_params.json)"
    )
    parser.add_argument(
        "--tactics",
        default="z3_tactics.json",
        help="Path to Z3 tactics JSON file (default: z3_tactics.json)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_report.json",
        help="Output JSON report path (default: comparison_report.json)"
    )
    parser.add_argument(
        "--text",
        "-t",
        action="store_true",
        help="Print text report to stdout"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write JSON output file"
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent

    params_path = Path(args.params)
    if not params_path.is_absolute():
        params_path = script_dir / params_path

    tactics_path = Path(args.tactics)
    if not tactics_path.is_absolute():
        tactics_path = script_dir / tactics_path

    # Run comparison
    comparator = FeatureComparator(
        z3_params_path=str(params_path) if params_path.exists() else None,
        z3_tactics_path=str(tactics_path) if tactics_path.exists() else None,
    )
    report = comparator.compare_all()

    # Generate text report
    if args.text or (not params_path.exists() and not tactics_path.exists()):
        print(generate_text_report(report))
        if not params_path.exists():
            print(f"Note: Z3 params file not found at {params_path}")
            print("Run extract_params.py first to generate Z3 data.")
        if not tactics_path.exists():
            print(f"Note: Z3 tactics file not found at {tactics_path}")
            print("Run extract_tactics.py first to generate Z3 data.")

    # Write JSON report
    if not args.no_json:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path

        json_report = generate_json_report(report)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        print(f"JSON report written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
