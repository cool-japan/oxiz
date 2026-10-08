#!/usr/bin/env python3
"""ladder.py - one script at a ladder of deterministic embedded-check budgets.

    ladder.py --probe probe_head --script corpus/named/q33_a01.body \
              --budgets 500,2000,5000,20000,50000 [--cap 130]

The script's `(set-logic ...)` line is kept first, `(set-option
:max-bv-embedded-checks N)` is inserted after it, and `(get-info
:all-statistics)` is appended, exactly as the in-tree pins build their
budgeted scripts.  Prints verdict, ms and the deterministic counters per rung;
the counters, not the clock, are what a rung is judged on.  A rung is run
under `timeout cap` (130 s; 900 only when a single named rung asks for it).
"""
from __future__ import annotations

import argparse
import json

import common as c


def budgeted(text: str, budget: int) -> str:
    lines = text.splitlines()
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("(set-logic"):
            out.append("(set-option :max-bv-embedded-checks %d)" % budget)
            inserted = True
    if not inserted:
        out.insert(0, "(set-option :max-bv-embedded-checks %d)" % budget)
    return "\n".join(out) + "\n(get-info :all-statistics)\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--budgets", default="500,2000,5000,20000,50000")
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--unbudgeted", action="store_true")
    args = ap.parse_args()
    with open(args.script) as fh:
        text = fh.read()
    rungs = [int(b) for b in args.budgets.split(",") if b]
    print("ladder probe=%s script=%s cap=%gs" % (args.probe, args.script, args.cap))
    for b in rungs + ([None] if args.unbudgeted else []):
        script = budgeted(text, b) if b is not None else text + "\n(get-info :all-statistics)\n"
        r = c.run_probe(args.probe, None, args.cap, extra_stdin=script)
        cnt = c.counters(r["stdout"])
        print(json.dumps({"budget": b, "verdict": c.verdict_of(r), "ms": r["ms"], "wall_s": round(r["wall"], 2),
                          "rounds": cnt.get(":array-refinement-rounds"),
                          "instances": cnt.get(":array-lemma-instances"),
                          "bv_checks": cnt.get(":bv-embedded-checks"),
                          "bv_conflicts": cnt.get(":bv-embedded-conflicts"),
                          "conflicts": cnt.get(":conflicts")}), flush=True)


if __name__ == "__main__":
    main()
