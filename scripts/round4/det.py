#!/usr/bin/env python3
"""det.py - determinism: two release runs of ONE probe over the same
scripts, compared response for response (verdicts and full text).

    det.py --probe probe_tree [--sweep] [--files 'corpus/qmbqi120/*.smt2' ...]
           [--cap 130] [--jobs 2]

--sweep adds the 217-script sweep composition (sweep.composition), run both
plain and with `(get-model)` after every `(check-sat)`.  Acceptance: 0
differences.

A script that TIMES OUT in both runs carries no determinism evidence, so it is
reported separately (`timeout_both`, named) instead of counting as identical,
and re-run twice with `--uncapped` seconds (default 900) as the cap; the
uncapped pair is compared like any other (decision (61)(8)).  A pair where one
run answered and the other hit the cap is a CAP-EDGE pair: named, kept out of
`differences`, and re-run uncapped the same way (re-fix pass 16).
"""
from __future__ import annotations

import argparse
import glob
import json

import common as c
import sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--root", default=c.REPO_ROOT)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--uncapped", type=float, default=900.0)
    args = ap.parse_args()
    items = []
    if args.sweep:
        for p in sweep.composition(args.root):
            with open(p) as fh:
                text = fh.read()
            items.append((p, text))
            items.append((p + "+get-model", text.replace("(check-sat)", "(check-sat)\n(get-model)")))
    for pat in args.files:
        for p in sorted(glob.glob(pat)):
            with open(p) as fh:
                items.append((p, fh.read()))

    def pair(item, cap):
        name, text = item
        r1 = c.run_probe(args.probe, None, cap, extra_stdin=text)
        r2 = c.run_probe(args.probe, None, cap, extra_stdin=text)
        return {"name": name, "item": item,
                "same": (r1["status"], r1["stdout"]) == (r2["status"], r2["stdout"]),
                "v1": c.verdict_of(r1), "v2": c.verdict_of(r2)}

    recs = c.pool_map(lambda item: pair(item, args.cap), items, args.jobs)
    timeout_both = [r for r in recs if r["v1"] == "TIMEOUT" and r["v2"] == "TIMEOUT"]
    # One run answered and the other hit the cap: a cap-edge pair, not a
    # different verdict (re-fix pass 16, recheck 15's minor 14) -- named and
    # re-run uncapped with the timeout-both scripts.
    cap_edge = [r for r in recs if not r["same"] and (r["v1"] == "TIMEOUT") != (r["v2"] == "TIMEOUT")]
    diffs = [r for r in recs if not r["same"] and r not in cap_edge]
    part = {}
    for r in recs:
        part[r["v1"]] = part.get(r["v1"], 0) + 1
    print("det probe=%s n=%d cap=%gs jobs=%d" % (args.probe, len(recs), args.cap, args.jobs))
    print(json.dumps({"n": len(recs), "differences": len(diffs), "timeout_both": len(timeout_both),
                      "cap_edge": len(cap_edge), "partition": dict(sorted(part.items()))}))
    for r in diffs:
        print("  DIFF %s %s / %s" % (r["name"], r["v1"], r["v2"]))
    for r in timeout_both:
        print("  TIMEOUT-BOTH %s (no determinism evidence at the cap)" % r["name"])
    for r in cap_edge:
        print("  CAP-EDGE %s %s / %s (one run at the cap)" % (r["name"], r["v1"], r["v2"]))
    rerun = timeout_both + cap_edge
    if rerun:
        reruns = c.pool_map(lambda r: pair(r["item"], args.uncapped), rerun, args.jobs)
        redge = [r for r in reruns if not r["same"] and (r["v1"] == "TIMEOUT") != (r["v2"] == "TIMEOUT")]
        rdiffs = [r for r in reruns if not r["same"] and r not in redge]
        print(json.dumps({"uncapped_reruns": len(reruns), "uncapped_cap_s": args.uncapped,
                          "uncapped_differences": len(rdiffs), "uncapped_cap_edge": len(redge)}))
        for r in reruns:
            tag = "same" if r["same"] else ("CAP-EDGE" if r in redge else "DIFF")
            print("  UNCAPPED %s %s / %s %s" % (r["name"], r["v1"], r["v2"], tag))


if __name__ == "__main__":
    main()
