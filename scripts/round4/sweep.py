#!/usr/bin/env python3
"""sweep.py - the response-for-response sweep (decision (42)(f)).

    sweep.py --a probe_head --b probe_tree [--root <repo>]
             [--get-model] [--cap 130] [--jobs 2] [--json out.json]

THE COMPOSITION.  The "217-benchmark sweep" of TODO.md / CHANGELOG.md is the
`bench/` corpus: CHANGELOG.md calls it "the 217-script `bench/` corpus", and
`bench/**/*.smt2` is exactly 217 files (170 z3_parity + 43 extended_theories
+ 4 regression).  The count is printed on every run and the run refuses to
report a figure if it is not 217, so a later bench/ edit cannot silently
change what "the sweep" means.

For every script the two probes run back to back (under `timeout cap`); the
comparison is the verdict sequence AND the full response text.  --get-model
appends `(get-model)` after every `(check-sat)` (the corpus-wide model
check), which is the only way a model change can show on bench/, none of
whose quantified-array scripts asks for a model.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import common as c


def composition(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "bench", "**", "*.smt2"), recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--root", default=c.REPO_ROOT)
    ap.add_argument("--get-model", action="store_true")
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    paths = composition(args.root)
    if len(paths) != 217:
        raise SystemExit("sweep composition is %d scripts, not 217 - refusing to report" % len(paths))

    def one(path):
        with open(path) as fh:
            text = fh.read()
        if args.get_model:
            text = text.replace("(check-sat)", "(check-sat)\n(get-model)")
        ra = c.run_probe(args.a, None, args.cap, extra_stdin=text)
        rb = c.run_probe(args.b, None, args.cap, extra_stdin=text)
        return {"path": os.path.relpath(path, args.root), "a": ra["stdout"], "b": rb["stdout"],
                "va": ra["verdicts"] if ra["status"] == "ok" else [ra["status"]],
                "vb": rb["verdicts"] if rb["status"] == "ok" else [rb["status"]],
                "ms_a": ra["ms"], "ms_b": rb["ms"], "sa": ra["status"], "sb": rb["status"]}

    recs = c.pool_map(one, paths, args.jobs)
    vdiff = [r for r in recs if r["va"] != r["vb"]]
    rdiff = [r for r in recs if r["a"] != r["b"]]
    ms_a = sum(r["ms_a"] or 0 for r in recs)
    ms_b = sum(r["ms_b"] or 0 for r in recs)
    print("sweep a=%s b=%s n=%d get_model=%s cap=%gs jobs=%d" % (
        args.a, args.b, len(recs), args.get_model, args.cap, args.jobs))
    print(json.dumps({"n": len(recs), "verdict_differences": len(vdiff), "response_differences": len(rdiff),
                      "ms_a": round(ms_a, 1), "ms_b": round(ms_b, 1)}))
    for r in vdiff:
        print("  VERDICT %s a=%s b=%s" % (r["path"], r["va"], r["vb"]))
    for r in rdiff:
        if r in vdiff:
            continue
        print("  RESPONSE %s" % r["path"])
        print("    --- a ---\n" + "\n".join("    " + l for l in r["a"].splitlines()))
        print("    --- b ---\n" + "\n".join("    " + l for l in r["b"].splitlines()))
    slow = sorted(recs, key=lambda r: -(r["ms_b"] or 0))[:8]
    print("slowest on b:", ", ".join("%s %.1f ms (a %.1f)" % (os.path.basename(r["path"]), r["ms_b"] or -1, r["ms_a"] or -1) for r in slow))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
