#!/usr/bin/env python3
"""pairverd.py - one probe against another over a corpus, per path.

    pairverd.py --base probe_base --tree probe_tree --files 'corpus/qmbqi120/q*.smt2' \
                [--cap 20] [--jobs 2] [--json out.json]

Per script: LOST (base decided, tree did not), GAINED (tree decided, base did
not), FLIPPED (both decided, different, with its direction).  Each script is
run with `(get-info :all-statistics)` appended, so the deterministic counters
(:conflicts, :bv-embedded-checks, :bv-embedded-conflicts,
:array-refinement-rounds, :array-lemma-instances) are recorded beside the
verdict and the time.  The two probes run back to back on each script.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import common as c


def run(probe, path, cap):
    with open(path) as fh:
        text = fh.read()
    r = c.run_probe(probe, None, cap, extra_stdin=text + "\n(get-info :all-statistics)\n")
    return {"v": c.verdict_of(r), "ms": r["ms"], "counters": c.counters(r["stdout"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--files", required=True, nargs="+")
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    paths = sorted({p for pat in args.files for p in glob.glob(pat)})

    def one(path):
        b = run(args.base, path, args.cap)
        t = run(args.tree, path, args.cap)
        return {"path": path, "base": b, "tree": t}

    recs = c.pool_map(one, paths, args.jobs)
    decided = ("sat", "unsat")
    lost, gained, flipped = [], [], []
    bp, tp = {}, {}
    for r in recs:
        b, t = r["base"]["v"], r["tree"]["v"]
        bp[b] = bp.get(b, 0) + 1
        tp[t] = tp.get(t, 0) + 1
        if b in decided and t not in decided:
            lost.append(r)
        elif t in decided and b not in decided:
            gained.append(r)
        elif b in decided and t in decided and b != t:
            flipped.append(r)
    print("pairverd base=%s tree=%s n=%d cap=%gs jobs=%d" % (args.base, args.tree, len(recs), args.cap, args.jobs))
    print("base", json.dumps(dict(sorted(bp.items()))), "tree", json.dumps(dict(sorted(tp.items()))))
    print(json.dumps({"LOST": len(lost), "GAINED": len(gained), "FLIPPED": len(flipped)}))
    for name, group in (("LOST", lost), ("GAINED", gained), ("FLIPPED", flipped)):
        for r in group:
            print("  %s %s base=%s(%s ms) tree=%s(%s ms) tree-counters=%s" % (
                name, os.path.basename(r["path"]), r["base"]["v"], r["base"]["ms"], r["tree"]["v"],
                r["tree"]["ms"], json.dumps(r["tree"]["counters"], sort_keys=True)))
    dirs = {}
    for r in flipped:
        key = "%s->%s" % (r["base"]["v"], r["tree"]["v"])
        dirs[key] = dirs.get(key, 0) + 1
    print("flip directions", json.dumps(dirs))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
