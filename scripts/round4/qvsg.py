#!/usr/bin/env python3
"""qvsg.py - quantified script against its own ground twin, one probe.

    qvsg.py --probe probe_tree --corpus corpus/qmbqi120 [--cap 20] [--jobs 2]

WRONG_SAT   = quantified `sat`, ground `unsat`;
WRONG_UNSAT = quantified `unsat`, ground `sat`;
agree       = both decided and equal.
The two partitions are printed beside them; TIMEOUT / PANIC are their own
classes and never counted as a verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    ks = sorted(int(m.group(1)) for f in os.listdir(args.corpus) if (m := re.match(r"q(\d{4})\.smt2$", f)))

    def one(k):
        rq = c.run_probe(args.probe, os.path.join(args.corpus, "q%04d.smt2" % k), args.cap)
        rg = c.run_probe(args.probe, os.path.join(args.corpus, "g%04d.smt2" % k), args.cap)
        return {"k": k, "q": c.verdict_of(rq), "g": c.verdict_of(rg), "q_ms": rq["ms"], "g_ms": rg["ms"]}

    recs = c.pool_map(one, ks, args.jobs)
    tot = {
        "n": len(recs),
        "WRONG_SAT": sum(r["q"] == "sat" and r["g"] == "unsat" for r in recs),
        "WRONG_UNSAT": sum(r["q"] == "unsat" and r["g"] == "sat" for r in recs),
        "agree": sum(r["q"] in ("sat", "unsat") and r["q"] == r["g"] for r in recs),
    }
    qp, gp = {}, {}
    for r in recs:
        qp[r["q"]] = qp.get(r["q"], 0) + 1
        gp[r["g"]] = gp.get(r["g"], 0) + 1
    print("qvsg probe=%s corpus=%s cap=%gs jobs=%d" % (args.probe, args.corpus, args.cap, args.jobs))
    print(json.dumps(tot))
    print("quantified", json.dumps(dict(sorted(qp.items()))), "ground", json.dumps(dict(sorted(gp.items()))))
    for r in recs:
        if (r["q"] == "sat" and r["g"] == "unsat") or (r["q"] == "unsat" and r["g"] == "sat"):
            print("  WRONG q%04d quantified=%s ground=%s" % (r["k"], r["q"], r["g"]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "totals": tot, "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
