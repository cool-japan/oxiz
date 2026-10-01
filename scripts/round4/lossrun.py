#!/usr/bin/env python3
"""lossrun.py - named base-decided losses of a corpus, run uncapped to a cap of
130 s on two probes, with the verdict, `:reason-unknown` and the deterministic
counters of every run (recheck 13's instrument, made path-independent).

    lossrun.py --corpus <dir> NAME [NAME ...] [--probes probe_tree probe_head]
               [--cap 130] [--jobs 2]

NAME is a script's stem inside --corpus (`g0013`).  `(get-model)` is dropped
and `(get-info :reason-unknown)` / `(get-info :all-statistics)` appended.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402

COUNTERS = [":conflicts", ":array-refinement-rounds", ":array-lemma-instances",
            ":bv-embedded-checks", ":bv-embedded-conflicts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--probes", nargs="+", default=["probe_tree", "probe_head"])
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    def one(name):
        with open(os.path.join(args.corpus, name + ".smt2")) as fh:
            text = fh.read().replace("(get-model)", "")
        text += "\n(get-info :reason-unknown)\n(get-info :all-statistics)\n"
        out = {}
        for probe in args.probes:
            t0 = time.time()
            r = subprocess.run(["timeout", str(args.cap), c.probe_path(probe)], input=text,
                               capture_output=True, text=True)
            so = r.stdout
            verdicts = [l for l in so.splitlines() if l in ("sat", "unsat", "unknown")]
            reason = [l for l in so.splitlines() if "reason-unknown" in l]
            counters = {}
            for key in COUNTERS:
                i = so.find(key + " ")
                if i >= 0:
                    counters[key] = int(so[i + len(key) + 1:].split()[0].rstrip(")"))
            verdict = verdicts[0] if verdicts else ("TIMEOUT" if r.returncode == 124 else "rc%d" % r.returncode)
            out[probe] = {"v": verdict, "reason": reason, "wall_s": round(time.time() - t0, 1), **counters}
        return name, out

    print("lossrun corpus=%s probes=%s cap=%gs jobs=%d" % (args.corpus, args.probes, args.cap, args.jobs))
    for name, out in c.pool_map(one, args.names, args.jobs):
        print(name, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
