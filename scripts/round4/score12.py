#!/usr/bin/env python3
"""score12.py - the width-1/2 paired corpora (q / qnoite) against an
exhaustive brute-force oracle.

    score12.py --probe probe_tree --corpus corpus/q [--cap 20] [--jobs 2]

The truth of each pair is decided WITHOUT a solver: common.Script.brute_force
enumerates every interpretation of every declared constant (at index widths
1-2 an array of bit-vector-1 elements has at most 16 values).  Both files of
each pair are run through the probe, and every published `sat` model is
evaluated exactly against the quantified script.

wrong_sat   : an answer `sat` where the oracle says `unsat` (either file);
wrong_unsat : an answer `unsat` where the oracle says `sat` (either file);
falsifying  : a `sat` whose published model evaluates to false;
model_na    : a `sat` whose model could not be evaluated (a value the
              evaluator cannot read) - reported, never counted as either;
unknown_q   : the quantified file answered `unknown` / TIMEOUT;
q_vs_g      : both files decided and they differ;
agree       : the quantified answer equals the oracle.
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
        qp = os.path.join(args.corpus, "q%04d.smt2" % k)
        gp = os.path.join(args.corpus, "g%04d.smt2" % k)
        with open(qp) as fh:
            script = c.Script(fh.read())
        truth, _ = script.brute_force()
        rec = {"k": k, "truth": truth}
        for tag, path in (("q", qp), ("g", gp)):
            r = c.run_probe(args.probe, path, args.cap)
            v = c.verdict_of(r)
            rec[tag] = v
            rec[tag + "_panic"] = r["status"] == "panic"
            if v == "sat" and r["models"]:
                ev = script.eval_model(c.model_values(r["models"][-1], script.decls))
                rec[tag + "_model"] = "na" if ev is None else ("ok" if ev else "false")
        return rec

    recs = c.pool_map(one, ks, args.jobs)
    tot = {"n": len(recs), "wrong_sat": 0, "wrong_unsat": 0, "falsifying": 0, "model_na": 0,
           "unknown_q": 0, "q_vs_g": 0, "panics": 0, "agree": 0, "truth_sat": 0, "truth_unsat": 0}
    bad = []
    for r in recs:
        tot["truth_" + r["truth"]] = tot.get("truth_" + r["truth"], 0) + 1
        for tag in ("q", "g"):
            if r[tag] == "sat" and r["truth"] == "unsat":
                tot["wrong_sat"] += 1
                bad.append(("wrong_sat", tag, r["k"]))
            if r[tag] == "unsat" and r["truth"] == "sat":
                tot["wrong_unsat"] += 1
                bad.append(("wrong_unsat", tag, r["k"]))
            if r.get(tag + "_model") == "false":
                tot["falsifying"] += 1
                bad.append(("falsifying", tag, r["k"]))
            if r.get(tag + "_model") == "na":
                tot["model_na"] += 1
            if r[tag + "_panic"]:
                tot["panics"] += 1
        if r["q"] not in ("sat", "unsat"):
            tot["unknown_q"] += 1
        if r["q"] in ("sat", "unsat") and r["g"] in ("sat", "unsat") and r["q"] != r["g"]:
            tot["q_vs_g"] += 1
        if r["q"] == r["truth"]:
            tot["agree"] += 1
    print("score12 probe=%s corpus=%s cap=%gs jobs=%d oracle=brute-force" % (args.probe, args.corpus, args.cap, args.jobs))
    print(json.dumps(tot))
    for kind, tag, k in bad:
        print("  %s %s%04d" % (kind, tag, k))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "totals": tot, "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
