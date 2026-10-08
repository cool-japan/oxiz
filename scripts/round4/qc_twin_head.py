#!/usr/bin/env python3
"""Re-judge fuzz_qc's tree-unsat checks on the ground twin solved by an independent build.

    qc_twin_head.py --seed N --count K [--probe probe_tree] [--judge probe_head]
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c, qc_eval as qe, fuzz_qc as fq
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--count", type=int, required=True)
ap.add_argument("--probe", default="probe_tree")
ap.add_argument("--judge", default="probe_head")
args = ap.parse_args()
seed, count = args.seed, args.count
def one(k):
    text = fq.make(seed, k)
    r = c.run_probe(args.probe, None, 20, extra_stdin=text)
    if r["status"] != "ok":
        return k, 0, {}
    res = qe.judge(text, r["stdout"])
    if not res["unsat_checks"]:
        return k, 0, {}
    rt = c.run_probe(args.judge, None, 20, extra_stdin=qe.ground_twin(text))
    if rt["status"] != "ok":
        return k, len(res["unsat_checks"]), {"unresolved": len(res["unsat_checks"])}
    return k, len(res["unsat_checks"]), qe.judge_twin(text, rt["stdout"], res["unsat_checks"])
tot = {"tree_unsat_checks": 0, "wrong_unsat": 0, "confirmed": 0, "unresolved": 0}
bad = []
for k, n, t in c.pool_map(one, list(range(count)), 2):
    tot["tree_unsat_checks"] += n
    for key in ("wrong_unsat", "confirmed", "unresolved"):
        tot[key] += t.get(key, 0)
    if t.get("wrong_unsat"):
        bad.append(k)
print("qc_twin_head seed=%d count=%d probe=%s judge=%s cap=20s jobs=2" % (seed, count, args.probe, args.judge))
print(json.dumps(tot), "bad:", bad)
