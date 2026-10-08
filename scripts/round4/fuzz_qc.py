#!/usr/bin/env python3
"""fuzz_qc.py - recheck13's quantified-completion fuzzer (decisions (40), (41), #P2b-63).

    fuzz_qc.py --probe probe_tree --seed N --count 2000 [--jobs 2] [--cap 20]
               [--twin-probe probe_tree] [--keep-dir DIR]

Each script: index sort (_ BitVec 7) (128 points, one above finite_expand's budget),
element sort (_ BitVec 1|2) or Bool, 1-5 arrays, 0-2 index constants, optionally an
uninterpreted f : I -> E used ONLY at ground arguments (the #P2b-63 ground-value path),
1-2 universals over the index sort (select-eq / not-b / f-ground / guarded shapes),
optionally a positive `exists`, optionally a NEGATED `exists` (a universal in
disguise, which the completion declines), ground reads / stores / array equalities,
optionally one push/pop around part of it, and 1-2 check-sat each followed by
(get-model).

Oracle (never trusts the probe):
  * every `sat` model is evaluated EXACTLY (recheck13/tools/qc_eval.py: arrays,
    f as a define-fun, quantifiers expanded over all 128 points) against the
    assertions active at that check -> `falsifying` if false (a published model
    that falsifies its own script);
  * every `unsat` is re-checked on the GROUND TWIN (each quantifier written out over
    all 128 points, same command sequence) by --twin-probe; a twin `sat` whose model
    evaluates TRUE against the quantified assertions is a `wrong_unsat`;
  * with `--judge z3` (re-fix pass 15, decision (69)(11)) z3 is the independent judge
    instead: every verdict is compared with z3's and every published model is replayed
    by z3 (`z3_*` counters, zjudge.py), beside the exact evaluation above.
`unknown` is counted, never a failure.  No (set-option :timeout ...) is emitted.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
import qc_eval as qe  # noqa: E402

W = 7
N = 1 << W


def bv(v, w=W):
    return "(_ bv%d %d)" % (v, w)


class Gen:
    def __init__(self, r: random.Random):
        self.r = r
        self.elem = r.choice(["bv1", "bv2", "bool"])
        self.na = r.choice([1, 2, 2, 3, 3, 4, 5])
        self.nk = r.choice([0, 1, 2])
        self.uf = r.random() < 0.35
        self.arrays = ["a%d" % i for i in range(self.na)]
        self.consts = ["k%d" % i for i in range(self.nk)]
        # a handful of index literals the script names
        self.lits = sorted(set(r.randrange(N) for _ in range(r.randint(1, 4))))

    def esort(self):
        return {"bv1": "(_ BitVec 1)", "bv2": "(_ BitVec 2)", "bool": "Bool"}[self.elem]

    def ev(self):
        if self.elem == "bool":
            return self.r.choice(["true", "false"])
        w = 1 if self.elem == "bv1" else 2
        return bv(self.r.randrange(1 << w), w)

    def enot(self, t):
        return "(not %s)" % t if self.elem == "bool" else "(bvnot %s)" % t

    def gidx(self):
        if self.consts and self.r.random() < 0.4:
            return self.r.choice(self.consts)
        return bv(self.r.choice(self.lits))

    def arr(self):
        return self.r.choice(self.arrays)

    def gelem(self):
        roll = self.r.random()
        if self.uf and roll < 0.3:
            return "(f %s)" % self.gidx()
        if roll < 0.6:
            return "(select %s %s)" % (self.arr(), self.gidx())
        return self.ev()

    def body(self):
        a = self.arr()
        b = self.arr()
        roll = self.r.random()
        if roll < 0.25:
            core = "(= (select %s i) %s)" % (a, self.ev())
        elif roll < 0.45:
            core = "(= (select %s i) %s)" % (a, self.enot("(select %s i)" % b)) if a != b else "(= (select %s i) %s)" % (a, self.ev())
        elif roll < 0.6:
            core = "(= (select %s i) (select %s i))" % (a, b)
        elif roll < 0.75:
            core = "(= (select %s i) %s)" % (a, self.gelem())
        else:
            core = "(distinct (select %s i) %s)" % (a, self.ev())
        g = self.r.random()
        if g < 0.35:
            return core
        if g < 0.55:
            return "(=> (distinct i %s) %s)" % (self.gidx(), core)
        if g < 0.75:
            op = self.r.choice(["bvuge", "bvult", "bvugt", "bvule"])
            return "(=> (%s i %s) %s)" % (op, bv(self.r.randrange(N)), core)
        return "(or (= i %s) %s)" % (self.gidx(), core)

    def ground(self):
        roll = self.r.random()
        a = self.arr()
        if roll < 0.45:
            return "(= (select %s %s) %s)" % (a, self.gidx(), self.gelem())
        if roll < 0.6 and len(self.arrays) > 1:
            b = self.r.choice([x for x in self.arrays if x != a])
            return "(= %s (store %s %s %s))" % (a, b, self.gidx(), self.ev())
        if roll < 0.7:
            return "(distinct (select %s %s) %s)" % (a, self.gidx(), self.gelem())
        if roll < 0.8 and self.consts:
            return "(= %s %s)" % (self.r.choice(self.consts), bv(self.r.choice(self.lits)))
        if roll < 0.9 and self.uf:
            return "(= (f %s) %s)" % (self.gidx(), self.ev())
        return "(= (select %s %s) %s)" % (a, self.gidx(), self.ev())

    def assertion(self):
        roll = self.r.random()
        if roll < 0.45:
            return "(forall ((i (_ BitVec %d))) %s)" % (W, self.body())
        if roll < 0.52:
            return "(exists ((i (_ BitVec %d))) (= (select %s i) %s))" % (W, self.arr(), self.ev())
        if roll < 0.58:
            return "(not (exists ((i (_ BitVec %d))) (= (select %s i) %s)))" % (W, self.arr(), self.ev())
        return self.ground()

    def script(self):
        lines = ["(set-logic ALL)"]
        for a in self.arrays:
            lines.append("(declare-const %s (Array (_ BitVec %d) %s))" % (a, W, self.esort()))
        for k in self.consts:
            lines.append("(declare-const %s (_ BitVec %d))" % (k, W))
        if self.uf:
            lines.append("(declare-fun f ((_ BitVec %d)) %s)" % (W, self.esort()))
        cmds = []
        n = self.r.randint(2, 5)
        quantified = False
        for _ in range(n):
            a = self.assertion()
            quantified |= a.startswith("(forall") or "exists" in a
            cmds.append("(assert %s)" % a)
        if not quantified:
            cmds.append("(assert (forall ((i (_ BitVec %d))) %s))" % (W, self.body()))
        if self.r.random() < 0.35:
            cut = self.r.randint(0, len(cmds))
            extra = ["(push 1)", "(assert %s)" % self.assertion(), "(check-sat)", "(get-model)", "(pop 1)"]
            cmds = cmds[:cut] + extra + cmds[cut:]
        cmds += ["(check-sat)", "(get-model)"]
        text = "\n".join(lines + cmds) + "\n"
        assert ":timeout" not in text
        return text


def make(seed, k):
    r = random.Random("qc/%d/%d" % (seed, k))
    return Gen(r).script()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--twin-probe", default="probe_tree")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--keep-dir")
    ap.add_argument("--judge", choices=["probe", "z3"], default="probe",
                    help="probe (default): every unsat is re-judged on the ground twin by "
                    "--twin-probe; z3: every verdict is judged by z3 on the quantified script "
                    "and every published model is replayed by z3 as well (zjudge.py)")
    ap.add_argument("--zcap", type=int, default=20, help="z3 cap per query with --judge z3")
    args = ap.parse_args()
    if args.keep_dir:
        os.makedirs(args.keep_dir, exist_ok=True)

    def one(k):
        text = make(args.seed, k)
        rec = {"k": k, "checks": 0, "sat": 0, "unsat": 0, "unknown": 0, "falsifying": 0,
               "model_unresolved": 0, "withheld": 0, "wrong_unsat": 0, "twin_unresolved": 0,
               "status": None}
        r = c.run_probe(args.probe, None, args.cap, extra_stdin=text)
        rec["status"] = r["status"]
        if r["status"] != "ok":
            return rec, text, r["stdout"]
        res = qe.judge(text, r["stdout"])
        for key in ("checks", "sat", "unsat", "unknown", "falsifying", "model_unresolved", "withheld"):
            rec[key] = res[key]
        if args.judge == "z3":
            import zjudge  # noqa: E402 - only with --judge z3
            zres = zjudge.judge(text, r["stdout"], args.zcap, True)
            for key in ("wrong_sat", "wrong_unsat", "z3_unknown", "model_ok", "model_unresolved",
                        "withheld"):
                rec["z3_" + key] = zres.get(key, 0)
            rec["z3_falsifying"] = zres.get("falsifying", 0)
            rec["wrong_unsat"] = zres.get("wrong_unsat", 0)
        elif res["unsat_checks"]:
            twin = qe.ground_twin(text)
            rt = c.run_probe(args.twin_probe, None, args.cap, extra_stdin=twin)
            if rt["status"] == "ok":
                tres = qe.judge_twin(text, rt["stdout"], res["unsat_checks"])
                rec["wrong_unsat"] = tres["wrong_unsat"]
                rec["twin_unresolved"] = tres["unresolved"]
            else:
                rec["twin_unresolved"] = len(res["unsat_checks"])
        return rec, text, r["stdout"]

    items = list(range(args.count))
    outs = c.pool_map(one, items, args.jobs)
    tot = {}
    bad = []
    for rec, text, out in outs:
        for key, v in rec.items():
            if isinstance(v, int) and key != "k":
                tot[key] = tot.get(key, 0) + v
        tot["status_" + str(rec["status"])] = tot.get("status_" + str(rec["status"]), 0) + 1
        if (rec["falsifying"] or rec["wrong_unsat"] or rec["withheld"] or rec.get("z3_falsifying")
                or rec.get("z3_wrong_sat") or rec["status"] == "panic"):
            bad.append(rec["k"])
            if args.keep_dir:
                with open(os.path.join(args.keep_dir, "qc_%d_%05d.smt2" % (args.seed, rec["k"])), "w") as fh:
                    fh.write(text)
                with open(os.path.join(args.keep_dir, "qc_%d_%05d.out" % (args.seed, rec["k"])), "w") as fh:
                    fh.write(out)
    print("fuzz_qc probe=%s judge=%s twin=%s seed=%d count=%d cap=%gs jobs=%d" % (
        args.probe, args.judge, args.twin_probe, args.seed, args.count, args.cap, args.jobs))
    print(json.dumps(tot, sort_keys=True))
    print("bad:", bad[:200])


if __name__ == "__main__":
    main()
