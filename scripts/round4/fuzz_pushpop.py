#!/usr/bin/env python3
"""fuzz_pushpop.py - incremental bit-vector scripts against a brute-force
oracle (decision (45)'s acceptance instrument).

    fuzz_pushpop.py --probe probe_tree --seed N --count 20000 [--jobs 2]
                    [--cap 130] [--keep-dir DIR]

Each script: 2-3 bit-vector variables of one width in 1..8 (at most 18 bits
of state in total, so the oracle enumerates every assignment), a random
command sequence of `assert`, `push 1` / `pop 1` (nesting depth 1 or 2) and
`check-sat` (at least two per script).  The oracle is numpy, evaluating every
active assertion over all 2^(n*w) assignments at each `check-sat`; it never
calls a solver.  Every seed is independent (`random.Random("seed/index")`).

Acceptance: wrong_sat 0, wrong_unsat 0, panics 0 (`unknown` is counted and
reported, never a failure).  Scripts that fail are written to --keep-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

import common as c

OPS2 = ["bvadd", "bvsub", "bvand", "bvor", "bvxor", "bvmul"]
OPS1 = ["bvnot", "bvneg"]
CMPS = ["=", "distinct", "bvult", "bvule", "bvslt", "bvsle"]


class G:
    def __init__(self, r, nv, w):
        self.r, self.nv, self.w = r, nv, w

    def const(self):
        return "(_ bv%d %d)" % (self.r.randrange(1 << self.w), self.w)

    def term(self, d):
        roll = self.r.random()
        if d <= 0 or roll < 0.35:
            return "x%d" % self.r.randrange(self.nv) if self.r.random() < 0.7 else self.const()
        if roll < 0.75:
            return "(%s %s %s)" % (self.r.choice(OPS2), self.term(d - 1), self.term(d - 1))
        if roll < 0.9:
            return "(%s %s)" % (self.r.choice(OPS1), self.term(d - 1))
        return "(ite %s %s %s)" % (self.atom(0), self.term(d - 1), self.term(d - 1))

    def atom(self, d):
        return "(%s %s %s)" % (self.r.choice(CMPS), self.term(d), self.term(d))

    def formula(self):
        roll = self.r.random()
        if roll < 0.55:
            return self.atom(2)
        if roll < 0.75:
            return "(or %s %s)" % (self.atom(1), self.atom(1))
        if roll < 0.9:
            return "(not %s)" % self.atom(2)
        return "(and %s %s)" % (self.atom(1), self.atom(1))


def make(seed, k):
    r = random.Random("%d/%d" % (seed, k))
    nv = r.choice([2, 3])
    w = r.randint(1, 8)
    if nv * w > 18:
        w = 18 // nv
    g = G(r, nv, w)
    cmds = []
    depth = 0
    checks = 0
    n_cmds = r.randint(6, 14)
    for _ in range(n_cmds):
        roll = r.random()
        if roll < 0.45:
            cmds.append(("assert", g.formula()))
        elif roll < 0.62 and depth < 2:
            cmds.append(("push",))
            depth += 1
        elif roll < 0.77 and depth > 0:
            cmds.append(("pop",))
            depth -= 1
        else:
            cmds.append(("check",))
            checks += 1
    while checks < 2:
        cmds.append(("check",))
        checks += 1
    if cmds[-1] != ("check",):
        cmds.append(("check",))
    text = "(set-logic QF_BV)\n" + "".join("(declare-fun x%d () (_ BitVec %d))\n" % (i, w) for i in range(nv))
    for cmd in cmds:
        if cmd[0] == "assert":
            text += "(assert %s)\n" % cmd[1]
        elif cmd[0] == "push":
            text += "(push 1)\n"
        elif cmd[0] == "pop":
            text += "(pop 1)\n"
        else:
            text += "(check-sat)\n"
    assert ":timeout" not in text
    return text, nv, w, cmds


def np_eval(e, env, w):
    """numpy value of a QF_BV expression over all assignments."""
    m = (1 << w) - 1
    if isinstance(e, str):
        if e in env:
            return env[e]
        if e == "true":
            return True
        if e == "false":
            return False
        raise ValueError(e)
    if e[0] == "_":
        return int(e[1][2:]) & m
    head, args = e[0], e[1:]
    if head == "ite":
        cnd = np_eval(args[0], env, w)
        return np.where(cnd, np_eval(args[1], env, w), np_eval(args[2], env, w))
    vs = [np_eval(a, env, w) for a in args]
    if head == "and":
        out = vs[0]
        for v in vs[1:]:
            out = np.logical_and(out, v)
        return out
    if head == "or":
        out = vs[0]
        for v in vs[1:]:
            out = np.logical_or(out, v)
        return out
    if head == "not":
        return np.logical_not(vs[0])
    if head == "=":
        return np.equal(vs[0], vs[1])
    if head == "distinct":
        return np.not_equal(vs[0], vs[1])
    if head == "bvult":
        return np.less(vs[0], vs[1])
    if head == "bvule":
        return np.less_equal(vs[0], vs[1])
    if head in ("bvslt", "bvsle"):
        def sg(v):
            v = np.asarray(v, dtype=np.int64)
            return v - (((v >> (w - 1)) & 1) << w)
        return np.less(sg(vs[0]), sg(vs[1])) if head == "bvslt" else np.less_equal(sg(vs[0]), sg(vs[1]))
    if head == "bvadd":
        return (vs[0] + vs[1]) & m
    if head == "bvsub":
        return (vs[0] - vs[1]) & m
    if head == "bvmul":
        return (vs[0] * vs[1]) & m
    if head == "bvand":
        return vs[0] & vs[1]
    if head == "bvor":
        return vs[0] | vs[1]
    if head == "bvxor":
        return vs[0] ^ vs[1]
    if head == "bvnot":
        return vs[0] ^ m
    if head == "bvneg":
        return (-vs[0]) & m
    raise ValueError(head)


def oracle(nv, w, cmds):
    n = 1 << (nv * w)
    base = np.arange(n, dtype=np.int64)
    env = {"x%d" % i: (base >> (i * w)) & ((1 << w) - 1) for i in range(nv)}
    stack = [[]]
    answers = []
    for cmd in cmds:
        if cmd[0] == "assert":
            v = np_eval(c.parse_all(cmd[1])[0], env, w)
            stack[-1].append(np.broadcast_to(np.asarray(v, dtype=bool), (n,)))
        elif cmd[0] == "push":
            stack.append([])
        elif cmd[0] == "pop":
            stack.pop()
        else:
            acc = np.ones(n, dtype=bool)
            for frame in stack:
                for v in frame:
                    acc &= v
            answers.append("sat" if acc.any() else "unsat")
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--keep-dir")
    args = ap.parse_args()

    def one(k):
        text, nv, w, cmds = make(args.seed, k)
        truth = oracle(nv, w, cmds)
        r = c.run_probe(args.probe, None, args.cap, extra_stdin=text)
        got = r["verdicts"]
        rec = {"k": k, "status": r["status"], "wrong_sat": 0, "wrong_unsat": 0, "unknown": 0,
               "checks": len(truth), "short": 0}
        if r["status"] != "ok":
            rec["short"] = 1
        for i, t in enumerate(truth):
            g = got[i] if i < len(got) else None
            if g is None:
                rec["short"] = 1
            elif g == "unknown":
                rec["unknown"] += 1
            elif g == "sat" and t == "unsat":
                rec["wrong_sat"] += 1
            elif g == "unsat" and t == "sat":
                rec["wrong_unsat"] += 1
        bad = rec["wrong_sat"] or rec["wrong_unsat"] or r["status"] == "panic"
        if bad and args.keep_dir:
            os.makedirs(args.keep_dir, exist_ok=True)
            with open(os.path.join(args.keep_dir, "f%d_%06d.smt2" % (args.seed, k)), "w") as fh:
                fh.write(text + "; truth " + " ".join(truth) + "\n; got " + " ".join(got) + "\n")
        return rec

    recs = c.pool_map(one, range(args.start, args.start + args.count), args.jobs)
    tot = {"scripts": len(recs), "checks": sum(r["checks"] for r in recs),
           "wrong_sat": sum(r["wrong_sat"] for r in recs), "wrong_unsat": sum(r["wrong_unsat"] for r in recs),
           "unknown": sum(r["unknown"] for r in recs), "panics": sum(r["status"] == "panic" for r in recs),
           "timeouts": sum(r["status"] == "timeout" for r in recs),
           "short_responses": sum(r["short"] for r in recs)}
    print("fuzz_pushpop probe=%s seed=%d start=%d count=%d cap=%gs jobs=%d" % (
        args.probe, args.seed, args.start, args.count, args.cap, args.jobs))
    print(json.dumps(tot))
    for r in recs:
        if r["wrong_sat"] or r["wrong_unsat"] or r["status"] == "panic":
            print("  BAD k=%d %s" % (r["k"], json.dumps(r)))


if __name__ == "__main__":
    main()
