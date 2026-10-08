#!/usr/bin/env python3
"""fuzz_abv.py - the recheck's incremental QF_AUFBV fuzzer (decision (45)
attack: "interleaved BV and array assertions across three levels").

    fuzz_abv.py --probe probe_tree --seed N --count 20000 [--jobs 2] [--cap 130]
                [--keep-dir DIR]

Each script declares, over an index sort I = (_ BitVec wi) (wi in 1..2) and an
element sort E = (_ BitVec we) (we in 1..3):
  * 1-2 index constants i0 i1, 0-1 element constant e0,
  * 1-2 arrays a0 a1 : (Array I E),
  * optionally one uninterpreted function f : I -> E,
with at most 16 bits of state in total, so a numpy oracle enumerates every
interpretation (arrays and f are packed integers over all 2^wi cells).

Terms: select / store / (as const) / array ite / array equality and
disequality / f applications / bvadd bvxor bvnot / bvult bvule / ite, over a
shared term pool so one circuit is asserted, popped and re-asserted later.
Commands: assert (named in a third of the scripts), push k / pop k (depth <=
3), check-sat (>= 2), and after every check-sat a (get-value ...) of every
constant, every array cell and every f cell, so each published `sat` model is
checked against the active assertions (bad_model); with cores on, every
`unsat` core must name active assertions only and be unsatisfiable alone
(bad_core).  `(get-info :all-statistics)` at the end.

The oracle never calls a solver.  `unknown` is counted, never a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402


class Gen:
    def __init__(self, r, wi, we, ni, ne, na, uf):
        self.r, self.wi, self.we, self.ni, self.ne, self.na, self.uf = r, wi, we, ni, ne, na, uf
        self.pool_e: list[str] = []
        self.pool_a: list[str] = []
        for _ in range(r.randint(2, 4)):
            self.pool_e.append(self.eterm(2))
        for _ in range(r.randint(1, 3)):
            self.pool_a.append(self.aterm(2))

    def ic(self):
        return "(_ bv%d %d)" % (self.r.randrange(1 << self.wi), self.wi)

    def ec(self):
        return "(_ bv%d %d)" % (self.r.randrange(1 << self.we), self.we)

    def iterm(self, d):
        roll = self.r.random()
        if d <= 0 or roll < 0.55:
            return "i%d" % self.r.randrange(self.ni) if self.r.random() < 0.7 else self.ic()
        if roll < 0.75:
            return "(bvadd %s %s)" % (self.iterm(d - 1), self.iterm(d - 1))
        if roll < 0.85:
            return "(bvnot %s)" % self.iterm(d - 1)
        return "(ite %s %s %s)" % (self.atom(0), self.iterm(d - 1), self.iterm(d - 1))

    def eterm(self, d):
        roll = self.r.random()
        if d <= 0 or roll < 0.2:
            if self.ne and self.r.random() < 0.5:
                return "e0"
            return self.ec()
        if roll < 0.55:
            return "(select %s %s)" % (self.aterm(d - 1), self.iterm(d - 1))
        if roll < 0.65 and self.uf:
            return "(f %s)" % self.iterm(d - 1)
        if roll < 0.75 and getattr(self, "pool_e", None):
            return self.r.choice(self.pool_e)
        if roll < 0.85:
            return "(%s %s %s)" % (self.r.choice(["bvadd", "bvxor"]), self.eterm(d - 1), self.eterm(d - 1))
        return "(ite %s %s %s)" % (self.atom(0), self.eterm(d - 1), self.eterm(d - 1))

    def aterm(self, d):
        roll = self.r.random()
        if d <= 0 or roll < 0.4:
            if self.r.random() < 0.9:
                return "a%d" % self.r.randrange(self.na)
            return "((as const (Array (_ BitVec %d) (_ BitVec %d))) %s)" % (self.wi, self.we, self.ec())
        if roll < 0.55 and getattr(self, "pool_a", None):
            return self.r.choice(self.pool_a)
        if roll < 0.85:
            return "(store %s %s %s)" % (self.aterm(d - 1), self.iterm(d - 1), self.eterm(d - 1))
        return "(ite %s %s %s)" % (self.atom(0), self.aterm(d - 1), self.aterm(d - 1))

    def atom(self, d):
        roll = self.r.random()
        if roll < 0.35:
            return "(= %s %s)" % (self.eterm(d), self.eterm(d))
        if roll < 0.5:
            return "(bvult %s %s)" % (self.eterm(d), self.eterm(d))
        if roll < 0.62:
            return "(= %s %s)" % (self.iterm(d), self.iterm(d))
        if roll < 0.7:
            return "(bvule %s %s)" % (self.iterm(d), self.iterm(d))
        if roll < 0.85:
            return "(= %s %s)" % (self.aterm(d), self.aterm(d))
        return "(distinct %s %s)" % (self.aterm(d), self.aterm(d))

    def formula(self):
        roll = self.r.random()
        if roll < 0.45:
            return self.atom(1)
        if roll < 0.6:
            return "(or %s %s)" % (self.atom(1), self.atom(1))
        if roll < 0.75:
            return "(not %s)" % self.atom(1)
        if roll < 0.87:
            return "(and %s %s)" % (self.atom(1), self.atom(0))
        return "(=> %s %s)" % (self.atom(0), self.atom(1))


def params(r):
    while True:
        wi = r.randint(1, 2)
        we = r.randint(1, 3)
        ni = r.randint(1, 2)
        ne = r.randint(0, 1)
        na = r.randint(1, 2)
        uf = r.random() < 0.35
        cells = 1 << wi
        bits = ni * wi + ne * we + na * cells * we + (cells * we if uf else 0)
        if bits <= 16:
            return wi, we, ni, ne, na, uf, bits


def make(seed, k):
    r = random.Random("abv/%d/%d" % (seed, k))
    wi, we, ni, ne, na, uf, bits = params(r)
    cores = r.random() < 0.33
    g = Gen(r, wi, we, ni, ne, na, uf)
    cmds = []
    depth = 0
    checks = 0
    names = 0
    for _ in range(r.randint(6, 16)):
        roll = r.random()
        if roll < 0.45:
            cmds.append(("assert", g.formula(), "n%d" % names if cores else None))
            names += 1
        elif roll < 0.6 and depth < 3:
            n = 1 if depth == 2 or r.random() < 0.7 else 2
            cmds.append(("push", n))
            depth += n
        elif roll < 0.74 and depth > 0:
            n = r.randint(1, depth)
            cmds.append(("pop", n))
            depth -= n
        else:
            cmds.append(("check",))
            checks += 1
    while checks < 2:
        cmds.append(("check",))
        checks += 1
    cells = 1 << wi
    idx = ["(_ bv%d %d)" % (j, wi) for j in range(cells)]
    gv = ["i%d" % j for j in range(ni)] + (["e0"] if ne else [])
    gv += ["(select a%d %s)" % (a, ix) for a in range(na) for ix in idx]
    if uf:
        gv += ["(f %s)" % ix for ix in idx]
    text = "(set-logic QF_AUFBV)\n"
    if cores:
        text += "(set-option :produce-unsat-cores true)\n"
    text += "".join("(declare-fun i%d () (_ BitVec %d))\n" % (j, wi) for j in range(ni))
    if ne:
        text += "(declare-fun e0 () (_ BitVec %d))\n" % we
    text += "".join("(declare-fun a%d () (Array (_ BitVec %d) (_ BitVec %d)))\n" % (a, wi, we) for a in range(na))
    if uf:
        text += "(declare-fun f ((_ BitVec %d)) (_ BitVec %d))\n" % (wi, we)
    for cmd in cmds:
        if cmd[0] == "assert":
            text += ("(assert (! %s :named %s))\n" % (cmd[1], cmd[2])) if cmd[2] else "(assert %s)\n" % cmd[1]
        elif cmd[0] == "push":
            text += "(push %d)\n" % cmd[1]
        elif cmd[0] == "pop":
            text += "(pop %d)\n" % cmd[1]
        else:
            text += "(check-sat)\n(get-value (%s))\n" % " ".join(gv)
            if cores:
                text += "(get-unsat-core)\n"
    text += "(get-info :all-statistics)\n"
    assert ":timeout" not in text
    return text, (wi, we, ni, ne, na, uf, bits), cmds, cores, gv


class Oracle:
    """Every interpretation, packed: i_j, e0, a_j (cells * we bits), f."""

    def __init__(self, p):
        wi, we, ni, ne, na, uf, bits = p
        self.wi, self.we = wi, we
        self.cells = 1 << wi
        self.n = 1 << bits
        base = np.arange(self.n, dtype=np.int64)
        self.env = {}
        self.layout = []
        off = 0

        def take(name, w):
            nonlocal off
            self.env[name] = (base >> off) & ((1 << w) - 1)
            self.layout.append((name, off, w))
            off += w

        for j in range(ni):
            take("i%d" % j, wi)
        if ne:
            take("e0", we)
        for a in range(na):
            take("a%d" % a, self.cells * we)
        if uf:
            take("f", self.cells * we)

    def sel(self, arr, i):
        return (arr >> (np.asarray(i, dtype=np.int64) * self.we)) & ((1 << self.we) - 1)

    def ev(self, e):
        we, wi = self.we, self.wi
        if isinstance(e, str):
            if e in self.env:
                return self.env[e]
            if e == "true":
                return np.ones(self.n, dtype=bool)
            if e == "false":
                return np.zeros(self.n, dtype=bool)
            raise ValueError(e)
        head = e[0]
        if head == "_":
            return np.full(self.n, int(e[1][2:]), dtype=np.int64)
        if isinstance(head, list):  # ((as const (Array I E)) v)
            v = self.ev(e[1])
            out = np.zeros(self.n, dtype=np.int64)
            for j in range(self.cells):
                out |= v << (j * we)
            return out
        args = e[1:]
        if head == "ite":
            return np.where(self.ev(args[0]), self.ev(args[1]), self.ev(args[2]))
        vs = [self.ev(a) for a in args]
        if head == "and":
            return np.logical_and(vs[0], vs[1])
        if head == "or":
            return np.logical_or(vs[0], vs[1])
        if head == "not":
            return np.logical_not(vs[0])
        if head == "=>":
            return np.logical_or(np.logical_not(vs[0]), vs[1])
        if head == "=":
            return np.equal(vs[0], vs[1])
        if head == "distinct":
            return np.not_equal(vs[0], vs[1])
        if head == "bvult":
            return np.less(vs[0], vs[1])
        if head == "bvule":
            return np.less_equal(vs[0], vs[1])
        if head == "select":
            return self.sel(vs[0], vs[1])
        if head == "f":
            return self.sel(self.env["f"], vs[0])
        if head == "store":
            m = (1 << we) - 1
            sh = vs[1] * we
            return (vs[0] & ~(np.int64(m) << sh)) | (vs[2] << sh)
        if head == "bvadd":
            w = self.width_of(args[0])
            return (vs[0] + vs[1]) & ((1 << w) - 1)
        if head == "bvxor":
            return vs[0] ^ vs[1]
        if head == "bvnot":
            w = self.width_of(args[0])
            return vs[0] ^ ((1 << w) - 1)
        raise ValueError(head)

    def width_of(self, e):
        if isinstance(e, str):
            if e.startswith("i"):
                return self.wi
            return self.we
        if e[0] == "_":
            return int(e[2])
        if e[0] in ("select", "f"):
            return self.we
        if e[0] == "ite":
            return self.width_of(e[2])
        return self.width_of(e[1])


def lit_value(text):
    if text.startswith("#b"):
        return int(text[2:], 2)
    if text.startswith("#x"):
        return int(text[2:], 16)
    return None


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
        text, p, cmds, cores, gv = make(args.seed, k)
        o = Oracle(p)
        r = c.run_probe(args.probe, None, args.cap, extra_stdin=text)
        rec = {"k": k, "status": r["status"], "checks": 0, "wrong_sat": 0, "wrong_unsat": 0, "unknown": 0,
               "bad_model": 0, "bad_core": 0, "short": 0, "models_checked": 0, "cores_checked": 0,
               "stats_missing": 0, "wild_cells": 0, "core_missing": 0}
        try:
            resp = c.parse_all(r["stdout"])
        except ValueError:
            resp = []
            rec["short"] = 1
        pos = 0

        def take():
            nonlocal pos
            if pos < len(resp):
                pos += 1
                return resp[pos - 1]
            return None

        stack = [[]]
        for cmd in cmds:
            if cmd[0] == "assert":
                v = np.broadcast_to(np.asarray(o.ev(c.parse_all(cmd[1])[0]), dtype=bool), (o.n,))
                stack[-1].append((cmd[2], v))
            elif cmd[0] == "push":
                for _ in range(cmd[1]):
                    stack.append([])
            elif cmd[0] == "pop":
                for _ in range(cmd[1]):
                    stack.pop()
            else:
                rec["checks"] += 1
                acc = np.ones(o.n, dtype=bool)
                for frame in stack:
                    for _nm, v in frame:
                        acc &= v
                truth = "sat" if acc.any() else "unsat"
                got = take()
                vals = take()
                core = take() if cores else None
                if got not in ("sat", "unsat", "unknown"):
                    rec["short"] = 1
                    continue
                if got == "unknown":
                    rec["unknown"] += 1
                elif got != truth:
                    rec["wrong_" + got] += 1
                if got == "sat":
                    # The published values, as constraints on the packed
                    # interpretation: a literal fixes a component, a value the
                    # solver echoes as a term (an unconstrained cell) is a
                    # wildcard.  The model is accepted iff SOME completion of
                    # the wildcards satisfies the active assertions (so a
                    # literal that contradicts them is caught); wildcards are
                    # counted separately.
                    ok = isinstance(vals, list) and len(vals) == len(gv)
                    assign = {}
                    wild = 0
                    if ok:
                        for pair in vals:
                            if not (isinstance(pair, list) and len(pair) == 2):
                                ok = False
                                break
                            val = lit_value(pair[1]) if isinstance(pair[1], str) else None
                            if val is None:
                                wild += 1
                                continue
                            assign[c.show(pair[0])] = val
                    if ok:
                        mask = np.ones(o.n, dtype=bool)
                        for name, off, w in o.layout:
                            comp = o.env[name]
                            if name.startswith("a") or name == "f":
                                for j in range(o.cells):
                                    lit = "(_ bv%d %d)" % (j, o.wi)
                                    key = ("(select %s %s)" % (name, lit)) if name != "f" else ("(f %s)" % lit)
                                    v = assign.get(key)
                                    if v is not None:
                                        mask &= ((comp >> (j * o.we)) & ((1 << o.we) - 1)) == v
                            else:
                                v = assign.get(name)
                                if v is not None:
                                    mask &= comp == v
                        rec["models_checked"] += 1
                        rec["wild_cells"] += wild
                        if not (acc & mask).any():
                            rec["bad_model"] += 1
                    else:
                        rec["bad_model"] += 1
                if cores and got == "unsat" and truth == "unsat":
                    active = {nm: v for frame in stack for nm, v in frame}
                    if isinstance(core, list) and core and core[0] == "error":
                        rec["core_missing"] += 1
                    elif isinstance(core, list) and all(isinstance(x, str) and x in active for x in core):
                        rec["cores_checked"] += 1
                        cacc = np.ones(o.n, dtype=bool)
                        for nm in core:
                            cacc &= active[nm]
                        if cacc.any():
                            rec["bad_core"] += 1
                    else:
                        rec["bad_core"] += 1
        if ":bv-embedded-checks" not in r["stdout"] or ":bv-embedded-conflicts" not in r["stdout"]:
            rec["stats_missing"] = 1
        bad = rec["wrong_sat"] or rec["wrong_unsat"] or rec["bad_model"] or rec["bad_core"] or r["status"] == "panic"
        if bad and args.keep_dir:
            os.makedirs(args.keep_dir, exist_ok=True)
            with open(os.path.join(args.keep_dir, "abv%d_%06d.smt2" % (args.seed, k)), "w") as fh:
                fh.write(text + "\n; got:\n; " + r["stdout"].replace("\n", "\n; ") + "\n")
        return rec

    recs = c.pool_map(one, range(args.start, args.start + args.count), args.jobs)
    keys = ["checks", "wrong_sat", "wrong_unsat", "unknown", "bad_model", "bad_core", "short",
            "models_checked", "cores_checked", "stats_missing", "wild_cells", "core_missing"]
    tot = {"scripts": len(recs)}
    for key in keys:
        tot[key] = sum(rr[key] for rr in recs)
    tot["panics"] = sum(rr["status"] == "panic" for rr in recs)
    tot["timeouts"] = sum(rr["status"] == "timeout" for rr in recs)
    print("fuzz_abv probe=%s seed=%d start=%d count=%d cap=%gs jobs=%d" % (
        args.probe, args.seed, args.start, args.count, args.cap, args.jobs))
    print(json.dumps(tot))
    for rr in recs:
        if rr["wrong_sat"] or rr["wrong_unsat"] or rr["bad_model"] or rr["bad_core"] or rr["status"] == "panic":
            print("  BAD %s" % json.dumps(rr))


if __name__ == "__main__":
    main()
