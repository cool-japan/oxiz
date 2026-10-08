#!/usr/bin/env python3
"""gen14.py - recheck 14's seeded generator of mixed-theory scripts for zjudge.py.

Families (chosen per script):
  qf_uf    : QF_UFLIA / QF_UFLRA with nested applications, ite, disequalities
  qf_auf   : QF_AUFLIA with nested selects / stores / UF over reads
  q_uf     : quantified UF (no arrays) over Int / BV7, forall and exists, guards
  q_arr    : quantified arrays over Int / BV7 with UF and arithmetic in bodies
  dt       : datatype (enum and record) indices / UF arguments, QF and quantified
  mix      : any of the above with push/pop and several check-sats
Every (check-sat) is followed by (get-model) and one (get-value) over ground terms.
No script sets :timeout.
"""
from __future__ import annotations

import argparse
import os
import random

INT_LITS = ["0", "1", "2", "3", "5", "7", "(- 1)", "(- 3)", "10"]
REAL_LITS = ["0.0", "1.0", "2.5", "(- 1.0)", "4.0", "(/ 1.0 3.0)"]
BV_W = 7


def bvlit(r, w=BV_W):
    return "(_ bv%d %d)" % (r.randrange(1 << w), w)


class G:
    def __init__(self, r, fam):
        self.r = r
        self.fam = fam
        self.num = "Real" if (fam == "qf_uf" and r.random() < 0.25) else "Int"
        self.decls = []
        self.use_bv = r.random() < 0.3
        self.use_dt = fam == "dt"
        self.consts = {"N": ["k", "j", "m"][: r.randint(2, 3)]}
        self.funs = {"N": [], "P": [], "BVF": [], "DF": []}
        self.arrays = {"AN": [], "AB": [], "AD": []}
        nsort = self.num
        for name in self.consts["N"]:
            self.decls.append("(declare-const %s %s)" % (name, nsort))
        if fam in ("qf_uf", "q_uf", "mix", "dt", "q_arr", "qf_auf"):
            for name in ["f", "g"][: r.randint(1, 2)]:
                self.funs["N"].append(name)
                self.decls.append("(declare-fun %s (%s) %s)" % (name, nsort, nsort))
            if r.random() < 0.4:
                self.funs["P"].append("p")
                self.decls.append("(declare-fun p (%s) Bool)" % nsort)
        if fam in ("qf_auf", "q_arr", "mix") and nsort == "Int":
            for name in ["a", "b"][: r.randint(1, 2)]:
                self.arrays["AN"].append(name)
                self.decls.append("(declare-const %s (Array Int Int))" % name)
        if self.use_bv:
            self.consts["B"] = ["x", "y"]
            for name in self.consts["B"]:
                self.decls.append("(declare-const %s (_ BitVec %d))" % (name, BV_W))
            self.funs["BVF"].append("h")
            self.decls.append("(declare-fun h ((_ BitVec %d)) (_ BitVec %d))" % (BV_W, BV_W))
            if fam in ("q_arr", "qf_auf", "mix"):
                self.arrays["AB"].append("c")
                self.decls.append("(declare-const c (Array (_ BitVec %d) (_ BitVec %d)))" % (BV_W, BV_W))
        else:
            self.consts["B"] = []
        if self.use_dt:
            self.prelude = ["(declare-datatypes ((Color 0) (Pair 0)) (((red) (green) (blue)) ((mk (fst Int) (snd Color)))))"]
            self.consts["D"] = ["u", "v"]
            for name in self.consts["D"]:
                self.decls.append("(declare-const %s Color)" % name)
            self.consts["PR"] = ["w"]
            self.decls.append("(declare-const w Pair)")
            self.funs["DF"].append("col")
            self.decls.append("(declare-fun col (Color) Int)")
            self.arrays["AD"].append("d")
            self.decls.append("(declare-const d (Array Color Int))")
            if nsort == "Int" and r.random() < 0.5:
                self.arrays["AD"].append("e")
                self.decls.append("(declare-const e (Array Pair Int))")
        else:
            self.prelude = []
            self.consts["D"] = []
            self.consts["PR"] = []

    # ---- terms -----------------------------------------------------------
    def nlit(self):
        return self.r.choice(REAL_LITS if self.num == "Real" else INT_LITS)

    def nterm(self, depth, bound=()):
        r = self.r
        leaves = list(self.consts["N"]) + [b for b, s in bound if s == "N"]
        if depth <= 0 or r.random() < 0.3:
            return r.choice(leaves) if r.random() < 0.8 else self.nlit()
        opts = ["leaf", "arith"]
        if self.funs["N"]:
            opts += ["app", "app"]
        if self.arrays["AN"]:
            opts += ["sel", "sel"]
        if self.arrays["AD"] and self.num == "Int":
            opts += ["dsel"]
        if self.funs["DF"] and self.num == "Int":
            opts += ["col"]
        if self.consts["PR"] and self.num == "Int":
            opts += ["fst"]
        opts += ["ite"]
        k = r.choice(opts)
        if k == "leaf":
            return r.choice(leaves)
        if k == "arith":
            op = r.choice(["+", "-", "+"])
            return "(%s %s %s)" % (op, self.nterm(depth - 1, bound), self.nlit())
        if k == "app":
            return "(%s %s)" % (r.choice(self.funs["N"]), self.nterm(depth - 1, bound))
        if k == "sel":
            return "(select %s %s)" % (self.aterm(depth - 1, bound), self.nterm(depth - 1, bound))
        if k == "dsel":
            arr = r.choice(self.arrays["AD"])
            idx = self.dterm(bound) if arr == "d" else self.pterm(depth - 1, bound)
            return "(select %s %s)" % (arr, idx)
        if k == "col":
            return "(col %s)" % self.dterm(bound)
        if k == "fst":
            return "(fst %s)" % self.pterm(depth - 1, bound)
        return "(ite %s %s %s)" % (self.atom(depth - 1, bound), self.nterm(depth - 1, bound), self.nterm(depth - 1, bound))

    def aterm(self, depth, bound=()):
        r = self.r
        base = r.choice(self.arrays["AN"])
        if depth > 0 and r.random() < 0.3:
            return "(store %s %s %s)" % (base, self.nterm(depth - 1, bound), self.nterm(depth - 1, bound))
        if depth > 0 and r.random() < 0.1:
            return "((as const (Array Int Int)) %s)" % self.nlit()
        return base

    def dterm(self, bound=()):
        leaves = list(self.consts["D"]) + [b for b, s in bound if s == "D"] + ["red", "green", "blue"]
        return self.r.choice(leaves)

    def pterm(self, depth, bound=()):
        r = self.r
        leaves = list(self.consts["PR"]) + [b for b, s in bound if s == "PR"]
        if depth <= 0 or r.random() < 0.6:
            return r.choice(leaves)
        return "(mk %s %s)" % (self.nterm(depth - 1, bound), self.dterm(bound))

    def bterm(self, depth, bound=()):
        r = self.r
        leaves = list(self.consts["B"]) + [b for b, s in bound if s == "B"]
        if not leaves:
            leaves = [bvlit(r)]
        if depth <= 0 or r.random() < 0.35:
            return r.choice(leaves) if r.random() < 0.8 else bvlit(r)
        opts = ["app", "add", "ite"]
        if self.arrays["AB"]:
            opts += ["sel", "sel"]
        k = r.choice(opts)
        if k == "app":
            return "(h %s)" % self.bterm(depth - 1, bound)
        if k == "add":
            return "(bvadd %s %s)" % (self.bterm(depth - 1, bound), bvlit(r))
        if k == "sel":
            return "(select c %s)" % self.bterm(depth - 1, bound)
        return "(ite %s %s %s)" % (self.atom(depth - 1, bound), self.bterm(depth - 1, bound), self.bterm(depth - 1, bound))

    def atom(self, depth, bound=()):
        r = self.r
        opts = ["eq", "eq", "neq", "lt", "le"]
        if self.funs["P"]:
            opts.append("p")
        if self.consts["B"] or any(s == "B" for _, s in bound):
            opts += ["beq", "bult"]
        if self.consts["D"]:
            opts += ["deq"]
        if self.arrays["AN"] and r.random() < 0.15:
            opts += ["aeq"]
        k = r.choice(opts)
        if k == "eq":
            return "(= %s %s)" % (self.nterm(depth, bound), self.nterm(depth, bound) if r.random() < 0.6 else self.nlit())
        if k == "neq":
            return "(distinct %s %s)" % (self.nterm(depth, bound), self.nterm(depth, bound))
        if k == "lt":
            return "(< %s %s)" % (self.nterm(depth, bound), self.nterm(depth, bound))
        if k == "le":
            return "(<= %s %s)" % (self.nterm(depth, bound), self.nlit())
        if k == "p":
            return "(p %s)" % self.nterm(depth, bound)
        if k == "beq":
            return "(= %s %s)" % (self.bterm(depth, bound), self.bterm(depth, bound))
        if k == "bult":
            return "(bvult %s %s)" % (self.bterm(depth, bound), self.bterm(depth, bound))
        if k == "deq":
            return "(= %s %s)" % (self.dterm(bound), self.dterm(bound))
        return "(= %s %s)" % (self.aterm(1, bound), self.aterm(1, bound))

    def formula(self, depth, bound=()):
        r = self.r
        if depth <= 0 or r.random() < 0.45:
            a = self.atom(r.randint(0, 2), bound)
            return a if r.random() < 0.8 else "(not %s)" % a
        k = r.choice(["and", "or", "=>", "not", "ite", "xor"])
        if k == "not":
            return "(not %s)" % self.formula(depth - 1, bound)
        if k == "ite":
            return "(ite %s %s %s)" % (self.formula(depth - 1, bound), self.formula(depth - 1, bound), self.formula(depth - 1, bound))
        return "(%s %s %s)" % (k, self.formula(depth - 1, bound), self.formula(depth - 1, bound))

    def quantified(self):
        r = self.r
        kinds = []
        if self.num == "Int" or self.fam != "q_arr":
            kinds.append(("N", self.num))
        if self.consts["B"]:
            kinds.append(("B", "(_ BitVec %d)" % BV_W))
        if self.consts["D"]:
            kinds.append(("D", "Color"))
        s, sort = r.choice(kinds)
        var = r.choice(["i", "xx", "q"])
        bound = ((var, s),)
        q = r.choice(["forall", "forall", "forall", "exists"])
        if r.random() < 0.5:
            # guarded shape
            if s == "N":
                guard = r.choice(["(> %s %s)", "(< %s %s)", "(= %s %s)", "(>= %s %s)"]) % (var, self.nlit())
            elif s == "B":
                guard = "(bvult %s %s)" % (var, bvlit(r))
            else:
                guard = "(= %s %s)" % (var, self.dterm())
            body = "(=> %s %s)" % (guard, self.formula(1, bound))
        else:
            body = self.formula(2, bound)
        text = "(%s ((%s %s)) %s)" % (q, var, sort, body)
        if r.random() < 0.15:
            text = "(not %s)" % text
        return text

    def assertion(self):
        if self.fam in ("q_uf", "q_arr", "dt", "mix") and self.r.random() < 0.4:
            return "(assert %s)" % self.quantified()
        return "(assert %s)" % self.formula(2)

    def gv_terms(self):
        r = self.r
        ts = []
        for _ in range(r.randint(2, 4)):
            ts.append(self.nterm(2))
        if self.consts["B"]:
            ts.append(self.bterm(1))
        if self.consts["D"]:
            ts.append(self.dterm())
        return ts


def script(seed_text: str) -> str:
    r = random.Random(seed_text)
    fam = r.choice(["qf_uf", "qf_uf", "qf_auf", "qf_auf", "q_uf", "q_uf", "q_arr", "q_arr", "dt", "mix"])
    g = G(r, fam)
    lines = ["(set-logic ALL)", "(set-option :produce-models true)"] + g.prelude + g.decls
    checks = r.randint(1, 3) if fam == "mix" or r.random() < 0.3 else 1
    depth = 0
    for ci in range(checks):
        for _ in range(r.randint(1, 4)):
            lines.append(g.assertion())
        lines.append("(check-sat)")
        lines.append("(get-model)")
        lines.append("(get-value (%s))" % " ".join(g.gv_terms()))
        if ci + 1 < checks:
            if depth > 0 and r.random() < 0.4:
                lines.append("(pop 1)")
                depth -= 1
            else:
                lines.append("(push 1)")
                depth += 1
    text = "\n".join(lines) + "\n"
    assert ":timeout" not in text
    return "; gen14 family %s\n" % fam + text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        with open(os.path.join(args.out, "s%05d.smt2" % i), "w") as fh:
            fh.write(script("%d/%d" % (args.seed, i)))


if __name__ == "__main__":
    main()
