#!/usr/bin/env python3
"""gen_dtite.py --seed N --count C --out DIR : re-fix pass 18's generator for the TODO.md #P2b-90 family
(decisions (84), (89)): testers, selectors and datatype equalities over datatype-sorted `ite`s.

The signature is gen_dt.py's (L = nil | cons(hd Int, tl L); C = red | green | blue; P = mk(px Int, pc C); an
uninterpreted sort U; h: Int -> L, g: Int -> C, k: L -> Int, q: C -> Int, w: Int -> U, v: L -> L; constants
x y z : Int, l1 l2 l3 : L, c1 c2 : C, p : P, u1 u2 : U) plus three Booleans a b e.  Every sort with a value
(L, C, U, Int) gets `ite` terms, and the shapes the family needs are weighted up: a tester over an `ite`
(`((_ is cons) (ite b l1 nil))`), a selector over one (under its tester or bare), an equality / `distinct` with an
`ite` on one or both sides, nested `ite`s and `ite` chains, an `ite` under an uninterpreted argument
(`(k (ite ...))`, `(v (ite ...))`) and under a constructor (`(cons 1 (ite ...))`), an enumeration and an
uninterpreted-sort `ite`, push/pop with 1-3 `check-sat`s, and in about a third of the scripts one quantified
assertion over Int (bounded or guarded universal, or an existential) whose body holds such a tester, an equality, a
selector chain only the body names, or a constructor over the bound variable.  Selectors appear only under their
tester (as in gen_dt.py), so a printed model fixes every term it is judged on.  A `(get-model)` follows every check.  No `:timeout`.
Judge with z3 (`camp14.py`).
"""
import argparse
import os
import random

HEAD = """(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun h (Int) L)
(declare-fun g (Int) C)
(declare-fun k (L) Int)
(declare-fun q (C) Int)
(declare-fun w (Int) U)
(declare-fun v (L) L)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const l1 L)
(declare-const l2 L)
(declare-const l3 L)
(declare-const c1 C)
(declare-const c2 C)
(declare-const p P)
(declare-const u1 U)
(declare-const u2 U)
(declare-const a Bool)
(declare-const b Bool)
(declare-const e Bool)
(assert (and (<= (- 3) x 3) (<= (- 3) y 3) (<= (- 3) z 3)))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1", action="store_true",
                    help="the first version (seed 30101802's corpus): selectors not guarded by their tester, "
                         "one bounded-universal shape, a quantifier in a fifth of the scripts")
    args = ap.parse_args()
    Gen.V1 = args.v1
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    for n in range(args.count):
        text = Gen(rng).script()
        assert ":timeout" not in text
        with open(os.path.join(args.out, "t%05d.smt2" % n), "w") as fh:
            fh.write(text)


class Gen:
    # Term depth past which every term is a leaf: the family lives in the
    # first two levels, and deeper terms only make the scripts slow.
    MAX_DEPTH = 2
    # The first version's choices (`--v1`), kept so its corpus regenerates.
    V1 = False

    def __init__(self, rng, bound=None):
        self.rng = rng
        # The bound variable of a quantifier body, usable as an Int leaf.
        self.bound = bound

    def pick(self, *items):
        return self.rng.choice(items)

    def cond(self, d):
        r = self.rng.random()
        if d >= self.MAX_DEPTH or r < 0.35:
            return self.pick("a", "b", "e", "(not a)", "(not b)")
        if r < 0.55:
            return "((_ is %s) %s)" % (self.pick("cons", "nil"), self.lterm(d + 1))
        if r < 0.65:
            return "((_ is %s) %s)" % (self.pick("red", "green", "blue"), self.cterm(d + 1))
        if r < 0.8:
            return "(%s %s %s)" % (self.pick("=", "<", "<="), self.iterm(d + 1), self.iterm(d + 1))
        if r < 0.9:
            return "(= %s %s)" % (self.lterm(d + 1), self.lterm(d + 1))
        return "(= %s %s)" % (self.uterm(d + 1), self.uterm(d + 1))

    def iterm(self, d):
        r = self.rng.random()
        leaves = ["x", "y", "z", "0", "1", "2", "(- 1)"]
        if self.bound:
            leaves.append(self.bound)
        if d >= self.MAX_DEPTH or r < 0.35:
            return self.pick(*leaves)
        if r < 0.5:
            return "(k %s)" % self.lterm(d + 1)
        if r < 0.58:
            return "(q %s)" % self.cterm(d + 1)
        if r < 0.64:
            return "(px p)"
        if r < 0.78:
            t = self.lterm(d + 1)
            return "(ite ((_ is cons) %s) (hd %s) 0)" % (t, t)
        if r < 0.86:
            if self.V1:
                return "(hd %s)" % self.lite(d + 1)
            t = self.lite(d + 1)
            return "(ite ((_ is cons) %s) (hd %s) 0)" % (t, t)
        if r < 0.94:
            return "(ite %s %s %s)" % (self.cond(d + 1), self.iterm(d + 1), self.iterm(d + 1))
        return "(+ %s %s)" % (self.iterm(d + 1), self.pick("1", "(- 1)", "x"))

    def lite(self, d):
        """A datatype-sorted `ite` (possibly a chain or a nest)."""
        r = self.rng.random()
        if r < 0.35:
            t = self.lterm(d + 1)
            return "(ite ((_ is cons) %s) (tl %s) nil)" % (t, t)
        if r < 0.6:
            return "(ite %s %s %s)" % (self.cond(d + 1), self.lterm(d + 1), self.lterm(d + 1))
        if r < 0.8:
            return "(ite %s %s (ite %s %s %s))" % (
                self.cond(d + 1), self.lterm(d + 1), self.cond(d + 1), self.lterm(d + 1), self.lterm(d + 1))
        return "(ite %s (ite %s %s %s) %s)" % (
            self.cond(d + 1), self.cond(d + 1), self.lterm(d + 1), self.lterm(d + 1), self.lterm(d + 1))

    def lterm(self, d):
        r = self.rng.random()
        if d >= self.MAX_DEPTH or r < 0.3:
            return self.pick("l1", "l2", "l3", "nil")
        if r < 0.45:
            return "(cons %s %s)" % (self.iterm(d + 1), self.lterm(d + 1))
        if r < 0.55:
            return "(h %s)" % self.iterm(d + 1)
        if r < 0.62:
            return "(v %s)" % self.lterm(d + 1)
        if r < 0.9:
            return self.lite(d)
        t = self.lterm(d + 1)
        return "(ite ((_ is cons) %s) (tl %s) %s)" % (t, t, self.lterm(d + 1))

    def cterm(self, d):
        r = self.rng.random()
        if d >= self.MAX_DEPTH or r < 0.45:
            return self.pick("c1", "c2", "red", "green", "blue")
        if r < 0.65:
            return "(g %s)" % self.iterm(d + 1)
        if r < 0.72:
            return "(pc p)"
        return "(ite %s %s %s)" % (self.cond(d + 1), self.cterm(d + 1), self.cterm(d + 1))

    def uterm(self, d):
        r = self.rng.random()
        if d >= self.MAX_DEPTH or r < 0.5:
            return self.pick("u1", "u2")
        if r < 0.7:
            return "(w %s)" % self.iterm(d + 1)
        return "(ite %s %s %s)" % (self.cond(d + 1), self.uterm(d + 1), self.uterm(d + 1))

    def atom(self):
        r = self.rng.random()
        if r < 0.22:
            return "((_ is %s) %s)" % (self.pick("cons", "nil"), self.lite(0))
        if r < 0.3:
            return "(< 0 (ite ((_ is cons) %s) 1 0))" % self.lite(0)
        if r < 0.45:
            return "(%s %s %s)" % (self.pick("=", "=", "distinct"), self.lite(0), self.lterm(0))
        if r < 0.52:
            return "(%s %s %s)" % (self.pick("=", "distinct"), self.lite(0), self.lite(0))
        if r < 0.58:
            return "(%s (k %s) %s)" % (self.pick("=", "<", "distinct"), self.lite(0), self.iterm(1))
        if r < 0.63:
            return "(%s (v %s) %s)" % (self.pick("=", "distinct"), self.lite(0), self.lterm(1))
        if r < 0.68:
            return "(= %s (cons %s %s))" % (self.lterm(1), self.iterm(1), self.lite(1))
        if r < 0.74:
            return "((_ is %s) %s)" % (self.pick("red", "green", "blue"), self.cterm(0))
        if r < 0.8:
            return "(%s %s %s)" % (self.pick("=", "distinct"), self.cterm(0), self.cterm(0))
        if r < 0.85:
            return "(%s %s %s)" % (self.pick("=", "distinct"), self.uterm(0), self.uterm(0))
        if r < 0.9:
            return "(= p (mk %s %s))" % (self.iterm(1), self.cterm(1))
        return "(%s %s %s)" % (self.pick("=", "<", "<=", "distinct"), self.iterm(0), self.iterm(0))

    def form(self, d=0):
        r = self.rng.random()
        if d >= 2 or r < 0.6:
            return self.atom()
        if r < 0.75:
            return "(or %s %s)" % (self.form(d + 1), self.form(d + 1))
        if r < 0.85:
            return "(not %s)" % self.form(d + 1)
        return "(and %s %s)" % (self.form(d + 1), self.form(d + 1))

    def quantified(self):
        """One quantified assertion over Int whose body holds datatype terms —
        a tester / equality over an `ite`, or a term only the body names (a
        selector chain of a constant, a constructor over the bound variable)."""
        body = Gen(self.rng, bound="n")
        if self.V1:
            inner = "((_ is %s) %s)" % (self.pick("cons", "nil"), body.lite(1))
            if self.rng.random() < 0.5:
                inner = "(%s %s %s)" % (self.pick("=", "distinct"), body.lite(1), body.lterm(1))
            return "(forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) %s))" % inner
        r = self.rng.random()
        if r < 0.35:
            inner = "((_ is %s) %s)" % (self.pick("cons", "nil"), body.lite(1))
        elif r < 0.6:
            inner = "(%s %s %s)" % (self.pick("=", "distinct"), body.lite(1), body.lterm(1))
        elif r < 0.8:
            t = self.pick("l1", "l2", "l3")
            inner = "(=> ((_ is cons) %s) ((_ is %s) (tl %s)))" % (t, self.pick("cons", "nil"), t)
        else:
            inner = "(distinct (cons n %s) %s)" % (body.lterm(1), body.lterm(1))
        q = self.rng.random()
        if q < 0.5:
            return "(forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) %s))" % inner
        if q < 0.8:
            return "(forall ((n Int)) (=> (> n %s) %s))" % (self.pick("0", "1", "x"), inner)
        return "(exists ((n Int)) (and (<= 0 n) %s))" % inner

    def script(self):
        lines = [HEAD.rstrip("\n")]
        for _ in range(self.rng.randint(2, 4)):
            lines.append("(assert %s)" % self.form())
        if self.rng.random() < (0.2 if self.V1 else 0.35):
            lines.append("(assert %s)" % self.quantified())
        depth = 0
        for i in range(self.rng.randint(1, 3)):
            if i > 0 and depth > 0 and self.rng.random() < 0.5:
                lines.append("(pop 1)")
                depth -= 1
            if self.rng.random() < 0.6:
                lines.append("(push 1)")
                depth += 1
                for _ in range(self.rng.randint(1, 2)):
                    lines.append("(assert %s)" % self.form())
            lines += ["(check-sat)", "(get-model)"]
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
