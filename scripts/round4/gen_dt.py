#!/usr/bin/env python3
"""gen_dt.py --seed N --count C --out DIR : recheck 15's quantifier-free datatype / uninterpreted-range generator
(attacks #P2b-76 / #P2b-77 / dt_refinement, #P2b-71's minting and uf_consistency's functional collisions).

Sorts: L = nil | cons(hd Int, tl L); C = red | green | blue; P = mk(px Int, pc C); an uninterpreted sort U.
Functions: h: Int -> L, g: Int -> C, k: L -> Int, q: C -> Int, w: Int -> U, v: L -> L.  Constants x y z : Int,
l1 l2 l3 : L, c1 c2 : C, p : P, u1 u2 : U.  Atoms: equalities / disequalities between constructor terms,
constants and applications (depth <= 3), testers, selectors only under their tester (ite), arithmetic bounds.
push/pop, 1-3 check-sat, a get-model after each.  QF only.  No :timeout.
"""
import argparse, os, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    for n in range(a.count):
        t = one(rng)
        assert ":timeout" not in t
        open(os.path.join(a.out, "d%05d.smt2" % n), "w").write(t)

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
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))
"""

def one(rng):
    def iterm(d=0):
        r = rng.random()
        if d >= 2 or r < 0.4: return rng.choice(["x", "y", "z", "0", "1", "2"])
        if r < 0.55: return "(k %s)" % lterm(d + 1)
        if r < 0.65: return "(q %s)" % cterm(d + 1)
        if r < 0.75: return "(px %s)" % "p"
        if r < 0.85: return "(ite ((_ is cons) %s) (hd %s) 0)" % ((lambda t: (t, t))(lterm(d + 1)))
        return "(+ %s %s)" % (iterm(d + 1), rng.choice(["1", "(- 1)", "x"]))
    def lterm(d=0):
        r = rng.random()
        if d >= 3 or r < 0.3: return rng.choice(["l1", "l2", "l3", "nil"])
        if r < 0.55: return "(cons %s %s)" % (iterm(d + 1), lterm(d + 1))
        if r < 0.7: return "(h %s)" % iterm(d + 1)
        if r < 0.8: return "(v %s)" % lterm(d + 1)
        if r < 0.9:
            t = lterm(d + 1); return "(ite ((_ is cons) %s) (tl %s) nil)" % (t, t)
        return rng.choice(["l1", "l2"])
    def cterm(d=0):
        r = rng.random()
        if d >= 2 or r < 0.5: return rng.choice(["c1", "c2", "red", "green", "blue"])
        if r < 0.8: return "(g %s)" % iterm(d + 1)
        return "(pc p)"
    def uterm():
        return rng.choice(["u1", "u2", "(w %s)" % iterm(1)])
    def atom():
        r = rng.random()
        if r < 0.35: op = rng.choice(["=", "=", "distinct"]); return "(%s %s %s)" % (op, lterm(), lterm())
        if r < 0.5: op = rng.choice(["=", "distinct"]); return "(%s %s %s)" % (op, cterm(), cterm())
        if r < 0.6: op = rng.choice(["=", "distinct"]); return "(%s %s %s)" % (op, uterm(), uterm())
        if r < 0.7: return "((_ is %s) %s)" % (rng.choice(["nil", "cons"]), lterm())
        if r < 0.78: return "(= p (mk %s %s))" % (iterm(), cterm())
        return "(%s %s %s)" % (rng.choice(["=", "<", "<=", "distinct"]), iterm(), iterm())
    def form(d=0):
        r = rng.random()
        if d >= 2 or r < 0.6: return atom()
        if r < 0.75: return "(or %s %s)" % (form(d + 1), form(d + 1))
        if r < 0.85: return "(not %s)" % form(d + 1)
        return "(and %s %s)" % (form(d + 1), form(d + 1))
    L = [HEAD.rstrip("\n")]
    for _ in range(rng.randint(2, 5)):
        L.append("(assert %s)" % form())
    depth = 0
    for i in range(rng.randint(1, 3)):
        if i > 0 and depth > 0 and rng.random() < 0.5:
            L.append("(pop 1)"); depth -= 1
        if rng.random() < 0.5:
            L.append("(push 1)"); depth += 1
            for _ in range(rng.randint(1, 2)):
                L.append("(assert %s)" % form())
        L += ["(check-sat)", "(get-model)"]
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    main()
