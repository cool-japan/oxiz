#!/usr/bin/env python3
"""gen_guard.py --seed N --count C --out DIR : recheck 15's guarded-universal generator (attacks #P2b-75,
decision (47)'s saturation hook and the published-model net).

Each script: one index sort from {Int, Real, BV16 (unsigned), BV16 (signed), BV32 (mixed)}, constants m, k
of that sort, f: S->Int (or S->S), optionally an array a: (Array S Int); 1-2 universals whose premise is a
random guard (x op t, t in {literal, m, k, (f lit), m+c}, under and/or/not, ite, xor spellings) and whose
body is false / (f x) op c / (= (select a x) c) / f x op f t; 0-3 ground facts at named points.  A
(get-model) after the check-sat.  No :timeout.
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
        open(os.path.join(a.out, "g%05d.smt2" % n), "w").write(t)

def one(rng):
    kind = rng.choice(["int", "int", "real", "bv16u", "bv16s", "bv32"])
    if kind == "int":
        S = "Int"; lit = lambda v: str(v) if v >= 0 else "(- %d)" % -v
        ops = ["<", "<=", ">", ">="]; add = lambda t, c: "(+ %s %s)" % (t, lit(c))
    elif kind == "real":
        S = "Real"; lit = lambda v: ("%d.0" % v) if v >= 0 else "(- %d.0)" % -v
        ops = ["<", "<=", ">", ">="]; add = lambda t, c: "(+ %s %s)" % (t, lit(c))
    else:
        w = 32 if kind == "bv32" else 16
        S = "(_ BitVec %d)" % w
        lit = lambda v: "(_ bv%d %d)" % (v % (1 << w), w)
        if kind == "bv16u": ops = ["bvult", "bvule", "bvugt", "bvuge"]
        elif kind == "bv16s": ops = ["bvslt", "bvsle", "bvsgt", "bvsge"]
        else: ops = ["bvult", "bvule", "bvugt", "bvuge", "bvslt", "bvsle", "bvsgt", "bvsge"]
        add = lambda t, c: "(bvadd %s %s)" % (t, lit(c))
    use_arr = rng.random() < 0.3
    use_ff = rng.random() < 0.3  # f: S -> S
    L = ["(set-logic ALL)"] if rng.random() < 0.7 else []
    L += ["(declare-const m %s)" % S, "(declare-const k %s)" % S]
    L.append("(declare-fun f (%s) %s)" % (S, S if use_ff else "Int"))
    if use_arr: L.append("(declare-const a (Array %s Int))" % S)
    vals = [-3, -1, 0, 1, 2, 5, 7, 8, 100, 32767, 32768, 65535, -32768]
    def ground():
        r = rng.random()
        if r < 0.45: return lit(rng.choice(vals))
        if r < 0.65: return rng.choice(["m", "k"])
        if r < 0.8: return add(rng.choice(["m", "k"]), rng.choice([1, -1, 2]))
        if r < 0.9 and use_ff: return "(f %s)" % lit(rng.choice(vals))
        return lit(rng.choice(vals))
    def cmp(x):
        op = rng.choice(ops + ["=", "distinct"])
        a_, b_ = (x, ground()) if rng.random() < 0.6 else (ground(), x)
        return "(%s %s %s)" % (op, a_, b_)
    def guard(x, d=0):
        r = rng.random()
        if d >= 2 or r < 0.45: return cmp(x)
        if r < 0.6: return "(not %s)" % guard(x, d + 1)
        if r < 0.75: return "(and %s %s)" % (guard(x, d + 1), guard(x, d + 1))
        if r < 0.87: return "(or %s %s)" % (guard(x, d + 1), guard(x, d + 1))
        if r < 0.94: return "(xor %s %s)" % (guard(x, d + 1), guard(x, d + 1))
        return "(ite %s %s %s)" % (guard(x, d + 1), guard(x, d + 1), guard(x, d + 1))
    def body(x):
        r = rng.random()
        ival = lambda: str(rng.randint(0, 3))
        if r < 0.2: return "false"
        if r < 0.45:
            if use_ff: return "(%s (f %s) %s)" % (rng.choice(ops + ["=", "distinct"]), x, ground())
            return "(%s (f %s) %s)" % (rng.choice(["=", "<", ">", "distinct"]), x, ival())
        if r < 0.65 and use_arr: return "(%s (select a %s) %s)" % (rng.choice(["=", "<", ">"]), x, ival())
        if r < 0.8: return "(= (f %s) (f %s))" % (x, ground())
        return "(%s %s %s)" % (rng.choice(ops + ["=", "distinct"]), x, ground())
    for q in range(rng.randint(1, 2)):
        x = "x%d" % q
        form = rng.random()
        if form < 0.7: L.append("(assert (forall ((%s %s)) (=> %s %s)))" % (x, S, guard(x), body(x)))
        elif form < 0.85: L.append("(assert (forall ((%s %s)) (or (not %s) %s)))" % (x, S, guard(x), body(x)))
        else: L.append("(assert (not (exists ((%s %s)) (and %s (not %s)))))" % (x, S, guard(x), body(x)))
    for _ in range(rng.randint(0, 3)):
        r = rng.random()
        if r < 0.3: L.append("(assert (= (f %s) %s))" % (ground(), ground() if use_ff else str(rng.randint(0, 3))))
        elif r < 0.5 and use_arr: L.append("(assert (= (select a %s) %d))" % (ground(), rng.randint(0, 3)))
        elif r < 0.8: L.append("(assert %s)" % cmp(rng.choice(["m", "k"])).replace("x0", "m"))
        else: L.append("(assert (distinct m k))")
    L += ["(check-sat)", "(get-model)"]
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    main()
