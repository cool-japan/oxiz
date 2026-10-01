#!/usr/bin/env python3
"""gen_mix.py --seed N --count C --out DIR : recheck 15's mixed Int/Real generator (attacks #P2b-79).

Each script: a logic from {ALL, none, QF_LIRA, QF_UFLIRA, AUFLIRA}; 2-4 Int, 0-2 Real constants,
optionally f: Int->Int, g: Int->Real; linear atoms with integral and rational coefficients (Int
terms coerced with to_real where a Real is needed), strict / non-strict / = / distinct, combined by
or / and / not / ite; push/pop with 1-3 check-sat and a get-model after each.  Every constant is
boxed in [-6, 6] so z3 decides every script.  No :timeout is emitted.
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
        text = one(rng)
        assert ":timeout" not in text
        with open(os.path.join(a.out, "m%05d.smt2" % n), "w") as fh:
            fh.write(text)

def one(rng):
    logic = rng.choice(["ALL", None, None, "QF_LIRA", "QF_UFLIRA", "AUFLIRA", "ALL"])
    ni = rng.randint(2, 4); nr = rng.randint(0, 2)
    ints = ["x%d" % i for i in range(ni)]; reals = ["r%d" % i for i in range(nr)]
    use_f = rng.random() < 0.35; use_g = rng.random() < 0.2
    quant = logic in (None, "ALL", "AUFLIRA") and rng.random() < 0.15
    lines = []
    if logic: lines.append("(set-logic %s)" % logic)
    for v in ints: lines.append("(declare-const %s Int)" % v)
    for v in reals: lines.append("(declare-const %s Real)" % v)
    if use_f: lines.append("(declare-fun f (Int) Int)")
    if use_g: lines.append("(declare-fun g (Int) Real)")
    for v in ints: lines.append("(assert (and (>= %s (- 6)) (<= %s 6)))" % (v, v))
    for v in reals: lines.append("(assert (and (>= %s (- 6.0)) (<= %s 6.0)))" % (v, v))
    def iterm():
        r = rng.random()
        if use_f and r < 0.2: return "(f %s)" % rng.choice(ints + [str(rng.randint(-2, 2))])
        if r < 0.3 and len(ints) > 1: return "(ite %s %s %s)" % (atom(depth=1, allow_ite=False), rng.choice(ints), rng.choice(ints))
        return rng.choice(ints)
    def icoef():
        c = rng.choice([1, 1, 2, 3, -1, -2])
        return str(c) if c >= 0 else "(- %d)" % -c
    def rcoef():
        return rng.choice(["0.5", "1.5", "2.0", "(- 0.5)", "3.0", "0.25", "(- 1.5)"])
    def ilin():
        k = rng.randint(1, 3)
        parts = ["(* %s %s)" % (icoef(), iterm()) for _ in range(k)]
        return parts[0] if k == 1 else "(+ %s)" % " ".join(parts)
    def rlin():
        k = rng.randint(1, 3); parts = []
        for _ in range(k):
            if reals and rng.random() < 0.4: parts.append("(* %s %s)" % (rcoef(), rng.choice(reals)))
            elif use_g and rng.random() < 0.2: parts.append("(* %s (g %s))" % (rcoef(), rng.choice(ints)))
            else: parts.append("(* %s (to_real %s))" % (rcoef(), iterm()))
        return parts[0] if k == 1 else "(+ %s)" % " ".join(parts)
    def iconst():
        c = rng.randint(-5, 5); return str(c) if c >= 0 else "(- %d)" % -c
    def rconst():
        c = rng.choice(["0.5", "1.5", "2.0", "(- 1.5)", "2.25", "0.0", "(- 2.5)", "3.0"]); return c
    def atom(depth=0, allow_ite=True):
        op = rng.choice(["<", "<=", "=", ">", ">=", "distinct", "<", ">"])
        if rng.random() < 0.55:
            lhs = ilin() if allow_ite else rng.choice(ints)
            rhs = rng.choice([iconst(), ilin() if allow_ite else rng.choice(ints)])
        else:
            lhs = rlin() if allow_ite else "(to_real %s)" % rng.choice(ints)
            rhs = rng.choice([rconst(), rlin() if allow_ite else rconst()])
        return "(%s %s %s)" % (op, lhs, rhs)
    def form(d=0):
        r = rng.random()
        if d >= 2 or r < 0.5: return atom()
        if r < 0.7: return "(or %s %s)" % (form(d + 1), form(d + 1))
        if r < 0.8: return "(not %s)" % form(d + 1)
        if r < 0.9: return "(and %s %s)" % (form(d + 1), form(d + 1))
        return "(=> %s %s)" % (form(d + 1), form(d + 1))
    if quant:
        v = rng.choice(ints)
        if use_f:
            lines.append("(assert (forall ((z Int)) (=> (and (>= z (- 3)) (<= z 3)) (and (>= (f z) (- 6)) (<= (f z) 6)))))")
        lines.append("(assert (forall ((z Int)) (=> (%s z %s) (%s (* 2 z) %s))))" % (rng.choice(["<", ">", "<=", ">="]), v, rng.choice(["distinct", "<", ">"]), rng.choice(ints)))
    for _ in range(rng.randint(1, 3)):
        lines.append("(assert %s)" % form())
    nchk = rng.randint(1, 3); depth = 0
    for k in range(nchk):
        if k > 0 and rng.random() < 0.5 and depth > 0:
            lines.append("(pop 1)"); depth -= 1
        if rng.random() < 0.6:
            lines.append("(push 1)"); depth += 1
            for _ in range(rng.randint(1, 2)):
                lines.append("(assert %s)" % form())
        lines.append("(check-sat)")
        lines.append("(get-model)")
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    main()
