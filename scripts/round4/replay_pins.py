#!/usr/bin/env python3
"""replay_pins.py - the in-tree pins of re-fix pass 12, replayed against any
probe through the SAME solver-free exact evaluation score.py uses.

    replay_pins.py --probe probe_iso [--env OXIZ_MUT_X=1]

Used for the mutation table (a mutation lives only in the isolated copy
behind `probe_iso`, switched on by an environment variable).  Each case is
(name, script, expected verdict, model check): a `sat` case's published model
is evaluated in Python over every point of its index sort; an `unsat` case
must answer `unsat`.  Prints one line per case (OK / RED with the reason).
"""
from __future__ import annotations

import argparse
import os
import subprocess

import common as c

K0 = "((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0)"
HDR = "(set-logic ALL)\n(set-option :produce-models true)\n"
A7 = "(declare-const a (Array (_ BitVec 7) (_ BitVec 1)))\n"

CASES = [
    ("pass11 three-line (inverted)", HDR + A7 + "(assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n", "sat"),
    ("pass11 Bool element (inverted)", HDR + "(declare-const a (Array (_ BitVec 7) Bool))\n(declare-const p Bool)\n"
     "(assert (or p (select a #b0000010)))\n(assert (forall ((i (_ BitVec 7))) (select a i)))\n", "sat"),
    ("pass6 six-line (inverted)", HDR + "(declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n"
     "(declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n(assert (= (select a1 (_ bv3 7)) #b0))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select a0 i) (bvxor (select a1 (_ bv3 7)) #b1))))\n", "sat"),
    ("pass11 one point off (inverted, (41))", HDR + A7 + "(declare-const p Bool)\n(assert (or p (= (select a #b0000010) #b1)))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select a i) (ite (= i #b0000000) #b0 #b1))))\n", "sat"),
    ("pass11 guarded (inverted, (41))", HDR + A7 + "(assert (= (select a #b0000101) #b0))\n"
     "(assert (forall ((i (_ BitVec 7))) (=> (bvult i #b0000100) (= (select a i) #b1))))\n", "sat"),
    ("pass10 two arrays one binder", HDR + A7 + "(declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select a i) (bvnot (select b i)))))\n(assert (= (select a #b0000000) #b1))\n", "sat"),
    ("pass10 M11 wrong-at-a-point #1", HDR + A7 + "(assert (= (select a #b0000001) #b0))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select a i) #b1)))\n", "unsat"),
    ("pass10 M11 wrong-at-a-point #2", HDR + A7 + "(declare-const d (_ BitVec 1))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select (store a #b0000010 #b0) i) #b1)))\n", "unsat"),
    ("pass12 pinned contradicted #1", HDR + A7 + "(assert (= (select a #b0000000) #b1))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select a i) (ite (= i #b0000000) #b0 #b1))))\n", "unsat"),
    ("pass12 negated universal", HDR + A7 + "(assert (not (forall ((i (_ BitVec 7))) (= (select a i) #b1))))\n"
     "(assert (= (select a #b0000000) #b1))\n", "sat"),
    ("P2b-60 wsat1", HDR + "(declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n(assert (= a1 (store " + K0 + " #b0000000 #b1)))\n"
     "(assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))\n", "unsat"),
    # Read lazily (decision (69)(11)): the corpus lives in the scratch, and
    # `--help` must not need it.
    ("qeq120/q0033 (saturation completion)", lambda: open(os.path.join(c.SCRATCH,
     "corpus/named/qeq_q0033.smt2")).read().replace("(check-sat)\n", "").replace("(set-option :produce-models true)\n", "")
     .replace("(set-logic ALL)\n", HDR), "sat"),
    ("P2b-60 wsat6 (Int)", HDR + "(declare-const a2 (Array Int Int))\n(assert (= a2 (store ((as const (Array Int Int)) 0) 0 1)))\n"
     "(assert (forall ((i Int)) (= (select a2 i) 1)))\n", "unsat"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--cap", type=float, default=60.0)
    args = ap.parse_args()
    env = dict(os.environ)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    red = 0
    for name, body, expected in CASES:
        if callable(body):
            body = body()
        text = body + "(check-sat)\n(get-model)\n"
        proc = subprocess.run(["timeout", str(args.cap), c.probe_path(args.probe)], input=text,
                              capture_output=True, text=True, env=env)
        verdicts, models, _ = c.parse_response(proc.stdout)
        got = verdicts[-1] if verdicts else ("TIMEOUT" if proc.returncode == 124 else "none")
        why = ""
        if got != expected:
            why = "verdict %s, expected %s" % (got, expected)
        elif got == "sat":
            script = c.Script(text)
            values = c.model_values(models[-1], script.decls) if models else {}
            ev = script.eval_model(values)
            if ev is not True:
                why = "published model evaluates %s" % ("FALSE" if ev is False else "n/a")
        red += bool(why)
        print("%-40s %s %s" % (name, "RED" if why else "OK", why))
    print("probe=%s env=%s red=%d of %d" % (args.probe, args.env, red, len(CASES)))


if __name__ == "__main__":
    main()
