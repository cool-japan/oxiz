#!/usr/bin/env python3
"""gen_qmbqi.py - the seeded, deterministic width-7/8 quantified/ground pair
corpus decision (42)(a) asks for (the regenerated `qmbqi120`).

    gen_qmbqi.py --seed N --count 120 --out <dir>

Writes q0000.smt2 .. (quantified) and g0000.smt2 .. (the same script with the
forall written out over every point of its index sort), plus MANIFEST.txt with
the seed, the generator's own sha256, and one line per file (name, sha256,
width, arrays, ground assertions, body shape).  Same seed, same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import os

from gen_core import make_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="7,8")
    ap.add_argument("--no-ite", action="store_true")
    ap.add_argument("--array-eq", action="store_true",
                    help="prefix every script with one ground array equality (= aK <store/const term>)")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    os.makedirs(args.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    gen_hash = hashlib.sha256()
    for name in ("gen_core.py", os.path.basename(__file__)):
        with open(os.path.join(here, name), "rb") as fh:
            gen_hash.update(fh.read())
    lines = [
        "seed %d" % args.seed,
        "count %d" % args.count,
        "widths %s" % args.widths,
        "ite %s" % ("no" if args.no_ite else "yes"),
        "array-eq %s" % ("yes" if args.array_eq else "no"),
        "generator-sha256 %s" % gen_hash.hexdigest(),
    ]
    for k in range(args.count):
        q_text, g_text, meta = make_pair(args.seed, k, widths, not args.no_ite, args.array_eq)
        # A `(set-option :timeout ...)` is never emitted (decision (16)).
        assert ":timeout" not in q_text and ":timeout" not in g_text
        for prefix, text in (("q", q_text), ("g", g_text)):
            name = "%s%04d.smt2" % (prefix, k)
            with open(os.path.join(args.out, name), "w") as fh:
                fh.write(text)
            lines.append("%s %s w=%d arrays=%d ground=%d shape=%s" % (
                name, hashlib.sha256(text.encode()).hexdigest()[:16], meta["width"],
                meta["arrays"], meta["ground"], meta["shape"]))
    with open(os.path.join(args.out, "MANIFEST.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %d pairs to %s" % (args.count, args.out))


if __name__ == "__main__":
    main()
