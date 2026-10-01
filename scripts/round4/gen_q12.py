#!/usr/bin/env python3
"""gen_q12.py - the width-1/2 paired corpora `q` and `qnoite` (decision
(42)(b)): the same shapes as gen_qmbqi.py at index widths 1 and 2, 400 pairs
each, `qnoite` with every `ite` (array selectors and ite bodies) left out.

At these widths every script is decided by brute force over every
interpretation (common.Script.brute_force), so the oracle needs no solver.

    gen_q12.py --seed N --out <corpus-root>     # writes <root>/q and <root>/qnoite
"""
from __future__ import annotations

import argparse
import subprocess
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    gen = os.path.join(here, "gen_qmbqi.py")
    for name, extra, seed in (("q", [], args.seed), ("qnoite", ["--no-ite"], args.seed + 1)):
        cmd = [sys.executable, gen, "--seed", str(seed), "--count", str(args.count),
               "--widths", "1,2", "--out", os.path.join(args.out, name)] + extra
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
