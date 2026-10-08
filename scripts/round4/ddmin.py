#!/usr/bin/env python3
"""ddmin.py FILE OUT 'PROP' [--probes tree,head] : greedy line reducer (recheck 15's ddmin2.py, moved
in-tree by re-fix pass 16 with the probes resolved through `common.probe_path`).

Deletes one line at a time (declarations included; a still-used declaration makes the script error,
so PROP rejects that trial) while PROP stays true.  PROP is a Python expression over `t` and `h`,
the verdict lists of the two probes named by --probes (default `tree,head`), e.g.
"t[-1] == 'unknown' and h[-1] == 'unsat'".  Every probe run is capped at 30 s.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402


def run(probe, text):
    out = subprocess.run(["timeout", "30", c.probe_path(probe)], input=text,
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip() in ("sat", "unsat", "unknown")] or ["none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("prop")
    ap.add_argument("--probes", default="probe_tree,probe_head")
    a = ap.parse_args()
    first, second = a.probes.split(",")

    def holds(text):
        t, h = run(first, text), run(second, text)
        try:
            return bool(eval(a.prop, {}, {"t": t, "h": h}))
        except Exception:  # noqa: BLE001
            return False

    lines = open(a.src).read().splitlines()
    if not holds("\n".join(lines)):
        sys.exit("the property does not hold on the input")
    changed = True
    while changed:
        changed = False
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("(check-sat"):
                continue
            trial = lines[:i] + lines[i + 1:]
            if holds("\n".join(trial)):
                lines = trial
                changed = True
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
