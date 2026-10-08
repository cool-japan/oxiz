#!/usr/bin/env python3
"""cmpcamp.py OLD NEW [--names] : per-check comparison of two camp14.py results.jsonl runs (old -> new).

A check's class in one run is its zjudge per_check value ("sat/ok", "sat/falsifying", "sat/withheld",
"sat/unresolved", "unsat", "unknown", ...), "WRONG:<class>" where the judge flagged a wrong verdict,
and, for a check the run did not answer, "TIMEOUT" (the script timed out and the check lies beyond
its judged prefix), "PANIC", "KILLED" or "none".  A timed-out script's answered prefix is compared
check by check like any other; its checks are marked "(prefix)" in the listings, because a prefix
is a different script from the whole one (Context::execute_script parses the whole text first, so
the terms of later commands are interned first and a check's trajectory can move; README.md,
"camp14 and prefix judging").

Counted (each listed with its script#check and both classes):
  LOST            old decided the check correctly (sat*/unsat, not WRONG) and new did not: unknown,
                  TIMEOUT (beyond the judged prefix), none, PANIC, or a WRONG verdict.
  GAINED          new decided correctly where old did not.
  MODEL_REGRESSED old printed a model the judge confirmed (sat/ok) and new's is withheld,
                  falsifying, unresolved or unjudged (each sub-counted).
  MODEL_IMPROVED  old's model was falsifying / withheld / unresolved and new's is sat/ok.
  NEW_WRONG       new gave a wrong verdict.
Then every class change with its count.  (Recheck 15's tool, moved in-tree by re-fix pass 16;
timeout-aware LOST and the model counters by re-fix pass 17, decision (79)(e).)"""
import json
import sys
from collections import Counter

MODEL_BAD = ("sat/withheld", "sat/falsifying", "sat/unresolved", "sat/falsifying_pin",
             "sat/uc_unresolved", "sat")


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["script"]] = rec
    return out


def classes(rec):
    """(per-check classes, the class of a check the run did not answer, prefix-judged?)."""
    if rec is None:
        return {}, "missing", False
    pc = {int(k): v for k, v in rec.get("per_check", {}).items()}
    for ev in rec.get("events", []):
        if ev[0] in ("WRONG_SAT", "WRONG_UNSAT"):
            pc[int(ev[1])] = "WRONG:" + pc.get(int(ev[1]), "?")
        elif ev[0] == "FALSIFYING_PIN":
            pc[int(ev[1])] = "sat/falsifying_pin"
        elif ev[0] == "UC_SAT":
            # A quantified script's refuted `@uc_` model is unresolved (the
            # universe's cardinality), whatever an older zjudge wrote in
            # per_check (adversarial recheck 17's minor 7; re-fix pass 18).
            pc[int(ev[1])] = "sat/uc_unresolved"
    fill = {"timeout": "TIMEOUT", "panic": "PANIC", "killed": "KILLED"}.get(rec.get("status"), "none")
    return pc, fill, "prefix_checks" in rec


def decided(v):
    return (v.startswith("sat") or v == "unsat") and not v.startswith("WRONG")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2 or "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        sys.exit(0 if ("-h" in sys.argv or "--help" in sys.argv) else 2)
    a, b = load(args[0]), load(args[1])
    lost, gained, regressed, improved, new_wrong = [], [], [], [], []
    sub = Counter()
    changes = {}
    for script in sorted(set(a) | set(b)):
        ca, fa, xa = classes(a.get(script))
        cb, fb, xb = classes(b.get(script))
        n = max([max(c) + 1 for c in (ca, cb) if c] or [0])
        for ci in range(n):
            va, vb = ca.get(ci, fa), cb.get(ci, fb)
            tag = "%s#%d%s" % (script, ci, " (prefix)" if (xa or xb) else "")
            row = (tag, va, vb)
            if decided(va) and not decided(vb):
                lost.append(row)
            if decided(vb) and not decided(va):
                gained.append(row)
            if va == "sat/ok" and vb in MODEL_BAD:
                regressed.append(row)
                sub[vb] += 1
            if vb == "sat/ok" and va in MODEL_BAD:
                improved.append(row)
            if vb.startswith("WRONG"):
                new_wrong.append(row)
            if va != vb:
                changes.setdefault((va, vb), []).append(tag)
    for name, rows in (("LOST (old decided correctly, new not; timeouts included)", lost),
                       ("GAINED", gained),
                       ("MODEL_REGRESSED (old sat/ok, new not)", regressed),
                       ("MODEL_IMPROVED (new sat/ok, old not)", improved),
                       ("NEW_WRONG", new_wrong)):
        print("%s: %d" % (name, len(rows)))
        if name.startswith("MODEL_REGRESSED") and sub:
            print("   by new class: %s" % ", ".join("%s %d" % kv for kv in sorted(sub.items())))
        if name.startswith("GAINED") or name.startswith("MODEL_IMPROVED"):
            if "--names" not in sys.argv:
                continue
        for tag, va, vb in rows:
            print("   %-28s %s -> %s" % (tag, va, vb))
    for k, v in sorted(changes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print("%-48s %5d  %s" % ("%s -> %s" % k, len(v), " ".join(v[:8])))


if __name__ == "__main__":
    main()
