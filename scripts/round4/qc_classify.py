#!/usr/bin/env python3
"""qc_classify.py DIR - bucket fuzz_qc's falsifying cases by the completion's decline rule."""
import glob, os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc_eval as qe

def classify(active):
    text = " ".join(qe.c.show(a) for a in active)
    has_forall = "(forall" in text
    negated_exists = bool(re.search(r"\(not \(exists", text))
    pos_exists = "(exists" in text and not negated_exists or text.count("(exists") > text.count("(not (exists")
    bound_arrays = set(re.findall(r"\(select (a\d) i\)", text))
    uf = "(f " in text
    rules = []
    if not has_forall:
        rules.append("no_universal")
    if negated_exists:
        rules.append("negated_exists")
    if len(bound_arrays) > 3:
        rules.append("gt3_arrays_under_binder")
    if not rules:
        rules.append("CLAIMED_HANDLED" + ("+uf" if uf else "") + ("+exists" if pos_exists else ""))
    return rules

import argparse
_ap = argparse.ArgumentParser(description="bucket fuzz_qc's kept falsifying cases (DIR/*.smt2 + .out) by shape")
_ap.add_argument("dir")
d = _ap.parse_args().dir
buckets = {}
cases = []
for smt in sorted(glob.glob(os.path.join(d, "*.smt2"))):
    out = open(smt[:-5] + ".out").read()
    text = open(smt).read()
    res = qe.judge(text, out)
    checks = [p for k, p in qe.walk(qe.commands(text)) if k == "check"]
    for ci in res["falsifying_checks"]:
        rules = classify(checks[ci])
        key = "+".join(rules)
        buckets[key] = buckets.get(key, 0) + 1
        cases.append((os.path.basename(smt), ci, key))
print(json.dumps(buckets, sort_keys=True))
for c in cases:
    if "CLAIMED" in c[2]:
        print("CLAIMED", c)
