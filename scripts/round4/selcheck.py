#!/usr/bin/env python3
"""selcheck.py PROBE SCRIPT... (re-fix pass 18) : for every check whose printed model zjudge calls falsifying, ask z3 whether the
model SATISFIES the assertions for some value of the terms it leaves unspecified (a selector applied to the wrong
constructor, e.g. `(hd nil)`): replay with the positive conjunction.  sat = the model is consistent (the
'falsification' is the judge choosing the unspecified selector's value), unsat = the model is false whatever the
unspecified terms are (a real falsification)."""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c, zjudge as z
probe = sys.argv[1]
for path in sys.argv[2:]:
    text = open(path).read()
    r = subprocess.run(["timeout", "60", c.probe_path(probe), path], capture_output=True, text=True)
    j = z.judge(text, r.stdout, 10)
    fals = [e[1] for e in j["events"] if e[0] == "FALSIFYING"]
    if not fals:
        print(os.path.basename(path), "no falsifying check", j["per_check"]); continue
    # rebuild per-check models: walk commands like zjudge
    cmds = [x for x in c.parse_all(text) if isinstance(x, list) and x]
    outs = [l for l in r.stdout.splitlines()]
    resp = c.parse_all(r.stdout)
    # zjudge exposes the replay only internally; re-run its loop with the positive goal
    items, scopes, ci, ri = [], [[]], -1, 0
    res = [x for x in resp]
    k = 0
    for cmd in cmds:
        h = cmd[0]
        if h == "push":
            scopes.append([])
        elif h == "pop":
            scopes.pop()
        elif h in ("declare-sort", "define-sort", "declare-datatypes", "declare-datatype", "declare-const", "declare-fun", "define-fun"):
            scopes[-1].append((h, cmd))
        elif h == "assert":
            scopes[-1].append(("assert", cmd))
        elif h == "check-sat":
            ci += 1; verdict = res[k]; k += 1
        elif h == "get-model":
            model = res[k]; k += 1
            if ci not in fals or not isinstance(model, list) or not model or model[0] != "model":
                continue
            defs = {}
            for d in model[1:]:
                if isinstance(d, list) and d and d[0] == "define-fun":
                    defs[d[1]] = d
            allitems = [it for s in scopes for it in s]
            asserts = [c.show(e[1]) for kd, e in allitems if kd == "assert"]
            goal = "(assert (and true %s))" % " ".join(asserts)
            txt, missing, _ = z.replay_text([(kd, e) for kd, e in allitems if kd != "assert"], defs, True, goal)
            zv = z.z3_run(txt, 20)
            print(os.path.basename(path), "check", ci, "positive replay:", zv, "(sat = consistent; unsat = REAL falsification)")
        elif h in ("get-value", "get-info", "get-option", "echo", "get-assignment", "get-unsat-core", "get-assertions", "get-proof"):
            k += 1
