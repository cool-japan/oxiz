#!/usr/bin/env python3
"""zjudge.py - recheck 14 (moved in-tree by re-fix pass 15, decision (69)(11)): judge a
probe's responses on an SMT-LIB script with z3 as an INDEPENDENT judge.

For every `(check-sat)` of the script:
  * the verdict is compared with z3's verdict on the same active assertions
    (WRONG_SAT: probe sat / z3 unsat; WRONG_UNSAT: probe unsat / z3 sat);
  * a `(get-model)` answered after a `sat` is REPLAYED: every declared symbol is
    replaced by the model's `define-fun`, and z3 decides `(not (and A1 .. An))`
    - `unsat` = the model satisfies every active assertion (OK),
    - `sat`   = the model FALSIFIES its own script,
    - anything else = unresolved;
    a model missing a declared symbol is re-checked as a pin instead (the
    assertions with the printed symbols fixed must stay satisfiable);
  * a `(get-value ...)` answered after a `sat` is replayed pair by pair
    against the same model: each value must be a literal and `(= term value)`
    must be valid under the model (GETVALUE_WRONG / GETVALUE_NONVALUE);
  * `(error "model not certified: ...")` is counted as WITHHELD.

Uninterpreted-sort witnesses `@uc_S_n` are replayed as distinct constants of S
(the universe's cardinality is not replayed, so a universal over a declared
sort can only be judged OK or unresolved, never FALSIFYING - such verdicts are
reported as `uc_unresolved`).  A QUANTIFIER-FREE script is closed once every
symbol is replaced, so there a refuted `@uc_` model is FALSIFYING (re-fix pass
16).

z3 runs as a subprocess under `-T:<cap>` and `-memory:2048` (MB).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402

Z3 = os.environ.get("Z3", "z3")
UC = re.compile(r"@uc_([A-Za-z0-9_.!~$%^&*+\-<>=/?]+?)_([0-9]+)")


def z3_run(text: str, cap: int) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        proc = subprocess.run([Z3, "-T:%d" % cap, "-memory:2048", path],
                              capture_output=True, text=True, timeout=cap + 30)
        out = proc.stdout.strip().splitlines()
        return out[0].strip() if out else "none"
    except subprocess.TimeoutExpired:
        return "timeout"
    finally:
        os.unlink(path)


class State:
    def __init__(self):
        self.frames = [[]]  # each: list of (kind, sexpr)

    def add(self, kind, expr):
        self.frames[-1].append((kind, expr))

    def push(self, n):
        for _ in range(n):
            self.frames.append([])

    def pop(self, n):
        for _ in range(n):
            if len(self.frames) > 1:
                self.frames.pop()

    def items(self):
        return [it for fr in self.frames for it in fr]


def decl_name(kind, expr):
    if kind in ("declare-const", "declare-fun"):
        return expr[1]
    return None


def replace_uc(text: str, ucs: dict) -> str:
    def sub(m):
        name = "uc__%s__%s" % (m.group(1), m.group(2))
        ucs[name] = m.group(1)
        return name
    return UC.sub(sub, text)


def base_prefix(items, logic):
    """Sort / datatype declarations and script-level define-funs."""
    out = []
    if logic:
        out.append("(set-logic ALL)")
    for kind, expr in items:
        if kind in ("declare-sort", "define-sort", "declare-datatypes", "declare-datatype", "define-fun", "define-fun-rec", "define-funs-rec"):
            out.append(c.show(expr))
    return out


def model_defs(model_item):
    defs = {}
    for d in model_item[1:] if model_item and model_item[0] == "model" else model_item:
        if isinstance(d, list) and d and d[0] == "define-fun" and len(d) == 5:
            defs[d[1]] = d
    return defs


def replay_text(items, defs, logic, goal):
    """Every item in script order: sorts / datatypes / script define-funs as
    written, each declared symbol as the model's define-fun (or its
    declaration when the model omits it), then the goal."""
    ucs = {}
    lines = ["(set-logic ALL)"] if logic else []
    missing = []
    body = []
    for kind, expr in items:
        if kind in ("declare-sort", "define-sort", "declare-datatypes", "declare-datatype", "define-fun",
                    "define-fun-rec", "define-funs-rec"):
            body.append(c.show(expr))
            continue
        name = decl_name(kind, expr)
        if name is None:
            continue
        d = defs.get(name)
        if d is None:
            missing.append(name)
            body.append(c.show(expr))
        else:
            body.append(replace_uc(c.show(d), ucs))
    uc_lines = []
    by_sort = {}
    for name, sort in sorted(ucs.items()):
        uc_lines.append("(declare-const %s %s)" % (name, sort))
        by_sort.setdefault(sort, []).append(name)
    for sort, names in sorted(by_sort.items()):
        if len(names) > 1:
            uc_lines.append("(assert (distinct %s))" % " ".join(names))
    goal = replace_uc(goal, ucs)
    # witnesses are declared after the sorts they belong to: put them right
    # before the goal (a witness is only mentioned by model values and goal)
    return "\n".join(lines + body[:0] + [x for x in body if x.startswith("(declare-sort") or x.startswith("(define-sort") or x.startswith("(declare-datatype")] + uc_lines + [x for x in body if not (x.startswith("(declare-sort") or x.startswith("(define-sort") or x.startswith("(declare-datatype"))] + [goal, "(check-sat)"]) + "\n", missing, bool(ucs)


def declared_names(items):
    return {decl_name(k, e) for k, e in items if decl_name(k, e)}


def mentions_declared(expr, names):
    if isinstance(expr, str):
        return expr in names
    return any(mentions_declared(x, names) for x in expr)


def quantified(asserts):
    """Whether any assertion (as printed) holds a binder."""
    return any(re.search(r"\((forall|exists)\b", a) for a in asserts)


def asserts_of(items):
    return [c.show(expr[1]) for kind, expr in items if kind == "assert"]


def judge(script_text: str, stdout: str, cap: int, check_verdicts=True):
    cmds = [x for x in c.parse_all(script_text) if isinstance(x, list) and x]
    try:
        resp = c.parse_all(stdout)
    except ValueError:
        return {"error": "unparsable stdout"}
    st = State()
    logic = None
    out = {"checks": 0, "sat": 0, "unsat": 0, "unknown": 0, "other": 0,
           "wrong_sat": 0, "wrong_unsat": 0, "z3_unknown": 0,
           "models": 0, "model_ok": 0, "falsifying": 0, "model_unresolved": 0, "uc_unresolved": 0,
           "withheld": 0, "incomplete_pin_fail": 0,
           "gv_pairs": 0, "gv_wrong": 0, "gv_nonvalue": 0, "gv_error": 0, "gv_unresolved": 0,
           "events": [], "per_check": {}}
    pos = 0
    last = None  # (verdict, items, check index)
    ci = -1
    for cmd in cmds:
        h = cmd[0]
        if h == "set-logic":
            logic = cmd[1]
            continue
        if h in ("declare-sort", "define-sort", "declare-datatypes", "declare-datatype", "define-fun",
                 "define-fun-rec", "define-funs-rec", "declare-const", "declare-fun"):
            st.add(h, cmd)
            continue
        if h == "assert":
            st.add("assert", cmd)
            continue
        if h == "push":
            st.push(int(cmd[1]) if len(cmd) > 1 else 1)
            continue
        if h == "pop":
            st.pop(int(cmd[1]) if len(cmd) > 1 else 1)
            continue
        if h in ("check-sat", "check-sat-assuming"):
            if pos >= len(resp):
                break
            item = resp[pos]
            pos += 1
            ci += 1
            out["checks"] += 1
            v = item if isinstance(item, str) else "other"
            out[v if v in ("sat", "unsat", "unknown") else "other"] += 1
            items = st.items()
            if h == "check-sat-assuming":
                items = items + [("assert", ["assert", a]) for a in cmd[1]]
            last = (v, items, ci)
            out["per_check"][ci] = v
            if check_verdicts and v in ("sat", "unsat"):
                txt = "\n".join((["(set-logic ALL)"] if logic else []) +
                                [c.show(e) for k, e in items if k != "assert"] +
                                ["(assert %s)" % a for a in asserts_of(items)] + ["(check-sat)"]) + "\n"
                zv = z3_run(txt, cap)
                if zv not in ("sat", "unsat"):
                    out["z3_unknown"] += 1
                elif v == "sat" and zv == "unsat":
                    out["wrong_sat"] += 1
                    out["events"].append(("WRONG_SAT", ci))
                elif v == "unsat" and zv == "sat":
                    out["wrong_unsat"] += 1
                    out["events"].append(("WRONG_UNSAT", ci))
            continue
        if h in ("get-model", "get-value", "get-info", "get-option", "echo", "get-assignment",
                 "get-unsat-core", "get-assertions", "get-proof"):
            if pos >= len(resp):
                break
            item = resp[pos]
            pos += 1
            if not last or last[0] != "sat":
                continue
            items = last[1]
            is_err = isinstance(item, list) and item and item[0] == "error"
            if h == "get-model":
                if is_err and "model not certified" in " ".join(str(x) for x in item[1:]):
                    out["withheld"] += 1
                    out["events"].append(("WITHHELD", last[2]))
                    out["per_check"][last[2]] = "sat/withheld"
                    continue
                if not (isinstance(item, list) and item and item[0] == "model"):
                    out["model_unresolved"] += 1
                    out["events"].append(("MODEL_ERR", last[2], c.show(item) if isinstance(item, list) else item))
                    continue
                out["models"] += 1
                defs = model_defs(item)
                asserts = asserts_of(items)
                goal = "(assert (not (and true %s)))" % " ".join(asserts)
                txt, missing, has_uc = replay_text(items, defs, logic, goal)
                if missing:
                    goal2 = "(assert (and true %s))" % " ".join(asserts)
                    txt2, _, _ = replay_text(items, defs, logic, goal2)
                    zv = z3_run(txt2, cap)
                    if zv == "unsat":
                        out["falsifying"] += 1
                        out["incomplete_pin_fail"] += 1
                        out["events"].append(("FALSIFYING_PIN", last[2], missing))
                    elif zv == "sat":
                        out["model_ok"] += 1
                    else:
                        out["model_unresolved"] += 1
                    continue
                zv = z3_run(txt, cap)
                out["per_check"][last[2]] = "sat/" + {"unsat": "ok", "sat": "falsifying"}.get(zv, "unresolved")
                if zv == "unsat":
                    out["model_ok"] += 1
                elif zv == "sat":
                    # A quantifier-free script is closed once every symbol is
                    # replaced, whatever the universe of a declared sort holds
                    # beyond the witnesses, so a refuted `@uc_` model of one is
                    # FALSIFYING (re-fix pass 16, recheck 15's minor 14); only a
                    # quantifier can see the universe's cardinality.
                    if has_uc and quantified(asserts):
                        out["uc_unresolved"] += 1
                        out["events"].append(("UC_SAT", last[2]))
                        # Unresolved, not falsifying: only a quantifier can see
                        # the universe's cardinality (adversarial recheck 17's
                        # minor 7; re-fix pass 18).
                        out["per_check"][last[2]] = "sat/uc_unresolved"
                    else:
                        out["falsifying"] += 1
                        out["events"].append(("FALSIFYING", last[2]))
                else:
                    out["model_unresolved"] += 1
                last = (last[0], last[1], last[2], defs)
                continue
            if h == "get-value":
                if is_err:
                    if "model not certified" in " ".join(str(x) for x in item[1:]):
                        out["withheld"] += 0
                    else:
                        out["gv_error"] += 1
                        out["events"].append(("GV_ERROR", last[2], c.show(item)))
                    continue
                defs = last[3] if len(last) > 3 else None
                if defs is None:
                    continue
                for pair in item if isinstance(item, list) else []:
                    if not (isinstance(pair, list) and len(pair) == 2):
                        continue
                    out["gv_pairs"] += 1
                    term, value = c.show(pair[0]), c.show(pair[1])
                    if mentions_declared(pair[1], declared_names(items)):
                        out["gv_nonvalue"] += 1
                        out["events"].append(("GV_NONVALUE", last[2], term, value))
                        continue
                    txt, missing, has_uc = replay_text(items, defs, logic, "(assert (not (= %s %s)))" % (term, value))
                    if missing:
                        out["gv_unresolved"] += 1
                        continue
                    zv = z3_run(txt, cap)
                    if zv == "unsat":
                        continue
                    if zv == "sat":
                        if has_uc and quantified(asserts_of(items)):
                            out["gv_unresolved"] += 1
                        else:
                            out["gv_wrong"] += 1
                            out["events"].append(("GV_WRONG", last[2], term, value))
                    else:
                        # z3 rejects `(= t v)` when v is not a value of t's sort
                        out["gv_unresolved"] += 1
                continue
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--probe", default="probe_tree")
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--zcap", type=int, default=20)
    ap.add_argument("--no-verdicts", action="store_true")
    args = ap.parse_args()
    for path in args.scripts:
        text = open(path).read()
        r = c.run_probe(args.probe, path, args.cap)
        if r["status"] != "ok":
            print(json.dumps({"script": path, "status": r["status"]}))
            continue
        res = judge(text, r["stdout"], args.zcap, not args.no_verdicts)
        res["script"] = path
        res["ms"] = r["ms"]
        print(json.dumps(res))


if __name__ == "__main__":
    main()
