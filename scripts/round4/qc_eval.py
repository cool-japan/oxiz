"""qc_eval.py - exact evaluation for fuzz_qc.py (recheck13).

Extends the shared tools/common.py compiler (decision (50) guard kept: quantifiers
over more than 2^16 points raise TooBig -> unresolved) with applications of
uninterpreted functions whose interpretation is a model's `define-fun` with
parameters.  Only ever evaluates source the Compiler emitted (see common.py).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402


class FCompiler(c.Compiler):
    def __init__(self, funs):
        super().__init__()
        self.funs = funs  # name -> (pid, range sort)

    def compile(self, e, env):
        if isinstance(e, list) and e and isinstance(e[0], str) and e[0] in self.funs and e[0] not in env:
            pid, rsort = self.funs[e[0]]
            args = [self.compile(a, env)[0] for a in e[1:]]
            return "%s(%s)" % (pid, ", ".join(args)), rsort
        return super().compile(e, env)


def commands(text):
    return [cmd for cmd in c.parse_all(text) if isinstance(cmd, list) and cmd]


def decls_of(cmds):
    consts, funs = [], []
    for cmd in cmds:
        if cmd[0] == "declare-const":
            consts.append((cmd[1], c.parse_sort(cmd[2])))
        elif cmd[0] == "declare-fun":
            if cmd[2] == []:
                consts.append((cmd[1], c.parse_sort(cmd[3])))
            else:
                funs.append((cmd[1], [c.parse_sort(s) for s in cmd[2]], c.parse_sort(cmd[3])))
    return consts, funs


def walk(cmds):
    """Yield (kind, payload): ('check', active assertions) / ('get-model', None)."""
    stack = [[]]
    for cmd in cmds:
        h = cmd[0]
        if h == "assert":
            stack[-1].append(cmd[1])
        elif h == "push":
            for _ in range(int(cmd[1]) if len(cmd) > 1 else 1):
                stack.append([])
        elif h == "pop":
            for _ in range(int(cmd[1]) if len(cmd) > 1 else 1):
                stack.pop()
        elif h == "check-sat":
            yield "check", [a for frame in stack for a in frame]
        elif h == "get-model":
            yield "get-model", None


def responses(stdout):
    try:
        return c.parse_all(stdout)
    except ValueError:
        return None


def eval_model(consts, funs, model_items, assertions):
    """True / False, or None when unresolved."""
    defs = {}
    for d in model_items:
        if isinstance(d, list) and len(d) == 5 and d[0] == "define-fun":
            defs[d[1]] = d
    try:
        # function interpretations first
        comp = FCompiler({})
        fenv = {}
        fglobals = dict(c.EVAL_GLOBALS)
        for name, dom, rng in funs:
            d = defs.get(name)
            if d is None:
                return None
            params = d[2]
            penv = {}
            pids = []
            for p in params:
                pid = comp.ident(p[0])
                penv[p[0]] = (pid, c.parse_sort(p[1]))
                pids.append(pid)
            src, _ = comp.compile(d[4], penv)
            fn = eval("lambda %s: %s" % (", ".join(pids), src), dict(fglobals))
            fid = comp.ident(name)
            fglobals[fid] = fn
            fenv[name] = (fid, rng)
        comp2 = FCompiler(fenv)
        comp2.fresh = comp.fresh + 1000
        env = {}
        params = []
        values = []
        for name, sort in consts:
            d = defs.get(name)
            if d is None:
                return None
            v = c.const_value(d[4], sort)
            if v is None:
                return None
            pid = comp2.ident(name)
            env[name] = (pid, sort)
            params.append(pid)
            values.append(v)
        parts = [comp2.compile(a, env)[0] for a in assertions]
        body = " and ".join("(%s)" % p for p in parts) if parts else "True"
        src = "lambda %s: %s" % (", ".join(params), body) if params else "lambda: %s" % body
        fn = eval(src, fglobals)
        return bool(fn(*values))
    except (c.TooBig, c.CompileError, ValueError, KeyError, IndexError, TypeError):
        return None


def judge(text, stdout):
    cmds = commands(text)
    consts, funs = decls_of(cmds)
    out = {"checks": 0, "sat": 0, "unsat": 0, "unknown": 0, "falsifying": 0,
           "model_unresolved": 0, "withheld": 0, "unsat_checks": [], "falsifying_checks": []}
    resp = responses(stdout)
    if resp is None:
        out["model_unresolved"] += 1
        return out
    pos = 0
    last = None
    check_index = -1
    for kind, payload in walk(cmds):
        if pos >= len(resp):
            break
        item = resp[pos]
        pos += 1
        if kind == "check":
            check_index += 1
            out["checks"] += 1
            v = item if isinstance(item, str) else "error"
            last = (v, payload, check_index)
            if v in ("sat", "unsat", "unknown"):
                out[v] += 1
            if v == "unsat":
                out["unsat_checks"].append(check_index)
        else:
            if last and last[0] == "sat":
                if isinstance(item, list) and item and item[0] == "model":
                    ok = eval_model(consts, funs, item[1:], last[1])
                    if ok is False:
                        out["falsifying"] += 1
                        out["falsifying_checks"].append(last[2])
                    elif ok is None:
                        out["model_unresolved"] += 1
                elif (isinstance(item, list) and item and item[0] == "error"
                      and "model not certified" in " ".join(str(x) for x in item[1:])):
                    # decision (54)(i): the model was withheld, not printed
                    out["withheld"] += 1
                else:
                    out["model_unresolved"] += 1
    return out


def _subst(e, name, val):
    if isinstance(e, str):
        return val if e == name else e
    if e and e[0] in ("forall", "exists") and any(b[0] == name for b in e[1]):
        return e
    return [_subst(x, name, val) for x in e]


def _expand(e):
    if isinstance(e, str):
        return e
    if e and e[0] in ("forall", "exists") and len(e[1]) == 1:
        name, sort = e[1][0][0], c.parse_sort(e[1][0][1])
        body = _expand(e[2])
        if sort[0] != "BV" or sort[1] > 10:
            raise c.TooBig("twin too big")
        pts = ["(_ bv%d %d)" % (v, sort[1]) for v in range(1 << sort[1])]
        parts = [_subst(body, name, c.parse_all(p)[0]) for p in pts]
        return ["and" if e[0] == "forall" else "or"] + parts
    return [_expand(x) for x in e]


def ground_twin(text):
    out = []
    for cmd in commands(text):
        if cmd[0] == "assert":
            cmd = ["assert", _expand(cmd[1])]
        out.append(c.show(cmd))
    return "\n".join(out) + "\n"


def judge_twin(text, twin_stdout, unsat_checks):
    cmds = commands(text)
    consts, funs = decls_of(cmds)
    res = {"wrong_unsat": 0, "unresolved": 0, "confirmed": 0}
    resp = responses(twin_stdout)
    if resp is None:
        res["unresolved"] = len(unsat_checks)
        return res
    pos = 0
    last = None
    ci = -1
    for kind, payload in walk(cmds):
        if pos >= len(resp):
            break
        item = resp[pos]
        pos += 1
        if kind == "check":
            ci += 1
            last = (item if isinstance(item, str) else "error", payload, ci)
            if ci in unsat_checks and last[0] == "unsat":
                res["confirmed"] += 1
            elif ci in unsat_checks and last[0] != "sat":
                res["unresolved"] += 1
        elif last and last[2] in unsat_checks and last[0] == "sat":
            ok = eval_model(consts, funs, item[1:], last[1]) if isinstance(item, list) and item and item[0] == "model" else None
            if ok is True:
                res["wrong_unsat"] += 1
            else:
                res["unresolved"] += 1
    return res
