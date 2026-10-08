#!/usr/bin/env python3
"""Shared machinery for the round-4 pass-9 (re-fix pass 12) tooling.

Everything here is solver-independent:

* an S-expression reader for SMT-LIB scripts and solver responses;
* a *compiler* from the generated fragment (Bool, fixed-width bit-vectors,
  arrays over a bit-vector or Bool index sort, `ite`, `forall`/`exists` over a
  finite sort) into Python source, so a script can be evaluated exactly under a
  given interpretation, or brute-forced over every interpretation when the
  sorts are small;
* the probe runner: one process per script, `timeout`-capped, stdout parsed
  into verdicts and models, stderr's `# elapsed_ms` read back, a panic
  recognised by its exit status / message.

Arrays are Python tuples indexed by the integer value of the index (a Bool
index sort is 0/1), so array equality is extensional equality for free.
Bit-vectors are Python ints in [0, 2^w).  Bool is Python bool.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Memory guard (decision (50)).  RLIMIT_AS / RLIMIT_DATA cannot be lowered on
# macOS (setrlimit raises ValueError "current limit exceeds maximum limit" and
# nothing is enforced), so the guard that actually protects the machine is the
# STRUCTURAL one below: nothing is ever materialised, compiled or enumerated
# over more than MAX_POINTS index points; such a script is 'unresolved' with a
# reason.  The rlimit is still attempted and its outcome recorded.
# ---------------------------------------------------------------------------
MAX_POINTS = 1 << 16
try:
    import resource as _resource
    _resource.setrlimit(_resource.RLIMIT_AS, (4 << 30, _resource.RLIM_INFINITY))
    RLIMIT_STATUS = "RLIMIT_AS=4GiB"
except Exception as _err:  # noqa: BLE001
    RLIMIT_STATUS = "RLIMIT_AS unavailable (%s); structural guard only" % _err


class TooBig(Exception):
    pass


# Where the tools live in the tree (`<repo>/scripts/round4/`), and the
# repository root they measure by default (`--root`).
REPO_ROOT = str(Path(__file__).resolve().parents[2])

# The scratch root: probes, corpora and campaign outputs.  Taken from the
# environment (`OXIZ_ROUND4_SCRATCH`) and defaulting to `<repo>/target/round4`,
# which `cargo clean` owns and git ignores.  No absolute path is written here.
SCRATCH = os.environ.get("OXIZ_ROUND4_SCRATCH") or os.path.join(REPO_ROOT, "target", "round4")
PROBES = os.path.join(SCRATCH, "probes")


def probe_path(name: str) -> str:
    """`probe_head` -> the release binary of that probe crate."""
    if os.path.sep in name:
        return name
    return os.path.join(PROBES, name, "target", "release", name)


# ---------------------------------------------------------------------------
# S-expressions
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r'\s*(?:(;[^\n]*)|(\()|(\))|("(?:[^"]|"")*")|(\|[^|]*\|)|([^\s()";|]+))')


def tokenize(text: str):
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            # trailing whitespace
            if text[pos:].strip() == "":
                return
            raise ValueError(f"cannot tokenize at {pos}: {text[pos:pos+40]!r}")
        pos = m.end()
        if m.group(1):
            continue
        if m.group(2):
            yield "("
        elif m.group(3):
            yield ")"
        elif m.group(4):
            yield ("str", m.group(4))
        elif m.group(5):
            yield m.group(5)[1:-1]
        elif m.group(6):
            yield m.group(6)


def parse_all(text: str) -> list:
    """Every top-level S-expression in `text`, as nested lists of str."""
    out: list = []
    stack: list = []
    for tok in tokenize(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced )")
            done = stack.pop()
            if stack:
                stack[-1].append(done)
            else:
                out.append(done)
        else:
            if isinstance(tok, tuple):
                tok = tok[1]
            if stack:
                stack[-1].append(tok)
            else:
                out.append(tok)
    if stack:
        raise ValueError("unbalanced (")
    return out


def show(expr) -> str:
    if isinstance(expr, list):
        return "(" + " ".join(show(e) for e in expr) + ")"
    return expr


# ---------------------------------------------------------------------------
# Sorts and the compiler
# ---------------------------------------------------------------------------

BOOL = ("Bool",)


def parse_sort(expr):
    if expr == "Bool":
        return BOOL
    if isinstance(expr, list):
        if len(expr) == 3 and expr[0] == "_" and expr[1] == "BitVec":
            return ("BV", int(expr[2]))
        if len(expr) == 3 and expr[0] == "Array":
            return ("Array", parse_sort(expr[1]), parse_sort(expr[2]))
    raise ValueError(f"unsupported sort {show(expr)}")


def domain_size(sort) -> int:
    if sort == BOOL:
        return 2
    if sort[0] == "BV":
        if sort[1] > 16:
            raise TooBig(f"index sort of width {sort[1]} exceeds 2^16 points")
        return 1 << sort[1]
    raise ValueError(f"not a finite index sort: {sort}")


def domain(sort):
    """Every value of a finite sort, in increasing order."""
    if sort == BOOL:
        return [False, True]
    if sort[0] == "BV":
        return list(range(domain_size(sort)))
    if sort[0] == "Array":
        idx = domain_size(sort[1])
        n = len(domain(sort[2])) ** idx if idx <= 64 else None
        if n is None or n > MAX_POINTS:
            raise TooBig(f"array domain {sort} has more than 2^16 values")
        return list(itertools.product(domain(sort[2]), repeat=idx))
    raise ValueError(f"no domain for {sort}")


def _mask(w: int) -> int:
    return (1 << w) - 1


def _store(arr, i, v):
    lst = list(arr)
    lst[int(i)] = v
    return tuple(lst)


def _signed(v, w):
    return v - (1 << w) if v >> (w - 1) else v


class CompileError(Exception):
    pass


class Compiler:
    """Compile an expression of the fragment into Python source.

    `env` maps SMT symbol -> (python identifier, sort)."""

    def __init__(self):
        self.fresh = 0

    def ident(self, name: str) -> str:
        self.fresh += 1
        return "v%d_%s" % (self.fresh, re.sub(r"[^A-Za-z0-9_]", "_", name))

    def compile(self, e, env):
        if isinstance(e, str):
            if e == "true":
                return "True", BOOL
            if e == "false":
                return "False", BOOL
            if e.startswith("#b"):
                return str(int(e[2:], 2)), ("BV", len(e) - 2)
            if e.startswith("#x"):
                return str(int(e[2:], 16)), ("BV", 4 * (len(e) - 2))
            if e in env:
                return env[e]
            raise CompileError(f"unknown symbol {e}")
        if not e:
            raise CompileError("empty application")
        head = e[0]
        if isinstance(head, list):
            # ((as const S) v) or ((_ extract i j) x) ...
            if len(head) == 3 and head[0] == "as" and head[1] == "const":
                sort = parse_sort(head[2])
                val, _ = self.compile(e[1], env)
                return "((%s,)*%d)" % (val, domain_size(sort[1])), sort
            if len(head) == 4 and head[0] == "_" and head[1] == "extract":
                hi, lo = int(head[2]), int(head[3])
                x, s = self.compile(e[1], env)
                return "((%s>>%d)&%d)" % (x, lo, _mask(hi - lo + 1)), ("BV", hi - lo + 1)
            if len(head) == 3 and head[0] == "_" and head[1] in ("zero_extend",):
                k = int(head[2])
                x, s = self.compile(e[1], env)
                return x, ("BV", s[1] + k)
            raise CompileError(f"unsupported head {show(head)}")
        if head == "_" and len(e) == 3 and e[1].startswith("bv"):
            w = int(e[2])
            return str(int(e[1][2:]) & _mask(w)), ("BV", w)
        args = e[1:]
        if head in ("forall", "exists"):
            binders = e[1]
            body = e[2]
            inner = dict(env)
            loops = []
            for b in binders:
                name, sort = b[0], parse_sort(b[1])
                pid = self.ident(name)
                inner[name] = (pid, sort)
                rng = "(False, True)" if sort == BOOL else "range(%d)" % domain_size(sort)
                loops.append("for %s in %s" % (pid, rng))
            prod = 1
            for b in binders:
                prod *= domain_size(parse_sort(b[1]))
            if prod > MAX_POINTS:
                raise TooBig(f"quantifier ranges over {prod} > 2^16 points")
            src, s = self.compile(body, inner)
            fn = "all" if head == "forall" else "any"
            return "%s((%s) %s)" % (fn, src, " ".join(loops)), BOOL
        if head == "!":
            return self.compile(e[1], env)
        if head == "let":
            inner = dict(env)
            binds = []
            for b in e[1]:
                src, s = self.compile(b[1], env)
                pid = self.ident(b[0])
                binds.append((pid, src))
                inner[b[0]] = (pid, s)
            body, s = self.compile(e[2], inner)
            for pid, src in reversed(binds):
                body = "(lambda %s: %s)(%s)" % (pid, body, src)
            return body, s
        cs = [self.compile(a, env) for a in args]
        srcs = [c[0] for c in cs]
        sorts = [c[1] for c in cs]
        if head == "and":
            return "(" + " and ".join(srcs) + ")" if srcs else "True", BOOL
        if head == "or":
            return "(" + " or ".join(srcs) + ")" if srcs else "False", BOOL
        if head == "not":
            return "(not %s)" % srcs[0], BOOL
        if head == "=>":
            acc = srcs[-1]
            for s in reversed(srcs[:-1]):
                acc = "((not %s) or %s)" % (s, acc)
            return acc, BOOL
        if head == "xor":
            acc = srcs[0]
            for s in srcs[1:]:
                acc = "(%s != %s)" % (acc, s)
            return acc, BOOL
        if head == "=":
            parts = ["(%s == %s)" % (srcs[i], srcs[i + 1]) for i in range(len(srcs) - 1)]
            return "(" + " and ".join(parts) + ")", BOOL
        if head == "distinct":
            parts = []
            for i in range(len(srcs)):
                for j in range(i + 1, len(srcs)):
                    parts.append("(%s != %s)" % (srcs[i], srcs[j]))
            return "(" + " and ".join(parts) + ")", BOOL
        if head == "ite":
            return "(%s if %s else %s)" % (srcs[1], srcs[0], srcs[2]), sorts[1]
        if head == "select":
            return "%s[%s]" % (srcs[0], srcs[1]), sorts[0][2]
        if head == "store":
            return "_store(%s, %s, %s)" % (srcs[0], srcs[1], srcs[2]), sorts[0]
        # bit-vectors
        if not sorts or sorts[0][0] != "BV":
            raise CompileError(f"unsupported operator {head}")
        w = sorts[0][1]
        m = _mask(w)
        if head == "bvxor":
            return "(" + " ^ ".join(srcs) + ")", ("BV", w)
        if head == "bvand":
            return "(" + " & ".join(srcs) + ")", ("BV", w)
        if head == "bvor":
            return "(" + " | ".join(srcs) + ")", ("BV", w)
        if head == "bvnot":
            return "(%s ^ %d)" % (srcs[0], m), ("BV", w)
        if head == "bvneg":
            return "((-%s) & %d)" % (srcs[0], m), ("BV", w)
        if head == "bvadd":
            return "((" + " + ".join(srcs) + ") & %d)" % m, ("BV", w)
        if head == "bvsub":
            return "((%s - %s) & %d)" % (srcs[0], srcs[1], m), ("BV", w)
        if head == "bvmul":
            return "((" + " * ".join(srcs) + ") & %d)" % m, ("BV", w)
        if head == "bvult":
            return "(%s < %s)" % (srcs[0], srcs[1]), BOOL
        if head == "bvule":
            return "(%s <= %s)" % (srcs[0], srcs[1]), BOOL
        if head == "bvugt":
            return "(%s > %s)" % (srcs[0], srcs[1]), BOOL
        if head == "bvuge":
            return "(%s >= %s)" % (srcs[0], srcs[1]), BOOL
        if head in ("bvslt", "bvsle", "bvsgt", "bvsge"):
            op = {"bvslt": "<", "bvsle": "<=", "bvsgt": ">", "bvsge": ">="}[head]
            return "(_signed(%s,%d) %s _signed(%s,%d))" % (srcs[0], w, op, srcs[1], w), BOOL
        if head == "concat":
            w2 = sorts[1][1]
            return "((%s << %d) | %s)" % (srcs[0], w2, srcs[1]), ("BV", w + w2)
        raise CompileError(f"unsupported operator {head}")


# `eval` below only ever sees source this Compiler emitted: identifiers are
# sanitised to `v<N>_<alnum>`, numerals come from `int()`, and every other
# token is a fixed operator template.  An SMT symbol it does not know raises
# CompileError instead of reaching the generated text, so no script or solver
# response can inject Python.
EVAL_GLOBALS = {"_store": _store, "_signed": _signed, "__builtins__": {"all": all, "any": any, "range": range, "True": True, "False": False}}


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


class Script:
    """The declarations, assertions and command sequence of a script."""

    def __init__(self, text: str):
        self.text = text
        self.commands = parse_all(text)
        self.decls: list[tuple[str, tuple]] = []
        self.unsupported = False
        self.assertions: list = []
        for c in self.commands:
            if not isinstance(c, list) or not c:
                continue
            try:
                if c[0] == "declare-const":
                    self.decls.append((c[1], parse_sort(c[2])))
                elif c[0] == "declare-fun" and c[2] == []:
                    self.decls.append((c[1], parse_sort(c[3])))
            except ValueError:
                self.unsupported = True
            if c[0] == "assert":
                self.assertions.append(c[1])

    def compile_conjunction(self, assertions=None):
        """A Python function over the declared constants, in declaration
        order, returning the conjunction of the assertions."""
        comp = Compiler()
        env = {}
        params = []
        for name, sort in self.decls:
            pid = comp.ident(name)
            env[name] = (pid, sort)
            params.append(pid)
        parts = [comp.compile(a, env)[0] for a in (self.assertions if assertions is None else assertions)]
        body = " and ".join("(%s)" % p for p in parts) if parts else "True"
        src = "lambda %s: %s" % (", ".join(params), body) if params else "lambda: %s" % body
        return eval(src, dict(EVAL_GLOBALS))

    def brute_force(self, limit: int = 1 << 22):
        """(`sat`, witness) or (`unsat`, None) by enumerating every
        interpretation, or (`too-big`, None)."""
        try:
            size = 1
            for _, sort in self.decls:
                if sort == BOOL:
                    size *= 2
                elif sort[0] == "BV":
                    size *= 1 << sort[1]
                else:
                    size *= len(domain(sort[2])) ** domain_size(sort[1])
                if size > limit:
                    return "too-big", None
            domains = [domain(sort) for _, sort in self.decls]
        except TooBig:
            return "too-big", None
        fn = self.compile_conjunction()
        for combo in itertools.product(*domains):
            if fn(*combo):
                return "sat", combo
        return "unsat", None

    def eval_model(self, values: dict):
        """True / False when every declared constant has a value in `values`;
        None otherwise."""
        args = []
        for name, sort in self.decls:
            if name not in values:
                return None
            args.append(values[name])
        try:
            fn = self.compile_conjunction()
        except (TooBig, CompileError, ValueError):
            return None
        return bool(fn(*args))


def const_value(expr, sort):
    """The Python value of a closed model value term of `sort`, or None."""
    comp = Compiler()
    try:
        src, s = comp.compile(expr, {})
    except (CompileError, ValueError, KeyError, IndexError, TooBig):
        return None
    try:
        return eval(src, dict(EVAL_GLOBALS))
    except Exception:  # noqa: BLE001 - an unevaluable value is "no value"
        return None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

VERDICTS = ("sat", "unsat", "unknown")


def parse_response(stdout: str):
    """(verdicts, models, errors) from a probe's stdout.

    `models` is a list of dicts name -> value-expr, one per model response,
    in order.  A response that cannot be tokenised is returned as one error.
    """
    verdicts: list[str] = []
    models: list[dict] = []
    errors: list[str] = []
    for line in stdout.splitlines():
        if line.strip() in VERDICTS:
            verdicts.append(line.strip())
    try:
        exprs = parse_all(stdout)
    except ValueError as err:
        return verdicts, models, [f"unparsable response: {err}"]
    for e in exprs:
        if isinstance(e, list) and e:
            if e[0] == "error":
                errors.append(show(e))
                continue
            items = e[1:] if e[0] == "model" else e
            defs = [x for x in items if isinstance(x, list) and x and x[0] == "define-fun"]
            if defs:
                model = {}
                for d in defs:
                    if len(d) == 5 and d[2] == []:
                        model[d[1]] = (d[3], d[4])
                models.append(model)
    return verdicts, models, errors


def model_values(model: dict, decls) -> dict:
    """Python values for the declared constants a model names."""
    out = {}
    sorts = dict(decls)
    for name, (sort_expr, value) in model.items():
        if name not in sorts:
            continue
        v = const_value(value, sorts[name])
        if v is not None:
            out[name] = v
    return out


def pins_for(model: dict) -> str:
    """`(assert (= name value))` for every closed entry of a model."""
    out = []
    for name, (_sort, value) in model.items():
        text = show(value)
        if text == "?" or "@" in text:
            continue
        out.append("(assert (= %s %s))" % (name if re.match(r"^[A-Za-z_~!@$%^&*+=<>.?/-][A-Za-z0-9_~!@$%^&*+=<>.?/-]*$", name) else "|%s|" % name, text))
    return "\n".join(out)


def run_probe(probe: str, path: str, cap: float, extra_stdin: str | None = None) -> dict:
    """Run one script through one probe under `timeout cap`."""
    exe = probe_path(probe)
    t0 = time.monotonic()
    try:
        if extra_stdin is not None:
            proc = subprocess.run(["timeout", str(cap), exe], input=extra_stdin, capture_output=True, text=True)
        else:
            proc = subprocess.run(["timeout", str(cap), exe, path], capture_output=True, text=True)
    except OSError as err:
        return {"status": "error", "verdicts": [], "models": [], "errors": [str(err)], "ms": None, "wall": 0.0, "stdout": "", "stderr": str(err)}
    wall = time.monotonic() - t0
    ms = None
    m = re.search(r"# elapsed_ms ([0-9.]+)", proc.stderr)
    if m:
        ms = float(m.group(1))
    status = "ok"
    if proc.returncode == 124:
        status = "timeout"
    elif proc.returncode == 137:
        status = "killed"
    elif proc.returncode != 0 or "panicked" in proc.stderr:
        status = "panic"
    verdicts, models, errors = parse_response(proc.stdout)
    return {
        "status": status,
        "rc": proc.returncode,
        "verdicts": verdicts,
        "models": models,
        "errors": errors,
        "ms": ms,
        "wall": wall,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-2000:],
    }


def verdict_of(result: dict) -> str:
    """The last verdict, or TIMEOUT / PANIC / KILLED / none."""
    if result["status"] == "timeout":
        return "TIMEOUT"
    if result["status"] == "killed":
        return "KILLED"
    if result["status"] == "panic":
        return "PANIC"
    return result["verdicts"][-1] if result["verdicts"] else "none"


STAT_KEYS = (
    ":conflicts",
    ":bv-embedded-checks",
    ":bv-embedded-conflicts",
    ":array-refinement-rounds",
    ":array-lemma-instances",
)


def counters(stdout: str) -> dict:
    out = {}
    for key in STAT_KEYS:
        m = re.search(re.escape(key) + r"\s+([0-9]+)", stdout)
        if m:
            out[key] = int(m.group(1))
    return out


def pool_map(fn, items, jobs: int):
    """Run `fn` over `items` with at most `jobs` concurrent probe processes,
    preserving order."""
    if jobs <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items))


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)
