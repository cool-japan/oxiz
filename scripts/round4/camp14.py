#!/usr/bin/env python3
"""camp14.py - run zjudge over a directory of scripts with at most --jobs workers
(recheck 14, moved in-tree by re-fix pass 15).

Each worker runs the probe, then z3, sequentially, so at most --jobs solver
processes run at once.  A script that times out is re-run prefix by prefix
(`check_prefixes`) and its longest answered prefix is judged (`prefix_checks`
in its record; re-fix pass 16, recheck 15's minor 14): the probe prints at
exit, so a timeout otherwise discards every check the script had answered.  Writes <out>/results.jsonl, <out>/summary.json and copies
every flagged script (and the probe's stdout) into <out>/flagged/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zjudge as z  # noqa: E402

KEYS = ["checks", "sat", "unsat", "unknown", "other", "wrong_sat", "wrong_unsat", "z3_unknown", "models",
        "model_ok", "falsifying", "model_unresolved", "uc_unresolved", "withheld", "gv_pairs", "gv_wrong", "gv_nonvalue",
        "gv_error", "gv_unresolved"]
FLAGS = ["wrong_sat", "wrong_unsat", "falsifying", "gv_wrong", "gv_error", "withheld"]


QUERIES = ("get-model", "get-value", "get-info", "get-option", "echo", "get-assignment",
           "get-unsat-core", "get-assertions", "get-proof")


def check_prefixes(text):
    """Each prefix of `text` that ends with one `check-sat` and the query
    commands straight after it, shortest first; the whole script is not one
    of them (it timed out)."""
    cmds = [x for x in z.c.parse_all(text) if isinstance(x, list) and x]
    ends = []
    for i, cmd in enumerate(cmds):
        if cmd[0] in ("check-sat", "check-sat-assuming"):
            j = i + 1
            while j < len(cmds) and cmds[j][0] in QUERIES:
                j += 1
            ends.append(j)
    out = []
    for end in ends[:-1]:
        out.append("\n".join(z.c.show(cmd) for cmd in cmds[:end]) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--probe", default="probe_tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--zcap", type=int, default=10)
    ap.add_argument("--no-prefixes", action="store_true",
                    help="do not judge the answered prefix of a timed-out script")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "flagged"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.dir, "*.smt2")))
    lock = threading.Lock()
    totals = {k: 0 for k in KEYS}
    totals.update({"scripts": 0, "timeout": 0, "panic": 0, "killed": 0})
    res_fh = open(os.path.join(args.out, "results.jsonl"), "w")

    def one(path):
        text = open(path).read()
        r = z.c.run_probe(args.probe, path, args.cap)
        rec = {"script": os.path.basename(path), "status": r["status"], "ms": r["ms"]}
        judged_text, judged_stdout = text, r["stdout"]
        if r["status"] == "timeout" and not args.no_prefixes:
            # The probe prints at exit, so a timeout discards every check the
            # script had answered: judge the longest prefix (up to and with the
            # queries of its k-th `check-sat`) that answers within the cap.
            for k, prefix in enumerate(check_prefixes(text), start=1):
                pr = z.c.run_probe(args.probe, None, args.cap, extra_stdin=prefix)
                if pr["status"] != "ok":
                    break
                judged_text, judged_stdout = prefix, pr["stdout"]
                rec["prefix_checks"] = k
        if r["status"] == "ok" or "prefix_checks" in rec:
            try:
                rec.update(z.judge(judged_text, judged_stdout, args.zcap))
            except Exception as err:  # noqa: BLE001
                rec["judge_error"] = repr(err)
        with lock:
            totals["scripts"] += 1
            if r["status"] in ("timeout", "panic", "killed"):
                totals[r["status"]] += 1
            for k in KEYS:
                if isinstance(rec.get(k), int):
                    totals[k] += rec[k]
            res_fh.write(json.dumps(rec) + "\n")
            res_fh.flush()
            if r["status"] in ("panic", "killed") or any(rec.get(k) for k in FLAGS):
                base = os.path.join(args.out, "flagged", os.path.basename(path))
                shutil.copy(path, base)
                with open(base + ".out", "w") as fh:
                    fh.write(r["stdout"])
                    fh.write("\n;; stderr\n" + r["stderr"])
            if totals["scripts"] % 200 == 0:
                print(json.dumps(totals), flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(one, files))
    res_fh.close()
    totals["probe"] = args.probe
    totals["cap"] = args.cap
    totals["zcap"] = args.zcap
    totals["jobs"] = args.jobs
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(totals, fh, indent=1)
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
