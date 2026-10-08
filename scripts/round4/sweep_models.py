#!/usr/bin/env python3
"""sweep_models.py - justify every changed response of a `--get-model` sweep.

    sweep_models.py --json fixA/after/sweep_gm.json --judge probe_tree [--root ...]
    sweep_models.py --json fixA/after/sweep_gm.json --judge z3 [--zcap 30]

For every script whose response differs between the two probes of the sweep,
both published models are pinned into the script (`(assert (= name value))`
for every `define-fun`), the script is re-solved by the JUDGE probe, and the
verdict is reported per side: `sat` = the model satisfies the script,
`unsat` = the model falsifies it.  Where the script's index sorts are finite
the Python exact evaluator is consulted as well.  The judge must be a probe
without the #P2b-60 wrong-`sat` (a pinned re-check through a probe that
confirms falsifying models proves nothing).
"""
from __future__ import annotations

import argparse
import json
import os

import common as c


def pinned(text: str, model: dict) -> str:
    pins = c.pins_for(model)
    body = text.replace("(get-model)", "").replace("(exit)", "")
    return body.replace("(check-sat)", pins + "\n(check-sat)", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--root", default=c.REPO_ROOT)
    ap.add_argument("--cap", type=float, default=130.0)
    ap.add_argument("--zcap", type=int, default=30, help="z3 cap per query with --judge z3")
    args = ap.parse_args()
    with open(args.json) as fh:
        data = json.load(fh)
    for rec in data["records"]:
        if rec["a"] == rec["b"]:
            continue
        with open(os.path.join(args.root, rec["path"])) as fh:
            text = fh.read()
        out = []
        if args.judge == "z3":
            # z3 as the independent judge (decision (69)(11)): every printed
            # model of each side is replayed with its define-funs.
            import zjudge  # noqa: E402 - only with --judge z3
            # The script as the sweep ran it: `(get-model)` after every
            # `(check-sat)` (sweep.py --get-model), so each printed model is
            # matched to the check it answers.
            swept = text.replace("(check-sat)", "(check-sat)\n(get-model)")
            for side in ("a", "b"):
                res = zjudge.judge(swept, rec[side], args.zcap, False)
                out.append("%s: z3 ok=%d falsifying=%d withheld=%d unresolved=%d" % (
                    side, res.get("model_ok", 0), res.get("falsifying", 0), res.get("withheld", 0),
                    res.get("model_unresolved", 0) + res.get("uc_unresolved", 0)))
            print("%-60s %s" % (rec["path"], " | ".join(out)))
            continue
        for side in ("a", "b"):
            _, models, _ = c.parse_response(rec[side])
            if not models:
                out.append("%s: no model" % side)
                continue
            r = c.run_probe(args.judge, None, args.cap, extra_stdin=pinned(text, models[-1]))
            verdict = c.verdict_of(r)
            try:
                script = c.Script(text.replace("(exit)", ""))
                ev = script.eval_model(c.model_values(models[-1], script.decls))
            except (ValueError, KeyError, c.CompileError):
                ev = None
            out.append("%s: pinned=%s eval=%s" % (side, verdict, {True: "true", False: "false", None: "n/a"}[ev]))
        print("%-60s %s" % (rec["path"], " | ".join(out)))


if __name__ == "__main__":
    main()
