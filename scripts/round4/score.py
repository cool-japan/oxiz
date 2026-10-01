#!/usr/bin/env python3
"""score.py - #P2b-51's instrument over a quantified/ground pair corpus.

    score.py --probe probe_head --corpus corpus/qmbqi120 [--cap 20] [--jobs 2]
             [--recheck-cap 20] [--json out.json]

For every pair (qNNNN, gNNNN) it runs the quantified script and its ground
twin, and for every `sat` answer on the quantified side with a published model
it re-checks the model in TWO independent ways:

* the pin re-check (the #P2b-51 instrument of record): every `define-fun` of
  the model is asserted into the quantified script as `(assert (= name v))`
  and the script is solved again under the same cap.  `unsat` is a real
  falsification (falsifying_models), `sat` a confirmation (models_confirmed),
  anything else - `unknown`, TIMEOUT - is models_unresolved and is NEVER
  counted as falsifying;
* the exact evaluation (solver-free): the model's values are evaluated in
  Python against the quantified script, the forall ranging over every point
  of its index sort (eval_true / eval_false / eval_na).  This is the DECISIVE
  figure: the pin re-check asks the probe under test, and a probe with a
  wrong-`sat` defect on pinned scripts confirms falsifying models
  (pinned_wrong_sat counts exactly those: eval false, every constant pinned,
  pin re-check `sat`).

Prints one JSON-ish line of totals with the probe, cap and job count, and the
per-pair details with --json.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import common as c


def pin_script(text: str, model: dict) -> str:
    pins = c.pins_for(model)
    text = text.replace("(get-model)\n", "")
    return text.replace("(check-sat)", pins + "\n(check-sat)", 1)


def score_pair(args, k):
    qp = os.path.join(args.corpus, "q%04d.smt2" % k)
    gp = os.path.join(args.corpus, "g%04d.smt2" % k)
    with open(qp) as fh:
        q_text = fh.read()
    rq = c.run_probe(args.probe, qp, args.cap)
    rg = c.run_probe(args.probe, gp, args.cap)
    rec = {"k": k, "q": c.verdict_of(rq), "g": c.verdict_of(rg), "q_ms": rq["ms"], "g_ms": rg["ms"],
           "q_stdout": rq["stdout"], "panic": rq["status"] == "panic" or rg["status"] == "panic",
           # A `sat` whose model the net withheld (decision (54)(i)) is neither
           # confirmed nor falsifying: counted apart (decision (69)(11)).
           "withheld": rq["status"] == "ok" and "model not certified" in (rq["stdout"] or "")}
    if rec["q"] == "sat" and rq["models"]:
        model = rq["models"][-1]
        rec["model"] = {n: c.show(v) for n, (_s, v) in model.items()}
        script = c.Script(q_text)
        values = c.model_values(model, script.decls)
        ev = script.eval_model(values)
        rec["eval"] = "na" if ev is None else ("true" if ev else "false")
        rec["all_pinned"] = all(name in values for name, _ in script.decls)
        rp = c.run_probe(args.probe, None, args.recheck_cap, extra_stdin=pin_script(q_text, model))
        pv = c.verdict_of(rp)
        rec["recheck"] = {"sat": "confirmed", "unsat": "falsifying"}.get(pv, "unresolved")
        rec["recheck_verdict"] = pv
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--recheck-cap", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.recheck_cap is None:
        args.recheck_cap = args.cap
    ks = sorted(int(m.group(1)) for f in os.listdir(args.corpus) if (m := re.match(r"q(\d{4})\.smt2$", f)))
    recs = c.pool_map(lambda k: score_pair(args, k), ks, args.jobs)
    tot = {
        "pairs": len(recs),
        "q_sat": sum(r["q"] == "sat" for r in recs),
        "q_sat_g_unsat": sum(r["q"] == "sat" and r["g"] == "unsat" for r in recs),
        "q_unsat_g_sat": sum(r["q"] == "unsat" and r["g"] == "sat" for r in recs),
        "falsifying_models": sum(r.get("recheck") == "falsifying" for r in recs),
        "models_confirmed": sum(r.get("recheck") == "confirmed" for r in recs),
        "models_unresolved": sum(r.get("recheck") == "unresolved" for r in recs),
        "models_withheld": sum(r["q"] == "sat" and bool(r.get("withheld")) for r in recs),
        "panics": sum(r["panic"] for r in recs),
        "agree": sum(r["q"] in ("sat", "unsat") and r["q"] == r["g"] for r in recs),
        "eval_false": sum(r.get("eval") == "false" for r in recs),
        "eval_true": sum(r.get("eval") == "true" for r in recs),
        "eval_na": sum(r.get("eval") == "na" for r in recs),
        # Exact evaluation says the model is false, every declared constant is
        # pinned, and the probe still answered `sat` to the pinned script: by
        # construction a WRONG `sat` on that pinned script.
        "pinned_wrong_sat": sum(r.get("eval") == "false" and bool(r.get("all_pinned")) and
                                r.get("recheck_verdict") == "sat" for r in recs),
    }
    part = {}
    for r in recs:
        part[r["q"]] = part.get(r["q"], 0) + 1
    print("score probe=%s corpus=%s cap=%gs recheck_cap=%gs jobs=%d" % (
        args.probe, args.corpus, args.cap, args.recheck_cap, args.jobs))
    print(json.dumps(tot))
    print("note: falsifying_models / models_confirmed come from the pin re-check, which asks the probe "
          "itself; wherever pinned_wrong_sat > 0 the probe confirmed a model the exact evaluation "
          "refutes, so eval_false / eval_true (solver-free) is the decisive figure.")
    print("q partition", json.dumps(dict(sorted(part.items()))))
    bad = [r["k"] for r in recs if r.get("recheck") == "falsifying" or r.get("eval") == "false"]
    print("falsifying (recheck or eval):", " ".join("q%04d" % k for k in bad))
    disagree = [r["k"] for r in recs if (r.get("recheck") == "falsifying") != (r.get("eval") == "false")
                and r.get("recheck") != "unresolved" and "eval" in r]
    if disagree:
        print("RECHECK/EVAL DISAGREE:", " ".join("q%04d" % k for k in disagree))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "totals": tot, "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
