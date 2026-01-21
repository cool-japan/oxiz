#!/usr/bin/env python3
"""
Basic SAT example using OxiZ

This example demonstrates:
- Creating boolean variables
- Building boolean formulas
- Checking satisfiability
- Extracting models
"""

import oxiz

def main():
    # Create term manager and solver
    tm = oxiz.TermManager()
    solver = oxiz.Solver()

    # Create boolean variables
    p = tm.mk_var("p", "Bool")
    q = tm.mk_var("q", "Bool")
    r = tm.mk_var("r", "Bool")

    print("Variables: p, q, r")

    # Assert: (p OR q) AND (NOT p OR r) AND (NOT q OR NOT r)
    clause1 = tm.mk_or([p, q])
    clause2 = tm.mk_or([tm.mk_not(p), r])
    clause3 = tm.mk_or([tm.mk_not(q), tm.mk_not(r)])

    solver.assert_term(clause1, tm)
    solver.assert_term(clause2, tm)
    solver.assert_term(clause3, tm)

    print("Constraints:")
    print("  (p OR q)")
    print("  (NOT p OR r)")
    print("  (NOT q OR NOT r)")
    print()

    # Check satisfiability
    result = solver.check_sat(tm)
    print(f"Result: {result}")

    if result == oxiz.SolverResult.Sat:
        model = solver.get_model(tm)
        print("Model found:")
        print(f"  p = {model['p']}")
        print(f"  q = {model['q']}")
        print(f"  r = {model['r']}")
    elif result == oxiz.SolverResult.Unsat:
        print("No solution exists (UNSAT)")
    else:
        print("Unknown (timeout or incomplete)")

if __name__ == "__main__":
    main()
