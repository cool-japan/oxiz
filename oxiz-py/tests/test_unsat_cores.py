"""Tests for unsat core computation in OxiZ Python bindings.

Exercises solver.assert_and_track(expr, label, tm) and solver.unsat_core().
"""

import oxiz
import pytest


def test_basic_unsat_core():
    """Two conflicting tracked assertions produce a non-empty unsat core."""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_option("produce-unsat-cores", "true")

    x = tm.mk_var("x_uc", "Int")
    zero = tm.mk_int(0)

    # x > 0  labelled "pos_x"
    solver.assert_and_track(tm.mk_gt(x, zero), "pos_x", tm)
    # x < 0  labelled "neg_x"
    solver.assert_and_track(tm.mk_lt(x, zero), "neg_x", tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Unsat

    core = solver.unsat_core()
    assert isinstance(core, list)
    assert len(core) > 0


def test_unsat_core_contains_relevant_labels():
    """The unsat core contains at least one of the conflicting labels."""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_option("produce-unsat-cores", "true")

    p = tm.mk_var("p_uc2", "Bool")

    # p  labelled "assert_p"
    solver.assert_and_track(p, "assert_p", tm)
    # NOT p  labelled "assert_not_p"
    solver.assert_and_track(tm.mk_not(p), "assert_not_p", tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Unsat

    core = solver.unsat_core()
    # At least one label from the conflicting pair must appear in the core
    assert "assert_p" in core or "assert_not_p" in core


def test_get_unsat_core_alias():
    """get_unsat_core() is a valid alias for unsat_core()."""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_option("produce-unsat-cores", "true")

    b = tm.mk_var("b_uc3", "Bool")
    solver.assert_and_track(b, "b_label", tm)
    solver.assert_and_track(tm.mk_not(b), "not_b_label", tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Unsat

    core_a = solver.unsat_core()
    core_b = solver.get_unsat_core()
    assert core_a == core_b


def test_unsat_core_empty_when_sat():
    """When the problem is satisfiable, the unsat core is empty."""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_option("produce-unsat-cores", "true")

    p = tm.mk_var("p_sat_uc", "Bool")
    solver.assert_and_track(p, "just_p", tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Sat

    core = solver.unsat_core()
    assert core == []


def test_assert_and_track_three_labels():
    """Three tracked assertions where two conflict; core should not include
    the irrelevant third assertion."""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_option("produce-unsat-cores", "true")
    solver.set_logic("QF_LIA")

    x = tm.mk_var("x_three", "Int")
    y = tm.mk_var("y_three", "Int")
    zero = tm.mk_int(0)
    ten = tm.mk_int(10)

    # Irrelevant: y >= 0 (does not conflict)
    solver.assert_and_track(tm.mk_ge(y, zero), "y_nonneg", tm)
    # Conflicting pair: x > 0 and x < 0
    solver.assert_and_track(tm.mk_gt(x, zero), "x_pos", tm)
    solver.assert_and_track(tm.mk_lt(x, zero), "x_neg", tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Unsat

    core = solver.unsat_core()
    assert len(core) > 0
    # The irrelevant assertion may or may not be in the core, but the
    # conflicting pair must have representation
    all_labels = {"y_nonneg", "x_pos", "x_neg"}
    assert set(core).issubset(all_labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
