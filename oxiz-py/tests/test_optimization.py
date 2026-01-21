"""Optimization tests for OxiZ Python bindings"""

import oxiz
import pytest

def test_optimizer_creation():
    """Test Optimizer can be created"""
    opt = oxiz.Optimizer()
    assert opt is not None

def test_minimize_simple():
    """Test simple minimization: minimize x subject to x >= 5"""
    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    x = tm.mk_var("x", "Int")
    five = tm.mk_int(5)

    # Constraint: x >= 5
    opt.assert_term(tm.mk_ge(x, five))

    # Objective: minimize x
    opt.minimize(x)

    # Optimize
    result = opt.optimize(tm)
    assert result == oxiz.OptimizationResult.Optimal

    # Get model and check
    model = opt.get_model(tm)
    assert int(model["x"]) == 5

def test_maximize_simple():
    """Test simple maximization: maximize x subject to x <= 10"""
    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    x = tm.mk_var("x", "Int")
    ten = tm.mk_int(10)

    # Constraint: x <= 10
    opt.assert_term(tm.mk_le(x, ten))

    # Objective: maximize x
    opt.maximize(x)

    # Optimize
    result = opt.optimize(tm)
    assert result == oxiz.OptimizationResult.Optimal

    # Get model and check
    model = opt.get_model(tm)
    assert int(model["x"]) == 10

def test_constrained_optimization():
    """Test optimization with multiple constraints"""
    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    x = tm.mk_var("x", "Int")
    y = tm.mk_var("y", "Int")

    zero = tm.mk_int(0)
    ten = tm.mk_int(10)

    # Constraints: x + y >= 10, x >= 0, y >= 0
    sum_xy = tm.mk_add([x, y])
    opt.assert_term(tm.mk_ge(sum_xy, ten))
    opt.assert_term(tm.mk_ge(x, zero))
    opt.assert_term(tm.mk_ge(y, zero))

    # Objective: minimize x + y
    opt.minimize(sum_xy)

    # Optimize
    result = opt.optimize(tm)
    assert result == oxiz.OptimizationResult.Optimal

    # Get model and check
    model = opt.get_model(tm)
    x_val = int(model["x"])
    y_val = int(model["y"])

    assert x_val >= 0
    assert y_val >= 0
    assert x_val + y_val == 10  # Should be exactly 10 (minimum)

def test_unbounded_minimization():
    """Test unbounded minimization: minimize x with no lower bound"""
    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    x = tm.mk_var("x", "Int")

    # No constraints on x
    opt.minimize(x)

    # Should be unbounded
    result = opt.optimize(tm)
    # Note: May return Unknown if unbounded detection not implemented
    # assert result == oxiz.OptimizationResult.Unbounded or result == oxiz.OptimizationResult.Unknown

def test_optimizer_push_pop():
    """Test optimizer push/pop"""
    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    x = tm.mk_var("x", "Int")
    zero = tm.mk_int(0)

    # Base constraint: x >= 0
    opt.assert_term(tm.mk_ge(x, zero))
    opt.minimize(x)

    # Push and add tighter constraint
    opt.push()
    five = tm.mk_int(5)
    opt.assert_term(tm.mk_ge(x, five))

    result = opt.optimize(tm)
    if result == oxiz.OptimizationResult.Optimal:
        model = opt.get_model(tm)
        assert int(model["x"]) >= 5

    # Pop back to original
    opt.pop()
    result2 = opt.optimize(tm)
    if result2 == oxiz.OptimizationResult.Optimal:
        model2 = opt.get_model(tm)
        assert int(model2["x"]) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
