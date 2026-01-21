#!/usr/bin/env python3
"""
Optimization example using OxiZ

This example demonstrates:
- Minimization and maximization
- Optimization with constraints
- Finding optimal values
"""

import oxiz

def minimize_example():
    """Minimize x subject to x >= 5"""
    print("=== Minimization Example ===")
    print("Minimize x subject to: x >= 5")
    print()

    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    # Create variable
    x = tm.mk_var("x", "Int")

    # Add constraint: x >= 5
    five = tm.mk_int(5)
    opt.assert_term(tm.mk_ge(x, five))

    # Minimize x
    opt.minimize(x)

    # Optimize
    result = opt.optimize(tm)
    print(f"Result: {result}")

    if result == oxiz.OptimizationResult.Optimal:
        model = opt.get_model(tm)
        print(f"Optimal value: x = {model['x']}")
        print("(Should be 5)\n")
    else:
        print(f"Optimization failed: {result}\n")

def maximize_example():
    """Maximize x subject to x <= 10 and x >= 0"""
    print("=== Maximization Example ===")
    print("Maximize x subject to: 0 <= x <= 10")
    print()

    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    # Create variable
    x = tm.mk_var("x", "Int")

    # Add constraints: 0 <= x <= 10
    zero = tm.mk_int(0)
    ten = tm.mk_int(10)
    opt.assert_term(tm.mk_ge(x, zero))
    opt.assert_term(tm.mk_le(x, ten))

    # Maximize x
    opt.maximize(x)

    # Optimize
    result = opt.optimize(tm)
    print(f"Result: {result}")

    if result == oxiz.OptimizationResult.Optimal:
        model = opt.get_model(tm)
        print(f"Optimal value: x = {model['x']}")
        print("(Should be 10)\n")
    else:
        print(f"Optimization failed: {result}\n")

def constrained_optimization():
    """Minimize x + 2*y subject to x + y >= 10, x >= 5, y >= 0"""
    print("=== Constrained Optimization ===")
    print("Minimize: x + 2*y")
    print("Subject to:")
    print("  x + y >= 10")
    print("  x >= 5")
    print("  y >= 0")
    print()

    tm = oxiz.TermManager()
    opt = oxiz.Optimizer()
    opt.set_logic("QF_LIA")

    # Create variables
    x = tm.mk_var("x", "Int")
    y = tm.mk_var("y", "Int")

    # Create constants
    zero = tm.mk_int(0)
    five = tm.mk_int(5)
    ten = tm.mk_int(10)
    two = tm.mk_int(2)

    # Constraints
    sum_xy = tm.mk_add([x, y])
    opt.assert_term(tm.mk_ge(sum_xy, ten))  # x + y >= 10
    opt.assert_term(tm.mk_ge(x, five))       # x >= 5
    opt.assert_term(tm.mk_ge(y, zero))       # y >= 0

    # Objective: minimize x + 2*y
    two_y = tm.mk_mul([two, y])
    objective = tm.mk_add([x, two_y])
    opt.minimize(objective)

    # Optimize
    result = opt.optimize(tm)
    print(f"Result: {result}")

    if result == oxiz.OptimizationResult.Optimal:
        model = opt.get_model(tm)
        print(f"Optimal solution:")
        print(f"  x = {model['x']}")
        print(f"  y = {model['y']}")
        x_val = int(model['x'])
        y_val = int(model['y'])
        obj_val = x_val + 2 * y_val
        print(f"  Objective value: {obj_val}")
        print(f"(Should be x=5, y=5, objective=15)\n")
    else:
        print(f"Optimization failed: {result}\n")

if __name__ == "__main__":
    minimize_example()
    maximize_example()
    constrained_optimization()
