#!/usr/bin/env python3
"""
Bitvector example using OxiZ

This example demonstrates:
- Creating bitvector variables and constants
- Bitvector arithmetic and bitwise operations
- Solving constraints over bitvectors
"""

import oxiz

def main():
    # Create term manager and solver
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_BV")

    # Create 8-bit bitvector variables
    x = tm.mk_var("x", "BitVec[8]")
    y = tm.mk_var("y", "BitVec[8]")

    print("8-bit bitvector variables: x, y")

    # Create bitvector constants
    ten = tm.mk_bv(10, 8)
    five = tm.mk_bv(5, 8)

    # Constraint 1: x + y = 10
    sum_xy = tm.mk_bv_add(x, y)
    constraint1 = tm.mk_eq(sum_xy, ten)

    # Constraint 2: x > 5 (unsigned)
    constraint2 = tm.mk_bv_ult(five, x)  # 5 < x

    solver.assert_term(constraint1, tm)
    solver.assert_term(constraint2, tm)

    print("Constraints:")
    print("  x + y = 10 (bitvector addition)")
    print("  x > 5 (unsigned)")
    print()

    # Check satisfiability
    result = solver.check_sat(tm)
    print(f"Result: {result}")

    if result == oxiz.SolverResult.Sat:
        model = solver.get_model(tm)
        print("Model found:")
        print(f"  x = {model['x']}")
        print(f"  y = {model['y']}")

def bitvec_operations_demo():
    """Demonstrate various bitvector operations"""
    print("\n=== Bitvector Operations Demo ===\n")

    tm = oxiz.TermManager()
    solver = oxiz.Solver()

    # Create 4-bit bitvectors
    a = tm.mk_bv(0b1010, 4)  # 10 in decimal
    b = tm.mk_bv(0b0011, 4)  # 3 in decimal

    # Bitwise operations
    and_result = tm.mk_bv_and(a, b)   # 1010 & 0011 = 0010
    or_result = tm.mk_bv_or(a, b)     # 1010 | 0011 = 1011
    not_a = tm.mk_bv_not(a)            # ~1010 = 0101

    # Arithmetic operations
    sum_ab = tm.mk_bv_add(a, b)       # 10 + 3 = 13 = 1101
    diff_ab = tm.mk_bv_sub(a, b)      # 10 - 3 = 7 = 0111
    prod_ab = tm.mk_bv_mul(a, b)      # 10 * 3 = 30 = 1110 (mod 16)

    print("4-bit bitvector operations:")
    print("  a = 0b1010 (10)")
    print("  b = 0b0011 (3)")
    print()
    print("Operations defined (values computable via solver):")
    print("  a & b  (bitwise AND)")
    print("  a | b  (bitwise OR)")
    print("  ~a     (bitwise NOT)")
    print("  a + b  (addition)")
    print("  a - b  (subtraction)")
    print("  a * b  (multiplication)")

    # Concatenation and extraction
    x = tm.mk_var("x", "BitVec[8]")
    high_bits = tm.mk_bv_extract(7, 4, x)  # Extract bits 7-4
    low_bits = tm.mk_bv_extract(3, 0, x)   # Extract bits 3-0
    concat = tm.mk_bv_concat(high_bits, low_bits)  # Should equal x

    solver.assert_term(tm.mk_eq(x, tm.mk_bv(0xA5, 8)), tm)

    if solver.check_sat(tm) == oxiz.SolverResult.Sat:
        model = solver.get_model(tm)
        print(f"\nExtraction demo: x = {model['x']}")

if __name__ == "__main__":
    main()
    bitvec_operations_demo()
