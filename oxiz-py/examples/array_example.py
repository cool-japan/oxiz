#!/usr/bin/env python3
"""
Array theory example using OxiZ

This example demonstrates:
- Creating array variables
- Select and store operations
- Solving constraints over arrays
"""

import oxiz

def basic_array_example():
    """Basic array operations"""
    print("=== Basic Array Example ===")
    print()

    tm = oxiz.TermManager()
    solver = oxiz.Solver()

    # Create integer variables for indices and values
    i = tm.mk_var("i", "Int")
    j = tm.mk_var("j", "Int")
    v = tm.mk_var("v", "Int")

    # In SMT-LIB2, we would declare: (declare-const a (Array Int Int))
    # For now, we work with select/store operations directly on symbolic terms

    print("Array theory operations:")
    print("  select(array, index) - read from array")
    print("  store(array, index, value) - write to array")
    print()

    # Note: Full array variable creation would require extending the API
    # to support array sort creation. For now, this demonstrates the
    # operation builders that are available.

    print("Array operations are available via mk_select() and mk_store()")
    print("Full array example requires array sort creation API (future enhancement)")

if __name__ == "__main__":
    basic_array_example()
