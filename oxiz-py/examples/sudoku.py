#!/usr/bin/env python3
"""
Sudoku solver using OxiZ

This example demonstrates:
- Solving combinatorial puzzles with SMT
- Using distinct constraints
- Extracting solutions from models
"""

import oxiz

def solve_sudoku(grid):
    """
    Solve a Sudoku puzzle using OxiZ SMT solver

    Args:
        grid: 9x9 list of lists with 0 for empty cells

    Returns:
        9x9 solution grid, or None if unsolvable
    """
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_LIA")

    # Create variables for each cell (1-9)
    cells = {}
    for i in range(9):
        for j in range(9):
            cells[(i, j)] = tm.mk_var(f"cell_{i}_{j}", "Int")

    one = tm.mk_int(1)
    nine = tm.mk_int(9)

    # Each cell is between 1 and 9
    for (i, j), cell in cells.items():
        solver.assert_term(tm.mk_ge(cell, one), tm)
        solver.assert_term(tm.mk_le(cell, nine), tm)

    # Row constraints: all different
    for i in range(9):
        row_cells = [cells[(i, j)] for j in range(9)]
        solver.assert_term(tm.mk_distinct(row_cells), tm)

    # Column constraints: all different
    for j in range(9):
        col_cells = [cells[(i, j)] for i in range(9)]
        solver.assert_term(tm.mk_distinct(col_cells), tm)

    # 3x3 box constraints: all different
    for box_i in range(3):
        for box_j in range(3):
            box_cells = []
            for i in range(3):
                for j in range(3):
                    box_cells.append(cells[(box_i*3 + i, box_j*3 + j)])
            solver.assert_term(tm.mk_distinct(box_cells), tm)

    # Given values
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                val = tm.mk_int(grid[i][j])
                solver.assert_term(tm.mk_eq(cells[(i, j)], val), tm)

    # Solve
    result = solver.check_sat(tm)
    if result == oxiz.SolverResult.Sat:
        model = solver.get_model(tm)
        solution = [[int(model[f"cell_{i}_{j}"]) for j in range(9)] for i in range(9)]
        return solution
    return None

def print_grid(grid):
    """Print a 9x9 grid in a readable format"""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        row = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row += "| "
            row += f"{grid[i][j]} " if grid[i][j] != 0 else ". "
        print(row)

def main():
    # Example puzzle (easy difficulty)
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    print("=== Sudoku Solver ===\n")
    print("Input puzzle:")
    print_grid(puzzle)
    print()

    print("Solving with OxiZ...")
    solution = solve_sudoku(puzzle)

    if solution:
        print("\nSolution found:")
        print_grid(solution)
    else:
        print("\nNo solution exists!")

if __name__ == "__main__":
    main()
