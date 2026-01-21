# OxiZ Python Bindings

Python bindings for the OxiZ SMT (Satisfiability Modulo Theories) solver.

## Installation

### From PyPI (when published)

```bash
pip install oxiz
```

### From Source

Requires [Rust](https://rustup.rs/) and [maturin](https://github.com/PyO3/maturin):

```bash
# Install maturin
pip install maturin

# Build and install
cd oxiz-py
maturin develop --release
```

## Quick Start

```python
import oxiz

# Create a term manager and solver
tm = oxiz.TermManager()
solver = oxiz.Solver()

# Create integer variables
x = tm.mk_var("x", "Int")
y = tm.mk_var("y", "Int")

# Create constants
ten = tm.mk_int(10)
five = tm.mk_int(5)

# Create constraints: x + y = 10, x > 5
sum_xy = tm.mk_add([x, y])
constraint1 = tm.mk_eq(sum_xy, ten)
constraint2 = tm.mk_gt(x, five)

# Assert constraints
solver.assert_term(constraint1, tm)
solver.assert_term(constraint2, tm)

# Check satisfiability
result = solver.check_sat(tm)
print(f"Result: {result}")  # sat

if result == oxiz.SolverResult.Sat:
    model = solver.get_model(tm)
    print(f"x = {model['x']}, y = {model['y']}")
```

## API Reference

### TermManager

The `TermManager` creates and manages terms (expressions).

```python
tm = oxiz.TermManager()

# Variables
x = tm.mk_var("x", "Int")      # Integer variable
b = tm.mk_var("b", "Bool")     # Boolean variable
r = tm.mk_var("r", "Real")     # Real variable

# Constants
t = tm.mk_bool(True)           # Boolean true
f = tm.mk_bool(False)          # Boolean false
n = tm.mk_int(42)              # Integer constant
q = tm.mk_real(3, 4)           # Rational 3/4

# Boolean operations
not_b = tm.mk_not(b)
and_term = tm.mk_and([b, t])
or_term = tm.mk_or([b, f])
implies = tm.mk_implies(b, t)

# Arithmetic operations
sum_term = tm.mk_add([x, n])
diff = tm.mk_sub(x, n)
prod = tm.mk_mul([x, n])
neg = tm.mk_neg(x)
div = tm.mk_div(x, n)
mod = tm.mk_mod(x, n)

# Comparisons
eq = tm.mk_eq(x, n)            # x = 42
lt = tm.mk_lt(x, n)            # x < 42
le = tm.mk_le(x, n)            # x <= 42
gt = tm.mk_gt(x, n)            # x > 42
ge = tm.mk_ge(x, n)            # x >= 42

# Other
ite = tm.mk_ite(b, x, n)       # if b then x else 42
distinct = tm.mk_distinct([x, n])  # x != 42
```

### Solver

The `Solver` checks satisfiability of constraints.

```python
solver = oxiz.Solver()

# Set logic (optional)
solver.set_logic("QF_LIA")  # Quantifier-free Linear Integer Arithmetic

# Add constraints
solver.assert_term(constraint, tm)

# Check satisfiability
result = solver.check_sat(tm)

# Get model (if sat)
if result == oxiz.SolverResult.Sat:
    model = solver.get_model(tm)

# Push/pop for incremental solving
solver.push()
solver.assert_term(additional_constraint, tm)
result = solver.check_sat(tm)
solver.pop()  # Removes additional_constraint

# Reset solver
solver.reset()
```

### SolverResult

```python
result = solver.check_sat(tm)

# Compare with enum values
if result == oxiz.SolverResult.Sat:
    print("Satisfiable")
elif result == oxiz.SolverResult.Unsat:
    print("Unsatisfiable")
elif result == oxiz.SolverResult.Unknown:
    print("Unknown")

# Use properties
if result.is_sat:
    model = solver.get_model(tm)
```

## Examples

### Sudoku Solver

```python
import oxiz

def solve_sudoku(grid):
    tm = oxiz.TermManager()
    solver = oxiz.Solver()

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
    if solver.check_sat(tm) == oxiz.SolverResult.Sat:
        model = solver.get_model(tm)
        result = [[int(model[f"cell_{i}_{j}"]) for j in range(9)] for i in range(9)]
        return result
    return None
```

### Boolean Satisfiability

```python
import oxiz

tm = oxiz.TermManager()
solver = oxiz.Solver()

# Create boolean variables
p = tm.mk_var("p", "Bool")
q = tm.mk_var("q", "Bool")
r = tm.mk_var("r", "Bool")

# Assert: (p OR q) AND (NOT p OR r) AND (NOT q OR NOT r)
clause1 = tm.mk_or([p, q])
clause2 = tm.mk_or([tm.mk_not(p), r])
clause3 = tm.mk_or([tm.mk_not(q), tm.mk_not(r)])

solver.assert_term(clause1, tm)
solver.assert_term(clause2, tm)
solver.assert_term(clause3, tm)

result = solver.check_sat(tm)
if result == oxiz.SolverResult.Sat:
    model = solver.get_model(tm)
    print(f"p={model['p']}, q={model['q']}, r={model['r']}")
```

## Supported Sorts

- `"Bool"` - Boolean values
- `"Int"` - Arbitrary precision integers
- `"Real"` - Rational numbers
- `"BitVec[N]"` - Bit vectors of width N (e.g., `"BitVec[32]"`)

## Bitvector Operations

```python
tm = oxiz.TermManager()

# Create bitvector constants
bv8 = tm.mk_bv(42, 8)       # 8-bit bitvector with value 42
bv16 = tm.mk_bv(1000, 16)   # 16-bit bitvector

# Bitwise operations
not_x = tm.mk_bv_not(x)         # Bitwise NOT
and_xy = tm.mk_bv_and(x, y)     # Bitwise AND
or_xy = tm.mk_bv_or(x, y)       # Bitwise OR

# Arithmetic operations
sum_xy = tm.mk_bv_add(x, y)     # Addition (modulo 2^width)
diff_xy = tm.mk_bv_sub(x, y)    # Subtraction
prod_xy = tm.mk_bv_mul(x, y)    # Multiplication
neg_x = tm.mk_bv_neg(x)         # Two's complement negation

# Comparisons (unsigned)
ult = tm.mk_bv_ult(x, y)        # x <u y (unsigned less than)
ule = tm.mk_bv_ule(x, y)        # x <=u y (unsigned less or equal)

# Comparisons (signed)
slt = tm.mk_bv_slt(x, y)        # x <s y (signed less than)
sle = tm.mk_bv_sle(x, y)        # x <=s y (signed less or equal)

# Division and remainder
udiv = tm.mk_bv_udiv(x, y)      # Unsigned division
sdiv = tm.mk_bv_sdiv(x, y)      # Signed division
urem = tm.mk_bv_urem(x, y)      # Unsigned remainder
srem = tm.mk_bv_srem(x, y)      # Signed remainder

# Bit manipulation
concat = tm.mk_bv_concat(high, low)    # Concatenate bitvectors
extract = tm.mk_bv_extract(7, 4, x)    # Extract bits[7:4]
```

## Array Operations

```python
tm = oxiz.TermManager()

# Array select and store
value = tm.mk_select(array, index)              # array[index]
new_array = tm.mk_store(array, index, value)    # array with array[index] = value
```

## Optimization

OxiZ supports optimization (minimize/maximize) using the `Optimizer` class:

```python
tm = oxiz.TermManager()
opt = oxiz.Optimizer()
opt.set_logic("QF_LIA")

# Variables
x = tm.mk_var("x", "Int")
y = tm.mk_var("y", "Int")

# Constraints
zero = tm.mk_int(0)
ten = tm.mk_int(10)
sum_xy = tm.mk_add([x, y])

opt.assert_term(tm.mk_ge(sum_xy, ten))  # x + y >= 10
opt.assert_term(tm.mk_ge(x, zero))       # x >= 0
opt.assert_term(tm.mk_ge(y, zero))       # y >= 0

# Objective: minimize x + y
opt.minimize(sum_xy)

# Solve
result = opt.optimize(tm)
if result == oxiz.OptimizationResult.Optimal:
    model = opt.get_model(tm)
    print(f"Optimal: x={model['x']}, y={model['y']}")  # Should be x=0, y=10 or x=10, y=0
```

### Optimization Results

- `OptimizationResult.Optimal` - Optimal solution found
- `OptimizationResult.Unbounded` - Objective is unbounded (no finite optimum)
- `OptimizationResult.Unsat` - No feasible solution exists
- `OptimizationResult.Unknown` - Timeout or incomplete

## Examples

See the `examples/` directory for complete demos:

- `basic_sat.py` - Boolean satisfiability
- `bitvec_example.py` - Bitvector operations
- `optimization_example.py` - Optimization (min/max)
- `sudoku.py` - Sudoku solver

## Testing

Run the test suite with pytest:

```bash
# Install the package in development mode
maturin develop --release

# Run tests
pytest tests/ -v
```

## License

MIT OR Apache-2.0
