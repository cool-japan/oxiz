# oxiz-cli

Command-line interface for OxiZ SMT solver.

## Installation

From crates.io:

```bash
cargo install oxiz-cli
```

Or build from source:

```bash
git clone https://github.com/cool-japan/oxiz
cd oxiz/oxiz-cli
cargo build --release
# Binary will be at: target/release/oxiz
```

## Usage

### Solve SMT-LIB2 Files

```bash
# Solve a single file
oxiz input.smt2

# Solve multiple files
oxiz file1.smt2 file2.smt2 file3.smt2

# Read from stdin
cat input.smt2 | oxiz -
```

### Interactive Mode

```bash
oxiz --interactive

# Or use short flag
oxiz -i
```

In interactive mode, enter SMT-LIB2 commands directly:

```
oxiz> (set-logic QF_LIA)
oxiz> (declare-const x Int)
oxiz> (assert (> x 0))
oxiz> (check-sat)
sat
oxiz> (exit)
```

### Options

```
USAGE:
    oxiz [OPTIONS] [FILES]...

ARGS:
    <FILES>...    Input SMT-LIB2 files (use - for stdin)

OPTIONS:
    -i, --interactive    Run in interactive mode
    -v, --verbose        Enable verbose output
    -t, --timeout <MS>   Set timeout in milliseconds
    -h, --help           Print help information
    -V, --version        Print version information
```

## Examples

### Basic Satisfiability

```bash
echo '
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(assert (< x 10))
(check-sat)
' | oxiz -
```

Output:
```
sat
```

### Unsatisfiable Problem

```bash
echo '
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 10))
(assert (< x 5))
(check-sat)
' | oxiz -
```

Output:
```
unsat
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success (satisfiable or completed) |
| 1    | Unsatisfiable |
| 2    | Unknown/Timeout |
| 3    | Parse error |
| 4    | Other error |

## License

MIT OR Apache-2.0
