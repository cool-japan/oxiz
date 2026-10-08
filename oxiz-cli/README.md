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
# Binary will be at: ../target/release/oxiz (the workspace's target directory)
```

## Usage

### Solve SMT-LIB2 Files

```bash
# Solve a single file
oxiz input.smt2

# Solve multiple files
oxiz file1.smt2 file2.smt2 file3.smt2

# Read from stdin (give no file argument)
cat input.smt2 | oxiz
```

### Interactive Mode

```bash
oxiz --interactive

# Or use short flag
oxiz -i
```

In interactive mode, enter SMT-LIB2 commands directly. On 0.3.4 a symbol declared on one line is
not visible on later lines (a later line that uses it answers `(error "… unknown constant or symbol: x")`),
while assertions do carry over to later lines, so declare a symbol on the same line as the commands that
use it:

```
oxiz> (set-logic QF_LIA) (declare-const x Int) (assert (> x 0)) (check-sat)
sat
oxiz> (exit)
```

### Options

A summary of selected options (`oxiz --help` lists all of them):

```
USAGE:
    oxiz [OPTIONS] [FILE]...

ARGS:
    [FILE]...    Input file(s) (SMT-LIB2 format); glob patterns are supported.
                 If no file is given, reads from stdin

OPTIONS:
    -i, --interactive              Run in interactive mode (REPL)
    -v, --verbosity <VERBOSITY>    Verbosity level: quiet, normal (default), verbose, debug, trace
    -t, --timeout <TIMEOUT>        Timeout in seconds (0 = no timeout, the default)
    -h, --help                     Print help information
    -V, --version                  Print version information
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
' | oxiz
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
' | oxiz
```

Output:
```
unsat
```

## Exit Codes

As measured on 0.3.4 in the default mode (without `--cicd`); the workspace `CHANGELOG.md` lists CLI
exit codes as a known open item:

| Code | Meaning |
|------|---------|
| 0    | The script ran — `sat`, `unsat` and `unknown` alike (the verdict is printed, not encoded in the exit code); also a script read from stdin that fails to parse (its `(error …)` line is printed) |
| 1    | No input file matches the arguments (for example `oxiz -`), or an input file fails to parse (its `(error …)` line is printed), also when it is one of several files |
| 2    | Invalid command line: an unknown flag or an invalid flag value |
| 124  | The `--timeout` deadline passed; `unknown` is printed |

An unknown top-level command is ignored without an error: measured on 0.3.4, `(foo-bar-unknown x)` followed by `(check-sat)` on a satisfiable script prints `sat` and exits 0.

## License

Apache-2.0
