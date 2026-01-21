# OxiZ Architecture

**Last Updated:** 2026-01-17

## 1. Project Overview

### What is OxiZ?

OxiZ is a next-generation **Satisfiability Modulo Theories (SMT) solver** written entirely in pure Rust. It implements a modular CDCL(T) architecture that closely follows the design of Z3 while leveraging Rust's safety guarantees and modern features.

**Project Statistics:**
- ~173,500 lines of Rust code
- 3,670+ tests
- 11 workspace crates
- ~90-95% Z3 feature parity

### Key Differentiators vs Z3

| Feature | Z3 | OxiZ |
|---------|-----|------|
| **Language** | C++ | Pure Rust (no FFI, memory-safe) |
| **WASM Bundle** | ~20MB | Target <2MB |
| **Machine-Checkable Proofs** | Generic proofs | DRAT, Alethe, LFSC + **Coq/Lean/Isabelle exports** |
| **Spacer/PDR** | Yes | Yes (missing in CVC5, Yices, most Z3 clones) |
| **Parallelism** | Limited | Native multi-core (Rayon portfolio, work-stealing) |
| **Memory Safety** | Manual C++ | Guaranteed by Rust ownership system |
| **Craig Interpolation** | Yes | McMillan, Pudlak, Huang + theory support |

---

## 2. Crate Dependency Graph

```
                              +-----------+
                              |   oxiz    |  (meta-crate: unified API)
                              +-----+-----+
                                    |
           +------------------------+-------------------------+
           |                        |                         |
           v                        v                         v
    +------------+           +------------+            +------------+
    | oxiz-cli   |           | oxiz-wasm  |            | oxiz-opt   |
    | (CLI front)|           | (WASM bind)|            | (MaxSAT/OMT)
    +-----+------+           +-----+------+            +-----+------+
          |                        |                         |
          +------------------------+-------------------------+
                                   |
                            +------+------+
                            | oxiz-solver |  (CDCL(T) orchestration)
                            +------+------+
                                   |
           +-----------------------+-----------------------+
           |                       |                       |
           v                       v                       v
    +-------------+        +---------------+       +--------------+
    | oxiz-spacer |        | oxiz-theories |       | oxiz-proof   |
    | (PDR/CHC)   |        | (EUF,LRA,BV..)|       | (DRAT,Alethe)|
    +------+------+        +-------+-------+       +------+-------+
           |                       |                      |
           |           +-----------+-----------+          |
           |           |                       |          |
           |           v                       v          |
           |     +----------+          +-----------+      |
           |     | oxiz-sat |          | oxiz-nlsat|      |
           |     | (CDCL)   |          | (CAD/NRA) |      |
           |     +-----+----+          +-----+-----+      |
           |           |                     |            |
           |           +----------+----------+            |
           |                      |                       |
           +----------------------+-----------------------+
                                  |
                           +------+------+
                           | oxiz-math   |  (polynomials, simplex, LP)
                           +------+------+
                                  |
                           +------+------+
                           | oxiz-core   |  (AST, sorts, parser, tactics)
                           +-------------+
```

### Dependency Summary

| Crate | Depends On | Provides |
|-------|------------|----------|
| **oxiz-core** | None (foundation) | AST, sorts, SMT-LIB2 parser, tactics, rewriters |
| **oxiz-math** | num-rational, num-bigint | Polynomials, simplex, intervals, LP, Grobner bases |
| **oxiz-sat** | oxiz-core | CDCL SAT solver with VSIDS/LRB/VMTF/CHB |
| **oxiz-theories** | oxiz-core, oxiz-sat | EUF, LRA, LIA, BV, Arrays, Strings, FP, Datatypes |
| **oxiz-nlsat** | oxiz-core, oxiz-math | Non-linear arithmetic (CAD, NRA/NIA) |
| **oxiz-proof** | oxiz-core, (oxiz-sat optional) | DRAT, Alethe, LFSC, Coq/Lean/Isabelle exports |
| **oxiz-solver** | oxiz-core, oxiz-sat, oxiz-theories | CDCL(T) orchestration, SMT-LIB2 execution |
| **oxiz-spacer** | oxiz-core, oxiz-solver | PDR/IC3, CHC solving, BMC, invariant synthesis |
| **oxiz-opt** | oxiz-core, oxiz-sat, oxiz-solver | MaxSAT, OMT, Pareto optimization |
| **oxiz-cli** | oxiz-core, oxiz-solver | Command-line interface, LSP server |
| **oxiz-wasm** | oxiz-core, oxiz-solver | WebAssembly bindings for browsers |

---

## 3. Key Algorithms by Crate

### oxiz-core (Foundation)

**Purpose:** Core data structures and utilities used by all other crates.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **AST** | Hash-consed terms, DAG traversal, pattern matching |
| **Sorts** | Parametric sorts, sort inference, recursive datatypes |
| **Parser** | Winnow-based SMT-LIB2 parser (define-fun, define-sort, datatypes) |
| **Tactics** | Simplify, propagate-values, ctx-simplify, bit-blast, ackermannize |
| **Congruence Closure** | Backtrackable union-find with explanation tracking |
| **E-graph** | Incremental merging, cost-based extraction |
| **Scripting** | Rhai-based custom tactics |

### oxiz-math (Mathematical Foundations)

**Purpose:** Mathematical algorithms for arithmetic theories.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **Simplex** | Revised simplex with Bland/Dantzig/Steepest-edge pivoting, dual simplex |
| **Polynomials** | Multivariate polynomial arithmetic, GCD, factorization, Karatsuba multiplication |
| **Root Isolation** | Sturm sequences, bisection, Descartes' rule of signs |
| **Intervals** | Interval arithmetic, bound propagation, convex hull |
| **Grobner Bases** | Buchberger's algorithm, F4/F5 |
| **Real Closure** | Algebraic number arithmetic, root isolation |
| **Number Theory** | Miller-Rabin primality, Pollard rho, Chinese Remainder Theorem |
| **Linear Algebra** | QR/Cholesky decomposition, matrix inverse, gradient/Hessian |

### oxiz-sat (CDCL SAT Solver)

**Purpose:** High-performance propositional SAT solving.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **Core CDCL** | Two-watched literals, First-UIP conflict analysis |
| **Branching** | VSIDS, LRB, VMTF, CHB, phase saving |
| **Restarts** | Luby, geometric, Glucose-style dynamic restarts |
| **Clause Management** | 3-tier system (Core/Mid/Local), LBD-based deletion |
| **Preprocessing** | Variable elimination, subsumption, BCE, BVE, failed literal probing |
| **Inprocessing** | Vivification, distillation, clause strengthening |
| **Local Search** | WalkSAT, ProbSAT integration |
| **Parallelism** | Cube-and-conquer, portfolio with clause sharing |
| **ML Integration** | Online learning for branching/restart decisions |
| **Proof Generation** | DRAT/LRAT proof logging |

### oxiz-theories (Theory Solvers)

**Purpose:** Modular theory solvers for SMT.

| Theory | Algorithm/Technique |
|--------|---------------------|
| **EUF** | Congruence closure, E-matching for quantifiers, MBQI |
| **LRA** | Simplex with strict inequalities (infinitesimals), Farkas lemmas |
| **LIA** | Branch-and-bound, Gomory/MIR/CG cuts, strong branching, feasibility pump |
| **Difference Logic** | Bellman-Ford graph-based algorithm |
| **UTVPI** | Doubled graph algorithm (Unit Two Variable Per Inequality) |
| **BitVectors** | Bit-blasting with AIGS, word-level propagators |
| **Arrays** | Read-over-write, extensionality, lazy axiom instantiation |
| **Strings** | Brzozowski derivatives, automata-based regex, word equations |
| **Floating-Point** | IEEE 754 with bit-blasting, rounding modes |
| **Datatypes** | ADT with constructors, testers, selectors |
| **Pseudo-Boolean** | Cardinality constraints, PB-specific propagation |
| **Theory Combination** | Nelson-Oppen, model-based, delayed, polite combination |

### oxiz-nlsat (Non-linear Arithmetic)

**Purpose:** Solving non-linear real/integer arithmetic.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **CAD** | Cylindrical Algebraic Decomposition (McCallum/Collins projection) |
| **Cell Decomposition** | Lifting phase, sample point selection |
| **Root Isolation** | Sturm sequences, interval bisection |
| **Variable Ordering** | Brown's heuristic, degree-based, activity-based (VSIDS-like) |
| **NIA** | Branch-and-bound for integers, Gomory/split cuts |
| **Parallel Projection** | Rayon-based parallel CAD projection |
| **Optimizations** | Polynomial evaluation cache, discriminant analysis, bound propagation |

### oxiz-proof (Proof Generation)

**Purpose:** Machine-checkable proof production and verification.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **DRAT** | SAT proof format (text/binary), clause addition/deletion recording |
| **Alethe** | SMT proof format with standard rules |
| **LFSC** | Logical Framework with Side Conditions (CVC5 compatible) |
| **Theorem Provers** | Coq, Lean 3/4, Isabelle/HOL exports |
| **Proof Checking** | Internal verification, Carcara compatibility |
| **Craig Interpolation** | McMillan, Pudlak, Huang algorithms |
| **Proof Operations** | Trimming, compression, merging, slicing, normalization |
| **Proof Learning** | Pattern extraction, template identification, fingerprinting |

### oxiz-solver (CDCL(T) Orchestration)

**Purpose:** Main SMT solving engine.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **CDCL(T)** | Lazy SMT with theory callbacks |
| **Encoding** | Tseitin transformation, polarity-aware |
| **Theory Integration** | Eager/lazy theory checking, propagation |
| **Preprocessing** | Unit propagation, pure literal detection, simplification |
| **Model Generation** | Model extraction, evaluation, minimization |
| **Unsat Core** | Core extraction, greedy minimization |
| **Optimization** | Lexicographic, Pareto, iterative optimization |
| **Context Management** | Push/pop, incremental solving |

### oxiz-spacer (PDR/CHC Solving)

**Purpose:** Property Directed Reachability for software verification.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **PDR/IC3** | Frame sequence management, proof obligations, propagation |
| **CHC** | Constrained Horn Clause representation, rule parsing |
| **Generalization** | MIC (Minimal Inductive Clause), CTG (Counterexample-guided) |
| **Invariant Synthesis** | Houdini algorithm, template-based inference |
| **BMC** | Bounded model checking, K-induction |
| **Parallelism** | Portfolio solver, parallel frame solving, distributed PDR |
| **Theories** | LIA/LRA, Arrays, BitVectors support |

### oxiz-opt (Optimization)

**Purpose:** MaxSAT and Optimization Modulo Theories.

| Component | Algorithm/Technique |
|-----------|---------------------|
| **MaxSAT** | Fu-Malik, OLL, MSU3, PMRES, RC2, MaxHS, IHS |
| **Weighted MaxSAT** | WMax (stratified), SortMax (sorting networks) |
| **Encodings** | Totalizer, sequential counter, pairwise, cardinality networks |
| **OMT** | Binary/linear/geometric search optimization |
| **Pareto** | Multi-objective, Pareto front enumeration |
| **LNS** | Large Neighborhood Search with restart strategies |
| **SLS** | Stochastic Local Search hybrid |
| **Preprocessing** | Soft clause preprocessing, BVE, hardening |
| **WCNF** | MaxSAT competition format parser |

### oxiz-cli (Command-Line Interface)

**Purpose:** User-facing command-line solver.

| Feature | Description |
|---------|-------------|
| **Input Formats** | SMT-LIB2, DIMACS CNF, QDIMACS |
| **Output Formats** | SMT-LIB2, JSON, YAML |
| **Interactive** | REPL with syntax highlighting, tab completion |
| **Parallelism** | Portfolio solving, parallel file processing |
| **Analysis** | Query complexity analysis, auto-tuning |
| **Integration** | LSP server, CI/CD helpers, shell completions |
| **Diagnostics** | Proof checking, dependency analysis, tutorial mode |

### oxiz-wasm (WebAssembly)

**Purpose:** Browser-based SMT solving.

| Feature | Description |
|---------|-------------|
| **API** | Full solver API (declare, assert, check-sat, get-model) |
| **Async** | True async with event loop yielding, cancellation |
| **Builders** | Formula builders (mkEq, mkAnd, mkOr, mkBvAdd, etc.) |
| **Optimization** | minimize, maximize, assertSoft (MaxSMT) |
| **Framework Wrappers** | React, Vue, Svelte, Deno |
| **TypeScript** | Full type declarations |

---

## 4. Data Flow

### Solving a Query (High-Level)

```
                          SMT-LIB2 Script
                                |
                                v
                    +------------------------+
                    |      oxiz-core         |
                    |    (Parser + AST)      |
                    +------------------------+
                                |
                         Parsed Terms
                                |
                                v
                    +------------------------+
                    |     oxiz-solver        |
                    |   (CDCL(T) Engine)     |
                    +------------------------+
                                |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
    +--------------+    +---------------+    +--------------+
    |   oxiz-sat   |    | oxiz-theories |    | oxiz-proof   |
    | (SAT Core)   |    | (Theory Check)|    | (Proof Log)  |
    +--------------+    +---------------+    +--------------+
            |                   |
            v                   v
    +------------------------------------------+
    |              SAT/UNSAT/UNKNOWN           |
    |         (+ Model/Core/Proof)             |
    +------------------------------------------+
```

### Detailed Solve Flow

1. **Parsing** (oxiz-core)
   - SMT-LIB2 script is tokenized and parsed
   - Terms are hash-consed for sharing
   - Sorts are inferred and validated

2. **Preprocessing** (oxiz-solver, oxiz-core)
   - Tactics applied (simplify, propagate-values)
   - Formula simplified and normalized
   - CNF-like structure prepared

3. **Encoding** (oxiz-solver)
   - Boolean skeleton created via Tseitin encoding
   - Theory atoms assigned Boolean variables
   - Constraints registered with theories

4. **CDCL(T) Loop** (oxiz-solver, oxiz-sat)
   ```
   while not done:
       # SAT solver makes decisions
       propagate()              # Boolean Constraint Propagation

       if conflict:
           analyze_conflict()   # First-UIP, learn clause
           backtrack()
       else:
           theory_check()       # Check theories (oxiz-theories)

           if theory_conflict:
               add_theory_lemma()
           elif all_assigned:
               return SAT with model
   ```

5. **Theory Checking** (oxiz-theories)
   - Each theory checks its constraints
   - EUF: Congruence closure
   - LRA: Simplex feasibility
   - BV: Bit-blasting or word-level
   - Conflicts generate theory lemmas

6. **Result Production**
   - **SAT**: Model extracted from assignments
   - **UNSAT**: Core extracted, proof generated
   - **UNKNOWN**: Resource limits or incomplete theory

---

## 5. Extension Points

### Adding a New Theory

To add a new theory solver:

1. **Create Theory Solver** in `oxiz-theories`
   ```rust
   pub struct MyTheorySolver {
       // State for your theory
   }

   impl TheorySolver for MyTheorySolver {
       fn assert_constraint(&mut self, term: TermId, value: bool);
       fn check(&mut self) -> TheoryResult;
       fn propagate(&mut self) -> Vec<Propagation>;
       fn explain_conflict(&self) -> Vec<TermId>;
       fn push(&mut self);
       fn pop(&mut self);
   }
   ```

2. **Register with Solver** in `oxiz-solver`
   - Add theory to `TheoryManager`
   - Handle theory-specific term kinds in encoding
   - Implement model extraction for theory sorts

3. **Add Sort/Term Support** in `oxiz-core`
   - Define new `SortKind` variants if needed
   - Define new `TermKind` variants for operations
   - Update SMT-LIB2 parser

4. **Write Tests**
   - Unit tests for theory solver
   - Integration tests with full solver
   - SMT-LIB2 benchmark tests

### Adding a New Tactic

To add a new preprocessing tactic:

1. **Implement Tactic Trait** in `oxiz-core`
   ```rust
   pub struct MyTactic;

   impl Tactic for MyTactic {
       fn apply(&self, goal: Goal) -> TacticResult {
           // Transform the goal
           // Return new subgoals or Solved/Failed
       }

       fn name(&self) -> &str { "my-tactic" }
   }
   ```

2. **Register Tactic**
   - Add to tactic registry
   - Make available in SMT-LIB2 `(apply my-tactic)`

3. **Combine with Other Tactics**
   ```rust
   // Use combinators
   let pipeline = ThenTactic::new(
       SimplifyTactic,
       MyTactic,
   );

   let fallback = OrElseTactic::new(
       MyTactic,
       PropagateValuesTactic,
   );
   ```

### Adding a New Proof Format

To add a new proof export format:

1. **Define Format** in `oxiz-proof`
   ```rust
   pub struct MyProofFormatter;

   impl ProofFormatter for MyProofFormatter {
       fn format_proof(&self, proof: &Proof) -> String {
           // Convert proof DAG to your format
       }

       fn file_extension(&self) -> &str { "myproof" }
   }
   ```

2. **Add Export Method**
   - Add `to_my_format()` convenience method
   - Register with proof output system

3. **Implement Proof Rules**
   - Map internal proof rules to format-specific rules
   - Handle theory-specific proof steps

---

## 6. Configuration and Tuning

### Solver Presets

| Preset | Use Case |
|--------|----------|
| `fast` | Quick results, minimal preprocessing |
| `balanced` | Default, good all-around performance |
| `thorough` | Hard problems, aggressive preprocessing |
| `minimal` | Debugging, minimal overhead |

### Key Parameters

| Parameter | Effect |
|-----------|--------|
| `restart_strategy` | Luby, geometric, Glucose-style |
| `branching_heuristic` | VSIDS, LRB, VMTF, CHB |
| `theory_mode` | Eager vs lazy theory checking |
| `clause_deletion` | LBD threshold, activity decay |
| `preprocessing` | Enable/disable BCE, BVE, etc. |

---

## 7. Performance Characteristics

### Complexity by Theory

| Theory | Complexity | Notes |
|--------|------------|-------|
| QF_UF | NP-complete | Congruence closure is O(n log n) |
| QF_LRA | Polynomial | Simplex is polynomial-time |
| QF_LIA | NP-complete | Branch-and-bound, cuts |
| QF_BV | NP-complete | Bit-blasting, word-level helps |
| QF_NRA | Doubly exponential | CAD is very expensive |
| QF_S | PSPACE-complete | Automata-based |

### Parallelization Points

- **Portfolio Solving**: Multiple solver configurations in parallel
- **Cube-and-Conquer**: Parallel search space partitioning
- **CAD Projection**: Parallel polynomial operations
- **File Processing**: Parallel benchmark execution
- **Clause Sharing**: Thread-safe clause exchange

---

## 8. Building and Testing

```bash
# Build all crates
cargo build --release

# Run all tests
cargo nextest run --all-features

# Build WASM
cd oxiz-wasm && wasm-pack build --target web

# Run CLI
cargo run --release -p oxiz-cli -- input.smt2
```

---

## 9. References

### Academic Papers

- **DPLL(T)**: Nieuwenhuis, Oliveras, Tinelli (2006)
- **CDCL**: Marques-Silva, Lynce, Malik (2009)
- **Simplex for SMT**: Dutertre, de Moura (2006)
- **E-matching**: de Moura, Bjorner (2007)
- **NLSAT/CAD**: Jovanovic, de Moura (2012)
- **PDR/IC3**: Bradley (2011), Hoder, Bjorner (2012)
- **Craig Interpolation**: McMillan (2003)

### Related Projects

- [Z3](https://github.com/Z3Prover/z3) - Microsoft Research
- [CVC5](https://github.com/cvc5/cvc5) - Stanford/Iowa
- [Yices](https://yices.csl.sri.com/) - SRI International
- [MiniSat](http://minisat.se/) - Chalmers

---

*This document is part of the OxiZ project documentation.*
