# OxiZ Feature Matrix: Targeting Z3 Parity & Beyond

This document maps Z3's capabilities (from `../z3/`) to OxiZ implementation tasks. Our goal is not just parity, but to surpass Z3 by leveraging Rust's strengths.

## Z3 Source Structure Reference

```
z3/src/
├── sat/           # CDCL SAT solver core
├── smt/           # SMT context, theory integration, 30+ theory implementations
├── nlsat/         # Non-linear arithmetic (CAD-based), ~180k lines
├── muz/           # μZ: Datalog/Fixpoint engine
│   ├── spacer/    # PDR (Property Directed Reachability)
│   ├── bmc/       # Bounded Model Checking
│   ├── clp/       # Constraint Logic Programming
│   ├── dataflow/  # Dataflow analysis
│   └── rel/       # Relational algebra
├── math/          # Mathematics libraries
│   ├── grobner/   # Gröbner bases
│   ├── polynomial/# Polynomial arithmetic
│   ├── simplex/   # Simplex algorithm
│   ├── lp/        # Linear programming
│   ├── interval/  # Interval arithmetic
│   └── realclosure/# Real closed fields
├── opt/           # Optimization: MaxSMT, OMT, Pareto
├── qe/            # Quantifier Elimination (arith, array, bv, datatype, dl)
├── tactic/        # Tactics framework (aig, arith, bv, core, fpa, portfolio, sls)
├── ast/           # AST, plugins, rewriters, simplifiers, proofs
├── solver/        # Combined solver, parallel tactical, MUS
└── parsers/       # SMT-LIB2 parser
```

---

## 1. Core Engines

| Z3 Component | Z3 Location | Description | OxiZ Strategy | Priority |
|:-------------|:------------|:------------|:--------------|:---------|
| **SAT Core** | `sat/` | CDCL with clause learning, two-watched literals, VSIDS | Implemented in `oxiz-sat` | **DONE** |
| **SMT Core** | `smt/smt_context.*` | Lazy SMT orchestrator (~5k lines) | Theory trait interaction, conflict resolution | **High** |
| **E-Graph** | `ast/euf/` | Congruence Closure for equality | Integrate `egg` crate or custom implementation | **High** |
| **NLSAT** | `nlsat/` | Non-linear real arithmetic (~180k lines) | CAD implementation, reference `nlsat_solver.cpp` | **Critical** |
| **Spacer** | `muz/spacer/` | PDR for Horn Clauses (CHC) | **Missing in most Z3 clones - Key Differentiator** | **Critical** |
| **BMC** | `muz/bmc/` | Bounded Model Checking | Verification pipeline | Medium |

---

## 2. Theories (from `smt/theory_*.cpp`)

| Theory | Z3 Files | SMT-LIB Logic | OxiZ Notes |
|:-------|:---------|:--------------|:-----------|
| **Uninterpreted Functions** | `theory_array_base.cpp` | QF_UF | E-Graph based |
| **Linear Real Arithmetic** | `theory_lra.cpp` | QF_LRA | Simplex, reference `math/simplex/` |
| **Linear Int Arithmetic** | `theory_arith*.cpp` | QF_LIA | Branch & bound, cuts |
| **Difference Logic** | `theory_diff_logic.cpp`, `theory_dense_diff_logic.cpp` | QF_IDL, QF_RDL | Bellman-Ford |
| **BitVectors** | `theory_bv.cpp`, `theory_intblast.cpp` | QF_BV | Bit-blasting + word-level |
| **Arrays** | `theory_array.cpp`, `theory_array_full.cpp` | QF_AX | Lazy axiom instantiation |
| **Floating Point** | `theory_fpa.cpp` | QF_FP | IEEE 754, reference `ast/fpa/` |
| **Sequences/Strings** | `theory_seq.cpp`, `seq_regex.cpp` | QF_S | Automata-based |
| **Datatypes (ADT)** | `theory_datatype.cpp` | Datatypes | Inductive structures |
| **RecFun** | `theory_recfun.cpp` | - | Recursive functions |
| **Pseudo-Boolean** | `theory_pb.cpp` | - | Cardinality, PB constraints |
| **Special Relations** | `theory_special_relations.cpp` | - | Transitive closure, etc. |
| **User Propagator** | `theory_user_propagator.cpp` | - | Custom theory plugin API |

---

## 3. Mathematics Libraries (from `math/`)

| Component | Z3 Location | OxiZ Strategy |
|:----------|:------------|:--------------|
| **Simplex** | `math/simplex/` | Core for LRA, implement in Pure Rust |
| **Linear Programming** | `math/lp/` | For optimization, dense matrix ops |
| **Polynomial Arithmetic** | `math/polynomial/` | Essential for NLSAT |
| **Gröbner Bases** | `math/grobner/` | Alternative to CAD for NRA |
| **Interval Arithmetic** | `math/interval/` | Bound propagation |
| **Real Closure** | `math/realclosure/` | Algebraic numbers |
| **Decision Diagrams** | `math/dd/` | BDD/ZDD operations |
| **Hilbert Basis** | `math/hilbert/` | Integer programming |

---

## 4. Optimization (from `opt/`)

| Feature | Z3 Files | OxiZ Strategy | Priority |
|:--------|:---------|:--------------|:---------|
| **MaxSMT** | `maxsmt.cpp`, `maxcore.cpp`, `wmax.cpp` | Core-guided MaxSAT algorithms | **Critical** |
| **OMT** | `optsmt.cpp`, `opt_solver.cpp` | Optimization Modulo Theories | **Critical** |
| **Pareto** | `opt_pareto.cpp` | Multi-objective optimization | Medium |
| **LNS** | `opt_lns.cpp` | Large Neighborhood Search | Medium |
| **PB-SLS** | `pb_sls.cpp` | Stochastic local search for PB | Low |

---

## 5. Quantifier Elimination (from `qe/`)

| Plugin | Z3 File | Description |
|:-------|:--------|:------------|
| **Arithmetic QE** | `qe_arith_plugin.cpp` (~98k lines) | Fourier-Motzkin, virtual term |
| **Array QE** | `qe_array_plugin.cpp` | Array quantifier elimination |
| **BV QE** | `qe_bv_plugin.cpp` | BitVector quantifier elimination |
| **Datatype QE** | `qe_datatype_plugin.cpp` | ADT quantifier elimination |
| **MBI** | `qe_mbi.cpp` | Model-Based Interpolation |
| **MBP** | `qe_mbp.cpp` | Model-Based Projection |
| **NLQSAT** | `nlqsat.cpp` | Non-linear quantified satisfiability |

---

## 6. Tactics System (from `tactic/`)

### Core Tactics (`tactic/core/`)
- `simplify`, `propagate-values`, `ctx-simplify`, `elim-uncnstr`, `solve-eqs`, `split-clause`

### Arithmetic Tactics (`tactic/arith/`)
- `normalize-bounds`, `lia2card`, `card2bv`, `nla2bv`, `fm`, `factor`, `purify-arith`

### BitVector Tactics (`tactic/bv/`)
- `bit-blast`, `bv-bounds`, `bv1-blast`, `bvarray2uf`, `dt2bv`

### Portfolio (`tactic/portfolio/`)
- Parallel tactic execution, competing strategies

### SLS (`tactic/sls/`)
- Stochastic Local Search tactics

### FPA (`tactic/fpa/`)
- Floating-point tactics, `fpa2bv`

### AIG (`tactic/aig/`)
- And-Inverter Graph transformations

---

## 7. Advanced Features - OxiZ Beyond Z3

| Feature | Z3 Status | OxiZ Opportunity |
|:--------|:----------|:-----------------|
| **Proof Generation** | Generic proofs | **Machine Checkable Proofs** (Alethe/LFSC format) by default |
| **Unsat Cores** | Supported | **Minimal Unsat Cores** with parallel reduction |
| **Parallelism** | Limited (Cube & Conquer) | **Native async/parallel** with Rayon/Tokio, work-stealing |
| **Interpolation** | Craig Interpolation | Clean Rust implementation for model checking |
| **WASM** | Heavy (~20MB), slow load | **Zero-cost WASM** - native browser/edge, <2MB |
| **Incremental** | Push/Pop | **Persistent data structures** for efficient backtracking |
| **User Propagator** | C++ API | **Rust trait-based** custom theory plugins |
| **Scripting** | Python API | **Embedded Lua/Rhai** for custom tactics |
| **Memory Safety** | Manual C++ | **Guaranteed by Rust** - no use-after-free, data races |

---

## 8. API Bindings (from `api/`)

Z3 provides bindings for: C, C++, Python, Java, .NET, Julia, OCaml, JavaScript, MCP

OxiZ target bindings (Pure Rust only - no C/C++ FFI):
- **Rust** (native)
- **Python** (PyO3 - compiles Rust directly, no C layer)
- **JavaScript/WASM** (wasm-bindgen) - **Priority**

---

## 9. Implementation Roadmap

### Phase 1: Core Parity
- [ ] Complete SMT core orchestration
- [ ] All QF theories (UF, LRA, LIA, BV, Arrays)
- [ ] Basic tactics pipeline
- [ ] SMT-LIB2 full compliance

### Phase 2: Advanced Theories
- [ ] NLSAT (non-linear arithmetic)
- [ ] String/Sequence theory
- [ ] Floating-point theory
- [ ] Datatype theory

### Phase 3: Beyond Z3
- [ ] Spacer (PDR) for CHC - **Key Differentiator**
- [ ] MaxSMT / OMT optimization
- [ ] Machine checkable proofs
- [ ] Native parallelism (portfolio, cube & conquer)
- [ ] Sub-2MB WASM bundle

### Phase 4: Ecosystem
- [ ] Python bindings (PyO3)
- [ ] VS Code extension
- [ ] SMT-COMP participation
- [ ] Benchmark suite matching SMT-LIB

---

## 10. Key Files to Study in Z3

| Purpose | Files |
|:--------|:------|
| SMT Core | `smt/smt_context.cpp`, `smt/smt_solver.cpp` |
| Theory Integration | `smt/smt_theory.cpp`, `smt/smt_setup.cpp` |
| Conflict Analysis | `smt/smt_conflict_resolution.cpp` |
| Model Generation | `smt/smt_model_generator.cpp` |
| NLSAT Core | `nlsat/nlsat_solver.cpp` (180k lines) |
| Spacer | `muz/spacer/` directory |
| Optimization | `opt/opt_context.cpp`, `opt/maxsmt.cpp` |
| Tactics | `tactic/tactical.cpp`, `tactic/tactic.cpp` |
