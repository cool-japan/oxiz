# OxiZ Solver Architecture

## Overview

OxiZ is a next-generation SMT (Satisfiability Modulo Theories) solver implemented in pure Rust. The architecture follows a modular CDCL(T) design with modern enhancements.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                    │
│  ┌────────────┬────────────┬────────────┬────────────────┐  │
│  │ SMT-LIB2   │ Rust API   │ C API      │ WASM Bindings  │  │
│  │ Parser     │            │ (oxiz-ffi) │ (oxiz-wasm)    │  │
│  └────────────┴────────────┴────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Preprocessing Layer                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Tactic Framework (oxiz-core/tactic)                │    │
│  │  • Simplification  • Bit-blasting  • Ackermann     │    │
│  │  • Solve-eqs       • Propagate-ineqs • NLA2BV      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Solver (oxiz-solver)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           CDCL SAT Engine (oxiz-sat)                 │   │
│  │  • Boolean propagation  • Clause learning           │   │
│  │  • Conflict analysis    • Restart strategies        │   │
│  │  • VSIDS/CHB heuristics • Phase saving              │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ▲ ▼                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Theory Combination (Nelson-Oppen)            │   │
│  │  • Equality propagation  • Delayed combination      │   │
│  │  • Shared term management                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ▲ ▼                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Theory Solvers (oxiz-theories)          │   │
│  │  ┌───────┬───────┬───────┬─────────┬─────────────┐  │   │
│  │  │ EUF   │ LIA   │ LRA   │ Arrays  │ Bitvectors  │  │   │
│  │  │ (UF)  │ (Int) │(Real) │ (Array) │ (BV)        │  │   │
│  │  └───────┴───────┴───────┴─────────┴─────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Mathematics Layer (oxiz-math)               │
│  • Linear Programming (Simplex, Dual Simplex)               │
│  • Polynomial Arithmetic  • GCD and Resultants              │
│  • Real Algebraic Numbers • CAD (Cylindrical Decomposition) │
│  • Farkas Lemma          • Cutting Planes                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Term Manager (oxiz-core/ast)

**Purpose**: Centralized term representation with hash consing.

**Key Features**:
- Arena-based allocation for performance
- Hash consing ensures structural sharing
- Immutable terms with lightweight `TermId` references
- Type checking via sort system

**Data Structures**:
```rust
pub struct TermManager {
    arena: Arena<Term>,
    hash_consing_table: HashMap<Term, TermId>,
    sorts: SortManager,
}

pub struct Term {
    kind: TermKind,
    sort: SortId,
    hash: u64,
}

pub enum TermKind {
    Var(Symbol),
    Const(Value),
    App { func: TermId, args: Vec<TermId> },
    Quantifier { kind: QuantKind, vars: Vec<TermId>, body: TermId },
    // ... other variants
}
```

**Performance Characteristics**:
- Term creation: O(1) amortized (hash consing)
- Term comparison: O(1) (pointer equality)
- Memory overhead: ~24 bytes per unique term

### 2. SAT Solver (oxiz-sat)

**Algorithm**: CDCL (Conflict-Driven Clause Learning) with modern enhancements

**Key Components**:

#### 2.1 Boolean Constraint Propagation (BCP)
```rust
fn propagate(&mut self) -> Result<(), Conflict> {
    while let Some(literal) = self.propagation_queue.pop() {
        for watched_clause in self.watches[literal] {
            let propagated = self.propagate_clause(watched_clause)?;
            if let Some(lit) = propagated {
                self.assign(lit, Reason::Clause(watched_clause));
            }
        }
    }
    Ok(())
}
```

**Watched Literals**: Two-watched literal scheme for efficient propagation
- Complexity: O(m + c) where m = assignments, c = conflicts
- Space: O(n * w) where w = average watch list size

#### 2.2 Conflict Analysis
```rust
fn analyze_conflict(&mut self, conflict: ClauseId) -> (LearnedClause, DecisionLevel) {
    let mut clause = vec![];
    let mut seen = HashSet::new();
    let mut current = conflict;

    // 1UIP (First Unique Implication Point)
    loop {
        for lit in self.clauses[current] {
            if !seen.contains(lit) && self.level[lit.var()] > 0 {
                if self.level[lit.var()] == self.decision_level() {
                    queue.push(lit);
                } else {
                    clause.push(lit);
                }
                seen.insert(lit);
            }
        }

        if queue.len() == 1 {
            break; // Found 1UIP
        }
        current = self.reason[queue.pop().var()];
    }

    let backtrack_level = clause.iter()
        .map(|l| self.level[l.var()])
        .max()
        .unwrap_or(0);

    (clause, backtrack_level)
}
```

**1UIP Strategy**: First Unique Implication Point
- Produces asserting clauses (guaranteed to propagate after backtracking)
- Typically smaller clauses than other UIP strategies
- Complexity: O(k) where k = conflict clause size

#### 2.3 Decision Heuristics

**VSIDS (Variable State Independent Decaying Sum)**:
```rust
fn pick_decision_variable(&self) -> Option<Var> {
    self.unassigned_vars()
        .max_by(|a, b| self.activity[a].partial_cmp(&self.activity[b]))
}

fn bump_activity(&mut self, var: Var) {
    self.activity[var] += self.activity_increment;
    if self.activity[var] > 1e100 {
        self.rescale_activities();
    }
}

fn decay_activities(&mut self) {
    self.activity_increment *= 1.0 / self.decay_factor; // typically 0.95
}
```

**CHB (Conflict History-Based)**:
- Alternative to VSIDS
- Maintains conflict participation counters
- Often outperforms VSIDS on industrial instances

#### 2.4 Restart Strategies

```rust
enum RestartStrategy {
    Fixed(usize),              // Fixed interval
    Geometric { base: usize, factor: f64 },  // Geometric growth
    Luby(usize),              // Luby sequence
    Glucose,                  // Glucose-style (LBD-based)
}
```

**Glucose-style (LBD)**:
- Literal Block Distance: number of different decision levels
- Low LBD = more likely to be useful
- Dynamic restart based on LBD statistics

### 3. Theory Solvers (oxiz-theories)

#### 3.1 Equality Logic with Uninterpreted Functions (EUF)

**Algorithm**: Congruence Closure

```rust
struct EUFSolver {
    uf: UnionFind,           // Union-find for equivalence classes
    use_list: HashMap<TermId, Vec<TermId>>,  // Congruence tracking
    pending: Vec<(TermId, TermId)>,  // Pending merges
}

fn merge(&mut self, a: TermId, b: TermId) -> Result<(), Conflict> {
    if self.uf.find(a) == self.uf.find(b) {
        return Ok(()); // Already equivalent
    }

    self.uf.union(a, b);
    self.pending.push((a, b));

    // Congruence propagation
    for (parent_a, parent_b) in self.find_parents(a, b) {
        if self.are_congruent(parent_a, parent_b) {
            self.merge(parent_a, parent_b)?;
        }
    }

    Ok(())
}
```

**Complexity**:
- Merge: O(α(n)) amortized (inverse Ackermann)
- Congruence check: O(k) where k = arity
- Space: O(n + e) where e = equality count

#### 3.2 Linear Integer Arithmetic (LIA)

**Algorithm**: Simplex + Branch-and-Bound

```rust
struct LIASolver {
    tableau: SimplexTableau,
    bounds: HashMap<Var, (Option<Int>, Option<Int>)>,
    branch_points: Vec<(Var, Int)>,
}

fn check_sat(&mut self) -> Result<(), Conflict> {
    // Phase 1: Solve LP relaxation
    self.tableau.optimize()?;

    // Phase 2: Check integer requirements
    for var in self.integer_vars() {
        let value = self.tableau.get_value(var);
        if !value.is_integer() {
            // Branch on fractional variable
            let branch_point = value.floor();
            self.branch_points.push((var, branch_point));
            return Err(Conflict::IntegerRequired);
        }
    }

    Ok(())
}
```

**Optimizations**:
- Gomory cuts for tighter bounds
- Branch-and-cut for faster convergence
- Preprocessing: GCD reduction, coefficient normalization

#### 3.3 Arrays (Theory of Arrays)

**Axioms**:
1. Read-over-write (same index): `select(store(a, i, v), i) = v`
2. Read-over-write (different): `i ≠ j → select(store(a, i, v), j) = select(a, j)`
3. Extensionality: `(∀i. select(a, i) = select(b, i)) → a = b`

**Lazy Instantiation**:
```rust
fn propagate_array_axioms(&mut self) -> Result<(), Conflict> {
    for (store_term, index, value) in self.pending_stores() {
        // Instantiate read-over-write axioms on demand
        for read_term in self.reads_of_array(store_term.base_array()) {
            if self.equal(read_term.index(), index) {
                self.assert_equality(read_term, value)?;
            }
        }
    }
    Ok(())
}
```

### 4. Theory Combination (Nelson-Oppen)

**Requirements**:
- Theories must be stably infinite
- Signatures must be disjoint (achieved via purification)

```rust
struct NelsonOppenCombiner {
    theories: Vec<Box<dyn Theory>>,
    shared_vars: HashMap<TermId, Vec<TheoryId>>,
    equality_graph: UnionFind,
}

fn propagate_equalities(&mut self) -> Result<(), Conflict> {
    let mut changed = true;
    while changed {
        changed = false;

        for theory in &mut self.theories {
            // Get new equalities from theory
            let new_eqs = theory.get_implied_equalities();

            for (a, b) in new_eqs {
                if self.equality_graph.find(a) != self.equality_graph.find(b) {
                    self.equality_graph.union(a, b);

                    // Propagate to all theories with shared variables
                    for other_theory in &mut self.theories {
                        other_theory.notify_equality(a, b)?;
                    }

                    changed = true;
                }
            }
        }
    }
    Ok(())
}
```

**Delayed Combination**:
- Defer equality propagation until necessary
- Reduces overhead for loosely-coupled formulas
- Check for conflicts before full propagation

## Control Flow

### Typical Solving Loop

```rust
fn solve(&mut self) -> SolverResult {
    loop {
        // 1. Boolean propagation
        match self.sat_solver.propagate() {
            Ok(()) => {},
            Err(conflict) => {
                if self.decision_level() == 0 {
                    return SolverResult::Unsat;
                }
                let (learned, level) = self.analyze_conflict(conflict);
                self.backtrack(level);
                self.add_clause(learned);
                continue;
            }
        }

        // 2. Check if boolean assignment is complete
        if self.sat_solver.all_assigned() {
            // 3. Theory check
            match self.theory_combination.check() {
                Ok(()) => return SolverResult::Sat,
                Err(theory_conflict) => {
                    let conflict_clause = self.explain_conflict(theory_conflict);
                    self.sat_solver.add_clause(conflict_clause);
                    continue;
                }
            }
        }

        // 4. Make decision
        match self.sat_solver.pick_decision() {
            Some(lit) => self.sat_solver.decide(lit),
            None => {
                // All variables assigned
                match self.theory_combination.check() {
                    Ok(()) => return SolverResult::Sat,
                    Err(conflict) => {
                        let clause = self.explain_conflict(conflict);
                        self.sat_solver.add_clause(clause);
                    }
                }
            }
        }

        // 5. Check resource limits
        if self.resources.check_limits().is_err() {
            return SolverResult::Unknown;
        }

        // 6. Consider restart
        if self.should_restart() {
            self.restart();
        }
    }
}
```

## Performance Optimizations

### 1. Memory Management

**Arena Allocation**:
- Bulk allocation for terms, clauses
- No per-object deallocation
- Better cache locality
- ~2-5x speedup for AST operations

**Object Pooling**:
- Reuse conflict clauses
- Preallocate watch lists
- Reduce allocation overhead

### 2. Incremental Solving

**Push/Pop Scopes**:
```rust
fn push(&mut self) {
    self.trail.push_scope();
    self.sat_solver.push_scope();
    for theory in &mut self.theories {
        theory.push_scope();
    }
}

fn pop(&mut self) {
    for theory in &mut self.theories {
        theory.pop_scope();
    }
    self.sat_solver.pop_scope();
    self.trail.pop_scope();
}
```

**Trail-based backtracking**:
- Record all state changes on trail
- O(1) backtracking by unwinding trail
- Efficient for incremental SMT

### 3. Parallel Solving

**Portfolio Approach**:
- Multiple solvers with different strategies
- Share learned clauses (unit clauses only)
- First to finish wins

**Cube-and-Conquer**:
- Lookahead phase: generate cubes (partial assignments)
- Conquer phase: solve cubes in parallel
- Work stealing for load balancing

## Proof Generation

```rust
enum ProofStep {
    Assumption(ClauseId),
    Resolution { left: ProofId, right: ProofId, pivot: Var },
    TheoryLemma { explanation: String },
    Conflict,
}

struct Proof {
    steps: Vec<ProofStep>,
    root: ProofId,
}
```

**Proof Formats**:
- Native: Binary, optimized for size
- DRAT: Deletion Resolution Asymmetric Tautology
- LFSC: Logical Framework with Side Conditions
- Lean: Interactive theorem prover format

## Statistics and Profiling

```rust
pub struct SolverStats {
    // SAT statistics
    pub decisions: u64,
    pub propagations: u64,
    pub conflicts: u64,
    pub restarts: u64,
    pub learned_clauses: usize,

    // Theory statistics
    pub theory_checks: u64,
    pub theory_propagations: u64,
    pub theory_conflicts: u64,

    // Performance
    pub time_sat: Duration,
    pub time_theory: Duration,
    pub time_preprocessing: Duration,
}
```

## References

1. **CDCL**: "Conflict-Driven Clause Learning SAT Solvers" (Handbook of SAT, 2009)
2. **VSIDS**: "Chaff: Engineering an Efficient SAT Solver" (DAC 2001)
3. **Nelson-Oppen**: "Simplification by Cooperating Decision Procedures" (TOPLAS 1979)
4. **DPLL(T)**: "Solving SAT and SAT Modulo Theories" (JACM 2006)
5. **Glucose**: "Predicting Learnt Clauses Quality in Modern SAT Solvers" (IJCAI 2009)
