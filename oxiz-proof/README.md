# oxiz-proof

Proof generation and checking for the OxiZ SMT solver.

## Overview

This crate provides machine-checkable proof output, enabling verification of solver results by external proof checkers. This is a **beyond Z3** feature - while Z3 supports generic proofs, OxiZ aims to generate standardized, machine-checkable proofs by default.

## Supported Formats

| Format | Description | Checker |
|:-------|:------------|:--------|
| **DRAT** | SAT resolution proofs | `drat-trim`, `cake_lpr` |
| **Alethe** | SMT-LIB proof format | `carcara`, `alethe-checker` |
| **LFSC** | Logical Framework proofs | `lfscc` |

## Why Machine-Checkable Proofs?

- **Trust**: Independent verification of solver correctness
- **Certification**: Required for safety-critical applications
- **Debugging**: Identify bugs in solver implementation
- **Interoperability**: Exchange proofs between tools

## Usage

```rust
use oxiz_proof::{Proof, ProofStep};

// Proofs are generated automatically during solving
let proof: Proof = solver.get_proof();

// Output in different formats
println!("{}", proof.to_drat());
println!("{}", proof.to_alethe());
```

## SMT-LIB2 Integration

```smt2
(set-option :produce-proofs true)
(assert (and p (not p)))
(check-sat)  ; Returns UNSAT
(get-proof)  ; Returns proof tree
```

## Proof Structure

```
(step t1 (cl (not p) q) :rule resolution :premises (t0))
(step t2 (cl (not q) r) :rule unit_resolution :premises (t1 h1))
(step t3 (cl false) :rule false_rule :premises (t2))
```

## References

- Barbosa, H., et al. (2022). Flexible proof production in an industrial-strength SMT solver. IJCAR.
- Wetzler, N., Heule, M., & Hunt, W. (2014). DRAT-trim: Efficient checking and trimming using expressive clausal proofs. SAT.

## License

MIT OR Apache-2.0
