# oxiz-spacer

Property Directed Reachability (PDR/IC3) engine for OxiZ - Horn clause solving.

## Overview

Spacer is a solver for Constrained Horn Clauses (CHC), essential for software verification. It implements the PDR (Property Directed Reachability) algorithm, also known as IC3.

**This is a key differentiator for OxiZ** - Spacer is missing in most Z3 clones, making OxiZ uniquely suitable for verification toolchains.

## Use Cases

- **Loop Invariant Inference**: Automatically discover loop invariants
- **Safety Verification**: Prove safety properties of programs
- **Model Checking**: Bounded and unbounded verification
- **Regression Verification**: Verify program equivalence
- **Interpolation**: Generate Craig interpolants for refinement

## Constrained Horn Clauses (CHC)

CHC is a fragment of first-order logic widely used in verification:

```
; Example: simple loop invariant problem
(declare-rel inv (Int))
(declare-var x Int)

; Initial: x = 0 => inv(x)
(rule (=> (= x 0) (inv x)))

; Transition: inv(x) ∧ x < 10 => inv(x+1)
(rule (=> (and (inv x) (< x 10)) (inv (+ x 1))))

; Query: inv(x) ∧ x >= 10 => x = 10
(query (and (inv x) (>= x 10) (not (= x 10))))
```

## Key Components

| Module | Description |
|:-------|:------------|
| `pdr` | Main PDR/IC3 algorithm |
| `chc` | CHC representation and parsing |
| `frames` | Frame sequence management |
| `pob` | Proof obligation tracking |
| `reach` | Reachability analysis |

## Algorithm Overview

1. **Initialize**: F_0 = Init, F_1 = ... = F_N = True
2. **Block**: For each frame, block bad states
3. **Propagate**: Push learned clauses forward
4. **Check Fixpoint**: If F_i = F_{i+1}, property holds
5. **Counterexample**: If bad state reaches F_0, property fails

## References

- Bradley, A. R. (2011). SAT-Based Model Checking without Unrolling. VMCAI.
- Een, N., Mishchenko, A., & Brayton, R. (2011). Efficient implementation of property directed reachability. FMCAD.
- Komuravelli, A., Gurfinkel, A., & Chaki, S. (2014). SMT-based model checking for recursive programs. CAV.

## License

MIT OR Apache-2.0
