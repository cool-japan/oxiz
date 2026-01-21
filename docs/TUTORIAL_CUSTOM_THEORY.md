# Tutorial: Extending OxiZ with Custom Theories

**Target Audience:** Developers familiar with Rust but new to SMT solvers

**Last Updated:** 2026-01-17

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theory Interface](#2-theory-interface)
3. [Step-by-Step Example: Set Theory](#3-step-by-step-example-set-theory)
4. [Integration with Solver](#4-integration-with-solver)
5. [Best Practices](#5-best-practices)
6. [Reference](#6-reference)

---

## 1. Introduction

### What is a Theory Solver in SMT?

An SMT (Satisfiability Modulo Theories) solver combines a SAT solver with specialized **theory solvers** to handle constraints beyond pure propositional logic. While the SAT solver manages Boolean variables and clauses, theory solvers understand domain-specific semantics.

For example, consider this formula:

```
(x > 5) AND (y = x + 2) AND (y < 3)
```

A pure SAT solver only sees three Boolean atoms (p1, p2, p3). It might assign:

```
p1 = true, p2 = true, p3 = true
```

But an **arithmetic theory solver** knows this is actually unsatisfiable:
- If x > 5, then x >= 6
- If y = x + 2, then y >= 8
- But y < 3 contradicts y >= 8

Theory solvers detect such semantic inconsistencies and generate **conflict clauses** (theory lemmas) that guide the SAT solver away from invalid assignments.

### Role of Theories in CDCL(T) Architecture

OxiZ implements the CDCL(T) architecture, which stands for **Conflict-Driven Clause Learning modulo Theories**. Here is how theories fit into the solving process:

```
                    +------------------+
                    |   SMT Formula    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Boolean Encoder |  (Tseitin transformation)
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
+--------+--------+                    +---------+--------+
|   CDCL SAT      |<--- Conflict ----->|  Theory Solvers  |
|   Solver        |      Lemmas        |  (EUF, LRA, ...) |
+-----------------+                    +------------------+
         |
         v
     SAT/UNSAT + Model/Core
```

**The main loop works as follows:**

1. The SAT solver makes **decisions** (assigns Boolean values to atoms)
2. Boolean Constraint Propagation (**BCP**) derives implied literals
3. Theory solvers **check** if current assignments are consistent
4. If inconsistent, theories produce a **conflict clause**
5. The SAT solver **learns** this clause and **backtracks**
6. Repeat until SAT (model found) or UNSAT (empty clause derived)

Theories also perform **propagation**: when they can derive new facts from partial assignments, they communicate these back to the SAT solver.

### Overview of OxiZ Theory Interface

OxiZ provides a clean trait-based interface for theory solvers in the `oxiz-theories` crate:

```rust
pub trait Theory: Send + Sync {
    /// Unique identifier for this theory
    fn id(&self) -> TheoryId;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Can this theory handle a given term?
    fn can_handle(&self, term: TermId) -> bool;

    /// Assert a term as true
    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult>;

    /// Assert a term as false
    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult>;

    /// Check consistency of current assertions
    fn check(&mut self) -> Result<TheoryResult>;

    /// Push a context level (for incremental solving)
    fn push(&mut self);

    /// Pop a context level
    fn pop(&mut self);

    /// Reset the theory solver
    fn reset(&mut self);

    /// Get the current model (if satisfiable)
    fn get_model(&self) -> Vec<(TermId, TermId)> { Vec::new() }
}
```

**Key concepts:**

- **TermId**: A unique identifier for AST terms (hash-consed)
- **TheoryResult**: Indicates SAT, UNSAT (with conflict), or propagations
- **Push/Pop**: Enable incremental solving with backtracking

---

## 2. Theory Interface

### The `Theory` Trait in Detail

Let us examine each method of the `Theory` trait:

```rust
/// Location: oxiz-theories/src/theory.rs

/// Unique identifier for a theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TheoryId {
    Bool,       // Boolean theory (always present)
    EUF,        // Equality with Uninterpreted Functions
    LRA,        // Linear Real Arithmetic
    LIA,        // Linear Integer Arithmetic
    BV,         // BitVectors
    Arrays,     // Extensional Arrays
    FP,         // Floating-Point (IEEE 754)
    Datatype,   // Algebraic Datatypes
    Strings,    // Strings and Regular Expressions
}

/// Result of a theory check
#[derive(Debug, Clone)]
pub enum TheoryResult {
    /// The theory constraints are satisfiable
    Sat,
    /// The theory constraints are unsatisfiable with explanation
    Unsat(Vec<TermId>),
    /// Need to propagate these literals: (consequence, reason)
    Propagate(Vec<(TermId, Vec<TermId>)>),
    /// Unknown (incomplete theory)
    Unknown,
}
```

### Key Methods Explained

#### `assert_true` and `assert_false`

These methods are called when the SAT solver assigns a value to a theory atom:

```rust
fn assert_true(&mut self, term: TermId) -> Result<TheoryResult>;
fn assert_false(&mut self, term: TermId) -> Result<TheoryResult>;
```

**Example:** If the SAT solver decides `(> x 5)` is true, the solver calls:

```rust
arith_solver.assert_true(term_for_x_gt_5)?;
```

The theory solver records this constraint and may immediately detect a conflict or trigger propagations.

#### `propagate`

Theory propagation derives new facts from current assignments. When a theory can infer that some unassigned atom must be true/false, it returns a `Propagate` result:

```rust
TheoryResult::Propagate(vec![
    (consequence_term, vec![reason_term1, reason_term2]),
])
```

**Example:** Given `x = y` and `y = z`, EUF propagates `x = z` (transitivity).

The **reason** is crucial for conflict analysis: if the propagation later leads to a conflict, the SAT solver needs to know which assertions caused it.

#### `check`

The main consistency check. Called after propagation reaches a fixpoint:

```rust
fn check(&mut self) -> Result<TheoryResult>;
```

Returns:
- `Sat`: All constraints are consistent
- `Unsat(conflict)`: Inconsistent; `conflict` contains the conflicting atoms
- `Propagate(...)`: New propagations discovered during check
- `Unknown`: Theory is incomplete (e.g., non-linear arithmetic heuristics)

#### `conflict` (via TheoryResult::Unsat)

When a theory detects inconsistency, it must explain **why**. The explanation is a set of term IDs representing the assertions that together cause the conflict:

```rust
TheoryResult::Unsat(vec![
    term_for_x_ge_10,  // x >= 10
    term_for_x_le_5,   // x <= 5
])
```

This becomes a clause: NOT(x >= 10) OR NOT(x <= 5), which the SAT solver learns.

### How Theory Lemmas Work

Theory lemmas are clauses derived from theory reasoning. They have two key properties:

1. **Soundness**: The lemma is logically valid (a tautology in the theory)
2. **Relevance**: The lemma helps exclude the current (partial) assignment

```
+-------------------+
|  SAT Assignment   |   p1=T, p2=T, p3=T (tentative)
+-------------------+
         |
         v
+-------------------+
|  Theory Check     |   Detects: p1 AND p2 => NOT p3
+-------------------+
         |
         v
+-------------------+
|  Theory Lemma     |   (NOT p1) OR (NOT p2) OR (NOT p3)
+-------------------+
         |
         v
+-------------------+
|  SAT Learning     |   Adds lemma, backtracks, avoids this assignment
+-------------------+
```

### Push/Pop for Incremental Solving

OxiZ supports incremental solving where constraints can be added/removed:

```rust
fn push(&mut self);  // Save current state
fn pop(&mut self);   // Restore to saved state
```

**Example usage:**

```rust
let mut solver = EufSolver::new();

// Base constraints
solver.merge(a, b, reason)?;  // a = b

solver.push();  // Save state

// Temporary constraint
solver.merge(b, c, reason)?;  // b = c
// Now a = b = c

solver.pop();   // Restore

// Back to just a = b
```

**Implementation typically uses:**
- A **trail** to record operations
- **Snapshot markers** at each push
- **Truncation** to restore state on pop

---

## 3. Step-by-Step Example: Set Theory

Let us implement a simple **Set Theory** solver that handles finite sets with operations like union, intersection, and membership.

### Defining Set Operations

Our set theory will support:

- **Membership**: `(element e S)` - e is in set S
- **Subset**: `(subset S1 S2)` - S1 is a subset of S2
- **Union**: `(union S1 S2)` - set union
- **Intersection**: `(inter S1 S2)` - set intersection
- **Singleton**: `(singleton e)` - set containing only e
- **Empty**: `empty` - the empty set

### Internal Representation

```rust
//! Set Theory Solver for OxiZ
//!
//! This is a tutorial implementation demonstrating how to create
//! a custom theory solver.

use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Set variable identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SetVar(pub u32);

/// Element identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Element(pub u32);

/// Set expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SetExpr {
    /// Named set variable
    Var(SetVar),
    /// Empty set
    Empty,
    /// Singleton {e}
    Singleton(Element),
    /// Union of two sets
    Union(Box<SetExpr>, Box<SetExpr>),
    /// Intersection of two sets
    Intersection(Box<SetExpr>, Box<SetExpr>),
}

/// A constraint in the set theory
#[derive(Debug, Clone)]
pub enum SetConstraint {
    /// Element e is in set S
    Member { element: Element, set: SetExpr },
    /// Element e is NOT in set S
    NotMember { element: Element, set: SetExpr },
    /// S1 is a subset of S2
    Subset { sub: SetExpr, sup: SetExpr },
    /// Two sets are equal
    Equal { lhs: SetExpr, rhs: SetExpr },
    /// Two sets are different
    NotEqual { lhs: SetExpr, rhs: SetExpr },
}

/// Constraint with its reason (for conflict explanation)
#[derive(Debug, Clone)]
struct TrackedConstraint {
    constraint: SetConstraint,
    reason: TermId,
}
```

### Implementing SetTheorySolver

```rust
/// Set Theory Solver
#[derive(Debug)]
pub struct SetTheorySolver {
    /// All constraints
    constraints: Vec<TrackedConstraint>,

    /// Known set memberships: set_var -> {elements}
    /// Positive memberships (e in S)
    positive_members: FxHashMap<SetVar, FxHashSet<Element>>,

    /// Negative memberships (e not in S)
    negative_members: FxHashMap<SetVar, FxHashSet<Element>>,

    /// Element universe (all elements mentioned)
    elements: FxHashSet<Element>,

    /// Set variables
    set_vars: FxHashSet<SetVar>,

    /// Term to set expression mapping
    term_to_expr: FxHashMap<TermId, SetExpr>,

    /// Term to element mapping
    term_to_element: FxHashMap<TermId, Element>,

    /// Context stack for push/pop
    context_stack: Vec<ContextState>,

    /// Next fresh element ID
    next_element: u32,

    /// Next fresh set variable ID
    next_set_var: u32,
}

/// State saved at each push
#[derive(Debug, Clone)]
struct ContextState {
    num_constraints: usize,
    positive_members_snapshot: FxHashMap<SetVar, usize>,
    negative_members_snapshot: FxHashMap<SetVar, usize>,
}

impl Default for SetTheorySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SetTheorySolver {
    /// Create a new set theory solver
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            positive_members: FxHashMap::default(),
            negative_members: FxHashMap::default(),
            elements: FxHashSet::default(),
            set_vars: FxHashSet::default(),
            term_to_expr: FxHashMap::default(),
            term_to_element: FxHashMap::default(),
            context_stack: Vec::new(),
            next_element: 0,
            next_set_var: 0,
        }
    }

    /// Create a fresh element
    pub fn fresh_element(&mut self) -> Element {
        let e = Element(self.next_element);
        self.next_element += 1;
        self.elements.insert(e);
        e
    }

    /// Create a fresh set variable
    pub fn fresh_set_var(&mut self) -> SetVar {
        let s = SetVar(self.next_set_var);
        self.next_set_var += 1;
        self.set_vars.insert(s);
        s
    }

    /// Intern a term as a set expression
    pub fn intern_set(&mut self, term: TermId) -> SetExpr {
        if let Some(expr) = self.term_to_expr.get(&term) {
            return expr.clone();
        }

        // In a real implementation, parse the term structure
        // For now, treat it as a fresh variable
        let var = self.fresh_set_var();
        let expr = SetExpr::Var(var);
        self.term_to_expr.insert(term, expr.clone());
        expr
    }

    /// Intern a term as an element
    pub fn intern_element(&mut self, term: TermId) -> Element {
        if let Some(&elem) = self.term_to_element.get(&term) {
            return elem;
        }

        let elem = self.fresh_element();
        self.term_to_element.insert(term, elem);
        elem
    }
```

### Assert Handling

```rust
    /// Assert: element e is a member of set S
    pub fn assert_member(
        &mut self,
        element: Element,
        set: SetExpr,
        reason: TermId,
    ) -> TheoryResult {
        self.constraints.push(TrackedConstraint {
            constraint: SetConstraint::Member {
                element,
                set: set.clone(),
            },
            reason,
        });

        // Record positive membership for base set variables
        if let SetExpr::Var(var) = &set {
            self.positive_members
                .entry(*var)
                .or_default()
                .insert(element);
        }

        // Eagerly check for conflicts
        if let Some(conflict) = self.check_member_conflict(element, &set) {
            return TheoryResult::Unsat(conflict);
        }

        TheoryResult::Sat
    }

    /// Assert: element e is NOT a member of set S
    pub fn assert_not_member(
        &mut self,
        element: Element,
        set: SetExpr,
        reason: TermId,
    ) -> TheoryResult {
        self.constraints.push(TrackedConstraint {
            constraint: SetConstraint::NotMember {
                element,
                set: set.clone(),
            },
            reason,
        });

        // Record negative membership
        if let SetExpr::Var(var) = &set {
            self.negative_members
                .entry(*var)
                .or_default()
                .insert(element);
        }

        // Check for conflicts
        if let Some(conflict) = self.check_member_conflict(element, &set) {
            return TheoryResult::Unsat(conflict);
        }

        TheoryResult::Sat
    }

    /// Assert: S1 is a subset of S2
    pub fn assert_subset(
        &mut self,
        sub: SetExpr,
        sup: SetExpr,
        reason: TermId,
    ) -> TheoryResult {
        self.constraints.push(TrackedConstraint {
            constraint: SetConstraint::Subset { sub, sup },
            reason,
        });

        // Subset checking happens in full check
        TheoryResult::Sat
    }
```

### Propagation Rules

```rust
    /// Perform theory propagation
    ///
    /// Propagation rules for sets:
    /// 1. If e in (union S1 S2), then e in S1 OR e in S2
    /// 2. If e in (inter S1 S2), then e in S1 AND e in S2
    /// 3. If e in (singleton e'), then e = e'
    /// 4. If S1 subset S2 and e in S1, then e in S2
    pub fn propagate(&mut self) -> Vec<(TermId, Vec<TermId>)> {
        let mut propagations = Vec::new();

        // Rule: e in (inter S1 S2) => e in S1 AND e in S2
        for tracked in &self.constraints {
            if let SetConstraint::Member { element, set } = &tracked.constraint {
                if let SetExpr::Intersection(s1, s2) = set {
                    // Propagate membership to both components
                    // (In full impl, would create actual propagation terms)
                    let _ = (element, s1, s2);
                }
            }
        }

        // Rule: S1 subset S2 and e in S1 => e in S2
        for tracked in &self.constraints {
            if let SetConstraint::Subset { sub, sup } = &tracked.constraint {
                if let SetExpr::Var(sub_var) = sub {
                    if let Some(members) = self.positive_members.get(sub_var) {
                        for &elem in members {
                            // elem is in sub, so it must be in sup
                            let _ = (elem, sup, tracked.reason);
                            // Would add propagation here
                        }
                    }
                }
            }
        }

        propagations
    }
```

### Conflict Detection

```rust
    /// Check for membership conflicts
    ///
    /// Conflicts arise when:
    /// - e in S and e not in S
    /// - e in empty
    fn check_member_conflict(
        &self,
        element: Element,
        set: &SetExpr,
    ) -> Option<Vec<TermId>> {
        match set {
            SetExpr::Empty => {
                // Element in empty set is always a conflict
                // Find the reason for this membership assertion
                for tracked in &self.constraints {
                    if let SetConstraint::Member { element: e, set: s } =
                        &tracked.constraint
                    {
                        if *e == element && s == &SetExpr::Empty {
                            return Some(vec![tracked.reason]);
                        }
                    }
                }
                None
            }

            SetExpr::Var(var) => {
                // Check if element is both in and not in the set
                let in_positive = self
                    .positive_members
                    .get(var)
                    .is_some_and(|members| members.contains(&element));

                let in_negative = self
                    .negative_members
                    .get(var)
                    .is_some_and(|members| members.contains(&element));

                if in_positive && in_negative {
                    // Find the reasons
                    let mut reasons = Vec::new();
                    for tracked in &self.constraints {
                        match &tracked.constraint {
                            SetConstraint::Member { element: e, set: s }
                                if *e == element && s == &SetExpr::Var(*var) =>
                            {
                                reasons.push(tracked.reason);
                            }
                            SetConstraint::NotMember { element: e, set: s }
                                if *e == element && s == &SetExpr::Var(*var) =>
                            {
                                reasons.push(tracked.reason);
                            }
                            _ => {}
                        }
                    }
                    if !reasons.is_empty() {
                        return Some(reasons);
                    }
                }
                None
            }

            SetExpr::Singleton(only_elem) => {
                // e in {e'} is only valid if e = e'
                // This would require equality theory integration
                if element != *only_elem {
                    // Conflict: trying to put wrong element in singleton
                    for tracked in &self.constraints {
                        if let SetConstraint::Member { element: e, set: s } =
                            &tracked.constraint
                        {
                            if *e == element && s == set {
                                return Some(vec![tracked.reason]);
                            }
                        }
                    }
                }
                None
            }

            _ => None, // Other cases handled in full check
        }
    }

    /// Full consistency check
    pub fn full_check(&mut self) -> TheoryResult {
        // Check all membership constraints
        for i in 0..self.constraints.len() {
            let tracked = self.constraints[i].clone();
            match &tracked.constraint {
                SetConstraint::Member { element, set } => {
                    if let Some(conflict) =
                        self.check_member_conflict(*element, set)
                    {
                        return TheoryResult::Unsat(conflict);
                    }
                }
                SetConstraint::NotMember { element, set } => {
                    // Check if we also assert membership
                    if self.is_member(*element, set) {
                        return TheoryResult::Unsat(vec![tracked.reason]);
                    }
                }
                SetConstraint::Subset { sub, sup } => {
                    // Check all elements of sub are in sup
                    if let Some(conflict) = self.check_subset(sub, sup) {
                        return TheoryResult::Unsat(conflict);
                    }
                }
                _ => {}
            }
        }

        TheoryResult::Sat
    }

    /// Check if element is known to be in set
    fn is_member(&self, element: Element, set: &SetExpr) -> bool {
        match set {
            SetExpr::Var(var) => self
                .positive_members
                .get(var)
                .is_some_and(|m| m.contains(&element)),
            SetExpr::Singleton(e) => element == *e,
            SetExpr::Empty => false,
            SetExpr::Union(s1, s2) => {
                self.is_member(element, s1) || self.is_member(element, s2)
            }
            SetExpr::Intersection(s1, s2) => {
                self.is_member(element, s1) && self.is_member(element, s2)
            }
        }
    }

    /// Check subset constraint
    fn check_subset(
        &self,
        sub: &SetExpr,
        sup: &SetExpr,
    ) -> Option<Vec<TermId>> {
        // Get all known members of sub
        if let SetExpr::Var(sub_var) = sub {
            if let Some(members) = self.positive_members.get(sub_var) {
                for &elem in members {
                    // Check elem is in sup
                    if !self.is_member(elem, sup) && self.is_not_member(elem, sup) {
                        // Found violation: elem in sub but not in sup
                        let mut reasons = Vec::new();
                        for tracked in &self.constraints {
                            match &tracked.constraint {
                                SetConstraint::Member { element, set }
                                    if *element == elem
                                        && set == &SetExpr::Var(*sub_var) =>
                                {
                                    reasons.push(tracked.reason);
                                }
                                SetConstraint::Subset { sub: s1, sup: s2 }
                                    if s1 == sub && s2 == sup =>
                                {
                                    reasons.push(tracked.reason);
                                }
                                SetConstraint::NotMember { element, set }
                                    if *element == elem && set == sup =>
                                {
                                    reasons.push(tracked.reason);
                                }
                                _ => {}
                            }
                        }
                        if !reasons.is_empty() {
                            return Some(reasons);
                        }
                    }
                }
            }
        }
        None
    }

    /// Check if element is known to NOT be in set
    fn is_not_member(&self, element: Element, set: &SetExpr) -> bool {
        match set {
            SetExpr::Var(var) => self
                .negative_members
                .get(var)
                .is_some_and(|m| m.contains(&element)),
            SetExpr::Empty => true,
            _ => false,
        }
    }
```

### Implementing the Theory Trait

```rust
use crate::theory::{Theory, TheoryId, TheoryResult};

impl Theory for SetTheorySolver {
    fn id(&self) -> TheoryId {
        // In a real implementation, you'd add a new TheoryId variant
        // For tutorial purposes, we'll use a placeholder
        TheoryId::Datatype // Placeholder
    }

    fn name(&self) -> &str {
        "Sets"
    }

    fn can_handle(&self, term: TermId) -> bool {
        // Check if term is a set operation
        // In real impl, inspect term structure
        self.term_to_expr.contains_key(&term)
            || self.term_to_element.contains_key(&term)
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // Parse term and add appropriate constraint
        // This is simplified - real impl parses term structure
        let elem = self.intern_element(term);
        let set = SetExpr::Empty; // Placeholder
        Ok(self.assert_member(elem, set, term))
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        let elem = self.intern_element(term);
        let set = SetExpr::Empty; // Placeholder
        Ok(self.assert_not_member(elem, set, term))
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // First propagate
        let _props = self.propagate();

        // Then do full check
        Ok(self.full_check())
    }

    fn push(&mut self) {
        // Save current state
        let mut pos_snapshot = FxHashMap::default();
        for (var, members) in &self.positive_members {
            pos_snapshot.insert(*var, members.len());
        }

        let mut neg_snapshot = FxHashMap::default();
        for (var, members) in &self.negative_members {
            neg_snapshot.insert(*var, members.len());
        }

        self.context_stack.push(ContextState {
            num_constraints: self.constraints.len(),
            positive_members_snapshot: pos_snapshot,
            negative_members_snapshot: neg_snapshot,
        });
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Restore constraints
            self.constraints.truncate(state.num_constraints);

            // Restore positive memberships
            for (var, size) in state.positive_members_snapshot {
                if let Some(members) = self.positive_members.get_mut(&var) {
                    // Note: FxHashSet doesn't support truncate
                    // In production, use a vector-backed set
                    while members.len() > size {
                        if let Some(&elem) = members.iter().next() {
                            members.remove(&elem);
                        }
                    }
                }
            }

            // Restore negative memberships (same approach)
            for (var, size) in state.negative_members_snapshot {
                if let Some(members) = self.negative_members.get_mut(&var) {
                    while members.len() > size {
                        if let Some(&elem) = members.iter().next() {
                            members.remove(&elem);
                        }
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.constraints.clear();
        self.positive_members.clear();
        self.negative_members.clear();
        self.context_stack.clear();
        self.term_to_expr.clear();
        self.term_to_element.clear();
    }
}
```

### Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_membership() {
        let mut solver = SetTheorySolver::new();

        let e1 = solver.fresh_element();
        let s1 = solver.fresh_set_var();
        let set = SetExpr::Var(s1);

        // Assert e1 in S1
        let result = solver.assert_member(e1, set.clone(), TermId::new(1));
        assert!(matches!(result, TheoryResult::Sat));

        // Check should pass
        let result = solver.full_check();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_membership_conflict() {
        let mut solver = SetTheorySolver::new();

        let e1 = solver.fresh_element();
        let s1 = solver.fresh_set_var();
        let set = SetExpr::Var(s1);

        // Assert e1 in S1
        solver.assert_member(e1, set.clone(), TermId::new(1));

        // Assert e1 NOT in S1 -> conflict!
        let result = solver.assert_not_member(e1, set, TermId::new(2));
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_empty_set_conflict() {
        let mut solver = SetTheorySolver::new();

        let e1 = solver.fresh_element();

        // Assert e1 in empty -> immediate conflict
        let result = solver.assert_member(e1, SetExpr::Empty, TermId::new(1));
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_subset_violation() {
        let mut solver = SetTheorySolver::new();

        let e1 = solver.fresh_element();
        let s1 = solver.fresh_set_var();
        let s2 = solver.fresh_set_var();

        // Assert S1 subset S2
        solver.assert_subset(
            SetExpr::Var(s1),
            SetExpr::Var(s2),
            TermId::new(1),
        );

        // Assert e1 in S1
        solver.assert_member(e1, SetExpr::Var(s1), TermId::new(2));

        // Assert e1 NOT in S2 -> violates subset!
        solver.assert_not_member(e1, SetExpr::Var(s2), TermId::new(3));

        let result = solver.full_check();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_push_pop() {
        let mut solver = SetTheorySolver::new();

        let e1 = solver.fresh_element();
        let s1 = solver.fresh_set_var();
        let set = SetExpr::Var(s1);

        // Assert e1 in S1
        solver.assert_member(e1, set.clone(), TermId::new(1));

        solver.push();

        // Assert e1 NOT in S1 -> conflict
        let result = solver.assert_not_member(e1, set.clone(), TermId::new(2));
        assert!(matches!(result, TheoryResult::Unsat(_)));

        solver.pop();

        // After pop, should be satisfiable again
        let result = solver.full_check();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_singleton_membership() {
        let mut solver = SetTheorySolver::new();

        let e1 = solver.fresh_element();
        let e2 = solver.fresh_element();

        // e1 in {e1} should be fine
        let result = solver.assert_member(
            e1,
            SetExpr::Singleton(e1),
            TermId::new(1),
        );
        assert!(matches!(result, TheoryResult::Sat));

        // e2 in {e1} is a conflict (e2 != e1)
        let result = solver.assert_member(
            e2,
            SetExpr::Singleton(e1),
            TermId::new(2),
        );
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }
}
```

---

## 4. Integration with Solver

### How to Register a Custom Theory

To use your theory with the CDCL(T) solver, you need to register it with the `TheoryCombiner`:

```rust
use oxiz_theories::combination::TheoryCombiner;
use oxiz_theories::TheoryId;

// 1. Create theory instances
let mut combiner = TheoryCombiner::new();
let set_solver = SetTheorySolver::new();

// 2. Register with the theory manager
// (In a full implementation, TheoryCombiner would be extended)
// combiner.register_theory(Box::new(set_solver));

// 3. The solver will now:
//    - Route set-related terms to your theory
//    - Call assert_true/assert_false on assignments
//    - Call check() periodically
//    - Learn from your conflict explanations
```

### Theory Combination (Nelson-Oppen)

When multiple theories share variables, they must cooperate. OxiZ implements **Nelson-Oppen** theory combination:

```
+----------+       Shared Variables       +----------+
|   EUF    | <--------------------------> |   Sets   |
| (x = y)  |      Equality Exchange       | (x in S) |
+----------+                              +----------+
```

**Key requirement:** Theories must exchange information about **shared variables**.

Implement the `TheoryCombination` trait:

```rust
use crate::theory::{EqualityNotification, TheoryCombination};

impl TheoryCombination for SetTheorySolver {
    fn notify_equality(&mut self, eq: EqualityNotification) -> bool {
        // Another theory derived that eq.lhs = eq.rhs
        // Update internal state accordingly

        // If we have e1 in S and e1 = e2, then e2 in S too
        let lhs = self.term_to_element.get(&eq.lhs);
        let rhs = self.term_to_element.get(&eq.rhs);

        if let (Some(&elem_lhs), Some(&elem_rhs)) = (lhs, rhs) {
            // Merge equivalence classes of elements
            // This would require union-find for elements
            return true;
        }

        false
    }

    fn get_shared_equalities(&self) -> Vec<EqualityNotification> {
        // Return equalities derived by this theory that should
        // be shared with other theories
        Vec::new()
    }

    fn is_relevant(&self, term: TermId) -> bool {
        self.can_handle(term)
    }
}
```

### Performance Considerations

**1. Incremental Checking**

Avoid re-checking everything on each `check()` call:

```rust
struct IncrementalSetSolver {
    // Track what changed since last check
    dirty_constraints: Vec<usize>,
    last_checked: usize,
}

impl IncrementalSetSolver {
    fn check(&mut self) -> TheoryResult {
        // Only check constraints added since last check
        for i in self.last_checked..self.constraints.len() {
            if let Some(conflict) = self.check_constraint(i) {
                return TheoryResult::Unsat(conflict);
            }
        }
        self.last_checked = self.constraints.len();
        TheoryResult::Sat
    }
}
```

**2. Watch Lists**

For large constraint sets, maintain watch lists:

```rust
// Watch list: for each element, which constraints mention it?
watches: FxHashMap<Element, Vec<usize>>,

fn on_membership_change(&self, elem: Element) {
    if let Some(constraint_indices) = self.watches.get(&elem) {
        for &idx in constraint_indices {
            // Only re-check relevant constraints
            self.dirty_constraints.push(idx);
        }
    }
}
```

**3. Lazy vs Eager Conflict Detection**

- **Eager:** Check for conflicts immediately on each assertion
  - Pro: Detects conflicts early, enables early pruning
  - Con: More overhead per assertion

- **Lazy:** Defer checking to explicit `check()` calls
  - Pro: Less overhead when many assertions happen together
  - Con: May do unnecessary work before finding conflicts

Choose based on your theory's characteristics.

---

## 5. Best Practices

### Efficient Data Structures

**1. Use Union-Find for Equivalences**

```rust
// Instead of maintaining explicit equivalence classes
use crate::euf::UnionFind;

struct ElementEquiv {
    uf: UnionFind,
    elem_to_node: FxHashMap<Element, u32>,
}

impl ElementEquiv {
    fn same_class(&mut self, e1: Element, e2: Element) -> bool {
        let n1 = self.elem_to_node[&e1];
        let n2 = self.elem_to_node[&e2];
        self.uf.same(n1, n2)
    }
}
```

**2. Use SmallVec for Small Collections**

```rust
use smallvec::SmallVec;

// Most constraints involve few elements
struct Constraint {
    elements: SmallVec<[Element; 4]>,  // Stack-allocated for <= 4 elements
}
```

**3. Use FxHashMap/FxHashSet**

```rust
use rustc_hash::{FxHashMap, FxHashSet};

// Faster than std HashMap for integer keys
members: FxHashMap<SetVar, FxHashSet<Element>>,
```

### Lazy vs Eager Propagation

**Eager Propagation:**
```rust
fn assert_member(&mut self, elem: Element, set: SetExpr, reason: TermId) {
    // Immediately propagate all consequences
    match set {
        SetExpr::Intersection(s1, s2) => {
            // e in (S1 & S2) => e in S1 AND e in S2
            self.assert_member(elem, *s1, reason);
            self.assert_member(elem, *s2, reason);
        }
        // ...
    }
}
```

**Lazy Propagation:**
```rust
fn assert_member(&mut self, elem: Element, set: SetExpr, reason: TermId) {
    // Just record the constraint
    self.constraints.push(MemberConstraint { elem, set, reason });
    // Propagation happens in propagate() call
}

fn propagate(&mut self) -> Vec<Propagation> {
    // Derive consequences lazily
    // ...
}
```

**Guideline:** Use eager propagation for cheap derivations, lazy for expensive ones.

### Explanation Generation

Good explanations are crucial for efficient conflict learning:

**1. Minimal Explanations**

```rust
fn explain_conflict(&self, conflict: &Conflict) -> Vec<TermId> {
    let mut reasons = Vec::new();

    // Only include relevant reasons
    for constraint in &self.constraints {
        if constraint.contributes_to(conflict) {
            reasons.push(constraint.reason);
        }
    }

    // Remove redundant reasons if possible
    self.minimize_reasons(&mut reasons);

    reasons
}
```

**2. Proof-Aware Explanations**

Track the derivation chain:

```rust
struct DerivedFact {
    fact: SetConstraint,
    derived_from: Vec<usize>,  // Indices of parent constraints
    reason: TermId,
}
```

### Testing Strategies

**1. Unit Tests for Core Operations**

```rust
#[test]
fn test_union_membership() {
    // e in S1 => e in (S1 union S2)
}

#[test]
fn test_intersection_membership() {
    // e in (S1 inter S2) <=> e in S1 AND e in S2
}
```

**2. Property-Based Testing**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn subset_transitivity(
        e in any::<u32>(),
        s1 in any::<u32>(),
        s2 in any::<u32>(),
        s3 in any::<u32>(),
    ) {
        // If S1 subset S2 and S2 subset S3, then S1 subset S3
    }
}
```

**3. Fuzzing with Random Constraints**

```rust
#[test]
fn fuzz_random_constraints() {
    let mut rng = rand::rng();

    for _ in 0..1000 {
        let mut solver = SetTheorySolver::new();
        // Generate random constraints
        // Check that solver doesn't crash
        // Verify consistency of results
    }
}
```

**4. Regression Tests from SMT-LIB Benchmarks**

Create `.smt2` files exercising your theory and run them through the full solver.

---

## 6. Reference

### Existing Theory Implementations in OxiZ

| Theory | Location | Key Algorithms |
|--------|----------|----------------|
| EUF | `oxiz-theories/src/euf/` | Congruence closure, union-find, E-matching |
| LRA/LIA | `oxiz-theories/src/arithmetic/` | Simplex, Farkas lemmas, branch-and-bound |
| BitVectors | `oxiz-theories/src/bv/` | Bit-blasting, word-level propagation |
| Arrays | `oxiz-theories/src/array/` | Read-over-write axioms |
| Strings | `oxiz-theories/src/string/` | Brzozowski derivatives, automata |
| Datatypes | `oxiz-theories/src/datatype/` | Algebraic datatypes, selectors |
| Diff Logic | `oxiz-theories/src/diff_logic/` | Bellman-Ford graph algorithm |
| UTVPI | `oxiz-theories/src/utvpi/` | Doubled graph for +-x +-y <= c |

### Relevant Papers

1. **Nelson-Oppen Theory Combination**
   - Nelson & Oppen, "Simplification by Cooperating Decision Procedures" (1979)
   - de Moura & Bjorner, "Model-based Theory Combination" (2007)

2. **DPLL(T) Architecture**
   - Nieuwenhuis, Oliveras, Tinelli, "Solving SAT and SAT Modulo Theories" (2006)

3. **Set Theory in SMT**
   - Cristia & Rossi, "A Decision Procedure for Restricted Intensional Sets" (2020)
   - Yessenov, Piskac, Kuncak, "Collections, Cardinalities, and Relations" (2010)

4. **Efficient Explanation Generation**
   - de Moura & Bjorner, "Proofs and Refutations..." (2008)

### API Documentation

- **oxiz-theories crate docs**: `cargo doc -p oxiz-theories --open`
- **Theory trait**: `oxiz_theories::theory::Theory`
- **Theory combination**: `oxiz_theories::combination::TheoryCombiner`
- **User propagator**: `oxiz_theories::user_propagator::UserPropagator`

### Getting Help

- Check existing theory implementations for patterns
- Look at test cases in `oxiz-theories/src/*/tests.rs`
- Review the architecture document: `docs/ARCHITECTURE.md`

---

*This tutorial is part of the OxiZ project documentation.*
