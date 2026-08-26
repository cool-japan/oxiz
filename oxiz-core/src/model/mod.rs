//! Model Subsystem
//!
//! Provides model construction, evaluation, and manipulation for SMT solving.
//!
//! # Components
//!
//! - **Evaluator**: Evaluates terms under a given model assignment
//! - **Completion**: Completes partial models with default values
//! - **Implicant**: Extracts minimal satisfying assignments (prime implicants)
//! - **Factory**: Creates default values for different sorts
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::model::{Model, ModelEvaluator};
//!
//! let mut model = Model::new();
//! model.assign(x, Value::Int(42));
//! model.assign(y, Value::Bool(true));
//!
//! let evaluator = ModelEvaluator::new(&model);
//! let result = evaluator.eval(expr)?;
//! ```

mod completion;
mod evaluator;
mod factory;
mod implicant;

pub use completion::{ModelCompletion, ModelCompletionConfig};
pub use evaluator::{EvalCache, EvalResult, ModelEvaluator};
pub use factory::{ValueFactory, ValueFactoryConfig};
pub use implicant::{ImplicantConfig, ImplicantExtractor, PrimeImplicant};

use crate::ast::{RoundingMode, TermId};
use crate::interner::Spur;
use crate::prelude::HashMap;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortKind, SortManager};
use num_rational::Rational64;

/// A value in the model
///
/// # Deeply nested values
///
/// [`Value::Array`] and [`Value::Datatype`] own their children *by value*, so
/// a value's nesting depth is bounded only by what built it — an array whose
/// exception values are themselves arrays nests one level per `store`. Every
/// structural trait is therefore written by hand and driven by an explicit
/// heap worklist instead of being derived: a derived `Drop`, `Clone`,
/// `PartialEq` or `Debug` recurses once per level and kills the *process* with
/// `fatal runtime error: stack overflow` — an abort, not an error a caller can
/// handle — from about 20 000 levels down on the ~1 MiB stack an embedder's
/// worker thread typically gets. `Drop` is the sharpest edge of the four: it
/// runs even when the value was built and walked entirely iteratively.
///
/// The representation itself is unchanged; only the trait implementations are.
pub enum Value {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Rational value
    Rational(Rational64),
    /// Bitvector value (width, value)
    BitVec(u32, u64),
    /// String value
    String(String),
    /// Array value (default, exceptions)
    Array(Box<Value>, Vec<(Value, Value)>),
    /// Datatype constructor (constructor id, arguments)
    Datatype(u32, Vec<Value>),
    /// Floating-point value (sign, exponent, mantissa)
    FloatingPoint(bool, u64, u64),
    /// One of the five IEEE 754 rounding modes.
    ///
    /// [`SortKind::RoundingMode`] is a *finite* built-in sort, so its values
    /// are named constants, not abstract domain elements. Modelling them as
    /// [`Value::Uninterpreted`] would print a `@uc_RoundingMode_0`-style
    /// witness and compare unequal to the very mode it stands for.
    RoundingMode(RoundingMode),
    /// Uninterpreted value (id for unique representation)
    Uninterpreted(u64),
    /// Undefined (no assignment)
    Undefined,
}

// ===== Structural traits: iterative, never recursive =====
//
// Everything in this section walks a `Value` with an explicit `Vec` worklist,
// for the reason spelled out on `Value` itself. The shared entry point is
// `Value::take_children`, which detaches a node's children so the caller can
// drop or inspect them at a flat depth.

impl Value {
    /// Whether this node owns child values at all.
    ///
    /// The leaf variants are the overwhelmingly common case, so every walk
    /// below short-circuits on them before allocating a worklist.
    fn has_children(&self) -> bool {
        // An array always owns its boxed default element, empty exception
        // list or not.
        matches!(self, Value::Array(_, _) | Value::Datatype(_, _))
    }

    /// Move this node's *compound* children onto `out`.
    ///
    /// Used by [`Drop`]: after this returns the node owns only leaves, whose
    /// own drop cannot recurse, so dropping the node is flat. Leaf children
    /// are deliberately left in place — pushing them would cost a worklist
    /// allocation per node and buy nothing.
    fn take_children(&mut self, out: &mut Vec<Value>) {
        match self {
            Value::Array(default, excs) => {
                if default.has_children() {
                    out.push(core::mem::replace(default.as_mut(), Value::Undefined));
                }
                out.extend(
                    excs.drain(..)
                        .flat_map(|(key, value)| [key, value])
                        .filter(Value::has_children),
                );
            }
            Value::Datatype(_, args) => {
                out.extend(args.drain(..).filter(Value::has_children));
            }
            _ => {}
        }
    }

    /// Shallow copy of a node, with its children left behind.
    ///
    /// Only ever called on a variant that owns no children, or by [`Clone`]
    /// on a compound node whose children it rebuilds separately.
    fn clone_leaf(&self) -> Value {
        match self {
            Value::Bool(b) => Value::Bool(*b),
            Value::Int(i) => Value::Int(*i),
            Value::Rational(r) => Value::Rational(*r),
            Value::BitVec(w, v) => Value::BitVec(*w, *v),
            Value::String(s) => Value::String(s.clone()),
            Value::FloatingPoint(sign, exp, mant) => Value::FloatingPoint(*sign, *exp, *mant),
            Value::RoundingMode(rm) => Value::RoundingMode(*rm),
            Value::Uninterpreted(id) => Value::Uninterpreted(*id),
            // The two compound variants are rebuilt by `Clone` from their
            // already-cloned children; reaching here would drop them.
            Value::Array(_, _) => Value::Array(Box::new(Value::Undefined), Vec::new()),
            Value::Datatype(id, _) => Value::Datatype(*id, Vec::new()),
            Value::Undefined => Value::Undefined,
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if !self.has_children() {
            return;
        }
        // Detach the children into a flat worklist and drop them one at a
        // time. Each popped node is itself detached before it goes out of
        // scope, so its own `drop` sees a childless husk and returns at a
        // constant native-stack depth.
        let mut worklist: Vec<Value> = Vec::new();
        self.take_children(&mut worklist);
        while let Some(mut node) = worklist.pop() {
            node.take_children(&mut worklist);
        }
    }
}

/// One step of the iterative [`Clone`] walk.
enum CloneStep<'a> {
    /// Clone this node, scheduling its children first.
    Visit(&'a Value),
    /// Rebuild a [`Value::Array`] from the child clones at `out[base..]`,
    /// which are laid out as `[default, key0, value0, key1, value1, ...]`.
    FinishArray {
        /// Where this node's child clones start in the output stack.
        base: usize,
    },
    /// Rebuild a [`Value::Datatype`] from the argument clones at `out[base..]`.
    FinishDatatype {
        /// The constructor id, carried across the walk.
        id: u32,
        /// Where this node's child clones start in the output stack.
        base: usize,
    },
}

impl Clone for Value {
    fn clone(&self) -> Self {
        if !self.has_children() {
            return self.clone_leaf();
        }
        let mut steps: Vec<CloneStep<'_>> = vec![CloneStep::Visit(self)];
        // Finished child clones, in completion order. A `Finish*` step owns
        // exactly the tail from the `base` it recorded when it was scheduled.
        let mut out: Vec<Value> = Vec::new();
        while let Some(step) = steps.pop() {
            match step {
                CloneStep::Visit(Value::Array(default, excs)) => {
                    steps.push(CloneStep::FinishArray { base: out.len() });
                    // Pushed in reverse so the children pop — and therefore
                    // finish — in `[default, key0, value0, ...]` order.
                    for (key, value) in excs.iter().rev() {
                        steps.push(CloneStep::Visit(value));
                        steps.push(CloneStep::Visit(key));
                    }
                    steps.push(CloneStep::Visit(default));
                }
                CloneStep::Visit(Value::Datatype(id, args)) => {
                    steps.push(CloneStep::FinishDatatype {
                        id: *id,
                        base: out.len(),
                    });
                    for arg in args.iter().rev() {
                        steps.push(CloneStep::Visit(arg));
                    }
                }
                CloneStep::Visit(leaf) => out.push(leaf.clone_leaf()),
                CloneStep::FinishArray { base } => {
                    let mut parts = out.split_off(base).into_iter();
                    match parts.next() {
                        Some(default) => {
                            let mut excs = Vec::with_capacity(parts.len() / 2);
                            while let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                                excs.push((key, value));
                            }
                            out.push(Value::Array(Box::new(default), excs));
                        }
                        // Unreachable: `FinishArray` is scheduled together
                        // with the `Visit` that produces its default element.
                        None => out.push(Value::Array(Box::new(Value::Undefined), Vec::new())),
                    }
                }
                CloneStep::FinishDatatype { id, base } => {
                    let args = out.split_off(base);
                    out.push(Value::Datatype(id, args));
                }
            }
        }
        match out.pop() {
            Some(root) => root,
            // Unreachable: the root `Visit` always produces exactly one value.
            None => Value::Undefined,
        }
    }
}

impl PartialEq for Value {
    /// Structural equality, with one deliberate numeric bridge: an
    /// [`Value::Int`] and an integral [`Value::Rational`] denoting the same
    /// number compare equal.
    ///
    /// Without it the evaluator contradicts itself on well-sorted input. A
    /// Real literal evaluates to `Rational(1/1)` while every computed Real
    /// result is funnelled through the `denom == 1 => Value::Int` normalization
    /// in the evaluator's `from_rational`, so `(= 1.0 (+ 0.5 0.5))` compared
    /// `Rational(1/1)` against `Int(1)` and answered `false` — a wrong verdict
    /// for `=`, and the same wrong answer for `distinct`, for `select`'s index
    /// lookup and for [`FuncInterp::evaluate`]'s argument table, all of which
    /// compare `Value`s with `==`. Nothing relies on the two shapes being
    /// distinguishable: SMT-LIB's `Int` and `Real` are separate sorts, so a
    /// well-sorted `=` never compares a value of one against a value of the
    /// other.
    fn eq(&self, other: &Self) -> bool {
        /// Compare two nodes without descending into children.
        ///
        /// Returns `false` for a compound node, which the caller handles
        /// instead, and for any mismatch of variants.
        fn leaf_eq(a: &Value, b: &Value) -> bool {
            match (a, b) {
                (Value::Bool(x), Value::Bool(y)) => x == y,
                (Value::Int(x), Value::Int(y)) => x == y,
                (Value::Rational(x), Value::Rational(y)) => x == y,
                (Value::Int(n), Value::Rational(r)) | (Value::Rational(r), Value::Int(n)) => {
                    r.is_integer() && r.numer() == n
                }
                (Value::BitVec(w1, v1), Value::BitVec(w2, v2)) => w1 == w2 && v1 == v2,
                (Value::String(x), Value::String(y)) => x == y,
                (Value::FloatingPoint(s1, e1, m1), Value::FloatingPoint(s2, e2, m2)) => {
                    s1 == s2 && e1 == e2 && m1 == m2
                }
                (Value::RoundingMode(x), Value::RoundingMode(y)) => x == y,
                (Value::Uninterpreted(x), Value::Uninterpreted(y)) => x == y,
                (Value::Undefined, Value::Undefined) => true,
                _ => false,
            }
        }

        let mut work: Vec<(&Value, &Value)> = vec![(self, other)];
        while let Some((a, b)) = work.pop() {
            match (a, b) {
                (Value::Array(d1, e1), Value::Array(d2, e2)) => {
                    if e1.len() != e2.len() {
                        return false;
                    }
                    work.push((d1, d2));
                    for ((k1, v1), (k2, v2)) in e1.iter().zip(e2.iter()) {
                        work.push((k1, k2));
                        work.push((v1, v2));
                    }
                }
                (Value::Datatype(id1, a1), Value::Datatype(id2, a2)) => {
                    if id1 != id2 || a1.len() != a2.len() {
                        return false;
                    }
                    for (x, y) in a1.iter().zip(a2.iter()) {
                        work.push((x, y));
                    }
                }
                _ => {
                    if !leaf_eq(a, b) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// One step of the iterative [`core::fmt::Debug`] walk: either a node still to
/// render, or literal punctuation already scheduled around it.
enum DebugStep<'a> {
    /// Render this node.
    Node(&'a Value),
    /// Emit this literal.
    Text(&'static str),
}

impl core::fmt::Debug for Value {
    /// Renders exactly what `#[derive(Debug)]` did, without its per-level
    /// recursion. `{:#?}` is deliberately *not* pretty-printed: a value deep
    /// enough to matter here is unreadable either way, and one shared code
    /// path is easier to keep faithful than two.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut steps: Vec<DebugStep<'_>> = vec![DebugStep::Node(self)];
        while let Some(step) = steps.pop() {
            match step {
                DebugStep::Text(text) => f.write_str(text)?,
                DebugStep::Node(Value::Array(default, excs)) => {
                    f.write_str("Array(")?;
                    // Scheduled in reverse of the emission order:
                    // `default, ", ", "[", (k0, v0), ", ", (k1, v1), "]", ")"`.
                    steps.push(DebugStep::Text(")"));
                    steps.push(DebugStep::Text("]"));
                    for (index, (key, value)) in excs.iter().enumerate().rev() {
                        steps.push(DebugStep::Text(")"));
                        steps.push(DebugStep::Node(value));
                        steps.push(DebugStep::Text(", "));
                        steps.push(DebugStep::Node(key));
                        steps.push(DebugStep::Text("("));
                        if index > 0 {
                            steps.push(DebugStep::Text(", "));
                        }
                    }
                    steps.push(DebugStep::Text("["));
                    steps.push(DebugStep::Text(", "));
                    steps.push(DebugStep::Node(default));
                }
                DebugStep::Node(Value::Datatype(id, args)) => {
                    write!(f, "Datatype({id}, [")?;
                    steps.push(DebugStep::Text(")"));
                    steps.push(DebugStep::Text("]"));
                    for (index, arg) in args.iter().enumerate().rev() {
                        steps.push(DebugStep::Node(arg));
                        if index > 0 {
                            steps.push(DebugStep::Text(", "));
                        }
                    }
                }
                DebugStep::Node(Value::Bool(b)) => write!(f, "Bool({b:?})")?,
                DebugStep::Node(Value::Int(i)) => write!(f, "Int({i:?})")?,
                DebugStep::Node(Value::Rational(r)) => write!(f, "Rational({r:?})")?,
                DebugStep::Node(Value::BitVec(w, v)) => write!(f, "BitVec({w:?}, {v:?})")?,
                DebugStep::Node(Value::String(s)) => write!(f, "String({s:?})")?,
                DebugStep::Node(Value::FloatingPoint(sign, exp, mant)) => {
                    write!(f, "FloatingPoint({sign:?}, {exp:?}, {mant:?})")?;
                }
                DebugStep::Node(Value::RoundingMode(rm)) => write!(f, "RoundingMode({rm:?})")?,
                DebugStep::Node(Value::Uninterpreted(id)) => write!(f, "Uninterpreted({id:?})")?,
                DebugStep::Node(Value::Undefined) => f.write_str("Undefined")?,
            }
        }
        Ok(())
    }
}

impl Value {
    /// Check if this is a boolean value
    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    /// Check if this is an integer value
    pub fn is_int(&self) -> bool {
        matches!(self, Value::Int(_))
    }

    /// Check if this is a rational value
    pub fn is_rational(&self) -> bool {
        matches!(self, Value::Rational(_))
    }

    /// Check if this is a bitvector value
    pub fn is_bitvec(&self) -> bool {
        matches!(self, Value::BitVec(_, _))
    }

    /// Check if this is undefined
    pub fn is_undefined(&self) -> bool {
        matches!(self, Value::Undefined)
    }

    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as rational
    pub fn as_rational(&self) -> Option<Rational64> {
        match self {
            Value::Rational(r) => Some(*r),
            Value::Int(i) => Some(Rational64::from_integer(*i)),
            _ => None,
        }
    }

    /// Get as bitvector (width, value)
    pub fn as_bitvec(&self) -> Option<(u32, u64)> {
        match self {
            Value::BitVec(w, v) => Some((*w, *v)),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Create a default value for `sort`, or `None` if `sort` cannot be
    /// soundly defaulted.
    ///
    /// Dispatches on `sort`'s [`SortKind`] (looked up in `sorts`), not on the
    /// raw [`SortId`] integer — see [`crate::model::ValueFactory::default_value`]
    /// for why that distinction matters. This associated function has no
    /// per-sort counter to draw from, so unlike `ValueFactory::default_value`
    /// it cannot mint a *fresh* uninterpreted element (two calls for the same
    /// uninterpreted sort would otherwise silently default to the same
    /// element, which is exactly the kind of wrong-looking value this
    /// dispatch exists to avoid); callers that need that should use
    /// `ValueFactory::default_value` instead.
    ///
    /// The array case is unrolled with a loop rather than recursion. `None`
    /// is a "cannot default this sort" answer, not an error channel a depth
    /// cap could report through, so a cap here could only return a *wrong*
    /// default. Array-sort nesting is bounded at 512 when it comes from
    /// SMT-LIB text, but `SortManager::array` is `pub` and interns in
    /// constant stack, so an embedder can build an arbitrarily deep one.
    pub fn default_for_sort(sort: SortId, sorts: &SortManager) -> Option<Self> {
        // Descend the chain of array *range* sorts first, then wrap the
        // innermost non-array default back up once per level.
        let mut array_levels = 0usize;
        let mut current = sort;
        let leaf = loop {
            let kind = sorts.get(current)?.kind.clone();
            match kind {
                SortKind::Bool => break Value::Bool(false),
                SortKind::Int => break Value::Int(0),
                SortKind::Real => break Value::Rational(Rational64::from_integer(0)),
                SortKind::String => break Value::String(String::new()),
                SortKind::BitVec(width) => break Value::BitVec(width, 0),
                SortKind::FloatingPoint { .. } => break Value::FloatingPoint(false, 0, 0),
                // `RoundingMode` is finite and inhabited, so unlike the
                // opaque sorts below it *does* have a canonical default:
                // round-to-nearest-ties-to-even, the IEEE 754 default mode.
                SortKind::RoundingMode => break Value::RoundingMode(RoundingMode::RNE),
                SortKind::Array { range, .. } => {
                    array_levels += 1;
                    current = range;
                }
                // See the doc comment: no counter here to guarantee freshness,
                // so an opaque domain element cannot be soundly minted. An
                // array over such a range has no default either, exactly as
                // the recursive version propagated `None` outward.
                SortKind::Uninterpreted(_)
                | SortKind::Parametric { .. }
                | SortKind::Parameter(_)
                | SortKind::Datatype(_) => return None,
            }
        };

        let mut value = leaf;
        for _ in 0..array_levels {
            value = Value::Array(Box::new(value), Vec::new());
        }
        Some(value)
    }
}

impl core::fmt::Display for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Rational(r) => {
                if *r.denom() == 1 {
                    write!(f, "{}", r.numer())
                } else {
                    write!(f, "(/ {} {})", r.numer(), r.denom())
                }
            }
            Value::BitVec(w, v) => write!(f, "#b{:0width$b}", v, width = *w as usize),
            // A string value is source text, not display text: the quotes and
            // any `\u{...}` escapes come from the one shared encoder the
            // SMT-LIB printers use, so a model value re-reads as itself.
            Value::String(s) => write!(f, "{}", crate::smtlib::format_string_literal(s)),
            // Only the default element is printed, so the nesting to unroll
            // is the chain of array-of-array defaults. Written as a loop for
            // the same reason the other structural traits are: a recursive
            // `write!(f, "{}", def)` per level aborts the process on a deep
            // value.
            Value::Array(_, _) => {
                let mut node = self;
                let mut depth = 0usize;
                while let Value::Array(default, excs) = node {
                    f.write_str(if excs.is_empty() {
                        "((as const) "
                    } else {
                        "(store ... "
                    })?;
                    depth += 1;
                    node = default;
                }
                // `node` is not an array, so this re-entry is the innermost
                // one and none of the remaining arms recurses.
                write!(f, "{}", node)?;
                for _ in 0..depth {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Value::Datatype(id, args) => {
                if args.is_empty() {
                    write!(f, "C{}", id)
                } else {
                    write!(f, "(C{} ...)", id)
                }
            }
            Value::FloatingPoint(sign, exp, mant) => {
                write!(
                    f,
                    "(fp {} {} {})",
                    if *sign { "#b1" } else { "#b0" },
                    exp,
                    mant
                )
            }
            // The canonical long spelling, which is what `get-model` prints
            // and what SMT-LIB accepts back as input.
            Value::RoundingMode(rm) => {
                write!(f, "{}", crate::ast::TermManager::rounding_mode_name(*rm))
            }
            Value::Uninterpreted(id) => write!(f, "u{}", id),
            Value::Undefined => write!(f, "undefined"),
        }
    }
}

/// One (input-args → output-value) entry in a function interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct FuncEntry {
    /// Argument values that trigger this entry.
    pub args: Vec<Value>,
    /// The output value for these arguments.
    pub value: Value,
}

/// Full interpretation of an uninterpreted function in a model.
///
/// Defines the function as a finite table of `(args → value)` entries plus an
/// `else_value` that applies to every input combination not covered by an entry.
/// This mirrors Z3's `FuncInterp` object.
#[derive(Debug, Clone)]
pub struct FuncInterp {
    /// Finite table entries (in order of insertion).
    pub entries: Vec<FuncEntry>,
    /// Value returned for all inputs not matched by any entry.
    pub else_value: Value,
    /// Number of argument positions this function accepts.
    pub arity: usize,
}

impl FuncInterp {
    /// Create a new empty `FuncInterp` with the given arity and `else_value`.
    #[must_use]
    pub fn new(arity: usize, else_value: Value) -> Self {
        Self {
            entries: Vec::new(),
            else_value,
            arity,
        }
    }

    /// Append an `(args → value)` entry to the interpretation table.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `args.len() != self.arity`.
    pub fn add_entry(&mut self, args: Vec<Value>, value: Value) {
        debug_assert_eq!(
            args.len(),
            self.arity,
            "FuncInterp::add_entry: arity mismatch (expected {}, got {})",
            self.arity,
            args.len()
        );
        self.entries.push(FuncEntry { args, value });
    }

    /// Return the number of explicit entries.
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Look up `args` in the entry table.
    ///
    /// Returns the value of the first matching entry, or `&self.else_value` if
    /// no entry matches.
    #[must_use]
    pub fn evaluate(&self, args: &[Value]) -> &Value {
        for entry in &self.entries {
            if entry.args == args {
                return &entry.value;
            }
        }
        &self.else_value
    }
}

/// A model: assignment of values to terms
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// Term to value assignments
    assignments: HashMap<TermId, Value>,
    /// Sort assignments (for uninterpreted sorts)
    sort_sizes: HashMap<SortId, u64>,
    /// Function interpretations for uninterpreted functions, keyed by the
    /// function *symbol* rather than any single [`TermId`].
    ///
    /// [`crate::ast::TermKind::Apply`] — the only term shape that names an
    /// uninterpreted function — carries the function as a `Spur` (its
    /// interned name), not a `TermId`: every `(f x)`, `(f y)`, ... gets its
    /// own distinct `TermId`, all sharing the same `Spur`. A `TermId` key
    /// would therefore need one `FuncInterp` per call site instead of one per
    /// function, so it cannot be what identifies "the function" here.
    func_interps: HashMap<Spur, FuncInterp>,
}

impl Model {
    /// Create a new empty model
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a value to a term
    pub fn assign(&mut self, term: TermId, value: Value) {
        self.assignments.insert(term, value);
    }

    /// Get the value of a term
    pub fn get(&self, term: TermId) -> Option<&Value> {
        self.assignments.get(&term)
    }

    /// Check if a term has an assignment
    pub fn has(&self, term: TermId) -> bool {
        self.assignments.contains_key(&term)
    }

    /// Remove an assignment
    pub fn remove(&mut self, term: TermId) -> Option<Value> {
        self.assignments.remove(&term)
    }

    /// Number of assignments
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Check if model is empty
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Iterate over assignments
    pub fn iter(&self) -> impl Iterator<Item = (&TermId, &Value)> {
        self.assignments.iter()
    }

    /// Set sort size (for uninterpreted sorts)
    pub fn set_sort_size(&mut self, sort: SortId, size: u64) {
        self.sort_sizes.insert(sort, size);
    }

    /// Get sort size
    pub fn get_sort_size(&self, sort: SortId) -> Option<u64> {
        self.sort_sizes.get(&sort).copied()
    }

    /// Clear all assignments
    pub fn clear(&mut self) {
        self.assignments.clear();
        self.sort_sizes.clear();
    }

    /// Merge another model into this one
    pub fn merge(&mut self, other: &Model) {
        for (term, value) in &other.assignments {
            self.assignments
                .entry(*term)
                .or_insert_with(|| value.clone());
        }
        for (sort, size) in &other.sort_sizes {
            self.sort_sizes.entry(*sort).or_insert(*size);
        }
        for (func_id, interp) in &other.func_interps {
            self.func_interps
                .entry(*func_id)
                .or_insert_with(|| interp.clone());
        }
    }

    /// Store a complete function interpretation for the function named
    /// `func_name` (an [`crate::ast::TermKind::Apply`] node's interned
    /// `func` field).
    ///
    /// Any previous interpretation for the same function is replaced.
    pub fn add_func_interp(&mut self, func_name: Spur, interp: FuncInterp) {
        self.func_interps.insert(func_name, interp);
    }

    /// Retrieve the function interpretation for `func_name`, if one was
    /// stored.
    #[must_use]
    pub fn get_func_interp(&self, func_name: Spur) -> Option<&FuncInterp> {
        self.func_interps.get(&func_name)
    }

    /// Iterate over all stored function interpretations.
    pub fn func_interps(&self) -> impl Iterator<Item = (&Spur, &FuncInterp)> {
        self.func_interps.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_bool() {
        let v = Value::Bool(true);
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(format!("{}", v), "true");
    }

    #[test]
    fn test_value_int() {
        let v = Value::Int(42);
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(format!("{}", v), "42");
    }

    #[test]
    fn test_value_rational() {
        let v = Value::Rational(Rational64::new(1, 2));
        assert!(v.is_rational());
        assert_eq!(v.as_rational(), Some(Rational64::new(1, 2)));
        assert_eq!(format!("{}", v), "(/ 1 2)");
    }

    #[test]
    fn test_value_bitvec() {
        let v = Value::BitVec(8, 255);
        assert!(v.is_bitvec());
        assert_eq!(v.as_bitvec(), Some((8, 255)));
        assert_eq!(format!("{}", v), "#b11111111");
    }

    /// A string value's `Display` is SMT-LIB *source text*, so it must carry
    /// the same escapes the printers emit — it used to carry none at all,
    /// which turned any value containing a quote into unreadable output.
    #[test]
    fn test_value_string_display_is_a_wellformed_literal() {
        for (value, expected) in [
            ("", r#""""#),
            ("hello world", r#""hello world""#),
            ("a\"b", r#""a""b""#),
            ("a\\b", r#""a\b""#),
            ("\\u0041", r#""\u{5c}u0041""#),
            ("\u{e9}", r#""\u{e9}""#),
            ("\u{0}", r#""\u{0}""#),
            ("\u{2ffff}", r#""\u{2ffff}""#),
        ] {
            let v = Value::String(value.to_string());
            assert_eq!(v.as_string(), Some(value));
            assert_eq!(format!("{v}"), expected, "{value:?} displayed wrongly");
        }
    }

    #[test]
    fn test_model_basic() {
        let mut model = Model::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        model.assign(t1, Value::Bool(true));
        model.assign(t2, Value::Int(42));

        assert_eq!(model.len(), 2);
        assert!(model.has(t1));
        assert!(model.has(t2));
        assert_eq!(model.get(t1), Some(&Value::Bool(true)));
        assert_eq!(model.get(t2), Some(&Value::Int(42)));
    }

    #[test]
    fn test_model_remove() {
        let mut model = Model::new();
        let t1 = TermId::from(1u32);

        model.assign(t1, Value::Bool(true));
        assert!(model.has(t1));

        model.remove(t1);
        assert!(!model.has(t1));
    }

    #[test]
    fn test_model_merge() {
        let mut m1 = Model::new();
        let mut m2 = Model::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        let t3 = TermId::from(3u32);

        m1.assign(t1, Value::Bool(true));
        m1.assign(t2, Value::Int(42));

        m2.assign(t2, Value::Int(100)); // Should not override
        m2.assign(t3, Value::Bool(false));

        m1.merge(&m2);

        assert_eq!(m1.len(), 3);
        assert_eq!(m1.get(t1), Some(&Value::Bool(true)));
        assert_eq!(m1.get(t2), Some(&Value::Int(42))); // Original preserved
        assert_eq!(m1.get(t3), Some(&Value::Bool(false)));
    }

    /// `Value::default_for_sort` unrolls nested array sorts with a loop.
    /// `None` means "cannot default this sort", not an error channel, so a
    /// depth cap could only ever return a wrong default; recursing aborted
    /// the process instead. `SortManager::array` is `pub` and interns in
    /// constant stack, so an embedder can reach any depth.
    ///
    /// Runs on a 128 KiB stack; the assertion is that the call returns.
    #[test]
    fn test_default_for_sort_survives_a_deeply_nested_array_sort() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut sorts = SortManager::new();
                let int_sort = sorts.int_sort;
                let mut sort = int_sort;
                for _ in 0..12_500 {
                    sort = sorts.array(int_sort, sort);
                }
                let value = Value::default_for_sort(sort, &sorts);
                let mut levels = 0usize;
                let mut node = value.as_ref();
                while let Some(Value::Array(default, _)) = node {
                    levels += 1;
                    node = Some(default.as_ref());
                }
                (levels, matches!(node, Some(Value::Int(0))))
            })
            .expect("spawn");
        let (levels, innermost_is_int_zero) =
            handle.join().expect("worker thread must not overflow");
        assert_eq!(levels, 12_500);
        assert!(innermost_is_int_zero);
    }

    #[test]
    fn test_value_default_for_sort() {
        let manager = crate::sort::SortManager::new();
        assert_eq!(
            Value::default_for_sort(manager.bool_sort, &manager),
            Some(Value::Bool(false))
        );
        assert_eq!(
            Value::default_for_sort(manager.int_sort, &manager),
            Some(Value::Int(0))
        );
    }

    /// Regression test mirroring
    /// `factory::tests::test_default_value_dispatches_on_sort_kind_not_raw_sort_id`:
    /// `default_for_sort` used to hardcode `sort.0 == 3 => ...`-style raw-id
    /// checks too (it had no `String`/`BitVec`/`Array` arms at all, so
    /// anything past `Real` silently fell through to `Undefined`). A BitVec
    /// sort landing on a non-built-in raw id must get a same-width zero
    /// BitVec, not `Undefined`.
    #[test]
    fn test_value_default_for_sort_dispatches_on_sort_kind() {
        let mut manager = crate::sort::SortManager::new();
        let bv16 = manager.bitvec(16);
        assert_eq!(
            Value::default_for_sort(bv16, &manager),
            Some(Value::BitVec(16, 0))
        );

        let arr = manager.array(manager.int_sort, manager.bool_sort);
        assert_eq!(
            Value::default_for_sort(arr, &manager),
            Some(Value::Array(Box::new(Value::Bool(false)), Vec::new()))
        );

        // A sort this associated function genuinely cannot default (no
        // per-sort counter to mint a fresh, sound uninterpreted element).
        let spur = manager.intern_str("S");
        let uninterpreted = manager.intern(crate::sort::SortKind::Uninterpreted(spur));
        assert_eq!(Value::default_for_sort(uninterpreted, &manager), None);

        // A `SortId` unknown to `manager` at all.
        assert_eq!(Value::default_for_sort(SortId(9_999), &manager), None);
    }

    // ── FuncInterp ────────────────────────────────────────────────────────────

    #[test]
    fn test_func_interp_new_empty() {
        let fi = FuncInterp::new(2, Value::Int(0));
        assert_eq!(fi.num_entries(), 0);
        assert_eq!(fi.arity, 2);
        assert_eq!(fi.else_value, Value::Int(0));
    }

    #[test]
    fn test_func_interp_add_entry_and_evaluate() {
        let mut fi = FuncInterp::new(1, Value::Int(-1));
        fi.add_entry(vec![Value::Int(0)], Value::Int(42));
        // Exact match
        assert_eq!(fi.evaluate(&[Value::Int(0)]), &Value::Int(42));
        // No match → else_value
        assert_eq!(fi.evaluate(&[Value::Int(99)]), &Value::Int(-1));
    }

    #[test]
    fn test_func_interp_evaluate_first_match_wins() {
        let mut fi = FuncInterp::new(1, Value::Int(0));
        fi.add_entry(vec![Value::Int(1)], Value::Int(10));
        fi.add_entry(vec![Value::Int(1)], Value::Int(20)); // duplicate key; first wins
        assert_eq!(fi.evaluate(&[Value::Int(1)]), &Value::Int(10));
    }

    #[test]
    fn test_func_interp_multi_arg() {
        let mut fi = FuncInterp::new(2, Value::Bool(false));
        fi.add_entry(vec![Value::Int(3), Value::Int(4)], Value::Bool(true));
        assert_eq!(
            fi.evaluate(&[Value::Int(3), Value::Int(4)]),
            &Value::Bool(true)
        );
        assert_eq!(
            fi.evaluate(&[Value::Int(3), Value::Int(5)]),
            &Value::Bool(false)
        );
    }

    // ── Model::add_func_interp / get_func_interp ──────────────────────────────

    #[test]
    fn test_model_add_and_get_func_interp() {
        let mut model = Model::new();
        let mut rodeo = crate::interner::Rodeo::default();
        let f = rodeo.get_or_intern("f");

        let mut fi = FuncInterp::new(1, Value::Int(0));
        fi.add_entry(vec![Value::Int(7)], Value::Int(49));
        model.add_func_interp(f, fi);

        let retrieved = model.get_func_interp(f);
        assert!(retrieved.is_some());
        let fi2 = retrieved.unwrap();
        assert_eq!(fi2.num_entries(), 1);
        assert_eq!(fi2.evaluate(&[Value::Int(7)]), &Value::Int(49));
        assert_eq!(fi2.evaluate(&[Value::Int(0)]), &Value::Int(0));
    }

    #[test]
    fn test_model_get_func_interp_missing_returns_none() {
        let model = Model::new();
        let mut rodeo = crate::interner::Rodeo::default();
        let g = rodeo.get_or_intern("g");
        assert!(model.get_func_interp(g).is_none());
    }

    // ── Deeply nested values ──────────────────────────────────────────────
    //
    // Regression tests for: "`Value` recurses structurally, so a deep value
    // aborts the process". Every case builds its value with an *iterative*
    // loop — a recursive helper would overflow before the assertion ran — and
    // runs on a thread with an explicitly small (128 KiB) stack, the size an
    // embedder's worker thread typically gets. A stack overflow aborts the
    // whole process rather than failing a test, so "the case returned at all"
    // *is* the assertion; the value checks on top of that make sure the
    // iterative implementations still compute the right answer.
    //
    // Before these traits were written by hand, the derived versions aborted
    // on this shape somewhere between 20 000 and 40 000 levels.

    /// Stack size every deep case runs under.
    const SMALL_STACK: usize = 1 << 17;

    /// A depth well past anything a native-stack recursion could survive.
    const DEEP: usize = 25_000;

    /// Run `body` on a thread with a deliberately small stack.
    ///
    /// A stack overflow inside `body` aborts the process rather than
    /// unwinding, so this helper cannot turn one into a test failure — that
    /// is the point. The test run itself fails, loudly.
    fn on_small_stack<T, F>(body: F) -> T
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        std::thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(body)
            .expect("spawn small-stack thread")
            .join()
            .expect("small-stack thread panicked")
    }

    /// `Array(Array(... Array(leaf) ...))`, `depth` levels of it, nested
    /// through the default-element slot and built without recursion.
    fn deep_array(depth: usize, leaf: Value) -> Value {
        let mut value = leaf;
        for _ in 0..depth {
            value = Value::Array(Box::new(value), Vec::new());
        }
        value
    }

    /// The same shape nested through an *exception value* instead, which is
    /// what a `store` chain in value position produces.
    fn deep_store_chain(depth: usize, leaf: Value) -> Value {
        let mut value = leaf;
        for i in 0..depth {
            value = Value::Array(
                Box::new(Value::Int(-1)),
                vec![(Value::Int(i as i64), value)],
            );
        }
        value
    }

    /// How many `Array`/`Datatype` levels deep `value` nests, counted with an
    /// explicit worklist.
    fn nesting_depth(value: &Value) -> usize {
        let mut deepest = 0usize;
        let mut work = vec![(value, 0usize)];
        while let Some((node, level)) = work.pop() {
            deepest = deepest.max(level);
            match node {
                Value::Array(default, excs) => {
                    work.push((default, level + 1));
                    for (key, item) in excs {
                        work.push((key, level + 1));
                        work.push((item, level + 1));
                    }
                }
                Value::Datatype(_, args) => {
                    for arg in args {
                        work.push((arg, level + 1));
                    }
                }
                _ => {}
            }
        }
        deepest
    }

    #[test]
    fn test_deep_value_drops_on_a_small_stack() {
        // The value must be *dropped inside* the small-stack thread for this
        // to mean anything: the drop glue is what recursed.
        let depth = on_small_stack(|| {
            let value = deep_array(DEEP, Value::Int(7));
            let depth = nesting_depth(&value);
            drop(value);
            depth
        });
        assert_eq!(depth, DEEP);
    }

    #[test]
    fn test_deep_store_shaped_value_drops_on_a_small_stack() {
        let depth = on_small_stack(|| {
            let value = deep_store_chain(DEEP, Value::Int(7));
            let depth = nesting_depth(&value);
            drop(value);
            depth
        });
        assert_eq!(depth, DEEP);
    }

    #[test]
    fn test_deep_value_clones_on_a_small_stack() {
        let (equal, depth) = on_small_stack(|| {
            let value = deep_store_chain(DEEP, Value::String("leaf".to_string()));
            let copy = value.clone();
            // Both the original and the clone are dropped here, on the small
            // stack, after being compared.
            (copy == value, nesting_depth(&copy))
        });
        assert!(
            equal,
            "the iterative clone must reproduce the value exactly"
        );
        assert_eq!(depth, DEEP);
    }

    #[test]
    fn test_deep_value_compares_on_a_small_stack() {
        let (same, differs) = on_small_stack(|| {
            let a = deep_array(DEEP, Value::Int(1));
            let b = deep_array(DEEP, Value::Int(1));
            let c = deep_array(DEEP, Value::Int(2));
            (a == b, a != c)
        });
        assert!(same, "identical deep values must compare equal");
        assert!(differs, "a differing leaf must still be detected");
    }

    #[test]
    fn test_deep_value_formats_on_a_small_stack() {
        let (display, debug) = on_small_stack(|| {
            let value = deep_array(DEEP, Value::Int(7));
            (format!("{value}"), format!("{value:?}"))
        });
        // `Display` unrolls the default chain: `depth` opening prefixes, the
        // leaf, then `depth` closing parens.
        let (prefix, suffix) = display.split_once('7').expect("the leaf is printed");
        assert_eq!(prefix.matches("((as const) ").count(), DEEP);
        assert_eq!(suffix, ")".repeat(DEEP));
        assert_eq!(debug.matches("Array(").count(), DEEP);
        assert!(debug.contains("Int(7)"));
    }

    #[test]
    fn test_value_formatting_matches_the_derived_shapes() {
        // The hand-written `Debug` has to render exactly what the derive did,
        // and `Display` exactly what the recursive version did.
        let leaf = Value::Array(Box::new(Value::Int(0)), Vec::new());
        assert_eq!(format!("{leaf:?}"), "Array(Int(0), [])");
        assert_eq!(format!("{leaf}"), "((as const) 0)");

        let stored = Value::Array(
            Box::new(Value::Int(0)),
            vec![
                (Value::Int(1), Value::Bool(true)),
                (Value::String("k".to_string()), Value::Undefined),
            ],
        );
        assert_eq!(
            format!("{stored:?}"),
            "Array(Int(0), [(Int(1), Bool(true)), (String(\"k\"), Undefined)])"
        );
        assert_eq!(format!("{stored}"), "(store ... 0)");

        let dt = Value::Datatype(3, vec![Value::Int(1), Value::Bool(false)]);
        assert_eq!(format!("{dt:?}"), "Datatype(3, [Int(1), Bool(false)])");
        assert_eq!(format!("{dt}"), "(C3 ...)");
        assert_eq!(
            format!("{:?}", Value::Datatype(4, Vec::new())),
            "Datatype(4, [])"
        );
        assert_eq!(format!("{:?}", Value::BitVec(8, 255)), "BitVec(8, 255)");
        assert_eq!(format!("{:?}", Value::Undefined), "Undefined");
        assert_eq!(
            format!("{:?}", Value::FloatingPoint(true, 1, 2)),
            "FloatingPoint(true, 1, 2)"
        );
    }

    #[test]
    fn test_value_clone_and_eq_preserve_nested_structure() {
        let value = Value::Array(
            Box::new(Value::Datatype(1, vec![Value::Int(5)])),
            vec![(
                Value::Int(2),
                Value::Array(Box::new(Value::Bool(true)), Vec::new()),
            )],
        );
        let copy = value.clone();
        assert_eq!(copy, value);
        assert_eq!(format!("{copy:?}"), format!("{value:?}"));

        // Mismatches at every structural position must still be detected.
        let other_default = Value::Array(
            Box::new(Value::Datatype(1, vec![Value::Int(6)])),
            vec![(
                Value::Int(2),
                Value::Array(Box::new(Value::Bool(true)), Vec::new()),
            )],
        );
        assert_ne!(other_default, value);
        let other_key = Value::Array(
            Box::new(Value::Datatype(1, vec![Value::Int(5)])),
            vec![(
                Value::Int(3),
                Value::Array(Box::new(Value::Bool(true)), Vec::new()),
            )],
        );
        assert_ne!(other_key, value);
        let shorter = Value::Array(
            Box::new(Value::Datatype(1, vec![Value::Int(5)])),
            Vec::new(),
        );
        assert_ne!(shorter, value);
        assert_ne!(
            Value::Datatype(1, Vec::new()),
            Value::Datatype(2, Vec::new())
        );
    }

    // ── Int / Real value-shape bridge ─────────────────────────────────────

    #[test]
    fn test_value_eq_bridges_int_and_integral_rational() {
        // The evaluator produces both shapes for the same Real number, so
        // they must compare equal; a non-integral rational must not.
        assert_eq!(Value::Int(1), Value::Rational(Rational64::from_integer(1)));
        assert_eq!(
            Value::Rational(Rational64::from_integer(-4)),
            Value::Int(-4)
        );
        assert_ne!(Value::Int(1), Value::Rational(Rational64::new(1, 2)));
        assert_ne!(Value::Int(2), Value::Rational(Rational64::from_integer(1)));
        assert_ne!(Value::Int(1), Value::Bool(true));
    }

    #[test]
    fn test_func_interp_evaluate_bridges_int_and_integral_rational() {
        // `FuncInterp::evaluate` matches its argument table with `==`, so it
        // inherits the bridge: an entry keyed on a Real literal's value is
        // found by a numerically equal computed argument.
        let mut fi = FuncInterp::new(1, Value::Int(0));
        fi.add_entry(
            vec![Value::Rational(Rational64::from_integer(3))],
            Value::Int(9),
        );
        assert_eq!(fi.evaluate(&[Value::Int(3)]), &Value::Int(9));
        assert_eq!(
            fi.evaluate(&[Value::Rational(Rational64::new(7, 2))]),
            &Value::Int(0)
        );
    }

    #[test]
    fn test_model_merge_preserves_func_interps() {
        let mut m1 = Model::new();
        let mut m2 = Model::new();
        let mut rodeo = crate::interner::Rodeo::default();
        let f1 = rodeo.get_or_intern("f1");
        let f2 = rodeo.get_or_intern("f2");

        let fi1 = FuncInterp::new(0, Value::Int(1));
        let fi2 = FuncInterp::new(0, Value::Int(2));
        m1.add_func_interp(f1, fi1);
        m2.add_func_interp(f2, fi2);

        m1.merge(&m2);
        assert!(m1.get_func_interp(f1).is_some());
        assert!(m1.get_func_interp(f2).is_some());
    }
}
