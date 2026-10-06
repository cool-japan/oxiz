//! LFSC proof format (Logical Framework with Side Conditions).
//!
//! LFSC is a typed first-order language with side conditions, used for
//! certified verification of SMT proofs. It was developed for use with
//! CVC4/CVC5 and is checkable by the LFSC checker.
//!
//! ## Structure
//!
//! LFSC proofs consist of:
//! - **Type declarations**: Define sorts and kinds
//! - **Term declarations**: Define functions and constants
//! - **Side conditions**: Computational side conditions for proof rules
//! - **Proof terms**: The actual proof derivation

use std::fmt;
use std::io::{self, Write};

/// An LFSC sort/type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LfscSort {
    /// Kind (type of types)
    Kind,
    /// Type (type of proofs)
    Type,
    /// Boolean sort
    Bool,
    /// Integer sort
    Int,
    /// Real sort
    Real,
    /// BitVector sort with width
    BitVec(u32),
    /// Arrow type (function type)
    Arrow(Box<LfscSort>, Box<LfscSort>),
    /// Named sort
    Named(String),
    /// Application of type constructor
    App(String, Vec<LfscSort>),
}

impl fmt::Display for LfscSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kind => write!(f, "kind"),
            Self::Type => write!(f, "type"),
            Self::Bool => write!(f, "bool"),
            Self::Int => write!(f, "mpz"),
            Self::Real => write!(f, "mpq"),
            Self::BitVec(w) => write!(f, "(bitvec {})", w),
            Self::Arrow(a, b) => write!(f, "(! _ {} {})", a, b),
            Self::Named(n) => write!(f, "{}", n),
            Self::App(n, args) => {
                write!(f, "({}", n)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// An LFSC term
///
/// # Depth invariant
///
/// There is deliberately no bound on how deep an `LfscTerm` may be: the
/// variants are public, so callers build values directly (e.g. via repeated
/// [`LfscTerm::App`] or [`LfscTerm::Lambda`] wrapping), and a proof converted
/// from an external source (see [`crate::conversion::FormatConverter`]) can
/// nest as deeply as that source does. [`Clone`] and [`Drop`] are therefore
/// iterative -- see their impls below -- rather than derived. Do **not**
/// replace either with a `derive`.
///
/// The one exception is the derived [`fmt::Debug`], which is still recursive:
/// it is a diagnostics-only formatter, is never invoked by this crate outside
/// tests, and hand-writing it would change `{:#?}` output. Prefer
/// [`fmt::Display`] when rendering a term whose depth is not known -- it is
/// also recursive today, so callers with depth-adversarial input should
/// prefer walking the term themselves rather than formatting it directly.
#[derive(Debug)]
pub enum LfscTerm {
    /// Variable reference
    Var(String),
    /// Integer literal
    IntLit(i64),
    /// Rational literal (numerator, denominator)
    RatLit(i64, i64),
    /// Boolean true
    True,
    /// Boolean false
    False,
    /// Application
    App(String, Vec<LfscTerm>),
    /// Lambda abstraction
    Lambda(String, Box<LfscSort>, Box<LfscTerm>),
    /// Pi type (dependent function type)
    Pi(String, Box<LfscSort>, Box<LfscTerm>),
    /// Side condition application
    SideCondition(String, Vec<LfscTerm>),
    /// Proof hold (assertion)
    Hold(Box<LfscTerm>),
    /// Type annotation
    Annotate(Box<LfscTerm>, Box<LfscSort>),
}

/// The shape of a node being rebuilt by the iterative [`Clone`] impl: which
/// variant it is, plus anything that is not one of the cloned children.
enum LfscCloneShape {
    /// `App`, carrying its function name and arity.
    App(String, usize),
    /// `Lambda`, carrying its bound variable name and (derive-cloned) sort.
    Lambda(String, Box<LfscSort>),
    /// `Pi`, carrying its bound variable name and (derive-cloned) sort.
    Pi(String, Box<LfscSort>),
    /// `SideCondition`, carrying its name and arity.
    SideCondition(String, usize),
    /// `Hold`, one child.
    Hold,
    /// `Annotate`, carrying its (derive-cloned) sort.
    Annotate(Box<LfscSort>),
}

/// Work item for the iterative [`Clone`] impl.
enum LfscCloneTask<'a> {
    /// Clone this subterm.
    Visit(&'a LfscTerm),
    /// Rebuild a node from the already-cloned children on the result stack.
    Rebuild(LfscCloneShape),
}

impl Clone for LfscTerm {
    /// Iterative clone.
    ///
    /// The derived recursive `Clone` walked the term with one native call
    /// frame per nesting level -- the same hazard the iterative [`Drop`]
    /// below exists to avoid, just triggered by a different standard-library
    /// entry point (`.clone()` / `#[derive(Clone)]` callers). `LfscSort`
    /// (the type of the `sort` field on `Lambda`/`Pi`/`Annotate`) keeps its
    /// ordinary derived `Clone`: this type's depth invariant is driven by
    /// term nesting, not sort nesting, and sorts built by this crate do not
    /// nest to comparable depths. Nodes are rebuilt with their plain variant
    /// constructors, exactly mirroring `InterpolantTerm` in
    /// `craig/term.rs`.
    fn clone(&self) -> Self {
        /// Detach the top `n` results, preserving their original order.
        fn take(results: &mut Vec<LfscTerm>, n: usize) -> Vec<LfscTerm> {
            let start = results.len().saturating_sub(n);
            results.split_off(start)
        }

        /// Rebuild a one-child node, or fall back to `False` if starved.
        fn one(results: &mut Vec<LfscTerm>) -> Box<LfscTerm> {
            let mut operand = take(results, 1);
            Box::new(operand.pop().unwrap_or(LfscTerm::False))
        }

        let mut tasks = vec![LfscCloneTask::Visit(self)];
        let mut results: Vec<Self> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                LfscCloneTask::Visit(term) => match term {
                    Self::Var(s) => results.push(Self::Var(s.clone())),
                    Self::IntLit(n) => results.push(Self::IntLit(*n)),
                    Self::RatLit(n, d) => results.push(Self::RatLit(*n, *d)),
                    Self::True => results.push(Self::True),
                    Self::False => results.push(Self::False),
                    Self::App(f, args) => {
                        tasks.push(LfscCloneTask::Rebuild(LfscCloneShape::App(
                            f.clone(),
                            args.len(),
                        )));
                        tasks.extend(args.iter().rev().map(LfscCloneTask::Visit));
                    }
                    Self::Lambda(var, sort, body) => {
                        tasks.push(LfscCloneTask::Rebuild(LfscCloneShape::Lambda(
                            var.clone(),
                            sort.clone(),
                        )));
                        tasks.push(LfscCloneTask::Visit(body));
                    }
                    Self::Pi(var, sort, body) => {
                        tasks.push(LfscCloneTask::Rebuild(LfscCloneShape::Pi(
                            var.clone(),
                            sort.clone(),
                        )));
                        tasks.push(LfscCloneTask::Visit(body));
                    }
                    Self::SideCondition(name, args) => {
                        tasks.push(LfscCloneTask::Rebuild(LfscCloneShape::SideCondition(
                            name.clone(),
                            args.len(),
                        )));
                        tasks.extend(args.iter().rev().map(LfscCloneTask::Visit));
                    }
                    Self::Hold(inner) => {
                        tasks.push(LfscCloneTask::Rebuild(LfscCloneShape::Hold));
                        tasks.push(LfscCloneTask::Visit(inner));
                    }
                    Self::Annotate(inner, sort) => {
                        tasks.push(LfscCloneTask::Rebuild(LfscCloneShape::Annotate(
                            sort.clone(),
                        )));
                        tasks.push(LfscCloneTask::Visit(inner));
                    }
                },
                LfscCloneTask::Rebuild(shape) => {
                    let rebuilt = match shape {
                        LfscCloneShape::App(f, n) => Self::App(f, take(&mut results, n)),
                        LfscCloneShape::Lambda(var, sort) => {
                            Self::Lambda(var, sort, one(&mut results))
                        }
                        LfscCloneShape::Pi(var, sort) => Self::Pi(var, sort, one(&mut results)),
                        LfscCloneShape::SideCondition(name, n) => {
                            Self::SideCondition(name, take(&mut results, n))
                        }
                        LfscCloneShape::Hold => Self::Hold(one(&mut results)),
                        LfscCloneShape::Annotate(sort) => Self::Annotate(one(&mut results), sort),
                    };
                    results.push(rebuilt);
                }
            }
        }

        results.pop().unwrap_or(Self::False)
    }
}

impl Drop for LfscTerm {
    /// Iterative drop.
    ///
    /// Every term built by this crate's converters and builders inherits the
    /// depth of its source, and the compiler-generated recursive
    /// `drop_in_place` would be the one remaining way to abort the process,
    /// at scope exit, with no diagnostic. Each node is dismantled into a
    /// shallow shell before being released, exactly mirroring
    /// `InterpolantTerm` in `craig/term.rs`.
    fn drop(&mut self) {
        /// Detach a node's children, leaving a shell that drops trivially.
        fn dismantle(node: &mut LfscTerm, out: &mut Vec<LfscTerm>) {
            /// Replace a boxed child with a leaf and hand the child over.
            fn take(slot: &mut Box<LfscTerm>, out: &mut Vec<LfscTerm>) {
                out.push(std::mem::replace(slot.as_mut(), LfscTerm::False));
            }

            match node {
                LfscTerm::Var(_)
                | LfscTerm::IntLit(_)
                | LfscTerm::RatLit(_, _)
                | LfscTerm::True
                | LfscTerm::False => {}
                LfscTerm::App(_, args) | LfscTerm::SideCondition(_, args) => out.append(args),
                LfscTerm::Lambda(_, _, body) | LfscTerm::Pi(_, _, body) => take(body, out),
                LfscTerm::Hold(inner) => take(inner, out),
                LfscTerm::Annotate(inner, _) => take(inner, out),
            }
        }

        let mut pending = Vec::new();
        dismantle(self, &mut pending);
        while let Some(mut node) = pending.pop() {
            dismantle(&mut node, &mut pending);
        }
    }
}

impl fmt::Display for LfscTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(v) => write!(f, "{}", v),
            Self::IntLit(n) => write!(f, "{}", n),
            Self::RatLit(n, d) => write!(f, "{}/{}", n, d),
            Self::True => write!(f, "tt"),
            Self::False => write!(f, "ff"),
            Self::App(func, args) => {
                write!(f, "({}", func)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
            Self::Lambda(var, sort, body) => {
                write!(f, "(\\ {} {} {})", var, sort, body)
            }
            Self::Pi(var, sort, body) => {
                write!(f, "(! {} {} {})", var, sort, body)
            }
            Self::SideCondition(name, args) => {
                write!(f, "(# {} (", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, "))")
            }
            Self::Hold(t) => write!(f, "(holds {})", t),
            Self::Annotate(t, s) => write!(f, "(: {} {})", t, s),
        }
    }
}

/// An LFSC declaration
#[derive(Debug, Clone)]
pub enum LfscDecl {
    /// Declare a new sort
    DeclareSort { name: String, arity: u32 },
    /// Declare a new constant/function
    DeclareConst { name: String, sort: LfscSort },
    /// Define a term
    Define {
        name: String,
        sort: LfscSort,
        value: LfscTerm,
    },
    /// Declare a proof rule
    DeclareRule {
        name: String,
        params: Vec<(String, LfscSort)>,
        conclusion: LfscTerm,
    },
    /// Side condition program
    SideCondition {
        name: String,
        params: Vec<(String, LfscSort)>,
        return_sort: LfscSort,
        body: String, // LFSC program text
    },
    /// Proof step (check)
    Check(LfscTerm),
}

impl fmt::Display for LfscDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeclareSort { name, arity } => {
                write!(f, "(declare {} ", name)?;
                for _ in 0..*arity {
                    write!(f, "(! _ type ")?;
                }
                write!(f, "type")?;
                for _ in 0..*arity {
                    write!(f, ")")?;
                }
                write!(f, ")")
            }
            Self::DeclareConst { name, sort } => {
                write!(f, "(declare {} {})", name, sort)
            }
            Self::Define { name, sort, value } => {
                write!(f, "(define {} (: {} {}))", name, value, sort)
            }
            Self::DeclareRule {
                name,
                params,
                conclusion,
            } => {
                write!(f, "(declare {} ", name)?;
                for (pname, psort) in params {
                    write!(f, "(! {} {} ", pname, psort)?;
                }
                write!(f, "{}", conclusion)?;
                for _ in params {
                    write!(f, ")")?;
                }
                write!(f, ")")
            }
            Self::SideCondition {
                name,
                params,
                return_sort,
                body,
            } => {
                write!(f, "(program {} (", name)?;
                for (i, (pname, psort)) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", pname, psort)?;
                }
                write!(f, ") {} {})", return_sort, body)
            }
            Self::Check(term) => {
                write!(f, "(check {})", term)
            }
        }
    }
}

/// An LFSC proof
#[derive(Debug, Default)]
pub struct LfscProof {
    /// Declarations and proof steps
    decls: Vec<LfscDecl>,
}

impl LfscProof {
    /// Create a new empty LFSC proof
    #[must_use]
    pub fn new() -> Self {
        Self { decls: Vec::new() }
    }

    /// Add a sort declaration
    pub fn declare_sort(&mut self, name: impl Into<String>, arity: u32) {
        self.decls.push(LfscDecl::DeclareSort {
            name: name.into(),
            arity,
        });
    }

    /// Add a constant declaration
    pub fn declare_const(&mut self, name: impl Into<String>, sort: LfscSort) {
        self.decls.push(LfscDecl::DeclareConst {
            name: name.into(),
            sort,
        });
    }

    /// Add a definition
    pub fn define(&mut self, name: impl Into<String>, sort: LfscSort, value: LfscTerm) {
        self.decls.push(LfscDecl::Define {
            name: name.into(),
            sort,
            value,
        });
    }

    /// Add a proof rule declaration
    pub fn declare_rule(
        &mut self,
        name: impl Into<String>,
        params: Vec<(String, LfscSort)>,
        conclusion: LfscTerm,
    ) {
        self.decls.push(LfscDecl::DeclareRule {
            name: name.into(),
            params,
            conclusion,
        });
    }

    /// Add a side condition program
    pub fn side_condition(
        &mut self,
        name: impl Into<String>,
        params: Vec<(String, LfscSort)>,
        return_sort: LfscSort,
        body: impl Into<String>,
    ) {
        self.decls.push(LfscDecl::SideCondition {
            name: name.into(),
            params,
            return_sort,
            body: body.into(),
        });
    }

    /// Add a proof check
    pub fn check(&mut self, term: LfscTerm) {
        self.decls.push(LfscDecl::Check(term));
    }

    /// Get the number of declarations
    #[must_use]
    pub fn len(&self) -> usize {
        self.decls.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }

    /// Get the declarations
    #[must_use]
    pub fn decls(&self) -> &[LfscDecl] {
        &self.decls
    }

    /// Clear all declarations
    pub fn clear(&mut self) {
        self.decls.clear();
    }

    /// Write the proof in LFSC format
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "; LFSC proof generated by OxiZ")?;
        writeln!(writer)?;

        for decl in &self.decls {
            writeln!(writer, "{}", decl)?;
        }

        Ok(())
    }

    /// Convert to string
    ///
    /// The text is the same as [`Self::write`] produces; it is built directly
    /// into a `String`, so no I/O or UTF-8 conversion step is involved.
    #[must_use]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut out = String::from("; LFSC proof generated by OxiZ\n\n");
        for decl in &self.decls {
            out.push_str(&decl.to_string());
            out.push('\n');
        }
        out
    }
}

/// Standard LFSC signatures for common theories
pub mod signatures {
    use super::*;

    /// Create declarations for the boolean theory
    pub fn boolean_theory() -> Vec<LfscDecl> {
        vec![
            LfscDecl::DeclareSort {
                name: "formula".to_string(),
                arity: 0,
            },
            LfscDecl::DeclareConst {
                name: "true".to_string(),
                sort: LfscSort::Named("formula".to_string()),
            },
            LfscDecl::DeclareConst {
                name: "false".to_string(),
                sort: LfscSort::Named("formula".to_string()),
            },
            LfscDecl::DeclareConst {
                name: "not".to_string(),
                sort: LfscSort::Arrow(
                    Box::new(LfscSort::Named("formula".to_string())),
                    Box::new(LfscSort::Named("formula".to_string())),
                ),
            },
            LfscDecl::DeclareConst {
                name: "and".to_string(),
                sort: LfscSort::Arrow(
                    Box::new(LfscSort::Named("formula".to_string())),
                    Box::new(LfscSort::Arrow(
                        Box::new(LfscSort::Named("formula".to_string())),
                        Box::new(LfscSort::Named("formula".to_string())),
                    )),
                ),
            },
            LfscDecl::DeclareConst {
                name: "or".to_string(),
                sort: LfscSort::Arrow(
                    Box::new(LfscSort::Named("formula".to_string())),
                    Box::new(LfscSort::Arrow(
                        Box::new(LfscSort::Named("formula".to_string())),
                        Box::new(LfscSort::Named("formula".to_string())),
                    )),
                ),
            },
            LfscDecl::DeclareConst {
                name: "impl".to_string(),
                sort: LfscSort::Arrow(
                    Box::new(LfscSort::Named("formula".to_string())),
                    Box::new(LfscSort::Arrow(
                        Box::new(LfscSort::Named("formula".to_string())),
                        Box::new(LfscSort::Named("formula".to_string())),
                    )),
                ),
            },
        ]
    }

    /// Create declarations for holds (proof type)
    pub fn holds_theory() -> Vec<LfscDecl> {
        vec![LfscDecl::DeclareConst {
            name: "holds".to_string(),
            sort: LfscSort::Arrow(
                Box::new(LfscSort::Named("formula".to_string())),
                Box::new(LfscSort::Type),
            ),
        }]
    }
}

/// Trait for solvers that can produce LFSC proofs
pub trait LfscProofProducer {
    /// Enable LFSC proof production
    fn enable_lfsc_proof(&mut self);

    /// Disable LFSC proof production
    fn disable_lfsc_proof(&mut self);

    /// Get the LFSC proof (if available)
    fn get_lfsc_proof(&self) -> Option<&LfscProof>;

    /// Take the LFSC proof, leaving None
    fn take_lfsc_proof(&mut self) -> Option<LfscProof>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfsc_sort_display() {
        assert_eq!(format!("{}", LfscSort::Bool), "bool");
        assert_eq!(format!("{}", LfscSort::Int), "mpz");
        assert_eq!(format!("{}", LfscSort::BitVec(32)), "(bitvec 32)");

        let arrow = LfscSort::Arrow(Box::new(LfscSort::Int), Box::new(LfscSort::Bool));
        assert_eq!(format!("{}", arrow), "(! _ mpz bool)");
    }

    #[test]
    fn test_lfsc_term_display() {
        assert_eq!(format!("{}", LfscTerm::Var("x".to_string())), "x");
        assert_eq!(format!("{}", LfscTerm::IntLit(42)), "42");
        assert_eq!(format!("{}", LfscTerm::True), "tt");
        assert_eq!(format!("{}", LfscTerm::False), "ff");

        let app = LfscTerm::App(
            "add".to_string(),
            vec![LfscTerm::IntLit(1), LfscTerm::IntLit(2)],
        );
        assert_eq!(format!("{}", app), "(add 1 2)");
    }

    #[test]
    fn test_lfsc_declare_sort() {
        let mut proof = LfscProof::new();
        proof.declare_sort("mySort", 0);
        proof.declare_sort("myParam", 1);

        let output = proof.to_string();
        assert!(output.contains("(declare mySort type)"));
        assert!(output.contains("(declare myParam (! _ type type))"));
    }

    #[test]
    fn test_lfsc_declare_const() {
        let mut proof = LfscProof::new();
        proof.declare_const("x", LfscSort::Int);

        let output = proof.to_string();
        assert!(output.contains("(declare x mpz)"));
    }

    #[test]
    fn test_lfsc_check() {
        let mut proof = LfscProof::new();
        proof.check(LfscTerm::Hold(Box::new(LfscTerm::True)));

        let output = proof.to_string();
        assert!(output.contains("(check (holds tt))"));
    }

    #[test]
    fn test_lfsc_boolean_theory() {
        let decls = signatures::boolean_theory();
        assert!(!decls.is_empty());

        // Check that we have the expected declarations
        let names: Vec<_> = decls
            .iter()
            .filter_map(|d| match d {
                LfscDecl::DeclareSort { name, .. } => Some(name.as_str()),
                LfscDecl::DeclareConst { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();

        assert!(names.contains(&"formula"));
        assert!(names.contains(&"true"));
        assert!(names.contains(&"false"));
        assert!(names.contains(&"not"));
        assert!(names.contains(&"and"));
        assert!(names.contains(&"or"));
    }

    #[test]
    fn test_lfsc_proof_clear() {
        let mut proof = LfscProof::new();
        proof.declare_sort("test", 0);
        assert!(!proof.is_empty());

        proof.clear();
        assert!(proof.is_empty());
    }

    #[test]
    fn test_lfsc_lambda() {
        let lambda = LfscTerm::Lambda(
            "x".to_string(),
            Box::new(LfscSort::Int),
            Box::new(LfscTerm::Var("x".to_string())),
        );

        assert_eq!(format!("{}", lambda), "(\\ x mpz x)");
    }

    #[test]
    fn test_lfsc_term_deep_clone_and_drop_small_stack() {
        // `LfscTerm` is public with public variants (unbounded construction)
        // and used to keep the derived recursive `Clone`/`Drop`; either one
        // recursing once per nesting level would overflow the native stack on
        // a term deep enough to build. Both are now iterative (mirroring
        // `InterpolantTerm` in `oxiz-proof/src/craig/term.rs`). Run on a
        // deliberately small (128 KiB) stack: a stack overflow aborts the
        // process, so "the thread returned at all" is part of the assertion.
        //
        // The stack size and `depth` are scaled together on purpose: what is
        // pinned is the ratio, ~21 bytes per frame, which no real call frame
        // fits into. Never raise one without raising the other.
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let depth = 6_250usize;
                let mut term = LfscTerm::Var("x".to_string());
                for _ in 0..depth {
                    term = LfscTerm::App("f".to_string(), vec![term]);
                }

                let cloned = term.clone();

                // Walk the spine iteratively to confirm the clone is
                // structurally identical, then let both the original and the
                // clone drop at scope exit -- exercising the iterative `Drop`
                // twice.
                let mut seen = 0usize;
                let mut node = &cloned;
                while let LfscTerm::App(func, args) = node {
                    assert_eq!(func, "f");
                    assert_eq!(args.len(), 1);
                    let Some(first) = args.first() else { break };
                    node = first;
                    seen += 1;
                }
                assert_eq!(seen, depth);
                assert!(matches!(node, LfscTerm::Var(v) if v == "x"));

                drop(term);
                drop(cloned);
            })
            .expect("spawning a thread with an explicit stack size must succeed");
        handle.join().expect(
            "cloning and dropping a deeply nested LfscTerm must not overflow a 128 KiB stack",
        );
    }
}
