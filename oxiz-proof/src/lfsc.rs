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
#[derive(Debug, Clone)]
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
    #[must_use]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut buf = Vec::new();
        self.write(&mut buf)
            .expect("writing to Vec should not fail");
        String::from_utf8(buf).expect("LFSC output is UTF-8")
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
}
