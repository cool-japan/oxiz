//! SMT-LIB2 Printer
//!
//! This module provides two printers:
//! - [`Printer`]: A basic printer that outputs terms on a single line
//! - [`PrettyPrinter`]: A configurable pretty printer with indentation support

use crate::ast::{
    TermId, TermKind, TermManager, model::Model, model::ModelValue, proof::Proof, proof::ProofRule,
};
use crate::sort::{SortId, SortKind};
use std::fmt::Write;

/// Configuration for pretty printing
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PrettyConfig {
    /// Number of spaces per indentation level
    pub indent_width: usize,
    /// Maximum line width before breaking
    pub max_width: usize,
    /// Whether to use tabs instead of spaces
    pub use_tabs: bool,
    /// Whether to print sorts for terms
    pub print_sorts: bool,
    /// Minimum depth before breaking
    pub break_depth: usize,
}

impl Default for PrettyConfig {
    fn default() -> Self {
        Self {
            indent_width: 2,
            max_width: 80,
            use_tabs: false,
            print_sorts: false,
            break_depth: 2,
        }
    }
}

#[allow(dead_code)]
impl PrettyConfig {
    /// Create a compact configuration (minimal whitespace)
    #[must_use]
    pub fn compact() -> Self {
        Self {
            indent_width: 0,
            max_width: usize::MAX,
            use_tabs: false,
            print_sorts: false,
            break_depth: usize::MAX,
        }
    }

    /// Create an expanded configuration (one term per line)
    #[must_use]
    pub fn expanded() -> Self {
        Self {
            indent_width: 2,
            max_width: 40,
            use_tabs: false,
            print_sorts: false,
            break_depth: 1,
        }
    }

    /// Set the indentation width
    #[must_use]
    pub fn with_indent_width(mut self, width: usize) -> Self {
        self.indent_width = width;
        self
    }

    /// Set the maximum line width
    #[must_use]
    pub fn with_max_width(mut self, width: usize) -> Self {
        self.max_width = width;
        self
    }

    /// Set whether to use tabs
    #[must_use]
    pub fn with_tabs(mut self, use_tabs: bool) -> Self {
        self.use_tabs = use_tabs;
        self
    }

    /// Set whether to print sorts
    #[must_use]
    pub fn with_print_sorts(mut self, print_sorts: bool) -> Self {
        self.print_sorts = print_sorts;
        self
    }

    /// Set the minimum depth before breaking
    #[must_use]
    pub fn with_break_depth(mut self, depth: usize) -> Self {
        self.break_depth = depth;
        self
    }
}

/// Pretty printer for SMT-LIB2 format with configurable formatting
#[allow(dead_code)]
pub struct PrettyPrinter<'a> {
    manager: &'a TermManager,
    config: PrettyConfig,
}

#[allow(dead_code)]
impl<'a> PrettyPrinter<'a> {
    /// Create a new pretty printer with default configuration
    #[must_use]
    pub fn new(manager: &'a TermManager) -> Self {
        Self {
            manager,
            config: PrettyConfig::default(),
        }
    }

    /// Create a pretty printer with custom configuration
    #[must_use]
    pub fn with_config(manager: &'a TermManager, config: PrettyConfig) -> Self {
        Self { manager, config }
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &PrettyConfig {
        &self.config
    }

    /// Get the indentation string for a given level
    fn indent_string(&self, level: usize) -> String {
        if self.config.use_tabs {
            "\t".repeat(level)
        } else {
            " ".repeat(level * self.config.indent_width)
        }
    }

    /// Estimate the width of a term when printed
    fn term_width(&self, term_id: TermId) -> usize {
        let basic = Printer::new(self.manager);
        basic.print_term(term_id).len()
    }

    /// Check if a term should be broken across lines
    fn should_break(&self, term_id: TermId, current_indent: usize) -> bool {
        let width = self.term_width(term_id);
        current_indent + width > self.config.max_width
    }

    /// Print a term to a string with pretty formatting
    #[must_use]
    pub fn print_term(&self, term_id: TermId) -> String {
        let mut buf = String::new();
        self.write_term(&mut buf, term_id, 0, 0);
        buf
    }

    /// Write a term with indentation
    pub fn write_term(&self, w: &mut impl Write, term_id: TermId, indent: usize, depth: usize) {
        let Some(term) = self.manager.get(term_id) else {
            let _ = write!(w, "?{}", term_id.0);
            return;
        };

        let break_here = depth >= self.config.break_depth && self.should_break(term_id, indent);

        match &term.kind {
            TermKind::True => {
                let _ = write!(w, "true");
            }
            TermKind::False => {
                let _ = write!(w, "false");
            }
            TermKind::IntConst(n) => {
                let _ = write!(w, "{n}");
            }
            TermKind::RealConst(r) => {
                let _ = write!(w, "{r}");
            }
            TermKind::BitVecConst { value, width } => {
                let _ = write!(
                    w,
                    "#x{:0>width$}",
                    value,
                    width = (*width as usize).div_ceil(4)
                );
            }
            TermKind::StringLit(s) => {
                let _ = write!(w, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
            }
            TermKind::Var(spur) => {
                let name = self.manager.resolve_str(*spur);
                let _ = write!(w, "{name}");
            }
            TermKind::Not(arg) => {
                let _ = write!(w, "(not ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => {
                let op = match &term.kind {
                    TermKind::And(_) => "and",
                    TermKind::Or(_) => "or",
                    TermKind::Add(_) => "+",
                    TermKind::Mul(_) => "*",
                    _ => unreachable!(),
                };
                self.write_nary_term(w, op, args, indent, depth, break_here);
            }
            TermKind::Distinct(args) => {
                self.write_nary_term(w, "distinct", args, indent, depth, break_here);
            }
            TermKind::Xor(lhs, rhs) => {
                self.write_binary_term(w, "xor", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Implies(lhs, rhs) => {
                self.write_binary_term(w, "=>", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Eq(lhs, rhs) => {
                self.write_binary_term(w, "=", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Sub(lhs, rhs) => {
                self.write_binary_term(w, "-", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Div(lhs, rhs) => {
                self.write_binary_term(w, "div", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Mod(lhs, rhs) => {
                self.write_binary_term(w, "mod", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Lt(lhs, rhs) => {
                self.write_binary_term(w, "<", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Le(lhs, rhs) => {
                self.write_binary_term(w, "<=", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Gt(lhs, rhs) => {
                self.write_binary_term(w, ">", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Ge(lhs, rhs) => {
                self.write_binary_term(w, ">=", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::Neg(arg) => {
                let _ = write!(w, "(- ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::StrLen(arg) => {
                let _ = write!(w, "(str.len ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::StrToInt(arg) => {
                let _ = write!(w, "(str.to_int ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::IntToStr(arg) => {
                let _ = write!(w, "(int.to_str ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::Ite(cond, then_b, else_b) => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _indent_str = self.indent_string(1);
                    let _ = write!(
                        w,
                        "(ite\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *cond, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *then_b, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *else_b, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(ite ");
                    self.write_term(w, *cond, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *then_b, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *else_b, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            TermKind::Select(array, index) => {
                self.write_binary_term(w, "select", *array, *index, indent, depth, break_here);
            }
            TermKind::StrConcat(s1, s2) => {
                self.write_binary_term(w, "str.++", *s1, *s2, indent, depth, break_here);
            }
            TermKind::StrAt(s, i) => {
                self.write_binary_term(w, "str.at", *s, *i, indent, depth, break_here);
            }
            TermKind::StrContains(s, sub) => {
                self.write_binary_term(w, "str.contains", *s, *sub, indent, depth, break_here);
            }
            TermKind::StrPrefixOf(prefix, s) => {
                self.write_binary_term(w, "str.prefixof", *prefix, *s, indent, depth, break_here);
            }
            TermKind::StrSuffixOf(suffix, s) => {
                self.write_binary_term(w, "str.suffixof", *suffix, *s, indent, depth, break_here);
            }
            TermKind::StrInRe(s, re) => {
                self.write_binary_term(w, "str.in_re", *s, *re, indent, depth, break_here);
            }
            TermKind::StrSubstr(s, i, n) => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(
                        w,
                        "(str.substr\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *s, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *i, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *n, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(str.substr ");
                    self.write_term(w, *s, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *i, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *n, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            TermKind::StrIndexOf(s, sub, offset) => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(
                        w,
                        "(str.indexof\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *s, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *sub, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *offset, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(str.indexof ");
                    self.write_term(w, *s, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *sub, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *offset, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            TermKind::StrReplace(s, from, to) => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(
                        w,
                        "(str.replace\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *s, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *from, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *to, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(str.replace ");
                    self.write_term(w, *s, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *from, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *to, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            TermKind::StrReplaceAll(s, from, to) => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(
                        w,
                        "(str.replace_all\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *s, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *from, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *to, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(str.replace_all ");
                    self.write_term(w, *s, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *from, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *to, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            TermKind::Store(array, index, value) => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(
                        w,
                        "(store\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *array, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *index, inner_indent, depth + 1);
                    let _ = write!(
                        w,
                        "\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *value, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(store ");
                    self.write_term(w, *array, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *index, indent, depth + 1);
                    let _ = write!(w, " ");
                    self.write_term(w, *value, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            TermKind::Apply { func, args } => {
                let name = self.manager.resolve_str(*func);
                if break_here && !args.is_empty() {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(w, "({name}");
                    for arg in args {
                        let _ = write!(
                            w,
                            "\n{}",
                            self.indent_string(inner_indent / self.config.indent_width.max(1))
                        );
                        self.write_term(w, *arg, inner_indent, depth + 1);
                    }
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "({name}");
                    for arg in args {
                        let _ = write!(w, " ");
                        self.write_term(w, *arg, indent, depth + 1);
                    }
                    let _ = write!(w, ")");
                }
            }
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                self.write_quantifier(
                    w, "forall", vars, *body, patterns, indent, depth, break_here,
                );
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                self.write_quantifier(
                    w, "exists", vars, *body, patterns, indent, depth, break_here,
                );
            }
            TermKind::Let { bindings, body } => {
                if break_here {
                    let inner_indent = indent + self.config.indent_width;
                    let _ = write!(w, "(let (");
                    for (i, (name, term)) in bindings.iter().enumerate() {
                        if i > 0 {
                            let _ = write!(
                                w,
                                "\n{}",
                                self.indent_string(
                                    (inner_indent + self.config.indent_width)
                                        / self.config.indent_width.max(1)
                                )
                            );
                        }
                        let name_str = self.manager.resolve_str(*name);
                        let _ = write!(w, "({name_str} ");
                        self.write_term(
                            w,
                            *term,
                            inner_indent + self.config.indent_width,
                            depth + 1,
                        );
                        let _ = write!(w, ")");
                    }
                    let _ = write!(
                        w,
                        ")\n{}",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    self.write_term(w, *body, inner_indent, depth + 1);
                    let _ = write!(w, ")");
                } else {
                    let _ = write!(w, "(let (");
                    for (i, (name, term)) in bindings.iter().enumerate() {
                        if i > 0 {
                            let _ = write!(w, " ");
                        }
                        let name_str = self.manager.resolve_str(*name);
                        let _ = write!(w, "({name_str} ");
                        self.write_term(w, *term, indent, depth + 1);
                        let _ = write!(w, ")");
                    }
                    let _ = write!(w, ") ");
                    self.write_term(w, *body, indent, depth + 1);
                    let _ = write!(w, ")");
                }
            }
            // BitVector operations (inline)
            TermKind::BvConcat(lhs, rhs) => {
                self.write_binary_term(w, "concat", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::BvExtract { high, low, arg } => {
                let _ = write!(w, "((_ extract {high} {low}) ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::BvNot(arg) => {
                let _ = write!(w, "(bvnot ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::BvAnd(lhs, rhs) => {
                self.write_binary_term(w, "bvand", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvOr(lhs, rhs) => {
                self.write_binary_term(w, "bvor", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvXor(lhs, rhs) => {
                self.write_binary_term(w, "bvxor", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvAdd(lhs, rhs) => {
                self.write_binary_term(w, "bvadd", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvSub(lhs, rhs) => {
                self.write_binary_term(w, "bvsub", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvMul(lhs, rhs) => {
                self.write_binary_term(w, "bvmul", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvUdiv(lhs, rhs) => {
                self.write_binary_term(w, "bvudiv", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvSdiv(lhs, rhs) => {
                self.write_binary_term(w, "bvsdiv", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvUrem(lhs, rhs) => {
                self.write_binary_term(w, "bvurem", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvSrem(lhs, rhs) => {
                self.write_binary_term(w, "bvsrem", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvShl(lhs, rhs) => {
                self.write_binary_term(w, "bvshl", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvLshr(lhs, rhs) => {
                self.write_binary_term(w, "bvlshr", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvAshr(lhs, rhs) => {
                self.write_binary_term(w, "bvashr", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvUlt(lhs, rhs) => {
                self.write_binary_term(w, "bvult", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvUle(lhs, rhs) => {
                self.write_binary_term(w, "bvule", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvSlt(lhs, rhs) => {
                self.write_binary_term(w, "bvslt", *lhs, *rhs, indent, depth, false);
            }
            TermKind::BvSle(lhs, rhs) => {
                self.write_binary_term(w, "bvsle", *lhs, *rhs, indent, depth, false);
            }
            // Floating-point literals and constants
            TermKind::FpLit { sign, exp, sig, .. } => {
                let _ = write!(
                    w,
                    "(fp #b{} #b{} #b{})",
                    if *sign { "1" } else { "0" },
                    exp,
                    sig
                );
            }
            TermKind::FpPlusInfinity { eb, sb } => {
                let _ = write!(w, "(_ +oo {eb} {sb})");
            }
            TermKind::FpMinusInfinity { eb, sb } => {
                let _ = write!(w, "(_ -oo {eb} {sb})");
            }
            TermKind::FpPlusZero { eb, sb } => {
                let _ = write!(w, "(_ +zero {eb} {sb})");
            }
            TermKind::FpMinusZero { eb, sb } => {
                let _ = write!(w, "(_ -zero {eb} {sb})");
            }
            TermKind::FpNaN { eb, sb } => {
                let _ = write!(w, "(_ NaN {eb} {sb})");
            }
            // Unary FP operations
            TermKind::FpAbs(arg) => {
                let _ = write!(w, "(fp.abs ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpNeg(arg) => {
                let _ = write!(w, "(fp.neg ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpSqrt(rm, arg) => {
                let _ = write!(w, "(fp.sqrt {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpRoundToIntegral(rm, arg) => {
                let _ = write!(w, "(fp.roundToIntegral {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            // Binary FP operations
            TermKind::FpAdd(rm, a, b) => {
                let _ = write!(w, "(fp.add {rm:?} ");
                self.write_term(w, *a, indent, depth + 1);
                let _ = write!(w, " ");
                self.write_term(w, *b, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpSub(rm, a, b) => {
                let _ = write!(w, "(fp.sub {rm:?} ");
                self.write_term(w, *a, indent, depth + 1);
                let _ = write!(w, " ");
                self.write_term(w, *b, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpMul(rm, a, b) => {
                let _ = write!(w, "(fp.mul {rm:?} ");
                self.write_term(w, *a, indent, depth + 1);
                let _ = write!(w, " ");
                self.write_term(w, *b, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpDiv(rm, a, b) => {
                let _ = write!(w, "(fp.div {rm:?} ");
                self.write_term(w, *a, indent, depth + 1);
                let _ = write!(w, " ");
                self.write_term(w, *b, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpRem(a, b) => {
                self.write_binary_term(w, "fp.rem", *a, *b, indent, depth, false);
            }
            TermKind::FpMin(a, b) => {
                self.write_binary_term(w, "fp.min", *a, *b, indent, depth, false);
            }
            TermKind::FpMax(a, b) => {
                self.write_binary_term(w, "fp.max", *a, *b, indent, depth, false);
            }
            TermKind::FpLeq(a, b) => {
                self.write_binary_term(w, "fp.leq", *a, *b, indent, depth, false);
            }
            TermKind::FpLt(a, b) => {
                self.write_binary_term(w, "fp.lt", *a, *b, indent, depth, false);
            }
            TermKind::FpGeq(a, b) => {
                self.write_binary_term(w, "fp.geq", *a, *b, indent, depth, false);
            }
            TermKind::FpGt(a, b) => {
                self.write_binary_term(w, "fp.gt", *a, *b, indent, depth, false);
            }
            TermKind::FpEq(a, b) => {
                self.write_binary_term(w, "fp.eq", *a, *b, indent, depth, false);
            }
            // Ternary FP operations
            TermKind::FpFma(rm, a, b, c) => {
                let _ = write!(w, "(fp.fma {rm:?} ");
                self.write_term(w, *a, indent, depth + 1);
                let _ = write!(w, " ");
                self.write_term(w, *b, indent, depth + 1);
                let _ = write!(w, " ");
                self.write_term(w, *c, indent, depth + 1);
                let _ = write!(w, ")");
            }
            // FP predicates
            TermKind::FpIsNormal(arg) => {
                let _ = write!(w, "(fp.isNormal ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpIsSubnormal(arg) => {
                let _ = write!(w, "(fp.isSubnormal ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpIsZero(arg) => {
                let _ = write!(w, "(fp.isZero ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpIsInfinite(arg) => {
                let _ = write!(w, "(fp.isInfinite ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpIsNaN(arg) => {
                let _ = write!(w, "(fp.isNaN ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpIsNegative(arg) => {
                let _ = write!(w, "(fp.isNegative ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpIsPositive(arg) => {
                let _ = write!(w, "(fp.isPositive ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            // FP conversions
            TermKind::FpToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpToSBV { rm, arg, width } => {
                let _ = write!(w, "((_ fp.to_sbv {width}) {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpToUBV { rm, arg, width } => {
                let _ = write!(w, "((_ fp.to_ubv {width}) {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::FpToReal(arg) => {
                let _ = write!(w, "(fp.to_real ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::RealToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::SBVToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::UBVToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp_unsigned {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }

            // Algebraic datatypes
            TermKind::DtConstructor { constructor, args } => {
                let name = self.manager.resolve_str(*constructor);
                let _ = write!(w, "({name}");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg, indent, depth + 1);
                }
                let _ = write!(w, ")");
            }
            TermKind::DtTester { constructor, arg } => {
                let name = self.manager.resolve_str(*constructor);
                let _ = write!(w, "(is-{name} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::DtSelector { selector, arg } => {
                let name = self.manager.resolve_str(*selector);
                let _ = write!(w, "({name} ");
                self.write_term(w, *arg, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::Match { scrutinee, cases } => {
                let _ = write!(w, "(match ");
                self.write_term(w, *scrutinee, indent, depth + 1);
                let _ = write!(w, " (");
                for (i, case) in cases.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    let _ = write!(w, "(");
                    if let Some(ctor) = case.constructor {
                        let ctor_name = self.manager.resolve_str(ctor);
                        if case.bindings.is_empty() {
                            let _ = write!(w, "{ctor_name}");
                        } else {
                            let _ = write!(w, "({ctor_name}");
                            for binding in &case.bindings {
                                let binding_name = self.manager.resolve_str(*binding);
                                let _ = write!(w, " {binding_name}");
                            }
                            let _ = write!(w, ")");
                        }
                    } else if let Some(first_binding) = case.bindings.first() {
                        let binding_name = self.manager.resolve_str(*first_binding);
                        let _ = write!(w, "{binding_name}");
                    }
                    let _ = write!(w, " ");
                    self.write_term(w, case.body, indent, depth + 1);
                    let _ = write!(w, ")");
                }
                let _ = write!(w, "))");
            }
        }
    }

    /// Helper for writing binary operations
    #[allow(clippy::too_many_arguments)]
    fn write_binary_term(
        &self,
        w: &mut impl Write,
        op: &str,
        lhs: TermId,
        rhs: TermId,
        indent: usize,
        depth: usize,
        break_here: bool,
    ) {
        if break_here {
            let inner_indent = indent + self.config.indent_width;
            let _ = write!(
                w,
                "({op}\n{}",
                self.indent_string(inner_indent / self.config.indent_width.max(1))
            );
            self.write_term(w, lhs, inner_indent, depth + 1);
            let _ = write!(
                w,
                "\n{}",
                self.indent_string(inner_indent / self.config.indent_width.max(1))
            );
            self.write_term(w, rhs, inner_indent, depth + 1);
            let _ = write!(w, ")");
        } else {
            let _ = write!(w, "({op} ");
            self.write_term(w, lhs, indent, depth + 1);
            let _ = write!(w, " ");
            self.write_term(w, rhs, indent, depth + 1);
            let _ = write!(w, ")");
        }
    }

    /// Helper for writing n-ary operations
    fn write_nary_term(
        &self,
        w: &mut impl Write,
        op: &str,
        args: &smallvec::SmallVec<[TermId; 4]>,
        indent: usize,
        depth: usize,
        break_here: bool,
    ) {
        if break_here && args.len() > 2 {
            let inner_indent = indent + self.config.indent_width;
            let _ = write!(w, "({op}");
            for arg in args {
                let _ = write!(
                    w,
                    "\n{}",
                    self.indent_string(inner_indent / self.config.indent_width.max(1))
                );
                self.write_term(w, *arg, inner_indent, depth + 1);
            }
            let _ = write!(w, ")");
        } else {
            let _ = write!(w, "({op}");
            for arg in args {
                let _ = write!(w, " ");
                self.write_term(w, *arg, indent, depth + 1);
            }
            let _ = write!(w, ")");
        }
    }

    /// Helper for writing quantifiers
    #[allow(clippy::too_many_arguments)]
    fn write_quantifier(
        &self,
        w: &mut impl Write,
        quantifier: &str,
        vars: &smallvec::SmallVec<[(lasso::Spur, crate::sort::SortId); 2]>,
        body: TermId,
        patterns: &smallvec::SmallVec<[smallvec::SmallVec<[TermId; 2]>; 2]>,
        indent: usize,
        depth: usize,
        break_here: bool,
    ) {
        // If there are patterns, we need to wrap the body in an annotation
        let has_patterns = !patterns.is_empty();

        if break_here {
            let inner_indent = indent + self.config.indent_width;
            let _ = write!(w, "({quantifier} (");
            for (i, (name, sort)) in vars.iter().enumerate() {
                if i > 0 {
                    let _ = write!(w, " ");
                }
                let name_str = self.manager.resolve_str(*name);
                let _ = write!(w, "({name_str} ");
                self.write_sort(w, *sort);
                let _ = write!(w, ")");
            }
            let _ = write!(
                w,
                ")\n{}",
                self.indent_string(inner_indent / self.config.indent_width.max(1))
            );

            if has_patterns {
                let _ = write!(w, "(! ");
                self.write_term(w, body, inner_indent, depth + 1);
                for pattern in patterns {
                    let _ = write!(
                        w,
                        "\n{}:pattern (",
                        self.indent_string(inner_indent / self.config.indent_width.max(1))
                    );
                    for (i, term) in pattern.iter().enumerate() {
                        if i > 0 {
                            let _ = write!(w, " ");
                        }
                        self.write_term(w, *term, inner_indent, depth + 2);
                    }
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ")");
            } else {
                self.write_term(w, body, inner_indent, depth + 1);
            }
            let _ = write!(w, ")");
        } else {
            let _ = write!(w, "({quantifier} (");
            for (i, (name, sort)) in vars.iter().enumerate() {
                if i > 0 {
                    let _ = write!(w, " ");
                }
                let name_str = self.manager.resolve_str(*name);
                let _ = write!(w, "({name_str} ");
                self.write_sort(w, *sort);
                let _ = write!(w, ")");
            }
            let _ = write!(w, ") ");

            if has_patterns {
                let _ = write!(w, "(! ");
                self.write_term(w, body, indent, depth + 1);
                for pattern in patterns {
                    let _ = write!(w, " :pattern (");
                    for (i, term) in pattern.iter().enumerate() {
                        if i > 0 {
                            let _ = write!(w, " ");
                        }
                        self.write_term(w, *term, indent, depth + 2);
                    }
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ")");
            } else {
                self.write_term(w, body, indent, depth + 1);
            }
            let _ = write!(w, ")");
        }
    }

    /// Write a sort
    pub fn write_sort(&self, w: &mut impl Write, sort_id: SortId) {
        let basic = Printer::new(self.manager);
        basic.write_sort(w, sort_id);
    }

    /// Print a sort to a string
    #[must_use]
    pub fn print_sort(&self, sort_id: SortId) -> String {
        let basic = Printer::new(self.manager);
        basic.print_sort(sort_id)
    }
}

/// Printer for SMT-LIB2 format
pub struct Printer<'a> {
    manager: &'a TermManager,
}

impl<'a> Printer<'a> {
    /// Create a new printer
    #[must_use]
    pub fn new(manager: &'a TermManager) -> Self {
        Self { manager }
    }

    /// Print a term to a string
    #[must_use]
    pub fn print_term(&self, term_id: TermId) -> String {
        let mut buf = String::new();
        self.write_term(&mut buf, term_id);
        buf
    }

    /// Write a term to a writer
    pub fn write_term(&self, w: &mut impl Write, term_id: TermId) {
        let Some(term) = self.manager.get(term_id) else {
            let _ = write!(w, "?{}", term_id.0);
            return;
        };

        match &term.kind {
            TermKind::True => {
                let _ = write!(w, "true");
            }
            TermKind::False => {
                let _ = write!(w, "false");
            }
            TermKind::IntConst(n) => {
                let _ = write!(w, "{n}");
            }
            TermKind::RealConst(r) => {
                let _ = write!(w, "{r}");
            }
            TermKind::BitVecConst { value, width } => {
                let _ = write!(
                    w,
                    "#x{:0>width$}",
                    value,
                    width = (*width as usize).div_ceil(4)
                );
            }
            TermKind::StringLit(s) => {
                let _ = write!(w, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
            }
            TermKind::Var(spur) => {
                let name = self.manager.resolve_str(*spur);
                let _ = write!(w, "{name}");
            }
            TermKind::Not(arg) => {
                let _ = write!(w, "(not ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::And(args) => {
                let _ = write!(w, "(and");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::Or(args) => {
                let _ = write!(w, "(or");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::Xor(lhs, rhs) => {
                let _ = write!(w, "(xor ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Implies(lhs, rhs) => {
                let _ = write!(w, "(=> ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Ite(cond, then_b, else_b) => {
                let _ = write!(w, "(ite ");
                self.write_term(w, *cond);
                let _ = write!(w, " ");
                self.write_term(w, *then_b);
                let _ = write!(w, " ");
                self.write_term(w, *else_b);
                let _ = write!(w, ")");
            }
            TermKind::Eq(lhs, rhs) => {
                let _ = write!(w, "(= ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Distinct(args) => {
                let _ = write!(w, "(distinct");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::Neg(arg) => {
                let _ = write!(w, "(- ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::StrLen(arg) => {
                let _ = write!(w, "(str.len ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::StrToInt(arg) => {
                let _ = write!(w, "(str.to_int ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::IntToStr(arg) => {
                let _ = write!(w, "(int.to_str ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::Add(args) => {
                let _ = write!(w, "(+");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::Sub(lhs, rhs) => {
                let _ = write!(w, "(- ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Mul(args) => {
                let _ = write!(w, "(*");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::Div(lhs, rhs) => {
                let _ = write!(w, "(div ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Mod(lhs, rhs) => {
                let _ = write!(w, "(mod ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Lt(lhs, rhs) => {
                let _ = write!(w, "(< ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Le(lhs, rhs) => {
                let _ = write!(w, "(<= ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Gt(lhs, rhs) => {
                let _ = write!(w, "(> ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Ge(lhs, rhs) => {
                let _ = write!(w, "(>= ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::Select(array, index) => {
                let _ = write!(w, "(select ");
                self.write_term(w, *array);
                let _ = write!(w, " ");
                self.write_term(w, *index);
                let _ = write!(w, ")");
            }
            TermKind::StrConcat(s1, s2) => {
                let _ = write!(w, "(str.++ ");
                self.write_term(w, *s1);
                let _ = write!(w, " ");
                self.write_term(w, *s2);
                let _ = write!(w, ")");
            }
            TermKind::StrAt(s, i) => {
                let _ = write!(w, "(str.at ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *i);
                let _ = write!(w, ")");
            }
            TermKind::StrContains(s, sub) => {
                let _ = write!(w, "(str.contains ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *sub);
                let _ = write!(w, ")");
            }
            TermKind::StrPrefixOf(prefix, s) => {
                let _ = write!(w, "(str.prefixof ");
                self.write_term(w, *prefix);
                let _ = write!(w, " ");
                self.write_term(w, *s);
                let _ = write!(w, ")");
            }
            TermKind::StrSuffixOf(suffix, s) => {
                let _ = write!(w, "(str.suffixof ");
                self.write_term(w, *suffix);
                let _ = write!(w, " ");
                self.write_term(w, *s);
                let _ = write!(w, ")");
            }
            TermKind::StrInRe(s, re) => {
                let _ = write!(w, "(str.in_re ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *re);
                let _ = write!(w, ")");
            }
            TermKind::StrSubstr(s, i, n) => {
                let _ = write!(w, "(str.substr ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *i);
                let _ = write!(w, " ");
                self.write_term(w, *n);
                let _ = write!(w, ")");
            }
            TermKind::StrIndexOf(s, sub, offset) => {
                let _ = write!(w, "(str.indexof ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *sub);
                let _ = write!(w, " ");
                self.write_term(w, *offset);
                let _ = write!(w, ")");
            }
            TermKind::StrReplace(s, from, to) => {
                let _ = write!(w, "(str.replace ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *from);
                let _ = write!(w, " ");
                self.write_term(w, *to);
                let _ = write!(w, ")");
            }
            TermKind::StrReplaceAll(s, from, to) => {
                let _ = write!(w, "(str.replace_all ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *from);
                let _ = write!(w, " ");
                self.write_term(w, *to);
                let _ = write!(w, ")");
            }
            TermKind::Store(array, index, value) => {
                let _ = write!(w, "(store ");
                self.write_term(w, *array);
                let _ = write!(w, " ");
                self.write_term(w, *index);
                let _ = write!(w, " ");
                self.write_term(w, *value);
                let _ = write!(w, ")");
            }
            TermKind::Apply { func, args } => {
                let name = self.manager.resolve_str(*func);
                let _ = write!(w, "({name}");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                let _ = write!(w, "(forall (");
                for (i, (name, sort)) in vars.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    let name_str = self.manager.resolve_str(*name);
                    let _ = write!(w, "({name_str} ");
                    self.write_sort(w, *sort);
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ") ");
                if patterns.is_empty() {
                    self.write_term(w, *body);
                } else {
                    let _ = write!(w, "(! ");
                    self.write_term(w, *body);
                    for pattern in patterns {
                        let _ = write!(w, " :pattern (");
                        for (i, term) in pattern.iter().enumerate() {
                            if i > 0 {
                                let _ = write!(w, " ");
                            }
                            self.write_term(w, *term);
                        }
                        let _ = write!(w, ")");
                    }
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ")");
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                let _ = write!(w, "(exists (");
                for (i, (name, sort)) in vars.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    let name_str = self.manager.resolve_str(*name);
                    let _ = write!(w, "({name_str} ");
                    self.write_sort(w, *sort);
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ") ");
                if patterns.is_empty() {
                    self.write_term(w, *body);
                } else {
                    let _ = write!(w, "(! ");
                    self.write_term(w, *body);
                    for pattern in patterns {
                        let _ = write!(w, " :pattern (");
                        for (i, term) in pattern.iter().enumerate() {
                            if i > 0 {
                                let _ = write!(w, " ");
                            }
                            self.write_term(w, *term);
                        }
                        let _ = write!(w, ")");
                    }
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ")");
            }
            TermKind::Let { bindings, body } => {
                let _ = write!(w, "(let (");
                for (i, (name, term)) in bindings.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    let name_str = self.manager.resolve_str(*name);
                    let _ = write!(w, "({name_str} ");
                    self.write_term(w, *term);
                    let _ = write!(w, ")");
                }
                let _ = write!(w, ") ");
                self.write_term(w, *body);
                let _ = write!(w, ")");
            }
            // BitVector operations
            TermKind::BvConcat(lhs, rhs) => {
                let _ = write!(w, "(concat ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvExtract { high, low, arg } => {
                let _ = write!(w, "((_ extract {high} {low}) ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::BvNot(arg) => {
                let _ = write!(w, "(bvnot ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::BvAnd(lhs, rhs) => {
                let _ = write!(w, "(bvand ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvOr(lhs, rhs) => {
                let _ = write!(w, "(bvor ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvXor(lhs, rhs) => {
                let _ = write!(w, "(bvxor ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvAdd(lhs, rhs) => {
                let _ = write!(w, "(bvadd ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvSub(lhs, rhs) => {
                let _ = write!(w, "(bvsub ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvMul(lhs, rhs) => {
                let _ = write!(w, "(bvmul ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvUdiv(lhs, rhs) => {
                let _ = write!(w, "(bvudiv ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvSdiv(lhs, rhs) => {
                let _ = write!(w, "(bvsdiv ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvUrem(lhs, rhs) => {
                let _ = write!(w, "(bvurem ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvSrem(lhs, rhs) => {
                let _ = write!(w, "(bvsrem ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvShl(lhs, rhs) => {
                let _ = write!(w, "(bvshl ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvLshr(lhs, rhs) => {
                let _ = write!(w, "(bvlshr ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvAshr(lhs, rhs) => {
                let _ = write!(w, "(bvashr ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvUlt(lhs, rhs) => {
                let _ = write!(w, "(bvult ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvUle(lhs, rhs) => {
                let _ = write!(w, "(bvule ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvSlt(lhs, rhs) => {
                let _ = write!(w, "(bvslt ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::BvSle(lhs, rhs) => {
                let _ = write!(w, "(bvsle ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            // Floating-point literals and constants
            TermKind::FpLit { sign, exp, sig, .. } => {
                let _ = write!(
                    w,
                    "(fp #b{} #b{} #b{})",
                    if *sign { "1" } else { "0" },
                    exp,
                    sig
                );
            }
            TermKind::FpPlusInfinity { eb, sb } => {
                let _ = write!(w, "(_ +oo {eb} {sb})");
            }
            TermKind::FpMinusInfinity { eb, sb } => {
                let _ = write!(w, "(_ -oo {eb} {sb})");
            }
            TermKind::FpPlusZero { eb, sb } => {
                let _ = write!(w, "(_ +zero {eb} {sb})");
            }
            TermKind::FpMinusZero { eb, sb } => {
                let _ = write!(w, "(_ -zero {eb} {sb})");
            }
            TermKind::FpNaN { eb, sb } => {
                let _ = write!(w, "(_ NaN {eb} {sb})");
            }
            // Unary FP operations
            TermKind::FpAbs(arg) => {
                let _ = write!(w, "(fp.abs ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpNeg(arg) => {
                let _ = write!(w, "(fp.neg ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpSqrt(rm, arg) => {
                let _ = write!(w, "(fp.sqrt {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpRoundToIntegral(rm, arg) => {
                let _ = write!(w, "(fp.roundToIntegral {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            // Binary FP operations
            TermKind::FpAdd(rm, lhs, rhs) => {
                let _ = write!(w, "(fp.add {rm:?} ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpSub(rm, lhs, rhs) => {
                let _ = write!(w, "(fp.sub {rm:?} ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpMul(rm, lhs, rhs) => {
                let _ = write!(w, "(fp.mul {rm:?} ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpDiv(rm, lhs, rhs) => {
                let _ = write!(w, "(fp.div {rm:?} ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpRem(lhs, rhs) => {
                let _ = write!(w, "(fp.rem ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpMin(lhs, rhs) => {
                let _ = write!(w, "(fp.min ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpMax(lhs, rhs) => {
                let _ = write!(w, "(fp.max ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpLeq(lhs, rhs) => {
                let _ = write!(w, "(fp.leq ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpLt(lhs, rhs) => {
                let _ = write!(w, "(fp.lt ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpGeq(lhs, rhs) => {
                let _ = write!(w, "(fp.geq ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpGt(lhs, rhs) => {
                let _ = write!(w, "(fp.gt ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::FpEq(lhs, rhs) => {
                let _ = write!(w, "(fp.eq ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            // Ternary FP operations
            TermKind::FpFma(rm, a, b, c) => {
                let _ = write!(w, "(fp.fma {rm:?} ");
                self.write_term(w, *a);
                let _ = write!(w, " ");
                self.write_term(w, *b);
                let _ = write!(w, " ");
                self.write_term(w, *c);
                let _ = write!(w, ")");
            }
            // FP predicates
            TermKind::FpIsNormal(arg) => {
                let _ = write!(w, "(fp.isNormal ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpIsSubnormal(arg) => {
                let _ = write!(w, "(fp.isSubnormal ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpIsZero(arg) => {
                let _ = write!(w, "(fp.isZero ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpIsInfinite(arg) => {
                let _ = write!(w, "(fp.isInfinite ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpIsNaN(arg) => {
                let _ = write!(w, "(fp.isNaN ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpIsNegative(arg) => {
                let _ = write!(w, "(fp.isNegative ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpIsPositive(arg) => {
                let _ = write!(w, "(fp.isPositive ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            // FP conversions
            TermKind::FpToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpToSBV { rm, arg, width } => {
                let _ = write!(w, "((_ fp.to_sbv {width}) {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpToUBV { rm, arg, width } => {
                let _ = write!(w, "((_ fp.to_ubv {width}) {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::FpToReal(arg) => {
                let _ = write!(w, "(fp.to_real ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::RealToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::SBVToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::UBVToFp { rm, arg, eb, sb } => {
                let _ = write!(w, "((_ to_fp_unsigned {eb} {sb}) {rm:?} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }

            // Algebraic datatypes
            TermKind::DtConstructor { constructor, args } => {
                let name = self.manager.resolve_str(*constructor);
                let _ = write!(w, "({name}");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_term(w, *arg);
                }
                let _ = write!(w, ")");
            }
            TermKind::DtTester { constructor, arg } => {
                let name = self.manager.resolve_str(*constructor);
                let _ = write!(w, "(is-{name} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }
            TermKind::DtSelector { selector, arg } => {
                let name = self.manager.resolve_str(*selector);
                let _ = write!(w, "({name} ");
                self.write_term(w, *arg);
                let _ = write!(w, ")");
            }

            // Match expressions
            TermKind::Match { scrutinee, cases } => {
                let _ = write!(w, "(match ");
                self.write_term(w, *scrutinee);
                let _ = write!(w, " (");
                for (i, case) in cases.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    let _ = write!(w, "(");
                    if let Some(constructor) = case.constructor {
                        let _ = write!(w, "(");
                        let name = self.manager.resolve_str(constructor);
                        let _ = write!(w, "{name}");
                        for binding in &case.bindings {
                            let binding_name = self.manager.resolve_str(*binding);
                            let _ = write!(w, " {binding_name}");
                        }
                        let _ = write!(w, ")");
                    } else {
                        // Wildcard pattern
                        let _ = write!(w, "_");
                    }
                    let _ = write!(w, " ");
                    self.write_term(w, case.body);
                    let _ = write!(w, ")");
                }
                let _ = write!(w, "))");
            }
        }
    }

    /// Write a sort to a writer
    pub fn write_sort(&self, w: &mut impl Write, sort_id: SortId) {
        let Some(sort) = self.manager.sorts.get(sort_id) else {
            let _ = write!(w, "?Sort{}", sort_id.0);
            return;
        };

        match &sort.kind {
            SortKind::Bool => {
                let _ = write!(w, "Bool");
            }
            SortKind::Int => {
                let _ = write!(w, "Int");
            }
            SortKind::Real => {
                let _ = write!(w, "Real");
            }
            SortKind::BitVec(width) => {
                let _ = write!(w, "(_ BitVec {width})");
            }
            SortKind::String => {
                let _ = write!(w, "String");
            }
            SortKind::FloatingPoint { eb, sb } => {
                let _ = write!(w, "(_ FloatingPoint {eb} {sb})");
            }
            SortKind::Array { domain, range } => {
                let _ = write!(w, "(Array ");
                self.write_sort(w, *domain);
                let _ = write!(w, " ");
                self.write_sort(w, *range);
                let _ = write!(w, ")");
            }
            SortKind::Uninterpreted(spur) => {
                let name = self.manager.resolve_str(*spur);
                let _ = write!(w, "{name}");
            }
            SortKind::Parameter(spur) => {
                let name = self.manager.resolve_str(*spur);
                let _ = write!(w, "{name}");
            }
            SortKind::Parametric { name, args } => {
                let name_str = self.manager.resolve_str(*name);
                let _ = write!(w, "({name_str}");
                for arg in args {
                    let _ = write!(w, " ");
                    self.write_sort(w, *arg);
                }
                let _ = write!(w, ")");
            }
            SortKind::Datatype(spur) => {
                let name = self.manager.resolve_str(*spur);
                let _ = write!(w, "{name}");
            }
        }
    }

    /// Print a sort to a string
    #[must_use]
    pub fn print_sort(&self, sort_id: SortId) -> String {
        let mut buf = String::new();
        self.write_sort(&mut buf, sort_id);
        buf
    }

    /// Print a model in SMT-LIB2 format
    #[must_use]
    pub fn print_model(&self, model: &Model) -> String {
        let mut buf = String::new();
        self.write_model(&mut buf, model);
        buf
    }

    /// Write a model in SMT-LIB2 format
    pub fn write_model(&self, w: &mut impl Write, model: &Model) {
        let _ = writeln!(w, "(model");

        // Print variable assignments as define-fun declarations
        for (term_id, value) in model.assignments() {
            if let Some(term) = self.manager.get(*term_id)
                && let crate::ast::TermKind::Var(name_spur) = term.kind
            {
                let var_name = self.manager.resolve_str(name_spur);
                let _ = write!(w, "  (define-fun {} () ", var_name);
                self.write_sort(w, term.sort);
                let _ = write!(w, " ");
                self.write_model_value(w, value);
                let _ = writeln!(w, ")");
            }
        }

        // Print function interpretations
        for (name_spur, func_interp) in model.functions() {
            let func_name = self.manager.resolve_str(*name_spur);
            let _ = write!(w, "  (define-fun {} ", func_name);

            // For now, we'll print a simplified version
            // A full implementation would need parameter sorts from the function signature
            if func_interp.table().is_empty() {
                // Just print the default value if there is one
                if let Some(default) = func_interp.default_value() {
                    let _ = write!(w, "() ");
                    // We'd need sort information here
                    let _ = write!(w, "Int "); // Placeholder
                    self.write_model_value(w, default);
                    let _ = writeln!(w, ")");
                }
            } else {
                // For functions with explicit table entries, we'd need to construct
                // an ITE chain or similar. This is a placeholder.
                let _ = writeln!(w, "...)");
            }
        }

        let _ = writeln!(w, ")");
    }

    /// Write a model value
    fn write_model_value(&self, w: &mut impl Write, value: &ModelValue) {
        match value {
            ModelValue::Bool(b) => {
                let _ = write!(w, "{}", b);
            }
            ModelValue::Int(n) => {
                let _ = write!(w, "{}", n);
            }
            ModelValue::Real(r) => {
                let _ = write!(w, "{}", r);
            }
            ModelValue::BitVec { value, width } => {
                let _ = write!(
                    w,
                    "#x{:0>width$x}",
                    value,
                    width = (*width as usize).div_ceil(4)
                );
            }
            ModelValue::Uninterpreted { sort, id } => {
                let _ = write!(w, "uninterp_{}_{}", sort.0, id);
            }
        }
    }

    /// Print a proof in SMT-LIB2 format
    #[must_use]
    pub fn print_proof(&self, proof: &Proof) -> String {
        let mut buf = String::new();
        self.write_proof(&mut buf, proof);
        buf
    }

    /// Write a proof in SMT-LIB2 format
    ///
    /// This outputs the proof as a tree of inference steps in a readable format.
    pub fn write_proof(&self, w: &mut impl Write, proof: &Proof) {
        let _ = writeln!(w, "(proof");

        // Get the root node and recursively write the proof tree
        let root_id = proof.root();
        self.write_proof_node(w, proof, root_id, 1);

        let _ = writeln!(w, ")");
    }

    /// Write a single proof node recursively
    fn write_proof_node(
        &self,
        w: &mut impl Write,
        proof: &Proof,
        node_id: crate::ast::proof::ProofId,
        indent: usize,
    ) {
        let Some(node) = proof.get_node(node_id) else {
            return;
        };

        let indent_str = "  ".repeat(indent);

        // Write the proof step
        let _ = write!(w, "{}", indent_str);
        let _ = write!(w, "(step @p{} ", node_id.0);

        // Write the rule
        self.write_proof_rule(w, &node.rule);

        // Write the conclusion
        let _ = write!(w, "\n{}  :conclusion ", indent_str);
        self.write_term(w, node.conclusion);

        // Write premises if any
        if !node.premises.is_empty() {
            let _ = write!(w, "\n{}  :premises (", indent_str);
            for (i, premise_id) in node.premises.iter().enumerate() {
                if i > 0 {
                    let _ = write!(w, " ");
                }
                let _ = write!(w, "@p{}", premise_id.0);
            }
            let _ = write!(w, ")");
        }

        // Write metadata if any
        if !node.metadata.is_empty() {
            let _ = write!(w, "\n{}  :metadata (", indent_str);
            for (i, (key, value)) in node.metadata.iter().enumerate() {
                if i > 0 {
                    let _ = write!(w, " ");
                }
                let _ = write!(w, ":{} \"{}\"", key, value);
            }
            let _ = write!(w, ")");
        }

        let _ = writeln!(w, ")");

        // Recursively write premises
        for premise_id in &node.premises {
            self.write_proof_node(w, proof, *premise_id, indent + 1);
        }
    }

    /// Write a proof rule
    fn write_proof_rule(&self, w: &mut impl Write, rule: &ProofRule) {
        match rule {
            ProofRule::Assume { name } => {
                if let Some(n) = name {
                    let _ = write!(w, ":rule assume :name \"{}\"", n);
                } else {
                    let _ = write!(w, ":rule assume");
                }
            }
            ProofRule::Resolution { pivot } => {
                let _ = write!(w, ":rule resolution :pivot ");
                self.write_term(w, *pivot);
            }
            ProofRule::ModusPonens => {
                let _ = write!(w, ":rule modus-ponens");
            }
            ProofRule::Tautology => {
                let _ = write!(w, ":rule tautology");
            }
            ProofRule::ArithInequality => {
                let _ = write!(w, ":rule arith-inequality");
            }
            ProofRule::TheoryLemma { theory } => {
                let _ = write!(w, ":rule theory-lemma :theory \"{}\"", theory);
            }
            ProofRule::Contradiction => {
                let _ = write!(w, ":rule contradiction");
            }
            ProofRule::Rewrite => {
                let _ = write!(w, ":rule rewrite");
            }
            ProofRule::Substitution => {
                let _ = write!(w, ":rule substitution");
            }
            ProofRule::Symmetry => {
                let _ = write!(w, ":rule symmetry");
            }
            ProofRule::Transitivity => {
                let _ = write!(w, ":rule transitivity");
            }
            ProofRule::Congruence => {
                let _ = write!(w, ":rule congruence");
            }
            ProofRule::Reflexivity => {
                let _ = write!(w, ":rule reflexivity");
            }
            ProofRule::Instantiation { terms } => {
                let _ = write!(w, ":rule instantiation :terms (");
                for (i, term_id) in terms.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    self.write_term(w, *term_id);
                }
                let _ = write!(w, ")");
            }
            ProofRule::Custom { name } => {
                let _ = write!(w, ":rule custom :name \"{}\"", name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_constants() {
        let manager = TermManager::new();
        let printer = Printer::new(&manager);

        assert_eq!(printer.print_term(manager.mk_true()), "true");
        assert_eq!(printer.print_term(manager.mk_false()), "false");
    }

    #[test]
    fn test_print_compound() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let printer = Printer::new(&manager);
        assert_eq!(printer.print_term(and), "(and x y)");
    }

    #[test]
    fn test_roundtrip() {
        let mut manager = TermManager::new();
        let input = "(and (or x y) (not z))";
        let term = crate::smtlib::parse_term(input, &mut manager).unwrap();

        let printer = Printer::new(&manager);
        let output = printer.print_term(term);

        // Note: Output might differ slightly due to canonicalization
        assert!(output.contains("and"));
        assert!(output.contains("or"));
        assert!(output.contains("not"));
    }

    // ==================== PrettyPrinter Tests ====================

    #[test]
    fn test_pretty_config_default() {
        let config = PrettyConfig::default();
        assert_eq!(config.indent_width, 2);
        assert_eq!(config.max_width, 80);
        assert!(!config.use_tabs);
        assert!(!config.print_sorts);
        assert_eq!(config.break_depth, 2);
    }

    #[test]
    fn test_pretty_config_compact() {
        let config = PrettyConfig::compact();
        assert_eq!(config.indent_width, 0);
        assert_eq!(config.max_width, usize::MAX);
        assert_eq!(config.break_depth, usize::MAX);
    }

    #[test]
    fn test_pretty_config_expanded() {
        let config = PrettyConfig::expanded();
        assert_eq!(config.max_width, 40);
        assert_eq!(config.break_depth, 1);
    }

    #[test]
    fn test_pretty_config_builder() {
        let config = PrettyConfig::default()
            .with_indent_width(4)
            .with_max_width(100)
            .with_tabs(true)
            .with_print_sorts(true)
            .with_break_depth(3);

        assert_eq!(config.indent_width, 4);
        assert_eq!(config.max_width, 100);
        assert!(config.use_tabs);
        assert!(config.print_sorts);
        assert_eq!(config.break_depth, 3);
    }

    #[test]
    fn test_pretty_printer_simple_term() {
        let manager = TermManager::new();
        let pretty = PrettyPrinter::new(&manager);

        let output = pretty.print_term(manager.mk_true());
        assert_eq!(output, "true");
    }

    #[test]
    fn test_pretty_printer_compound_term() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let pretty = PrettyPrinter::new(&manager);
        let output = pretty.print_term(and);
        assert_eq!(output, "(and x y)");
    }

    #[test]
    fn test_pretty_printer_compact() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y, z]);
        let prod = manager.mk_mul([sum, x]);

        let config = PrettyConfig::compact();
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(prod);

        // Compact mode should not break lines
        assert!(!output.contains('\n'));
        assert!(output.contains("(* (+ x y z) x)"));
    }

    #[test]
    fn test_pretty_printer_expanded() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);
        let w = manager.mk_var("w", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y, z, w]);

        let config = PrettyConfig::expanded();
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(sum);

        // Expanded mode with many terms should break lines
        // The exact format depends on the width calculation
        assert!(output.contains("+"));
        assert!(output.contains("x"));
        assert!(output.contains("y"));
        assert!(output.contains("z"));
        assert!(output.contains("w"));
    }

    #[test]
    fn test_pretty_printer_nested_ite() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let a = manager.mk_int(1);
        let b = manager.mk_int(2);
        let ite = manager.mk_ite(x, a, b);

        let config = PrettyConfig::default()
            .with_max_width(10)
            .with_break_depth(0);
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(ite);

        // Should break due to small max_width
        assert!(output.contains("ite"));
        assert!(output.contains("x"));
    }

    // ==================== Model Printing Tests ====================

    #[test]
    fn test_print_empty_model() {
        let manager = TermManager::new();
        let model = Model::new();
        let printer = Printer::new(&manager);

        let output = printer.print_model(&model);
        assert!(output.contains("(model"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_model_with_bool_assignment() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut model = Model::new();
        model.assign_bool(x, true);

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("define-fun x () Bool true"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_model_with_int_assignment() {
        let mut manager = TermManager::new();
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let mut model = Model::new();
        model.assign_int(y, num_bigint::BigInt::from(42));

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("define-fun y () Int 42"));
    }

    #[test]
    fn test_print_model_with_multiple_assignments() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let mut model = Model::new();
        model.assign_bool(x, false);
        model.assign_int(y, num_bigint::BigInt::from(10));

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("x"));
        assert!(output.contains("Bool"));
        assert!(output.contains("false"));
        assert!(output.contains("y"));
        assert!(output.contains("Int"));
        assert!(output.contains("10"));
    }

    #[test]
    fn test_print_model_with_bitvec_assignment() {
        let mut manager = TermManager::new();
        let bv_sort = manager.sorts.bitvec(8);
        let z = manager.mk_var("z", bv_sort);

        let mut model = Model::new();
        model.assign_bitvec(z, 0xFF, 8);

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("z"));
        assert!(output.contains("#xff"));
    }

    // ==================== Proof Printing Tests ====================

    #[test]
    fn test_print_empty_proof() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let false_term = manager.mk_false();

        let root = ProofNode::new(ProofId(0), ProofRule::Contradiction, false_term);
        proof.add_node(root);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("contradiction"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_proof_with_assumption() {
        use crate::ast::proof::*;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut proof = Proof::new();
        let assume_node = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            x,
        );
        proof.add_node(assume_node);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("assume"));
        assert!(output.contains("H1"));
        assert!(output.contains("conclusion"));
    }

    #[test]
    fn test_print_proof_with_resolution() {
        use crate::ast::proof::*;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let p1 = manager.mk_var("p", manager.sorts.bool_sort);
        let p2 = manager.mk_var("q", manager.sorts.bool_sort);

        let mut proof = Proof::new();

        // Add premise nodes
        let node1 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("A1".to_string()),
            },
            p1,
        );
        let node2 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("A2".to_string()),
            },
            p2,
        );

        // Add resolution node
        let resolution_node = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: x },
            manager.mk_true(),
            vec![ProofId(0), ProofId(1)],
        );

        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(resolution_node);
        proof.set_root(ProofId(2));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("resolution"));
        assert!(output.contains("premises"));
        assert!(output.contains("@p0"));
        assert!(output.contains("@p1"));
    }

    #[test]
    fn test_print_proof_with_metadata() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();

        let mut node = ProofNode::new(
            ProofId(0),
            ProofRule::TheoryLemma {
                theory: "LIA".to_string(),
            },
            manager.mk_false(),
        );
        node.add_metadata("source".to_string(), "farkas".to_string());

        proof.add_node(node);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("theory-lemma"));
        assert!(output.contains("LIA"));
        assert!(output.contains("metadata"));
        assert!(output.contains("source"));
        assert!(output.contains("farkas"));
    }
}
