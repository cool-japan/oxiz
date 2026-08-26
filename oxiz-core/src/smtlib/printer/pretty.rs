//\! Pretty printer with indentation and formatting

use super::basic::Printer;
use super::config::PrettyConfig;
use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use core::fmt::Write;

/// Maximum [`PrettyPrinter::write_term`] recursion depth before printing
/// degrades to a truncation marker (`...`) instead of descending further.
///
/// Same mechanism as `super::basic::MAX_PRINT_DEPTH`, but deliberately a
/// quarter of its value, because a pretty-print level is *far* more expensive
/// in native stack than a basic-print level: at every level past
/// `PrettyConfig::break_depth`, [`PrettyPrinter::should_break`] calls
/// [`PrettyPrinter::term_width`], which runs a whole nested
/// `Printer::print_term` walk over that subterm — up to `MAX_PRINT_DEPTH`
/// (2000) further frames of ~250 bytes each, i.e. about half of a 1 MiB
/// thread stack, *inside* the deepest pretty frame. Measured directly: with
/// this constant at 2000, `deeply_nested_term_truncates_instead_of_overflowing`
/// aborted the test process with `fatal runtime error: stack overflow` on the
/// 1 MiB stack this workspace's deep-term regressions use; at 512 it passes in
/// both the `dev` and `release` profiles, leaving the nested basic walk its
/// full budget with room to spare.
///
/// Before this existed, `PrettyPrinter::write_term`'s `depth` parameter was a
/// *formatting* input only — compared against `config.break_depth` to decide
/// line breaking — and nothing terminated the recursion at all, so printing a
/// deep term aborted the process outright.
const MAX_PRETTY_PRINT_DEPTH: usize = 512;

pub struct PrettyPrinter<'a> {
    manager: &'a TermManager,
    config: PrettyConfig,
    /// Memo for [`PrettyPrinter::term_width`].
    ///
    /// `term_width` re-prints a whole subterm with the basic printer just to
    /// measure it, and [`PrettyPrinter::should_break`] asks for that
    /// measurement at every level it descends through — so an unmemoised
    /// pretty-print costs O(size x depth) basic-print work, and a shared DAG
    /// node reachable by many paths is re-printed once per path. The measured
    /// width is a pure function of the `TermId` (the nested basic walk always
    /// restarts at depth 0), so caching it changes nothing but the cost.
    width_memo: core::cell::RefCell<FxHashMap<TermId, usize>>,
}

#[allow(dead_code)]
impl<'a> PrettyPrinter<'a> {
    /// Create a new pretty printer with default configuration
    #[must_use]
    pub fn new(manager: &'a TermManager) -> Self {
        Self {
            manager,
            config: PrettyConfig::default(),
            width_memo: core::cell::RefCell::default(),
        }
    }

    /// Create a pretty printer with custom configuration
    #[must_use]
    pub fn with_config(manager: &'a TermManager, config: PrettyConfig) -> Self {
        Self {
            manager,
            config,
            width_memo: core::cell::RefCell::default(),
        }
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
        if let Some(&width) = self.width_memo.borrow().get(&term_id) {
            return width;
        }
        // The borrow above is released before the nested print, which never
        // re-enters `term_width`, so no borrow can overlap.
        let basic = Printer::new(self.manager);
        let width = basic.print_term(term_id).len();
        self.width_memo.borrow_mut().insert(term_id, width);
        width
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
        // Every recursive descent below (directly, or through
        // `write_binary_term` / `write_nary_term` / `write_ternary_term` /
        // `write_quantifier`, all of which pass `depth + 1` or `depth + 2`)
        // re-enters here, so this single check bounds the whole mutually
        // recursive group. See [`MAX_PRETTY_PRINT_DEPTH`].
        if depth >= MAX_PRETTY_PRINT_DEPTH {
            let _ = write!(w, "...");
            return;
        }

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
                let _ = write!(w, "{}", super::format_bitvec_literal(value, *width));
            }
            TermKind::StringLit(s) => {
                let _ = write!(w, "{}", super::format_string_literal(s));
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
            TermKind::StrLt(lhs, rhs) => {
                self.write_binary_term(w, "str.<", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::StrLe(lhs, rhs) => {
                self.write_binary_term(w, "str.<=", *lhs, *rhs, indent, depth, break_here);
            }
            TermKind::StrToCode(s) => {
                let _ = write!(w, "(str.to_code ");
                self.write_term(w, *s, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::StrFromCode(n) => {
                let _ = write!(w, "(str.from_code ");
                self.write_term(w, *n, indent, depth + 1);
                let _ = write!(w, ")");
            }
            TermKind::StrReplaceRe(s, re, to) => {
                self.write_ternary_term(
                    w,
                    "str.replace_re",
                    *s,
                    *re,
                    *to,
                    indent,
                    depth,
                    break_here,
                );
            }
            TermKind::StrReplaceReAll(s, re, to) => {
                self.write_ternary_term(
                    w,
                    "str.replace_re_all",
                    *s,
                    *re,
                    *to,
                    indent,
                    depth,
                    break_here,
                );
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

            // Algebraic datatypes.  A *nullary* constructor is a plain symbol in
            // SMT-LIB (`nil`, `red`), not a one-element application: `(nil)` is
            // not a term the grammar accepts, so a model printed with it could
            // not be fed back to a solver.
            TermKind::DtConstructor { constructor, args } => {
                let name = self.manager.resolve_str(*constructor);
                if args.is_empty() {
                    let _ = write!(w, "{name}");
                } else {
                    let _ = write!(w, "({name}");
                    for arg in args {
                        let _ = write!(w, " ");
                        self.write_term(w, *arg, indent, depth + 1);
                    }
                    let _ = write!(w, ")");
                }
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

    /// Write `(op a b c)`, breaking one operand per line when `break_here`.
    #[allow(clippy::too_many_arguments)]
    fn write_ternary_term(
        &self,
        w: &mut impl Write,
        op: &str,
        first: TermId,
        second: TermId,
        third: TermId,
        indent: usize,
        depth: usize,
        break_here: bool,
    ) {
        if break_here {
            let inner_indent = indent + self.config.indent_width;
            let separator = self.indent_string(inner_indent / self.config.indent_width.max(1));
            let _ = write!(w, "({op}\n{separator}");
            self.write_term(w, first, inner_indent, depth + 1);
            let _ = write!(w, "\n{separator}");
            self.write_term(w, second, inner_indent, depth + 1);
            let _ = write!(w, "\n{separator}");
            self.write_term(w, third, inner_indent, depth + 1);
            let _ = write!(w, ")");
        } else {
            let _ = write!(w, "({op} ");
            self.write_term(w, first, indent, depth + 1);
            let _ = write!(w, " ");
            self.write_term(w, second, indent, depth + 1);
            let _ = write!(w, " ");
            self.write_term(w, third, indent, depth + 1);
            let _ = write!(w, ")");
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
        vars: &smallvec::SmallVec<[(crate::interner::Spur, crate::sort::SortId); 2]>,
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

#[cfg(test)]
mod tests {
    use super::{MAX_PRETTY_PRINT_DEPTH, PrettyPrinter};
    use crate::ast::{TermId, TermKind, TermManager};

    /// Build a chain of `levels` nested `not`s.
    ///
    /// Interned directly rather than through `mk_not`, whose double-negation
    /// simplification would collapse the chain to depth 1.
    fn nested_nots(manager: &mut TermManager, levels: usize) -> TermId {
        let mut term = manager.mk_true();
        let bool_sort = manager.sorts.bool_sort;
        for _ in 0..levels {
            term = manager.intern(TermKind::Not(term), bool_sort);
        }
        term
    }

    /// `PrettyPrinter::write_term`'s `depth` parameter used to be a
    /// *formatting* input only — compared against `config.break_depth` to
    /// decide line breaking — and nothing terminated the recursion, so
    /// pretty-printing a deep term aborted the process. It now degrades to
    /// the same `...` truncation marker the basic printer uses.
    ///
    /// The assertion is that the call returns at all on a 1 MiB stack.
    // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — the printer
    // truncates at `MAX_PRETTY_PRINT_DEPTH` (512) via bounded native
    // recursion, so stack usage and term depth are decoupled; 128 KiB would
    // abort inside the cap instead of exercising the truncation path.
    #[test]
    fn deeply_nested_term_truncates_instead_of_overflowing() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let mut manager = TermManager::new();
                let term = nested_nots(&mut manager, 12_500);
                let printer = PrettyPrinter::new(&manager);
                printer.print_term(term)
            })
            .expect("spawn");
        let printed = handle.join().expect("worker thread must not overflow");
        assert!(
            printed.contains("..."),
            "a term deeper than the print cap must end in the truncation marker"
        );
        // One `(not ` per level printed, up to the cap.
        assert_eq!(printed.matches("(not").count(), MAX_PRETTY_PRINT_DEPTH);
    }

    /// Semantic pin: a term shallower than the cap prints in full, exactly as
    /// before the guard was added.
    #[test]
    fn shallow_terms_print_in_full() {
        let mut manager = TermManager::new();
        let term = nested_nots(&mut manager, 3);
        let printer = PrettyPrinter::new(&manager);
        let printed = printer.print_term(term);
        assert_eq!(printed, "(not (not (not true)))");
    }

    /// The `term_width` memo must not change what is printed: the same
    /// printer reused across calls, and a fresh printer per call, agree — on
    /// a term whose subterms are shared (the same `TermId` reachable from
    /// several parents), which is exactly the case the memo collapses.
    #[test]
    fn width_memo_does_not_change_printed_output_for_shared_subterms() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", bool_sort);
        let y = manager.mk_var("y", bool_sort);
        // `shared` is reached from both arms of every level below, so an
        // unmemoised width walk re-prints it once per path.
        let shared = manager.mk_and([x, y]);
        let mut term = shared;
        for _ in 0..4 {
            term = manager.mk_or([term, shared]);
        }

        let reused = PrettyPrinter::new(&manager);
        let first = reused.print_term(term);
        let second = reused.print_term(term);
        let fresh = PrettyPrinter::new(&manager).print_term(term);

        assert_eq!(first, second, "the memo must not perturb a repeated print");
        assert_eq!(first, fresh, "the memo must not perturb a first print");
        assert!(first.contains("(and x y)"));
    }

    /// A deep chain makes `should_break` measure a subterm at every level:
    /// unmemoised that is O(size x depth) basic-print work, memoised it is
    /// linear. The assertion is only that the print is correct and returns —
    /// the memo is a cost fix, not a semantic one.
    #[test]
    fn deep_chain_prints_with_one_width_measurement_per_node() {
        let mut manager = TermManager::new();
        const LEVELS: usize = 400;
        let term = nested_nots(&mut manager, LEVELS);
        let printer = PrettyPrinter::new(&manager);
        let printed = printer.print_term(term);
        assert_eq!(printed.matches("(not").count(), LEVELS);
        assert!(!printed.contains("..."));
    }
}
