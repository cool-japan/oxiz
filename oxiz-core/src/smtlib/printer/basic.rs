//\! Basic SMT-LIB2 printer

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortKind};
use core::cell::Cell;
use core::fmt::Write;

/// Maximum `write_term` recursion depth before printing degrades to a
/// truncation marker instead of continuing to descend, mirroring the depth
/// cap `simplification`'s `SIMPLIFICATION_MAX_DEPTH` applies elsewhere in
/// the crate. (An earlier version of this comment also pointed to
/// `ast::manager::query::MAX_QUERY_RECURSION_DEPTH`; that cap was removed
/// when `TermManager::substitute`/`simplify`/`free_vars` were converted to
/// use an explicit heap stack instead, which needs no depth cap at all --
/// it no longer exists.) Terms built directly via the `mk_*` builder API
/// bypass parser-/rewriter-side caps like `SIMPLIFICATION_MAX_DEPTH`, so
/// without a bound here, printing a pathologically deep (but validly
/// constructed) term could overflow the native call stack. `write_term`
/// itself is *not* converted to an explicit-stack walk the way the
/// `TermManager` methods above were -- unlike those, a decision was made
/// to keep this guard rather than rewrite the printer, based on a direct
/// measurement (see below) rather than assumption.
///
/// # Measured stack cost (why 2000 is not just an assumed-safe round number)
///
/// A prior audit suspected 2000 frames of `write_term`/`write_term_at_depth`
/// might not fit in the 1 MiB thread stacks this workspace's own deep-term
/// regression tests deliberately constrain themselves to (see
/// `run_on_1mib_stack` in `ast/manager/query/tests.rs`), which is exactly
/// the failure mode that forced the SMT-LIB parser's own recursion-to-
/// iteration rewrite (there, roughly 2944 bytes/frame x 1024 levels needed
/// ~2.9 MiB, well past a 1-2 MiB budget). That suspicion does not hold here:
/// measured by binary-searching (via a chain of nested `not`s, built with
/// `intern` to bypass `mk_not`'s double-negation simplification, printed on
/// a thread with an explicitly bounded stack, with `MAX_PRINT_DEPTH`
/// temporarily raised far out of the way to observe genuine unguarded
/// native recursion) for the exact depth at which a real stack overflow
/// occurs:
///
/// | Profile (this workspace's `Cargo.toml`) | Stack size | Last safe depth | First crashing depth |
/// |---|---|---|---|
/// | `release` (`opt-level = "z"`, the profile `--release` actually uses here) | 1 MiB | 4,414 | 4,415 |
/// | `release` | 2 MiB | 8,783 | 8,784 |
/// | `dev` (`opt-level = 1`, what plain `cargo test` uses here) | 1 MiB | 3,999 | 4,000 |
///
/// The release-mode data points solve to **~240 bytes/frame** (per
/// print-recursion level, i.e. one `write_term` + `write_term_at_depth`
/// pair combined: `(2 MiB - 1 MiB) / (8784 - 4415) ~ 240`), an order of
/// magnitude leaner than the parser's ~2944 bytes/frame (printing a term
/// only ever holds a handful of small locals per level -- an `&str`
/// keyword and the args being written -- versus a parser frame's richer
/// per-level state). `MAX_PRINT_DEPTH = 2000` therefore leaves roughly a
/// 2x safety margin below the measured crash point in *both* profiles this
/// workspace's own commands exercise (`cargo test` and `cargo test
/// --release`), not merely in whichever one happens to be faster to
/// build -- so, unlike the stale suspicion above, this guard is confirmed
/// to actually fire before a real overflow rather than merely being
/// assumed to. See `write_term_truncates_on_a_constrained_stack_without_overflowing`
/// in this file's tests for the pinned regression coverage.
const MAX_PRINT_DEPTH: u32 = 2000;

/// Printer for SMT-LIB2 format
pub struct Printer<'a> {
    pub(super) manager: &'a TermManager,
    /// Current `write_term` recursion depth on this printer. `Cell`
    /// because `write_term` takes `&self` (it is called recursively and
    /// re-entrantly from many call sites, all of which stay unchanged —
    /// see `write_term`'s depth-guard wrapper below).
    depth: Cell<u32>,
}

/// RAII guard that decrements a [`Printer`]'s `depth` counter on drop
/// (including on early return), keeping the depth bookkeeping accurate
/// regardless of which path through `write_term_at_depth` returns.
struct PrintDepthGuard<'p, 'a>(&'p Printer<'a>);

impl Drop for PrintDepthGuard<'_, '_> {
    fn drop(&mut self) {
        self.0.depth.set(self.0.depth.get().saturating_sub(1));
    }
}

impl<'a> Printer<'a> {
    /// Create a new printer
    #[must_use]
    pub fn new(manager: &'a TermManager) -> Self {
        Self {
            manager,
            depth: Cell::new(0),
        }
    }

    /// Print a term to a string
    #[must_use]
    pub fn print_term(&self, term_id: TermId) -> String {
        let mut buf = String::new();
        self.write_term(&mut buf, term_id);
        buf
    }

    /// Write a term to a writer.
    ///
    /// Bounds recursion at `MAX_PRINT_DEPTH`: past that depth, printing
    /// degrades gracefully to a truncation marker (`...`) instead of
    /// recursing further and risking a native stack overflow. See
    /// `write_term_at_depth` for the actual
    /// per-`TermKind` printing logic — every recursive call within it
    /// still goes back through this wrapper, so the depth check applies
    /// at every level without needing to touch each call site.
    pub fn write_term(&self, w: &mut impl Write, term_id: TermId) {
        let depth = self.depth.get();
        if depth >= MAX_PRINT_DEPTH {
            let _ = write!(w, "...");
            return;
        }
        self.depth.set(depth + 1);
        let _guard = PrintDepthGuard(self);
        self.write_term_at_depth(w, term_id);
    }

    /// Per-`TermKind` printing logic for [`write_term`](Self::write_term);
    /// see that method's doc comment for the depth-guarding this relies
    /// on.
    fn write_term_at_depth(&self, w: &mut impl Write, term_id: TermId) {
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
            TermKind::StrReplaceRe(s, re, to) => {
                let _ = write!(w, "(str.replace_re ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *re);
                let _ = write!(w, " ");
                self.write_term(w, *to);
                let _ = write!(w, ")");
            }
            TermKind::StrReplaceReAll(s, re, to) => {
                let _ = write!(w, "(str.replace_re_all ");
                self.write_term(w, *s);
                let _ = write!(w, " ");
                self.write_term(w, *re);
                let _ = write!(w, " ");
                self.write_term(w, *to);
                let _ = write!(w, ")");
            }
            TermKind::StrLt(lhs, rhs) => {
                let _ = write!(w, "(str.< ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::StrLe(lhs, rhs) => {
                let _ = write!(w, "(str.<= ");
                self.write_term(w, *lhs);
                let _ = write!(w, " ");
                self.write_term(w, *rhs);
                let _ = write!(w, ")");
            }
            TermKind::StrToCode(s) => {
                let _ = write!(w, "(str.to_code ");
                self.write_term(w, *s);
                let _ = write!(w, ")");
            }
            TermKind::StrFromCode(n) => {
                let _ = write!(w, "(str.from_code ");
                self.write_term(w, *n);
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
                        self.write_term(w, *arg);
                    }
                    let _ = write!(w, ")");
                }
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

    /// Write a sort to a writer.
    ///
    /// Driven by an explicit worklist rather than by recursion. Unlike
    /// [`Printer::write_term`], this walk has no depth cap at all and needs
    /// none: it returns `()`, so a cap could only ever truncate a sort name
    /// into a *different, silently wrong* sort name. `SortManager::array` is
    /// `pub` and interns a nested sort in constant stack, so an embedder can
    /// build a million-deep `(Array (Array ...))` and print it; recursing here
    /// aborted the process (`write_term_at_depth` also calls this directly for
    /// quantifier variable sorts, bypassing `write_term`'s depth guard
    /// entirely).
    pub fn write_sort(&self, w: &mut impl Write, sort_id: SortId) {
        /// One pending step: a sort still to render, or literal punctuation
        /// already scheduled to follow one.
        enum Step {
            /// Render this sort.
            Sort(SortId),
            /// Emit this literal.
            Text(&'static str),
        }

        let mut stack = vec![Step::Sort(sort_id)];
        while let Some(step) = stack.pop() {
            let id = match step {
                Step::Text(text) => {
                    let _ = w.write_str(text);
                    continue;
                }
                Step::Sort(id) => id,
            };
            let Some(sort) = self.manager.sorts.get(id) else {
                let _ = write!(w, "?Sort{}", id.0);
                continue;
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
                SortKind::RoundingMode => {
                    let _ = write!(w, "RoundingMode");
                }
                SortKind::Array { domain, range } => {
                    let _ = write!(w, "(Array ");
                    // Pushed in reverse of emission order.
                    stack.push(Step::Text(")"));
                    stack.push(Step::Sort(*range));
                    stack.push(Step::Text(" "));
                    stack.push(Step::Sort(*domain));
                }
                // An uninterpreted sort's name is interned by the *term*
                // manager: both producers (`Parser::parse_sort` for
                // `declare-sort` names and `TermManager::reglan_sort`) call
                // `TermManager::intern_str`.
                SortKind::Uninterpreted(spur) => {
                    let name = self.manager.resolve_str(*spur);
                    let _ = write!(w, "{name}");
                }
                // A sort *parameter*'s name, by contrast, is interned by the
                // sort manager: `mk_sort_parameter` and
                // `define_parametric_sort` (whose `params` feed
                // `instantiate_parametric_sort`'s `SortKind::Parameter`) both
                // go through `SortManager::interner`.
                SortKind::Parameter(spur) => {
                    let name = self.manager.sorts.resolve_spur(*spur);
                    let _ = write!(w, "{name}");
                }
                // A parametric sort's head name is interned by `SortManager`
                // itself (`declare_parametric_sort` /
                // `instantiate_parametric_sort` / `mk_sort_parameter` all go
                // through `self.interner`), not by the term manager. The two
                // are separate `Rodeo`s, so resolving one's key through the
                // other yields an unrelated string or indexes out of range and
                // panics. Mirrors `Parser::sort_id_to_string`.
                SortKind::Parametric { name, args } => {
                    let name_str = self.manager.sorts.resolve_spur(*name);
                    let _ = write!(w, "({name_str}");
                    stack.push(Step::Text(")"));
                    for arg in args.iter().rev() {
                        stack.push(Step::Sort(*arg));
                        stack.push(Step::Text(" "));
                    }
                }
                // Likewise a datatype sort's name: `mk_datatype_sort` and
                // `declare_datatype` intern it in the sort manager's interner.
                SortKind::Datatype(spur) => {
                    let name = self.manager.sorts.resolve_spur(*spur);
                    let _ = write!(w, "{name}");
                }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression tests for: "Recursive term printers lack an explicit
    // depth cap" — `write_term` must degrade gracefully (truncate) past
    // `MAX_PRINT_DEPTH` instead of recursing without bound. Superseded the
    // original (default-thread-only) version of this test with one that
    // additionally constrains the stack, per this session's general
    // regression-test requirement (see `run_on_1mib_stack` below).

    /// Run `f` to completion on a dedicated thread with a 1 MiB stack --
    /// deliberately far smaller than the default (several-MiB) main-thread
    /// stack -- and return whatever it returns. Mirrors
    /// `ast/manager/query/tests.rs`'s `run_on_1mib_stack`: a stack overflow
    /// aborts the whole process rather than failing a single test
    /// gracefully, so for the deep-nesting test below, the call *returning
    /// at all* is itself part of what is being asserted.
    fn run_on_1mib_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(f)
            .expect("spawning the constrained-stack test thread should succeed")
            .join()
            .expect("the constrained-stack thread must not panic")
    }

    #[test]
    fn write_term_truncates_on_a_constrained_stack_without_overflowing() {
        // Regression pin for the measurement in `MAX_PRINT_DEPTH`'s doc
        // comment: on a 1 MiB stack, unguarded `write_term`-shaped native
        // recursion crashes somewhere around depth 4,000 (measured: 3,999
        // survives / 4,000 crashes in this workspace's `dev` profile;
        // 4,414 / 4,415 in `release`). This builds a chain twice that deep
        // and confirms `MAX_PRINT_DEPTH = 2000` still truncates cleanly,
        // without overflowing, even on a stack far smaller than a default
        // thread's -- not just on whatever generous stack the default test
        // thread happens to have (see the removed, now-superseded
        // `write_term_truncates_past_max_print_depth_instead_of_overflowing_stack`
        // above, which only ever ran on the default thread).
        const DEPTH: usize = 8_000;

        let printed = run_on_1mib_stack(|| {
            let mut manager = TermManager::new();
            let bool_sort = manager.sorts.bool_sort;
            let mut term = manager.mk_true();
            for _ in 0..DEPTH {
                term = manager.intern(TermKind::Not(term), bool_sort);
            }
            let printer = Printer::new(&manager);
            printer.print_term(term)
        });

        assert!(
            printed.contains("..."),
            "expected a truncation marker once MAX_PRINT_DEPTH is exceeded, got a string of len {}",
            printed.len()
        );
    }

    #[test]
    fn write_term_prints_normal_depth_terms_without_truncation() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let printer = Printer::new(&manager);
        let printed = printer.print_term(and);
        assert_eq!(printed, "(and x y)");
        assert!(!printed.contains("..."));
    }

    #[test]
    fn write_term_depth_counter_resets_between_independent_calls() {
        // The depth counter must not leak across independent top-level
        // `write_term` calls on the same `Printer` — verified by printing
        // a normal term successfully more than once in a row.
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let not_x = manager.mk_not(x);

        let printer = Printer::new(&manager);
        assert_eq!(printer.print_term(not_x), "(not x)");
        assert_eq!(printer.print_term(not_x), "(not x)");
    }

    /// The SMT-LIB Unicode Strings operators that gained a term kind print
    /// back under their standard names, so a printed term re-parses.
    #[test]
    fn string_theory_operators_print_under_their_smtlib_names() {
        let mut manager = TermManager::new();
        let string_sort = manager.sorts.string_sort();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", string_sort);
        let y = manager.mk_var("y", string_sort);
        let n = manager.mk_var("n", int_sort);
        let re = manager.mk_str_to_re(x);

        let cases = [
            (manager.mk_str_lt(x, y), "(str.< x y)"),
            (manager.mk_str_le(x, y), "(str.<= x y)"),
            (manager.mk_str_to_code(x), "(str.to_code x)"),
            (manager.mk_str_from_code(n), "(str.from_code n)"),
            (
                manager.mk_str_replace_re(x, re, y),
                "(str.replace_re x (str.to_re x) y)",
            ),
            (
                manager.mk_str_replace_re_all(x, re, y),
                "(str.replace_re_all x (str.to_re x) y)",
            ),
        ];

        let printer = Printer::new(&manager);
        for (term, expected) in cases {
            assert_eq!(printer.print_term(term), expected);
        }
    }

    /// `TermManager` and `SortManager` own *separate* string interners, so a
    /// `Spur` minted by one is meaningless to the other: resolving a datatype
    /// or parametric sort name (both interned by `SortManager`) through the
    /// term manager's interner yields whatever unrelated string happens to sit
    /// at that index — or panics with an out-of-range index when the term
    /// interner is shorter. This pins the sort names to the sort manager's
    /// interner, with the two interners deliberately driven out of sync first
    /// so a crossed resolution cannot accidentally agree.
    #[test]
    fn datatype_and_parametric_sort_names_resolve_through_the_sort_interner() {
        let mut manager = TermManager::new();

        // Drive the *term* interner's keys out of alignment with the sort
        // interner's: whatever key the sort names get, the term interner holds
        // a different string there.
        for decoy in ["term_side_zero", "term_side_one", "term_side_two"] {
            let _ = manager.intern_str(decoy);
        }

        let colour = manager.sorts.mk_datatype_sort("Colour");
        manager.sorts.declare_parametric_sort("List", 1);
        let int_sort = manager.sorts.int_sort;
        let list_int = manager
            .sorts
            .instantiate_parametric_sort("List", &[int_sort])
            .expect("List/1 was just declared, so instantiating it must succeed");
        let elem = manager.sorts.mk_sort_parameter("Elem");

        let printer = Printer::new(&manager);
        assert_eq!(printer.print_sort(colour), "Colour");
        assert_eq!(printer.print_sort(list_int), "(List Int)");
        assert_eq!(printer.print_sort(elem), "Elem");
    }

    /// The narrower half of the same bug: with *nothing* interned term-side,
    /// crossing the interners indexed past the end of the term manager's
    /// string table and panicked rather than merely printing the wrong name.
    #[test]
    fn sort_names_print_when_the_term_interner_is_empty() {
        let mut manager = TermManager::new();
        let tree = manager.sorts.mk_datatype_sort("Tree");
        let printer = Printer::new(&manager);
        assert_eq!(printer.print_sort(tree), "Tree");
    }

    /// An *uninterpreted* sort's name, by contrast, really is term-interned
    /// (`Parser::parse_sort` and `TermManager::reglan_sort` both mint it with
    /// `TermManager::intern_str`), so it must keep resolving there. Pinned so
    /// the fix above is not over-applied to this arm.
    #[test]
    fn uninterpreted_sort_names_resolve_through_the_term_interner() {
        let mut manager = TermManager::new();
        // Decoys on the sort side this time, to unalign the interners the
        // other way round.
        for decoy in ["sort_side_zero", "sort_side_one"] {
            let _ = manager.sorts.intern_str(decoy);
        }
        let spur = manager.intern_str("MyUninterpretedSort");
        let sort = manager.sorts.intern(SortKind::Uninterpreted(spur));

        let printer = Printer::new(&manager);
        assert_eq!(printer.print_sort(sort), "MyUninterpretedSort");
    }
}
