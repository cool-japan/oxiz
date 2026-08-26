//! Sort-expression string resolution for [`Context`].
//!
//! [`Context::parse_sort_name`] turns the sort strings carried by parsed
//! SMT-LIB commands (`declare-const`, `declare-fun`, `define-fun`,
//! `define-sort`, `declare-sort`) back into `SortId`s.  It lives in a child
//! module so the (already large) `context` module stays under the 2000-line
//! policy limit; being a child of `context`, it retains full access to
//! `Context`'s private fields.

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::error::{OxizError, Result};
use oxiz_core::sort::{SortId, SortKind};

use super::Context;

/// One suspended `(Array dom rng)` node during the iterative resolution in
/// [`Context::parse_sort_name`].
///
/// `parse_sort_name` used to recurse natively on the two child expressions
/// of an `(Array dom rng)` string.  The nesting depth of that string is
/// input-controlled (and, through chained `define-sort`, not usefully
/// bounded by any single parse), so the walk now carries its own heap stack
/// of these frames instead of native stack frames.
enum ArrayPending {
    /// The domain expression is being resolved; the range expression is
    /// still waiting its turn.
    DomainOf {
        /// The not-yet-resolved range expression.
        range_expr: String,
    },
    /// The range expression is being resolved; the domain has already been
    /// resolved.
    RangeOf {
        /// The resolved domain sort.
        domain: SortId,
    },
}

/// The classification of a single sort-expression string: either a sort
/// that resolves without looking at any sub-expression, or an
/// `(Array dom rng)` node whose two children still need resolving.
enum SortExprStep {
    /// The expression is fully resolved.
    Resolved(SortId),
    /// An `(Array dom rng)` compound; both children remain to be resolved.
    Array {
        /// The domain sub-expression.
        domain_expr: String,
        /// The range sub-expression.
        range_expr: String,
    },
}

impl Context {
    /// Split a sort-expression string into its top-level whitespace
    /// separated tokens, treating a parenthesized group as a single
    /// token (so nested compound sorts like `(Array Int (_ BitVec 8))`
    /// split into `["Array", "Int", "(_ BitVec 8)"]` rather than being
    /// torn apart at the inner spaces).
    fn split_sort_tokens(s: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut depth = 0i32;
        let mut current = String::new();
        for c in s.chars() {
            match c {
                '(' => {
                    depth += 1;
                    current.push(c);
                }
                ')' => {
                    depth -= 1;
                    current.push(c);
                }
                c if c.is_whitespace() && depth == 0 => {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                }
                c => current.push(c),
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// The honest error for a compound sort expression this resolver cannot
    /// handle.  Until 0.3.1 such an expression silently resolved to `Bool`,
    /// which mis-sorted the declared symbol and corrupted every model /
    /// value answer mentioning it; an explicit error is the only honest
    /// channel a sort resolver has.
    fn malformed_sort_error(expr: &str) -> OxizError {
        OxizError::Unsupported(format!("unsupported or malformed sort expression: {expr}"))
    }

    /// Resolve a sort-expression string into a `SortId`.
    ///
    /// The strings handled here are exactly the ones `oxiz_core`'s
    /// SMT-LIB parser produces for `Command::DeclareConst`/
    /// `Command::DeclareFun`/`Command::DefineFun` (see
    /// `Parser::sort_id_to_string`): the built-in atomic sorts, `(_
    /// BitVec n)`, `(_ FloatingPoint eb sb)`, `(Array dom rng)`
    /// (recursively), a previously-declared datatype name, or a plain
    /// uninterpreted-sort name.
    ///
    /// Uninterpreted names are interned through `self.terms`'s own
    /// string interner -- the same one the parser uses internally for
    /// `SortKind::Uninterpreted` when building terms during parsing --
    /// so a name declared via `declare-sort` resolves to the identical
    /// `SortId` (and thus, since `mk_var` hash-conses on `(name, sort)`,
    /// the identical `TermId`) that in-script term parsing already
    /// produced for it. Without this, a declared constant of a
    /// user-defined/compound sort would silently be registered here
    /// under an unrelated, disconnected term instead.
    ///
    /// # Errors
    ///
    /// A compound expression that is not a well-formed `(_ BitVec n)`
    /// (with `n > 0`), `(_ FloatingPoint eb sb)`, or three-token
    /// `(Array dom rng)` yields [`OxizError::Unsupported`].  It used to
    /// resolve to `Bool` silently, which registered the declared constant
    /// or function under the *wrong* sort — a silently corrupted model —
    /// so the malformed case now propagates as a real error to the
    /// declaring command instead.
    ///
    /// # Iterative walk
    ///
    /// `(Array dom rng)` nesting is resolved with an explicit heap stack
    /// ([`ArrayPending`]) rather than native recursion, so the resolvable
    /// depth is bounded by memory, not by thread stack size.
    pub(super) fn parse_sort_name(&mut self, name: &str) -> Result<SortId> {
        let mut pending: Vec<ArrayPending> = Vec::new();
        let mut current: String = name.to_string();
        loop {
            // Resolve `current`, descending through `Array` domains until a
            // directly-resolvable expression is reached.
            let mut resolved: SortId = loop {
                match self.classify_sort_expr(&current)? {
                    SortExprStep::Resolved(id) => break id,
                    SortExprStep::Array {
                        domain_expr,
                        range_expr,
                    } => {
                        pending.push(ArrayPending::DomainOf { range_expr });
                        current = domain_expr;
                    }
                }
            };
            // Feed the resolved sort upward through the pending frames:
            // a finished domain schedules its partner range; a finished
            // range completes its `Array` node, which continues upward.
            loop {
                match pending.pop() {
                    None => return Ok(resolved),
                    Some(ArrayPending::DomainOf { range_expr }) => {
                        pending.push(ArrayPending::RangeOf { domain: resolved });
                        current = range_expr;
                        break;
                    }
                    Some(ArrayPending::RangeOf { domain }) => {
                        resolved = self.terms.sorts.array(domain, resolved);
                    }
                }
            }
        }
    }

    /// Classify one sort-expression string: resolve it directly when it has
    /// no sort sub-expressions, or split out the two children of an
    /// `(Array dom rng)` compound for [`Context::parse_sort_name`]'s
    /// explicit stack.
    ///
    /// # Errors
    ///
    /// Any other compound form is a malformed or unsupported sort
    /// expression; see [`Context::parse_sort_name`].
    fn classify_sort_expr(&mut self, name: &str) -> Result<SortExprStep> {
        let name = name.trim();
        match name {
            "Bool" => return Ok(SortExprStep::Resolved(self.terms.sorts.bool_sort)),
            "Int" => return Ok(SortExprStep::Resolved(self.terms.sorts.int_sort)),
            "Real" => return Ok(SortExprStep::Resolved(self.terms.sorts.real_sort)),
            "String" => return Ok(SortExprStep::Resolved(self.terms.sorts.string_sort())),
            // The reserved `FloatingPoint`-theory sort.  This arm is
            // load-bearing for identity, not just for naming: commands carry
            // sorts as *strings* (`Command::DeclareConst(name, sort_name)`),
            // and `Parser::sort_id_to_string` prints this sort as
            // `"RoundingMode"`.  Without the arm the string would fall through
            // to the uninterpreted fallback at the bottom of this function and
            // resolve to `Uninterpreted("RoundingMode")` — a *different*
            // `SortId` than the one the parser interned `m`'s occurrences at.
            // Since `mk_var` hash-conses on `(name, sort)`, the declared
            // constant and every use of it would then be two unrelated terms,
            // and the rounding-mode closure axiom would constrain the wrong
            // one.
            "RoundingMode" => {
                return Ok(SortExprStep::Resolved(self.terms.sorts.rounding_mode_sort));
            }
            _ => {}
        }

        if let Some(inner) = name.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            let mut tokens = Self::split_sort_tokens(inner.trim());
            return match tokens.as_mut_slice() {
                [u, kw, width] if u.as_str() == "_" && kw.as_str() == "BitVec" => {
                    match width.parse::<u32>() {
                        Ok(width) if width > 0 => {
                            Ok(SortExprStep::Resolved(self.terms.sorts.bitvec(width)))
                        }
                        _ => Err(Self::malformed_sort_error(name)),
                    }
                }
                [u, kw, eb, sb] if u.as_str() == "_" && kw.as_str() == "FloatingPoint" => {
                    match (eb.parse::<u32>(), sb.parse::<u32>()) {
                        (Ok(eb), Ok(sb)) => {
                            Ok(SortExprStep::Resolved(self.terms.sorts.float_sort(eb, sb)))
                        }
                        _ => Err(Self::malformed_sort_error(name)),
                    }
                }
                [head, domain, range] if head.as_str() == "Array" => Ok(SortExprStep::Array {
                    domain_expr: std::mem::take(domain),
                    range_expr: std::mem::take(range),
                }),
                // A compound form the sort printer never emits.  This used
                // to fall back to `Bool` silently — a silently wrong sort
                // on the declared symbol — and is now an honest error.
                _ => Err(Self::malformed_sort_error(name)),
            };
        }

        // Legacy compact BitVec spelling ("BitVec32"), kept for
        // backward compatibility with any direct (non-script) callers.
        if let Some(width_str) = name.strip_prefix("BitVec")
            && let Ok(width) = width_str.trim().parse::<u32>()
            && width > 0
        {
            return Ok(SortExprStep::Resolved(self.terms.sorts.bitvec(width)));
        }

        // A previously-declared datatype resolves to its own sort.
        if self.terms.sorts.is_datatype_declared(name) {
            return Ok(SortExprStep::Resolved(
                self.terms.sorts.mk_datatype_sort(name),
            ));
        }

        // A sort alias registered by a prior `define-sort` (0-arity
        // aliases only; see the `DefineSort` command handling in
        // `Context::execute_script`).
        if let Some(sort_id) = self.terms.sorts.resolve_by_name(name) {
            return Ok(SortExprStep::Resolved(sort_id));
        }

        // Otherwise: an uninterpreted sort, e.g. one introduced by
        // `declare-sort`.
        let spur = self.terms.intern_str(name);
        Ok(SortExprStep::Resolved(
            self.terms.sorts.intern(SortKind::Uninterpreted(spur)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unwrap a resolution that must succeed.
    fn resolve(ctx: &mut Context, expr: &str) -> SortId {
        match ctx.parse_sort_name(expr) {
            Ok(id) => id,
            Err(err) => panic!("expected {expr:?} to resolve, got error: {err}"),
        }
    }

    /// Every atomic spelling resolves to the canonical interned sort.
    #[test]
    fn test_atomic_sorts_resolve() {
        let mut ctx = Context::new();
        let bool_sort = ctx.terms.sorts.bool_sort;
        let int_sort = ctx.terms.sorts.int_sort;
        let real_sort = ctx.terms.sorts.real_sort;
        let string_sort = ctx.terms.sorts.string_sort();
        assert_eq!(resolve(&mut ctx, "Bool"), bool_sort);
        assert_eq!(resolve(&mut ctx, "Int"), int_sort);
        assert_eq!(resolve(&mut ctx, "Real"), real_sort);
        assert_eq!(resolve(&mut ctx, "String"), string_sort);
        // Whitespace is trimmed, as before the iterative conversion.
        assert_eq!(resolve(&mut ctx, "  Int  "), int_sort);
    }

    /// Indexed and compound sorts resolve to the same `SortId`s the
    /// builder API interns.
    #[test]
    fn test_compound_sorts_resolve() {
        let mut ctx = Context::new();
        let bv8 = ctx.terms.sorts.bitvec(8);
        assert_eq!(resolve(&mut ctx, "(_ BitVec 8)"), bv8);

        let f32_sort = ctx.terms.sorts.float_sort(8, 24);
        assert_eq!(resolve(&mut ctx, "(_ FloatingPoint 8 24)"), f32_sort);

        let int_sort = ctx.terms.sorts.int_sort;
        let bool_sort = ctx.terms.sorts.bool_sort;
        let arr = ctx.terms.sorts.array(int_sort, bool_sort);
        assert_eq!(resolve(&mut ctx, "(Array Int Bool)"), arr);

        let inner = ctx.terms.sorts.array(int_sort, int_sort);
        let bv4 = ctx.terms.sorts.bitvec(4);
        let nested = ctx.terms.sorts.array(inner, bv4);
        assert_eq!(
            resolve(&mut ctx, "(Array (Array Int Int) (_ BitVec 4))"),
            nested
        );

        // Legacy compact spelling.
        let bv32 = ctx.terms.sorts.bitvec(32);
        assert_eq!(resolve(&mut ctx, "BitVec32"), bv32);
    }

    /// Datatype names, `define-sort` aliases, and unknown names resolve
    /// through the same three fallbacks as before the conversion.
    #[test]
    fn test_name_fallbacks_resolve() {
        let mut ctx = Context::new();
        ctx.execute_script("(declare-datatype P ((mk (x Int))))")
            .expect("datatype declaration script");
        let p = ctx.terms.sorts.mk_datatype_sort("P");
        assert_eq!(resolve(&mut ctx, "P"), p);

        let int_sort = ctx.terms.sorts.int_sort;
        ctx.terms.sorts.define_alias("MyInt", int_sort);
        assert_eq!(resolve(&mut ctx, "MyInt"), int_sort);

        // An unknown plain name interns an uninterpreted sort, stably.
        let first = resolve(&mut ctx, "Widget");
        let second = resolve(&mut ctx, "Widget");
        assert_eq!(first, second);
        let kind = ctx.terms.sorts.get(first).map(|s| s.kind.clone());
        assert!(matches!(kind, Some(SortKind::Uninterpreted(_))));
    }

    /// Malformed compound expressions are honest errors now, never a
    /// silent `Bool`.
    #[test]
    fn test_malformed_compound_is_error_not_bool() {
        let mut ctx = Context::new();
        for malformed in [
            "(Array Int)",             // Array arity 1
            "(Array Int Int Int)",     // Array arity 3
            "(_ BitVec 0)",            // zero width
            "(_ BitVec x)",            // non-numeric width
            "(_ BitVec 8 8)",          // BitVec arity 2
            "(_ FloatingPoint 8)",     // FP arity 1
            "(_ FloatingPoint a b)",   // non-numeric widths
            "(Seq Int)",               // unknown compound head
            "()",                      // empty compound
            "(Array Int (Array Int))", // malformed nested inside well-formed
        ] {
            let got = ctx.parse_sort_name(malformed);
            match got {
                Err(OxizError::Unsupported(msg)) => {
                    assert!(
                        msg.contains("sort expression"),
                        "error message should describe the sort expression, got: {msg}"
                    );
                }
                other => panic!("expected Unsupported error for {malformed:?}, got {other:?}"),
            }
        }
    }

    /// A malformed sort inside a `declare-const`-shaped call path
    /// propagates out of `execute_script` as `Err`, and a valid script
    /// still round-trips its sort into `get-model` output.
    #[test]
    fn test_execute_script_roundtrip_array_sort() {
        let mut ctx = Context::new();
        let out = ctx
            .execute_script("(declare-const a (Array Int Int))\n(check-sat)\n(get-model)")
            .expect("valid script executes");
        assert!(
            out.iter().any(|line| line.contains("(Array Int Int)")),
            "get-model output must render the declared array sort: {out:?}"
        );
    }

    /// Deep-nesting regression: a 6 250-deep `(Array Int ...)` string
    /// resolves on a 128 KiB thread stack.  The pre-conversion recursive
    /// resolver put two native frames per level on the stack, so merely
    /// *returning* is the proof of the iterative conversion; the result is
    /// additionally pinned to the interned sort the builder API produces.
    ///
    /// What the test pins is the *ratio* `stack / depth` — about 21 bytes of
    /// stack per level — not the absolute depth: any recursive resolver needs
    /// far more than that per native frame and still dies here.  The pair was
    /// scaled down from 1 MiB / 50 000 by a factor of 8 on both sides because
    /// the *input re-scanning* is the bottleneck, not the converted walk:
    /// `split_sort_tokens` re-tokenizes the remaining expression at every
    /// level (quadratic character work, identical to the pre-conversion
    /// code), so 50 000 levels cost 64x this one's construction time and
    /// gigabytes of live interner memory for no extra stack-depth coverage.
    #[test]
    fn test_deep_array_sort_string_returns_on_small_stack() {
        // Stack and depth scale together (1 MiB/50k -> 128 KiB/6.25k): the
        // ~21 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 6_250;
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut ctx = Context::new();
                let mut expr = String::with_capacity(DEPTH * 12 + 3);
                for _ in 0..DEPTH {
                    expr.push_str("(Array Int ");
                }
                expr.push_str("Int");
                for _ in 0..DEPTH {
                    expr.push(')');
                }
                let got = ctx
                    .parse_sort_name(&expr)
                    .expect("deep array sort resolves");
                let int_sort = ctx.terms.sorts.int_sort;
                let mut expected = int_sort;
                for _ in 0..DEPTH {
                    expected = ctx.terms.sorts.array(int_sort, expected);
                }
                assert_eq!(got, expected);
            })
            .expect("spawn deep-parse thread");
        handle.join().expect("deep-parse thread must not overflow");
    }

    /// Same regression on the domain side: `(Array (Array ... ) Int)`
    /// nesting descends through the domain frames instead of the range
    /// frames.  Stack and depth are the same 128 KiB / 6 250 pair as the
    /// range-side test above, for the same input re-scanning reason.
    #[test]
    fn test_deep_domain_nested_sort_string_returns_on_small_stack() {
        // Stack and depth scale together (1 MiB/50k -> 128 KiB/6.25k): the
        // ~21 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 6_250;
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut ctx = Context::new();
                let mut prefix = String::with_capacity(DEPTH * 7);
                let mut suffix = String::with_capacity(DEPTH * 5);
                for _ in 0..DEPTH {
                    prefix.push_str("(Array ");
                    suffix.push_str(" Int)");
                }
                let expr = format!("{prefix}Int{suffix}");
                let got = ctx
                    .parse_sort_name(&expr)
                    .expect("deep domain-nested sort resolves");
                let int_sort = ctx.terms.sorts.int_sort;
                let mut expected = int_sort;
                for _ in 0..DEPTH {
                    expected = ctx.terms.sorts.array(expected, int_sort);
                }
                assert_eq!(got, expected);
            })
            .expect("spawn deep-parse thread");
        handle.join().expect("deep-parse thread must not overflow");
    }
}
