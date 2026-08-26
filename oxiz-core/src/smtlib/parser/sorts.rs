//! Sort parsing for the SMT-LIB2 parser

use super::super::lexer::TokenKind;
use super::Parser;
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use std::cell::Cell;

/// Maximum recursive nesting depth accepted by [`Parser::parse_sort`].
/// Mirrors the identical guard on [`Parser::parse_term`] in `terms.rs`:
/// adversarial input with pathologically deep nested parametric sorts (e.g.
/// millions of `(Array (Array (Array ...) ...) ...)`) would otherwise
/// overflow the native call stack; once this bound is exceeded we surface an
/// honest [`OxizError::ParseError`] instead of aborting the process.
const MAX_SORT_PARSE_DEPTH: u32 = 512;

thread_local! {
    /// Current recursion depth of [`Parser::parse_sort`] on this thread.
    static SORT_PARSE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII guard that decrements the [`SORT_PARSE_DEPTH`] counter when it
/// leaves scope, including on error unwinding, so the depth stays accurate
/// across every return path of `parse_sort`.
struct SortDepthGuard;

impl Drop for SortDepthGuard {
    fn drop(&mut self) {
        SORT_PARSE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

impl<'a> Parser<'a> {
    /// Expect a closing parenthesis ')'
    pub(super) fn expect_rparen(&mut self) -> Result<()> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected ')', found end of input".to_string(),
            })?;

        if !matches!(token.kind, TokenKind::RParen) {
            return Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected ')', found {:?}", token.kind),
            });
        }
        Ok(())
    }

    /// Parse a list of sorted variable bindings: ((name sort) ...)
    /// Consumes from the first variable up to and including the closing ')'
    pub(super) fn parse_sorted_vars(&mut self) -> Result<Vec<(String, SortId)>> {
        let mut vars = Vec::new();
        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }

            self.expect_lparen()?;
            let name = self.expect_symbol()?;
            let sort = self.parse_sort()?;
            self.expect_rparen()?;
            vars.push((name, sort));
        }
        Ok(vars)
    }

    /// Resolve a sort name to a SortId (handles built-in sorts, aliases, bitvec, uninterpreted)
    pub(super) fn parse_sort_name(&mut self, name: &str) -> Result<SortId> {
        match name {
            "Bool" => Ok(self.manager.sorts.bool_sort),
            "Int" => Ok(self.manager.sorts.int_sort),
            "Real" => Ok(self.manager.sorts.real_sort),
            "String" => Ok(self.manager.sorts.string_sort()),
            // `RoundingMode` is a first-class sort (`SortKind::RoundingMode`):
            // `(declare-const m RoundingMode)` builds a real, solvable
            // rounding-mode constant.  Its five inhabitants are nullary `Var`
            // terms at this sort, so EUF decides equalities between them; the
            // finiteness of the domain is enforced by the solver layer, which
            // asserts a closure axiom per declared constant plus the pairwise
            // distinctness of the five modes.
            //
            // Because those axioms attach per *nullary declaration*, the sort
            // is accepted here and then rejected in the positions no axiom can
            // reach — see `reject_unsupported_rounding_mode_position`.
            "RoundingMode" => {
                self.manager.note_rounding_mode_used();
                Ok(self.manager.sorts.rounding_mode_sort)
            }
            // `RegLan` is *implemented*, just not user-declarable.
            //
            // The regular-language sublanguage itself works end to end:
            // `str.to_re`, `re.++`, `re.union`, `re.inter`, `re.*`, `re.none`,
            // `re.all`, `re.allchar`, `re.range`, `re.opt`, `re.+`, `(_ re.loop
            // l u)` and `str.in_re` all parse, and the strings theory compiles
            // them into a Brzozowski-derivative automaton for membership
            // solving. What makes the sort name itself off limits is that the
            // encoding *reserves* it: every regex term is interned at the
            // built-in sort `TermManager::reglan_sort()` (see
            // `ast/manager/builder.rs`, "Regular-expression (RegLan) terms"),
            // which is `Uninterpreted("RegLan")`. Accepting `RegLan` here would
            // let a user-declared symbol be minted at that very sort, aliasing a
            // free constant into the regex encoding's own namespace — a regex
            // operand the derivative engine cannot compile and has no way to
            // recognise as foreign. This rejection is therefore load-bearing,
            // not a placeholder: it is what keeps the reserved name reserved.
            "RegLan" => Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "sort 'RegLan' is a reserved SMT-LIB Strings theory sort name: it names \
                    the built-in encoding OxiZ interns every regular-expression term at, so it \
                    cannot also name a user-declared constant or function. The regular-language \
                    operators themselves are fully supported — build regular expressions with \
                    str.to_re / re.++ / re.union / re.inter / re.* / re.range / re.none / re.all \
                    / re.allchar and test membership with str.in_re"
                    .to_string(),
            }),
            _ => {
                // Check for sort alias first. The chain is followed
                // *iteratively* by `resolve_sort_alias_chain`, which returns a
                // name that is itself no longer an alias — so the recursive
                // call below descends exactly one level. Re-entering this
                // function per link is what turned a cyclic alias table into
                // unbounded recursion (`(define-sort A () A)` then
                // `(declare-const x (Array A Int))` aborted the process with a
                // stack overflow).
                if let Some(base_sort) = self.resolve_sort_alias_chain(name)? {
                    return self.parse_sort_name(&base_sort);
                }

                // A name introduced by `declare-datatype(s)` denotes that
                // datatype's own sort, not a fresh uninterpreted one.
                //
                // Falling through to the `Uninterpreted` case below gave
                // `(declare-const l Lst)` a sort unrelated to the one
                // `mk_dt_constructor` stamps on `(cons 1 nil)`, so `l` and the
                // constructor applications lived in different sorts and nothing
                // downstream could tell that `l` was a datatype at all: the
                // solver's datatype axiomatisation found no terms to axiomatise
                // and every structural property (exhaustiveness, reconstruction,
                // acyclicity, …) silently went missing.
                //
                // `is_datatype_declared` covers a datatype declared in an
                // earlier text fragment parsed against the same `TermManager`;
                // `dt_sorts` additionally covers the datatype *currently* being
                // declared, whose own name appears in its recursive fields
                // (`(tail Lst)`) before the declaration is complete.
                if let Some(&sort_id) = self.dt_sorts.get(name) {
                    return Ok(sort_id);
                }
                if self.manager.sorts.is_datatype_declared(name) {
                    let sort_id = self.manager.sorts.mk_datatype_sort(name);
                    self.dt_sorts.insert(name.to_string(), sort_id);
                    return Ok(sort_id);
                }

                // Check for BitVec
                // Note: Proper SMT-LIB2 syntax is `(_ BitVec n)` which requires
                // parsing an indexed identifier. For now, we support simple names
                // like "BitVec32" as a compromise.
                if let Some(width_str) = name.strip_prefix("BitVec") {
                    if let Ok(width) = width_str.parse::<u32>() {
                        if width > 0 && width <= 65536 {
                            Ok(self.manager.sorts.bitvec(width))
                        } else {
                            Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("invalid BitVec width: {width} (must be 1-65536)"),
                            })
                        }
                    } else if width_str.is_empty() {
                        // Just "BitVec" without width - use default 32
                        Ok(self.manager.sorts.bitvec(32))
                    } else {
                        Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!("invalid BitVec sort name: {name}"),
                        })
                    }
                } else {
                    // Uninterpreted sort
                    let spur = self.manager.intern_str(name);
                    Ok(self
                        .manager
                        .sorts
                        .intern(crate::sort::SortKind::Uninterpreted(spur)))
                }
            }
        }
    }

    /// Follow a chain of 0-arity `define-sort` aliases from `name` to the
    /// first name that is not itself such an alias.
    ///
    /// Returns `Ok(None)` when `name` names no 0-arity alias at all (so the
    /// caller keeps its own resolution), and `Ok(Some(base))` for the end of
    /// the chain, which is guaranteed not to be an alias.
    ///
    /// A chain that takes more steps than the table has entries must revisit a
    /// name, i.e. the table is cyclic. `define-sort` rejects cycles when they
    /// are defined (see the `"define-sort"` handler in `commands.rs`), so this
    /// bound never fires on a table this parser built; it is kept so the walk
    /// is *total* by construction rather than by the caller's discipline.
    fn resolve_sort_alias_chain(&self, name: &str) -> Result<Option<String>> {
        match self.sort_aliases.get(name) {
            // Only 0-arity aliases resolve by name; a parametric one needs
            // its arguments substituted and is rejected at definition time.
            Some((params, base)) if params.is_empty() => {
                let mut current = base.clone();
                for _ in 0..self.sort_aliases.len() {
                    match self.sort_aliases.get(&current) {
                        Some((params, base)) if params.is_empty() => current = base.clone(),
                        _ => return Ok(Some(current)),
                    }
                }
                Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!(
                        "sort alias '{name}' is cyclic: resolving it does not terminate"
                    ),
                })
            }
            _ => Ok(None),
        }
    }

    /// Whether `sort` *is*, or structurally contains, an uninterpreted or
    /// datatype sort spelled `name`.
    ///
    /// The `define-sort` handler uses this to reject an abbreviation whose
    /// body names the abbreviation itself — directly (`(define-sort A () A)`),
    /// through another abbreviation (`(define-sort A () B)` then
    /// `(define-sort B () A)`, whose body resolves back to `B`), or nested
    /// inside a compound body (`(define-sort A () (Array A Int))`, where the
    /// inner `A` would otherwise become a fresh free sort unrelated to the
    /// abbreviation).
    ///
    /// The walk uses an explicit worklist rather than recursion: `Array`
    /// sorts are interned bottom-up so the structure cannot itself be cyclic,
    /// but it can be [`MAX_SORT_PARSE_DEPTH`] levels deep, and a definition
    /// check has no business consuming that much native stack.
    pub(super) fn sort_mentions_name(&self, sort: SortId, name: &str) -> bool {
        use crate::sort::SortKind;
        let mut work = vec![sort];
        let mut seen = FxHashSet::default();
        while let Some(current) = work.pop() {
            if !seen.insert(current) {
                continue;
            }
            let Some(s) = self.manager.sorts.get(current) else {
                continue;
            };
            match &s.kind {
                // Interned by the *term* manager's interner.
                SortKind::Uninterpreted(spur) => {
                    if self.manager.resolve_str(*spur) == name {
                        return true;
                    }
                }
                // A sort parameter's name is interned by `SortManager` itself
                // (`mk_sort_parameter` / `define_parametric_sort`), so it must
                // be resolved through the sort manager's own interner — see
                // the matching arm split in `Printer::write_sort`.
                SortKind::Parameter(spur) => {
                    if self.manager.sorts.resolve_spur(*spur) == name {
                        return true;
                    }
                }
                // Interned by the *sort* manager's own interner; the two are
                // separate `Rodeo`s (see `sort_id_to_string`).
                SortKind::Datatype(spur) => {
                    if self.manager.sorts.resolve_spur(*spur) == name {
                        return true;
                    }
                }
                SortKind::Array { domain, range } => {
                    work.push(*domain);
                    work.push(*range);
                }
                SortKind::Parametric { name: head, args } => {
                    if self.manager.sorts.resolve_spur(*head) == name {
                        return true;
                    }
                    work.extend(args.iter().copied());
                }
                // `RoundingMode` is a built-in leaf whose name is reserved:
                // `parse_sort_name` never lets a `define-sort` alias or any
                // other user declaration bind that spelling, so it can never
                // be the `name` this walk is hunting for.
                SortKind::Bool
                | SortKind::Int
                | SortKind::Real
                | SortKind::String
                | SortKind::BitVec(_)
                | SortKind::FloatingPoint { .. }
                | SortKind::RoundingMode => {}
            }
        }
        false
    }

    /// Parse an indexed identifier: (_ name index1 index2 ...)
    /// Returns (name, indices). LParen already consumed by caller; consumes trailing RParen.
    pub(super) fn parse_indexed_identifier(&mut self) -> Result<(String, Vec<u32>)> {
        // Expect underscore symbol
        let underscore = self.expect_symbol()?;
        if underscore != "_" {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("expected '_', found '{underscore}'"),
            });
        }

        // Get the identifier name
        let name = self.expect_symbol()?;

        // Parse indices (numerals)
        let mut indices = Vec::new();
        loop {
            if let Some(token) = self.lexer.peek() {
                match &token.kind {
                    TokenKind::RParen => {
                        self.lexer.next_token(); // consume rparen
                        break;
                    }
                    TokenKind::Numeral(n) => {
                        let n = n.clone();
                        self.lexer.next_token();
                        let idx = n.parse::<u32>().map_err(|_| OxizError::ParseError {
                            position: token.start,
                            message: format!("invalid index: {n}"),
                        })?;
                        indices.push(idx);
                    }
                    _ => {
                        return Err(OxizError::ParseError {
                            position: token.start,
                            message: format!("expected numeral or ')', found {:?}", token.kind),
                        });
                    }
                }
            } else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input in indexed identifier".to_string(),
                });
            }
        }

        Ok((name, indices))
    }

    /// Parse a sort expression (simple name, indexed identifier, or parametric sort).
    ///
    /// This wraps the actual recursive-descent logic in a depth guard so
    /// that deeply nested sort input cannot overflow the stack; see
    /// [`MAX_SORT_PARSE_DEPTH`].
    pub(super) fn parse_sort(&mut self) -> Result<SortId> {
        let depth = SORT_PARSE_DEPTH.with(|d| {
            let next = d.get().saturating_add(1);
            d.set(next);
            next
        });
        let _guard = SortDepthGuard;
        if depth > MAX_SORT_PARSE_DEPTH {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "sort nesting too deep".to_string(),
            });
        }
        self.parse_sort_inner()
    }

    /// Inner sort parser; callers must go through [`Parser::parse_sort`] so
    /// the recursion-depth guard stays in effect.
    fn parse_sort_inner(&mut self) -> Result<SortId> {
        if let Some(token) = self.lexer.peek() {
            match &token.kind {
                TokenKind::Symbol(s) => {
                    let s = s.clone();
                    self.lexer.next_token();
                    self.parse_sort_name(&s)
                }
                TokenKind::LParen => {
                    self.lexer.next_token(); // consume lparen

                    // Check if this is an indexed identifier or a parametric sort like Array
                    let next_token = self.lexer.peek().ok_or_else(|| OxizError::ParseError {
                        position: self.lexer.position(),
                        message: "unexpected end of input in sort".to_string(),
                    })?;

                    if matches!(next_token.kind, TokenKind::Symbol(ref s) if s == "_") {
                        // Indexed identifier: (_ BitVec 32)
                        let (name, indices) = self.parse_indexed_identifier()?;

                        match name.as_str() {
                            "BitVec" => {
                                if indices.len() != 1 {
                                    return Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "BitVec requires exactly 1 index, got {}",
                                            indices.len()
                                        ),
                                    });
                                }
                                let width = indices[0];
                                if width > 0 && width <= 65536 {
                                    Ok(self.manager.sorts.bitvec(width))
                                } else {
                                    Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "invalid BitVec width: {width} (must be 1-65536)"
                                        ),
                                    })
                                }
                            }
                            "FloatingPoint" => {
                                if indices.len() != 2 {
                                    return Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "FloatingPoint requires exactly 2 indices (eb, sb), got {}",
                                            indices.len()
                                        ),
                                    });
                                }
                                let eb = indices[0]; // exponent bits
                                let sb = indices[1]; // significand bits
                                Ok(self.manager.sorts.float_sort(eb, sb))
                            }
                            _ => Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("unknown indexed sort: {name}"),
                            }),
                        }
                    } else if let TokenKind::Symbol(s) = &next_token.kind {
                        // Parametric sort like (Array Int Int)
                        let sort_name = s.clone();
                        self.lexer.next_token(); // consume the symbol

                        match sort_name.as_str() {
                            "Array" => {
                                // Parse domain and range sorts
                                let domain = self.parse_sort()?;
                                let range = self.parse_sort()?;
                                self.expect_rparen()?;
                                self.reject_unsupported_rounding_mode_position(
                                    domain,
                                    "an array domain",
                                )?;
                                self.reject_unsupported_rounding_mode_position(
                                    range,
                                    "an array range",
                                )?;
                                Ok(self.manager.sorts.array(domain, range))
                            }
                            _ => Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("unknown parametric sort: {sort_name}"),
                            }),
                        }
                    } else {
                        Err(OxizError::ParseError {
                            position: next_token.start,
                            message: format!("unexpected token in sort: {:?}", next_token.kind),
                        })
                    }
                }
                _ => Err(OxizError::ParseError {
                    position: token.start,
                    message: format!("expected sort, found {:?}", token.kind),
                }),
            }
        } else {
            Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected sort, found end of input".to_string(),
            })
        }
    }

    /// Reject `RoundingMode` in a position whose finiteness OxiZ cannot yet
    /// enforce.
    ///
    /// `RoundingMode` has exactly five elements, but nothing about the *sort*
    /// says so: the cardinality lives in axioms the solver asserts — a closure
    /// axiom `(or (= m RNE) ... (= m RTZ))` per declared rounding-mode
    /// constant, plus one pairwise-distinctness axiom over the five modes.
    /// The closure axiom can only be attached where there is a single symbol
    /// to attach it to, i.e. a *nullary* declaration.
    ///
    /// Everywhere else — a function's argument or result sort, an array's
    /// domain or range, a datatype field — the rounding modes reachable
    /// through the sort are unboundedly many and unnamed, so no closure axiom
    /// can be written down. Accepting such a declaration would silently make
    /// `RoundingMode` behave as an *infinite free sort*: `(distinct (g 1)
    /// (g 2) ... (g 6))` over `(declare-fun g (Int) RoundingMode)` would
    /// answer `sat`, which is wrong. Rejecting is the honest alternative.
    ///
    /// `position` names the offending position for the diagnostic and reads as
    /// the object of "not as ...", e.g. `"an array range"`.
    pub(super) fn reject_unsupported_rounding_mode_position(
        &self,
        sort: SortId,
        position: &str,
    ) -> Result<()> {
        if !self.manager.sorts.sort_mentions_rounding_mode(sort) {
            return Ok(());
        }
        Err(OxizError::ParseError {
            position: self.lexer.position(),
            message: format!(
                "sort 'RoundingMode' is supported only in nullary declaration position — \
                 (declare-const m RoundingMode) or (declare-fun m () RoundingMode) — not as \
                 {position}. The sort's five-element domain is enforced by a closure axiom \
                 attached to each declared rounding-mode constant, and there is no way to attach \
                 that axiom to the rounding modes reachable through {position}; accepting it \
                 would silently treat RoundingMode as an infinite free sort"
            ),
        })
    }

    /// Convert a SortId to its canonical SMT-LIB2 string representation.
    ///
    /// Driven by an explicit worklist rather than by recursion. The return
    /// type is a plain `String` — there is no error channel a depth cap could
    /// report through, so a cap here could only ever produce a *silently
    /// wrong* sort name. `(Array (Array (Array ...)))` nesting is bounded at
    /// [`MAX_SORT_PARSE_DEPTH`] when it comes from SMT-LIB text, but
    /// `SortManager::array` is `pub` and interns in constant stack, so an
    /// embedder can hand this function an arbitrarily deep sort.
    pub(super) fn sort_id_to_string(&self, sort_id: SortId) -> String {
        use crate::sort::SortKind;

        /// One pending step of the walk: a sort still to render, or literal
        /// punctuation already scheduled to follow one.
        enum Step {
            /// Render this sort.
            Sort(SortId),
            /// Emit this literal.
            Text(&'static str),
        }

        let mut out = String::new();
        let mut stack = vec![Step::Sort(sort_id)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Text(text) => out.push_str(text),
                Step::Sort(id) => {
                    let Some(sort) = self.manager.sorts.get(id) else {
                        out.push_str("Unknown");
                        continue;
                    };
                    match &sort.kind {
                        SortKind::Bool => out.push_str("Bool"),
                        SortKind::Int => out.push_str("Int"),
                        SortKind::Real => out.push_str("Real"),
                        SortKind::String => out.push_str("String"),
                        SortKind::BitVec(w) => out.push_str(&format!("(_ BitVec {w})")),
                        SortKind::FloatingPoint { eb, sb } => {
                            out.push_str(&format!("(_ FloatingPoint {eb} {sb})"));
                        }
                        // The spelling must be exactly the one
                        // `Parser::parse_sort_name` and the solver's
                        // `Context::parse_sort_name` both accept: these strings
                        // are what `Command::DeclareConst` carries, so a
                        // mismatch here would resolve the declared constant at
                        // a *different* `SortId` than the one its occurrences
                        // in terms were interned at, silently splitting the
                        // symbol in two.
                        SortKind::RoundingMode => out.push_str("RoundingMode"),
                        SortKind::Array { domain, range } => {
                            out.push_str("(Array ");
                            // Pushed in reverse of emission order.
                            stack.push(Step::Text(")"));
                            stack.push(Step::Sort(*range));
                            stack.push(Step::Text(" "));
                            stack.push(Step::Sort(*domain));
                        }
                        SortKind::Uninterpreted(spur) => {
                            out.push_str(self.manager.resolve_str(*spur));
                        }
                        // A sort parameter's name is interned by `SortManager`
                        // itself (`mk_sort_parameter` /
                        // `define_parametric_sort`), so it resolves through
                        // the sort manager's interner, like `Datatype` below.
                        SortKind::Parameter(spur) => {
                            out.push_str(self.manager.sorts.resolve_spur(*spur));
                        }
                        // A datatype sort's name is interned by `SortManager`
                        // itself (`mk_datatype_sort` / `declare_datatype`), not
                        // by the term manager, so it must be resolved through
                        // the sort manager's own interner — the two are
                        // separate `Rodeo`s and crossing them yields the wrong
                        // string or an out-of-range key. Same for a parametric
                        // sort's head name (`declare_parametric_sort`).
                        SortKind::Datatype(spur) => {
                            out.push_str(self.manager.sorts.resolve_spur(*spur));
                        }
                        SortKind::Parametric { name, args } => {
                            out.push('(');
                            out.push_str(self.manager.sorts.resolve_spur(*name));
                            stack.push(Step::Text(")"));
                            for arg in args.iter().rev() {
                                stack.push(Step::Sort(*arg));
                                stack.push(Step::Text(" "));
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::Parser;
    use crate::ast::TermManager;
    use crate::sort::SortKind;

    /// `sort_id_to_string` renders `(Array ...)` with an explicit worklist.
    /// It returns a plain `String`, so a depth cap could only ever have
    /// produced a *different, wrong* sort name; recursing instead aborted the
    /// process, and `SortManager::array` is `pub` and interns in constant
    /// stack, so nothing bounds the depth an embedder can reach.
    ///
    /// Runs on a 128 KiB stack: the assertion is that the call returns at all.
    #[test]
    fn sort_id_to_string_survives_twelve_thousand_five_hundred_array_levels() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let mut sort = int_sort;
                for _ in 0..12_500 {
                    sort = manager.sorts.array(int_sort, sort);
                }
                let parser = Parser::new("", &mut manager);
                parser.sort_id_to_string(sort).len()
            })
            .expect("spawn");
        let len = handle.join().expect("worker thread must not overflow");
        assert_eq!(len, 12_500 * 11 + 3 + 12_500);
    }

    /// Semantic pin for the shapes the recursive version already handled.
    #[test]
    fn sort_id_to_string_output_is_unchanged_for_shallow_sorts() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let bv = manager.sorts.bitvec(8);
        let inner = manager.sorts.array(bv, bool_sort);
        let nested = manager.sorts.array(int_sort, inner);
        let parser = Parser::new("", &mut manager);
        assert_eq!(parser.sort_id_to_string(int_sort), "Int");
        assert_eq!(parser.sort_id_to_string(bool_sort), "Bool");
        assert_eq!(parser.sort_id_to_string(bv), "(_ BitVec 8)");
        assert_eq!(
            parser.sort_id_to_string(nested),
            "(Array Int (Array (_ BitVec 8) Bool))"
        );
    }

    /// A sort *parameter* and a parametric sort application used to both
    /// render as the literal string `"Unknown"` — two different sorts
    /// collapsing onto one name, which `define-sort`/`define-fun` then stored
    /// as the parameter's declared sort text. They now render honestly.
    #[test]
    fn parameter_and_parametric_sorts_render_their_real_names() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        // A sort parameter's name spur is minted by the *sort* manager in
        // every real producer (`mk_sort_parameter`,
        // `instantiate_parametric_sort`, `define_parametric_sort`) — build it
        // the same way here so the pin reflects reality.
        let param = manager.sorts.mk_sort_parameter("T");
        // A parametric sort's head name lives in the *sort* manager's own
        // interner, which is what `sort_id_to_string` must resolve it through.
        let list_spur = manager.sorts.intern_str("List");
        let list = manager.sorts.intern(SortKind::Parametric {
            name: list_spur,
            args: smallvec::smallvec![int_sort],
        });
        let parser = Parser::new("", &mut manager);
        assert_eq!(parser.sort_id_to_string(param), "T");
        assert_eq!(parser.sort_id_to_string(list), "(List Int)");
    }
}
