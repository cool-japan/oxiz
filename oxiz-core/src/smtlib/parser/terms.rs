//! Term parsing for the SMT-LIB2 parser.
//!
//! Terms are parsed **iteratively**, by an explicit frame stack held on the
//! heap, rather than by recursive descent. `(- (- (- ... )))` nested a million
//! levels deep therefore costs a million `Vec` entries and a constant number of
//! native stack frames, instead of a million native frames.
//!
//! That matters because a library cannot know how much stack its caller has.
//! The recursive-descent version of this parser needed roughly 2.9 KiB of
//! native stack per nesting level in the release profile, so the nesting limit
//! below (1024) needed a 3 MiB stack to reach — more than the ~2 MiB a libtest
//! thread gets, and far more than the ~1 MiB an embedder's worker thread may
//! have. Past that point the process died of a stack overflow *before* the
//! limit could report an error, which is exactly the failure mode the limit
//! exists to prevent. Making the walk iterative removes the dependence on the
//! caller's stack size entirely; the limit stays as a resource bound.
//!
//! Reference: Z3's `smt2parser.cpp` parses expressions the same way, with an
//! explicit frame stack (`m_stack`) and result stack (`expr_stack()`) driven by
//! a single loop in `parse_expr`.

use super::super::lexer::{Token, TokenKind};
use super::build::operand_plan;
use super::commands::RmOperand;
use super::{Attribute, AttributeValue, Parser, parse_decimal_to_rational};
use crate::ast::{RoundingMode, TermId};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use num_bigint::BigInt;
use smallvec::SmallVec;
use std::cell::Cell;

/// Maximum depth of the **term this parser builds**, as accepted by
/// [`Parser::parse_term`].
///
/// This bounds the resulting AST, not the input's parenthesis nesting. The two
/// are not the same: SMT-LIB's n-ary `=>`, `xor`, `-`, `div`, `/` and `str.++`
/// have no n-ary term representation and are folded into binary chains, so a
/// syntactically flat `(str.++ x1 … x100000)` — nesting depth 2 — used to build
/// a 100 000-deep term and sail straight through this limit. Every downstream
/// mitigation that assumes "the parser bounds term depth" was void as a result,
/// from a single-line input file. [`Parser::charge_fold_depth`] now charges the
/// chain a fold will build against this same budget, so the name means what it
/// says. (`and`, `or`, `distinct`, `+`, `*`, `re.++`, `re.union` and `re.inter`
/// need no charge: they build genuinely n-ary `TermKind`s of depth 1.)
///
/// This is a *resource* bound, not a stack bound: the parser is iterative, so
/// exceeding it can no longer overflow the parser's own native stack. It is
/// kept because unbounded nesting is still unbounded work and memory, and
/// because the resulting [`OxizError::ParseError`] ("term nesting too deep") is
/// part of the observable contract.
///
/// The value is chosen to stay below the caps of the walks that later consume a
/// parsed term, so that anything this parser accepts can also be printed:
///
/// * `smtlib::printer::basic::MAX_PRINT_DEPTH` = 2000 (printing truncates
///   past it).
///
/// Substitution and evaluation no longer need such a margin at all.
/// `TermManager::substitute` and `TermManager::simplify` used to share a
/// depth cap, `ast::manager::query::MAX_QUERY_RECURSION_DEPTH` = 1000, that
/// bailed out past it -- that constant has been deleted; both were converted
/// to an explicit heap stack instead (see `ast/manager/query.rs`'s module
/// doc comment) and now accept any depth this parser can produce, with no
/// limit at all. `model::ModelEvaluator::eval` likewise no longer constrains
/// it: that walk is iterative too.
const MAX_PARSE_DEPTH: u32 = 1024;

thread_local! {
    /// Current *logical* term nesting depth on this thread.
    ///
    /// The driver keeps this in step with its frame stack. It is a thread-local
    /// rather than a plain local because [`Parser::parse_term`] can be re-entered
    /// while a parse is in progress (an attribute value such as
    /// `:pattern (...)` contains terms), and the nesting budget has to be shared
    /// across those re-entries rather than restarting at zero for each.
    static PARSE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Restores [`PARSE_DEPTH`] to the value it had when a driver was entered,
/// on every exit path including an early `?` return.
struct DepthReset(u32);

impl Drop for DepthReset {
    fn drop(&mut self) {
        PARSE_DEPTH.with(|d| d.set(self.0));
    }
}

/// How many operand terms a pending compound term still expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Plan {
    /// Exactly `n` operand terms, followed by the closing `)`.
    Fixed(u8),
    /// Operand terms until the closing `)`.
    Variadic,
}

/// The outer bindings a binder shadowed, restored when it closes.
#[derive(Debug, Default)]
struct SavedScope {
    bindings: Vec<(String, TermId)>,
    constants: Vec<(String, SortId)>,
}

/// A `(let ((x e) ...) body)` in progress.
///
/// `let` is the one form whose operands are not all parsed in the same scope:
/// the binding values belong to the enclosing scope and the body to the
/// extended one, so the frame tracks which half it is in.
#[derive(Debug, Default)]
struct LetFrame {
    /// Bindings whose value has already been parsed.
    bindings: Vec<(String, TermId)>,
    /// Name of the binding whose value is currently being parsed.
    pending: Option<String>,
    /// Outer bindings shadowed by this `let`.
    saved: Vec<(String, TermId)>,
    /// Whether the binding list is closed and the body is being parsed.
    in_body: bool,
    /// The parsed body, once available.
    body: Option<TermId>,
}

/// A `(! t :attr ...)` annotation in progress.
///
/// Annotations get their own frame because an attribute *value* can itself be a
/// list of terms (`:pattern (f x)`), so the attribute grammar sits between two
/// term positions. Parsing those nested terms with a recursive call would
/// reintroduce exactly the native recursion this module exists to avoid.
#[derive(Debug, Default)]
struct AnnotFrame {
    /// The annotated term, once its operand has been parsed.
    term: Option<TermId>,
    /// Attributes read so far.
    attrs: Vec<Attribute>,
    /// Key of the attribute whose `( ... )` value is currently being collected.
    pending_key: Option<String>,
    /// Elements of that value collected so far.
    sexpr: Vec<AttributeValue>,
    /// Whether an s-expression attribute value is currently open.
    in_sexpr: bool,
}

/// Everything a compound term's builder needs that was determined when its head
/// was read, before any operand existed.
#[derive(Debug)]
enum Head {
    /// A built-in operator dispatched by name; see `build::operand_plan`.
    Builtin(String),
    /// A floating-point operator whose head carries a rounding mode.
    FpRounded { op: String, rm: RmOperand },
    /// `((_ name i ...) args)`, or the flattened `(_ name i ... args)`.
    Indexed { name: String, indices: Vec<String> },
    /// An indexed floating-point conversion, whose head carries both indices
    /// and a rounding-mode symbol: `((_ to_fp eb sb) RM x)` and friends.
    IndexedFpConv {
        name: String,
        indices: Vec<String>,
        rm: RmOperand,
    },
    /// The flattened bit-vector extract spelling `(_ extract i j x)`.
    BvExtractFlat { high: u32, low: u32 },
    /// A datatype tester, `((_ is ctor) x)` or `(_ is ctor) x`.
    DtTester { ctor: String },
    /// A qualified identifier, `((as name Sort) args)`.
    Qualified { name: String, sort: SortId },
    /// `forall` (`universal`) or `exists`, with its variables already bound.
    Binder {
        universal: bool,
        vars: Vec<(String, SortId)>,
        saved: SavedScope,
    },
    /// An application of a `define-fun` definition, expanded by substitution.
    DefinedFun(String),
    /// An applied datatype constructor.
    DtConstructor { name: String, sort: SortId },
    /// An applied datatype selector.
    DtSelector { name: String, sort: SortId },
    /// An application of a symbol declared by `declare-fun`.
    DeclaredFun { name: String, ret: SortId },
    /// The Bool-sorted uninterpreted-application fallback.
    GenericApply(String),
}

/// One entry of the driver's explicit stack: a compound term whose head has
/// been read and whose operands are still being collected.
#[derive(Debug)]
enum Frame {
    Op {
        head: Head,
        plan: Plan,
        args: SmallVec<[TermId; 4]>,
    },
    Let(LetFrame),
    Annot(AnnotFrame),
}

impl Frame {
    fn op(head: Head, plan: Plan) -> Self {
        Frame::Op {
            head,
            plan,
            args: SmallVec::new(),
        }
    }
}

/// The result of reading a compound term's head: either the head was a complete
/// term by itself (`(_ bv5 8)`, `(_ NaN 8 24)`, ...) or it opened a frame that
/// now needs operands.
enum Opened {
    Value(TermId),
    Frame(Frame),
}

/// Symbol namespaces reserved by the SMT-LIB theories OxiZ parses.
///
/// SMT-LIB reserves theory operator names, so an *undeclared* symbol in one of
/// these namespaces is never a user-introduced function: it is either a typo or
/// a theory operator OxiZ has not implemented. Either way it must be reported
/// instead of being minted as an unconstrained uninterpreted function, which
/// would let the solver answer confidently and wrongly (e.g. `(not (str.< "abc"
/// "abd"))` reported `sat`). Declarations are always consulted first, so a
/// script that genuinely declares such a name keeps working.
fn is_reserved_theory_symbol(name: &str) -> bool {
    const RESERVED_PREFIXES: [&str; 7] = ["str.", "re.", "seq.", "char.", "fp.", "int.", "bv"];
    RESERVED_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
        || matches!(name, "nat2bv" | "int2bv" | "ubv2int" | "sbv2int")
}

impl Parser<'_> {
    /// Parse a term.
    ///
    /// Runs the explicit frame stack described in the module documentation: a
    /// single loop that alternates between reading the next operand and handing
    /// finished operands back to the innermost pending compound term. Native
    /// stack usage is constant in the nesting depth of the input.
    pub fn parse_term(&mut self) -> Result<TermId> {
        let outer = PARSE_DEPTH.with(Cell::get);
        let _restore = DepthReset(outer);
        let mut frames: Vec<Frame> = Vec::new();

        loop {
            // --- read one operand ------------------------------------------
            let depth = outer.saturating_add(frame_depth(&frames)).saturating_add(1);
            if depth > MAX_PARSE_DEPTH {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "term nesting too deep".to_string(),
                });
            }
            PARSE_DEPTH.with(|d| d.set(depth));

            let token = self
                .lexer
                .next_token()
                .ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input".to_string(),
                })?;

            let mut value = match token.kind {
                TokenKind::LParen => match self.open_compound()? {
                    Opened::Value(term) => term,
                    Opened::Frame(frame) => {
                        frames.push(frame);
                        continue;
                    }
                },
                _ => self.parse_leaf(token)?,
            };

            // --- hand it to the innermost pending frame --------------------
            loop {
                PARSE_DEPTH.with(|d| d.set(outer.saturating_add(frame_depth(&frames))));
                let Some(top) = frames.last_mut() else {
                    return Ok(value);
                };
                if !self.accept_operand(top, value)? {
                    break;
                }
                let Some(frame) = frames.pop() else {
                    break;
                };
                value = self.close_frame(frame)?;
            }
        }
    }

    /// Charge `extra` further levels of term nesting against the same budget
    /// as syntactic nesting, and fail with the usual "term nesting too deep"
    /// error when they do not fit.
    ///
    /// Called from [`Parser::build_variadic`](super::build) just before it
    /// folds an n-ary operator into a binary chain. At that point the driver
    /// has already set [`PARSE_DEPTH`] to the depth of the node being closed,
    /// and `extra` is the depth of the chain that will hang below it — so the
    /// sum is the depth of the term about to be built, which is exactly what
    /// [`MAX_PARSE_DEPTH`] promises to bound.
    pub(super) fn charge_fold_depth(&self, extra: u32) -> Result<()> {
        let depth = PARSE_DEPTH.with(Cell::get).saturating_add(extra);
        if depth > MAX_PARSE_DEPTH {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "term nesting too deep".to_string(),
            });
        }
        Ok(())
    }

    /// Charge the depth of an already-built term that is about to be
    /// installed under `what` as a *name binding*, against the same
    /// [`MAX_PARSE_DEPTH`] budget as syntactic nesting.
    ///
    /// A nullary `(define-fun a () Int <body>)` inlines `<body>` into
    /// `self.bindings`, and [`Parser::parse_symbol`] then hands that `TermId`
    /// straight back wherever `a` occurs — it is a *substitution*, not a
    /// variable reference. So a chain
    ///
    /// ```text
    /// (define-fun a0 () Int 0)
    /// (define-fun a1 () Int (+ a0 a0))
    /// (define-fun a2 () Int (+ a1 a1))
    /// ...
    /// ```
    ///
    /// adds one level of term depth *per command* while every individual
    /// command's parse stays two parens deep. The per-parse depth counter
    /// cannot see that: it is reset at each command, so a 100 000-command
    /// chain sailed through `MAX_PARSE_DEPTH` and handed a 100 000-deep term
    /// to whatever consumed it next. This closes the hole the same way
    /// [`Parser::charge_fold_depth`] closes the n-ary-fold hole: by charging
    /// the depth of the term that is actually being built, so the error is an
    /// honest `ParseError` here rather than a stack death downstream.
    pub(super) fn charge_binding_depth(&self, what: &str, term: TermId) -> Result<()> {
        let used = PARSE_DEPTH.with(Cell::get);
        let budget = MAX_PARSE_DEPTH.saturating_sub(used);
        if self.term_depth_exceeds(term, budget) {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!(
                    "term nesting too deep: the definition inlined for '{what}' nests more \
                     than {MAX_PARSE_DEPTH} levels deep (definitions that reference earlier \
                     definitions accumulate depth even though each command is short)"
                ),
            });
        }
        Ok(())
    }

    /// Whether `root`'s depth exceeds `budget` levels.
    ///
    /// Iterative, and memoized on `TermId` so a shared subterm of the
    /// hash-consed DAG is measured once rather than re-expanded per
    /// occurrence — without the memo, `(+ a a)` chains would take exponential
    /// time. Aborts as soon as any node's depth passes `budget`, so the cost
    /// of the check on an over-deep input is bounded by the budget rather
    /// than by the input.
    fn term_depth_exceeds(&self, root: TermId, budget: u32) -> bool {
        use crate::ast::traversal::get_children;

        /// One step of the post-order depth walk.
        enum Step {
            /// Schedule this node's children, then its own `Exit`.
            Enter(TermId),
            /// All children are measured; combine them into this node's depth.
            Exit(TermId),
        }

        let mut memo: FxHashMap<TermId, u32> = FxHashMap::default();
        let mut stack = vec![Step::Enter(root)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(id) => {
                    if memo.contains_key(&id) {
                        continue;
                    }
                    let children = match self.manager.get(id) {
                        Some(term) => get_children(&term.kind),
                        // A dangling id has no structure to descend into;
                        // treat it as a leaf rather than guessing a depth.
                        None => SmallVec::new(),
                    };
                    if children.is_empty() {
                        memo.insert(id, 1);
                        continue;
                    }
                    stack.push(Step::Exit(id));
                    for &child in &children {
                        stack.push(Step::Enter(child));
                    }
                }
                Step::Exit(id) => {
                    let children = match self.manager.get(id) {
                        Some(term) => get_children(&term.kind),
                        None => SmallVec::new(),
                    };
                    let mut deepest = 0u32;
                    for child in &children {
                        deepest = deepest.max(memo.get(child).copied().unwrap_or(0));
                    }
                    let depth = deepest.saturating_add(1);
                    if depth > budget {
                        return true;
                    }
                    memo.insert(id, depth);
                }
            }
        }
        false
    }

    /// Parse a non-`(` token into a leaf term.
    fn parse_leaf(&mut self, token: Token) -> Result<TermId> {
        match token.kind {
            TokenKind::Symbol(s) => self.parse_symbol(&s),
            TokenKind::Numeral(n) => {
                let value: BigInt = n.parse().map_err(|_| OxizError::ParseError {
                    position: token.start,
                    message: format!("invalid numeral: {n}"),
                })?;
                Ok(self.manager.mk_int(value))
            }
            TokenKind::Hexadecimal(h) => {
                let value =
                    BigInt::parse_bytes(h.as_bytes(), 16).ok_or_else(|| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid hexadecimal: {h}"),
                    })?;
                let width = (h.len() * 4) as u32;
                Ok(self.manager.mk_bitvec(value, width))
            }
            TokenKind::Binary(b) => {
                let value =
                    BigInt::parse_bytes(b.as_bytes(), 2).ok_or_else(|| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid binary: {b}"),
                    })?;
                let width = b.len() as u32;
                Ok(self.manager.mk_bitvec(value, width))
            }
            TokenKind::Decimal(d) => {
                let rational =
                    parse_decimal_to_rational(&d).map_err(|e| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid decimal: {d} - {e}"),
                    })?;
                Ok(self.manager.mk_real(rational))
            }
            TokenKind::StringLit(s) => Ok(self.manager.mk_string_lit(&s)),
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("unexpected token: {:?}", token.kind),
            }),
        }
    }

    /// Deliver a finished operand to `frame`.
    ///
    /// Returns `true` when the frame has everything it needs (the driver then
    /// pops and closes it) and `false` when another operand must be parsed
    /// first.
    fn accept_operand(&mut self, frame: &mut Frame, value: TermId) -> Result<bool> {
        match frame {
            Frame::Op { plan, args, .. } => {
                args.push(value);
                let complete = match *plan {
                    Plan::Fixed(n) => {
                        if args.len() >= usize::from(n) {
                            self.expect_rparen()?;
                            true
                        } else {
                            false
                        }
                    }
                    Plan::Variadic => self.try_consume_rparen(),
                };
                Ok(complete)
            }
            Frame::Let(state) => {
                if state.in_body {
                    self.expect_rparen()?; // closes `(let ...)`
                    state.body = Some(value);
                    return Ok(true);
                }
                self.expect_rparen()?; // closes the `(name value)` pair
                let name = state.pending.take().ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "internal: let binding without a name".to_string(),
                })?;
                state.bindings.push((name, value));
                if self.try_consume_rparen() {
                    // The binding list is closed; the bindings come into scope
                    // for the body only.
                    let saved: Vec<(String, TermId)> = state
                        .bindings
                        .iter()
                        .filter_map(|(name, _)| self.bindings.get(name).map(|&t| (name.clone(), t)))
                        .collect();
                    state.saved = saved;
                    for (name, term) in &state.bindings {
                        self.bindings.insert(name.clone(), *term);
                    }
                    state.in_body = true;
                } else {
                    self.expect_lparen()?;
                    state.pending = Some(self.expect_symbol()?);
                }
                Ok(false)
            }
            Frame::Annot(state) => {
                if state.term.is_none() {
                    state.term = Some(value);
                } else {
                    state.sexpr.push(AttributeValue::Term(value));
                }
                let needs_term = self.advance_annotation(state)?;
                Ok(!needs_term)
            }
        }
    }

    /// Consume as much of an annotation's attribute list as possible without a
    /// term.
    ///
    /// Returns `true` when a term is needed next (the next element of an
    /// s-expression attribute value) and `false` when the whole annotation —
    /// including its closing `)` — has been consumed.
    fn advance_annotation(&mut self, state: &mut AnnotFrame) -> Result<bool> {
        loop {
            if state.in_sexpr {
                if !self.try_consume_rparen() {
                    return Ok(true);
                }
                state.in_sexpr = false;
                let values = core::mem::take(&mut state.sexpr);
                let key = state
                    .pending_key
                    .take()
                    .ok_or_else(|| OxizError::ParseError {
                        position: self.lexer.position(),
                        message: "internal: attribute value without a keyword".to_string(),
                    })?;
                state.attrs.push(Attribute {
                    key,
                    value: Some(AttributeValue::SExpr(values)),
                });
                continue;
            }

            let Some(token) = self.lexer.peek() else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input in annotation".to_string(),
                });
            };
            if matches!(token.kind, TokenKind::RParen) {
                self.lexer.next_token(); // closes `(! ...)`
                return Ok(false);
            }
            // Attributes start with a keyword (e.g. `:named`, `:pattern`).
            let TokenKind::Keyword(key) = &token.kind else {
                return Err(OxizError::ParseError {
                    position: token.start,
                    message: format!("expected keyword in annotation, found {:?}", token.kind),
                });
            };
            let key = key.clone();
            self.lexer.next_token(); // consume the keyword

            match self.lexer.peek().map(|t| t.kind) {
                // A keyword or `)` next means this attribute has no value.
                None | Some(TokenKind::Keyword(_)) | Some(TokenKind::RParen) => {
                    state.attrs.push(Attribute { key, value: None });
                }
                // `( ... )`: a list whose elements are terms (`:pattern`).
                // Collect them through the driver rather than recursing.
                Some(TokenKind::LParen) => {
                    self.lexer.next_token(); // consume `(`
                    state.pending_key = Some(key);
                    state.in_sexpr = true;
                    state.sexpr.clear();
                }
                Some(_) => {
                    let value = self.parse_simple_attribute_value()?;
                    state.attrs.push(Attribute {
                        key,
                        value: Some(value),
                    });
                }
            }
        }
    }

    /// Build the term of a frame whose operands are all present.
    fn close_frame(&mut self, frame: Frame) -> Result<TermId> {
        match frame {
            Frame::Op { head, plan, args } => self.close_op(head, plan, &args),
            Frame::Let(state) => {
                let LetFrame {
                    bindings,
                    saved,
                    body,
                    ..
                } = state;
                let body = body.ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "internal: let closed without a body".to_string(),
                })?;
                for (name, _) in &bindings {
                    self.bindings.remove(name);
                }
                for (name, term) in saved {
                    self.bindings.insert(name, term);
                }
                // Return the body ALONE — deliberately not `mk_let(bindings, body)`.
                //
                // This parser resolves a `let`-bound name by *substitution*, not by
                // reference: `open_let_frame` inserts `name -> TermId` into
                // `self.bindings` before the body is parsed, and `parse_symbol`
                // hands that `TermId` straight back at every occurrence (see the
                // `self.bindings.get(s)` arm). By the time the frame closes, the
                // body already *is* the fully expanded term; the value of every
                // binding is physically present inside it and the bound name
                // occurs nowhere. A `Let` node wrapped around that body therefore
                // binds nothing — it is pure decoration.
                //
                // Decoration with teeth, though: `TermKind::Let` is a real
                // capture-avoiding binder everywhere else in the AST layer
                // (`substitute.rs`, `alpha.rs`, `structural.rs`), so every
                // consumer that walks a term treats it as an opaque scope and
                // refuses to descend. `collect_ground_subterms`
                // (`oxiz-solver/src/solver/encode/bool_euf_encoding.rs`) lists it
                // alongside `Forall`/`Exists` for exactly that reason. The result
                // was that a single vacuous wrapper silently disabled every
                // assert-time pre-pass — `eliminate_nonbool_ite`,
                // `abstract_compound_bool_args`, `flatten_lookup_spines`,
                // `purify_numeric_uf_args`, the quantified model gate and the
                // arithmetic-disequality split — for the whole assertion
                // beneath it.
                // Two formulas differing only by an *unused* `(let ((q 0)) ...)`
                // wrapper got different verdicts, one of them a wrong `sat`.
                //
                // Only the parser is changed here. `TermManager::mk_let` keeps
                // building a genuine `Let` for programmatic construction, where
                // the body may really contain free occurrences of the bound name.
                Ok(body)
            }
            Frame::Annot(state) => {
                let AnnotFrame { term, attrs, .. } = state;
                let term = term.ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "internal: annotation closed without a term".to_string(),
                })?;
                if !attrs.is_empty() {
                    self.annotations.insert(term, attrs);
                }
                Ok(term)
            }
        }
    }

    /// Build the term of a completed operator application.
    fn close_op(&mut self, head: Head, plan: Plan, args: &[TermId]) -> Result<TermId> {
        match head {
            Head::Builtin(op) => match (plan, args) {
                (Plan::Fixed(1), [x]) => self.build_unary(&op, *x),
                (Plan::Fixed(2), [x, y]) => self.build_binary(&op, *x, *y),
                (Plan::Fixed(3), [x, y, z]) => self.build_ternary(&op, *x, *y, *z),
                (Plan::Variadic, _) => self.build_variadic(&op, args),
                _ => Err(self.operand_mismatch(&op, args.len())),
            },
            Head::FpRounded { op, rm } => self.build_fp_rounded(&op, rm, args),
            Head::Indexed { name, indices } => {
                if let Some(term) = self.build_indexed_op(&name, &indices, args)? {
                    return Ok(term);
                }
                // An indexed identifier can only come from a theory, never from
                // a user declaration, so an unrecognised one is an unimplemented
                // operator; reject it rather than minting an unconstrained
                // uninterpreted function.
                let func_name = indexed_display_name(&name, &indices);
                self.reject_unknown_symbol(&func_name, &name)?;
                let sort = self.manager.sorts.bool_sort;
                Ok(self
                    .manager
                    .mk_apply(&func_name, args.iter().copied(), sort))
            }
            Head::IndexedFpConv { name, indices, rm } => {
                let [arg] = args else {
                    return Err(self.operand_mismatch(&name, args.len()));
                };
                self.build_indexed_fp_conv(&name, &indices, rm, *arg)
            }
            Head::BvExtractFlat { high, low } => {
                let [arg] = args else {
                    return Err(self.operand_mismatch("extract", args.len()));
                };
                Ok(self.manager.mk_bv_extract(high, low, *arg))
            }
            Head::DtTester { ctor } => {
                let [arg] = args else {
                    return Err(self.operand_mismatch("(_ is)", args.len()));
                };
                Ok(self.manager.mk_dt_tester(&ctor, *arg))
            }
            Head::Qualified { name, sort } => {
                // For known forms like `(as const (Array D R))` we represent the
                // qualified application as an `Apply` node whose function name
                // records the qualifier and whose sort is the annotated one.
                let func_name = format!("(as {name})");
                Ok(self
                    .manager
                    .mk_apply(&func_name, args.iter().copied(), sort))
            }
            Head::Binder {
                universal,
                vars,
                saved,
            } => {
                let [body] = args else {
                    return Err(self.operand_mismatch("binder", args.len()));
                };
                Ok(self.close_binder(universal, &vars, saved, *body))
            }
            Head::DefinedFun(name) => self.expand_defined_fun(&name, args),
            Head::DtConstructor { name, sort } => {
                Ok(self
                    .manager
                    .mk_dt_constructor(&name, args.iter().copied(), sort))
            }
            Head::DtSelector { name, sort } => {
                let [arg] = args else {
                    return Err(self.operand_mismatch(&name, args.len()));
                };
                Ok(self.manager.mk_dt_selector(&name, *arg, sort))
            }
            Head::DeclaredFun { name, ret } => {
                Ok(self.manager.mk_apply(&name, args.iter().copied(), ret))
            }
            Head::GenericApply(name) => {
                let sort = self.manager.sorts.bool_sort;
                Ok(self.manager.mk_apply(&name, args.iter().copied(), sort))
            }
        }
    }

    /// Error for an operand count the plan should have made impossible.
    fn operand_mismatch(&self, op: &str, got: usize) -> OxizError {
        OxizError::ParseError {
            position: self.lexer.position(),
            message: format!("wrong number of arguments for {op}: got {got}"),
        }
    }

    /// Expand an application of a `define-fun` definition by substituting the
    /// arguments for the parameters in the recorded body.
    fn expand_defined_fun(&mut self, name: &str, args: &[TermId]) -> Result<TermId> {
        let Some(macro_def) = self.function_defs.get(name).cloned() else {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("internal: no definition recorded for {name}"),
            });
        };
        if args.len() != macro_def.formal_vars.len() {
            return Err(OxizError::ParseError {
                position: 0,
                message: format!(
                    "wrong number of arguments for {}: expected {}, got {}",
                    name,
                    macro_def.formal_vars.len(),
                    args.len()
                ),
            });
        }
        // Substitute directly by the formal parameters' own `TermId`s, taken
        // verbatim from where they were bound while the body was parsed. No
        // sort has to be looked up or guessed here, so there is no way for
        // this map's keys to miss the occurrences actually in the body — see
        // `FunctionMacro::formal_vars`'s doc comment for the failure mode
        // this sidesteps.
        let mut substitution = FxHashMap::default();
        for (&formal, &actual) in macro_def.formal_vars.iter().zip(args.iter()) {
            substitution.insert(formal, actual);
        }
        Ok(self.manager.substitute(macro_def.expansion, &substitution))
    }

    /// Restore the scope a binder shadowed and build the quantifier.
    fn close_binder(
        &mut self,
        universal: bool,
        vars: &[(String, SortId)],
        saved: SavedScope,
        body: TermId,
    ) -> TermId {
        for (name, _) in vars {
            self.bindings.remove(name);
        }
        for (name, term) in saved.bindings {
            self.bindings.insert(name, term);
        }
        for (name, sort) in saved.constants {
            self.constants.insert(name, sort);
        }
        let body = self.relativize_rounding_mode_binders(universal, vars, body);
        let var_refs: Vec<(&str, SortId)> = vars.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        if universal {
            self.manager.mk_forall(var_refs, body)
        } else {
            self.manager.mk_exists(var_refs, body)
        }
    }

    /// Restrict a quantifier's `RoundingMode`-sorted variables to the five
    /// real rounding modes, syntactically, before the binder is built.
    ///
    /// The solver's closure axiom (`oxiz-solver`'s `context::rounding_mode`)
    /// is asserted per *declared constant*; a bound variable has no
    /// declaration to hang it on, and asserting a closure about a bound name
    /// at top level would be nonsense. Relativizing the body is the standard
    /// answer and the only sound one here: without it, `(forall ((m
    /// RoundingMode)) phi)` would quantify over an unconstrained free sort —
    /// strictly *stronger* than the intended "for all five modes", so a
    /// formula true of every real mode could still come back `unsat`; and
    /// `(exists ((m RoundingMode)) phi)` would be strictly *weaker*, able to
    /// witness `phi` with a sixth value that is no rounding mode at all.
    ///
    /// - `forall`: `body` becomes `(=> (and closure(v1) ... ) body)`
    /// - `exists`: `body` becomes `(and closure(v1) ... body)`
    ///
    /// where `closure(v)` is `(or (= v RNE) (= v RNA) (= v RTP) (= v RTN)
    /// (= v RTZ))`. Non-`RoundingMode` binders are untouched and the body is
    /// returned unchanged, so this costs nothing for every other quantifier.
    ///
    /// The modes' pairwise *distinctness* is still the solver's job; a
    /// relativized `forall` is sound without it, and the solver asserts it
    /// whenever any rounding mode is in play.
    fn relativize_rounding_mode_binders(
        &mut self,
        universal: bool,
        vars: &[(String, SortId)],
        body: TermId,
    ) -> TermId {
        let rm_sort = self.manager.sorts.rounding_mode_sort;
        let bound: Vec<String> = vars
            .iter()
            .filter(|(_, sort)| *sort == rm_sort)
            .map(|(name, _)| name.clone())
            .collect();
        if bound.is_empty() {
            return body;
        }

        let mut guards: Vec<TermId> = Vec::with_capacity(bound.len());
        for name in &bound {
            let var = self.manager.mk_var(name, rm_sort);
            let alternatives: Vec<TermId> = RoundingMode::ALL
                .iter()
                .map(|&rm| {
                    let mode = self.manager.mk_rounding_mode(rm);
                    self.manager.mk_eq(var, mode)
                })
                .collect();
            guards.push(self.manager.mk_or(alternatives));
        }

        if universal {
            let guard = self.manager.mk_and(guards);
            self.manager.mk_implies(guard, body)
        } else {
            guards.push(body);
            self.manager.mk_and(guards)
        }
    }

    /// Read the head of a compound term, just after its `(` was consumed.
    fn open_compound(&mut self) -> Result<Opened> {
        let op_token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "unexpected end of input".to_string(),
            })?;

        // A head that is itself an S-expression: an indexed identifier
        // `((_ to_fp 8 24) RNE 1.5)` or a qualified one
        // `((as const (Array Int Int)) 0)`.
        if matches!(op_token.kind, TokenKind::LParen) {
            return self.open_sexpr_head();
        }
        // The flattened indexed spelling `(_ extract 7 4 x)`.
        if matches!(op_token.kind, TokenKind::Symbol(ref s) if s == "_") {
            return self.open_flat_indexed_head();
        }

        let op = match &op_token.kind {
            TokenKind::Symbol(s) => s.clone(),
            TokenKind::Keyword(k) => format!(":{k}"),
            _ => {
                return Err(OxizError::ParseError {
                    position: op_token.start,
                    message: format!("expected operator, found {:?}", op_token.kind),
                });
            }
        };
        self.open_named_head(op)
    }

    /// Head of the form `((as name Sort) ...)` or `((_ name i ...) ...)`.
    fn open_sexpr_head(&mut self) -> Result<Opened> {
        let qualifier = self.expect_symbol()?;
        if qualifier == "as" {
            // SMT-LIB qualified identifier: `(as <symbol> <sort>)`.
            let symbol = self.expect_symbol()?;
            let sort = self.parse_sort()?;
            self.expect_rparen()?; // closes the `(as ...)` form
            return Ok(Opened::Frame(Frame::op(
                Head::Qualified { name: symbol, sort },
                Plan::Variadic,
            )));
        }
        if qualifier != "_" {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("expected '_' or 'as' in compound operator, found '{qualifier}'"),
            });
        }

        let name = self.expect_symbol()?;
        let indices = self.parse_indices_closing()?;

        if name == "is" {
            // Datatype tester: `((_ is constructor) arg)`.
            let [ctor] = indices.as_slice() else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!(
                        "(_ is) requires exactly 1 constructor name, got {}",
                        indices.len()
                    ),
                });
            };
            return Ok(Opened::Frame(Frame::op(
                Head::DtTester { ctor: ctor.clone() },
                Plan::Fixed(1),
            )));
        }

        // Indexed FP conversions take a leading rounding-mode *symbol*, which
        // is not itself a term, so the head has to consume it now.
        if let Some(rm) = self.open_indexed_fp_conv(&name, &indices)? {
            return Ok(Opened::Frame(Frame::op(
                Head::IndexedFpConv { name, indices, rm },
                Plan::Fixed(1),
            )));
        }

        Ok(Opened::Frame(Frame::op(
            Head::Indexed { name, indices },
            Plan::Variadic,
        )))
    }

    /// Collect the index parts of an `(_ name i ...)` head, consuming the `)`
    /// that closes it.
    fn parse_indices_closing(&mut self) -> Result<Vec<String>> {
        let mut indices = Vec::new();
        while let Some(token) = self.lexer.peek() {
            match &token.kind {
                TokenKind::RParen => {
                    self.lexer.next_token();
                    break;
                }
                TokenKind::Numeral(n) => {
                    let n = n.clone();
                    self.lexer.next_token();
                    indices.push(n);
                }
                // For datatype testers like `(_ is nil)` the constructor name
                // is a symbol rather than a numeral.
                TokenKind::Symbol(s) => {
                    let s = s.clone();
                    self.lexer.next_token();
                    indices.push(s);
                }
                _ => {
                    return Err(OxizError::ParseError {
                        position: token.start,
                        message: format!(
                            "expected numeral, symbol, or ')' in indexed identifier, found {:?}",
                            token.kind
                        ),
                    });
                }
            }
        }
        Ok(indices)
    }

    /// Head of the flattened form `(_ name i ... args)`, just after the `_`.
    fn open_flat_indexed_head(&mut self) -> Result<Opened> {
        let name = self.expect_symbol()?;

        // Indices, leaving the closing `)` in place.
        let mut numeric: Vec<u32> = Vec::new();
        let mut indices: Vec<String> = Vec::new();
        while let Some(token) = self.lexer.peek() {
            match &token.kind {
                TokenKind::RParen => break,
                TokenKind::Numeral(n) => {
                    let n = n.clone();
                    let start = token.start;
                    self.lexer.next_token();
                    let idx = n.parse::<u32>().map_err(|_| OxizError::ParseError {
                        position: start,
                        message: format!("invalid index: {n}"),
                    })?;
                    numeric.push(idx);
                    indices.push(n);
                }
                TokenKind::Symbol(s) => {
                    let s = s.clone();
                    self.lexer.next_token();
                    indices.push(s);
                }
                _ => break,
            }
        }

        // `(_ bvN M)` is a bit-vector literal with value N and width M.
        if let Some(bv_val_str) = name.strip_prefix("bv") {
            let [width] = numeric.as_slice() else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!(
                        "bitvector literal (_ {name} ...) requires exactly 1 index (width), got {}",
                        numeric.len()
                    ),
                });
            };
            // SMT-LIB places no bound on the literal's magnitude (only the
            // *width* `M` is capped below), so the value must be parsed with
            // arbitrary precision rather than truncated to `i64`.
            let value: BigInt = bv_val_str.parse().map_err(|_| OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("invalid bitvector literal value: {bv_val_str}"),
            })?;
            let width = *width;
            if width == 0 || width > 65536 {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!("invalid bitvector width: {width} (must be 1-65536)"),
                });
            }
            self.expect_rparen()?;
            return Ok(Opened::Value(self.manager.mk_bitvec(value, width)));
        }

        // Indexed floating-point special-value constants. Like `(_ bvN M)`
        // these are complete nullary terms, so no operands follow.
        if matches!(name.as_str(), "+oo" | "-oo" | "+zero" | "-zero" | "NaN") {
            let [eb, sb] = numeric.as_slice() else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!(
                        "(_ {name} eb sb) requires exactly 2 indices, got {}",
                        numeric.len()
                    ),
                });
            };
            let (eb, sb) = (*eb, *sb);
            self.expect_rparen()?;
            let term = match name.as_str() {
                "+oo" => self.manager.mk_fp_plus_infinity(eb, sb),
                "-oo" => self.manager.mk_fp_minus_infinity(eb, sb),
                "+zero" => self.manager.mk_fp_plus_zero(eb, sb),
                "-zero" => self.manager.mk_fp_minus_zero(eb, sb),
                _ => self.manager.mk_fp_nan(eb, sb),
            };
            return Ok(Opened::Value(term));
        }

        match name.as_str() {
            "extract" => {
                let [high, low] = numeric.as_slice() else {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "extract requires exactly 2 indices, got {}",
                            numeric.len()
                        ),
                    });
                };
                Ok(Opened::Frame(Frame::op(
                    Head::BvExtractFlat {
                        high: *high,
                        low: *low,
                    },
                    Plan::Fixed(1),
                )))
            }
            "is" => {
                let [ctor] = indices.as_slice() else {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "(_ is) requires exactly 1 constructor name, got {}",
                            indices.len()
                        ),
                    });
                };
                let ctor = ctor.clone();
                self.expect_rparen()?; // closes the `(_ is X)` head
                Ok(Opened::Frame(Frame::op(
                    Head::DtTester { ctor },
                    Plan::Fixed(1),
                )))
            }
            _ => {
                // For unrecognized indexed identifiers, consume the closing
                // paren of the `(_ name indices)` head and treat whatever
                // follows as the argument list.
                self.expect_rparen()?;
                Ok(Opened::Frame(Frame::op(
                    Head::Indexed { name, indices },
                    Plan::Variadic,
                )))
            }
        }
    }

    /// Head that is a plain operator or symbol.
    fn open_named_head(&mut self, op: String) -> Result<Opened> {
        // Forms whose head carries syntax of its own.
        match op.as_str() {
            "!" => return Ok(Opened::Frame(Frame::Annot(AnnotFrame::default()))),
            "let" => return self.open_let(),
            "forall" => return self.open_binder(true),
            "exists" => return self.open_binder(false),
            // Floating-point operators taking a leading rounding mode.
            "fp.add" | "fp.sub" | "fp.mul" | "fp.div" => {
                let rm = self.parse_rounding_mode_operand()?;
                return Ok(Opened::Frame(Frame::op(
                    Head::FpRounded { op, rm },
                    Plan::Fixed(2),
                )));
            }
            "fp.sqrt" | "fp.roundToIntegral" => {
                let rm = self.parse_rounding_mode_operand()?;
                return Ok(Opened::Frame(Frame::op(
                    Head::FpRounded { op, rm },
                    Plan::Fixed(1),
                )));
            }
            "fp.fma" => {
                let rm = self.parse_rounding_mode_operand()?;
                return Ok(Opened::Frame(Frame::op(
                    Head::FpRounded { op, rm },
                    Plan::Fixed(3),
                )));
            }
            _ => {}
        }

        // Plain built-in operators.
        if let Some(plan) = operand_plan(&op) {
            return Ok(Opened::Frame(Frame::op(Head::Builtin(op), plan)));
        }

        // User symbols, resolved against every declaration table in turn.
        if self.function_defs.contains_key(&op) {
            return Ok(Opened::Frame(Frame::op(
                Head::DefinedFun(op),
                Plan::Variadic,
            )));
        }
        if let Some(&sort) = self.dt_constructors.get(&op) {
            // Applied datatype constructor, e.g. `(cons 1 nil)`.
            return Ok(Opened::Frame(Frame::op(
                Head::DtConstructor { name: op, sort },
                Plan::Variadic,
            )));
        }
        if let Some(&sort) = self.dt_selectors.get(&op) {
            // Applied datatype selector, e.g. `(head l)`. Building a real
            // `DtSelector` (rather than an uninterpreted apply) is what gives
            // the term its declared result sort, so `(= (head l) 10)` reaches
            // the arithmetic theory.
            return Ok(Opened::Frame(Frame::op(
                Head::DtSelector { name: op, sort },
                Plan::Fixed(1),
            )));
        }
        if let Some(&(_, ret)) = self.functions.get(&op) {
            // Declared uninterpreted function. The declared return sort is
            // essential for theory reasoning: without it an expression like
            // `(> (f k) 10)` would build `f(k)` with `Bool` sort and the
            // arithmetic theory would ignore it.
            return Ok(Opened::Frame(Frame::op(
                Head::DeclaredFun { name: op, ret },
                Plan::Variadic,
            )));
        }
        if self.constants.contains_key(&op) || self.bindings.contains_key(&op) {
            // A symbol that *was* declared (or `let`-bound) but is used in head
            // position. That is a sort error rather than an unknown symbol;
            // keep the permissive behavior and let the sort checker speak.
            return Ok(Opened::Frame(Frame::op(
                Head::GenericApply(op),
                Plan::Variadic,
            )));
        }
        // Undeclared head symbol. See `reject_unknown_symbol` for the rule;
        // when it declines to reject (bare-term mode, non-reserved name) we
        // keep the historical uninterpreted-application fallback.
        self.reject_unknown_symbol(&op, &op)?;
        Ok(Opened::Frame(Frame::op(
            Head::GenericApply(op),
            Plan::Variadic,
        )))
    }

    /// Open a `(let ((x e) ...) body)` frame, consuming the head of the
    /// binding list up to the point where the first binding's value begins.
    fn open_let(&mut self) -> Result<Opened> {
        self.expect_lparen()?; // opens the binding list
        let mut state = LetFrame::default();
        if self.try_consume_rparen() {
            // An empty binding list: nothing comes into scope, the body is next.
            state.in_body = true;
        } else {
            self.expect_lparen()?; // opens the first `(name value)` pair
            state.pending = Some(self.expect_symbol()?);
        }
        Ok(Opened::Frame(Frame::Let(state)))
    }

    /// Open a `forall` / `exists` frame, binding the quantified variables so
    /// that body references resolve with their declared sorts. Without this a
    /// bound variable like `i` would fall through to the default
    /// `mk_var(name, bool_sort)` path.
    fn open_binder(&mut self, universal: bool) -> Result<Opened> {
        self.expect_lparen()?;
        let vars = self.parse_sorted_vars()?;

        let saved = SavedScope {
            bindings: vars
                .iter()
                .filter_map(|(name, _)| self.bindings.get(name).map(|&t| (name.clone(), t)))
                .collect(),
            constants: vars
                .iter()
                .filter_map(|(name, _)| self.constants.get(name).map(|&s| (name.clone(), s)))
                .collect(),
        };
        for (name, sort) in &vars {
            let var_term = self.manager.mk_var(name, *sort);
            self.bindings.insert(name.clone(), var_term);
            // Remove from constants to avoid shadowing issues.
            self.constants.remove(name);
        }

        Ok(Opened::Frame(Frame::op(
            Head::Binder {
                universal,
                vars,
                saved,
            },
            Plan::Fixed(1),
        )))
    }

    pub(super) fn parse_symbol(&mut self, s: &str) -> Result<TermId> {
        match s {
            "true" => Ok(self.manager.mk_true()),
            "false" => Ok(self.manager.mk_false()),
            // Nullary regular-expression constants (SMT-LIB Strings theory).
            // These are leaf symbols (not compound applications), so they must
            // be recognised here, ahead of the strict "unknown symbol" reject
            // in the `_` arm below.
            "re.none" => Ok(self.manager.mk_re_none()),
            "re.all" => Ok(self.manager.mk_re_all()),
            "re.allchar" => Ok(self.manager.mk_re_all_char()),
            // The five rounding modes of the SMT-LIB `FloatingPoint` theory,
            // in both their short and long spellings. Both spellings of a mode
            // route through `mk_rounding_mode`, which interns under the
            // canonical *long* name, so `RNE` and `roundNearestTiesToEven` are
            // literally the same `TermId`.
            //
            // These sit in the top match, ahead of the declaration tables and
            // ahead of the strict unknown-symbol reject in the `_` arm: the
            // names are SMT-LIB-reserved, so no user declaration may shadow
            // them, and in script mode the `_` arm errors out before any
            // reserved-name handling could run.
            "RNE" | "roundNearestTiesToEven" => {
                Ok(self.manager.mk_rounding_mode(RoundingMode::RNE))
            }
            "RNA" | "roundNearestTiesToAway" => {
                Ok(self.manager.mk_rounding_mode(RoundingMode::RNA))
            }
            "RTP" | "roundTowardPositive" => Ok(self.manager.mk_rounding_mode(RoundingMode::RTP)),
            "RTN" | "roundTowardNegative" => Ok(self.manager.mk_rounding_mode(RoundingMode::RTN)),
            "RTZ" | "roundTowardZero" => Ok(self.manager.mk_rounding_mode(RoundingMode::RTZ)),
            _ => {
                // Check bindings first
                if let Some(&term) = self.bindings.get(s) {
                    return Ok(term);
                }
                // Check if this is a datatype constructor (e.g., Monday, nil, cons, etc.)
                if let Some(&dt_sort) = self.dt_constructors.get(s) {
                    return Ok(self.manager.mk_dt_constructor(s, vec![], dt_sort));
                }
                // Check constants
                if let Some(&sort) = self.constants.get(s) {
                    return Ok(self.manager.mk_var(s, sort));
                }
                // Z3-compatible shortcut: the SMT-LIB 2 grammar does not treat
                // `-3` or `-3.0` as a numeric literal (they parse as symbols
                // because `-` is a valid symbol char), but Z3 accepts them as
                // negative numbers.  If the symbol matches the pattern of a
                // negative numeral or decimal and has not been bound to
                // anything else, interpret it as such — otherwise arithmetic
                // constraints like `(* -3.0 x)` silently become nonsense
                // boolean-sorted variables.
                if let Some(rest) = s.strip_prefix('-')
                    && !rest.is_empty()
                    && rest.chars().next().is_some_and(|c| c.is_ascii_digit())
                {
                    if let Some(dot_idx) = rest.find('.') {
                        let (int_part, frac_part) = rest.split_at(dot_idx);
                        let frac_part = &frac_part[1..];
                        if int_part.chars().all(|c| c.is_ascii_digit())
                            && !frac_part.is_empty()
                            && frac_part.chars().all(|c| c.is_ascii_digit())
                        {
                            let rational =
                                super::parse_decimal_to_rational(rest).map_err(|_| {
                                    OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!("invalid negative decimal: {s}"),
                                    }
                                })?;
                            return Ok(self.manager.mk_real(-rational));
                        }
                    } else if rest.chars().all(|c| c.is_ascii_digit()) {
                        let value: BigInt = rest.parse().map_err(|_| OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!("invalid negative numeral: {s}"),
                        })?;
                        return Ok(self.manager.mk_int(-value));
                    }
                }
                // At this point `s` is not a bound variable, a datatype
                // constructor, a declared constant, or a negative numeric
                // literal. In a genuine SMT-LIB script every symbol must be
                // declared before use, so an unknown symbol here is a typo or
                // a missing `declare-const`/`declare-fun`. Silently minting a
                // fresh Bool-sorted variable (the old behavior) makes such a
                // script solve a *different* problem and can report `sat` with
                // a meaningless model. Reject it, matching Z3's "unknown
                // constant" error.
                //
                // The one exception is the bare `parse_term` convenience path
                // used to build ad-hoc terms with no declarations at all: when
                // parsing an isolated term (not a script) we stay lenient so
                // that free variables can still be constructed.
                if self.script_mode {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!("unknown constant or symbol: {s}"),
                    });
                }
                // Even in bare-term mode a reserved theory name (`str.foo`,
                // `re.bar`, ...) is never a legitimate free variable, so it is
                // rejected too — see `Parser::reject_unknown_symbol`.
                self.reject_unknown_symbol(s, s)?;
                // Lenient fallback (bare-term mode): boolean variable.
                let sort = self.manager.sorts.bool_sort;
                Ok(self.manager.mk_var(s, sort))
            }
        }
    }

    /// Decide whether an *undeclared* symbol used in head position must be
    /// rejected, and reject it if so.
    ///
    /// The rule, applied only after every declaration table (`define-fun`,
    /// datatype constructors and selectors, `declare-fun`, `declare-const`,
    /// `let` bindings) has already been consulted, so a symbol the user really
    /// declared always wins and `QF_UF` is unaffected:
    ///
    /// 1. In **script mode** (`parse_script`, or an embedder that seeded a real
    ///    context) every symbol must be declared before use, exactly as in Z3.
    ///    Any undeclared head symbol is an error — the same rule
    ///    [`Parser::parse_symbol`] already applies to undeclared *nullary*
    ///    symbols.
    /// 2. In **bare-term mode** (`parse_term`, which intentionally allows
    ///    free variables in an ad-hoc fragment) an undeclared symbol is still
    ///    an error when it lives in a reserved SMT-LIB theory namespace — no
    ///    such name can ever be a user-introduced function, so it is either a
    ///    typo or an operator OxiZ has not implemented.
    ///
    /// `display` is the name to quote in the message; `namespace_key` is the
    /// bare symbol used for the reserved-namespace test (they differ for
    /// indexed identifiers, where `display` is the whole `(_ f i)` form).
    ///
    /// Returning `Ok(())` means the caller may keep the historical
    /// uninterpreted-application fallback.
    fn reject_unknown_symbol(&self, display: &str, namespace_key: &str) -> Result<()> {
        if self.script_mode {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("unknown function/constant {display}"),
            });
        }
        if is_reserved_theory_symbol(namespace_key) {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!(
                    "unknown theory operator {display}: the symbol is in a reserved SMT-LIB \
                     theory namespace but OxiZ does not implement it"
                ),
            });
        }
        Ok(())
    }

    /// Returns the bit-vector width of `term`, if it has a bit-vector sort.
    pub(super) fn bv_width(&self, term: TermId) -> Option<u32> {
        let sort = self.manager.get(term)?.sort;
        self.manager.sorts.get(sort)?.bitvec_width()
    }

    /// If the next token is a closing `)`, consume it and return `true`;
    /// otherwise leave the stream untouched and return `false`.
    pub(super) fn try_consume_rparen(&mut self) -> bool {
        if let Some(token) = self.lexer.peek()
            && matches!(token.kind, TokenKind::RParen)
        {
            self.lexer.next_token();
            return true;
        }
        false
    }

    /// Read an attribute value that cannot contain a term.
    ///
    /// The `( ... )` form is *not* handled here: its elements are terms, so
    /// [`Parser::advance_annotation`] collects them through the driver's frame
    /// stack instead of recursing.
    fn parse_simple_attribute_value(&mut self) -> Result<AttributeValue> {
        let token = self.lexer.peek().ok_or_else(|| OxizError::ParseError {
            position: self.lexer.position(),
            message: "unexpected end of input in attribute value".to_string(),
        })?;

        match &token.kind {
            TokenKind::Symbol(s) => {
                let s = s.clone();
                self.lexer.next_token();
                Ok(AttributeValue::Symbol(s))
            }
            TokenKind::Numeral(n) => {
                let n = n.clone();
                self.lexer.next_token();
                Ok(AttributeValue::Numeral(n))
            }
            TokenKind::StringLit(s) => {
                let s = s.clone();
                self.lexer.next_token();
                Ok(AttributeValue::String(s))
            }
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("unexpected token in attribute value: {:?}", token.kind),
            }),
        }
    }
}

/// The nesting contributed by the frames currently on the driver's stack,
/// saturating rather than wrapping on the (unreachable, given
/// [`MAX_PARSE_DEPTH`]) 4-billion-frame case.
fn frame_depth(frames: &[Frame]) -> u32 {
    u32::try_from(frames.len()).unwrap_or(u32::MAX)
}

/// Spell an indexed identifier the way it is recorded in a generic `Apply`.
fn indexed_display_name(name: &str, indices: &[String]) -> String {
    if indices.is_empty() {
        format!("(_ {name})")
    } else {
        format!("(_ {name} {})", indices.join(" "))
    }
}
