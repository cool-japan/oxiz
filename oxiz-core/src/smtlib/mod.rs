//! SMT-LIB2 Parser and Printer
//!
//! This module provides parsing and printing for the SMT-LIB2 standard format.

#[allow(unused_imports)]
use crate::prelude::*;

mod lexer;
mod parser;
mod printer;

pub use lexer::{Lexer, Token, TokenKind};
pub use parser::{Command, RecFunDecl, parse_script, parse_term};
pub use printer::Printer;

// The single encoder that turns a string *value* back into SMT-LIB source
// text.  Re-exported here so `model::Value`'s `Display` shares it with the
// term printers instead of keeping its own copy of the escape rules.
//
// `pub` (not `pub(crate)`): reused outside `oxiz-core` too, by any site in
// the workspace that emits SMT-LIB-shaped text containing a string value —
// e.g. `(error "...")` responses in `oxiz-solver`/`oxiz-cli` and proof
// metadata in `oxiz-core::smtlib::printer::proof`, so a `"` or control
// character in user-supplied text cannot break the surrounding syntax.
pub use printer::format_string_literal;

// The single encoder that turns a bit-vector *value* back into SMT-LIB source
// text, and therefore the one place the `#x`-iff-`width % 4 == 0` radix rule
// is stated.  Re-exported for the same reason as `format_string_literal`
// above: `oxiz-solver` renders bit-vector values on paths that hold no
// `TermManager` (`Context::default_value`), and a second copy of the rule is
// exactly how the two spellings of finding U-Z13 came about.
pub use printer::format_bitvec_literal;

/// The interned function symbol the parser gives the SMT-LIB array constant
/// `((as const (Array D R)) d)`.
///
/// There is no dedicated `TermKind` for an array constant: `parser::terms`
/// (`Head::Qualified`) turns the qualified identifier into an ordinary
/// uninterpreted application, and the *name* of that application is the only
/// record that it is an array constant rather than a user function.  The name
/// therefore has to be one no SMT-LIB script can spell, or a script that
/// declares the same symbol makes the two indistinguishable — a read of the
/// user's function would be decided by the array-constant axiom and a
/// satisfiable formula would answer `unsat`.
///
/// A backslash is what makes it unspellable, and it is unspellable in both
/// of SMT-LIB 2.6's symbol forms at once: a *simple* symbol's character set
/// (section 3.1) excludes `\`, and a *quoted* symbol "may not contain `|` or
/// `\`" — the lexer rejects one outright ([`Lexer`]), so the two together
/// leave no way to write this name.
///
/// Printing is the mirror image: the term printers special-case exactly this
/// name back to `((as const (Array D R)) d)` using the application's own
/// sort, so the reserved spelling never reaches a user-visible response.
pub const CONST_ARRAY_FUNC: &str = "\\oxiz.as-const";

/// Name prefix of the *extensionality witness index* the array theory mints
/// for an unordered pair of array terms (`oxiz-solver`'s
/// `solver::array_axioms`).
///
/// Reserved for the same reason as [`CONST_ARRAY_FUNC`] and by the same
/// mechanism — the backslash — but against a different failure: the witness is
/// a fresh index *variable*, and `TermManager::mk_var` interns on
/// `(name, sort)`, so a user declaration of the same name at the index sort
/// **is** the same term.  The lemmas this index appears in
/// (`a = b ∨ select(a,k) != select(b,k)`) are valid for every index, so a
/// collision here costs precision rather than soundness; it is reserved
/// anyway, because "harmless today" is not a property to leave resting on the
/// shape of the current lemma set.
pub const ARRAY_EXT_WITNESS_PREFIX: &str = "\\oxiz.ext!";

/// Name prefix of the *off-chain Skolem index* the array theory mints for an
/// unordered pair of array terms whose store chains bottom out at different
/// arrays (`oxiz-solver`'s `solver::array_axioms`).
///
/// This one is reserved for soundness, not tidiness.  The rule asserts
/// `d != i_k` for every store index `i_k` of the pair's chains — a constraint
/// on the *symbol itself*, satisfiable only because the symbol is fresh.  A
/// script that declares the same name at the index sort interns the same term
/// and inherits those constraints, which is a wrong `unsat` on a formula in
/// which the declared constant is free: six lines of plain SMT-LIB were enough
/// while the prefix was `!oxiz!off!` (`!` is a legal simple-symbol character,
/// so the name was spellable).  The backslash is what makes it unspellable in
/// both SMT-LIB 2.6 symbol forms at once.
pub const ARRAY_OFF_CHAIN_PREFIX: &str = "\\oxiz.off!";
