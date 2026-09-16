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
