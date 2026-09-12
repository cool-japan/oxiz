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
