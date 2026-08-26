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
