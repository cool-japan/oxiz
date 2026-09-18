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

// The single encoder that writes a *symbol* back as SMT-LIB source text, and
// therefore the one place the simple-symbol character set of section 3.1 is
// stated.  Re-exported for the same reason as `format_string_literal` above:
// `oxiz-solver`'s model printer holds declaration names as plain strings and
// has no `TermManager` to route them through, and a second copy of the rule is
// how `(get-model)` came to print `|a b|` without its bars while `(get-value)`
// printed it with them.
pub use printer::{format_symbol, is_simple_symbol};

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

/// The one prefix every solver-internal symbol carries.
///
/// A solver that mints a fresh symbol and then *asserts something about it* —
/// a purification proxy tied to its argument by `v = arg`, a Skolem constant
/// standing for an existential witness, a datatype size measure carrying the
/// well-founded-ordering lemmas — has made that symbol's name part of its
/// soundness argument.  `TermManager::mk_var` interns on `(name, sort)`, so a
/// user declaration of the same name at the same sort *is* the minted symbol
/// and inherits its side conditions; a formula in which the declared constant
/// is free then answers `unsat`.  Eight lines of plain SMT-LIB were enough
/// while the encoder proxies were spelled `$encode-numarg!{id}` — `$`, `!`,
/// `-`, letters and digits are all in SMT-LIB 2.6 section 3.1's simple-symbol
/// character set, so the name was spellable.
///
/// The backslash is what makes the whole class unspellable, and it is
/// unspellable in both of SMT-LIB 2.6's symbol forms at once: a *simple*
/// symbol's character set excludes `\`, and a *quoted* symbol "may not contain
/// `|` or `\`" — the lexer rejects one outright.  Reserving by this single
/// prefix, rather than by a list of names, is what makes a *new* mint site
/// safe by construction: [`reserved_name`] is the only way to spell one, and
/// the parser's reserved-symbol check is one `starts_with`
/// against this constant.
///
/// [`CONST_ARRAY_FUNC`], [`ARRAY_EXT_WITNESS_PREFIX`] and
/// [`ARRAY_OFF_CHAIN_PREFIX`] are the three members that predate the helper
/// and are spelled out in full because they are matched by name elsewhere;
/// every one of them begins with this prefix.
pub const RESERVED_PREFIX: &str = "\\oxiz.";

/// Mint the name of a solver-internal symbol: `\oxiz.<tag>!<suffix>`.
///
/// The single constructor for the reserved class documented at
/// [`RESERVED_PREFIX`].  `tag` names the mint site (`numarg`, `sk`, `dtsize`,
/// …) and `suffix` distinguishes the instances that site produces — a term id,
/// a counter, a pair of ids.  Neither may contain a backslash of its own; they
/// never do, because every caller builds them from integers.
///
/// Call this rather than `format!`-ing a name by hand: a site that forgets the
/// prefix reintroduces the capture hole, and this is the function that makes
/// forgetting impossible to do accidentally.
///
/// ```
/// # use oxiz_core::smtlib::{RESERVED_PREFIX, reserved_name};
/// let name = reserved_name("numarg", "17");
/// assert_eq!(name, "\\oxiz.numarg!17");
/// assert!(name.starts_with(RESERVED_PREFIX));
/// ```
#[must_use]
pub fn reserved_name(tag: &str, suffix: &str) -> String {
    let mut name = String::with_capacity(RESERVED_PREFIX.len() + tag.len() + 1 + suffix.len());
    name.push_str(RESERVED_PREFIX);
    name.push_str(tag);
    name.push('!');
    name.push_str(suffix);
    name
}

/// Does `name` belong to the reserved family [`reserved_name`] mints for `tag`?
///
/// The read-side twin of [`reserved_name`], and the only supported way to ask
/// the question: a site that instead tested `name.starts_with("sk")` — as the
/// MBQI candidate filters did — is a filter that silently stops matching the
/// moment a mint site is respelled, and one that matches a *user's* `skew` in
/// the meantime.  Allocation-free, so it is usable on the hot walk.
///
/// ```
/// # use oxiz_core::smtlib::{is_reserved_tag, reserved_name};
/// let sk = reserved_name("sk", "0");
/// assert!(is_reserved_tag(&sk, "sk"));
/// assert!(!is_reserved_tag(&sk, "skf"));
/// assert!(!is_reserved_tag("skew", "sk"));
/// ```
#[must_use]
pub fn is_reserved_tag(name: &str, tag: &str) -> bool {
    name.strip_prefix(RESERVED_PREFIX)
        .and_then(|rest| rest.strip_prefix(tag))
        .is_some_and(|rest| rest.starts_with('!'))
}
