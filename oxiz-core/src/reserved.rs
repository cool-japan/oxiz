//! The reserved-symbol helpers, in the one module that is compiled with and
//! without the `std` feature.
//!
//! This is the `no_std`-reachable home of `ARRAY_EXT_WITNESS_PREFIX`,
//! `ARRAY_OFF_CHAIN_PREFIX`, `RESERVED_PREFIX`, `reserved_name` and
//! `is_reserved_tag`.  They are defined here, and not in `smtlib`, because
//! `smtlib` is `std`-only while `ast` -- which mints its Skolem constants
//! through `reserved_name` -- is not.  `smtlib` re-exports all five, so every
//! `oxiz_core::smtlib::<item>` path is unchanged and each definition (and each
//! string constant) still exists exactly once.
//!
//! Only `RESERVED_PREFIX` and `reserved_name` are used without `std`.  The other
//! three are compiled with `std` only: their one way out of this crate is the
//! `smtlib` re-export, and a `pub` item of a private module that a `no_std`
//! build never reaches is dead code there.

#[cfg(not(feature = "std"))]
use crate::prelude::String;

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
///
/// [`CONST_ARRAY_FUNC`]: crate::smtlib::CONST_ARRAY_FUNC
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
///
/// [`CONST_ARRAY_FUNC`]: crate::smtlib::CONST_ARRAY_FUNC
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
#[cfg(feature = "std")]
#[must_use]
pub fn is_reserved_tag(name: &str, tag: &str) -> bool {
    name.strip_prefix(RESERVED_PREFIX)
        .and_then(|rest| rest.strip_prefix(tag))
        .is_some_and(|rest| rest.starts_with('!'))
}
