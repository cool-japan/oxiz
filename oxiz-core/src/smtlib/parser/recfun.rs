//! `define-fun-rec` / `define-funs-rec` parsing.
//!
//! # Why this is not just another `define-fun`
//!
//! A `define-fun` is a *macro*: the parser records its body and substitutes it
//! at every call site ([`Parser::expand_defined_fun`]). Doing that for a
//! recursive definition would not terminate — `f`'s body mentions `f`, so each
//! expansion produces a fresh call site to expand.
//!
//! So a recursive definition is handled the other way round. The function is
//! registered as an ordinary *declared* symbol — in [`Parser::functions`] when
//! it takes arguments, in [`Parser::constants`] when it is nullary — **before
//! its body is read**. That single ordering trick is what makes the whole
//! feature work:
//!
//! * the self-reference inside the body resolves through the normal declared-
//!   symbol path and builds a real `Apply` (or `Var`) node with the declared
//!   result sort, instead of tripping the strict undeclared-symbol rejection;
//! * every later call site builds the *same* node, so the solver sees one
//!   uninterpreted symbol it can constrain;
//! * nothing is ever inserted into [`Parser::bindings`] or
//!   [`Parser::function_defs`], the two inlining paths — an inlined recursive
//!   body would loop forever.
//!
//! For `define-funs-rec` the same ordering is applied group-wide: *all*
//! signatures are registered before *any* body is parsed, which is exactly what
//! makes mutual recursion (`is-even` / `is-odd`) parse.
//!
//! The bodies leave here inside [`Command::DefineFunsRec`]; the definitional
//! axiom `forall x. f(x) = body` is discharged by the solver, by fuel-bounded
//! unfolding.

use super::{Parser, RecFunDecl};
use crate::ast::TermId;
use crate::error::{OxizError, Result};
use crate::sort::SortId;

/// A parsed signature, held between the signature pass and the body pass of a
/// `define-funs-rec` group.
struct RecSignature {
    /// The function's name.
    name: String,
    /// `(name, sort-string)` per formal parameter, in declaration order.
    params: Vec<(String, String)>,
    /// The already-resolved sort of each formal parameter, parallel to
    /// `params`. Kept so the body pass can mint each parameter's `Var` with the
    /// sort the signature actually declared, rather than re-resolving the
    /// stringified sort (which cannot round-trip a compound sort such as
    /// `(Array Int Int)` — see the same note on `define-fun` in `commands.rs`).
    param_sort_ids: Vec<SortId>,
    /// The declared return sort, stringified.
    ret_sort: String,
}

impl Parser<'_> {
    /// Parse `(define-fun-rec name ((p S) ..) R body)`, with the opening paren
    /// and the command name already consumed.
    pub(super) fn parse_define_fun_rec(&mut self) -> Result<RecFunDecl> {
        let signature = self.parse_rec_signature()?;
        let decl = self.parse_rec_body(signature)?;
        self.expect_rparen()?;
        self.rec_function_defs
            .insert(decl.name.clone(), decl.clone());
        Ok(decl)
    }

    /// Parse
    /// `(define-funs-rec ((f1 ((p S) ..) R) ..) (body1 ..))`,
    /// with the opening paren and the command name already consumed.
    ///
    /// Every signature is registered before any body is read, so a body may
    /// call any function of the group — including one declared after it. That
    /// is the whole point of the two-list syntax.
    pub(super) fn parse_define_funs_rec(&mut self) -> Result<Vec<RecFunDecl>> {
        // ---- pass 1: the signature list -------------------------------
        self.expect_lparen()?;
        let mut signatures = Vec::new();
        loop {
            if self.try_consume_rparen() {
                break;
            }
            self.expect_lparen()?;
            let signature = self.parse_rec_signature()?;
            self.expect_rparen()?;
            signatures.push(signature);
        }

        if signatures.is_empty() {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "define-funs-rec: the function declaration list is empty".to_string(),
            });
        }

        // ---- pass 2: the body list ------------------------------------
        self.expect_lparen()?;
        let mut decls = Vec::with_capacity(signatures.len());
        for signature in signatures {
            if self.try_consume_rparen() {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!(
                        "define-funs-rec: fewer bodies than function declarations ({} declared, \
                         {} bodies)",
                        decls.len() + 1,
                        decls.len()
                    ),
                });
            }
            decls.push(self.parse_rec_body(signature)?);
        }
        if !self.try_consume_rparen() {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!(
                    "define-funs-rec: more bodies than function declarations ({} declared)",
                    decls.len()
                ),
            });
        }
        self.expect_rparen()?;

        for decl in &decls {
            self.rec_function_defs
                .insert(decl.name.clone(), decl.clone());
        }
        Ok(decls)
    }

    /// Parse `name ((p S) ..) R` and **register the function before returning**.
    ///
    /// The registration has to happen here, ahead of the body pass, for the
    /// reason spelled out in this module's doc comment: it is what lets the
    /// body's own recursive calls resolve. An arity-0 recursive function is
    /// legal SMT-LIB (`(define-fun-rec c () Int (+ c 1))`), and goes into
    /// `constants` so that its self-reference resolves as a `Var` — the solver
    /// then saturates it at a single instance.
    fn parse_rec_signature(&mut self) -> Result<RecSignature> {
        let name = self.expect_symbol()?;
        self.expect_lparen()?;

        let mut params: Vec<(String, String)> = Vec::new();
        let mut param_sort_ids: Vec<SortId> = Vec::new();
        loop {
            if self.try_consume_rparen() {
                break;
            }
            self.expect_lparen()?;
            let param_name = self.expect_symbol()?;
            let param_sort_id = self.resolve_sort()?;
            let param_sort = self.sort_id_to_string(param_sort_id);
            self.expect_rparen()?;
            params.push((param_name, param_sort));
            param_sort_ids.push(param_sort_id);
        }

        let ret_sort_id = self.resolve_sort()?;
        let ret_sort = self.sort_id_to_string(ret_sort_id);

        if param_sort_ids.is_empty() {
            self.constants.insert(name.clone(), ret_sort_id);
        } else {
            // Same restriction `declare-fun` applies: the closure axiom that
            // gives `RoundingMode` its five-element domain attaches only to a
            // nullary declaration, so accepting the sort here would silently
            // treat it as an infinite free sort.
            for &arg_sort_id in &param_sort_ids {
                self.reject_unsupported_rounding_mode_position(
                    arg_sort_id,
                    "a function argument sort",
                )?;
            }
            self.reject_unsupported_rounding_mode_position(
                ret_sort_id,
                "the result sort of a function with arguments",
            )?;
            self.functions
                .insert(name.clone(), (param_sort_ids.clone(), ret_sort_id));
        }

        Ok(RecSignature {
            name,
            params,
            param_sort_ids,
            ret_sort,
        })
    }

    /// Bind the formal parameters, parse exactly one body term, then restore the
    /// scope the parameters shadowed.
    ///
    /// The save/restore mirrors `define-fun`'s in `commands.rs`: a parameter
    /// named like an outer `let` binding must shadow it for the body and only
    /// for the body.
    fn parse_rec_body(&mut self, signature: RecSignature) -> Result<RecFunDecl> {
        let RecSignature {
            name,
            params,
            param_sort_ids,
            ret_sort,
        } = signature;

        let old_bindings: Vec<(String, TermId)> = params
            .iter()
            .filter_map(|(pname, _)| self.bindings.get(pname).map(|&t| (pname.clone(), t)))
            .collect();

        // Mint each parameter's `Var` with the sort the signature resolved, and
        // keep the resulting `TermId`s: they are the only sound substitution
        // keys for the body (see `RecFunDecl::formal_vars`).
        let mut formal_vars: Vec<TermId> = Vec::with_capacity(params.len());
        for ((pname, _psort), &sort_id) in params.iter().zip(param_sort_ids.iter()) {
            let param_term = self.manager.mk_var(pname, sort_id);
            self.bindings.insert(pname.clone(), param_term);
            formal_vars.push(param_term);
        }

        let body = self.parse_term()?;

        for (pname, _) in &params {
            self.bindings.remove(pname);
        }
        for (pname, term) in old_bindings {
            self.bindings.insert(pname, term);
        }

        Ok(RecFunDecl {
            name,
            params,
            formal_vars,
            ret_sort,
            body,
        })
    }
}
