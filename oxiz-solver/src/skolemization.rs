//! Skolemization: transforms existential quantifiers by replacing
//! existentially quantified variables with Skolem constants or functions.
//!
//! For `exists x. phi(x)` with no outer universals: replace x with a fresh constant c,
//! yielding `phi(c)`.
//!
//! For `forall y. exists x. phi(x, y)` with outer universals: replace x with `f(y)`
//! where f is a fresh function symbol (Skolem function).
//!
//! This module performs NNF conversion first, then Skolemization, ensuring that
//! quantifier polarities are correctly determined before replacement.

use oxiz_core::ast::TermManager;
use oxiz_core::ast::{TermId, TermKind};
use oxiz_core::sort::SortId;
use std::collections::HashMap;
use std::fmt;

/// Errors that can occur during Skolemization
#[derive(Debug, Clone)]
pub enum SkolemizationError {
    /// A term ID could not be resolved in the TermManager
    UnknownTerm(TermId),
    /// Sort information could not be retrieved
    UnknownSort(SortId),
    /// The Skolem counter overflowed (extremely unlikely)
    CounterOverflow,
    /// Internal error during term construction
    TermConstructionFailed(String),
}

impl fmt::Display for SkolemizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkolemizationError::UnknownTerm(id) => {
                write!(f, "unknown term with id {}", id.raw())
            }
            SkolemizationError::UnknownSort(id) => {
                write!(f, "unknown sort with id {}", id.0)
            }
            SkolemizationError::CounterOverflow => {
                write!(f, "Skolem counter overflow")
            }
            SkolemizationError::TermConstructionFailed(msg) => {
                write!(f, "term construction failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for SkolemizationError {}

/// Represents a generated Skolem symbol (constant or function)
#[derive(Debug, Clone)]
pub struct SkolemSymbol {
    /// The generated name (e.g., "sk!0", "skf!1")
    pub name: String,
    /// The result sort of the Skolem constant or function
    pub sort: SortId,
    /// The term created for this Skolem symbol
    pub term: TermId,
    /// For Skolem functions, the sorts of the arguments (universal variables).
    /// Empty for Skolem constants.
    pub arg_sorts: Vec<SortId>,
}

/// Context that tracks state during Skolemization.
///
/// Maintains the stack of outer universal variables, the mapping from
/// existential variables to their Skolem replacements, and a counter
/// for generating unique Skolem names.
#[derive(Debug)]
pub struct SkolemizationContext {
    /// Stack of outer universal variables: (sort, term_id) pairs.
    /// When we enter a Forall scope, the bound variables are pushed here;
    /// when we leave, they are popped.
    outer_universals: Vec<(SortId, TermId)>,
    /// Map from original existential variable TermId to its Skolem replacement TermId.
    skolem_map: HashMap<TermId, TermId>,
    /// Counter for generating unique Skolem names
    skolem_counter: u64,
    /// All generated Skolem symbols, for inspection/tracking
    skolem_symbols: Vec<SkolemSymbol>,
    /// Cache for NNF conversion: (term_id, negated) -> result_id
    nnf_cache: HashMap<(TermId, bool), TermId>,
    /// Cache for skolemize_inner to avoid reprocessing subterms
    skolem_cache: HashMap<TermId, TermId>,
}

impl Default for SkolemizationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SkolemizationContext {
    /// Create a new Skolemization context
    pub fn new() -> Self {
        SkolemizationContext {
            outer_universals: Vec::new(),
            skolem_map: HashMap::new(),
            skolem_counter: 0,
            skolem_symbols: Vec::new(),
            nnf_cache: HashMap::new(),
            skolem_cache: HashMap::new(),
        }
    }

    /// Get the list of generated Skolem symbols
    pub fn skolem_symbols(&self) -> &[SkolemSymbol] {
        &self.skolem_symbols
    }

    /// Get the number of Skolem symbols generated so far
    pub fn skolem_count(&self) -> u64 {
        self.skolem_counter
    }

    /// Main entry point: Skolemize a term, returning the transformed term.
    ///
    /// This performs NNF conversion first (to ensure quantifier polarities are
    /// correctly determined), then Skolemization.
    ///
    /// # Errors
    ///
    /// Returns `SkolemizationError` if terms cannot be looked up, sorts are
    /// missing, or the Skolem counter overflows.
    pub fn skolemize(
        &mut self,
        tm: &mut TermManager,
        term: TermId,
    ) -> Result<TermId, SkolemizationError> {
        // 1. Convert to NNF (push negations inward so quantifier polarities are clear)
        let nnf = self.convert_nnf(tm, term, false)?;
        // 2. Skolemize the NNF term
        self.skolemize_inner(tm, nnf)
    }

    /// Convert to Negation Normal Form.
    ///
    /// When `negated` is true, we are converting under a negation context, which
    /// flips AND/OR and swaps Forall/Exists.
    fn convert_nnf(
        &mut self,
        tm: &mut TermManager,
        term: TermId,
        negated: bool,
    ) -> Result<TermId, SkolemizationError> {
        // Check cache
        if let Some(&cached) = self.nnf_cache.get(&(term, negated)) {
            return Ok(cached);
        }

        let t = tm.get(term).ok_or(SkolemizationError::UnknownTerm(term))?;
        let kind = t.kind.clone();

        let result = match kind {
            TermKind::True => {
                if negated {
                    Ok(tm.mk_false())
                } else {
                    Ok(tm.mk_true())
                }
            }
            TermKind::False => {
                if negated {
                    Ok(tm.mk_true())
                } else {
                    Ok(tm.mk_false())
                }
            }
            TermKind::Var(_)
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => {
                if negated {
                    Ok(tm.mk_not(term))
                } else {
                    Ok(term)
                }
            }
            TermKind::Not(arg) => {
                // Double negation elimination: push negation through
                self.convert_nnf(tm, arg, !negated)
            }
            TermKind::And(args) => {
                let mut nnf_args = Vec::with_capacity(args.len());
                for &a in &args {
                    nnf_args.push(self.convert_nnf(tm, a, negated)?);
                }
                if negated {
                    // De Morgan: NOT(a AND b) = (NOT a) OR (NOT b)
                    Ok(tm.mk_or(nnf_args))
                } else {
                    Ok(tm.mk_and(nnf_args))
                }
            }
            TermKind::Or(args) => {
                let mut nnf_args = Vec::with_capacity(args.len());
                for &a in &args {
                    nnf_args.push(self.convert_nnf(tm, a, negated)?);
                }
                if negated {
                    // De Morgan: NOT(a OR b) = (NOT a) AND (NOT b)
                    Ok(tm.mk_and(nnf_args))
                } else {
                    Ok(tm.mk_or(nnf_args))
                }
            }
            TermKind::Implies(lhs, rhs) => {
                // (a -> b) = (NOT a) OR b
                // NOT(a -> b) = a AND (NOT b)
                let lhs_nnf = self.convert_nnf(tm, lhs, !negated)?;
                let rhs_nnf = self.convert_nnf(tm, rhs, negated)?;
                if negated {
                    Ok(tm.mk_and([lhs_nnf, rhs_nnf]))
                } else {
                    Ok(tm.mk_or([lhs_nnf, rhs_nnf]))
                }
            }
            TermKind::Xor(lhs, rhs) => {
                // a XOR b = (a OR b) AND (NOT a OR NOT b)
                let a = self.convert_nnf(tm, lhs, false)?;
                let b = self.convert_nnf(tm, rhs, false)?;
                let not_a = self.convert_nnf(tm, lhs, true)?;
                let not_b = self.convert_nnf(tm, rhs, true)?;
                let clause1 = tm.mk_or([a, b]);
                let clause2 = tm.mk_or([not_a, not_b]);
                let xor_nnf = tm.mk_and([clause1, clause2]);
                if negated {
                    // NOT(a XOR b) = (a AND b) OR (NOT a AND NOT b)
                    // Re-derive under negation to keep NNF:
                    // Simply negate the XOR NNF
                    self.convert_nnf(tm, xor_nnf, true)
                } else {
                    Ok(xor_nnf)
                }
            }
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                let body_nnf = self.convert_nnf(tm, body, negated)?;

                // Resolve variable names first to avoid borrowing issues
                let var_names: Vec<(String, SortId)> = vars
                    .iter()
                    .map(|(s, sort)| (tm.resolve_str(*s).to_string(), *sort))
                    .collect();

                if negated {
                    // NOT(forall x. P(x)) = exists x. NOT P(x)
                    Ok(tm.mk_exists_with_patterns(
                        var_names
                            .iter()
                            .map(|(s, sort): &(String, SortId)| (s.as_str(), *sort)),
                        body_nnf,
                        patterns,
                    ))
                } else {
                    Ok(tm.mk_forall_with_patterns(
                        var_names
                            .iter()
                            .map(|(s, sort): &(String, SortId)| (s.as_str(), *sort)),
                        body_nnf,
                        patterns,
                    ))
                }
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                let body_nnf = self.convert_nnf(tm, body, negated)?;

                // Resolve variable names first to avoid borrowing issues
                let var_names: Vec<(String, SortId)> = vars
                    .iter()
                    .map(|(s, sort)| (tm.resolve_str(*s).to_string(), *sort))
                    .collect();

                if negated {
                    // NOT(exists x. P(x)) = forall x. NOT P(x)
                    Ok(tm.mk_forall_with_patterns(
                        var_names
                            .iter()
                            .map(|(s, sort): &(String, SortId)| (s.as_str(), *sort)),
                        body_nnf,
                        patterns,
                    ))
                } else {
                    Ok(tm.mk_exists_with_patterns(
                        var_names
                            .iter()
                            .map(|(s, sort): &(String, SortId)| (s.as_str(), *sort)),
                        body_nnf,
                        patterns,
                    ))
                }
            }
            TermKind::Eq(_, _)
            | TermKind::Distinct(_)
            | TermKind::Lt(_, _)
            | TermKind::Le(_, _)
            | TermKind::Gt(_, _)
            | TermKind::Ge(_, _)
            | TermKind::Apply { .. }
            | TermKind::Ite(_, _, _) => {
                // Atoms and other non-boolean-connective terms: just negate if needed
                if negated {
                    Ok(tm.mk_not(term))
                } else {
                    Ok(term)
                }
            }
            // All remaining term kinds (arithmetic, bitvec, string ops, FP, etc.)
            // are treated as atoms in the boolean sense
            _ => {
                if negated {
                    Ok(tm.mk_not(term))
                } else {
                    Ok(term)
                }
            }
        }?;

        self.nnf_cache.insert((term, negated), result);
        Ok(result)
    }

    /// Inner recursive Skolemization.
    ///
    /// Traverses the NNF term tree:
    /// - Forall: pushes bound variables onto `outer_universals`, recurses on body, pops
    /// - Exists: for each bound variable, creates a Skolem constant or function,
    ///   adds mapping to `skolem_map`, recurses on body
    /// - Variable: if in `skolem_map`, returns the replacement; otherwise returns as-is
    /// - Other: recurses on children
    fn skolemize_inner(
        &mut self,
        tm: &mut TermManager,
        term: TermId,
    ) -> Result<TermId, SkolemizationError> {
        // Check cache
        if let Some(&cached) = self.skolem_cache.get(&term) {
            return Ok(cached);
        }

        let t = tm.get(term).ok_or(SkolemizationError::UnknownTerm(term))?;
        let kind = t.kind.clone();

        let result = match kind {
            // Base cases: constants are unchanged
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => Ok(term),

            // Variables: check if this var has a Skolem replacement
            TermKind::Var(_) => {
                if let Some(&replacement) = self.skolem_map.get(&term) {
                    Ok(replacement)
                } else {
                    Ok(term)
                }
            }

            // Universal quantifier: push vars onto outer_universals, recurse, pop
            TermKind::Forall {
                vars,
                body,
                patterns: _,
            } => {
                // Record how many universals we are pushing so we can pop exactly that many
                let push_count = vars.len();

                // Push each bound variable onto the outer_universals stack.
                // We need the actual TermIds for these variables so that Skolem functions
                // can reference them as arguments.
                for (name_spur, sort) in &vars {
                    let var_name = tm.resolve_str(*name_spur).to_string();
                    let var_id = tm.mk_var(&var_name, *sort);
                    self.outer_universals.push((*sort, var_id));
                }

                // Recurse on the body
                let sk_body = self.skolemize_inner(tm, body)?;

                // Pop the universal variables we pushed
                for _ in 0..push_count {
                    self.outer_universals.pop();
                }

                // Reconstruct the Forall with the Skolemized body (patterns are dropped
                // since Skolemization may have changed the variable structure)
                let var_names: Vec<(String, SortId)> = vars
                    .iter()
                    .map(|(s, sort)| (tm.resolve_str(*s).to_string(), *sort))
                    .collect();

                Ok(tm.mk_forall(
                    var_names
                        .iter()
                        .map(|(s, sort): &(String, SortId)| (s.as_str(), *sort)),
                    sk_body,
                ))
            }

            // Existential quantifier: create Skolem constants/functions for each bound var
            TermKind::Exists { vars, body, .. } => {
                // For each existentially quantified variable, create a Skolem replacement
                for (name_spur, sort) in &vars {
                    let var_name = tm.resolve_str(*name_spur).to_string();
                    let var_id = tm.mk_var(&var_name, *sort);

                    let skolem_term = if self.outer_universals.is_empty() {
                        // No outer universal variables: create a Skolem constant
                        self.mk_skolem_constant(tm, *sort)?
                    } else {
                        // There are outer universal variables: create a Skolem function
                        // applied to the outer universal variables
                        self.mk_skolem_function(tm, *sort)?
                    };

                    self.skolem_map.insert(var_id, skolem_term);
                }

                // Recurse on the body (the existential quantifier is eliminated)
                self.skolemize_inner(tm, body)
            }

            // Boolean connectives: recurse on children
            TermKind::Not(arg) => {
                let sk_arg = self.skolemize_inner(tm, arg)?;
                Ok(tm.mk_not(sk_arg))
            }
            TermKind::And(args) => {
                let mut sk_args = Vec::with_capacity(args.len());
                for &a in &args {
                    sk_args.push(self.skolemize_inner(tm, a)?);
                }
                Ok(tm.mk_and(sk_args))
            }
            TermKind::Or(args) => {
                let mut sk_args = Vec::with_capacity(args.len());
                for &a in &args {
                    sk_args.push(self.skolemize_inner(tm, a)?);
                }
                Ok(tm.mk_or(sk_args))
            }
            TermKind::Implies(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_implies(sk_lhs, sk_rhs))
            }
            TermKind::Xor(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_xor(sk_lhs, sk_rhs))
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let sk_cond = self.skolemize_inner(tm, cond)?;
                let sk_then = self.skolemize_inner(tm, then_br)?;
                let sk_else = self.skolemize_inner(tm, else_br)?;
                Ok(tm.mk_ite(sk_cond, sk_then, sk_else))
            }

            // Equality and comparison: recurse on children
            TermKind::Eq(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_eq(sk_lhs, sk_rhs))
            }
            TermKind::Distinct(args) => {
                let mut sk_args = Vec::with_capacity(args.len());
                for &a in &args {
                    sk_args.push(self.skolemize_inner(tm, a)?);
                }
                Ok(tm.mk_distinct(sk_args))
            }
            TermKind::Lt(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_lt(sk_lhs, sk_rhs))
            }
            TermKind::Le(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_le(sk_lhs, sk_rhs))
            }
            TermKind::Gt(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_gt(sk_lhs, sk_rhs))
            }
            TermKind::Ge(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_ge(sk_lhs, sk_rhs))
            }

            // Arithmetic: recurse on children
            TermKind::Neg(arg) => {
                let sk_arg = self.skolemize_inner(tm, arg)?;
                Ok(tm.mk_neg(sk_arg))
            }
            TermKind::Add(args) => {
                let mut sk_args = Vec::with_capacity(args.len());
                for &a in &args {
                    sk_args.push(self.skolemize_inner(tm, a)?);
                }
                Ok(tm.mk_add(sk_args))
            }
            TermKind::Sub(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_sub(sk_lhs, sk_rhs))
            }
            TermKind::Mul(args) => {
                let mut sk_args = Vec::with_capacity(args.len());
                for &a in &args {
                    sk_args.push(self.skolemize_inner(tm, a)?);
                }
                Ok(tm.mk_mul(sk_args))
            }
            TermKind::Div(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_div(sk_lhs, sk_rhs))
            }
            TermKind::Mod(lhs, rhs) => {
                let sk_lhs = self.skolemize_inner(tm, lhs)?;
                let sk_rhs = self.skolemize_inner(tm, rhs)?;
                Ok(tm.mk_mod(sk_lhs, sk_rhs))
            }

            // Uninterpreted function application: recurse on arguments
            TermKind::Apply { func, args } => {
                let mut sk_args = Vec::with_capacity(args.len());
                for &a in &args {
                    sk_args.push(self.skolemize_inner(tm, a)?);
                }
                let func_name = tm.resolve_str(func).to_string();
                let result_sort = tm
                    .get(term)
                    .ok_or(SkolemizationError::UnknownTerm(term))?
                    .sort;
                Ok(tm.mk_apply(&func_name, sk_args, result_sort))
            }

            // Array operations
            TermKind::Select(arr, idx) => {
                let sk_arr = self.skolemize_inner(tm, arr)?;
                let sk_idx = self.skolemize_inner(tm, idx)?;
                Ok(tm.mk_select(sk_arr, sk_idx))
            }
            TermKind::Store(arr, idx, val) => {
                let sk_arr = self.skolemize_inner(tm, arr)?;
                let sk_idx = self.skolemize_inner(tm, idx)?;
                let sk_val = self.skolemize_inner(tm, val)?;
                Ok(tm.mk_store(sk_arr, sk_idx, sk_val))
            }

            // Let bindings: recurse on bindings and body
            TermKind::Let { bindings, body } => {
                let mut sk_bindings = Vec::with_capacity(bindings.len());
                for (name, val) in &bindings {
                    let sk_val = self.skolemize_inner(tm, *val)?;
                    let name_str = tm.resolve_str(*name).to_string();
                    sk_bindings.push((name_str, sk_val));
                }
                let sk_body = self.skolemize_inner(tm, body)?;
                Ok(tm.mk_let(
                    sk_bindings
                        .iter()
                        .map(|(n, v): &(String, TermId)| (n.as_str(), *v)),
                    sk_body,
                ))
            }

            // For all other term kinds (BV ops, FP ops, String ops, DT ops, Match, etc.),
            // we treat them as opaque and return them as-is. These do not contain
            // quantifier-relevant substructure at the boolean level. If deeper
            // Skolemization is needed in theory-specific terms, the theory solver
            // should handle it.
            _ => Ok(term),
        }?;

        self.skolem_cache.insert(term, result);
        Ok(result)
    }

    /// Create a Skolem constant (no outer universals).
    ///
    /// Generates a fresh uninterpreted constant with the name `sk!N` where N
    /// is the current counter value.
    fn mk_skolem_constant(
        &mut self,
        tm: &mut TermManager,
        sort: SortId,
    ) -> Result<TermId, SkolemizationError> {
        let counter = self.skolem_counter;
        self.skolem_counter = self
            .skolem_counter
            .checked_add(1)
            .ok_or(SkolemizationError::CounterOverflow)?;

        let name = format!("sk!{}", counter);
        let term = tm.mk_var(&name, sort);

        self.skolem_symbols.push(SkolemSymbol {
            name,
            sort,
            term,
            arg_sorts: Vec::new(),
        });

        Ok(term)
    }

    /// Create a Skolem function applied to outer universals.
    ///
    /// Generates a fresh uninterpreted function `skf!N` and returns the
    /// application `skf!N(y1, y2, ...)` where y1, y2, ... are the current
    /// outer universal variables.
    fn mk_skolem_function(
        &mut self,
        tm: &mut TermManager,
        result_sort: SortId,
    ) -> Result<TermId, SkolemizationError> {
        let counter = self.skolem_counter;
        self.skolem_counter = self
            .skolem_counter
            .checked_add(1)
            .ok_or(SkolemizationError::CounterOverflow)?;

        let name = format!("skf!{}", counter);

        // Collect the argument sorts and term IDs from outer universals
        let arg_sorts: Vec<SortId> = self.outer_universals.iter().map(|(s, _)| *s).collect();
        let arg_terms: Vec<TermId> = self.outer_universals.iter().map(|(_, t)| *t).collect();

        // Create a function application term: skf!N(y1, y2, ...)
        let term = tm.mk_apply(&name, arg_terms, result_sort);

        self.skolem_symbols.push(SkolemSymbol {
            name,
            sort: result_sort,
            term,
            arg_sorts,
        });

        Ok(term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    /// Helper: check that a term does not contain any Exists quantifiers
    fn is_existential_free(tm: &TermManager, term: TermId) -> bool {
        let Some(t) = tm.get(term) else {
            return true;
        };
        match &t.kind {
            TermKind::Exists { .. } => false,
            TermKind::Not(arg) => is_existential_free(tm, *arg),
            TermKind::And(args) | TermKind::Or(args) => {
                args.iter().all(|&a| is_existential_free(tm, a))
            }
            TermKind::Implies(lhs, rhs) => {
                is_existential_free(tm, *lhs) && is_existential_free(tm, *rhs)
            }
            TermKind::Forall { body, .. } => is_existential_free(tm, *body),
            _ => true,
        }
    }

    #[test]
    fn test_skolemize_simple_exists() {
        // exists x : Bool. x
        // Should become: sk!0
        let mut tm = TermManager::new();
        let bool_sort = tm.sorts.bool_sort;
        let x = tm.mk_var("x", bool_sort);
        let exists = tm.mk_exists([("x", bool_sort)], x);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, exists);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        // The result should be existential-free
        assert!(is_existential_free(&tm, result_id));

        // Should have generated one Skolem symbol
        assert_eq!(ctx.skolem_count(), 1);
        let sym = &ctx.skolem_symbols()[0];
        assert_eq!(sym.name, "sk!0");
        assert!(sym.arg_sorts.is_empty());
    }

    #[test]
    fn test_skolemize_exists_with_body() {
        // exists x : Int. x > 0
        // Should become: sk!0 > 0
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let zero = tm.mk_int(BigInt::from(0));
        let gt = tm.mk_gt(x, zero);
        let exists = tm.mk_exists([("x", int_sort)], gt);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, exists);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        assert!(is_existential_free(&tm, result_id));
        assert_eq!(ctx.skolem_count(), 1);
    }

    #[test]
    fn test_skolemize_forall_exists() {
        // forall y : Int. exists x : Int. x > y
        // Should become: forall y : Int. skf!0(y) > y
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let gt = tm.mk_gt(x, y);
        let exists = tm.mk_exists([("x", int_sort)], gt);
        let forall = tm.mk_forall([("y", int_sort)], exists);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, forall);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        assert!(is_existential_free(&tm, result_id));

        // Should have generated one Skolem function
        assert_eq!(ctx.skolem_count(), 1);
        let sym = &ctx.skolem_symbols()[0];
        assert_eq!(sym.name, "skf!0");
        assert_eq!(sym.arg_sorts.len(), 1);
        assert_eq!(sym.arg_sorts[0], int_sort);
    }

    #[test]
    fn test_skolemize_nested_exists() {
        // exists x : Int. exists y : Int. x > y
        // Should become: sk!0 > sk!1
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let gt = tm.mk_gt(x, y);
        let exists_y = tm.mk_exists([("y", int_sort)], gt);
        let exists_x = tm.mk_exists([("x", int_sort)], exists_y);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, exists_x);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        assert!(is_existential_free(&tm, result_id));
        assert_eq!(ctx.skolem_count(), 2);
        // Both should be constants (no outer universals)
        assert!(ctx.skolem_symbols()[0].arg_sorts.is_empty());
        assert!(ctx.skolem_symbols()[1].arg_sorts.is_empty());
    }

    #[test]
    fn test_skolemize_negated_forall() {
        // NOT(forall x : Bool. x) should become, after NNF:
        // exists x : Bool. NOT x
        // Then Skolemized to: NOT sk!0
        let mut tm = TermManager::new();
        let bool_sort = tm.sorts.bool_sort;
        let x = tm.mk_var("x", bool_sort);
        let forall = tm.mk_forall([("x", bool_sort)], x);
        let neg_forall = tm.mk_not(forall);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, neg_forall);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        assert!(is_existential_free(&tm, result_id));
        assert_eq!(ctx.skolem_count(), 1);
    }

    #[test]
    fn test_skolemize_multiple_universal_vars() {
        // forall y : Int, z : Int. exists x : Int. x > y + z
        // Should generate skf!0(y, z) with two argument sorts
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let z = tm.mk_var("z", int_sort);
        let sum = tm.mk_add([y, z]);
        let gt = tm.mk_gt(x, sum);
        let exists = tm.mk_exists([("x", int_sort)], gt);
        let forall = tm.mk_forall([("y", int_sort), ("z", int_sort)], exists);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, forall);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        assert!(is_existential_free(&tm, result_id));
        assert_eq!(ctx.skolem_count(), 1);
        let sym = &ctx.skolem_symbols()[0];
        assert_eq!(sym.name, "skf!0");
        assert_eq!(sym.arg_sorts.len(), 2);
        assert_eq!(sym.arg_sorts[0], int_sort);
        assert_eq!(sym.arg_sorts[1], int_sort);
    }

    #[test]
    fn test_skolemize_preserves_ground_terms() {
        // A term with no quantifiers should be unchanged
        let mut tm = TermManager::new();
        let bool_sort = tm.sorts.bool_sort;
        let p = tm.mk_var("p", bool_sort);
        let q = tm.mk_var("q", bool_sort);
        let and = tm.mk_and([p, q]);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, and);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        // No Skolem symbols should be generated
        assert_eq!(ctx.skolem_count(), 0);
        // The result should be the same term
        assert_eq!(result_id, and);
    }

    #[test]
    fn test_skolemize_mixed_sorts() {
        // forall y : Int. exists x : Bool. x AND (y > 0)
        // The Skolem function should have Int argument sort and Bool result sort
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bool_sort = tm.sorts.bool_sort;
        let x = tm.mk_var("x", bool_sort);
        let y = tm.mk_var("y", int_sort);
        let zero = tm.mk_int(BigInt::from(0));
        let gt = tm.mk_gt(y, zero);
        let and = tm.mk_and([x, gt]);
        let exists = tm.mk_exists([("x", bool_sort)], and);
        let forall = tm.mk_forall([("y", int_sort)], exists);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, forall);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        assert!(is_existential_free(&tm, result_id));
        assert_eq!(ctx.skolem_count(), 1);
        let sym = &ctx.skolem_symbols()[0];
        assert_eq!(sym.sort, bool_sort);
        assert_eq!(sym.arg_sorts.len(), 1);
        assert_eq!(sym.arg_sorts[0], int_sort);
    }

    #[test]
    fn test_nnf_conversion_via_skolemize() {
        // NOT(p AND q) should be converted to (NOT p) OR (NOT q) before Skolemization
        // (though no quantifiers present, the NNF step still runs)
        let mut tm = TermManager::new();
        let bool_sort = tm.sorts.bool_sort;
        let p = tm.mk_var("p", bool_sort);
        let q = tm.mk_var("q", bool_sort);
        let and = tm.mk_and([p, q]);
        let neg = tm.mk_not(and);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, neg);
        assert!(result.is_ok());
        let result_id = result.expect("skolemize should succeed");

        // The result should be an OR (due to De Morgan)
        let t = tm.get(result_id);
        assert!(t.is_some());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::Or(_))));
    }

    #[test]
    fn test_skolemize_error_on_unknown_term() {
        let mut tm = TermManager::new();
        let bogus = TermId::new(999_999);

        let mut ctx = SkolemizationContext::new();
        let result = ctx.skolemize(&mut tm, bogus);
        assert!(result.is_err());
    }
}
