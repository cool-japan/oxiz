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

use crate::prelude::FxHashMap;
use oxiz_core::ast::TermManager;
use oxiz_core::ast::{TermId, TermKind};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;
use std::collections::HashMap;
use std::fmt;

/// Memo key for the Skolemization walk: the subterm, the universal variables
/// currently in scope, and the size of the Skolem map.
type SkolemKey = (TermId, Vec<TermId>, usize);

/// Resolve a quantifier's bound variables to owned `(name, sort)` pairs.
///
/// The names must be owned before the term manager is borrowed mutably again
/// to rebuild the quantifier.
fn resolve_bound_vars(tm: &TermManager, vars: &[(Spur, SortId)]) -> Vec<(String, SortId)> {
    vars.iter()
        .map(|(name, sort)| (tm.resolve_str(*name).to_string(), *sort))
        .collect()
}

/// Copy a quantifier's trigger patterns out of their `SmallVec` representation
/// so they can be parked in a work-stack frame.
fn flatten_patterns<P, Q>(patterns: P) -> Vec<Vec<TermId>>
where
    P: IntoIterator<Item = Q>,
    Q: IntoIterator<Item = TermId>,
{
    patterns
        .into_iter()
        .map(|pattern| pattern.into_iter().collect())
        .collect()
}

/// Pop one finished child result off a work-stack machine's value stack.
///
/// An empty stack means the machine pushed a build step without the matching
/// number of child evaluations -- an internal inconsistency, reported as an
/// error rather than by indexing into an empty `Vec`.
fn pop_value(values: &mut Vec<TermId>, what: &str) -> Result<TermId, SkolemizationError> {
    values.pop().ok_or_else(|| {
        SkolemizationError::TermConstructionFailed(format!("{what}: missing operand"))
    })
}

/// Pop the `arity` most recent child results, oldest first.
fn pop_values(
    values: &mut Vec<TermId>,
    arity: usize,
    what: &str,
) -> Result<Vec<TermId>, SkolemizationError> {
    let start = values.len().checked_sub(arity).ok_or_else(|| {
        SkolemizationError::TermConstructionFailed(format!(
            "{what}: expected {arity} operands, found {}",
            values.len()
        ))
    })?;
    Ok(values.split_off(start))
}

/// Read exactly one already-built child result.
fn arg1(args: &[TermId], what: &str) -> Result<TermId, SkolemizationError> {
    match args {
        [a] => Ok(*a),
        _ => Err(SkolemizationError::TermConstructionFailed(format!(
            "{what}: expected 1 operand, found {}",
            args.len()
        ))),
    }
}

/// Read exactly two already-built child results.
fn arg2(args: &[TermId], what: &str) -> Result<(TermId, TermId), SkolemizationError> {
    match args {
        [a, b] => Ok((*a, *b)),
        _ => Err(SkolemizationError::TermConstructionFailed(format!(
            "{what}: expected 2 operands, found {}",
            args.len()
        ))),
    }
}

/// Read exactly three already-built child results.
fn arg3(args: &[TermId], what: &str) -> Result<(TermId, TermId, TermId), SkolemizationError> {
    match args {
        [a, b, c] => Ok((*a, *b, *c)),
        _ => Err(SkolemizationError::TermConstructionFailed(format!(
            "{what}: expected 3 operands, found {}",
            args.len()
        ))),
    }
}

/// Schedule `op`'s rebuild and then its children, so that the children are
/// evaluated left to right and their results sit under the build step.
fn push_children<I>(steps: &mut Vec<SkolemStep>, key: SkolemKey, op: SkolemOp, children: I)
where
    I: IntoIterator<Item = TermId>,
    I::IntoIter: DoubleEndedIterator,
{
    steps.push(SkolemStep::Build { key, op });
    for child in children.into_iter().rev() {
        steps.push(SkolemStep::Eval { term: child });
    }
}

/// One step of the iterative Skolemization machine driven by
/// [`SkolemizationContext::skolemize_inner`].
enum SkolemStep {
    /// Skolemize one subterm.
    Eval {
        /// Subterm to process.
        term: TermId,
    },
    /// Adopt the single child result as the result of `key` (used where a node
    /// disappears, i.e. for `exists`).
    Memo {
        /// Memo key of the node being completed.
        key: SkolemKey,
    },
    /// Rebuild a node from its already-Skolemized children.
    Build {
        /// Memo key of the node being completed.
        key: SkolemKey,
        /// How to recombine the children.
        op: SkolemOp,
    },
}

/// How a [`SkolemStep::Build`] recombines its children.
enum SkolemOp {
    /// Logical negation.
    Not,
    /// Conjunction of the given number of children.
    And(usize),
    /// Disjunction of the given number of children.
    Or(usize),
    /// Implication.
    Implies,
    /// Exclusive or.
    Xor,
    /// If-then-else.
    Ite,
    /// Equality.
    Eq,
    /// Pairwise distinctness of the given number of children.
    Distinct(usize),
    /// Strict less-than.
    Lt,
    /// Less-or-equal.
    Le,
    /// Strict greater-than.
    Gt,
    /// Greater-or-equal.
    Ge,
    /// Arithmetic negation.
    Neg,
    /// Sum of the given number of children.
    Add(usize),
    /// Difference.
    Sub,
    /// Product of the given number of children.
    Mul(usize),
    /// Division.
    Div,
    /// Modulo.
    Mod,
    /// Uninterpreted function application.
    Apply {
        /// Function name.
        name: String,
        /// Result sort.
        sort: SortId,
        /// Number of arguments.
        arity: usize,
    },
    /// Array select.
    Select,
    /// Array store.
    Store,
    /// `let` binding; the children are the bound values followed by the body.
    Let {
        /// Names of the bound variables.
        names: Vec<String>,
    },
    /// Universal quantifier; also pops the universals its `Eval` step pushed.
    Forall {
        /// Bound variables, already resolved to owned names.
        vars: Vec<(String, SortId)>,
    },
}

impl SkolemOp {
    /// Number of child results this build step consumes.
    fn arity(&self) -> usize {
        match self {
            SkolemOp::Not | SkolemOp::Neg | SkolemOp::Forall { .. } => 1,
            SkolemOp::Implies
            | SkolemOp::Xor
            | SkolemOp::Eq
            | SkolemOp::Lt
            | SkolemOp::Le
            | SkolemOp::Gt
            | SkolemOp::Ge
            | SkolemOp::Sub
            | SkolemOp::Div
            | SkolemOp::Mod
            | SkolemOp::Select => 2,
            SkolemOp::Ite | SkolemOp::Store => 3,
            SkolemOp::And(n)
            | SkolemOp::Or(n)
            | SkolemOp::Distinct(n)
            | SkolemOp::Add(n)
            | SkolemOp::Mul(n)
            | SkolemOp::Apply { arity: n, .. } => *n,
            SkolemOp::Let { names } => names.len() + 1,
        }
    }
}

/// One step of the iterative NNF conversion machine driven by
/// [`SkolemizationContext::convert_nnf`].
enum NnfStep {
    /// Convert `term` under the given negation polarity.
    Eval {
        /// Subterm to convert.
        term: TermId,
        /// Whether the conversion happens under a negation.
        negated: bool,
    },
    /// Adopt the single child result as the result of `key`.
    Memo {
        /// Cache key of the node being completed.
        key: (TermId, bool),
    },
    /// Recombine `arity` child results into a conjunction or disjunction.
    Junction {
        /// Cache key of the node being completed.
        key: (TermId, bool),
        /// Number of child results to consume.
        arity: usize,
        /// Build an `and` when true, an `or` when false.
        build_and: bool,
    },
    /// Recombine the four child results of an `Xor`.
    Xor {
        /// Cache key of the node being completed.
        key: (TermId, bool),
        /// Whether the `Xor` occurs under a negation.
        negated: bool,
    },
    /// Rebuild a quantifier around the converted body.
    Quantifier {
        /// Cache key of the node being completed.
        key: (TermId, bool),
        /// Build a `forall` when true, an `exists` when false.
        build_forall: bool,
        /// Bound variables, already resolved to owned names.
        vars: Vec<(String, SortId)>,
        /// Trigger patterns carried over unchanged.
        patterns: Vec<Vec<TermId>>,
    },
}

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
    /// Cache for skolemize_inner to avoid reprocessing subterms; keyed on the
    /// binding context as well as the subterm (see
    /// [`SkolemizationContext::skolem_key`]).
    skolem_cache: HashMap<SkolemKey, TermId>,
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

    /// Create a Skolemization context whose fresh-symbol counter starts at
    /// `first_id`.
    ///
    /// Skolem symbols are named positionally (`sk!N` / `skf!N`), so two
    /// contexts that both start at zero mint the *same* names — and, because
    /// names are interned, the *same* symbols.  Two unrelated existentials
    /// sharing one witness symbol is a strengthening of the assertion set: it
    /// can turn a satisfiable problem unsatisfiable.  A caller that
    /// Skolemizes more than once (the solver Skolemizes per assertion) must
    /// therefore thread a monotone counter through this constructor and read
    /// it back with [`SkolemizationContext::skolem_count`].
    pub fn with_first_id(first_id: u64) -> Self {
        SkolemizationContext {
            skolem_counter: first_id,
            ..SkolemizationContext::new()
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

    /// Record `value` as the NNF of `key` and hand it to the pending build step.
    fn nnf_yield(&mut self, key: (TermId, bool), value: TermId, values: &mut Vec<TermId>) {
        self.nnf_cache.insert(key, value);
        values.push(value);
    }

    /// Convert to Negation Normal Form.
    ///
    /// When `negated` is true, we are converting under a negation context, which
    /// flips AND/OR and swaps Forall/Exists.
    ///
    /// # Implementation
    ///
    /// The conversion is driven by an explicit heap stack of [`NnfStep`]s
    /// rather than by recursion. Nesting depth here is entirely
    /// caller-controlled -- a chain of `n` negations, conjunctions or
    /// quantifiers used to cost `n` native stack frames -- and the depth at
    /// which a formula overflows depends on the build profile, so no honest
    /// cap exists. `Eval` steps expand one subterm into its children plus the
    /// build step that recombines them; each finished child pushes its result
    /// onto `values`, which the build step pops from. Evaluation order,
    /// negation propagation and every `nnf_cache` insertion are the same as in
    /// the recursive version, so the produced term is identical.
    fn convert_nnf(
        &mut self,
        tm: &mut TermManager,
        term: TermId,
        negated: bool,
    ) -> Result<TermId, SkolemizationError> {
        let mut steps: Vec<NnfStep> = vec![NnfStep::Eval { term, negated }];
        let mut values: Vec<TermId> = Vec::new();

        while let Some(step) = steps.pop() {
            match step {
                NnfStep::Eval { term, negated } => {
                    if let Some(&cached) = self.nnf_cache.get(&(term, negated)) {
                        values.push(cached);
                        continue;
                    }

                    let kind = tm
                        .get(term)
                        .ok_or(SkolemizationError::UnknownTerm(term))?
                        .kind
                        .clone();
                    let key = (term, negated);

                    match kind {
                        TermKind::True => {
                            let value = if negated { tm.mk_false() } else { tm.mk_true() };
                            self.nnf_yield(key, value, &mut values);
                        }
                        TermKind::False => {
                            let value = if negated { tm.mk_true() } else { tm.mk_false() };
                            self.nnf_yield(key, value, &mut values);
                        }
                        TermKind::Var(_)
                        | TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. }
                        | TermKind::StringLit(_) => {
                            let value = if negated { tm.mk_not(term) } else { term };
                            self.nnf_yield(key, value, &mut values);
                        }
                        TermKind::Not(arg) => {
                            // Double negation elimination: push negation through.
                            steps.push(NnfStep::Memo { key });
                            steps.push(NnfStep::Eval {
                                term: arg,
                                negated: !negated,
                            });
                        }
                        TermKind::And(args) => {
                            // De Morgan: NOT(a AND b) = (NOT a) OR (NOT b)
                            steps.push(NnfStep::Junction {
                                key,
                                arity: args.len(),
                                build_and: !negated,
                            });
                            for &a in args.iter().rev() {
                                steps.push(NnfStep::Eval { term: a, negated });
                            }
                        }
                        TermKind::Or(args) => {
                            // De Morgan: NOT(a OR b) = (NOT a) AND (NOT b)
                            steps.push(NnfStep::Junction {
                                key,
                                arity: args.len(),
                                build_and: negated,
                            });
                            for &a in args.iter().rev() {
                                steps.push(NnfStep::Eval { term: a, negated });
                            }
                        }
                        TermKind::Implies(lhs, rhs) => {
                            // (a -> b) = (NOT a) OR b; NOT(a -> b) = a AND (NOT b)
                            steps.push(NnfStep::Junction {
                                key,
                                arity: 2,
                                build_and: negated,
                            });
                            steps.push(NnfStep::Eval { term: rhs, negated });
                            steps.push(NnfStep::Eval {
                                term: lhs,
                                negated: !negated,
                            });
                        }
                        TermKind::Xor(lhs, rhs) => {
                            // a XOR b = (a OR b) AND (NOT a OR NOT b)
                            steps.push(NnfStep::Xor { key, negated });
                            steps.push(NnfStep::Eval {
                                term: rhs,
                                negated: true,
                            });
                            steps.push(NnfStep::Eval {
                                term: lhs,
                                negated: true,
                            });
                            steps.push(NnfStep::Eval {
                                term: rhs,
                                negated: false,
                            });
                            steps.push(NnfStep::Eval {
                                term: lhs,
                                negated: false,
                            });
                        }
                        TermKind::Forall {
                            vars,
                            body,
                            patterns,
                        } => {
                            // NOT(forall x. P(x)) = exists x. NOT P(x)
                            steps.push(NnfStep::Quantifier {
                                key,
                                build_forall: !negated,
                                vars: resolve_bound_vars(tm, &vars),
                                patterns: flatten_patterns(patterns),
                            });
                            steps.push(NnfStep::Eval {
                                term: body,
                                negated,
                            });
                        }
                        TermKind::Exists {
                            vars,
                            body,
                            patterns,
                        } => {
                            // NOT(exists x. P(x)) = forall x. NOT P(x)
                            steps.push(NnfStep::Quantifier {
                                key,
                                build_forall: negated,
                                vars: resolve_bound_vars(tm, &vars),
                                patterns: flatten_patterns(patterns),
                            });
                            steps.push(NnfStep::Eval {
                                term: body,
                                negated,
                            });
                        }
                        TermKind::Eq(_, _)
                        | TermKind::Distinct(_)
                        | TermKind::Lt(_, _)
                        | TermKind::Le(_, _)
                        | TermKind::Gt(_, _)
                        | TermKind::Ge(_, _)
                        | TermKind::Apply { .. }
                        | TermKind::Ite(_, _, _) => {
                            // Atoms and other non-boolean-connective terms:
                            // just negate if needed.
                            let value = if negated { tm.mk_not(term) } else { term };
                            self.nnf_yield(key, value, &mut values);
                        }
                        // All remaining term kinds (arithmetic, bitvec, string
                        // ops, FP, etc.) are treated as atoms in the boolean
                        // sense.
                        _ => {
                            let value = if negated { tm.mk_not(term) } else { term };
                            self.nnf_yield(key, value, &mut values);
                        }
                    }
                }
                NnfStep::Memo { key } => {
                    let value = pop_value(&mut values, "NNF pass-through")?;
                    self.nnf_yield(key, value, &mut values);
                }
                NnfStep::Junction {
                    key,
                    arity,
                    build_and,
                } => {
                    let args = pop_values(&mut values, arity, "NNF junction")?;
                    let value = if build_and {
                        tm.mk_and(args)
                    } else {
                        tm.mk_or(args)
                    };
                    self.nnf_yield(key, value, &mut values);
                }
                NnfStep::Xor { key, negated } => {
                    let parts = pop_values(&mut values, 4, "NNF xor")?;
                    let [a, b, not_a, not_b] = parts[..] else {
                        return Err(SkolemizationError::TermConstructionFailed(
                            "NNF xor expected four operands".to_string(),
                        ));
                    };
                    let clause1 = tm.mk_or([a, b]);
                    let clause2 = tm.mk_or([not_a, not_b]);
                    let xor_nnf = tm.mk_and([clause1, clause2]);
                    if negated {
                        // NOT(a XOR b) = (a AND b) OR (NOT a AND NOT b):
                        // re-enter the machine on the built term so the result
                        // stays in NNF.
                        steps.push(NnfStep::Memo { key });
                        steps.push(NnfStep::Eval {
                            term: xor_nnf,
                            negated: true,
                        });
                    } else {
                        self.nnf_yield(key, xor_nnf, &mut values);
                    }
                }
                NnfStep::Quantifier {
                    key,
                    build_forall,
                    vars,
                    patterns,
                } => {
                    let body_nnf = pop_value(&mut values, "NNF quantifier body")?;
                    let bound = vars
                        .iter()
                        .map(|(s, sort): &(String, SortId)| (s.as_str(), *sort));
                    let value = if build_forall {
                        tm.mk_forall_with_patterns(bound, body_nnf, patterns)
                    } else {
                        tm.mk_exists_with_patterns(bound, body_nnf, patterns)
                    };
                    self.nnf_yield(key, value, &mut values);
                }
            }
        }

        pop_value(&mut values, "NNF conversion")
    }

    /// Cache key for [`Self::skolemize_inner`].
    ///
    /// Skolemization is *not* a function of the subterm alone: the result also
    /// depends on which universals are currently in scope (they become the
    /// arguments of any Skolem function created underneath) and on which
    /// existential variables have already been mapped. Keying the memo on the
    /// term id alone -- as this cache used to -- lets a subterm skolemized
    /// under one quantifier scope be reused verbatim under a different one,
    /// which silently produces a term with the wrong Skolem arguments, and
    /// lets an occurrence of `x` that was replaced inside `exists x. P(x)` be
    /// replaced again in a later, genuinely free occurrence of `P(x)`.
    ///
    /// The key therefore carries the current universal scope and the size of
    /// the Skolem map (which only ever grows, so its size identifies its
    /// state). Within one scope the key is stable, so a shared sub-DAG is
    /// still skolemized once.
    fn skolem_key(&self, term: TermId) -> SkolemKey {
        (
            term,
            self.outer_universals.iter().map(|(_, id)| *id).collect(),
            self.skolem_map.len(),
        )
    }

    /// Record `value` as the Skolemization of `key` and hand it to the pending
    /// build step.
    fn skolem_yield(&mut self, key: SkolemKey, value: TermId, values: &mut Vec<TermId>) {
        self.skolem_cache.insert(key, value);
        values.push(value);
    }

    /// Inner Skolemization pass.
    ///
    /// Traverses the NNF term tree:
    /// - Forall: pushes bound variables onto `outer_universals`, walks the body, pops
    /// - Exists: for each bound variable, creates a Skolem constant or function,
    ///   adds mapping to `skolem_map`, walks the body
    /// - Variable: if in `skolem_map`, returns the replacement; otherwise returns as-is
    /// - Other: rebuilds the node around its Skolemized children
    ///
    /// # Implementation
    ///
    /// Like [`Self::convert_nnf`] this runs on an explicit heap stack of
    /// [`SkolemStep`]s. Depth is caller-controlled (one native frame per
    /// nesting level in the recursive version), and no depth cap could be
    /// honest here: silently returning an unskolemized subterm would leave an
    /// existential quantifier in a formula the caller believes is
    /// quantifier-free on the existential side.
    ///
    /// The `outer_universals` stack is pushed when a `Forall` node is expanded
    /// and popped by that node's build step, and the machine is a strict
    /// depth-first walk, so the scope nesting is exactly the one the recursive
    /// version maintained.
    fn skolemize_inner(
        &mut self,
        tm: &mut TermManager,
        term: TermId,
    ) -> Result<TermId, SkolemizationError> {
        let mut steps: Vec<SkolemStep> = vec![SkolemStep::Eval { term }];
        let mut values: Vec<TermId> = Vec::new();

        while let Some(step) = steps.pop() {
            match step {
                SkolemStep::Eval { term } => {
                    let key = self.skolem_key(term);
                    if let Some(&cached) = self.skolem_cache.get(&key) {
                        values.push(cached);
                        continue;
                    }

                    let kind = tm
                        .get(term)
                        .ok_or(SkolemizationError::UnknownTerm(term))?
                        .kind
                        .clone();

                    match kind {
                        // Base cases: constants are unchanged
                        TermKind::True
                        | TermKind::False
                        | TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. }
                        | TermKind::StringLit(_) => self.skolem_yield(key, term, &mut values),

                        // Variables: check if this var has a Skolem replacement
                        TermKind::Var(_) => {
                            let value = self.skolem_map.get(&term).copied().unwrap_or(term);
                            self.skolem_yield(key, value, &mut values);
                        }

                        // Universal quantifier: push vars onto outer_universals,
                        // walk the body, pop them again in the build step.
                        TermKind::Forall {
                            vars,
                            body,
                            patterns: _,
                        } => {
                            // Push each bound variable onto the outer_universals
                            // stack. We need the actual TermIds for these
                            // variables so that Skolem functions can reference
                            // them as arguments.
                            let var_names = resolve_bound_vars(tm, &vars);
                            for (name, sort) in &var_names {
                                let var_id = tm.mk_var(name, *sort);
                                self.outer_universals.push((*sort, var_id));
                            }

                            // Patterns are dropped because Skolemization may
                            // have changed the variable structure.
                            steps.push(SkolemStep::Build {
                                key,
                                op: SkolemOp::Forall { vars: var_names },
                            });
                            steps.push(SkolemStep::Eval { term: body });
                        }

                        // Existential quantifier: create Skolem constants or
                        // functions for each bound var, then keep only the body.
                        TermKind::Exists { vars, body, .. } => {
                            for (name, sort) in resolve_bound_vars(tm, &vars) {
                                let var_id = tm.mk_var(&name, sort);

                                let skolem_term = if self.outer_universals.is_empty() {
                                    // No outer universal variables: create a
                                    // Skolem constant.
                                    self.mk_skolem_constant(tm, sort)?
                                } else {
                                    // There are outer universal variables:
                                    // create a Skolem function applied to them.
                                    self.mk_skolem_function(tm, sort)?
                                };

                                self.skolem_map.insert(var_id, skolem_term);
                            }

                            // The existential quantifier itself is eliminated.
                            steps.push(SkolemStep::Memo { key });
                            steps.push(SkolemStep::Eval { term: body });
                        }

                        // Boolean connectives
                        TermKind::Not(arg) => {
                            push_children(&mut steps, key, SkolemOp::Not, [arg]);
                        }
                        TermKind::And(args) => {
                            let op = SkolemOp::And(args.len());
                            push_children(&mut steps, key, op, args.iter().copied());
                        }
                        TermKind::Or(args) => {
                            let op = SkolemOp::Or(args.len());
                            push_children(&mut steps, key, op, args.iter().copied());
                        }
                        TermKind::Implies(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Implies, [lhs, rhs]);
                        }
                        TermKind::Xor(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Xor, [lhs, rhs]);
                        }
                        TermKind::Ite(cond, then_br, else_br) => {
                            push_children(&mut steps, key, SkolemOp::Ite, [cond, then_br, else_br]);
                        }

                        // Equality and comparison
                        TermKind::Eq(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Eq, [lhs, rhs]);
                        }
                        TermKind::Distinct(args) => {
                            let op = SkolemOp::Distinct(args.len());
                            push_children(&mut steps, key, op, args.iter().copied());
                        }
                        TermKind::Lt(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Lt, [lhs, rhs]);
                        }
                        TermKind::Le(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Le, [lhs, rhs]);
                        }
                        TermKind::Gt(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Gt, [lhs, rhs]);
                        }
                        TermKind::Ge(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Ge, [lhs, rhs]);
                        }

                        // Arithmetic
                        TermKind::Neg(arg) => {
                            push_children(&mut steps, key, SkolemOp::Neg, [arg]);
                        }
                        TermKind::Add(args) => {
                            let op = SkolemOp::Add(args.len());
                            push_children(&mut steps, key, op, args.iter().copied());
                        }
                        TermKind::Sub(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Sub, [lhs, rhs]);
                        }
                        TermKind::Mul(args) => {
                            let op = SkolemOp::Mul(args.len());
                            push_children(&mut steps, key, op, args.iter().copied());
                        }
                        TermKind::Div(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Div, [lhs, rhs]);
                        }
                        TermKind::Mod(lhs, rhs) => {
                            push_children(&mut steps, key, SkolemOp::Mod, [lhs, rhs]);
                        }

                        // Uninterpreted function application
                        TermKind::Apply { func, args } => {
                            let op = SkolemOp::Apply {
                                name: tm.resolve_str(func).to_string(),
                                sort: tm
                                    .get(term)
                                    .ok_or(SkolemizationError::UnknownTerm(term))?
                                    .sort,
                                arity: args.len(),
                            };
                            push_children(&mut steps, key, op, args.iter().copied());
                        }

                        // Array operations
                        TermKind::Select(arr, idx) => {
                            push_children(&mut steps, key, SkolemOp::Select, [arr, idx]);
                        }
                        TermKind::Store(arr, idx, val) => {
                            push_children(&mut steps, key, SkolemOp::Store, [arr, idx, val]);
                        }

                        // Let bindings: the bound values, then the body
                        TermKind::Let { bindings, body } => {
                            let names: Vec<String> = bindings
                                .iter()
                                .map(|(name, _)| tm.resolve_str(*name).to_string())
                                .collect();
                            let children: Vec<TermId> = bindings
                                .iter()
                                .map(|(_, value)| *value)
                                .chain(std::iter::once(body))
                                .collect();
                            push_children(&mut steps, key, SkolemOp::Let { names }, children);
                        }

                        // All other term kinds (BV ops, FP ops, string ops,
                        // datatype ops, `Match`, ...) carry no quantifier
                        // structure of their own, but they may still contain
                        // variables that an enclosing `exists` has mapped to a
                        // Skolem term, so the mapping is applied through
                        // `TermManager::substitute` (exhaustive over every
                        // kind, capture-avoiding, and itself iterative) rather
                        // than returning the subterm untouched -- returning it
                        // untouched used to drop the replacement silently, so
                        // `exists x. (= (bvadd x y) z)` kept a free `x` that no
                        // quantifier bound any more.
                        _ => {
                            let value = self.apply_skolem_map(tm, term);
                            self.skolem_yield(key, value, &mut values);
                        }
                    }
                }

                SkolemStep::Memo { key } => {
                    let value = pop_value(&mut values, "Skolemization pass-through")?;
                    self.skolem_yield(key, value, &mut values);
                }

                SkolemStep::Build { key, op } => {
                    let args = pop_values(&mut values, op.arity(), "Skolemization")?;
                    let value = self.build_skolemized(tm, op, &args)?;
                    self.skolem_yield(key, value, &mut values);
                }
            }
        }

        pop_value(&mut values, "Skolemization")
    }

    /// Replace every Skolem-mapped variable inside `term`.
    ///
    /// Used for the term kinds the walk treats as opaque leaves.
    fn apply_skolem_map(&self, tm: &mut TermManager, term: TermId) -> TermId {
        if self.skolem_map.is_empty() {
            return term;
        }
        let subst: FxHashMap<TermId, TermId> =
            self.skolem_map.iter().map(|(&k, &v)| (k, v)).collect();
        tm.substitute(term, &subst)
    }

    /// Rebuild one node from its already-Skolemized children.
    fn build_skolemized(
        &mut self,
        tm: &mut TermManager,
        op: SkolemOp,
        args: &[TermId],
    ) -> Result<TermId, SkolemizationError> {
        let value = match op {
            SkolemOp::Not => tm.mk_not(arg1(args, "not")?),
            SkolemOp::And(_) => tm.mk_and(args.to_vec()),
            SkolemOp::Or(_) => tm.mk_or(args.to_vec()),
            SkolemOp::Implies => {
                let (lhs, rhs) = arg2(args, "implies")?;
                tm.mk_implies(lhs, rhs)
            }
            SkolemOp::Xor => {
                let (lhs, rhs) = arg2(args, "xor")?;
                tm.mk_xor(lhs, rhs)
            }
            SkolemOp::Ite => {
                let (cond, then_br, else_br) = arg3(args, "ite")?;
                tm.mk_ite(cond, then_br, else_br)
            }
            SkolemOp::Eq => {
                let (lhs, rhs) = arg2(args, "eq")?;
                tm.mk_eq(lhs, rhs)
            }
            SkolemOp::Distinct(_) => tm.mk_distinct(args.to_vec()),
            SkolemOp::Lt => {
                let (lhs, rhs) = arg2(args, "lt")?;
                tm.mk_lt(lhs, rhs)
            }
            SkolemOp::Le => {
                let (lhs, rhs) = arg2(args, "le")?;
                tm.mk_le(lhs, rhs)
            }
            SkolemOp::Gt => {
                let (lhs, rhs) = arg2(args, "gt")?;
                tm.mk_gt(lhs, rhs)
            }
            SkolemOp::Ge => {
                let (lhs, rhs) = arg2(args, "ge")?;
                tm.mk_ge(lhs, rhs)
            }
            SkolemOp::Neg => tm.mk_neg(arg1(args, "neg")?),
            SkolemOp::Add(_) => tm.mk_add(args.to_vec()),
            SkolemOp::Sub => {
                let (lhs, rhs) = arg2(args, "sub")?;
                tm.mk_sub(lhs, rhs)
            }
            SkolemOp::Mul(_) => tm.mk_mul(args.to_vec()),
            SkolemOp::Div => {
                let (lhs, rhs) = arg2(args, "div")?;
                tm.mk_div(lhs, rhs)
            }
            SkolemOp::Mod => {
                let (lhs, rhs) = arg2(args, "mod")?;
                tm.mk_mod(lhs, rhs)
            }
            SkolemOp::Apply { name, sort, .. } => tm.mk_apply(&name, args.to_vec(), sort),
            SkolemOp::Select => {
                let (arr, idx) = arg2(args, "select")?;
                tm.mk_select(arr, idx)
            }
            SkolemOp::Store => {
                let (arr, idx, val) = arg3(args, "store")?;
                tm.mk_store(arr, idx, val)
            }
            SkolemOp::Let { names } => {
                let (bound, body) = args.split_at(names.len());
                let body = arg1(body, "let body")?;
                let bindings: Vec<(&str, TermId)> = names
                    .iter()
                    .zip(bound.iter())
                    .map(|(name, value)| (name.as_str(), *value))
                    .collect();
                tm.mk_let(bindings, body)
            }
            SkolemOp::Forall { vars } => {
                // Pop exactly the universals this node pushed.
                let remaining = self.outer_universals.len().saturating_sub(vars.len());
                self.outer_universals.truncate(remaining);

                let body = arg1(args, "forall body")?;
                tm.mk_forall(
                    vars.iter()
                        .map(|(name, sort): &(String, SortId)| (name.as_str(), *sort)),
                    body,
                )
            }
        };

        Ok(value)
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

        let name = oxiz_core::smtlib::reserved_name("sk", &counter.to_string());
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

        let name = oxiz_core::smtlib::reserved_name("skf", &counter.to_string());

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
        assert_eq!(sym.name, oxiz_core::smtlib::reserved_name("sk", "0"));
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
        assert_eq!(sym.name, oxiz_core::smtlib::reserved_name("skf", "0"));
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
        assert_eq!(sym.name, oxiz_core::smtlib::reserved_name("skf", "0"));
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

    /// Both passes (NNF conversion and Skolemization proper) used to recurse
    /// once per nesting level. Returning at all is the assertion.
    #[test]
    fn skolemize_survives_a_deep_negation_chain_on_a_small_stack() {
        // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
        // ~10 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 12_500;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let bool_sort = tm.sorts.bool_sort;
                let mut term = tm.mk_var("p", bool_sort);
                for _ in 0..DEPTH {
                    term = tm.mk_not(term);
                }

                let mut ctx = SkolemizationContext::new();
                ctx.skolemize(&mut tm, term).is_ok()
            })
            .expect("spawning the worker thread should succeed");

        assert!(handle.join().expect("the walk must not overflow"));
    }

    /// A deep conjunction nest exercises the n-ary junction frames.
    #[test]
    fn skolemize_survives_a_deep_conjunction_nest_on_a_small_stack() {
        // Stack and depth scale together (1 MiB/50k -> 128 KiB/6.25k): the
        // ~21 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 6_250;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let bool_sort = tm.sorts.bool_sort;
                // `mk_and` flattens nested conjunctions, so alternate the
                // connective to build depth rather than width.
                let leaf = tm.mk_var("q", bool_sort);
                let mut term = tm.mk_var("p", bool_sort);
                for level in 0..DEPTH {
                    term = if level % 2 == 0 {
                        tm.mk_and([term, leaf])
                    } else {
                        tm.mk_or([term, leaf])
                    };
                }

                let mut ctx = SkolemizationContext::new();
                ctx.skolemize(&mut tm, term).is_ok()
            })
            .expect("spawning the worker thread should succeed");

        assert!(handle.join().expect("the walk must not overflow"));
    }

    /// A doubling DAG has 2^levels paths but only `levels` distinct subterms.
    /// Both memos must collapse it, or this never finishes.
    #[test]
    fn skolemize_collapses_a_shared_dag() {
        let mut tm = TermManager::new();
        let bool_sort = tm.sorts.bool_sort;
        let mut term = tm.mk_var("p", bool_sort);
        for level in 0..55 {
            term = if level % 2 == 0 {
                tm.mk_and([term, term])
            } else {
                tm.mk_or([term, term])
            };
        }

        let mut ctx = SkolemizationContext::new();
        assert!(ctx.skolemize(&mut tm, term).is_ok());
    }

    /// Semantic pin: the same existential subterm occurring twice gets a
    /// single Skolem constant, and an occurrence of the same variable name
    /// under an unrelated universal keeps its own identity.
    #[test]
    fn skolem_scope_is_part_of_the_memo_key() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bool_sort = tm.sorts.bool_sort;

        // (and (exists ((x Int)) (p x)) (forall ((y Int)) (exists ((x Int)) (p x))))
        let x = tm.mk_var("x", int_sort);
        let body = tm.mk_apply("p", vec![x], bool_sort);
        let inner_exists = tm.mk_exists([("x", int_sort)], body);
        let nested = tm.mk_exists([("x", int_sort)], body);
        let forall = tm.mk_forall([("y", int_sort)], nested);
        let term = tm.mk_and([inner_exists, forall]);

        let mut ctx = SkolemizationContext::new();
        let result = ctx
            .skolemize(&mut tm, term)
            .expect("skolemizing a quantifier-free-after-NNF term should succeed");

        // Two distinct scopes: a Skolem constant outside, a Skolem function
        // under the universal. The context must not have reused one for both.
        assert!(
            ctx.skolem_symbols().len() >= 2,
            "each scope needs its own Skolem symbol, got {:?}",
            ctx.skolem_symbols().len()
        );
        assert!(tm.get(result).is_some());
    }
}
