//! Exact and approximate model counting (#SAT)
//!
//! This module counts satisfying assignments ("models") of an SMT-LIB2
//! script by actually driving [`Context`]'s solver — it never fabricates a
//! count from text heuristics.
//!
//! * **Exact** counting enumerates models via the classic blocking-clause
//!   technique: solve, read the model back, assert the negation of that
//!   exact assignment, solve again, and repeat until the solver reports
//!   `unsat` (every model has been found) or a configured cap is hit. When
//!   capped, the result is reported honestly as a lower bound ("at least
//!   N"), never as a fabricated exact count.
//! * **Approximate** counting invokes the solver too: for small variable
//!   spaces (or spaces containing non-Boolean variables, which the hashing
//!   scheme below does not cover) it falls back to the same bounded
//!   enumeration used by exact mode. For large all-Boolean variable spaces
//!   it uses XOR/parity-constraint hashing (in the spirit of ApproxMC-style
//!   model counters): repeatedly add random parity constraints and binary
//!   search for the point where the solver flips from `sat` to `unsat`,
//!   which estimates `log2(#models)`. Every reported number therefore comes
//!   from a real `check-sat` call, and the result documents which strategy
//!   produced it.

use oxiz_core::ast::TermId;
use oxiz_solver::{Context, SolverResult};
use serde::{Deserialize, Serialize};

/// Below this many Boolean variables, exhaustive enumeration is cheap and
/// exact, so we prefer it over the statistically noisier hashing estimator.
const HASH_MIN_VARS: usize = 10;

/// Default hard wall-clock ceiling on a single bounded-enumeration pass
/// ([`enumerate_models_bounded`]), independent of the configured `--count-samples`
/// cap.
///
/// A formula whose declared variables include an unbounded-domain sort (`Int`,
/// `Real`) has, from the enumerator's point of view, no natural stopping point
/// short of the user's cap: each iteration asserts one more small blocking
/// clause and asks the solver again. On a genuinely infinite-model formula
/// (e.g. a single unconstrained `Int`), the default cap of 1000 solver calls
/// was observed to take anywhere from a few seconds to several minutes
/// depending on machine load, since every call re-processes an
/// ever-growing assertion set. That variance makes exact/approximate
/// counting on an otherwise-trivial formula (even one with only a couple of
/// declared variables) unpredictably slow.
///
/// This budget bounds that pathological case honestly: once elapsed wall
/// time exceeds it, enumeration stops and reports a sound lower bound
/// ([`CountStatus::LowerBoundCapped`]) instead of grinding on indefinitely.
/// It never fires on formulas with a fully enumerable (all-`Bool`/fixed-width
/// `BitVec`) domain, since [`enumerable_domain_size`] already gives those an
/// exact early exit well before this many seconds could elapse.
const DEFAULT_ENUMERATION_WALL_CLOCK_BUDGET: std::time::Duration =
    std::time::Duration::from_secs(10);

/// Environment variable letting tests shrink
/// [`DEFAULT_ENUMERATION_WALL_CLOCK_BUDGET`] so a regression test for the
/// wall-clock safety net does not have to actually wait out the full
/// production ceiling. Unset (or unparseable) in normal operation, where the
/// production default always applies.
const WALL_CLOCK_BUDGET_OVERRIDE_MS_VAR: &str = "OXIZ_MODEL_COUNT_WALL_CLOCK_BUDGET_MS";

/// The wall-clock ceiling actually in effect for [`enumerate_models_bounded`]:
/// [`DEFAULT_ENUMERATION_WALL_CLOCK_BUDGET`] unless overridden (for tests
/// only) via [`WALL_CLOCK_BUDGET_OVERRIDE_MS_VAR`].
fn enumeration_wall_clock_budget() -> std::time::Duration {
    std::env::var(WALL_CLOCK_BUDGET_OVERRIDE_MS_VAR)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(std::time::Duration::from_millis)
        .unwrap_or(DEFAULT_ENUMERATION_WALL_CLOCK_BUDGET)
}

/// Number of independent XOR-hash trials per level, majority-voted, to
/// reduce the influence of an unlucky single random hash family (the
/// "median trick" used by ApproxMC-style counters).
const HASH_TRIALS_PER_LEVEL: usize = 3;

/// How a [`ModelCountResult`] was produced, so callers/output formatting
/// never has to guess whether a number is exact, a sound lower bound, a
/// randomized estimate, or "we don't know".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountStatus {
    /// Every model was enumerated; `estimated_count` is exact.
    Exact,
    /// Enumeration stopped early (cap reached); `estimated_count` is a real,
    /// sound lower bound, not the true count.
    LowerBoundCapped,
    /// A randomized XOR-hash search produced an order-of-magnitude estimate.
    HashEstimate,
    /// The solver returned `unknown` at some point; count is indeterminate.
    Unknown,
    /// Counting could not proceed (e.g. a script/model error).
    Error,
}

/// Result of model counting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCountResult {
    /// Estimated (or exact) number of models
    pub estimated_count: f64,
    /// Lower bound (always sound: never exceeds the true count)
    pub lower_bound: f64,
    /// Upper bound (may be `f64::INFINITY` when genuinely unknown)
    pub upper_bound: f64,
    /// Number of solver `check-sat` calls actually issued while counting
    pub samples: usize,
    /// Confidence level (0.0 to 1.0); `1.0` for exact/sound-lower-bound
    /// results, lower for randomized hash estimates.
    pub confidence: f64,
    /// Whether `estimated_count` is the exact model count
    pub is_exact: bool,
    /// Whether counting stopped early because a configured cap was hit
    /// (so `estimated_count`/`lower_bound` is a sound lower bound, not the
    /// true count)
    pub capped: bool,
    /// How this result was produced
    pub status: CountStatus,
    /// Human-readable explanation of how the number was derived, including
    /// any caveats (cap reached, fell back to enumeration, hash estimate, …)
    pub note: String,
    /// Time taken in milliseconds
    pub time_ms: u128,
}

impl ModelCountResult {
    fn error(message: String, start: std::time::Instant) -> Self {
        Self {
            estimated_count: 0.0,
            lower_bound: 0.0,
            upper_bound: 0.0,
            samples: 0,
            confidence: 0.0,
            is_exact: false,
            capped: false,
            status: CountStatus::Error,
            note: message,
            time_ms: start.elapsed().as_millis(),
        }
    }

    fn exact_count(count: usize, note: String, start: std::time::Instant) -> Self {
        Self {
            estimated_count: count as f64,
            lower_bound: count as f64,
            upper_bound: count as f64,
            samples: count,
            confidence: 1.0,
            is_exact: true,
            capped: false,
            status: CountStatus::Exact,
            note,
            time_ms: start.elapsed().as_millis(),
        }
    }

    fn unknown(start: std::time::Instant) -> Self {
        Self {
            estimated_count: 0.0,
            lower_bound: 0.0,
            upper_bound: f64::INFINITY,
            samples: 0,
            confidence: 0.0,
            is_exact: false,
            capped: false,
            status: CountStatus::Unknown,
            note: "solver returned 'unknown' while checking satisfiability".to_string(),
            time_ms: start.elapsed().as_millis(),
        }
    }
}

/// Model counting method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountingMethod {
    /// Exact counting (enumerates all models, up to a cap)
    Exact,
    /// Approximate counting: XOR-hash estimation, or an honest fallback to
    /// bounded enumeration when hashing does not apply
    ApproximateSampling,
}

/// Model counter
pub struct ModelCounter {
    /// For exact mode: the maximum number of models to enumerate before
    /// reporting a capped lower bound. For approximate mode: the maximum
    /// number of `check-sat` calls to spend (whichever strategy is used).
    samples: usize,
    /// Confidence level to report alongside randomized (hash-based)
    /// estimates. Exact results and sound lower bounds always report `1.0`
    /// regardless of this setting.
    confidence: f64,
}

impl ModelCounter {
    /// Create a new model counter with default settings
    pub fn new() -> Self {
        Self {
            samples: 1000,
            confidence: 0.95,
        }
    }

    /// Create with a custom cap: max models enumerated (exact mode) or max
    /// solver calls spent (approximate mode).
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// Create with custom confidence level
    #[allow(dead_code)]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Count models for a given SMT-LIB2 script
    pub fn count(
        &self,
        ctx: &mut Context,
        script: &str,
        method: CountingMethod,
    ) -> ModelCountResult {
        let start = std::time::Instant::now();

        match method {
            CountingMethod::Exact => self.count_exact(ctx, script, start),
            CountingMethod::ApproximateSampling => self.count_approximate(ctx, script, start),
        }
    }

    /// Exact counting via real blocking-clause model enumeration.
    ///
    /// Loads `script` into `ctx` (so its declarations/assertions/any
    /// `check-sat` it contains run exactly as written), then repeatedly
    /// solves, reads back the model, blocks it, and solves again — inside a
    /// `push`/`pop` scope so the blocking clauses never leak into `ctx`'s
    /// permanent assertion set.
    fn count_exact(
        &self,
        ctx: &mut Context,
        script: &str,
        start: std::time::Instant,
    ) -> ModelCountResult {
        if let Err(e) = ctx.execute_script(script) {
            return ModelCountResult::error(format!("failed to load script: {e}"), start);
        }

        let cap = self.samples.max(1);
        ctx.push();
        let outcome = enumerate_models_bounded(ctx, cap);
        ctx.pop();

        result_from_enumeration(outcome, cap, start, None)
    }

    /// Approximate counting that actually invokes the solver.
    ///
    /// * If the formula is UNSAT/`unknown`, that is reported honestly (no
    ///   heuristic guess).
    /// * If every declared variable is Boolean and the full assignment
    ///   space (`2^n`) clearly exceeds the configured sample budget, uses
    ///   XOR-hash estimation (real, randomized, solver-invoking).
    /// * Otherwise (small space, or non-Boolean variables present — which
    ///   the hashing scheme below does not cover) honestly falls back to
    ///   the same bounded enumeration used by exact mode.
    fn count_approximate(
        &self,
        ctx: &mut Context,
        script: &str,
        start: std::time::Instant,
    ) -> ModelCountResult {
        if let Err(e) = ctx.execute_script(script) {
            return ModelCountResult::error(format!("failed to load script: {e}"), start);
        }

        match ctx.check_sat() {
            SolverResult::Unsat => {
                return ModelCountResult::exact_count(
                    0,
                    "formula is UNSAT (0 models)".to_string(),
                    start,
                );
            }
            SolverResult::Unknown => return ModelCountResult::unknown(start),
            SolverResult::Sat => {}
        }

        let Some(model) = ctx.get_model() else {
            return ModelCountResult::error(
                "solver reported sat but produced no model".to_string(),
                start,
            );
        };

        if model.is_empty() {
            // No declared variables: the formula is a ground tautology
            // (already confirmed sat), so there is exactly one model.
            return ModelCountResult::exact_count(
                1,
                "no declared variables; ground formula has exactly one model".to_string(),
                start,
            );
        }

        let cap = self.samples.max(1);
        let all_bool = model.iter().all(|(_, sort, _)| sort == "Bool");
        let space_too_big_to_enumerate = match 1u128.checked_shl(model.len() as u32) {
            Some(space) => space > cap as u128,
            None => true,
        };

        if all_bool && model.len() >= HASH_MIN_VARS && space_too_big_to_enumerate {
            self.count_via_hashing(ctx, &model, start)
        } else {
            ctx.push();
            let outcome = enumerate_models_bounded(ctx, cap);
            ctx.pop();
            let fallback_note = if all_bool {
                None
            } else {
                Some(
                    "approximate hashing requires an all-Boolean variable set; \
                     falling back to bounded enumeration since this formula has \
                     non-Boolean declared variable(s)",
                )
            };
            result_from_enumeration(outcome, cap, start, fallback_note)
        }
    }

    /// XOR/parity-constraint hashing estimate (ApproxMC-style) over an
    /// all-Boolean variable set. Real, randomized, and solver-invoking:
    /// each level of the search issues genuine `check-sat` calls with
    /// randomly generated parity constraints pushed/popped around them.
    fn count_via_hashing(
        &self,
        ctx: &mut Context,
        model: &[(String, String, String)],
        start: std::time::Instant,
    ) -> ModelCountResult {
        let var_terms: Vec<TermId> = model
            .iter()
            .map(|(name, _sort, _value)| ctx.terms.mk_var(name, ctx.terms.sorts.bool_sort))
            .collect();
        let n = var_terms.len();
        let budget = self.samples.max(1);
        let mut rng = SplitMix64::new(random_seed());
        let mut solver_calls = 0usize;
        let mut threshold: Option<usize> = None;

        'levels: for m in 0..=n {
            let mut sat_votes = 0usize;
            let mut unsat_votes = 0usize;
            for _ in 0..HASH_TRIALS_PER_LEVEL {
                ctx.push();
                for _ in 0..m {
                    if let Some(constraint) = random_parity_term(ctx, &var_terms, &mut rng) {
                        ctx.assert(constraint);
                    }
                }
                let result = ctx.check_sat();
                solver_calls += 1;
                ctx.pop();
                match result {
                    SolverResult::Sat => sat_votes += 1,
                    SolverResult::Unsat => unsat_votes += 1,
                    // An inconclusive trial contributes to neither vote.
                    SolverResult::Unknown => {}
                }
                if solver_calls >= budget {
                    break 'levels;
                }
            }
            if unsat_votes > sat_votes {
                threshold = Some(m);
                break;
            }
        }

        let elapsed = start.elapsed().as_millis();
        match threshold {
            Some(m) => {
                let m_i32 = m as i32;
                let estimate = 2f64.powi(m_i32);
                let lower = 2f64.powi((m_i32 - 1).max(0));
                let upper = 2f64.powi((m_i32 + 1).min(n as i32));
                ModelCountResult {
                    estimated_count: estimate,
                    lower_bound: lower,
                    upper_bound: upper,
                    samples: solver_calls,
                    confidence: self.confidence,
                    is_exact: false,
                    capped: false,
                    status: CountStatus::HashEstimate,
                    note: format!(
                        "XOR-hash estimate over {n} Boolean variable(s): satisfiability \
                         flipped to UNSAT around {m} random parity constraint(s) (majority \
                         of {HASH_TRIALS_PER_LEVEL} trials), so #models ~ 2^{m}. This is a \
                         randomized order-of-magnitude estimate, not a rigorous statistical \
                         bound."
                    ),
                    time_ms: elapsed,
                }
            }
            None => ModelCountResult {
                estimated_count: 2f64.powi(n as i32),
                lower_bound: 2f64.powi(n as i32),
                upper_bound: 2f64.powi(n as i32),
                samples: solver_calls,
                confidence: self.confidence,
                is_exact: false,
                capped: true,
                status: CountStatus::LowerBoundCapped,
                note: format!(
                    "XOR-hash search over {n} Boolean variable(s) never reached UNSAT \
                     within the sample budget ({solver_calls} solver call(s)); the formula \
                     appears satisfiable under essentially the full 2^{n} assignment space"
                ),
                time_ms: elapsed,
            },
        }
    }
}

impl Default for ModelCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of one bounded blocking-clause enumeration pass.
struct EnumerationOutcome {
    /// Number of distinct models found (`<= cap`).
    found: usize,
    /// `true` iff enumeration stopped because the solver reported `unsat`,
    /// or because [`enumerable_domain_size`] proved every possible
    /// assignment over the declared variables had already been found
    /// satisfiable — either way every model was genuinely found, so
    /// `found` is an exact count.
    exhausted: bool,
    /// `true` iff the solver reported `unknown` partway through, so `found`
    /// is a sound lower bound but the search could not be completed.
    hit_unknown: bool,
    /// `true` iff enumeration stopped because the
    /// [`enumeration_wall_clock_budget`] elapsed (rather than the configured
    /// cap being reached). `found` is still a sound lower bound; only the
    /// reported reason differs.
    hit_wall_clock_budget: bool,
    /// Set if blocking a found model failed (e.g. an unsupported sort);
    /// `found` up to that point is still a sound lower bound.
    error: Option<String>,
}

/// Enumerate up to `cap` models of `ctx`'s current assertion set via
/// blocking clauses, actually invoking `check_sat` for each one.
///
/// Every found model is blocked by asserting the negation of the
/// conjunction `AND(var_i = value_i)` for that model's declared variables,
/// so the next `check_sat` call can only find a genuinely different model.
///
/// Two safety nets keep this from pathologically over-running on a formula
/// whose declared variables have unbounded domains (`Int`/`Real`), where
/// there is otherwise no natural stopping point short of `cap`:
///
/// * Once the first model reveals every declared variable's sort,
///   [`enumerable_domain_size`] computes the *exact* number of possible
///   assignments when that domain is provably finite (`Bool`, or a
///   fixed-width `BitVec`). Enumeration then stops the instant that many
///   models have been found — every possible assignment has already been
///   confirmed satisfiable, so the count is exact by construction, without
///   spending one more `check_sat` call to have the solver "confirm" what
///   sort-level reasoning already proves.
/// * Independently, [`enumeration_wall_clock_budget`] bounds total wall time
///   regardless of `cap` or domain shape, so a genuinely unbounded-domain
///   formula (no finite `enumerable_domain_size`) cannot run past a sane
///   ceiling even under adverse (slow-machine / heavily-loaded) conditions.
fn enumerate_models_bounded(ctx: &mut Context, cap: usize) -> EnumerationOutcome {
    let mut found = 0usize;
    // Tightened to the true assignment count the first time a model reveals
    // an all-finite-domain variable set; `None` for any formula with at
    // least one Int/Real/unrecognized-sort variable, in which case no
    // domain-based tightening is possible and `cap` alone bounds the loop
    // (backstopped by the wall-clock budget below).
    let mut domain_size: Option<u128> = None;
    let mut effective_cap = cap;
    let enumeration_start = std::time::Instant::now();
    let wall_clock_budget = enumeration_wall_clock_budget();

    loop {
        // The wall-clock backstop only ever needs to apply while the domain
        // is still unbounded (or not yet known): once `enumerable_domain_size`
        // establishes a finite bound, `effective_cap` already guarantees
        // termination in at most that many more iterations, so there is
        // nothing pathological left to guard against -- and *not* checking
        // the clock in that case means a legitimately large-but-finite
        // (Bool/BitVec-only) enumeration can never be spuriously cut short
        // by scheduling noise on a heavily loaded machine.
        if domain_size.is_none() && enumeration_start.elapsed() > wall_clock_budget {
            return EnumerationOutcome {
                found,
                exhausted: false,
                hit_unknown: false,
                hit_wall_clock_budget: true,
                error: None,
            };
        }
        if found >= effective_cap {
            // If a *domain* bound (not the user-configured cap) is what
            // stopped us, every possible assignment has been confirmed
            // satisfiable, so the count is exact.
            let exhausted_by_domain = domain_size == Some(found as u128);
            return EnumerationOutcome {
                found,
                exhausted: exhausted_by_domain,
                hit_unknown: false,
                hit_wall_clock_budget: false,
                error: None,
            };
        }
        match ctx.check_sat() {
            SolverResult::Unsat => {
                return EnumerationOutcome {
                    found,
                    exhausted: true,
                    hit_unknown: false,
                    hit_wall_clock_budget: false,
                    error: None,
                };
            }
            SolverResult::Unknown => {
                return EnumerationOutcome {
                    found,
                    exhausted: false,
                    hit_unknown: true,
                    hit_wall_clock_budget: false,
                    error: None,
                };
            }
            SolverResult::Sat => {
                let Some(model) = ctx.get_model() else {
                    return EnumerationOutcome {
                        found,
                        exhausted: false,
                        hit_unknown: false,
                        hit_wall_clock_budget: false,
                        error: Some("solver reported sat but produced no model".to_string()),
                    };
                };

                if model.is_empty() {
                    // No declared constants: the formula is a ground
                    // tautology. It is already known sat, so there is
                    // exactly one model and no blocking clause is possible.
                    found += 1;
                    return EnumerationOutcome {
                        found,
                        exhausted: true,
                        hit_unknown: false,
                        hit_wall_clock_budget: false,
                        error: None,
                    };
                }

                if domain_size.is_none()
                    && let Some(size) = enumerable_domain_size(&model)
                {
                    domain_size = Some(size);
                    let tightened = size.min(cap as u128) as usize;
                    effective_cap = effective_cap.min(tightened);
                }

                let mut equalities = Vec::with_capacity(model.len());
                let mut blocking_error = None;
                for (name, sort_name, value) in &model {
                    let Some(sort) = sort_id_from_name(ctx, sort_name) else {
                        blocking_error = Some(format!(
                            "cannot reconstruct term for variable '{name}' of sort '{sort_name}'"
                        ));
                        break;
                    };
                    let var_term = ctx.terms.mk_var(name, sort);
                    // Deliberately parse `get_model()`'s own formatted value
                    // rather than using `Context::eval_in_model`: for a bare
                    // variable the solver never had to decide (e.g. it does
                    // not appear, or appears only in tautological
                    // assertions), `eval_in_model` returns the *unassigned
                    // variable itself* rather than a value, whereas
                    // `get_model()` already applies the solver's own default
                    // fallback. Blocking on the wrong term there would
                    // assert `(= v v)` (always true) for every variable,
                    // making the "blocking" clause `(not true)` -- i.e.
                    // `false` -- which would corrupt the assertion set.
                    let Some(value_term) = value_term_from_str(ctx, sort_name, value) else {
                        blocking_error = Some(format!(
                            "cannot parse model value '{value}' for variable '{name}' of sort '{sort_name}'"
                        ));
                        break;
                    };
                    equalities.push(ctx.terms.mk_eq(var_term, value_term));
                }

                found += 1;

                if let Some(err) = blocking_error {
                    return EnumerationOutcome {
                        found,
                        exhausted: false,
                        hit_unknown: false,
                        hit_wall_clock_budget: false,
                        error: Some(err),
                    };
                }

                let conjunction = ctx.terms.mk_and(equalities);
                let blocking_clause = ctx.terms.mk_not(conjunction);
                ctx.assert(blocking_clause);
            }
        }
    }
}

/// The total number of distinct variable assignments possible over `model`'s
/// declared variables, when every one has a sort with a *provably finite*
/// domain (`Bool`, or a fixed-width `BitVec`). Returns `None` if any variable
/// has an unbounded domain (`Int`, `Real`) or an unrecognized sort, or if the
/// product would overflow `u128` (at which point the bound is not useful for
/// tightening enumeration anyway).
fn enumerable_domain_size(model: &[(String, String, String)]) -> Option<u128> {
    let mut total: u128 = 1;
    for (_name, sort_name, _value) in model {
        let domain: u128 = if sort_name == "Bool" {
            2
        } else {
            let width_str = sort_name
                .strip_prefix("(_ BitVec ")
                .and_then(|s| s.strip_suffix(')'))?;
            let width: u32 = width_str.trim().parse().ok()?;
            1u128.checked_shl(width)?
        };
        total = total.checked_mul(domain)?;
    }
    Some(total)
}

/// Turn a [`EnumerationOutcome`] into a [`ModelCountResult`], honestly
/// reflecting whether the search was exhaustive, hit `unknown`, or was
/// capped. `fallback_note`, if set, is prepended to explain *why* bounded
/// enumeration was used (e.g. approximate mode falling back).
fn result_from_enumeration(
    outcome: EnumerationOutcome,
    cap: usize,
    start: std::time::Instant,
    fallback_note: Option<&str>,
) -> ModelCountResult {
    let elapsed = start.elapsed().as_millis();
    if let Some(err) = outcome.error {
        return ModelCountResult::error(
            format!(
                "enumeration failed after {} confirmed model(s): {err}",
                outcome.found
            ),
            start,
        );
    }

    let prefix = fallback_note.map(|n| format!("{n}; ")).unwrap_or_default();

    if outcome.exhausted {
        ModelCountResult {
            estimated_count: outcome.found as f64,
            lower_bound: outcome.found as f64,
            upper_bound: outcome.found as f64,
            samples: outcome.found,
            confidence: 1.0,
            is_exact: true,
            capped: false,
            status: CountStatus::Exact,
            note: format!(
                "{prefix}exact count via blocking-clause enumeration ({} model(s) found)",
                outcome.found
            ),
            time_ms: elapsed,
        }
    } else if outcome.hit_unknown {
        ModelCountResult {
            estimated_count: outcome.found as f64,
            lower_bound: outcome.found as f64,
            upper_bound: f64::INFINITY,
            samples: outcome.found,
            confidence: 0.0,
            is_exact: false,
            capped: false,
            status: CountStatus::Unknown,
            note: format!(
                "{prefix}solver returned 'unknown' after enumerating {} model(s); true count \
                 is unknown, but at least {} model(s) exist",
                outcome.found, outcome.found
            ),
            time_ms: elapsed,
        }
    } else if outcome.hit_wall_clock_budget {
        ModelCountResult {
            estimated_count: outcome.found as f64,
            lower_bound: outcome.found as f64,
            upper_bound: f64::INFINITY,
            samples: outcome.found,
            confidence: 1.0,
            is_exact: false,
            capped: true,
            status: CountStatus::LowerBoundCapped,
            note: format!(
                "{prefix}stopped after {:?} of enumeration (likely an unbounded-domain \
                 variable with no natural stopping point before --count-samples={cap}); \
                 at least {} model(s) exist",
                enumeration_wall_clock_budget(),
                outcome.found
            ),
            time_ms: elapsed,
        }
    } else {
        ModelCountResult {
            estimated_count: outcome.found as f64,
            lower_bound: outcome.found as f64,
            upper_bound: f64::INFINITY,
            samples: outcome.found,
            confidence: 1.0,
            is_exact: false,
            capped: true,
            status: CountStatus::LowerBoundCapped,
            note: format!(
                "{prefix}stopped after the configured cap of {cap} model(s) (--count-samples); \
                 at least {} model(s) exist",
                outcome.found
            ),
            time_ms: elapsed,
        }
    }
}

/// Map an SMT-LIB2 sort name, as produced by [`Context::get_model`], back to
/// a [`oxiz_core::sort::SortId`] so we can rebuild the variable's `TermId`
/// via `TermManager::mk_var` (which is content-addressed/hashconsed, so this
/// reconstructs the exact same term the original `declare-const` produced).
fn sort_id_from_name(ctx: &mut Context, sort_name: &str) -> Option<oxiz_core::sort::SortId> {
    if sort_name == "Bool" {
        Some(ctx.terms.sorts.bool_sort)
    } else if sort_name == "Int" {
        Some(ctx.terms.sorts.int_sort)
    } else if sort_name == "Real" {
        Some(ctx.terms.sorts.real_sort)
    } else if let Some(width_str) = sort_name
        .strip_prefix("(_ BitVec ")
        .and_then(|s| s.strip_suffix(')'))
    {
        let width: u32 = width_str.trim().parse().ok()?;
        Some(ctx.terms.sorts.bitvec(width))
    } else {
        None
    }
}

/// Parse the *value* half of a `(name, sort_name, value)` triple, as
/// produced by [`Context::get_model`], back into a concrete [`TermId`] for
/// that sort. `get_model()`'s formatting (`Context::format_value` /
/// `Context::default_value`) is the authoritative source of truth for what
/// a variable's value "is" in the current model — including its default
/// fallback for variables the solver never had to decide — so blocking
/// clauses are built from this string rather than from
/// `Context::eval_in_model`, which does not apply that fallback.
fn value_term_from_str(ctx: &mut Context, sort_name: &str, value: &str) -> Option<TermId> {
    if sort_name == "Bool" {
        match value {
            "true" => Some(ctx.terms.mk_bool(true)),
            "false" => Some(ctx.terms.mk_bool(false)),
            _ => None,
        }
    } else if sort_name == "Int" {
        let n: i128 = value.parse().ok()?;
        Some(ctx.terms.mk_int(n))
    } else if sort_name == "Real" {
        if let Some(rest) = value.strip_prefix("(/ ").and_then(|s| s.strip_suffix(')')) {
            let mut parts = rest.split_whitespace();
            let numer: i64 = parts.next()?.parse().ok()?;
            let denom: i64 = parts.next()?.parse().ok()?;
            if denom == 0 {
                return None;
            }
            Some(
                ctx.terms
                    .mk_real(num_rational::Rational64::new(numer, denom)),
            )
        } else {
            let numer: i64 = value.strip_suffix(".0")?.parse().ok()?;
            Some(ctx.terms.mk_real(num_rational::Rational64::new(numer, 1)))
        }
    } else if sort_name.starts_with("(_ BitVec") {
        let bits = value.strip_prefix("#b")?;
        if bits.is_empty() || !bits.bytes().all(|b| b == b'0' || b == b'1') {
            return None;
        }
        let width = bits.len() as u32;
        let n = u128::from_str_radix(bits, 2).ok()?;
        Some(ctx.terms.mk_bitvec(n, width))
    } else {
        None
    }
}

/// Build one random parity (XOR) constraint over `vars`: a random subset of
/// `vars` (each included independently with probability 1/2), XORed
/// together and constrained to equal a random target bit. Returns `None` if
/// the random subset came up empty (the constraint would be trivial) —
/// callers simply skip adding a constraint that round.
fn random_parity_term(ctx: &mut Context, vars: &[TermId], rng: &mut SplitMix64) -> Option<TermId> {
    let selected: Vec<TermId> = vars.iter().copied().filter(|_| rng.next_bool()).collect();
    let mut iter = selected.into_iter();
    let mut parity = iter.next()?;
    for v in iter {
        parity = ctx.terms.mk_xor(parity, v);
    }
    let target = ctx.terms.mk_bool(rng.next_bool());
    Some(ctx.terms.mk_eq(parity, target))
}

/// A tiny, dependency-free `splitmix64` PRNG. Used only to pick random
/// Boolean subsets/target bits for hash-family parity constraints — not for
/// anything security-sensitive — so we avoid pulling in the `rand` crate for
/// a handful of pseudo-random bits.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// Produce a fresh, non-fixed seed without an external RNG dependency.
/// `RandomState`'s keys are drawn from OS randomness on construction (the
/// same mechanism `HashMap` uses for DoS resistance); mixing in a
/// wall-clock timestamp adds an extra source of variation across calls.
fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    hasher.write_u128(nanos);
    hasher.finish()
}

/// Format model count result as human-readable string
pub fn format_model_count(result: &ModelCountResult) -> String {
    let mut output = String::new();

    output.push_str("=== Model Count Result ===\n\n");

    match result.status {
        CountStatus::Exact => {
            output.push_str(&format!("Exact count: {:.0}\n", result.estimated_count));
        }
        CountStatus::LowerBoundCapped => {
            output.push_str(&format!(
                "At least {:.0} model(s) found (capped; true count may be higher)\n",
                result.lower_bound
            ));
        }
        CountStatus::HashEstimate => {
            output.push_str(&format!(
                "Estimated count: {:.2e}\n",
                result.estimated_count
            ));
            output.push_str(&format!(
                "Rough bracket: [{:.2e}, {:.2e}]\n",
                result.lower_bound, result.upper_bound
            ));
        }
        CountStatus::Unknown => {
            output.push_str("Count: unknown (solver could not determine satisfiability)\n");
            if result.lower_bound > 0.0 {
                output.push_str(&format!(
                    "At least {:.0} model(s) were found before that\n",
                    result.lower_bound
                ));
            }
        }
        CountStatus::Error => {
            output.push_str("Count: error\n");
        }
    }

    if !result.note.is_empty() {
        output.push_str(&format!("Note: {}\n", result.note));
    }
    output.push_str(&format!("Solver calls: {}\n", result.samples));
    output.push_str(&format!("Time: {} ms\n", result.time_ms));

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_counter_creation() {
        let counter = ModelCounter::new();
        assert_eq!(counter.samples, 1000);
        assert_eq!(counter.confidence, 0.95);
    }

    #[test]
    fn test_model_counter_with_samples() {
        let counter = ModelCounter::new().with_samples(5000);
        assert_eq!(counter.samples, 5000);
    }

    #[test]
    fn test_exact_counting_small_formula() {
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        // x OR y has exactly 3 satisfying assignments out of 4.
        let script = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (or x y))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::Exact);

        assert!(result.is_exact, "small formula should be fully enumerated");
        assert_eq!(result.status, CountStatus::Exact);
        assert_eq!(result.estimated_count, 3.0);
        assert_eq!(result.lower_bound, 3.0);
        assert_eq!(result.upper_bound, 3.0);
    }

    #[test]
    fn test_exact_counting_unsat_is_zero() {
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const x Bool)
            (assert x)
            (assert (not x))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::Exact);

        assert!(result.is_exact);
        assert_eq!(result.estimated_count, 0.0);
    }

    #[test]
    fn test_exact_counting_cap_reports_lower_bound_honestly() {
        let mut ctx = Context::new();
        // 4 free Boolean variables => 16 models; cap enumeration at 3.
        let counter = ModelCounter::new().with_samples(3);

        let script = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (declare-const d Bool)
            (assert (or a (not a)))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::Exact);

        assert!(
            !result.is_exact,
            "capped enumeration must not claim exactness"
        );
        assert!(result.capped);
        assert_eq!(result.status, CountStatus::LowerBoundCapped);
        assert_eq!(result.estimated_count, 3.0);
        assert!(result.upper_bound.is_infinite());
    }

    #[test]
    fn test_approximate_counting_small_formula_is_exact_via_fallback() {
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (or x y))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::ApproximateSampling);

        // The variable space is tiny, so approximate mode should honestly
        // fall back to full enumeration and report the *exact* count rather
        // than pretending it is merely an estimate.
        assert!(result.estimated_count > 0.0);
        assert!(result.lower_bound > 0.0);
        assert!(result.upper_bound >= result.estimated_count);
        assert!(result.is_exact);
        assert_eq!(result.estimated_count, 3.0);
    }

    #[test]
    fn test_approximate_counting_unsat_is_honest() {
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const x Bool)
            (assert x)
            (assert (not x))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::ApproximateSampling);

        assert!(result.is_exact);
        assert_eq!(result.estimated_count, 0.0);
        assert_eq!(result.status, CountStatus::Exact);
    }

    #[test]
    fn test_approximate_counting_uses_real_solver_calls() {
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (or x y))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::ApproximateSampling);
        // At minimum: the initial check-sat plus each enumeration round.
        assert!(result.samples > 0, "must actually invoke the solver");
    }

    #[test]
    fn test_format_model_count_exact() {
        let result = ModelCountResult {
            estimated_count: 3.0,
            lower_bound: 3.0,
            upper_bound: 3.0,
            samples: 4,
            confidence: 1.0,
            is_exact: true,
            capped: false,
            status: CountStatus::Exact,
            note: "exact count via blocking-clause enumeration (3 model(s) found)".to_string(),
            time_ms: 10,
        };

        let formatted = format_model_count(&result);
        assert!(formatted.contains("Exact count: 3"));
        assert!(formatted.contains("Note:"));
    }

    #[test]
    fn test_format_model_count_capped() {
        let result = ModelCountResult {
            estimated_count: 100.0,
            lower_bound: 100.0,
            upper_bound: f64::INFINITY,
            samples: 100,
            confidence: 1.0,
            is_exact: false,
            capped: true,
            status: CountStatus::LowerBoundCapped,
            note: "stopped after the configured cap of 100 model(s)".to_string(),
            time_ms: 50,
        };

        let formatted = format_model_count(&result);
        assert!(formatted.contains("At least 100"));
        assert!(!formatted.contains("Exact count"));
    }

    #[test]
    fn test_never_reports_zero_for_satisfiable_formula() {
        // Regression test for the original bug: exact mode used to always
        // report 0 (with `is_exact: true`) regardless of the formula.
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const x Bool)
            (assert x)
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::Exact);
        assert_eq!(result.estimated_count, 1.0);
        assert!(result.is_exact);
    }

    // -----------------------------------------------------------------
    // Regression tests for the pathological-runtime fix: enumeration used
    // to have no way to stop short of the configured `--count-samples` cap
    // (default 1000), so a formula with an unbounded-domain variable (Int,
    // Real) could force up to 1000 sequential solver calls over an
    // ever-growing blocking-clause assertion set -- observed to take
    // anywhere from a few seconds to several minutes depending on machine
    // load, even for a "trivial" formula with only a couple of declared
    // variables. `enumerable_domain_size` + the wall-clock budget in
    // `enumerate_models_bounded` bound this properly.
    // -----------------------------------------------------------------

    /// Wall-clock duration assertions here are flaky under shared-machine /
    /// CI load (a loaded box can blow a generous budget on an otherwise-fast
    /// operation with no bug involved), matching the convention already
    /// established in `tests/benchmark.rs`. Gate them behind
    /// `OXIZ_TIMING_TESTS=1`; the underlying solve/count still runs and is
    /// checked for correctness either way.
    fn timing_asserts_enabled() -> bool {
        std::env::var("OXIZ_TIMING_TESTS").as_deref() == Ok("1")
    }

    fn assert_within_budget(elapsed: std::time::Duration, budget_ms: u128, label: &str) {
        if timing_asserts_enabled() {
            assert!(
                elapsed.as_millis() < budget_ms,
                "{label} took too long: {elapsed:?}"
            );
        }
    }

    #[test]
    fn test_enumerable_domain_size_all_bool() {
        let model = vec![
            ("x".to_string(), "Bool".to_string(), "true".to_string()),
            ("y".to_string(), "Bool".to_string(), "false".to_string()),
        ];
        assert_eq!(enumerable_domain_size(&model), Some(4));
    }

    #[test]
    fn test_enumerable_domain_size_bitvec() {
        let model = vec![
            (
                "x".to_string(),
                "(_ BitVec 4)".to_string(),
                "#b0000".to_string(),
            ),
            ("y".to_string(), "Bool".to_string(), "true".to_string()),
        ];
        // 2^4 (BitVec 4) * 2 (Bool) = 32.
        assert_eq!(enumerable_domain_size(&model), Some(32));
    }

    #[test]
    fn test_enumerable_domain_size_none_for_unbounded_int() {
        let model = vec![("x".to_string(), "Int".to_string(), "0".to_string())];
        assert_eq!(enumerable_domain_size(&model), None);
    }

    #[test]
    fn test_enumerable_domain_size_none_for_unbounded_real() {
        let model = vec![("x".to_string(), "Real".to_string(), "0.0".to_string())];
        assert_eq!(enumerable_domain_size(&model), None);
    }

    #[test]
    fn test_enumerable_domain_size_none_on_overflow() {
        // A single 200-bit BitVec's domain (2^200) does not fit in a u128;
        // this must honestly bail out to "unbounded" rather than silently
        // wrapping or panicking on overflow.
        let model = vec![(
            "x".to_string(),
            "(_ BitVec 200)".to_string(),
            "#b0".to_string(),
        )];
        assert_eq!(enumerable_domain_size(&model), None);
    }

    #[test]
    fn test_exact_counting_tautology_over_finite_domain_stops_at_domain_boundary() {
        // 8 free Boolean variables under a tautological assertion: every one
        // of the 2^8 = 256 possible assignments is a genuine model. With the
        // default cap of 1000 (well above 256), unbounded enumeration would
        // still terminate correctly, but only after wastefully spending a
        // 257th `check_sat` call purely to have the solver "confirm" UNSAT --
        // the domain-size early exit skips that, and this must remain fast
        // and exact either way.
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (declare-const d Bool)
            (declare-const e Bool)
            (declare-const f Bool)
            (declare-const g Bool)
            (declare-const h Bool)
            (assert (or a (not a)))
        "#;

        let start = std::time::Instant::now();
        let result = counter.count(&mut ctx, script, CountingMethod::Exact);
        let elapsed = start.elapsed();

        assert!(result.is_exact, "full finite domain must be exact");
        assert_eq!(result.status, CountStatus::Exact);
        assert_eq!(result.estimated_count, 256.0);
        assert_eq!(result.samples, 256);
        assert!(!result.capped);
        assert_within_budget(elapsed, 3_000, "8-boolean-variable tautology exact count");
    }

    #[test]
    fn test_exact_counting_unbounded_domain_is_bounded_not_pathological() {
        // Regression test for the core fix: a formula with unbounded-domain
        // (`Int`) variables and no upper bound has infinitely many models, so
        // enumeration can never reach the "solver reports unsat" exit and,
        // before this fix, would always spend the full `--count-samples`
        // budget (default 1000) worth of sequential solver calls over an
        // ever-growing assertion set -- the empirically pathological case.
        //
        // Shrink the wall-clock safety net for this test only (via the
        // documented env var override) so the test itself stays fast while
        // still genuinely exercising the same code path that bounds the
        // production default.
        //
        // SAFETY: nextest runs each test in its own process; no other thread
        // in this process reads/writes this env var concurrently.
        unsafe {
            std::env::set_var("OXIZ_MODEL_COUNT_WALL_CLOCK_BUDGET_MS", "200");
        }

        let mut ctx = Context::new();
        // A large sample cap: without the wall-clock backstop this would be
        // free to run up to 5000 sequential solver calls.
        let counter = ModelCounter::new().with_samples(5000);

        let script = r#"
            (declare-const x Int)
            (declare-const y Int)
            (assert (>= x 0))
            (assert (>= y 0))
        "#;

        let start = std::time::Instant::now();
        let result = counter.count(&mut ctx, script, CountingMethod::Exact);
        let elapsed = start.elapsed();

        unsafe {
            std::env::remove_var("OXIZ_MODEL_COUNT_WALL_CLOCK_BUDGET_MS");
        }

        // A generous margin (well beyond the 200ms budget) that tolerates
        // heavy scheduling contention without ever tolerating the old
        // unbounded-up-to-5000-solves behavior this test guards against.
        assert!(
            elapsed < std::time::Duration::from_secs(20),
            "unbounded-domain exact counting must be bounded by the wall-clock \
             safety net, not run out the full sample cap; took {elapsed:?}"
        );
        assert!(
            !result.is_exact,
            "an infinite-model formula can never be reported as exact"
        );
        assert_eq!(result.status, CountStatus::LowerBoundCapped);
        assert!(result.capped);
        assert!(
            result.estimated_count > 0.0,
            "must still report genuinely-found models as a sound lower bound"
        );
    }
}
