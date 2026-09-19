//! Datatype Case Analysis for QE.
//!
//! Performs systematic constructor case analysis during quantifier
//! elimination.
//!
//! ## Strategy
//!
//! For `∃ x:T. φ(x)` where `T` has constructors `{C₁, ..., Cₙ}`, the analyzer
//! produces one case per constructor,
//!
//! ```text
//! φ[x := C_i(ȳ_i)]   (with fresh arguments ȳ_i existentially bound)
//! ```
//!
//! so that the disjunction of the cases is equivalent to the original
//! quantified formula. Unlike a plain reduction, each case actually performs
//! the capture-avoiding substitution using the term manager, and fresh
//! argument variables get globally unique names so they never alias.
//!
//! ## References
//!
//! - "Datatypes with Shared Selectors" (Reynolds & Blanchette, 2017)
//! - Z3's `qe/qe_datatypes.cpp`

use crate::Sort;
use crate::ast::{TermId, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::smtlib::reserved_name;
use crate::sort::SortId;

/// Variable identifier (legacy alias, kept for API compatibility).
pub type VarId = usize;

/// Constructor identifier.
pub type ConstructorId = usize;

/// A datatype constructor.
#[derive(Debug, Clone)]
pub struct Constructor {
    /// Constructor ID.
    pub id: ConstructorId,
    /// Constructor name.
    pub name: String,
    /// Argument sorts (empty for a nullary/enumeration constructor).
    pub arg_sorts: Vec<Sort>,
}

impl Constructor {
    /// Arity (number of arguments).
    pub fn arity(&self) -> usize {
        self.arg_sorts.len()
    }
}

/// Case analysis result.
#[derive(Debug, Clone)]
pub struct CaseAnalysisResult {
    /// Cases (one per constructor). Each is `φ[x := C_i(ȳ_i)]` with the fresh
    /// arguments existentially bound; the disjunction of `cases` is equivalent
    /// to the original `∃ x. φ(x)`.
    pub cases: Vec<TermId>,
    /// Whether `cases` is a semantically complete, sound case split. This is
    /// `true` exactly when every constructor of the datatype was expanded via
    /// real substitution (which is what [`CaseAnalyzer::analyze`] now does).
    pub complete: bool,
}

/// Configuration for case analysis.
#[derive(Debug, Clone)]
pub struct CaseAnalysisConfig {
    /// Enable case pruning (eliminate impossible cases).
    pub enable_pruning: bool,
    /// Enable case merging (combine similar cases).
    pub enable_merging: bool,
    /// Maximum case depth.
    pub max_depth: usize,
}

impl Default for CaseAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_pruning: true,
            enable_merging: true,
            max_depth: 5,
        }
    }
}

/// Statistics for case analysis.
#[derive(Debug, Clone, Default)]
pub struct CaseAnalysisStats {
    /// Cases generated.
    pub cases_generated: u64,
    /// Cases pruned.
    pub cases_pruned: u64,
    /// Cases merged.
    pub cases_merged: u64,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
}

/// Case analysis engine.
#[derive(Debug)]
pub struct CaseAnalyzer {
    /// Known constructors by datatype name.
    constructors: FxHashMap<String, Vec<Constructor>>,
    /// Configuration.
    config: CaseAnalysisConfig,
    /// Statistics.
    stats: CaseAnalysisStats,
    /// Fresh-variable counter.
    next_id: usize,
}

impl CaseAnalyzer {
    /// Create a new case analyzer.
    pub fn new(config: CaseAnalysisConfig) -> Self {
        Self {
            constructors: FxHashMap::default(),
            config,
            stats: CaseAnalysisStats::default(),
            next_id: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CaseAnalysisConfig::default())
    }

    /// Register constructors for a datatype.
    pub fn register_datatype(&mut self, datatype_name: String, constructors: Vec<Constructor>) {
        self.constructors.insert(datatype_name, constructors);
    }

    /// Perform case analysis on a quantified datatype variable.
    ///
    /// `var` is the `Var` term being eliminated and `formula` the body
    /// `φ(var)`. Returns one case per constructor (see [`CaseAnalysisResult`]).
    pub fn analyze(
        &mut self,
        var: TermId,
        datatype_name: &str,
        formula: TermId,
        tm: &mut TermManager,
    ) -> CaseAnalysisResult {
        let constructors = match self.constructors.get(datatype_name) {
            Some(ctors) => ctors.clone(),
            None => {
                return CaseAnalysisResult {
                    cases: Vec::new(),
                    complete: false,
                };
            }
        };

        let dt_sort = match tm.get(var) {
            Some(t) => t.sort,
            None => {
                return CaseAnalysisResult {
                    cases: Vec::new(),
                    complete: false,
                };
            }
        };

        let mut cases = Vec::new();

        for ctor in &constructors {
            self.stats.cases_generated += 1;

            let case = self.generate_case(ctor, var, dt_sort, formula, tm);

            if self.config.enable_pruning && self.is_trivially_false(case, tm) {
                self.stats.cases_pruned += 1;
                continue;
            }

            cases.push(case);
        }

        if self.config.enable_merging {
            cases = self.merge_cases(cases);
        }

        CaseAnalysisResult {
            cases,
            complete: true,
        }
    }

    /// Generate the case `φ[x := C(ȳ)]` for a specific constructor, with the
    /// fresh arguments `ȳ` existentially bound.
    fn generate_case(
        &mut self,
        ctor: &Constructor,
        var: TermId,
        dt_sort: SortId,
        formula: TermId,
        tm: &mut TermManager,
    ) -> TermId {
        // Fresh, uniquely named arguments of the constructor's sorts.
        let mut arg_terms = Vec::with_capacity(ctor.arg_sorts.len());
        let mut binders: Vec<(String, SortId)> = Vec::with_capacity(ctor.arg_sorts.len());
        for arg_sort in &ctor.arg_sorts {
            let name = self.fresh_name();
            let arg_term = tm.mk_var(&name, arg_sort.id);
            arg_terms.push(arg_term);
            binders.push((name, arg_sort.id));
        }

        let ctor_term = tm.mk_dt_constructor(&ctor.name, arg_terms.iter().copied(), dt_sort);
        let mut subst = FxHashMap::default();
        subst.insert(var, ctor_term);
        let case = tm.substitute(formula, &subst);

        if binders.is_empty() {
            case
        } else {
            let binder_refs: Vec<(&str, SortId)> =
                binders.iter().map(|(n, s)| (n.as_str(), *s)).collect();
            tm.mk_exists(binder_refs, case)
        }
    }

    /// Generate a globally unique fresh variable name.
    ///
    /// Minted through [`reserved_name`] for the reason its sibling in
    /// `plugin.rs` documents: `!dtca0` was an ordinary SMT-LIB simple symbol a
    /// caller of the public `oxiz_core::qe::datatype` API could already hold.
    fn fresh_name(&mut self) -> String {
        let n = self.next_id;
        self.next_id += 1;
        reserved_name("dtca", &n.to_string())
    }

    /// Check if a case is trivially false.
    ///
    /// Only the syntactic constant `false` is recognized; no theory reasoning
    /// is performed, so this never prunes a case that is merely semantically
    /// unsatisfiable (which would require a solver).
    fn is_trivially_false(&self, case: TermId, tm: &TermManager) -> bool {
        matches!(
            tm.get(case).map(|t| &t.kind),
            Some(crate::ast::TermKind::False)
        )
    }

    /// Merge similar cases.
    ///
    /// No structural merging is performed (it would require semantic
    /// equivalence checks); the cases are returned unchanged.
    fn merge_cases(&mut self, cases: Vec<TermId>) -> Vec<TermId> {
        cases
    }

    /// Get constructors for a datatype.
    pub fn get_constructors(&self, datatype_name: &str) -> Option<&Vec<Constructor>> {
        self.constructors.get(datatype_name)
    }

    /// Get statistics.
    pub fn stats(&self) -> &CaseAnalysisStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CaseAnalysisStats::default();
    }
}

impl Default for CaseAnalyzer {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermKind;

    fn either_ctors() -> Vec<Constructor> {
        vec![
            Constructor {
                id: 0,
                name: "Nothing".to_string(),
                arg_sorts: vec![],
            },
            Constructor {
                id: 1,
                name: "Just".to_string(),
                arg_sorts: vec![],
            },
        ]
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = CaseAnalyzer::default_config();
        assert_eq!(analyzer.stats().cases_generated, 0);
    }

    #[test]
    fn test_register_datatype() {
        let mut analyzer = CaseAnalyzer::default_config();
        analyzer.register_datatype("Maybe".to_string(), either_ctors());
        let ctors = analyzer
            .get_constructors("Maybe")
            .expect("test operation should succeed");
        assert_eq!(ctors.len(), 2);
        assert_eq!(ctors[0].arity(), 0);
    }

    #[test]
    fn test_analyze_enum_produces_real_cases() {
        // ∃ x:Maybe. (x = Nothing). Two nullary constructors -> two cases.
        let mut tm = TermManager::new();
        let maybe = tm.sorts.mk_datatype_sort("Maybe");
        let x = tm.mk_var("x", maybe);
        let nothing = tm.mk_dt_constructor("Nothing", core::iter::empty(), maybe);
        let phi = tm.mk_eq(x, nothing);

        let mut analyzer = CaseAnalyzer::default_config();
        analyzer.register_datatype("Maybe".to_string(), either_ctors());

        let result = analyzer.analyze(x, "Maybe", phi, &mut tm);
        assert!(result.complete);
        assert_eq!(result.cases.len(), 2);
        assert_eq!(analyzer.stats().cases_generated, 2);

        // The variable must not appear free in any case (real substitution).
        for &case in &result.cases {
            assert!(!tm.free_vars(case).contains(&x));
        }

        // Case for Nothing is `Nothing = Nothing` which folds to true.
        assert!(matches!(
            tm.get(result.cases[0]).map(|t| &t.kind),
            Some(TermKind::True)
        ));
    }

    #[test]
    fn test_analyze_unknown_datatype() {
        let mut tm = TermManager::new();
        let s = tm.sorts.mk_datatype_sort("Unknown");
        let x = tm.mk_var("x", s);
        let t = tm.mk_true();
        let mut analyzer = CaseAnalyzer::default_config();
        let result = analyzer.analyze(x, "Unknown", t, &mut tm);
        assert!(!result.complete);
        assert!(result.cases.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut analyzer = CaseAnalyzer::default_config();
        analyzer.stats.cases_generated = 10;
        analyzer.stats.cases_pruned = 3;

        assert_eq!(analyzer.stats().cases_generated, 10);
        assert_eq!(analyzer.stats().cases_pruned, 3);

        analyzer.reset_stats();
        assert_eq!(analyzer.stats().cases_generated, 0);
    }
}
