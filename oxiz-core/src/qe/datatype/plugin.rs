//! Datatype Quantifier Elimination Plugin.
//!
//! Eliminates existential quantifiers over algebraic-datatype variables by
//! *constructor case splitting*:
//!
//! ```text
//! ∃ x:T. φ(x)   ≡   ⋁_i  ∃ ȳ_i.  φ[x := C_i(ȳ_i)]
//! ```
//!
//! where `C_1, ..., C_n` are the constructors of `T` and `ȳ_i` are fresh
//! variables of the argument sorts of `C_i`. For an *enumeration* datatype
//! (all constructors nullary) the residual `∃ ȳ_i` vanish and the result is
//! quantifier-free. For constructors that carry arguments the residual
//! existentials remain, except that arguments whose sort is itself a
//! registered datatype are expanded recursively up to a configurable depth
//! budget; beyond the budget the argument is left as an honest residual
//! `∃ y:T. …` rather than being dropped or fabricated.
//!
//! Substitution is capture-avoiding and every fresh argument variable is given
//! a globally unique name (the term manager hash-conses variables by name and
//! sort), so distinct constructor arguments never alias.
//!
//! Reference: Barrett et al., "A Decision Procedure for Datatypes"; Z3's
//! `qe/qe_datatype_plugin.cpp`.

use crate::Sort;
use crate::ast::{TermId, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::smtlib::reserved_name;
use crate::sort::SortId;

/// Variable identifier.
pub type VarId = usize;

/// Constructor identifier.
pub type ConstructorId = usize;

/// Datatype constructor.
#[derive(Debug, Clone)]
pub struct Constructor {
    /// Constructor ID.
    pub id: ConstructorId,
    /// Constructor name.
    pub name: String,
    /// Argument sorts.
    pub arg_sorts: Vec<Sort>,
}

/// Datatype definition.
#[derive(Debug, Clone)]
pub struct Datatype {
    /// Datatype name.
    pub name: String,
    /// Constructors.
    pub constructors: Vec<Constructor>,
}

/// Datatype constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatatypeConstraint {
    /// x = constructor(args...)
    IsConstructor(VarId, ConstructorId, Vec<VarId>),
    /// x != y
    Neq(VarId, VarId),
    /// Conjunction.
    And(Vec<DatatypeConstraint>),
    /// Disjunction.
    Or(Vec<DatatypeConstraint>),
}

/// Configuration for datatype quantifier elimination.
#[derive(Debug, Clone)]
pub struct DatatypeQeConfig {
    /// Enable case splitting.
    pub enable_case_split: bool,
    /// Maximum recursive case-split depth for datatype-typed constructor
    /// arguments (bounds expansion of recursive datatypes).
    pub max_case_depth: usize,
    /// Enable acyclicity constraints.
    pub enable_acyclicity: bool,
}

impl Default for DatatypeQeConfig {
    fn default() -> Self {
        Self {
            enable_case_split: true,
            max_case_depth: 3,
            enable_acyclicity: true,
        }
    }
}

/// Statistics for datatype quantifier elimination.
#[derive(Debug, Clone, Default)]
pub struct DatatypeQeStats {
    /// Number of quantifiers eliminated.
    pub quantifiers_eliminated: u64,
    /// Number of case splits.
    pub case_splits: u64,
    /// Fresh variables introduced.
    pub fresh_vars: u64,
}

/// Datatype quantifier elimination plugin.
#[derive(Debug)]
pub struct DatatypeQePlugin {
    /// Configuration.
    config: DatatypeQeConfig,
    /// Known datatypes.
    datatypes: FxHashMap<String, Datatype>,
    /// Fresh variable counter.
    next_var_id: VarId,
    /// Statistics.
    stats: DatatypeQeStats,
}

impl DatatypeQePlugin {
    /// Create a new datatype QE plugin.
    pub fn new(config: DatatypeQeConfig) -> Self {
        Self {
            config,
            datatypes: FxHashMap::default(),
            next_var_id: 0,
            stats: DatatypeQeStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(DatatypeQeConfig::default())
    }

    /// Register a datatype.
    pub fn register_datatype(&mut self, datatype: Datatype) {
        self.datatypes.insert(datatype.name.clone(), datatype);
    }

    /// Eliminate an existential quantifier over `var` from `formula`.
    ///
    /// `var` must be a `Var` term whose sort is the datatype named
    /// `datatype_name`. Returns a formula equivalent to `∃ var. formula`, or
    /// `None` if the datatype is not registered or case splitting is disabled.
    pub fn eliminate(
        &mut self,
        var: TermId,
        datatype_name: &str,
        formula: TermId,
        tm: &mut TermManager,
    ) -> Option<TermId> {
        let datatype = self.datatypes.get(datatype_name)?.clone();

        if !self.config.enable_case_split {
            return None;
        }

        let dt_sort = tm.get(var)?.sort;
        self.stats.quantifiers_eliminated += 1;

        let depth = self.config.max_case_depth;
        self.eliminate_via_case_split(var, &datatype, dt_sort, formula, tm, depth)
    }

    /// Eliminate `∃ var:datatype. formula` via constructor case splitting.
    fn eliminate_via_case_split(
        &mut self,
        var: TermId,
        datatype: &Datatype,
        dt_sort: SortId,
        formula: TermId,
        tm: &mut TermManager,
        depth: usize,
    ) -> Option<TermId> {
        self.stats.case_splits += 1;

        let mut disjuncts = Vec::with_capacity(datatype.constructors.len());

        for constructor in &datatype.constructors {
            // Fresh arguments for this constructor.
            let mut arg_terms = Vec::with_capacity(constructor.arg_sorts.len());
            let mut arg_names = Vec::with_capacity(constructor.arg_sorts.len());
            for arg_sort in &constructor.arg_sorts {
                let name = self.fresh_name();
                let arg_term = tm.mk_var(&name, arg_sort.id);
                arg_terms.push((arg_term, arg_sort.id));
                arg_names.push(name);
            }

            // φ[x := C_i(ȳ_i)]
            let ctor_term = tm.mk_dt_constructor(
                &constructor.name,
                arg_terms.iter().map(|(t, _)| *t),
                dt_sort,
            );
            let mut subst = FxHashMap::default();
            subst.insert(var, ctor_term);
            let mut case = tm.substitute(formula, &subst);

            // Handle residual argument existentials.
            let mut residual_binders: Vec<(String, SortId)> = Vec::new();
            for ((arg_term, arg_sort_id), name) in arg_terms.into_iter().zip(arg_names) {
                // If the argument is itself a registered datatype and we still
                // have budget, expand it recursively; otherwise leave an
                // honest residual `∃ y:T. …`.
                let arg_dt = if depth > 0 {
                    tm.sorts
                        .datatype_name(arg_sort_id)
                        .map(|s| s.to_string())
                        .and_then(|nm| self.datatypes.get(&nm).cloned())
                } else {
                    None
                };

                if let Some(arg_dt) = arg_dt
                    && let Some(elim) = self.eliminate_via_case_split(
                        arg_term,
                        &arg_dt,
                        arg_sort_id,
                        case,
                        tm,
                        depth - 1,
                    )
                {
                    case = elim;
                    continue;
                }
                residual_binders.push((name, arg_sort_id));
            }

            if !residual_binders.is_empty() {
                let binders: Vec<(&str, SortId)> = residual_binders
                    .iter()
                    .map(|(n, s)| (n.as_str(), *s))
                    .collect();
                case = tm.mk_exists(binders, case);
            }

            disjuncts.push(case);
        }

        match disjuncts.len() {
            0 => None,
            // Exactly one constructor: the single case is the whole answer.
            1 => disjuncts.into_iter().next(),
            _ => Some(tm.mk_or(disjuncts)),
        }
    }

    /// Generate a globally unique fresh variable name.
    ///
    /// Minted through [`reserved_name`], so the name carries
    /// [`oxiz_core::smtlib::RESERVED_PREFIX`] and the parser refuses its
    /// spelling outright: a backslash is in neither of SMT-LIB 2.6's two
    /// symbol forms, so a user cannot write one even inside `|…|`.  The
    /// monotonically increasing counter is what keeps (name, sort)
    /// hash-consing in the term manager from aliasing two distinct fresh
    /// arguments.
    ///
    /// This used to be `format!("!dtqe{n}")`, and its doc used to claim the
    /// prefix "cannot collide with user variables".  `!` *is* in SMT-LIB 2.6's
    /// simple-symbol character set and this parser accepts it, so
    /// `(declare-const !dtqe0 Int)` parsed and printed — the claim was false.
    /// Nothing in `oxiz-solver` calls this plugin, so it could not produce a
    /// wrong verdict, but `oxiz_core::qe::datatype` is public API and a caller
    /// with a `!dtqe0` of its own had it captured.
    fn fresh_name(&mut self) -> String {
        let n = self.next_var_id;
        self.next_var_id += 1;
        self.stats.fresh_vars += 1;
        reserved_name("dtqe", &n.to_string())
    }

    /// Extract datatype constraints on `var` from `formula`.
    ///
    /// Constraint extraction over the legacy [`VarId`]-based
    /// [`DatatypeConstraint`] model is not implemented; this returns an empty
    /// vector (honestly: "no constraints extracted"), never fabricated ones.
    /// The case-split path above does not depend on it.
    pub fn extract_constraints(&self, _formula: TermId, _var: TermId) -> Vec<DatatypeConstraint> {
        Vec::new()
    }

    /// Get statistics.
    pub fn stats(&self) -> &DatatypeQeStats {
        &self.stats
    }

    /// Reset plugin state.
    pub fn reset(&mut self) {
        self.stats = DatatypeQeStats::default();
        self.next_var_id = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermKind;
    use crate::sort::SortKind;

    /// Evaluate a *ground* datatype formula built from constructors,
    /// equalities, testers, and Boolean connectives.
    fn ctor_name(term: TermId, tm: &TermManager) -> Option<String> {
        match &tm.get(term)?.kind {
            TermKind::DtConstructor { constructor, args } if args.is_empty() => {
                Some(tm.resolve_str(*constructor).to_string())
            }
            _ => None,
        }
    }

    fn eval_dt(term: TermId, tm: &TermManager) -> Option<bool> {
        match &tm.get(term)?.kind {
            TermKind::True => Some(true),
            TermKind::False => Some(false),
            TermKind::And(args) => {
                let args: Vec<TermId> = args.iter().copied().collect();
                let mut acc = true;
                for a in args {
                    acc &= eval_dt(a, tm)?;
                }
                Some(acc)
            }
            TermKind::Or(args) => {
                let args: Vec<TermId> = args.iter().copied().collect();
                let mut acc = false;
                for a in args {
                    acc |= eval_dt(a, tm)?;
                }
                Some(acc)
            }
            TermKind::Not(a) => Some(!eval_dt(*a, tm)?),
            TermKind::Eq(a, b) => Some(ctor_name(*a, tm)? == ctor_name(*b, tm)?),
            TermKind::DtTester { constructor, arg } => {
                Some(ctor_name(*arg, tm)? == tm.resolve_str(*constructor))
            }
            _ => None,
        }
    }

    fn color_datatype() -> Datatype {
        Datatype {
            name: "Color".to_string(),
            constructors: vec![
                Constructor {
                    id: 0,
                    name: "Red".to_string(),
                    arg_sorts: vec![],
                },
                Constructor {
                    id: 1,
                    name: "Green".to_string(),
                    arg_sorts: vec![],
                },
                Constructor {
                    id: 2,
                    name: "Blue".to_string(),
                    arg_sorts: vec![],
                },
            ],
        }
    }

    #[test]
    fn test_plugin_creation() {
        let plugin = DatatypeQePlugin::default_config();
        assert_eq!(plugin.stats().quantifiers_eliminated, 0);
    }

    #[test]
    fn test_register_datatype() {
        let mut plugin = DatatypeQePlugin::default_config();
        plugin.register_datatype(color_datatype());
        assert!(plugin.datatypes.contains_key("Color"));
    }

    #[test]
    fn test_fresh_name() {
        let mut plugin = DatatypeQePlugin::default_config();
        let v1 = plugin.fresh_name();
        let v2 = plugin.fresh_name();
        assert_eq!(v1, "!dtqe0");
        assert_eq!(v2, "!dtqe1");
        assert_eq!(plugin.stats().fresh_vars, 2);
    }

    #[test]
    fn test_enum_eliminate_true_case() {
        // ∃ x:Color. (x = Red)  ≡  true.
        let mut tm = TermManager::new();
        let color = tm.sorts.mk_datatype_sort("Color");
        let x = tm.mk_var("x", color);
        let red = tm.mk_dt_constructor("Red", core::iter::empty(), color);
        let phi = tm.mk_eq(x, red);

        let mut plugin = DatatypeQePlugin::default_config();
        plugin.register_datatype(color_datatype());

        let result = plugin
            .eliminate(x, "Color", phi, &mut tm)
            .expect("elimination should succeed");

        // The quantified variable must be gone.
        assert!(!tm.free_vars(result).contains(&x));
        // Ground result evaluates to the semantic value of ∃x. (x=Red) = true.
        assert_eq!(eval_dt(result, &tm), Some(true));
    }

    #[test]
    fn test_enum_eliminate_false_case() {
        // ∃ x:Color. (x = Red ∧ x = Green)  ≡  false.
        let mut tm = TermManager::new();
        let color = tm.sorts.mk_datatype_sort("Color");
        let x = tm.mk_var("x", color);
        let red = tm.mk_dt_constructor("Red", core::iter::empty(), color);
        let green = tm.mk_dt_constructor("Green", core::iter::empty(), color);
        let eq_red = tm.mk_eq(x, red);
        let eq_green = tm.mk_eq(x, green);
        let phi = tm.mk_and([eq_red, eq_green]);

        let mut plugin = DatatypeQePlugin::default_config();
        plugin.register_datatype(color_datatype());

        let result = plugin
            .eliminate(x, "Color", phi, &mut tm)
            .expect("elimination should succeed");

        assert!(!tm.free_vars(result).contains(&x));
        assert_eq!(eval_dt(result, &tm), Some(false));
    }

    #[test]
    fn test_enum_eliminate_tester() {
        // ∃ x:Color. is_Green(x)  ≡  true.
        let mut tm = TermManager::new();
        let color = tm.sorts.mk_datatype_sort("Color");
        let x = tm.mk_var("x", color);
        let phi = tm.mk_dt_tester("Green", x);

        let mut plugin = DatatypeQePlugin::default_config();
        plugin.register_datatype(color_datatype());

        let result = plugin
            .eliminate(x, "Color", phi, &mut tm)
            .expect("elimination should succeed");

        assert!(!tm.free_vars(result).contains(&x));
        assert_eq!(eval_dt(result, &tm), Some(true));
    }

    #[test]
    fn test_constructor_with_args_eliminates_var() {
        // Pair datatype: mk(a: Int, b: Int). ∃ x:Pair. (x = mk(fst)).
        // Only checks the target variable is eliminated and a residual
        // existential over the fresh args is produced.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let int_kind = SortKind::Int;
        let pair = tm.sorts.mk_datatype_sort("Pair");
        let x = tm.mk_var("x", pair);
        // A concrete pair mk(1,2)
        let one = tm.mk_int(1);
        let two = tm.mk_int(2);
        let concrete = tm.mk_dt_constructor("mk", [one, two], pair);
        let phi = tm.mk_eq(x, concrete);

        let pair_dt = Datatype {
            name: "Pair".to_string(),
            constructors: vec![Constructor {
                id: 0,
                name: "mk".to_string(),
                arg_sorts: vec![
                    Sort {
                        id: int_sort,
                        kind: int_kind.clone(),
                    },
                    Sort {
                        id: int_sort,
                        kind: int_kind,
                    },
                ],
            }],
        };

        let mut plugin = DatatypeQePlugin::default_config();
        plugin.register_datatype(pair_dt);

        let result = plugin
            .eliminate(x, "Pair", phi, &mut tm)
            .expect("elimination should succeed");

        // x eliminated.
        assert!(!tm.free_vars(result).contains(&x));
        // Result is a residual existential over the fresh arguments.
        assert!(matches!(
            tm.get(result).map(|t| &t.kind),
            Some(TermKind::Exists { .. })
        ));
    }
}
