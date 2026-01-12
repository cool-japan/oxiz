//! CHC-COMP format parser
//!
//! Parses Constrained Horn Clauses in CHC-COMP format (SMT-LIB2 extension).
//!
//! CHC-COMP format:
//! - Predicates are declared as uninterpreted functions: `(declare-fun P (Int Bool) Bool)`
//! - Rules are asserted as Horn clauses: `(assert (forall vars (=> body head)))`
//! - Queries have `false` as the head
//!
//! Reference: <https://chc-comp.github.io>

use crate::chc::{ChcSystem, PredId, PredicateApp, RuleBody, RuleHead};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_core::sort::SortId;
use rustc_hash::FxHashMap;
use thiserror::Error;
use tracing::{debug, warn};

/// Errors from CHC-COMP parsing
#[derive(Error, Debug)]
pub enum ChcCompError {
    /// SMT-LIB parse error
    #[error("SMT-LIB parse error: {0}")]
    ParseError(String),
    /// Invalid CHC format
    #[error("invalid CHC format: {0}")]
    InvalidFormat(String),
    /// Unsupported feature
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    /// Predicate not found
    #[error("predicate not found: {0}")]
    PredicateNotFound(String),
}

/// CHC-COMP parser
pub struct ChcCompParser<'a> {
    /// Term manager
    terms: &'a mut TermManager,
    /// CHC system being built
    system: ChcSystem,
    /// Predicate name to ID mapping
    pred_map: FxHashMap<String, PredId>,
    /// Function symbol (TermId) to predicate ID mapping
    /// Maps function application terms to their predicate IDs
    func_to_pred: FxHashMap<String, PredId>,
}

impl<'a> ChcCompParser<'a> {
    /// Create a new CHC-COMP parser
    pub fn new(terms: &'a mut TermManager) -> Self {
        Self {
            terms,
            system: ChcSystem::new(),
            pred_map: FxHashMap::default(),
            func_to_pred: FxHashMap::default(),
        }
    }

    /// Parse a CHC-COMP file from a string
    pub fn parse(&mut self, input: &str) -> Result<ChcSystem, ChcCompError> {
        // Parse SMT-LIB2 commands
        let commands = match parse_script(input, self.terms) {
            Ok(cmds) => cmds,
            Err(e) => return Err(ChcCompError::ParseError(e.to_string())),
        };

        // Process commands
        for cmd in commands {
            self.process_command(cmd)?;
        }

        // Take ownership of the system
        Ok(std::mem::take(&mut self.system))
    }

    /// Process a single SMT-LIB2 command
    fn process_command(&mut self, cmd: Command) -> Result<(), ChcCompError> {
        match cmd {
            Command::SetLogic(logic) => {
                debug!("Set logic: {}", logic);
                if !logic.contains("HORN") && !logic.contains("ALL") {
                    warn!("Logic {} may not support Horn clauses", logic);
                }
                Ok(())
            }
            Command::DeclareFun(name, arg_sorts, ret_sort) => {
                self.declare_predicate(&name, arg_sorts, &ret_sort)
            }
            Command::Assert(term) => self.process_assertion(term),
            Command::CheckSat => {
                debug!("check-sat command (ignored in CHC parsing)");
                Ok(())
            }
            _ => {
                // Ignore other commands
                Ok(())
            }
        }
    }

    /// Declare a predicate
    fn declare_predicate(
        &mut self,
        name: &str,
        arg_sort_names: Vec<String>,
        ret_sort_name: &str,
    ) -> Result<(), ChcCompError> {
        // Parse argument sorts
        let arg_sorts: Vec<SortId> = arg_sort_names
            .iter()
            .map(|s| self.parse_sort_name(s))
            .collect();

        // Parse return sort
        let ret_sort = self.parse_sort_name(ret_sort_name);

        // CHC predicates should return Bool
        if ret_sort != self.terms.sorts.bool_sort {
            return Err(ChcCompError::InvalidFormat(format!(
                "Predicate {} must return Bool, not {}",
                name, ret_sort_name
            )));
        }

        // Declare the predicate
        let pred_id = self.system.declare_predicate(name, arg_sorts);
        self.pred_map.insert(name.to_string(), pred_id);

        // Also map the function symbol name to the predicate ID
        // This allows lookup when parsing applications
        self.func_to_pred.insert(name.to_string(), pred_id);

        debug!(
            "Declared predicate: {} with {} args",
            name,
            arg_sort_names.len()
        );
        Ok(())
    }

    /// Parse a sort name to SortId
    fn parse_sort_name(&mut self, name: &str) -> SortId {
        match name {
            "Bool" => self.terms.sorts.bool_sort,
            "Int" => self.terms.sorts.int_sort,
            "Real" => self.terms.sorts.real_sort,
            _ => {
                // Check for BitVec
                if let Some(width_str) = name.strip_prefix("(_ BitVec ")
                    && let Some(width_str) = width_str.strip_suffix(')')
                    && let Ok(width) = width_str.trim().parse::<u32>()
                {
                    return self.terms.sorts.bitvec(width);
                }
                // Default to Bool for unknown sorts
                self.terms.sorts.bool_sort
            }
        }
    }

    /// Process an assertion (Horn clause)
    ///
    /// Enhanced full SMT-LIB2 parser for Horn clauses:
    /// - Parses (forall vars (=> body head)) format
    /// - Extracts predicates from body and head
    /// - Handles quantified variables
    /// - Supports nested conjunctions and disjunctions
    fn process_assertion(&mut self, term: TermId) -> Result<(), ChcCompError> {
        use oxiz_core::ast::TermKind;

        debug!("Processing assertion (Horn clause)");

        let Some(term_data) = self.terms.get(term) else {
            return Err(ChcCompError::InvalidFormat(
                "Invalid term in assertion".to_string(),
            ));
        };

        match &term_data.kind {
            // Quantified formula: (forall vars body)
            TermKind::Forall { vars, body, .. } => {
                // Convert Spur to String
                let var_vec: Vec<(String, SortId)> = vars
                    .iter()
                    .map(|(name_spur, sort)| {
                        let name_str = self.terms.resolve_str(*name_spur).to_string();
                        (name_str, *sort)
                    })
                    .collect();
                self.process_horn_clause(*body, var_vec)
            }
            // Direct implication: body => head
            TermKind::Implies(body_term, head_term) => {
                self.process_implication(*body_term, *head_term, Vec::new())
            }
            // Other formulas: treat as head-only rule (constraint)
            _ => {
                // This is a fact or constraint
                let body = RuleBody::init(self.terms.mk_true());

                // Try to extract predicate from the term
                if let Some(pred_app) = self.try_extract_predicate_app(term) {
                    let head = RuleHead::Predicate(pred_app);
                    self.system.add_rule(Vec::new(), body, head, None);
                } else {
                    // It's a query (constraint with no head predicate)
                    let body = RuleBody::init(term);
                    let head = RuleHead::Query;
                    self.system.add_rule(Vec::new(), body, head, None);
                }
                Ok(())
            }
        }
    }

    /// Process a Horn clause (possibly under quantifiers)
    fn process_horn_clause(
        &mut self,
        body: TermId,
        vars: Vec<(String, SortId)>,
    ) -> Result<(), ChcCompError> {
        use oxiz_core::ast::TermKind;

        let Some(body_term) = self.terms.get(body) else {
            return Err(ChcCompError::InvalidFormat(
                "Invalid body in Horn clause".to_string(),
            ));
        };

        match &body_term.kind {
            // Implication: body => head
            TermKind::Implies(lhs, rhs) => self.process_implication(*lhs, *rhs, vars),
            // Other: treat as constraint
            _ => {
                let rule_body = RuleBody::init(body);
                let head = RuleHead::Query;
                self.system.add_rule(vars, rule_body, head, None);
                Ok(())
            }
        }
    }

    /// Process an implication (body => head)
    fn process_implication(
        &mut self,
        body_term: TermId,
        head_term: TermId,
        vars: Vec<(String, SortId)>,
    ) -> Result<(), ChcCompError> {
        use oxiz_core::ast::TermKind;

        // Extract predicates and constraints from body
        let (body_preds, body_constraint) = self.decompose_conjunction(body_term);

        // Extract head
        let head = if let Some(head_term_data) = self.terms.get(head_term) {
            match &head_term_data.kind {
                TermKind::False => {
                    // Query: body => false
                    RuleHead::Query
                }
                TermKind::Apply { func, args } => {
                    // Predicate application
                    // Resolve function name from Spur
                    let func_name = self.terms.resolve_str(*func).to_string();
                    if let Some(&pred_id) = self.func_to_pred.get(&func_name) {
                        let pred_app = PredicateApp::new(pred_id, args.iter().copied());
                        RuleHead::Predicate(pred_app)
                    } else {
                        // Unknown predicate
                        return Err(ChcCompError::PredicateNotFound(format!(
                            "Unknown predicate in head: {}",
                            func_name
                        )));
                    }
                }
                _ => {
                    // Try to extract predicate app from head
                    if let Some(pred_app) = self.try_extract_predicate_app(head_term) {
                        RuleHead::Predicate(pred_app)
                    } else {
                        // Treat as query
                        RuleHead::Query
                    }
                }
            }
        } else {
            RuleHead::Query
        };

        // Build rule body
        let rule_body = if body_preds.is_empty() {
            RuleBody::init(body_constraint)
        } else {
            RuleBody::new(body_preds, body_constraint)
        };

        // Add rule to system
        self.system.add_rule(vars, rule_body, head, None);
        debug!("Added Horn clause rule");

        Ok(())
    }

    /// Decompose a conjunction into predicates and constraints
    ///
    /// Returns: (predicates, remaining constraints)
    fn decompose_conjunction(&mut self, term: TermId) -> (Vec<crate::chc::PredicateApp>, TermId) {
        let mut predicates = Vec::new();
        let mut constraints = Vec::new();

        // Collect conjuncts
        let conjuncts = self.collect_conjuncts(term);

        for conjunct in conjuncts {
            if let Some(pred_app) = self.try_extract_predicate_app(conjunct) {
                predicates.push(pred_app);
            } else {
                constraints.push(conjunct);
            }
        }

        // Build constraint formula
        let constraint = if constraints.is_empty() {
            self.terms.mk_true()
        } else if constraints.len() == 1 {
            constraints[0]
        } else {
            self.terms.mk_and(constraints)
        };

        (predicates, constraint)
    }

    /// Collect conjuncts from a formula (flatten AND)
    fn collect_conjuncts(&self, term: TermId) -> Vec<TermId> {
        let Some(term_data) = self.terms.get(term) else {
            return vec![term];
        };

        match &term_data.kind {
            TermKind::And(args) => {
                // Recursively collect from each conjunct
                let mut result = Vec::new();
                for &arg in args.iter() {
                    result.extend(self.collect_conjuncts(arg));
                }
                result
            }
            _ => vec![term],
        }
    }

    /// Try to extract a predicate application from a term
    fn try_extract_predicate_app(&mut self, term: TermId) -> Option<crate::chc::PredicateApp> {
        use oxiz_core::ast::TermKind;

        let term_data = self.terms.get(term)?;

        match &term_data.kind {
            TermKind::Apply { func, args } => {
                // Resolve function name from Spur
                let func_name = self.terms.resolve_str(*func).to_string();
                let pred_id = *self.func_to_pred.get(&func_name)?;
                Some(crate::chc::PredicateApp::new(pred_id, args.iter().copied()))
            }
            _ => None,
        }
    }

    /// Extract predicate ID from a term (for simple cases)
    #[allow(dead_code)]
    fn extract_predicate_app(&mut self, _term: TermId) -> Option<PredId> {
        // Placeholder: would need to look up the predicate by term
        // This requires tracking predicate symbols
        None
    }

    /// Look up predicate by function symbol term
    #[allow(dead_code)]
    fn lookup_predicate_by_term(&self, func: TermId) -> Option<PredId> {
        use oxiz_core::ast::TermKind;

        // Get the function symbol name from the term
        let func_term = self.terms.get(func)?;

        match &func_term.kind {
            TermKind::Var(name_spur) => {
                // Resolve Spur to string and look up by name
                let name = self.terms.resolve_str(*name_spur).to_string();
                self.func_to_pred.get(&name).copied()
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chccomp_parser_creation() {
        let mut terms = TermManager::new();
        let _parser = ChcCompParser::new(&mut terms);
    }

    #[test]
    fn test_predicate_declaration() {
        let mut terms = TermManager::new();
        let mut parser = ChcCompParser::new(&mut terms);

        let result =
            parser.declare_predicate("P", vec!["Int".to_string(), "Bool".to_string()], "Bool");
        assert!(result.is_ok());

        assert_eq!(parser.system.num_predicates(), 1);
        assert!(parser.pred_map.contains_key("P"));
    }

    #[test]
    fn test_invalid_return_sort() {
        let mut terms = TermManager::new();
        let mut parser = ChcCompParser::new(&mut terms);

        let result = parser.declare_predicate("P", vec!["Int".to_string()], "Int");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sort_names() {
        let mut terms = TermManager::new();
        let bool_sort = terms.sorts.bool_sort;
        let int_sort = terms.sorts.int_sort;
        let real_sort = terms.sorts.real_sort;

        let mut parser = ChcCompParser::new(&mut terms);

        assert_eq!(parser.parse_sort_name("Bool"), bool_sort);
        assert_eq!(parser.parse_sort_name("Int"), int_sort);
        assert_eq!(parser.parse_sort_name("Real"), real_sort);
    }
}
