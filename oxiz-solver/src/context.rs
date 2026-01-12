//! Solver context

use crate::solver::{Solver, SolverResult};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::Result;
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_core::sort::SortId;

/// A declared constant
#[derive(Debug, Clone)]
struct DeclaredConst {
    /// The term ID for this constant
    term: TermId,
    /// The sort of this constant
    sort: SortId,
    /// The name of this constant
    name: String,
}

/// A declared function
#[derive(Debug, Clone)]
struct DeclaredFun {
    /// The function name
    name: String,
    /// Argument sorts
    arg_sorts: Vec<SortId>,
    /// Return sort
    ret_sort: SortId,
}

/// Solver context for managing the solving process
///
/// The `Context` provides a high-level API for SMT solving, similar to
/// the SMT-LIB2 standard. It manages declarations, assertions, and solver state.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use oxiz_solver::Context;
///
/// let mut ctx = Context::new();
/// ctx.set_logic("QF_UF");
///
/// // Declare boolean constants
/// let p = ctx.declare_const("p", ctx.terms.sorts.bool_sort);
/// let q = ctx.declare_const("q", ctx.terms.sorts.bool_sort);
///
/// // Assert p AND q
/// let formula = ctx.terms.mk_and(vec![p, q]);
/// ctx.assert(formula);
///
/// // Check satisfiability
/// ctx.check_sat();
/// ```
///
/// ## SMT-LIB2 Script Execution
///
/// ```
/// use oxiz_solver::Context;
///
/// let mut ctx = Context::new();
///
/// let script = r#"
/// (set-logic QF_LIA)
/// (declare-const x Int)
/// (assert (>= x 0))
/// (assert (<= x 10))
/// (check-sat)
/// "#;
///
/// let _ = ctx.execute_script(script);
/// ```
#[derive(Debug)]
pub struct Context {
    /// Term manager
    pub terms: TermManager,
    /// Solver instance
    solver: Solver,
    /// Current logic
    logic: Option<String>,
    /// Assertions
    assertions: Vec<TermId>,
    /// Assertion stack for push/pop
    assertion_stack: Vec<usize>,
    /// Declared constants
    declared_consts: Vec<DeclaredConst>,
    /// Declared constants stack for push/pop
    const_stack: Vec<usize>,
    /// Mapping from constant names to indices (for efficient removal)
    const_name_to_index: std::collections::HashMap<String, usize>,
    /// Declared functions
    declared_funs: Vec<DeclaredFun>,
    /// Declared functions stack for push/pop
    fun_stack: Vec<usize>,
    /// Mapping from function names to indices
    fun_name_to_index: std::collections::HashMap<String, usize>,
    /// Last check-sat result
    last_result: Option<SolverResult>,
    /// Options
    options: std::collections::HashMap<String, String>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Create a new context
    #[must_use]
    pub fn new() -> Self {
        Self {
            terms: TermManager::new(),
            solver: Solver::new(),
            logic: None,
            assertions: Vec::new(),
            assertion_stack: Vec::new(),
            declared_consts: Vec::new(),
            const_stack: Vec::new(),
            const_name_to_index: std::collections::HashMap::new(),
            declared_funs: Vec::new(),
            fun_stack: Vec::new(),
            fun_name_to_index: std::collections::HashMap::new(),
            last_result: None,
            options: std::collections::HashMap::new(),
        }
    }

    /// Declare a constant
    pub fn declare_const(&mut self, name: &str, sort: SortId) -> TermId {
        let term = self.terms.mk_var(name, sort);
        let index = self.declared_consts.len();
        self.declared_consts.push(DeclaredConst {
            term,
            sort,
            name: name.to_string(),
        });
        self.const_name_to_index.insert(name.to_string(), index);
        term
    }

    /// Declare a function
    ///
    /// Registers a function signature in the context. For nullary functions (constants),
    /// use `declare_const` instead.
    pub fn declare_fun(&mut self, name: &str, arg_sorts: Vec<SortId>, ret_sort: SortId) {
        let index = self.declared_funs.len();
        self.declared_funs.push(DeclaredFun {
            name: name.to_string(),
            arg_sorts,
            ret_sort,
        });
        self.fun_name_to_index.insert(name.to_string(), index);
    }

    /// Get function signature if it exists
    pub fn get_fun_signature(&self, name: &str) -> Option<(Vec<SortId>, SortId)> {
        self.fun_name_to_index.get(name).and_then(|&idx| {
            self.declared_funs
                .get(idx)
                .map(|f| (f.arg_sorts.clone(), f.ret_sort))
        })
    }

    /// Set the logic
    pub fn set_logic(&mut self, logic: &str) {
        self.logic = Some(logic.to_string());
        self.solver.set_logic(logic);
    }

    /// Get the current logic
    #[must_use]
    pub fn logic(&self) -> Option<&str> {
        self.logic.as_deref()
    }

    /// Add an assertion
    pub fn assert(&mut self, term: TermId) {
        self.assertions.push(term);
        self.solver.assert(term, &mut self.terms);
    }

    /// Check satisfiability
    pub fn check_sat(&mut self) -> SolverResult {
        let result = self.solver.check(&mut self.terms);
        self.last_result = Some(result);
        result
    }

    /// Get the model (if SAT)
    /// Returns a list of (name, sort, value) tuples
    pub fn get_model(&self) -> Option<Vec<(String, String, String)>> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }

        let mut model = Vec::new();
        let solver_model = self.solver.model()?;

        for decl in &self.declared_consts {
            let value = if let Some(val) = solver_model.get(decl.term) {
                self.format_value(val)
            } else {
                // Default value based on sort
                self.default_value(decl.sort)
            };
            let sort_name = self.format_sort_name(decl.sort);
            model.push((decl.name.clone(), sort_name, value));
        }

        Some(model)
    }

    /// Format a sort ID to its SMT-LIB2 name
    fn format_sort_name(&self, sort: SortId) -> String {
        if sort == self.terms.sorts.bool_sort {
            "Bool".to_string()
        } else if sort == self.terms.sorts.int_sort {
            "Int".to_string()
        } else if sort == self.terms.sorts.real_sort {
            "Real".to_string()
        } else if let Some(s) = self.terms.sorts.get(sort) {
            if let Some(w) = s.bitvec_width() {
                format!("(_ BitVec {})", w)
            } else {
                "Unknown".to_string()
            }
        } else {
            "Unknown".to_string()
        }
    }

    /// Format a model value
    fn format_value(&self, term: TermId) -> String {
        match self.terms.get(term).map(|t| &t.kind) {
            Some(TermKind::True) => "true".to_string(),
            Some(TermKind::False) => "false".to_string(),
            Some(TermKind::IntConst(n)) => n.to_string(),
            Some(TermKind::RealConst(r)) => {
                if *r.denom() == 1 {
                    format!("{}.0", r.numer())
                } else {
                    format!("(/ {} {})", r.numer(), r.denom())
                }
            }
            Some(TermKind::BitVecConst { value, width }) => {
                format!(
                    "#b{:0>width$}",
                    format!("{:b}", value),
                    width = *width as usize
                )
            }
            _ => "?".to_string(),
        }
    }

    /// Get a default value for a sort
    fn default_value(&self, sort: SortId) -> String {
        if sort == self.terms.sorts.bool_sort {
            "false".to_string()
        } else if sort == self.terms.sorts.int_sort {
            "0".to_string()
        } else if sort == self.terms.sorts.real_sort {
            "0.0".to_string()
        } else if let Some(s) = self.terms.sorts.get(sort) {
            if let Some(w) = s.bitvec_width() {
                format!("#b{:0>width$}", "0", width = w as usize)
            } else {
                "?".to_string()
            }
        } else {
            "?".to_string()
        }
    }

    /// Format the model as SMT-LIB2
    pub fn format_model(&self) -> String {
        match self.get_model() {
            None => "(error \"No model available\")".to_string(),
            Some(model) if model.is_empty() => "(model)".to_string(),
            Some(model) => {
                let mut lines = vec!["(model".to_string()];
                for (name, sort, value) in model {
                    lines.push(format!("  (define-fun {} () {} {})", name, sort, value));
                }
                lines.push(")".to_string());
                lines.join("\n")
            }
        }
    }

    /// Push a context level
    pub fn push(&mut self) {
        self.assertion_stack.push(self.assertions.len());
        self.const_stack.push(self.declared_consts.len());
        self.fun_stack.push(self.declared_funs.len());
        self.solver.push();
    }

    /// Pop a context level with incremental declaration removal
    pub fn pop(&mut self) {
        if let Some(len) = self.assertion_stack.pop() {
            self.assertions.truncate(len);
            if let Some(const_len) = self.const_stack.pop() {
                // Remove constants from the name-to-index mapping
                while self.declared_consts.len() > const_len {
                    if let Some(decl) = self.declared_consts.pop() {
                        self.const_name_to_index.remove(&decl.name);
                    }
                }
            }
            if let Some(fun_len) = self.fun_stack.pop() {
                // Remove functions from the name-to-index mapping
                while self.declared_funs.len() > fun_len {
                    if let Some(decl) = self.declared_funs.pop() {
                        self.fun_name_to_index.remove(&decl.name);
                    }
                }
            }
            self.solver.pop();
        }
    }

    /// Reset the context
    pub fn reset(&mut self) {
        self.solver.reset();
        self.assertions.clear();
        self.assertion_stack.clear();
        self.declared_consts.clear();
        self.const_stack.clear();
        self.const_name_to_index.clear();
        self.declared_funs.clear();
        self.fun_stack.clear();
        self.fun_name_to_index.clear();
        self.logic = None;
        self.last_result = None;
        self.options.clear();
    }

    /// Reset assertions (keep declarations and options)
    pub fn reset_assertions(&mut self) {
        self.solver.reset();
        self.assertions.clear();
        self.assertion_stack.clear();
        // Keep declared_consts, const_stack, const_name_to_index,
        // declared_funs, fun_stack, and fun_name_to_index
        // Re-assert nothing - solver is fresh
        self.last_result = None;
    }

    /// Get all current assertions
    #[must_use]
    pub fn get_assertions(&self) -> &[TermId] {
        &self.assertions
    }

    /// Format assertions as SMT-LIB2
    pub fn format_assertions(&self) -> String {
        if self.assertions.is_empty() {
            return "()".to_string();
        }
        let printer = oxiz_core::smtlib::Printer::new(&self.terms);
        let mut parts = Vec::new();
        for &term in &self.assertions {
            parts.push(printer.print_term(term));
        }
        format!("({})", parts.join("\n "))
    }

    /// Set an option
    pub fn set_option(&mut self, key: &str, value: &str) {
        self.options.insert(key.to_string(), value.to_string());

        // Handle special options that affect the solver
        match key {
            "produce-proofs" => {
                let mut config = self.solver.config().clone();
                config.proof = value == "true";
                self.solver.set_config(config);
            }
            "produce-unsat-cores" => {
                self.solver.set_produce_unsat_cores(value == "true");
            }
            _ => {}
        }
    }

    /// Get an option
    #[must_use]
    pub fn get_option(&self, key: &str) -> Option<&str> {
        self.options.get(key).map(String::as_str)
    }

    /// Format an option value
    fn format_option(&self, key: &str) -> String {
        match self.get_option(key) {
            Some(val) => val.to_string(),
            None => {
                // Return default values for well-known options
                match key {
                    "produce-models" => "false".to_string(),
                    "produce-unsat-cores" => "false".to_string(),
                    "produce-proofs" => "false".to_string(),
                    "produce-assignments" => "false".to_string(),
                    "print-success" => "true".to_string(),
                    _ => "unsupported".to_string(),
                }
            }
        }
    }

    /// Get assignment (for propositional variables with :named attribute)
    /// Returns an empty list as we don't track named literals yet
    pub fn get_assignment(&self) -> String {
        "()".to_string()
    }

    /// Get proof (if proof generation is enabled and result is unsat)
    pub fn get_proof(&self) -> String {
        if self.last_result != Some(SolverResult::Unsat) {
            return "(error \"Proof is only available after unsat result\")".to_string();
        }

        match self.solver.get_proof() {
            Some(proof) => proof.format(),
            None => {
                "(error \"Proof generation not enabled. Set :produce-proofs to true\")".to_string()
            }
        }
    }

    /// Get solver statistics
    /// Returns statistics about the last solving run
    pub fn get_statistics(&self) -> String {
        let stats = self.solver.get_statistics();
        format!(
            "(:decisions {} :conflicts {} :propagations {} :restarts {} :learned-clauses {} :theory-propagations {} :theory-conflicts {})",
            stats.decisions,
            stats.conflicts,
            stats.propagations,
            stats.restarts,
            stats.learned_clauses,
            stats.theory_propagations,
            stats.theory_conflicts
        )
    }

    /// Parse a sort name and return its SortId
    fn parse_sort_name(&mut self, name: &str) -> SortId {
        match name {
            "Bool" => self.terms.sorts.bool_sort,
            "Int" => self.terms.sorts.int_sort,
            "Real" => self.terms.sorts.real_sort,
            _ => {
                // Check for BitVec
                if let Some(width_str) = name.strip_prefix("BitVec")
                    && let Ok(width) = width_str.trim().parse::<u32>()
                {
                    return self.terms.sorts.bitvec(width);
                }
                // Default to Bool for unknown sorts
                self.terms.sorts.bool_sort
            }
        }
    }

    /// Execute an SMT-LIB2 script
    pub fn execute_script(&mut self, script: &str) -> Result<Vec<String>> {
        let commands = parse_script(script, &mut self.terms)?;
        let mut output = Vec::new();

        for cmd in commands {
            match cmd {
                Command::SetLogic(logic) => {
                    self.set_logic(&logic);
                }
                Command::DeclareConst(name, sort_name) => {
                    let sort = self.parse_sort_name(&sort_name);
                    self.declare_const(&name, sort);
                }
                Command::DeclareFun(name, arg_sorts, ret_sort) => {
                    // Treat nullary functions as constants
                    if arg_sorts.is_empty() {
                        let sort = self.parse_sort_name(&ret_sort);
                        self.declare_const(&name, sort);
                    } else {
                        // Parse argument sorts and return sort
                        let parsed_arg_sorts: Vec<SortId> =
                            arg_sorts.iter().map(|s| self.parse_sort_name(s)).collect();
                        let parsed_ret_sort = self.parse_sort_name(&ret_sort);
                        self.declare_fun(&name, parsed_arg_sorts, parsed_ret_sort);
                    }
                }
                Command::Assert(term) => {
                    self.assert(term);
                }
                Command::CheckSat => {
                    let result = self.check_sat();
                    output.push(match result {
                        SolverResult::Sat => "sat".to_string(),
                        SolverResult::Unsat => "unsat".to_string(),
                        SolverResult::Unknown => "unknown".to_string(),
                    });
                }
                Command::Push(n) => {
                    for _ in 0..n {
                        self.push();
                    }
                }
                Command::Pop(n) => {
                    for _ in 0..n {
                        self.pop();
                    }
                }
                Command::Reset => {
                    self.reset();
                }
                Command::ResetAssertions => {
                    self.reset_assertions();
                }
                Command::Exit => {
                    break;
                }
                Command::Echo(msg) => {
                    output.push(msg);
                }
                Command::GetModel => {
                    output.push(self.format_model());
                }
                Command::GetAssertions => {
                    output.push(self.format_assertions());
                }
                Command::GetAssignment => {
                    output.push(self.get_assignment());
                }
                Command::GetProof => {
                    output.push(self.get_proof());
                }
                Command::GetOption(key) => {
                    output.push(self.format_option(&key));
                }
                Command::SetOption(key, value) => {
                    self.set_option(&key, &value);
                }
                Command::CheckSatAssuming(assumptions) => {
                    // For now, we push, assert all assumptions, check, then pop
                    self.push();
                    for assumption in assumptions {
                        self.assert(assumption);
                    }
                    let result = self.check_sat();
                    self.pop();
                    output.push(match result {
                        SolverResult::Sat => "sat".to_string(),
                        SolverResult::Unsat => "unsat".to_string(),
                        SolverResult::Unknown => "unknown".to_string(),
                    });
                }
                Command::Simplify(term) => {
                    // Simplify and output the term
                    let simplified = self.terms.simplify(term);
                    let printer = oxiz_core::smtlib::Printer::new(&self.terms);
                    output.push(printer.print_term(simplified));
                }
                Command::GetUnsatCore => {
                    if let Some(core) = self.solver.get_unsat_core() {
                        if core.names.is_empty() {
                            output.push("()".to_string());
                        } else {
                            output.push(format!("({})", core.names.join(" ")));
                        }
                    } else {
                        output.push("(error \"No unsat core available\")".to_string());
                    }
                }
                Command::GetValue(terms) => {
                    if self.last_result != Some(SolverResult::Sat) {
                        output.push("(error \"No model available\")".to_string());
                    } else if let Some(model) = self.solver.model() {
                        let mut values = Vec::new();
                        for term in terms {
                            // Evaluate the term in the model first
                            let value = model.eval(term, &mut self.terms);
                            // Then create printer and format
                            let printer = oxiz_core::smtlib::Printer::new(&self.terms);
                            let term_str = printer.print_term(term);
                            let value_str = printer.print_term(value);
                            values.push(format!("({} {})", term_str, value_str));
                        }
                        output.push(format!("({})", values.join("\n ")));
                    } else {
                        output.push("(error \"No model available\")".to_string());
                    }
                }
                Command::GetInfo(keyword) => {
                    // Handle get-info command
                    if keyword == ":all-statistics" {
                        output.push(self.get_statistics());
                    } else {
                        output.push(format!("(error \"Unsupported info keyword: {}\")", keyword));
                    }
                }
                Command::SetInfo(_, _)
                | Command::DeclareSort(_, _)
                | Command::DefineSort(_, _, _)
                | Command::DefineFun(_, _, _, _)
                | Command::DeclareDatatype { .. } => {
                    // Ignore these commands for now
                }
            }
        }

        Ok(output)
    }

    /// Get solver statistics
    #[must_use]
    pub fn stats(&self) -> &oxiz_sat::SolverStats {
        self.solver.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_basic() {
        let mut ctx = Context::new();

        ctx.set_logic("QF_UF");
        assert_eq!(ctx.logic(), Some("QF_UF"));

        let t = ctx.terms.mk_true();
        ctx.assert(t);

        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_context_push_pop() {
        let mut ctx = Context::new();

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        ctx.push();

        let f = ctx.terms.mk_false();
        ctx.assert(f);

        // Should be unsat with false asserted
        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Unsat);

        ctx.pop();

        // After pop, should be sat again
        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_execute_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (check-sat)
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output, vec!["sat"]);
    }

    #[test]
    fn test_declare_const() {
        let mut ctx = Context::new();

        let bool_sort = ctx.terms.sorts.bool_sort;
        let int_sort = ctx.terms.sorts.int_sort;

        ctx.declare_const("x", bool_sort);
        ctx.declare_const("y", int_sort);

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        let result = ctx.check_sat();
        assert_eq!(result, SolverResult::Sat);

        // Model should include both constants
        let model = ctx.get_model();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_format_model() {
        let mut ctx = Context::new();

        let bool_sort = ctx.terms.sorts.bool_sort;
        ctx.declare_const("p", bool_sort);

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        let _ = ctx.check_sat();

        let model_str = ctx.format_model();
        assert!(model_str.contains("(model"));
        assert!(model_str.contains("define-fun p () Bool"));
    }

    #[test]
    fn test_get_model_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Bool)
            (assert true)
            (check-sat)
            (get-model)
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "sat");
        assert!(output[1].contains("(model"));
        assert!(output[1].contains("Int"));
        assert!(output[1].contains("Bool"));
    }

    #[test]
    fn test_push_pop_consts() {
        let mut ctx = Context::new();

        let bool_sort = ctx.terms.sorts.bool_sort;
        ctx.declare_const("a", bool_sort);
        ctx.push();
        ctx.declare_const("b", bool_sort);

        let t = ctx.terms.mk_true();
        ctx.assert(t);
        let _ = ctx.check_sat();

        let model = ctx.get_model().unwrap();
        assert_eq!(model.len(), 2);

        ctx.pop();
        let _ = ctx.check_sat();

        let model = ctx.get_model().unwrap();
        assert_eq!(model.len(), 1);
        assert_eq!(model[0].0, "a");
    }

    #[test]
    fn test_get_assertions() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (assert (not p))
            (get-assertions)
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].starts_with('('));
        // Should contain both assertions
        assert!(output[0].contains("p"));
    }

    #[test]
    fn test_check_sat_assuming_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (declare-const q Bool)
            (assert p)
            (check-sat-assuming (q))
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], "sat");
    }

    #[test]
    fn test_get_option_script() {
        let mut ctx = Context::new();

        let script = r#"
            (set-option :produce-models true)
            (get-option :produce-models)
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], "true");
    }

    #[test]
    fn test_reset_assertions() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (reset-assertions)
            (get-assertions)
            (check-sat)
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "()"); // No assertions after reset
        assert_eq!(output[1], "sat"); // Empty formula is SAT
    }

    #[test]
    fn test_simplify_command() {
        let mut ctx = Context::new();

        let script = r#"
            (simplify (+ 1 2))
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 1);
        // Should simplify to 3
        assert_eq!(output[0], "3");
    }

    #[test]
    fn test_simplify_complex() {
        let mut ctx = Context::new();

        let script = r#"
            (simplify (* 2 3 4))
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 1);
        // Should simplify to 24
        assert_eq!(output[0], "24");
    }

    #[test]
    fn test_get_value() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (declare-const q Bool)
            (assert p)
            (assert (not q))
            (check-sat)
            (get-value (p q (and p q) (or p q)))
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "sat");

        // Parse the get-value output
        let value_output = &output[1];
        assert!(value_output.contains("p"));
        assert!(value_output.contains("q"));
        // p should evaluate to true
        assert!(value_output.contains("true"));
        // q should evaluate to false
        assert!(value_output.contains("false"));
    }

    #[test]
    fn test_get_value_no_model() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (get-value (p))
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].contains("error") || output[0].contains("No model"));
    }

    #[test]
    fn test_get_value_after_unsat() {
        let mut ctx = Context::new();

        let script = r#"
            (set-logic QF_UF)
            (declare-const p Bool)
            (assert p)
            (assert (not p))
            (check-sat)
            (get-value (p))
        "#;

        let output = ctx.execute_script(script).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "unsat");
        assert!(output[1].contains("error") || output[1].contains("No model"));
    }
}
