//! Model Completion Algorithms
//!
//! This module implements model completion for MBQI. Model completion is the process
//! of taking a partial model (which may only define values for some terms) and
//! completing it to a total model that assigns values to all terms.
//!
//! The key challenge is handling function symbols and uninterpreted sorts, which may
//! have infinitely many possible interpretations. We use several strategies:
//!
//! 1. **Macro Solving**: Identify quantifiers that can be solved as macros
//! 2. **Projection Functions**: Map infinite domains to finite representatives
//! 3. **Default Values**: Assign sensible defaults for undefined terms
//! 4. **Finite Universes**: Restrict uninterpreted sorts to finite sets
//!
//! # References
//!
//! - Z3's model_fixer.cpp and q_model_fixer.cpp
//! - "Complete Quantifier Instantiation" (Ge & de Moura, 2009)

#![allow(missing_docs)]
#![allow(dead_code)]

use lasso::Spur;
use num_bigint::BigInt;
use num_rational::Rational64;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt;

use super::QuantifiedFormula;

/// A completed model that assigns values to all relevant terms
#[derive(Debug, Clone)]
pub struct CompletedModel {
    /// Term assignments (term -> value)
    pub assignments: FxHashMap<TermId, TermId>,
    /// Function interpretations
    pub function_interps: FxHashMap<Spur, FunctionInterpretation>,
    /// Universes for uninterpreted sorts (sort -> finite set of values)
    pub universes: FxHashMap<SortId, Vec<TermId>>,
    /// Default values for each sort
    pub defaults: FxHashMap<SortId, TermId>,
    /// Generation number
    pub generation: u32,
}

impl CompletedModel {
    /// Create a new empty completed model
    pub fn new() -> Self {
        Self {
            assignments: FxHashMap::default(),
            function_interps: FxHashMap::default(),
            universes: FxHashMap::default(),
            defaults: FxHashMap::default(),
            generation: 0,
        }
    }

    /// Get the value of a term in this model
    pub fn eval(&self, term: TermId) -> Option<TermId> {
        self.assignments.get(&term).copied()
    }

    /// Set the value of a term
    pub fn set(&mut self, term: TermId, value: TermId) {
        self.assignments.insert(term, value);
    }

    /// Get the universe for a sort
    pub fn universe(&self, sort: SortId) -> Option<&[TermId]> {
        self.universes.get(&sort).map(|v| v.as_slice())
    }

    /// Add a value to a sort's universe
    pub fn add_to_universe(&mut self, sort: SortId, value: TermId) {
        self.universes.entry(sort).or_default().push(value);
    }

    /// Get the default value for a sort
    pub fn default_value(&self, sort: SortId) -> Option<TermId> {
        self.defaults.get(&sort).copied()
    }

    /// Set the default value for a sort
    pub fn set_default(&mut self, sort: SortId, value: TermId) {
        self.defaults.insert(sort, value);
    }

    /// Check if a sort has an uninterpreted universe
    pub fn has_uninterpreted_sort(&self, sort: SortId) -> bool {
        self.universes.contains_key(&sort)
    }
}

impl Default for CompletedModel {
    fn default() -> Self {
        Self::new()
    }
}

/// A function interpretation (finite representation of function mapping)
#[derive(Debug, Clone)]
pub struct FunctionInterpretation {
    /// Function name
    pub name: Spur,
    /// Arity
    pub arity: usize,
    /// Domain sorts
    pub domain: SmallVec<[SortId; 4]>,
    /// Range sort
    pub range: SortId,
    /// Explicit entries (args -> result)
    pub entries: Vec<FunctionEntry>,
    /// Default/else value (for arguments not in entries)
    pub else_value: Option<TermId>,
    /// Projection functions for arguments (if any)
    pub projections: Vec<Option<ProjectionFunctionDef>>,
}

impl FunctionInterpretation {
    /// Create a new function interpretation
    pub fn new(name: Spur, domain: SmallVec<[SortId; 4]>, range: SortId) -> Self {
        let arity = domain.len();
        Self {
            name,
            arity,
            domain,
            range,
            entries: Vec::new(),
            else_value: None,
            projections: vec![None; arity],
        }
    }

    /// Add an entry to the function table
    pub fn add_entry(&mut self, args: Vec<TermId>, result: TermId) {
        if args.len() == self.arity {
            self.entries.push(FunctionEntry { args, result });
        }
    }

    /// Lookup a value in the function table
    pub fn lookup(&self, args: &[TermId]) -> Option<TermId> {
        for entry in &self.entries {
            if entry.args == args {
                return Some(entry.result);
            }
        }
        self.else_value
    }

    /// Check if this is a constant function
    pub fn is_constant(&self) -> bool {
        self.arity == 0
    }

    /// Check if the interpretation is partial (missing else value or entries)
    pub fn is_partial(&self) -> bool {
        self.else_value.is_none() && !self.entries.is_empty()
    }

    /// Get the most common result value
    pub fn max_occurrence_result(&self) -> Option<TermId> {
        if self.entries.is_empty() {
            return None;
        }

        let mut counts: FxHashMap<TermId, usize> = FxHashMap::default();
        for entry in &self.entries {
            *counts.entry(entry.result).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(term, _)| term)
    }
}

/// A single entry in a function interpretation
#[derive(Debug, Clone)]
pub struct FunctionEntry {
    /// Arguments
    pub args: Vec<TermId>,
    /// Result value
    pub result: TermId,
}

/// Definition of a projection function for argument position
#[derive(Debug, Clone)]
pub struct ProjectionFunctionDef {
    /// Argument index this projection is for
    pub arg_index: usize,
    /// Sort being projected
    pub sort: SortId,
    /// Sorted values that appear in function applications
    pub values: Vec<TermId>,
    /// Mapping from value to representative term
    pub value_to_term: FxHashMap<TermId, TermId>,
    /// Mapping from term to value
    pub term_to_value: FxHashMap<TermId, TermId>,
}

impl ProjectionFunctionDef {
    /// Create a new projection function definition
    pub fn new(arg_index: usize, sort: SortId) -> Self {
        Self {
            arg_index,
            sort,
            values: Vec::new(),
            value_to_term: FxHashMap::default(),
            term_to_value: FxHashMap::default(),
        }
    }

    /// Add a value to the projection
    pub fn add_value(&mut self, value: TermId, term: TermId) {
        if !self.values.contains(&value) {
            self.values.push(value);
        }
        self.value_to_term.insert(value, term);
        self.term_to_value.insert(term, value);
    }

    /// Project a value to its representative
    pub fn project(&self, value: TermId) -> Option<TermId> {
        self.value_to_term.get(&value).copied()
    }
}

/// Model completer that takes partial models and makes them complete
#[derive(Debug)]
pub struct ModelCompleter {
    /// Macro solver
    macro_solver: MacroSolver,
    /// Model fixer for function interpretations
    model_fixer: ModelFixer,
    /// Handler for uninterpreted sorts
    uninterp_handler: UninterpretedSortHandler,
    /// Cache of completed models
    cache: FxHashMap<u64, CompletedModel>,
    /// Statistics
    stats: CompletionStats,
}

impl ModelCompleter {
    /// Create a new model completer
    pub fn new() -> Self {
        Self {
            macro_solver: MacroSolver::new(),
            model_fixer: ModelFixer::new(),
            uninterp_handler: UninterpretedSortHandler::new(),
            cache: FxHashMap::default(),
            stats: CompletionStats::default(),
        }
    }

    /// Complete a partial model
    pub fn complete(
        &mut self,
        partial_model: &FxHashMap<TermId, TermId>,
        quantifiers: &[QuantifiedFormula],
        manager: &mut TermManager,
    ) -> Result<CompletedModel, CompletionError> {
        self.stats.num_completions += 1;

        // Start with the partial model
        let mut completed = CompletedModel::new();
        completed.assignments = partial_model.clone();

        // Try to solve some quantifiers as macros
        let macro_results = self.macro_solver.solve_macros(quantifiers, manager)?;
        for (func_name, interp) in macro_results {
            completed.function_interps.insert(func_name, interp);
        }

        // Complete function interpretations
        self.model_fixer
            .fix_model(&mut completed, quantifiers, manager)?;

        // Handle uninterpreted sorts
        self.uninterp_handler
            .complete_universes(&mut completed, manager)?;

        // Set default values for all sorts
        self.set_default_values(&mut completed, manager)?;

        Ok(completed)
    }

    /// Set default values for all sorts in the model
    fn set_default_values(
        &mut self,
        model: &mut CompletedModel,
        manager: &mut TermManager,
    ) -> Result<(), CompletionError> {
        // Boolean
        if !model.defaults.contains_key(&manager.sorts.bool_sort) {
            model.set_default(manager.sorts.bool_sort, manager.mk_false());
        }

        // Integer
        if !model.defaults.contains_key(&manager.sorts.int_sort) {
            model.set_default(manager.sorts.int_sort, manager.mk_int(BigInt::from(0)));
        }

        // Real
        if !model.defaults.contains_key(&manager.sorts.real_sort) {
            model.set_default(
                manager.sorts.real_sort,
                manager.mk_real(Rational64::from_integer(0)),
            );
        }

        // Uninterpreted sorts - use first element from universe
        // Collect defaults first to avoid borrow conflict
        let defaults_to_set: Vec<(SortId, TermId)> = model
            .universes
            .iter()
            .filter_map(|(sort, universe)| {
                if !model.defaults.contains_key(sort) {
                    universe.first().map(|&first| (*sort, first))
                } else {
                    None
                }
            })
            .collect();

        for (sort, value) in defaults_to_set {
            model.set_default(sort, value);
        }

        Ok(())
    }

    /// Get completion statistics
    pub fn stats(&self) -> &CompletionStats {
        &self.stats
    }
}

impl Default for ModelCompleter {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro solver that identifies quantifiers that can be solved as macros
///
/// A quantifier can be solved as a macro if it has the form:
/// ∀x. f(x) = body(x)
/// where f is an uninterpreted function and body doesn't contain f
#[derive(Debug)]
pub struct MacroSolver {
    /// Detected macros
    macros: FxHashMap<Spur, MacroDefinition>,
    /// Statistics
    stats: MacroStats,
}

impl MacroSolver {
    /// Create a new macro solver
    pub fn new() -> Self {
        Self {
            macros: FxHashMap::default(),
            stats: MacroStats::default(),
        }
    }

    /// Try to solve quantifiers as macros
    pub fn solve_macros(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        manager: &mut TermManager,
    ) -> Result<FxHashMap<Spur, FunctionInterpretation>, CompletionError> {
        let mut results = FxHashMap::default();

        for quant in quantifiers {
            if let Some(macro_def) = self.try_extract_macro(quant, manager)? {
                self.stats.num_macros_found += 1;
                let interp = self.macro_to_interpretation(&macro_def, manager)?;
                results.insert(macro_def.func_name, interp);
                self.macros.insert(macro_def.func_name, macro_def);
            }
        }

        Ok(results)
    }

    /// Try to extract a macro from a quantified formula
    fn try_extract_macro(
        &self,
        quant: &QuantifiedFormula,
        manager: &TermManager,
    ) -> Result<Option<MacroDefinition>, CompletionError> {
        // Look for pattern: ∀x. f(x) = body(x)
        let Some(body_term) = manager.get(quant.body) else {
            return Ok(None);
        };

        // Check if body is an equality
        if let TermKind::Eq(lhs, rhs) = &body_term.kind {
            // Try both directions
            if let Some(macro_def) = self.try_extract_macro_from_eq(*lhs, *rhs, quant, manager)? {
                return Ok(Some(macro_def));
            }
            if let Some(macro_def) = self.try_extract_macro_from_eq(*rhs, *lhs, quant, manager)? {
                return Ok(Some(macro_def));
            }
        }

        Ok(None)
    }

    /// Try to extract macro from equality lhs = rhs
    fn try_extract_macro_from_eq(
        &self,
        lhs: TermId,
        rhs: TermId,
        quant: &QuantifiedFormula,
        manager: &TermManager,
    ) -> Result<Option<MacroDefinition>, CompletionError> {
        let Some(lhs_term) = manager.get(lhs) else {
            return Ok(None);
        };

        // Check if lhs is f(x1, ..., xn) where f is uninterpreted
        if let TermKind::Apply { func, args } = &lhs_term.kind {
            // Check if all args are bound variables
            let mut is_macro = true;
            for &arg in args.iter() {
                if let Some(arg_term) = manager.get(arg)
                    && !matches!(arg_term.kind, TermKind::Var(_))
                {
                    is_macro = false;
                    break;
                }
            }

            if is_macro {
                // Check if rhs doesn't contain f
                if !self.contains_function(rhs, *func, manager) {
                    return Ok(Some(MacroDefinition {
                        quantifier: quant.term,
                        func_name: *func,
                        bound_vars: quant.bound_vars.clone(),
                        body: rhs,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Check if term contains a function application
    fn contains_function(&self, term: TermId, func: Spur, manager: &TermManager) -> bool {
        let mut visited = FxHashSet::default();
        self.contains_function_rec(term, func, manager, &mut visited)
    }

    fn contains_function_rec(
        &self,
        term: TermId,
        func: Spur,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) -> bool {
        if visited.contains(&term) {
            return false;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return false;
        };

        match &t.kind {
            TermKind::Apply { func: f, args } => {
                if *f == func {
                    return true;
                }
                for &arg in args.iter() {
                    if self.contains_function_rec(arg, func, manager, visited) {
                        return true;
                    }
                }
                false
            }
            _ => {
                // Recursively check children
                let children = self.get_children(term, manager);
                for child in children {
                    if self.contains_function_rec(child, func, manager, visited) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Get children of a term
    fn get_children(&self, term: TermId, manager: &TermManager) -> Vec<TermId> {
        let Some(t) = manager.get(term) else {
            return vec![];
        };

        match &t.kind {
            TermKind::Not(arg) | TermKind::Neg(arg) => vec![*arg],
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args.to_vec(),
            TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs)
            | TermKind::Mod(lhs, rhs)
            | TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Implies(lhs, rhs) => vec![*lhs, *rhs],
            TermKind::Ite(cond, then_br, else_br) => vec![*cond, *then_br, *else_br],
            TermKind::Apply { args, .. } => args.to_vec(),
            _ => vec![],
        }
    }

    /// Convert a macro definition to a function interpretation
    fn macro_to_interpretation(
        &self,
        macro_def: &MacroDefinition,
        manager: &mut TermManager,
    ) -> Result<FunctionInterpretation, CompletionError> {
        // For now, create an empty interpretation
        // In a full implementation, we would evaluate the body for various inputs
        let func_name = macro_def.func_name;

        // Get function signature (this is simplified - real implementation would look it up)
        let domain = SmallVec::new();
        let range = manager.sorts.bool_sort; // Placeholder

        let interp = FunctionInterpretation::new(func_name, domain, range);
        Ok(interp)
    }

    /// Get statistics
    pub fn stats(&self) -> &MacroStats {
        &self.stats
    }
}

impl Default for MacroSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// A macro definition extracted from a quantifier
#[derive(Debug, Clone)]
pub struct MacroDefinition {
    /// Original quantifier
    pub quantifier: TermId,
    /// Function being defined
    pub func_name: Spur,
    /// Bound variables
    pub bound_vars: SmallVec<[(Spur, SortId); 4]>,
    /// Definition body
    pub body: TermId,
}

/// Model fixer that completes function interpretations
#[derive(Debug)]
pub struct ModelFixer {
    /// Projection functions by sort
    projections: FxHashMap<SortId, Box<dyn ProjectionFunction>>,
    /// Statistics
    stats: FixerStats,
}

impl ModelFixer {
    /// Create a new model fixer
    pub fn new() -> Self {
        Self {
            projections: FxHashMap::default(),
            stats: FixerStats::default(),
        }
    }

    /// Fix a model by completing function interpretations
    pub fn fix_model(
        &mut self,
        model: &mut CompletedModel,
        quantifiers: &[QuantifiedFormula],
        manager: &mut TermManager,
    ) -> Result<(), CompletionError> {
        self.stats.num_fixes += 1;

        // Collect all partial functions from quantifiers
        let partial_functions = self.collect_partial_functions(quantifiers, manager);

        // For each partial function, add projection functions
        // Process one at a time to avoid borrow conflicts
        for func_name in partial_functions.iter() {
            // Check if function exists first (immutable borrow)
            let has_interp = model.function_interps.contains_key(func_name);
            if has_interp {
                // Get mutable reference in separate scope
                if let Some(interp) = model.function_interps.get_mut(func_name) {
                    // Create a minimal projection without full model access
                    // This is a simplified version - full implementation would cache model data
                    for arg_idx in 0..interp.arity {
                        let sort = interp.domain[arg_idx];
                        if self.needs_projection(sort, manager) {
                            // Placeholder: would need model data extracted first
                            interp.projections[arg_idx] = None;
                        }
                    }
                }
            }
        }

        // Complete partial interpretations
        for interp in model.function_interps.values_mut() {
            if interp.is_partial() {
                // Use most common value as default
                if let Some(default) = interp.max_occurrence_result() {
                    interp.else_value = Some(default);
                }
            }
        }

        Ok(())
    }

    /// Collect partial function symbols from quantifiers
    fn collect_partial_functions(
        &self,
        quantifiers: &[QuantifiedFormula],
        manager: &TermManager,
    ) -> FxHashSet<Spur> {
        let mut functions = FxHashSet::default();

        for quant in quantifiers {
            self.collect_partial_functions_rec(quant.body, &mut functions, manager);
        }

        functions
    }

    fn collect_partial_functions_rec(
        &self,
        term: TermId,
        functions: &mut FxHashSet<Spur>,
        manager: &TermManager,
    ) {
        let Some(t) = manager.get(term) else {
            return;
        };

        if let TermKind::Apply { func, args } = &t.kind {
            // Check if any arg contains variables (not ground)
            let has_vars = args.iter().any(|&arg| {
                if let Some(arg_t) = manager.get(arg) {
                    matches!(arg_t.kind, TermKind::Var(_))
                } else {
                    false
                }
            });

            if has_vars {
                functions.insert(*func);
            }

            // Recurse into args
            for &arg in args.iter() {
                self.collect_partial_functions_rec(arg, functions, manager);
            }
        }

        // Recurse into other children
        match &t.kind {
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.collect_partial_functions_rec(*arg, functions, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args.iter() {
                    self.collect_partial_functions_rec(arg, functions, manager);
                }
            }
            TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) | TermKind::Le(lhs, rhs) => {
                self.collect_partial_functions_rec(*lhs, functions, manager);
                self.collect_partial_functions_rec(*rhs, functions, manager);
            }
            _ => {}
        }
    }

    /// Add projection functions for a function interpretation
    fn add_projection_functions(
        &mut self,
        interp: &mut FunctionInterpretation,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Result<(), CompletionError> {
        // For each argument position, create a projection if needed
        for arg_idx in 0..interp.arity {
            let sort = interp.domain[arg_idx];

            // Check if we need a projection for this sort
            if self.needs_projection(sort, manager) {
                let proj_def = self.create_projection(interp, arg_idx, model, manager)?;
                interp.projections[arg_idx] = Some(proj_def);
            }
        }

        Ok(())
    }

    /// Check if a sort needs projection
    fn needs_projection(&self, sort: SortId, manager: &TermManager) -> bool {
        // Arithmetic sorts benefit from projection
        sort == manager.sorts.int_sort || sort == manager.sorts.real_sort
    }

    /// Create a projection function for an argument position
    fn create_projection(
        &mut self,
        interp: &FunctionInterpretation,
        arg_idx: usize,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Result<ProjectionFunctionDef, CompletionError> {
        let sort = interp.domain[arg_idx];
        let mut proj_def = ProjectionFunctionDef::new(arg_idx, sort);

        // Collect all values that appear at this argument position
        for entry in &interp.entries {
            if let Some(&arg_term) = entry.args.get(arg_idx) {
                // Evaluate the argument in the model
                let value = model.eval(arg_term).unwrap_or(arg_term);
                proj_def.add_value(value, arg_term);
            }
        }

        // Sort the values
        proj_def
            .values
            .sort_by(|a, b| self.compare_values(*a, *b, sort, manager));

        Ok(proj_def)
    }

    /// Compare two values for a given sort
    fn compare_values(
        &self,
        a: TermId,
        b: TermId,
        _sort: SortId,
        manager: &TermManager,
    ) -> Ordering {
        let a_term = manager.get(a);
        let b_term = manager.get(b);

        if let (Some(at), Some(bt)) = (a_term, b_term) {
            // Integer comparison
            if let (TermKind::IntConst(av), TermKind::IntConst(bv)) = (&at.kind, &bt.kind) {
                return av.cmp(bv);
            }

            // Real comparison
            if let (TermKind::RealConst(av), TermKind::RealConst(bv)) = (&at.kind, &bt.kind) {
                return av.cmp(bv);
            }

            // Boolean comparison (false < true)
            match (&at.kind, &bt.kind) {
                (TermKind::False, TermKind::True) => return Ordering::Less,
                (TermKind::True, TermKind::False) => return Ordering::Greater,
                (TermKind::False, TermKind::False) | (TermKind::True, TermKind::True) => {
                    return Ordering::Equal;
                }
                _ => {}
            }
        }

        // Fall back to ID comparison
        a.0.cmp(&b.0)
    }

    /// Get statistics
    pub fn stats(&self) -> &FixerStats {
        &self.stats
    }
}

impl Default for ModelFixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for projection functions (maps infinite domain to finite representatives)
pub trait ProjectionFunction: fmt::Debug + Send + Sync {
    /// Compare two values (for sorting)
    fn compare(&self, a: TermId, b: TermId, manager: &TermManager) -> bool;

    /// Create a less-than term
    fn mk_lt(&self, x: TermId, y: TermId, manager: &mut TermManager) -> TermId;
}

/// Arithmetic projection function
#[derive(Debug)]
pub struct ArithmeticProjection {
    /// Whether this is for integers (vs reals)
    is_int: bool,
}

impl ArithmeticProjection {
    pub fn new(is_int: bool) -> Self {
        Self { is_int }
    }
}

impl ProjectionFunction for ArithmeticProjection {
    fn compare(&self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        let a_term = manager.get(a);
        let b_term = manager.get(b);

        if let (Some(at), Some(bt)) = (a_term, b_term) {
            if let (TermKind::IntConst(av), TermKind::IntConst(bv)) = (&at.kind, &bt.kind) {
                return av < bv;
            }
            if let (TermKind::RealConst(av), TermKind::RealConst(bv)) = (&at.kind, &bt.kind) {
                return av < bv;
            }
        }

        a.0 < b.0
    }

    fn mk_lt(&self, x: TermId, y: TermId, manager: &mut TermManager) -> TermId {
        manager.mk_lt(x, y)
    }
}

/// Handler for uninterpreted sorts
#[derive(Debug)]
pub struct UninterpretedSortHandler {
    /// Maximum universe size for each sort
    max_universe_size: usize,
    /// Statistics
    stats: UninterpStats,
}

impl UninterpretedSortHandler {
    /// Create a new handler
    pub fn new() -> Self {
        Self {
            max_universe_size: 8,
            stats: UninterpStats::default(),
        }
    }

    /// Create with custom universe size limit
    pub fn with_max_size(max_size: usize) -> Self {
        let mut handler = Self::new();
        handler.max_universe_size = max_size;
        handler
    }

    /// Complete universes for uninterpreted sorts
    pub fn complete_universes(
        &mut self,
        model: &mut CompletedModel,
        manager: &mut TermManager,
    ) -> Result<(), CompletionError> {
        // Identify uninterpreted sorts
        let uninterp_sorts = self.identify_uninterpreted_sorts(model, manager);

        for sort in uninterp_sorts {
            if let std::collections::hash_map::Entry::Vacant(e) = model.universes.entry(sort) {
                // Create a finite universe for this sort
                let universe = self.create_finite_universe(sort, manager)?;
                e.insert(universe);
                self.stats.num_universes_created += 1;
            }
        }

        Ok(())
    }

    /// Identify uninterpreted sorts in the model
    fn identify_uninterpreted_sorts(
        &self,
        model: &CompletedModel,
        manager: &TermManager,
    ) -> Vec<SortId> {
        let mut sorts = Vec::new();

        // Collect sorts from function interpretations
        for interp in model.function_interps.values() {
            for &sort in &interp.domain {
                if self.is_uninterpreted(sort, manager) && !sorts.contains(&sort) {
                    sorts.push(sort);
                }
            }
            if self.is_uninterpreted(interp.range, manager) && !sorts.contains(&interp.range) {
                sorts.push(interp.range);
            }
        }

        sorts
    }

    /// Check if a sort is uninterpreted
    fn is_uninterpreted(&self, sort: SortId, manager: &TermManager) -> bool {
        // A sort is uninterpreted if it's not a built-in sort
        sort != manager.sorts.bool_sort
            && sort != manager.sorts.int_sort
            && sort != manager.sorts.real_sort
    }

    /// Create a finite universe for a sort
    fn create_finite_universe(
        &self,
        sort: SortId,
        manager: &mut TermManager,
    ) -> Result<Vec<TermId>, CompletionError> {
        let mut universe = Vec::new();

        // Create fresh constants for the universe
        for i in 0..self.max_universe_size {
            let name = format!("u!{}", i);
            let const_id = manager.mk_var(&name, sort);
            universe.push(const_id);
        }

        Ok(universe)
    }

    /// Get statistics
    pub fn stats(&self) -> &UninterpStats {
        &self.stats
    }
}

impl Default for UninterpretedSortHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Error during model completion
#[derive(Debug, Clone)]
pub enum CompletionError {
    /// Could not complete the model
    CompletionFailed(String),
    /// Resource limit exceeded
    ResourceLimit,
    /// Invalid model
    InvalidModel(String),
}

impl fmt::Display for CompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompletionFailed(msg) => write!(f, "Model completion failed: {}", msg),
            Self::ResourceLimit => write!(f, "Resource limit exceeded during completion"),
            Self::InvalidModel(msg) => write!(f, "Invalid model: {}", msg),
        }
    }
}

impl std::error::Error for CompletionError {}

/// Statistics for model completion
#[derive(Debug, Clone, Default)]
pub struct CompletionStats {
    pub num_completions: usize,
    pub num_failures: usize,
}

/// Statistics for macro solving
#[derive(Debug, Clone, Default)]
pub struct MacroStats {
    pub num_macros_found: usize,
    pub num_macros_applied: usize,
}

/// Statistics for model fixing
#[derive(Debug, Clone, Default)]
pub struct FixerStats {
    pub num_fixes: usize,
    pub num_projections_created: usize,
}

/// Statistics for uninterpreted sort handling
#[derive(Debug, Clone, Default)]
pub struct UninterpStats {
    pub num_universes_created: usize,
    pub total_universe_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::Key;

    #[test]
    fn test_completed_model_creation() {
        let model = CompletedModel::new();
        assert_eq!(model.assignments.len(), 0);
        assert_eq!(model.function_interps.len(), 0);
    }

    #[test]
    fn test_completed_model_eval() {
        let mut model = CompletedModel::new();
        let term = TermId::new(1);
        let value = TermId::new(2);

        model.set(term, value);
        assert_eq!(model.eval(term), Some(value));
        assert_eq!(model.eval(TermId::new(99)), None);
    }

    #[test]
    fn test_function_interpretation_lookup() {
        // Create a function with arity 2 (domain has 2 sorts)
        let mut domain = SmallVec::new();
        domain.push(SortId::new(1));
        domain.push(SortId::new(1));

        let mut interp = FunctionInterpretation::new(
            Spur::try_from_usize(1).expect("valid spur"),
            domain,
            SortId::new(1),
        );

        let args = vec![TermId::new(1), TermId::new(2)];
        let result = TermId::new(10);
        interp.add_entry(args.clone(), result);

        assert_eq!(interp.lookup(&args), Some(result));
        assert_eq!(interp.lookup(&[TermId::new(99)]), None);
    }

    #[test]
    fn test_function_interpretation_else_value() {
        let mut interp = FunctionInterpretation::new(
            Spur::try_from_usize(1).expect("valid spur"),
            SmallVec::new(),
            SortId::new(1),
        );

        let else_val = TermId::new(42);
        interp.else_value = Some(else_val);

        assert_eq!(interp.lookup(&[TermId::new(99)]), Some(else_val));
    }

    #[test]
    fn test_function_interpretation_max_occurrence() {
        // Create a function with arity 1 (domain has 1 sort)
        let mut domain = SmallVec::new();
        domain.push(SortId::new(1));

        let mut interp = FunctionInterpretation::new(
            Spur::try_from_usize(1).expect("valid spur"),
            domain,
            SortId::new(1),
        );

        let result1 = TermId::new(10);
        let result2 = TermId::new(20);

        interp.add_entry(vec![TermId::new(1)], result1);
        interp.add_entry(vec![TermId::new(2)], result1);
        interp.add_entry(vec![TermId::new(3)], result2);

        assert_eq!(interp.max_occurrence_result(), Some(result1));
    }

    #[test]
    fn test_projection_function_def() {
        let mut proj = ProjectionFunctionDef::new(0, SortId::new(1));

        let value1 = TermId::new(1);
        let term1 = TermId::new(10);
        proj.add_value(value1, term1);

        assert_eq!(proj.project(value1), Some(term1));
        assert_eq!(proj.values.len(), 1);
    }

    #[test]
    fn test_model_completer_creation() {
        let completer = ModelCompleter::new();
        assert_eq!(completer.stats.num_completions, 0);
    }

    #[test]
    fn test_macro_solver_creation() {
        let solver = MacroSolver::new();
        assert_eq!(solver.stats.num_macros_found, 0);
    }

    #[test]
    fn test_model_fixer_creation() {
        let fixer = ModelFixer::new();
        assert_eq!(fixer.stats.num_fixes, 0);
    }

    #[test]
    fn test_uninterpreted_sort_handler_creation() {
        let handler = UninterpretedSortHandler::new();
        assert_eq!(handler.max_universe_size, 8);
    }

    #[test]
    fn test_uninterpreted_sort_handler_custom_size() {
        let handler = UninterpretedSortHandler::with_max_size(16);
        assert_eq!(handler.max_universe_size, 16);
    }

    #[test]
    fn test_arithmetic_projection() {
        let proj = ArithmeticProjection::new(true);
        assert!(proj.is_int);
    }

    #[test]
    fn test_completion_error_display() {
        let err = CompletionError::CompletionFailed("test".to_string());
        assert!(format!("{}", err).contains("test"));
    }
}
