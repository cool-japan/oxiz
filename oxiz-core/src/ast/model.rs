//! Model representation for satisfying assignments
//!
//! A model represents a satisfying assignment for a set of assertions.
//! It maps variables to concrete values and provides function interpretations.

use crate::ast::TermId;
use crate::sort::SortId;
use num_bigint::BigInt;
use num_rational::BigRational;
use rustc_hash::FxHashMap;

/// A model represents a satisfying assignment
#[derive(Debug, Clone)]
pub struct Model {
    /// Variable assignments: maps variable TermId to its value
    assignments: FxHashMap<TermId, ModelValue>,
    /// Function interpretations
    functions: FxHashMap<lasso::Spur, FunctionInterpretation>,
}

/// A value in a model
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(BigInt),
    /// Real value
    Real(BigRational),
    /// BitVector value
    BitVec {
        /// The bitvector value
        value: u64,
        /// The bitvector width in bits
        width: u32,
    },
    /// Uninterpreted constant (for sorts without concrete values)
    Uninterpreted {
        /// The sort of the uninterpreted value
        sort: SortId,
        /// Unique identifier for this uninterpreted value
        id: u64,
    },
}

/// Function interpretation in a model
#[derive(Debug, Clone)]
pub struct FunctionInterpretation {
    /// Explicit function table: maps argument tuples to results
    table: Vec<(Vec<ModelValue>, ModelValue)>,
    /// Default value for arguments not in the table
    default: Option<ModelValue>,
}

impl Model {
    /// Create an empty model
    #[must_use]
    pub fn new() -> Self {
        Self {
            assignments: FxHashMap::default(),
            functions: FxHashMap::default(),
        }
    }

    /// Assign a boolean value to a variable
    pub fn assign_bool(&mut self, var: TermId, value: bool) {
        self.assignments.insert(var, ModelValue::Bool(value));
    }

    /// Assign an integer value to a variable
    pub fn assign_int(&mut self, var: TermId, value: BigInt) {
        self.assignments.insert(var, ModelValue::Int(value));
    }

    /// Assign a real value to a variable
    pub fn assign_real(&mut self, var: TermId, value: BigRational) {
        self.assignments.insert(var, ModelValue::Real(value));
    }

    /// Assign a bitvector value to a variable
    pub fn assign_bitvec(&mut self, var: TermId, value: u64, width: u32) {
        self.assignments
            .insert(var, ModelValue::BitVec { value, width });
    }

    /// Assign an uninterpreted value to a variable
    pub fn assign_uninterpreted(&mut self, var: TermId, sort: SortId, id: u64) {
        self.assignments
            .insert(var, ModelValue::Uninterpreted { sort, id });
    }

    /// Get the value assigned to a variable
    #[must_use]
    pub fn get_assignment(&self, var: TermId) -> Option<&ModelValue> {
        self.assignments.get(&var)
    }

    /// Add a function interpretation
    pub fn add_function(&mut self, name: lasso::Spur, interp: FunctionInterpretation) {
        self.functions.insert(name, interp);
    }

    /// Get a function interpretation
    #[must_use]
    pub fn get_function(&self, name: lasso::Spur) -> Option<&FunctionInterpretation> {
        self.functions.get(&name)
    }

    /// Evaluate a function application
    #[must_use]
    pub fn eval_function(&self, name: lasso::Spur, args: &[ModelValue]) -> Option<ModelValue> {
        let func = self.functions.get(&name)?;
        func.eval(args)
    }

    /// Check if the model is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty() && self.functions.is_empty()
    }

    /// Get the number of variable assignments
    #[must_use]
    pub fn num_assignments(&self) -> usize {
        self.assignments.len()
    }

    /// Get the number of function interpretations
    #[must_use]
    pub fn num_functions(&self) -> usize {
        self.functions.len()
    }

    /// Iterate over all variable assignments
    pub fn assignments(&self) -> impl Iterator<Item = (&TermId, &ModelValue)> {
        self.assignments.iter()
    }

    /// Iterate over all function interpretations
    pub fn functions(&self) -> impl Iterator<Item = (&lasso::Spur, &FunctionInterpretation)> {
        self.functions.iter()
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionInterpretation {
    /// Create a new function interpretation
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            default: None,
        }
    }

    /// Create a function interpretation with a default value
    #[must_use]
    pub fn with_default(default: ModelValue) -> Self {
        Self {
            table: Vec::new(),
            default: Some(default),
        }
    }

    /// Add an entry to the function table
    pub fn add_entry(&mut self, args: Vec<ModelValue>, result: ModelValue) {
        self.table.push((args, result));
    }

    /// Set the default value
    pub fn set_default(&mut self, default: ModelValue) {
        self.default = Some(default);
    }

    /// Evaluate the function for given arguments
    #[must_use]
    pub fn eval(&self, args: &[ModelValue]) -> Option<ModelValue> {
        // Search the explicit table first
        for (table_args, result) in &self.table {
            if table_args.as_slice() == args {
                return Some(result.clone());
            }
        }
        // Fall back to default value
        self.default.clone()
    }

    /// Get the function table
    #[must_use]
    pub fn table(&self) -> &[(Vec<ModelValue>, ModelValue)] {
        &self.table
    }

    /// Get the default value
    #[must_use]
    pub fn default_value(&self) -> Option<&ModelValue> {
        self.default.as_ref()
    }
}

impl Default for FunctionInterpretation {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ModelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelValue::Bool(b) => write!(f, "{}", b),
            ModelValue::Int(n) => write!(f, "{}", n),
            ModelValue::Real(r) => write!(f, "{}", r),
            ModelValue::BitVec { value, width } => {
                write!(f, "#b{:0width$b}", value, width = *width as usize)
            }
            ModelValue::Uninterpreted { sort, id } => write!(f, "uninterp!{:?}!{}", sort, id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermId;

    #[test]
    fn test_empty_model() {
        let model = Model::new();
        assert!(model.is_empty());
        assert_eq!(model.num_assignments(), 0);
        assert_eq!(model.num_functions(), 0);
    }

    #[test]
    fn test_bool_assignment() {
        let mut model = Model::new();
        let var = TermId(0);

        model.assign_bool(var, true);
        assert!(!model.is_empty());
        assert_eq!(model.num_assignments(), 1);

        let value = model.get_assignment(var);
        assert_eq!(value, Some(&ModelValue::Bool(true)));
    }

    #[test]
    fn test_int_assignment() {
        let mut model = Model::new();
        let var = TermId(0);

        model.assign_int(var, BigInt::from(42));

        let value = model.get_assignment(var);
        assert_eq!(value, Some(&ModelValue::Int(BigInt::from(42))));
    }

    #[test]
    fn test_real_assignment() {
        let mut model = Model::new();
        let var = TermId(0);

        let rational = BigRational::new(BigInt::from(3), BigInt::from(2));
        model.assign_real(var, rational.clone());

        let value = model.get_assignment(var);
        assert_eq!(value, Some(&ModelValue::Real(rational)));
    }

    #[test]
    fn test_bitvec_assignment() {
        let mut model = Model::new();
        let var = TermId(0);

        model.assign_bitvec(var, 42, 8);

        let value = model.get_assignment(var);
        assert_eq!(
            value,
            Some(&ModelValue::BitVec {
                value: 42,
                width: 8
            })
        );
    }

    #[test]
    fn test_function_interpretation() {
        let mut func = FunctionInterpretation::new();

        // Add f(0) = 1
        func.add_entry(
            vec![ModelValue::Int(BigInt::from(0))],
            ModelValue::Int(BigInt::from(1)),
        );

        // Add f(1) = 2
        func.add_entry(
            vec![ModelValue::Int(BigInt::from(1))],
            ModelValue::Int(BigInt::from(2)),
        );

        // Set default to 0
        func.set_default(ModelValue::Int(BigInt::from(0)));

        // Test evaluation
        assert_eq!(
            func.eval(&[ModelValue::Int(BigInt::from(0))]),
            Some(ModelValue::Int(BigInt::from(1)))
        );
        assert_eq!(
            func.eval(&[ModelValue::Int(BigInt::from(1))]),
            Some(ModelValue::Int(BigInt::from(2)))
        );
        assert_eq!(
            func.eval(&[ModelValue::Int(BigInt::from(5))]),
            Some(ModelValue::Int(BigInt::from(0)))
        );
    }

    #[test]
    fn test_model_with_function() {
        let mut model = Model::new();
        let mut rodeo = lasso::Rodeo::default();
        let f_name = rodeo.get_or_intern("f");

        let mut func = FunctionInterpretation::with_default(ModelValue::Int(BigInt::from(0)));
        func.add_entry(
            vec![ModelValue::Int(BigInt::from(1))],
            ModelValue::Int(BigInt::from(10)),
        );

        model.add_function(f_name, func);

        assert_eq!(model.num_functions(), 1);
        assert!(model.get_function(f_name).is_some());

        let result = model.eval_function(f_name, &[ModelValue::Int(BigInt::from(1))]);
        assert_eq!(result, Some(ModelValue::Int(BigInt::from(10))));
    }

    #[test]
    fn test_model_value_display() {
        assert_eq!(ModelValue::Bool(true).to_string(), "true");
        assert_eq!(ModelValue::Int(BigInt::from(42)).to_string(), "42");
        assert_eq!(
            ModelValue::BitVec { value: 5, width: 4 }.to_string(),
            "#b0101"
        );
    }

    #[test]
    fn test_multiple_assignments() {
        let mut model = Model::new();

        model.assign_bool(TermId(0), true);
        model.assign_int(TermId(1), BigInt::from(10));
        model.assign_bitvec(TermId(2), 255, 8);

        assert_eq!(model.num_assignments(), 3);

        let mut count = 0;
        for _ in model.assignments() {
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
