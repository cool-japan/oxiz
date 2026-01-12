//! Sort system for OxiZ
//!
//! Sorts represent the types of terms in SMT-LIB2. Common sorts include:
//! - Bool: Boolean values
//! - Int: Arbitrary precision integers
//! - Real: Real numbers
//! - BitVec(n): Bit vectors of width n
//! - Array(domain, range): Arrays mapping domain to range

pub mod inference;

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Unique identifier for a sort
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SortId(pub u32);

impl SortId {
    /// Create a new SortId from a raw value
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// The kind of a sort
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SortKind {
    /// Boolean sort
    Bool,
    /// Integer sort (arbitrary precision)
    Int,
    /// Real number sort
    Real,
    /// String sort
    String,
    /// Bit vector sort with specified width
    BitVec(u32),
    /// Floating-point sort with exponent and significand widths
    /// FloatingPoint(exponent_bits, significand_bits)
    /// Examples: Float32 = FloatingPoint(8, 24), Float64 = FloatingPoint(11, 53)
    FloatingPoint {
        /// Exponent width in bits
        eb: u32,
        /// Significand width in bits
        sb: u32,
    },
    /// Array sort with domain and range sorts
    Array {
        /// Domain sort
        domain: SortId,
        /// Range sort
        range: SortId,
    },
    /// Uninterpreted sort with a name
    Uninterpreted(lasso::Spur),
    /// Sort parameter (used in parametric sort definitions)
    /// The Spur is the parameter name
    Parameter(lasso::Spur),
    /// Parametric sort application: name applied to argument sorts
    /// Example: (List Int) where List is the name and \[Int\] are the args
    Parametric {
        /// Name of the parametric sort
        name: lasso::Spur,
        /// Sort arguments
        args: smallvec::SmallVec<[SortId; 2]>,
    },
    /// Datatype sort
    /// Reference to a datatype definition by name
    Datatype(lasso::Spur),
}

/// A sort in the SMT-LIB2 sense
#[derive(Debug, Clone)]
pub struct Sort {
    /// Unique identifier
    pub id: SortId,
    /// The kind of this sort
    pub kind: SortKind,
}

impl Sort {
    /// Check if this is a boolean sort
    #[must_use]
    pub fn is_bool(&self) -> bool {
        matches!(self.kind, SortKind::Bool)
    }

    /// Check if this is an integer sort
    #[must_use]
    pub fn is_int(&self) -> bool {
        matches!(self.kind, SortKind::Int)
    }

    /// Check if this is a real sort
    #[must_use]
    pub fn is_real(&self) -> bool {
        matches!(self.kind, SortKind::Real)
    }

    /// Check if this is a string sort
    #[must_use]
    pub fn is_string(&self) -> bool {
        matches!(self.kind, SortKind::String)
    }

    /// Check if this is a bit vector sort
    #[must_use]
    pub fn is_bitvec(&self) -> bool {
        matches!(self.kind, SortKind::BitVec(_))
    }

    /// Get the bit vector width if this is a bit vector sort
    #[must_use]
    pub fn bitvec_width(&self) -> Option<u32> {
        match self.kind {
            SortKind::BitVec(w) => Some(w),
            _ => None,
        }
    }

    /// Check if this is a floating-point sort
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self.kind, SortKind::FloatingPoint { .. })
    }

    /// Get the exponent and significand widths if this is a floating-point sort
    #[must_use]
    pub fn float_format(&self) -> Option<(u32, u32)> {
        match self.kind {
            SortKind::FloatingPoint { eb, sb } => Some((eb, sb)),
            _ => None,
        }
    }
}

/// Datatype constructor for algebraic datatypes.
///
/// Represents a constructor for an algebraic datatype like `cons` in a list type.
#[derive(Debug, Clone)]
pub struct DataTypeConstructor {
    /// Name of the constructor
    pub name: lasso::Spur,
    /// Selector names and their sorts (field name, field sort)
    pub selectors: smallvec::SmallVec<[(lasso::Spur, SortId); 4]>,
}

/// Parametric sort declaration
///
/// Represents a declared parametric sort like `(declare-sort List 1)`
/// where the sort takes a specified number of parameters.
#[derive(Debug, Clone)]
pub struct ParametricSortDecl {
    /// Name of the parametric sort
    pub name: lasso::Spur,
    /// Number of parameters this sort takes
    pub arity: usize,
}

/// Parametric sort definition
///
/// Represents a defined parametric sort like `(define-sort (List T) (Array Int T))`
/// where T is a sort parameter.
#[derive(Debug, Clone)]
pub struct ParametricSortDef {
    /// Name of the parametric sort
    pub name: lasso::Spur,
    /// Parameter names
    pub params: smallvec::SmallVec<[lasso::Spur; 2]>,
    /// The body sort expression (may reference parameters)
    pub body: SortId,
}

/// Datatype definition
///
/// Represents an algebraic datatype like List or Tree.
/// Can be recursive (e.g., `(List (cons (car T) (cdr (List T)))) (nil)`).
#[derive(Debug, Clone)]
pub struct DataTypeDef {
    /// Name of the datatype
    pub name: lasso::Spur,
    /// The sort ID for this datatype (self-reference for recursive types)
    pub sort_id: SortId,
    /// Constructors for this datatype
    pub constructors: Vec<DataTypeConstructor>,
}

/// Manager for sorts, handling canonicalization and interning
#[derive(Debug)]
pub struct SortManager {
    sorts: Vec<Sort>,
    cache: FxHashMap<SortKind, SortId>,
    next_id: AtomicU32,
    /// Pre-allocated common sorts
    pub bool_sort: SortId,
    /// Integer sort
    pub int_sort: SortId,
    /// Real sort
    pub real_sort: SortId,
    /// String sort (cached)
    string_sort_cached: Option<SortId>,
    /// Floating-point sorts (cached for common sizes)
    float16_sort_cached: Option<SortId>,
    float32_sort_cached: Option<SortId>,
    float64_sort_cached: Option<SortId>,
    float128_sort_cached: Option<SortId>,
    /// Sort aliases: maps alias names to their underlying sorts
    aliases: FxHashMap<lasso::Spur, SortId>,
    /// String interner for alias names
    interner: lasso::Rodeo,
    /// Declared parametric sorts (name -> arity)
    parametric_decls: FxHashMap<lasso::Spur, ParametricSortDecl>,
    /// Defined parametric sorts (name -> definition)
    parametric_defs: FxHashMap<lasso::Spur, ParametricSortDef>,
    /// Datatype definitions (name -> definition)
    datatypes: FxHashMap<lasso::Spur, DataTypeDef>,
}

impl Default for SortManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SortManager {
    /// Create a new sort manager with pre-allocated common sorts
    #[must_use]
    pub fn new() -> Self {
        let mut manager = Self {
            sorts: Vec::with_capacity(64),
            cache: FxHashMap::default(),
            next_id: AtomicU32::new(0),
            bool_sort: SortId(0),
            int_sort: SortId(1),
            real_sort: SortId(2),
            string_sort_cached: None,
            float16_sort_cached: None,
            float32_sort_cached: None,
            float64_sort_cached: None,
            float128_sort_cached: None,
            aliases: FxHashMap::default(),
            interner: lasso::Rodeo::default(),
            parametric_decls: FxHashMap::default(),
            parametric_defs: FxHashMap::default(),
            datatypes: FxHashMap::default(),
        };

        // Pre-allocate common sorts
        manager.bool_sort = manager.intern(SortKind::Bool);
        manager.int_sort = manager.intern(SortKind::Int);
        manager.real_sort = manager.intern(SortKind::Real);

        manager
    }

    /// Intern a sort kind, returning its unique ID
    pub fn intern(&mut self, kind: SortKind) -> SortId {
        if let Some(&id) = self.cache.get(&kind) {
            return id;
        }

        let id = SortId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let sort = Sort {
            id,
            kind: kind.clone(),
        };
        self.sorts.push(sort);
        self.cache.insert(kind, id);
        id
    }

    /// Get a sort by its ID
    #[must_use]
    pub fn get(&self, id: SortId) -> Option<&Sort> {
        self.sorts.get(id.0 as usize)
    }

    /// Create a bit vector sort with the given width
    pub fn bitvec(&mut self, width: u32) -> SortId {
        self.intern(SortKind::BitVec(width))
    }

    /// Create an array sort with the given domain and range
    pub fn array(&mut self, domain: SortId, range: SortId) -> SortId {
        self.intern(SortKind::Array { domain, range })
    }

    /// Get or create the string sort
    pub fn string_sort(&mut self) -> SortId {
        if let Some(id) = self.string_sort_cached {
            id
        } else {
            let id = self.intern(SortKind::String);
            self.string_sort_cached = Some(id);
            id
        }
    }

    /// Create a floating-point sort with custom exponent and significand widths
    pub fn float_sort(&mut self, eb: u32, sb: u32) -> SortId {
        self.intern(SortKind::FloatingPoint { eb, sb })
    }

    /// Get or create Float16 sort (IEEE 754 half precision: 5 exponent, 11 significand)
    pub fn float16_sort(&mut self) -> SortId {
        if let Some(id) = self.float16_sort_cached {
            id
        } else {
            let id = self.intern(SortKind::FloatingPoint { eb: 5, sb: 11 });
            self.float16_sort_cached = Some(id);
            id
        }
    }

    /// Get or create Float32 sort (IEEE 754 single precision: 8 exponent, 24 significand)
    pub fn float32_sort(&mut self) -> SortId {
        if let Some(id) = self.float32_sort_cached {
            id
        } else {
            let id = self.intern(SortKind::FloatingPoint { eb: 8, sb: 24 });
            self.float32_sort_cached = Some(id);
            id
        }
    }

    /// Get or create Float64 sort (IEEE 754 double precision: 11 exponent, 53 significand)
    pub fn float64_sort(&mut self) -> SortId {
        if let Some(id) = self.float64_sort_cached {
            id
        } else {
            let id = self.intern(SortKind::FloatingPoint { eb: 11, sb: 53 });
            self.float64_sort_cached = Some(id);
            id
        }
    }

    /// Get or create Float128 sort (IEEE 754 quad precision: 15 exponent, 113 significand)
    pub fn float128_sort(&mut self) -> SortId {
        if let Some(id) = self.float128_sort_cached {
            id
        } else {
            let id = self.intern(SortKind::FloatingPoint { eb: 15, sb: 113 });
            self.float128_sort_cached = Some(id);
            id
        }
    }

    /// Define a sort alias
    ///
    /// # Example
    /// ```ignore
    /// // type Word = BitVec(32)
    /// let bv32 = manager.bitvec(32);
    /// manager.define_alias("Word", bv32);
    /// ```
    pub fn define_alias(&mut self, name: &str, sort: SortId) {
        let key = self.interner.get_or_intern(name);
        self.aliases.insert(key, sort);
    }

    /// Look up a sort alias by name
    ///
    /// Returns the underlying sort if the alias exists, None otherwise
    #[must_use]
    pub fn resolve_alias(&self, name: &str) -> Option<SortId> {
        let key = self.interner.get(name)?;
        self.aliases.get(&key).copied()
    }

    /// Check if a name is defined as an alias
    #[must_use]
    pub fn is_alias(&self, name: &str) -> bool {
        self.resolve_alias(name).is_some()
    }

    /// Get all defined aliases
    #[must_use]
    pub fn aliases(&self) -> Vec<(String, SortId)> {
        self.aliases
            .iter()
            .map(|(key, &sort)| (self.interner.resolve(key).to_string(), sort))
            .collect()
    }

    /// Remove an alias definition
    pub fn undefine_alias(&mut self, name: &str) -> Option<SortId> {
        let key = self.interner.get(name)?;
        self.aliases.remove(&key)
    }

    /// Resolve a sort by name, checking aliases first
    ///
    /// This is useful for parsing where you want to support both
    /// built-in sorts and user-defined aliases
    #[must_use]
    pub fn resolve_by_name(&self, name: &str) -> Option<SortId> {
        // Check aliases first
        if let Some(sort_id) = self.resolve_alias(name) {
            return Some(sort_id);
        }

        // Check built-in sorts
        match name {
            "Bool" => Some(self.bool_sort),
            "Int" => Some(self.int_sort),
            "Real" => Some(self.real_sort),
            _ => None,
        }
    }

    // ========================== Parametric Sorts ==========================

    /// Declare a parametric sort
    ///
    /// This corresponds to `(declare-sort Name arity)` in SMT-LIB2.
    /// A declared parametric sort is an uninterpreted sort constructor.
    ///
    /// # Example
    /// ```ignore
    /// // (declare-sort List 1)
    /// manager.declare_parametric_sort("List", 1);
    ///
    /// // (declare-sort Pair 2)
    /// manager.declare_parametric_sort("Pair", 2);
    /// ```
    pub fn declare_parametric_sort(&mut self, name: &str, arity: usize) {
        let key = self.interner.get_or_intern(name);
        self.parametric_decls
            .insert(key, ParametricSortDecl { name: key, arity });
    }

    /// Check if a parametric sort with the given name is declared
    #[must_use]
    pub fn is_parametric_sort_declared(&self, name: &str) -> bool {
        self.interner
            .get(name)
            .is_some_and(|key| self.parametric_decls.contains_key(&key))
    }

    /// Get the arity of a declared parametric sort
    #[must_use]
    pub fn parametric_sort_arity(&self, name: &str) -> Option<usize> {
        let key = self.interner.get(name)?;
        self.parametric_decls.get(&key).map(|d| d.arity)
    }

    /// Define a parametric sort
    ///
    /// This corresponds to `(define-sort (Name params...) body)` in SMT-LIB2.
    /// A defined parametric sort is a sort alias with parameters.
    ///
    /// # Example
    /// ```ignore
    /// // (define-sort (List T) (Array Int T))
    /// // First create a parameter sort for T
    /// let t_param = manager.mk_sort_parameter("T");
    /// let array_sort = manager.array(manager.int_sort, t_param);
    /// manager.define_parametric_sort("List", &["T"], array_sort);
    /// ```
    pub fn define_parametric_sort(&mut self, name: &str, params: &[&str], body: SortId) {
        let key = self.interner.get_or_intern(name);
        let param_spurs: smallvec::SmallVec<[lasso::Spur; 2]> = params
            .iter()
            .map(|p| self.interner.get_or_intern(*p))
            .collect();

        self.parametric_defs.insert(
            key,
            ParametricSortDef {
                name: key,
                params: param_spurs,
                body,
            },
        );
    }

    /// Check if a parametric sort with the given name is defined
    #[must_use]
    pub fn is_parametric_sort_defined(&self, name: &str) -> bool {
        self.interner
            .get(name)
            .is_some_and(|key| self.parametric_defs.contains_key(&key))
    }

    /// Create a sort parameter
    ///
    /// Used when building parametric sort definitions.
    pub fn mk_sort_parameter(&mut self, name: &str) -> SortId {
        let key = self.interner.get_or_intern(name);
        self.intern(SortKind::Parameter(key))
    }

    /// Instantiate a parametric sort with concrete sort arguments
    ///
    /// # Example
    /// ```ignore
    /// // Create (List Int)
    /// let list_int = manager.instantiate_parametric_sort("List", &[manager.int_sort]);
    /// ```
    pub fn instantiate_parametric_sort(
        &mut self,
        name: &str,
        args: &[SortId],
    ) -> Result<SortId, String> {
        let key = self.interner.get_or_intern(name);

        // Check if it's a declared parametric sort (uninterpreted)
        if let Some(decl) = self.parametric_decls.get(&key).cloned() {
            if args.len() != decl.arity {
                return Err(format!(
                    "Sort {} expects {} arguments, got {}",
                    name,
                    decl.arity,
                    args.len()
                ));
            }

            // Create a parametric sort application
            let args_vec: smallvec::SmallVec<[SortId; 2]> = args.iter().copied().collect();
            let sort_id = self.intern(SortKind::Parametric {
                name: key,
                args: args_vec,
            });
            return Ok(sort_id);
        }

        // Check if it's a defined parametric sort (with body)
        if let Some(def) = self.parametric_defs.get(&key).cloned() {
            if args.len() != def.params.len() {
                return Err(format!(
                    "Sort {} expects {} arguments, got {}",
                    name,
                    def.params.len(),
                    args.len()
                ));
            }

            // Build substitution map: param -> arg
            let subst: FxHashMap<SortId, SortId> = def
                .params
                .iter()
                .zip(args.iter())
                .map(|(&param_spur, &arg_sort)| {
                    let param_sort = self.intern(SortKind::Parameter(param_spur));
                    (param_sort, arg_sort)
                })
                .collect();

            // Apply substitution to the body
            let instantiated = self.substitute_sort(def.body, &subst);
            return Ok(instantiated);
        }

        Err(format!("Unknown parametric sort: {}", name))
    }

    /// Substitute sort parameters with concrete sorts
    pub fn substitute_sort(
        &mut self,
        sort_id: SortId,
        subst: &FxHashMap<SortId, SortId>,
    ) -> SortId {
        // Check if this sort is directly substituted
        if let Some(&replacement) = subst.get(&sort_id) {
            return replacement;
        }

        // Otherwise, recursively substitute in the sort's structure
        let sort = match self.get(sort_id) {
            Some(s) => s.kind.clone(),
            None => return sort_id,
        };

        match sort {
            SortKind::Bool
            | SortKind::Int
            | SortKind::Real
            | SortKind::String
            | SortKind::BitVec(_)
            | SortKind::FloatingPoint { .. } => sort_id,
            SortKind::Uninterpreted(_) => sort_id,
            SortKind::Datatype(_) => sort_id,
            SortKind::Parameter(_) => {
                // If not in subst map, return as-is (free parameter)
                sort_id
            }
            SortKind::Array { domain, range } => {
                let new_domain = self.substitute_sort(domain, subst);
                let new_range = self.substitute_sort(range, subst);
                if new_domain == domain && new_range == range {
                    sort_id
                } else {
                    self.array(new_domain, new_range)
                }
            }
            SortKind::Parametric { name, args } => {
                let new_args: smallvec::SmallVec<[SortId; 2]> = args
                    .iter()
                    .map(|&arg| self.substitute_sort(arg, subst))
                    .collect();

                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    sort_id
                } else {
                    self.intern(SortKind::Parametric {
                        name,
                        args: new_args,
                    })
                }
            }
        }
    }

    /// Get the name of a sort if it has one
    #[must_use]
    pub fn sort_name(&self, sort_id: SortId) -> Option<String> {
        let sort = self.get(sort_id)?;
        match &sort.kind {
            SortKind::Bool => Some("Bool".to_string()),
            SortKind::Int => Some("Int".to_string()),
            SortKind::Real => Some("Real".to_string()),
            SortKind::String => Some("String".to_string()),
            SortKind::BitVec(w) => Some(format!("BitVec({})", w)),
            SortKind::FloatingPoint { eb, sb } => Some(format!("FloatingPoint({}, {})", eb, sb)),
            SortKind::Array { .. } => Some("Array".to_string()),
            SortKind::Uninterpreted(spur) => Some(self.interner.resolve(spur).to_string()),
            SortKind::Parameter(spur) => Some(self.interner.resolve(spur).to_string()),
            SortKind::Parametric { name, .. } => Some(self.interner.resolve(name).to_string()),
            SortKind::Datatype(spur) => Some(self.interner.resolve(spur).to_string()),
        }
    }

    /// Check if a sort is a parametric sort (instantiated)
    #[must_use]
    pub fn is_parametric(&self, sort_id: SortId) -> bool {
        self.get(sort_id)
            .is_some_and(|s| matches!(s.kind, SortKind::Parametric { .. }))
    }

    /// Check if a sort is a sort parameter
    #[must_use]
    pub fn is_sort_parameter(&self, sort_id: SortId) -> bool {
        self.get(sort_id)
            .is_some_and(|s| matches!(s.kind, SortKind::Parameter(_)))
    }

    /// Get the parameter name if this is a sort parameter
    #[must_use]
    pub fn parameter_name(&self, sort_id: SortId) -> Option<&str> {
        let sort = self.get(sort_id)?;
        if let SortKind::Parameter(spur) = &sort.kind {
            Some(self.interner.resolve(spur))
        } else {
            None
        }
    }

    /// Get the arguments of a parametric sort
    #[must_use]
    pub fn parametric_args(&self, sort_id: SortId) -> Option<&[SortId]> {
        let sort = self.get(sort_id)?;
        if let SortKind::Parametric { args, .. } = &sort.kind {
            Some(args.as_slice())
        } else {
            None
        }
    }

    // ========================== Datatype Support ==========================

    /// Declare a datatype
    ///
    /// Creates a new datatype sort with the given name and constructors.
    /// Supports recursive datatypes (e.g., List, Tree).
    ///
    /// # Example
    /// ```ignore
    /// // Declare a List datatype:
    /// // (declare-datatype List
    /// //   ((cons (car Int) (cdr List))
    /// //    (nil)))
    /// let list_sort = manager.mk_datatype_sort("List");
    /// let cons = DataTypeConstructor {
    ///     name: manager.intern_str("cons"),
    ///     selectors: smallvec![
    ///         (manager.intern_str("car"), manager.int_sort),
    ///         (manager.intern_str("cdr"), list_sort),
    ///     ],
    /// };
    /// let nil = DataTypeConstructor {
    ///     name: manager.intern_str("nil"),
    ///     selectors: smallvec![],
    /// };
    /// manager.declare_datatype("List", vec![cons, nil]);
    /// ```
    pub fn declare_datatype(&mut self, name: &str, constructors: Vec<DataTypeConstructor>) {
        let key = self.interner.get_or_intern(name);

        // Create the sort for this datatype
        let sort_id = self.intern(SortKind::Datatype(key));

        // Store the datatype definition
        self.datatypes.insert(
            key,
            DataTypeDef {
                name: key,
                sort_id,
                constructors,
            },
        );
    }

    /// Create a datatype sort reference
    ///
    /// This is used when building recursive datatypes.
    /// It creates a forward reference to a datatype that will be defined later.
    pub fn mk_datatype_sort(&mut self, name: &str) -> SortId {
        let key = self.interner.get_or_intern(name);
        self.intern(SortKind::Datatype(key))
    }

    /// Get a datatype definition by name
    #[must_use]
    pub fn get_datatype(&self, name: &str) -> Option<&DataTypeDef> {
        let key = self.interner.get(name)?;
        self.datatypes.get(&key)
    }

    /// Check if a datatype with the given name is declared
    #[must_use]
    pub fn is_datatype_declared(&self, name: &str) -> bool {
        self.interner
            .get(name)
            .is_some_and(|key| self.datatypes.contains_key(&key))
    }

    /// Check if a sort is a datatype sort
    #[must_use]
    pub fn is_datatype(&self, sort_id: SortId) -> bool {
        self.get(sort_id)
            .is_some_and(|s| matches!(s.kind, SortKind::Datatype(_)))
    }

    /// Get the datatype name if this is a datatype sort
    #[must_use]
    pub fn datatype_name(&self, sort_id: SortId) -> Option<&str> {
        let sort = self.get(sort_id)?;
        if let SortKind::Datatype(spur) = &sort.kind {
            Some(self.interner.resolve(spur))
        } else {
            None
        }
    }

    /// Intern a string into the interner
    ///
    /// This is used for creating Spur values for datatype constructors and selectors.
    pub fn intern_str(&mut self, s: &str) -> lasso::Spur {
        self.interner.get_or_intern(s)
    }

    /// Resolve a Spur back to a string
    #[must_use]
    pub fn resolve_spur(&self, spur: lasso::Spur) -> &str {
        self.interner.resolve(&spur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_manager_common_sorts() {
        let manager = SortManager::new();
        assert!(manager.get(manager.bool_sort).unwrap().is_bool());
        assert!(manager.get(manager.int_sort).unwrap().is_int());
        assert!(manager.get(manager.real_sort).unwrap().is_real());
    }

    #[test]
    fn test_bitvec_sort() {
        let mut manager = SortManager::new();
        let bv32 = manager.bitvec(32);
        let bv64 = manager.bitvec(64);
        let bv32_dup = manager.bitvec(32);

        assert_eq!(bv32, bv32_dup);
        assert_ne!(bv32, bv64);

        let sort = manager.get(bv32).unwrap();
        assert!(sort.is_bitvec());
        assert_eq!(sort.bitvec_width(), Some(32));
    }

    #[test]
    fn test_sort_alias() {
        let mut manager = SortManager::new();

        // Define Word = BitVec(32)
        let bv32 = manager.bitvec(32);
        manager.define_alias("Word", bv32);

        // Resolve the alias
        assert_eq!(manager.resolve_alias("Word"), Some(bv32));
        assert!(manager.is_alias("Word"));

        // Check that it doesn't exist for undefined aliases
        assert_eq!(manager.resolve_alias("Undefined"), None);
        assert!(!manager.is_alias("Undefined"));
    }

    #[test]
    fn test_alias_shadowing() {
        let mut manager = SortManager::new();

        let bv32 = manager.bitvec(32);
        let bv64 = manager.bitvec(64);

        // Define MyType = BitVec(32)
        manager.define_alias("MyType", bv32);
        assert_eq!(manager.resolve_alias("MyType"), Some(bv32));

        // Redefine MyType = BitVec(64)
        manager.define_alias("MyType", bv64);
        assert_eq!(manager.resolve_alias("MyType"), Some(bv64));
    }

    #[test]
    fn test_undefine_alias() {
        let mut manager = SortManager::new();

        let bv32 = manager.bitvec(32);
        manager.define_alias("Word", bv32);
        assert!(manager.is_alias("Word"));

        // Undefine the alias
        let removed = manager.undefine_alias("Word");
        assert_eq!(removed, Some(bv32));
        assert!(!manager.is_alias("Word"));

        // Undefining again returns None
        assert_eq!(manager.undefine_alias("Word"), None);
    }

    #[test]
    fn test_resolve_by_name() {
        let mut manager = SortManager::new();

        // Built-in sorts
        assert_eq!(manager.resolve_by_name("Bool"), Some(manager.bool_sort));
        assert_eq!(manager.resolve_by_name("Int"), Some(manager.int_sort));
        assert_eq!(manager.resolve_by_name("Real"), Some(manager.real_sort));

        // User-defined alias
        let bv32 = manager.bitvec(32);
        manager.define_alias("Word", bv32);
        assert_eq!(manager.resolve_by_name("Word"), Some(bv32));

        // Undefined
        assert_eq!(manager.resolve_by_name("Undefined"), None);
    }

    #[test]
    fn test_aliases_list() {
        let mut manager = SortManager::new();

        let bv32 = manager.bitvec(32);
        let int = manager.int_sort;

        manager.define_alias("Word", bv32);
        manager.define_alias("MyInt", int);

        let aliases = manager.aliases();
        assert_eq!(aliases.len(), 2);

        // Check that both aliases are in the list
        assert!(
            aliases
                .iter()
                .any(|(name, sort)| name == "Word" && *sort == bv32)
        );
        assert!(
            aliases
                .iter()
                .any(|(name, sort)| name == "MyInt" && *sort == int)
        );
    }

    // ========================== Parametric Sort Tests ==========================

    #[test]
    fn test_declare_parametric_sort() {
        let mut manager = SortManager::new();

        // Declare (List 1) - List takes 1 argument
        manager.declare_parametric_sort("List", 1);

        assert!(manager.is_parametric_sort_declared("List"));
        assert_eq!(manager.parametric_sort_arity("List"), Some(1));
        assert!(!manager.is_parametric_sort_declared("Pair"));
    }

    #[test]
    fn test_instantiate_declared_sort() {
        let mut manager = SortManager::new();

        // Declare (List 1)
        manager.declare_parametric_sort("List", 1);

        // Instantiate (List Int)
        let list_int = manager
            .instantiate_parametric_sort("List", &[manager.int_sort])
            .unwrap();

        assert!(manager.is_parametric(list_int));
        assert_eq!(manager.sort_name(list_int), Some("List".to_string()));

        let args = manager.parametric_args(list_int).unwrap();
        assert_eq!(args.len(), 1);
        assert_eq!(args[0], manager.int_sort);
    }

    #[test]
    fn test_instantiate_arity_mismatch() {
        let mut manager = SortManager::new();

        // Declare (List 1)
        manager.declare_parametric_sort("List", 1);

        // Try to instantiate with wrong number of args
        let result = manager.instantiate_parametric_sort("List", &[]);
        assert!(result.is_err());

        let result =
            manager.instantiate_parametric_sort("List", &[manager.int_sort, manager.bool_sort]);
        assert!(result.is_err());
    }

    #[test]
    fn test_define_parametric_sort() {
        let mut manager = SortManager::new();

        // Define (MyList T) = (Array Int T)
        let t_param = manager.mk_sort_parameter("T");
        let array_sort = manager.array(manager.int_sort, t_param);
        manager.define_parametric_sort("MyList", &["T"], array_sort);

        assert!(manager.is_parametric_sort_defined("MyList"));
        assert!(!manager.is_parametric_sort_defined("Other"));
    }

    #[test]
    fn test_instantiate_defined_sort() {
        let mut manager = SortManager::new();

        // Define (MyList T) = (Array Int T)
        let t_param = manager.mk_sort_parameter("T");
        let array_sort = manager.array(manager.int_sort, t_param);
        manager.define_parametric_sort("MyList", &["T"], array_sort);

        // Instantiate (MyList Bool)
        let mylist_bool = manager
            .instantiate_parametric_sort("MyList", &[manager.bool_sort])
            .unwrap();

        // Should expand to (Array Int Bool)
        let sort = manager.get(mylist_bool).unwrap();
        if let SortKind::Array { domain, range } = &sort.kind {
            assert_eq!(*domain, manager.int_sort);
            assert_eq!(*range, manager.bool_sort);
        } else {
            panic!("Expected Array sort");
        }
    }

    #[test]
    fn test_nested_parametric_sort() {
        let mut manager = SortManager::new();

        // Declare (Pair 2)
        manager.declare_parametric_sort("Pair", 2);

        // Create (Pair Int Bool)
        let pair_int_bool = manager
            .instantiate_parametric_sort("Pair", &[manager.int_sort, manager.bool_sort])
            .unwrap();

        // Create (Pair (Pair Int Bool) Real)
        let nested = manager
            .instantiate_parametric_sort("Pair", &[pair_int_bool, manager.real_sort])
            .unwrap();

        assert!(manager.is_parametric(nested));
        let args = manager.parametric_args(nested).unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], pair_int_bool);
        assert_eq!(args[1], manager.real_sort);
    }

    #[test]
    fn test_sort_parameter() {
        let mut manager = SortManager::new();

        let t = manager.mk_sort_parameter("T");
        let u = manager.mk_sort_parameter("U");

        assert!(manager.is_sort_parameter(t));
        assert!(manager.is_sort_parameter(u));
        assert!(!manager.is_sort_parameter(manager.int_sort));

        assert_eq!(manager.parameter_name(t), Some("T"));
        assert_eq!(manager.parameter_name(u), Some("U"));
    }

    #[test]
    fn test_substitute_sort() {
        let mut manager = SortManager::new();

        // Create (Array T U)
        let t = manager.mk_sort_parameter("T");
        let u = manager.mk_sort_parameter("U");
        let array_tu = manager.array(t, u);

        // Substitute T -> Int, U -> Bool
        let t_id = manager.intern(SortKind::Parameter(manager.interner.get("T").unwrap()));
        let u_id = manager.intern(SortKind::Parameter(manager.interner.get("U").unwrap()));

        let mut subst = FxHashMap::default();
        subst.insert(t_id, manager.int_sort);
        subst.insert(u_id, manager.bool_sort);

        let result = manager.substitute_sort(array_tu, &subst);

        // Should be (Array Int Bool)
        let sort = manager.get(result).unwrap();
        if let SortKind::Array { domain, range } = &sort.kind {
            assert_eq!(*domain, manager.int_sort);
            assert_eq!(*range, manager.bool_sort);
        } else {
            panic!("Expected Array sort");
        }
    }

    #[test]
    fn test_parametric_sort_interning() {
        let mut manager = SortManager::new();

        manager.declare_parametric_sort("List", 1);

        // Create two (List Int) instances
        let list_int_1 = manager
            .instantiate_parametric_sort("List", &[manager.int_sort])
            .unwrap();
        let list_int_2 = manager
            .instantiate_parametric_sort("List", &[manager.int_sort])
            .unwrap();

        // Should be the same due to interning
        assert_eq!(list_int_1, list_int_2);

        // (List Bool) should be different
        let list_bool = manager
            .instantiate_parametric_sort("List", &[manager.bool_sort])
            .unwrap();
        assert_ne!(list_int_1, list_bool);
    }

    // ========================== Datatype Tests ==========================

    #[test]
    fn test_simple_datatype() {
        let mut manager = SortManager::new();

        // Create a simple datatype with no recursion
        // (declare-datatype Color ((red) (green) (blue)))
        let red = DataTypeConstructor {
            name: manager.intern_str("red"),
            selectors: smallvec::SmallVec::new(),
        };
        let green = DataTypeConstructor {
            name: manager.intern_str("green"),
            selectors: smallvec::SmallVec::new(),
        };
        let blue = DataTypeConstructor {
            name: manager.intern_str("blue"),
            selectors: smallvec::SmallVec::new(),
        };

        manager.declare_datatype("Color", vec![red, green, blue]);

        assert!(manager.is_datatype_declared("Color"));
        assert!(!manager.is_datatype_declared("Unknown"));

        let color_def = manager.get_datatype("Color").unwrap();
        assert_eq!(manager.resolve_spur(color_def.name), "Color");
        assert_eq!(color_def.constructors.len(), 3);
    }

    #[test]
    fn test_recursive_datatype_list() {
        let mut manager = SortManager::new();

        // Create a recursive list datatype:
        // (declare-datatype List
        //   ((cons (car Int) (cdr List))
        //    (nil)))

        // First create a forward reference to List
        let list_sort = manager.mk_datatype_sort("List");

        let cons = DataTypeConstructor {
            name: manager.intern_str("cons"),
            selectors: smallvec::smallvec![
                (manager.intern_str("car"), manager.int_sort),
                (manager.intern_str("cdr"), list_sort),
            ],
        };
        let nil = DataTypeConstructor {
            name: manager.intern_str("nil"),
            selectors: smallvec::SmallVec::new(),
        };

        manager.declare_datatype("List", vec![cons, nil]);

        assert!(manager.is_datatype_declared("List"));
        assert!(manager.is_datatype(list_sort));
        assert_eq!(manager.datatype_name(list_sort), Some("List"));

        let list_def = manager.get_datatype("List").unwrap();
        assert_eq!(list_def.constructors.len(), 2);
        assert_eq!(manager.resolve_spur(list_def.constructors[0].name), "cons");
        assert_eq!(manager.resolve_spur(list_def.constructors[1].name), "nil");

        // Check the cons constructor's selectors
        let cons_ctor = &list_def.constructors[0];
        assert_eq!(cons_ctor.selectors.len(), 2);
        assert_eq!(manager.resolve_spur(cons_ctor.selectors[0].0), "car");
        assert_eq!(cons_ctor.selectors[0].1, manager.int_sort);
        assert_eq!(manager.resolve_spur(cons_ctor.selectors[1].0), "cdr");
        assert_eq!(cons_ctor.selectors[1].1, list_sort);
    }

    #[test]
    fn test_recursive_datatype_tree() {
        let mut manager = SortManager::new();

        // Create a recursive binary tree datatype:
        // (declare-datatype Tree
        //   ((leaf (value Int))
        //    (node (left Tree) (right Tree))))

        let tree_sort = manager.mk_datatype_sort("Tree");

        let leaf = DataTypeConstructor {
            name: manager.intern_str("leaf"),
            selectors: smallvec::smallvec![(manager.intern_str("value"), manager.int_sort),],
        };
        let node = DataTypeConstructor {
            name: manager.intern_str("node"),
            selectors: smallvec::smallvec![
                (manager.intern_str("left"), tree_sort),
                (manager.intern_str("right"), tree_sort),
            ],
        };

        manager.declare_datatype("Tree", vec![leaf, node]);

        assert!(manager.is_datatype_declared("Tree"));

        let tree_def = manager.get_datatype("Tree").unwrap();
        assert_eq!(tree_def.constructors.len(), 2);

        // Verify the node constructor has two Tree-typed selectors
        let node_ctor = &tree_def.constructors[1];
        assert_eq!(node_ctor.selectors.len(), 2);
        assert_eq!(node_ctor.selectors[0].1, tree_sort);
        assert_eq!(node_ctor.selectors[1].1, tree_sort);
    }

    #[test]
    fn test_datatype_sort_interning() {
        let mut manager = SortManager::new();

        let sort1 = manager.mk_datatype_sort("MyType");
        let sort2 = manager.mk_datatype_sort("MyType");

        // Should be interned to the same ID
        assert_eq!(sort1, sort2);

        let sort3 = manager.mk_datatype_sort("OtherType");
        assert_ne!(sort1, sort3);
    }
}
