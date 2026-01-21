//! Term representation

use crate::sort::SortId;
use lasso::Spur;
use num_bigint::BigInt;
use num_rational::Rational64;
use smallvec::SmallVec;

/// IEEE 754 rounding modes for floating-point operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum RoundingMode {
    /// Round to nearest, ties to even
    RNE,
    /// Round to nearest, ties away from zero
    RNA,
    /// Round toward positive infinity
    RTP,
    /// Round toward negative infinity
    RTN,
    /// Round toward zero
    RTZ,
}

/// Unique identifier for a term in the arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(pub u32);

impl TermId {
    /// Create a new TermId from a raw value
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

impl From<u32> for TermId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// The kind of a term
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TermKind {
    // Constants
    /// Boolean true
    True,
    /// Boolean false
    False,
    /// Integer constant
    IntConst(BigInt),
    /// Rational constant
    RealConst(Rational64),
    /// Bit vector constant (value, width)
    BitVecConst {
        /// Value of the bit vector
        value: BigInt,
        /// Width in bits
        width: u32,
    },

    // Variables
    /// Named variable
    Var(Spur),

    // Boolean operations
    /// Logical NOT
    Not(TermId),
    /// Logical AND
    And(SmallVec<[TermId; 4]>),
    /// Logical OR
    Or(SmallVec<[TermId; 4]>),
    /// Logical XOR
    Xor(TermId, TermId),
    /// Logical implication
    Implies(TermId, TermId),
    /// If-then-else
    Ite(TermId, TermId, TermId),

    // Equality and comparison
    /// Equality
    Eq(TermId, TermId),
    /// Distinct (all different)
    Distinct(SmallVec<[TermId; 4]>),

    // Arithmetic operations
    /// Negation
    Neg(TermId),
    /// Addition
    Add(SmallVec<[TermId; 4]>),
    /// Subtraction
    Sub(TermId, TermId),
    /// Multiplication
    Mul(SmallVec<[TermId; 4]>),
    /// Integer division
    Div(TermId, TermId),
    /// Modulo
    Mod(TermId, TermId),
    /// Less than
    Lt(TermId, TermId),
    /// Less than or equal
    Le(TermId, TermId),
    /// Greater than
    Gt(TermId, TermId),
    /// Greater than or equal
    Ge(TermId, TermId),

    // BitVector operations
    /// Bit vector concatenation
    BvConcat(TermId, TermId),
    /// Bit vector extraction (high, low, arg)
    BvExtract {
        /// High bit index
        high: u32,
        /// Low bit index
        low: u32,
        /// Term to extract from
        arg: TermId,
    },
    /// Bit vector NOT
    BvNot(TermId),
    /// Bit vector AND
    BvAnd(TermId, TermId),
    /// Bit vector OR
    BvOr(TermId, TermId),
    /// Bit vector XOR
    BvXor(TermId, TermId),
    /// Bit vector addition
    BvAdd(TermId, TermId),
    /// Bit vector subtraction
    BvSub(TermId, TermId),
    /// Bit vector multiplication
    BvMul(TermId, TermId),
    /// Bit vector unsigned division
    BvUdiv(TermId, TermId),
    /// Bit vector signed division
    BvSdiv(TermId, TermId),
    /// Bit vector unsigned remainder
    BvUrem(TermId, TermId),
    /// Bit vector signed remainder
    BvSrem(TermId, TermId),
    /// Bit vector shift left
    BvShl(TermId, TermId),
    /// Bit vector logical shift right
    BvLshr(TermId, TermId),
    /// Bit vector arithmetic shift right
    BvAshr(TermId, TermId),
    /// Bit vector unsigned less than
    BvUlt(TermId, TermId),
    /// Bit vector unsigned less than or equal
    BvUle(TermId, TermId),
    /// Bit vector signed less than
    BvSlt(TermId, TermId),
    /// Bit vector signed less than or equal
    BvSle(TermId, TermId),

    // Array operations
    /// Array select
    Select(TermId, TermId),
    /// Array store
    Store(TermId, TermId, TermId),

    // String operations
    /// String literal
    StringLit(String),
    /// String concatenation
    StrConcat(TermId, TermId),
    /// String length
    StrLen(TermId),
    /// Substring (string, start, length)
    StrSubstr(TermId, TermId, TermId),
    /// Character at index
    StrAt(TermId, TermId),
    /// Contains substring
    StrContains(TermId, TermId),
    /// Prefix of
    StrPrefixOf(TermId, TermId),
    /// Suffix of
    StrSuffixOf(TermId, TermId),
    /// Index of (string, substring, offset)
    StrIndexOf(TermId, TermId, TermId),
    /// Replace (string, pattern, replacement)
    StrReplace(TermId, TermId, TermId),
    /// Replace all (string, pattern, replacement)
    StrReplaceAll(TermId, TermId, TermId),
    /// String to integer conversion
    StrToInt(TermId),
    /// Integer to string conversion
    IntToStr(TermId),
    /// String in regular expression
    StrInRe(TermId, TermId),

    // Floating-point operations
    /// Floating-point numeral (sign_bit, exponent_bitvec, significand_bitvec)
    /// Represented as three bitvectors
    FpLit {
        /// Sign bit (1 bit)
        sign: bool,
        /// Exponent bitvector
        exp: BigInt,
        /// Significand bitvector
        sig: BigInt,
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Floating-point positive infinity
    FpPlusInfinity {
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Floating-point negative infinity
    FpMinusInfinity {
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Floating-point positive zero
    FpPlusZero {
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Floating-point negative zero
    FpMinusZero {
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Floating-point NaN
    FpNaN {
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },

    // FP unary operations
    /// Floating-point absolute value
    FpAbs(TermId),
    /// Floating-point negation
    FpNeg(TermId),
    /// Floating-point square root (rounding mode, arg)
    FpSqrt(RoundingMode, TermId),
    /// Round to integral (rounding mode, arg)
    FpRoundToIntegral(RoundingMode, TermId),

    // FP binary operations
    /// Floating-point addition (rounding mode, lhs, rhs)
    FpAdd(RoundingMode, TermId, TermId),
    /// Floating-point subtraction (rounding mode, lhs, rhs)
    FpSub(RoundingMode, TermId, TermId),
    /// Floating-point multiplication (rounding mode, lhs, rhs)
    FpMul(RoundingMode, TermId, TermId),
    /// Floating-point division (rounding mode, lhs, rhs)
    FpDiv(RoundingMode, TermId, TermId),
    /// Floating-point remainder (lhs, rhs)
    FpRem(TermId, TermId),
    /// Floating-point minimum
    FpMin(TermId, TermId),
    /// Floating-point maximum
    FpMax(TermId, TermId),
    /// Floating-point less than or equal
    FpLeq(TermId, TermId),
    /// Floating-point less than
    FpLt(TermId, TermId),
    /// Floating-point greater than or equal
    FpGeq(TermId, TermId),
    /// Floating-point greater than
    FpGt(TermId, TermId),
    /// Floating-point equality
    FpEq(TermId, TermId),

    // FP ternary operations
    /// Fused multiply-add: (x * y) + z (rounding mode, x, y, z)
    FpFma(RoundingMode, TermId, TermId, TermId),

    // FP predicates
    /// Check if normal floating-point number
    FpIsNormal(TermId),
    /// Check if subnormal floating-point number
    FpIsSubnormal(TermId),
    /// Check if zero
    FpIsZero(TermId),
    /// Check if infinite
    FpIsInfinite(TermId),
    /// Check if NaN
    FpIsNaN(TermId),
    /// Check if negative
    FpIsNegative(TermId),
    /// Check if positive
    FpIsPositive(TermId),

    // FP conversions
    /// Convert to floating-point from another FP format (rm, arg, target_eb, target_sb)
    FpToFp {
        /// Rounding mode
        rm: RoundingMode,
        /// Term to convert
        arg: TermId,
        /// Target exponent width
        eb: u32,
        /// Target significand width
        sb: u32,
    },
    /// Convert floating-point to signed bitvector (rm, arg, width)
    FpToSBV {
        /// Rounding mode
        rm: RoundingMode,
        /// Term to convert
        arg: TermId,
        /// Target bitvector width
        width: u32,
    },
    /// Convert floating-point to unsigned bitvector (rm, arg, width)
    FpToUBV {
        /// Rounding mode
        rm: RoundingMode,
        /// Term to convert
        arg: TermId,
        /// Target bitvector width
        width: u32,
    },
    /// Convert floating-point to real
    FpToReal(TermId),
    /// Convert real to floating-point (rm, arg, eb, sb)
    RealToFp {
        /// Rounding mode
        rm: RoundingMode,
        /// Term to convert
        arg: TermId,
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Convert signed bitvector to floating-point (rm, arg, eb, sb)
    SBVToFp {
        /// Rounding mode
        rm: RoundingMode,
        /// Term to convert
        arg: TermId,
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },
    /// Convert unsigned bitvector to floating-point (rm, arg, eb, sb)
    UBVToFp {
        /// Rounding mode
        rm: RoundingMode,
        /// Term to convert
        arg: TermId,
        /// Exponent width
        eb: u32,
        /// Significand width
        sb: u32,
    },

    // Uninterpreted functions
    /// Function application
    Apply {
        /// Function symbol
        func: Spur,
        /// Arguments
        args: SmallVec<[TermId; 4]>,
    },

    // Algebraic datatypes
    /// Datatype constructor application
    DtConstructor {
        /// Constructor name (e.g., "cons", "nil")
        constructor: Spur,
        /// Arguments to the constructor
        args: SmallVec<[TermId; 4]>,
    },
    /// Datatype tester/discriminator (checks if term was built with a specific constructor)
    DtTester {
        /// Constructor name to test for (e.g., "is-cons")
        constructor: Spur,
        /// Term to test
        arg: TermId,
    },
    /// Datatype selector/accessor (extracts a field from a constructor)
    DtSelector {
        /// Selector name (e.g., "head", "tail")
        selector: Spur,
        /// Term to extract from
        arg: TermId,
    },

    // Quantifiers
    /// Universal quantification
    Forall {
        /// Bound variables (name, sort)
        vars: SmallVec<[(Spur, SortId); 2]>,
        /// Body
        body: TermId,
        /// Instantiation patterns (triggers)
        /// Each pattern is a list of terms that guide instantiation
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    },
    /// Existential quantification
    Exists {
        /// Bound variables (name, sort)
        vars: SmallVec<[(Spur, SortId); 2]>,
        /// Body
        body: TermId,
        /// Instantiation patterns (triggers)
        /// Each pattern is a list of terms that guide instantiation
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    },

    // Let bindings
    /// Let expression
    Let {
        /// Bindings (name, value)
        bindings: SmallVec<[(Spur, TermId); 2]>,
        /// Body
        body: TermId,
    },

    // Match expressions
    /// Pattern matching on algebraic datatypes
    Match {
        /// The term being matched (scrutinee)
        scrutinee: TermId,
        /// The match cases (patterns and bodies)
        cases: SmallVec<[MatchCase; 4]>,
    },
}

/// A case in a match expression.
///
/// Each case consists of a pattern (constructor with optional bindings)
/// and a body expression to evaluate when the pattern matches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchCase {
    /// The constructor to match, or None for a variable/wildcard pattern
    pub constructor: Option<Spur>,
    /// Variable bindings for constructor arguments
    pub bindings: SmallVec<[Spur; 4]>,
    /// The body expression to evaluate when matched
    pub body: TermId,
}

/// A term in the SMT-LIB2 sense
#[derive(Debug, Clone)]
pub struct Term {
    /// Unique identifier
    pub id: TermId,
    /// The kind of this term
    pub kind: TermKind,
    /// The sort of this term
    pub sort: SortId,
}

/// Instantiation pattern for quantifier triggers.
/// Each pattern is a list of terms that guide quantifier instantiation.
pub type InstPattern = SmallVec<[TermId; 2]>;

impl Term {
    /// Check if this term is a constant
    #[must_use]
    pub fn is_const(&self) -> bool {
        matches!(
            self.kind,
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
        )
    }

    /// Check if this term is a variable
    #[must_use]
    pub fn is_var(&self) -> bool {
        matches!(self.kind, TermKind::Var(_))
    }

    /// Check if this term is a boolean constant
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self.kind {
            TermKind::True => Some(true),
            TermKind::False => Some(false),
            _ => None,
        }
    }
}
