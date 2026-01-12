//! Format conversion utilities.
//!
//! This module provides conversion utilities between different proof formats:
//! - DRAT to Alethe: Convert SAT proofs to SMT format
//! - Alethe to LFSC: Convert SMT proofs to typed proof format

use crate::alethe::{AletheProof, AletheRule, AletheStep};
use crate::drat::{DratProof, DratStep};
use crate::lfsc::{LfscProof, LfscSort, LfscTerm};
use std::fmt;

/// Result of format conversion.
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Errors that can occur during format conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionError {
    /// Unsupported conversion
    UnsupportedConversion { from: String, to: String },
    /// Information loss during conversion
    InformationLoss { reason: String },
    /// Invalid source format
    InvalidSource { reason: String },
    /// Conversion failed
    ConversionFailed { reason: String },
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::UnsupportedConversion { from, to } => {
                write!(f, "Unsupported conversion from {} to {}", from, to)
            }
            ConversionError::InformationLoss { reason } => {
                write!(f, "Information loss during conversion: {}", reason)
            }
            ConversionError::InvalidSource { reason } => {
                write!(f, "Invalid source format: {}", reason)
            }
            ConversionError::ConversionFailed { reason } => {
                write!(f, "Conversion failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for ConversionError {}

/// Converter for proof formats.
///
/// This is a placeholder for future conversion implementations.
/// Currently supports basic error types and scaffolding.
pub struct FormatConverter {
    /// Preserve all information (may fail if not possible)
    strict_mode: bool,
    /// Generate comments in target format
    add_comments: bool,
}

impl Default for FormatConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatConverter {
    /// Create a new format converter.
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            add_comments: true,
        }
    }

    /// Enable strict mode (fail on information loss).
    pub fn strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Enable/disable comment generation.
    pub fn add_comments(mut self, add: bool) -> Self {
        self.add_comments = add;
        self
    }

    /// Convert DRAT proof to Alethe proof.
    ///
    /// This conversion maps:
    /// - DRAT Add steps to Alethe Input steps (for clauses from the problem)
    /// - Subsequent Add steps to Resolution steps
    /// - Delete steps are skipped (Alethe is monotonic)
    ///
    /// Note: This is a best-effort conversion. DRAT proofs are untyped and
    /// don't contain all the information needed for a complete Alethe proof.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_proof::conversion::{FormatConverter};
    /// use oxiz_proof::drat::DratProof;
    ///
    /// let converter = FormatConverter::new();
    /// let mut drat = DratProof::new();
    ///
    /// // Add some clauses
    /// drat.add_clause(vec![1, 2, -3]);
    /// drat.add_clause(vec![-1, 4]);
    ///
    /// // Convert to Alethe
    /// let alethe = converter.drat_to_alethe(&drat).unwrap();
    /// assert_eq!(alethe.len(), 2);
    /// ```
    pub fn drat_to_alethe(&self, drat: &DratProof) -> ConversionResult<AletheProof> {
        let mut alethe = AletheProof::new();

        // Track step indices for premises
        let mut step_indices = Vec::new();

        for (i, step) in drat.steps().iter().enumerate() {
            match step {
                DratStep::Add(clause) => {
                    // Convert DRAT literals to Alethe terms
                    let alethe_clause: Vec<String> = clause
                        .iter()
                        .map(|&lit| {
                            if lit == 0 {
                                // Terminator - skip
                                return String::new();
                            }
                            if lit > 0 {
                                format!("p{}", lit)
                            } else {
                                format!("(not p{})", -lit)
                            }
                        })
                        .filter(|s| !s.is_empty())
                        .collect();

                    // First few clauses are inputs, later ones are derived
                    let rule = if i < 5 {
                        // Heuristic: first clauses are likely from the problem
                        AletheRule::Input
                    } else {
                        // Derived clauses use resolution
                        AletheRule::Resolution
                    };

                    // Add the step
                    let premises = if rule == AletheRule::Resolution && !step_indices.is_empty() {
                        // Use last two steps as premises (simplified heuristic)
                        let len = step_indices.len();
                        if len >= 2 {
                            vec![step_indices[len - 2], step_indices[len - 1]]
                        } else {
                            vec![step_indices[len - 1]]
                        }
                    } else {
                        Vec::new()
                    };

                    let idx = alethe.step(alethe_clause, rule, premises, Vec::new());
                    step_indices.push(idx);
                }
                DratStep::Delete(_clause) => {
                    // Alethe is monotonic, so deletions are not represented
                    // In strict mode, this is an error
                    if self.strict_mode {
                        return Err(ConversionError::InformationLoss {
                            reason:
                                "DRAT deletions cannot be represented in Alethe (monotonic format)"
                                    .to_string(),
                        });
                    }
                    // Otherwise, skip the deletion
                }
            }
        }

        Ok(alethe)
    }

    /// Convert Alethe proof to LFSC proof.
    ///
    /// This conversion maps:
    /// - Alethe Assume steps to LFSC variable declarations
    /// - Alethe Step steps to LFSC proof term applications
    /// - Alethe rules to corresponding LFSC proof rules
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiz_proof::conversion::FormatConverter;
    /// use oxiz_proof::alethe::{AletheProof, AletheRule};
    ///
    /// let converter = FormatConverter::new();
    /// let mut alethe = AletheProof::new();
    ///
    /// // Add assumptions and a resolution step
    /// let a1 = alethe.assume("(or p q)");
    /// let a2 = alethe.assume("(not p)");
    /// alethe.resolution(vec!["q".to_string()], vec![a1, a2]);
    ///
    /// // Convert to LFSC
    /// let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();
    /// assert!(!lfsc.is_empty());
    /// ```
    pub fn alethe_to_lfsc(&self, alethe: &AletheProof) -> ConversionResult<LfscProof> {
        let mut lfsc = LfscProof::new();

        // Add standard LFSC signatures for boolean reasoning
        self.add_boolean_signature(&mut lfsc);

        // Convert each Alethe step
        for step in alethe.steps() {
            match step {
                AletheStep::Assume { index, term } => {
                    // Declare assumption as a proof variable
                    let var_name = format!("t{}", index);
                    let formula = Self::term_to_lfsc_term(term)?;

                    // Declare as a proof that the formula holds
                    lfsc.declare_const(&var_name, LfscSort::Named("proof".to_string()));

                    if self.add_comments {
                        // Add a comment about the assumption (as a definition)
                        lfsc.define(
                            format!("{}_formula", var_name),
                            LfscSort::Named("formula".to_string()),
                            formula,
                        );
                    }
                }
                AletheStep::Step {
                    index,
                    clause,
                    rule,
                    premises,
                    args: _args,
                } => {
                    let var_name = format!("t{}", index);

                    // Build LFSC proof term
                    let proof_term = self.alethe_rule_to_lfsc(*rule, clause, premises)?;

                    // Define the proof step
                    lfsc.define(var_name, LfscSort::Named("proof".to_string()), proof_term);
                }
                AletheStep::Anchor { step, args } => {
                    // Anchors represent local scopes in Alethe
                    // In LFSC, we can represent this as a lambda abstraction
                    let var_name = format!("anchor{}", step);

                    // Create lambda for each argument
                    let mut body = LfscTerm::Var(var_name.clone());
                    for (arg_name, arg_sort) in args.iter().rev() {
                        let sort = Self::parse_sort(arg_sort)?;
                        body = LfscTerm::Lambda(arg_name.clone(), Box::new(sort), Box::new(body));
                    }

                    lfsc.define(var_name, LfscSort::Named("proof".to_string()), body);
                }
                AletheStep::DefineFun {
                    name,
                    args,
                    return_sort,
                    body,
                } => {
                    // Convert to LFSC definition
                    let lfsc_sort = Self::parse_sort(return_sort)?;
                    let lfsc_body = Self::term_to_lfsc_term(body)?;

                    // Wrap in lambdas for each argument
                    let mut def_body = lfsc_body;
                    for (arg_name, arg_sort) in args.iter().rev() {
                        let sort = Self::parse_sort(arg_sort)?;
                        def_body =
                            LfscTerm::Lambda(arg_name.clone(), Box::new(sort), Box::new(def_body));
                    }

                    lfsc.define(name.clone(), lfsc_sort, def_body);
                }
            }
        }

        Ok(lfsc)
    }

    /// Add standard boolean signature to LFSC proof
    fn add_boolean_signature(&self, lfsc: &mut LfscProof) {
        // Declare formula sort
        lfsc.declare_sort("formula", 0);

        // Declare proof sort
        lfsc.declare_sort("proof", 0);

        // Declare holds predicate
        lfsc.declare_const(
            "holds",
            LfscSort::Arrow(
                Box::new(LfscSort::Named("formula".to_string())),
                Box::new(LfscSort::Type),
            ),
        );

        // Declare boolean connectives
        lfsc.declare_const("true", LfscSort::Named("formula".to_string()));
        lfsc.declare_const("false", LfscSort::Named("formula".to_string()));

        lfsc.declare_const(
            "not",
            LfscSort::Arrow(
                Box::new(LfscSort::Named("formula".to_string())),
                Box::new(LfscSort::Named("formula".to_string())),
            ),
        );

        lfsc.declare_const(
            "and",
            LfscSort::Arrow(
                Box::new(LfscSort::Named("formula".to_string())),
                Box::new(LfscSort::Arrow(
                    Box::new(LfscSort::Named("formula".to_string())),
                    Box::new(LfscSort::Named("formula".to_string())),
                )),
            ),
        );

        lfsc.declare_const(
            "or",
            LfscSort::Arrow(
                Box::new(LfscSort::Named("formula".to_string())),
                Box::new(LfscSort::Arrow(
                    Box::new(LfscSort::Named("formula".to_string())),
                    Box::new(LfscSort::Named("formula".to_string())),
                )),
            ),
        );
    }

    /// Convert Alethe rule to LFSC proof term
    fn alethe_rule_to_lfsc(
        &self,
        rule: AletheRule,
        _clause: &[String],
        premises: &[u32],
    ) -> ConversionResult<LfscTerm> {
        // Map Alethe rules to LFSC proof rule applications
        let rule_name = match rule {
            AletheRule::Resolution => "resolution",
            AletheRule::Trans => "trans",
            AletheRule::Cong => "cong",
            AletheRule::Refl => "refl",
            AletheRule::Symm => "symm",
            AletheRule::EqRefl => "eq_refl",
            AletheRule::EqSymm => "eq_symm",
            AletheRule::EqTrans => "eq_trans",
            AletheRule::EqCong => "eq_cong",
            AletheRule::Input => "input",
            _ => {
                return Err(ConversionError::UnsupportedConversion {
                    from: format!("Alethe rule {:?}", rule),
                    to: "LFSC".to_string(),
                });
            }
        };

        // Build application of rule to premises
        let premise_terms: Vec<LfscTerm> = premises
            .iter()
            .map(|&idx| LfscTerm::Var(format!("t{}", idx)))
            .collect();

        Ok(LfscTerm::App(rule_name.to_string(), premise_terms))
    }

    /// Convert SMT-LIB term string to LFSC term
    ///
    /// This is an improved parser that handles:
    /// - Nested s-expressions
    /// - Negative numbers
    /// - Rational literals
    /// - Proper tokenization
    fn term_to_lfsc_term(term: &str) -> ConversionResult<LfscTerm> {
        let trimmed = term.trim();

        // Handle boolean literals
        if trimmed == "true" {
            return Ok(LfscTerm::True);
        } else if trimmed == "false" {
            return Ok(LfscTerm::False);
        }

        // Handle s-expressions
        if trimmed.starts_with('(') && trimmed.ends_with(')') {
            return Self::parse_sexpr(trimmed);
        }

        // Handle numeric literals
        if let Some(term) = Self::try_parse_number(trimmed) {
            return Ok(term);
        }

        // Everything else is a variable/symbol
        Ok(LfscTerm::Var(trimmed.to_string()))
    }

    /// Try to parse a number (integer or rational)
    fn try_parse_number(s: &str) -> Option<LfscTerm> {
        // Try parsing as rational (e.g., "3/4")
        if let Some(slash_pos) = s.find('/') {
            let num_str = &s[..slash_pos];
            let den_str = &s[slash_pos + 1..];

            if let (Ok(num), Ok(den)) = (num_str.parse::<i64>(), den_str.parse::<i64>()) {
                return Some(LfscTerm::RatLit(num, den));
            }
        }

        // Try parsing as integer
        if let Ok(n) = s.parse::<i64>() {
            return Some(LfscTerm::IntLit(n));
        }

        None
    }

    /// Parse an s-expression into an LFSC term
    fn parse_sexpr(s: &str) -> ConversionResult<LfscTerm> {
        let inner = &s[1..s.len() - 1].trim();

        if inner.is_empty() {
            return Err(ConversionError::InvalidSource {
                reason: "Empty s-expression".to_string(),
            });
        }

        // Tokenize the s-expression (handling nested parens)
        let tokens = Self::tokenize_sexpr(inner)?;

        if tokens.is_empty() {
            return Err(ConversionError::InvalidSource {
                reason: "Empty s-expression".to_string(),
            });
        }

        let func = &tokens[0];
        let args: ConversionResult<Vec<LfscTerm>> = tokens[1..]
            .iter()
            .map(|arg| Self::term_to_lfsc_term(arg))
            .collect();

        Ok(LfscTerm::App(func.to_string(), args?))
    }

    /// Tokenize an s-expression, respecting nested parentheses
    fn tokenize_sexpr(s: &str) -> ConversionResult<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        let mut in_string = false;

        for ch in s.chars() {
            match ch {
                '"' => {
                    in_string = !in_string;
                    current.push(ch);
                }
                '(' if !in_string => {
                    if depth == 0 && !current.trim().is_empty() {
                        tokens.push(current.trim().to_string());
                        current.clear();
                    }
                    depth += 1;
                    current.push(ch);
                }
                ')' if !in_string => {
                    depth -= 1;
                    current.push(ch);
                    if depth == 0 {
                        tokens.push(current.trim().to_string());
                        current.clear();
                    }
                }
                ' ' | '\t' | '\n' if !in_string && depth == 0 => {
                    if !current.trim().is_empty() {
                        tokens.push(current.trim().to_string());
                        current.clear();
                    }
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        if in_string {
            return Err(ConversionError::InvalidSource {
                reason: "Unterminated string literal".to_string(),
            });
        }

        if depth != 0 {
            return Err(ConversionError::InvalidSource {
                reason: "Unbalanced parentheses in s-expression".to_string(),
            });
        }

        if !current.trim().is_empty() {
            tokens.push(current.trim().to_string());
        }

        Ok(tokens)
    }

    /// Parse SMT-LIB sort string to LFSC sort
    fn parse_sort(sort: &str) -> ConversionResult<LfscSort> {
        match sort.trim() {
            "Bool" => Ok(LfscSort::Bool),
            "Int" => Ok(LfscSort::Int),
            "Real" => Ok(LfscSort::Real),
            s if s.starts_with("(_ BitVec ") => {
                // Parse bitvector width
                let width_str = s.trim_start_matches("(_ BitVec ").trim_end_matches(')');
                let width =
                    width_str
                        .parse::<u32>()
                        .map_err(|_| ConversionError::InvalidSource {
                            reason: format!("Invalid bitvector width: {}", width_str),
                        })?;
                Ok(LfscSort::BitVec(width))
            }
            other => Ok(LfscSort::Named(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_converter_new() {
        let converter = FormatConverter::new();
        assert!(!converter.strict_mode);
        assert!(converter.add_comments);
    }

    #[test]
    fn test_converter_with_settings() {
        let converter = FormatConverter::new().strict_mode(true).add_comments(false);
        assert!(converter.strict_mode);
        assert!(!converter.add_comments);
    }

    #[test]
    fn test_conversion_error_display() {
        let err = ConversionError::UnsupportedConversion {
            from: "DRAT".to_string(),
            to: "Coq".to_string(),
        };
        assert!(err.to_string().contains("Unsupported conversion"));
    }

    // DRAT to Alethe conversion tests

    #[test]
    fn test_drat_to_alethe_empty() {
        let converter = FormatConverter::new();
        let drat = DratProof::new();
        let alethe = converter.drat_to_alethe(&drat).unwrap();

        assert!(alethe.is_empty());
    }

    #[test]
    fn test_drat_to_alethe_single_clause() {
        let converter = FormatConverter::new();
        let mut drat = DratProof::new();
        drat.add_clause(vec![1, 2, -3]);

        let alethe = converter.drat_to_alethe(&drat).unwrap();

        assert_eq!(alethe.len(), 1);
        let output = alethe.to_string();
        assert!(output.contains("p1"));
        assert!(output.contains("p2"));
        assert!(output.contains("(not p3)"));
    }

    #[test]
    fn test_drat_to_alethe_multiple_clauses() {
        let converter = FormatConverter::new();
        let mut drat = DratProof::new();
        drat.add_clause(vec![1, 2]);
        drat.add_clause(vec![-1, 3]);
        drat.add_clause(vec![-2, 4]);
        drat.add_clause(vec![2, -3]);

        let alethe = converter.drat_to_alethe(&drat).unwrap();

        assert_eq!(alethe.len(), 4);
        // First clauses should be Input
        let output = alethe.to_string();
        assert!(output.contains(":rule input"));
    }

    #[test]
    fn test_drat_to_alethe_with_resolution() {
        let converter = FormatConverter::new();
        let mut drat = DratProof::new();

        // Add enough clauses to trigger resolution heuristic
        for i in 1..=6 {
            drat.add_clause(vec![i, -i]);
        }

        let alethe = converter.drat_to_alethe(&drat).unwrap();

        assert_eq!(alethe.len(), 6);
        let output = alethe.to_string();
        // Later clauses should use resolution
        assert!(output.contains(":rule resolution"));
    }

    #[test]
    fn test_drat_to_alethe_with_deletion_non_strict() {
        let converter = FormatConverter::new();
        let mut drat = DratProof::new();
        drat.add_clause(vec![1, 2]);
        drat.delete_clause(vec![1, 2]);
        drat.add_clause(vec![3, 4]);

        // Should succeed in non-strict mode, skipping deletion
        let alethe = converter.drat_to_alethe(&drat).unwrap();

        // Only the additions should be present
        assert_eq!(alethe.len(), 2);
    }

    #[test]
    fn test_drat_to_alethe_with_deletion_strict() {
        let converter = FormatConverter::new().strict_mode(true);
        let mut drat = DratProof::new();
        drat.add_clause(vec![1, 2]);
        drat.delete_clause(vec![1, 2]);

        // Should fail in strict mode
        let result = converter.drat_to_alethe(&drat);
        assert!(result.is_err());

        if let Err(ConversionError::InformationLoss { reason }) = result {
            assert!(reason.contains("monotonic"));
        } else {
            panic!("Expected InformationLoss error");
        }
    }

    #[test]
    fn test_drat_to_alethe_empty_clause() {
        let converter = FormatConverter::new();
        let mut drat = DratProof::new();
        drat.add_clause(vec![1]);
        drat.add_clause(vec![-1]);
        drat.add_clause(vec![]); // Empty clause (contradiction)

        let alethe = converter.drat_to_alethe(&drat).unwrap();

        assert_eq!(alethe.len(), 3);
        let output = alethe.to_string();
        // Empty clause represented as (cl)
        assert!(output.contains("(cl)"));
    }

    // Alethe to LFSC conversion tests

    #[test]
    fn test_alethe_to_lfsc_empty() {
        let converter = FormatConverter::new();
        let alethe = AletheProof::new();
        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        // Should have the boolean signature
        assert!(!lfsc.is_empty());
        let output = lfsc.to_string();
        assert!(output.contains("formula"));
        assert!(output.contains("proof"));
    }

    #[test]
    fn test_alethe_to_lfsc_assumption() {
        let converter = FormatConverter::new();
        let mut alethe = AletheProof::new();
        alethe.assume("(= x 5)");

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(output.contains("t1"));
        assert!(output.contains("proof"));
    }

    #[test]
    fn test_alethe_to_lfsc_resolution() {
        let converter = FormatConverter::new();
        let mut alethe = AletheProof::new();
        let a1 = alethe.assume("(or p q)");
        let a2 = alethe.assume("(not p)");
        alethe.resolution(vec!["q".to_string()], vec![a1, a2]);

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(output.contains("resolution"));
        assert!(output.contains("t1"));
        assert!(output.contains("t2"));
        assert!(output.contains("t3"));
    }

    #[test]
    fn test_alethe_to_lfsc_equality_rules() {
        let converter = FormatConverter::new();
        let mut alethe = AletheProof::new();

        alethe.step_simple(vec!["(= x x)".to_string()], AletheRule::EqRefl);
        alethe.step_simple(vec!["(= y x)".to_string()], AletheRule::EqSymm);

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(output.contains("eq_refl"));
        assert!(output.contains("eq_symm"));
    }

    #[test]
    fn test_alethe_to_lfsc_anchor() {
        let converter = FormatConverter::new();
        let mut alethe = AletheProof::new();

        alethe.anchor(vec![("x".to_string(), "Int".to_string())]);

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(output.contains("anchor1"));
    }

    #[test]
    fn test_alethe_to_lfsc_define_fun() {
        let converter = FormatConverter::new();
        let mut alethe = AletheProof::new();

        alethe.define_fun(
            "f",
            vec![("x".to_string(), "Int".to_string())],
            "Int",
            "(+ x 1)",
        );

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(output.contains("(define f"));
    }

    #[test]
    fn test_alethe_to_lfsc_with_comments() {
        let converter = FormatConverter::new().add_comments(true);
        let mut alethe = AletheProof::new();
        alethe.assume("(= x 5)");

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(output.contains("t1_formula"));
    }

    #[test]
    fn test_alethe_to_lfsc_without_comments() {
        let converter = FormatConverter::new().add_comments(false);
        let mut alethe = AletheProof::new();
        alethe.assume("(= x 5)");

        let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

        let output = lfsc.to_string();
        assert!(!output.contains("t1_formula"));
    }

    #[test]
    fn test_alethe_to_lfsc_unsupported_rule() {
        let converter = FormatConverter::new();
        let mut alethe = AletheProof::new();

        // Use a rule that doesn't have LFSC mapping
        alethe.step_simple(vec!["p".to_string()], AletheRule::Skolem);

        let result = converter.alethe_to_lfsc(&alethe);
        assert!(result.is_err());

        if let Err(ConversionError::UnsupportedConversion { from, to }) = result {
            assert!(from.contains("Skolem"));
            assert_eq!(to, "LFSC");
        } else {
            panic!("Expected UnsupportedConversion error");
        }
    }

    // Helper function tests

    #[test]
    fn test_term_to_lfsc_term_literals() {
        assert!(matches!(
            FormatConverter::term_to_lfsc_term("true"),
            Ok(LfscTerm::True)
        ));
        assert!(matches!(
            FormatConverter::term_to_lfsc_term("false"),
            Ok(LfscTerm::False)
        ));
        assert!(matches!(
            FormatConverter::term_to_lfsc_term("42"),
            Ok(LfscTerm::IntLit(42))
        ));
    }

    #[test]
    fn test_term_to_lfsc_term_variable() {
        if let Ok(LfscTerm::Var(name)) = FormatConverter::term_to_lfsc_term("x") {
            assert_eq!(name, "x");
        } else {
            panic!("Expected Var term");
        }
    }

    #[test]
    fn test_term_to_lfsc_term_application() {
        if let Ok(LfscTerm::App(func, args)) = FormatConverter::term_to_lfsc_term("(+ x y)") {
            assert_eq!(func, "+");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected App term");
        }
    }

    #[test]
    fn test_term_to_lfsc_term_nested() {
        // Test nested s-expressions
        if let Ok(LfscTerm::App(func, args)) = FormatConverter::term_to_lfsc_term("(+ (* 2 x) y)") {
            assert_eq!(func, "+");
            assert_eq!(args.len(), 2);

            // First arg should be (* 2 x)
            if let LfscTerm::App(inner_func, inner_args) = &args[0] {
                assert_eq!(inner_func, "*");
                assert_eq!(inner_args.len(), 2);
            } else {
                panic!("Expected nested App term");
            }
        } else {
            panic!("Expected App term");
        }
    }

    #[test]
    fn test_term_to_lfsc_term_negative_number() {
        if let Ok(LfscTerm::IntLit(n)) = FormatConverter::term_to_lfsc_term("-42") {
            assert_eq!(n, -42);
        } else {
            panic!("Expected IntLit term");
        }
    }

    #[test]
    fn test_term_to_lfsc_term_rational() {
        if let Ok(LfscTerm::RatLit(num, den)) = FormatConverter::term_to_lfsc_term("3/4") {
            assert_eq!(num, 3);
            assert_eq!(den, 4);
        } else {
            panic!("Expected RatLit term");
        }
    }

    #[test]
    fn test_term_to_lfsc_term_negative_rational() {
        if let Ok(LfscTerm::RatLit(num, den)) = FormatConverter::term_to_lfsc_term("-5/7") {
            assert_eq!(num, -5);
            assert_eq!(den, 7);
        } else {
            panic!("Expected RatLit term");
        }
    }

    #[test]
    fn test_term_to_lfsc_term_complex_nested() {
        // (= (+ x (* 3 y)) z)
        let term = "(= (+ x (* 3 y)) z)";
        let result = FormatConverter::term_to_lfsc_term(term);

        assert!(result.is_ok());
        if let Ok(LfscTerm::App(func, args)) = result {
            assert_eq!(func, "=");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected App term");
        }
    }

    #[test]
    fn test_tokenize_sexpr_simple() {
        let tokens = FormatConverter::tokenize_sexpr("+ x y").unwrap();
        assert_eq!(tokens, vec!["+", "x", "y"]);
    }

    #[test]
    fn test_tokenize_sexpr_nested() {
        let tokens = FormatConverter::tokenize_sexpr("+ (- x y) z").unwrap();
        assert_eq!(tokens, vec!["+", "(- x y)", "z"]);
    }

    #[test]
    fn test_tokenize_sexpr_deeply_nested() {
        let tokens = FormatConverter::tokenize_sexpr("f (g (h x)) y").unwrap();
        assert_eq!(tokens, vec!["f", "(g (h x))", "y"]);
    }

    #[test]
    fn test_tokenize_sexpr_unbalanced() {
        let result = FormatConverter::tokenize_sexpr("+ (x y z");
        assert!(result.is_err());

        if let Err(ConversionError::InvalidSource { reason }) = result {
            assert!(reason.contains("Unbalanced parentheses"));
        } else {
            panic!("Expected InvalidSource error");
        }
    }

    #[test]
    fn test_parse_sort_basic() {
        assert!(matches!(
            FormatConverter::parse_sort("Bool"),
            Ok(LfscSort::Bool)
        ));
        assert!(matches!(
            FormatConverter::parse_sort("Int"),
            Ok(LfscSort::Int)
        ));
        assert!(matches!(
            FormatConverter::parse_sort("Real"),
            Ok(LfscSort::Real)
        ));
    }

    #[test]
    fn test_parse_sort_bitvector() {
        if let Ok(LfscSort::BitVec(width)) = FormatConverter::parse_sort("(_ BitVec 32)") {
            assert_eq!(width, 32);
        } else {
            panic!("Expected BitVec sort");
        }
    }

    #[test]
    fn test_parse_sort_named() {
        if let Ok(LfscSort::Named(name)) = FormatConverter::parse_sort("MySort") {
            assert_eq!(name, "MySort");
        } else {
            panic!("Expected Named sort");
        }
    }

    #[test]
    fn test_information_loss_error_display() {
        let err = ConversionError::InformationLoss {
            reason: "test reason".to_string(),
        };
        assert!(err.to_string().contains("Information loss"));
        assert!(err.to_string().contains("test reason"));
    }

    #[test]
    fn test_invalid_source_error_display() {
        let err = ConversionError::InvalidSource {
            reason: "bad input".to_string(),
        };
        assert!(err.to_string().contains("Invalid source"));
        assert!(err.to_string().contains("bad input"));
    }

    #[test]
    fn test_conversion_failed_error_display() {
        let err = ConversionError::ConversionFailed {
            reason: "conversion issue".to_string(),
        };
        assert!(err.to_string().contains("Conversion failed"));
        assert!(err.to_string().contains("conversion issue"));
    }

    // Property-based tests

    proptest! {
        /// DRAT to Alethe conversion should never lose steps (except deletions in non-strict mode)
        #[test]
        fn prop_drat_to_alethe_preserves_additions(
            clauses in prop::collection::vec(
                prop::collection::vec(-100i32..100i32, 0..5),
                0..20
            )
        ) {
            let converter = FormatConverter::new();
            let mut drat = DratProof::new();

            let mut add_count = 0;
            for clause in clauses {
                drat.add_clause(clause);
                add_count += 1;
            }

            let alethe = converter.drat_to_alethe(&drat).unwrap();

            // Alethe should have the same number of steps as DRAT additions
            prop_assert_eq!(alethe.len(), add_count);
        }

        /// DRAT to Alethe conversion should fail in strict mode with deletions
        #[test]
        fn prop_drat_deletion_strict_mode_fails(
            clauses in prop::collection::vec(-100i32..100i32, 1..5)
        ) {
            let converter = FormatConverter::new().strict_mode(true);
            let mut drat = DratProof::new();

            drat.add_clause(clauses.clone());
            drat.delete_clause(clauses);

            let result = converter.drat_to_alethe(&drat);
            prop_assert!(result.is_err());
        }

        /// Alethe to LFSC conversion should always produce non-empty LFSC (due to signature)
        #[test]
        fn prop_alethe_to_lfsc_nonempty(
            num_assumptions in 0usize..10
        ) {
            let converter = FormatConverter::new();
            let mut alethe = AletheProof::new();

            for i in 0..num_assumptions {
                alethe.assume(format!("p{}", i));
            }

            let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

            // LFSC should never be empty because of the boolean signature
            prop_assert!(!lfsc.is_empty());
        }

        /// Term parsing should be deterministic
        #[test]
        fn prop_term_parsing_deterministic(
            term in "[a-z][a-z0-9]{0,5}"
        ) {
            let result1 = FormatConverter::term_to_lfsc_term(&term);
            let result2 = FormatConverter::term_to_lfsc_term(&term);

            prop_assert_eq!(
                result1.is_ok(),
                result2.is_ok(),
                "Parsing determinism violated"
            );
        }

        /// Sort parsing should accept basic sorts
        #[test]
        fn prop_sort_parsing_basic(
            sort in prop::sample::select(vec!["Bool", "Int", "Real"])
        ) {
            let result = FormatConverter::parse_sort(sort);
            prop_assert!(result.is_ok());
        }

        /// BitVec parsing should handle various widths
        #[test]
        fn prop_bitvec_parsing(
            width in 1u32..256
        ) {
            let sort_str = format!("(_ BitVec {})", width);
            let result = FormatConverter::parse_sort(&sort_str);

            prop_assert!(result.is_ok());
            if let Ok(LfscSort::BitVec(w)) = result {
                prop_assert_eq!(w, width);
            }
        }

        /// Conversion errors should always have non-empty messages
        #[test]
        fn prop_conversion_errors_nonempty(
            reason in "[a-zA-Z ]{1,50}"
        ) {
            let errors = vec![
                ConversionError::InformationLoss { reason: reason.clone() },
                ConversionError::InvalidSource { reason: reason.clone() },
                ConversionError::ConversionFailed { reason },
            ];

            for err in errors {
                let msg = err.to_string();
                prop_assert!(!msg.is_empty());
            }
        }

        /// DRAT clauses with zero literals should be handled correctly
        #[test]
        fn prop_drat_empty_clause_handling(
            prefix_clauses in prop::collection::vec(
                prop::collection::vec(-10i32..10i32, 1..3),
                0..5
            )
        ) {
            let converter = FormatConverter::new();
            let mut drat = DratProof::new();

            // Add prefix clauses
            for clause in prefix_clauses {
                drat.add_clause(clause);
            }

            // Add empty clause (contradiction)
            drat.add_clause(vec![]);

            let alethe = converter.drat_to_alethe(&drat).unwrap();

            // Should successfully convert
            prop_assert!(!alethe.is_empty());
        }

        /// Alethe to LFSC with multiple resolution steps
        #[test]
        fn prop_alethe_resolution_chain(
            chain_length in 1usize..10
        ) {
            let converter = FormatConverter::new();
            let mut alethe = AletheProof::new();

            // Build a chain of resolution steps
            let mut indices = Vec::new();
            for i in 0..chain_length {
                let idx = alethe.assume(format!("p{}", i));
                indices.push(idx);
            }

            // Add resolution step using first two assumptions
            if indices.len() >= 2 {
                alethe.resolution(
                    vec!["q".to_string()],
                    vec![indices[0], indices[1]]
                );
            }

            let lfsc = converter.alethe_to_lfsc(&alethe).unwrap();

            // Should succeed
            prop_assert!(!lfsc.is_empty());
        }

        /// FormatConverter settings should be preserved
        #[test]
        fn prop_converter_settings_preserved(
            strict in prop::bool::ANY,
            comments in prop::bool::ANY
        ) {
            let converter = FormatConverter::new()
                .strict_mode(strict)
                .add_comments(comments);

            prop_assert_eq!(converter.strict_mode, strict);
            prop_assert_eq!(converter.add_comments, comments);
        }
    }
}
