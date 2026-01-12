//! SMT-LIB2 optimization commands.
//!
//! This module provides support for SMT-LIB2 optimization extensions:
//! - `minimize` and `maximize` commands for objectives
//! - `assert-soft` for soft constraints with weights
//! - `get-objectives` for retrieving optimization results
//!
//! Reference: SMT-LIB2 optimization extensions, Z3's `opt/opt_cmds.cpp`

use crate::context::{OptContext, OptError, OptResult};
use crate::maxsat::Weight;
use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_core::ast::TermId;
use std::fmt;

/// SMT-LIB2 optimization command
#[derive(Debug, Clone, PartialEq)]
pub enum OptCommand {
    /// Minimize an objective expression
    Minimize {
        /// The term to minimize
        term: TermId,
        /// Optional identifier for this objective
        id: Option<String>,
    },
    /// Maximize an objective expression
    Maximize {
        /// The term to maximize
        term: TermId,
        /// Optional identifier for this objective
        id: Option<String>,
    },
    /// Assert a soft constraint with optional weight
    AssertSoft {
        /// The soft constraint term
        term: TermId,
        /// Weight (default is 1)
        weight: Weight,
        /// Optional group identifier
        group: Option<String>,
    },
    /// Get the values of all objectives
    GetObjectives,
    /// Check satisfiability and optimize
    CheckSat,
}

/// Response to get-objectives command
#[derive(Debug, Clone)]
pub struct ObjectivesResponse {
    /// Objective values
    pub objectives: Vec<ObjectiveValue>,
}

/// Value of a single objective
#[derive(Debug, Clone)]
pub struct ObjectiveValue {
    /// Objective identifier (if provided)
    pub id: Option<String>,
    /// The objective term
    pub term: TermId,
    /// Whether this is minimization or maximization
    pub is_minimize: bool,
    /// The optimal value found
    pub value: ObjectiveValueKind,
}

/// Kind of objective value
#[derive(Debug, Clone)]
pub enum ObjectiveValueKind {
    /// Integer value
    Int(BigInt),
    /// Rational value
    Rational(BigRational),
    /// Unbounded (infinity)
    Unbounded,
    /// Unknown (optimization not completed)
    Unknown,
}

impl fmt::Display for ObjectiveValueKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveValueKind::Int(n) => write!(f, "{n}"),
            ObjectiveValueKind::Rational(r) => write!(f, "{r}"),
            ObjectiveValueKind::Unbounded => write!(f, "oo"),
            ObjectiveValueKind::Unknown => write!(f, "unknown"),
        }
    }
}

/// SMT-LIB2 optimization command processor
#[derive(Debug)]
pub struct OptCommandProcessor {
    /// The optimization context
    context: OptContext,
    /// Objective identifiers (index -> id)
    objective_ids: Vec<Option<String>>,
}

impl OptCommandProcessor {
    /// Create a new command processor
    pub fn new() -> Self {
        Self {
            context: OptContext::new(),
            objective_ids: Vec::new(),
        }
    }

    /// Process a command
    pub fn process(&mut self, command: OptCommand) -> Result<CommandResponse, OptError> {
        match command {
            OptCommand::Minimize { term, id } => {
                let _obj_id = self.context.minimize(term);
                self.objective_ids.push(id);
                Ok(CommandResponse::Success)
            }
            OptCommand::Maximize { term, id } => {
                let _obj_id = self.context.maximize(term);
                self.objective_ids.push(id);
                Ok(CommandResponse::Success)
            }
            OptCommand::AssertSoft {
                term,
                weight,
                group,
            } => {
                self.context.add_soft_grouped(term, weight, group);
                Ok(CommandResponse::Success)
            }
            OptCommand::GetObjectives => {
                let objectives = self.get_objectives_response();
                Ok(CommandResponse::Objectives(objectives))
            }
            OptCommand::CheckSat => {
                let result = self.context.optimize()?;
                Ok(CommandResponse::CheckSatResult(result))
            }
        }
    }

    /// Get objectives response
    fn get_objectives_response(&self) -> ObjectivesResponse {
        let mut objectives = Vec::new();

        for (idx, id) in self.objective_ids.iter().enumerate() {
            if let Some(obj_value) = self
                .context
                .objective_value(crate::objective::ObjectiveId(idx as u32))
            {
                let value = weight_to_value_kind(obj_value);

                // Get the objective to determine if it's minimize or maximize
                // This is a simplified version - in reality we'd track this properly
                let is_minimize = true; // Default assumption

                objectives.push(ObjectiveValue {
                    id: id.clone(),
                    term: TermId::from(0), // Would need to track this
                    is_minimize,
                    value,
                });
            }
        }

        ObjectivesResponse { objectives }
    }

    /// Get the underlying context
    pub fn context(&self) -> &OptContext {
        &self.context
    }

    /// Get mutable access to the underlying context
    pub fn context_mut(&mut self) -> &mut OptContext {
        &mut self.context
    }

    /// Reset the processor
    pub fn reset(&mut self) {
        self.context.reset();
        self.objective_ids.clear();
    }
}

impl Default for OptCommandProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Response from processing a command
#[derive(Debug, Clone)]
pub enum CommandResponse {
    /// Command succeeded
    Success,
    /// Check-sat result
    CheckSatResult(OptResult),
    /// Objectives response
    Objectives(ObjectivesResponse),
}

/// Convert Weight to ObjectiveValueKind
fn weight_to_value_kind(weight: &Weight) -> ObjectiveValueKind {
    match weight {
        Weight::Int(n) => ObjectiveValueKind::Int(n.clone()),
        Weight::Rational(r) => ObjectiveValueKind::Rational(r.clone()),
        Weight::Infinite => ObjectiveValueKind::Unbounded,
    }
}

/// Parse weight from SMT-LIB2 format
pub fn parse_weight(s: &str) -> Result<Weight, String> {
    if s == "oo" || s == "inf" || s == "infinity" {
        return Ok(Weight::Infinite);
    }

    // Try parsing as integer
    if let Ok(n) = s.parse::<i64>() {
        return Ok(Weight::from(n));
    }

    // Try parsing as rational (format: "num/den")
    if let Some((num_str, den_str)) = s.split_once('/') {
        let num: BigInt = num_str
            .parse()
            .map_err(|_| format!("Invalid numerator: {num_str}"))?;
        let den: BigInt = den_str
            .parse()
            .map_err(|_| format!("Invalid denominator: {den_str}"))?;
        let rational = BigRational::new(num, den);
        return Ok(Weight::Rational(rational));
    }

    Err(format!("Invalid weight format: {s}"))
}

/// Format weight for SMT-LIB2 output
pub fn format_weight(weight: &Weight) -> String {
    match weight {
        Weight::Int(n) => n.to_string(),
        Weight::Rational(r) => {
            if r.is_integer() {
                r.numer().to_string()
            } else {
                format!("{}/{}", r.numer(), r.denom())
            }
        }
        Weight::Infinite => "oo".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_weight_integer() {
        let w = parse_weight("42").unwrap();
        assert_eq!(w, Weight::from(42));
    }

    #[test]
    fn test_parse_weight_negative() {
        let w = parse_weight("-10").unwrap();
        assert_eq!(w, Weight::from(-10));
    }

    #[test]
    fn test_parse_weight_rational() {
        let w = parse_weight("3/4").unwrap();
        match w {
            Weight::Rational(r) => {
                assert_eq!(*r.numer(), BigInt::from(3));
                assert_eq!(*r.denom(), BigInt::from(4));
            }
            _ => panic!("Expected rational weight"),
        }
    }

    #[test]
    fn test_parse_weight_infinite() {
        assert_eq!(parse_weight("oo").unwrap(), Weight::Infinite);
        assert_eq!(parse_weight("inf").unwrap(), Weight::Infinite);
        assert_eq!(parse_weight("infinity").unwrap(), Weight::Infinite);
    }

    #[test]
    fn test_parse_weight_invalid() {
        assert!(parse_weight("abc").is_err());
        assert!(parse_weight("1.5").is_err());
    }

    #[test]
    fn test_format_weight_integer() {
        let w = Weight::from(42);
        assert_eq!(format_weight(&w), "42");
    }

    #[test]
    fn test_format_weight_rational() {
        let w = Weight::Rational(BigRational::new(BigInt::from(3), BigInt::from(4)));
        assert_eq!(format_weight(&w), "3/4");
    }

    #[test]
    fn test_format_weight_infinite() {
        let w = Weight::Infinite;
        assert_eq!(format_weight(&w), "oo");
    }

    #[test]
    fn test_command_processor_minimize() {
        let mut proc = OptCommandProcessor::new();
        let term = TermId::from(1);

        let cmd = OptCommand::Minimize {
            term,
            id: Some("obj1".to_string()),
        };

        let result = proc.process(cmd);
        assert!(matches!(result, Ok(CommandResponse::Success)));
        assert_eq!(proc.context().num_objectives(), 1);
    }

    #[test]
    fn test_command_processor_maximize() {
        let mut proc = OptCommandProcessor::new();
        let term = TermId::from(2);

        let cmd = OptCommand::Maximize { term, id: None };

        let result = proc.process(cmd);
        assert!(matches!(result, Ok(CommandResponse::Success)));
        assert_eq!(proc.context().num_objectives(), 1);
    }

    #[test]
    fn test_command_processor_assert_soft() {
        let mut proc = OptCommandProcessor::new();
        let term = TermId::from(3);

        let cmd = OptCommand::AssertSoft {
            term,
            weight: Weight::from(5),
            group: Some("g1".to_string()),
        };

        let result = proc.process(cmd);
        assert!(matches!(result, Ok(CommandResponse::Success)));
        assert_eq!(proc.context().num_soft(), 1);
    }

    #[test]
    fn test_command_processor_multiple_objectives() {
        let mut proc = OptCommandProcessor::new();

        proc.process(OptCommand::Minimize {
            term: TermId::from(1),
            id: Some("obj1".to_string()),
        })
        .unwrap();

        proc.process(OptCommand::Maximize {
            term: TermId::from(2),
            id: Some("obj2".to_string()),
        })
        .unwrap();

        assert_eq!(proc.context().num_objectives(), 2);
        assert_eq!(proc.objective_ids.len(), 2);
    }

    #[test]
    fn test_command_processor_reset() {
        let mut proc = OptCommandProcessor::new();

        proc.process(OptCommand::Minimize {
            term: TermId::from(1),
            id: None,
        })
        .unwrap();

        proc.process(OptCommand::AssertSoft {
            term: TermId::from(2),
            weight: Weight::one(),
            group: None,
        })
        .unwrap();

        assert_eq!(proc.context().num_objectives(), 1);
        assert_eq!(proc.context().num_soft(), 1);

        proc.reset();

        assert_eq!(proc.context().num_objectives(), 0);
        assert_eq!(proc.context().num_soft(), 0);
        assert_eq!(proc.objective_ids.len(), 0);
    }
}
