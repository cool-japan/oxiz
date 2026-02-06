//! TPTP (Thousands of Problems for Theorem Provers) format parser and converter
//!
//! TPTP is a standard format for representing first-order logic problems.
//! This module supports the FOF (First-Order Formula) sublanguage.
//!
//! Format specification:
//! - fof declarations: `fof(name, role, formula).`
//! - Roles: axiom, hypothesis, conjecture, negated_conjecture
//! - Formulas: & (and), | (or), ~ (not), => (implies), <=> (iff), ! (forall), ? (exists)
//! - Terms: constants (lowercase), variables (uppercase), functions

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::BufRead;

/// TPTP formula role
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TptpRole {
    /// Axiom - assumed to be true
    Axiom,
    /// Hypothesis - assumed for the problem
    Hypothesis,
    /// Conjecture - to be proven
    Conjecture,
    /// Negated conjecture - negation of conjecture (for refutation)
    NegatedConjecture,
    /// Lemma
    Lemma,
    /// Definition
    Definition,
    /// Type declaration
    Type,
    /// Unknown/other role
    Unknown,
}

impl TptpRole {
    /// Parse role from string
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "axiom" => TptpRole::Axiom,
            "hypothesis" => TptpRole::Hypothesis,
            "conjecture" => TptpRole::Conjecture,
            "negated_conjecture" => TptpRole::NegatedConjecture,
            "lemma" => TptpRole::Lemma,
            "definition" => TptpRole::Definition,
            "type" => TptpRole::Type,
            _ => TptpRole::Unknown,
        }
    }
}

impl fmt::Display for TptpRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TptpRole::Axiom => write!(f, "axiom"),
            TptpRole::Hypothesis => write!(f, "hypothesis"),
            TptpRole::Conjecture => write!(f, "conjecture"),
            TptpRole::NegatedConjecture => write!(f, "negated_conjecture"),
            TptpRole::Lemma => write!(f, "lemma"),
            TptpRole::Definition => write!(f, "definition"),
            TptpRole::Type => write!(f, "type"),
            TptpRole::Unknown => write!(f, "unknown"),
        }
    }
}

/// TPTP term (constant, variable, or function application)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TptpTerm {
    /// Variable (uppercase identifier)
    Variable(String),
    /// Constant (lowercase identifier)
    Constant(String),
    /// Function application
    Function(String, Vec<TptpTerm>),
}

#[allow(dead_code)]
impl TptpTerm {
    /// Check if this term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, TptpTerm::Variable(_))
    }

    /// Get all variables in this term
    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match self {
            TptpTerm::Variable(name) => {
                vars.insert(name.clone());
            }
            TptpTerm::Constant(_) => {}
            TptpTerm::Function(_, args) => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
        }
    }

    /// Convert to SMT-LIB2 term
    fn to_smtlib2(&self) -> String {
        match self {
            TptpTerm::Variable(name) => name.clone(),
            TptpTerm::Constant(name) => name.clone(),
            TptpTerm::Function(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| a.to_smtlib2()).collect();
                    format!("({} {})", name, args_str.join(" "))
                }
            }
        }
    }
}

/// TPTP formula
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TptpFormula {
    /// Atomic formula (predicate application)
    Atom(String, Vec<TptpTerm>),
    /// Equality
    Equality(TptpTerm, TptpTerm),
    /// Inequality
    Inequality(TptpTerm, TptpTerm),
    /// Negation
    Not(Box<TptpFormula>),
    /// Conjunction
    And(Vec<TptpFormula>),
    /// Disjunction
    Or(Vec<TptpFormula>),
    /// Implication
    Implies(Box<TptpFormula>, Box<TptpFormula>),
    /// Equivalence (iff)
    Iff(Box<TptpFormula>, Box<TptpFormula>),
    /// Universal quantification
    Forall(Vec<String>, Box<TptpFormula>),
    /// Existential quantification
    Exists(Vec<String>, Box<TptpFormula>),
    /// True constant
    True,
    /// False constant
    False,
}

impl TptpFormula {
    /// Get all free variables in this formula
    fn free_variables(&self) -> HashSet<String> {
        let mut free = HashSet::new();
        self.collect_free_variables(&mut free, &HashSet::new());
        free
    }

    fn collect_free_variables(&self, free: &mut HashSet<String>, bound: &HashSet<String>) {
        match self {
            TptpFormula::Atom(_, args) => {
                for arg in args {
                    let mut term_vars = HashSet::new();
                    arg.collect_variables(&mut term_vars);
                    for v in term_vars {
                        if !bound.contains(&v) {
                            free.insert(v);
                        }
                    }
                }
            }
            TptpFormula::Equality(t1, t2) | TptpFormula::Inequality(t1, t2) => {
                let mut term_vars = HashSet::new();
                t1.collect_variables(&mut term_vars);
                t2.collect_variables(&mut term_vars);
                for v in term_vars {
                    if !bound.contains(&v) {
                        free.insert(v);
                    }
                }
            }
            TptpFormula::Not(f) => f.collect_free_variables(free, bound),
            TptpFormula::And(fs) | TptpFormula::Or(fs) => {
                for f in fs {
                    f.collect_free_variables(free, bound);
                }
            }
            TptpFormula::Implies(f1, f2) | TptpFormula::Iff(f1, f2) => {
                f1.collect_free_variables(free, bound);
                f2.collect_free_variables(free, bound);
            }
            TptpFormula::Forall(vars, f) | TptpFormula::Exists(vars, f) => {
                let mut new_bound = bound.clone();
                for v in vars {
                    new_bound.insert(v.clone());
                }
                f.collect_free_variables(free, &new_bound);
            }
            TptpFormula::True | TptpFormula::False => {}
        }
    }

    /// Get all predicates used in this formula with their arities
    fn collect_predicates(&self, predicates: &mut HashMap<String, usize>) {
        match self {
            TptpFormula::Atom(name, args) => {
                predicates.insert(name.clone(), args.len());
            }
            TptpFormula::Equality(_, _) | TptpFormula::Inequality(_, _) => {}
            TptpFormula::Not(f) => f.collect_predicates(predicates),
            TptpFormula::And(fs) | TptpFormula::Or(fs) => {
                for f in fs {
                    f.collect_predicates(predicates);
                }
            }
            TptpFormula::Implies(f1, f2) | TptpFormula::Iff(f1, f2) => {
                f1.collect_predicates(predicates);
                f2.collect_predicates(predicates);
            }
            TptpFormula::Forall(_, f) | TptpFormula::Exists(_, f) => {
                f.collect_predicates(predicates);
            }
            TptpFormula::True | TptpFormula::False => {}
        }
    }

    /// Get all functions used in this formula with their arities
    fn collect_functions(&self, functions: &mut HashMap<String, usize>) {
        match self {
            TptpFormula::Atom(_, args) => {
                for arg in args {
                    Self::collect_functions_from_term(arg, functions);
                }
            }
            TptpFormula::Equality(t1, t2) | TptpFormula::Inequality(t1, t2) => {
                Self::collect_functions_from_term(t1, functions);
                Self::collect_functions_from_term(t2, functions);
            }
            TptpFormula::Not(f) => f.collect_functions(functions),
            TptpFormula::And(fs) | TptpFormula::Or(fs) => {
                for f in fs {
                    f.collect_functions(functions);
                }
            }
            TptpFormula::Implies(f1, f2) | TptpFormula::Iff(f1, f2) => {
                f1.collect_functions(functions);
                f2.collect_functions(functions);
            }
            TptpFormula::Forall(_, f) | TptpFormula::Exists(_, f) => {
                f.collect_functions(functions);
            }
            TptpFormula::True | TptpFormula::False => {}
        }
    }

    fn collect_functions_from_term(term: &TptpTerm, functions: &mut HashMap<String, usize>) {
        match term {
            TptpTerm::Variable(_) => {}
            TptpTerm::Constant(name) => {
                functions.entry(name.clone()).or_insert(0);
            }
            TptpTerm::Function(name, args) => {
                functions.insert(name.clone(), args.len());
                for arg in args {
                    Self::collect_functions_from_term(arg, functions);
                }
            }
        }
    }

    /// Convert to SMT-LIB2 formula string
    fn to_smtlib2(&self) -> String {
        match self {
            TptpFormula::Atom(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| a.to_smtlib2()).collect();
                    format!("({} {})", name, args_str.join(" "))
                }
            }
            TptpFormula::Equality(t1, t2) => {
                format!("(= {} {})", t1.to_smtlib2(), t2.to_smtlib2())
            }
            TptpFormula::Inequality(t1, t2) => {
                format!("(not (= {} {}))", t1.to_smtlib2(), t2.to_smtlib2())
            }
            TptpFormula::Not(f) => format!("(not {})", f.to_smtlib2()),
            TptpFormula::And(fs) => {
                if fs.is_empty() {
                    "true".to_string()
                } else if fs.len() == 1 {
                    fs[0].to_smtlib2()
                } else {
                    let fs_str: Vec<String> = fs.iter().map(|f| f.to_smtlib2()).collect();
                    format!("(and {})", fs_str.join(" "))
                }
            }
            TptpFormula::Or(fs) => {
                if fs.is_empty() {
                    "false".to_string()
                } else if fs.len() == 1 {
                    fs[0].to_smtlib2()
                } else {
                    let fs_str: Vec<String> = fs.iter().map(|f| f.to_smtlib2()).collect();
                    format!("(or {})", fs_str.join(" "))
                }
            }
            TptpFormula::Implies(f1, f2) => {
                format!("(=> {} {})", f1.to_smtlib2(), f2.to_smtlib2())
            }
            TptpFormula::Iff(f1, f2) => {
                format!("(= {} {})", f1.to_smtlib2(), f2.to_smtlib2())
            }
            TptpFormula::Forall(vars, f) => {
                let bindings: Vec<String> = vars.iter().map(|v| format!("({} U)", v)).collect();
                format!("(forall ({}) {})", bindings.join(" "), f.to_smtlib2())
            }
            TptpFormula::Exists(vars, f) => {
                let bindings: Vec<String> = vars.iter().map(|v| format!("({} U)", v)).collect();
                format!("(exists ({}) {})", bindings.join(" "), f.to_smtlib2())
            }
            TptpFormula::True => "true".to_string(),
            TptpFormula::False => "false".to_string(),
        }
    }
}

/// A single TPTP statement (fof declaration)
#[derive(Debug, Clone)]
pub struct TptpStatement {
    /// Name of the formula
    pub name: String,
    /// Role of the formula
    pub role: TptpRole,
    /// The formula itself
    pub formula: TptpFormula,
}

/// TPTP problem (collection of statements)
#[derive(Debug, Clone)]
pub struct TptpProblem {
    /// All statements in the problem
    pub statements: Vec<TptpStatement>,
    /// Comments from the file
    pub comments: Vec<String>,
}

/// TPTP parser
pub struct TptpParser {
    input: Vec<char>,
    pos: usize,
}

impl TptpParser {
    /// Create a new parser for the given input
    pub fn new(input: &str) -> Self {
        TptpParser {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    /// Parse a TPTP problem from a reader
    pub fn parse_reader<R: BufRead>(reader: R) -> Result<TptpProblem, String> {
        let mut input = String::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            input.push_str(&line);
            input.push('\n');
        }
        let mut parser = TptpParser::new(&input);
        parser.parse_problem()
    }

    /// Parse the entire TPTP problem
    pub fn parse_problem(&mut self) -> Result<TptpProblem, String> {
        let mut statements = Vec::new();
        let mut comments = Vec::new();

        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.pos >= self.input.len() {
                break;
            }

            // Check for comment
            if self.peek() == Some('%') {
                let comment = self.parse_line_comment();
                comments.push(comment);
                continue;
            }

            // Check for block comment
            if self.peek() == Some('/') && self.peek_ahead(1) == Some('*') {
                let comment = self.parse_block_comment()?;
                comments.push(comment);
                continue;
            }

            // Try to parse an fof statement
            if self.try_consume("fof") || self.try_consume("cnf") {
                let stmt = self.parse_statement()?;
                statements.push(stmt);
            } else if self.try_consume("include") {
                // Skip include statements for now
                self.skip_until('.');
                self.consume_char('.')?;
            } else if self.pos < self.input.len() {
                // Unknown content, try to skip
                self.pos += 1;
            }
        }

        Ok(TptpProblem {
            statements,
            comments,
        })
    }

    /// Parse a single fof statement
    fn parse_statement(&mut self) -> Result<TptpStatement, String> {
        self.skip_whitespace();
        self.consume_char('(')?;
        self.skip_whitespace();

        // Parse name
        let name = self.parse_identifier()?;
        self.skip_whitespace();
        self.consume_char(',')?;
        self.skip_whitespace();

        // Parse role
        let role_str = self.parse_identifier()?;
        let role = TptpRole::from_str(&role_str);
        self.skip_whitespace();
        self.consume_char(',')?;
        self.skip_whitespace();

        // Parse formula
        let formula = self.parse_formula()?;
        self.skip_whitespace();

        // Optional annotations
        if self.peek() == Some(',') {
            self.consume_char(',')?;
            self.skip_annotations()?;
        }

        self.skip_whitespace();
        self.consume_char(')')?;
        self.skip_whitespace();
        self.consume_char('.')?;

        Ok(TptpStatement {
            name,
            role,
            formula,
        })
    }

    /// Parse a formula
    fn parse_formula(&mut self) -> Result<TptpFormula, String> {
        self.parse_iff()
    }

    /// Parse iff (<=>)
    fn parse_iff(&mut self) -> Result<TptpFormula, String> {
        let mut left = self.parse_implies()?;

        loop {
            self.skip_whitespace();
            if self.try_consume("<=>") {
                self.skip_whitespace();
                let right = self.parse_implies()?;
                left = TptpFormula::Iff(Box::new(left), Box::new(right));
            } else if self.try_consume("<~>") {
                // XOR (not iff)
                self.skip_whitespace();
                let right = self.parse_implies()?;
                left =
                    TptpFormula::Not(Box::new(TptpFormula::Iff(Box::new(left), Box::new(right))));
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse implies (=>)
    fn parse_implies(&mut self) -> Result<TptpFormula, String> {
        let left = self.parse_or()?;

        self.skip_whitespace();
        if self.try_consume("=>") {
            self.skip_whitespace();
            let right = self.parse_implies()?;
            Ok(TptpFormula::Implies(Box::new(left), Box::new(right)))
        } else if self.peek() == Some('<') {
            // Check for reverse implies (<=) but not <=> or <~>
            let next_chars: String = self.input[self.pos..].iter().take(3).collect();
            if next_chars.starts_with("<=")
                && !next_chars.starts_with("<=>")
                && !next_chars.starts_with("<~>")
            {
                self.try_consume("<=");
                self.skip_whitespace();
                let right = self.parse_implies()?;
                Ok(TptpFormula::Implies(Box::new(right), Box::new(left)))
            } else {
                Ok(left)
            }
        } else {
            Ok(left)
        }
    }

    /// Parse or (|)
    fn parse_or(&mut self) -> Result<TptpFormula, String> {
        let mut operands = vec![self.parse_and()?];

        loop {
            self.skip_whitespace();
            if self.try_consume("|") {
                self.skip_whitespace();
                operands.push(self.parse_and()?);
            } else {
                break;
            }
        }

        if operands.len() == 1 {
            // Safety: len() == 1 ensures pop() succeeds, use into_iter for no-unwrap policy
            Ok(operands.into_iter().next().unwrap_or(TptpFormula::True))
        } else {
            Ok(TptpFormula::Or(operands))
        }
    }

    /// Parse and (&)
    fn parse_and(&mut self) -> Result<TptpFormula, String> {
        let mut operands = vec![self.parse_unary()?];

        loop {
            self.skip_whitespace();
            if self.try_consume("&") {
                self.skip_whitespace();
                operands.push(self.parse_unary()?);
            } else {
                break;
            }
        }

        if operands.len() == 1 {
            // Safety: len() == 1 ensures pop() succeeds, use into_iter for no-unwrap policy
            Ok(operands.into_iter().next().unwrap_or(TptpFormula::True))
        } else {
            Ok(TptpFormula::And(operands))
        }
    }

    /// Parse unary operators (~ for not, ! for forall, ? for exists)
    fn parse_unary(&mut self) -> Result<TptpFormula, String> {
        self.skip_whitespace();

        // Negation
        if self.try_consume("~") {
            self.skip_whitespace();
            let inner = self.parse_unary()?;
            return Ok(TptpFormula::Not(Box::new(inner)));
        }

        // Universal quantifier
        if self.try_consume("!") {
            return self.parse_quantifier(true);
        }

        // Existential quantifier
        if self.try_consume("?") {
            return self.parse_quantifier(false);
        }

        self.parse_atomic()
    }

    /// Parse a quantified formula
    fn parse_quantifier(&mut self, is_universal: bool) -> Result<TptpFormula, String> {
        self.skip_whitespace();
        self.consume_char('[')?;
        self.skip_whitespace();

        let mut vars = Vec::new();
        loop {
            let var = self.parse_variable()?;
            vars.push(var);
            self.skip_whitespace();

            // Optional type annotation - only if next char is ':' and not followed by ']'
            // This distinguishes between `[X:Type]` (typed) and `[X]` (untyped)
            if self.peek() == Some(':') && self.pos + 1 < self.input.len() {
                // Look ahead to check if this is a type annotation or body separator
                // Type annotation: [X:Type] or [X : Type]
                // Body separator: [X]: formula
                let next_non_ws = self.input[self.pos + 1..]
                    .iter()
                    .find(|&&c| !c.is_whitespace());
                if next_non_ws != Some(&']') && next_non_ws != Some(&'(') {
                    self.try_consume(":");
                    self.skip_whitespace();
                    let _type = self.parse_identifier()?;
                    self.skip_whitespace();
                }
            }

            if self.try_consume(",") {
                self.skip_whitespace();
            } else {
                break;
            }
        }

        self.consume_char(']')?;
        self.skip_whitespace();
        // Body separator - may or may not have ':'
        self.try_consume(":");
        self.skip_whitespace();

        let body = self.parse_unary()?;

        if is_universal {
            Ok(TptpFormula::Forall(vars, Box::new(body)))
        } else {
            Ok(TptpFormula::Exists(vars, Box::new(body)))
        }
    }

    /// Parse atomic formula
    fn parse_atomic(&mut self) -> Result<TptpFormula, String> {
        self.skip_whitespace();

        // Check for true/false
        if self.try_consume("$true") {
            return Ok(TptpFormula::True);
        }
        if self.try_consume("$false") {
            return Ok(TptpFormula::False);
        }

        // Check for parenthesized formula
        if self.peek() == Some('(') {
            self.consume_char('(')?;
            let formula = self.parse_formula()?;
            self.skip_whitespace();
            self.consume_char(')')?;
            return Ok(formula);
        }

        // Parse term or predicate
        let first_term = self.parse_term()?;

        self.skip_whitespace();

        // Check for equality/inequality
        // Note: Must check != before =, and must not consume = if it's part of => or <=>
        if self.try_consume("!=") {
            self.skip_whitespace();
            let second_term = self.parse_term()?;
            return Ok(TptpFormula::Inequality(first_term, second_term));
        }

        // Check for = but not => or <=>
        if self.peek() == Some('=') {
            // Look ahead to make sure it's not => or part of <=>
            let next_char = if self.pos + 1 < self.input.len() {
                Some(self.input[self.pos + 1])
            } else {
                None
            };
            if next_char != Some('>') {
                self.try_consume("=");
                self.skip_whitespace();
                let second_term = self.parse_term()?;
                return Ok(TptpFormula::Equality(first_term, second_term));
            }
        }

        // Convert term to atom
        match first_term {
            TptpTerm::Function(name, args) => Ok(TptpFormula::Atom(name, args)),
            TptpTerm::Constant(name) => Ok(TptpFormula::Atom(name, vec![])),
            TptpTerm::Variable(name) => Ok(TptpFormula::Atom(name, vec![])),
        }
    }

    /// Parse a term
    fn parse_term(&mut self) -> Result<TptpTerm, String> {
        self.skip_whitespace();

        let name = self.parse_identifier()?;

        self.skip_whitespace();

        // Check for function application
        if self.peek() == Some('(') {
            self.consume_char('(')?;
            self.skip_whitespace();

            let mut args = Vec::new();
            if self.peek() != Some(')') {
                args.push(self.parse_term()?);
                self.skip_whitespace();

                while self.try_consume(",") {
                    self.skip_whitespace();
                    args.push(self.parse_term()?);
                    self.skip_whitespace();
                }
            }

            self.consume_char(')')?;

            Ok(TptpTerm::Function(name, args))
        } else if name.chars().next().is_some_and(|c| c.is_uppercase()) {
            // Variable (uppercase)
            Ok(TptpTerm::Variable(name))
        } else {
            // Constant (lowercase)
            Ok(TptpTerm::Constant(name))
        }
    }

    /// Parse a variable (must start with uppercase)
    fn parse_variable(&mut self) -> Result<String, String> {
        let name = self.parse_identifier()?;
        if name.chars().next().is_some_and(|c| c.is_uppercase()) {
            Ok(name)
        } else {
            Err(format!("Expected variable (uppercase), found '{}'", name))
        }
    }

    /// Parse an identifier
    fn parse_identifier(&mut self) -> Result<String, String> {
        self.skip_whitespace();

        let mut name = String::new();

        // Handle quoted identifiers
        if self.peek() == Some('\'') {
            self.consume_char('\'')?;
            while let Some(c) = self.peek() {
                if c == '\'' {
                    self.consume_char('\'')?;
                    break;
                }
                name.push(c);
                self.pos += 1;
            }
            return Ok(name);
        }

        // Handle regular identifiers
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '$' {
                name.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }

        if name.is_empty() {
            Err(format!(
                "Expected identifier at position {}, found {:?}",
                self.pos,
                self.peek()
            ))
        } else {
            Ok(name)
        }
    }

    /// Parse a line comment (starting with %)
    fn parse_line_comment(&mut self) -> String {
        let mut comment = String::new();
        self.pos += 1; // Skip %

        while let Some(c) = self.peek() {
            if c == '\n' {
                self.pos += 1;
                break;
            }
            comment.push(c);
            self.pos += 1;
        }

        comment.trim().to_string()
    }

    /// Parse a block comment (/* ... */)
    fn parse_block_comment(&mut self) -> Result<String, String> {
        let mut comment = String::new();
        self.pos += 2; // Skip /*

        while self.pos + 1 < self.input.len() {
            if self.peek() == Some('*') && self.peek_ahead(1) == Some('/') {
                self.pos += 2;
                break;
            }
            if let Some(c) = self.peek() {
                comment.push(c);
            }
            self.pos += 1;
        }

        Ok(comment.trim().to_string())
    }

    /// Skip annotations (after formula in parentheses)
    fn skip_annotations(&mut self) -> Result<(), String> {
        let mut depth = 0;
        while let Some(c) = self.peek() {
            match c {
                '(' | '[' => {
                    depth += 1;
                    self.pos += 1;
                }
                ')' | ']' => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.pos += 1;
                }
                _ => self.pos += 1,
            }
        }
        Ok(())
    }

    /// Skip until a specific character
    fn skip_until(&mut self, target: char) {
        while let Some(c) = self.peek() {
            if c == target {
                break;
            }
            self.pos += 1;
        }
    }

    /// Skip whitespace
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Peek at current character
    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    /// Peek ahead by n characters
    fn peek_ahead(&self, n: usize) -> Option<char> {
        self.input.get(self.pos + n).copied()
    }

    /// Try to consume a string
    fn try_consume(&mut self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        if self.pos + chars.len() <= self.input.len() {
            for (i, c) in chars.iter().enumerate() {
                if self.input[self.pos + i] != *c {
                    return false;
                }
            }
            self.pos += chars.len();
            true
        } else {
            false
        }
    }

    /// Consume a specific character
    fn consume_char(&mut self, expected: char) -> Result<(), String> {
        if self.peek() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!(
                "Expected '{}' at position {}, found {:?}",
                expected,
                self.pos,
                self.peek()
            ))
        }
    }
}

#[allow(dead_code)]
impl TptpProblem {
    /// Parse a TPTP problem from a string
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut parser = TptpParser::new(input);
        parser.parse_problem()
    }

    /// Convert the TPTP problem to SMT-LIB2 format
    pub fn to_smtlib2(&self) -> String {
        let mut output = String::new();

        // Collect all predicates and functions
        let mut predicates: HashMap<String, usize> = HashMap::new();
        let mut functions: HashMap<String, usize> = HashMap::new();
        let mut free_vars: HashSet<String> = HashSet::new();

        for stmt in &self.statements {
            stmt.formula.collect_predicates(&mut predicates);
            stmt.formula.collect_functions(&mut functions);
            free_vars.extend(stmt.formula.free_variables());
        }

        // Set logic (use UF for uninterpreted functions)
        output.push_str("(set-logic UF)\n\n");

        // Add comments
        for comment in &self.comments {
            output.push_str(&format!("; {}\n", comment));
        }
        if !self.comments.is_empty() {
            output.push('\n');
        }

        // Declare the universal sort U
        output.push_str("(declare-sort U 0)\n\n");

        // Declare all functions (constants have arity 0)
        for (name, arity) in &functions {
            if *arity == 0 {
                output.push_str(&format!("(declare-const {} U)\n", name));
            } else {
                let args: Vec<&str> = (0..*arity).map(|_| "U").collect();
                output.push_str(&format!("(declare-fun {} ({}) U)\n", name, args.join(" ")));
            }
        }

        // Declare free variables as constants
        for var in &free_vars {
            output.push_str(&format!("(declare-const {} U)\n", var));
        }

        if !functions.is_empty() || !free_vars.is_empty() {
            output.push('\n');
        }

        // Declare all predicates
        for (name, arity) in &predicates {
            if *arity == 0 {
                output.push_str(&format!("(declare-const {} Bool)\n", name));
            } else {
                let args: Vec<&str> = (0..*arity).map(|_| "U").collect();
                output.push_str(&format!(
                    "(declare-fun {} ({}) Bool)\n",
                    name,
                    args.join(" ")
                ));
            }
        }

        if !predicates.is_empty() {
            output.push('\n');
        }

        // Add assertions
        let mut has_conjecture = false;

        for stmt in &self.statements {
            let formula_str = stmt.formula.to_smtlib2();

            match stmt.role {
                TptpRole::Conjecture => {
                    // For refutation-based proving, we negate the conjecture
                    output.push_str(&format!(
                        "; {} ({})\n(assert (not {}))\n\n",
                        stmt.name, stmt.role, formula_str
                    ));
                    has_conjecture = true;
                }
                TptpRole::NegatedConjecture => {
                    // Already negated
                    output.push_str(&format!(
                        "; {} ({})\n(assert {})\n\n",
                        stmt.name, stmt.role, formula_str
                    ));
                    has_conjecture = true;
                }
                _ => {
                    output.push_str(&format!(
                        "; {} ({})\n(assert {})\n\n",
                        stmt.name, stmt.role, formula_str
                    ));
                }
            }
        }

        // Add check-sat
        output.push_str("(check-sat)\n");

        // Add comment about interpretation
        if has_conjecture {
            output.push_str("; If unsat, the conjecture is a theorem\n");
            output.push_str("; If sat, the conjecture has a counter-example\n");
        }

        output
    }

    /// Check if the problem has a conjecture
    pub fn has_conjecture(&self) -> bool {
        self.statements
            .iter()
            .any(|s| matches!(s.role, TptpRole::Conjecture | TptpRole::NegatedConjecture))
    }
}

/// SZS status codes for TPTP output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SzsStatus {
    /// The conjecture is a theorem (proven)
    Theorem,
    /// The conjecture is not a theorem (counter-satisfiable)
    CounterSatisfiable,
    /// The problem is satisfiable (no conjecture)
    Satisfiable,
    /// The problem is unsatisfiable (no conjecture)
    Unsatisfiable,
    /// Unknown result
    Unknown,
    /// Timeout
    Timeout,
    /// Error
    Error,
    /// Resource out (memory, etc.)
    ResourceOut,
    /// Given up
    GaveUp,
}

impl fmt::Display for SzsStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SzsStatus::Theorem => write!(f, "Theorem"),
            SzsStatus::CounterSatisfiable => write!(f, "CounterSatisfiable"),
            SzsStatus::Satisfiable => write!(f, "Satisfiable"),
            SzsStatus::Unsatisfiable => write!(f, "Unsatisfiable"),
            SzsStatus::Unknown => write!(f, "Unknown"),
            SzsStatus::Timeout => write!(f, "Timeout"),
            SzsStatus::Error => write!(f, "Error"),
            SzsStatus::ResourceOut => write!(f, "ResourceOut"),
            SzsStatus::GaveUp => write!(f, "GaveUp"),
        }
    }
}

/// Format SMT-LIB2 result as TPTP SZS status
pub fn format_tptp_result(smtlib_result: &str, has_conjecture: bool) -> String {
    let status = if smtlib_result.contains("unsat") {
        if has_conjecture {
            SzsStatus::Theorem
        } else {
            SzsStatus::Unsatisfiable
        }
    } else if smtlib_result.contains("sat") && !smtlib_result.contains("unsat") {
        if has_conjecture {
            SzsStatus::CounterSatisfiable
        } else {
            SzsStatus::Satisfiable
        }
    } else if smtlib_result.contains("timeout") {
        SzsStatus::Timeout
    } else if smtlib_result.contains("error") {
        SzsStatus::Error
    } else {
        SzsStatus::Unknown
    };

    format!("% SZS status {}", status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_fof() {
        let input = r#"
            fof(ax1, axiom, ![X]: (human(X) => mortal(X))).
            fof(ax2, axiom, human(socrates)).
            fof(conj, conjecture, mortal(socrates)).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 3);
        assert_eq!(problem.statements[0].role, TptpRole::Axiom);
        assert_eq!(problem.statements[1].role, TptpRole::Axiom);
        assert_eq!(problem.statements[2].role, TptpRole::Conjecture);
    }

    #[test]
    fn test_parse_complex_formula() {
        let input = r#"
            fof(test, axiom, ![X,Y]: ((p(X) & q(Y)) => r(X,Y))).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 1);
    }

    #[test]
    fn test_parse_equality() {
        let input = r#"
            fof(eq_test, axiom, ![X]: (X = X)).
            fof(neq_test, axiom, a != b).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 2);
    }

    #[test]
    fn test_to_smtlib2() {
        let input = r#"
            fof(ax1, axiom, ![X]: (human(X) => mortal(X))).
            fof(ax2, axiom, human(socrates)).
            fof(conj, conjecture, mortal(socrates)).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        let smtlib = problem.to_smtlib2();

        assert!(smtlib.contains("(set-logic UF)"));
        assert!(smtlib.contains("(declare-sort U 0)"));
        assert!(smtlib.contains("(declare-fun human (U) Bool)"));
        assert!(smtlib.contains("(declare-fun mortal (U) Bool)"));
        assert!(smtlib.contains("(declare-const socrates U)"));
        assert!(smtlib.contains("(check-sat)"));
        // Conjecture should be negated
        assert!(smtlib.contains("(assert (not"));
    }

    #[test]
    fn test_szs_status_theorem() {
        let result = format_tptp_result("unsat", true);
        assert_eq!(result, "% SZS status Theorem");
    }

    #[test]
    fn test_szs_status_counter_satisfiable() {
        let result = format_tptp_result("sat", true);
        assert_eq!(result, "% SZS status CounterSatisfiable");
    }

    #[test]
    fn test_szs_status_satisfiable() {
        let result = format_tptp_result("sat", false);
        assert_eq!(result, "% SZS status Satisfiable");
    }

    #[test]
    fn test_szs_status_unsatisfiable() {
        let result = format_tptp_result("unsat", false);
        assert_eq!(result, "% SZS status Unsatisfiable");
    }

    #[test]
    fn test_parse_comments() {
        let input = r#"
            % This is a comment
            fof(ax1, axiom, p).
            /* Block comment */
            fof(ax2, axiom, q).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 2);
        assert!(!problem.comments.is_empty());
    }

    #[test]
    fn test_parse_existential() {
        let input = r#"
            fof(ex_test, axiom, ?[X]: p(X)).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 1);

        let smtlib = problem.to_smtlib2();
        assert!(smtlib.contains("exists"));
    }

    #[test]
    fn test_parse_iff() {
        let input = r#"
            fof(iff_test, axiom, p <=> q).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 1);
    }

    #[test]
    fn test_parse_function_terms() {
        let input = r#"
            fof(func_test, axiom, p(f(a, g(b)))).
        "#;

        let problem = TptpProblem::parse(input).unwrap();
        assert_eq!(problem.statements.len(), 1);

        let smtlib = problem.to_smtlib2();
        assert!(smtlib.contains("(declare-fun f (U U) U)"));
        assert!(smtlib.contains("(declare-fun g (U) U)"));
    }
}
