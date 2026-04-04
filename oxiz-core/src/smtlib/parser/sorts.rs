//! Sort parsing for the SMT-LIB2 parser

use super::super::lexer::TokenKind;
use super::Parser;
use crate::ast::TermId;
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;

impl<'a> Parser<'a> {
    /// Expect a closing parenthesis ')'
    pub(super) fn expect_rparen(&mut self) -> Result<()> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected ')', found end of input".to_string(),
            })?;

        if !matches!(token.kind, TokenKind::RParen) {
            return Err(OxizError::ParseError {
                position: token.start,
                message: format!("expected ')', found {:?}", token.kind),
            });
        }
        Ok(())
    }

    /// Parse a let binding: (let ((name term) ...) body)
    /// Called after consuming the "let" symbol
    pub(super) fn parse_let(&mut self) -> Result<TermId> {
        // Parse bindings: ((name term) ...)
        self.expect_lparen()?;

        let mut new_bindings: Vec<(String, TermId)> = Vec::new();

        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }

            self.expect_lparen()?;
            let name = self.expect_symbol()?;
            let term = self.parse_term()?;
            self.expect_rparen()?;
            new_bindings.push((name, term));
        }

        // Add bindings to scope
        let old_bindings: Vec<_> = new_bindings
            .iter()
            .filter_map(|(name, _)| self.bindings.get(name).map(|&t| (name.clone(), t)))
            .collect();

        for (name, term) in &new_bindings {
            self.bindings.insert(name.clone(), *term);
        }

        // Parse body
        let body = self.parse_term()?;
        self.expect_rparen()?;

        // Restore old bindings
        for (name, _) in &new_bindings {
            self.bindings.remove(name);
        }
        for (name, term) in old_bindings {
            self.bindings.insert(name, term);
        }

        // Create let term
        let bindings: Vec<_> = new_bindings.iter().map(|(n, t)| (n.as_str(), *t)).collect();
        Ok(self.manager.mk_let(bindings, body))
    }

    /// Parse a forall binder: (forall ((name sort) ...) body)
    /// Called after consuming the "forall" symbol
    pub(super) fn parse_forall(&mut self) -> Result<TermId> {
        // Parse sorted vars: ((name sort) ...)
        self.expect_lparen()?;
        let vars = self.parse_sorted_vars()?;

        // Bind quantifier variables so body references resolve correctly.
        // This ensures that a bound variable like `i` is looked up from
        // bindings (with the declared sort) rather than falling through to
        // the default `mk_var(name, bool_sort)` path.
        let old_bindings: Vec<_> = vars
            .iter()
            .filter_map(|(name, _)| self.bindings.get(name).map(|&t| (name.clone(), t)))
            .collect();
        let old_constants: Vec<_> = vars
            .iter()
            .filter_map(|(name, _)| self.constants.get(name).map(|&s| (name.clone(), s)))
            .collect();
        for (name, sort) in &vars {
            let var_term = self.manager.mk_var(name, *sort);
            self.bindings.insert(name.clone(), var_term);
            // Remove from constants to avoid shadowing issues
            self.constants.remove(name);
        }

        // Parse body (bound variables now resolve with the correct sort)
        let body = self.parse_term()?;
        self.expect_rparen()?;

        // Restore old bindings and constants
        for (name, _) in &vars {
            self.bindings.remove(name);
        }
        for (name, term) in old_bindings {
            self.bindings.insert(name, term);
        }
        for (name, sort) in old_constants {
            self.constants.insert(name, sort);
        }

        let var_refs: Vec<_> = vars.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        Ok(self.manager.mk_forall(var_refs, body))
    }

    /// Parse an exists binder: (exists ((name sort) ...) body)
    /// Called after consuming the "exists" symbol
    pub(super) fn parse_exists(&mut self) -> Result<TermId> {
        // Parse sorted vars: ((name sort) ...)
        self.expect_lparen()?;
        let vars = self.parse_sorted_vars()?;

        // Bind quantifier variables (same scoping as forall).
        let old_bindings: Vec<_> = vars
            .iter()
            .filter_map(|(name, _)| self.bindings.get(name).map(|&t| (name.clone(), t)))
            .collect();
        let old_constants: Vec<_> = vars
            .iter()
            .filter_map(|(name, _)| self.constants.get(name).map(|&s| (name.clone(), s)))
            .collect();
        for (name, sort) in &vars {
            let var_term = self.manager.mk_var(name, *sort);
            self.bindings.insert(name.clone(), var_term);
            self.constants.remove(name);
        }

        // Parse body
        let body = self.parse_term()?;
        self.expect_rparen()?;

        // Restore old bindings and constants
        for (name, _) in &vars {
            self.bindings.remove(name);
        }
        for (name, term) in old_bindings {
            self.bindings.insert(name, term);
        }
        for (name, sort) in old_constants {
            self.constants.insert(name, sort);
        }

        let var_refs: Vec<_> = vars.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        Ok(self.manager.mk_exists(var_refs, body))
    }

    /// Parse a list of sorted variable bindings: ((name sort) ...)
    /// Consumes from the first variable up to and including the closing ')'
    pub(super) fn parse_sorted_vars(&mut self) -> Result<Vec<(String, SortId)>> {
        let mut vars = Vec::new();
        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }

            self.expect_lparen()?;
            let name = self.expect_symbol()?;
            let sort = self.parse_sort()?;
            self.expect_rparen()?;
            vars.push((name, sort));
        }
        Ok(vars)
    }

    /// Resolve a sort name to a SortId (handles built-in sorts, aliases, bitvec, uninterpreted)
    pub(super) fn parse_sort_name(&mut self, name: &str) -> Result<SortId> {
        match name {
            "Bool" => Ok(self.manager.sorts.bool_sort),
            "Int" => Ok(self.manager.sorts.int_sort),
            "Real" => Ok(self.manager.sorts.real_sort),
            "String" => Ok(self.manager.sorts.string_sort()),
            _ => {
                // Check for sort alias first
                if let Some((params, base_sort)) = self.sort_aliases.get(name).cloned() {
                    // For now, only support 0-arity sort aliases
                    if params.is_empty() {
                        return self.parse_sort_name(&base_sort);
                    }
                }

                // Check for BitVec
                // Note: Proper SMT-LIB2 syntax is `(_ BitVec n)` which requires
                // parsing an indexed identifier. For now, we support simple names
                // like "BitVec32" as a compromise.
                if let Some(width_str) = name.strip_prefix("BitVec") {
                    if let Ok(width) = width_str.parse::<u32>() {
                        if width > 0 && width <= 65536 {
                            Ok(self.manager.sorts.bitvec(width))
                        } else {
                            Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("invalid BitVec width: {width} (must be 1-65536)"),
                            })
                        }
                    } else if width_str.is_empty() {
                        // Just "BitVec" without width - use default 32
                        Ok(self.manager.sorts.bitvec(32))
                    } else {
                        Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!("invalid BitVec sort name: {name}"),
                        })
                    }
                } else {
                    // Uninterpreted sort
                    let spur = self.manager.intern_str(name);
                    Ok(self
                        .manager
                        .sorts
                        .intern(crate::sort::SortKind::Uninterpreted(spur)))
                }
            }
        }
    }

    /// Parse an indexed identifier: (_ name index1 index2 ...)
    /// Returns (name, indices). LParen already consumed by caller; consumes trailing RParen.
    pub(super) fn parse_indexed_identifier(&mut self) -> Result<(String, Vec<u32>)> {
        // Expect underscore symbol
        let underscore = self.expect_symbol()?;
        if underscore != "_" {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("expected '_', found '{underscore}'"),
            });
        }

        // Get the identifier name
        let name = self.expect_symbol()?;

        // Parse indices (numerals)
        let mut indices = Vec::new();
        loop {
            if let Some(token) = self.lexer.peek() {
                match &token.kind {
                    TokenKind::RParen => {
                        self.lexer.next_token(); // consume rparen
                        break;
                    }
                    TokenKind::Numeral(n) => {
                        let n = n.clone();
                        self.lexer.next_token();
                        let idx = n.parse::<u32>().map_err(|_| OxizError::ParseError {
                            position: token.start,
                            message: format!("invalid index: {n}"),
                        })?;
                        indices.push(idx);
                    }
                    _ => {
                        return Err(OxizError::ParseError {
                            position: token.start,
                            message: format!("expected numeral or ')', found {:?}", token.kind),
                        });
                    }
                }
            } else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input in indexed identifier".to_string(),
                });
            }
        }

        Ok((name, indices))
    }

    /// Parse a sort expression (simple name, indexed identifier, or parametric sort)
    pub(super) fn parse_sort(&mut self) -> Result<SortId> {
        if let Some(token) = self.lexer.peek() {
            match &token.kind {
                TokenKind::Symbol(s) => {
                    let s = s.clone();
                    self.lexer.next_token();
                    self.parse_sort_name(&s)
                }
                TokenKind::LParen => {
                    self.lexer.next_token(); // consume lparen

                    // Check if this is an indexed identifier or a parametric sort like Array
                    let next_token = self.lexer.peek().ok_or_else(|| OxizError::ParseError {
                        position: self.lexer.position(),
                        message: "unexpected end of input in sort".to_string(),
                    })?;

                    if matches!(next_token.kind, TokenKind::Symbol(ref s) if s == "_") {
                        // Indexed identifier: (_ BitVec 32)
                        let (name, indices) = self.parse_indexed_identifier()?;

                        match name.as_str() {
                            "BitVec" => {
                                if indices.len() != 1 {
                                    return Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "BitVec requires exactly 1 index, got {}",
                                            indices.len()
                                        ),
                                    });
                                }
                                let width = indices[0];
                                if width > 0 && width <= 65536 {
                                    Ok(self.manager.sorts.bitvec(width))
                                } else {
                                    Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "invalid BitVec width: {width} (must be 1-65536)"
                                        ),
                                    })
                                }
                            }
                            "FloatingPoint" => {
                                if indices.len() != 2 {
                                    return Err(OxizError::ParseError {
                                        position: self.lexer.position(),
                                        message: format!(
                                            "FloatingPoint requires exactly 2 indices (eb, sb), got {}",
                                            indices.len()
                                        ),
                                    });
                                }
                                let eb = indices[0]; // exponent bits
                                let sb = indices[1]; // significand bits
                                Ok(self.manager.sorts.float_sort(eb, sb))
                            }
                            _ => Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("unknown indexed sort: {name}"),
                            }),
                        }
                    } else if let TokenKind::Symbol(s) = &next_token.kind {
                        // Parametric sort like (Array Int Int)
                        let sort_name = s.clone();
                        self.lexer.next_token(); // consume the symbol

                        match sort_name.as_str() {
                            "Array" => {
                                // Parse domain and range sorts
                                let domain = self.parse_sort()?;
                                let range = self.parse_sort()?;
                                self.expect_rparen()?;
                                Ok(self.manager.sorts.array(domain, range))
                            }
                            _ => Err(OxizError::ParseError {
                                position: self.lexer.position(),
                                message: format!("unknown parametric sort: {sort_name}"),
                            }),
                        }
                    } else {
                        Err(OxizError::ParseError {
                            position: next_token.start,
                            message: format!("unexpected token in sort: {:?}", next_token.kind),
                        })
                    }
                }
                _ => Err(OxizError::ParseError {
                    position: token.start,
                    message: format!("expected sort, found {:?}", token.kind),
                }),
            }
        } else {
            Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: "expected sort, found end of input".to_string(),
            })
        }
    }

    /// Convert a SortId to its canonical SMT-LIB2 string representation
    pub(super) fn sort_id_to_string(&self, sort_id: SortId) -> String {
        if let Some(sort) = self.manager.sorts.get(sort_id) {
            match &sort.kind {
                crate::sort::SortKind::Bool => "Bool".to_string(),
                crate::sort::SortKind::Int => "Int".to_string(),
                crate::sort::SortKind::Real => "Real".to_string(),
                crate::sort::SortKind::String => "String".to_string(),
                crate::sort::SortKind::BitVec(w) => format!("(_ BitVec {w})"),
                crate::sort::SortKind::FloatingPoint { eb, sb } => {
                    format!("(_ FloatingPoint {eb} {sb})")
                }
                crate::sort::SortKind::Array { domain, range } => {
                    let domain_str = self.sort_id_to_string(*domain);
                    let range_str = self.sort_id_to_string(*range);
                    format!("(Array {domain_str} {range_str})")
                }
                crate::sort::SortKind::Uninterpreted(spur) => {
                    self.manager.resolve_str(*spur).to_string()
                }
                crate::sort::SortKind::Datatype(spur) => {
                    self.manager.resolve_str(*spur).to_string()
                }
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        }
    }
}
