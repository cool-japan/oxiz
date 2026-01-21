//! Constraint addition methods for LIA solver

use super::super::simplex::LinExpr;
use super::types::LiaSolver;

impl LiaSolver {
    /// Add a linear constraint: expr <= 0
    pub fn add_le(&mut self, expr: LinExpr, reason: u32) {
        self.simplex.add_le(expr, reason);
    }

    /// Add a linear constraint: expr >= 0
    pub fn add_ge(&mut self, expr: LinExpr, reason: u32) {
        self.simplex.add_ge(expr, reason);
    }

    /// Add an equality constraint: expr = 0
    pub fn add_eq(&mut self, expr: LinExpr, reason: u32) {
        self.simplex.add_eq(expr, reason);
    }
}
