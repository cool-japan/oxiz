//! Spacer-oriented preprocessing tactics.
//!
//! ## `BmcUnrollTactic` vs. the production `oxiz_spacer::Bmc` solver
//!
//! [`BmcUnrollTactic`] is a **lightweight, goal-level preprocessing tactic**.
//! It takes a [`oxiz_core::tactic::Goal`] encoding an `(init, trans, property)`
//! triple as a flat list of [`oxiz_core::ast::TermId`] assertions and rewrites
//! it into an unrolled BMC formula up to a configurable depth.  It does *not*
//! invoke any SMT solver; the resulting goal is handed back to the caller for
//! further processing.
//!
//! The production [`oxiz_spacer::Bmc`] solver (in `oxiz_spacer::bmc`) is
//! completely distinct: it operates on a full [`oxiz_spacer::chc::ChcSystem`],
//! runs CHC/PDR-based model checking internally, and returns a
//! [`oxiz_spacer::bmc::BmcResult`].  Use `BmcUnrollTactic` for tactic pipeline
//! preprocessing, and `Bmc` for end-to-end model-checking queries.
//!
//! [`oxiz_spacer::Bmc`]: crate::Bmc
//! [`oxiz_spacer::bmc::BmcResult`]: crate::BmcResult

mod bmc_unroll;

pub use bmc_unroll::{BmcEngine, BmcUnrollTactic};
