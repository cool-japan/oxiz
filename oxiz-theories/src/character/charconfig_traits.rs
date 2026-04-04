//! # CharConfig - Trait Implementations
//!
//! This module contains trait implementations for `CharConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CharConfig;
#[allow(unused_imports)]
use crate::prelude::*;

impl Default for CharConfig {
    fn default() -> Self {
        Self {
            max_code_point: 0x10FFFF,
            normalize_unicode: true,
            case_folding: true,
        }
    }
}
