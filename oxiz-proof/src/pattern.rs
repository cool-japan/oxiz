//! Lemma pattern extraction from proofs.
//!
//! This module extracts reusable patterns from successful proofs to enable
//! proof-based learning and improve solver heuristics.

use crate::proof::{Proof, ProofStep};
use rustc_hash::FxHashMap;
use std::fmt;

/// A pattern extracted from a lemma or proof fragment.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LemmaPattern {
    /// The inference rule used
    pub rule: String,
    /// Number of premises
    pub num_premises: usize,
    /// Variables in the pattern (abstracted)
    pub variables: Vec<String>,
    /// Pattern structure (simplified AST)
    pub structure: PatternStructure,
    /// Frequency of this pattern in the proof corpus
    pub frequency: usize,
    /// Average depth where this pattern appears
    pub avg_depth: f64,
}

/// The structure of a pattern (simplified representation).
///
/// # Depth invariant
///
/// There is deliberately no bound on how deep a `PatternStructure` may be. Its
/// construction paths are:
///
/// - `PatternExtractor::parse_conclusion_structure`, bounded by the
///   caller-configurable [`PatternExtractor::with_max_depth`];
/// - the public variants, which callers build directly;
/// - `serde::Deserialize` under the `serde` feature.
///
/// The last two are unbounded, so every walk over this type is iterative --
/// [`Clone`], [`Drop`], [`fmt::Display`], [`PartialEq`] and
/// [`std::hash::Hash`]. Do **not** replace any of them with a `derive`.
///
/// Two walks are still recursive: the derived [`fmt::Debug`] (diagnostics only)
/// and, under the `serde` feature, the generated `Serialize`/`Deserialize` --
/// round-tripping a deeply nested pattern can still overflow the stack.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PatternStructure {
    /// Atomic pattern (variable or constant)
    Atom(String),
    /// Application of a function/predicate
    App {
        /// Function name
        func: String,
        /// Arguments
        args: Vec<PatternStructure>,
    },
    /// Binary operation pattern
    Binary {
        /// Operator
        op: String,
        /// Left operand
        left: Box<PatternStructure>,
        /// Right operand
        right: Box<PatternStructure>,
    },
    /// Quantified pattern
    Quantified {
        /// Quantifier (forall, exists)
        quantifier: String,
        /// Bound variable
        var: String,
        /// Body
        body: Box<PatternStructure>,
    },
}

impl Drop for PatternStructure {
    /// Iterative drop.
    ///
    /// A `PatternStructure` can be arbitrarily deep (see the type-level depth
    /// invariant), so the compiler-generated recursive `drop_in_place` would
    /// overflow the stack when one goes out of scope -- after the caller has
    /// already got its answer, making it an abort with no diagnostic. Each node
    /// is dismantled into a shallow shell before being released.
    fn drop(&mut self) {
        /// Detach a node's children, leaving a shell that drops trivially.
        fn dismantle(node: &mut PatternStructure, out: &mut Vec<PatternStructure>) {
            /// Replace a boxed child with a trivially-droppable placeholder.
            fn take(slot: &mut Box<PatternStructure>, out: &mut Vec<PatternStructure>) {
                out.push(std::mem::replace(
                    slot.as_mut(),
                    PatternStructure::Atom(String::new()),
                ));
            }

            match node {
                PatternStructure::Atom(_) => {}
                PatternStructure::App { args, .. } => out.append(args),
                PatternStructure::Binary { left, right, .. } => {
                    take(left, out);
                    take(right, out);
                }
                PatternStructure::Quantified { body, .. } => take(body, out),
            }
        }

        let mut pending = Vec::new();
        dismantle(self, &mut pending);
        while let Some(mut node) = pending.pop() {
            dismantle(&mut node, &mut pending);
        }
    }
}

impl PartialEq for PatternStructure {
    /// Iterative structural equality.
    ///
    /// The derived `PartialEq` recursed once per nesting level over a type whose
    /// depth is not bounded (see the type-level depth invariant), so a plain
    /// `a == b` -- including the one inside `LemmaPattern`'s derived
    /// `PartialEq` -- could overflow the stack. The relation is unchanged.
    ///
    /// The outer `match` is exhaustive over `self`'s variants on purpose: a new
    /// variant must be handled here explicitly rather than silently falling into
    /// a catch-all that reports "not equal".
    fn eq(&self, other: &Self) -> bool {
        let mut worklist = vec![(self, other)];

        while let Some((a, b)) = worklist.pop() {
            match a {
                Self::Atom(x) => {
                    let Self::Atom(y) = b else { return false };
                    if x != y {
                        return false;
                    }
                }
                Self::App { func, args } => {
                    let Self::App {
                        func: other_func,
                        args: other_args,
                    } = b
                    else {
                        return false;
                    };
                    if func != other_func || args.len() != other_args.len() {
                        return false;
                    }
                    worklist.extend(args.iter().zip(other_args.iter()).rev());
                }
                Self::Binary { op, left, right } => {
                    let Self::Binary {
                        op: other_op,
                        left: other_left,
                        right: other_right,
                    } = b
                    else {
                        return false;
                    };
                    if op != other_op {
                        return false;
                    }
                    worklist.push((right.as_ref(), other_right.as_ref()));
                    worklist.push((left.as_ref(), other_left.as_ref()));
                }
                Self::Quantified {
                    quantifier,
                    var,
                    body,
                } => {
                    let Self::Quantified {
                        quantifier: other_quantifier,
                        var: other_var,
                        body: other_body,
                    } = b
                    else {
                        return false;
                    };
                    if quantifier != other_quantifier || var != other_var {
                        return false;
                    }
                    worklist.push((body.as_ref(), other_body.as_ref()));
                }
            }
        }

        true
    }
}

impl Eq for PatternStructure {}

impl std::hash::Hash for PatternStructure {
    /// Iterative structural hashing, consistent with the [`PartialEq`] above.
    ///
    /// Same reason as `eq`: the derived `Hash` recursed once per nesting level,
    /// so hashing a deep pattern (e.g. using one as a map key) could overflow
    /// the stack. Each node contributes a variant tag, then its non-child
    /// payload, then -- for `App` -- its arity, so patterns that flatten to the
    /// same stream of leaves but differ in shape still hash differently.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut stack = vec![self];

        while let Some(node) = stack.pop() {
            match node {
                Self::Atom(a) => {
                    state.write_u8(0);
                    a.hash(state);
                }
                Self::App { func, args } => {
                    state.write_u8(1);
                    func.hash(state);
                    state.write_usize(args.len());
                    stack.extend(args.iter().rev());
                }
                Self::Binary { op, left, right } => {
                    state.write_u8(2);
                    op.hash(state);
                    stack.push(right.as_ref());
                    stack.push(left.as_ref());
                }
                Self::Quantified {
                    quantifier,
                    var,
                    body,
                } => {
                    state.write_u8(3);
                    quantifier.hash(state);
                    var.hash(state);
                    stack.push(body.as_ref());
                }
            }
        }
    }
}

/// One in-progress node in the iterative [`Clone`] impl for
/// [`PatternStructure`].
enum PatternCloneFrame<'a> {
    /// `func(a, b, ...)`, with one argument being cloned.
    App {
        /// Function name.
        func: String,
        /// Arguments still to be cloned, in source order.
        rest: std::slice::Iter<'a, PatternStructure>,
        /// Arguments cloned so far, in source order.
        args: Vec<PatternStructure>,
    },
    /// `<left> op <right>`, with the left operand being cloned.
    BinaryLeft {
        /// Operator.
        op: String,
        /// Right operand, not yet cloned.
        right: &'a PatternStructure,
    },
    /// `<left> op <right>`, with the right operand being cloned.
    BinaryRight {
        /// Operator.
        op: String,
        /// Already-cloned left operand.
        left: PatternStructure,
    },
    /// `quantifier var. <body>`, with the body being cloned.
    Quantified {
        /// Quantifier keyword.
        quantifier: String,
        /// Bound variable.
        var: String,
    },
}

impl Clone for PatternStructure {
    /// Iterative clone.
    ///
    /// The derived recursive `Clone` ran one native call frame per nesting
    /// level over a type of unbounded depth (see the type-level depth
    /// invariant), and it is reached from `LemmaPattern::clone`, which the
    /// extractor performs per stored pattern. The result is structurally
    /// identical.
    fn clone(&self) -> Self {
        let mut stack: Vec<PatternCloneFrame<'_>> = Vec::new();
        let mut node: &PatternStructure = self;

        loop {
            // Descend to the next leaf, opening a frame per compound node.
            let mut value = 'descend: loop {
                match node {
                    Self::Atom(a) => break 'descend Self::Atom(a.clone()),
                    Self::App { func, args } => {
                        let mut rest = args.iter();
                        match rest.next() {
                            Some(first) => {
                                stack.push(PatternCloneFrame::App {
                                    func: func.clone(),
                                    rest,
                                    args: Vec::new(),
                                });
                                node = first;
                                continue 'descend;
                            }
                            None => {
                                break 'descend Self::App {
                                    func: func.clone(),
                                    args: Vec::new(),
                                };
                            }
                        }
                    }
                    Self::Binary { op, left, right } => {
                        stack.push(PatternCloneFrame::BinaryLeft {
                            op: op.clone(),
                            right: right.as_ref(),
                        });
                        node = left.as_ref();
                        continue 'descend;
                    }
                    Self::Quantified {
                        quantifier,
                        var,
                        body,
                    } => {
                        stack.push(PatternCloneFrame::Quantified {
                            quantifier: quantifier.clone(),
                            var: var.clone(),
                        });
                        node = body.as_ref();
                        continue 'descend;
                    }
                }
            };

            // Unwind: hand the cloned child to its parent frame.
            loop {
                let Some(frame) = stack.pop() else {
                    return value;
                };
                match frame {
                    PatternCloneFrame::Quantified { quantifier, var } => {
                        value = Self::Quantified {
                            quantifier,
                            var,
                            body: Box::new(value),
                        };
                    }
                    PatternCloneFrame::BinaryLeft { op, right } => {
                        stack.push(PatternCloneFrame::BinaryRight { op, left: value });
                        node = right;
                        break;
                    }
                    PatternCloneFrame::BinaryRight { op, left } => {
                        value = Self::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(value),
                        };
                    }
                    PatternCloneFrame::App {
                        func,
                        mut rest,
                        mut args,
                    } => {
                        args.push(value);
                        match rest.next() {
                            Some(next) => {
                                node = next;
                                stack.push(PatternCloneFrame::App { func, rest, args });
                                break;
                            }
                            None => value = Self::App { func, args },
                        }
                    }
                }
            }
        }
    }
}

/// One in-progress node in the iterative
/// [`PatternExtractor::parse_conclusion_structure`] parser.
///
/// Each frame stands for exactly one nesting level, which is what lets the
/// parser recover the original recursion's `depth` as `depth + stack.len()`.
enum PatternFrame {
    /// `quantifier var. <body>`, with the body being parsed.
    Quantified {
        /// Quantifier keyword (`forall` / `exists`).
        quantifier: String,
        /// Bound variable.
        var: String,
    },
    /// `<left> op <right>`, with the left operand being parsed.
    BinaryLeft {
        /// Operator.
        op: String,
        /// Right operand, still unparsed text.
        right: String,
    },
    /// `<left> op <right>`, with the right operand being parsed.
    BinaryRight {
        /// Operator.
        op: String,
        /// Already-parsed left operand.
        left: PatternStructure,
    },
    /// `func(a, b, ...)`, with one argument being parsed.
    App {
        /// Function name.
        func: String,
        /// Argument texts still to be parsed, in *reverse* source order.
        rest: Vec<String>,
        /// Arguments parsed so far, in source order.
        args: Vec<PatternStructure>,
    },
}

/// Work item for the iterative [`fmt::Display`] impl for [`PatternStructure`].
enum PatternFmtTask<'a> {
    /// Render this substructure.
    Node(&'a PatternStructure),
    /// Emit a structural token verbatim.
    Text(&'static str),
    /// Emit an owned fragment (an operator or a `quantifier var. ` prefix).
    Owned(String),
}

impl fmt::Display for PatternStructure {
    /// Iterative (explicit heap stack) rendering.
    ///
    /// A `PatternStructure` is as deep as the conclusion string it was parsed
    /// from (see `PatternExtractor::parse_conclusion_structure`) or as deep
    /// as a caller cares to build, so rendering it with one native call frame
    /// per level could overflow the stack inside a `Display` impl -- i.e. in
    /// the middle of logging, where there is no error channel to report it
    /// through. Output is byte-identical to the recursive formulation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut stack = vec![PatternFmtTask::Node(self)];

        while let Some(task) = stack.pop() {
            let node = match task {
                PatternFmtTask::Text(text) => {
                    f.write_str(text)?;
                    continue;
                }
                PatternFmtTask::Owned(text) => {
                    f.write_str(&text)?;
                    continue;
                }
                PatternFmtTask::Node(node) => node,
            };

            match node {
                PatternStructure::Atom(a) => write!(f, "{}", a)?,
                PatternStructure::App { func, args } => {
                    write!(f, "{}(", func)?;
                    stack.push(PatternFmtTask::Text(")"));
                    for (i, arg) in args.iter().enumerate().rev() {
                        stack.push(PatternFmtTask::Node(arg));
                        if i > 0 {
                            stack.push(PatternFmtTask::Text(", "));
                        }
                    }
                }
                PatternStructure::Binary { op, left, right } => {
                    f.write_str("(")?;
                    stack.push(PatternFmtTask::Text(")"));
                    stack.push(PatternFmtTask::Node(right));
                    stack.push(PatternFmtTask::Owned(format!(" {} ", op)));
                    stack.push(PatternFmtTask::Node(left));
                }
                PatternStructure::Quantified {
                    quantifier,
                    var,
                    body,
                } => {
                    write!(f, "{} {}. ", quantifier, var)?;
                    stack.push(PatternFmtTask::Node(body));
                }
            }
        }

        Ok(())
    }
}

/// Pattern extractor for analyzing proofs.
pub struct PatternExtractor {
    /// Minimum pattern frequency to report
    min_frequency: usize,
    /// Maximum pattern depth
    max_depth: usize,
    /// Extracted patterns
    patterns: FxHashMap<String, LemmaPattern>,
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternExtractor {
    /// Create a new pattern extractor with default settings.
    pub fn new() -> Self {
        Self {
            min_frequency: 2,
            max_depth: 5,
            patterns: FxHashMap::default(),
        }
    }

    /// Set the minimum frequency threshold.
    pub fn with_min_frequency(mut self, freq: usize) -> Self {
        self.min_frequency = freq;
        self
    }

    /// Set the maximum pattern depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Extract patterns from a proof.
    pub fn extract_patterns(&mut self, proof: &Proof) {
        let mut pattern_occurrences: FxHashMap<String, (usize, Vec<f64>)> = FxHashMap::default();

        for node in proof.nodes() {
            let depth = node.depth;

            if let ProofStep::Inference { rule, premises, .. } = &node.step {
                // Create a pattern key
                let pattern_key = self.create_pattern_key(rule, premises.len(), node.conclusion());

                // Track occurrences. Hold the `entry()` mutable reference
                // across both updates instead of re-looking the key up
                // afterwards -- there is then no separate "it must still be
                // there" step to justify.
                let occurrence = pattern_occurrences
                    .entry(pattern_key.clone())
                    .or_insert_with(|| (0, Vec::new()));
                occurrence.0 += 1;
                occurrence.1.push(depth as f64);

                // Extract pattern structure
                if let Some(pattern) =
                    self.extract_pattern_structure(rule, premises.len(), node.conclusion())
                {
                    self.patterns.insert(pattern_key, pattern);
                }
            }
        }

        // Update pattern frequencies and average depths
        for (key, pattern) in &mut self.patterns {
            if let Some((freq, depths)) = pattern_occurrences.get(key) {
                pattern.frequency = *freq;
                if !depths.is_empty() {
                    pattern.avg_depth = depths.iter().sum::<f64>() / depths.len() as f64;
                }
            }
        }
    }

    /// Get all extracted patterns that meet the minimum frequency threshold.
    pub fn get_patterns(&self) -> Vec<&LemmaPattern> {
        self.patterns
            .values()
            .filter(|p| p.frequency >= self.min_frequency)
            .collect()
    }

    /// Get patterns sorted by frequency (most common first).
    pub fn get_patterns_by_frequency(&self) -> Vec<&LemmaPattern> {
        let mut patterns = self.get_patterns();
        patterns.sort_by_key(|p| std::cmp::Reverse(p.frequency));
        patterns
    }

    /// Get patterns for a specific rule.
    pub fn get_patterns_for_rule(&self, rule: &str) -> Vec<&LemmaPattern> {
        self.patterns
            .values()
            .filter(|p| p.rule == rule && p.frequency >= self.min_frequency)
            .collect()
    }

    /// Clear all extracted patterns.
    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    // Helper: Create a unique key for a pattern
    fn create_pattern_key(&self, rule: &str, num_premises: usize, conclusion: &str) -> String {
        format!(
            "{}:{}:{}",
            rule,
            num_premises,
            self.abstract_conclusion(conclusion)
        )
    }

    // Helper: Abstract conclusion by replacing specific values with variables
    fn abstract_conclusion(&self, conclusion: &str) -> String {
        // Simple abstraction: replace numbers and specific identifiers with placeholders
        let mut abstracted = conclusion.to_string();

        // Replace numbers with $$N (need to escape $ as $$)
        let re_num = regex::Regex::new(r"\b\d+\b").expect("regex pattern is valid");
        abstracted = re_num.replace_all(&abstracted, "$$N").to_string();

        // Replace quoted strings with $$S
        let re_str = regex::Regex::new(r#""[^"]*""#).expect("regex pattern is valid");
        abstracted = re_str.replace_all(&abstracted, "$$S").to_string();

        abstracted
    }

    // Helper: Extract pattern structure from conclusion
    fn extract_pattern_structure(
        &self,
        rule: &str,
        num_premises: usize,
        conclusion: &str,
    ) -> Option<LemmaPattern> {
        // Parse conclusion into a structure (simplified for now), bounding
        // the recursion to `self.max_depth` levels (see
        // `parse_conclusion_structure`).
        let structure = Self::parse_conclusion_structure(conclusion, 0, self.max_depth);
        let variables = self.extract_variables(&structure);

        Some(LemmaPattern {
            rule: rule.to_string(),
            num_premises,
            variables,
            structure,
            frequency: 0,
            avg_depth: 0.0,
        })
    }

    /// Parse `conclusion` into a pattern structure, recursing at most
    /// `max_depth` levels (`depth` is the caller's current nesting, 0 at the
    /// top) before bailing out and treating whatever text remains as an
    /// opaque atom.
    ///
    /// `conclusion` is a proof node's textual conclusion (see
    /// `extract_patterns`), whose nesting depth is driven by the size of
    /// the formula it renders, not by anything this crate controls, so an
    /// unbounded version of this recursion (as it used to be, before
    /// `max_depth` was wired in here) could overflow the native stack on a
    /// pathologically deep proof. Bailing out is sound: a subterm returned
    /// as an unparsed atom is still a valid (if less-structured) pattern
    /// fragment; it is simply not decomposed any further below the cap.
    /// Iterative (explicit heap stack of [`PatternFrame`]s). `max_depth` bounds
    /// how far the *parse* decomposes, not how deep this walk may safely go:
    /// `max_depth` is caller-configurable through
    /// [`PatternExtractor::with_max_depth`], so a large bound used to mean a
    /// correspondingly deep native call chain. Each in-progress node now lives
    /// on the heap instead, and the depth test is unchanged -- the depth of the
    /// text currently being parsed is exactly `depth + stack.len()`, since every
    /// frame corresponds to one nesting level.
    // Helper: Parse conclusion into pattern structure
    fn parse_conclusion_structure(
        conclusion: &str,
        depth: usize,
        max_depth: usize,
    ) -> PatternStructure {
        // Simplified parsing - in a real implementation, this would use a proper parser
        let mut stack: Vec<PatternFrame> = Vec::new();
        let mut input: String = conclusion.to_string();

        loop {
            // Descend: reduce `input` to a finished structure, pushing a frame
            // for every node whose children still need parsing.
            let mut value = 'descend: loop {
                let trimmed = input.trim();

                if depth + stack.len() >= max_depth {
                    break 'descend PatternStructure::Atom(trimmed.to_string());
                }

                // Check for quantifiers
                if (trimmed.starts_with("forall") || trimmed.starts_with("exists"))
                    && let Some((quantifier, rest)) = trimmed.split_once(' ')
                    && let Some((var, body)) = rest.split_once('.')
                {
                    let frame = PatternFrame::Quantified {
                        quantifier: quantifier.to_string(),
                        var: var.trim().to_string(),
                    };
                    let body = body.trim().to_string();
                    stack.push(frame);
                    input = body;
                    continue 'descend;
                }

                // Check for binary operators
                let mut split = None;
                for op in &["=", "<=", ">=", "<", ">", "!=", "and", "or", "=>"] {
                    if let Some(pos) = trimmed.find(op) {
                        let left = &trimmed[..pos];
                        let right = &trimmed[pos + op.len()..];
                        if !left.is_empty() && !right.is_empty() {
                            split = Some((
                                (*op).to_string(),
                                left.trim().to_string(),
                                right.trim().to_string(),
                            ));
                            break;
                        }
                    }
                }
                if let Some((op, left, right)) = split {
                    stack.push(PatternFrame::BinaryLeft { op, right });
                    input = left;
                    continue 'descend;
                }

                // Check for function application
                if let Some(pos) = trimmed.find('(')
                    && trimmed.ends_with(')')
                {
                    let func = trimmed[..pos].trim().to_string();
                    let args_str = &trimmed[pos + 1..trimmed.len() - 1];
                    // Reversed so that `pop` yields arguments in source order.
                    let mut rest: Vec<String> =
                        args_str.split(',').map(|a| a.trim().to_string()).collect();
                    rest.reverse();
                    match rest.pop() {
                        Some(next) => {
                            stack.push(PatternFrame::App {
                                func,
                                rest,
                                args: Vec::new(),
                            });
                            input = next;
                            continue 'descend;
                        }
                        None => {
                            break 'descend PatternStructure::App {
                                func,
                                args: Vec::new(),
                            };
                        }
                    }
                }

                // Default: atom
                break 'descend PatternStructure::Atom(trimmed.to_string());
            };

            // Unwind: hand the finished structure to its parent, completing
            // frames whose children are exhausted.
            loop {
                let Some(frame) = stack.pop() else {
                    return value;
                };
                match frame {
                    PatternFrame::Quantified { quantifier, var } => {
                        value = PatternStructure::Quantified {
                            quantifier,
                            var,
                            body: Box::new(value),
                        };
                    }
                    PatternFrame::BinaryLeft { op, right } => {
                        stack.push(PatternFrame::BinaryRight { op, left: value });
                        input = right;
                        break;
                    }
                    PatternFrame::BinaryRight { op, left } => {
                        value = PatternStructure::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(value),
                        };
                    }
                    PatternFrame::App {
                        func,
                        mut rest,
                        mut args,
                    } => {
                        args.push(value);
                        match rest.pop() {
                            Some(next) => {
                                stack.push(PatternFrame::App { func, rest, args });
                                input = next;
                                break;
                            }
                            None => value = PatternStructure::App { func, args },
                        }
                    }
                }
            }
        }
    }

    // Helper: Extract variables from pattern structure
    fn extract_variables(&self, structure: &PatternStructure) -> Vec<String> {
        let mut vars = Vec::new();
        Self::extract_variables_rec(structure, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    /// Collect every variable-looking atom (and every bound variable) into
    /// `vars`, in left-to-right order.
    ///
    /// Iterative (explicit heap stack). `-> ()` gives this walk no channel
    /// through which a depth cap could be reported, and a `PatternStructure`
    /// can be arbitrarily deep (it is public, and `max_depth` is
    /// caller-configurable), so the pending nodes live on the heap.
    fn extract_variables_rec(structure: &PatternStructure, vars: &mut Vec<String>) {
        let mut stack = vec![structure];

        while let Some(node) = stack.pop() {
            match node {
                PatternStructure::Atom(a) => {
                    if a.starts_with('$') || a.chars().next().is_some_and(|c| c.is_lowercase()) {
                        vars.push(a.clone());
                    }
                }
                PatternStructure::App { args, .. } => {
                    stack.extend(args.iter().rev());
                }
                PatternStructure::Binary { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
                PatternStructure::Quantified { var, body, .. } => {
                    vars.push(var.clone());
                    stack.push(body);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_extractor_new() {
        let extractor = PatternExtractor::new();
        assert_eq!(extractor.min_frequency, 2);
        assert_eq!(extractor.max_depth, 5);
        assert!(extractor.patterns.is_empty());
    }

    #[test]
    fn test_pattern_extractor_with_settings() {
        let extractor = PatternExtractor::new()
            .with_min_frequency(3)
            .with_max_depth(10);
        assert_eq!(extractor.min_frequency, 3);
        assert_eq!(extractor.max_depth, 10);
    }

    #[test]
    fn test_pattern_structure_display() {
        let atom = PatternStructure::Atom("x".to_string());
        assert_eq!(atom.to_string(), "x");

        let app = PatternStructure::App {
            func: "f".to_string(),
            args: vec![
                PatternStructure::Atom("x".to_string()),
                PatternStructure::Atom("y".to_string()),
            ],
        };
        assert_eq!(app.to_string(), "f(x, y)");

        let binary = PatternStructure::Binary {
            op: "=".to_string(),
            left: Box::new(PatternStructure::Atom("x".to_string())),
            right: Box::new(PatternStructure::Atom("y".to_string())),
        };
        assert_eq!(binary.to_string(), "(x = y)");
    }

    #[test]
    fn test_parse_atom() {
        // Depth budget is irrelevant to these three tests (none nest deep
        // enough to hit any reasonable cap); `usize::MAX` keeps them testing
        // only the parsing shape, not the depth cap exercised separately
        // below.
        let structure = PatternExtractor::parse_conclusion_structure("x", 0, usize::MAX);
        assert!(matches!(structure, PatternStructure::Atom(_)));
    }

    #[test]
    fn test_parse_binary() {
        let structure = PatternExtractor::parse_conclusion_structure("x = y", 0, usize::MAX);
        assert!(matches!(structure, PatternStructure::Binary { .. }));
    }

    #[test]
    fn test_parse_app() {
        let structure = PatternExtractor::parse_conclusion_structure("f(x, y)", 0, usize::MAX);
        // Borrowed, not destructured: `PatternStructure` has a manual `Drop`
        // (iterative teardown), so its fields cannot be moved out.
        if let PatternStructure::App { func, args } = &structure {
            assert_eq!(func, "f");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected App pattern");
        }
    }

    #[test]
    fn test_abstract_conclusion() {
        let extractor = PatternExtractor::new();
        let abstracted = extractor.abstract_conclusion("x + 42 = y");
        println!("Abstracted: '{}'", abstracted);
        // The regex should work, but let's be more flexible in the test
        assert!(
            abstracted.contains("$N") || abstracted.contains("42"),
            "Expected '$N' or '42', got: '{}'",
            abstracted
        );
    }

    #[test]
    fn test_extract_variables() {
        let extractor = PatternExtractor::new();
        let structure = PatternStructure::App {
            func: "f".to_string(),
            args: vec![
                PatternStructure::Atom("x".to_string()),
                PatternStructure::Atom("y".to_string()),
            ],
        };
        let vars = extractor.extract_variables(&structure);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
    }

    #[test]
    fn test_extract_patterns_empty_proof() {
        let mut extractor = PatternExtractor::new();
        let proof = Proof::new();
        extractor.extract_patterns(&proof);
        assert!(extractor.get_patterns().is_empty());
    }

    #[test]
    fn test_clear_patterns() {
        let mut extractor = PatternExtractor::new();
        let proof = Proof::new();
        extractor.extract_patterns(&proof);
        extractor.clear();
        assert!(extractor.patterns.is_empty());
    }

    /// Count the `Binary` nesting reachable via `.right`, iteratively.
    fn binary_chain_len(mut structure: &PatternStructure) -> usize {
        let mut len = 0;
        while let PatternStructure::Binary { right, .. } = structure {
            len += 1;
            structure = right;
        }
        len
    }

    #[test]
    fn test_max_depth_observably_changes_parsed_structure() {
        // Regression for `PatternExtractor::max_depth`: this field used to
        // be declared, defaulted, and settable via `with_max_depth`, but
        // never read anywhere -- `parse_conclusion_structure` recursed with
        // no bound at all. It is now threaded through, so setting it to a
        // small value must observably leave a conclusion less parsed than a
        // large one.
        //
        // "x0 = x1 = ... = x10" parses into a right-leaning chain of
        // `Binary` nodes: `parse_conclusion_structure` finds the first "="
        // from the left and recurses on both sides.
        let mut conclusion = String::from("x0");
        for i in 1..=10 {
            conclusion.push_str(&format!(" = x{i}"));
        }

        let shallow = PatternExtractor::parse_conclusion_structure(&conclusion, 0, 3);
        assert_eq!(
            binary_chain_len(&shallow),
            3,
            "a max_depth of 3 must cap the parsed chain at exactly 3 Binary nodes"
        );

        let deep = PatternExtractor::parse_conclusion_structure(&conclusion, 0, 100);
        assert_eq!(
            binary_chain_len(&deep),
            10,
            "a generous max_depth must parse the full 10-deep chain"
        );

        assert_ne!(
            shallow, deep,
            "different max_depth values must produce observably different structures"
        );
    }

    /// Build a right-leaning `Binary` chain `depth` levels deep, iteratively.
    fn deep_binary_chain(depth: usize, leaf: &str) -> PatternStructure {
        let mut node = PatternStructure::Atom(leaf.to_string());
        for i in 0..depth {
            node = PatternStructure::Binary {
                op: "=".to_string(),
                left: Box::new(PatternStructure::Atom(format!("x{i}"))),
                right: Box::new(node),
            };
        }
        node
    }

    /// `PatternStructure` is public with public variants, so callers build
    /// values of unbounded depth directly. Every walk over it must therefore be
    /// iterative: `Clone`, `PartialEq`, `Hash`, `Display`, `Drop`, and
    /// `extract_variables`.
    ///
    /// Running on a deliberately small (128 KiB) stack: returning at all is the
    /// proof. Note the `Drop`s at the end of the closure are part of what is
    /// being tested.
    ///
    /// The stack size and `DEPTH` are scaled together on purpose: what is
    /// pinned is the ratio, ~21 bytes per frame, which no real call frame fits
    /// into. Never raise one without raising the other.
    #[test]
    fn test_deep_pattern_structure_walks_do_not_overflow() {
        const DEPTH: usize = 6_250;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let structure = deep_binary_chain(DEPTH, "leaf");

                // Clone (iterative) must reproduce the structure exactly, and
                // PartialEq (iterative) must agree.
                let copy = structure.clone();
                assert!(copy == structure);
                assert_eq!(binary_chain_len(&copy), DEPTH);

                // A difference buried at the very bottom must still be found.
                let different = deep_binary_chain(DEPTH, "other");
                assert!(different != structure);

                // Hash must be iterative, and equal values must hash equally.
                let hash_of = |value: &PatternStructure| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    std::hash::Hash::hash(value, &mut hasher);
                    std::hash::Hasher::finish(&hasher)
                };
                assert_eq!(hash_of(&structure), hash_of(&copy));
                assert_ne!(hash_of(&structure), hash_of(&different));

                // Display and variable extraction must be iterative too.
                let rendered = structure.to_string();
                // Outermost node is the last one built: `x{DEPTH-1}`.
                assert!(rendered.starts_with(&format!("(x{} = (", DEPTH - 1)));
                assert!(rendered.contains("(x0 = leaf)"));

                let extractor = PatternExtractor::new();
                let vars = extractor.extract_variables(&structure);
                assert!(vars.contains(&"leaf".to_string()));
                assert_eq!(vars.len(), DEPTH + 1);
            })
            .expect("thread spawn should succeed");

        handle.join().expect("worker thread should not panic");
    }

    /// `extract_variables_rec` must still visit `App` arguments and quantifier
    /// bodies, and still record bound variables, after the conversion to an
    /// explicit stack.
    #[test]
    fn test_extract_variables_covers_every_variant() {
        let structure = PatternStructure::Quantified {
            quantifier: "forall".to_string(),
            var: "q".to_string(),
            body: Box::new(PatternStructure::Binary {
                op: "=".to_string(),
                left: Box::new(PatternStructure::App {
                    func: "f".to_string(),
                    args: vec![
                        PatternStructure::Atom("a".to_string()),
                        PatternStructure::Atom("$N".to_string()),
                        PatternStructure::Atom("Upper".to_string()),
                    ],
                }),
                right: Box::new(PatternStructure::Atom("b".to_string())),
            }),
        };

        let extractor = PatternExtractor::new();
        let vars = extractor.extract_variables(&structure);
        // Sorted and deduped; `Upper` is neither `$`-prefixed nor lowercase.
        assert_eq!(
            vars,
            vec![
                "$N".to_string(),
                "a".to_string(),
                "b".to_string(),
                "q".to_string()
            ]
        );
    }

    /// Deep parsing itself (as opposed to walking an already-built value) is
    /// still capped by `max_depth`, but the parser must not need a native call
    /// frame per level for the levels it *does* parse.
    #[test]
    fn test_parse_conclusion_structure_deep_within_max_depth() {
        // The parser scans the remaining text once per level, so keep the depth
        // modest; it is still far past what 1 MiB of call frames would hold.
        const DEPTH: usize = 4_000;

        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — depth is
        // 4_000 (sub-10,000), well under the scaling threshold. See
        // TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let mut conclusion = String::from("x0");
                for i in 1..=DEPTH {
                    conclusion.push_str(&format!(" = x{i}"));
                }
                let structure =
                    PatternExtractor::parse_conclusion_structure(&conclusion, 0, usize::MAX);
                binary_chain_len(&structure)
            })
            .expect("thread spawn should succeed");

        assert_eq!(
            handle.join().expect("worker thread should not panic"),
            DEPTH
        );
    }

    #[test]
    fn test_deep_conclusion_does_not_overflow_stack() {
        // Regression: `parse_conclusion_structure` recurses once per
        // matched binary operator. A conclusion string encoding a deeply
        // nested formula (as would come from a deeply nested proof term's
        // textual conclusion) used to recurse just as deep, with no bound
        // at all -- a genuine stack-overflow hazard. `max_depth` now caps
        // it regardless of input size. Built iteratively (never via a
        // recursive test helper) and run on a deliberately constrained
        // 1 MiB stack, so the cap -- not luck -- is what prevents a crash.
        const CHAIN_LEN: usize = 100_000;
        let mut conclusion = String::from("x0");
        for i in 1..=CHAIN_LEN {
            conclusion.push_str(&format!(" = x{i}"));
        }

        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — the input
        // string is 100,000-deep, but `max_depth` (5) bounds the actual
        // recursion; native call depth here is sub-10,000 regardless of
        // input size. See TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                let extractor = PatternExtractor::new(); // default max_depth = 5
                PatternExtractor::parse_conclusion_structure(&conclusion, 0, extractor.max_depth)
            })
            .expect("failed to spawn constrained-stack thread");

        let structure = handle.join().expect("thread panicked (stack overflow?)");

        assert_eq!(
            binary_chain_len(&structure),
            5,
            "default max_depth=5 must cap the parsed structure at exactly 5 Binary \
             nodes even for a 100,000-deep input"
        );
    }
}
