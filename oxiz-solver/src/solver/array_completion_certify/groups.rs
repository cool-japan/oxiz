//! Arrays under **independent** binders are completed per group (`#P2b-51`,
//! decision (54)(ii)).
//!
//! The constant search is a product over one default pool per array, which
//! is why `MAX_COMPLETED_ARRAYS` bounds it.  But the product is only needed
//! over arrays some assertion mentions together: four arrays each read under
//! its own `forall` (recheck 13, `d03`) are four independent one-array
//! searches, and declining the goal because it holds four arrays kept a
//! candidate model that printed all four as `#b0` beside `∀i. a[i] = #b1`.
//!
//! So the arrays are partitioned by co-occurrence — two arrays are in one
//! group when one assertion mentions both (an array equality `a = b` links
//! them like a shared universal does) — the bound applies to the size of a
//! group, and each group is searched on its own sub-goal (the assertions that
//! mention it).  The searches only *propose*: the union of their completions
//! goes through the one certificate over every assertion before anything is
//! installed, and that certificate is the soundness argument, exactly as for
//! a single search.

use oxiz_core::ast::{TermId, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};

use super::Goal;

/// The arrays of `arrays` partitioned by co-occurrence in one of
/// `assertions`, each group in `arrays`' order and the groups ordered by
/// their first member.
pub(super) fn array_groups(
    arrays: &[TermId],
    assertions: &[TermId],
    manager: &TermManager,
) -> Vec<Vec<TermId>> {
    let position: FxHashMap<TermId, usize> = arrays
        .iter()
        .enumerate()
        .map(|(index, &array)| (array, index))
        .collect();
    let mut parent: Vec<usize> = (0..arrays.len()).collect();
    for &assertion in assertions {
        let mut members: Vec<usize> = manager
            .free_vars_including_patterns(assertion)
            .into_iter()
            .filter_map(|var| position.get(&var).copied())
            .collect();
        members.sort_unstable();
        if let Some((&first, rest)) = members.split_first() {
            for &other in rest {
                union(&mut parent, first, other);
            }
        }
    }
    let mut groups: Vec<Vec<TermId>> = Vec::new();
    let mut group_of_root: FxHashMap<usize, usize> = FxHashMap::default();
    for (index, &array) in arrays.iter().enumerate() {
        let root = find(&mut parent, index);
        let slot = *group_of_root.entry(root).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        if let Some(group) = groups.get_mut(slot) {
            group.push(array);
        }
    }
    groups
}

/// The size of the largest group [`array_groups`] forms.
pub(super) fn largest_group(
    arrays: &[TermId],
    assertions: &[TermId],
    manager: &TermManager,
) -> usize {
    array_groups(arrays, assertions, manager)
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while let Some(&up) = parent.get(node) {
        if up == node {
            break;
        }
        // Path halving.
        let grand = parent.get(up).copied().unwrap_or(up);
        if let Some(slot) = parent.get_mut(node) {
            *slot = grand;
        }
        node = grand;
    }
    node
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        // The smaller index is the root, so the grouping is a property of the
        // array order and not of the union order.
        let (root, child) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        if let Some(slot) = parent.get_mut(child) {
            *slot = root;
        }
    }
}

impl Goal {
    /// The sub-goal of one group: the assertions that mention one of its
    /// arrays, their quantifiers, and the group as the arrays to complete.
    /// `None` when the sub-goal's quantifiers cannot be read.
    pub(super) fn restricted_to(
        &self,
        group: &[TermId],
        manager: &mut TermManager,
    ) -> Option<Goal> {
        let members: FxHashSet<TermId> = group.iter().copied().collect();
        let assertions: Vec<TermId> = self
            .assertions
            .iter()
            .copied()
            .filter(|&assertion| {
                manager
                    .free_vars_including_patterns(assertion)
                    .into_iter()
                    .any(|var| members.contains(&var))
            })
            .collect();
        let quantifiers = super::polarity::collect_quantifiers(&assertions, manager)?;
        Some(Goal {
            assertions,
            universals: quantifiers.universals,
            arrays: group.to_vec(),
            scalar_pins: self.scalar_pins.clone(),
            existentials: quantifiers.existentials,
            ground_values: self.ground_values.clone(),
            negations: quantifiers.negations,
            hypotheses: self.hypotheses.clone(),
        })
    }
}
