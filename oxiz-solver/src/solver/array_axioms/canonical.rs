//! Canonical structural keys for the array refinement's scheduling choices
//! (decision (43)).
//!
//! # Why the refinement needs a key of its own
//!
//! Every choice the lazy array refinement makes — which equality pair a lazy
//! family is built for first, which array constant meets which pair, the
//! order in which a round's lemmas reach the SAT core (and therefore which
//! SAT variables they are given) — used to follow either the *collection
//! pre-order* of the term-graph walk (which follows the order the assertions
//! were written in) or the raw [`TermId`] (which follows the order the terms
//! were interned in, i.e. the parse order again).  `#P2b-59`'s own repro with
//! its two assertions swapped (`round4_pass10_recheck_pins`,
//! `…_swapped…`) is the same formula and took a different trajectory through
//! the same lemma set: 66 rounds and a verdict in one order, no verdict in the
//! other.
//!
//! A [`StructuralKey`] is a function of a term's *structure* only: its
//! operator, its sort (by structure, not by interned [`SortId`]), the names
//! and literal values at its leaves and the keys of its children.  Two
//! scripts that spell the same terms in a different order give every term the
//! same key, so an order by key is independent of assertion position, of
//! term-id allocation order and of hash iteration order.
//!
//! # What is canonicalised beyond the plain structure
//!
//! * **Commutative operators** (`=`, `distinct`, `and`, `or`, `xor`, `+`,
//!   `*`, `bvand`, `bvor`, `bvxor`, `bvadd`, `bvmul`) hash their children's
//!   keys as a *multiset*.  `mk_eq` orders its operands by raw id, so without
//!   this `(= a b)` would hash differently depending on which of `a`, `b`
//!   was interned first.
//! * **Minted names.**  An extensionality witness `\oxiz.ext!{lo}!{hi}` and
//!   an off-chain index `\oxiz.off!{lo}!{hi}` are named after the raw ids of
//!   their array pair; they are keyed by the (unordered) keys of that pair
//!   instead.  An `ite`-elimination proxy is keyed by the `ite` it names
//!   (`Solver::ite_elim_aliases`) rather than by its counter.
//!
//! # Collisions
//!
//! The key is a 128-bit hash, so two distinct terms can in principle share
//! one.  Every consumer breaks a tie by raw id, which makes a collision cost
//! at most a scheduling difference between two spellings — never a lemma,
//! never a verdict: every instance the refinement asserts is a theorem of the
//! array theory whatever order it is asserted in.  Operators outside the
//! array / bit-vector / arithmetic / Boolean fragment that carry a payload the
//! walk does not read (a datatype selector's name, a `match` case) hash that
//! payload through its `Debug` form, which names no term id.

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::SortKind;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;

/// A term's canonical structural key; see the module documentation.
pub(super) type StructuralKey = u128;

/// Memoised [`StructuralKey`]s for one refinement round.
///
/// Built per round rather than cached on the solver: the terms a round keys
/// are the ones it collected, a few hundred on the scripts this was measured
/// on, and a per-round table needs no journal across `push`/`pop`.
pub(super) struct StructuralKeys {
    /// `ite`-elimination proxy → the `ite` term it names.
    aliases: FxHashMap<TermId, TermId>,
    terms: FxHashMap<TermId, StructuralKey>,
    sorts: FxHashMap<SortId, StructuralKey>,
}

/// Two independent 64-bit lanes of a SplitMix-style absorb.  Deterministic
/// (no per-process seed), so the order it induces is the same on every run
/// and every machine.
#[derive(Clone, Copy)]
struct Mix {
    lo: u64,
    hi: u64,
}

impl Mix {
    fn new(tag: u64) -> Self {
        let mut mix = Self {
            lo: 0x9e37_79b9_7f4a_7c15,
            hi: 0xc2b2_ae3d_27d4_eb4f,
        };
        mix.word(tag);
        mix
    }

    fn scramble(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn word(&mut self, value: u64) {
        self.lo = Self::scramble(self.lo ^ value).wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.hi = Self::scramble(self.hi.rotate_left(17) ^ value ^ 0x1656_67b1_9e37_79f9)
            .wrapping_add(0x27d4_eb2f_1656_67c5);
    }

    fn key(&mut self, key: StructuralKey) {
        self.word(key as u64);
        self.word((key >> 64) as u64);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.word(bytes.len() as u64);
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.word(u64::from_le_bytes(buf));
        }
    }

    fn finish(self) -> StructuralKey {
        (u128::from(self.hi) << 64) | u128::from(self.lo)
    }
}

/// Operators whose operand order carries no meaning.
fn is_commutative(kind: &TermKind) -> bool {
    matches!(
        kind,
        TermKind::Eq(..)
            | TermKind::Distinct(_)
            | TermKind::And(_)
            | TermKind::Or(_)
            | TermKind::Xor(..)
            | TermKind::Add(_)
            | TermKind::Mul(_)
            | TermKind::BvAnd(..)
            | TermKind::BvOr(..)
            | TermKind::BvXor(..)
            | TermKind::BvAdd(..)
            | TermKind::BvMul(..)
    )
}

/// The array pair a minted `\oxiz.ext!` / `\oxiz.off!` name was built from.
fn minted_pair(name: &str, prefix: &str) -> Option<(TermId, TermId)> {
    let rest = name.strip_prefix(prefix)?;
    let (lo, hi) = rest.split_once('!')?;
    Some((
        TermId::new(lo.parse::<u32>().ok()?),
        TermId::new(hi.parse::<u32>().ok()?),
    ))
}

impl StructuralKeys {
    /// A fresh table; `aliases` is `Solver::ite_elim_aliases`.
    pub(super) fn new(aliases: &FxHashMap<TermId, TermId>) -> Self {
        Self {
            aliases: aliases.clone(),
            terms: FxHashMap::default(),
            sorts: FxHashMap::default(),
        }
    }

    /// The key of `sort`, by structure.
    fn sort_key(&mut self, sort: SortId, manager: &TermManager) -> StructuralKey {
        if let Some(&key) = self.sorts.get(&sort) {
            return key;
        }
        let Some(kind) = manager.sorts.get(sort).map(|s| s.kind.clone()) else {
            let mut mix = Mix::new(0x50);
            mix.word(u64::from(sort.0));
            return mix.finish();
        };
        let key = match kind {
            SortKind::Bool => Mix::new(0x51).finish(),
            SortKind::Int => Mix::new(0x52).finish(),
            SortKind::Real => Mix::new(0x53).finish(),
            SortKind::String => Mix::new(0x54).finish(),
            SortKind::RoundingMode => Mix::new(0x55).finish(),
            SortKind::BitVec(width) => {
                let mut mix = Mix::new(0x56);
                mix.word(u64::from(width));
                mix.finish()
            }
            SortKind::FloatingPoint { eb, sb } => {
                let mut mix = Mix::new(0x57);
                mix.word(u64::from(eb));
                mix.word(u64::from(sb));
                mix.finish()
            }
            SortKind::Array { domain, range } => {
                let domain = self.sort_key(domain, manager);
                let range = self.sort_key(range, manager);
                let mut mix = Mix::new(0x58);
                mix.key(domain);
                mix.key(range);
                mix.finish()
            }
            SortKind::Uninterpreted(name) => {
                let mut mix = Mix::new(0x59);
                mix.bytes(manager.resolve_str(name).as_bytes());
                mix.finish()
            }
            SortKind::Parameter(name) => {
                let mut mix = Mix::new(0x5a);
                mix.bytes(manager.resolve_str(name).as_bytes());
                mix.finish()
            }
            SortKind::Parametric { name, args } => {
                let mut mix = Mix::new(0x5b);
                mix.bytes(manager.resolve_str(name).as_bytes());
                for arg in args {
                    let arg = self.sort_key(arg, manager);
                    mix.key(arg);
                }
                mix.finish()
            }
            SortKind::Datatype(name) => {
                let mut mix = Mix::new(0x5c);
                mix.bytes(manager.resolve_str(name).as_bytes());
                mix.finish()
            }
        };
        self.sorts.insert(sort, key);
        key
    }

    /// The canonical key of `term`.
    ///
    /// Iterative (an explicit post-order stack), so a deep term costs heap,
    /// not native stack, exactly like the structure walk it serves.
    pub(super) fn key(&mut self, term: TermId, manager: &TermManager) -> StructuralKey {
        if let Some(&key) = self.terms.get(&term) {
            return key;
        }
        let mut stack: Vec<(TermId, bool)> = vec![(term, false)];
        let mut children: Vec<TermId> = Vec::new();
        while let Some((current, expanded)) = stack.pop() {
            if self.terms.contains_key(&current) {
                continue;
            }
            let Some(data) = manager.get(current) else {
                let mut mix = Mix::new(0x01);
                mix.word(u64::from(current.raw()));
                self.terms.insert(current, mix.finish());
                continue;
            };
            children.clear();
            self.dependencies(current, &data.kind, manager, &mut children);
            if !expanded {
                stack.push((current, true));
                for &child in children.iter().rev() {
                    if !self.terms.contains_key(&child) {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            let key = self.combine(current, manager, &children);
            self.terms.insert(current, key);
        }
        self.terms.get(&term).copied().unwrap_or_default()
    }

    /// The terms `term`'s key is computed from: its structural children, or
    /// — for a minted name — the terms the name stands for.
    fn dependencies(
        &self,
        term: TermId,
        kind: &TermKind,
        manager: &TermManager,
        out: &mut Vec<TermId>,
    ) {
        if let TermKind::Var(name) = kind {
            let name = manager.resolve_str(*name);
            for prefix in [
                oxiz_core::smtlib::ARRAY_EXT_WITNESS_PREFIX,
                oxiz_core::smtlib::ARRAY_OFF_CHAIN_PREFIX,
            ] {
                if let Some((a, b)) = minted_pair(name, prefix) {
                    out.push(a);
                    out.push(b);
                    return;
                }
            }
            if let Some(&ite) = self.aliases.get(&term)
                && ite != term
            {
                out.push(ite);
            }
            return;
        }
        super::super::term_walk::collect_structural_children(kind, out);
    }

    /// The key of `term`, every dependency already keyed.
    fn combine(
        &mut self,
        term: TermId,
        manager: &TermManager,
        dependencies: &[TermId],
    ) -> StructuralKey {
        let Some(data) = manager.get(term) else {
            return 0;
        };
        let kind = data.kind.clone();
        let sort = self.sort_key(data.sort, manager);
        let mut child_keys: Vec<StructuralKey> = dependencies
            .iter()
            .map(|child| self.terms.get(child).copied().unwrap_or_default())
            .collect();
        let mut mix = Mix::new(0x10);
        mix.key(sort);
        match &kind {
            TermKind::Var(name) => {
                let name = manager.resolve_str(*name);
                if name.starts_with(oxiz_core::smtlib::ARRAY_EXT_WITNESS_PREFIX)
                    || name.starts_with(oxiz_core::smtlib::ARRAY_OFF_CHAIN_PREFIX)
                {
                    // A minted pair index: the prefix plus the pair, unordered.
                    let prefix = if name.starts_with(oxiz_core::smtlib::ARRAY_EXT_WITNESS_PREFIX) {
                        0x21
                    } else {
                        0x22
                    };
                    mix.word(prefix);
                    child_keys.sort_unstable();
                    for key in child_keys {
                        mix.key(key);
                    }
                } else if !child_keys.is_empty() {
                    // An `ite`-elimination proxy: the `ite` it names.
                    mix.word(0x23);
                    for key in child_keys {
                        mix.key(key);
                    }
                } else {
                    mix.word(0x24);
                    mix.bytes(name.as_bytes());
                }
                return mix.finish();
            }
            TermKind::True => mix.word(0x30),
            TermKind::False => mix.word(0x31),
            TermKind::IntConst(value) => {
                mix.word(0x32);
                mix.bytes(value.to_string().as_bytes());
            }
            TermKind::RealConst(value) => {
                mix.word(0x33);
                mix.bytes(format!("{value}").as_bytes());
            }
            TermKind::BitVecConst { value, width } => {
                mix.word(0x34);
                mix.word(u64::from(*width));
                mix.bytes(value.to_string().as_bytes());
            }
            TermKind::StringLit(text) => {
                mix.word(0x35);
                mix.bytes(text.as_bytes());
            }
            TermKind::Apply { func, .. } => {
                mix.word(0x36);
                mix.bytes(manager.resolve_str(*func).as_bytes());
            }
            TermKind::BvExtract { high, low, .. } => {
                mix.word(0x37);
                mix.word(u64::from(*high));
                mix.word(u64::from(*low));
            }
            TermKind::DtConstructor { constructor, .. }
            | TermKind::DtTester { constructor, .. } => {
                mix.word(0x38);
                mix.bytes(manager.resolve_str(*constructor).as_bytes());
            }
            TermKind::DtSelector { selector, .. } => {
                mix.word(0x39);
                mix.bytes(manager.resolve_str(*selector).as_bytes());
            }
            TermKind::Forall { vars, .. } | TermKind::Exists { vars, .. } => {
                mix.word(if matches!(kind, TermKind::Forall { .. }) {
                    0x3a
                } else {
                    0x3b
                });
                for &(name, var_sort) in vars.iter() {
                    mix.bytes(manager.resolve_str(name).as_bytes());
                    let var_sort = self.sort_key(var_sort, manager);
                    mix.key(var_sort);
                }
            }
            TermKind::Let { bindings, .. } => {
                mix.word(0x3c);
                for &(name, _) in bindings.iter() {
                    mix.bytes(manager.resolve_str(name).as_bytes());
                }
            }
            _ => {
                // Discriminant, plus any payload that is not a child: the
                // `Debug` form of an operator with no term-valued payload
                // other than its children names no term id once the children
                // are elided, so hash the variant name alone and rely on the
                // sort and the children for the rest.
                let debug = format!("{kind:?}");
                let variant = debug
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .next()
                    .unwrap_or_default();
                mix.word(0x3f);
                mix.bytes(variant.as_bytes());
                for rounding in ["RNE", "RNA", "RTP", "RTN", "RTZ"] {
                    if debug.contains(rounding) {
                        mix.bytes(rounding.as_bytes());
                    }
                }
            }
        }
        if is_commutative(&kind) {
            child_keys.sort_unstable();
        }
        mix.word(child_keys.len() as u64);
        for key in child_keys {
            mix.key(key);
        }
        mix.finish()
    }

    /// `(key, raw id)`: the canonical order of `term`, with the raw id as the
    /// collision tie-break the module documentation describes.
    pub(super) fn order(&mut self, term: TermId, manager: &TermManager) -> (StructuralKey, u32) {
        (self.key(term, manager), term.raw())
    }

    /// The canonical order of an unordered pair: its two keys, smaller first.
    pub(super) fn pair_order(
        &mut self,
        pair: (TermId, TermId),
        manager: &TermManager,
    ) -> ((StructuralKey, u32), (StructuralKey, u32)) {
        let a = self.order(pair.0, manager);
        let b = self.order(pair.1, manager);
        if a <= b { (a, b) } else { (b, a) }
    }
}

impl super::ArrayStructure {
    /// Sort every list a family is built from by canonical key (decision
    /// (43)), and orient every equality pair smaller-key first.
    ///
    /// Collection itself stays in the term-graph walk's pre-order (its unit
    /// tests pin that order); the scheduling reads the lists only after this.
    pub(super) fn canonicalize(&mut self, keys: &mut StructuralKeys, manager: &TermManager) {
        self.selects
            .sort_by_cached_key(|&(select, _, _)| keys.order(select, manager));
        for pair in &mut self.eq_pairs {
            if keys.order(pair.1, manager) < keys.order(pair.0, manager) {
                *pair = (pair.1, pair.0);
            }
        }
        self.eq_pairs
            .sort_by_cached_key(|&pair| keys.pair_order(pair, manager));
        self.eq_atoms
            .sort_by_cached_key(|&(atom, _, _)| keys.order(atom, manager));
        self.distinct_atoms
            .sort_by_cached_key(|&atom| keys.order(atom, manager));
        self.stores
            .sort_by_cached_key(|&(store, _, _)| keys.order(store, manager));
        self.const_arrays
            .sort_by_cached_key(|&constant| keys.order(constant, manager));
        self.foreign
            .sort_by_cached_key(|&term| keys.order(term, manager));
        self.array_ites
            .sort_by_cached_key(|&(ite, _, _, _)| keys.order(ite, manager));
        for indices in self.read_indices.values_mut() {
            indices.sort_by_cached_key(|&index| keys.order(index, manager));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key does not depend on the order in which the two spellings of
    /// one formula interned their terms.
    #[test]
    fn two_interning_orders_give_every_term_the_same_key() {
        let aliases = FxHashMap::default();
        let build = |swap: bool| {
            let mut tm = TermManager::new();
            let bv8 = tm.sorts.bitvec(8);
            let bv1 = tm.sorts.bitvec(1);
            let arr = tm.sorts.array(bv8, bv1);
            let (a0, a1) = if swap {
                let a1 = tm.mk_var("a1", arr);
                let a0 = tm.mk_var("a0", arr);
                (a0, a1)
            } else {
                let a0 = tm.mk_var("a0", arr);
                let a1 = tm.mk_var("a1", arr);
                (a0, a1)
            };
            let eq = tm.mk_eq(a0, a1);
            let idx = tm.mk_bitvec(3u32, 8);
            let read = tm.mk_select(a1, idx);
            let mut keys = StructuralKeys::new(&aliases);
            (keys.key(eq, &tm), keys.key(read, &tm), keys.key(a0, &tm))
        };
        assert_eq!(build(false), build(true));
    }

    /// Distinct structures get distinct keys (on these, at least), and the
    /// operand order of a non-commutative operator is part of the structure.
    #[test]
    fn distinct_structures_get_distinct_keys() {
        let aliases = FxHashMap::default();
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let lt_xy = tm.mk_bv_ult(x, y);
        let lt_yx = tm.mk_bv_ult(y, x);
        let mut keys = StructuralKeys::new(&aliases);
        assert_ne!(keys.key(x, &tm), keys.key(y, &tm));
        assert_ne!(keys.key(lt_xy, &tm), keys.key(lt_yx, &tm));
    }
}
