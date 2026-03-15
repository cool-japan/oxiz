//! Prelude for no_std / std compatibility.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
pub(crate) use alloc::{
    boxed::Box,
    collections::{BTreeMap, BinaryHeap, VecDeque},
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
pub(crate) use portable_atomic_util::Arc;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
pub(crate) use hashbrown::{HashMap, HashSet, hash_map};

#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque, hash_map};

#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) use std::sync::Arc;

// FxHash types: use rustc_hash in std mode, hashbrown+FxHasher in no_std
#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) use rustc_hash::{FxHashMap, FxHashSet};

#[cfg(not(feature = "std"))]
pub(crate) type FxHashMap<K, V> =
    hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
#[cfg(not(feature = "std"))]
pub(crate) type FxHashSet<K> =
    hashbrown::HashSet<K, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
