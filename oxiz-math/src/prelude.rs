//! Prelude for no_std / std compatibility.

#[cfg(not(feature = "std"))]
pub(crate) use alloc::{
    boxed::Box,
    collections::BinaryHeap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
pub(crate) use hashbrown::{HashMap, HashSet};

#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) use std::collections::{BinaryHeap, HashMap, HashSet};

#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) use rustc_hash::{FxHashMap, FxHashSet};

#[cfg(not(feature = "std"))]
pub(crate) type FxHashMap<K, V> =
    hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
#[cfg(not(feature = "std"))]
pub(crate) type FxHashSet<K> =
    hashbrown::HashSet<K, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
