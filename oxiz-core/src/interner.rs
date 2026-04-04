//! String interning types for sort names and identifiers.
//!
//! In std mode, delegates to `lasso` for efficient thread-safe interning.
//! In no_std mode, provides a simple replacement.

#[cfg(feature = "std")]
pub use lasso::{Key, Rodeo, Spur};

#[cfg(not(feature = "std"))]
mod no_std_interner {
    use crate::prelude::FxHashMap;
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::num::NonZeroU32;

    /// A key type representing an interned string.
    ///
    /// Compatible with `lasso::Spur` API surface.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Spur(NonZeroU32);

    impl Spur {
        /// Try to create a Spur from a usize index.
        pub fn try_from_usize(index: usize) -> Option<Self> {
            NonZeroU32::new((index as u32).wrapping_add(1)).map(Spur)
        }

        /// Get the inner NonZeroU32 value.
        pub fn into_inner(self) -> NonZeroU32 {
            self.0
        }

        fn index(self) -> usize {
            (self.0.get() - 1) as usize
        }
    }

    /// Trait for key types used with the interner.
    pub trait Key: Copy + Eq + core::hash::Hash {
        /// Try to create a key from a usize.
        fn try_from_usize(index: usize) -> Option<Self>;
        /// Convert the key to a usize.
        fn into_usize(self) -> usize;
    }

    impl Key for Spur {
        fn try_from_usize(index: usize) -> Option<Self> {
            Spur::try_from_usize(index)
        }
        fn into_usize(self) -> usize {
            self.index()
        }
    }

    /// A string interner that stores strings and returns compact keys.
    ///
    /// Compatible with `lasso::Rodeo` API surface.
    pub struct Rodeo {
        strings: Vec<String>,
        map: FxHashMap<String, Spur>,
    }

    impl core::fmt::Debug for Rodeo {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Rodeo")
                .field("len", &self.strings.len())
                .finish()
        }
    }

    impl Default for Rodeo {
        fn default() -> Self {
            Self {
                strings: Vec::new(),
                map: FxHashMap::default(),
            }
        }
    }

    impl Rodeo {
        /// Intern a string, returning an existing key if already interned.
        pub fn get_or_intern(&mut self, s: &str) -> Spur {
            if let Some(&spur) = self.map.get(s) {
                return spur;
            }
            let spur = Spur::try_from_usize(self.strings.len()).expect("too many interned strings");
            self.strings.push(String::from(s));
            self.map.insert(String::from(s), spur);
            spur
        }

        /// Look up an already-interned string, returning None if not found.
        pub fn get(&self, s: &str) -> Option<Spur> {
            self.map.get(s).copied()
        }

        /// Resolve a key to its interned string.
        pub fn resolve(&self, spur: &Spur) -> &str {
            &self.strings[spur.index()]
        }
    }
}

#[cfg(not(feature = "std"))]
pub use no_std_interner::{Key, Rodeo, Spur};
