//! Arena Allocator for AST Nodes.
//!
//! Provides bump allocation for short-lived AST nodes with batch deallocation.

#![allow(unsafe_code, clippy::non_canonical_clone_impl)]

#[allow(unused_imports)]
use crate::prelude::*;
use core::cell::Cell;
use core::marker::PhantomData;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// Configuration for arena allocator.
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Initial chunk size in bytes
    pub initial_chunk_size: usize,
    /// Maximum chunk size in bytes
    pub max_chunk_size: usize,
    /// Growth factor for chunks
    pub growth_factor: f64,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            initial_chunk_size: 4096,    // 4KB
            max_chunk_size: 1024 * 1024, // 1MB
            growth_factor: 2.0,
        }
    }
}

/// A chunk of memory in the arena.
struct Chunk {
    ptr: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    used: Cell<usize>,
}

impl Chunk {
    /// Allocate a chunk of `size` bytes (at least one) aligned to `align` and
    /// to 8, so that an allocation of that alignment fits at its start.
    fn new(size: usize, align: usize) -> Result<Self, ArenaError> {
        // Never zero: allocating zero bytes is undefined behaviour, and a
        // zero-sized value asks for it whenever the chunk size has become 0
        // (a zero `initial_chunk_size`, or a `growth_factor` that rounds the
        // next size down to 0).
        let size = size.max(1);
        let layout =
            Layout::from_size_align(size, align.max(8)).map_err(|_| ArenaError::LayoutError)?;

        // SAFETY: `layout.size()` is `size`, which the `max(1)` above makes
        // non-zero, the one requirement `alloc` places on its layout.
        let ptr = unsafe { alloc(layout) };
        let non_null_ptr = NonNull::new(ptr).ok_or(ArenaError::AllocationFailed)?;

        Ok(Self {
            ptr: non_null_ptr,
            layout,
            capacity: size,
            used: Cell::new(0),
        })
    }

    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Align the address, not the offset: `align_offset` gives the padding
        // that makes the address at `used` a multiple of `align` (a power of
        // two, a `Layout`'s alignment), or `usize::MAX` when it cannot, which
        // the checked addition turns into "no room here" and a fresh chunk.
        // The base is aligned to 8 and to the alignment the chunk was made
        // for, so for `align <= 8` the padding is the one that rounds `used`
        // itself up to `align`.
        let used = self.used.get();
        let padding = self.ptr.as_ptr().wrapping_add(used).align_offset(align);
        let aligned_offset = used.checked_add(padding)?;
        let new_used = aligned_offset.checked_add(size)?;

        if new_used > self.capacity {
            return None;
        }

        self.used.set(new_used);

        // SAFETY: `aligned_offset <= new_used <= self.capacity`, the size of
        // the allocation `self.ptr` heads (`Chunk::new` made it with
        // `self.layout`, whose size is `self.capacity`), so the result is in
        // bounds of that allocation or one past its end, as `add` requires.
        Some(unsafe { self.ptr.add(aligned_offset) })
    }

    fn reset(&self) {
        self.used.set(0);
    }

    fn used(&self) -> usize {
        self.used.get()
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Arena allocator for fast bump allocation.
pub struct Arena {
    config: ArenaConfig,
    chunks: Vec<Chunk>,
    current_chunk_size: usize,
}

impl Arena {
    /// Create a new arena with default configuration.
    pub fn new() -> Self {
        Self::with_config(ArenaConfig::default())
    }

    /// Create a new arena with custom configuration.
    pub fn with_config(config: ArenaConfig) -> Self {
        Self {
            current_chunk_size: config.initial_chunk_size,
            config,
            chunks: Vec::new(),
        }
    }

    /// Allocate memory in the arena.
    pub fn alloc<T>(&mut self, value: T) -> ArenaHandle<T> {
        let layout = Layout::new::<T>();
        let ptr = self
            .alloc_raw(layout.size(), layout.align())
            .expect("arena allocation failed");

        unsafe {
            ptr.as_ptr().cast::<T>().write(value);
        }

        ArenaHandle {
            ptr: ptr.cast::<T>(),
            _phantom: PhantomData,
        }
    }

    /// Allocate raw memory.
    fn alloc_raw(&mut self, size: usize, align: usize) -> Result<NonNull<u8>, ArenaError> {
        // Try current chunk
        if let Some(chunk) = self.chunks.last()
            && let Some(ptr) = chunk.allocate(size, align)
        {
            return Ok(ptr);
        }

        // Need new chunk
        self.grow(size, align)?;

        self.chunks
            .last()
            .and_then(|chunk| chunk.allocate(size, align))
            .ok_or(ArenaError::AllocationFailed)
    }

    /// Grow the arena by adding a new chunk.
    fn grow(&mut self, min_size: usize, align: usize) -> Result<(), ArenaError> {
        let new_size = self.current_chunk_size.max(min_size);
        let chunk = Chunk::new(new_size, align)?;
        self.chunks.push(chunk);

        // Grow chunk size for next allocation
        let next_size = (self.current_chunk_size as f64 * self.config.growth_factor) as usize;
        self.current_chunk_size = next_size.min(self.config.max_chunk_size);

        Ok(())
    }

    /// Reset the arena, reusing existing chunks.
    pub fn reset(&mut self) {
        for chunk in &self.chunks {
            chunk.reset();
        }
        self.current_chunk_size = self.config.initial_chunk_size;
    }

    /// Clear the arena, deallocating all chunks.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.current_chunk_size = self.config.initial_chunk_size;
    }

    /// Get total capacity across all chunks.
    pub fn capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity).sum()
    }

    /// Get total used memory across all chunks.
    pub fn used(&self) -> usize {
        self.chunks.iter().map(|c| c.used()).sum()
    }

    /// Get number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to an arena-allocated value.
pub struct ArenaHandle<T> {
    ptr: NonNull<T>,
    _phantom: PhantomData<T>,
}

impl<T> ArenaHandle<T> {
    /// Get a reference to the value.
    pub fn get(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }

    /// Get a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Clone for ArenaHandle<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _phantom: PhantomData,
        }
    }
}

impl<T> Copy for ArenaHandle<T> {}

impl<T: core::fmt::Debug> core::fmt::Debug for ArenaHandle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArenaHandle")
            .field("value", self.get())
            .finish()
    }
}

impl<T> core::ops::Deref for ArenaHandle<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

/// Errors that can occur during arena allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// Failed to create memory layout
    LayoutError,
    /// Allocation failed
    AllocationFailed,
}

impl core::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LayoutError => write!(f, "invalid memory layout"),
            Self::AllocationFailed => write!(f, "allocation failed"),
        }
    }
}

impl core::error::Error for ArenaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena = Arena::new();
        let handle = arena.alloc(42);
        assert_eq!(*handle, 42);
    }

    #[test]
    fn test_arena_multiple() {
        let mut arena = Arena::new();
        let h1 = arena.alloc(1);
        let h2 = arena.alloc(2);
        let h3 = arena.alloc(3);

        assert_eq!(*h1, 1);
        assert_eq!(*h2, 2);
        assert_eq!(*h3, 3);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new();
        arena.alloc(100);
        let used_before = arena.used();

        arena.reset();
        let used_after = arena.used();

        assert!(used_before > 0);
        assert_eq!(used_after, 0);
    }

    #[test]
    fn test_arena_growth() {
        let config = ArenaConfig {
            initial_chunk_size: 64,
            max_chunk_size: 1024,
            growth_factor: 2.0,
        };
        let mut arena = Arena::with_config(config);

        // Allocate enough to trigger growth
        for _ in 0..100 {
            arena.alloc(10u64);
        }

        assert!(arena.num_chunks() > 1);
    }

    /// For alignments up to 8 the offsets are those of the plain
    /// offset-rounding layout: `used()` after each allocation.
    #[test]
    fn test_small_alignment_offsets() {
        let mut arena = Arena::new();
        let mut used = Vec::new();
        arena.alloc(1u8);
        used.push(arena.used());
        arena.alloc(2u32);
        used.push(arena.used());
        arena.alloc(3u16);
        used.push(arena.used());
        arena.alloc(4u64);
        used.push(arena.used());
        arena.alloc([5u8; 3]);
        used.push(arena.used());
        arena.alloc(6u64);
        used.push(arena.used());
        // Each value at its predecessor's end rounded up to its alignment.
        let mut end: usize = 0;
        let want: Vec<usize> = [
            (1, 1),
            (4, core::mem::align_of::<u32>()),
            (2, core::mem::align_of::<u16>()),
            (8, core::mem::align_of::<u64>()),
            (3, 1),
            (8, core::mem::align_of::<u64>()),
        ]
        .iter()
        .map(|&(size, align): &(usize, usize)| {
            end = end.next_multiple_of(align) + size;
            end
        })
        .collect();
        assert_eq!(used, want);
    }

    /// A value aligned more strictly than 8 is placed at an address of its
    /// alignment, at the start of a fresh chunk and inside a used one.
    #[test]
    fn test_over_aligned_value_is_aligned() {
        #[repr(align(4096))]
        struct Page(u64);

        let mut arena = Arena::new();
        arena.alloc(1u8);
        let page = arena.alloc(Page(7));
        assert_eq!((page.get() as *const Page).addr() % 4096, 0);
        assert_eq!(page.get().0, 7);

        let mut roomy = Arena::with_config(ArenaConfig {
            initial_chunk_size: 1 << 16,
            max_chunk_size: 1 << 16,
            growth_factor: 2.0,
        });
        roomy.alloc(1u8);
        let page = roomy.alloc(Page(9));
        assert_eq!((page.get() as *const Page).addr() % 4096, 0);
        assert_eq!(page.get().0, 9);
    }

    /// A zero chunk size and a zero-sized value never ask the allocator for
    /// zero bytes.
    #[test]
    fn test_zero_sized_requests() {
        let mut arena = Arena::with_config(ArenaConfig {
            initial_chunk_size: 0,
            max_chunk_size: 0,
            growth_factor: 2.0,
        });
        let unit = arena.alloc(());
        let () = *unit.get();
        let byte = arena.alloc(5u8);
        assert_eq!(*byte.get(), 5);
    }
}
