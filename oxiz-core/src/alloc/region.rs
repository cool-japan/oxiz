//! Region-Based Memory Management.
//!
//! Provides hierarchical memory regions with automatic cleanup on scope exit.

#![allow(unsafe_code, missing_docs)] // Memory management - docs in progress

#[allow(unused_imports)]
use crate::prelude::*;
use core::cell::RefCell;
use core::marker::PhantomData;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// A memory region that can be nested.
pub struct Region {
    allocator: RefCell<RegionAllocator>,
}

impl Region {
    /// Create a new region.
    pub fn new() -> Self {
        Self {
            allocator: RefCell::new(RegionAllocator::new()),
        }
    }

    /// Create a region with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            allocator: RefCell::new(RegionAllocator::with_capacity(capacity)),
        }
    }

    /// Allocate a value in this region.
    pub fn alloc<T>(&self, value: T) -> RegionRef<'_, T> {
        let mut allocator = self.allocator.borrow_mut();
        let ptr = allocator.alloc(value);
        RegionRef {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// Allocate a slice in this region.
    pub fn alloc_slice<T: Clone>(&self, slice: &[T]) -> RegionSlice<'_, T> {
        let mut allocator = self.allocator.borrow_mut();
        let ptr = allocator.alloc_slice(slice);
        RegionSlice {
            ptr,
            len: slice.len(),
            _phantom: PhantomData,
        }
    }

    /// Get total allocated bytes.
    pub fn allocated(&self) -> usize {
        self.allocator.borrow().allocated()
    }

    /// Get number of allocations.
    pub fn num_allocations(&self) -> usize {
        self.allocator.borrow().num_allocations()
    }

    /// Reset the region, deallocating all memory.
    pub fn reset(&mut self) {
        self.allocator.borrow_mut().reset();
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        self.allocator.borrow_mut().reset();
    }
}

/// Internal region allocator.
pub struct RegionAllocator {
    blocks: Vec<Block>,
    current_block_size: usize,
    total_allocated: usize,
    num_allocations: usize,
}

impl RegionAllocator {
    const INITIAL_BLOCK_SIZE: usize = 4096;
    const MAX_BLOCK_SIZE: usize = 1024 * 1024;

    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            current_block_size: Self::INITIAL_BLOCK_SIZE,
            total_allocated: 0,
            num_allocations: 0,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        let mut allocator = Self::new();
        allocator.current_block_size = capacity;
        allocator
    }

    fn alloc<T>(&mut self, value: T) -> NonNull<T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout);

        // SAFETY: `alloc_raw` returned the start of `layout.size()` bytes of a
        // live block reserved for this call alone (`Block::allocate` and
        // `Block::with_first` move `used` past them, so no later allocation
        // overlaps them, and the block is freed only by `reset` or drop), and
        // aligned to `layout.align()`, which is `align_of::<T>()`: from an
        // existing block, `Block::allocate` returns only an address it has
        // rounded up to that alignment with `align_offset` (and `None` when it
        // cannot); a fresh block is the start of an allocation whose layout
        // `Block::with_first` gave at least that alignment. So writing a `T`
        // there is in bounds and aligned, and it overwrites no value.
        unsafe {
            ptr.as_ptr().cast::<T>().write(value);
        }

        self.total_allocated += layout.size();
        self.num_allocations += 1;

        ptr.cast::<T>()
    }

    fn alloc_slice<T: Clone>(&mut self, slice: &[T]) -> NonNull<T> {
        if slice.is_empty() {
            return NonNull::dangling();
        }

        // The layout of the slice value itself: `slice.len()` elements of
        // `T`, which exists because the slice does, so there is no overflow
        // case to handle.
        let layout = Layout::for_value(slice);
        let ptr = self.alloc_raw(layout);

        // SAFETY: as in `alloc`, `ptr` starts `layout.size()` =
        // `slice.len() * size_of::<T>()` bytes reserved for this call alone in
        // a live block, aligned to `layout.align()` = `align_of::<T>()`; every
        // `dest.add(i)` with `i < slice.len()` therefore stays inside them and
        // is aligned, and each element is written once into bytes that held
        // no value.
        unsafe {
            let dest = ptr.as_ptr().cast::<T>();
            for (i, item) in slice.iter().enumerate() {
                dest.add(i).write(item.clone());
            }
        }

        self.total_allocated += layout.size();
        self.num_allocations += 1;

        ptr.cast::<T>()
    }

    fn alloc_raw(&mut self, layout: Layout) -> NonNull<u8> {
        // Try to allocate from existing blocks
        for block in &mut self.blocks {
            if let Some(ptr) = block.allocate(layout) {
                return ptr;
            }
        }

        // Need a new block, which holds this allocation at its start
        let block_size = self.current_block_size.max(layout.size() * 2);
        let (block, ptr) = Block::with_first(block_size, layout).expect("failed to allocate block");

        self.blocks.push(block);

        // Update block size for next allocation
        self.current_block_size = (self.current_block_size * 2).min(Self::MAX_BLOCK_SIZE);

        ptr
    }

    fn allocated(&self) -> usize {
        self.total_allocated
    }

    fn num_allocations(&self) -> usize {
        self.num_allocations
    }

    fn reset(&mut self) {
        self.blocks.clear();
        self.current_block_size = Self::INITIAL_BLOCK_SIZE;
        self.total_allocated = 0;
        self.num_allocations = 0;
    }
}

/// A block of memory in the region.
struct Block {
    ptr: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    used: usize,
}

impl Block {
    /// Allocate a block of `size` bytes (at least one, and at least
    /// `first.size()`) aligned to `first.align()` and to 8, and reserve its
    /// first `first.size()` bytes for `first`, whose start it returns.
    fn with_first(size: usize, first: Layout) -> Option<(Self, NonNull<u8>)> {
        // Never zero: allocating zero bytes is undefined behaviour.
        let size = size.max(first.size()).max(1);
        let layout = Layout::from_size_align(size, first.align().max(8)).ok()?;

        // SAFETY: `layout.size()` is `size`, which the `max(1)` above makes
        // non-zero, the one requirement `alloc` places on its layout.
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr)?;

        // The block's base is aligned to `layout.align()`, which is at least
        // `first.align()`, so `first` fits at offset 0, the start.
        Some((
            Self {
                ptr,
                layout,
                capacity: size,
                used: first.size(),
            },
            ptr,
        ))
    }

    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align();
        let size = layout.size();

        // Align the address, not the offset: `align_offset` gives the padding
        // that makes the address at `used` a multiple of `align` (a power of
        // two, a `Layout`'s alignment), or `usize::MAX` when it cannot, which
        // the checked addition turns into "no room here" and a fresh block.
        // The base is aligned to 8 and to the first allocation's alignment,
        // so for `align <= 8` the padding is the one that rounds `used` itself
        // up to `align`.
        let padding = self
            .ptr
            .as_ptr()
            .wrapping_add(self.used)
            .align_offset(align);
        let aligned_offset = self.used.checked_add(padding)?;
        let new_used = aligned_offset.checked_add(size)?;

        if new_used > self.capacity {
            return None;
        }

        self.used = new_used;

        // SAFETY: `aligned_offset <= new_used <= self.capacity`, the size of
        // the allocation `self.ptr` heads (`with_first` made it with
        // `self.layout`, whose size is `self.capacity`), so the result is in
        // bounds of that allocation or one past its end, as `add` requires.
        Some(unsafe { self.ptr.add(aligned_offset) })
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by `alloc(self.layout)` in
        // `with_first` and both fields are set only there; a `Block` is not
        // `Clone`, so this is the one deallocation of that pointer, with the
        // layout it was allocated with.
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Reference to a value in a region.
pub struct RegionRef<'a, T> {
    ptr: NonNull<T>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> RegionRef<'a, T> {
    pub fn get(&self) -> &T {
        // SAFETY: `self.ptr` is the aligned, initialized `T` that
        // `RegionAllocator::alloc` wrote for this `RegionRef` alone. Its block
        // is freed only by `Region::reset(&mut self)` or by dropping the
        // `Region`, and both need the `Region` unborrowed, while for as long
        // as this value exists the `&'a Region` borrow that
        // `Region::alloc(&'a self)` took is live: the value holds no
        // reference itself, but its `PhantomData<&'a T>` carries `'a`, and
        // `alloc` returns `RegionRef<'a, T>`. Nothing else writes those
        // bytes, so a shared reference for the duration of `&self` is valid.
        unsafe { self.ptr.as_ref() }
    }

    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: as in `get`, the `T` is aligned, initialized and alive for
        // `'a`; a `RegionRef` is neither `Clone` nor `Copy` and is the only
        // handle to those bytes, so `&mut self` makes this reference unique.
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T> core::ops::Deref for RegionRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<'a, T> core::ops::DerefMut for RegionRef<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

/// Reference to a slice in a region.
pub struct RegionSlice<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _phantom: PhantomData<&'a [T]>,
}

impl<'a, T> RegionSlice<'a, T> {
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: for `len == 0`, `ptr` is `NonNull::dangling()`, non-null and
        // aligned as an empty slice needs; otherwise it starts the `len`
        // aligned, initialized elements `RegionAllocator::alloc_slice` wrote
        // (their total size is the size of the source slice, at most
        // `isize::MAX`), alive for `'a` and written by nothing else, as in
        // `RegionRef::get`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and a `RegionSlice` is neither `Clone` nor
        // `Copy`, the only handle to those elements, so `&mut self` makes this
        // slice unique.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> core::ops::Deref for RegionSlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> core::ops::DerefMut for RegionSlice<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_basic() {
        let region = Region::new();
        let val = region.alloc(42);
        assert_eq!(*val, 42);
    }

    #[test]
    fn test_region_multiple() {
        let region = Region::new();
        let v1 = region.alloc(1);
        let v2 = region.alloc(2);
        let v3 = region.alloc(3);

        assert_eq!(*v1, 1);
        assert_eq!(*v2, 2);
        assert_eq!(*v3, 3);
    }

    #[test]
    fn test_region_slice() {
        let region = Region::new();
        let data = vec![1, 2, 3, 4, 5];
        let slice = region.alloc_slice(&data);

        assert_eq!(slice.len(), 5);
        assert_eq!(&slice[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_region_stats() {
        let region = Region::new();
        region.alloc(100);
        region.alloc(200);

        assert_eq!(region.num_allocations(), 2);
        assert!(region.allocated() > 0);
    }

    #[test]
    fn test_region_reset() {
        let mut region = Region::new();
        region.alloc(42);

        let allocated_before = region.allocated();
        assert!(allocated_before > 0);

        region.reset();

        let allocated_after = region.allocated();
        assert_eq!(allocated_after, 0);
        assert_eq!(region.num_allocations(), 0);
    }

    /// For alignments up to 8 the offsets are those of the plain
    /// offset-rounding layout, counted from the first allocation of a block.
    #[test]
    fn test_small_alignment_offsets() {
        let region = Region::new();
        let a = region.alloc(1u8);
        let b = region.alloc(2u32);
        let c = region.alloc(3u16);
        let d = region.alloc(4u64);
        let e = region.alloc([5u8; 3]);
        let f = region.alloc(6u64);
        let start = (a.get() as *const u8).addr();
        let offsets = [
            0,
            (b.get() as *const u32).addr() - start,
            (c.get() as *const u16).addr() - start,
            (d.get() as *const u64).addr() - start,
            (e.get() as *const [u8; 3]).addr() - start,
            (f.get() as *const u64).addr() - start,
        ];
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
            let start = end.next_multiple_of(align);
            end = start + size;
            start
        })
        .collect();
        assert_eq!(offsets.to_vec(), want);
        assert_eq!((*b, *c, *d, *e, *f), (2, 3, 4, [5; 3], 6));
    }

    /// A value aligned more strictly than 8 is placed at an address of its
    /// alignment, at the start of a fresh block and inside a used one.
    #[test]
    fn test_over_aligned_value_is_aligned() {
        #[repr(align(4096))]
        struct Page(u64);

        let region = Region::new();
        region.alloc(1u8);
        let page = region.alloc(Page(7));
        assert_eq!((page.get() as *const Page).addr() % 4096, 0);
        assert_eq!(page.get().0, 7);

        let roomy = Region::with_capacity(1 << 16);
        roomy.alloc(1u8);
        let page = roomy.alloc(Page(9));
        assert_eq!((page.get() as *const Page).addr() % 4096, 0);
        assert_eq!(page.get().0, 9);

        let pages = roomy.alloc_slice(&[0u8, 1, 2]);
        assert_eq!(&pages[..], &[0, 1, 2]);
    }

    /// A zero capacity and zero-sized values never ask the allocator for zero
    /// bytes.
    #[test]
    fn test_zero_sized_requests() {
        let region = Region::with_capacity(0);
        let unit = region.alloc(());
        let () = *unit;
        let units = region.alloc_slice(&[(), ()]);
        assert_eq!(units.len(), 2);
        let byte = region.alloc(5u8);
        assert_eq!(*byte, 5);
    }

    #[test]
    fn test_region_large_allocation() {
        let region = Region::new();
        let large_data: Vec<u64> = (0..1000).collect();
        let slice = region.alloc_slice(&large_data);

        assert_eq!(slice.len(), 1000);
        assert_eq!(slice[0], 0);
        assert_eq!(slice[999], 999);
    }
}
