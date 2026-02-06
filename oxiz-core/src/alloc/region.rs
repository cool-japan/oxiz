//! Region-Based Memory Management.
//!
//! Provides hierarchical memory regions with automatic cleanup on scope exit.

#![allow(unsafe_code, missing_docs)] // Memory management - docs in progress

use std::alloc::{Layout, alloc, dealloc};
use std::cell::RefCell;
use std::marker::PhantomData;
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

        let layout = Layout::array::<T>(slice.len()).expect("layout overflow");
        let ptr = self.alloc_raw(layout);

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

        // Need a new block
        let block_size = self.current_block_size.max(layout.size() * 2);
        let block = Block::new(block_size).expect("failed to allocate block");

        self.blocks.push(block);

        // Update block size for next allocation
        self.current_block_size = (self.current_block_size * 2).min(Self::MAX_BLOCK_SIZE);

        // Allocate from new block
        self.blocks
            .last_mut()
            .and_then(|block| block.allocate(layout))
            .expect("allocation failed")
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
    fn new(size: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, 8).ok()?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return None;
        }

        Some(Self {
            ptr: NonNull::new(ptr)?,
            layout,
            capacity: size,
            used: 0,
        })
    }

    fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align();
        let size = layout.size();

        let aligned_offset = align_up(self.used, align);
        let new_used = aligned_offset.checked_add(size)?;

        if new_used > self.capacity {
            return None;
        }

        self.used = new_used;

        unsafe {
            Some(NonNull::new_unchecked(
                self.ptr.as_ptr().add(aligned_offset),
            ))
        }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
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
        unsafe { self.ptr.as_ref() }
    }

    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T> std::ops::Deref for RegionRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<'a, T> std::ops::DerefMut for RegionRef<'a, T> {
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
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> std::ops::Deref for RegionSlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> std::ops::DerefMut for RegionSlice<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
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
