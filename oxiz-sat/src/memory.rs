//! Memory management and clause compaction
//!
//! This module provides efficient memory management for clauses,
//! including arena allocation, compaction, and cache-friendly layouts.

#![allow(unsafe_code)]

use crate::literal::Lit;

/// Clause reference in the memory arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClauseRef(u32);

impl ClauseRef {
    /// Create a null reference
    pub const fn null() -> Self {
        Self(u32::MAX)
    }

    /// Check if this is a null reference
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

/// Clause header stored in the arena
#[repr(C, align(8))]
struct ClauseHeader {
    /// Length of the clause
    len: u32,
    /// Activity score
    activity: f32,
    /// LBD (Literal Block Distance)
    lbd: u32,
    /// Flags (deleted, learned, etc.)
    flags: u32,
}

const FLAG_DELETED: u32 = 1 << 0;
const FLAG_LEARNED: u32 = 1 << 1;

impl ClauseHeader {
    fn new(len: u32, learned: bool) -> Self {
        Self {
            len,
            activity: 0.0,
            lbd: len,
            flags: if learned { FLAG_LEARNED } else { 0 },
        }
    }

    fn is_deleted(&self) -> bool {
        (self.flags & FLAG_DELETED) != 0
    }

    fn mark_deleted(&mut self) {
        self.flags |= FLAG_DELETED;
    }

    #[allow(dead_code)]
    fn is_learned(&self) -> bool {
        (self.flags & FLAG_LEARNED) != 0
    }
}

/// Memory arena for clause storage
pub struct ClauseArena {
    /// Raw memory buffer
    buffer: Vec<u8>,
    /// Current write position
    pos: usize,
    /// Number of clauses
    num_clauses: usize,
    /// Number of deleted clauses
    num_deleted: usize,
    /// Total wasted space from deleted clauses
    wasted_bytes: usize,
}

impl ClauseArena {
    /// Create a new clause arena with initial capacity
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(initial_capacity),
            pos: 0,
            num_clauses: 0,
            num_deleted: 0,
            wasted_bytes: 0,
        }
    }

    /// Allocate a clause in the arena
    pub fn alloc(&mut self, lits: &[Lit], learned: bool) -> ClauseRef {
        let header = ClauseHeader::new(lits.len() as u32, learned);
        let header_size = std::mem::size_of::<ClauseHeader>();
        let lits_size = std::mem::size_of_val(lits);
        let total_size = header_size + lits_size;

        // Ensure alignment
        let align_offset = (8 - (self.pos % 8)) % 8;
        let aligned_pos = self.pos + align_offset;

        // Grow buffer if needed
        while aligned_pos + total_size > self.buffer.capacity() {
            let new_capacity = if self.buffer.capacity() == 0 {
                4096
            } else {
                self.buffer.capacity() * 2
            };
            self.buffer.reserve(new_capacity - self.buffer.capacity());
        }

        // Write header
        let clause_ref = ClauseRef(aligned_pos as u32);
        unsafe {
            let header_ptr = self.buffer.as_mut_ptr().add(aligned_pos) as *mut ClauseHeader;
            std::ptr::write(header_ptr, header);

            // Write literals
            let lits_ptr = header_ptr.add(1) as *mut Lit;
            std::ptr::copy_nonoverlapping(lits.as_ptr(), lits_ptr, lits.len());
        }

        self.pos = aligned_pos + total_size;
        if self.pos > self.buffer.len() {
            unsafe {
                self.buffer.set_len(self.pos);
            }
        }
        self.num_clauses += 1;

        clause_ref
    }

    /// Get a clause by reference
    pub fn get(&self, cref: ClauseRef) -> Option<&[Lit]> {
        if cref.is_null() || cref.0 as usize >= self.buffer.len() {
            return None;
        }

        unsafe {
            let header_ptr = self.buffer.as_ptr().add(cref.0 as usize) as *const ClauseHeader;
            let header = &*header_ptr;

            if header.is_deleted() {
                return None;
            }

            let lits_ptr = header_ptr.add(1) as *const Lit;
            Some(std::slice::from_raw_parts(lits_ptr, header.len as usize))
        }
    }

    /// Get clause activity
    pub fn get_activity(&self, cref: ClauseRef) -> Option<f32> {
        if cref.is_null() || cref.0 as usize >= self.buffer.len() {
            return None;
        }

        unsafe {
            let header_ptr = self.buffer.as_ptr().add(cref.0 as usize) as *const ClauseHeader;
            Some((*header_ptr).activity)
        }
    }

    /// Set clause activity
    pub fn set_activity(&mut self, cref: ClauseRef, activity: f32) {
        if cref.is_null() || cref.0 as usize >= self.buffer.len() {
            return;
        }

        unsafe {
            let header_ptr = self.buffer.as_mut_ptr().add(cref.0 as usize) as *mut ClauseHeader;
            (*header_ptr).activity = activity;
        }
    }

    /// Get clause LBD
    pub fn get_lbd(&self, cref: ClauseRef) -> Option<u32> {
        if cref.is_null() || cref.0 as usize >= self.buffer.len() {
            return None;
        }

        unsafe {
            let header_ptr = self.buffer.as_ptr().add(cref.0 as usize) as *const ClauseHeader;
            Some((*header_ptr).lbd)
        }
    }

    /// Set clause LBD
    pub fn set_lbd(&mut self, cref: ClauseRef, lbd: u32) {
        if cref.is_null() || cref.0 as usize >= self.buffer.len() {
            return;
        }

        unsafe {
            let header_ptr = self.buffer.as_mut_ptr().add(cref.0 as usize) as *mut ClauseHeader;
            (*header_ptr).lbd = lbd;
        }
    }

    /// Delete a clause
    pub fn delete(&mut self, cref: ClauseRef) {
        if cref.is_null() || cref.0 as usize >= self.buffer.len() {
            return;
        }

        unsafe {
            let header_ptr = self.buffer.as_mut_ptr().add(cref.0 as usize) as *mut ClauseHeader;
            let header = &mut *header_ptr;

            if !header.is_deleted() {
                header.mark_deleted();
                self.num_deleted += 1;

                let header_size = std::mem::size_of::<ClauseHeader>();
                let lits_size = header.len as usize * std::mem::size_of::<Lit>();
                self.wasted_bytes += header_size + lits_size;
            }
        }
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        if self.num_clauses == 0 {
            return false;
        }
        let waste_ratio = self.wasted_bytes as f64 / self.buffer.len() as f64;
        waste_ratio > 0.3 || self.num_deleted > self.num_clauses / 2
    }

    /// Compact the arena by removing deleted clauses
    /// Returns a mapping from old refs to new refs
    pub fn compact(&mut self) -> std::collections::HashMap<ClauseRef, ClauseRef> {
        let mut mapping = std::collections::HashMap::new();
        let mut new_buffer: Vec<u8> = Vec::with_capacity(self.buffer.len() - self.wasted_bytes);
        let mut new_pos = 0;

        let mut offset = 0;
        while offset < self.pos {
            // Align to 8 bytes
            let align_offset = (8 - (offset % 8)) % 8;
            offset += align_offset;

            if offset >= self.pos {
                break;
            }

            unsafe {
                let header_ptr = self.buffer.as_ptr().add(offset) as *const ClauseHeader;
                let header = &*header_ptr;

                let header_size = std::mem::size_of::<ClauseHeader>();
                let lits_size = header.len as usize * std::mem::size_of::<Lit>();
                let clause_size = header_size + lits_size;

                if !header.is_deleted() {
                    // Copy clause to new buffer
                    let old_ref = ClauseRef(offset as u32);
                    let new_align = (8 - (new_pos % 8)) % 8;
                    new_pos += new_align;

                    let new_ref = ClauseRef(new_pos as u32);
                    mapping.insert(old_ref, new_ref);

                    // Ensure capacity
                    while new_pos + clause_size > new_buffer.capacity() {
                        new_buffer.reserve(4096);
                    }

                    // Copy data
                    let src = self.buffer.as_ptr().add(offset);
                    let dst = new_buffer.as_mut_ptr().add(new_pos);
                    std::ptr::copy_nonoverlapping(src, dst, clause_size);

                    new_pos += clause_size;
                }

                offset += clause_size;
            }
        }

        unsafe {
            new_buffer.set_len(new_pos);
        }

        self.buffer = new_buffer;
        self.pos = new_pos;
        self.num_clauses -= self.num_deleted;
        self.num_deleted = 0;
        self.wasted_bytes = 0;

        mapping
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_bytes: self.buffer.capacity(),
            used_bytes: self.pos,
            wasted_bytes: self.wasted_bytes,
            num_clauses: self.num_clauses,
            num_deleted: self.num_deleted,
        }
    }
}

impl Drop for ClauseArena {
    fn drop(&mut self) {
        // Vec will handle deallocation
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total allocated bytes
    pub total_bytes: usize,
    /// Bytes currently in use
    pub used_bytes: usize,
    /// Bytes wasted by deleted clauses
    pub wasted_bytes: usize,
    /// Number of active clauses
    pub num_clauses: usize,
    /// Number of deleted clauses
    pub num_deleted: usize,
}

impl MemoryStats {
    /// Get memory efficiency (used / total)
    pub fn efficiency(&self) -> f64 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        (self.used_bytes - self.wasted_bytes) as f64 / self.total_bytes as f64
    }

    /// Get waste ratio
    pub fn waste_ratio(&self) -> f64 {
        if self.used_bytes == 0 {
            return 0.0;
        }
        self.wasted_bytes as f64 / self.used_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let mut arena = ClauseArena::new(1024);

        let lits = vec![
            Lit::pos(crate::literal::Var(0)),
            Lit::pos(crate::literal::Var(1)),
        ];
        let cref = arena.alloc(&lits, false);

        assert!(!cref.is_null());
        assert_eq!(arena.get(cref), Some(&lits[..]));
    }

    #[test]
    fn test_arena_delete() {
        let mut arena = ClauseArena::new(1024);

        let lits = vec![
            Lit::pos(crate::literal::Var(0)),
            Lit::pos(crate::literal::Var(1)),
        ];
        let cref = arena.alloc(&lits, false);

        arena.delete(cref);
        assert_eq!(arena.get(cref), None);
    }

    #[test]
    fn test_arena_activity() {
        let mut arena = ClauseArena::new(1024);

        let lits = vec![
            Lit::pos(crate::literal::Var(0)),
            Lit::pos(crate::literal::Var(1)),
        ];
        let cref = arena.alloc(&lits, false);

        arena.set_activity(cref, 1.5);
        assert_eq!(arena.get_activity(cref), Some(1.5));
    }

    #[test]
    fn test_arena_compact() {
        let mut arena = ClauseArena::new(1024);

        let lits1 = vec![Lit::pos(crate::literal::Var(0))];
        let lits2 = vec![Lit::pos(crate::literal::Var(1))];
        let lits3 = vec![Lit::pos(crate::literal::Var(2))];

        let cref1 = arena.alloc(&lits1, false);
        let cref2 = arena.alloc(&lits2, false);
        let cref3 = arena.alloc(&lits3, false);

        arena.delete(cref2);

        let mapping = arena.compact();

        // cref1 and cref3 should be remapped
        let new_cref1 = mapping.get(&cref1).copied().unwrap_or(cref1);
        let new_cref3 = mapping.get(&cref3).copied().unwrap_or(cref3);

        assert_eq!(arena.get(new_cref1), Some(&lits1[..]));
        assert_eq!(arena.get(new_cref3), Some(&lits3[..]));
        assert_eq!(arena.num_deleted, 0);
    }

    #[test]
    fn test_memory_stats() {
        let arena = ClauseArena::new(1024);
        let stats = arena.stats();

        assert_eq!(stats.num_clauses, 0);
        assert!(stats.efficiency() >= 0.0);
        assert_eq!(stats.waste_ratio(), 0.0);
    }
}
