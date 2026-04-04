; Basic array read/write with bitvector indices and values
; Expected: sat
; Tests read-over-write axiom with BV32 arrays

(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun i () (_ BitVec 32))
(declare-fun v () (_ BitVec 32))

; Store v at index i, then read back at the same index
(assert (= (select (store a i v) i) v))

; v must be nonzero
(assert (not (= v (_ bv0 32))))

; i is a specific index
(assert (= i (_ bv42 32)))

(check-sat)
; expected: sat
(exit)
