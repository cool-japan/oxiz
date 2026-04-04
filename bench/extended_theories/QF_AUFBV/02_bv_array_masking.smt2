; Test: Bitvector array with masking operations
;; expected: sat
; Pattern: Store masked values, verify bitwise properties

(set-logic QF_AUFBV)
(declare-const buf (Array (_ BitVec 4) (_ BitVec 8)))

; Store values with known bit patterns
(declare-const buf1 (Array (_ BitVec 4) (_ BitVec 8)))
(assert (= buf1 (store buf #x0 #xFF)))

(declare-const buf2 (Array (_ BitVec 4) (_ BitVec 8)))
(assert (= buf2 (store buf1 #x1 #x0F)))

(declare-const buf3 (Array (_ BitVec 4) (_ BitVec 8)))
(assert (= buf3 (store buf2 #x2 #xF0)))

; AND of values at index 0 and 1 should be 0x0F
(assert (= (bvand (select buf3 #x0) (select buf3 #x1)) #x0F))

; OR of values at index 1 and 2 should be 0xFF
(assert (= (bvor (select buf3 #x1) (select buf3 #x2)) #xFF))

; XOR of values at index 0 and 2 should be 0x0F
(assert (= (bvxor (select buf3 #x0) (select buf3 #x2)) #x0F))

(check-sat)
(exit)
