; Test: Bitvector array overwrite semantics
;; expected: sat
; Pattern: Multiple writes to same index, last write wins

(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 4) (_ BitVec 16)))

; Write 0x1111 at index 0
(declare-const arr1 (Array (_ BitVec 4) (_ BitVec 16)))
(assert (= arr1 (store arr #x0 #x1111)))

; Overwrite with 0x2222 at same index
(declare-const arr2 (Array (_ BitVec 4) (_ BitVec 16)))
(assert (= arr2 (store arr1 #x0 #x2222)))

; Overwrite again with 0x3333
(declare-const arr3 (Array (_ BitVec 4) (_ BitVec 16)))
(assert (= arr3 (store arr2 #x0 #x3333)))

; Only the last write should be visible
(assert (= (select arr3 #x0) #x3333))

; Write to different index should not affect index 0
(declare-const arr4 (Array (_ BitVec 4) (_ BitVec 16)))
(assert (= arr4 (store arr3 #x1 #xFFFF)))
(assert (= (select arr4 #x0) #x3333))
(assert (= (select arr4 #x1) #xFFFF))

(check-sat)
(exit)
