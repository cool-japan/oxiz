; Test: QF_ABV store-select roundtrip - satisfiable
; Expected: sat
; Storing #x42 at #x00 and then selecting gives #x42

(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(declare-const b (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (store a #x00 #x42) b))
(assert (= (select b #x00) #x42))
(check-sat)
; expected: sat
(exit)
