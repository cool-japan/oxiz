; Array extensionality with bitvector indices/values
; Expected: unsat
; If two arrays differ at some index, they cannot be equal

(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun b () (Array (_ BitVec 32) (_ BitVec 32)))

; Assert arrays are equal
(assert (= a b))

; But their values at index 7 differ
(assert (not (= (select a (_ bv7 32)) (select b (_ bv7 32)))))

(check-sat)
; expected: unsat
(exit)
