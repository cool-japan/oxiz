; Basic integer array operations
; Expected: sat
; Tests store/select with integer indices and values

(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))
(declare-fun x () Int)

; Store x at index 0, read it back
(assert (= (select (store a 0 x) 0) x))

; x is positive
(assert (> x 0))
(assert (< x 100))

(check-sat)
; expected: sat
(exit)
