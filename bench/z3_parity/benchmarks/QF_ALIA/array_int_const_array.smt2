; Constant array pattern
; Expected: unsat
; Tests that a constant array has the same value everywhere

(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))

; All accessible elements equal 5
(assert (= (select a 0) 5))
(assert (= (select a 1) 5))
(assert (= (select a 2) 5))

; Store at index 3 preserves the value at index 0
(define-fun b () (Array Int Int) (store a 3 5))
(assert (= (select b 0) 5))

; But claim b at index 3 is not 5 (contradiction with store axiom)
(assert (not (= (select b 3) 5)))

(check-sat)
; expected: unsat
(exit)
