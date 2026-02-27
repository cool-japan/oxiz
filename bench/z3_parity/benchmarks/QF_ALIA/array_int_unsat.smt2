; Unsatisfiable integer array problem
; Expected: unsat
; Conflicting bounds on array elements

(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))

; a[0] + a[1] = 10
(assert (= (+ (select a 0) (select a 1)) 10))

; a[0] > 7
(assert (> (select a 0) 7))

; a[1] > 7
(assert (> (select a 1) 7))

; Impossible: if a[0] > 7 and a[1] > 7, then a[0] + a[1] > 14, not 10

(check-sat)
; expected: unsat
(exit)
