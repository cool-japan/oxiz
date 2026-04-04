; Test: QF_NIRA unsatisfiable - no integer square root of 3
; Expected: unsat
; x^2 = 3 has no integer solution

(set-logic QF_NIRA)
(declare-const x Int)
(assert (= (* x x) 3))
(check-sat)
; expected: unsat
(exit)
