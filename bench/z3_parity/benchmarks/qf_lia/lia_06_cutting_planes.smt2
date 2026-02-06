; Test: Cutting planes scenario
; Expected: unsat
; Pattern: Fractional relaxation has solution, but no integer solution

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ (* 2 x) (* 2 y)) 7))
(assert (>= x 0))
(assert (>= y 0))

(check-sat)
