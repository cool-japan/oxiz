; Test: Simple multiplication
; Expected: sat
; Pattern: Basic nonlinear constraint

(set-logic QF_NIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (* x y) 12))
(assert (>= x 1))
(assert (<= x 12))
(assert (>= y 1))

(check-sat)
