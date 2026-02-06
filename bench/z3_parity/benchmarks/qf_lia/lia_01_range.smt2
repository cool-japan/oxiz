; Test: Simple range constraints
; Expected: sat
; Pattern: Basic bounds checking

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (>= x 0))
(assert (<= x 10))
(assert (>= y 5))
(assert (<= y 15))
(assert (< x y))

(check-sat)
