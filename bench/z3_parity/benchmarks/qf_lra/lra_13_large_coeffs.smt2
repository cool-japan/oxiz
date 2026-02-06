; Test: Large coefficients
; Expected: sat
; Pattern: Numerical range test

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ (* 1000.0 x) (* 2000.0 y)) 50000.0))
(assert (>= x 0.0))
(assert (<= x 100.0))
(assert (>= y 0.0))
(assert (<= y 100.0))

(check-sat)
