; Test: Large coefficients
; Expected: sat
; Pattern: Stress test with large numbers

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ (* 1000 x) (* 2000 y)) 50000))
(assert (>= x 0))
(assert (<= x 100))
(assert (>= y 0))
(assert (<= y 100))

(check-sat)
