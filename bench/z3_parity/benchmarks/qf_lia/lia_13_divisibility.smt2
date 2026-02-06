; Test: Divisibility constraints via multiples
; Expected: sat
; Pattern: x must be divisible by 5

(set-logic QF_LIA)
(declare-const x Int)
(declare-const k Int)

(assert (= x (* 5 k)))
(assert (>= x 10))
(assert (<= x 30))

(check-sat)
