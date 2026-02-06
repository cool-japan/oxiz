; Test: Branch and bound scenario
; Expected: sat
; Pattern: Integer optimization-like constraints

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

(assert (= (+ (* 3 x) (* 2 y) z) 100))
(assert (>= x 0))
(assert (<= x 20))
(assert (>= y 0))
(assert (<= y 30))
(assert (>= z 0))
(assert (<= z 50))

(check-sat)
