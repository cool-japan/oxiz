; Test: Many variables
; Expected: sat
; Pattern: Scaling test with multiple variables

(set-logic QF_LIA)
(declare-const x1 Int)
(declare-const x2 Int)
(declare-const x3 Int)
(declare-const x4 Int)
(declare-const x5 Int)

(assert (= (+ x1 x2 x3 x4 x5) 50))
(assert (>= x1 0))
(assert (>= x2 0))
(assert (>= x3 0))
(assert (>= x4 0))
(assert (>= x5 0))
(assert (<= x1 20))
(assert (<= x2 20))
(assert (<= x3 20))
(assert (<= x4 20))
(assert (<= x5 20))

(check-sat)
