; Test: Many variables
; Expected: sat
; Pattern: Scaling test

(set-logic QF_LRA)
(declare-const x1 Real)
(declare-const x2 Real)
(declare-const x3 Real)
(declare-const x4 Real)
(declare-const x5 Real)
(declare-const x6 Real)

(assert (= (+ x1 x2 x3 x4 x5 x6) 60.0))
(assert (>= x1 0.0))
(assert (>= x2 0.0))
(assert (>= x3 0.0))
(assert (>= x4 0.0))
(assert (>= x5 0.0))
(assert (>= x6 0.0))
(assert (<= x1 20.0))
(assert (<= x2 20.0))
(assert (<= x3 20.0))
(assert (<= x4 20.0))
(assert (<= x5 20.0))
(assert (<= x6 20.0))

(check-sat)
