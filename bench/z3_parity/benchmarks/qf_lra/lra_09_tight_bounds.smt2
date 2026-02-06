; Test: Tight bounds still feasible
; Expected: sat
; Pattern: Solution at boundary

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ x y) 10.0))
(assert (>= x 0.0))
(assert (<= x 10.0))
(assert (>= y 0.0))
(assert (<= y 10.0))

(check-sat)
