; Test: Simple real arithmetic
; Expected: sat
; Pattern: Basic feasibility

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (>= x 0.0))
(assert (<= x 10.0))
(assert (>= y 0.0))
(assert (= (+ x y) 5.5))

(check-sat)
