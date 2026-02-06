; Test: Infeasible due to bounds
; Expected: unsat
; Pattern: Over-constrained system

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ x y) 100.0))
(assert (<= x 40.0))
(assert (<= y 40.0))

(check-sat)
