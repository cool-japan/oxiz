; Test: Unsatisfiable due to tight bounds
; Expected: unsat
; Pattern: Bounds that make system infeasible

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ x y) 100))
(assert (<= x 40))
(assert (<= y 40))

(check-sat)
