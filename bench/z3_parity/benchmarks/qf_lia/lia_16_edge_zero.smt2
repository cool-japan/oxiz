; Test: Edge case with zero
; Expected: sat
; Pattern: Variables can be zero

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ x y) 0))
(assert (>= x -10))
(assert (<= x 10))
(assert (>= y -10))
(assert (<= y 10))

(check-sat)
