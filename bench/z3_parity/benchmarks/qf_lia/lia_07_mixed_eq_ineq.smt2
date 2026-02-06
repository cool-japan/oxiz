; Test: Mixed equalities and inequalities
; Expected: sat
; Pattern: Combined constraint types

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

(assert (= (+ x y) 20))
(assert (>= (+ y z) 15))
(assert (<= (+ x z) 25))
(assert (>= x 5))
(assert (>= y 5))
(assert (>= z 5))

(check-sat)
