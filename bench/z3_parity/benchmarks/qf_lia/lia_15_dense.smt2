; Test: Dense constraint matrix
; Expected: sat
; Pattern: Each constraint involves all variables

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

(assert (>= (+ x y z) 10))
(assert (<= (+ x y z) 30))
(assert (>= (+ (* 2 x) y (* -1 z)) 5))
(assert (<= (+ x (* -1 y) (* 2 z)) 20))
(assert (>= x 0))
(assert (>= y 0))
(assert (>= z 0))

(check-sat)
