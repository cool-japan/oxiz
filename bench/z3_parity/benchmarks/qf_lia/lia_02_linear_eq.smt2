; Test: Linear equations
; Expected: sat
; Pattern: System of linear equations

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

(assert (= (+ x y) 10))
(assert (= (+ y z) 15))
(assert (= (+ x z) 13))

(check-sat)
