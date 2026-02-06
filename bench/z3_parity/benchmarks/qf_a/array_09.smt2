; Test: Array with arithmetic constraints
; Expected: sat
; Pattern: Combine array theory with LIA

(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const x Int)
(declare-const y Int)

(assert (= (select a 0) x))
(assert (= (select a 1) y))
(assert (= (+ x y) 100))
(assert (>= x 0))
(assert (>= y 0))

(check-sat)
