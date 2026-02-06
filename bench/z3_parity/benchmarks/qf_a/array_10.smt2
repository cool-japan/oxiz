; Test: Array values with constraints
; Expected: sat
; Pattern: Store with computed values

(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const x Int)

(assert (= (select a 0) (* 2 x)))
(assert (= (select a 1) (+ x 10)))
(assert (= (select (store a 2 100) 2) 100))
(assert (= x 5))

(check-sat)
