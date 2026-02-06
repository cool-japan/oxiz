; Test: Different arrays
; Expected: sat
; Pattern: Arrays differ at specific index

(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))

(assert (not (= a b)))
(assert (= (select a 0) 10))
(assert (= (select b 0) 20))

(check-sat)
