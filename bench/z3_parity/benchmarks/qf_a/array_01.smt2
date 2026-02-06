; Test: Basic store/select operations
; Expected: sat
; Pattern: Read-over-write axiom

(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))

(assert (= b (store a 0 10)))
(assert (= (select b 0) 10))
(assert (= (select b 1) (select a 1)))

(check-sat)
