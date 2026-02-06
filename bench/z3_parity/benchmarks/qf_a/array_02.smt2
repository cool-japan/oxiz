; Test: Store chain operations
; Expected: sat
; Pattern: Multiple sequential stores

(set-logic QF_ALIA)
(declare-const a (Array Int Int))

(assert (= (select (store (store a 0 10) 1 20) 0) 10))
(assert (= (select (store (store a 0 10) 1 20) 1) 20))

(check-sat)
