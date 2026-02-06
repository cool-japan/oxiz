; Test: Multi-dimensional store
; Expected: unsat
; Pattern: Nested array with contradiction

(set-logic QF_ALIA)
(declare-const matrix (Array Int (Array Int Int)))

(assert (= (select (select matrix 0) 0) 10))
(assert (= (select (select (store matrix 0 (store (select matrix 0) 0 20)) 0) 0) 10))

(check-sat)
