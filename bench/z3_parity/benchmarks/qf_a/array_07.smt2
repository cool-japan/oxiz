; Test: Two-dimensional array (nested)
; Expected: sat
; Pattern: Array of arrays

(set-logic QF_ALIA)
(declare-const matrix (Array Int (Array Int Int)))
(declare-const row0 (Array Int Int))

(assert (= row0 (select matrix 0)))
(assert (= (select row0 0) 42))
(assert (= (select (select matrix 0) 0) 42))

(check-sat)
