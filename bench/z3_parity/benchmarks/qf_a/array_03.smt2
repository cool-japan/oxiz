; Test: Read-over-write with different index
; Expected: unsat
; Pattern: Contradictory store/select

(set-logic QF_ALIA)
(declare-const a (Array Int Int))

(assert (= (select a 0) 5))
(assert (= (select (store a 0 10) 0) 5))

(check-sat)
