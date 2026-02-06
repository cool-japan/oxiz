; Test: Extensionality contradiction
; Expected: unsat
; Pattern: Arrays equal but differ at index

(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))

(assert (= a b))
(assert (not (= (select a 0) (select b 0))))

(check-sat)
