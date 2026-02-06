; Test: Array extensionality
; Expected: sat
; Pattern: Arrays are equal if all elements are equal

(set-logic QF_ALIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(declare-const i Int)

(assert (= (select a i) (select b i)))
(assert (= a b))

(check-sat)
