; Test: Array extensionality
; Expected: sat
; Pattern: forall i. a[i] = b[i] implies a = b

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))

; Pointwise equality
(assert (forall ((i Int)) (= (select a i) (select b i))))

; By extensionality, a = b
(assert (= a b))

; Specific element values
(assert (= (select a 0) 10))
(assert (= (select a 1) 20))
(assert (= (select b 0) 10))
(assert (= (select b 1) 20))

(check-sat)
(exit)
