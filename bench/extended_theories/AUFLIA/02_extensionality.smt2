; Test: Array extensionality
;; expected: sat
; Pattern: (forall i. a[i] = b[i]) => a = b

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))

; Pointwise equality implies structural equality
(assert (forall ((i Int)) (= (select a i) (select b i))))
(assert (= a b))

; Both arrays have specific values
(assert (= (select a 0) 10))
(assert (= (select a 1) 20))
(assert (= (select a 2) 30))

; b must agree (since a = b)
(assert (= (select b 0) 10))
(assert (= (select b 1) 20))
(assert (= (select b 2) 30))

(check-sat)
(exit)
