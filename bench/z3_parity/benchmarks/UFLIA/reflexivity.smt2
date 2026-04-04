; Test: Trivial reflexivity
; Expected: sat
; Pattern: forall x. f(x) = f(x) -- trivially true

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Reflexivity of equality applied to function outputs
(assert (forall ((x Int)) (= (f x) (f x))))

; Some concrete values
(assert (= (f 0) 7))
(assert (= (f 1) 13))
(assert (> (f 2) 0))

(check-sat)
(exit)
