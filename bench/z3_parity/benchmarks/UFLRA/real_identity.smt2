; Test: Identity function
; Expected: sat
; Pattern: forall x. f(x) = x

(set-logic UFLRA)
(declare-fun f (Real) Real)

; f is the identity function
(assert (forall ((x Real)) (= (f x) x)))

; Verify with specific values
(assert (= (f 3.14) 3.14))
(assert (= (f (- 2.5)) (- 2.5)))
(assert (= (f 0.0) 0.0))

(check-sat)
(exit)
