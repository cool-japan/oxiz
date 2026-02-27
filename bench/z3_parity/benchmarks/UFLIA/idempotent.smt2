; Test: Idempotency of an uninterpreted function
; Expected: sat
; Pattern: forall x. f(f(x)) = f(x)

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Idempotency axiom
(assert (forall ((x Int)) (= (f (f x)) (f x))))

; Specific values
(assert (= (f 0) 5))
; By idempotency, f(5) must equal 5 (since f(f(0)) = f(0) = 5, so f(5) = 5)
(assert (= (f 5) 5))

; Another fixed point
(assert (= (f 3) 3))

(check-sat)
(exit)
