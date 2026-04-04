; Test: Mixed real/integer-like reasoning with quantifiers
; Expected: sat
; Pattern: Quantified constraints mixing real and integer-like values

(set-logic UFLRA)
(declare-fun f (Real) Real)
(declare-fun g (Real) Real)

; f is non-negative for non-negative inputs
(assert (forall ((x Real))
  (=> (>= x 0.0) (>= (f x) 0.0))))

; g is an upper bound on f
(assert (forall ((x Real))
  (=> (and (>= x 0.0) (<= x 10.0))
      (<= (f x) (g x)))))

; g is affine: g(x) = 2x + 1
(assert (forall ((x Real))
  (=> (and (>= x 0.0) (<= x 10.0))
      (= (g x) (+ (* 2.0 x) 1.0)))))

; Concrete values
(assert (= (f 0.0) 0.5))
(assert (= (f 5.0) 8.0))     ; f(5) = 8 <= g(5) = 11, OK
(assert (= (g 0.0) 1.0))
(assert (= (g 5.0) 11.0))

(check-sat)
(exit)
