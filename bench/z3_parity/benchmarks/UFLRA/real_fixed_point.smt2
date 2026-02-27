; Test: Fixed point existence
; Expected: sat
; Pattern: exists x. f(x) = x with bounded range

(set-logic UFLRA)
(declare-fun f (Real) Real)

; f maps [0, 1] to [0, 1]
(assert (forall ((x Real))
  (=> (and (>= x 0.0) (<= x 1.0))
      (and (>= (f x) 0.0) (<= (f x) 1.0)))))

; There exists a fixed point
(assert (exists ((x Real))
  (and (>= x 0.0) (<= x 1.0) (= (f x) x))))

; Specific value: f(0.5) = 0.5 is a fixed point
(assert (= (f 0.5) 0.5))

(check-sat)
(exit)
