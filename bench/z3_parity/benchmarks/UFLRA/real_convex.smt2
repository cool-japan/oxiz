; Test: Convexity property
; Expected: sat
; Pattern: forall x y t. 0<=t /\ t<=1 => f(t*x + (1-t)*y) <= t*f(x) + (1-t)*f(y)

(set-logic UFLRA)
(declare-fun f (Real) Real)

; Convexity (tested at specific points rather than fully quantified for decidability)
; We check: f(midpoint) <= average of endpoints
; For x=0, y=2, t=0.5: f(1) <= 0.5*f(0) + 0.5*f(2)
(assert (<= (f 1.0) (+ (* 0.5 (f 0.0)) (* 0.5 (f 2.0)))))

; For x=0, y=4, t=0.5: f(2) <= 0.5*f(0) + 0.5*f(4)
(assert (<= (f 2.0) (+ (* 0.5 (f 0.0)) (* 0.5 (f 4.0)))))

; For x=0, y=4, t=0.25: f(1) <= 0.25*f(0) + 0.75*f(4)
(assert (<= (f 1.0) (+ (* 0.25 (f 0.0)) (* 0.75 (f 4.0)))))

; Concrete values consistent with convexity (f(x) = x^2 would work)
(assert (= (f 0.0) 0.0))
(assert (= (f 1.0) 1.0))
(assert (= (f 2.0) 4.0))
(assert (= (f 4.0) 16.0))

(check-sat)
(exit)
