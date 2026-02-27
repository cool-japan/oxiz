; Test: Homogeneity-like property for even arguments
; Expected: sat
; Pattern: forall x. f(2*x) = 2*f(x) with specific values (bounded)

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Doubling property for bounded inputs
(assert (forall ((x Int))
  (=> (and (>= x 0) (<= x 5))
      (= (f (* 2 x)) (* 2 (f x))))))

; f(0) = 0 (forced by axiom: f(0) = f(2*0) = 2*f(0), so f(0) = 0)
(assert (= (f 0) 0))

; Set f(1) = 3
(assert (= (f 1) 3))
; Then f(2) = 2*f(1) = 6
(assert (= (f 2) 6))
; And f(4) = 2*f(2) = 12
(assert (= (f 4) 12))

(check-sat)
(exit)
