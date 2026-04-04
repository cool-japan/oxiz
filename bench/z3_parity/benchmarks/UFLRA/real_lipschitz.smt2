; Test: Lipschitz continuity (linear encoding without nonlinear terms)
; Expected: sat
; Pattern: |f(x) - f(y)| <= K * |x - y| for specific point pairs

(set-logic UFLRA)
(declare-fun f (Real) Real)

; Lipschitz with K=2, encoded for specific pairs without nonlinear multiplication
; For pair (0, 1): |f(0) - f(1)| <= 2 * |0 - 1| = 2
(assert (<= (- (f 1.0) (f 0.0)) 2.0))
(assert (<= (- (f 0.0) (f 1.0)) 2.0))

; For pair (1, 3): |f(1) - f(3)| <= 2 * |1 - 3| = 4
(assert (<= (- (f 3.0) (f 1.0)) 4.0))
(assert (<= (- (f 1.0) (f 3.0)) 4.0))

; For pair (0, 3): |f(0) - f(3)| <= 2 * |0 - 3| = 6
(assert (<= (- (f 3.0) (f 0.0)) 6.0))
(assert (<= (- (f 0.0) (f 3.0)) 6.0))

; General bounded Lipschitz: for all x, y in [0, 5], |f(x) - f(y)| <= 10
(assert (forall ((x Real) (y Real))
  (=> (and (>= x 0.0) (<= x 5.0) (>= y 0.0) (<= y 5.0))
      (and (<= (- (f x) (f y)) 10.0)
           (<= (- (f y) (f x)) 10.0)))))

; Concrete values respecting Lipschitz with K=2
(assert (= (f 0.0) 0.0))
(assert (= (f 1.0) 1.5))
(assert (= (f 3.0) 4.0))

(check-sat)
(exit)
