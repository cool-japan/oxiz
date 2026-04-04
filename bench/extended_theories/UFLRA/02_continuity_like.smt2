; Test: Continuity-like axioms with UF
;; expected: sat
; Pattern: |f(x) - f(y)| <= K * |x - y| (Lipschitz-like condition)

(set-logic UFLRA)
(declare-fun f (Real) Real)
(declare-const K Real)

; K is a positive constant
(assert (= K 2.0))

; Lipschitz condition on bounded domain (ground instances)
(declare-const x1 Real)
(declare-const x2 Real)
(declare-const x3 Real)

(assert (= x1 0.0))
(assert (= x2 1.0))
(assert (= x3 0.5))

; |f(x1) - f(x2)| <= K * |x1 - x2|
(assert (<= (- (f x1) (f x2)) (* K (- x2 x1))))
(assert (>= (- (f x1) (f x2)) (* (- 0.0 K) (- x2 x1))))

; |f(x2) - f(x3)| <= K * |x2 - x3|
(assert (<= (- (f x2) (f x3)) (* K (- x2 x3))))
(assert (>= (- (f x2) (f x3)) (* (- 0.0 K) (- x2 x3))))

; |f(x1) - f(x3)| <= K * |x3 - x1|
(assert (<= (- (f x3) (f x1)) (* K (- x3 x1))))
(assert (>= (- (f x3) (f x1)) (* (- 0.0 K) (- x3 x1))))

; Concrete assignment
(assert (= (f 0.0) 0.0))
(assert (= (f 1.0) 1.5))
(assert (= (f 0.5) 0.8))

(check-sat)
(exit)
