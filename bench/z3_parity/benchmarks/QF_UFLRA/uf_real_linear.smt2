; Linear functions with UF over reals
; Expected: sat
; Tests combining UF with linear real arithmetic constraints

(set-logic QF_UFLRA)
(declare-fun f (Real) Real)
(declare-fun g (Real) Real)
(declare-fun x () Real)

; f(x) + g(x) = 1.0
(assert (= (+ (f x) (g x)) 1.0))

; f(x) >= 0.0 and g(x) >= 0.0
(assert (>= (f x) 0.0))
(assert (>= (g x) 0.0))

; f(x) > g(x)
(assert (> (f x) (g x)))

; x = 0.5
(assert (= x 0.5))

(check-sat)
; expected: sat
(exit)
