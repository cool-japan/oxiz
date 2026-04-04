; Test: Mixed UF equality and real arithmetic
;; expected: sat
; Pattern: UF congruence with real arithmetic side conditions

(set-logic UFLRA)
(declare-fun f (Real) Real)
(declare-fun g (Real) Real)
(declare-const a Real)
(declare-const b Real)

; a and b are equal
(assert (= a b))

; By congruence: f(a) = f(b) and g(a) = g(b)
; We test this implicitly
(assert (= a 3.14))
(assert (>= b 3.0))
(assert (<= b 4.0))

; Arithmetic relation involving f and g
(assert (= (+ (f a) (g a)) 10.0))
(assert (= (f a) 4.0))
(assert (= (g b) 6.0))

; Since a = b, f(a) = f(b) = 4.0 and g(a) = g(b) = 6.0
; 4.0 + 6.0 = 10.0 -- consistent
(assert (= (+ (f b) (g b)) 10.0))

(check-sat)
(exit)
