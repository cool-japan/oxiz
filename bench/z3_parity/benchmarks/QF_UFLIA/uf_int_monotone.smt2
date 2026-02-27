; Monotone function with arithmetic
; Expected: sat
; Tests UF with monotonicity constraints over integers

(set-logic QF_UFLIA)
(declare-fun f (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)
(declare-fun z () Int)

; f is monotone on the relevant domain
(assert (=> (< x y) (< (f x) (f y))))
(assert (=> (< y z) (< (f y) (f z))))

; x < y < z
(assert (< x y))
(assert (< y z))

; Bounds
(assert (= x 1))
(assert (= z 5))

; f values are bounded
(assert (> (f x) 0))
(assert (< (f z) 100))

(check-sat)
; expected: sat
(exit)
