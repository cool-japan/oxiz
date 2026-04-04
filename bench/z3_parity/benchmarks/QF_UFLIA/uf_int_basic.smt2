; Basic UF + integer arithmetic
; Expected: sat
; Tests uninterpreted function with integer constraints

(set-logic QF_UFLIA)
(declare-fun f (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)

; f(x) = 2 * x + 1
(assert (= (f x) (+ (* 2 x) 1)))
(assert (= (f y) (+ (* 2 y) 1)))

; x and y are different
(assert (not (= x y)))

; Values are within bounds
(assert (>= x 0))
(assert (<= x 10))
(assert (>= y 0))
(assert (<= y 10))

(check-sat)
; expected: sat
(exit)
