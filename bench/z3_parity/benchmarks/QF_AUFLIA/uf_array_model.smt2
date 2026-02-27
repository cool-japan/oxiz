; Satisfiable problem requiring combined model from UF + Array + LIA
; Expected: sat
; Tests that a satisfying model can be constructed across all three theories

(set-logic QF_AUFLIA)
(declare-fun a () (Array Int Int))
(declare-fun f (Int) Int)
(declare-fun g (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)

; f and g are different functions at x
(assert (not (= (f x) (g x))))

; Array stores using f(x) and g(x) as indices
(define-fun b () (Array Int Int) (store (store a (f x) 100) (g x) 200))

; Read back both values
(assert (= (select b (f x)) 100))
(assert (= (select b (g x)) 200))

; Arithmetic constraints
(assert (> x 0))
(assert (< x 10))
(assert (> (f x) 10))
(assert (> (g x) 20))

(check-sat)
; expected: sat
(exit)
