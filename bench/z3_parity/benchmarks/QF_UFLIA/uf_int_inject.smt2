; Injectivity with integers
; Expected: sat
; Tests injective UF with integer domain constraints

(set-logic QF_UFLIA)
(declare-fun f (Int) Int)
(declare-fun a () Int)
(declare-fun b () Int)
(declare-fun c () Int)

; f is injective
(assert (=> (= (f a) (f b)) (= a b)))
(assert (=> (= (f a) (f c)) (= a c)))
(assert (=> (= (f b) (f c)) (= b c)))

; a, b, c are distinct
(assert (not (= a b)))
(assert (not (= a c)))
(assert (not (= b c)))

; f(a), f(b), f(c) must also be distinct (follows from injectivity)
; Additional arithmetic: all values in [0, 10]
(assert (>= a 0))
(assert (<= a 10))
(assert (>= b 0))
(assert (<= b 10))
(assert (>= c 0))
(assert (<= c 10))

; f maps into [100, 200]
(assert (>= (f a) 100))
(assert (<= (f a) 200))
(assert (>= (f b) 100))
(assert (<= (f b) 200))
(assert (>= (f c) 100))
(assert (<= (f c) 200))

(check-sat)
; expected: sat
(exit)
