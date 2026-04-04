; Test: Transitivity through function composition
;; expected: sat
; Pattern: f(a)=b, f(b)=c => f(f(a))=c

(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)

; f(a) = b
(assert (= (f a) b))
; f(b) = c
(assert (= (f b) c))

; Therefore f(f(a)) = c must hold
(assert (= (f (f a)) c))

; Ground constraints
(assert (= a 1))
(assert (= b 2))
(assert (= c 3))

; Additional: f(f(f(a))) should equal f(c)
(assert (= (f (f (f a))) (f c)))

(check-sat)
(exit)
