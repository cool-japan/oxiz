; Store with function application indices
; Expected: sat
; Tests store at f(x) and select at f(y) with appropriate constraints

(set-logic QF_AUFLIA)
(declare-fun a () (Array Int Int))
(declare-fun f (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)

; Store 42 at index f(x)
(define-fun b () (Array Int Int) (store a (f x) 42))

; f(x) = f(y) because x = y
(assert (= x y))

; So reading b at f(y) should give 42
(assert (= (select b (f y)) 42))

; Additional constraint to make interesting
(assert (> (f x) 0))

(check-sat)
; expected: sat
(exit)
