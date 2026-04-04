; Test: Integer array swap operation
;; expected: sat
; Pattern: Swap two elements and verify

(set-logic QF_ALIA)
(declare-const a (Array Int Int))

; Initial values
(assert (= (select a 0) 10))
(assert (= (select a 1) 20))
(assert (= (select a 2) 30))

; Swap a[0] and a[2]
(declare-const tmp Int)
(assert (= tmp (select a 0)))

(declare-const a1 (Array Int Int))
(assert (= a1 (store a 0 (select a 2))))

(declare-const a2 (Array Int Int))
(assert (= a2 (store a1 2 tmp)))

; After swap: a2[0] = 30, a2[1] = 20, a2[2] = 10
(assert (= (select a2 0) 30))
(assert (= (select a2 1) 20))
(assert (= (select a2 2) 10))

(check-sat)
(exit)
