; Test: Basic nonlinear integer/real system
;; expected: sat
; Pattern: x*y = 6, x + y = 5 => (x=2,y=3) or (x=3,y=2)

(set-logic QF_NIRA)

(declare-const x Int)
(declare-const y Int)

(assert (= (* x y) 6))
(assert (= (+ x y) 5))

; Both positive
(assert (> x 0))
(assert (> y 0))

(check-sat)
(exit)
