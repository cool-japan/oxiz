; Test: Modular arithmetic constraints
;; expected: sat
; Pattern: x mod 3 = 1, y mod 5 = 2, with bounds

(set-logic QF_NIA)

(declare-const x Int)
(declare-const y Int)

; x mod 3 = 1
(assert (= (mod x 3) 1))

; y mod 5 = 2
(assert (= (mod y 5) 2))

; Both positive and bounded
(assert (> x 0))
(assert (> y 0))
(assert (<= x 20))
(assert (<= y 20))

; Additional: x + y > 10
(assert (> (+ x y) 10))

; e.g., x=4 (4 mod 3 = 1), y=7 (7 mod 5 = 2), sum=11 > 10
(check-sat)
(exit)
