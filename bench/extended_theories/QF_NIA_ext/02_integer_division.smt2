; Test: Properties of integer division and modulus
;; expected: unknown
; Pattern: div/mod relationship: a = (div a b) * b + (mod a b)

(set-logic QF_NIA)

(declare-const a Int)
(declare-const b Int)

; b is a positive divisor
(assert (> b 0))
(assert (<= b 10))

; a is positive
(assert (> a 0))
(assert (<= a 100))

; Verify the fundamental div/mod relationship
(assert (= a (+ (* (div a b) b) (mod a b))))

; mod is non-negative and less than b
(assert (>= (mod a b) 0))
(assert (< (mod a b) b))

; Specific: a divided by b gives quotient 7 with remainder 3
(assert (= (div a b) 7))
(assert (= (mod a b) 3))

; So a = 7*b + 3, b > 3 (since mod < b)
(assert (> b 3))

(check-sat)
(exit)
