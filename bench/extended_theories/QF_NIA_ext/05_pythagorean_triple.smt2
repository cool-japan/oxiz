; Test: Pythagorean triple
;; expected: unknown
; Pattern: x^2 + y^2 = z^2 with positive integers

(set-logic QF_NIA)

(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

; Pythagorean relation
(assert (= (+ (* x x) (* y y)) (* z z)))

; All positive
(assert (> x 0))
(assert (> y 0))
(assert (> z 0))

; Bound to find small triples: (3,4,5), (5,12,13), (6,8,10), etc.
(assert (<= x 20))
(assert (<= y 20))
(assert (<= z 20))

; Require x <= y for canonicality
(assert (<= x y))

(check-sat)
(exit)
