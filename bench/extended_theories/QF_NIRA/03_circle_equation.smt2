; Test: Circle equation with integer lattice points
;; expected: sat
; Pattern: x^2 + y^2 <= r^2 with r = 5

(set-logic QF_NIRA)

(declare-const x Int)
(declare-const y Int)
(declare-const r Real)

; Radius is 5.0
(assert (= r 5.0))

; Point (x, y) inside circle of radius r
(assert (<= (to_real (+ (* x x) (* y y))) (* r r)))

; Require non-trivial point
(assert (not (= x 0)))
(assert (not (= y 0)))

; Both positive quadrant
(assert (> x 0))
(assert (> y 0))

; Require reasonably large coordinates
(assert (>= x 3))
(assert (>= y 3))

; (3,3): 9+9=18 <= 25, (3,4): 9+16=25 <= 25, (4,3): same
(check-sat)
(exit)
