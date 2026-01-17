; Simple LRA benchmark for real arithmetic solver performance
; Tests basic linear real arithmetic constraints

(set-logic QF_LRA)

(declare-const x Real)
(declare-const y Real)
(declare-const z Real)

; Basic range constraints
(assert (>= x 0.0))
(assert (<= x 10.0))
(assert (>= y 0.0))
(assert (<= y 10.0))
(assert (>= z 0.0))
(assert (<= z 10.0))

; Linear constraints with rationals
(assert (= (+ x y) 5.5))
(assert (= (+ y z) 7.25))
(assert (< (+ x z) 8.0))

; Additional constraints
(assert (> (- x y) (- 1.0)))
(assert (< (- y z) 2.0))

; This should be satisfiable
(check-sat)
(get-model)
(exit)
