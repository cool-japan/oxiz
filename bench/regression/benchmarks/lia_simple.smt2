; Simple LIA benchmark for theory solver performance
; Tests basic linear integer arithmetic constraints

(set-logic QF_LIA)

(declare-const x Int)
(declare-const y Int)
(declare-const z Int)

; Basic range constraints
(assert (>= x 0))
(assert (<= x 100))
(assert (>= y 0))
(assert (<= y 100))
(assert (>= z 0))
(assert (<= z 100))

; Linear constraints
(assert (= (+ x y) 50))
(assert (= (+ y z) 75))
(assert (< (+ x z) 80))

; This should be satisfiable
(check-sat)
(get-model)
(exit)
