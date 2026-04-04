; Test: QF_NIRA integer product satisfiable
; Expected: sat
; x * y = 12 with x > 0 and y > 0 has solutions (e.g. x=3, y=4)

(set-logic QF_NIRA)
(declare-const x Int)
(declare-const y Int)
(assert (= (* x y) 12))
(assert (> x 0))
(assert (> y 0))
(check-sat)
; expected: sat
(exit)
