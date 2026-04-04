; Test: AUFLIRA mixed integer/real with UF - unsatisfiable
; Expected: unsat
; h(1) > 2.5 and h(1) < 2.5 is a contradiction

(set-logic AUFLIRA)
(declare-fun h (Int) Real)
(assert (> (h 1) 2.5))
(assert (< (h 1) 2.5))
(check-sat)
; expected: unsat
(exit)
