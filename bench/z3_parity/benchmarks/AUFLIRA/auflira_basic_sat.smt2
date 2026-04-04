; Test: Basic AUFLIRA satisfiable formula
; Expected: sat
; Arrays + Uninterpreted Functions + Linear Integer/Real Arithmetic

(set-logic AUFLIRA)
(declare-sort U 0)
(declare-fun f (U) Int)
(declare-const a U)
(declare-const b U)
(declare-fun arr (Int) Real)
(assert (= (f a) 5))
(assert (> (arr 0) 1.5))
(assert (not (= a b)))
(check-sat)
; expected: sat
(exit)
