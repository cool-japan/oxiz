; Test: QF_ABV bitvector range contradiction - unsatisfiable
; Expected: unsat
; x < #xf and x > #xf cannot both hold (strict inequality both ways)

(set-logic QF_ABV)
(declare-const x (_ BitVec 4))
(assert (bvult x #xf))
(assert (bvugt x #xf))
(check-sat)
; expected: unsat
(exit)
