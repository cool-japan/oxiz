; Word-level reasoning: Subtraction (16-bit)
; Expected: UNSAT
; Tests contradictory subtraction constraints

(set-logic QF_BV)
(declare-const x (_ BitVec 16))
(declare-const y (_ BitVec 16))

; x - y = 100
; y - x = 100
; These two constraints are contradictory
(assert (= (bvsub x y) #x0064))
(assert (= (bvsub y x) #x0064))

; Should be UNSAT: cannot have both x-y=100 and y-x=100 unless both are 0
(check-sat)
(exit)
