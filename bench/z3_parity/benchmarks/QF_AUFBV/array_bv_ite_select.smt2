; Conditional array access with bitvector conditions
; Expected: sat
; Tests ITE with array select and BV operations

(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun b () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun idx () (_ BitVec 32))
(declare-fun flag () (_ BitVec 1))

; Choose array based on flag
(declare-fun result () (_ BitVec 32))
(assert (= result (ite (= flag (_ bv1 1))
                       (select a idx)
                       (select b idx))))

; Constrain values
(assert (= (select a idx) (_ bv100 32)))
(assert (= (select b idx) (_ bv200 32)))
(assert (= flag (_ bv1 1)))

; result should be 100
(assert (= result (_ bv100 32)))

(check-sat)
; expected: sat
(exit)
