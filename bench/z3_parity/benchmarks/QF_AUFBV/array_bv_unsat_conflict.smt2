; Unsatisfiable: conflicting array constraints with BV
; Expected: unsat
; Tests that store-then-select with same index must yield stored value

(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 8) (_ BitVec 8)))
(declare-fun i () (_ BitVec 8))
(declare-fun v () (_ BitVec 8))

; Store v at index i
(define-fun b () (Array (_ BitVec 8) (_ BitVec 8))
  (store a i v))

; Read back at the same index must equal v (by read-over-write axiom)
; But we assert it equals something different from v
(assert (= v (_ bv42 8)))
(assert (not (= (select b i) (_ bv42 8))))

(check-sat)
; expected: unsat
(exit)
