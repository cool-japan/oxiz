; Multiple stores and reads on BV arrays
; Expected: sat
; Tests chain of store operations with different indices

(set-logic QF_AUFBV)
(declare-fun a () (Array (_ BitVec 16) (_ BitVec 16)))

; Store values at three different indices
(define-fun a1 () (Array (_ BitVec 16) (_ BitVec 16))
  (store a (_ bv0 16) (_ bv10 16)))
(define-fun a2 () (Array (_ BitVec 16) (_ BitVec 16))
  (store a1 (_ bv1 16) (_ bv20 16)))
(define-fun a3 () (Array (_ BitVec 16) (_ BitVec 16))
  (store a2 (_ bv2 16) (_ bv30 16)))

; Read back all three values
(assert (= (select a3 (_ bv0 16)) (_ bv10 16)))
(assert (= (select a3 (_ bv1 16)) (_ bv20 16)))
(assert (= (select a3 (_ bv2 16)) (_ bv30 16)))

(check-sat)
; expected: sat
(exit)
