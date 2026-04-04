; Test: Bitvector array store and select
;; expected: sat
; Pattern: Store BV values in array, read back with UF constraints

(set-logic QF_AUFBV)
(declare-const mem (Array (_ BitVec 8) (_ BitVec 32)))

; Store word at address 0x00
(declare-const mem1 (Array (_ BitVec 8) (_ BitVec 32)))
(assert (= mem1 (store mem #x00 #x0000002A)))

; Store word at address 0x01
(declare-const mem2 (Array (_ BitVec 8) (_ BitVec 32)))
(assert (= mem2 (store mem1 #x01 #xDEADBEEF)))

; Read back
(assert (= (select mem2 #x00) #x0000002A))
(assert (= (select mem2 #x01) #xDEADBEEF))

; BV arithmetic on loaded value
(declare-const val (_ BitVec 32))
(assert (= val (bvadd (select mem2 #x00) #x00000001)))
(assert (= val #x0000002B))

(check-sat)
(exit)
