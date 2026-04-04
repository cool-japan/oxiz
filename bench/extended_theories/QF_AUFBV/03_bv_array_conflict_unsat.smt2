; Test: BV array conflict
;; expected: unsat
; Pattern: Store a value and require a different value at same address

(set-logic QF_AUFBV)
(declare-const mem (Array (_ BitVec 8) (_ BitVec 16)))

; Store 0xCAFE at address 0x10
(declare-const mem1 (Array (_ BitVec 8) (_ BitVec 16)))
(assert (= mem1 (store mem #x10 #xCAFE)))

; Require reading 0xBEEF at the same address
(assert (= (select mem1 #x10) #xBEEF))

; 0xCAFE != 0xBEEF => unsat
(check-sat)
(exit)
