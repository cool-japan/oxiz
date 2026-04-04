; Test: Real array with conflicting integer-like constraints
;; expected: unsat
; Pattern: Require a real value to be simultaneously > 3.0 and < 2.0

(set-logic AUFLIRA)

(declare-const a (Array Int Real))

; Store a value at index 0
(declare-const a1 (Array Int Real))
(declare-const v Real)
(assert (= a1 (store a 0 v)))

; Require the stored value to be > 3.0
(assert (> (select a1 0) 3.0))

; Also require the stored value to be < 2.0
(assert (< (select a1 0) 2.0))

; This is contradictory: no real can be both > 3.0 and < 2.0
(check-sat)
(exit)
