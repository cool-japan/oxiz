; QF_S Benchmark: Replace Operation
; Expected Result: SAT
; Description: Single string replacement

(set-logic QF_S)
(declare-const input String)
(declare-const output String)

; Test: Replace "old" with "new" in input to get output
(assert (= output (str.replace input "old" "new")))
(assert (= input "the old way"))
(assert (= output "the new way"))

(check-sat)
; Expected: sat
