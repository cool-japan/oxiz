; QF_S Benchmark: Replace All Operation
; Expected Result: UNSAT
; Description: Contradictory replace_all constraint

(set-logic QF_S)
(declare-const s String)
(declare-const result String)

; Test: Replace all "a" with "b", but result contradicts
(assert (= result (str.replace_all s "a" "b")))
(assert (= s "banana"))
(assert (= result "banana"))  ; Should be "bbnbnb" if all 'a's are replaced

(check-sat)
; Expected: unsat (result cannot be "banana" if we replace all 'a's)
