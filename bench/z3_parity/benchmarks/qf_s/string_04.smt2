; QF_S Benchmark: String Length Constraints
; Expected Result: UNSAT
; Description: Contradictory length constraints

(set-logic QF_S)
(declare-const x String)

; Test: x cannot simultaneously satisfy conflicting length constraints
(assert (= (str.len x) 10))
(assert (= x "short"))  ; "short" has length 5

(check-sat)
; Expected: unsat (length 10 != length 5)
