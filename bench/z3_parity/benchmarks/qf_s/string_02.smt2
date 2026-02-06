; QF_S Benchmark: Chain Concatenation
; Expected Result: UNSAT
; Description: Conflicting chain concatenation constraints

(set-logic QF_S)
(declare-const a String)
(declare-const b String)
(declare-const c String)

; Test: a ++ b ++ c = "abc" but length constraints make it impossible
(assert (= (str.++ a b c) "abc"))
(assert (= (str.len a) 2))
(assert (= (str.len b) 2))
(assert (= (str.len c) 1))

(check-sat)
; Expected: unsat (total length would be 5, but "abc" has length 3)
