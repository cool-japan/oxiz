; QF_S Benchmark: Basic String Concatenation
; Expected Result: SAT
; Description: Simple concatenation with equality constraint

(set-logic QF_S)
(declare-const x String)
(declare-const y String)
(declare-const z String)

; Test: x ++ y = "hello" and x = "hel" implies y = "lo"
(assert (= (str.++ x y) "hello"))
(assert (= x "hel"))
(assert (= y "lo"))

(check-sat)
; Expected: sat
