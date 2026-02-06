; QF_S Benchmark: Contains and Prefix
; Expected Result: SAT
; Description: String containment and prefix operations

(set-logic QF_S)
(declare-const s String)

; Test: Find a string that contains "test" and has prefix "my"
(assert (str.contains s "test"))
(assert (str.prefixof "my" s))
(assert (>= (str.len s) 6))

(check-sat)
; Expected: sat (e.g., s = "mytest")
