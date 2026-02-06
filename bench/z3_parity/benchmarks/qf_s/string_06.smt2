; QF_S Benchmark: Suffix Operations
; Expected Result: SAT
; Description: Suffix and contains combined

(set-logic QF_S)
(declare-const text String)

; Test: Find a string with suffix ".txt" that contains "file"
(assert (str.suffixof ".txt" text))
(assert (str.contains text "file"))
(assert (<= (str.len text) 15))

(check-sat)
; Expected: sat (e.g., text = "myfile.txt")
