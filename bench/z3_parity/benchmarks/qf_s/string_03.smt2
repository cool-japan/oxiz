; QF_S Benchmark: String Length Basic
; Expected Result: SAT
; Description: Basic string length constraints

(set-logic QF_S)
(declare-const s String)
(declare-const t String)

; Test: Find strings where s has length 5 and t has length 3
; and their concatenation equals a known string
(assert (= (str.len s) 5))
(assert (= (str.len t) 3))
(assert (= (str.++ s t) "worldfoo"))

(check-sat)
; Expected: sat (s = "world", t = "foo")
