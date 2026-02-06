; QF_S Benchmark: Basic Regex - Character Range
; Expected Result: SAT
; Description: Match strings with lowercase letters

(set-logic QF_S)
(declare-const word String)

; Test: Word contains only lowercase letters (at least 3, at most 8)
(assert (str.in_re word
    (re.++
        (re.range "a" "z")
        (re.++
            (re.range "a" "z")
            (re.++
                (re.range "a" "z")
                (re.* (re.range "a" "z")))))))
(assert (>= (str.len word) 3))
(assert (<= (str.len word) 8))
(assert (str.contains word "test"))

(check-sat)
; Expected: sat (e.g., word = "test" or "testing")
