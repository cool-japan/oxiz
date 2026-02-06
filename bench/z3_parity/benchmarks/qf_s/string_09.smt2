; QF_S Benchmark: Basic Regex - Digit Pattern
; Expected Result: SAT
; Description: Match strings containing digits

(set-logic QF_S)
(declare-const phone String)

; Test: Phone number matches pattern with digits
; Pattern: any chars followed by 3 digits
(assert (str.in_re phone
    (re.++
        (re.* re.allchar)
        (re.++ (re.range "0" "9")
               (re.++ (re.range "0" "9")
                      (re.range "0" "9"))))))
(assert (= (str.len phone) 10))
(assert (str.prefixof "call" phone))

(check-sat)
; Expected: sat (e.g., phone = "call me 123")
