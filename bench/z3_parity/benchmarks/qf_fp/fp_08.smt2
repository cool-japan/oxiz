; Benchmark: Format Conversion - Type Conversions with Constraints
; Expected: UNSAT
; Description: Tests conversions with impossible precision constraints

(set-logic QF_FP)
(set-info :status unsat)

; Declare Float32 and Float64 variables
(declare-fun x32 () (_ FloatingPoint 8 24))
(declare-fun x64_1 () (_ FloatingPoint 11 53))
(declare-fun x64_2 () (_ FloatingPoint 11 53))

; x32 represents a value that cannot be exactly represented in Float32
; Use a value with many decimal places
(assert (= x32 ((_ to_fp 8 24) RNE 1.23456789)))

; Convert x32 to Float64 with RNE
(assert (= x64_1 ((_ to_fp 11 53) RNE x32)))

; Create x64_2 directly from the decimal value
(assert (= x64_2 ((_ to_fp 11 53) RNE 1.23456789)))

; Impossible constraint: require that converting Float32->Float64
; gives the same result as creating Float64 directly from decimal
; This is impossible due to Float32's limited precision
(assert (= x64_1 x64_2))

; Additional impossible constraint
; Require x64_1 to have more precision than Float32 can represent
(declare-fun diff () (_ FloatingPoint 11 53))
(assert (= diff (fp.sub RNE x64_1 x64_2)))
(assert (fp.isZero diff))  ; diff must be zero
(assert (not (= x32 ((_ to_fp 8 24) RNE x64_2))))  ; but back-conversion differs

(check-sat)
; (exit)
