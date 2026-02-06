; Benchmark: Format Conversion - Float32 to Float64
; Expected: SAT
; Description: Tests conversion from Float32 to Float64 with precision preservation

(set-logic QF_FP)
(set-info :status sat)

; Declare Float32 and Float64 variables
(declare-fun x32 () (_ FloatingPoint 8 24))
(declare-fun y32 () (_ FloatingPoint 8 24))
(declare-fun x64 () (_ FloatingPoint 11 53))
(declare-fun y64 () (_ FloatingPoint 11 53))

; x32 = 3.14159 (approximation in Float32)
(assert (= x32 ((_ to_fp 8 24) RNE 3.14159)))

; y32 = -2.71828 (approximation in Float32)
(assert (= y32 ((_ to_fp 8 24) RNE (- 2.71828))))

; Convert to Float64
(assert (= x64 ((_ to_fp 11 53) RNE x32)))
(assert (= y64 ((_ to_fp 11 53) RNE y32)))

; Check that signs are preserved
(assert (fp.isPositive x64))
(assert (fp.isNegative y64))

; Check that conversions preserve relative ordering
(assert (fp.gt x64 y64))

; Verify bounds after conversion
(assert (fp.gt x64 ((_ to_fp 11 53) RNE 3.14)))
(assert (fp.lt x64 ((_ to_fp 11 53) RNE 3.15)))

; Test arithmetic on converted values
(declare-fun sum64 () (_ FloatingPoint 11 53))
(assert (= sum64 (fp.add RNE x64 y64)))
(assert (fp.gt sum64 ((_ to_fp 11 53) RNE 0.4)))
(assert (fp.lt sum64 ((_ to_fp 11 53) RNE 0.5)))

(check-sat)
; (exit)
