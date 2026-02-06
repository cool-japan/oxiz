; Benchmark: Rounding Modes - RNE and RTZ
; Expected: SAT
; Description: Tests Round to Nearest Even (RNE) and Round to Zero (RTZ) modes
; with basic floating point operations

(set-logic QF_FP)
(set-info :status sat)

; Declare Float32 variables
(declare-fun x () (_ FloatingPoint 8 24))
(declare-fun y () (_ FloatingPoint 8 24))
(declare-fun z () (_ FloatingPoint 8 24))

; x = 1.5 in Float32
(assert (= x ((_ to_fp 8 24) RNE 1.5)))

; y = 2.3 in Float32
(assert (= y ((_ to_fp 8 24) RNE 2.3)))

; z = x + y with RNE rounding
(assert (= z (fp.add RNE x y)))

; Check that z is approximately 3.8
(assert (fp.gt z ((_ to_fp 8 24) RNE 3.7)))
(assert (fp.lt z ((_ to_fp 8 24) RNE 3.9)))

; Additional constraint using RTZ
(declare-fun w () (_ FloatingPoint 8 24))
(assert (= w (fp.mul RTZ x y)))

; w should be positive
(assert (fp.gt w ((_ to_fp 8 24) RTZ 0.0)))

(check-sat)
; (exit)
