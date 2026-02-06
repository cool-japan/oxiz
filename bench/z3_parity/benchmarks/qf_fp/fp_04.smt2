; Benchmark: Special Values - NaN Handling
; Expected: SAT
; Description: Tests NaN propagation and detection

(set-logic QF_FP)
(set-info :status sat)

; Declare Float32 variables
(declare-fun x () (_ FloatingPoint 8 24))
(declare-fun y () (_ FloatingPoint 8 24))
(declare-fun z () (_ FloatingPoint 8 24))

; x is NaN
(assert (fp.isNaN x))

; y is a normal number
(assert (= y ((_ to_fp 8 24) RNE 5.0)))
(assert (not (fp.isNaN y)))

; z = x + y (NaN propagates)
(assert (= z (fp.add RNE x y)))

; z should be NaN
(assert (fp.isNaN z))

; Additional NaN operations
(declare-fun w () (_ FloatingPoint 8 24))
(assert (= w (fp.mul RNE x y)))
(assert (fp.isNaN w))

; Check that y is positive and finite
(assert (fp.isPositive y))
(assert (not (fp.isInfinite y)))
(assert (not (fp.isZero y)))

(check-sat)
; (exit)
