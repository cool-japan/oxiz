; Benchmark: Special Values - Infinity Handling
; Expected: SAT
; Description: Tests positive and negative infinity arithmetic

(set-logic QF_FP)
(set-info :status sat)

; Declare Float64 variables
(declare-fun inf_pos () (_ FloatingPoint 11 53))
(declare-fun inf_neg () (_ FloatingPoint 11 53))
(declare-fun x () (_ FloatingPoint 11 53))
(declare-fun y () (_ FloatingPoint 11 53))

; inf_pos is positive infinity
(assert (fp.isInfinite inf_pos))
(assert (fp.isPositive inf_pos))

; inf_neg is negative infinity
(assert (fp.isInfinite inf_neg))
(assert (fp.isNegative inf_neg))

; x is a finite positive number
(assert (= x ((_ to_fp 11 53) RNE 42.0)))
(assert (not (fp.isInfinite x)))

; y = inf_pos + x (should still be positive infinity)
(assert (= y (fp.add RNE inf_pos x)))
(assert (fp.isInfinite y))
(assert (fp.isPositive y))

; Check infinity arithmetic properties
(declare-fun z () (_ FloatingPoint 11 53))
(assert (= z (fp.mul RNE inf_pos x)))
(assert (fp.isInfinite z))
(assert (fp.isPositive z))

; inf_pos > any finite number
(assert (fp.gt inf_pos x))

(check-sat)
; (exit)
