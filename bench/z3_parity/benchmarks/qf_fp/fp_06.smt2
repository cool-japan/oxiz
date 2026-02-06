; Benchmark: Special Values - Zero Handling (Positive/Negative)
; Expected: UNSAT
; Description: Tests positive and negative zero with conflicting constraints

(set-logic QF_FP)
(set-info :status unsat)

; Declare Float32 variables
(declare-fun pzero () (_ FloatingPoint 8 24))
(declare-fun nzero () (_ FloatingPoint 8 24))
(declare-fun x () (_ FloatingPoint 8 24))

; pzero is positive zero
(assert (fp.isZero pzero))
(assert (fp.isPositive pzero))

; nzero is negative zero
(assert (fp.isZero nzero))
(assert (fp.isNegative nzero))

; x = pzero + nzero (should be positive zero in RNE mode)
(assert (= x (fp.add RNE pzero nzero)))

; Check that x is zero
(assert (fp.isZero x))

; Conflicting constraint: require x to be negative zero AND positive zero
; is not the same bit pattern
(assert (fp.isNegative x))

; Additional impossible constraint
(declare-fun y () (_ FloatingPoint 8 24))
(assert (= y (fp.div RNE pzero pzero)))
; Division by zero produces NaN
(assert (not (fp.isNaN y)))  ; This should make it UNSAT

(check-sat)
; (exit)
