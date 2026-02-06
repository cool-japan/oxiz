; Benchmark: Rounding Modes - Conflicting Constraints
; Expected: UNSAT
; Description: Creates conflicting constraints with different rounding modes

(set-logic QF_FP)
(set-info :status unsat)

; Declare Float32 variables
(declare-fun x () (_ FloatingPoint 8 24))
(declare-fun y () (_ FloatingPoint 8 24))
(declare-fun z1 () (_ FloatingPoint 8 24))
(declare-fun z2 () (_ FloatingPoint 8 24))

; x = 1.1
(assert (= x ((_ to_fp 8 24) RNE 1.1)))

; y = 2.2
(assert (= y ((_ to_fp 8 24) RNE 2.2)))

; z1 = x + y with RTP (round to positive)
(assert (= z1 (fp.add RTP x y)))

; z2 = x + y with RTN (round to negative)
(assert (= z2 (fp.add RTN x y)))

; Conflicting constraint: require z1 < z2 (impossible for positive operands with RTP and RTN)
(assert (fp.lt z1 z2))

; Additional impossible constraint
(assert (fp.gt z1 ((_ to_fp 8 24) RNE 3.3)))
(assert (fp.lt z2 ((_ to_fp 8 24) RNE 3.3)))

(check-sat)
; (exit)
