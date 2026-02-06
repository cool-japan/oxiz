; Benchmark: Rounding Modes - RTP and RTN
; Expected: SAT
; Description: Tests Round to Positive (RTP) and Round to Negative (RTN) modes
; with division operations

(set-logic QF_FP)
(set-info :status sat)

; Declare Float64 variables
(declare-fun a () (_ FloatingPoint 11 53))
(declare-fun b () (_ FloatingPoint 11 53))
(declare-fun c_rtp () (_ FloatingPoint 11 53))
(declare-fun c_rtn () (_ FloatingPoint 11 53))

; a = 10.0
(assert (= a ((_ to_fp 11 53) RNE 10.0)))

; b = 3.0
(assert (= b ((_ to_fp 11 53) RNE 3.0)))

; c_rtp = a / b with round to positive
(assert (= c_rtp (fp.div RTP a b)))

; c_rtn = a / b with round to negative
(assert (= c_rtn (fp.div RTN a b)))

; RTP result should be >= RTN result for positive operands
(assert (fp.geq c_rtp c_rtn))

; Both should be close to 3.333...
(assert (fp.gt c_rtp ((_ to_fp 11 53) RNE 3.333)))
(assert (fp.lt c_rtn ((_ to_fp 11 53) RNE 3.334)))

(check-sat)
; (exit)
