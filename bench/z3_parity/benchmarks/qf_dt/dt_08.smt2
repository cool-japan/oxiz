; QF_DT Benchmark: Enum with Impossible Constraints
; Expected: UNSAT
; Description: Test enumeration with contradictory constraints

(set-logic ALL)

; Define a weekday enumeration
(declare-datatypes ((Weekday 0)) (
  ((Monday) (Tuesday) (Wednesday) (Thursday) (Friday) (Saturday) (Sunday))
))

; Declare variable
(declare-const day Weekday)

; day is Monday
(assert (= day Monday))

; day is also Tuesday - contradiction!
(assert (= day Tuesday))

(check-sat)
(exit)
