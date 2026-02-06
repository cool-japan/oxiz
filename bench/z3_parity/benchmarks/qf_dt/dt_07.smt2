; QF_DT Benchmark: Simple Enumeration Types
; Expected: SAT
; Description: Test simple enumeration types with equality

(set-logic ALL)

; Define a color enumeration
(declare-datatypes ((Color 0)) (
  ((Red) (Green) (Blue) (Yellow))
))

; Define a direction enumeration
(declare-datatypes ((Direction 0)) (
  ((North) (South) (East) (West))
))

; Declare variables
(declare-const c1 Color)
(declare-const c2 Color)
(declare-const c3 Color)
(declare-const d1 Direction)
(declare-const d2 Direction)

; c1 is Red
(assert (= c1 Red))

; c2 is not Red and not Green
(assert (not (= c2 Red)))
(assert (not (= c2 Green)))

; c2 is Blue
(assert (= c2 Blue))

; c3 is either Red or Yellow
(assert (or (= c3 Red) (= c3 Yellow)))

; c3 is not c2
(assert (not (= c3 c2)))

; d1 is North
(assert (= d1 North))

; d2 is not North and not South
(assert (not (= d2 North)))
(assert (not (= d2 South)))

; d2 must be East or West
(assert (or (= d2 East) (= d2 West)))

(check-sat)
(exit)
