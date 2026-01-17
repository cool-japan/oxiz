; Simple arrays benchmark for array theory solver performance
; Tests basic array operations (select/store)

(set-logic QF_AUFLIA)

(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(declare-const i Int)
(declare-const j Int)
(declare-const v Int)

; Basic constraints on indices
(assert (>= i 0))
(assert (<= i 10))
(assert (>= j 0))
(assert (<= j 10))
(assert (not (= i j)))

; Array operations
(assert (= (select a i) 42))
(assert (= (select a j) 17))

; Store and read back
(assert (= b (store a i 100)))
(assert (= (select b i) 100))
(assert (= (select b j) (select a j)))

; Value constraint
(assert (= v (+ (select a i) (select a j))))
(assert (= v 59))

; This should be satisfiable
(check-sat)
(get-model)
(exit)
