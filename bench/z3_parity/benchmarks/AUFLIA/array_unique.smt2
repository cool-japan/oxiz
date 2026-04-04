; Test: Array injectivity -- too many distinct values for bounded range
; Expected: unsat
; Z3 result: unsat (pigeonhole principle)
; Pattern: forall i j. i != j => a[i] != a[j] but values bounded to fewer options

(set-logic AUFLIA)
(declare-const a (Array Int Int))

; 4 positions must all have distinct values
(assert (forall ((i Int) (j Int))
  (=> (and (>= i 0) (<= i 3) (>= j 0) (<= j 3) (not (= i j)))
      (not (= (select a i) (select a j))))))

; But all values must be in {0, 1, 2} -- only 3 slots for 4 distinct values
(assert (forall ((i Int))
  (=> (and (>= i 0) (<= i 3))
      (and (>= (select a i) 0) (<= (select a i) 2)))))

(check-sat)
(exit)
