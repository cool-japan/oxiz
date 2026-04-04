; Test: Ackermann-like function properties
; Expected: sat
; Pattern: Properties inspired by the Ackermann function (bounded version)

(set-logic UFLIA)
(declare-fun ack (Int Int) Int)

; Ackermann-like base cases and recursive properties (bounded)
; ack(0, n) = n + 1
(assert (forall ((n Int))
  (=> (and (>= n 0) (<= n 5))
      (= (ack 0 n) (+ n 1)))))

; ack(m, 0) = ack(m-1, 1) for m > 0 (bounded)
(assert (= (ack 1 0) (ack 0 1)))
(assert (= (ack 2 0) (ack 1 1)))

; ack(1, n) = n + 2 for small n
(assert (forall ((n Int))
  (=> (and (>= n 0) (<= n 3))
      (= (ack 1 n) (+ n 2)))))

; Verify: ack(0, 0) = 1, ack(1, 0) = 2, ack(1, 1) = 3
(assert (= (ack 0 0) 1))
(assert (= (ack 1 0) 2))
(assert (= (ack 1 1) 3))

; Ackermann function is always positive for non-negative inputs
(assert (forall ((m Int) (n Int))
  (=> (and (>= m 0) (<= m 2) (>= n 0) (<= n 5))
      (> (ack m n) 0))))

(check-sat)
(exit)
