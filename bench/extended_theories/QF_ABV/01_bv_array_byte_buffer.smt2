; Test: Bitvector array as byte buffer
;; expected: sat
; Pattern: Sequential writes to a byte-addressed buffer

(set-logic QF_ABV)
(declare-const buf (Array (_ BitVec 8) (_ BitVec 8)))

; Write ASCII "Hi!" to buffer at offsets 0, 1, 2
(declare-const buf1 (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= buf1 (store buf #x00 #x48)))  ; 'H' = 0x48

(declare-const buf2 (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= buf2 (store buf1 #x01 #x69)))  ; 'i' = 0x69

(declare-const buf3 (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= buf3 (store buf2 #x02 #x21)))  ; '!' = 0x21

; Read back and verify
(assert (= (select buf3 #x00) #x48))
(assert (= (select buf3 #x01) #x69))
(assert (= (select buf3 #x02) #x21))

; Arithmetic check: sum of first two chars
(assert (= (bvadd (select buf3 #x00) (select buf3 #x01)) #xB1))

(check-sat)
(exit)
