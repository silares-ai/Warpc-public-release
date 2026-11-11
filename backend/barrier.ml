(* mbarriers: the word a barrier is initialised with, the parity flip, and
   the parities a wait on a fresh barrier waits at. *)

open Core

(* The word SYNCS.EXCH.64 initialises a barrier with, for a phase that
   completes after [count] arrivals: the pending count 2^20 - count at bit 1
   of the low word and at bit 11 of the high one, as ptxas lowers
   mbarrier.init. *)
let init_word (w : ur2 reg) ~count =
  let pending = 0x100000 - count in
  [ umov (upart w 0) ((pending lsl 1) land 0xffffffff); umov (upart w 1) ((pending lsl 11) land 0xffffffff) ]

(* the parity of the next phase *)
let flip parity = lop3_xor_imm parity parity 0x80000000

(* The phase a wait on a barrier that starts free passes at once, and the
   one that waits for the first completion. *)
let free_parity = 0x80000000
let full_parity = 0
