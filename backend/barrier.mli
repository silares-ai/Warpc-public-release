(** mbarriers: the word a barrier is initialised with, the parity flip, and
   the parities a wait on a fresh barrier waits at. *)

(** The word SYNCS.EXCH.64 initialises a barrier with, for a phase that
   completes after [count] arrivals: the pending count 2^20 - count at bit 1
   of the low word and at bit 11 of the high one, as ptxas lowers
   mbarrier.init. *)
val init_word : Core.ur2 Core.reg -> count:int -> Core.stmt list

(** the parity of the next phase *)
val flip : Core.r Core.reg -> Core.stmt

(** The phase a wait on a barrier that starts free passes at once, and the
   one that waits for the first completion. *)
val free_parity : int

val full_parity : int
