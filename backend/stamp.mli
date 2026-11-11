(** A timing build: where a warp passes a point the program marks, it writes
   the global timer's low word to the stamp buffer -- the kernel's last
   parameter, 64 words a CTA: 16 events for each of its first four tiles,
   the tile counted mod 4 (run_tma.py's .stamps). A stamp is a construct:
   its instructions are the timing build's, not the program's. *)

val at : Prog.t -> nparams:int -> ev:int -> ?count:Core.r Core.reg -> unit -> unit

(** A stamp each time a stage lands: 4096 words a CTA, the [count]-th at
   word count mod 4096 (run_tma.py's .stagestamps) *)
val stage :
  Prog.t -> ?half:int -> nparams:int -> count:Core.r Core.reg -> unit -> unit
