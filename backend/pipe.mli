(** Pipes between warps: [depth] stages, each with a barrier pair -- full
    once its producers have filled it, empty once its consumers have freed
    it -- and a ring position at each end.

    The two ends are different types, so a producer can only wait for a
    stage to be empty and fill it, a consumer only wait for it to be full
    and free it; each end's ring starts at the phase its first wait passes
    at, and flips it as it wraps. Every operation is one instruction, or a
    spin on one. *)

type t
type producer
type consumer

(** A pipe of [depth] stages, a stage filled by [fills] arrivals and freed
    by [frees]; its barriers are the kernel's. *)
val make : Mem.kernel -> name:string -> depth:int -> fills:int -> frees:int -> t

(** This warp's end of the pipe: to the producer every stage starts free,
    to the consumer none starts full. *)
val producer : Mem.kernel -> t -> producer

val consumer : Mem.kernel -> t -> consumer

(** One stage at this end: [body] runs at it, and the end moves on after. *)
val produce : producer -> (unit -> 'a) -> 'a

val consume : consumer -> (unit -> 'a) -> 'a

(** [buf]'s copy for the end's stage, in this CTA. *)
val slot_p : producer -> (Mem.shared, 'k) Mem.buffer -> (Mem.shared, 'k) Mem.term

val slot_c : consumer -> (Mem.shared, 'k) Mem.buffer -> (Mem.shared, 'k) Mem.term

(** The producer's side: until the stage is empty; the barrier the stage's
    fill reports to -- this CTA's, or at [peer] another's. *)
val wait_empty : producer -> unit

val full : ?peer:Mem.shared Mem.window -> producer -> (Mem.shared, Mem.barrier) Mem.term

(** The consumer's side: until the stage is full; a question about it whose
    answer lands in [into] later; the barrier its release reports to. *)
val wait_full : consumer -> unit

val probe_full : consumer -> into:Core.p Core.reg -> unit
val empty : ?peer:Mem.shared Mem.window -> consumer -> (Mem.shared, Mem.barrier) Mem.term
