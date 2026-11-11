(** The program warpc gemm chooses for a shape (fp16 in, fp32 out), by the
   rule in programs.ml: the swapped orientation with cluster launch control,
   stacked atoms or multicast clusters where the tiles fill the machine; K
   split, stream-K or the M = 128 pair where they do not; the pair in
   CUTLASS's orientation, or one CTA a tile, otherwise. *)

(** the GPU the chosen program is for: sm_100a (the B200, the default) or
    sm_103a (the B300) *)
val target : string ref

val choose : m:int -> n:int -> k:int -> Core.kernel
