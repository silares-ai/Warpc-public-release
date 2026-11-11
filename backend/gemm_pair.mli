(** The CUTLASS-orientation pair: a cluster of two CTAs along M shares one
   tcgen05.mma.cta_group::2 of M = 256, each CTA holding 128 rows of A, 64
   rows of B^T and 128 rows of the result. C = A B^T, f16 in, f32 out through
   staging copies and the copy engine.

   Warp 0 loads, warp 1 issues the pair's MMAs (the leader's does), warps 4-7
   store; warp 1 also sets up the barriers and tensor memory. The ring runs
   on across tiles. Every memory operand is a buffer at coordinates: this
   CTA's ring stage, the leader's barrier, a warp's staging copy. *)

val build : m:int -> n:int -> k:int -> depth:int -> unit -> Core.kernel
