(** A GEMM: one CTA per 128 x 128 output tile, C = A B^T with A M x K and B^T
   N x K (f16, K-major), C f32 row-major. The MMA takes the B^T tile as its A
   operand, so tensor-memory lanes are output columns and each epilogue
   register a row: the store goes straight from the registers. Warps 0-3
   store, warp 4 loads, warp 5 issues the MMAs and owns the barriers and
   tensor memory.

   The loads reach the MMAs through a pipe; every memory operand is a buffer
   at coordinates, so no line below computes an address. *)

val build : m:int -> n:int -> k:int -> depth:int -> unit -> Core.kernel
