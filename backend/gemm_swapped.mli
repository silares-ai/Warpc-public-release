(** The swapped orientation: a CTA pair
   shares a tcgen05.mma.cta_group::2 whose A operand is the tile of B^T, so a
   CTA's accumulator is output columns by output rows. C = A B^T, f16 in,
   f32 out.

   The cluster is the pair (x, along the output's columns) times [cluster_n]
   pairs (y, along its rows) that share the tile of B^T: each CTA along y
   loads a slice of it and multicasts it. A CTA's tile is [tile_m] x
   [tile_n]; a taller MMA stacks atoms. One or two accumulators; tiles by a
   fixed stride, by cluster launch control, or as stream-K segments; the
   store through staging copies and the copy engine, or straight from the
   registers.

   Warp 0 loads, warp 1 issues the pair's MMAs (the leader's does) and sets
   up the barriers and tensor memory, warp 2 schedules (with cluster launch
   control, in the cluster's first CTA), warps 4-7 store. Every memory
   operand is a buffer at coordinates: a pipe's stage, an accumulator
   buffer, a peer's barrier, a warp's staging copy, a CTA's workspace slot.
   The warps meet through three pipes: the operand stages, the accumulators,
   and the scheduler's answer. *)

(** the clusters of so many CTAs resident at once *)
val resident_clusters : int -> int

val build :
  ?clc:bool ->
  ?ask_ahead:bool ->
  ?direct:bool ->
  ?serpentine:bool ->
  ?cluster_n:int ->
  ?epi_rows:int ->
  ?bufs:int ->
  ?group:int ->
  ?stamps:bool ->
  ?stage_stamps:bool ->
  ?streamk:bool ->
  ?split:int ->
  ?split_grid:int ->
  ?split_add:bool ->
  ?late_launch:bool ->
  tile_m:int ->
  tile_n:int -> depth:int -> m:int -> n:int -> k:int -> unit -> Core.kernel
