(** Which output tile a CTA index stands for. The map is a layout: the
   index's digits -- the CTA's rank in its cluster, its place in a group of
   rows, the column, the group -- name the tile, so consecutive clusters walk
   down [group] rows before moving across and the tiles in flight share their
   operand rows in L2. What a program emits is the decided strided form of the
   map to each coordinate of the tile's origin. *)

type origin = {
  e_row : Layouts.Expr.t;
  e_col : Layouts.Expr.t;
  row_of : int -> int;
  col_of : int -> int;
  ntiles : int;
}

(** [tiles_m] x [tiles_n] tiles of [tile_m] x [tile_n]. A cluster is [pair]
   CTAs that split one MMA -- consecutive ranks, on neighbouring tiles down
   the rows, or across the columns when [pair_on_cols] (the accumulator's
   lanes stand for output columns) -- times [pairs] such groups along the
   other coordinate. Clusters walk [group] cluster rows down before moving
   across -- or, [along_n], [group] cluster columns across before moving
   down -- and with [serpentine] every other band runs backwards. The
   origin's row and column as expressions in the CTA index "c". *)
val origin :
  tiles_m:int ->
  tiles_n:int ->
  tile_m:int ->
  tile_n:int ->
  pair:int ->
  pairs:int ->
  pair_on_cols:bool ->
  ?group:int -> ?along_n:bool -> ?serpentine:bool -> unit -> origin
