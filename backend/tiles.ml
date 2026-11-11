(* Which output tile a CTA index stands for. The map is a layout: the
   index's digits -- the CTA's rank in its cluster, its place in a group of
   rows, the column, the group -- name the tile, so consecutive clusters walk
   down [group] rows before moving across and the tiles in flight share their
   operand rows in L2. What a program emits is the decided strided form of the
   map to each coordinate of the tile's origin. *)

open Layouts

let pow2 n = n > 0 && n land (n - 1) = 0

(* The origin of a CTA's tile: its row and column as expressions in the CTA
   index "c" -- what a program's derived instructions compute -- and as
   functions of it, evaluated on the map itself -- what the checks compare
   them with. *)
type origin =
  { e_row : Expr.t
  ; e_col : Expr.t
  ; row_of : int -> int
  ; col_of : int -> int
  ; ntiles : int
  }

(* [tiles_m] x [tiles_n] tiles of [tile_m] x [tile_n]. A cluster is [pair]
   CTAs that split one MMA -- consecutive ranks, on neighbouring tiles down
   the rows, or across the columns when [pair_on_cols] (the accumulator's
   lanes stand for output columns) -- times [pairs] such groups along the
   other coordinate. Clusters walk [group] cluster rows down before moving
   across -- or, [along_n], [group] cluster columns across before moving
   down -- and with [serpentine] every other band runs backwards. The
   origin's row and column as expressions in the CTA index "c". *)
let origin ~tiles_m ~tiles_n ~tile_m ~tile_n ~pair ~pairs ~pair_on_cols ?(group = 1) ?(along_n = false) ?(serpentine = false) () =
  let cm, cn = if pair_on_cols then pairs, pair else pair, pairs in
  let cl_m = tiles_m / cm and cl_n = tiles_n / cn in
  if cl_m * cm <> tiles_m || cl_n * cn <> tiles_n then failwith "Tiles: the clusters do not tile the output";
  let walked, other = if along_n then cl_n, cl_m else cl_m, cl_n in
  let group =
    let rec g n = if n * 2 <= walked && n < group then g (n * 2) else n in
    let g = g 1 in
    if pow2 other && walked mod g = 0 then g else 1
  in
  let width = tiles_n in
  let grid : (Space.thread_value, Space.logical) Layout.t =
    let row = cm * width and col = cn in
    let walk : Linear.t list =
      if along_n
      then [ Axis { size = cl_n / group; stride = group * col }; Axis { size = cl_m; stride = row }; Axis { size = group; stride = col } ]
      else [ Axis { size = cl_m / group; stride = group * row }; Axis { size = cl_n; stride = col }; Axis { size = group; stride = row } ]
    in
    (* the rank's digits, the pair partner fastest *)
    let within : Linear.t list =
      if pair_on_cols then [ Axis { size = pairs; stride = width }; Axis { size = pair; stride = 1 } ]
      else [ Axis { size = pairs; stride = 1 }; Axis { size = pair; stride = width } ]
    in
    Layout.of_linear (Group (walk @ within))
  in
  let ntiles = tiles_m * tiles_n in
  if not (Layout.is_bijection_onto grid ~size:ntiles) then failwith "Tiles: the CTA-to-tile map misses a tile";
  let tile_of t = Layout.offset grid (Coord.unflatten (Layout.shape grid) t) in
  let e_row = Addr.form ~name:"tile row" ~n:ntiles (fun i -> tile_of i / width * tile_m) in
  let e_col = Addr.form ~name:"tile column" ~n:ntiles (fun i -> tile_of i mod width * tile_n) in
  (* a serpentine walk reverses every other band a whole cluster at a time:
     the column of a cluster's CTAs xored with the last cluster column *)
  let serp = serpentine && (not along_n) && pow2 cl_n && cl_n > 1 in
  let e_col =
    if serp
    then begin
      let band = Expr.modulo (Expr.div (Expr.div e_row tile_m) (group * cm)) 2 in
      let col_tile = Expr.div e_col tile_n in
      Expr.scale tile_n (Expr.xor col_tile (Expr.scale ((cl_n - 1) * cn) band))
    end
    else e_col
  in
  let row_of t = tile_of t / width * tile_m in
  let col_of t =
    let col_tile = tile_of t mod width in
    let band = tile_of t / width / (group * cm) mod 2 in
    tile_n * (if serp then col_tile lxor ((cl_n - 1) * cn * band) else col_tile)
  in
  { e_row; e_col; row_of; col_of; ntiles }
