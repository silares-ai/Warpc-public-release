(* The program warpc gemm chooses for a shape (fp16 in, fp32 out):

   - 8192 and up in every dimension: 256 x 256 a CTA, two stacked M = 256,
     N = 256 atoms, ring depth 4, cluster launch control, asking a stage
     ahead, the tiles walked as a serpentine.
   - otherwise, when 256-row tiles still fill the machine: up to K = 1024,
     256 x 128 a CTA at depth 5, the next grid let in once each producer's
     last load is out; beyond, 192 x 128 at depth 7, taking tiles by
     cluster launch control from K = 4096 on.
   - when 128 x 128 tiles would fill at most half the machine: the M = 128
     pair, 64 x 128 a CTA, depth 13, asking ahead, each warp storing its
     registers straight to C, on the B300 the next grid let in once each
     producer's last load is out; from K = 16384 on, 128 x 128 a CTA at
     depth 9 with K split in two in a 2 x 2 cluster, the second pair's
     partial added by the first; from K = 65536 on, 256 x 256 a CTA, depth
     4, asking ahead, K split over as many pairs a tile as are resident, each
     run adding its accumulator to C in the L2 once the tile's runs have
     zeroed it.
   - when 128 x 128 tiles fill one wave: 128 x 128 a CTA on the pairs
     alone, depth 6, each warp storing its registers straight to C, the next
     grid let in once each producer's last load is out.
   - from K = 16384 on, where 256 x 128 tiles in 2 x 2 clusters fit in one
     wave: on the B200, stream-K over all 74 pairs, the segment with a
     tile's first k step storing it and the tile's other segments adding to
     it in the L2 once its flag is up; on the B300, 256 x 128 tiles at depth
     6 in 2 x 2 clusters, the next grid let in once each producer's last load
     is out.
   - when 128 x 128 tiles take more than one wave: 128 x 128 a CTA. The
     2 x 4 cluster when it needs no more waves than the 2 x 2, the next grid
     let in once each producer's last load is out; otherwise the pairs alone
     when they need no more waves than the 2 x 2; otherwise the 2 x 2. A
     cluster's pairs share the tile of B^T, each CTA loading a slice and
     multicasting it, and take tiles by cluster launch control; the pairs
     alone are a fixed grid of as many pairs as are resident, walking the
     tiles by stride. Ring depth 9 from K = 8192 on, 8 below; the tiles
     walked in groups of 16.
   - the pair in CUTLASS's orientation on 128 x 128 tiles, depth 8, where it
     fits; one CTA a 128 x 128 tile otherwise. *)

(* the GPU the chosen program is for: sm_100a, the B200, or sm_103a, the B300 *)
let target = ref "sm_100a"

let sms = 148
let resident_clusters = Gemm_swapped.resident_clusters

let choose ~m ~n ~k : Core.kernel =
  let tiles = m / 128 * (n / 128) in
  let pair_fits = m mod 256 = 0 && n mod 128 = 0 && k mod 64 = 0 in
  let swapped_fits = n mod 256 = 0 && m mod 64 = 0 && k mod 64 = 0 && (m + 255) / 256 * (n / 128) >= sms in
  let waves cn = if m mod (128 * cn) = 0 then (tiles / (2 * cn) + resident_clusters (2 * cn) - 1) / resident_clusters (2 * cn) else max_int in
  (* K split over the grid: the pairs of 256 x 256 tiles, and the runs a tile
     when all are resident *)
  let pair_tiles = if m mod 256 = 0 && n mod 512 = 0 then m / 256 * (n / 512) else 0 in
  let runs = if pair_tiles > 0 then resident_clusters 2 / pair_tiles else 0 in
  if swapped_fits && n mod 512 = 0 && m mod 256 = 0 && min m (min n k) >= 8192
  then Gemm_swapped.build ~clc:true ~ask_ahead:true ~serpentine:true ~tile_m:256 ~tile_n:256 ~depth:4 ~m ~n ~k ()
  else if swapped_fits && k <= 1024 && m mod 256 = 0
  then Gemm_swapped.build ~late_launch:true ~tile_m:256 ~tile_n:128 ~depth:5 ~m ~n ~k ()
  else if swapped_fits
  then Gemm_swapped.build ~clc:(k >= 4096) ~tile_m:192 ~tile_n:128 ~depth:7 ~m ~n ~k ()
  else if 2 * tiles <= sms && n mod 128 = 0 && m mod 128 = 0 && k mod 64 = 0
  then
    if k >= 65536 && runs >= 2 && k / 64 >= runs
    then Gemm_swapped.build ~split_grid:runs ~split_add:true ~ask_ahead:true ~tile_m:256 ~tile_n:256 ~depth:4 ~m ~n ~k ()
    else if k >= 16384 && n / 128 mod 2 = 0 && k mod 128 = 0 && tiles / 2 <= resident_clusters 4
    then Gemm_swapped.build ~split:2 ~cluster_n:2 ~direct:true ~tile_m:128 ~tile_n:128 ~depth:9 ~m ~n ~k ()
    else Gemm_swapped.build ~direct:true ~ask_ahead:true ~late_launch:(!target = "sm_103a") ~tile_m:128 ~tile_n:64 ~depth:13 ~m ~n ~k ()
  else if tiles <= sms && n mod 256 = 0 && m mod 128 = 0 && k mod 64 = 0
  then Gemm_swapped.build ~direct:true ~late_launch:true ~tile_m:128 ~tile_n:128 ~depth:6 ~m ~n ~k ()
  else if tiles > sms && k >= 16384 && m mod 512 = 0 && n mod 256 = 0 && m / 512 * (n / 256) <= resident_clusters 4
  then
    if !target = "sm_100a"
    then Gemm_swapped.build ~streamk:true ~split_add:true ~bufs:2 ~ask_ahead:true ~tile_m:256 ~tile_n:128 ~depth:6 ~m ~n ~k ()
    else Gemm_swapped.build ~cluster_n:2 ~late_launch:true ~tile_m:256 ~tile_n:128 ~depth:6 ~m ~n ~k ()
  else if tiles > sms && n mod 256 = 0 && k mod 64 = 0 && waves 2 <> max_int
  then
    let cluster_n = if waves 4 <= waves 2 then 4 else if waves 1 <= waves 2 then 1 else 2 in
    Gemm_swapped.build ~clc:(cluster_n > 1) ~cluster_n ~late_launch:(cluster_n = 4) ~group:16 ~tile_m:128 ~tile_n:128 ~depth:(if k >= 8192 then 9 else 8) ~m ~n ~k ()
  else if pair_fits then Gemm_pair.build ~m ~n ~k ~depth:8 ()
  else Gemm_one.build ~m ~n ~k ~depth:4 ()
