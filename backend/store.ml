(* The accumulator's way out, as plans: what each warp loads from tensor
   memory and where each register goes, read off the layouts --
   the load's fragment composed with the accumulator's image, taken to output
   coordinates and through the matrix's layout. *)

open Layouts

(* [rows] x [cols] indices read as the index of the transposed [cols] x [rows] *)
let transpose ~rows ~cols : (Space.logical, Space.logical) Layout.t =
  Layout.of_linear (Group [ Axis { size = rows; stride = 1 }; Axis { size = cols; stride = rows } ])

(* A store straight from the registers. With the accumulator transposed -- the
   MMA's A operand is the second matrix -- a warp's lanes stand for 32
   neighbouring output columns and each register for an output row, so one
   store instruction writes 32 contiguous elements of a row. *)
type direct =
  { ld : Expr.t (* the block's tensor-memory address, over the lane quarter "c" *)
  ; imm_ld : int array (* and each of the warp's blocks from it *)
  ; y : Expr.t
  ; imm_y : int array
  ; x : Expr.t
  ; imm_x : int array
  ; lane : Expr.t (* a lane's byte offset in the block, over "laneid" *)
  ; reg_off : int array (* and each register's from it *)
  ; n : int (* registers a load *)
  ; blocks : int (* blocks a warp *)
  ; rep_ld : int array
  ; rep_y : int array
  ; rep_x : int array
  ; cols : int (* the matrix's columns *)
  ; elem : int
  ; ref_ld : int -> int * int -> int
    (* the references, read off the layouts at each point: a quarter's block's
       tensor-memory address from the accumulator's *)
  ; ref_off : int -> int * int -> int -> int -> int
    (* and lane l's register v of it, in bytes from the tile's origin in the matrix *)
  }

let direct_n = 32 (* one tcgen05.ld.32x32b.x32 a block *)

(* [image]: the accumulator's tensor-memory layout, [trows] x [tcols] -- its
   lanes are output columns; [reps] such accumulators stacked [span] columns
   apart; the output matrix has [cols] columns of [elem] bytes. *)
let direct_plan ~image ~trows ~tcols ~reps ~span ~cols ~elem =
  let lanes = Atom.ldtm_block and n = direct_n in
  let frag = Atom.ldtm_32x32b ~n in
  let block = Shape.Product [ Bound lanes; Bound n ] in
  let blocks_w = trows / lanes and blocks_ch = tcols / n in
  let blocked = Layout.divide ~by:block (Layout.of_linear (Linear.canonical (Product [ Bound trows; Bound tcols ]))) in
  let ld =
    Layout.compose
      (Layout.interleave ~by:(Linear.canonical (Product [ Bound blocks_w; Bound blocks_ch ])) frag)
      (Layout.compose blocked image)
  in
  let at w ch l r = Layout.offset ld (Coord.Tuple [ Tuple [ Idx w; Idx ch ]; Tuple [ Idx l; Idx r ] ]) in
  Atom.check_ldtm ~at ~blocks_w ~blocks_ch ~n;
  let quarters = 4 in
  let of_quarter =
    Array.init quarters (fun q ->
      List.concat_map
        (fun w -> List.filter_map (fun ch -> if Atom.ldtm_quarter ~at w ch = q then Some (w, ch) else None) (List.init blocks_ch Fun.id))
        (List.init blocks_w Fun.id))
  in
  let per_warp = List.length of_quarter.(0) in
  if Array.exists (fun l -> List.length l <> per_warp) of_quarter then failwith "Store: blocks not spread over the lane quarters";
  let block_of q j = List.nth of_quarter.(q) j in
  let rt = Shape.Bound quarters in
  let w_of = function Coord.Idx w -> w | Tuple _ -> assert false in
  let ld_e, imm_ld = Addr.split ~name:"tensor-memory load" ~rt ~n:per_warp (fun c j -> let w, ch = block_of (w_of c) j in at w ch 0 0) in
  let out_cols = trows in
  let tile : (Space.logical, Space.logical) Layout.t =
    Layout.compose
      (Layout.divide ~by:block (Layout.of_linear (Linear.canonical (Product [ Bound trows; Bound tcols ]))))
      (transpose ~rows:trows ~cols:tcols)
  in
  let origin_of c j = let w, ch = block_of (w_of c) j in Layout.offset tile (Coord.Tuple [ Tuple [ Idx 0; Idx 0 ]; Tuple [ Idx w; Idx ch ] ]) in
  let y, imm_y = Addr.split ~name:"store row" ~rt ~n:per_warp (fun c j -> origin_of c j / out_cols) in
  let x, imm_x = Addr.split ~name:"store column" ~rt ~n:per_warp (fun c j -> origin_of c j mod out_cols) in
  (* lane l's register r, in the block's output coordinates (n rows of 32
     columns), then through the matrix's row-major layout *)
  let out_block = Layout.compose frag (transpose ~rows:lanes ~cols:n) in
  let matrix = Layout.storage (Group [ Axis { size = n; stride = cols * elem }; Axis { size = lanes; stride = elem } ]) in
  let addr = Layout.compose out_block matrix in
  let lane, reg_off = Addr.split ~name:"register store" ~rt:(Shape.Bound lanes) ~n (fun c r -> Layout.offset addr (Coord.Tuple [ Idx (w_of c); Idx r ])) in
  if Expr.eval lane (fun _ -> 1) - Expr.eval lane (fun _ -> 0) <> elem then failwith "Store: a register's store is not contiguous across the lanes";
  Array.iter (fun d -> if abs d >= 1 lsl 23 then failwith "Store: a register's offset does not fit the store's immediate") reg_off;
  let full_rows = reps * trows in
  let rep_origin r = Layout.offset (transpose ~rows:full_rows ~cols:tcols) (Coord.Tuple [ Idx (r * trows); Idx 0 ]) in
  let rep_y = Array.init reps (fun r -> rep_origin r / full_rows) and rep_x = Array.init reps (fun r -> rep_origin r mod full_rows) in
  let ref_ld q (r, j) = let w, ch = block_of q j in at w ch 0 0 + (r * span) in
  let ref_off q (r, j) l v =
    let o = origin_of (Coord.Idx q) j in
    (((((o / out_cols) + rep_y.(r)) * cols) + (o mod out_cols) + rep_x.(r)) * elem) + Layout.offset addr (Coord.Tuple [ Idx l; Idx v ])
  in
  { ld = ld_e; imm_ld; y; imm_y; x; imm_x; lane = Expr.bind "c" (Expr.var "laneid") lane; reg_off; n; blocks = per_warp
  ; rep_ld = Array.init reps (fun r -> r * span); rep_y; rep_x; cols; elem; ref_ld; ref_off }

(* the lane quarter a warp's tensor-memory loads are confined to *)
let quarter e = Expr.bind "c" (Expr.modulo (Expr.var "warpid") 4) e

(* A buffer only the kernel reads back holds the accumulator in the order its
   stores leave the registers: block after block, a lane's registers four at
   a time, consecutive lanes' fours side by side -- so each 16-byte store and
   load of a warp is one contiguous 512 bytes. The layout is (block, vector,
   lane, element). *)
let frag_layout (p : direct) : (Space.logical, Space.physical) Layout.t =
  let reps = Array.length p.rep_ld and vec = 16 / p.elem in
  let lanes = Atom.ldtm_block in
  Layout.storage
    (Group
       [ Axis { size = reps * p.blocks; stride = lanes * p.n * p.elem }
       ; Axis { size = p.n / vec; stride = lanes * 16 }
       ; Axis { size = lanes; stride = 16 }
       ; Axis { size = vec; stride = p.elem } ])

let frag_at (p : direct) ~blk ~lane ~reg =
  let vec = 16 / p.elem in
  if blk = Array.length p.rep_ld * p.blocks then blk * Atom.ldtm_block * p.n * p.elem
  else Layout.offset (frag_layout p) (Coord.Tuple [ Idx blk; Idx (reg / vec); Idx lane; Idx (reg mod vec) ])

(* this warp's slice of such a buffer: its lane quarter's blocks *)
let frag_lane (p : direct) =
  let e = Addr.form ~name:"fragment lane" ~n:Atom.ldtm_block (fun l -> frag_at p ~blk:0 ~lane:l ~reg:0) in
  Expr.bind "c" (Expr.var "laneid") e

(* ---- through a staging tile and the copy engine ----

   The accumulator's lanes are output rows, or -- when the MMA's A operand is
   the second matrix -- output columns. One tcgen05.ld.32x32b covers 32 lanes
   by n columns: in output coordinates, the staging tile, whose layout is the
   SWIZZLE_128B box the copy engine reads. Each warp has [copies] of it. *)
type staged =
  { s_ld : Expr.t
  ; s_imm_ld : int array
  ; s_copy : Expr.t (* the warp's first copy, over the lane quarter "c" *)
  ; s_y : Expr.t
  ; s_imm_y : int array
  ; s_x : Expr.t
  ; s_imm_x : int array
  ; sts : (Space.thread_value, Space.physical) Layout.t (* lane, register -> staging address *)
  ; s_n : int
  ; vec : int (* elements a shared-memory store *)
  ; s_elem : int
  ; copy : int (* bytes a copy *)
  ; copies : int
  ; s_blocks : int
  ; s_rep_ld : int array
  ; s_rep_y : int array
  ; s_rep_x : int array
  ; s_ref_ld : int -> int * int -> int (* as the direct plan's *)
  ; ref_sts : int -> int -> int (* lane l's register v, in bytes into a staging copy *)
  ; ref_origin : int -> int * int -> int * int (* a quarter's block's first row and column from the tile's origin *)
  }

let footprint l ~elem = elem + List.fold_left (fun m c -> max m (Layout.offset l c)) 0 (Coord.enumerate (Layout.shape l))
let round_up a m = (a + m - 1) / m * m

let staged_plan ~image ~trows ~tcols ~reps ~span ~lanes_are_cols ~srows ~scols ~elem ~copies =
  let box = Atom.swizzled_rows ~rows:srows ~cols:scols ~elem in
  ignore (Atom.tma_box box ~elem);
  let copy = round_up (footprint box ~elem) Atom.sw128_period in
  let lanes = Atom.ldtm_block in
  let out_cols = if lanes_are_cols then trows else tcols in
  let to_out ~rows ~cols = if lanes_are_cols then transpose ~rows ~cols else Layout.of_linear (Linear.canonical (Product [ Bound rows; Bound cols ])) in
  let n = if lanes_are_cols then srows else scols in
  if (if lanes_are_cols then scols else srows) <> lanes then failwith "Store: the staging tile spans the 32 lanes of a warp's quarter";
  let frag = Atom.ldtm_32x32b ~n in
  let block = Shape.Product [ Bound lanes; Bound n ] in
  let blocks_w = trows / lanes and blocks_ch = tcols / n in
  let blocked = Layout.divide ~by:block (Layout.of_linear (Linear.canonical (Product [ Bound trows; Bound tcols ]))) in
  let ld =
    Layout.compose
      (Layout.interleave ~by:(Linear.canonical (Product [ Bound blocks_w; Bound blocks_ch ])) frag)
      (Layout.compose blocked image)
  in
  let at w ch l r = Layout.offset ld (Coord.Tuple [ Tuple [ Idx w; Idx ch ]; Tuple [ Idx l; Idx r ] ]) in
  Atom.check_ldtm ~at ~blocks_w ~blocks_ch ~n;
  let quarters = 4 in
  let of_quarter =
    Array.init quarters (fun q ->
      List.concat_map
        (fun w -> List.filter_map (fun ch -> if Atom.ldtm_quarter ~at w ch = q then Some (w, ch) else None) (List.init blocks_ch Fun.id))
        (List.init blocks_w Fun.id))
  in
  let per_warp = List.length of_quarter.(0) in
  if Array.exists (fun l -> List.length l <> per_warp) of_quarter || per_warp * quarters <> blocks_w * blocks_ch
  then failwith "Store: the accumulator's blocks are not spread evenly over the four lane quarters";
  let block_of q j = List.nth of_quarter.(q) j in
  let rt = Shape.Bound quarters in
  let w_of = function Coord.Idx w -> w | Tuple _ -> assert false in
  let s_ld, s_imm_ld = Addr.split ~name:"tensor-memory load" ~rt ~n:per_warp (fun c j -> let w, ch = block_of (w_of c) j in at w ch 0 0) in
  (* the staging write is the fragment taken to output coordinates and
     composed with the staging tile's layout *)
  let sts = Layout.compose frag (Layout.compose (to_out ~rows:lanes ~cols:n) box) in
  if not (Layout.is_injective sts) then failwith "Store: two values of a fragment land on one address";
  let reg_addr l r = Layout.offset sts (Coord.Tuple [ Idx l; Idx r ]) in
  let fits v =
    n mod v = 0
    && List.for_all
         (fun l ->
           List.for_all
             (fun q ->
               let a0 = reg_addr l (v * q) in
               a0 mod (v * elem) = 0 && List.for_all (fun e -> reg_addr l ((v * q) + e) = a0 + (e * elem)) (List.init v Fun.id))
             (List.init (n / v) Fun.id))
         (List.init lanes Fun.id)
  in
  let vec = List.find fits (List.filter (fun v -> v >= 1) [ 16 / elem; 8 / elem; 4 / elem; 1 ]) in
  let tile : (Space.logical, Space.logical) Layout.t =
    Layout.compose
      (Layout.divide ~by:block (Layout.of_linear (Linear.canonical (Product [ Bound trows; Bound tcols ]))))
      (to_out ~rows:trows ~cols:tcols)
  in
  let origin_of c j = let w, ch = block_of (w_of c) j in Layout.offset tile (Coord.Tuple [ Tuple [ Idx 0; Idx 0 ]; Tuple [ Idx w; Idx ch ] ]) in
  let s_y, s_imm_y = Addr.split ~name:"store row" ~rt ~n:per_warp (fun c j -> origin_of c j / out_cols) in
  let s_x, s_imm_x = Addr.split ~name:"store column" ~rt ~n:per_warp (fun c j -> origin_of c j mod out_cols) in
  let full_rows = reps * trows in
  let full_out_cols = if lanes_are_cols then full_rows else tcols in
  let rep_origin r = Layout.offset (to_out ~rows:full_rows ~cols:tcols) (Coord.Tuple [ Idx (r * trows); Idx 0 ]) in
  let s_rep_y = Array.init reps (fun r -> rep_origin r / full_out_cols) and s_rep_x = Array.init reps (fun r -> rep_origin r mod full_out_cols) in
  let s_ref_ld q (r, j) = let w, ch = block_of q j in at w ch 0 0 + (r * span) in
  let ref_origin q (r, j) = let o = origin_of (Coord.Idx q) j in (o / out_cols) + s_rep_y.(r), (o mod out_cols) + s_rep_x.(r) in
  { s_ld; s_imm_ld; s_copy = Expr.scale (copies * copy) (Expr.var "c"); s_y; s_imm_y; s_x; s_imm_x; sts; s_n = n; vec; s_elem = elem
  ; copy; copies; s_blocks = per_warp
  ; s_rep_ld = Array.init reps (fun r -> r * span); s_rep_y; s_rep_x; s_ref_ld; ref_sts = reg_addr; ref_origin }

(* the blocks a warp reads out, in order: every block of an accumulator
   before the next accumulator's *)
let blocks (p : direct) = List.concat_map (fun r -> List.init p.blocks (fun j -> r, j)) (List.init (Array.length p.rep_ld) Fun.id)
let staged_blocks (p : staged) = List.concat_map (fun r -> List.init p.s_blocks (fun j -> r, j)) (List.init (Array.length p.s_rep_ld) Fun.id)
