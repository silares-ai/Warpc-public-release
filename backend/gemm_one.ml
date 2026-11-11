(* A GEMM: one CTA per 128 x 128 output tile, C = A B^T with A M x K and B^T
   N x K (f16, K-major), C f32 row-major. The MMA takes the B^T tile as its A
   operand, so tensor-memory lanes are output columns and each epilogue
   register a row: the store goes straight from the registers. Warps 0-3
   store, warp 4 loads, warp 5 issues the MMAs and owns the barriers and
   tensor memory.

   The loads reach the MMAs through a pipe; every memory operand is a buffer
   at coordinates, so no line below computes an address. *)

open Core

let tile_m = 128
let tile_n = 128
let tile_k = 64

let build ~m ~n ~k ~depth () =
  if m mod tile_m <> 0 || n mod tile_n <> 0 || k mod tile_k <> 0 then failwith "gemm_one: the shape is a whole number of tiles";
  let elem = 2 and out_elem = 4 in
  (* the instruction, and the layouts it fixes *)
  let atom = Atom.umma ~ab:F16 ~acc:F32 ~m:tile_n ~n:tile_m ~ctas:1 ~a_major:K_major ~b_major:K_major ~a_src:Smem_desc in
  let mk = Atom.umma_k atom in
  let l_bt = Atom.swizzled_rows ~rows:tile_n ~cols:tile_k ~elem and l_a = Atom.swizzled_rows ~rows:tile_m ~cols:tile_k ~elem in
  let box_a = Atom.tma_box l_a ~elem and box_bt = Atom.tma_box l_bt ~elem in
  let kt = k / tile_k in
  let acc_cols = Atom.tmem_columns (Atom.tmem_acc_columns atom) in
  (* the tile a CTA owns: CTA t takes tile row t / (n / tile_n), column t mod it *)
  let tiles = Tiles.origin ~tiles_m:(m / tile_m) ~tiles_n:(n / tile_n) ~tile_m ~tile_n ~pair:1 ~pairs:1 ~pair_on_cols:false () in

  let p = Prog.create () in
  (* what the hardware and the launch give *)
  let tid = r "tid" and ctaid = r "ctaid" and lane0 = Core.p "lane0" in
  let gdesc = ur2 "gdesc" and c_ptr = ur2 "c_ptr" and map_a = ur2 "map_a" and map_bt = ur2 "map_bt" in
  (* the next grid in the stream may begin as soon as this one has *)
  Prog.emits p
    [ preexit (); stack_pointer (); s2r_tid tid; ldcu64 gdesc gmem_desc_off
    ; ldcu64 c_ptr (param_off 0); ldcu64 map_a (param_off 1); ldcu64 map_bt (param_off 2) ];
  let km = Mem.create p in
  let warpid, laneid = Prog.warp_lane p ~tid ~warps:6 in
  let warp = Prog.reg warpid in
  Prog.emits p [ isetp_eq_u32 lane0 (Prog.reg laneid) 0; s2r_ctaid ctaid ~axis:"X" ];
  let tile = Prog.fixed p ~name:"tile" ~range:tiles.ntiles ~cls:General ctaid in
  let row = Mem.of_map ~e:tiles.e_row ~f:tiles.row_of tile and col = Mem.of_map ~e:tiles.e_col ~f:tiles.col_of tile in

  (* shared memory: the operand rings and the pipe their stages go through,
     the barrier the accumulator is ready on, the word its address is left in *)
  let sbt = Mem.ring km ~name:"sbt" ~depth ~elem l_bt in
  let sa = Mem.ring km ~name:"sa" ~depth ~elem l_a in
  let stages = Pipe.make km ~name:"stage" ~depth ~fills:1 ~frees:1 in
  let ready = Mem.barriers km ~name:"ready" ~n:1 ~count:1 in
  let slot = Mem.words km ~name:"tmem_word" ~n:1 ~bytes:4 in
  let at buf c = Mem.at buf [ c ] in
  let tx = Mem.footprint l_bt ~elem + Mem.footprint l_a ~elem in
  let sync_epi () = bar_sync_n ~bar:1 ~count:(32 * 5) in

  (* warp 5 initialises the barriers; the CTA's barrier publishes them *)
  Prog.role p ~name:"init" ~skip:(fun q -> isetp_ne_u32 q warp 5) (fun () ->
    Prog.emits p [ utmacctl_pf ~map:map_a; utmacctl_pf ~map:map_bt ];
    Mem.init_barriers km);
  Prog.emit p (bar_sync_inits ());
  (* the grids before this one have finished, their memory visible, before
     the first load or store *)
  Prog.emit p (acqbulk ());

  (* warp 4: the loads *)
  Prog.role p ~name:"producer" ~skip:(fun q -> isetp_ne_u32 q warp 4) (fun () ->
    let txr = r "tx" in
    Prog.emit p (mov_imm txr tx);
    let out = Pipe.producer km stages in
    Prog.for_ p ~name:"k" ~count:(Times kt) (fun kk ->
      Pipe.produce out (fun () ->
        Pipe.wait_empty out;
        Prog.when_ p lane0 (fun () -> Mem.arrive_tx km (Pipe.full out) ~tx:txr);
        Mem.tma_load km ~map:map_bt ~dst:(Pipe.slot_p out sbt) ~bar:(Pipe.full out) ~x:Mem.(scale box_bt.box_cols (idx kk)) ~y:col ();
        Mem.tma_load km ~map:map_a ~dst:(Pipe.slot_p out sa) ~bar:(Pipe.full out) ~x:Mem.(scale box_a.box_cols (idx kk)) ~y:row ()));
    Prog.emit p (exit ()));

  (* warp 5: tensor memory and the MMAs *)
  Prog.role p ~name:"mma" ~skip:(fun q -> isetp_ne_u32 q warp 5) (fun () ->
    let tmem = Mem.tmem_alloc km ~ncols:acc_cols () and word = r "tmem_word" in
    Prog.emit p (mov_ur word tmem);
    Mem.sts km (at slot (C 0)) ~data:word;
    Prog.emits p [ uvirtcount_dealloc (); sync_epi () ];
    let regs = Mem.mma_regs km ~atom ~la:l_bt ~lb:l_a in
    let first = ur "first" and accum = up "accumulate" in
    Prog.emit p (umov first 0);
    let inp = Pipe.consumer km stages in
    Prog.for_ p ~name:"k" ~count:(Times kt) (fun _ ->
      Pipe.consume inp (fun () ->
        Prog.emit p (uisetp_ne accum first);
        Pipe.wait_full inp;
        (* the stage's k steps, one MMA each; the tile's first overwrites *)
        for j = 0 to (tile_k / mk) - 1 do
          Mem.mma km regs ~a:(Mem.sub (Pipe.slot_c inp sbt) (0, j * mk)) ~b:(Mem.sub (Pipe.slot_c inp sa) (0, j * mk)) ~d:(tmem, 0)
            ~accumulate:(if j = 0 then `If accum else `Always)
        done;
        Prog.emit p (umov first 1);
        Mem.commit km (Pipe.empty inp)));
    Mem.commit km (at ready (C 0));
    (* the epilogue has read the accumulator out: free it *)
    Prog.emit p (sync_epi ());
    Mem.tmem_free km ~base:tmem ~ncols:acc_cols;
    Prog.emit p (exit ()));

  (* warps 0-3: the accumulator out, straight to C *)
  Prog.role p (fun () ->
    let acc_r = r "acc_r" and acc = ur "acc" and parity = r "parity" in
    Prog.emit p (sync_epi ());
    Mem.lds km acc_r (at slot (C 0));
    Prog.emits p [ r2ur acc acc_r; mov_imm parity Barrier.full_parity ];
    Mem.wait km (at ready (C 0)) ~parity;
    let dp =
      Store.direct_plan ~image:(Atom.tmem_acc atom) ~trows:tile_n ~tcols:tile_m ~reps:1 ~span:(Atom.tmem_acc_columns atom) ~cols:n
        ~elem:out_elem
    in
    let out = Mem.direct dp ~acc ~mat:(`UR c_ptr) ~desc:gdesc ~row ~col ~warp:warpid ~lane:laneid in
    (* each block loaded into one half of the registers while the block
       before it is stored from the other, a store a register *)
    let data = rn (2 * dp.n) "data" in
    let half i = slice data (i mod 2 * dp.n) dp.n in
    let blocks = Store.blocks dp in
    Mem.ldtm km out (List.hd blocks) (half 0);
    List.iteri
      (fun i b ->
        for v = 0 to dp.n - 1 do Mem.stg km out b v (part (half i) v) done;
        if i + 1 < List.length blocks then Mem.ldtm km out (List.nth blocks (i + 1)) (half (i + 1)))
      blocks;
    Prog.emits p [ sync_epi (); exit () ]);

  Mem.finish km ~name:(Printf.sprintf "one_%d_%d_%d_s%d" m n k depth)
    ~params:
      [ Ptr { pname = "c"; rows = m; cols = n; elem = out_elem }
      ; Tmap { pname = "a"; rows = m; cols = k; elem; box_rows = box_a.box_rows; box_cols = box_a.box_cols; swizzle = box_a.box_swizzle }
      ; Tmap { pname = "bt"; rows = n; cols = k; elem; box_rows = box_bt.box_rows; box_cols = box_bt.box_cols; swizzle = box_bt.box_swizzle }
      ]
    ~grid:(tiles.ntiles, 1) ~tile:(tile_m, tile_n, tile_k)
    ~notes:[ Printf.sprintf "gemm_one %dx%dx%d, ring depth %d, one CTA a 128x128 tile" m n k depth ]
    ()
