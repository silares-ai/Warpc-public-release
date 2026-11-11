(* The CUTLASS-orientation pair: a cluster of two CTAs along M shares one
   tcgen05.mma.cta_group::2 of M = 256, each CTA holding 128 rows of A, 64
   rows of B^T and 128 rows of the result. C = A B^T, f16 in, f32 out through
   staging copies and the copy engine.

   Warp 0 loads, warp 1 issues the pair's MMAs (the leader's does), warps 4-7
   store; warp 1 also sets up the barriers and tensor memory. The ring runs
   on across tiles. Every memory operand is a buffer at coordinates: this
   CTA's ring stage, the leader's barrier, a warp's staging copy. *)

open Core

let tile_k = 64
let sms = 148

let build ~m ~n ~k ~depth () =
  let cluster = 2 and elem = 2 and out_elem = 4 and epi_warps = 4 in
  let atom = Atom.umma ~ab:F16 ~acc:F32 ~m:256 ~n:128 ~ctas:cluster ~a_major:K_major ~b_major:K_major ~a_src:Smem_desc in
  let mk = Atom.umma_k atom in
  (* the rows of each operand, and of the result, the atom gives a CTA *)
  let a_rows = snd (Atom.cta_rows (Atom.umma_a atom) ~cols:mk ~v:0) in
  let b_rows = snd (Atom.cta_rows (Atom.umma_b atom) ~cols:mk ~v:0) in
  let tile_m = snd (Atom.cta_rows (Atom.umma_c atom) ~cols:atom.n ~v:0) and tile_n = atom.n in
  (* B^T's rows split over the pair: rank v loads the atom's rows v on *)
  let b_stride = fst (Atom.cta_rows (Atom.umma_b atom) ~cols:mk ~v:1) - fst (Atom.cta_rows (Atom.umma_b atom) ~cols:mk ~v:0) in
  if m mod (cluster * tile_m) <> 0 || n mod tile_n <> 0 || k mod tile_k <> 0 then failwith "gemm_pair: the shape is a whole number of cluster tiles";
  let l_a = Atom.swizzled_rows ~rows:a_rows ~cols:tile_k ~elem and l_b = Atom.swizzled_rows ~rows:b_rows ~cols:tile_k ~elem in
  let box_a = Atom.tma_box l_a ~elem and box_b = Atom.tma_box l_b ~elem in
  let span = Atom.tmem_acc_columns atom in
  let plan =
    Store.staged_plan ~image:(Atom.tmem_acc atom) ~trows:tile_m ~tcols:tile_n ~reps:1 ~span ~lanes_are_cols:false ~srows:32 ~scols:32
      ~elem:out_elem ~copies:2
  in
  let l_c = Atom.swizzled_rows ~rows:32 ~cols:32 ~elem:out_elem in
  let box_c = Atom.tma_box l_c ~elem:out_elem in
  (* the pair's whole stage lands on the leader's barrier *)
  let tx = cluster * (Mem.footprint l_a ~elem + Mem.footprint l_b ~elem) in
  let kt = k / tile_k in
  let acc_cols = Atom.tmem_columns span in
  let tiles_m = m / tile_m and tiles_n = n / tile_n in
  let tiles = Tiles.origin ~tiles_m ~tiles_n ~tile_m ~tile_n ~pair:cluster ~pairs:1 ~pair_on_cols:false ~group:8 () in
  let ntiles = tiles.ntiles in
  let grid = if ntiles <= sms then ntiles else sms / cluster * cluster in
  let sync_epi () = bar_sync_n ~bar:1 ~count:(32 * (1 + epi_warps)) in

  let p = Prog.create () in
  let tid = r "tid" and tile_r = r "tile" and lane0 = Core.p "lane0" in
  let gdesc = ur2 "gdesc" and map_c = ur2 "map_c" and map_a = ur2 "map_a" and map_b = ur2 "map_bt" in
  (* the next grid in the stream may begin as soon as this one has *)
  Prog.emits p
    [ preexit (); stack_pointer (); s2r_tid tid; ldcu64 gdesc gmem_desc_off
    ; ldcu64 map_c (param_off 0); ldcu64 map_a (param_off 1); ldcu64 map_b (param_off 2) ];
  let km = Mem.create ~cluster:(cluster, 1) p in
  let lead = Mem.peer km ~x:0 () and rank = Mem.rank km in
  let warpid, laneid = Prog.warp_lane p ~tid ~warps:8 in
  let warp = Prog.reg warpid in
  let leader = up "leader" and p_leader = Core.p "is_leader" in
  Prog.emits p
    [ s2r_ctaid tile_r ~axis:"X"; isetp_eq_u32 lane0 (Prog.reg laneid) 0; uisetp_eq0 leader (Prog.reg rank); plop3_up p_leader leader ];
  let row tile = Mem.of_map ~e:tiles.e_row ~f:tiles.row_of tile and col tile = Mem.of_map ~e:tiles.e_col ~f:tiles.col_of tile in

  (* shared memory: the rings of A and B^T tiles, a row of staging copies a
     storing warp, the barriers, the word the accumulator's address is left in *)
  let sa = Mem.ring km ~name:"sa" ~depth ~elem l_a in
  let sb = Mem.ring km ~name:"sbt" ~depth ~elem l_b in
  let staging = Mem.shared km ~name:"staging" ~modes:[ epi_warps; plan.copies ] ~elem:out_elem l_c in
  (* a stage is filled by the leader's one arrival stating the pair's bytes,
     freed by one commit; the accumulator filled by one commit, freed once
     every storing warp of the pair has released it *)
  let stages = Pipe.make km ~name:"stage" ~depth ~fills:1 ~frees:1 in
  let accs = Pipe.make km ~name:"acc" ~depth:1 ~fills:1 ~frees:(epi_warps * cluster) in
  let slot = Mem.words km ~name:"tmem_word" ~n:1 ~bytes:4 in
  let at buf c = Mem.at buf [ c ] in
  (* the pair: where its commits land *)
  let mask = Mem.commit_mask km [ X ] in

  (* warp 3 initialises the barriers while warp 1 allocates the
     accumulator; the cluster's barrier publishes both -- the leader's MMAs
     write the peer's tensor memory, so the peer's allocation comes before
     the arrival *)
  Prog.role p ~name:"init" ~skip:(fun q -> isetp_ne_u32 q warp 3) (fun () ->
    Prog.emits p [ utmacctl_pf ~map:map_c; utmacctl_pf ~map:map_a; utmacctl_pf ~map:map_b ];
    Mem.init_barriers km);
  Prog.role p ~name:"alloc" ~skip:(fun q -> isetp_ne_u32 q warp 1) (fun () ->
    let tm = Mem.tmem_alloc km ~two:true ~ncols:acc_cols () and w = r "tmem_word" in
    Prog.emit p (mov_ur w tm);
    Mem.sts km (at slot (C 0)) ~data:w;
    Prog.emit p (uvirtcount_dealloc ()));
  Prog.emits p [ cluster_arrive_inits (); cluster_wait () ];
  (* the grids before this one have finished, their memory visible, before
     the first load or store *)
  Prog.emit p (acqbulk ());

  (* warp 0: the loads, into this CTA's stage, reporting to the leader's
     barrier; the leader states the pair's bytes *)
  Prog.role p ~name:"producer" ~skip:(fun q -> isetp_ne_u32 q warp 0) (fun () ->
    let txr = r "tx" in
    Prog.emit p (mov_imm txr tx);
    let out = Pipe.producer km stages in
    Prog.stride p ~name:"tiles" ~reg:tile_r ~until:ntiles ~step:grid (fun tile ->
      let b_row = Mem.plus (col tile) (Mem.scale b_stride (Mem.idx rank)) in
      Prog.for_ p ~name:"k" ~count:(Times kt) (fun kk ->
        Pipe.produce out (fun () ->
          Pipe.wait_empty out;
          let not_leader = Prog.fresh "NOT_LEADER" in
          Prog.emit p (bra ~neg:true p_leader not_leader);
          Prog.when_ p lane0 (fun () -> Mem.arrive_tx km (Pipe.full out) ~tx:txr);
          Prog.label p not_leader;
          Mem.tma_load km ~two:true ~map:map_a ~dst:(Pipe.slot_p out sa) ~bar:(Pipe.full ~peer:lead out) ~x:Mem.(scale box_a.box_cols (idx kk)) ~y:(row tile) ();
          Mem.tma_load km ~two:true ~map:map_b ~dst:(Pipe.slot_p out sb) ~bar:(Pipe.full ~peer:lead out) ~x:Mem.(scale box_b.box_cols (idx kk)) ~y:b_row ())));
    (* before leaving, every stage this CTA filled has been released: the
       commits that release them are the arrivals still in flight towards it *)
    Prog.for_ p ~name:"drain" ~count:(Times depth) (fun _ -> Pipe.produce out (fun () -> Pipe.wait_empty out));
    Prog.emit p (exit ()));

  (* warp 1: the pair's MMAs, the leader's to issue *)
  Prog.role p ~name:"mma" ~skip:(fun q -> isetp_ne_u32 q warp 1) (fun () ->
    let done_ = Prog.fresh "MMA_DONE" in
    Prog.emits p [ sync_epi (); bra ~neg:true p_leader done_ ];
    let acc_r = r "acc_r" and acc = ur "acc" in
    Mem.lds km acc_r (at slot (C 0));
    Prog.emit p (r2ur acc acc_r);
    let regs = Mem.mma_regs km ~atom ~la:l_a ~lb:l_b in
    let first = ur "first" and accum = up "accumulate" in
    let inp = Pipe.consumer km stages and acc_out = Pipe.producer km accs in
    Prog.stride p ~name:"tiles" ~reg:tile_r ~until:ntiles ~step:grid (fun _ ->
      Pipe.produce acc_out (fun () ->
        Pipe.wait_empty acc_out;
        Prog.emit p (umov first 0);
        Prog.for_ p ~name:"k" ~count:(Times kt) (fun _ ->
          Pipe.consume inp (fun () ->
            Prog.emit p (uisetp_ne accum first);
            Pipe.wait_full inp;
            for j = 0 to (tile_k / mk) - 1 do
              Mem.mma km ~leader regs ~a:(Mem.sub (Pipe.slot_c inp sa) (0, j * mk)) ~b:(Mem.sub (Pipe.slot_c inp sb) (0, j * mk)) ~d:(acc, 0)
                ~accumulate:(if j = 0 then `If accum else `Always)
            done;
            Prog.emit p (umov first 1);
            (* the stage is free in both CTAs once these MMAs have read it *)
            Mem.commit2_mc km ~leader (Pipe.empty inp) ~mask));
        Mem.commit2_mc km ~leader (Pipe.full acc_out) ~mask));
    (* the last accumulator has been read out in both CTAs before leaving *)
    Pipe.produce acc_out (fun () -> Pipe.wait_empty acc_out);
    Prog.label p done_;
    (* every warp that reads tensor memory is done with it: free it *)
    let base = ur "free_base" and fw = r "free_word" in
    Prog.emit p (sync_epi ());
    Mem.lds km fw (at slot (C 0));
    Prog.emit p (r2ur base fw);
    Mem.tmem_free km ~base ~ncols:acc_cols;
    Prog.emit p (exit ()));

  (* warps 4-7: each tile's accumulator through the staging copies to C; once
     the last block is out of tensor memory, it is released on the leader's
     barrier *)
  Prog.role p ~name:"epilogue" ~skip:(fun q -> isetp_lt_u32 q warp 4) (fun () ->
    let acc_r = r "acc_r" and acc = ur "acc" in
    Prog.emit p (sync_epi ());
    Mem.lds km acc_r (at slot (C 0));
    Prog.emit p (r2ur acc acc_r);
    let acc_in = Pipe.consumer km accs in
    let data = rn (2 * plan.s_n) "data" in
    let half i = slice data (i mod 2 * plan.s_n) plan.s_n in
    let blocks = Store.staged_blocks plan in
    let nblk = List.length blocks in
    Prog.stride p ~name:"tiles" ~reg:tile_r ~until:ntiles ~step:grid (fun tile -> Pipe.consume acc_in (fun () ->
      let out = Mem.staged plan ~staging ~acc ~map:map_c ~row:(row tile) ~col:(col tile) ~warp:warpid ~lane:laneid in
      Pipe.wait_full acc_in;
      (* each block loaded while the one before it is staged and stored *)
      Mem.ldtm_staged km out (List.hd blocks) (half 0);
      List.iteri
        (fun i b ->
          let copy = i mod plan.copies in
          (* the copy this block goes into was last read by the store
             [copies] blocks back; stores finish in order *)
          Prog.emit p (depbar_le ~n:(plan.copies - 1));
          for v = 0 to (plan.s_n / plan.vec) - 1 do Mem.sts_staged km out ~slot:copy v (half i) done;
          if i + 1 < nblk then Mem.ldtm_staged km out (List.nth blocks (i + 1)) (half (i + 1));
          if i = nblk - 1 then Prog.when_ p lane0 (fun () -> Mem.arrive_red km (Pipe.empty ~peer:lead acc_in));
          (* the writes land before the copy engine reads them: the fence
             publishes them to the async proxy *)
          Prog.emits p [ fence_view_async (); warpsync () ];
          Mem.tma_store km out ~slot:copy b;
          Prog.emit p (utmacmdflush ()))
        blocks));
    (* the last stores have read their staging copies before the warp leaves *)
    Prog.emits p [ depbar_drain (); sync_epi (); exit () ]);
  Prog.emit p (exit ());

  Mem.finish km ~name:(Printf.sprintf "pair_%d_%d_%d_s%d" m n k depth)
    ~params:
      [ Tmap { pname = "c"; rows = m; cols = n; elem = out_elem; box_rows = box_c.box_rows; box_cols = box_c.box_cols; swizzle = box_c.box_swizzle }
      ; Tmap { pname = "a"; rows = m; cols = k; elem; box_rows = box_a.box_rows; box_cols = box_a.box_cols; swizzle = box_a.box_swizzle }
      ; Tmap { pname = "bt"; rows = n; cols = k; elem; box_rows = box_b.box_rows; box_cols = box_b.box_cols; swizzle = box_b.box_swizzle }
      ]
    ~grid:(grid, 1) ~tile:(tile_m, tile_n, tile_k)
    ~notes:[ Printf.sprintf "gemm_pair %dx%dx%d, ring depth %d, a 2x1 cluster sharing a cta_group::2 M=256 MMA" m n k depth ]
    ()
