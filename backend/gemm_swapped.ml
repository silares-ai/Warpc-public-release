(* The swapped orientation: a CTA pair
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

open Layouts
open Core

let tile_k = 64
let sms = 148
let resident_clusters = function 1 -> sms | 2 -> 74 | 4 -> 33 | 8 -> 15 | c -> failwith (Printf.sprintf "resident_clusters %d" c)
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)

let build ?(clc = false) ?(ask_ahead = false) ?(direct = false) ?(serpentine = false) ?(cluster_n = 1) ?(epi_rows = 8) ?bufs
    ?(group = 8) ?(stamps = false) ?(stage_stamps = false) ?(streamk = false) ?(split = 1) ?(split_grid = 1) ?(split_add = false) ?(late_launch = false) ~tile_m ~tile_n ~depth ~m ~n ~k () =
  if streamk && (clc || cluster_n > 1 || direct = split_add) then
    failwith "gemm_swapped: stream-K takes a pair alone, a fixed grid, and the store from the registers or, adding in the L2, through the staging copies";
  (* split-K in the cluster: its pairs along y take the same tile's K in
     [split] runs, and the pairs past the first leave their partials for the
     first to add *)
  if split > 1 && (split > 4 || cluster_n <> split || clc || streamk || not direct) then
    failwith "gemm_swapped: split-K takes two to four pairs, a fixed grid and the store from the registers";
  (* split-K over the grid: [split_grid] clusters take the same tiles' K in
     runs, each leaves its partial in its slot and raises its flag, and each
     adds up its share of the tiles' blocks from all the partials *)
  if split_grid > 1 && (split > 1 || clc || streamk || direct = split_add) then
    failwith "gemm_swapped: split-K over the grid takes a fixed grid, no split in the cluster, and the store from the registers or, adding in the L2, through the staging copies";
  if split_add && split_grid = 1 && not streamk then failwith "gemm_swapped: adding in the L2 is split-K's, over the grid or stream-K";
  (* [split_add]: each run adds its whole accumulator to C in the L2 (the copy
     engine's reducing store), after the tile's runs have each zeroed their
     share of it; no workspace, and the sum's order is the order the adds
     reach the L2 *)
  let nparams = if (streamk || split_grid > 1) && not split_add then 5 else if split > 1 || split_add then 4 else 3 in
  let pair = 2 and elem = 2 and out_elem = 4 and epi_warps = 4 in
  let ctas = pair * cluster_n in
  (* the MMA: its rows are the pair's output columns, stacked in atoms of at
     most 256 *)
  let rows_needed = tile_n * pair in
  let atom_m = min rows_needed 256 in
  if rows_needed mod atom_m <> 0 then failwith "gemm_swapped: the MMA's rows are not whole atoms";
  let reps = rows_needed / atom_m in
  let atom = Atom.umma ~ab:F16 ~acc:F32 ~m:atom_m ~n:tile_m ~ctas:pair ~a_major:K_major ~b_major:K_major ~a_src:Smem_desc in
  let mk = Atom.umma_k atom in
  let rows_of l = snd (Atom.cta_rows l ~cols:mk ~v:0) in
  let start_of l v = fst (Atom.cta_rows l ~cols:mk ~v) in
  (* the MMA's A operand, the tile of B^T: every stacked block's rows; its B
     operand, the tile of A: the pair's split of the atom's N *)
  let a_rows = rows_of (Atom.umma_a atom) in
  let sb_rows = reps * a_rows and sa_rows = rows_of (Atom.umma_b atom) in
  let acc_rows = snd (Atom.cta_rows (Atom.umma_c atom) ~cols:atom.n ~v:0) and acc_cols = atom.n in
  if reps * acc_rows <> tile_n || acc_cols <> tile_m then failwith "gemm_swapped: the accumulator is not the CTA's tile";
  (* a tile past the matrix's edge is loaded zero-filled and stored clipped
     by the copy engine; the store from the registers has no such bound *)
  let tiles_m = (m + tile_m - 1) / tile_m and tiles_n = (n + tile_n - 1) / tile_n in
  let sharing = if split > 1 then 1 else cluster_n in
  if tiles_m mod sharing <> 0 || tiles_n mod pair <> 0 || k mod (tile_k * split) <> 0 || k / tile_k < split_grid then failwith "gemm_swapped: the tiles are not a whole number of clusters";
  if direct && (m mod tile_m <> 0 || n mod tile_n <> 0) then failwith "gemm_swapped: the store from the registers needs whole tiles";
  (* A's rows split over the pair: rank x loads its share of the atom's N *)
  let a_stride = start_of (Atom.umma_b atom) 1 - start_of (Atom.umma_b atom) 0 in
  let l_sa = Atom.swizzled_rows ~rows:sa_rows ~cols:tile_k ~elem and l_sb = Atom.swizzled_rows ~rows:sb_rows ~cols:tile_k ~elem in
  let slice_box, _, _ = Atom.load_slice l_sb ~elem ~group:sharing in
  let box_sa = Atom.tma_box l_sa ~elem and box_sb = Atom.tma_box slice_box ~elem in
  let acc_image = Atom.tmem_acc atom and span = Atom.tmem_acc_columns atom in
  let ncols = Atom.tmem_columns (reps * span) in
  let total = tiles_m * tiles_n in
  let nbuf = match bufs with Some b -> b | None -> if 2 * ncols <= 512 && total > sms then 2 else 1 in
  if ncols * nbuf > 512 then failwith "gemm_swapped: tensor memory";
  let alloc_cols = Atom.tmem_columns (ncols * nbuf) in
  let persistent = split = 1 && split_grid = 1 && (clc || total > sms) in
  let grid =
    if streamk then sms / ctas * ctas
    else if split > 1 then total * split
    else if split_grid > 1 then total * split_grid
    else if clc || total <= sms then total
    else
      (* a fixed grid walks the tiles by stride: every cluster resident *)
      resident_clusters ctas * ctas
  in
  (* a CTA waits on the partials of others: every CTA is resident at once *)
  if split_grid > 1 && grid / ctas > resident_clusters ctas then failwith "gemm_swapped: split-K over the grid takes more clusters than are resident at once";
  let l_c = Atom.swizzled_rows ~rows:epi_rows ~cols:32 ~elem:out_elem in
  let plan =
    if direct then None
    else
      Some
        (Store.staged_plan ~image:acc_image ~trows:acc_rows ~tcols:acc_cols ~reps ~span ~lanes_are_cols:true ~srows:epi_rows ~scols:32
           ~elem:out_elem ~copies:2)
  in
  let dp = Store.direct_plan ~image:acc_image ~trows:acc_rows ~tcols:acc_cols ~reps ~span ~cols:n ~elem:out_elem in
  (* a stage lands on the leader's barrier: the pair's whole share *)
  let tx = pair * (Mem.footprint l_sa ~elem + Mem.footprint l_sb ~elem) in
  let kt = k / tile_k / split in
  (* split over the grid, run s of K is [s kt / S, (s + 1) kt / S), CTA c's
     run s = (c / ctas) mod S: s q + s r / S, kt = q S + r *)
  let run_of c =
    let s = Expr.modulo (Expr.div c ctas) split_grid in
    let at s = Expr.add (Expr.scale (kt / split_grid) s) (Expr.div (Expr.scale (kt mod split_grid) s) split_grid) in
    s, at s, Expr.add (at (Expr.add s (Expr.Const 1))) (Expr.scale (-1) (at s))
  in
  let even_runs = kt mod split_grid = 0 in
  let tiles = Tiles.origin ~tiles_m ~tiles_n ~tile_m ~tile_n ~pair ~pairs:sharing ~pair_on_cols:true ~group ~serpentine () in
  (* split, a cluster's pairs share its tile: CTA c is pair c / ctas's
     partner c mod pair *)
  let tiles =
    if split_grid > 1 then begin
      (* split over the grid, cluster c / ctas is tile cluster c / (ctas S)'s
         run (c / ctas) mod S *)
      let c = Expr.var "c" in
      let i = Expr.add (Expr.scale ctas (Expr.div c (ctas * split_grid))) (Expr.modulo c ctas) and f c = (c / (ctas * split_grid) * ctas) + (c mod ctas) in
      Tiles.{ e_row = Expr.bind "c" i tiles.e_row; e_col = Expr.bind "c" i tiles.e_col; row_of = (fun c -> tiles.row_of (f c)); col_of = (fun c -> tiles.col_of (f c)); ntiles = grid }
    end
    else if split = 1 then tiles
    else begin
      let c = Expr.var "c" in
      let i = Expr.add (Expr.scale pair (Expr.div c ctas)) (Expr.modulo c pair) and f c = (c / ctas * pair) + (c mod pair) in
      Tiles.{ e_row = Expr.bind "c" i tiles.e_row; e_col = Expr.bind "c" i tiles.e_col; row_of = (fun c -> tiles.row_of (f c)); col_of = (fun c -> tiles.col_of (f c)); ntiles = grid }
    end
  in
  let ntiles = tiles.ntiles in
  (* the warps that take tiles from the scheduler, across the cluster: in a
     pair only the leader's MMA warp does *)
  let takers = ctas + cluster_n + (epi_warps * ctas) in
  let sync_epi () = bar_sync_n ~bar:1 ~count:(32 * (1 + epi_warps)) in

  let p = Prog.create () in
  (* a timing build's marks, a construct each: [st_tile] in the slot of the
     role's tile -- stream-K's segment, or the tiles the role has counted --
     and [st_first] only on a k loop's first pass *)
  let st ev = if stamps then Stamp.at p ~nparams ~ev () in
  let seg_t = r "seg_tile" in
  let tile_count = ref None in
  let st_tile ev =
    if stamps then
      match streamk, !tile_count with
      | true, _ -> Stamp.at p ~nparams ~ev ~count:seg_t ()
      | false, Some c -> Stamp.at p ~nparams ~ev ~count:c ()
      | false, None -> st ev
  in
  let st_first (kk : (Prog.counter, r) Prog.index) ev =
    if stamps then begin
      let skip = Prog.fresh "STAMP_SKIP" and pf = Core.p "stamp_first" in
      Prog.constructs p [ isetp_ne_u32 pf (Prog.reg kk) 1; bra pf skip ];
      st_tile ev;
      Prog.label p skip
    end
  in

  let tid = r "tid" and tile_r = r "tile" in
  let lane0 = Core.p "lane0" and more = Core.p "more" in
  let gdesc = ur2 "gdesc" and map_c = ur2 "map_c" and map_a = ur2 "map_a" and map_b = ur2 "map_bt" in
  (* the next grid in the stream may begin as soon as this one has; a timing
     build's first mark is a store, so it waits for the grids before here *)
  (* [late_launch]: the producer signals once its last load is out, so the
     next grid's CTAs are not waiting on idle SMs through this one *)
  Prog.emits p ((if late_launch then [] else [ preexit () ]) @ [ stack_pointer () ]);
  if stamps then Prog.emit p (acqbulk ());
  st 0;
  Prog.emits p
    [ s2r_tid tid; ldcu64 gdesc gmem_desc_off; ldcu64 map_c (param_off 0); ldcu64 map_a (param_off 1); ldcu64 map_b (param_off 2) ];
  let km = Mem.create ~cluster:(pair, cluster_n) p in
  let lead = Mem.peer km ~x:0 () and first_cta = Mem.peer km ~x:0 ~y:0 () in
  let warpid, laneid = Prog.warp_lane p ~tid ~warps:8 in
  let warp = Prog.reg warpid in
  let leader = up "leader" and p_leader = Core.p "is_leader" in
  Prog.emits p
    [ s2r_ctaid tile_r ~axis:"X"; isetp_eq_u32 lane0 (Prog.reg laneid) 0
    ; uisetp_eq0 leader (Mem.get km ~cls:Uniform (Mem.coord_of km X)); plop3_up p_leader leader ];
  let row tile = Mem.of_map ~e:tiles.e_row ~f:tiles.row_of tile and col tile = Mem.of_map ~e:tiles.e_col ~f:tiles.col_of tile in

  (* shared memory: the rings of A and B^T tiles, a row of staging copies a
     storing warp, the pipes, the words the accumulators' addresses are left
     in, the scheduler's answer *)
  let sa = Mem.ring km ~name:"sa" ~depth ~elem l_sa in
  let sb = Mem.ring km ~name:"sbt" ~depth ~elem l_sb in
  let staging = Option.map (fun (pl : Store.staged) -> Mem.shared km ~name:"staging" ~modes:[ epi_warps; pl.copies ] ~elem:out_elem l_c) plan in
  (* a stage is filled by the leader's one arrival stating the pair's bytes
     and freed by one commit from each pair's leader; an accumulator filled by
     one commit and freed by every storing warp of the pair; the answer
     filled by the scheduler's request and freed by every taker *)
  let stages = Pipe.make km ~name:"stage" ~depth ~fills:1 ~frees:sharing in
  let accs = Pipe.make km ~name:"acc" ~depth:nbuf ~fills:1 ~frees:(epi_warps * pair) in
  let answers = if clc then Some (Pipe.make km ~name:"answer" ~depth:1 ~fills:1 ~frees:takers) else None in
  let slot = Mem.words km ~name:"tmem_word" ~n:nbuf ~bytes:4 in
  (* split, the first pair's barrier the other pairs' partials report to:
     each storing warp's arrival *)
  let partials = if split > 1 then Some (Mem.barriers km ~name:"partials" ~n:1 ~count:((split - 1) * epi_warps)) else None in
  let answer = Mem.words km ~name:"answer" ~n:1 ~bytes:16 in
  let at buf c = Mem.at buf [ c ] in
  (* the cluster's CTAs refilling a stage, a pair's accumulator *)
  let mask_all = Mem.commit_mask km [ X; Y ] and mask_pair = Mem.commit_mask km [ X ] in

  (* warp 3 initialises the barriers while warp 1 allocates the
     accumulators, leaving their addresses in the slot words *)
  Prog.role p ~name:"init" ~skip:(fun q -> isetp_ne_u32 q warp 3) (fun () ->
    Prog.emits p [ utmacctl_pf ~map:map_c; utmacctl_pf ~map:map_a; utmacctl_pf ~map:map_b ];
    Mem.init_barriers km);
  (* the cluster's barrier publishes the allocations too: the leader's MMAs
     write the peer's tensor memory. One allocation for every accumulator,
     buffer b its columns from b ncols on: each allocation is a round trip
     between the pair's CTAs. *)
  Prog.role p ~name:"alloc" ~skip:(fun q -> isetp_ne_u32 q warp 1) (fun () ->
    let tm = Mem.tmem_alloc km ~two:true ~ncols:alloc_cols () in
    for b = 0 to nbuf - 1 do
      let w = r "tmem_word" in
      if b = 0 then Prog.emit p (mov_ur w tm)
      else begin
        let tb = ur "tmem_buf" in
        Prog.emits p [ uiadd3 tb tm (b * ncols); mov_ur w tb ]
      end;
      Mem.sts km (at slot (C b)) ~data:w
    done;
    Prog.emit p (uvirtcount_dealloc ()));
  Prog.emits p [ cluster_arrive_inits (); cluster_wait () ];
  (* the grids before this one have finished, their memory visible, before
     the first load or store *)
  if not stamps then Prog.emit p (acqbulk ());
  st 1;

  (* ---- the tiles a role walks ---- *)
  let ans = r4 "answer" and none_left = Core.p "no_tile" in
  (* The answer to the scheduler's last request: wait for it, read the first
     CTA of the cluster it cancelled and whether there was one, give the slot
     back to the scheduler and take that cluster's CTA of this rank. The next
     answer is written by the async proxy: the proxy fence orders this read
     before it -- without it a warp whose stores hold up its read can let the
     scheduler's next answer land first. *)
  let take inbox ~done_ =
    Pipe.consume inbox (fun () ->
      Pipe.wait_full inbox;
      Mem.lds128 km ans (at answer (C 0));
      Prog.emits p [ lea_ur tile_r (part ans 0) (Mem.cta km) 0; lop3_and (part ans 2) (part ans 2) 1; fence_view_async () ];
      Prog.when_ p lane0 (fun () -> Mem.arrive_red km (Pipe.empty ~peer:first_cta inbox)));
    Prog.emits p [ isetp_eq_u32 none_left (part ans 2) 0; bra none_left done_ ]
  in
  (* Stream-K: the tiles' k steps, cluster tile after cluster tile, split
     evenly over the clusters -- cluster c takes units [c U / G, (c + 1) U / G)
     -- each walking its range from its last tile to its first. The last is a
     tile it only starts: it leaves the partial sum in its slot of the
     workspace and raises its flag first thing. The first is a tile it
     finishes: it takes the cluster before's partial, which that cluster left
     first thing too, so no cluster waits on another's end. A range spans at
     least a tile, so a tile has at most two contributors. *)
  let clusters = grid / ctas and cl_tiles = total / ctas in
  let units = cl_tiles * kt in
  (* with the partial handed on, a tile has two contributors at most; adding
     in the L2, any number *)
  if streamk && (not split_add) && units / clusters < kt then failwith "gemm_swapped: a cluster's share of stream-K is less than a tile";
  let kb = r "k_begin" and ke = r "k_end" and kcount = r "k_count" in
  let segments body =
    let cl_r = r "cluster" in
    Prog.emits p [ s2r_ctaid cl_r ~axis:"X"; shf_r cl_r cl_r (log2 ctas) ];
    let cl = Prog.fixed p ~name:"cl" ~range:clusters ~cls:General cl_r in
    let e v = Expr.var v in
    (* c U / G as c q + c r / G, U = q G + r: the division's numerator stays
       below G^2, where Emit has an exact reciprocal *)
    let q = units / clusters and rem = units mod clusters in
    let split c = Expr.add (Expr.scale q c) (Expr.div (Expr.scale rem c) clusters) in
    let bc = split (e "cl") and last = Expr.add (split (Expr.add (e "cl") (Expr.Const 1))) (Expr.Const (-1)) in
    let v name ex = Prog.value p ~name ~cls:General ~vars:[ "cl", Prog.Var cl ] ex in
    let t_first = v "t_first" (Expr.div bc kt) and kb_first = v "kb_first" (Expr.modulo bc kt) in
    let t_last = v "t_last" (Expr.div last kt) and ke_last = v "ke_last" (Expr.add (Expr.modulo last kt) (Expr.Const 1)) in
    let pf = Core.p "is_first" and pl = Core.p "is_last" and pm = Core.p "more_segments" in
    Prog.emit p (mov_rr seg_t t_last);
    Prog.over p ~name:"seg" ~reg:seg_t ~range:cl_tiles
      ~next:(fun top -> Prog.constructs p [ isetp_rr ~cmp:"GT" pm seg_t t_first; iadd3_c seg_t seg_t (-1); bra pm top ])
      (fun seg ->
        Prog.emits p
          [ mov_rz kb; isetp_rr ~cmp:"EQ" pf seg_t t_first; on pf (mov_rr kb kb_first)
          ; mov_imm ke kt; isetp_rr ~cmp:"EQ" pl seg_t t_last; on pl (mov_rr ke ke_last); iadd3_sub kcount ke kb ];
        let kb_i = Prog.fixed p ~name:"kb" ~range:kt ~cls:General kb in
        (* the cluster tile's CTA of this rank *)
        let tile = Mem.plus (Mem.scale ctas (Mem.idx seg)) (Mem.idx (Mem.rank km)) in
        let tile_i = Prog.fixed p ~name:"tile" ~range:ntiles ~cls:General (Mem.get km ~cls:General tile) in
        body tile_i (Some kb_i))
  in
  (* a role's tile loop; a timing build counts the role's tiles *)
  let tiles_loop ~done_ body =
    let body =
      if stamps && not streamk then begin
        let c = r "stamp_tile" in
        Prog.construct p (mov_rz c);
        fun t kb ->
          tile_count := Some c;
          body t kb;
          Prog.construct p (iadd3_c c c 1);
          tile_count := None
      end
      else body
    in
    if streamk then segments body
    else
      match answers with
      | Some a ->
        let inbox = Pipe.consumer km a in
        Prog.over p ~name:"tile" ~reg:tile_r ~range:ntiles ~next:(fun top -> take inbox ~done_; Prog.construct p (jmp top)) (fun t -> body t None)
      | None ->
        if persistent then Prog.stride p ~name:"tile" ~reg:tile_r ~until:ntiles ~step:grid (fun t -> body t None)
        else body (Prog.fixed p ~name:"tile" ~range:ntiles ~cls:General tile_r) None
  in
  (* a tile's k loop: its own k steps, or a segment's *)
  let k_loop body =
    let count = if streamk || (split_grid > 1 && not even_runs) then Prog.Reg_times kcount else Times (if split_grid > 1 then kt / split_grid else kt) in
    Prog.for_ p ~name:"k" ~pred:more ~count body
  in
  (* split over the grid, unevenly: the run's length, from the CTA *)
  let run_count tile =
    if split_grid > 1 && not even_runs then
      let _, _, count = run_of (Expr.var "c") in
      ignore (Prog.value p ~name:"k_count" ~cls:General ~into:kcount ~vars:[ "c", Prog.Var tile ] count)
  in

  (* warp 0: the loads, into this CTA's stage, reporting to the leader's
     barrier; the leader states the pair's bytes *)
  Prog.role p ~name:"producer" ~skip:(fun q -> isetp_ne_u32 q warp 0) (fun () ->
    let txr = r "tx" and done_ = Prog.fresh "PRODUCER_DONE" in
    Prog.emit p (mov_imm txr tx);
    let out = Pipe.producer km stages in
    st 2;
    tiles_loop ~done_ (fun tile kb_i ->
      let a_row = Mem.(plus (row tile) (scale a_stride (coord_of km X))) in
      (* the k column the segment starts at; split, the pair's run of K *)
      let k0 =
        match kb_i with
        | Some i -> Mem.(scale tile_k (idx i))
        | None when split_grid > 1 ->
          (* the run's first k step in a uniform register: the coordinate
             is a uniform sum of it and the step's *)
          let _, kb, _ = run_of (Expr.var "c") in
          let kb_r = Prog.value p ~name:"k_begin" ~cls:General ~vars:[ "c", Prog.Var tile ] kb and kb_u = ur "k_begin" in
          Prog.emit p (r2ur kb_u kb_r);
          Mem.(scale tile_k (idx (Prog.fixed p ~name:"kb" ~range:kt ~cls:Uniform kb_u)))
        | None when split > 1 ->
          let y = Prog.fixed p ~name:"split_y" ~range:split ~cls:Uniform (Mem.get km ~cls:Uniform (Mem.coord_of km Y)) in
          Mem.(scale (kt * tile_k) (idx y))
        | None -> Mem.const 0
      in
      run_count tile;
      k_loop (fun kk ->
        let x = Mem.(plus k0 (scale tile_k (idx kk))) in
        Pipe.produce out (fun () ->
          Pipe.wait_empty out;
          let not_leader = Prog.fresh "NOT_LEADER" in
          Prog.emit p (bra ~neg:true p_leader not_leader);
          Prog.when_ p lane0 (fun () -> Mem.arrive_tx km (Pipe.full out) ~tx:txr);
          Prog.label p not_leader;
          Mem.tma_load km ~two:true ~map:map_a ~dst:(Pipe.slot_p out sa) ~bar:(Pipe.full ~peer:lead out) ~x ~y:a_row ();
          Mem.tma_load km ~two:true ?share:(if sharing > 1 then Some Mem.Y else None) ~map:map_b ~dst:(Pipe.slot_p out sb)
            ~bar:(Pipe.full ~peer:lead out) ~x ~y:(col tile) ();
          st_first kk 10)));
    (* every stage this CTA filled has been released before it leaves *)
    Prog.label p done_;
    if late_launch then Prog.emit p (preexit ());
    Prog.for_ p ~name:"drain" ~count:(Times depth) (fun _ -> Pipe.produce out (fun () -> Pipe.wait_empty out));
    st 3;
    Prog.emit p (exit ()));

  (* warp 1: the pair's MMAs, the leader's to issue *)
  Prog.role p ~name:"mma" ~skip:(fun q -> isetp_ne_u32 q warp 1) (fun () ->
    let done_ = Prog.fresh "MMA_DONE" and leave = Prog.fresh "MMA_LEAVE" in
    Prog.emits p [ sync_epi (); bra ~neg:true p_leader leave ];
    let regs = Mem.mma_regs km ~atom ~la:l_sb ~lb:l_sa in
    let first = ur "first" and accum = up "accumulate" and probe = Core.p "probe" and acc_r = r "acc_r" and acc = ur "acc" in
    let inp = Pipe.consumer km stages and acc_out = Pipe.producer km accs in
    st 4;
    tiles_loop ~done_ (fun tile _ ->
      run_count tile;
      Pipe.produce acc_out (fun () ->
        Mem.lds km acc_r (Pipe.slot_p acc_out slot);
        Prog.emit p (r2ur acc acc_r);
        Pipe.wait_empty acc_out;
        st_tile 15;
        Prog.emit p (umov first 0);
        (* asking a stage ahead: a tile asks about its first stage as it
           starts, each stage about the next once its work is out -- the last
           asks nothing, since a stage no load is coming for would keep the
           question out until the hardware gives up -- and a stage the
           question found complete is not waited for again *)
        if ask_ahead then Pipe.probe_full inp ~into:probe;
        k_loop (fun kk ->
          Pipe.consume inp (fun () ->
            Prog.emit p (uisetp_ne accum first);
            let landed = Prog.fresh "READY" in
            if ask_ahead then Prog.emit p (bra probe landed);
            Pipe.wait_full inp;
            Prog.label p landed;
            st_first kk 9;
            if stage_stamps then Stamp.stage p ~nparams ~count:(Prog.reg kk) ();
            (* block after block of the stacked atoms, every k step of a block
               before the next: block r reads B^T's rows r atoms on, into the
               accumulator's columns r blocks on *)
            for r = 0 to reps - 1 do
              for j = 0 to (tile_k / mk) - 1 do
                Mem.mma km ~leader regs ~a:(Mem.sub (Pipe.slot_c inp sb) (r * a_rows, j * mk)) ~b:(Mem.sub (Pipe.slot_c inp sa) (0, j * mk))
                  ~d:(acc, r * span) ~accumulate:(if j = 0 then `If accum else `Always)
              done
            done;
            (* the stage is free in every CTA of the cluster once these MMAs have read it *)
            Prog.emit p (umov first 1);
            Mem.commit2_mc km ~leader (Pipe.empty inp) ~mask:(if sharing > 1 then mask_all else mask_pair);
            if stage_stamps then Stamp.stage p ~half:1 ~nparams:3 ~count:(Prog.reg kk) ());
          if ask_ahead then Prog.when_ p more (fun () -> Pipe.probe_full inp ~into:probe));
        (* the accumulator is ready in both CTAs of the pair *)
        st_tile 12;
        Mem.commit2_mc km ~leader (Pipe.full acc_out) ~mask:mask_pair));
    (* every accumulator has been read out in both CTAs before leaving *)
    Prog.label p done_;
    Prog.for_ p ~name:"drain" ~count:(Times nbuf) (fun _ -> Pipe.produce acc_out (fun () -> Pipe.wait_empty acc_out));
    st 5;
    Prog.label p leave;
    (* every warp that reads tensor memory is done with it: free it *)
    Prog.emit p (sync_epi ());
    if ncols * nbuf = 512 && nbuf = 1 then Prog.emit p (utcatomsws_clear ())
    else begin
      let base = ur "free_base" and w = r "free_word" in
      Mem.lds km w (at slot (C 0));
      Prog.emit p (r2ur base w);
      Mem.tmem_free km ~base ~ncols:alloc_cols
    end;
    Prog.emit p (exit ()));

  (* warp 2, with cluster launch control, in the cluster's first CTA: the
     scheduler, one request at a time; it fills the answer and reads it back *)
  Option.iter
    (fun a ->
      Prog.role p ~name:"scheduler" ~skip:(fun q -> isetp_ne_u32 q warp 2) (fun () ->
        let first_u = up "first_cta" and p_first = Core.p "is_first" and tx16 = r "tx16" in
        let top = Prog.fresh "SCHED" and last = Prog.fresh "SCHED_LAST" and stop = Prog.fresh "SCHED_END" in
        Prog.emits p [ uisetp_eq0 first_u (Mem.cta km); plop3_up p_first first_u; bra ~neg:true p_first stop ];
        let peers = List.init ctas (Mem.peer_rank km) in
        let outbox = Pipe.producer km a and inbox = Pipe.consumer km a in
        Prog.emit p (mov_imm tx16 16);
        Prog.label p top;
        (* the answer's 16 bytes, stated on every CTA's copy of the slot *)
        Pipe.produce outbox (fun () ->
          Pipe.wait_empty outbox;
          Prog.when_ p lane0 (fun () -> List.iter (fun w -> Mem.arrive_tx_red km (Pipe.full ~peer:w outbox) ~tx:tx16) peers));
        Mem.clc_request km ~answer:(at answer (C 0)) ~bar:(Pipe.full outbox);
        Pipe.consume inbox (fun () ->
          Pipe.wait_full inbox;
          Mem.lds128 km ans (at answer (C 0));
          Prog.emits p [ lop3_and (part ans 2) (part ans 2) 1; isetp_eq_u32 none_left (part ans 2) 0 ]);
        Prog.emits p [ bra none_left last; jmp top ];
        (* none left: once the takers have read that answer, stop *)
        Prog.label p last;
        Pipe.produce outbox (fun () -> Pipe.wait_empty outbox);
        Prog.label p stop;
        Prog.emit p (exit ())))
    answers;

  (* warps 4-7: each tile's accumulator out to C; once the last block is out
     of tensor memory, it is released on the leader's barrier *)
  Prog.role p ~name:"epilogue" ~skip:(fun q -> isetp_lt_u32 q warp 4) (fun () ->
    let done_ = Prog.fresh "EPI_DONE" and acc_r = r "acc_r" and acc = ur "acc" and mat = r2 "c_addr" in
    Prog.emit p (sync_epi ());
    if direct then Prog.emit p (ldc64 mat (param_off 0));
    let acc_in = Pipe.consumer km accs in
    let release () = Prog.when_ p lane0 (fun () -> Mem.arrive_red km (Pipe.empty ~peer:lead acc_in)); st_tile 11 in
    let ready_wait () = Pipe.wait_full acc_in; st_tile 7 in
    (* a block's registers: each block loaded into one half while the block
       before it goes out from the other *)
    let n = match plan with Some pl -> pl.s_n | None -> dp.n in
    let data = rn (2 * n) "data" in
    let half i = slice data (i mod 2 * n) n in
    (* straight from the registers: a store a register; with [adds], the
       partials of the same tile in those slots are added first, block by
       block, each (block, slot) loaded past the L1 two steps ahead *)
    let direct_out ?(adds = []) ~before_ld out =
      let blocks = Store.blocks dp in
      let nblk = List.length blocks in
      let partial = rn (2 * dp.n) "partial" in
      let phalf j = slice partial (j mod 2 * dp.n) dp.n in
      let steps = Array.of_list (List.concat (List.init nblk (fun i -> List.map (fun slot -> i, slot) adds))) in
      let fetch j =
        if j < Array.length steps then begin
          let i, slot = steps.(j) in
          for v = 0 to (dp.n / 4) - 1 do Mem.ldg_frag km out ~slot i (4 * v) (slice (phalf j) (4 * v) 4) done
        end
      in
      fetch 0;
      fetch 1;
      before_ld ();
      Mem.ldtm km out (List.hd blocks) (half 0);
      List.iteri
        (fun i b ->
          List.iteri
            (fun s _ ->
              let j = (i * List.length adds) + s in
              for v = 0 to dp.n - 1 do Prog.emit p (fadd (part (half i) v) (part (half i) v) (part (phalf j) v)) done;
              fetch (j + 2))
            adds;
          for v = 0 to dp.n - 1 do Mem.stg km out b v (part (half i) v) done;
          if i + 1 < nblk then Mem.ldtm km out (List.nth blocks (i + 1)) (half (i + 1));
          if i = nblk - 1 then release ())
        blocks
    in
    (* a tile's partial, in fragment order, to [slot] *)
    let frag_out out slot =
      let blocks = Store.blocks dp in
      let nblk = List.length blocks in
      Mem.ldtm km out (List.hd blocks) (half 0);
      List.iteri
        (fun i _ ->
          for v = 0 to (dp.n / 4) - 1 do Mem.stg_frag km out ~slot i (4 * v) (slice (half i) (4 * v) 4) done;
          if i + 1 < nblk then Mem.ldtm km out (List.nth blocks (i + 1)) (half (i + 1));
          if i = nblk - 1 then release ())
        blocks
    in
    (* split over the grid, a run's part of the tile: the blocks of the other
       runs out to its [slot], then [signal]; once [wait] has seen the other
       runs' partials, each of [mine] the sum of tensor memory's and theirs at
       [slots] in order, each (block, slot) loaded past the L1 [ahead] steps
       ahead; the accumulator released after its last block *)
    let run_out out ~mine ~slot ~slots ~signal ~wait =
      let blocks = List.mapi (fun i b -> i, b) (Store.blocks dp) in
      let theirs = List.filter (fun (i, _) -> not (List.mem i mine)) blocks and mine = List.filter (fun (i, _) -> List.mem i mine) blocks in
      let ld k l = match List.nth_opt l k with Some (_, b) -> Mem.ldtm km out b (half k) | None -> () in
      (* block k + 1 out of tensor memory while block k's stores issue *)
      ld 0 theirs;
      List.iteri
        (fun k (i, _) ->
          ld (k + 1) theirs;
          for v = 0 to (dp.n / 4) - 1 do Mem.stg_frag km out ~slot i (4 * v) (slice (half k) (4 * v) 4) done)
        theirs;
      signal ();
      if mine = [] then release ()
      else begin
        wait ();
        let ns = List.length slots in
        let steps = Array.of_list (List.concat_map (fun (i, _) -> List.map (fun sl -> i, sl) slots) mine) in
        let ahead = max 1 (min (Array.length steps) ((if stamps then 64 else 96) / dp.n)) in
        let partial = rn (ahead * dp.n) "partial" in
        let buf j = slice partial (j mod ahead * dp.n) dp.n in
        let fetch j =
          if j < Array.length steps then begin
            let i, sl = steps.(j) in
            for v = 0 to (dp.n / 4) - 1 do Mem.ldg_frag km out ~slot:sl i (4 * v) (slice (buf j) (4 * v) 4) done
          end
        in
        for j = 0 to ahead - 1 do fetch j done;
        ld 0 mine;
        List.iteri
          (fun k (_, b) ->
            let sum = half k in
            ld (k + 1) mine;
            for t = 0 to ns - 1 do
              let j = (k * ns) + t in
              for v = 0 to dp.n - 1 do Prog.emit p (fadd (part sum v) (part sum v) (part (buf j) v)) done;
              fetch (j + ahead)
            done;
            for v = 0 to dp.n - 1 do Mem.stg km out b v (part sum v) done;
            if k = List.length mine - 1 then release ())
          mine
      end
    in
    (* through the staging copies: each block staged, then stored by the
       copy engine *)
    let staged_out ?(add = false) pl out =
      let blocks = Store.staged_blocks pl in
      let nblk = List.length blocks in
      Mem.ldtm_staged km out (List.hd blocks) (half 0);
      List.iteri
        (fun i b ->
          let copy = i mod pl.copies in
          (* the copy this block goes into was last read by the store
             [copies] blocks back; stores finish in order *)
          Prog.emit p (depbar_le ~n:(pl.copies - 1));
          for v = 0 to (pl.s_n / pl.vec) - 1 do Mem.sts_staged km out ~slot:copy v (half i) done;
          if i + 1 < nblk then Mem.ldtm_staged km out (List.nth blocks (i + 1)) (half (i + 1));
          if i = nblk - 1 then release ();
          (* the writes land before the copy engine reads them: the fence
             publishes them to the async proxy *)
          Prog.emits p [ fence_view_async (); warpsync () ];
          Mem.tma_store ~add km out ~slot:copy b;
          Prog.emit p (utmacmdflush ()))
        blocks
    in
    (* stream-K's workspace and flags, a slot a CTA; split-K's workspace *)
    let ws_ptr = r2 "ws" and flags_ptr = r2 "flags" and cta_i = ref None in
    if streamk || split > 1 || split_grid > 1 then begin
      let cta_r = r "cta_id" in
      Prog.emits p
        (s2r_ctaid cta_r ~axis:"X"
        :: (if split_add then [ ldc64 flags_ptr (param_off 3) ]
            else ldc64 ws_ptr (param_off 3) :: (if streamk || split_grid > 1 then [ ldc64 flags_ptr (param_off 4) ] else [])));
      cta_i := Some (Prog.fixed p ~name:"cta_id" ~range:grid ~cls:General cta_r)
    end;
    let ws = Mem.global_slabs ~name:"ws" ~ptr:ws_ptr ~n:grid ~bytes:(tile_m * tile_n * out_elem) in
    (* stream-K adding in the L2: a word a CTA, then a word a tile *)
    let nflags = if streamk && split_add then grid + total else grid in
    let flags = Mem.global_words ~name:"flags" ~ptr:flags_ptr ~n:nflags ~bytes:4 in
    st 6;
    (* stream-K adding in the L2: the segment with a tile's first k step
       stores it and raises the tile's flag; the tile's other segments add to
       it once the flag is up. The flag's value this launch: one past what
       each CTA's own word held, raised once a launch, so the same in every
       CTA *)
    let epoch = ref None in
    (match plan, staging with
     | Some _, Some _ when streamk && split_add ->
       let cta = Option.get !cta_i in
       let own = r "own_count" and target = r "flag_target" and first_w = Core.p "first_warp" in
       Prog.emit p (isetp_eq_u32 first_w warp 4);
       Mem.ldg_strong km ~desc:gdesc own (Mem.at flags [ I cta ]);
       Prog.emit p (iadd3_c target own 1);
       let skip_s = Prog.fresh "NOT_FIRST_WARP" in
       Prog.emit p (bra ~neg:true first_w skip_s);
       (* every lane of the first warp, the same word *)
       Mem.stg_strong km ~desc:gdesc (Mem.at flags [ I cta ]) ~data:target;
       Prog.label p skip_s;
       epoch := Some (target, first_w)
     | _ -> ());
    tiles_loop ~done_ (fun tile _ -> Pipe.consume acc_in (fun () ->
      Mem.lds km acc_r (Pipe.slot_c acc_in slot);
      Prog.emit p (r2ur acc acc_r);
      let row = row tile and col = col tile in
      (match plan, staging with
       | Some pl, Some sg when split_add && streamk ->
         let target, first_w = Option.get !epoch in
         let out = Mem.staged pl ~staging:sg ~acc ~map:map_c ~row ~col ~warp:warpid ~lane:laneid in
         let tile_flag = Mem.at flags [ V (Mem.plus (Mem.const grid) (Mem.idx tile)) ] in
         let adds = Core.p "adds" and add_l = Prog.fresh "ADD" and done_l = Prog.fresh "SEGMENT_STORED" in
         Prog.emits p [ isetp_ne_u32 adds kb 0; bra adds add_l ];
         (* the tile's first k steps: stored, in the L2, then the flag *)
         ready_wait ();
         staged_out pl out;
         let skip_s = Prog.fresh "NOT_FIRST_WARP" in
         Prog.emits p
           [ depbar_drain (); fence_view_async_global (); membar_gpu (); bar_sync_n ~bar:2 ~count:(32 * epi_warps); bra ~neg:true first_w skip_s ];
         Mem.stg_strong km ~desc:gdesc tile_flag ~data:target;
         Prog.label p skip_s;
         Prog.emit p (jmp done_l);
         (* the first warp waits for the tile's flag *)
         Prog.label p add_l;
         let wait_l = Prog.fresh "STORED" and skip_r = Prog.fresh "NOT_FIRST_WARP" and got = r "tile_flag" and spin = Core.p "spin" in
         Prog.emit p (bra ~neg:true first_w skip_r);
         Prog.label p wait_l;
         Mem.ldg_strong km ~desc:gdesc got tile_flag;
         Prog.emits p [ isetp_rr ~cmp:"NE" spin got target; bra spin wait_l ];
         Prog.label p skip_r;
         Prog.emits p [ membar_gpu (); fence_view_async_global (); bar_sync_n ~bar:2 ~count:(32 * epi_warps) ];
         ready_wait ();
         staged_out ~add:true pl out;
         Prog.label p done_l
       | Some pl, Some sg when split_add ->
         (* the tile's runs zero it, block i run i S / blocks's, from a staging
            copy of zeros, while the k loop runs; each raises its flag to one
            past what it held once its zeros are in the L2; once every run's
            flag is up and its own accumulator is ready, each adds the whole
            of it *)
         let out = Mem.staged pl ~staging:sg ~acc ~map:map_c ~row ~col ~warp:warpid ~lane:laneid in
         let cta = Option.get !cta_i in
         let c = Expr.var "c" in
         let s_r = Prog.value p ~name:"run" ~cls:General ~vars:[ "c", Prog.Var cta ] (let s, _, _ = run_of c in s) in
         (* run j's CTA of this rank *)
         let flag_of j =
           Mem.at flags [ V (Mem.of_expr (Expr.add (Expr.scale ctas (Expr.add (Expr.scale split_grid (Expr.div c (ctas * split_grid))) (Expr.Const j))) (Expr.modulo c ctas)) [ "c", Prog.Var cta ]) ]
         in
         let own = r "own_flag" and target = r "flag_target" in
         let first_w = Core.p "first_warp" and spin = Core.p "spin" and theirs = Core.p "not_mine" in
         Prog.emit p (isetp_eq_u32 first_w warp 4);
         Mem.ldg_strong km ~desc:gdesc own (Mem.at flags [ I cta ]);
         Prog.emit p (iadd3_c target own 1);
         let zeros = half 0 in
         for v = 0 to n - 1 do Prog.emit p (mov_rz (part zeros v)) done;
         for v = 0 to (pl.s_n / pl.vec) - 1 do Mem.sts_staged km out ~slot:0 v zeros done;
         Prog.emits p [ fence_view_async (); warpsync () ];
         let blocks = Store.staged_blocks pl in
         let nblk = List.length blocks in
         List.iteri
           (fun i b ->
             let skip = Prog.fresh "NOT_MINE" in
             Prog.emits p [ isetp_ne_u32 theirs s_r (i * split_grid / nblk); bra theirs skip ];
             Mem.tma_store km out ~slot:0 b;
             Prog.emit p (utmacmdflush ());
             Prog.label p skip)
           blocks;
         (* the zeros in the L2, then the flag, the first warp's every lane
            storing the same word *)
         let skip_s = Prog.fresh "NOT_FIRST_WARP" in
         Prog.emits p
           [ depbar_drain (); fence_view_async_global (); membar_gpu (); bar_sync_n ~bar:2 ~count:(32 * epi_warps); bra ~neg:true first_w skip_s ];
         Mem.stg_strong km ~desc:gdesc (Mem.at flags [ I cta ]) ~data:target;
         Prog.label p skip_s;
         ready_wait ();
         (* the first warp waits for the tile's flags whole, all loaded at once *)
         let wait_l = Prog.fresh "FLAGS" and skip_r = Prog.fresh "NOT_FIRST_WARP" in
         Prog.emit p (bra ~neg:true first_w skip_r);
         Prog.label p wait_l;
         let got = List.init split_grid (fun j -> r (Printf.sprintf "flag_%d" j)) in
         List.iteri (fun j g -> Mem.ldg_strong km ~desc:gdesc g (flag_of j)) got;
         List.iter (fun g -> Prog.emits p [ isetp_rr ~cmp:"NE" spin g target; bra spin wait_l ]) got;
         Prog.label p skip_r;
         Prog.emits p [ membar_gpu (); fence_view_async_global (); bar_sync_n ~bar:2 ~count:(32 * epi_warps) ];
         staged_out ~add:true pl out
       | Some pl, Some sg ->
         ready_wait ();
         staged_out pl (Mem.staged pl ~staging:sg ~acc ~map:map_c ~row ~col ~warp:warpid ~lane:laneid)
       | _ ->
         let out = Mem.direct dp ~acc ~mat:(`R mat) ~desc:gdesc ~row ~col ~warp:warpid ~lane:laneid in
         if split > 1 then begin
           (* the first pair waits for the other's partial and adds it; the
              other leaves it in its slot, fenced for the GPU, and reports *)
           let cta = Option.get !cta_i and bar = Option.get partials in
           let first_pair = Core.p "first_pair" and parity = r "partials_parity" in
           let other_l = Prog.fresh "PARTIAL_OUT" and done_l = Prog.fresh "SPLIT_STORED" in
           Prog.emits p [ isetp_eq_u32 first_pair (Mem.get km ~cls:General (Mem.coord_of km Y)) 0; bra ~neg:true first_pair other_l ];
           ready_wait ();
           Prog.emit p (mov_imm parity Barrier.full_parity);
           Mem.wait km (Mem.at bar [ C 0 ]) ~parity;
           (* pair j of the cluster is the CTAs [pair j] past the first's *)
           let adds = List.init (split - 1) (fun j -> Mem.at ws [ V (Mem.plus (Mem.idx cta) (Mem.const (pair * (j + 1)))) ]) in
           direct_out ~adds ~before_ld:(fun () -> ()) out;
           Prog.emit p (jmp done_l);
           Prog.label p other_l;
           ready_wait ();
           frag_out out (Mem.at ws [ I cta ]);
           Prog.emit p (membar_gpu ());
           Prog.when_ p lane0 (fun () -> Mem.arrive_red km (Mem.at ~peer:(Mem.peer km ~y:0 ()) bar [ C 0 ]));
           Prog.label p done_l
         end
         else if split_grid > 1 then begin
           (* every CTA of the tile leaves its partial in its slot and raises
              its flag to one past what it held -- the tile's flags all hold
              the same, each raised once a launch -- and, once the other runs'
              flags are up, adds up block i of the tile if i S / blocks is its
              run: a copy of the epilogue for each run *)
           let cta = Option.get !cta_i in
           let s_r = Prog.value p ~name:"run" ~cls:General ~vars:[ "c", Prog.Var cta ] (let s, _, _ = run_of (Expr.var "c") in s) in
           (* run j's CTA of this rank, from run s's *)
           let peer s j = Mem.V (Mem.plus (Mem.idx cta) (Mem.const ((j - s) * ctas))) in
           let own = r "own_flag" and target = r "flag_target" in
           let first_w = Core.p "first_warp" and spin = Core.p "spin" and other = Core.p "other_run" in
           Prog.emit p (isetp_eq_u32 first_w warp 4);
           Mem.ldg_strong km ~desc:gdesc own (Mem.at flags [ I cta ]);
           Prog.emit p (iadd3_c target own 1);
           ready_wait ();
           let nblk = List.length (Store.blocks dp) and done_l = Prog.fresh "SPLIT_STORED" in
           for s = 0 to split_grid - 1 do
             let next = Prog.fresh "OTHER_RUN" in
             if s < split_grid - 1 then Prog.emits p [ isetp_ne_u32 other s_r s; bra other next ];
             let others = List.filter (fun j -> j <> s) (List.init split_grid Fun.id) in
             (* every store of the partial is visible GPU-wide before the flag *)
             let signal () =
               let skip_s = Prog.fresh "NOT_FIRST_WARP" in
               Prog.emits p [ membar_gpu (); bar_sync_n ~bar:2 ~count:(32 * epi_warps); bra ~neg:true first_w skip_s ];
               (* every lane of the first warp, the same word *)
               Mem.stg_strong km ~desc:gdesc (Mem.at flags [ I cta ]) ~data:target;
               Prog.label p skip_s;
               st_tile 14
             in
             (* the first warp waits for the other runs' flags whole, all
                loaded at once *)
             let wait () =
               let wait_l = Prog.fresh "FLAGS" and skip_r = Prog.fresh "NOT_FIRST_WARP" in
               Prog.emit p (bra ~neg:true first_w skip_r);
               Prog.label p wait_l;
               let got = List.map (fun j -> r (Printf.sprintf "flag_%d" j)) others in
               List.iter2 (fun j g -> Mem.ldg_strong km ~desc:gdesc g (Mem.at flags [ peer s j ])) others got;
               List.iter (fun g -> Prog.emits p [ isetp_rr ~cmp:"NE" spin g target; bra spin wait_l ]) got;
               Prog.label p skip_r;
               Prog.emits p [ membar_gpu (); bar_sync_n ~bar:2 ~count:(32 * epi_warps) ];
               st_tile 13
             in
             run_out out
               ~mine:(List.filter (fun i -> i * split_grid / nblk = s) (List.init nblk Fun.id))
               ~slot:(Mem.at ws [ I cta ])
               ~slots:(List.map (fun j -> Mem.at ws [ peer s j ]) others)
               ~signal ~wait;
             if s < split_grid - 1 then begin
               Prog.emit p (jmp done_l);
               Prog.label p next
             end
           done;
           Prog.label p done_l
         end
         else if not streamk then (ready_wait (); direct_out ~before_ld:(fun () -> ()) out)
         else begin
           (* a segment that ends before the tile does leaves its partial in
              this CTA's slot of the workspace; one that starts after the
              tile does adds the slot of the cluster before *)
           let cta = Option.get !cta_i in
           let before = Mem.V (Mem.plus (Mem.idx cta) (Mem.const (-ctas))) in
           let zero = r "zero" and one = r "one_flag" and got = r "got" and ktr = r "kt" in
           let first_t = Core.p "first_thread" and pp = Core.p "partial" and pa = Core.p "adds" and spin = Core.p "spin" in
           let first_w = Core.p "first_warp" in
           let partial_l = Prog.fresh "PARTIAL" and add_l = Prog.fresh "ADD" and done_l = Prog.fresh "SEGMENT_STORED" in
           let wait_l = Prog.fresh "FLAG" and skip_r = Prog.fresh "NOT_FIRST_WARP" in
           Prog.emits p
             [ mov_rz zero; mov_imm one 1; mov_imm ktr kt; isetp_eq_u32 first_t tid (32 * 4); isetp_eq_u32 first_w warp 4
             ; isetp_rr ~cmp:"LT" pp ke ktr; bra pp partial_l; isetp_ne_u32 pa kb 0; bra pa add_l ];
           direct_out ~before_ld:ready_wait out;
           Prog.emit p (jmp done_l);
           Prog.label p partial_l;
           ready_wait ();
           frag_out out (Mem.at ws [ I cta ]);
           (* every store of the partial is visible GPU-wide before the flag;
              a warp meets the others whole, so one lane's store is
              predicated, not branched around *)
           Prog.emits p [ membar_gpu (); bar_sync_n ~bar:2 ~count:(32 * epi_warps) ];
           Prog.when_ p first_t (fun () -> Mem.stg_strong km ~desc:gdesc (Mem.at flags [ I cta ]) ~data:one);
           Prog.emit p (jmp done_l);
           (* the cluster before's flag -- the first warp waits for it whole,
              every lane reading the same word -- then its partial *)
           Prog.label p add_l;
           let flag = Mem.at flags [ before ] in
           Prog.emit p (bra ~neg:true first_w skip_r);
           Prog.label p wait_l;
           Mem.ldg_strong km ~desc:gdesc got flag;
           Prog.emits p [ isetp_eq_u32 spin got 0; bra spin wait_l ];
           Prog.when_ p first_t (fun () -> Mem.stg_strong km ~desc:gdesc flag ~data:zero);
           Prog.label p skip_r;
           Prog.emits p [ membar_gpu (); bar_sync_n ~bar:2 ~count:(32 * epi_warps) ];
           direct_out ~adds:[ Mem.at ws [ before ] ] ~before_ld:ready_wait out;
           Prog.label p done_l;
           st_tile 14
         end);
      if split_grid = 1 then st_tile 13));
    (* the last stores have read their staging copies before the warp leaves *)
    Prog.label p done_;
    Prog.emit p (depbar_drain ());
    st 8;
    Prog.emits p [ sync_epi (); exit () ]);
  Prog.emit p (exit ());

  let c_param =
    match plan with
    | Some _ ->
      let bx = Atom.tma_box l_c ~elem:out_elem in
      Tmap { pname = "c"; rows = m; cols = n; elem = out_elem; box_rows = bx.box_rows; box_cols = bx.box_cols; swizzle = bx.box_swizzle }
    | None -> Ptr { pname = "c"; rows = m; cols = n; elem = out_elem }
  in
  let sk_params =
    if split_add then [ Ptr { pname = "flags"; rows = 1; cols = (if streamk then grid + total else grid); elem = 4 } ]
    else if streamk then [ Ptr { pname = "ws"; rows = grid; cols = tile_m * tile_n; elem = out_elem }; Ptr { pname = "flags"; rows = 1; cols = grid; elem = 4 } ]
    else if split_grid > 1 then [ Ptr { pname = "ws"; rows = grid; cols = tile_m * tile_n; elem = out_elem }; Ptr { pname = "flags"; rows = 1; cols = grid; elem = 4 } ]
    else if split > 1 then [ Ptr { pname = "ws"; rows = grid; cols = tile_m * tile_n; elem = out_elem } ]
    else []
  in
  Mem.finish km ~name:(Printf.sprintf "swapped_%d_%d_%d_%dx%d_s%d" m n k tile_m tile_n depth)
    ~params:
      ([ c_param
       ; Tmap { pname = "a"; rows = m; cols = k; elem; box_rows = box_sa.box_rows; box_cols = box_sa.box_cols; swizzle = box_sa.box_swizzle }
       ; Tmap { pname = "bt"; rows = n; cols = k; elem; box_rows = box_sb.box_rows; box_cols = box_sb.box_cols; swizzle = box_sb.box_swizzle }
       ]
      @ sk_params)
    ~grid:(grid, 1) ~tile:(tile_m, tile_n, tile_k)
    ~together:(if streamk && split_add then grid / ctas else split_grid)
    ~timing:
      (if stage_stamps then Stage_stamps
       else if stamps then
         Stamps [ 0, "entry"; 1, "barriers initialised, cluster met"; 2, "producer starts"; 10, "producer: first stage issued"; 3, "producer done"
         ; 4, "MMA warp starts"; 15, "MMA: accumulator free"; 9, "MMA: first stage landed"; 12, "MMA: last stage issued"; 5, "MMA warp done"
         ; 6, "epilogue starts"; 7, "epilogue: accumulator ready"; 11, "epilogue: accumulator released"
         ; 14, (if split_grid > 1 then "epilogue: partial out, flag up" else "epilogue: segment stored")
         ; 13, (if split_grid > 1 then "epilogue: the other runs' flags seen" else "epilogue: tile's stores issued"); 8, "epilogue done" ]
       else No_timing)
    ~notes:
      [ Printf.sprintf "gemm_swapped %dx%dx%d: %dx%d a CTA, ring depth %d, cluster 2x%d, %d accumulator%s, %s%s%s" m n k tile_m tile_n depth
          cluster_n nbuf (if nbuf > 1 then "s" else "") (if clc then "cluster launch control" else "a fixed grid")
          (if ask_ahead then ", asking a stage ahead" else "") (if direct then ", the store from the registers" else "")
        ^ (if split > 1 then Printf.sprintf ", K split %d ways in the cluster" split else "")
        ^ (if split_grid > 1 then Printf.sprintf ", K split %d ways over the grid" split_grid else "")
        ^ if split_add then ", the runs adding in the L2" else "" ]
    ()
