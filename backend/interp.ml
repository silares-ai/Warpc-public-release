(* The host-side interpreter: a kernel's instructions run for every warp of
   a few clusters' CTAs together, with each instruction's effect on the
   registers (Sass.sem) and on what orders the warps (Sass.act). A warp runs
   its lanes together, splitting where a branch divides them and joining
   where they meet again; general registers and predicates are a lane's,
   uniform ones the warp's. The warps take turns, each running until it
   waits for something not yet there -- or spins on a word of global memory
   that has not changed.

   What the hardware does outside the registers is modelled as the
   protocols need it. An mbarrier counts its phase's arrivals and bytes and
   completes it when both reach what the phase takes; a wait passes once the
   phase of its parity has completed, as the hardware's try-wait answers. A
   named barrier releases its warps once its count of threads has arrived,
   the cluster's barrier once every warp of the cluster has. The copy
   engine's loads land, report their bytes and the tensor core's commits
   arrive as they are issued -- the earliest the hardware could; tensor
   memory is handed out in order; shared and global memory hold what was
   stored; a word of global memory no CTA of the run writes reads as 1 -- a
   flag raised, by a CTA the run does not include; the scheduler has no work
   to hand out.

   A run takes the first cluster and, separately, the last; a kernel whose
   CTAs read what others write in global memory runs its first two clusters
   together, so the hand-off between them runs too.

   Every operand the compiler derived is checked where its instruction runs
   against what the layouts it came from say it must be (Core.check). The
   coloured program is run too, and each checked operand must hold the same
   value there: the colouring kept every derived value alive.

   Every named barrier: the k-th arrivals of the warps that reach it state
   one thread count, 32 times how many warps they are, and each warp arrives
   with all 32 lanes.

   The barriers' protocol: the run must end with every warp out -- a warp
   still waiting is a hang, reported with what it waits for and where the
   barrier stands -- and every mbarrier at the start of a phase; nothing may
   arrive at an address no barrier was initialised at, and no warp may wait
   on another CTA's barrier.

   Memory as declared: a constant the kernel reads is a parameter it
   declares; a global access lands inside the pointer parameter it
   addresses; a copy-engine operation takes a tensor-map parameter, and its
   box fits the shared copy it fills or empties. Tensor memory: an MMA that
   adds to an accumulator adds to one an MMA has written since it was
   allocated or last read out -- a tile's first MMA overwrites -- one that
   overwrites does not overwrite results no load has read, and a load reads
   what an MMA wrote.

   And the happens-before order the run's synchronisation makes: each
   thread -- a warp; the tensor core's queue of a warp's MMAs and commits,
   which run in order; each copy-engine load or store and each request to
   the scheduler, which run on their own -- carries a vector clock. A
   barrier's phase passes the clocks of what arrived to whoever waits for
   it; a named barrier, the cluster's barrier, a wait for the copy engine's
   stores and a strong load of what a strong store wrote do the same. Every
   access to shared memory and global memory (4 bytes at a time), tensor
   memory (a column of a lane quarter) and the barriers' words is checked
   against the accesses before it: two to the same place, one a write, must
   be ordered; a write through the generic proxy must be fenced
   (FENCE.VIEW.ASYNC) before the copy engine or the tensor core reads it; a
   global write another CTA reads must be fenced at the GPU's scope
   (MEMBAR.ALL.GPU) before what orders the read. What touches a CTA's memory
   from outside its warps must be ordered before that CTA's warps have all
   exited. Each ordered pair is a row of the happens-before table, with the
   synchronisation that ordered it. *)

open Core
module IM = Map.Make (Int)

type failure =
  { what : string
  ; at : string (* the instruction *)
  ; cta : int
  ; warp : int
  ; lane : int
  ; got : int
  ; want : int
  }

type row =
  { buffer : string
  ; first : string
  ; second : string
  ; via : string
  ; count : int
  }

type report =
  { checked : int (* operands compared with their references *)
  ; failures : failure list
  ; mismatches : int (* checked operands the coloured program disagrees on *)
  ; barriers : string list (* named barriers whose count the arriving warps do not make *)
  ; protocol : string list (* hangs, barriers left part-way, arrivals at no barrier, waits on another CTA's *)
  ; memory : string list (* parameters and memory used past what was declared; accumulators added to or read out of turn *)
  ; races : string list (* accesses nothing orders, unfenced reads, touches after a CTA's exit *)
  ; table : row list (* the happens-before table: every ordered pair of accesses, by what ordered it *)
  ; idle : string list (* synchronisation the run passed that ordered no pair *)
  }

(* the threads a BAR.SYNC waits for: its count, or without one the CTA's *)
let bar_count (k : kernel) (ins : Sass.insn) =
  let t = Sass.to_string ins in
  match String.rindex_opt t ',' with
  | Some i -> int_of_string (String.trim (String.sub t (i + 1) (String.length t - i - 1)))
  | None -> k.threads

let m32 x = x land 0xffffffff
let step_limit = 50_000_000
let cap = 20

(* ---- the happens-before order ---- *)

(* a vector clock: for each thread, the latest of its events this one is
   ordered after, how it learned of it -- the synchronisation, or [program]
   (its own), or [issued] (the warp that issued it) -- and what it knew of
   that thread before, so the synchronisation that first ordered an event
   can be read back *)
type entry =
  { c : int
  ; via : int
  ; prev : entry option
  }

type vc = entry IM.t

let program = -1
let issued = -2

type thread =
  { tid : int
  ; mutable clk : int
  ; mutable vc : vc
  ; tslot : int (* the CTA of the warp it belongs to, by its place in the run *)
  ; twarp : int
  ; warp_thread : bool
  }

(* one instruction's execution, as an access: by which thread at which
   clock, through which proxy, at the GPU's coherence point or not *)
type event =
  { eid : int
  ; eth : int
  ; eclk : int
  ; ins : Sass.insn
  ; eslot : int
  ; ewarp : int
  ; async : bool
  ; strong : bool
  }

type cell =
  { mutable w : event option
  ; mutable rs : event list
  ; mutable adds : event list (* the reduce-adds since the last write: they commute with each other *)
  }

let clock (v : vc) t = match IM.find_opt t v with Some e -> e.c | None -> 0

(* the synchronisation through which [v] first knew clock [c] *)
let first_via (v : vc) t c =
  let rec go e = match e.prev with Some p when p.c >= c -> go p | _ -> e.via in
  go (IM.find t v)

let vc_join (a : vc) (b : vc) = IM.union (fun _ x y -> Some (if y.c > x.c then y else x)) a b

type mbar =
  { mutable count : int
  ; mutable pending : int
  ; mutable tx : int
  ; mutable phase : int
  ; mutable acc : vc (* what has arrived in the phase under way *)
  ; mutable last : vc (* what the phase completed last had *)
  ; init : event
  }

type nbar =
  { mutable arrived : int
  ; mutable gen : int
  ; mutable nacc : vc
  ; gens : (int, vc) Hashtbl.t
  }

(* a column of tensor memory: free, holding what an MMA wrote, or read out
   since *)
let t_free = 0
let t_live = 1
let t_read = 2
let pointer_bits = 40

(* One run: the warps of [clusters]' CTAs together. [on_check] sees each
   checked operand of each active lane; with [hb], the accesses are checked
   against the happens-before order and tabled. A global word in [written]
   -- the words the run's CTAs write, when known -- reads 0 until it is
   stored; any other word not yet stored reads 1. [writes] collects the
   words the run stores. *)
let run ?(hb = false) ?written ?writes ?on_exec (k : kernel) (body : Sass.item array) ~clusters ~on_check
    ~(rows : (string * (string * int * bool) * (string * int * bool) * string, int ref) Hashtbl.t) ~(passed : (string, unit) Hashtbl.t) =
  let n = Array.length body in
  let nc = k.cluster in
  let cta_of = Array.of_list (List.concat clusters) in
  let nslots = Array.length cta_of in
  let labels = Hashtbl.create 64 in
  Array.iteri (fun i it -> match it with Sass.Label l -> Hashtbl.replace labels l i | I _ -> ()) body;
  let checks = Hashtbl.create 256 in
  List.iter (fun (id, c) -> Hashtbl.replace checks id (c :: Option.value (Hashtbl.find_opt checks id) ~default:[])) k.checks;
  let touches = Hashtbl.create 64 in
  List.iter (fun (id, t) -> Hashtbl.replace touches id (t :: Option.value (Hashtbl.find_opt touches id) ~default:[])) k.touches;
  let shared = Hashtbl.create 256 and gmem = Hashtbl.create 256 in
  let next_chunk = Array.make nslots 0 in
  let nw = k.threads / 32 in
  let problems = ref [] and protocol = ref [] and memory = ref [] and races = ref [] in
  let say l s = if List.length !l < cap && not (List.mem s !l) then l := s :: !l in
  (* each named barrier's arrivals: by warp, the counts it arrived with, latest first *)
  let arrivals = Array.init nslots (fun _ -> (Hashtbl.create 4 : (int, (int, int list) Hashtbl.t) Hashtbl.t)) in

  (* ---- addresses: a shared::cluster address of the j-th cluster's CTAs is
     j << 32 | the address here ---- *)
  let in_run j a = (j lsl 32) lor m32 a in
  let slot_of ra = ((ra lsr 32) * nc) + ((ra lsr 24) land 0xff) in
  let rank_ok ra = (ra lsr 24) land 0xff < nc in

  (* ---- names, for the reports ---- *)
  let region off = List.find_opt (fun (r : region) -> off >= r.first && off < r.first + (r.copies * r.stride)) k.regions in
  let cta_name s = if nslots > 1 then Printf.sprintf "CTA %d's " cta_of.(s) else "" in
  let place ra =
    let off = ra land 0xffffff in
    match region off with
    | Some g ->
      let i = (off - g.first) / g.stride and w = (off - g.first) mod g.stride in
      Printf.sprintf "%s%s%s%s" (cta_name (slot_of ra)) g.rname (if g.copies > 1 then Printf.sprintf "[%d]" i else "") (if w > 0 then Printf.sprintf " +0x%x" w else "")
    | None -> Printf.sprintf "%sshared 0x%x" (cta_name (slot_of ra)) off
  in
  let buffer_of ra = match region (ra land 0xffffff) with Some g -> g.rname | None -> "shared memory" in
  (* the declared parameters, and a timing build's stamp buffer after them:
     64 words a CTA, or 4096 (Stamp) *)
  let params =
    k.params
    @ match k.timing with
      | No_timing -> []
      | Stamps _ -> [ Ptr { pname = "stamps"; rows = fst k.grid; cols = 64; elem = 4 } ]
      | Stage_stamps -> [ Ptr { pname = "stamps"; rows = fst k.grid; cols = 4096; elem = 4 } ]
  in
  let param i = List.nth_opt params i in
  let pname = function Tmap t -> t.pname | Ptr p -> p.pname in
  let global_place a =
    let i = (a lsr pointer_bits) - 1 in
    match param i with Some p -> Printf.sprintf "%s+0x%x" (pname p) (a land ((1 lsl pointer_bits) - 1)) | None -> Printf.sprintf "global 0x%x" a
  in
  let global_buffer a = match param ((a lsr pointer_bits) - 1) with Some p -> pname p | None -> "global memory" in
  (* an access in the table: the instruction, its warp, whether it is
     another CTA's *)
  let side (e : event) loc = Sass.opcode e.ins, e.ewarp, e.eslot <> loc in
  let who (e : event) loc = Printf.sprintf "%s, warp %d%s" (Sass.opcode e.ins) e.ewarp (if e.eslot <> loc then " of another CTA" else "") in

  (* ---- threads ---- *)
  let next_tid = ref 0 and next_eid = ref 0 in
  let thread ~slot ~warp ~warp_thread vc =
    incr next_tid;
    { tid = !next_tid; clk = 1; vc = IM.add !next_tid { c = 1; via = program; prev = None } vc; tslot = slot; twarp = warp; warp_thread }
  in
  let tick t = t.clk <- t.clk + 1; t.vc <- IM.add t.tid { c = t.clk; via = program; prev = None } t.vc in
  (* what [t] learns at a synchronisation: every clock above its own,
     through it *)
  let syncs = Hashtbl.create 16 and sync_names = ref [||] in
  let sync_id s =
    match Hashtbl.find_opt syncs s with
    | Some i -> i
    | None -> let i = Hashtbl.length syncs in Hashtbl.replace syncs s i; sync_names := Array.append !sync_names [| s |]; i
  in
  let learn t (src : vc) via =
    if hb then begin
      let v = sync_id via in
      Hashtbl.replace passed via ();
      t.vc <- IM.fold (fun u (e : entry) acc -> if u <> t.tid && e.c > clock acc u then IM.add u { c = e.c; via = v; prev = IM.find_opt u acc } acc else acc) src t.vc
    end
  in
  (* what a thread issued by [t] starts from: [t]'s knowledge, its own
     events through the issue *)
  let descend (q : thread) (t : thread) =
    if hb then
      q.vc <-
        IM.fold
          (fun u (e : entry) acc ->
            if u <> q.tid && e.c > clock acc u then IM.add u (if u = t.tid then { c = e.c; via = issued; prev = IM.find_opt u acc } else e) acc else acc)
          t.vc q.vc
  in
  let via_name v = if v = issued then "issued after it" else if v = program then "program order" else !sync_names.(v) in
  let event ?(strong = false) (t : thread) ins ~async =
    incr next_eid;
    { eid = !next_eid; eth = t.tid; eclk = t.clk; ins; eslot = t.tslot; ewarp = t.twarp; async; strong }
  in

  (* ---- shadow memory: shared and global memory 4 bytes a cell, tensor
     memory a column of a lane quarter ---- *)
  let smem = Array.make nslots [||] and tmem = Array.make nslots [||] and gcells = Hashtbl.create 1024 in
  let tstate = Array.init nslots (fun _ -> Array.make (4 * 512) t_free) in
  let cells arr s size = if Array.length arr.(s) = 0 then arr.(s) <- Array.init size (fun _ -> { w = None; rs = []; adds = [] }); arr.(s) in
  (* the clocks at which each thread fenced its writes: for the copy engine
     and the tensor core, and at the GPU's scope *)
  let fences = Hashtbl.create 8 and gfences = Hashtbl.create 8 in
  let fenced tbl (e : event) (v : vc) =
    let upto = clock v e.eth in
    List.exists (fun f -> f > e.eclk && f <= upto) (Option.value (Hashtbl.find_opt tbl e.eth) ~default:[])
  in
  let last_pair = ref (0, 0) and pairs = Hashtbl.create 1024 in
  let count_row key = match Hashtbl.find_opt rows key with Some r -> incr r | None -> Hashtbl.replace rows key (ref 1) in
  (* [e1], then [e2] by [t2], at the same place: ordered, and fenced where
     the place needs it *)
  let order ?(proxied = false) ?(global = false) ~buffer ~loc ~at ~w1 ~w2 (e1 : event) (e2 : event) (t2 : thread) =
    (* two strong accesses are the synchronisation itself *)
    if e1.eth <> t2.tid && !last_pair <> (e1.eid, e2.eid) && not (e1.strong && e2.strong) then begin
      last_pair := (e1.eid, e2.eid);
      match IM.find_opt e1.eth t2.vc with
      | Some x when x.c >= e1.eclk ->
        if proxied && w1 && (not e1.async) && e2.async && not (fenced fences e1 t2.vc) then
          say races
            (Printf.sprintf "%s: %s writes it and %s reads it through the async proxy, with no proxy fence between" (loc ()) (who e1 at) (who e2 at));
        if global && w1 && (not e1.strong) && (not e1.async) && e1.eslot <> e2.eslot && not (fenced gfences e1 t2.vc) then
          say races
            (Printf.sprintf "%s: %s writes it and %s of another CTA reads it, with no MEMBAR.ALL.GPU after the write ordered before the read" (loc ())
               (who e1 e1.eslot) (who e2 e1.eslot));
        if not (Hashtbl.mem pairs (e1.eid, e2.eid)) then begin
          Hashtbl.replace pairs (e1.eid, e2.eid) ();
          count_row (buffer, side e1 at, side e2 at, via_name (first_via t2.vc e1.eth e1.eclk))
        end
      | _ ->
        let does w = if w then "writes" else "reads" in
        say races (Printf.sprintf "%s: %s %s it and %s %s it, and nothing orders the two" (loc ()) (who e1 at) (does w1) (who e2 at) (does w2))
    end
  in
  (* the latest event of each thread from outside a CTA's warps that
     touched its memory, for the exit check *)
  let foreign = Array.init nslots (fun _ -> (Hashtbl.create 64 : (int, event * string) Hashtbl.t)) in
  let note_foreign s (t : thread) e buffer = if not (t.warp_thread && t.tslot = s) then Hashtbl.replace foreign.(s) t.tid (e, buffer) in
  let touch_cell ?proxied ?global ?(add = false) ~buffer ~loc ~at (c : cell) (e : event) t ~write =
    (match c.w with Some w -> order ?proxied ?global ~buffer ~loc ~at ~w1:true ~w2:(write || add) w e t | None -> ());
    (* a reduce-add is ordered against every other access, but not against
       the other adds *)
    if not add then List.iter (fun a -> order ?proxied ?global ~buffer ~loc ~at ~w1:true ~w2:write a e t) c.adds;
    if add then begin
      List.iter (fun r -> order ?proxied ?global ~buffer ~loc ~at ~w1:false ~w2:true r e t) c.rs;
      c.adds <- e :: List.filter (fun (a : event) -> a.eth <> e.eth) c.adds
    end
    else if write then begin
      List.iter (fun r -> order ?proxied ?global ~buffer ~loc ~at ~w1:false ~w2:true r e t) c.rs;
      c.w <- Some e;
      c.rs <- [];
      c.adds <- []
    end
    else c.rs <- e :: List.filter (fun (r : event) -> r.eth <> e.eth) c.rs
  in
  let rank_in_cluster ra (e : event) =
    rank_ok ra || (say protocol (Printf.sprintf "%s names CTA %d, outside the cluster of %d" (who e e.eslot) ((ra lsr 24) land 0xff) nc); false)
  in
  (* [len] bytes of shared memory from run address [ra] *)
  let shared_access t e ra len ~write =
    if hb then begin
      let off = ra land 0xffffff in
      if off + len > 0x40000 then say memory (Printf.sprintf "%s touches shared 0x%x, past shared memory" (who e e.eslot) off)
      else if rank_in_cluster ra e then begin
        let s = slot_of ra in
        let cs = cells smem s 0x10000 in
        let buffer = buffer_of ra in
        note_foreign s t e buffer;
        for i = off / 4 to (off + len - 1) / 4 do
          touch_cell ~proxied:true ~buffer ~loc:(fun () -> place ((ra land lnot 0xffffff) lor (4 * i))) ~at:s cs.(i) e t ~write
        done
      end
    end
  in
  (* [len] bytes of global memory from [a]; not the stamp buffer's, where
     several warps write an event's word and any one's time is the event's *)
  let stamps = match k.timing with No_timing -> -1 | _ -> List.length k.params in
  let global_access ?add t e a len ~write =
    if hb && (a lsr pointer_bits) - 1 <> stamps then begin
      let buffer = global_buffer a in
      for i = a / 4 to (a + len - 1) / 4 do
        let c = match Hashtbl.find_opt gcells i with Some c -> c | None -> let c = { w = None; rs = []; adds = [] } in Hashtbl.replace gcells i c; c in
        touch_cell ~global:true ?add ~buffer ~loc:(fun () -> global_place (4 * i)) ~at:e.eslot c e t ~write
      done
    end
  in
  let tmem_name s q c = Printf.sprintf "%stensor memory lanes %d-%d column %d" (cta_name s) (32 * q) ((32 * q) + 31) c in
  (* [len] columns of a lane quarter from run address CTA << 24 | lane << 16
     | column; [state] what the access does to the columns' contents *)
  let tmem_access ?state t e ra len ~write =
    if hb && rank_in_cluster ra e then begin
      let s = slot_of ra and q = ((ra lsr 16) land 0x7f) / 32 and col = ra land 0xffff in
      let cs = cells tmem s (4 * 512) in
      note_foreign s t e "tensor memory";
      for c = col to min 511 (col + len - 1) do
        touch_cell ~buffer:"tensor memory" ~loc:(fun () -> tmem_name s q c) ~at:s cs.((q * 512) + c) e t ~write;
        Option.iter (fun f -> f s q c) state
      done
    end
  in
  (* what an MMA, a load, an allocation does to the columns' contents *)
  let said = Hashtbl.create 8 in
  let once (e : event) kind msg = if not (Hashtbl.mem said (e.ins.id, kind)) then (Hashtbl.replace said (e.ins.id, kind) (); say memory (msg ())) in
  let mma_state (e : event) ~accumulate s q c =
    let st = tstate.(s).((q * 512) + c) in
    if accumulate && st = t_free then
      once e `Fresh (fun () -> Printf.sprintf "%s adds to %s, which no MMA has written since its allocation" (who e s) (tmem_name s q c))
    else if accumulate && st = t_read then
      once e `Read (fun () -> Printf.sprintf "%s adds to %s, which has been read out: a tile's first MMA overwrites" (who e s) (tmem_name s q c))
    else if (not accumulate) && st = t_live then
      once e `Live (fun () -> Printf.sprintf "%s overwrites %s, which holds results no load has read" (who e s) (tmem_name s q c));
    tstate.(s).((q * 512) + c) <- t_live
  in
  let load_state (e : event) s q c =
    let st = tstate.(s).((q * 512) + c) in
    if st = t_free then once e `Unwritten (fun () -> Printf.sprintf "%s reads %s, which no MMA has written" (who e s) (tmem_name s q c))
    else tstate.(s).((q * 512) + c) <- t_read
  in
  let fresh_state s q c = tstate.(s).((q * 512) + c) <- t_free in

  (* ---- global memory, the parameters ---- *)
  let gload a = match Hashtbl.find_opt gmem a with Some v -> v | None -> (match written with Some w when Hashtbl.mem w a -> 0 | _ -> 1) in
  (* a store hands a word over unless its warp read the word first -- a
     flag taken and put down again *)
  let gstore ~loaded a v = Hashtbl.replace gmem a v; if not (Hashtbl.mem loaded a) then Option.iter (fun w -> Hashtbl.replace w a ()) writes in
  (* the clocks a strong store left at a word *)
  let released = Hashtbl.create 16 in
  (* a global access from [a]: inside the pointer parameter it addresses *)
  let in_param (e : event) a bytes =
    let i = (a lsr pointer_bits) - 1 and off = a land ((1 lsl pointer_bits) - 1) in
    match param i with
    | Some (Ptr p) ->
      if off + bytes > p.rows * p.cols * p.elem then
        once e `Past (fun () -> Printf.sprintf "%s touches %s+0x%x, past its %d bytes" (who e e.eslot) p.pname off (p.rows * p.cols * p.elem))
    | Some (Tmap t) -> once e `Map (fun () -> Printf.sprintf "%s addresses %s, a tensor map, as memory" (who e e.eslot) t.pname)
    | None -> once e `Nowhere (fun () -> Printf.sprintf "%s touches global 0x%x, which no parameter gives" (who e e.eslot) a)
  in
  (* a copy-engine operation's map: a tensor-map parameter, its box *)
  let tmap (e : event) map =
    match param ((map lsr pointer_bits) - 1) with
    | Some (Tmap t) -> Some (t.rows, t.cols, t.elem, t.box_rows, t.box_cols)
    | Some (Ptr p) -> once e `Ptr (fun () -> Printf.sprintf "%s takes %s, a pointer, as its tensor map" (who e e.eslot) p.pname); None
    | None -> once e `Nomap (fun () -> Printf.sprintf "%s takes 0x%x as its tensor map, which no parameter gives" (who e e.eslot) map); None
  in
  (* a box at shared run address [ra]: inside the copy it starts in *)
  let fits (e : event) ra bytes =
    match region (ra land 0xffffff) with
    | Some g ->
      let within = ((ra land 0xffffff) - g.first) mod g.stride in
      if within + bytes > g.stride then
        once e `Box (fun () -> Printf.sprintf "%s moves %d bytes at %s, past the copy's %d" (who e e.eslot) bytes (place ra) g.stride)
    | None -> once e `Unplaced (fun () -> Printf.sprintf "%s moves a box at %s, in no buffer the kernel declares" (who e e.eslot) (place ra))
  in

  (* ---- mbarriers, at their run addresses ---- *)
  let mbars : (int, mbar) Hashtbl.t = Hashtbl.create 32 in
  let settle b = if b.pending = 0 && b.tx = 0 then begin b.last <- b.acc; b.acc <- IM.empty; b.phase <- b.phase + 1; b.pending <- b.count end in
  (* the barrier at [ra], its use ordered after its initialisation *)
  let barrier t e ra =
    match Hashtbl.find_opt mbars ra with
    | None -> None
    | Some b ->
      if hb then begin
        let s = slot_of ra and buffer = buffer_of ra in
        order ~buffer ~loc:(fun () -> place ra) ~at:s ~w1:true ~w2:false b.init e t;
        note_foreign s t e buffer
      end;
      Some b
  in
  let arrive t e ra ~tx =
    match barrier t e ra with
    | None -> say protocol (Printf.sprintf "%s arrives at %s, where no barrier was initialised" (who e e.eslot) (place ra))
    | Some b ->
      b.tx <- b.tx + tx;
      b.pending <- b.pending - 1;
      if hb then b.acc <- vc_join b.acc t.vc;
      if b.pending < 0 then
        say protocol (Printf.sprintf "%s arrives at %s in phase %d after its %d arrivals, while %d bytes are still to come" (who e e.eslot) (place ra) b.phase b.count b.tx);
      settle b
  in
  let complete_tx t e ra bytes =
    match barrier t e ra with
    | None -> say protocol (Printf.sprintf "%s reports %d bytes to %s, where no barrier was initialised" (who e e.eslot) bytes (place ra))
    | Some b ->
      b.tx <- b.tx - bytes;
      if hb then b.acc <- vc_join b.acc t.vc;
      settle b
  in
  let parity_passes b parity = b.phase land 1 <> (parity lsr 31) land 1 in
  let ranks_of mask = List.filter (fun r -> mask land (1 lsl r) <> 0) (List.init nc Fun.id) in

  (* ---- named barriers, the clusters' barriers ---- *)
  let nbars = Array.init nslots (fun _ -> Hashtbl.create 4) in
  let nbar s b =
    match Hashtbl.find_opt nbars.(s) b with
    | Some x -> x
    | None -> let x = { arrived = 0; gen = 0; nacc = IM.empty; gens = Hashtbl.create 4 } in Hashtbl.replace nbars.(s) b x; x
  in
  let carv = Array.make (nslots * nw) 0 and cwait = Array.make (nslots * nw) 0 and cgens = Hashtbl.create 4 in
  let finished = Array.make (nslots * nw) false in
  let cluster_warps j = List.init (nc * nw) (fun x -> (j * nc * nw) + x) in

  (* ---- the warps ---- *)
  let warps =
    Array.init (nslots * nw) (fun i ->
      let s = i / nw and w = i mod nw in
      let j = s / nc and r = s mod nc in
      let cta = cta_of.(s) in
      let th = thread ~slot:s ~warp:w ~warp_thread:true IM.empty in
      let pipe = thread ~slot:s ~warp:w ~warp_thread:false IM.empty in
      let uniform = Hashtbl.create 64 in
      let lanes = Array.init 32 (fun _ -> Hashtbl.create 128) in
      let file lane = function Sass.UR _ | UP _ -> uniform | _ -> lanes.(lane) in
      let get lane reg = Option.value (Hashtbl.find_opt (file lane reg) reg) ~default:0 in
      let cur = ref None and cur_ins = ref None and loaded = Hashtbl.create 8 in
      (* this instruction's execution, as the warp's access *)
      let ev () =
        match !cur with
        | Some e -> e
        | None -> let e = event th (Option.get !cur_ins) ~async:false in cur := Some e; e
      in
      let machine lane : Sass.machine =
        { get = get lane
        ; set = (fun reg v -> Hashtbl.replace (file lane reg) reg v)
        ; special =
            (function
              | "SR_TID.X" -> (32 * w) + lane
              | "SR_CTAID.X" -> cta
              | "SR_CgaCtaId" -> r
              | _ -> 0)
        ; const =
            (fun off ->
              (* parameter i, a pointer: (i + 1) << 40 *)
              if off >= 0x380 && off < 0x380 + (8 * 64) then begin
                let p = (off - 0x380) / 8 in
                if p >= List.length params then
                  once (ev ()) `Param (fun () -> Printf.sprintf "%s reads parameter %d, and the kernel declares %d" (who (ev ()) s) p (List.length params));
                if (off - 0x380) mod 8 = 4 then (p + 1) lsl (pointer_bits - 32) else 0
              end
              else if off = 0x358 || off = 0x35c || off = 0x37c then if off = 0x358 then 0x1000 else 0
              else begin
                once (ev ()) `Const (fun () -> Printf.sprintf "%s reads the constant bank at 0x%x, which holds nothing the kernel declares" (who (ev ()) s) off);
                0
              end)
        ; load = (fun a -> let ra = in_run j a in shared_access th (ev ()) ra 4 ~write:false; Option.value (Hashtbl.find_opt shared ra) ~default:0)
        ; store = (fun a v -> let ra = in_run j a in shared_access th (ev ()) ra 4 ~write:true; Hashtbl.replace shared ra v)
        ; gload = (fun a -> Hashtbl.replace loaded a (); gload a)
        ; gstore = gstore ~loaded
        ; alloc =
            (fun chunks ->
              let c0 = next_chunk.(s) in
              next_chunk.(s) <- c0 + chunks;
              for q = 0 to 3 do
                tmem_access ~state:fresh_state th (ev ()) (in_run j ((r lsl 24) lor ((32 * q) lsl 16) lor (32 * c0))) (32 * chunks) ~write:true
              done;
              c0)
        ; lane }
      in
      let machines = Array.init 32 machine in
      let read l cls width u =
        let unit u = match cls with R -> Sass.R u | UR -> Sass.UR u | P -> Sass.P u | UP -> Sass.UP u in
        if width = 1 then get l (unit u) else get l (unit u) lor (get l (unit (u + 1)) lsl 32)
      in
      let value l (x : raw) = read l x.cls x.width x.id in
      let groups = ref [ 0, List.init 32 Fun.id ] in
      let steps = ref 0 in
      let at_bar = ref None and checked_at = ref (-1) in
      let open_group = ref [] and closed = Queue.create () in
      let exit_vc = ref IM.empty in
      (* a strong load's place, value and step: read again unchanged a few
         instructions later, the warp spins *)
      let last_strong = ref (-1, 0, 0) and spin = ref None in
      let ctx = function "cta" -> cta | "rank" -> r | _ -> 0 in
      let env l = { value = (fun x -> value l (Core.raw x)); ctx } in
      (* the instruction at the group furthest behind: run it, or say the
         warp waits *)
      let step () =
        let pc = List.fold_left (fun a (p, _) -> min a p) max_int !groups in
        let here, rest = List.partition (fun (p, _) -> p = pc) !groups in
        let lanes_here = List.concat_map snd here in
        incr steps;
        if !steps > step_limit then failwith (Printf.sprintf "Interp: CTA %d warp %d ran past %d steps" cta w step_limit);
        if pc >= n then failwith (Printf.sprintf "Interp: CTA %d warp %d ran off the end" cta w);
        match body.(pc) with
        | Sass.Label _ -> groups := (pc + 1, lanes_here) :: rest; `Ran
        | I ins ->
          let active l = match ins.guard with None -> true | Some (g, neg) -> (get l g <> 0) <> neg in
          let on, off = List.partition active lanes_here in
          let first = match on with l :: _ -> Some machines.(l) | [] -> None in
          cur := None;
          cur_ins := Some ins;
          let act = match ins.act, first with Some f, Some m -> Some (f m) | _ -> None in
          (* the operands as the warp reaches the instruction, whether or
             not it then waits *)
          if !checked_at <> pc then begin
            checked_at := pc;
            List.iter (fun l -> List.iter (fun c -> on_check ~cta ~w ~l ~ins ~c ~got:(value l c.operand) ~read:(read l) ~env:(env l)) (Option.value (Hashtbl.find_opt checks ins.id) ~default:[])) on
          end;
          (* a wait that has not happened: the warp waits *)
          let blocked =
            match act, ins.named_barrier with
            | Some (Sass.Wait { bar; parity; ahead = false }), _ ->
              (match Hashtbl.find_opt mbars (in_run j bar) with Some b -> not (parity_passes b parity) | None -> true)
            | Some Sass.Cluster_wait, _ ->
              let g = cwait.(i) in
              not (List.for_all (fun x -> finished.(x) || carv.(x) > g) (cluster_warps j))
            | _, Some b when on <> [] ->
              let nb = nbar s b in
              let g =
                match !at_bar with
                | Some (b', g) when b' = b -> g
                | _ ->
                  (* the arrival, once *)
                  if List.length on < 32 then
                    problems := Printf.sprintf "CTA %d warp %d arrives at barrier %d with %d of its 32 lanes -- %s" cta w b (List.length on) (Sass.to_string ins) :: !problems;
                  let by_warp = match Hashtbl.find_opt arrivals.(s) b with Some t -> t | None -> let t = Hashtbl.create 8 in Hashtbl.replace arrivals.(s) b t; t in
                  let count = bar_count k ins in
                  Hashtbl.replace by_warp w (count :: Option.value (Hashtbl.find_opt by_warp w) ~default:[]);
                  let g = nb.gen in
                  nb.arrived <- nb.arrived + 32;
                  if hb then nb.nacc <- vc_join nb.nacc th.vc;
                  if nb.arrived >= count then begin
                    Hashtbl.replace nb.gens g nb.nacc;
                    nb.gen <- g + 1;
                    nb.arrived <- 0;
                    nb.nacc <- IM.empty
                  end;
                  at_bar := Some (b, g);
                  g
              in
              nb.gen <= g
            | _ -> false
          in
          if blocked then `Blocked
          else begin
            groups := rest;
            checked_at := -1;
            (match on_exec with Some f when on <> [] -> f s w ins | _ -> ());
            (match ins.named_barrier, !at_bar with
             | Some b, Some (b', g) when b = b' ->
               learn th (Hashtbl.find (nbar s b).gens g) (Printf.sprintf "named barrier %d" b);
               at_bar := None
             | _ -> ());
            (* what the instruction does to the barriers, the copy engine,
               the tensor core and global memory *)
            let answer = ref None and spun = ref 0 in
            (match act with
             | None -> ()
             | Some a ->
               (match a with
                | Sass.Init { bar; count } ->
                  let e = ev () and ra = in_run j bar in
                  if hb then note_foreign (slot_of ra) th e (buffer_of ra);
                  Hashtbl.replace mbars ra { count; pending = count; tx = 0; phase = 0; acc = IM.empty; last = IM.empty; init = e }
                | Arrive _ ->
                  List.iter
                    (fun l ->
                      match ins.act with
                      | Some f -> (match f machines.(l) with Sass.Arrive { bar; tx } -> arrive th (ev ()) (in_run j bar) ~tx | _ -> ())
                      | None -> ())
                    on
                | Global _ ->
                  List.iter
                    (fun l ->
                      match ins.act with
                      | Some f ->
                        (match f machines.(l) with
                         | Sass.Global { addr; bytes; write; strong } ->
                           let e = if strong then event ~strong th ins ~async:false else ev () in
                           in_param e addr bytes;
                           (* a strong load takes what the strong store it reads released *)
                           if strong && not write then
                             (match Hashtbl.find_opt released addr with
                              | Some v -> learn th v ("strong load of " ^ global_buffer addr)
                              | None -> ());
                           global_access th e addr bytes ~write;
                           if strong && write && hb then Hashtbl.replace released addr th.vc;
                           if strong && not write then begin
                             (* the first lane's word: read again unchanged, a spin *)
                             if l = List.hd on then begin
                               let v = gload addr and p, v0, st = !last_strong in
                               if p = pc && v = v0 && !steps - st < 16 then (spun := !steps - st; spin := Some (addr, v));
                               last_strong := (pc, v, !steps)
                             end
                           end
                         | _ -> ())
                      | None -> ())
                    on
                | Gpu_fence -> if hb then Hashtbl.replace gfences th.tid (th.clk :: Option.value (Hashtbl.find_opt gfences th.tid) ~default:[])
                | Wait { bar; parity; ahead } ->
                  let ra = in_run j bar in
                  if (bar lsr 24) land 0xff <> r then say protocol (Printf.sprintf "CTA %d warp %d waits on %s, another CTA's barrier" cta w (place ra));
                  (match barrier th (ev ()) ra with
                   | Some b when parity_passes b parity ->
                     learn th b.last ("wait on " ^ buffer_of ra);
                     answer := Some 1
                   | Some _ -> answer := Some 0
                   | None ->
                     if ahead then (say protocol (Printf.sprintf "CTA %d warp %d asks about %s, where no barrier was initialised" cta w (place ra)); answer := Some 0))
                | Load { dst; bar; map; mask; two } ->
                  let q = thread ~slot:s ~warp:w ~warp_thread:false IM.empty in
                  descend q th;
                  let e = event q ins ~async:true in
                  (match tmap e map with
                   | None -> ()
                   | Some (_, _, elem, box_rows, box_cols) ->
                     let bytes = box_rows * box_cols * elem in
                     let dests = match mask with None -> [ (dst lsr 24) land 0xff ] | Some m -> ranks_of ((m lsr 16) land 0xffff) in
                     List.iter
                       (fun d ->
                         let ra = in_run j ((d lsl 24) lor (dst land 0xffffff)) in
                         fits e ra bytes;
                         shared_access q e ra bytes ~write:true;
                         (* a multicast's bytes report at the barrier's offset in
                            each CTA it lands in -- the pair's first, for a load
                            that feeds a two-CTA MMA *)
                         let target = if mask = None then bar else ((if two then d land lnot 1 else d) lsl 24) lor (bar land 0xffffff) in
                         complete_tx q e (in_run j target) bytes)
                       dests)
                | Store { src; map; x; y; add } ->
                  let q = thread ~slot:s ~warp:w ~warp_thread:false IM.empty in
                  descend q th;
                  let e = event q ins ~async:true in
                  (match tmap e map with
                   | None -> ()
                   | Some (rows, mcols, elem, box_rows, box_cols) ->
                     let ra = in_run j src in
                     fits e ra (box_rows * box_cols * elem);
                     shared_access q e ra (box_rows * box_cols * elem) ~write:false;
                     (* the box's rows inside the matrix: the copy engine clips the rest *)
                     let base = map land lnot ((1 lsl pointer_bits) - 1) in
                     let cols = max 0 (min box_cols (mcols - x)) in
                     for row = y to min (rows - 1) (y + box_rows - 1) do
                       if cols > 0 then global_access ~add q e (base + (((row * mcols) + x) * elem)) (cols * elem) ~write:(not add)
                     done);
                  open_group := q.vc :: !open_group
                | Flush -> Queue.push !open_group closed; open_group := []
                | Drain m ->
                  while Queue.length closed > m do
                    List.iter (fun v -> learn th v (Sass.to_string ins)) (Queue.pop closed)
                  done
                | Proxy_fence -> if hb then Hashtbl.replace fences th.tid (th.clk :: Option.value (Hashtbl.find_opt fences th.tid) ~default:[])
                | Mma { accumulate } ->
                  descend pipe th;
                  let e = event pipe ins ~async:true in
                  let l = List.hd on in
                  List.iter
                    (fun (t : touch) ->
                      List.iter
                        (fun (a, len) ->
                          match t.space with
                          | Smem -> shared_access pipe e (in_run j a) len ~write:t.write
                          | Tmem -> tmem_access ~state:(mma_state e ~accumulate) pipe e (in_run j a) len ~write:t.write)
                        (t.runs (env l)))
                    (Option.value (Hashtbl.find_opt touches ins.id) ~default:[]);
                  tick pipe
                | Commit { bar; mask } ->
                  descend pipe th;
                  let e = event pipe ins ~async:true in
                  let targets = match mask with None -> [ bar ] | Some m -> List.map (fun d -> (d lsl 24) lor (bar land 0xffffff)) (ranks_of (m land 0xffff)) in
                  List.iter (fun a -> arrive pipe e (in_run j a) ~tx:0) targets;
                  tick pipe
                | Cluster_arrive ->
                  let g = carv.(i) in
                  carv.(i) <- g + 1;
                  if hb then Hashtbl.replace cgens (j, g) (vc_join (Option.value (Hashtbl.find_opt cgens (j, g)) ~default:IM.empty) th.vc)
                | Cluster_wait ->
                  let g = cwait.(i) in
                  cwait.(i) <- g + 1;
                  learn th (Option.value (Hashtbl.find_opt cgens (j, g)) ~default:IM.empty) "cluster barrier"
                | Request { answer = ans; bar } ->
                  let q = thread ~slot:s ~warp:w ~warp_thread:false IM.empty in
                  descend q th;
                  let e = event q ins ~async:true in
                  List.iter
                    (fun d ->
                      let ra = in_run j ((d lsl 24) lor ans) in
                      shared_access q e ra 16 ~write:true;
                      (* none left: the answer's words are zero *)
                      for x = 0 to 3 do Hashtbl.replace shared (ra + (4 * x)) 0 done;
                      complete_tx q e (in_run j ((d lsl 24) lor bar)) 16)
                    (List.init nc Fun.id)
                | Tmem_load { addr; n } -> tmem_access ~state:(load_state (ev ())) th (ev ()) (in_run j ((r lsl 24) lor addr)) n ~write:false
                | Tmem_free chunks ->
                  for c = 0 to 15 do
                    if chunks land (1 lsl c) <> 0 then
                      for q = 0 to 3 do tmem_access ~state:fresh_state th (ev ()) (in_run j ((r lsl 24) lor ((32 * q) lsl 16) lor (32 * c))) 32 ~write:true done
                  done));
            (* an instruction that writes uniform registers alone runs once
               for the warp; one that writes a lane's registers, for each
               lane *)
            let uniform_only = ins.defs <> [] && List.for_all (function Sass.UR _ | UP _ -> true | _ -> false) ins.defs in
            Option.iter (fun f -> match on with l :: _ when uniform_only -> f machines.(l) | _ -> List.iter (fun l -> f machines.(l)) on) ins.sem;
            (* a question asked ahead is answered as the barrier stands *)
            (match !answer, ins.defs with Some v, [ p ] -> List.iter (fun l -> machines.(l).set p v) on | _ -> ());
            tick th;
            let next = pc + 1 in
            let add p ls = if ls <> [] then groups := (p, ls) :: !groups in
            (match ins.branch with
             | `No -> add next lanes_here
             | `Exit -> add next off
             | `Bra l ->
               let t = match Hashtbl.find_opt labels l with Some t -> t | None -> failwith ("Interp: no label " ^ l) in
               add t on;
               add next off);
            if !groups = [] then begin
              finished.(i) <- true;
              exit_vc := th.vc;
              `Done
            end
            else if !spun > 0 then `Spun !spun
            else (spin := None; `Ran)
          end
      in
      let describe_wait () =
        let pc = List.fold_left (fun a (p, _) -> min a p) max_int !groups in
        match !spin, body.(pc) with
        | Some (a, v), _ -> Printf.sprintf "CTA %d warp %d spins on %s, which holds %d and nothing in the run changes" cta w (global_place a) v
        | None, Sass.I ins ->
          let lane = List.hd (List.concat_map snd (List.filter (fun (p, _) -> p = pc) !groups)) in
          (match Option.map (fun f -> f machines.(lane)) ins.act, ins.named_barrier with
           | Some (Sass.Wait { bar; parity; _ }), _ ->
             let ra = in_run j bar in
             (match Hashtbl.find_opt mbars ra with
              | Some b ->
                Printf.sprintf "CTA %d warp %d waits on %s for the phase of parity %d: it is in phase %d, %d of %d arrivals to come, %d bytes" cta w (place ra)
                  ((parity lsr 31) land 1) b.phase b.pending b.count b.tx
              | None -> Printf.sprintf "CTA %d warp %d waits on %s, where no barrier was ever initialised" cta w (place ra))
           | Some Sass.Cluster_wait, _ ->
             let g = cwait.(i) in
             let missing = List.filter (fun x -> not (finished.(x) || carv.(x) > g)) (cluster_warps j) in
             Printf.sprintf "CTA %d warp %d waits at the cluster barrier for warps %s" cta w
               (String.concat ", " (List.map (fun x -> Printf.sprintf "%d of CTA %d" (x mod nw) cta_of.(x / nw)) missing))
           | _, Some b ->
             let nb = nbar s b in
             Printf.sprintf "CTA %d warp %d waits at named barrier %d: %d of %d threads have arrived" cta w b nb.arrived (bar_count k ins)
           | _ -> Printf.sprintf "CTA %d warp %d waits at %s" cta w (Sass.to_string ins))
        | None, Label l -> Printf.sprintf "CTA %d warp %d stopped at %s" cta w l
      in
      (step, describe_wait, (fun () -> !exit_vc)))
  in
  (* each warp in turn, until it waits or spins; a round in which none moves
     with some not out is a hang *)
  let quantum = 100_000 in
  let rec rounds () =
    let moved = ref false in
    Array.iteri
      (fun i (step, _, _) ->
        if not finished.(i) then begin
          (* a turn that only went once round a spin loop moved nothing *)
          let go = ref true and m = ref 0 in
          while !go && !m < quantum do
            match step () with
            | `Ran -> incr m
            | `Done -> incr m; go := false
            | `Blocked -> go := false
            | `Spun loop -> if !m + 1 > loop then moved := true; m := 0; go := false
          done;
          if !m > 0 then moved := true
        end)
      warps;
    if Array.exists not finished then if !moved then rounds () else Array.iteri (fun i (_, d, _) -> if not finished.(i) then say protocol ("hang: " ^ d ())) warps
  in
  rounds ();
  if Array.for_all Fun.id finished then begin
    (* every mbarrier at the start of a phase *)
    Hashtbl.fold (fun a b acc -> (a, b) :: acc) mbars []
    |> List.sort compare
    |> List.iter (fun (a, b) ->
         if b.pending <> b.count || b.tx <> 0 then
           say protocol
             (Printf.sprintf "%s is left part-way through phase %d: %d of its %d arrivals made, %d bytes %s" (place a) b.phase (b.count - b.pending) b.count
                (abs b.tx) (if b.tx > 0 then "still to come" else "more than it was told")));
    (* what touched a CTA's memory from outside its warps, ordered before
       they have all exited *)
    if hb then
      Array.iteri
        (fun s tbl ->
          let exits = List.init nw (fun w -> let _, _, x = warps.((s * nw) + w) in x ()) in
          Hashtbl.iter
            (fun _ ((e : event), buffer) ->
              (* the warp that issued it first: what it waited for before leaving *)
              let own = if e.eslot = s then [ List.nth exits e.ewarp ] else [] in
              match List.find_opt (fun v -> clock v e.eth >= e.eclk) (own @ exits) with
              | Some v -> count_row (buffer, side e s, ("the CTA's last exit", -1, false), via_name (first_via v e.eth e.eclk))
              | None ->
                say races (Printf.sprintf "%s touches %s%s, and nothing orders it before that CTA's warps have all exited" (who e s) (cta_name s) buffer))
            tbl)
        foreign
  end;
  (* the named barriers, phase by phase: the k-th arrival of every warp that
     arrives k times *)
  Array.iteri
    (fun s tbl ->
      let cta = cta_of.(s) in
      Hashtbl.iter
        (fun b by_warp ->
          let seqs = Hashtbl.fold (fun w cs acc -> (w, Array.of_list (List.rev cs)) :: acc) by_warp [] |> List.sort compare in
          let phases = List.fold_left (fun m (_, a) -> max m (Array.length a)) 0 seqs in
          for i = 0 to phases - 1 do
            let here = List.filter_map (fun (w, a) -> if i < Array.length a then Some (w, a.(i)) else None) seqs in
            let names = String.concat "," (List.map (fun (w, _) -> string_of_int w) here) in
            match List.sort_uniq compare (List.map snd here) with
            | [ c ] when c = 32 * List.length here -> ()
            | [ c ] ->
              problems :=
                Printf.sprintf "CTA %d barrier %d, arrival %d: its count is %d threads, and warps %s arrive -- %d threads" cta b (i + 1) c names (32 * List.length here)
                :: !problems
            | cs ->
              problems :=
                Printf.sprintf "CTA %d barrier %d, arrival %d: warps %s arrive with different counts (%s)" cta b (i + 1) names (String.concat ", " (List.map string_of_int cs))
                :: !problems
          done)
        tbl)
    arrivals;
  List.rev !problems, List.rev !protocol, List.rev !memory, List.rev !races

(* The runs a check makes: the first cluster and the last; a kernel that
   reads global memory -- what another CTA left there -- runs its first two
   clusters together. *)
let runs_of (k : kernel) =
  let g = fst k.grid and c = k.cluster and t = k.together in
  let cluster i = List.init c (fun x -> (i * c) + x) in
  (* the clusters that wait on one another, run together *)
  let group i = List.init t (fun j -> cluster ((i * t) + j)) in
  let last = (g / c / t) - 1 in
  let reads_global = List.exists (function Sass.I i -> String.length (Sass.opcode i) >= 3 && String.sub (Sass.opcode i) 0 3 = "LDG" | Label _ -> false) k.body in
  let first = if reads_global && last >= 1 && t = 1 then group 0 @ group 1 else group 0 in
  let rest = if List.mem (List.hd (group last)) first then [] else [ group last ] in
  (first :: rest), reads_global

(* Every derived operand of [k] against its reference, in the program as
   written and as coloured; the barriers' protocol, memory as declared, and
   the happens-before order in the program as written. *)
let check ?(colored = true) (k : kernel) =
  let body = Array.of_list k.body in
  let checked = ref 0 and failures = ref [] and barriers = ref [] and protocol = ref [] and memory = ref [] and races = ref [] in
  let rows = Hashtbl.create 64 and passed = Hashtbl.create 16 in
  let runs, reads_global = runs_of k in
  (* the global words each run writes, from a run before: what reads 0 until
     it is stored *)
  let written =
    List.map
      (fun clusters ->
        if reads_global then begin
          let w = Hashtbl.create 1024 in
          ignore (run k body ~clusters ~writes:w ~rows:(Hashtbl.create 1) ~passed:(Hashtbl.create 1) ~on_check:(fun ~cta:_ ~w:_ ~l:_ ~ins:_ ~c:_ ~got:_ ~read:_ ~env:_ -> ()));
          Some w
        end
        else None)
      runs
  in
  (* what each checked operand held, in order, for the coloured run *)
  let seen = Hashtbl.create 4096 in
  let queue key = match Hashtbl.find_opt seen key with Some q -> q | None -> let q = Queue.create () in Hashtbl.replace seen key q; q in
  List.iter2
    (fun clusters written ->
      let b, p, m, r =
        run ~hb:true ?written k body ~clusters ~rows ~passed ~on_check:(fun ~cta ~w ~l ~ins ~c ~got ~read:_ ~env ->
          incr checked;
          let wide = c.operand.width = 2 in
          let got = if wide then got + c.imm else m32 (got + c.imm) and want = if wide then c.expect env else m32 (c.expect env) in
          Queue.push got (queue (cta, w, l, ins.Sass.id, c.what));
          if got <> want && List.length !failures < 20 then
            failures := { what = c.what; at = Sass.to_string ins; cta; warp = w; lane = l; got; want } :: !failures)
      in
      let add l xs = l := !l @ List.filter (fun x -> not (List.mem x !l)) xs in
      add barriers b;
      add protocol p;
      add memory m;
      add races r)
    runs written;
  (* the coloured program: each checked operand, through its colour *)
  let mismatches = ref 0 in
  if colored then begin
    let body, _, phys = Compile.colored k in
    List.iter2
      (fun clusters written ->
        ignore
        @@ run ?written { k with body } (Array.of_list body) ~clusters ~rows:(Hashtbl.create 1) ~passed:(Hashtbl.create 1)
             ~on_check:(fun ~cta ~w ~l ~ins ~c ~got:_ ~read ~env:_ ->
               let got = read c.operand.cls c.operand.width (phys c.operand.id) in
               let got = if c.operand.width = 2 then got + c.imm else m32 (got + c.imm) in
               let q = queue (cta, w, l, ins.Sass.id, c.what) in
               if Queue.is_empty q || Queue.pop q <> got then incr mismatches))
      runs written
  end;
  (* the rows that differ in their warps alone, as one *)
  let merged = Hashtbl.create 64 in
  Hashtbl.iter
    (fun (buffer, (o1, w1, x1), (o2, w2, x2), via) c ->
      let key = buffer, (o1, x1), (o2, x2), via in
      let ws1, ws2, n = Option.value (Hashtbl.find_opt merged key) ~default:([], [], 0) in
      Hashtbl.replace merged key (w1 :: ws1, w2 :: ws2, n + !c))
    rows;
  let side (op, remote) ws =
    let ws = List.sort_uniq compare ws in
    let rec spans = function
      | [] -> []
      | a :: rest ->
        let rec run b = function x :: r when x = b + 1 -> run x r | r -> b, r in
        let b, r = run a rest in
        (if b = a then string_of_int a else Printf.sprintf "%d-%d" a b) :: spans r
    in
    match ws with
    | [ -1 ] -> op
    | [ w ] -> Printf.sprintf "%s, warp %d%s" op w (if remote then " of another CTA" else "")
    | _ -> Printf.sprintf "%s, warps %s%s" op (String.concat "," (spans ws)) (if remote then " of another CTA" else "")
  in
  let table =
    Hashtbl.fold (fun (buffer, s1, s2, via) (ws1, ws2, n) acc -> { buffer; first = side s1 ws1; second = side s2 ws2; via; count = n } :: acc) merged []
    |> List.sort (fun a b -> compare (a.buffer, a.first, a.second) (b.buffer, b.first, b.second))
  in
  let used = List.sort_uniq compare (List.map (fun (r : row) -> r.via) table) in
  let idle = List.filter (fun s -> not (List.mem s used)) (List.sort compare (Hashtbl.fold (fun s () acc -> s :: acc) passed [])) in
  { checked = !checked; failures = List.rev !failures; mismatches = !mismatches; barriers = !barriers; protocol = !protocol; memory = !memory; races = !races
  ; table; idle }

let describe (r : report) =
  Printf.sprintf
    "%d derived operands checked, %d wrong, %d changed by the colouring; %d named-barrier arrivals that do not make their counts; %d protocol problems; %d uses of memory past what was declared; %d accesses nothing orders"
    r.checked (List.length r.failures) r.mismatches (List.length r.barriers) (List.length r.protocol) (List.length r.memory) (List.length r.races)
  :: List.map
       (fun f -> Printf.sprintf "  CTA %d warp %d lane %d: %s is 0x%x, the layout says 0x%x -- %s" f.cta f.warp f.lane f.what f.got f.want f.at)
       r.failures
  @ List.map (fun b -> "  " ^ b) (r.barriers @ r.protocol @ r.memory @ r.races)

(* the happens-before table, a line a row *)
let table (r : report) =
  let w f = List.fold_left (fun m x -> max m (String.length (f x))) 0 r.table in
  let wb = max 6 (w (fun x -> x.buffer)) and w1 = max 7 (w (fun x -> x.first)) and w2 = max 6 (w (fun x -> x.second)) in
  Printf.sprintf "%-*s  %-*s  %-*s  %s" wb "buffer" w1 "earlier" w2 "later" "ordered by"
  :: List.map (fun x -> Printf.sprintf "%-*s  %-*s  %-*s  %s  (%d)" wb x.buffer w1 x.first w2 x.second x.via x.count) r.table
  @ if r.idle = [] then [] else "synchronisation that ordered no pair:" :: List.map (fun s -> "  " ^ s) r.idle

(* Which warps of which CTA of the first cluster run each instruction, in
   the host run: by the instruction's place among the kernel's
   instructions, its (rank, warp) pairs *)
let executions (k : kernel) =
  let body = Array.of_list k.body in
  let place = Hashtbl.create 1024 in
  List.iteri (fun i (ins : Sass.insn) -> Hashtbl.replace place ins.id i) (List.filter_map (function Sass.I i -> Some i | Label _ -> None) k.body);
  let seen = Hashtbl.create 4096 in
  let runs, _ = runs_of k in
  ignore
    (run k body ~clusters:(List.hd runs) ~rows:(Hashtbl.create 1) ~passed:(Hashtbl.create 1)
       ~on_check:(fun ~cta:_ ~w:_ ~l:_ ~ins:_ ~c:_ ~got:_ ~read:_ ~env:_ -> ())
       ~on_exec:(fun s w (ins : Sass.insn) -> Hashtbl.replace seen (Hashtbl.find place ins.id, s, w) ()));
  let out = Hashtbl.create 1024 in
  Hashtbl.iter (fun (i, s, w) () -> Hashtbl.replace out i ((s, w) :: Option.value (Hashtbl.find_opt out i) ~default:[])) seen;
  fun i -> List.sort compare (Option.value (Hashtbl.find_opt out i) ~default:[])

