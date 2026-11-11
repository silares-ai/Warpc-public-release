(* Memory as the program sees it: buffers declared with their layouts, and
   operands that are a buffer at coordinates. No program writes an address.
   A buffer's layout takes its index coordinates -- a ring's stage, a copy
   per warp, an accumulator buffer -- to a byte, and its element layout the
   coordinates inside one copy; an instruction's thread-value layout takes a
   lane and a value to element coordinates. The address an instruction needs
   is read off the composite: the strided form over the run-time indices is
   derived into registers (Prog.value), the part a lane's position fixes is
   decided over the warp's lanes, and what the value index adds is the
   instruction's immediate.

   Every operand derived here carries a check: what it must hold where its
   instruction runs, computed from the buffer's layout at the coordinates --
   not from the expression the derivation compiled -- which the host-side
   interpreter (Interp) compares with what the instructions compute. *)

open Layouts
open Core

(* A buffer's space and kind are its type: a barrier operation takes a
   shared barrier, a load of a word a word of its space, an MMA a shared
   tile. *)
type shared = [ `Shared ]
type global = [ `Global ]
type tile = [ `Tile ]
type barrier = [ `Barrier ]
type word = [ `Word ]
type slab = [ `Slab ]

type space =
  | Shared
  | Global

(* what a term's address adds to: a CTA's shared window, or a pointer *)
type base =
  | Win of ur reg
  | Ptr of r2 reg

type buf =
  { bname : string
  ; space : space
  ; modes : (int * int) list (* each index mode: its range, and the bytes a step of it moves *)
  ; elem : (Space.logical, Space.physical) Layout.t option (* one copy; none for a barrier *)
  ; esize : int (* bytes an element *)
  ; copy : int (* bytes one copy takes *)
  ; off : int (* shared: from the shared-memory base register *)
  ; home : base (* this CTA's window, or the buffer's pointer *)
  }

type ('s, 'k) buffer = buf

(* a CTA's shared window, as this CTA addresses it *)
type 's window = ur reg

(* a value over indices, for the operands that are coordinates: its
   expression, the indices it names, and its reference where it has one of
   its own (a map's value read off the map) *)
type value =
  { e : Expr.t
  ; vars : (string * Prog.var) list
  ; ref : (env -> int) option
  }

(* a coordinate of an index mode: fixed at compile time, an index, or a
   value of indices *)
type coord =
  | C of int
  | I : ('k, 'c) Prog.index -> coord
  | V of value

type tm =
  { buf : buf
  ; at : coord list (* one per index mode *)
  ; base : base
  ; el : int * int (* inside one copy: the row and column a sub-tile starts at *)
  }

type ('s, 'k) term = tm

(* [buf] at the coordinates of its index modes, in this CTA or at a peer's
   window *)
let at ?peer (buf : buf) coords : tm = { buf; at = coords; base = (match peer with Some w -> Win w | None -> buf.home); el = (0, 0) }

(* the sub-tile of a term's copy from row [r], column [c] on *)
let sub (t : tm) (r, c) = { t with el = (r, c) }

(* ---- the kernel's memory ---- *)

type kernel =
  { p : Prog.t
  ; smem : ur reg (* this CTA's shared window at 0x400, in the cluster's address form *)
  ; cta : ur reg (* this CTA's rank in the cluster, the CTA field of its addresses *)
  ; k400 : ur reg
  ; cluster : int * int (* CTAs along x, the fastest, and along y *)
  ; rank : (Prog.fixed, ur) Prog.index option (* this CTA's rank in its cluster *)
  ; mutable peers : (int * (int * int)) list (* a peer window's register: the rank bits it fixes, and to what *)
  ; mutable static_used : int (* bytes of the static window, from [smem] *)
  ; mutable dyn_used : int (* bytes of the dynamic region, from [smem] + 0x400 *)
  ; mutable barriers : (buf * int) list (* each barrier array and its arrival count *)
  ; mutable regions : region list (* the shared buffers by name, for the interpreter's reports *)
  }

let static_bytes = 0x400

(* The kernel's shared memory: this CTA's window, and its rank in a cluster
   of [nx] x [ny] CTAs, x the fastest -- the CTA field of the window's
   cluster address. *)
let create ?(cluster = (1, 1)) p =
  let nx, ny = cluster in
  let pow2 n = n land (n - 1) = 0 in
  if not (pow2 nx && pow2 ny) then failwith "Mem.create: a cluster of powers of two";
  let cta = ur "cta" and smem = ur "smem" and k400 = ur "k400" in
  Prog.emit p (s2ur_cta cta);
  Prog.derived p [ umov k400 0x400; ulea smem cta k400 0x18 ];
  let rank = if nx * ny = 1 then None else Some (Prog.fixed p ~name:"rank" ~range:(nx * ny) ~cls:Uniform cta) in
  { p; smem; cta; k400; cluster; rank; peers = []; static_used = 0; dyn_used = 0; barriers = []; regions = [] }

let prog k = k.p
let cta k = k.cta
let rank k = match k.rank with Some r -> r | None -> failwith "Mem.rank: one CTA a cluster"
let ctas k = fst k.cluster * snd k.cluster

type axis =
  | X
  | Y

let extent k = function X -> fst k.cluster | Y -> snd k.cluster
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)

(* the rank bits an axis takes *)
let bits k = function X -> fst k.cluster - 1 | Y -> (snd k.cluster - 1) lsl log2 (fst k.cluster)

(* ---- references: what an operand is, read off the layouts ---- *)

let window rank = (rank lsl 24) + static_bytes
let rank_of (env : env) = env.ctx "rank"

(* a base register's value: this CTA's window, a peer's, or the register's *)
let base_ref k (env : env) (b : base) =
  match b with
  | Ptr p -> env.value p
  | Win w when id w = id k.smem -> window (rank_of env)
  | Win w ->
    (match List.assoc_opt (id w) k.peers with
     | Some (fixed, c) -> window (rank_of env land lnot fixed lor c)
     | None -> env.value w)

let rec value_ref (env : env) (v : value) =
  match v.ref with
  | Some f -> f env
  | None -> Expr.eval v.e (fun n -> match List.assoc_opt n v.vars with Some i -> Prog.var_value env i | None -> failwith ("Mem: unbound " ^ n))

and coord_ref env = function C c -> c | I i -> Prog.index_value env i | V v -> value_ref env v

(* the byte a term's copy starts at *)
let term_ref k env (t : tm) =
  base_ref k env t.base
  + (match t.buf.space with Shared -> t.buf.off | Global -> 0)
  + List.fold_left2 (fun a c (_, stride) -> a + (stride * coord_ref env c)) 0 t.at t.buf.modes

let check what operand imm expect = { what; operand = raw operand; imm; expect }

(* this CTA's coordinate along an axis, a value of its rank *)
let coord_of k a =
  let r = rank k in
  let nx = fst k.cluster in
  let n = Prog.index_name r in
  let e = match a with X -> Expr.modulo (Expr.var n) nx | Y -> Expr.div (Expr.var n) nx in
  let f env = match a with X -> rank_of env mod nx | Y -> rank_of env / nx in
  { e; vars = [ n, Prog.Var r ]; ref = Some f }

(* The window of the cluster's CTA that agrees with this one except where
   [x] or [y] fixes a coordinate, as this CTA addresses it. *)
let peer k ?x ?y () =
  let nx = fst k.cluster in
  let fixed = (match x with Some _ -> bits k X | None -> 0) lor match y with Some _ -> bits k Y | None -> 0 in
  let c = Option.value x ~default:0 + (nx * Option.value y ~default:0) in
  let name = Printf.sprintf "smem_%s%s" (match x with Some v -> "x" ^ string_of_int v | None -> "") (match y with Some v -> "y" ^ string_of_int v | None -> "") in
  let w = ur name in
  if fixed = ctas k - 1 then Prog.derived k.p [ umov w ((c lsl 24) + static_bytes) ]
  else begin
    let t = ur (name ^ "_cta") in
    Prog.derived k.p ([ ulop3_and t k.cta (lnot fixed land 0xff) ] @ (if c = 0 then [] else [ uiadd3 t t c ]) @ [ ulea w t k.k400 0x18 ])
  end;
  k.peers <- (id w, (fixed, c)) :: k.peers;
  (w : shared window)

(* the window of the cluster's CTA of rank [r] *)
let peer_rank k r = let nx = fst k.cluster in peer k ~x:(r mod nx) ~y:(r / nx) ()

(* The CTAs that agree with this one except along [vary], as a mask of
   ranks: the pattern of the axes' ranks, shifted to this CTA's place along
   the others. [shift] moves it where an instruction takes it. *)
let vary_bits k vary = List.fold_left (fun a ax -> a lor bits k ax) 0 vary

let mask k ?(shift = 0) vary =
  let vb = vary_bits k vary in
  let pattern = List.fold_left (fun a r -> if r land lnot vb = 0 then a lor (1 lsl r) else a) 0 (List.init (ctas k) Fun.id) in
  let key = Printf.sprintf "mask %x %d" vb shift in
  if vb = ctas k - 1 then Prog.derive k.p ~name:"mask" ~kind:D_ur ~over:[] ~key (fun d -> [ umov d (pattern lsl shift) ])
  else
    Prog.derive k.p ~name:"mask" ~kind:D_ur ~over:[ Var (rank k) ] ~key (fun d ->
      let t = ur "mask_shift" in
      [ ulop3_and t k.cta (lnot vb land 0xff); umov d (pattern lsl shift); ushf_l_ur d d t ])

(* the reference: every rank that agrees with this CTA's off [vary] *)
let mask_ref k ?(shift = 0) vary env =
  let vb = vary_bits k vary and me = rank_of env in
  List.fold_left (fun a r -> if r land lnot vb = me land lnot vb then a lor (1 lsl r) else a) 0 (List.init (ctas k) Fun.id) lsl shift

let round_up a m = (a + m - 1) / m * m

(* a shared buffer, named for the interpreter's reports *)
let named k (b : buf) =
  let n, stride = match b.modes with (n, s) :: _ -> n, s | [] -> 1, b.copy in
  k.regions <- { rname = b.bname; first = static_bytes + b.off; copies = n; stride } :: k.regions;
  b
let footprint l ~elem = elem + List.fold_left (fun m c -> max m (Layout.offset l c)) 0 (Coord.enumerate (Layout.shape l))

(* ---- buffers ---- *)

(* A ring of [depth] copies of a shared tile of layout [l], each copy aligned
   to the swizzle's period so every copy starts where the pattern starts. *)
let ring k ~name ~depth ~elem (l : (Space.logical, Space.physical) Layout.t) =
  let copy = round_up (footprint l ~elem) Atom.sw128_period in
  let off = round_up (static_bytes + k.dyn_used) Atom.sw128_period in
  k.dyn_used <- off - static_bytes + (depth * copy);
  if k.dyn_used + static_bytes + 0x400 > 232448 then failwith (name ^ ": shared memory");
  (named k { bname = name; space = Shared; modes = [ depth, copy ]; elem = Some l; esize = elem; copy; off; home = Win k.smem } : (shared, tile) buffer)

let strides ~copy modes =
  let rec go = function [] -> [] | n :: rest -> let s = go rest in (n, match s with [] -> copy | (m, st) :: _ -> m * st) :: s in
  go modes

(* a block of shared memory the kernel holds per warp or per slot: [modes]
   outermost first, each copy of layout [l] *)
let shared k ~name ~modes ~elem (l : (Space.logical, Space.physical) Layout.t) =
  let copy = round_up (footprint l ~elem) Atom.sw128_period in
  let off = round_up (static_bytes + k.dyn_used) Atom.sw128_period in
  let ms = strides ~copy modes in
  let total = match ms with [] -> copy | (n, s) :: _ -> n * s in
  k.dyn_used <- off - static_bytes + total;
  if k.dyn_used + static_bytes + 0x400 > 232448 then failwith (name ^ ": shared memory");
  (named k { bname = name; space = Shared; modes = ms; elem = Some l; esize = elem; copy; off; home = Win k.smem } : (shared, tile) buffer)

(* [n] barriers of [count] arrivals, 8 bytes each, in the static window *)
let barriers k ~name ~n ~count =
  let off = round_up k.static_used 8 in
  k.static_used <- off + (8 * n);
  if k.static_used > static_bytes then failwith (name ^ ": the static window");
  let b = named k { bname = name; space = Shared; modes = [ n, 8 ]; elem = None; esize = 8; copy = 8; off; home = Win k.smem } in
  k.barriers <- (b, count) :: k.barriers;
  (b : (shared, barrier) buffer)

(* [n] words of [bytes] each in the static window *)
let words k ~name ~n ~bytes =
  let off = round_up k.static_used (max 4 bytes) in
  k.static_used <- off + (bytes * n);
  if k.static_used > static_bytes then failwith (name ^ ": the static window");
  (named k { bname = name; space = Shared; modes = [ n, bytes ]; elem = None; esize = bytes; copy = bytes; off; home = Win k.smem } : (shared, word) buffer)

(* [n] words, or slabs, of [bytes] each in global memory, at [ptr] *)
let in_global ~name ~ptr ~n ~bytes = { bname = name; space = Global; modes = [ n, bytes ]; elem = None; esize = bytes; copy = bytes; off = 0; home = Ptr ptr }
let global_words ~name ~ptr ~n ~bytes : (global, word) buffer = in_global ~name ~ptr ~n ~bytes
let global_slabs ~name ~ptr ~n ~bytes : (global, slab) buffer = in_global ~name ~ptr ~n ~bytes

let dynsmem k = round_up k.dyn_used Atom.sw128_period
let barrier_count k = List.fold_left (fun a (b, _) -> a + fst (List.hd b.modes)) 0 k.barriers

(* ---- addresses ---- *)

(* the byte offset of a term's index coordinates, as an expression over the
   indices, and the indices it names; an index of range one is the
   coordinate 0 *)
let index_part ?within (t : tm) =
  if List.length t.at <> List.length t.buf.modes then failwith (t.buf.bname ^ ": one coordinate an index mode");
  let parts =
    List.map2
      (fun c (range, stride) ->
        match c with
        | C v -> if v < 0 || v >= range then failwith (t.buf.bname ^ ": a coordinate out of range"); Expr.Const (v * stride), []
        | I i when Prog.index_range i = 1 -> Expr.Const 0, []
        | I i ->
          let n = Prog.index_name i in
          if Prog.index_range i > range then failwith (Printf.sprintf "%s: %s ranges past the mode" t.buf.bname n);
          Expr.scale stride (Expr.var n), [ n, Prog.Var i ]
        | V v -> Expr.scale stride v.e, v.vars)
      t.at t.buf.modes
  in
  let off = match t.buf.space with Shared -> t.buf.off | Global -> 0 in
  let parts = match within with Some v -> (v.e, v.vars) :: parts | None -> parts in
  Expr.add (Expr.Const off) (Expr.sum (List.map fst parts)), List.concat_map snd parts

let win (t : tm) = match t.base with Win w -> w | Ptr _ -> failwith (t.buf.bname ^ ": not in shared memory")
let ptr (t : tm) = match t.base with Ptr p -> p | Win _ -> failwith (t.buf.bname ^ ": not in global memory")

(* the address of a shared term's first element, in a register of [cls]:
   the window plus the strided form over its indices *)
let addr k ~cls ?into ?within (t : tm) =
  let e, vars = index_part ?within t in
  Prog.value k.p ~name:t.buf.bname ~cls ?into ~base:(Some_reg (Uniform, win t)) ~vars e

(* the same with the constant part left for the instruction's immediate (a
   barrier's [UR+imm]) *)
let addr_imm k (t : tm) : ur reg * int =
  let name = if t.buf.elem = None && t.buf.copy = 8 then "bar" else t.buf.bname in
  let base = Prog.Some_reg (Uniform, win t) in
  match index_part t with
  | Expr.Const c, _ -> win t, c
  | Expr.Sum (ts, c), vars -> Prog.value k.p ~name ~cls:Uniform ~base ~vars (Expr.Sum (ts, 0)), c
  | e, vars -> Prog.value k.p ~name ~cls:Uniform ~base ~vars e, 0

let index_list vars = List.sort_uniq (fun a b -> compare (Prog.var_name a) (Prog.var_name b)) (List.map snd vars)

(* the 64-bit address of a global term: its pointer plus the offset, formed
   once where its indices are set *)
let gaddr k ?within (t : tm) : r2 reg =
  let p = ptr t in
  let e, vars = index_part ?within t in
  let off = Prog.value k.p ~name:(t.buf.bname ^ "_off") ~cls:General ~vars e in
  Prog.derive k.p ~name:(t.buf.bname ^ "_addr") ~after:[ raw p ] ~kind:D_r2 ~over:(index_list vars)
    ~key:(Printf.sprintf "gaddr %d %d" (id p) (id off)) (fun d -> [ imad_wide d off 1 p ])

(* a term the shared window names by offset alone, as a request takes it *)
let window_imm (t : tm) =
  match index_part t with
  | Expr.Const c, _ when t.buf.space = Shared -> static_bytes + c
  | _ -> failwith (t.buf.bname ^ ": a window offset is fixed")

(* ---- barriers ---- *)

(* A construct: every barrier the kernel declared, initialised with its
   arrival count. *)
let init_barriers k =
  let word = ur2 "init_word" in
  let last = ref (-1) in
  List.iter
    (fun (b, count) ->
      if count <> !last then Prog.derived k.p (Barrier.init_word word ~count);
      last := count;
      for i = 0 to fst (List.hd b.modes) - 1 do
        let t = at b [ C i ] in
        let base, imm = addr_imm k t in
        let s = syncs_exch (Ubar (base, imm)) ~v:word in
        Prog.check k.p s [ check ("barrier " ^ b.bname) base imm (fun env -> term_ref k env t) ];
        Prog.construct k.p s
      done)
    (List.rev k.barriers)

let bar k t =
  let base, imm = addr_imm k t in
  Ubar (base, imm), [ check ("barrier " ^ t.buf.bname) base imm (fun env -> term_ref k env t) ]

(* the parity a wait on [t] waits for: given, or the ring's whose stage
   names the barrier *)
let parity_of ?parity (t : tm) =
  match parity with
  | Some q -> q
  | None ->
    (match List.find_map (function I i -> Prog.ring_parity (Var i) | _ -> None) t.at with
     | Some q -> q
     | None -> failwith (t.buf.bname ^ ": a barrier no ring names is waited for at a given parity"))

(* until [t] has completed the phase: a spin on one try-wait *)
let wait k ?ns ?parity t =
  let b, checks = bar k t in
  let parity = parity_of ?parity t in
  Prog.spin k.p ?ns ~checks (fun q -> trywait q b ~parity)

(* ask whether [t] has completed the phase, the answer in [into] later: the
   question reads its operands until it is answered *)
let probe k ?parity t ~into =
  let b, checks = bar k t in
  Prog.emit_checked k.p (trywait_ahead into b ~parity:(parity_of ?parity t)) checks

let arrival k t f = let b, checks = bar k t in Prog.emit_checked k.p (f b) checks
let arrive_tx_red k t ~tx = arrival k t (fun b -> arrive_tx_red b ~tx)
let arrive_tx k t ~tx = arrival k t (fun b -> arrive_tx b ~tx)
let arrive_red k t = arrival k t arrive_red

(* the tensor core's commit names its barrier by a register alone, which it
   reads once the MMAs ahead of it have run *)
let commit k t =
  let a = addr k ~cls:Uniform t in
  Prog.emit_checked k.p (utcbar ~bar:a) [ check ("commit " ^ t.buf.bname) a 0 (fun env -> term_ref k env t) ]

(* a mask of ranks, and the axes it varies along *)
type mask = ur reg * axis list

let commit2_mc k ~leader t ~mask:(m, vary) =
  let a = addr k ~cls:Uniform t in
  Prog.emit_checked k.p (utcbar2_mc ~leader ~bar:a ~mask:m)
    [ check ("commit " ^ t.buf.bname) a 0 (fun env -> term_ref k env t); check "commit mask" m 0 (mask_ref k vary) ]

(* a mask and the axes it varies along, as a commit takes it *)
let commit_mask k vary = mask k vary, vary

(* ---- values over indices, for the operands that are coordinates ---- *)

let idx i = let n = Prog.index_name i in { e = Expr.var n; vars = [ n, Prog.Var i ]; ref = Some (fun env -> Prog.index_value env i) }
let const c = { e = Expr.Const c; vars = []; ref = Some (fun _ -> c) }
let scale k v = { e = Expr.scale k v.e; vars = v.vars; ref = Some (fun env -> k * value_ref env v) }
let plus a b = { e = Expr.add a.e b.e; vars = a.vars @ b.vars; ref = Some (fun env -> value_ref env a + value_ref env b) }

(* an expression over indices, its reference the expression's own value *)
let of_expr e vars = { e; vars; ref = None }

(* a map's value at an index: its expression over the variable "c", and the
   map itself for the reference *)
let of_map ~e ~f i = let n = Prog.index_name i in { e = Expr.bind "c" (Expr.var n) e; vars = [ n, Prog.Var i ]; ref = Some (fun env -> f (Prog.index_value env i)) }

let get k ?name ~cls ?into (v : value) = Prog.value k.p ?name ~cls ?into ~vars:v.vars v.e

(* ---- the copy engine ---- *)

(* A 2-D tensor-map box into a shared tile: [dst] the copy it fills (its
   layout the box's image, checked where the map is declared), [bar] where its
   bytes report, [x] and [y] the box's origin in the matrix. Each operand is
   derived straight into its place in the instruction's aligned quad. With
   [share], the box is shared by the cluster's CTAs along that axis: each
   loads its slice of the rows -- at its coordinate along the axis -- and
   multicasts it to them. *)
let tma_load k ?(two = false) ?share ~map ~dst ~bar ~x ~y () =
  let g = ur4 ("load_" ^ dst.buf.bname) in
  let l = match dst.buf.elem with Some l -> l | None -> failwith (dst.buf.bname ^ ": not a tile") in
  let within, y, mask, slice_ref =
    match share with
    | None -> None, y, None, fun _ -> 0
    | Some ax ->
      let _, rows, stride = Atom.load_slice l ~elem:dst.buf.esize ~group:(extent k ax) in
      let c = coord_of k ax in
      (* the multicast form takes its mask in the upper half *)
      Some (scale stride c), plus y (scale rows c), Some (mask k ~shift:16 [ ax ], ax), fun env -> Atom.at2 l (rows * value_ref env c) 0
  in
  ignore (addr k ~cls:Uniform ~into:(upart g 0) ?within dst);
  ignore (addr k ~cls:Uniform ~into:(upart g 1) bar);
  ignore (get k ~cls:Uniform ~into:(upart g 2) x);
  ignore (get k ~cls:Uniform ~into:(upart g 3) y);
  let checks =
    [ check ("load into " ^ dst.buf.bname) (upart g 0) 0 (fun env -> term_ref k env dst + slice_ref env)
    ; check "load's barrier" (upart g 1) 0 (fun env -> term_ref k env bar)
    ; check "load's x" (upart g 2) 0 (fun env -> value_ref env x)
    ; check "load's y" (upart g 3) 0 (fun env -> value_ref env y) ]
    @ match mask with Some (m, ax) -> [ check "load's mask" m 0 (mask_ref k ~shift:16 [ ax ]) ] | None -> []
  in
  Prog.emit_checked k.p (match mask with None -> utmaldg ~two ~g ~map () | Some (m, _) -> utmaldg_mc ~two ~g ~map ~mask:m ()) checks

(* ---- the tensor core ---- *)

(* An operand's descriptor start, read off its tile's layout at its
   coordinates: the window address over 16, which is affine in the indices
   when every stride is a whole number of 16-byte units and the window stays
   below 256 KB -- then a step of an index is a step of the start. *)
let desc_start k (t : tm) =
  (match t.buf.elem with Some _ -> () | None -> failwith (t.buf.bname ^ ": not an operand tile"));
  let e, vars = index_part t in
  let window = Expr.add (Expr.Const 0x400) e in
  let ts, c = match window with Expr.Sum (ts, c) -> ts, c | Expr.Const c -> [], c | _ -> failwith "desc_start" in
  if c mod 16 <> 0 || List.exists (fun (s, _) -> s mod 16 <> 0) ts then failwith (t.buf.bname ^ ": a descriptor moves in 16-byte units");
  let top = c + List.fold_left (fun a (s, v) -> a + (s * ((match v with Expr.Var n -> Prog.var_range (List.assoc n vars) | _ -> 1) - 1))) 0 ts in
  if top >= 1 lsl 18 then failwith (t.buf.bname ^ ": past the descriptor's reach");
  let e16 = Expr.add (Expr.Const ((c / 16) lor Atom.desc_low_fixed)) (Expr.sum (List.map (fun (s, v) -> Expr.scale (s / 16) v) ts)) in
  Prog.value k.p ~name:(t.buf.bname ^ "_desc") ~cls:Uniform ~vars e16

(* The registers the MMAs take: the instruction descriptor, and two pairs of
   each operand's descriptor, which consecutive MMAs alternate between -- the
   tensor core reads an MMA's descriptors when it takes the MMA, so
   rewriting one waits for the MMA two back. *)
type mma_regs =
  { atom : Atom.umma
  ; da : Atom.desc
  ; db : Atom.desc
  ; das : ur2 reg array
  ; dbs : ur2 reg array
  ; ei : ur2 reg
  ; mutable next : int
  }

let mma_regs k ~atom ~la ~lb =
  let mk = Atom.umma_k atom in
  let da = Atom.umma_kmajor la ~elem:(Atom.elt_bytes atom.Atom.ab) ~mma_k:mk and db = Atom.umma_kmajor lb ~elem:(Atom.elt_bytes atom.ab) ~mma_k:mk in
  let m = { atom; da; db; das = [| ur2 "desc_a0"; ur2 "desc_a1" |]; dbs = [| ur2 "desc_b0"; ur2 "desc_b1" |]; ei = ur2 "zero_idesc"; next = 0 } in
  Prog.derived k.p
    ([ umov (upart m.ei 0) 0; umov (upart m.ei 1) (Atom.idesc atom) ]
    @ List.concat_map (fun i -> [ umov (upart m.das.(i) 1) (Atom.desc_high da); umov (upart m.dbs.(i) 1) (Atom.desc_high db) ]) [ 0; 1 ]);
  m

(* where a sub-tile starts in its copy, as a descriptor's start moves: in
   16-byte units, from a whole swizzle row *)
let sub_start (t : tm) =
  let l = match t.buf.elem with Some l -> l | None -> failwith (t.buf.bname ^ ": not an operand tile") in
  let r, c = t.el in
  let off = Atom.at2 l r c - Atom.at2 l 0 0 in
  if r mod 8 <> 0 || off mod 16 <> 0 then failwith (t.buf.bname ^ ": an MMA's operand starts on a swizzle row and a 16-byte chunk");
  off / 16

(* the reference: a sub-tile's descriptor low word, its window address over
   16 in bits 0-13 and the fixed fields above *)
let desc_low_ref k env (t : tm) =
  let l = Option.get t.buf.elem in
  let r, c = t.el in
  let w = term_ref k env t - base_ref k env t.base + static_bytes + (Atom.at2 l r c - Atom.at2 l 0 0) in
  ((w lsr 4) land 0x3fff) lor Atom.desc_low_fixed

(* What an MMA touches, read off the layouts: in each CTA it runs in -- the
   pair, for a two-CTA atom -- the rows of each operand's sub-tile that CTA
   holds, [mma_k] columns of them, as runs of bytes from the copy's start;
   and the accumulator's columns in each lane quarter its image takes. *)
let runs_of offs =
  let sorted = List.sort_uniq compare offs in
  List.rev
    (List.fold_left
       (fun acc (o, n) -> match acc with (s, m) :: rest when s + m = o -> (s, m + n) :: rest | _ -> (o, n) :: acc)
       [] (List.map (fun o -> o, 1) sorted))

let footprints = Hashtbl.create 16

let operand_runs (t : tm) ~rows ~cols =
  let l = match t.buf.elem with Some l -> l | None -> failwith (t.buf.bname ^ ": not an operand tile") in
  let key = t.buf.bname, t.buf.off, t.el, rows, cols in
  match Hashtbl.find_opt footprints key with
  | Some r -> r
  | None ->
    let r0, c0 = t.el in
    let bytes = List.concat (List.init rows (fun r -> List.concat (List.init cols (fun c -> let o = Atom.at2 l (r0 + r) (c0 + c) in List.init t.buf.esize (fun b -> o + b))))) in
    let r = runs_of bytes in
    Hashtbl.replace footprints key r;
    r

let images = Hashtbl.create 4

(* each lane quarter the accumulator's image takes, and its columns *)
let acc_quarters (atom : Atom.umma) =
  match Hashtbl.find_opt images atom with
  | Some q -> q
  | None ->
    let l = Atom.tmem_acc atom in
    let offs = List.map (Layout.offset l) (Coord.enumerate (Layout.shape l)) in
    let q =
      List.filter_map
        (fun q ->
          let cols = List.filter_map (fun o -> if (o lsr 16) / 32 = q then Some (o land 0xffff) else None) offs in
          match cols with
          | [] -> None
          | c :: _ ->
            let lo = List.fold_left min c cols and hi = List.fold_left max c cols in
            if List.length (List.sort_uniq compare cols) <> hi - lo + 1 then failwith "Mem: an accumulator's columns in a lane quarter are not one run";
            Some (q, lo, hi - lo + 1))
        [ 0; 1; 2; 3 ]
    in
    Hashtbl.replace images atom q;
    q

let mma_touches k (m : mma_regs) ~(a : tm) ~(b : tm) ~acc ~col =
  let mk = Atom.umma_k m.atom in
  let rows l = snd (Atom.cta_rows l ~cols:mk ~v:0) in
  let ra = operand_runs a ~rows:(rows (Atom.umma_a m.atom)) ~cols:mk and rb = operand_runs b ~rows:(rows (Atom.umma_b m.atom)) ~cols:mk in
  let quarters = acc_quarters m.atom in
  let ranks env = let r = rank_of env in if m.atom.ctas = 2 then [ r land lnot 1; r lor 1 ] else [ r ] in
  let in_cta r a = (r lsl 24) lor (a land 0xffffff) in
  let smem (t : tm) runs env =
    let start = term_ref k env { t with el = (0, 0) } in
    List.concat_map (fun r -> List.map (fun (o, n) -> in_cta r (start + o), n) runs) (ranks env)
  in
  [ { space = Smem; write = false; runs = smem a ra }
  ; { space = Smem; write = false; runs = smem b rb }
  ; { space = Tmem; write = true
    ; runs = (fun env -> let d = (env.value acc + col) land 0xffff in
               List.concat_map (fun r -> List.map (fun (q, lo, n) -> (r lsl 24) lor ((32 * q) lsl 16) lor (d + lo), n) quarters) (ranks env)) } ]

(* One MMA: [a] and [b] its operands' sub-tiles, [d] the accumulator's
   address and the column its block starts at; [accumulate] the predicate
   that says whether it adds to what is there, or [`Always]. A two-CTA
   atom's MMA is issued by the pair's [leader]. *)
let mma k ?leader (m : mma_regs) ~(a : tm) ~(b : tm) ~d:(acc, col) ~accumulate =
  let a0 = desc_start k { a with el = (0, 0) } and b0 = desc_start k { b with el = (0, 0) } in
  let pa = m.das.(m.next mod 2) and pb = m.dbs.(m.next mod 2) in
  m.next <- m.next + 1;
  Prog.derived k.p [ uiadd3 (upart pb 0) b0 (sub_start b); uiadd3 (upart pa 0) a0 (sub_start a) ];
  let d = if col = 0 then acc else Prog.value k.p ~name:"acc_col" ~cls:Uniform ~base:(Some_reg (Uniform, acc)) ~vars:[] (Expr.Const col) in
  let leader () = match leader with Some l -> l | None -> failwith "Mem.mma: a two-CTA MMA names the pair's leader" in
  let checks =
    [ check ("MMA's A, " ^ a.buf.bname) (upart pa 0) 0 (fun env -> desc_low_ref k env a)
    ; check "MMA's A, the high word" (upart pa 1) 0 (fun _ -> Atom.desc_high m.da)
    ; check ("MMA's B, " ^ b.buf.bname) (upart pb 0) 0 (fun env -> desc_low_ref k env b)
    ; check "MMA's B, the high word" (upart pb 1) 0 (fun _ -> Atom.desc_high m.db)
    ; check "MMA's accumulator" d 0 (fun env -> env.value acc + col)
    ; check "MMA's instruction descriptor" (upart m.ei 1) 0 (fun _ -> Atom.idesc m.atom) ]
  in
  let s =
    match m.atom.ctas, accumulate with
    | 2, `If up -> utchmma2_up ~leader:(leader ()) ~a:pa ~bb:pb ~d ~ei:m.ei ~up
    | 2, `Always -> utchmma2 ~leader:(leader ()) ~a:pa ~bb:pb ~d ~ei:m.ei ~acc:true
    | _, `If up -> utchmma_up ~a:pa ~bb:pb ~d ~ei:m.ei ~up
    | _, `Always -> utchmma ~a:pa ~bb:pb ~d ~ei:m.ei ~acc:true
  in
  Prog.touch k.p s (mma_touches k m ~a ~b ~acc ~col);
  Prog.emit_checked k.p s checks

(* ---- tensor memory ---- *)

(* A construct: one lane asks the allocator for [ncols] columns until it has
   them; the answer, a 32-column chunk, becomes the first column's address
   for the whole warp, as ptxas writes it. [two] allocates for the CTA pair. *)
let tmem_alloc k ?two ~ncols () =
  let addr = ur "tmem" and elected = Core.p "elected" and ok = Core.p "alloc_ok" and got = up "alloc_up" in
  let top = Prog.fresh "ALLOC" and done_ = Prog.fresh "ALLOC_OK" and sync = Prog.fresh "ALLOC_SYNC" in
  let p = k.p in
  Prog.constructs p [ umov addr (ncols / 32); elect elected; bra ~neg:true elected sync ];
  Prog.label p top;
  Prog.constructs p [ umov addr (ncols / 32) ];
  Prog.emit p (utcatomsws_fas ?two ~up:got addr);
  Prog.constructs p [ plop3_up ok got; bra ok done_; nanosleep (); jmp top ];
  Prog.label p done_;
  Prog.derived p [ ushf_l addr addr 5 ];
  Prog.label p sync;
  Prog.construct p (warpsync ());
  addr

(* Free [ncols] columns allocated at [base]: clear their 32-column chunks and
   the allocation's start bit, 16 above its first chunk -- the mask ptxas
   builds (ref/talloc512.ptx). *)
let tmem_free k ~base ~ncols =
  let chunk = ur "free_chunk" and mask = ur "free_mask" and bit = ur "free_bit" in
  Prog.derived k.p
    [ ulop3_and chunk base 0xffff; ushf_r chunk chunk 5; umov mask ((1 lsl (ncols / 32)) - 1); ushf_l_ur mask mask chunk; uiadd3 chunk chunk 16
    ; umov bit 1; ushf_l_ur bit bit chunk; ulop3_or_ur mask mask bit; ulop3_not mask mask ];
  Prog.emit k.p (utcatomsws_and mask)

(* ---- shared memory by a uniform address ---- *)

let word_check k t base imm what = [ check what base imm (fun env -> term_ref k env t) ]
let lds k d (t : tm) = let base, imm = addr_imm k t in Prog.emit_checked k.p (lds d ~base ~imm) (word_check k t base imm ("load of " ^ t.buf.bname))
let sts k (t : tm) ~data = let base, imm = addr_imm k t in Prog.emit_checked k.p (sts_ur ~base ~imm ~data) (word_check k t base imm ("store to " ^ t.buf.bname))

let lds128 k d (t : tm) =
  let a = addr k ~cls:General t in
  Prog.emit_checked k.p (lds128_r d ~addr:a) (word_check k t a 0 ("load of " ^ t.buf.bname))

(* a request to the scheduler: its answer's 16 bytes into [answer], stated
   on [bar] -- window offsets, the instruction issued once for the warp *)
let clc_request k ~answer ~bar =
  let rb = ur2 "request" in
  let win t env = term_ref k env t - base_ref k env t.base + static_bytes in
  Prog.derived k.p [ umov (upart rb 0) (window_imm answer); umov (upart rb 1) (window_imm bar) ];
  Prog.emit k.p (warpsync ());
  Prog.emit_checked k.p (ugetnextworkid ~resp_bar:rb)
    [ check "the request's answer" (upart rb 0) 0 (win answer); check "the request's barrier" (upart rb 1) 0 (win bar) ]

(* a word of global memory past the L1, read and written *)
let ldg_strong k ~desc d t = let a = gaddr k t in Prog.emit_checked k.p (ldg_strong ~desc d ~base:a ~imm:0) [ check ("load of " ^ t.buf.bname) a 0 (fun env -> term_ref k env t) ]
let stg_strong k ~desc t ~data = let a = gaddr k t in Prog.emit_checked k.p (stg_strong ~desc ~base:a ~imm:0 ~data) [ check ("store to " ^ t.buf.bname) a 0 (fun env -> term_ref k env t) ]

(* ---- the accumulator's way out, an instruction a call ----

   A tile's accumulator in tensor memory goes out block by block as a plan
   reads it: the load's fragment composed with the accumulator's layout, and
   with the matrix's or a staging copy's (Store.direct_plan,
   Store.staged_plan). What varies with the warp and the lane is formed once
   a tile; a block and a register add an immediate. *)

(* the accumulator at [acc] (a uniform register) into the matrix at [mat]
   (a pair, general or uniform), the tile's origin at [row], [col] *)
type lane_index = (Prog.fixed, r) Prog.index

type direct =
  { dp : Store.direct
  ; acc : ur reg
  ; mat : [ `R of r2 reg | `UR of ur2 reg ]
  ; desc : ur2 reg
  ; row : value
  ; col : value
  ; warp : lane_index
  ; lane : lane_index
  ; wl : (string * Prog.var) list
  }

let direct (dp : Store.direct) ~acc ~mat ~desc ~row ~col ~warp ~lane =
  { dp; acc; mat; desc; row; col; warp; lane; wl = [ "warpid", Var warp; "laneid", Var lane ] }

let mat_value (env : env) = function `R m -> env.value m | `UR m -> env.value m
let mat_raw = function `R m -> raw m | `UR m -> raw m

let quarter_of env w = Prog.index_value env w mod 4

(* block [(r, j)] of the accumulator into [data]: one tcgen05.ld, from the
   warp's lane quarter *)
let ldtm k (c : direct) (r, j) data =
  let ld = Prog.value k.p ~name:"ld" ~cls:Uniform ~base:(Some_reg (Uniform, c.acc)) ~vars:c.wl (Store.quarter c.dp.ld) in
  let imm = c.dp.imm_ld.(j) + c.dp.rep_ld.(r) in
  Prog.emit_checked k.p (Core.ldtm data ~n:c.dp.n ~addr:ld ~imm)
    [ check "tensor-memory load" ld imm (fun env -> env.value c.acc + c.dp.ref_ld (quarter_of env c.warp) (r, j)) ]

(* the lane's first element of the tile, as a wide address: (row + y) *
   pitch + (col + x + the lane's) * elem *)
let tile_addr k (c : direct) =
  let dp = c.dp and q = Store.quarter in
  let vars = c.wl @ c.row.vars @ c.col.vars in
  let e =
    Expr.add (Expr.scale (dp.cols * dp.elem) (Expr.add c.row.e (q dp.y))) (Expr.add (Expr.scale dp.elem (Expr.add c.col.e (q dp.x))) dp.lane)
  in
  let off = Prog.value k.p ~name:"offset" ~cls:General ~vars e in
  Prog.derive k.p ~name:"tile_addr" ~after:[ mat_raw c.mat ] ~kind:D_r2 ~over:(index_list vars) ~key:(Printf.sprintf "tile_addr %d" (id off))
    (fun d ->
      match c.mat with
      | `R m -> [ imad_wide d off 1 m ]
      | `UR u -> let m = r2 "mat_r" in [ mov_ur (part m 0) (upart u 0); mov_ur (part m 1) (upart u 1); imad_wide d off 1 m ])

(* register [v] of block [(r, j)], [data], to the matrix: one store. The
   block's offset and the register's are its immediate while they fit it,
   and a block's own address otherwise. *)
let stg k (c : direct) (r, j) v data =
  let dp = c.dp in
  let base = tile_addr k c in
  let boff = (((dp.imm_y.(j) + dp.rep_y.(r)) * dp.cols) + dp.imm_x.(j) + dp.rep_x.(r)) * dp.elem in
  let b, bo =
    if Array.for_all (fun o -> abs (boff + o) < 1 lsl 23) dp.reg_off then base, boff
    else
      ( Prog.derive k.p ~name:"block_addr" ~after:[ raw base ] ~kind:D_r2 ~over:(index_list (c.wl @ c.row.vars @ c.col.vars))
          ~key:(Printf.sprintf "block %d %d %d" (id base) r j)
          (fun d -> let one = Core.r "one" in [ mov_imm one 1; imad_wide d one boff base ]),
        0 )
  in
  let imm = bo + dp.reg_off.(v) in
  let expect env =
    mat_value env c.mat + ((((value_ref env c.row * dp.cols) + value_ref env c.col) * dp.elem) + dp.ref_off (quarter_of env c.warp) (r, j) (Prog.index_value env c.lane) v)
  in
  Prog.emit_checked k.p (stg32 ~desc:c.desc ~base:b ~imm ~data) [ check "store to the matrix" b imm expect ]

(* A tile's partial in fragment order at [slot] (a global term): block
   after block, a lane's registers four at a time, consecutive lanes' fours
   side by side (Store.frag_layout), this warp's lane quarter's blocks. *)
let frag_addr k (c : direct) slot =
  let dp = c.dp in
  let nblk = List.length (Store.blocks dp) in
  let per_quarter = Store.frag_at dp ~blk:nblk ~lane:0 ~reg:0 in
  let lane_e = Expr.add (Store.frag_lane dp) (Expr.scale per_quarter (Expr.modulo (Expr.var "warpid") 4)) in
  gaddr k ~within:{ e = lane_e; vars = c.wl; ref = None } slot, per_quarter

let frag_check k (c : direct) ~slot a i v what =
  let _, per_quarter = frag_addr k c slot in
  let imm = Store.frag_at c.dp ~blk:i ~lane:0 ~reg:v in
  check what a imm (fun env ->
    term_ref k env slot + (per_quarter * quarter_of env c.warp) + Store.frag_at c.dp ~blk:i ~lane:(Prog.index_value env c.lane) ~reg:v)

(* registers [v] .. [v] + 3 of block [i] of the partial at [slot]: one load
   past the L1, one store *)
let ldg_frag k (c : direct) ~slot i v data =
  let a, _ = frag_addr k c slot in
  Prog.emit_checked k.p (ldg_strong128 ~desc:c.desc data ~base:a ~imm:(Store.frag_at c.dp ~blk:i ~lane:0 ~reg:v)) [ frag_check k c ~slot a i v "load of the partial" ]

let stg_frag k (c : direct) ~slot i v data =
  let a, _ = frag_addr k c slot in
  Prog.emit_checked k.p (stg128 ~desc:c.desc ~base:a ~imm:(Store.frag_at c.dp ~blk:i ~lane:0 ~reg:v) ~data) [ frag_check k c ~slot a i v "store of the partial" ]

(* the accumulator at [acc] through [staging] -- a row of copies a lane
   quarter, each the copy engine's box -- to the matrix [map] names *)
type staged =
  { sp : Store.staged
  ; staging : buf
  ; sacc : ur reg
  ; map : ur2 reg
  ; srow : value
  ; scol : value
  ; swarp : lane_index
  ; slane : lane_index
  ; swl : (string * Prog.var) list
  ; g : ur4 reg (* the copy engine's operand quad *)
  }

let staged (sp : Store.staged) ~staging ~acc ~map ~row ~col ~warp ~lane =
  { sp; staging; sacc = acc; map; srow = row; scol = col; swarp = warp; slane = lane; swl = [ "warpid", Var warp; "laneid", Var lane ]; g = ur4 "store" }

(* this warp's row of staging copies *)
let copies (s : staged) =
  let quarter = { e = Expr.modulo (Expr.var "warpid") 4; vars = [ "warpid", Var s.swarp ]; ref = Some (fun env -> quarter_of env s.swarp) } in
  at s.staging [ V quarter; C 0 ]

let ldtm_staged k (s : staged) (r, ch) data =
  let ld = Prog.value k.p ~name:"ld" ~cls:Uniform ~base:(Some_reg (Uniform, s.sacc)) ~vars:s.swl (Store.quarter s.sp.s_ld) in
  let imm = s.sp.s_imm_ld.(ch) + s.sp.s_rep_ld.(r) in
  Prog.emit_checked k.p (Core.ldtm data ~n:s.sp.s_n ~addr:ld ~imm)
    [ check "tensor-memory load" ld imm (fun env -> env.value s.sacc + s.sp.s_ref_ld (quarter_of env s.swarp) (r, ch)) ]

(* vector [v] of a block's registers [data] into the warp's copy [slot]:
   one shared-memory store, at the lane's place in the swizzled box *)
let sts_staged k (s : staged) ~slot v data =
  let sp = s.sp in
  let cp = copies s in
  let src_r = addr k ~cls:General cp in
  let e = Restricted.expr (Restricted.restrict ~at:(Parts [ Free; At (sp.vec * v) ]) sp.sts) in
  let a = Prog.value k.p ~name:(Printf.sprintf "swz%d" v) ~cls:General ~base:(Some_reg (General, src_r)) ~vars:s.swl (Expr.bind "c0" (Expr.var "laneid") e) in
  let imm = slot * sp.copy in
  Prog.emit_checked k.p (sts_r ~width:(8 * sp.vec * sp.s_elem) ~base:a ~imm ~data:(slice data (sp.vec * v) sp.vec))
    [ check "store to the staging copy" a imm (fun env -> term_ref k env cp + imm + sp.ref_sts (Prog.index_value env s.slane) (sp.vec * v)) ]

(* the warp's copy [slot] to block [(r, ch)]'s place in the matrix: one
   copy-engine store. A block coordinate the warp does not move is the
   tile's origin plus an immediate. *)
let tma_store ?(add = false) k (s : staged) ~slot (r, ch) =
  let sp = s.sp in
  let cp = copies s in
  let src = addr k ~cls:Uniform cp in
  let coord e (o : value) name =
    match e with
    | Expr.Const c -> Prog.value k.p ~name ~cls:Uniform ~vars:o.vars o.e, c
    | e -> Prog.value k.p ~name ~cls:Uniform ~vars:(s.swl @ o.vars) (Expr.add (Store.quarter e) o.e), 0
  in
  let uy, y0 = coord sp.s_y s.srow "block_y" and ux, x0 = coord sp.s_x s.scol "block_x" in
  Prog.derived k.p
    [ uiadd3 (upart s.g 0) src (slot * sp.copy)
    ; uiadd3 (upart s.g 1) ux (x0 + sp.s_imm_x.(ch) + sp.s_rep_x.(r))
    ; uiadd3 (upart s.g 2) uy (y0 + sp.s_imm_y.(ch) + sp.s_rep_y.(r)) ];
  let origin env = sp.ref_origin (quarter_of env s.swarp) (r, ch) in
  Prog.emit_checked k.p ((if add then utmaredg else utmastg) ~g:s.g ~map:s.map)
    [ check "store's staging copy" (upart s.g 0) 0 (fun env -> term_ref k env cp + (slot * sp.copy))
    ; check "store's x" (upart s.g 1) 0 (fun env -> value_ref env s.scol + snd (origin env))
    ; check "store's y" (upart s.g 2) 0 (fun env -> value_ref env s.srow + fst (origin env)) ]

(* ---- the kernel ---- *)

(* The finished kernel: its launch read off the program -- the threads its
   warps take, the named barriers it meets at, whether it uses the tensor
   core, the shared memory and barriers declared here -- and the
   declarations its body names. *)
let finish k ~name ~params ~grid ~tile ?(together = 1) ?(timing = No_timing) ~notes () : Core.kernel =
  let body = Prog.finish k.p in
  let insns = List.filter_map (function Sass.I i -> Some i | Label _ -> None) body in
  let named = List.fold_left (fun a (i : Sass.insn) -> match i.named_barrier with Some b -> max a (b + 1) | None -> a) 0 insns in
  let regs = regs_of body in
  { kname = name; params; threads = 32 * Prog.warps k.p; grid; cluster = ctas k; together; smem = static_bytes; dynsmem = dynsmem k
  ; mbarriers = barrier_count k; named_barriers = named; tile; tcgen05 = List.exists (fun (i : Sass.insn) -> let o = Sass.opcode i in List.exists (fun p -> String.starts_with ~prefix:p o) [ "UTC"; "LDTM"; "UVIRTCOUNT" ]) insns
  ; body; regs; checks = Prog.checks k.p; touches = Prog.touches k.p; regions = List.rev k.regions; notes; timing }
