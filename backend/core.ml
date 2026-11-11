(* The core language: a program is a list of statements, and a statement is
   one SASS instruction or a label -- nothing else. Its operands are
   registers the program declares by name. The compiler colours them (Color)
   and stops with an error when they do not fit, and writes the header from
   what the program declares. It decides
   nothing else: the loops, the stages, the protocols between warps are the
   program's text. An address is not: a memory operand is a buffer at
   coordinates (Mem), and the instructions that form its address are derived
   from the buffer's layout into the program's holes (Prog). The expansion
   prints one line per instruction.

   The instructions are Sass's constructors, under the same names, taking
   declared registers where Sass takes numbers. *)

type cls =
  | R
  | UR
  | P
  | UP

type raw =
  { id : int (* a placeholder number until coloured; a wide register takes consecutive ones *)
  ; name : string
  ; cls : cls
  ; width : int
  ; align : int
  }

(* A register's class and width are its type: one general register, an
   aligned pair or quad, a vector of general registers (its length a value),
   the same in the uniform file, a predicate, a uniform predicate. *)
type r = [ `R ]
type r2 = [ `R2 ]
type r4 = [ `R4 ]
type rv = [ `Rv ]
type ur = [ `UR ]
type ur2 = [ `UR2 ]
type ur4 = [ `UR4 ]
type p = [ `P ]
type up = [ `UP ]
type 'k reg = raw

let raw x = x
let id (x : raw) = x.id
let name (x : raw) = x.name
let cls_of (x : raw) = x.cls
let width (x : raw) = x.width

(* the file a derived value is computed in *)
type _ vclass =
  | General : r vclass
  | Uniform : ur vclass

let cls_of_vclass : type k. k vclass -> cls = function General -> R | Uniform -> UR

type stmt = Sass.item

(* Declarations are numbered from one counter, so a placeholder names one
   register whatever program it is in; [owner] finds the declaration a
   placeholder unit belongs to while a program is built. A finished kernel
   carries the declarations its body names. *)
let first_placeholder = 1000
let next = ref first_placeholder
let owners : (int, raw) Hashtbl.t = Hashtbl.create 256
let owner u = Hashtbl.find_opt owners u

let declare ?align cls width name =
  let align =
    match align with
    | Some a -> a
    | None ->
      (* a vector operand starts at a multiple of its size, up to 8 *)
      let rec up a = if a >= width || a >= 8 then a else up (2 * a) in
      up 1
  in
  let r = { id = !next; name; cls; width; align } in
  next := !next + width;
  for i = 0 to width - 1 do Hashtbl.replace owners (r.id + i) r done;
  r

(* a register of the file a value is computed in *)
let fresh_in : type k. k vclass -> string -> k reg = fun c name -> match c with General -> declare R 1 name | Uniform -> declare UR 1 name

let r name = declare R 1 name
let r2 name = declare R 2 name
let r4 name = declare R 4 name
let rn ?align n name = declare ?align R n name
let ur name = declare UR 1 name
let ur2 name = declare UR 2 name
let ur4 name = declare UR 4 name
let p name = declare P 1 name
let up name = declare UP 1 name

let cls_name = function R -> "R" | UR -> "UR" | P -> "P" | UP -> "UP"

(* the i-th register of a wide one, as a register of its own for an operand
   that names one part *)
let part (x : raw) i =
  if i < 0 || i >= x.width then failwith (Printf.sprintf "%s: no part %d of %d" x.name i x.width);
  { x with id = x.id + i; name = Printf.sprintf "%s.%d" x.name i; width = 1; align = 1 }

let upart = part


(* [w] consecutive registers of a wide one from its [i]-th, for an operand
   that names a vector inside it *)
let slice (x : raw) i w =
  if i < 0 || w < 1 || i + w > x.width then failwith (Printf.sprintf "%s: no slice %d+%d of %d" x.name i w x.width);
  if i mod w <> 0 then failwith (Printf.sprintf "%s: a slice of %d at %d is misaligned" x.name w i);
  { x with id = x.id + i; name = Printf.sprintf "%s.%d" x.name i; width = w; align = w }

let want cls (x : raw) =
  if x.cls <> cls then failwith (Printf.sprintf "%s is %s, the operand is %s" x.name (cls_name x.cls) (cls_name cls));
  x.id

let want_w cls w (x : raw) =
  if x.width < w then failwith (Printf.sprintf "%s holds %d registers, the operand takes %d" x.name x.width w);
  want cls x

let rr = want R
let uu = want UR
let pp = want P
let upp = want UP

(* the one instruction a Sass constructor pushes *)
let one f : stmt =
  let b = Sass.create () in
  f b;
  match Sass.items b with
  | [ (Sass.I _ as it) ] -> it
  | its -> failwith (Printf.sprintf "Core.one: %d items" (List.length its))

(* a constructor that always guards its instruction, taken without the guard:
   guards are put on by [on] *)
let dummy = 999

let unguarded f : stmt =
  match one f with
  | Sass.I ins -> Sass.I { ins with guard = None; uses = List.filter (fun r -> r <> Sass.P dummy) ins.uses }
  | l -> l

let on ?(neg = false) (g : raw) (s : stmt) : stmt =
  match s with
  | Sass.Label l -> failwith ("on: label " ^ l ^ " takes no guard")
  | Sass.I ins ->
    if ins.guard <> None then failwith ("on: already guarded: " ^ Sass.to_string ins);
    let r = match g.cls with P -> Sass.P g.id | UP -> Sass.UP g.id | _ -> failwith (g.name ^ ": a guard is a predicate") in
    Sass.I { ins with guard = Some (r, neg); uses = r :: ins.uses }

let label_count = ref 0

let fresh_label prefix =
  incr label_count;
  Printf.sprintf "%s_%d" prefix !label_count

let label l : stmt = Sass.Label l

(* ---- the instructions ---- *)

(* the stack pointer, R1, the one register a program does not declare *)
let stack_pointer () = one (fun b -> Sass.ldc b 1 0x37c)

(* where parameter [i] sits in the constant bank *)
let param_off i = 0x380 + (8 * i)

(* the global-memory descriptor's place in the constant bank *)
let gmem_desc_off = 0x358

let ldc64 d off = one (fun b -> Sass.ldc64 b (want_w R 2 d) off)
let ldcu64 d off = one (fun b -> Sass.ldcu64 b (want_w UR 2 d) off)
let s2r_tid d = one (fun b -> Sass.s2r_tid b (rr d))
let s2r_timer d = one (fun b -> Sass.s2r_timer b (rr d))
let cs2r_clock d = one (fun b -> Sass.cs2r_clock b (want_w R 2 d))
let s2r_ctaid d ~axis = one (fun b -> Sass.s2r_ctaid b (rr d) ~axis)
let s2ur_cta d = one (fun b -> Sass.s2ur_cta b (uu d))

let stg32 ~desc ~base ~imm ~data = one (fun b -> Sass.stg32 ~desc:(want_w UR 2 desc) b ~base:(want_w R 2 base) ~imm ~data:(rr data))
let stg128 ~desc ~base ~imm ~data = one (fun b -> Sass.stg128 ~desc:(want_w UR 2 desc) b ~base:(want_w R 2 base) ~imm ~data:(want_w R 4 data))

let imad_wide d a imm c = one (fun b -> Sass.imad_wide b (want_w R 2 d) (rr a) imm (want_w R 2 c))
let imad d a imm c = one (fun b -> Sass.imad b (rr d) (rr a) imm (rr c))
let imad_rz d a imm = one (fun b -> Sass.imad_rz b (rr d) (rr a) imm)
let mov_rz d = one (fun b -> Sass.mov_rz b (rr d))
let lop3_and d a imm = one (fun b -> Sass.lop3_and b (rr d) (rr a) imm)
let shf_r d a sh = one (fun b -> Sass.shf_r b (rr d) (rr a) sh)
let lea_ur d a u sh = one (fun b -> Sass.lea_ur b (rr d) (rr a) (uu u) sh)
let mov_imm d imm = one (fun b -> Sass.mov_imm b (rr d) imm)
let mov_rr d a = one (fun b -> Sass.mov_rr b (rr d) (rr a))
let mov_ur d u = one (fun b -> Sass.mov_ur b (rr d) (uu u))
let lop3_xor_imm d a imm = one (fun b -> Sass.lop3_xor_imm b (rr d) (rr a) imm)
let iadd3 d a c = one (fun b -> Sass.iadd3 b (rr d) (rr a) (rr c))
let iadd3_c d a imm = one (fun b -> Sass.iadd3_c b (rr d) (rr a) imm)
let isetp_lt_u32 p a imm = one (fun b -> Sass.isetp_lt_u32_imm b (pp p) (rr a) imm)
let isetp_ne_u32 p a imm = one (fun b -> Sass.isetp_ne_u32 b (pp p) (rr a) imm)
let isetp_eq_u32 p a imm = one (fun b -> Sass.isetp_eq_u32 b (pp p) (rr a) imm)
let elect p = one (fun b -> Sass.elect b (pp p))
let r2ur u a = one (fun b -> Sass.r2ur b (uu u) (rr a))

let umov u imm = one (fun b -> Sass.umov b (uu u) imm)
let uiadd3 d a imm = one (fun b -> Sass.uiadd3 b (uu d) (uu a) imm)
let uiadd3_uu d a c = one (fun b -> Sass.uiadd3_uu b (uu d) (uu a) (uu c))
let ushf_l d a sh = one (fun b -> Sass.ushf_l b (uu d) (uu a) sh)
let ushf_r d a sh = one (fun b -> Sass.ushf_r b (uu d) (uu a) sh)
let ushf_l_ur d a sh = one (fun b -> Sass.ushf_l_ur b (uu d) (uu a) (uu sh))
let ulop3_and d a imm = one (fun b -> Sass.ulop3_and b (uu d) (uu a) imm)
let ulop3_or_ur d a c = one (fun b -> Sass.ulop3_or_ur b (uu d) (uu a) (uu c))
let ulop3_not d a = one (fun b -> Sass.ulop3_not b (uu d) (uu a))
let ulea d a c sh = one (fun b -> Sass.ulea b (uu d) (uu a) (uu c) sh)
let uimad_imm d a imm c = one (fun b -> Sass.uimad_imm b (uu d) (uu a) imm (uu c))
let uisetp_ne up a = one (fun b -> Sass.uisetp_ne b (upp up) (uu a))
let uisetp_eq0 up a = one (fun b -> Sass.uisetp_eq0 b (upp up) (uu a))
let uisetp_ne_imm up a imm = one (fun b -> Sass.uisetp_ne_imm b (upp up) (uu a) imm)
let plop3_up p up = one (fun b -> Sass.plop3_up b (pp p) ~up:(upp up))

let bra ?(neg = false) p l = one (fun b -> Sass.bra b ~neg (pp p) l)
let jmp l = one (fun b -> Sass.jmp b l)
let exit () = one Sass.exit

(* shared memory *)
let lds d ~base ~imm = one (fun b -> Sass.lds b (rr d) ~ur:(uu base) ~imm)
let lds128_r d ~addr = one (fun b -> Sass.lds128_r b (want_w R 4 d) ~addr:(rr addr))
let sts ~width ~r ~base ~imm ~data = one (fun b -> Sass.sts b ~width ~r:(rr r) ~ur:(uu base) ~imm ~data:(want_w R (width / 32) data))
let sts_ur ~base ~imm ~data = one (fun b -> Sass.sts_ur b ~ur:(uu base) ~imm ~data:(rr data))
let sts_r ~width ~base ~imm ~data = one (fun b -> Sass.sts_r b ~width ~r:(rr base) ~imm ~data:(want_w R (width / 32) data))

let fence_view_async () = one Sass.fence_view_async
let fence_view_async_global () = one Sass.fence_view_async_global
let bar_sync_n ~bar ~count = one (fun b -> Sass.bar_sync_n b ~bar ~count)
let bar_sync_inits () = one Sass.bar_sync_inits
let cluster_arrive_inits () = one Sass.cluster_arrive_inits
let cluster_wait () = one Sass.cluster_wait
let warpsync () = one Sass.warpsync
let preexit () = one Sass.preexit
let acqbulk () = one Sass.acqbulk
let depbar_drain () = one Sass.depbar_drain
let depbar_le ~n = one (fun b -> Sass.depbar_le b ~n)
let nanosleep () = one Sass.nanosleep

(* ---- mbarriers ----

   A barrier operand is its shared::cluster address: a uniform register and
   an immediate, [URn+imm], or a general register, [Rn+URZ]. *)
type bar =
  | Ubar of ur reg * int
  | Rbar of r reg

let syncs_exch bar ~v =
  match bar with
  | Ubar (u, imm) -> one (fun b -> Sass.syncs_exch b ~base:(uu u) ~imm ~v:(want_w UR 2 v))
  | Rbar _ -> failwith "syncs_exch: a barrier is initialised through a uniform register"

let trywait p bar ~parity =
  match bar with
  | Ubar (u, imm) -> one (fun b -> Sass.syncs_trywait b (pp p) ~base:(uu u) ~imm ~parity_reg:(Some (rr parity)))
  | Rbar a -> one (fun b -> Sass.syncs_trywait_r b (pp p) ~addr:(rr a) ~parity:(rr parity))

(* ask now, read the answer later: see Sass.syncs_trywait_ahead *)
let trywait_ahead p bar ~parity =
  match bar with
  | Ubar (u, imm) -> one (fun b -> Sass.syncs_trywait_ahead b (pp p) ~base:(uu u) ~imm ~parity:(rr parity))
  | Rbar _ -> failwith "trywait_ahead: through a uniform register"

let arrive_tx bar ~tx =
  match bar with
  | Ubar (u, imm) -> unguarded (fun b -> Sass.syncs_arrive_tx b ~guard:dummy ~base:(uu u) ~imm ~tx:(rr tx))
  | Rbar a -> unguarded (fun b -> Sass.syncs_arrive_tx_r b ~guard:dummy ~addr:(rr a) ~tx:(rr tx))

(* on another CTA's copy: the address carries that CTA in its top byte *)
let arrive_red bar =
  match bar with
  | Ubar (u, imm) -> unguarded (fun b -> Sass.syncs_arrive_red b ~guard:dummy ~base:(uu u) ~imm)
  | Rbar a -> unguarded (fun b -> Sass.syncs_arrive_red_r b ~guard:dummy ~addr:(rr a))

let arrive_tx_red bar ~tx =
  match bar with
  | Ubar (u, imm) -> unguarded (fun b -> Sass.syncs_arrive_tx_red b ~guard:dummy ~base:(uu u) ~imm ~tx:(rr tx))
  | Rbar a -> unguarded (fun b -> Sass.syncs_arrive_tx_red_r b ~guard:dummy ~addr:(rr a) ~tx:(rr tx))

let sleep_syncs ?ns () = unguarded (fun b -> Sass.nanosleep_syncs ?ns b ~neg:false dummy)

(* ---- tensor memory, the tensor core, the copy engine ---- *)

let utcatomsws_fas ?two ~up u = one (fun b -> Sass.utcatomsws_fas ?two ~up:(upp up) b (uu u))
let utcatomsws_and u = one (fun b -> Sass.utcatomsws_and b (uu u))
let utcatomsws_clear () = one Sass.utcatomsws_clear
let uvirtcount_dealloc () = one Sass.uvirtcount_dealloc

(* The second tensor-memory operand and the instruction descriptor are an
   aligned pair of uniform registers, the descriptor one above, as ptxas
   allocates them -- so they are one declared pair here: [ei] = (e, idesc). *)
let e_of ei = want_w UR 2 ei
let idesc_of ei = want_w UR 2 ei + 1

let utchmma ~a ~bb ~d ~ei ~acc =
  one (fun b -> Sass.utchmma_acc b ~a:(want_w UR 2 a) ~bb:(want_w UR 2 bb) ~d:(uu d) ~e:(e_of ei) ~idesc:(idesc_of ei) ~acc)

let utchmma_up ~a ~bb ~d ~ei ~up =
  one (fun b -> Sass.utchmma_up b ~a:(want_w UR 2 a) ~bb:(want_w UR 2 bb) ~d:(uu d) ~e:(e_of ei) ~idesc:(idesc_of ei) ~up:(upp up))

(* the two-CTA form is issued by the pair's leader, under its predicate *)
let utchmma2 ~leader ~a ~bb ~d ~ei ~acc =
  one (fun b ->
    Sass.utchmma2 b ~guard:(upp leader) ~a:(want_w UR 2 a) ~bb:(want_w UR 2 bb) ~d:(uu d) ~e:(e_of ei) ~idesc:(idesc_of ei) ~acc)

let utchmma2_up ~leader ~a ~bb ~d ~ei ~up =
  one (fun b ->
    Sass.utchmma2_up b ~guard:(upp leader) ~a:(want_w UR 2 a) ~bb:(want_w UR 2 bb) ~d:(uu d) ~e:(e_of ei) ~idesc:(idesc_of ei)
      ~up:(upp up))

let utcbar ~bar = one (fun b -> Sass.utcbar b ~mbar:(uu bar))
let utcbar2_mc ~leader ~bar ~mask = one (fun b -> Sass.utcbar2_mc b ~guard:(upp leader) ~mbar:(uu bar) ~mask:(uu mask))

let ldtm d ~n ~addr ~imm = one (fun b -> Sass.ldtm_off b (want_w R n d) ~n ~addr:(uu addr) ~imm)

let utmaldg ?two ~g ~map () = one (fun b -> Sass.utmaldg ?two b ~g:(want_w UR 4 g) ~map:(want_w UR 2 map))
let utmaldg_mc ?two ~g ~map ~mask () = one (fun b -> Sass.utmaldg_mc ?two b ~g:(want_w UR 4 g) ~map:(want_w UR 2 map) ~mask:(uu mask))
let utmastg ~g ~map = one (fun b -> Sass.utmastg b ~g:(want_w UR 4 g) ~map:(want_w UR 2 map))
let utmaredg ~g ~map = one (fun b -> Sass.utmastg ~add:true b ~g:(want_w UR 4 g) ~map:(want_w UR 2 map))
let utmacmdflush () = one Sass.utmacmdflush
let utmacctl_pf ~map = one (fun b -> Sass.utmacctl_pf b ~map:(want_w UR 2 map))
(* the answer's address and its barrier's are one field too: an aligned pair *)
let ugetnextworkid ~resp_bar = one (fun b -> Sass.ugetnextworkid b ~resp:(want_w UR 2 resp_bar) ~mbar:(want_w UR 2 resp_bar + 1))

(* ---- the kernel ---- *)

type param =
  | Tmap of
      { pname : string
      ; rows : int
      ; cols : int
      ; elem : int
      ; box_rows : int
      ; box_cols : int
      ; swizzle : int
      }
  | Ptr of
      { pname : string
      ; rows : int
      ; cols : int
      ; elem : int
      }

(* What a derived operand must hold where its instruction runs: [operand]
   plus [imm] -- a pair's 64 bits for a wide operand -- equals [expect],
   computed from the layouts the operand was derived from. [env] reads the
   value a declared register holds there, and the launch's facts: "cta",
   the CTA's index; "rank", its rank in the cluster. *)
type env =
  { value : 'k. 'k reg -> int
  ; ctx : string -> int
  }

type check =
  { what : string
  ; operand : raw
  ; imm : int
  ; expect : env -> int
  }

(* What an instruction touches that its registers alone do not say -- an
   MMA's operand tiles and its accumulator -- read off the layouts it came
   from: runs of bytes of shared memory at shared::cluster addresses, or runs
   of tensor-memory columns from a lane quarter's first lane, at CTA << 24 |
   lane << 16 | column, where it runs. *)
type space =
  | Smem
  | Tmem

type touch =
  { space : space
  ; write : bool
  ; runs : env -> (int * int) list
  }

(* a named block of shared memory: its first byte in a CTA's window, how
   many copies, the bytes between them *)
type region =
  { rname : string
  ; first : int
  ; copies : int
  ; stride : int
  }

(* the declarations a body names *)
let regs_of (body : stmt list) =
  let units (i : Sass.insn) = List.map (function Sass.R u | UR u | P u | UP u -> u) (i.defs @ i.uses @ i.late_uses) in
  List.sort_uniq (fun (a : raw) b -> compare a.id b.id)
    (List.filter_map (fun u -> if u >= first_placeholder then owner u else None)
       (List.concat_map (function Sass.I i -> units i | Label _ -> []) body))

(* A kernel: its launch, read off the program -- the threads its warps take,
   the barriers and shared memory it declares, whether it uses the tensor
   core -- its parameters, its body, the declarations the body names, and
   each derived operand's check, keyed by instruction. *)
type kernel =
  { kname : string
  ; params : param list
  ; threads : int
  ; grid : int * int
  ; cluster : int
  ; together : int (* consecutive clusters that wait on one another's global writes *)
  ; smem : int (* the static window past the reserved 0x400 *)
  ; dynsmem : int
  ; mbarriers : int
  ; named_barriers : int
  ; tile : int * int * int (* the output tile and K step, for the host's check *)
  ; tcgen05 : bool
  ; body : stmt list
  ; regs : raw list
  ; checks : (int * check) list
  ; touches : (int * touch) list
  ; regions : region list
  ; notes : string list
  ; timing : timing
  }

(* A timing build: the events the program stamps, or a stamp each time a
   stage lands (Stamp) *)
and timing =
  | No_timing
  | Stamps of (int * string) list
  | Stage_stamps

let isetp_rr ~cmp p a c = one (fun b -> Sass.isetp_rr b ~cmp (pp p) (rr a) (rr c))
let iadd3_sub d a c = one (fun b -> Sass.iadd3_sub b (rr d) (rr a) (rr c))
let fadd d a c = one (fun b -> Sass.fadd b (rr d) (rr a) (rr c))
let ldg_strong ~desc d ~base ~imm = one (fun b -> Sass.ldg_strong ~desc:(want_w UR 2 desc) b (rr d) ~base:(want_w R 2 base) ~imm)
let stg_strong ~desc ~base ~imm ~data = one (fun b -> Sass.stg_strong ~desc:(want_w UR 2 desc) b ~base:(want_w R 2 base) ~imm ~data:(rr data))
let membar_gpu () = one Sass.membar_gpu
let ldg_strong128 ~desc d ~base ~imm = one (fun b -> Sass.ldg_strong128 ~desc:(want_w UR 2 desc) b (want_w R 4 d) ~base:(want_w R 2 base) ~imm)
let uimad_z d a imm = one (fun b -> Sass.uimad_z b (uu d) (uu a) imm)
