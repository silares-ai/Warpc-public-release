(* SASS instructions as data. An instruction is its text as cuobjdump prints
   it -- literal pieces and register operands -- and what the compiler needs
   to know about it: which registers it writes, reads, and reads late (after
   issue), its guard, its effect on the registers for the host-side
   interpreter, and the control-flow it takes. *)

type reg =
  | R of int
  | UR of int
  | P of int
  | UP of int

type branch =
  [ `No
  | `Bra of string
  | `Exit
  ]

type piece =
  | Text of string
  | Reg of reg
  | Lab of string (* a branch's target *)

(* where an instruction came from: the program's own line, an address the
   compiler derived from a layout, or the control a construct of the
   language (a loop, a ring's step, a role, a spin) is made of *)
type origin =
  | Program
  | Derived
  | Construct

(* what the interpreter gives an instruction's effect: register values are
   32-bit unsigned, predicates 0 or 1 *)
type machine =
  { get : reg -> int
  ; set : reg -> int -> unit
  ; special : string -> int (* SR_TID.X, SR_CTAID.X, SR_CgaCtaId, ... *)
  ; const : int -> int (* a word of the constant bank *)
  ; load : int -> int (* a word of shared memory, at a cluster address *)
  ; store : int -> int -> unit
  ; gload : int -> int (* a word of global memory *)
  ; gstore : int -> int -> unit
  ; alloc : int -> int (* tensor memory: chunks of 32 columns asked for -> the slot *)
  ; lane : int
  }

(* What an instruction does outside the registers that orders the warps --
   the barriers' phases, the copy engine, the tensor core, the proxies --
   with its operands' values, for the interpreter's run of a cluster's warps
   together (Interp). Addresses are shared::cluster addresses, the CTA in
   the top byte; a tensor-memory address is lane << 16 | column. *)
type act =
  | Init of { bar : int; count : int } (* SYNCS.EXCH: a barrier, the arrivals its phases take *)
  | Arrive of { bar : int; tx : int } (* a lane's arrival, stating [tx] bytes the phase also waits for *)
  | Wait of { bar : int; parity : int; ahead : bool } (* until the phase of [parity] has completed; [ahead], asked now and read later *)
  | Load of { dst : int; bar : int; map : int; mask : int option; two : bool } (* the copy engine's load of [map]'s box *)
  | Store of { src : int; map : int; x : int; y : int; add : bool }
    (* its store of a box from shared memory, to [map]'s matrix at column [x], row [y]; [add], added to what is there, in the L2 *)
  | Mma of { accumulate : bool } (* the tensor core's MMA, adding to the accumulator or overwriting it; what it reads and writes, the kernel's touches *)
  | Global of { addr : int; bytes : int; write : bool; strong : bool } (* a lane's access to global memory; [strong], at the GPU's coherence point *)
  | Gpu_fence (* this thread's writes so far, visible to the whole GPU before any after *)
  | Commit of { bar : int; mask : int option } (* an arrival once the MMAs before it have run, on [bar] or at its offset in each CTA of [mask] *)
  | Flush (* the copy engine's stores so far, a group *)
  | Drain of int (* until at most this many groups are still reading shared memory *)
  | Proxy_fence (* this thread's writes so far, visible to the copy engine and the tensor core *)
  | Cluster_arrive
  | Cluster_wait
  | Request of { answer : int; bar : int } (* the scheduler's answer, 16 bytes at a window offset in every CTA, reported on [bar] there *)
  | Tmem_load of { addr : int; n : int } (* [n] columns of the warp's lane quarter *)
  | Tmem_free of int (* the 32-column chunks freed, a mask *)

type insn =
  { id : int (* unique: what the checks an operand carries are keyed by *)
  ; asm : piece list
  ; guard : (reg * bool) option (* a predicate, and whether negated *)
  ; defs : reg list
  ; uses : reg list
  ; late_uses : reg list (* read after issue: the operands of a memory or asynchronous instruction *)
  ; branch : branch
  ; named_barrier : int option (* BAR.SYNC's barrier *)
  ; sem : (machine -> unit) option
  ; act : (machine -> act) option
  ; origin : origin
  }

type item =
  | Label of string
  | I of insn

type t = { mutable items : item list }

let create () = { items = [] }
let push b it = b.items <- it :: b.items
let items b = List.rev b.items

(* The text as pieces: every register token -- R, UR, P or UP followed by
   digits, not inside a longer word -- a register operand. *)
let pieces s =
  let n = String.length s in
  let ident c = match c with 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '_' -> true | _ -> false in
  let digit c = c >= '0' && c <= '9' in
  let out = ref [] and lit = Buffer.create 16 in
  let flush () = if Buffer.length lit > 0 then (out := Text (Buffer.contents lit) :: !out; Buffer.clear lit) in
  let rec go i =
    if i < n then begin
      let starts = i = 0 || not (ident s.[i - 1]) in
      let k =
        if not starts then 0
        else if i + 1 < n && s.[i] = 'U' && (s.[i + 1] = 'R' || s.[i + 1] = 'P') then 2
        else if s.[i] = 'R' || s.[i] = 'P' then 1
        else 0
      in
      let j = ref (i + k) in
      if k > 0 then while !j < n && digit s.[!j] do incr j done;
      if k > 0 && !j > i + k && not (!j < n && ident s.[!j]) then begin
        flush ();
        let v = int_of_string (String.sub s (i + k) (!j - i - k)) in
        out := Reg (match String.sub s i k with "R" -> R v | "UR" -> UR v | "P" -> P v | _ -> UP v) :: !out;
        go !j
      end
      else begin
        Buffer.add_char lit s.[i];
        go (i + 1)
      end
    end
  in
  go 0;
  flush ();
  List.rev !out

let reg_text = function R v -> "R" ^ string_of_int v | UR v -> "UR" ^ string_of_int v | P v -> "P" ^ string_of_int v | UP v -> "UP" ^ string_of_int v

(* the instruction as cuobjdump prints it, each register as [name] spells it *)
let to_string ?(name = reg_text) (ins : insn) =
  let g = match ins.guard with None -> "" | Some (r, neg) -> "@" ^ (if neg then "!" else "") ^ name r ^ " " in
  g ^ String.concat "" (List.map (function Text t | Lab t -> t | Reg r -> name r) ins.asm)

(* an instruction's name: its first word, the guard apart *)
let opcode ins =
  match ins.asm with
  | Text t :: _ -> (match String.index_opt t ' ' with Some i -> String.sub t 0 i | None -> t)
  | _ -> ""

(* The operand of [asm] that reads [r], by its place among the register
   operands in the text: the last naming it, or else the last whose pair or
   quad takes it in. *)
let slot ?(regs = []) asm r =
  let named = List.filter_map (function Reg g -> Some g | _ -> None) asm in
  let off g = match g, r with R x, R y | UR x, UR y | P x, P y | UP x, UP y -> Some (y - x) | _ -> None in
  let last p = List.fold_left (fun (i, b) g -> i + 1, match p g with Some d -> i, d | None -> b) (0, (-1, 0)) named |> snd in
  (* a group covers r when every register from its first to r is one the
     instruction touches and no other operand names one of them *)
  let covers g d =
    let shift k = match g with R x -> R (x + k) | UR x -> UR (x + k) | P x -> P (x + k) | UP x -> UP (x + k) in
    List.for_all (fun k -> List.mem (shift k) regs && (k = 0 || not (List.mem (shift k) named))) (List.init (d + 1) Fun.id)
  in
  match last (fun g -> if off g = Some 0 then Some 0 else None) with
  | -1, _ -> last (fun g -> match off g with Some d when d > 0 && d < 32 && covers g d -> Some d | _ -> None)
  | x -> x

let next_id = ref 0

let mk ?guard ?(defs = []) ?(uses = []) ?(late = []) ?(branch = `No) ?bar ?sem ?act text =
  incr next_id;
  { id = !next_id; asm = pieces text; guard; defs; uses; late_uses = late; branch; named_barrier = bar; sem; act; origin = Program }

let regs r n = List.init n (fun i -> R (r + i))
let urs u n = List.init n (fun i -> UR (u + i))
let hex n = if n < 0 then Printf.sprintf "-0x%x" (-n) else Printf.sprintf "0x%x" n
let addr base imm = if imm = 0 then Printf.sprintf "[R%d.64]" base else Printf.sprintf "[R%d.64+%s]" base (hex imm)
let pf = Printf.sprintf

(* ---- the semantics' arithmetic: 32-bit words, 64-bit pairs ---- *)

let m32 x = x land 0xffffffff
let b2i b = if b then 1 else 0
let pair (m : machine) (lo : reg) = match lo with R r -> m.get (R r) lor (m.get (R (r + 1)) lsl 32) | _ -> assert false
let set_pair (m : machine) d v = m.set (R d) (m32 v); m.set (R (d + 1)) (m32 (v lsr 32))
let ur_pair (m : machine) u = m.get (UR u) lor (m.get (UR (u + 1)) lsl 32)

(* constant bank / special registers *)
let ldc b d off = push b (I (mk ~defs:[ R d ] ~sem:(fun m -> m.set (R d) (m.const off)) (pf "LDC R%d, c[0x0][%s]" d (hex off))))
let special b d sr = push b (I (mk ~defs:[ R d ] ~sem:(fun m -> m.set (R d) (m.special sr)) (pf "S2R R%d, %s" d sr)))
let s2r_tid b d = special b d "SR_TID.X"
let s2r_timer b d = special b d "SR_GLOBALTIMERLO"

(* the SM's cycle counter, a pair *)
let cs2r_clock b d = push b (I (mk ~defs:(regs d 2) ~sem:(fun m -> m.set (R d) 0; m.set (R (d + 1)) 0) (pf "CS2R R%d, SR_CLOCKLO" d)))

let ldc64 b d off =
  push b (I (mk ~defs:(regs d 2) ~sem:(fun m -> m.set (R d) (m.const off); m.set (R (d + 1)) (m.const (off + 4)))
               (pf "LDC.64 R%d, c[0x0][%s]" d (hex off))))

let ldcu64 b u off =
  push b (I (mk ~defs:(urs u 2) ~sem:(fun m -> m.set (UR u) (m.const off); m.set (UR (u + 1)) (m.const (off + 4)))
               (pf "LDCU.64 UR%d, c[0x0][%s]" u (hex off))))

(* global memory; the descriptor is a uniform pair, UR4:UR5 unless named *)
(* d:d+1 = a * imm + c:c+1, unsigned *)
let imad_wide b d a imm c =
  push b (I (mk ~defs:(regs d 2) ~uses:(R a :: regs c 2)
               ~sem:(fun m -> set_pair m d ((m.get (R a) * imm) + pair m (R c)))
               (pf "IMAD.WIDE.U32 R%d, R%d, %s, R%d" d a (hex imm) c)))

(* a lane's global access of [bytes] from base + imm, the data's words in
   order *)
let global_store ~base ~imm ~data ~n ~strong =
  let at m = pair m (R base) + imm in
  (fun m -> for i = 0 to n - 1 do m.gstore (at m + (4 * i)) (m.get (R (data + i))) done),
  fun m -> Global { addr = at m; bytes = 4 * n; write = true; strong }

let stg32 ?(desc = 4) b ~base ~imm ~data =
  let sem, act = global_store ~base ~imm ~data ~n:1 ~strong:false in
  push b (I (mk ~uses:(urs desc 2) ~late:(regs base 2 @ [ R data ]) ~sem ~act (pf "STG.E desc[UR%d]%s, R%d" desc (addr base imm) data)))

let alu_op b ~defs ~uses ~sem text = push b (I (mk ~defs ~uses ~sem text))
let ualu_op = alu_op

(* d = a * imm + c *)
let imad b d a imm c = alu_op b ~defs:[ R d ] ~uses:[ R a; R c ] ~sem:(fun m -> m.set (R d) (m32 ((m.get (R a) * imm) + m.get (R c)))) (pf "IMAD R%d, R%d, %s, R%d" d a (hex imm) c)
let imad_rz b d a imm = alu_op b ~defs:[ R d ] ~uses:[ R a ] ~sem:(fun m -> m.set (R d) (m32 (m.get (R a) * imm))) (pf "IMAD R%d, R%d, %s, RZ" d a (hex imm))
let mov_rz b d = alu_op b ~defs:[ R d ] ~uses:[] ~sem:(fun m -> m.set (R d) 0) (pf "MOV R%d, RZ" d)
let lop3_and b d a imm = alu_op b ~defs:[ R d ] ~uses:[ R a ] ~sem:(fun m -> m.set (R d) (m.get (R a) land m32 imm)) (pf "LOP3.LUT R%d, R%d, %s, RZ, 0xc0, !PT" d a (hex imm))
let shf_r b d a sh = alu_op b ~defs:[ R d ] ~uses:[ R a ] ~sem:(fun m -> m.set (R d) (m.get (R a) lsr sh)) (pf "SHF.R.U32.HI R%d, RZ, %s, R%d" d (hex sh) a)

let branch_to ins l = { ins with asm = [ Text "BRA "; Lab l ] }
let bra b ?(neg = false) p l = push b (I (branch_to (mk ~guard:(P p, neg) ~uses:[ P p ] ~branch:(`Bra l) "BRA") l))
let exit b = push b (I (mk ~branch:`Exit "EXIT"))

(* ---- the tcgen05 path: uniform datapath, shared memory, mbarriers, TMEM ---- *)

let s2ur_cta b u = push b (I (mk ~defs:[ UR u ] ~sem:(fun m -> m.set (UR u) (m.special "SR_CgaCtaId")) (pf "S2UR UR%d, SR_CgaCtaId" u)))
let umov b u imm = push b (I (mk ~defs:[ UR u ] ~sem:(fun m -> m.set (UR u) (m32 imm)) (pf "UMOV UR%d, %s" u (hex imm))))
let uiadd3 b d a imm = ualu_op b ~defs:[ UR d ] ~uses:[ UR a ] ~sem:(fun m -> m.set (UR d) (m32 (m.get (UR a) + imm))) (pf "UIADD3 UR%d, UPT, UPT, UR%d, %s, URZ" d a (hex imm))
let uiadd3_uu b d a c = ualu_op b ~defs:[ UR d ] ~uses:[ UR a; UR c ] ~sem:(fun m -> m.set (UR d) (m32 (m.get (UR a) + m.get (UR c)))) (pf "UIADD3 UR%d, UPT, UPT, UR%d, UR%d, URZ" d a c)
let ushf_l b d a sh = ualu_op b ~defs:[ UR d ] ~uses:[ UR a ] ~sem:(fun m -> m.set (UR d) (m32 (m.get (UR a) lsl sh))) (pf "USHF.L.U32 UR%d, UR%d, %s, URZ" d a (hex sh))
let ushf_r b d a sh = ualu_op b ~defs:[ UR d ] ~uses:[ UR a ] ~sem:(fun m -> m.set (UR d) (m.get (UR a) lsr sh)) (pf "USHF.R.U32.HI UR%d, URZ, %s, UR%d" d (hex sh) a)
let ulop3_and b d a imm = ualu_op b ~defs:[ UR d ] ~uses:[ UR a ] ~sem:(fun m -> m.set (UR d) (m.get (UR a) land m32 imm)) (pf "ULOP3.LUT UR%d, UR%d, %s, URZ, 0xc0, !UPT" d a (hex imm))
let ulop3_not b d a = ualu_op b ~defs:[ UR d ] ~uses:[ UR a ] ~sem:(fun m -> m.set (UR d) (m32 (lnot (m.get (UR a))))) (pf "ULOP3.LUT UR%d, URZ, UR%d, URZ, 0x33, !UPT" d a)

(* d = (a << sh) + c *)
let ulea b d a c sh = ualu_op b ~defs:[ UR d ] ~uses:[ UR a; UR c ] ~sem:(fun m -> m.set (UR d) (m32 ((m.get (UR a) lsl sh) + m.get (UR c)))) (pf "ULEA UR%d, UR%d, UR%d, %s" d a c (hex sh))

(* d = a * imm + c, as ptxas spells it (UIMAD UR5, UR5, -0xa0, URZ) *)
let uimad_imm b d a imm c =
  ualu_op b ~defs:[ UR d ] ~uses:[ UR a; UR c ] ~sem:(fun m -> m.set (UR d) (m32 ((m.get (UR a) * imm) + m.get (UR c)))) (pf "UIMAD UR%d, UR%d, %s, UR%d" d a (hex imm) c)

(* d = a * imm, uniform: UIMAD UR, UR, imm, URZ *)
let uimad_z b d a imm = ualu_op b ~defs:[ UR d ] ~uses:[ UR a ] ~sem:(fun m -> m.set (UR d) (m32 (m.get (UR a) * imm))) (pf "UIMAD UR%d, UR%d, %s, URZ" d a (hex imm))

let r2ur b u r = push b (I (mk ~defs:[ UR u ] ~uses:[ R r ] ~sem:(fun m -> m.set (UR u) (m.get (R r))) (pf "R2UR UR%d, R%d" u r)))
let mov_ur b d u = alu_op b ~defs:[ R d ] ~uses:[ UR u ] ~sem:(fun m -> m.set (R d) (m.get (UR u))) (pf "IMAD.U32 R%d, RZ, RZ, UR%d" d u)

(* d = (a << sh) + c *)
let lea_ur b d a u sh = alu_op b ~defs:[ R d ] ~uses:[ R a; UR u ] ~sem:(fun m -> m.set (R d) (m32 ((m.get (R a) lsl sh) + m.get (UR u)))) (pf "LEA R%d, R%d, UR%d, %s" d a u (hex sh))
let mov_imm b d imm = alu_op b ~defs:[ R d ] ~uses:[] ~sem:(fun m -> m.set (R d) (m32 imm)) (pf "MOV R%d, %s" d (hex imm))
let lop3_xor b d a c = alu_op b ~defs:[ R d ] ~uses:[ R a; R c ] ~sem:(fun m -> m.set (R d) (m.get (R a) lxor m.get (R c))) (pf "LOP3.LUT R%d, R%d, R%d, RZ, 0x3c, !PT" d a c)
let iadd3 b d a c = alu_op b ~defs:[ R d ] ~uses:[ R a; R c ] ~sem:(fun m -> m.set (R d) (m32 (m.get (R a) + m.get (R c)))) (pf "IADD3 R%d, PT, PT, R%d, R%d, RZ" d a c)
let iadd3_c b d a imm = alu_op b ~defs:[ R d ] ~uses:[ R a ] ~sem:(fun m -> m.set (R d) (m32 (m.get (R a) + imm))) (pf "IADD3 R%d, PT, PT, R%d, %s, RZ" d a (hex imm))

let setp b ~p ~a ~sem text = push b (I (mk ~defs:[ P p ] ~uses:[ R a ] ~sem text))

let isetp_ne_u32 b p a imm =
  let rhs = if imm = 0 then "RZ" else hex imm in
  setp b ~p ~a ~sem:(fun m -> m.set (P p) (b2i (m.get (R a) <> m32 imm))) (pf "ISETP.NE.U32.AND P%d, PT, R%d, %s, PT" p a rhs)

let jmp b l = push b (I (branch_to (mk ~branch:(`Bra l) "BRA") l))

(* shared memory; addresses are the cluster form (cta << 24 | offset) held in a UR *)
let lds b d ~ur ~imm =
  push b (I (mk ~defs:[ R d ] ~late:[ UR ur ] ~sem:(fun m -> m.set (R d) (m.load (m32 (m.get (UR ur) + imm)))) (pf "LDS R%d, [UR%d+%s]" d ur (hex imm))))

let sts b ~width ~r ~ur ~imm ~data =
  let n = width / 32 in
  let suffix = if width = 32 then "" else pf ".%d" width in
  push b (I (mk ~uses:[ UR ur ] ~late:(R r :: regs data n)
               ~sem:(fun m -> for i = 0 to n - 1 do m.store (m32 (m.get (R r) + m.get (UR ur) + imm + (4 * i))) (m.get (R (data + i))) done)
               (pf "STS%s [R%d+UR%d+%s], R%d" suffix r ur (hex imm) data)))

let sts_ur b ~ur ~imm ~data =
  push b (I (mk ~uses:[ UR ur ] ~late:[ R data ] ~sem:(fun m -> m.store (m32 (m.get (UR ur) + imm)) (m.get (R data)))
               (pf "STS [UR%d+%s], R%d" ur (hex imm) data)))

let stg128 ?(desc = 4) b ~base ~imm ~data =
  let sem, act = global_store ~base ~imm ~data ~n:4 ~strong:false in
  push b (I (mk ~uses:(urs desc 2) ~late:(regs base 2 @ regs data 4) ~sem ~act (pf "STG.E.128 desc[UR%d]%s, R%d" desc (addr base imm) data)))

let fence_view_async b = push b (I (mk ~act:(fun _ -> Proxy_fence) "FENCE.VIEW.ASYNC.S"))

(* fence.proxy.async.global: global memory the generic proxy wrote, visible
   to the copy engine's accesses after it, and the other way round *)
let fence_view_async_global b = push b (I (mk "FENCE.VIEW.ASYNC.G"))

(* a named barrier over a subset of the block: the warps that store share one,
   and the warps that feed the tensor core never arrive at it *)
let bar_sync_n b ~bar ~count =
  push b (I (mk ~bar
               (pf "BAR.SYNC.DEFER_BLOCKING %s, %s" (hex bar) (hex count))))

let bar_addr base imm = if imm = 0 then pf "[UR%d]" base else pf "[UR%d+%s]" base (hex imm)

(* A barrier's initialisation; what publishes the initialisations to the
   other warps -- the cluster's arrival, or the CTA's barrier -- comes after
   every one. *)
let syncs_exch b ~base ~imm ~v =
  (* the word's low half holds 2^20 - count at bit 1 (Barrier.init_word) *)
  let act m = Init { bar = m32 (m.get (UR base) + imm); count = 0x100000 - ((m.get (UR v) lsr 1) land 0xfffff) } in
  push b (I (mk ~late:[ UR base; UR v; UR (v + 1) ] ~act (pf "SYNCS.EXCH.64 URZ, %s, UR%d" (bar_addr base imm) v)))

(* a test of a barrier's phase, the answer in a predicate *)
let syncs_trywait b p ~base ~imm ~parity_reg =
  let rb = match parity_reg with None -> "RZ" | Some r -> pf "R%d" r in
  let uses = match parity_reg with None -> [ UR base ] | Some r -> [ UR base; R r ] in
  let act m = Wait { bar = m32 (m.get (UR base) + imm); parity = (match parity_reg with Some r -> m.get (R r) | None -> 0); ahead = false } in
  push b (I (mk ~defs:[ P p ] ~uses ~sem:(fun m -> m.set (P p) 1) ~act (pf "SYNCS.PHASECHK.TRANS64.TRYWAIT P%d, %s, %s" p (bar_addr base imm) rb)))

(* A try-wait whose answer is read later, a stage ahead: the barrier unit
   reads its operands until it answers, so they are read late. *)
let syncs_trywait_ahead b p ~base ~imm ~parity =
  push b (I (mk ~defs:[ P p ] ~late:[ UR base; R parity ] ~sem:(fun m -> m.set (P p) 1)
               ~act:(fun m -> Wait { bar = m32 (m.get (UR base) + imm); parity = m.get (R parity); ahead = true })
               (pf "SYNCS.PHASECHK.TRANS64.TRYWAIT P%d, %s, R%d" p (bar_addr base imm) parity)))

(* [two] asks the allocator for the CTA PAIR, which is what makes a two-CTA
   MMA work: both CTAs get the same tensor-memory columns, and the accumulator
   the instruction writes half into each lands at one address. *)
let utcatomsws_fas ?(two = false) ?(up = 0) b u =
  push b (I (mk ~defs:[ UP up; UR u ] ~late:[ UR u ]
               ~sem:(fun m -> m.set (UR u) (m.alloc (m.get (UR u))); m.set (UP up) 1)
               (pf "UTCATOMSWS%s.FIND_AND_SET.ALIGN UP%d, UR%d, UR%d" (if two then ".2CTA" else "") up u u)))

let plop3_up b p ~up =
  push b (I (mk ~defs:[ P p ] ~uses:[ UP up ] ~sem:(fun m -> m.set (P p) (m.get (UP up))) (pf "PLOP3.LUT P%d, PT, PT, PT, UP%d, 0x80, 0x8" p up)))

let nanosleep b = push b (I (mk "NANOSLEEP 0x64"))

(* a failed barrier test sleeps until the barrier unit has news,
   instead of asking again at once *)
let nanosleep_syncs ?(ns = 0xc350) b ~neg p =
  push b (I (mk ~guard:(P p, neg) ~uses:[ P p ] (pf "NANOSLEEP.SYNCS %s" (hex ns))))

let uvirtcount_dealloc b = push b (I (mk "UVIRTCOUNT.DEALLOC.SMPOOL 0x80"))
(* the allocation's chunks are the low 16 bits; the mask keeps what it does not free *)
let utcatomsws_and b u = push b (I (mk ~late:[ UR u ] ~act:(fun m -> Tmem_free (lnot (m.get (UR u)) land 0xffff)) (pf "UTCATOMSWS.AND URZ, UR%d" u)))

(* clear the whole allocation bitmap: all 512 columns freed at once *)
let utcatomsws_clear b = push b (I (mk ~act:(fun _ -> Tmem_free 0xffff) "UTCATOMSWS.AND URZ, URZ"))

let utcbar b ~mbar =
  push b (I (mk ~late:[ UR mbar ] ~act:(fun m -> Commit { bar = m.get (UR mbar); mask = None })
               (pf "UTCBAR [UR%d], URZ" mbar)))

let ushf_l_ur b d a sh =
  (* a shift of 32 or more leaves nothing *)
  ualu_op b ~defs:[ UR d ] ~uses:[ UR a; UR sh ] ~sem:(fun m -> let n = m.get (UR sh) in m.set (UR d) (if n >= 32 then 0 else m32 (m.get (UR a) lsl n))) (pf "USHF.L.U32 UR%d, UR%d, UR%d, URZ" d a sh)
let ulop3_or_ur b d a c = ualu_op b ~defs:[ UR d ] ~uses:[ UR a; UR c ] ~sem:(fun m -> m.set (UR d) (m.get (UR a) lor m.get (UR c))) (pf "ULOP3.LUT UR%d, UR%d, UR%d, URZ, 0xfc, !UPT" d a c)

(* ---- the pipelined path: TMA, per-stage mbarriers, uniform predicates ---- *)

let s2r_ctaid b d ~axis = special b d ("SR_CTAID." ^ axis)

let isetp_eq_u32 b p a imm =
  let rhs = if imm = 0 then "RZ" else hex imm in
  setp b ~p ~a ~sem:(fun m -> m.set (P p) (b2i (m.get (R a) = m32 imm))) (pf "ISETP.EQ.U32.AND P%d, PT, R%d, %s, PT" p a rhs)

let isetp_lt_u32_imm b p a imm =
  setp b ~p ~a ~sem:(fun m -> m.set (P p) (b2i (m.get (R a) < m32 imm))) (pf "ISETP.LT.U32.AND P%d, PT, R%d, %s, PT" p a (hex imm))

let lop3_xor_imm b d a imm = alu_op b ~defs:[ R d ] ~uses:[ R a ] ~sem:(fun m -> m.set (R d) (m.get (R a) lxor m32 imm)) (pf "LOP3.LUT R%d, R%d, %s, RZ, 0x3c, !PT" d a (hex imm))

(* UTMALDG.2D [URg], [URmap]: URg = smem destination, URg+1 = mbarrier, URg+2 = x, URg+3 = y.
   With .MULTICAST the load also names a CTA mask: the slice it fetches lands in
   the shared memory of every CTA of the cluster the mask selects. *)
(* [two] marks a load that feeds a two-CTA MMA: its bytes are reported to the
   pair, so neither CTA can start the MMA until both halves have landed *)
let load_act ~g ~map ~mask ~two m = Load { dst = m.get (UR g); bar = m.get (UR (g + 1)); map = ur_pair m map; mask = Option.map (fun u -> m.get (UR u)) mask; two }

let utmaldg ?(two = false) b ~g ~map =
  push b (I (mk ~late:(urs g 4 @ urs map 2) ~act:(load_act ~g ~map ~mask:None ~two)
               (pf "UTMALDG.2D%s [UR%d], [UR%d]" (if two then ".2CTA" else "") g map)))

let utmaldg_mc ?guard ?(two = false) b ~g ~map ~mask =
  let gd, gu = match guard with None -> (None, []) | Some u -> (Some (UP u, false), [ UP u ]) in
  push b (I (mk ?guard:gd ~uses:gu ~late:(urs g 4 @ urs map 2 @ [ UR mask ]) ~act:(load_act ~g ~map ~mask:(Some mask) ~two)
               (pf "UTMALDG.2D.MULTICAST%s [UR%d], [UR%d], UR%d" (if two then ".2CTA" else "") g map mask)))

(* the arrivals: one lane's, with an expected transaction count or plainly,
   on this CTA's barrier or -- a reduction, the address carrying the CTA in
   its top byte -- on another CTA's copy, which is the form ptxas emits for a
   remote arrive (a remote WAIT is illegal) *)
let arrival b ~guard ~late ~bar ~tx text =
  push b (I (mk ~guard:(P guard, false) ~uses:[ P guard ] ~late ~act:(fun m -> Arrive { bar = m32 (bar m); tx = tx m }) text))

let at_ur base imm m = m.get (UR base) + imm
let in_r r m = m.get (R r)
let none _ = 0
let syncs_arrive_tx_red b ~guard ~base ~imm ~tx = arrival b ~guard ~late:[ UR base; R tx ] ~bar:(at_ur base imm) ~tx:(in_r tx) (pf "SYNCS.ARRIVE.TRANS64.RED RZ, %s, R%d" (bar_addr base imm) tx)
let syncs_arrive_tx b ~guard ~base ~imm ~tx = arrival b ~guard ~late:[ UR base; R tx ] ~bar:(at_ur base imm) ~tx:(in_r tx) (pf "SYNCS.ARRIVE.TRANS64 RZ, %s, R%d" (bar_addr base imm) tx)
let syncs_arrive_red b ~guard ~base ~imm = arrival b ~guard ~late:[ UR base ] ~bar:(at_ur base imm) ~tx:none (pf "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, %s, RZ" (bar_addr base imm))
let syncs_arrive b ~guard ~base ~imm = arrival b ~guard ~late:[ UR base ] ~bar:(at_ur base imm) ~tx:none (pf "SYNCS.ARRIVE.TRANS64.A1T0 RZ, %s, RZ" (bar_addr base imm))

(* the same with the address in a register, [Rn+URZ], as ptxas addresses a
   barrier it indexes at run time (ref/clcloop.ptx) *)
let syncs_arrive_r b ~guard ~addr = arrival b ~guard ~late:[ R addr ] ~bar:(in_r addr) ~tx:none (pf "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R%d+URZ], RZ" addr)
let syncs_arrive_tx_r b ~guard ~addr ~tx = arrival b ~guard ~late:[ R addr; R tx ] ~bar:(in_r addr) ~tx:(in_r tx) (pf "SYNCS.ARRIVE.TRANS64 RZ, [R%d+URZ], R%d" addr tx)
let syncs_arrive_red_r b ~guard ~addr = arrival b ~guard ~late:[ R addr ] ~bar:(in_r addr) ~tx:none (pf "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R%d+URZ], RZ" addr)
let syncs_arrive_tx_red_r b ~guard ~addr ~tx = arrival b ~guard ~late:[ R addr; R tx ] ~bar:(in_r addr) ~tx:(in_r tx) (pf "SYNCS.ARRIVE.TRANS64.RED RZ, [R%d+URZ], R%d" addr tx)

let syncs_trywait_r b p ~addr ~parity =
  push b (I (mk ~defs:[ P p ] ~uses:[ R addr; R parity ] ~sem:(fun m -> m.set (P p) 1)
               ~act:(fun m -> Wait { bar = m.get (R addr); parity = m.get (R parity); ahead = false })
               (pf "SYNCS.PHASECHK.TRANS64.TRYWAIT P%d, [R%d+URZ], R%d" p addr parity)))

(* A CTA of a cluster may not leave while a peer can still write its shared
   memory, so the cluster meets once before the exits.  This is what
   barrier.cluster.arrive / barrier.cluster.wait lower to (ref/remarrive.ptx).
   The meeting split in two: the arrival publishes what came before it to
   the cluster -- the barriers' initialisations, and the tensor-memory
   allocations, which the pair's leader's MMAs write into -- and the wait
   comes before the first use of either. *)
let cluster_arrive_inits b = push b (I (mk ~act:(fun _ -> Cluster_arrive) "UCGABAR_ARV"))
let cluster_wait b = push b (I (mk ~act:(fun _ -> Cluster_wait) "UCGABAR_WAIT"))

(* without a cluster, the CTA's barrier publishes the initialisations *)
let bar_sync_inits b =
  push b (I (mk ~bar:0 "BAR.SYNC.DEFER_BLOCKING 0x0"))

(* cluster launch control: cancel a cluster not yet launched, its first
   CTA's id written to [resp] in every CTA of this cluster, each completing 16
   bytes on its own copy of [mbar] -- clusterlaunchcontrol.try_cancel ...
   multicast::cluster::all, as ptxas emits it *)
let ugetnextworkid b ~resp ~mbar =
  push b (I (mk ~uses:[ UR resp; UR mbar ] ~act:(fun m -> Request { answer = m.get (UR resp); bar = m.get (UR mbar) })
               (pf "UGETNEXTWORKID.BROADCAST [UR%d], [UR%d]" resp mbar)))

let lds128_r b d ~addr =
  push b (I (mk ~defs:(regs d 4) ~late:[ R addr ]
               ~sem:(fun m -> for i = 0 to 3 do m.set (R (d + i)) (m.load (m32 (m.get (R addr) + (4 * i)))) done)
               (pf "LDS.128 R%d, [R%d]" d addr)))

(* a uniform predicate takes as long to become readable as a regular one *)
let usetp b ~up ~a ~sem text = push b (I (mk ~defs:[ UP up ] ~uses:[ UR a ] ~sem text))
let uisetp_ne b up a = usetp b ~up ~a ~sem:(fun m -> m.set (UP up) (b2i (m.get (UR a) <> 0))) (pf "UISETP.NE.U32.AND UP%d, UPT, UR%d, URZ, UPT" up a)
let uisetp_ne_imm b up a imm = usetp b ~up ~a ~sem:(fun m -> m.set (UP up) (b2i (m.get (UR a) <> m32 imm))) (pf "UISETP.NE.U32.AND UP%d, UPT, UR%d, %s, UPT" up a (hex imm))
let uisetp_eq0 b up a = usetp b ~up ~a ~sem:(fun m -> m.set (UP up) (b2i (m.get (UR a) = 0))) (pf "UISETP.EQ.U32.AND UP%d, UPT, UR%d, URZ, UPT" up a)

(* the MMAs: the one-CTA form, its accumulate flag a constant or a uniform
   predicate; the two-CTA form, issued by the pair's leader alone and so
   guarded on the leader predicate *)
let mma_operands a bb d e idesc = [ UR a; UR (a + 1); UR bb; UR (bb + 1); UR d; UR e; UR idesc ]

let utchmma_up b ~a ~bb ~d ~e ~idesc ~up =
  push b (I (mk ~uses:[ UP up ] ~late:(mma_operands a bb d e idesc)
               ~act:(fun m -> Mma { accumulate = m.get (UP up) <> 0 })
               (pf "UTCHMMA gdesc[UR%d], gdesc[UR%d], tmem[UR%d], tmem[UR%d], idesc[UR%d], UP%d" a bb d e idesc up)))

let utchmma_acc b ~a ~bb ~d ~e ~idesc ~acc =
  push b (I (mk ~late:(mma_operands a bb d e idesc) ~act:(fun _ -> Mma { accumulate = acc })
               (pf "UTCHMMA gdesc[UR%d], gdesc[UR%d], tmem[UR%d], tmem[UR%d], idesc[UR%d], %s" a bb d e idesc (if acc then "UPT" else "!UPT"))))

let utchmma2 b ~guard ~a ~bb ~d ~e ~idesc ~acc =
  push b (I (mk ~guard:(UP guard, false) ~uses:[ UP guard ] ~late:(mma_operands a bb d e idesc) ~act:(fun _ -> Mma { accumulate = acc })
               (pf "UTCHMMA.2CTA gdesc[UR%d], gdesc[UR%d], tmem[UR%d], tmem[UR%d], idesc[UR%d], %s" a bb d e idesc (if acc then "UPT" else "!UPT"))))

let utchmma2_up b ~guard ~a ~bb ~d ~e ~idesc ~up =
  push b (I (mk ~guard:(UP guard, false) ~uses:[ UP guard; UP up ] ~late:(mma_operands a bb d e idesc)
               ~act:(fun m -> Mma { accumulate = m.get (UP up) <> 0 }) (pf "UTCHMMA.2CTA gdesc[UR%d], gdesc[UR%d], tmem[UR%d], tmem[UR%d], idesc[UR%d], UP%d" a bb d e idesc up)))

let utcbar2_mc b ~guard ~mbar ~mask =
  push b (I (mk ~guard:(UP guard, false) ~uses:[ UP guard ] ~late:[ UR mbar; UR mask ]
               ~act:(fun m -> Commit { bar = m.get (UR mbar); mask = Some (m.get (UR mask)) })
               (pf "UTCBAR.2CTA.MULTICAST [UR%d], URZ, UR%d" mbar mask)))

let ldtm_off b d ~n ~addr ~imm =
  let a = if imm = 0 then pf "tmem[UR%d]" addr else pf "tmem[UR%d+%s]" addr (hex imm) in
  push b (I (mk ~defs:(regs d n) ~uses:[ UR addr ] ~sem:(fun m -> for i = 0 to n - 1 do m.set (R (d + i)) 0 done)
               ~act:(fun m -> Tmem_load { addr = m32 (m.get (UR addr) + imm); n })
               (pf "LDTM.x%d R%d, %s" n d a)))

let utmacctl_pf b ~map = push b (I (mk ~late:(urs map 2) (pf "UTMACCTL.PF [UR%d]" map)))

(* the lowest active lane, for the code that must run once per warp *)
let elect b p = push b (I (mk ~defs:[ P p ] ~sem:(fun m -> m.set (P p) (b2i (m.lane = 0))) (pf "ELECT P%d, URZ, PT" p)))

(* shared-memory staging for the epilogue: a warp writes its rows, then reads
   them back so that consecutive lanes hold consecutive addresses *)
let sts_r b ~width ~r ~imm ~data =
  let n = width / 32 in
  let suffix = if width = 32 then "" else pf ".%d" width in
  push b (I (mk ~late:(R r :: regs data n)
               ~sem:(fun m -> for i = 0 to n - 1 do m.store (m32 (m.get (R r) + imm + (4 * i))) (m.get (R (data + i))) done)
               (pf "STS%s [R%d+%s], R%d" suffix r (hex imm) data)))

let warpsync b = push b (I (mk "WARPSYNC.ALL"))

(* Dependent launch (griddepcontrol): PREEXIT lets the grid launched after
   this one in the stream begin -- its CTAs take SMs as this grid's leave --
   and ACQBULK waits until every grid before this one has finished and its
   memory is visible. A kernel launched with the programmatic stream
   serialization attribute touches no global memory before its ACQBULK. *)
let preexit b = push b (I (mk "PREEXIT"))
let acqbulk b = push b (I (mk "ACQBULK"))
let mov_rr b d a = alu_op b ~defs:[ R d ] ~uses:[ R a ] ~sem:(fun m -> m.set (R d) (m.get (R a))) (pf "IMAD.MOV.U32 R%d, RZ, RZ, R%d" d a)

(* UTMASTG.2D [URg], [URmap]: URg = shared source, URg+1 = x, URg+2 = y --
   three of the aligned quad. The copy engine writes global memory, so a warp
   issues one instruction for its whole block instead of a store per row. *)
let utmastg ?(add = false) b ~g ~map =
  push b (I (mk ~late:(urs g 3 @ urs map 2) ~act:(fun m -> Store { src = m.get (UR g); map = ur_pair m map; x = m.get (UR (g + 1)); y = m.get (UR (g + 2)); add })
               (if add then pf "UTMAREDG.2D.ADD [UR%d], [UR%d]" g map else pf "UTMASTG.2D [UR%d], [UR%d]" g map)))

(* the copy engine's stores so far, closed as a group (cp.async.bulk.commit_group) *)
let utmacmdflush b = push b (I (mk ~act:(fun _ -> Flush) "UTMACMDFLUSH"))

(* wait for the copy engine to have taken the staged block before it is reused *)
let depbar_drain b = push b (I (mk ~act:(fun _ -> Drain 0) "DEPBAR.LE SB0, 0x0"))

(* wait until at most [n] groups of tensor-map stores are still reading
   shared memory (cp.async.bulk.wait_group.read): the groups finish in order,
   so this frees the staging copy the group [n + 1] back was reading *)
let depbar_le b ~n = push b (I (mk ~act:(fun _ -> Drain n) (pf "DEPBAR.LE SB0, %s" (hex n))))

(* ---- for work split over CTAs: register compares, the partials' adds, and
   the flags one CTA hands another through global memory ---- *)

let isetp_rr b ~cmp p a c =
  let f = match cmp with "EQ" -> ( = ) | "NE" -> ( <> ) | "LT" -> ( < ) | "GT" -> ( > ) | "LE" -> ( <= ) | "GE" -> ( >= ) | c -> failwith ("isetp_rr: " ^ c) in
  push b (I (mk ~defs:[ P p ] ~uses:[ R a; R c ] ~sem:(fun m -> m.set (P p) (b2i (f (m.get (R a)) (m.get (R c)))))
               (pf "ISETP.%s.U32.AND P%d, PT, R%d, R%d, PT" cmp p a c)))

let iadd3_sub b d a c = alu_op b ~defs:[ R d ] ~uses:[ R a; R c ] ~sem:(fun m -> m.set (R d) (m32 (m.get (R a) - m.get (R c)))) (pf "IADD3 R%d, PT, PT, R%d, -R%d, RZ" d a c)
(* the sum in double precision, rounded once to single: FADD's own result
   whenever the exact sum is a single-precision number, as small integers'
   are *)
let fadd b d a c =
  let f x = Int32.float_of_bits (Int32.of_int x) and bits v = Int32.to_int (Int32.bits_of_float v) land 0xffffffff in
  alu_op b ~defs:[ R d ] ~uses:[ R a; R c ] ~sem:(fun m -> m.set (R d) (bits (f (m.get (R a)) +. f (m.get (R c))))) (pf "FADD R%d, R%d, R%d" d a c)

(* a load and a store at the GPU's coherence point, past the L1: what
   ld.relaxed.gpu / st.relaxed.gpu are *)
let global_load ~d ~base ~imm ~n =
  (fun m -> for i = 0 to n - 1 do m.set (R (d + i)) (m.gload (pair m (R base) + imm + (4 * i))) done),
  fun m -> Global { addr = pair m (R base) + imm; bytes = 4 * n; write = false; strong = true }

let ldg_strong ?(desc = 4) b d ~base ~imm =
  let sem, act = global_load ~d ~base ~imm ~n:1 in
  push b (I (mk ~defs:[ R d ] ~uses:(urs desc 2) ~late:(regs base 2) ~sem ~act (pf "LDG.E.STRONG.GPU R%d, desc[UR%d]%s" d desc (addr base imm))))

let stg_strong ?(desc = 4) b ~base ~imm ~data =
  let sem, act = global_store ~base ~imm ~data ~n:1 ~strong:true in
  push b (I (mk ~uses:(urs desc 2) ~late:(regs base 2 @ [ R data ]) ~sem ~act (pf "STG.E.STRONG.GPU desc[UR%d]%s, R%d" desc (addr base imm) data)))

(* every memory access before it is visible to the whole GPU before any after
   it; the L1's lines are dropped, so what another multiprocessor wrote is
   read from L2 -- the fence and the invalidate ptxas puts around a release
   and an acquire at GPU scope *)
let membar_gpu b = push b (I (mk ~act:(fun _ -> Gpu_fence) "MEMBAR.ALL.GPU"))

let ldg_strong128 ?(desc = 4) b d ~base ~imm =
  let sem, act = global_load ~d ~base ~imm ~n:4 in
  push b (I (mk ~defs:(regs d 4) ~uses:(urs desc 2) ~late:(regs base 2) ~sem ~act (pf "LDG.E.128.STRONG.GPU R%d, desc[UR%d]%s" d desc (addr base imm))))
